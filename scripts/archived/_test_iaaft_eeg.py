"""IAAFT surrogates for EEG PLV pipeline — semisynthetic comparison.

EEG signals are continuous and near-Gaussian (unlike zero-inflated BL AUs),
so IAAFT should converge better. Compare null z-distributions and FA rates
against circular shift at kappa=0.1/0.2 with spatial decay.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy import stats
from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial)
from cadence.surrogates import iaaft_surrogate, iaaft_surrogate_batched
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0; DURATION = 1800; N_CH = 14; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_eeg(cfg):
    entries = discover_cached_sessions(cfg['session_cache'])
    sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
    sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
    s1, s2 = sess[-1][1], sess[0][1]
    window = find_valid_window(s1, min_duration=DURATION)

    p1_ts = s1['p1_eeg_ts']; p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
    p1_raw = s1['p1_eeg'][p1_m].copy()
    p2_ts = s2['p2_eeg_ts']; p2_s = float(p2_ts[0])
    p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + DURATION)
    p2_raw = s2['p2_eeg'][p2_m].copy()
    N = min(len(p1_raw), len(p2_raw), int(DURATION * FS))
    t_target = np.linspace(0, DURATION, N)
    p2_tw = p2_ts[p2_m] - p2_s
    p2_raw = np.stack([np.interp(t_target, p2_tw, p2_raw[:, c])
                        for c in range(N_CH)], axis=1).astype(np.float32)
    p1_raw = p1_raw[:N].astype(np.float32)

    # Average re-reference
    p1_raw -= p1_raw.mean(axis=1, keepdims=True)
    p2_raw -= p2_raw.mean(axis=1, keepdims=True)
    # Z-score per channel
    for ch in range(N_CH):
        for sig in [p1_raw, p2_raw]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd
    return p1_raw, p2_raw, N


def test_iaaft_properties_eeg(p1):
    """Verify IAAFT convergence for EEG (continuous, near-Gaussian)."""
    print("\n--- IAAFT Property Verification (EEG) ---")
    x = p1[:, :5]  # 5 channels

    t0 = time.perf_counter()
    surr = iaaft_surrogate(x, seed=42, max_iter=100)
    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.2f}s for 5ch x {len(x)} samples")

    print(f"  {'Ch':>3} {'KS':>8} {'PSD_diff':>10} {'ACF_orig':>9} {'ACF_IAAFT':>10}")
    for ch in range(5):
        orig = x[:, ch]
        ia = surr[:, ch]
        ks = stats.ks_2samp(orig, ia).statistic
        psd_o = np.abs(np.fft.rfft(orig)) ** 2
        psd_i = np.abs(np.fft.rfft(ia)) ** 2
        psd_diff = np.sqrt(np.mean((psd_i/psd_o.mean() - psd_o/psd_o.mean())**2))
        acf_o = np.corrcoef(orig[1:], orig[:-1])[0, 1]
        acf_i = np.corrcoef(ia[1:], ia[:-1])[0, 1]
        print(f"  {EPOC_CHANNEL_NAMES[ch]:>3} {ks:>8.4f} {psd_diff:>10.4f} "
              f"{acf_o:>9.4f} {acf_i:>10.4f}")


def run_plv_pipeline(p1, p2, oracle_ch, freqs, label=""):
    """Run PLV pipeline, return z-scores and diagnostics."""
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1, p2, FS, channels=oracle_ch,
        center_freqs=freqs, n_surrogates=100, n_cycles=[3, 7],
        window_s=20.0, stride_s=0.5, target_fa=0.03,
        min_event_s=5.0, metric='plv', seed=SEED, device=device,
        smooth_s=15)
    return mask, z_agg, diag


def test_null_comparison(p1, p2, oracle_ch, freqs):
    """Compare null z-distributions: PLV with circular shift vs IAAFT P1."""
    print("\n--- Null Distribution: Circular Shift vs IAAFT (no coupling) ---")

    # Standard PLV pipeline (uses circular shift internally)
    t0 = time.perf_counter()
    mask_cs, z_cs, diag_cs = run_plv_pipeline(p1, p2, oracle_ch, freqs, "CS")
    t_cs = time.perf_counter() - t0
    print(f"  Circ shift: z_mean={z_cs.mean():.3f} z_std={z_cs.std():.3f} "
          f"z_max={z_cs.max():.2f} FA={mask_cs.mean():.1%} ({t_cs:.1f}s)")

    # Generate IAAFT surrogates of P1, run PLV on each
    print("  Generating IAAFT surrogates of P1...")
    t0 = time.perf_counter()
    # Only the selected channels
    p1_sel = p1[:, oracle_ch]
    surr_p1 = iaaft_surrogate_batched(p1_sel, n_surrogates=10, seed=SEED, max_iter=100)
    t_gen = time.perf_counter() - t0
    print(f"  IAAFT generation: {t_gen:.1f}s for 10 surrogates of {p1_sel.shape}")

    # Run PLV on each IAAFT surrogate P1 vs real P2
    def _run_iaaft_plv(k):
        # Create full-channel P1 with IAAFT-replaced oracle channels
        p1_ia = p1.copy()
        for i, ch in enumerate(oracle_ch):
            p1_ia[:, ch] = surr_p1[k, :, i]
        _, z, _ = run_plv_pipeline(p1_ia, p2, oracle_ch, freqs)
        return z

    t0 = time.perf_counter()
    ia_results = Parallel(n_jobs=5)(delayed(_run_iaaft_plv)(k) for k in range(10))
    t_ia = time.perf_counter() - t0
    z_ia = np.concatenate(ia_results)
    print(f"  IAAFT PLV: z_mean={z_ia.mean():.3f} z_std={z_ia.std():.3f} "
          f"z_max={z_ia.max():.2f} ({t_ia:.1f}s)")

    print(f"\n  Null should be ~N(0,1):")
    print(f"    CS:    mean={z_cs.mean():.3f} std={z_cs.std():.3f} "
          f"skew={stats.skew(z_cs):.2f} kurt={stats.kurtosis(z_cs):.2f}")
    print(f"    IAAFT: mean={z_ia.mean():.3f} std={z_ia.std():.3f} "
          f"skew={stats.skew(z_ia):.2f} kurt={stats.kurtosis(z_ia):.2f}")


def test_coupled_comparison(p1_raw, p2_raw, N):
    """Compare detection at kappa=0.1 and 0.2 with spatial decay."""
    print("\n--- Coupled Detection: kappa sweep ---")
    freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)

    for kappa in [0.1, 0.2, 0.4]:
        gate = generate_coupling_gate(N, FS, {
            'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0
        }, seed=SEED)
        gate_mask = gate > 0.5

        p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
            p1_raw, p2_raw, gate, kappa, center_ch=2, decay_sigma=0.5, lag_samp=8)

        # Re-zscore after coupling
        p1_c = p1_raw.copy()
        for ch in range(N_CH):
            for sig in [p1_c, p2_coupled]:
                mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
                sig[:, ch] = (sig[:, ch] - mu) / sd

        oracle_ch = [i for i in np.argsort(-kappa_per_ch) if kappa_per_ch[i] > 0.03]
        t_eeg = np.arange(N) / FS

        mask, z_agg, diag = run_plv_pipeline(p1_c, p2_coupled, oracle_ch, freqs)
        wt = diag['win_times']
        gw = np.interp(wt, t_eeg, gate.astype(float)) > 0.5

        zc = z_agg[gw].mean() if gw.any() else 0
        zn = z_agg[~gw].mean() if (~gw).any() else 0

        # Hit/FA sweep
        win_rate = 1.0 / 0.5
        min_ev = max(1, int(5.0 * win_rate))
        best_hit, best_fa = 0, 1
        for thr in np.arange(0.6, 4.0, 0.2):
            m = _min_event_filter(z_agg > thr, min_ev)
            h = float((m & gw).sum() / max(gw.sum(), 1))
            f = float((m & ~gw).sum() / max((~gw).sum(), 1))
            if 0.03 <= f <= 0.07:
                if h > best_hit:
                    best_hit, best_fa = h, f

        print(f"  kappa={kappa:.1f}: z_coupled={zc:.2f} z_null={zn:.2f} "
              f"contrast={zc-zn:.2f} "
              f"hit@FA~5%={best_hit:.1%} oracle_ch={len(oracle_ch)} "
              f"thr={diag.get('z_threshold', 0):.2f}")


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    print("Loading real EEG signals...")
    p1, p2, N = load_eeg(cfg)
    print(f"  P1: {p1.shape}, P2: {p2.shape}, N={N}, duration={N/FS:.0f}s")

    freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)
    oracle_ch = list(range(N_CH))  # all channels for null test

    test_iaaft_properties_eeg(p1)
    test_null_comparison(p1, p2, oracle_ch, freqs)
    test_coupled_comparison(p1, p2, N)
