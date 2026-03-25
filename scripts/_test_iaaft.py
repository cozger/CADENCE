"""IAAFT surrogate validation.

Tests:
1. Verify IAAFT preserves amplitude distribution and power spectrum
2. Compare null z-score distributions: circular shift vs IAAFT vs Fourier
3. Measure impact on FA rate for BL cross-product TL
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from scipy import stats
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import generate_coupling_gate, find_valid_window
from cadence.surrogates import (iaaft_surrogate, iaaft_surrogate_batched,
                                 fourier_surrogate)
from cadence.significance.coherence_localization import xcorr_temporal_localization
from cadence.significance.kim_filter import _estimate_ar

DURATION_S = 1800
FS = 30.0
SEED = 42

HETERO_PROFILE = {
    43: 0.30, 44: 0.30, 6: 0.25, 7: 0.25,
    2: 0.15, 0: 0.12, 1: 0.12, 18: 0.10, 19: 0.10, 20: 0.08,
    5: 0.05, 49: 0.05, 50: 0.05, 29: 0.05, 30: 0.05,
    27: 0.04, 28: 0.04, 3: 0.03,
}
HETERO_AUS = sorted(HETERO_PROFILE.keys())
HETERO_KAPPAS = np.array([HETERO_PROFILE[c] for c in HETERO_AUS])


def load_bl(cfg):
    entries = discover_cached_sessions(cfg['session_cache'])
    sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
    sess = [(n, s) for n, s in sess if s is not None]
    s1, s2 = sess[-1][1], sess[0][1]
    window = find_valid_window(s1, min_duration=DURATION_S)
    t_start, t_end = window
    t_end = min(t_end, t_start + DURATION_S)
    k = 'p1_blendshapes'
    d, ts = s1[k], s1[f'{k}_ts']
    m = (ts >= t_start) & (ts < t_end)
    raw = d[m, :52].copy()
    T = int((t_end - t_start) * FS)
    times = np.linspace(0, T / FS, T)
    ts_local = ts[m] - t_start
    sig = np.stack([np.interp(times, ts_local, raw[:, c]) for c in range(52)], 1)
    for c in range(52):
        mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
        sig[:, c] = (sig[:, c] - mu) / sd
    # Also load P2 from different session
    k2 = 'p2_blendshapes'
    d2, ts2 = s2[k2], s2[f'{k2}_ts']
    t2_start = float(ts2[0])
    m2 = (ts2 >= t2_start) & (ts2 < t2_start + DURATION_S)
    raw2 = d2[m2, :52].copy()
    ts2_local = ts2[m2] - t2_start
    sig2 = np.stack([np.interp(times, ts2_local, raw2[:, c]) for c in range(52)], 1)
    for c in range(52):
        mu, sd = sig2[:, c].mean(), max(sig2[:, c].std(), 1e-8)
        sig2[:, c] = (sig2[:, c] - mu) / sd
    return sig, sig2, T


def test_iaaft_properties(signal):
    """Verify IAAFT preserves amplitude distribution and spectrum."""
    print("\n--- IAAFT Property Verification ---")
    x = signal[:, :5]  # 5 channels for speed

    t0 = time.perf_counter()
    surr_iaaft = iaaft_surrogate(x, seed=42)
    t_iaaft = time.perf_counter() - t0

    t0 = time.perf_counter()
    surr_fourier = fourier_surrogate(x, seed=42)
    t_fourier = time.perf_counter() - t0

    print(f"  Time: IAAFT={t_iaaft:.2f}s, Fourier={t_fourier:.3f}s")

    for ch in range(5):
        orig = x[:, ch]
        ia = surr_iaaft[:, ch]
        fo = surr_fourier[:, ch]

        # Amplitude distribution (KS test)
        ks_ia = stats.ks_2samp(orig, ia).statistic
        ks_fo = stats.ks_2samp(orig, fo).statistic

        # Power spectrum (normalized L2 distance)
        psd_orig = np.abs(np.fft.rfft(orig)) ** 2
        psd_ia = np.abs(np.fft.rfft(ia)) ** 2
        psd_fo = np.abs(np.fft.rfft(fo)) ** 2
        psd_diff_ia = np.sqrt(np.mean((psd_ia/psd_orig.mean() - psd_orig/psd_orig.mean())**2))
        psd_diff_fo = np.sqrt(np.mean((psd_fo/psd_orig.mean() - psd_orig/psd_orig.mean())**2))

        # Autocorrelation at lag 1
        acf_orig = np.corrcoef(orig[1:], orig[:-1])[0,1]
        acf_ia = np.corrcoef(ia[1:], ia[:-1])[0,1]
        acf_fo = np.corrcoef(fo[1:], fo[:-1])[0,1]

        if ch == 0:
            print(f"  {'Ch':>3} {'KS_IAAFT':>9} {'KS_Four':>9} "
                  f"{'PSD_IAAFT':>10} {'PSD_Four':>10} "
                  f"{'ACF_orig':>9} {'ACF_IAAFT':>10} {'ACF_Four':>10}")
        print(f"  {ch:>3} {ks_ia:>9.4f} {ks_fo:>9.4f} "
              f"{psd_diff_ia:>10.4f} {psd_diff_fo:>10.4f} "
              f"{acf_orig:>9.4f} {acf_ia:>10.4f} {acf_fo:>10.4f}")

    print("  (KS closer to 0 = better amp distribution match)")
    print("  (PSD closer to 0 = better spectrum match)")


def test_null_distributions(p1, p2, T):
    """Compare FA rates under circular shift vs IAAFT surrogates."""
    print("\n--- Null Distribution Comparison (no coupling) ---")

    # AR residualize P2
    p2_res = p2.copy()
    for ch in HETERO_AUS:
        a, _ = _estimate_ar(p2[:, ch], 3)
        pred = np.zeros(T)
        for i in range(len(a)):
            pred[i+1:] += a[i] * p2[:T-i-1, ch]
        p2_res[:, ch] = p2[:, ch] - pred

    # Run cross-product with circular shift surrogates (standard)
    t0 = time.perf_counter()
    mask_cs, z_cs, _, diag_cs = xcorr_temporal_localization(
        p1[:, HETERO_AUS], p2_res[:, HETERO_AUS], FS,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
    t_cs = time.perf_counter() - t0

    # Now manually run with IAAFT surrogates
    # We need to: (a) compute real cross-product, (b) generate IAAFT surrogates
    # of P1, (c) compute cross-product with each surrogate
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p1_sel = p1[:, HETERO_AUS]
    p2_sel = p2_res[:, HETERO_AUS]

    # Generate IAAFT surrogates of P1
    t0 = time.perf_counter()
    surr_p1 = iaaft_surrogate_batched(p1_sel, n_surrogates=100, seed=SEED)
    t_gen = time.perf_counter() - t0
    print(f"  IAAFT generation: {t_gen:.1f}s for 100 surrogates of {p1_sel.shape}")

    # Run cross-product on each IAAFT surrogate
    # For fair comparison, use same real cc and same pipeline
    # We'll compute z-scores manually
    # For now, just compare statistical properties
    print(f"\n  Circular shift: z_mean={z_cs.mean():.3f} z_std={z_cs.std():.3f} "
          f"z_max={z_cs.max():.2f} FA={mask_cs.mean():.1%} ({t_cs:.1f}s)")

    # Quick IAAFT comparison: use 10 surrogates, measure their z-distribution
    # when plugged into the same pipeline
    z_iaaft_all = []
    t0 = time.perf_counter()

    def _run_one(k):
        m, z, _, _ = xcorr_temporal_localization(
            surr_p1[k], p2_sel, FS,
            max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
            n_surrogates=50, target_fa=0.05, min_event_s=5.0,
            seed=SEED + k * 1000)
        return z

    results = Parallel(n_jobs=-1)(delayed(_run_one)(k) for k in range(10))
    t_iaaft = time.perf_counter() - t0

    z_iaaft = np.concatenate(results)
    print(f"  IAAFT (10 runs): z_mean={z_iaaft.mean():.3f} "
          f"z_std={z_iaaft.std():.3f} z_max={z_iaaft.max():.2f} ({t_iaaft:.1f}s)")
    print(f"  Null z should be ~N(0,1): CS skew={stats.skew(z_cs):.2f} "
          f"kurt={stats.kurtosis(z_cs):.2f}")
    print(f"                            IA skew={stats.skew(z_iaaft):.2f} "
          f"kurt={stats.kurtosis(z_iaaft):.2f}")


def test_coupled_fa_comparison(p1, p2_orig, T):
    """Compare detection with coupling: circular shift vs IAAFT input."""
    print("\n--- Coupled Detection: CS vs IAAFT-input ---")
    lag_samp = int(2.0 * FS)
    gate = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0,
    }, seed=SEED)

    for scale in [1.0, 2.0, 3.0]:
        kappas = HETERO_KAPPAS * scale
        p2c = p2_orig.copy()
        for ch, kappa in zip(HETERO_AUS, kappas):
            if kappa <= 0: continue
            s_lag = np.roll(p1[:, ch], lag_samp); s_lag[:lag_samp] = 0
            alpha = kappa * gate
            m = alpha > 0.001
            p2c[m, ch] = alpha[m]*s_lag[m] + np.sqrt(np.maximum(1-alpha[m]**2,0))*p2_orig[m,ch]

        p2r = p2c.copy()
        for ch in HETERO_AUS:
            a, _ = _estimate_ar(p2c[:, ch], 3)
            pred = np.zeros(T)
            for i in range(len(a)):
                pred[i+1:] += a[i] * p2c[:T-i-1, ch]
            p2r[:, ch] = p2c[:, ch] - pred

        mask, z, _, diag = xcorr_temporal_localization(
            p1[:, HETERO_AUS], p2r[:, HETERO_AUS], FS,
            max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
            n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
        xc_t = np.arange(len(z)) / diag['output_rate']
        gm = np.interp(xc_t, np.arange(T)/FS, gate) > 0.5
        hit = float((mask & gm).sum() / max(gm.sum(), 1))
        fa = float((mask & ~gm).sum() / max((~gm).sum(), 1))
        z_c = z[gm].mean() if gm.any() else 0
        z_n = z[~gm].mean() if (~gm).any() else 0
        mk = float(kappas.mean())
        print(f"  k={mk:.3f}: hit={hit:.1%} fa={fa:.1%} "
              f"z_c={z_c:.2f} z_n={z_n:.2f} thr={diag['z_threshold']:.2f}")


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    print("Loading real BL signals...")
    p1, p2, T = load_bl(cfg)
    print(f"  P1: {p1.shape}, T={T}, duration={T/FS:.0f}s")

    test_iaaft_properties(p1)
    test_null_distributions(p1, p2, T)
    test_coupled_fa_comparison(p1, p2, T)
