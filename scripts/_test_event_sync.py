"""Event Synchronization TL — semi-synthetic validation.

Tests both coupling models:
1. Amplitude mixing (standard): P2 = κ·P1[t-lag] + √(1-κ²)·P2_orig
2. Event mimicry (new): P1 peak → P2 mimicry response with probability κ

Compares event sync vs cross-product on both models.
All tests parallelized via joblib.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_bl_event_coupling)
from cadence.significance.event_sync import event_sync_temporal_localization
from cadence.significance.coherence_localization import xcorr_temporal_localization
from cadence.significance.kim_filter import _estimate_ar

DURATION_S = 1800
FS = 30.0
SEED = 42
N_JOBS = -1

HETERO_PROFILE = {
    43: 0.30, 44: 0.30, 6: 0.25, 7: 0.25,
    2: 0.15, 0: 0.12, 1: 0.12, 18: 0.10, 19: 0.10, 20: 0.08,
    5: 0.05, 49: 0.05, 50: 0.05, 29: 0.05, 30: 0.05,
    27: 0.04, 28: 0.04, 3: 0.03,
}
HETERO_AUS = sorted(HETERO_PROFILE.keys())
HETERO_KAPPAS = np.array([HETERO_PROFILE[c] for c in HETERO_AUS])


def load_bl_signals(cfg):
    entries = discover_cached_sessions(cfg['session_cache'])
    sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
    sess = [(n, s) for n, s in sess if s is not None]
    s1, s2 = sess[-1][1], sess[0][1]
    window = find_valid_window(s1, min_duration=DURATION_S)
    t_start, t_end = window
    t_end = min(t_end, t_start + DURATION_S)

    def extract(s, pfx):
        k = f'{pfx}_blendshapes'
        d, ts = s[k], s[f'{k}_ts']
        m = (ts >= t_start) & (ts < t_end)
        return d[m, :52].copy(), ts[m] - t_start

    p1, p1t = extract(s1, 'p1')
    p2, p2t = extract(s2, 'p2')
    t_s = max(float(p1t[0]), float(p2t[0]))
    t_e = min(float(p1t[-1]), float(p2t[-1]))
    T = int((t_e - t_s) * FS)
    times = np.linspace(t_s, t_e, T)
    p1s = np.stack([np.interp(times, p1t, p1[:, c]) for c in range(52)], 1)
    p2s = np.stack([np.interp(times, p2t, p2[:, c]) for c in range(52)], 1)
    for c in range(52):
        for sig in [p1s, p2s]:
            mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
            sig[:, c] = (sig[:, c] - mu) / sd
    return p1s, p2s, T


def inject_amplitude_mixing(src, tgt, gate, coupled_aus, kappas, lag_samp):
    out = tgt.copy()
    for ch, kappa in zip(coupled_aus, kappas):
        if kappa <= 0: continue
        s_lag = np.roll(src[:, ch], lag_samp); s_lag[:lag_samp] = 0
        alpha = kappa * gate
        m = alpha > 0.001
        out[m, ch] = alpha[m]*s_lag[m] + np.sqrt(np.maximum(1-alpha[m]**2, 0))*tgt[m, ch]
    return out


def ar_residualize(signal, aus, order=3):
    T = signal.shape[0]
    res = signal.copy()
    for ch in aus:
        a, _ = _estimate_ar(signal[:, ch], order)
        pred = np.zeros(T)
        for i in range(len(a)):
            pred[i+1:] += a[i] * signal[:T-i-1, ch]
        res[:, ch] = signal[:, ch] - pred
    return res


def metrics(mask, gate_mask):
    if gate_mask.sum() == 0: return {'hit': 0, 'fa': 0, 'f1': 0}
    hit = float((mask & gate_mask).sum() / gate_mask.sum())
    nm = ~gate_mask
    fa = float((mask & nm).sum() / max(nm.sum(), 1))
    prec = (mask & gate_mask).sum() / max(mask.sum(), 1)
    f1 = 2*prec*hit / max(prec+hit, 1e-10)
    return {'hit': hit, 'fa': fa, 'f1': float(f1)}


def run_both_detectors(p1, p2_coupled, gate, T, label=""):
    """Run event sync + cross-product, return results dict."""
    # Event sync
    t0 = time.perf_counter()
    mask_es, z_es, diag_es = event_sync_temporal_localization(
        p1, p2_coupled, FS, channels=HETERO_AUS,
        event_method='peaks',
        event_kwargs={'prominence_sigma': 0.5, 'min_iei_s': 0.3},
        window_s=20.0, stride_s=3.0,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
    t_es = time.perf_counter() - t0

    wt_es = diag_es['win_times']
    gm_es = np.interp(wt_es, np.arange(T)/FS, gate) > 0.5
    m_es = metrics(mask_es, gm_es)

    # Cross-product
    p2_res = ar_residualize(p2_coupled, HETERO_AUS)
    t0 = time.perf_counter()
    mask_xc, z_xc, _, diag_xc = xcorr_temporal_localization(
        p1[:, HETERO_AUS], p2_res[:, HETERO_AUS], FS,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
    t_xc = time.perf_counter() - t0

    xc_times = np.arange(len(z_xc)) / diag_xc['output_rate']
    gm_xc = np.interp(xc_times, np.arange(T)/FS, gate) > 0.5
    m_xc = metrics(mask_xc, gm_xc)

    # Correlation
    z_xc_rs = np.interp(wt_es, xc_times, z_xc)
    corr = 0.0
    if len(z_es) > 3 and np.std(z_es) > 1e-8 and np.std(z_xc_rs) > 1e-8:
        corr = np.corrcoef(z_es, z_xc_rs)[0, 1]

    z_es_c = z_es[gm_es].mean() if gm_es.any() else 0
    z_es_n = z_es[~gm_es].mean() if (~gm_es).any() else 0
    z_xc_c = z_xc_rs[gm_es].mean() if gm_es.any() else 0
    z_xc_n = z_xc_rs[~gm_es].mean() if (~gm_es).any() else 0

    return {
        'label': label,
        'es': m_es, 'xc': m_xc,
        'z_es_contrast': z_es_c - z_es_n,
        'z_xc_contrast': z_xc_c - z_xc_n,
        'corr': corr,
        't_es': t_es, 't_xc': t_xc,
        'n_events_p1': diag_es['total_events_p1'],
        'n_events_p2': diag_es['total_events_p2'],
    }


def run_test_config(p1, p2_orig, T, coupling_model, kappa_scale, gate, lag_samp):
    """Run one test configuration (for parallel dispatch)."""
    kappas = HETERO_KAPPAS * kappa_scale
    mean_k = float(kappas.mean())

    if coupling_model == 'amplitude':
        p2_coupled = inject_amplitude_mixing(
            p1, p2_orig, gate, HETERO_AUS, kappas, lag_samp)
        label = f"amp  k={mean_k:.3f}"
    elif coupling_model == 'event':
        kappa_dict = {ch: k for ch, k in zip(HETERO_AUS, kappas)}
        p2_coupled, n_trig = inject_bl_event_coupling(
            p1, p2_orig, gate, kappa_dict, lag_samp=lag_samp,
            seed=SEED + int(kappa_scale*100),
            peak_prominence=0.5, response_half_s=0.5, fs=FS)
        total_trig = sum(n_trig.values())
        label = f"evt  k={mean_k:.3f} trig={total_trig}"
    else:
        raise ValueError(f"Unknown model: {coupling_model}")

    return run_both_detectors(p1, p2_coupled, gate, T, label)


def print_result(r):
    print(f"  {r['label']:<28} "
          f"ES: hit={r['es']['hit']:>5.1%} fa={r['es']['fa']:>5.1%}  "
          f"XC: hit={r['xc']['hit']:>5.1%} fa={r['xc']['fa']:>5.1%}  "
          f"z_ES={r['z_es_contrast']:>5.2f} z_XC={r['z_xc_contrast']:>5.2f}  "
          f"corr={r['corr']:.2f}")


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    print("Loading real BL signals...")
    p1, p2, T = load_bl_signals(cfg)
    print(f"  P1: {p1.shape}, P2: {p2.shape}, duration={T/FS:.0f}s")

    lag_samp = int(2.0 * FS)
    gate = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0,
    }, seed=SEED)

    # ── Sparsity analysis ──
    print("\n--- Event Sparsity Analysis ---")
    from cadence.significance.event_sync import detect_events_multichannel
    for sigma in [1.0, 1.5, 2.0, 2.5, 3.0]:
        evts, cnts = detect_events_multichannel(
            p1, FS, method='threshold', channels=HETERO_AUS,
            threshold_sigma=sigma, min_iei_s=0.5)
        rates = cnts / (T / FS)
        total_rate = rates.sum()
        print(f"  sigma={sigma:.1f}: {total_rate:.1f} total ev/s "
              f"({rates.mean():.3f} /s/ch), "
              f"expected random coincidences/20s = {(total_rate*20)**2 * 0.5/20:.0f}")

    # ── Sparse event sync: top 5 AUs only, high threshold ──
    TOP5 = [43, 44, 6, 7, 2]  # highest kappa AUs
    TOP5_KAPPAS = np.array([HETERO_PROFILE[c] for c in TOP5])

    print("\n--- Sparse Event Sync (top5 AUs, sigma=2.0) ---")
    print(f"  {'Config':<28} {'ES: hit':>10} {'fa':>5}  {'XC: hit':>10} {'fa':>5}  "
          f"{'z_ES':>5} {'z_XC':>5}  {'corr':>5}")

    def run_sparse_test(kappa_scale, model):
        kappas = TOP5_KAPPAS * kappa_scale
        mean_k = float(kappas.mean())
        if model == 'amplitude':
            p2c = inject_amplitude_mixing(p1, p2, gate, TOP5, kappas, lag_samp)
            lbl = f"amp  k={mean_k:.3f}"
        else:
            kd = {ch: k for ch, k in zip(TOP5, kappas)}
            p2c, nt = inject_bl_event_coupling(
                p1, p2, gate, kd, lag_samp=lag_samp,
                seed=SEED+int(kappa_scale*100), fs=FS)
            lbl = f"evt  k={mean_k:.3f} trig={sum(nt.values())}"

        # Event sync with sparse settings
        mask_es, z_es, diag_es = event_sync_temporal_localization(
            p1, p2c, FS, channels=TOP5,
            event_method='threshold',
            event_kwargs={'threshold_sigma': 2.0, 'min_iei_s': 0.5},
            window_s=30.0, stride_s=5.0,
            n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
        wt = diag_es['win_times']
        gm = np.interp(wt, np.arange(T)/FS, gate) > 0.5
        m_es = metrics(mask_es, gm)

        # Cross-product on same channels
        p2r = ar_residualize(p2c, TOP5)
        mask_xc, z_xc, _, diag_xc = xcorr_temporal_localization(
            p1[:, TOP5], p2r[:, TOP5], FS,
            max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
            n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
        xc_t = np.arange(len(z_xc)) / diag_xc['output_rate']
        gm_xc = np.interp(xc_t, np.arange(T)/FS, gate) > 0.5
        m_xc = metrics(mask_xc, gm_xc)

        z_xc_rs = np.interp(wt, xc_t, z_xc)
        corr = 0
        if len(z_es) > 3 and np.std(z_es) > 1e-8 and np.std(z_xc_rs) > 1e-8:
            corr = np.corrcoef(z_es, z_xc_rs)[0, 1]
        z_c = z_es[gm].mean() if gm.any() else 0
        z_n = z_es[~gm].mean() if (~gm).any() else 0
        z_xcc = z_xc_rs[gm].mean() if gm.any() else 0
        z_xcn = z_xc_rs[~gm].mean() if (~gm).any() else 0

        return {'label': lbl, 'es': m_es, 'xc': m_xc,
                'z_es_contrast': z_c-z_n, 'z_xc_contrast': z_xcc-z_xcn,
                'corr': corr, 'n_events_p1': diag_es['total_events_p1'],
                'n_events_p2': diag_es['total_events_p2'],
                't_es': 0, 't_xc': 0}

    # Amplitude mixing
    for s in [1.0, 2.0, 3.0]:
        r = run_sparse_test(s, 'amplitude')
        print_result(r)
        print(f"    (events: P1={r['n_events_p1']}, P2={r['n_events_p2']})")

    # Event mimicry
    print()
    for s in [3.0, 5.0, 8.0]:
        r = run_sparse_test(s, 'event')
        print_result(r)
        print(f"    (events: P1={r['n_events_p1']}, P2={r['n_events_p2']})")

    print("\nDone.")
