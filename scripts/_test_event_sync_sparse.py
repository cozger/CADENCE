"""Event Sync TL — Sparse high-kappa event regime.

Models real facial mimicry: major expressions (laughter, significant smiles)
occurring every 10-30s with strong mimicry (kappa=0.5-1.0) when they occur.

This is the natural regime for event synchronization — sparse events where
random coincidences are rare and genuine coupling is detectable.

Tests both:
- Event-mimicry coupling (P1 peak → P2 response at lag)
- Amplitude mixing at high kappa with low duty
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

# Major expression AUs — strong mimicry when they occur
SPARSE_AUS = [43, 44, 6, 7, 2]  # smile L/R, cheekSquint L/R, browInnerUp
SPARSE_KAPPAS = np.array([0.70, 0.70, 0.70, 0.70, 0.70])
N_JOBS = -1


def load_bl(cfg):
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


def inject_amplitude_mixing(src, tgt, gate, aus, kappas, lag_samp):
    out = tgt.copy()
    for ch, kappa in zip(aus, kappas):
        if kappa <= 0: continue
        s_lag = np.roll(src[:, ch], lag_samp); s_lag[:lag_samp] = 0
        alpha = kappa * gate
        m = alpha > 0.001
        out[m, ch] = alpha[m]*s_lag[m] + np.sqrt(np.maximum(1-alpha[m]**2,0))*tgt[m,ch]
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


def run_one_config(p1, p2_orig, T, gate, lag_samp,
                   coupling_model, kappa_scale,
                   es_method, es_kwargs, es_window, es_stride):
    """Run a single test config (event sync + cross-product)."""
    kappas = SPARSE_KAPPAS * kappa_scale

    if coupling_model == 'amplitude':
        p2c = inject_amplitude_mixing(p1, p2_orig, gate, SPARSE_AUS, kappas, lag_samp)
        label = f"amp k={kappas.mean():.2f}"
    else:
        kd = {ch: k for ch, k in zip(SPARSE_AUS, kappas)}
        p2c, nt = inject_bl_event_coupling(
            p1, p2_orig, gate, kd, lag_samp=lag_samp,
            seed=SEED + int(kappa_scale*100),
            peak_prominence=0.5, response_half_s=0.5, fs=FS)
        label = f"evt k={kappas.mean():.2f} trig={sum(nt.values())}"

    # Event sync
    mask_es, z_es, diag_es = event_sync_temporal_localization(
        p1, p2c, FS, channels=SPARSE_AUS,
        event_method=es_method, event_kwargs=es_kwargs,
        window_s=es_window, stride_s=es_stride,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
    wt = diag_es['win_times']
    gm_es = np.interp(wt, np.arange(T)/FS, gate) > 0.5
    m_es = metrics(mask_es, gm_es)
    z_es_c = z_es[gm_es].mean() if gm_es.any() else 0
    z_es_n = z_es[~gm_es].mean() if (~gm_es).any() else 0

    # Cross-product
    p2r = ar_residualize(p2c, SPARSE_AUS)
    mask_xc, z_xc, _, diag_xc = xcorr_temporal_localization(
        p1[:, SPARSE_AUS], p2r[:, SPARSE_AUS], FS,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0, seed=SEED)
    xc_t = np.arange(len(z_xc)) / diag_xc['output_rate']
    gm_xc = np.interp(xc_t, np.arange(T)/FS, gate) > 0.5
    m_xc = metrics(mask_xc, gm_xc)
    z_xc_rs = np.interp(wt, xc_t, z_xc)
    z_xc_c = z_xc_rs[gm_es].mean() if gm_es.any() else 0
    z_xc_n = z_xc_rs[~gm_es].mean() if (~gm_es).any() else 0

    corr = 0
    if len(z_es) > 3 and np.std(z_es) > 1e-8 and np.std(z_xc_rs) > 1e-8:
        corr = np.corrcoef(z_es, z_xc_rs)[0, 1]

    return {
        'label': label, 'es': m_es, 'xc': m_xc,
        'z_es_c': z_es_c, 'z_es_n': z_es_n,
        'z_xc_c': z_xc_c, 'z_xc_n': z_xc_n,
        'corr': corr,
        'n_ev_p1': diag_es['total_events_p1'],
        'n_ev_p2': diag_es['total_events_p2'],
        'es_method': f"{es_method}({es_kwargs})",
    }


def print_result(r):
    print(f"  {r['label']:<30} "
          f"ES:{r['es']['hit']:>5.1%}/{r['es']['fa']:>4.1%} "
          f"XC:{r['xc']['hit']:>5.1%}/{r['xc']['fa']:>4.1%} "
          f"z_ES={r['z_es_c']-r['z_es_n']:>5.2f} "
          f"z_XC={r['z_xc_c']-r['z_xc_n']:>5.2f} "
          f"r={r['corr']:.2f} "
          f"ev={r['n_ev_p1']}/{r['n_ev_p2']}")


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    print("Loading real BL signals...")
    p1, p2, T = load_bl(cfg)
    print(f"  P1: {p1.shape}, duration={T/FS:.0f}s")
    print(f"  AUs: {SPARSE_AUS}, kappas: {SPARSE_KAPPAS}")

    lag_samp = int(2.0 * FS)

    # ── Test 1: Very sparse coupling — major expressions every 1-5 min ──
    # κ=0.7 uniform, duty 1-5%, events 3-15s long
    header = (f"  {'Config':<30} {'ES: hit/FA':>12} {'XC: hit/FA':>12} "
              f"{'z_ES':>5} {'z_XC':>5} {'r':>5} {'events':>10}")

    duty_cycles = [0.01, 0.02, 0.03, 0.05]

    print(f"\n{'='*80}")
    print(f"Test 1: Amplitude mixing, kappa=0.7, very sparse gates")
    print(f"{'='*80}")

    for duty in duty_cycles:
        gate = generate_coupling_gate(T, FS, {
            'duty_cycle': duty, 'event_range_s': (3, 15), 'ramp_s': 0.5,
        }, seed=SEED)
        coupled_s = gate.sum() / FS
        print(f"\n  Duty={duty:.0%} ({coupled_s:.0f}s of {T/FS:.0f}s):")
        print(header)

        # Run with various event detection thresholds
        configs = [
            ('thr_2.0 w30', 'threshold', {'threshold_sigma': 2.0, 'min_iei_s': 0.5}, 30, 5),
            ('thr_2.5 w30', 'threshold', {'threshold_sigma': 2.5, 'min_iei_s': 1.0}, 30, 5),
            ('thr_3.0 w60', 'threshold', {'threshold_sigma': 3.0, 'min_iei_s': 1.0}, 60, 10),
            ('thr_3.0 w120', 'threshold', {'threshold_sigma': 3.0, 'min_iei_s': 1.0}, 120, 20),
            ('peak_1.0 w60', 'peaks', {'prominence_sigma': 1.0, 'min_iei_s': 1.0}, 60, 10),
            ('peak_1.5 w120', 'peaks', {'prominence_sigma': 1.5, 'min_iei_s': 1.0}, 120, 20),
        ]
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_config)(p1, p2, T, gate, lag_samp, 'amplitude', 1.0,
                                    m, kw, ws, ss)
            for _, m, kw, ws, ss in configs)
        for r in results:
            print_result(r)

    # ── Test 2: Event mimicry model, very sparse ──
    print(f"\n{'='*80}")
    print(f"Test 2: Event mimicry model, duty=2%, kappa sweep")
    print(f"{'='*80}")
    gate = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.02, 'event_range_s': (3, 10), 'ramp_s': 0.5,
    }, seed=SEED)
    print(f"  Gate: {gate.sum()/FS:.0f}s coupled")
    print(header)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_config)(p1, p2, T, gate, lag_samp, 'event', ks,
                                'threshold', {'threshold_sigma': 2.5, 'min_iei_s': 1.0},
                                60, 10)
        for ks in [1.0, 2.0, 3.0, 5.0, 8.0])
    for r in results:
        print_result(r)

    print("\nDone.")
