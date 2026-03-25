"""Per-event mimicry characterization — sparse major expression regime.

Models: major facial expressions (laughter, big smiles) every 1-5 min,
with kappa=0.7 mimicry probability when they occur.

Tests:
1. Event-mimicry coupling: P1 peak → P2 mimicry response (kappa=probability)
2. Amplitude-mixing coupling: P2 = kappa*P1[t-lag] + noise (at sparse duty)
3. Null: no coupling → chance match rate
4. Significance via surrogate calibration
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_bl_event_coupling)
from cadence.significance.event_sync import characterize_mimicry_events

DURATION_S = 1800
FS = 30.0
SEED = 42

SPARSE_AUS = [43, 44, 6, 7, 2]  # smile L/R, cheekSquint L/R, browInnerUp


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


def inject_amp(src, tgt, gate, aus, kappas, lag_samp):
    out = tgt.copy()
    for ch, kappa in zip(aus, kappas):
        if kappa <= 0: continue
        s_lag = np.roll(src[:, ch], lag_samp); s_lag[:lag_samp] = 0
        alpha = kappa * gate
        m = alpha > 0.001
        out[m, ch] = alpha[m]*s_lag[m] + np.sqrt(np.maximum(1-alpha[m]**2,0))*tgt[m,ch]
    return out


def run_characterization(p1, p2c, label, prom=1.5, iei=2.0,
                          lag_range=(0.5, 5.0)):
    """Run per-event characterization and print results."""
    t0 = time.perf_counter()
    result = characterize_mimicry_events(
        p1, p2c, FS, channels=SPARSE_AUS,
        event_method='peaks',
        event_kwargs={'prominence_sigma': prom, 'min_iei_s': iei},
        lag_range_s=lag_range,
        n_surrogates=200, seed=SEED)
    elapsed = time.perf_counter() - t0

    s = result['summary']
    sig = '*' if s['p_value'] < 0.05 else ' '
    sig2 = '**' if s['p_value'] < 0.01 else sig

    print(f"  {label:<35} "
          f"P1ev={s['n_p1_events']:>4} P2ev={s['n_p2_events']:>4}  "
          f"match={s['n_matched_p1_to_p2']:>3}/{s['n_p1_events']}"
          f"={s['match_rate_p1_to_p2']:.1%}  "
          f"null={s['null_match_rate']:.1%}  "
          f"z={s['z_score']:>5.2f} p={s['p_value']:.3f}{sig2}  "
          f"rev={s['match_rate_p2_to_p1']:.1%}  "
          f"lag={s['mean_lag']:.2f}s" if s['mean_lag'] else
          f"  {label:<35} "
          f"P1ev={s['n_p1_events']:>4} P2ev={s['n_p2_events']:>4}  "
          f"match={s['n_matched_p1_to_p2']:>3}/{s['n_p1_events']}"
          f"={s['match_rate_p1_to_p2']:.1%}  "
          f"null={s['null_match_rate']:.1%}  "
          f"z={s['z_score']:>5.2f} p={s['p_value']:.3f}{sig2}  "
          f"rev={s['match_rate_p2_to_p1']:.1%}")
    return result


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    print("Loading real BL signals...")
    p1, p2, T = load_bl(cfg)
    print(f"  P1: {p1.shape}, duration={T/FS:.0f}s, AUs: {SPARSE_AUS}")

    lag_samp = int(2.0 * FS)

    # ── Null: no coupling ──
    print(f"\n{'='*90}")
    print("Null: no coupling (chance coincidence rate)")
    print(f"{'='*90}")
    for prom in [1.0, 1.5, 2.0, 2.5]:
        run_characterization(p1, p2, f"null prom={prom}", prom=prom)

    # ── Event mimicry: sparse gate, varying kappa ──
    print(f"\n{'='*90}")
    print("Event mimicry coupling: duty=2%, kappa=probability of response")
    print(f"{'='*90}")
    gate_sparse = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.02, 'event_range_s': (3, 10), 'ramp_s': 0.5,
    }, seed=SEED)
    print(f"  Gate: {gate_sparse.sum()/FS:.0f}s coupled of {T/FS:.0f}s")

    for kappa in [0.3, 0.5, 0.7, 1.0]:
        kd = {ch: kappa for ch in SPARSE_AUS}
        p2c, nt = inject_bl_event_coupling(
            p1, p2, gate_sparse, kd, lag_samp=lag_samp,
            seed=SEED, peak_prominence=0.5, response_half_s=0.5, fs=FS)
        trig = sum(nt.values())
        run_characterization(p1, p2c, f"evt k={kappa:.1f} trig={trig}", prom=1.5)

    # ── Event mimicry: varying duty cycle ──
    print(f"\n{'='*90}")
    print("Event mimicry coupling: kappa=0.7, varying duty cycle")
    print(f"{'='*90}")
    for duty in [0.01, 0.02, 0.05, 0.10]:
        gate = generate_coupling_gate(T, FS, {
            'duty_cycle': duty, 'event_range_s': (3, 10), 'ramp_s': 0.5,
        }, seed=SEED)
        kd = {ch: 0.7 for ch in SPARSE_AUS}
        p2c, nt = inject_bl_event_coupling(
            p1, p2, gate, kd, lag_samp=lag_samp,
            seed=SEED, peak_prominence=0.5, response_half_s=0.5, fs=FS)
        trig = sum(nt.values())
        run_characterization(p1, p2c,
                             f"duty={duty:.0%} trig={trig} ({gate.sum()/FS:.0f}s)",
                             prom=1.5)

    # ── Amplitude mixing: sparse gate, k=0.7 ──
    print(f"\n{'='*90}")
    print("Amplitude mixing: duty=2%, k=0.7 (for comparison)")
    print(f"{'='*90}")
    kappas = np.full(len(SPARSE_AUS), 0.7)
    gate_sparse = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.02, 'event_range_s': (3, 10), 'ramp_s': 0.5,
    }, seed=SEED)
    p2c = inject_amp(p1, p2, gate_sparse, SPARSE_AUS, kappas, lag_samp)
    run_characterization(p1, p2c, "amp k=0.7 duty=2%", prom=1.5)
    # Also try with various prominence levels
    for prom in [1.0, 2.0, 2.5]:
        run_characterization(p1, p2c, f"amp k=0.7 duty=2% prom={prom}", prom=prom)

    # ── Event detection parameter sensitivity ──
    print(f"\n{'='*90}")
    print("Event detection sensitivity: event mimicry k=0.7 duty=5%")
    print(f"{'='*90}")
    gate = generate_coupling_gate(T, FS, {
        'duty_cycle': 0.05, 'event_range_s': (3, 15), 'ramp_s': 0.5,
    }, seed=SEED)
    kd = {ch: 0.7 for ch in SPARSE_AUS}
    p2c, nt = inject_bl_event_coupling(
        p1, p2, gate, kd, lag_samp=lag_samp,
        seed=SEED, peak_prominence=0.5, response_half_s=0.5, fs=FS)
    for prom in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for iei in [1.0, 2.0, 5.0]:
            run_characterization(p1, p2c,
                                 f"prom={prom} iei={iei}s",
                                 prom=prom, iei=iei)

    print("\nDone.")
