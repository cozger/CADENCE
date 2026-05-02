"""Empirical test: 60 Hz vs 30 Hz impact on Stage 2 event detection.

Uses the clustered pseudo-dyad with mimicry injection at both rates.
Compares: n_events detected, co-occurrence count, p-value, lag precision.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from cadence.synthetic import generate_coupling_gate
from cadence.significance.bl_coupling import bl_two_stage_coupling

DURATION = 600; SEED = 42; N_SEEDS = 3; N_SURR = 200
SMILE_AUS = [43, 44, 17]
BUMP_AMP_RANGE = (0.3, 0.8)
IPI_RANGE = (10, 25)
BUMP_HALF_S = 0.4
LAG_S = 1.0


def generate_clustered_dyad(duration, fs, seed=42):
    """Generate pseudo-dyad at specified sample rate."""
    rng = np.random.default_rng(seed)
    N = int(duration * fs)
    p1 = np.zeros((N, 52), dtype=np.float32)
    p2 = np.zeros((N, 52), dtype=np.float32)

    shared = generate_coupling_gate(N, fs, {
        'duty_cycle': 0.30, 'event_range_s': (8, 25), 'ramp_s': 0.5
    }, seed=seed)

    half_samp = int(BUMP_HALF_S * fs)
    p1_events, p2_events = [], []

    for pseed, signal, evts in [(seed, p1, p1_events),
                                 (seed + 7777, p2, p2_events)]:
        prng = np.random.default_rng(pseed)
        active = shared > 0.5
        edges = np.diff(active.astype(np.int8), prepend=0)
        onsets = np.where(edges == 1)[0]
        offsets = np.where(edges == -1)[0]
        if len(offsets) < len(onsets):
            offsets = np.append(offsets, N)

        for on, off in zip(onsets, offsets):
            pos = on + int(prng.uniform(1, 5) * fs)
            while pos < off - half_samp:
                amp = prng.uniform(*BUMP_AMP_RANGE)
                s = max(0, pos - half_samp)
                e = min(N, pos + half_samp + 1)
                tt = np.arange(s, e) - pos
                env = np.exp(-tt**2 / (2 * (half_samp / 2.5)**2))
                for au in SMILE_AUS:
                    signal[s:e, au] += amp * env
                evts.append(pos)
                pos += int(prng.uniform(*IPI_RANGE) * fs)

    return p1, p2, shared, np.array(p1_events), np.array(p2_events)


def inject_mimicry(p1, p2_base, p1_events, gate, kappa, fs, seed=42):
    rng = np.random.default_rng(seed)
    N = len(p1)
    p2 = p2_base.copy()
    lag_samp = int(LAG_S * fs)
    half_samp = int(BUMP_HALF_S * fs)
    n_inj = 0
    p1_comp = sum(p1[:, au] for au in SMILE_AUS)

    for evt in p1_events:
        if evt >= N or gate[evt] < 0.5:
            continue
        if rng.random() > kappa:
            continue
        resp = evt + lag_samp + int(rng.normal(0, 0.3 * fs))
        if resp < half_samp or resp >= N - half_samp:
            continue
        src_amp = p1_comp[evt] / len(SMILE_AUS)
        amp = max(0.2, src_amp * rng.uniform(0.6, 1.0))
        s, e = resp - half_samp, resp + half_samp + 1
        tt = np.arange(s, e) - resp
        env = np.exp(-tt**2 / (2 * (half_samp / 2.5)**2))
        for au in SMILE_AUS:
            p2[s:e, au] += amp * env
        n_inj += 1
    return p2, n_inj


def run_one(fs, kappa, seed):
    """Run full pipeline at given fs and kappa."""
    p1, p2, shared, p1_ev, p2_ev = generate_clustered_dyad(DURATION, fs, seed=SEED)

    if kappa > 0:
        p2, n_inj = inject_mimicry(p1, p2, p1_ev, shared, kappa, fs, seed=seed)
    else:
        n_inj = 0

    # Check raw composite peak count
    comp = gaussian_filter1d(sum(p1[:, au] for au in SMILE_AUS), 0.3 * fs)
    pks, props = find_peaks(comp, prominence=0.3, distance=int(3 * fs))
    mean_prom = props['prominences'].mean() if len(pks) > 0 else 0

    r = bl_two_stage_coupling(
        p1, p2, fs,
        max_lag_s=5.0, lag_step_s=0.1,
        smooth_s=3.0,  # fixed for fair comparison
        n_surrogates=N_SURR, target_fa=0.05,
        event_prominence=0.3, min_event_iei_s=3.0,
        n_surrogates_event=N_SURR,
        population_prior=(1.5, 1.0), seed=seed)

    cats = getattr(r, 'catalogs', {})
    sc = cats.get('smile')
    return {
        'fs': fs, 'kappa': kappa, 'seed': seed,
        'p1_peaks': len(pks),
        'mean_prominence': mean_prom,
        'n_events_a': sc.n_events_a if sc else 0,
        'n_events_b': sc.n_events_b if sc else 0,
        'co_occ': sc.n_cooccurrences if sc else 0,
        'p_value': sc.session_p_value if sc else 1.0,
        'mean_lag': sc.mean_lag if sc and sc.mean_lag else 0,
        'n_mimicry': sc.n_mimicry if sc else 0,
        'n_inj': n_inj,
    }


# ── Run ──────────────────────────────────────────────────────────────────
print(f"Comparing 30 Hz vs 60 Hz, Stage 2 event detection")
print(f"Duration={DURATION}s, seeds={N_SEEDS}, surrogates={N_SURR}\n")

jobs = []
for fs in [30.0, 60.0]:
    for kappa in [0.0, 0.3, 0.5, 1.0]:
        for si in range(N_SEEDS):
            jobs.append((fs, kappa, SEED + si * 100))

# Sequential — bl_two_stage_coupling uses GPU internally
print(f"Running {len(jobs)} jobs sequentially (GPU-bound)...")
import time as _t
_t0 = _t.perf_counter()
results = [run_one(fs, k, s) for fs, k, s in jobs]
print(f"Done in {_t.perf_counter()-_t0:.0f}s.\n")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"{'fs':>5} {'kappa':>6} | {'P1_pks':>7} {'prom':>6} {'ev_a':>5} "
      f"{'ev_b':>5} {'co_oc':>6} {'p_val':>7} {'mean_lag':>9} {'mim':>4} {'inj':>4}")
print(f"{'-'*80}")

for fs in [30.0, 60.0]:
    for kappa in [0.0, 0.3, 0.5, 1.0]:
        kr = [r for r in results if r['fs'] == fs and r['kappa'] == kappa]
        pks = np.mean([r['p1_peaks'] for r in kr])
        prom = np.mean([r['mean_prominence'] for r in kr])
        ea = np.mean([r['n_events_a'] for r in kr])
        eb = np.mean([r['n_events_b'] for r in kr])
        co = np.mean([r['co_occ'] for r in kr])
        pv = np.mean([r['p_value'] for r in kr])
        ml = np.mean([r['mean_lag'] for r in kr if r['mean_lag']])
        mim = np.mean([r['n_mimicry'] for r in kr])
        inj = np.mean([r['n_inj'] for r in kr])
        det = np.mean([r['p_value'] < 0.05 for r in kr])
        print(f"{fs:>5.0f} {kappa:>6.1f} | {pks:>7.0f} {prom:>6.2f} {ea:>5.0f} "
              f"{eb:>5.0f} {co:>6.0f} {pv:>7.3f} {ml:>8.2f}s {mim:>4.0f} {inj:>4.0f}")
    print()

# Direct comparison
print(f"\n{'='*60}")
print(f"  DIRECT COMPARISON at kappa=0.5")
print(f"{'='*60}")
for fs in [30.0, 60.0]:
    kr = [r for r in results if r['fs'] == fs and r['kappa'] == 0.5]
    co = np.mean([r['co_occ'] for r in kr])
    pv = np.mean([r['p_value'] for r in kr])
    det = np.mean([r['p_value'] < 0.05 for r in kr])
    ea = np.mean([r['n_events_a'] for r in kr])
    eb = np.mean([r['n_events_b'] for r in kr])
    prom = np.mean([r['mean_prominence'] for r in kr])
    ml = np.mean([r['mean_lag'] for r in kr if r['mean_lag']])
    print(f"  {fs:.0f} Hz: events={ea:.0f}/{eb:.0f}, co-occ={co:.1f}, "
          f"p={pv:.3f}, det={det:.0%}, prom={prom:.3f}, lag={ml:.3f}s")
print(f"{'='*60}")
