"""BL semi-synthetic AUC sweep with clustered pseudo-dyad.

Three-layer model:
  Layer 1: Shared stimulus windows (30% duty — both active)
  Layer 2: Independent events within windows (each participant)
  Layer 3: Mimicry injection (P1 event → P2 response at lag, prob=kappa)

Parallelized: seeds run via joblib, surrogates internal to bl_two_stage_coupling.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from cadence.synthetic import generate_coupling_gate
from cadence.significance.bl_coupling import bl_two_stage_coupling

FS = 30.0; SEED = 42; N_SEEDS = 5; N_SURR = 300; DURATION = 600

SMILE_AUS = [43, 44, 17]
BUMP_AMP_RANGE = (0.3, 0.8)
IPI_RANGE = (10, 25)
LAG_S = 1.0  # Short lag — matches significant real sessions (lag window ≤0.6s)
BUMP_HALF_S = 0.4


def generate_clustered_dyad(duration, fs, seed=42):
    """Generate pseudo-dyad with shared-stimulus temporal clustering."""
    rng = np.random.default_rng(seed)
    N = int(duration * fs)
    n_au = 52
    p1 = np.zeros((N, n_au), dtype=np.float32)
    p2 = np.zeros((N, n_au), dtype=np.float32)

    shared = generate_coupling_gate(N, fs, {
        'duty_cycle': 0.30, 'event_range_s': (8, 25), 'ramp_s': 0.5
    }, seed=seed)

    half_samp = int(BUMP_HALF_S * fs)
    p1_event_times = []
    p2_event_times = []

    for person_seed, signal, event_list in [
        (seed, p1, p1_event_times),
        (seed + 7777, p2, p2_event_times),
    ]:
        prng = np.random.default_rng(person_seed)
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
                event_list.append(pos)
                pos += int(prng.uniform(*IPI_RANGE) * fs)

    return p1, p2, shared, np.array(p1_event_times), np.array(p2_event_times)


def inject_mimicry(p1, p2_base, p1_events, coupling_gate, kappa, seed=42):
    """Inject mimicry: P1 event → P2 bump at lag with prob kappa."""
    rng = np.random.default_rng(seed)
    N = len(p1)
    p2 = p2_base.copy()
    lag_samp = int(LAG_S * FS)
    half_samp = int(BUMP_HALF_S * FS)
    n_inj = 0
    p1_comp = sum(p1[:, au] for au in SMILE_AUS)

    for evt in p1_events:
        if evt >= N or coupling_gate[evt] < 0.5:
            continue
        if rng.random() > kappa:
            continue
        resp = evt + lag_samp + int(rng.normal(0, 0.3 * FS))
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


def run_one(p1_base, p2_base, p1_events, shared, kappa, seed):
    """Run one kappa × seed combination. Returns result dict."""
    if kappa == 0:
        p2_test = p2_base.copy()
        n_inj = 0
    else:
        p2_test, n_inj = inject_mimicry(
            p1_base, p2_base, p1_events, shared, kappa, seed=seed)

    r = bl_two_stage_coupling(
        p1_base, p2_test, FS,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
        n_surrogates=N_SURR, target_fa=0.05,
        event_prominence=0.3, min_event_iei_s=3.0,
        n_surrogates_event=N_SURR,
        population_prior=(1.5, 1.0), seed=seed)

    cats = getattr(r, 'catalogs', {})
    sc = cats.get('smile')
    # Get lag estimation details
    est_lag = getattr(r, 'estimated_lag_s', None)
    lag_conf = getattr(r, 'lag_confidence', None)
    lag_window = getattr(r, 'lag_window_s', None)

    return {
        'kappa': kappa, 'seed': seed,
        'smile_p': sc.session_p_value if sc else 1.0,
        'co_occ': sc.n_cooccurrences if sc else 0,
        'mimicry': sc.n_mimicry if sc else 0,
        'shared_stim': sc.n_shared_stimulus if sc else 0,
        'n_events_a': sc.n_events_a if sc else 0,
        'n_events_b': sc.n_events_b if sc else 0,
        'mean_lag': sc.mean_lag if sc and sc.mean_lag else 0,
        'n_inj': n_inj,
        'est_lag': est_lag,
        'lag_conf': lag_conf,
        'lag_window': lag_window,
    }


# ── Generate base dyad ──────────────────────────────────────────────────
p1_base, p2_base, shared, p1_events, p2_events = \
    generate_clustered_dyad(DURATION, FS, seed=SEED)

p1_comp = gaussian_filter1d(sum(p1_base[:, au] for au in SMILE_AUS), 0.3 * FS)
p2_comp = gaussian_filter1d(sum(p2_base[:, au] for au in SMILE_AUS), 0.3 * FS)
p1_pks, _ = find_peaks(p1_comp, prominence=0.3, distance=int(3 * FS))
p2_pks, _ = find_peaks(p2_comp, prominence=0.3, distance=int(3 * FS))

print(f"Duration: {DURATION}s, surrogates: {N_SURR}, seeds: {N_SEEDS}")
print(f"Shared window duty: {(shared > 0.5).mean():.1%}")
print(f"P1 events: {len(p1_events)} (composite peaks: {len(p1_pks)})")
print(f"P2 events: {len(p2_events)} (composite peaks: {len(p2_pks)})")
n_gated = sum(1 for e in p1_events if shared[e] > 0.5)
print(f"P1 events in coupling gate: {n_gated}")

# ── Build all jobs ───────────────────────────────────────────────────────
kappas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
jobs = [(kappa, SEED + si * 100) for kappa in kappas for si in range(N_SEEDS)]

print(f"\nRunning {len(jobs)} jobs in parallel...")
t0 = time.perf_counter()
all_results = Parallel(n_jobs=-1)(
    delayed(run_one)(p1_base, p2_base, p1_events, shared, k, s)
    for k, s in jobs)
total_time = time.perf_counter() - t0
print(f"Done in {total_time:.1f}s\n")

# ── Print detailed results ───────────────────────────────────────────────
print(f"{'kappa':>6} {'seed':>5} | {'smile_p':>8} {'co_occ':>7} {'mim':>4} "
      f"{'sh_stim':>8} {'P1_ev':>6} {'P2_ev':>6} {'n_inj':>6} "
      f"{'est_lag':>8} {'lag_window':>12} {'mean_lag':>9}")
print(f"{'-'*105}")

for r in sorted(all_results, key=lambda x: (x['kappa'], x['seed'])):
    sig = "*" if r['smile_p'] < 0.05 else " "
    el = f"{r['est_lag']:.2f}" if r['est_lag'] else "n/a"
    ml = f"{r['mean_lag']:.2f}" if r['mean_lag'] else "n/a"
    lw = (f"[{r['lag_window'][0]:.1f},{r['lag_window'][1]:.1f}]"
          if r['lag_window'] else "n/a")
    print(f"{r['kappa']:>6.1f} {r['seed']:>5} | {r['smile_p']:>7.3f}{sig} "
          f"{r['co_occ']:>7} {r['mimicry']:>4} {r['shared_stim']:>8} "
          f"{r['n_events_a']:>6} {r['n_events_b']:>6} {r['n_inj']:>6} "
          f"{el:>8} {lw:>12} {ml:>9}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY (mean over {N_SEEDS} seeds)")
print(f"{'='*70}")
print(f"  {'kappa':>6} {'mean_p':>8} {'det%':>6} {'co_occ':>7} {'mim':>5} "
      f"{'P2_ev':>6} {'n_inj':>6} {'mean_lag':>9}")
print(f"  {'-'*60}")

for kappa in kappas:
    kr = [r for r in all_results if r['kappa'] == kappa]
    mp = np.mean([r['smile_p'] for r in kr])
    det = np.mean([r['smile_p'] < 0.05 for r in kr])
    co = np.mean([r['co_occ'] for r in kr])
    mim = np.mean([r['mimicry'] for r in kr])
    p2e = np.mean([r['n_events_b'] for r in kr])
    inj = np.mean([r['n_inj'] for r in kr])
    ml = np.mean([r['mean_lag'] for r in kr if r['mean_lag']])
    print(f"  {kappa:>6.1f} {mp:>8.3f} {det:>5.0%} {co:>7.0f} {mim:>5.0f} "
          f"{p2e:>6.0f} {inj:>6.0f} {ml:>8.2f}s")

print(f"{'='*70}")
