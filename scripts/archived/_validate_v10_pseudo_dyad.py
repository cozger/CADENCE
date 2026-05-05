"""V10 Pseudo-Dyad Burst Rate Validation (Standard).

Definitive test for genuine coupling vs shared activity/noise.

For each cross-session pair (P1 from session A, P2 from session B):
  1. Build pseudo-dyad from raw cached data (no coupling by construction)
  2. Run full V10 scaffold on pseudo-dyad raw data
  3. Compute full-session burst rates per modality group
  4. Compare real-dyad vs pseudo-dyad rates (Mann-Whitney U)

Features are RECOMPUTED from misaligned raw signals — not circular-shifted
derived features. Any coupling detected in pseudo-dyads is noise/artifact.

Additionally tests per-condition coupling by mapping session A's condition
timing (as relative offsets from session start) onto the pseudo-dyad.

Usage:
    python scripts/_validate_v10_pseudo_dyad.py               # 10 pairs
    python scripts/_validate_v10_pseudo_dyad.py --quick        # 5 pairs
"""

import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic_v82 import build_v82_pseudo_dyad
from cadence.constants import V10_MODALITY_KEYS

FS_OUT = 2.0
OUT_DIR = 'results/v10/pseudo_dyad_validation'

GROUPS = {
    'EEG Phase':    [0, 1, 2],
    'EEG Power':    [3, 4, 5],
    'EEG Dynamics': [6, 7, 8],
    'Face+Body':    [12, 13, 17],
    'Autonomic':    [14, 15, 16],
    'LZ Complex':   [18, 19],
}

COND_TYPES = {
    'baseline':     ['base_EO', 'base_EC'],
    'conversation': ['conv_1', 'conv_2'],
    'intervention': ['meditate_B', 'meditate_K', 'PE_1', 'PE_2'],
}


def compute_intensity(Y, group_idx, smooth_s=3.0, fs=FS_OUT):
    return gf1d(np.sqrt(np.mean(Y[:, group_idx] ** 2, axis=1)), sigma=smooth_s * fs)


def count_bursts(intensity, threshold_pctl=90, min_dur_s=2.0, fs=FS_OUT):
    thresh = np.percentile(intensity, threshold_pctl)
    above = intensity > thresh
    n_bursts = 0
    in_burst = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i; in_burst = True
        elif not above[i] and in_burst:
            if (i - start) / fs >= min_dur_s:
                n_bursts += 1
            in_burst = False
    if in_burst and (len(above) - start) / fs >= min_dur_s:
        n_bursts += 1
    return n_bursts


def session_burst_rates(Y, t):
    """Compute full-session burst rate per group. Returns dict group -> rate/min."""
    dur_min = (t[-1] - t[0]) / 60.0
    if dur_min < 1.0:
        return {}
    rates = {}
    for gname, gidx in GROUPS.items():
        intensity = compute_intensity(Y, gidx)
        nb = count_bursts(intensity)
        rates[gname] = nb / dur_min
    return rates


def session_burst_rates_by_condtype(Y, t, segments):
    """Burst rates grouped by condition type (baseline/conversation/intervention)."""
    rates = {}
    for ctype, cond_names in COND_TYPES.items():
        # Merge all matching segments
        merged_mask = np.zeros(len(t), dtype=bool)
        for cn, t0, t1 in segments:
            if cn in cond_names:
                merged_mask |= (t >= t0) & (t <= t1)
        dur_min = merged_mask.sum() / FS_OUT / 60.0
        if dur_min < 0.5:
            continue
        Y_seg = Y[merged_mask]
        for gname, gidx in GROUPS.items():
            intensity_full = compute_intensity(Y, gidx)
            intensity_seg = intensity_full[merged_mask]
            thresh = np.percentile(intensity_full, 90)  # full-session threshold
            above = intensity_seg > thresh
            n_bursts = 0
            in_burst = False
            start = 0
            for i in range(len(above)):
                if above[i] and not in_burst:
                    start = i; in_burst = True
                elif not above[i] and in_burst:
                    if (i - start) / FS_OUT >= 2.0:
                        n_bursts += 1
                    in_burst = False
            if in_burst and (len(above) - start) / FS_OUT >= 2.0:
                n_bursts += 1
            rates[(ctype, gname)] = n_bursts / dur_min
    return rates


def run_one_pseudo_pair(cached_a, cached_b, segments_rel_a, pair_id):
    """Build pseudo-dyad from raw data, run V10 scaffold, return burst rates."""
    from scripts._run_scaffold_v10 import run_from_raw_v10

    ts_a = cached_a.get('p1_eeg_ts', np.array([0, 300]))
    ts_b = cached_b.get('p2_eeg_ts', cached_b.get('p1_eeg_ts', np.array([0, 300])))
    t_start = max(ts_a[0], ts_b[0])
    t_end = min(ts_a[-1], ts_b[-1])
    dur = t_end - t_start
    if dur < 120:
        return None

    t_common = np.arange(t_start, t_end, 1.0 / FS_OUT)

    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)
    n_eeg = min(len(base.get('p1_eeg', [])), len(base.get('p2_eeg', [])))
    if n_eeg < 1000:
        return None

    # Map session A's relative segments onto pseudo-dyad time
    pseudo_segments = []
    for cname, rel_start, rel_end in segments_rel_a:
        abs_start = t_start + rel_start
        abs_end = t_start + rel_end
        if abs_end <= t_end and abs_start >= t_start:
            pseudo_segments.append((cname, abs_start, abs_end))

    try:
        z_v10, _, _, _, _ = run_from_raw_v10(
            base, t_common, segments=pseudo_segments,
            label=f'pseudo_{pair_id}')
    except Exception as e:
        return None

    result = {
        'pair_id': pair_id,
        'full': session_burst_rates(z_v10, t_common),
    }
    if pseudo_segments:
        result['by_condtype'] = session_burst_rates_by_condtype(
            z_v10, t_common, pseudo_segments)
    return result


def main(n_pairs=10, n_jobs=4):
    print("=" * 80)
    print("  V10 Pseudo-Dyad Burst Rate Validation (Standard)")
    print("  Features recomputed from misaligned raw data")
    print("=" * 80)

    t_wall = time.time()
    config = load_config()

    # ── Load sessions ────────────────────────────────────────────────
    print("\nLoading sessions...")
    cached_sessions = discover_cached_sessions(config['session_cache'])
    sessions = []
    for name, path in cached_sessions:
        try:
            cached = load_session_from_cache(path, config)
            if 'p1_eeg' in cached and 'p2_eeg' in cached:
                sessions.append((name, cached))
        except:
            continue
    print(f"  {len(sessions)} sessions loaded")

    # ── Real-dyad burst rates ────────────────────────────────────────
    print("\nComputing real-dyad burst rates from V10 scaffolds...")
    real_full = []  # list of {group: rate}
    real_by_cond = []  # list of {(condtype, group): rate}
    real_sessions = []

    v10_npzs = sorted(glob.glob('results/v10/*/scaffold_v10_ztimecourses.npz'))
    for npz_path in v10_npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        json_path = f'results/v10/{name}/scaffold_v10_results.json'
        if not os.path.exists(json_path):
            continue
        try:
            data = np.load(npz_path)
            with open(json_path) as f:
                info = json.load(f)
            Y = np.column_stack([data[f'z_{k}'] for k in V10_MODALITY_KEYS])
            t = data['t_common']
            segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

            # Full-session rates
            rates = session_burst_rates(Y, t)
            real_full.append(rates)

            # Per-condtype rates
            cond_rates = session_burst_rates_by_condtype(Y, t, segments)
            real_by_cond.append(cond_rates)

            # Store relative segments for pseudo-dyad mapping
            t0_session = t[0]
            rel_segments = [(cn, t0s - t0_session, t1s - t0_session)
                            for cn, t0s, t1s in segments]
            real_sessions.append({'name': name, 'rel_segments': rel_segments})
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"  {len(real_full)} real sessions")

    # ── Pseudo-dyad pairs ────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n = len(sessions)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((i, j))

    # Use session A's relative segments for condition mapping
    tasks = []
    for pi, (i, j) in enumerate(pairs):
        name_a = sessions[i][0]
        match = [r for r in real_sessions if name_a.lower() in r['name'].lower()]
        rel_segs = match[0]['rel_segments'] if match else []
        tasks.append((sessions[i][1], sessions[j][1], rel_segs, pi))

    print(f"\nRunning {len(tasks)} pseudo-dyad pairs (n_jobs={n_jobs})...")
    pseudo_results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_pseudo_pair)(*t) for t in tasks
    )
    pseudo_results = [r for r in pseudo_results if r is not None]
    print(f"  {len(pseudo_results)} completed")

    pseudo_full = [r['full'] for r in pseudo_results]
    pseudo_by_cond = [r.get('by_condtype', {}) for r in pseudo_results]

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Full-Session Burst Rates: Real vs Pseudo-Dyad")
    print("=" * 80)
    print(f"\n  {'Group':>14s} | {'Real':>10s} | {'Pseudo':>10s} | "
          f"{'Ratio':>6s} | {'p':>6s} | {'Verdict':>10s}")
    print(f"  " + "-" * 70)

    results = {'full_session': {}, 'by_condition_type': {}}

    for gname in GROUPS:
        rv = [r[gname] for r in real_full if gname in r]
        pv = [r[gname] for r in pseudo_full if gname in r]
        if len(rv) < 3 or len(pv) < 3:
            continue
        rm, pm = np.mean(rv), np.mean(pv)
        ratio = rm / pm if pm > 0.01 else 99.0
        U, p = mannwhitneyu(rv, pv, alternative='two-sided')
        verdict = 'COUPLING' if ratio > 1.3 and p < 0.05 else ('trend' if ratio > 1.2 and p < 0.10 else 'n.s.')
        print(f"  {gname:>14s} | {rm:>5.2f}±{np.std(rv)/np.sqrt(len(rv)):.2f} | "
              f"{pm:>5.2f}±{np.std(pv)/np.sqrt(len(pv)):.2f} | "
              f"{ratio:>6.2f} | {p:>6.3f} | {verdict:>10s}")
        results['full_session'][gname] = {
            'real_mean': float(rm), 'pseudo_mean': float(pm),
            'ratio': float(ratio), 'p_value': float(p),
            'n_real': len(rv), 'n_pseudo': len(pv), 'verdict': verdict,
        }

    # Per-condition-type comparison
    print("\n" + "=" * 80)
    print("  Per-Condition-Type Burst Rates: Real vs Pseudo-Dyad")
    print("=" * 80)

    for ctype in COND_TYPES:
        print(f"\n  {ctype.upper()}:")
        print(f"  {'Group':>14s} | {'Real':>10s} | {'Pseudo':>10s} | "
              f"{'Ratio':>6s} | {'p':>6s} | {'Verdict':>10s}")
        print(f"  " + "-" * 70)

        for gname in GROUPS:
            key = (ctype, gname)
            rv = [r[key] for r in real_by_cond if key in r]
            pv = [r[key] for r in pseudo_by_cond if key in r]
            if len(rv) < 3 or len(pv) < 3:
                continue
            rm, pm = np.mean(rv), np.mean(pv)
            ratio = rm / pm if pm > 0.01 else 99.0
            U, p = mannwhitneyu(rv, pv, alternative='two-sided')
            verdict = 'COUPLING' if ratio > 1.3 and p < 0.05 else ('trend' if ratio > 1.2 and p < 0.10 else 'n.s.')
            print(f"  {gname:>14s} | {rm:>5.2f}±{np.std(rv)/np.sqrt(len(rv)):.2f} | "
                  f"{pm:>5.2f}±{np.std(pv)/np.sqrt(len(pv)):.2f} | "
                  f"{ratio:>6.2f} | {p:>6.3f} | {verdict:>10s}")
            results['by_condition_type'][f'{ctype}_{gname}'] = {
                'real_mean': float(rm), 'pseudo_mean': float(pm),
                'ratio': float(ratio), 'p_value': float(p),
                'n_real': len(rv), 'n_pseudo': len(pv), 'verdict': verdict,
            }

    # ── Summary ──────────────────────────────────────────────────────
    all_verdicts = [v['verdict'] for v in results['full_session'].values()]
    all_verdicts += [v['verdict'] for v in results['by_condition_type'].values()]
    print(f"\n{'='*80}")
    print(f"  Summary: {all_verdicts.count('COUPLING')} COUPLING, "
          f"{all_verdicts.count('trend')} trend, {all_verdicts.count('n.s.')} n.s.")
    print(f"{'='*80}")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    output = {
        'version': 'v10', 'n_real': len(real_full), 'n_pseudo': len(pseudo_full),
        'groups': {g: idx for g, idx in GROUPS.items()},
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(OUT_DIR, 'pseudo_dyad_burst_validation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {time.time()-t_wall:.0f}s")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pairs', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    main(n_pairs=5 if args.quick else args.n_pairs, n_jobs=args.n_jobs)
