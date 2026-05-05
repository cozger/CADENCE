"""Hierarchical rSLDS on V8 9D features with constrained Viterbi.

Fits shared-transition rSLDS across all V8 sessions:
  - Shared: dynamics (A), transitions (W, S), recurrent weights (R)
  - Session-specific: emissions (C, d, R_emit, F)
  - Post-hoc: constrained Viterbi (min_dwell=20 = 10s)

State identifiability solved by Hungarian alignment + shared transitions.

Usage:
    python scripts/_run_hierarchical_v8.py
"""

import sys, os, time, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, iohmm_flexibility_metrics,
    build_observation_mask, fit_slds, fit_hierarchical_slds,
    _align_states_to_reference,
)
from scripts._run_rslds_phase2 import MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS, FS_OUT

MIN_DWELL = 20  # 10s at 2Hz
K = 4
D_LATENT = 3


def load_all_v8():
    """Load all V8 sessions as (Y, U, obs_mask, t, name) tuples."""
    npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_v8_ztimecourses.npz'))
    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']
        Y = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
        T, D = Y.shape
        U = np.zeros((T, 2), dtype=np.float64)
        obs_mask = np.ones((T, D), dtype=bool)
        sessions.append((Y, U, obs_mask, t, name))
    return sessions


def load_conditions(session_name):
    """Load condition segments from V8 JSON."""
    json_path = f'results/rslds/{session_name}/rslds_scaffold_v8_results.json'
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        info = json.load(f)
    return [(s[0], s[1], s[2]) for s in info.get('segments', [])]


def constrained_viterbi_from_gamma(gamma, min_dwell=MIN_DWELL):
    """Apply min-dwell constraint to soft posteriors via DP.

    Simple approach: argmax gamma, then merge short segments into neighbors.
    """
    T, K = gamma.shape
    path = np.argmax(gamma, axis=1)

    if min_dwell <= 1:
        return path

    # Iteratively merge short segments
    for _ in range(10):  # max iterations
        changed = False
        # Find segment boundaries
        seg_starts = [0]
        for t in range(1, T):
            if path[t] != path[t-1]:
                seg_starts.append(t)
        seg_starts.append(T)

        for i in range(len(seg_starts) - 1):
            s, e = seg_starts[i], seg_starts[i+1]
            if e - s < min_dwell:
                # Short segment — merge with neighbor that has highest mean posterior
                current_state = path[s]
                # Check left and right neighbors
                if i > 0:
                    left_state = path[seg_starts[i] - 1]
                    left_score = gamma[s:e, left_state].mean()
                else:
                    left_state = current_state
                    left_score = -np.inf
                if i < len(seg_starts) - 2:
                    right_state = path[seg_starts[i+1]]
                    right_score = gamma[s:e, right_state].mean()
                else:
                    right_state = current_state
                    right_score = -np.inf

                best = left_state if left_score >= right_score else right_state
                if best != current_state:
                    path[s:e] = best
                    changed = True

        if not changed:
            break

    return path


def analyze_session(name, gamma, path, t_common):
    """Compute per-session metrics from aligned state path."""
    segments = load_conditions(name)
    T = len(path)

    # Dwell stats
    n_trans = int(np.sum(path[1:] != path[:-1]))
    dwells = []
    run = 1
    for i in range(1, T):
        if path[i] == path[i-1]:
            run += 1
        else:
            dwells.append(run / FS_OUT)
            run = 1
    dwells.append(run / FS_OUT)

    # Per-condition usage
    condition_usage = {}
    if segments:
        conv_mask = np.zeros(T, dtype=bool)
        base_mask = np.zeros(T, dtype=bool)
        med_mask = np.zeros(T, dtype=bool)
        gap_mask = np.ones(T, dtype=bool)

        for sname, t0, t1 in segments:
            seg = (t_common >= t0) & (t_common <= t1)
            gap_mask[seg] = False
            if 'conv' in sname:
                conv_mask |= seg
            elif 'base' in sname:
                base_mask |= seg
            elif 'meditate' in sname:
                med_mask |= seg

        for pn, pm in [('conversation', conv_mask), ('baseline', base_mask),
                       ('meditation', med_mask), ('gaps', gap_mask)]:
            if pm.sum() >= 5:
                condition_usage[pn] = [float((path[pm] == k).mean()) for k in range(K)]

    return {
        'n_transitions': n_trans,
        'mean_dwell_s': float(np.mean(dwells)),
        'median_dwell_s': float(np.median(dwells)),
        'usage': [float((path == k).mean()) for k in range(K)],
        'condition_usage': condition_usage,
    }


def main():
    print("=" * 70)
    print("  Hierarchical rSLDS + Constrained Viterbi (V8, 8 sessions)")
    print("=" * 70)
    t_wall = time.time()

    # Load all V8 sessions
    all_sessions = load_all_v8()
    print(f"\n  Loaded {len(all_sessions)} sessions:")
    for Y, U, mask, t, name in all_sessions:
        print(f"    {name}: {Y.shape[0]} pts, {Y.shape[0]/FS_OUT:.0f}s")

    # Prepare session tuples for hierarchical fit
    session_tuples = [(Y, U, mask) for Y, U, mask, t, name in all_sessions]

    # Fit hierarchical rSLDS
    print(f"\n  Fitting hierarchical rSLDS (K={K}, D_latent={D_LATENT}, recurrent=True)...")
    cfg = IOHMMConfig(
        K=K, D_obs=len(MODALITY_KEYS), D_input=2, D_latent=D_LATENT,
        n_factors=2, recurrent=True,
        n_restarts=2, max_em_iter=80,
        sticky_strength=3.0,  # moderate sticky prior
    )

    result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)

    # Extract shared emission profiles (d_emit = between-state means)
    shared = result['shared']
    print(f"\n  Hierarchical BIC: {result['bic']:.0f}")
    print(f"  Total LL: {result['ll_trace'][-1]:.0f}")

    # Per-session: extract gamma, apply constrained Viterbi
    print(f"\n  Applying constrained Viterbi (min_dwell={MIN_DWELL}, {MIN_DWELL/FS_OUT:.0f}s)...")

    session_results = []
    for i, (Y, U, mask, t, name) in enumerate(all_sessions):
        sess = result['sessions'][i]
        gamma = sess['gamma']

        # Constrained Viterbi from posteriors
        path = constrained_viterbi_from_gamma(gamma, min_dwell=MIN_DWELL)

        # Analyze
        metrics = analyze_session(name, gamma, path, t)
        metrics['session'] = name

        # Per-state emission means from data (not model)
        state_means = {}
        for k in range(K):
            m = path == k
            if m.sum() > 0:
                state_means[k] = {mod: float(Y[m, d].mean())
                                  for d, mod in enumerate(MODALITY_KEYS)}
                state_means[k]['usage'] = float(m.mean())
        metrics['state_means'] = state_means

        session_results.append(metrics)

    # ── Per-session summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PER-SESSION SUMMARY (Hierarchical rSLDS)")
    print(f"{'='*70}")
    print(f"  {'Session':>16s} | Trans | Dwell_s | " + ' '.join(f'  S{k}  ' for k in range(K)))
    print(f"  " + "-" * 65)
    for r in session_results:
        usage_str = ' '.join(f'{u:6.0%}' for u in r['usage'])
        print(f"  {r['session']:>16s} | {r['n_transitions']:5d} | {r['mean_dwell_s']:7.1f} | {usage_str}")

    # ── Cross-session condition usage ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATE USAGE BY CONDITION (aligned across sessions)")
    print(f"{'='*70}")
    for period in ['conversation', 'baseline', 'meditation', 'gaps']:
        usages = []
        for r in session_results:
            cu = r['condition_usage'].get(period)
            if cu:
                usages.append(cu)
        if usages:
            usages = np.array(usages)
            mean_u = usages.mean(axis=0)
            std_u = usages.std(axis=0)
            print(f"  {period:>14s} (n={len(usages)}): "
                  + '  '.join(f'S{k}={mean_u[k]:.0%}+/-{std_u[k]:.0%}' for k in range(K)))

    # ── Aligned state profiles ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ALIGNED STATE EMISSION PROFILES")
    print(f"{'='*70}")
    for k in range(K):
        means_list = []
        usages = []
        for r in session_results:
            sm = r['state_means'].get(k)
            if sm and sm.get('usage', 0) > 0.01:
                means_list.append([sm[mod] for mod in MODALITY_KEYS])
                usages.append(sm['usage'])
        if means_list:
            means = np.array(means_list)
            mean_profile = means.mean(axis=0)
            std_profile = means.std(axis=0)
            mean_usage = np.mean(usages)
            print(f"\n  S{k} (usage={mean_usage:.0%}, n={len(means_list)} sessions):")
            for d, mod in enumerate(MODALITY_KEYS):
                print(f"    {mod:>12s}: {mean_profile[d]:+.4f} +/- {std_profile[d]:.4f}")

    # ── Gap detection ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GAP DETECTION (aligned states)")
    print(f"{'='*70}")
    for r in session_results:
        cu = r['condition_usage'].get('gaps')
        if cu:
            dom = np.argmax(cu)
            print(f"  {r['session']:>16s}: gap dominant = S{dom} ({cu[dom]:.0%}), "
                  f"all: {' '.join(f'S{k}={v:.0%}' for k, v in enumerate(cu))}")

    # Count consistency
    gap_doms = []
    for r in session_results:
        cu = r['condition_usage'].get('gaps')
        if cu:
            gap_doms.append(np.argmax(cu))
    if gap_doms:
        from collections import Counter
        c = Counter(gap_doms)
        top = c.most_common(1)[0]
        print(f"\n  Gap state consistency: S{top[0]} in {top[1]}/{len(gap_doms)} sessions ({top[1]/len(gap_doms):.0%})")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = 'results/rslds'
    os.makedirs(out_dir, exist_ok=True)

    save_results = {
        'model': 'hierarchical_rslds',
        'K': K,
        'D_latent': D_LATENT,
        'min_dwell': MIN_DWELL,
        'sticky_strength': cfg.sticky_strength,
        'bic': float(result['bic']),
        'sessions': session_results,
    }
    with open(os.path.join(out_dir, 'hierarchical_v8_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved {out_dir}/hierarchical_v8_results.json")


if __name__ == '__main__':
    main()
