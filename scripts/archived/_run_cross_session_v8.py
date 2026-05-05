"""Cross-session constrained Viterbi (min_dwell=20, 10s) analysis.

Fits IOHMM K=4 + constrained Viterbi on all V8 sessions.
Reports per-condition state usage, inter-condition gap detection,
and cross-session consistency of state profiles.

Usage:
    python scripts/_run_cross_session_v8.py
"""

import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from cadence.significance.rslds_model import IOHMM, IOHMMConfig, build_observation_mask
from scripts._run_rslds_phase2 import MODALITY_KEYS, MODALITY_NAMES, FS_OUT


def process_one_session(session_name):
    """Fit IOHMM + constrained Viterbi on one session. Returns results dict."""
    npz_path = f'results/rslds/{session_name}/rslds_scaffold_v8_ztimecourses.npz'
    json_path = f'results/rslds/{session_name}/rslds_scaffold_v8_results.json'
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    t = data['t_common']
    Y = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
    T, D = Y.shape
    U = np.zeros((T, 2), dtype=np.float64)
    obs_mask = np.ones((T, D), dtype=bool)

    # Load condition segments
    segments = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            info = json.load(f)
        segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

    # Fit IOHMM
    cfg = IOHMMConfig(K=4, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
    model = IOHMM(cfg)
    params, hist = model.fit(Y, U, obs_mask, seed=42, verbose=False)

    # Constrained Viterbi (10s min dwell = 20 samples at 2Hz)
    labels = model.viterbi_min_dwell(Y, U, obs_mask, params, min_dwell=20)

    # Dwell stats
    n_trans = int(np.sum(labels[1:] != labels[:-1]))
    dwells = []
    run = 1
    for i in range(1, T):
        if labels[i] == labels[i-1]:
            run += 1
        else:
            dwells.append(run / FS_OUT)
            run = 1
    dwells.append(run / FS_OUT)

    # Per-state emission means (from data)
    state_means = {}
    for k in range(4):
        m = labels == k
        if m.sum() > 0:
            state_means[k] = {mod: float(Y[m, d].mean())
                              for d, mod in enumerate(MODALITY_KEYS)}
            state_means[k]['usage'] = float(m.mean())
        else:
            state_means[k] = {mod: 0.0 for mod in MODALITY_KEYS}
            state_means[k]['usage'] = 0.0

    # Per-condition usage + gaps
    condition_usage = {}
    if segments:
        conv_mask = np.zeros(T, dtype=bool)
        base_mask = np.zeros(T, dtype=bool)
        med_mask = np.zeros(T, dtype=bool)
        gap_mask = np.ones(T, dtype=bool)

        for name, t0, t1 in segments:
            seg = (t >= t0) & (t <= t1)
            gap_mask[seg] = False
            if 'conv' in name:
                conv_mask |= seg
            elif 'base' in name:
                base_mask |= seg
            elif 'meditate' in name:
                med_mask |= seg

        for pn, pm in [('conversation', conv_mask), ('baseline', base_mask),
                       ('meditation', med_mask), ('gaps', gap_mask)]:
            if pm.sum() >= 5:
                condition_usage[pn] = {
                    'usage': [float((labels[pm] == k).mean()) for k in range(4)],
                    'duration_s': float(pm.sum() / FS_OUT),
                }

        # Per-segment detail
        per_segment = {}
        for name, t0, t1 in segments:
            seg = (t >= t0) & (t <= t1)
            if seg.sum() >= 5:
                per_segment[name] = [float((labels[seg] == k).mean()) for k in range(4)]

        condition_usage['per_segment'] = per_segment

    # Condition alignment permutation test
    p_value = None
    if segments and len(segments) >= 3:
        usage_matrix = []
        for name, t0, t1 in segments:
            seg = (t >= t0) & (t <= t1)
            if seg.sum() >= 10:
                usage_matrix.append([float((labels[seg] == k).mean()) for k in range(4)])
        if len(usage_matrix) >= 3:
            usage_matrix = np.array(usage_matrix)
            uniform = 0.25
            chi2 = np.sum((usage_matrix - uniform) ** 2) / uniform
            rng = np.random.default_rng(42)
            null_chi2s = []
            for _ in range(200):
                perm = rng.permutation(labels)
                pu = []
                for name, t0, t1 in segments:
                    seg = (t >= t0) & (t <= t1)
                    if seg.sum() >= 10:
                        pu.append([float((perm[seg] == k).mean()) for k in range(4)])
                if pu:
                    pm = np.array(pu)
                    null_chi2s.append(np.sum((pm - uniform) ** 2) / uniform)
            p_value = float(np.mean(np.array(null_chi2s) >= chi2))

    return {
        'session': session_name,
        'T': T,
        'duration_s': float(T / FS_OUT),
        'n_transitions': n_trans,
        'mean_dwell_s': float(np.mean(dwells)),
        'median_dwell_s': float(np.median(dwells)),
        'state_means': state_means,
        'condition_usage': condition_usage,
        'cond_p_value': p_value,
        'bic': float(hist['bic']),
    }


def main():
    print("=" * 70)
    print("  Cross-Session V8 Constrained Viterbi Analysis (min_dwell=10s)")
    print("=" * 70)

    # Find all V8 sessions
    npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_v8_ztimecourses.npz'))
    sessions = [os.path.basename(os.path.dirname(p)) for p in npzs]
    print(f"\n  V8 sessions: {len(sessions)}")

    # Parallel fitting
    t0 = time.time()
    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(process_one_session)(s) for s in sessions)
    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"  Fitted {len(results)} sessions in {elapsed:.0f}s")

    # ── Per-session summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PER-SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Session':>16s} | dur_s | Trans | Dwell_s | cond_p")
    print(f"  " + "-" * 60)
    for r in results:
        p_str = f"{r['cond_p_value']:.4f}" if r['cond_p_value'] is not None else "  N/A "
        print(f"  {r['session']:>16s} | {r['duration_s']:5.0f} | {r['n_transitions']:5d} | "
              f"{r['mean_dwell_s']:7.1f} | {p_str}")

    # ── Per-condition usage across sessions ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATE USAGE BY CONDITION (across sessions)")
    print(f"{'='*70}")

    for period in ['conversation', 'baseline', 'meditation', 'gaps']:
        usages = []
        for r in results:
            cu = r['condition_usage'].get(period)
            if cu:
                usages.append(cu['usage'])
        if usages:
            usages = np.array(usages)
            mean_u = usages.mean(axis=0)
            std_u = usages.std(axis=0)
            print(f"\n  {period:>14s} (n={len(usages)}):  "
                  + '  '.join(f'S{k}={mean_u[k]:.0%}+/-{std_u[k]:.0%}' for k in range(4)))

    # ── State profile consistency ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATE EMISSION PROFILES (mean across sessions)")
    print(f"{'='*70}")

    # Collect per-state means across sessions
    for k in range(4):
        means = []
        for r in results:
            sm = r['state_means'].get(k)
            if sm and sm['usage'] > 0.01:
                means.append([sm[mod] for mod in MODALITY_KEYS])
        if means:
            means = np.array(means)
            mean_profile = means.mean(axis=0)
            std_profile = means.std(axis=0)
            usage_mean = np.mean([r['state_means'][k]['usage'] for r in results
                                  if r['state_means'].get(k, {}).get('usage', 0) > 0.01])
            print(f"\n  S{k} (usage={usage_mean:.0%}, n={len(means)} sessions):")
            for d, mod in enumerate(MODALITY_KEYS):
                bar = '+' * int(abs(mean_profile[d]) * 50)
                sign = '+' if mean_profile[d] > 0 else '-'
                print(f"    {mod:>12s}: {mean_profile[d]:+.3f} +/- {std_profile[d]:.3f} {sign}{bar}")

    # ── Gap detection consistency ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GAP DETECTION CONSISTENCY")
    print(f"{'='*70}")

    # Which state dominates gaps in each session?
    gap_dominant = []
    for r in results:
        cu = r['condition_usage'].get('gaps')
        if cu:
            dominant = np.argmax(cu['usage'])
            gap_dominant.append(dominant)
            print(f"  {r['session']:>16s}: gap dominant = S{dominant} "
                  f"({cu['usage'][dominant]:.0%})")

    if gap_dominant:
        from collections import Counter
        counts = Counter(gap_dominant)
        most_common = counts.most_common(1)[0]
        print(f"\n  Most common gap state: S{most_common[0]} "
              f"({most_common[1]}/{len(gap_dominant)} sessions = {most_common[1]/len(gap_dominant):.0%})")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = 'results/rslds/cross_session_v8_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")


if __name__ == '__main__':
    main()
