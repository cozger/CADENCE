"""V10 Coupling Excess Analysis.

Surrogate-calibrated coupling detection for all sessions. Replaces
session-adaptive burst rates (invalid) with continuous coupling excess
z-scores against circular-shift null.

Usage:
    python scripts/run_coupling_excess_analysis.py            # All sessions
    python scripts/run_coupling_excess_analysis.py --session y_06  # Single
"""

import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import mannwhitneyu
from cadence.significance.coupling_bursts import (
    analyze_session, COUPLING_GROUPS, BIDIRECTIONAL_GROUPS,
)
from cadence.constants import V10_MODALITY_KEYS

OUT_DIR = 'results/v10/coupling_excess'
CONDS = ['base_EO', 'base_EC', 'conv_1', 'conv_2',
         'meditate_B', 'meditate_K', 'PE_1', 'PE_2']


def main(session_name=None):
    print("=" * 80)
    print("  V10 Coupling Excess Analysis")
    print("  (Surrogate-calibrated, replaces session-adaptive burst rates)")
    print("=" * 80)

    t_wall = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find sessions
    if session_name:
        npzs = [f'results/v10/{session_name}/scaffold_v10_ztimecourses.npz']
    else:
        npzs = sorted(glob.glob('results/v10/*/scaffold_v10_ztimecourses.npz'))

    # Determine session IDs (exclude hierarchical dir)
    session_ids = []
    for p in npzs:
        sid = os.path.basename(os.path.dirname(p))
        if sid != 'hierarchical' and os.path.exists(p):
            session_ids.append(sid)

    print(f"\n  {len(session_ids)} sessions to analyze")

    # Analyze each session
    all_results = []
    for sid in session_ids:
        print(f"\n  {sid}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = analyze_session(sid)
            all_results.append(result)
            n_events = sum(v['n_excitation'] + v['n_suppression']
                           for v in result['event_summary'].values())
            print(f"{time.time()-t0:.1f}s, {n_events} coupling events")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_results:
        print("No results.")
        return

    # ── Per-condition coupling excess (cross-session) ────────────────
    print(f"\n{'='*80}")
    print(f"  Per-Condition Coupling Excess (mean z across sessions)")
    print(f"{'='*80}")

    # Determine protocol per session
    for r in all_results:
        conds = list(r['per_condition_excess'].keys())
        if any('meditate' in c for c in conds):
            r['protocol'] = 'med'
        elif any('PE' in c for c in conds):
            r['protocol'] = 'pe'
        else:
            r['protocol'] = '?'

    # Print per-condition excess
    tier1_groups = [g for g, info in COUPLING_GROUPS.items() if info['tier'] == 1]
    print(f"\n  Tier 1 (coupling-specific):")
    print(f"  {'Condition':>12s} |", end='')
    for g in tier1_groups:
        print(f" {g:>12s}", end='')
    print(f" | n")
    print(f"  " + "-" * (15 + 13 * len(tier1_groups) + 5))

    cond_excess_agg = {}
    for cond in CONDS:
        vals = {g: [] for g in tier1_groups}
        for r in all_results:
            if cond in r['per_condition_excess']:
                for g in tier1_groups:
                    if g in r['per_condition_excess'][cond]:
                        vals[g].append(r['per_condition_excess'][cond][g])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 2:
            continue

        cond_excess_agg[cond] = vals
        print(f"  {cond:>12s} |", end='')
        for g in tier1_groups:
            if vals[g]:
                m = np.mean(vals[g])
                print(f" {m:>+12.3f}", end='')
            else:
                print(f" {'':>12s}", end='')
        print(f" | {n}")

    # ── Med vs PE comparison ─────────────────────────────────────────
    print(f"\n  Meditation vs PE protocol comparison:")
    for g in tier1_groups:
        conv1_med = [r['per_condition_excess'].get('conv_1', {}).get(g)
                     for r in all_results if r['protocol'] == 'med'
                     and 'conv_1' in r['per_condition_excess']
                     and g in r['per_condition_excess'].get('conv_1', {})]
        conv1_pe = [r['per_condition_excess'].get('conv_1', {}).get(g)
                    for r in all_results if r['protocol'] == 'pe'
                    and 'conv_1' in r['per_condition_excess']
                    and g in r['per_condition_excess'].get('conv_1', {})]
        conv1_med = [v for v in conv1_med if v is not None]
        conv1_pe = [v for v in conv1_pe if v is not None]
        if len(conv1_med) >= 2 and len(conv1_pe) >= 2:
            U, p = mannwhitneyu(conv1_med, conv1_pe, alternative='two-sided')
            print(f"    {g}: conv_1 med={np.mean(conv1_med):+.3f} pe={np.mean(conv1_pe):+.3f} p={p:.3f}")

    # ── Event rates ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Coupling Event Rates (events/min, z > 2.0)")
    print(f"{'='*80}")

    for g in COUPLING_GROUPS:
        rates = [r['event_summary'][g]['rate_excitation'] for r in all_results
                 if g in r['event_summary']]
        sup_rates = [r['event_summary'][g]['rate_suppression'] for r in all_results
                     if g in r['event_summary']]
        if rates:
            print(f"  {g:>14s}: excitation={np.mean(rates):.2f}±{np.std(rates)/np.sqrt(len(rates)):.2f}/min", end='')
            if any(s > 0 for s in sup_rates):
                print(f"  suppression={np.mean(sup_rates):.2f}/min", end='')
            print()

    # ── ECA Coincidence ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Cross-Modal Coincidence (ECA, tau=5s)")
    print(f"{'='*80}")

    # Aggregate coincidence matrices
    all_trigger = []
    all_p = []
    group_names = None
    for r in all_results:
        if 'coincidence' in r and r['coincidence']['trigger']:
            all_trigger.append(np.array(r['coincidence']['trigger']))
            all_p.append(np.array(r['coincidence']['p_trigger']))
            group_names = r['coincidence']['group_names']

    if all_trigger and group_names:
        mean_trigger = np.mean(all_trigger, axis=0)
        mean_p = np.mean(all_p, axis=0)  # harmonic mean would be better, but this is informative
        n_sig = np.sum(np.array(all_p) < 0.05, axis=0)

        print(f"\n  Mean trigger rate (fraction of A events coinciding with B):")
        print(f"  {'':>14s} |", end='')
        for g in group_names:
            print(f" {g[:10]:>10s}", end='')
        print()
        for i, ga in enumerate(group_names):
            print(f"  {ga:>14s} |", end='')
            for j, gb in enumerate(group_names):
                if i == j:
                    print(f" {'---':>10s}", end='')
                else:
                    sig = '*' if n_sig[i, j] > len(all_trigger) // 2 else ' '
                    print(f" {mean_trigger[i,j]:>9.2f}{sig}", end='')
            print()
        print(f"  (* = significant in >50% of sessions)")

    # ── Asymmetry during coupling events ─────────────────────────────
    print(f"\n{'='*80}")
    print(f"  EEG Asymmetry During Coupling Events (positive = therapist higher)")
    print(f"{'='*80}")

    print(f"\n  {'Condition':>12s} | {'theta':>8s} {'alpha':>8s} {'beta':>8s} | n_events")
    print(f"  " + "-" * 55)
    for cond in CONDS:
        thetas, alphas, betas, ns = [], [], [], []
        for r in all_results:
            if cond in r.get('per_condition_asymmetry', {}):
                a = r['per_condition_asymmetry'][cond]
                thetas.append(a['theta']['mean'])
                alphas.append(a['alpha']['mean'])
                betas.append(a['beta']['mean'])
                ns.append(a['theta']['n'])
        if thetas:
            print(f"  {cond:>12s} | {np.mean(thetas):>+8.3f} {np.mean(alphas):>+8.3f} "
                  f"{np.mean(betas):>+8.3f} | {int(np.mean(ns))}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        'version': 'v10_coupling_excess',
        'n_sessions': len(all_results),
        'sessions': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(OUT_DIR, 'coupling_excess_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default=None)
    args = parser.parse_args()
    main(session_name=args.session)
