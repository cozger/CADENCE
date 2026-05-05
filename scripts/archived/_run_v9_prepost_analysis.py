"""V9 Pre/Post Intervention Analysis — Mixed-Effects Design.

Tests whether meditation vs PE differentially changes graph metrics
from conv_1 (pre-intervention) to conv_2 (post-intervention).

Design:
    DV: flexibility, λ₂, n_edges, cross-modal edges
    Within-subject: time (conv_1 vs conv_2)
    Between-subject: protocol (meditation vs PE)
    Key test: protocol × time interaction

Usage:
    python scripts/_run_v9_prepost_analysis.py
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, wilcoxon, permutation_test
from itertools import combinations

# ── Load all session results ──────────────────────────────────────

def load_all_v9_results():
    """Load per-condition graph metrics from all V9 session results."""
    rows = []
    for f in sorted(glob.glob('results/v9/*/v9_results.json')):
        sname = os.path.basename(os.path.dirname(f))
        data = json.load(open(f))
        app3 = data.get('app3_modality_graph', {})
        tv = app3.get('per_condition_timevarying', {})
        edges = app3.get('per_condition_edges', {})
        protocol = app3.get('protocol', data.get('protocol', 'unknown'))

        if not tv:
            continue

        for cond_name, metrics in tv.items():
            row = {
                'session': sname,
                'protocol': protocol,
                'condition': cond_name,
                'flexibility': metrics.get('flex_mean', np.nan),
                'lambda2': metrics.get('lambda2_mean', np.nan),
                'n_components': metrics.get('n_components_mean', np.nan),
                'n_edges': metrics.get('n_edges_mean', np.nan),
            }
            if cond_name in edges:
                row['cross_modal'] = edges[cond_name].get('cross_modal', np.nan)
                row['within_modal'] = edges[cond_name].get('within_modal', np.nan)
            else:
                row['cross_modal'] = np.nan
                row['within_modal'] = np.nan
            rows.append(row)

    return rows


def get_paired_data(rows, cond_pre='conv_1', cond_post='conv_2'):
    """Extract sessions that have BOTH pre and post condition data."""
    # Group by session
    by_session = {}
    for r in rows:
        s = r['session']
        if s not in by_session:
            by_session[s] = {}
        by_session[s][r['condition']] = r

    paired = []
    for sname, conds in by_session.items():
        if cond_pre in conds and cond_post in conds:
            pre = conds[cond_pre]
            post = conds[cond_post]
            paired.append({
                'session': sname,
                'protocol': pre['protocol'],
                'pre': pre,
                'post': post,
            })
    return paired


def interaction_permutation_test(deltas_a, deltas_b, n_perms=10000, seed=42):
    """Permutation test for interaction: is mean(deltas_a) != mean(deltas_b)?"""
    all_deltas = np.concatenate([deltas_a, deltas_b])
    n_a = len(deltas_a)
    observed = np.mean(deltas_a) - np.mean(deltas_b)

    rng = np.random.default_rng(seed)
    # Exhaustive if small enough
    n_total = len(all_deltas)
    if n_total <= 15:
        combos = list(combinations(range(n_total), n_a))
        null_diffs = np.array([
            all_deltas[list(c)].mean() - all_deltas[[i for i in range(n_total) if i not in c]].mean()
            for c in combos
        ])
        n_used = len(combos)
    else:
        null_diffs = np.empty(n_perms)
        for i in range(n_perms):
            perm = rng.permutation(n_total)
            null_diffs[i] = all_deltas[perm[:n_a]].mean() - all_deltas[perm[n_a:]].mean()
        n_used = n_perms

    p_val = float((np.abs(null_diffs) >= np.abs(observed)).mean())
    return observed, null_diffs, p_val, n_used


# ── Main analysis ─────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  V9 Pre/Post Intervention Analysis")
    print("  conv_1 (pre) → intervention → conv_2 (post)")
    print("=" * 70)

    rows = load_all_v9_results()
    print(f"\n  Loaded {len(rows)} condition-level observations from "
          f"{len(set(r['session'] for r in rows))} sessions")

    paired = get_paired_data(rows, 'conv_1', 'conv_2')
    print(f"  Sessions with both conv_1 and conv_2: {len(paired)}")

    med_paired = [p for p in paired if p['protocol'] == 'meditation']
    pe_paired = [p for p in paired if p['protocol'] == 'PE']
    print(f"    Meditation: {len(med_paired)} — {[p['session'] for p in med_paired]}")
    print(f"    PE:         {len(pe_paired)} — {[p['session'] for p in pe_paired]}")

    if len(med_paired) < 2 or len(pe_paired) < 2:
        print("\n  WARNING: Need ≥2 sessions per protocol for interaction test")

    # Also get baseline → conv_2 for broader analysis
    # Use base_EO and base_EC as baseline (average if both present)
    by_session = {}
    for r in rows:
        s = r['session']
        if s not in by_session:
            by_session[s] = {}
        by_session[s][r['condition']] = r

    # Build extended dataset: baseline, conv_1, intervention, conv_2
    extended = []
    for sname, conds in by_session.items():
        protocol = list(conds.values())[0]['protocol']
        # Baseline = average of base_EO and base_EC
        base_vals = [conds[c] for c in ['base_EO', 'base_EC', 'baseline'] if c in conds]
        if base_vals:
            base_flex = np.mean([v['flexibility'] for v in base_vals])
        else:
            base_flex = np.nan
        # Intervention = meditation or PE blocks
        int_conds = [c for c in ['meditate_B', 'meditate_K', 'PE', 'PE_1', 'PE_2'] if c in conds]
        if int_conds:
            int_flex = np.mean([conds[c]['flexibility'] for c in int_conds])
        else:
            int_flex = np.nan

        extended.append({
            'session': sname,
            'protocol': protocol,
            'baseline': base_flex,
            'conv_1': conds.get('conv_1', {}).get('flexibility', np.nan),
            'intervention': int_flex,
            'conv_2': conds.get('conv_2', {}).get('flexibility', np.nan),
        })

    # ── Print extended table ──────────────────────────────────────
    metrics = ['flexibility', 'lambda2', 'n_edges', 'cross_modal']

    print(f"\n{'─' * 85}")
    print(f"  {'Session':>20s} | {'Protocol':>10s} | {'Baseline':>8s} | "
          f"{'conv_1':>8s} | {'Interv':>8s} | {'conv_2':>8s} | {'Δ(c2-c1)':>8s}")
    print(f"{'─' * 85}")
    for e in sorted(extended, key=lambda x: x['protocol']):
        delta = e['conv_2'] - e['conv_1'] if np.isfinite(e['conv_1']) and np.isfinite(e['conv_2']) else np.nan
        print(f"  {e['session']:>20s} | {e['protocol']:>10s} | "
              f"{e['baseline']:8.3f} | {e['conv_1']:8.3f} | "
              f"{e['intervention']:8.3f} | {e['conv_2']:8.3f} | "
              f"{delta:+8.3f}" if np.isfinite(delta) else
              f"  {e['session']:>20s} | {e['protocol']:>10s} | "
              f"{'—':>8s} | {'—':>8s} | {'—':>8s} | {'—':>8s} | {'—':>8s}")

    # ── Interaction tests per metric ──────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Interaction Tests: protocol × time (conv_1 → conv_2)")
    print(f"{'=' * 70}")

    out_dir = 'results/v9/prepost_analysis'
    os.makedirs(out_dir, exist_ok=True)

    interaction_results = {}
    for metric in metrics:
        med_pre = [p['pre'][metric] for p in med_paired if np.isfinite(p['pre'].get(metric, np.nan))]
        med_post = [p['post'][metric] for p in med_paired if np.isfinite(p['post'].get(metric, np.nan))]
        pe_pre = [p['pre'][metric] for p in pe_paired if np.isfinite(p['pre'].get(metric, np.nan))]
        pe_post = [p['post'][metric] for p in pe_paired if np.isfinite(p['post'].get(metric, np.nan))]

        if len(med_pre) < 2 or len(pe_pre) < 2:
            print(f"\n  {metric}: insufficient data (med={len(med_pre)}, PE={len(pe_pre)})")
            continue

        # Compute deltas
        med_deltas = np.array([p['post'][metric] - p['pre'][metric] for p in med_paired
                               if np.isfinite(p['pre'].get(metric, np.nan))
                               and np.isfinite(p['post'].get(metric, np.nan))])
        pe_deltas = np.array([p['post'][metric] - p['pre'][metric] for p in pe_paired
                              if np.isfinite(p['pre'].get(metric, np.nan))
                              and np.isfinite(p['post'].get(metric, np.nan))])

        if len(med_deltas) < 1 or len(pe_deltas) < 1:
            continue

        # Interaction permutation test
        obs_diff, null_diffs, p_val, n_used = interaction_permutation_test(
            med_deltas, pe_deltas)

        # Effect size (Cohen's d on deltas)
        pooled_std = np.sqrt((np.var(med_deltas) + np.var(pe_deltas)) / 2)
        d = obs_diff / pooled_std if pooled_std > 1e-8 else 0

        # Within-group pre-post tests
        if len(med_deltas) >= 2:
            _, p_med_within = wilcoxon(med_deltas, alternative='two-sided') if len(med_deltas) >= 5 else (0, np.nan)
        else:
            p_med_within = np.nan

        print(f"\n  {metric}")
        print(f"    Meditation Δ: {med_deltas.mean():+.4f} ± {med_deltas.std():.4f} "
              f"(n={len(med_deltas)}) {[round(d, 3) for d in med_deltas]}")
        print(f"    PE Δ:         {pe_deltas.mean():+.4f} ± {pe_deltas.std():.4f} "
              f"(n={len(pe_deltas)}) {[round(d, 3) for d in pe_deltas]}")
        print(f"    Interaction:  diff={obs_diff:+.4f}, d={d:+.2f}, "
              f"p={p_val:.4f} (perm, {n_used} arrangements)")
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"    {sig}")

        interaction_results[metric] = {
            'med_deltas': med_deltas.tolist(),
            'pe_deltas': pe_deltas.tolist(),
            'med_delta_mean': float(med_deltas.mean()),
            'pe_delta_mean': float(pe_deltas.mean()),
            'interaction_diff': float(obs_diff),
            'cohens_d': float(d),
            'p_value': float(p_val),
            'n_perms': n_used,
            'n_med': len(med_deltas),
            'n_pe': len(pe_deltas),
        }

    # ── Visualization ─────────────────────────────────────────────

    # Plot 1: Pre-post spaghetti plot for each metric
    n_metrics = len([m for m in metrics if m in interaction_results])
    if n_metrics == 0:
        print("\n  No metrics to plot")
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    mi = 0
    for metric in metrics:
        if metric not in interaction_results:
            continue
        ax = axes[mi]
        ir = interaction_results[metric]

        # Plot individual session lines
        for p in med_paired:
            pre_v = p['pre'].get(metric, np.nan)
            post_v = p['post'].get(metric, np.nan)
            if np.isfinite(pre_v) and np.isfinite(post_v):
                ax.plot([0, 1], [pre_v, post_v], 'o-', color='#9C27B0',
                        alpha=0.6, linewidth=1.5, markersize=8)
                ax.annotate(p['session'], (1.05, post_v), fontsize=6, alpha=0.5)

        for p in pe_paired:
            pre_v = p['pre'].get(metric, np.nan)
            post_v = p['post'].get(metric, np.nan)
            if np.isfinite(pre_v) and np.isfinite(post_v):
                ax.plot([0, 1], [pre_v, post_v], 's-', color='#E53935',
                        alpha=0.6, linewidth=1.5, markersize=8)
                ax.annotate(p['session'], (1.05, post_v), fontsize=6, alpha=0.5)

        # Group means
        med_pre_vals = [p['pre'][metric] for p in med_paired
                        if np.isfinite(p['pre'].get(metric, np.nan))]
        med_post_vals = [p['post'][metric] for p in med_paired
                         if np.isfinite(p['post'].get(metric, np.nan))]
        pe_pre_vals = [p['pre'][metric] for p in pe_paired
                       if np.isfinite(p['pre'].get(metric, np.nan))]
        pe_post_vals = [p['post'][metric] for p in pe_paired
                        if np.isfinite(p['post'].get(metric, np.nan))]

        if med_pre_vals and med_post_vals:
            ax.plot([0, 1], [np.mean(med_pre_vals), np.mean(med_post_vals)],
                    'o-', color='#9C27B0', linewidth=3, markersize=12,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        if pe_pre_vals and pe_post_vals:
            ax.plot([0, 1], [np.mean(pe_pre_vals), np.mean(pe_post_vals)],
                    's-', color='#E53935', linewidth=3, markersize=12,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['conv_1\n(pre-intervention)', 'conv_2\n(post-intervention)'],
                           fontsize=10)
        ax.set_xlim(-0.3, 1.6)
        metric_label = {'flexibility': 'Coupling Flexibility',
                        'lambda2': 'λ₂ (Algebraic Connectivity)',
                        'n_edges': '# Edges',
                        'cross_modal': '# Cross-Modal Edges'}.get(metric, metric)
        ax.set_ylabel(metric_label, fontsize=11)

        p_str = f"p={ir['p_value']:.3f}" if ir['p_value'] >= 0.001 else f"p<.001"
        sig = ' *' if ir['p_value'] < 0.05 else ''
        ax.set_title(f"{metric_label}\ninteraction d={ir['cohens_d']:+.2f}, {p_str}{sig}",
                     fontsize=11)
        mi += 1

    # Legend
    axes[0].legend(handles=[
        plt.Line2D([0], [0], color='#9C27B0', marker='o', linewidth=2,
                   label=f'Meditation (n={len(med_paired)})'),
        plt.Line2D([0], [0], color='#E53935', marker='s', linewidth=2,
                   label=f'PE (n={len(pe_paired)})'),
    ], loc='upper right', fontsize=9)

    fig.suptitle('Pre/Post Intervention: conv_1 → [meditation/PE] → conv_2',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_prepost_interaction.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {out_dir}/v9_prepost_interaction.png")

    # Plot 2: Permutation null distribution for flexibility interaction
    if 'flexibility' in interaction_results:
        ir = interaction_results['flexibility']
        fig, ax = plt.subplots(figsize=(8, 5))
        _, null_diffs, _, _ = interaction_permutation_test(
            np.array(ir['med_deltas']), np.array(ir['pe_deltas']))
        ax.hist(null_diffs, bins=30, color='#90CAF9', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Null distribution')
        ax.axvline(ir['interaction_diff'], color='red', linewidth=2.5,
                   linestyle='--', label=f"Observed: {ir['interaction_diff']:+.3f}")
        ax.set_xlabel('Interaction effect (Δ meditation − Δ PE)', fontsize=12)
        ax.set_ylabel('Count')
        ax.set_title(f"Protocol × Time Interaction (Flexibility)\n"
                     f"p={ir['p_value']:.4f}, d={ir['cohens_d']:+.2f}, "
                     f"n_med={ir['n_med']}, n_PE={ir['n_pe']}",
                     fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'v9_prepost_permutation_null.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir}/v9_prepost_permutation_null.png")

    # Plot 3: Full session trajectory (baseline → conv_1 → intervention → conv_2)
    fig, ax = plt.subplots(figsize=(10, 6))
    timepoints = ['baseline', 'conv_1', 'intervention', 'conv_2']
    x_pos = [0, 1, 2, 3]

    for e in extended:
        vals = [e.get(t, np.nan) for t in timepoints]
        if sum(np.isfinite(v) for v in vals) < 3:
            continue
        color = '#9C27B0' if e['protocol'] == 'meditation' else (
            '#E53935' if e['protocol'] == 'PE' else '#BDBDBD')
        marker = 'o' if e['protocol'] == 'meditation' else 's'
        # Connect only finite values
        valid_x = [x for x, v in zip(x_pos, vals) if np.isfinite(v)]
        valid_v = [v for v in vals if np.isfinite(v)]
        ax.plot(valid_x, valid_v, marker=marker, color=color,
                alpha=0.4, linewidth=1, markersize=6)
        ax.annotate(e['session'], (valid_x[-1] + 0.1, valid_v[-1]),
                    fontsize=5, alpha=0.4)

    # Group means
    for protocol, color, marker in [('meditation', '#9C27B0', 'o'), ('PE', '#E53935', 's')]:
        group = [e for e in extended if e['protocol'] == protocol]
        if len(group) < 2:
            continue
        means = []
        valid_x = []
        for ti, t in enumerate(timepoints):
            vals = [e[t] for e in group if np.isfinite(e.get(t, np.nan))]
            if vals:
                means.append(np.mean(vals))
                valid_x.append(x_pos[ti])
        ax.plot(valid_x, means, marker=marker, color=color,
                linewidth=3, markersize=14, markeredgecolor='black',
                markeredgewidth=1.5, zorder=5, label=protocol.capitalize())

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Baseline', 'Conv 1\n(pre)', 'Intervention\n(med/PE)', 'Conv 2\n(post)'],
                       fontsize=11)
    ax.set_ylabel('Coupling Flexibility', fontsize=12)
    ax.set_title('Full Session Trajectory: Flexibility Across Conditions', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 4)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_prepost_trajectory.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir}/v9_prepost_trajectory.png")

    # Save results
    results = {
        'n_sessions_paired': len(paired),
        'n_meditation': len(med_paired),
        'n_pe': len(pe_paired),
        'interaction_tests': interaction_results,
        'extended_trajectories': extended,
    }
    json_path = os.path.join(out_dir, 'v9_prepost_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {json_path}")


if __name__ == '__main__':
    main()
