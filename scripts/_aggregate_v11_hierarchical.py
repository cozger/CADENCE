"""V11 Hierarchical rSLDS: Grand average aggregation across sessions.

Reads v11_hierarchical_results.json and produces:
  1. Per-condition state usage (mean ± std across sessions)
  2. Protocol comparison (meditation vs PE)
  3. Grand average bar plots
  4. Summary JSON

Usage:
    python scripts/_aggregate_v11_hierarchical.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Auto-detect protocol from segment markers ────────────────────────────
def detect_protocol(session_data):
    """Determine protocol from segment names in results."""
    seg_names = {p[0] for p in session_data.get('periods', [])}
    if seg_names & {'meditate_B', 'meditate_K'}:
        return 'Meditation'
    elif seg_names & {'PE', 'PE_1', 'PE_2'}:
        return 'PE'
    else:
        return 'Unknown'  # baselines only, truncated session

# Condition grouping for aggregation
CONDITION_GROUPS = {
    'Baseline EO': ['base_EO'],
    'Baseline EC': ['base_EC', 'baseline'],
    'Conversation 1': ['conv_1'],
    'Conversation 2': ['conv_2'],
    'Meditation B': ['meditate_B'],
    'Meditation K': ['meditate_K'],
    'PE 1': ['PE_1', 'PE'],
    'PE 2': ['PE_2'],
}

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']


def load_results(results_path):
    """Load hierarchical results JSON."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_by_condition(data):
    """Aggregate state usage by condition across sessions."""
    state_labels = data['state_labels']
    K = len(state_labels)

    # Collect per-condition usage across sessions
    condition_usage = {}  # group_name -> list of [K] usage arrays
    condition_sessions = {}  # group_name -> list of session names

    for sess in data['sessions']:
        name = sess['session']
        for period_name, duration, usage in sess['periods']:
            # Find which group this period belongs to
            for group, aliases in CONDITION_GROUPS.items():
                if period_name in aliases:
                    if group not in condition_usage:
                        condition_usage[group] = []
                        condition_sessions[group] = []
                    condition_usage[group].append(usage)
                    condition_sessions[group].append(name)
                    break

    # Compute statistics
    stats = {}
    for group in CONDITION_GROUPS:
        if group not in condition_usage:
            continue
        arr = np.array(condition_usage[group])
        stats[group] = {
            'n': len(arr),
            'mean': arr.mean(axis=0).tolist(),
            'std': arr.std(axis=0, ddof=1).tolist() if len(arr) > 1 else [0.0] * K,
            'sessions': condition_sessions[group],
        }

    return stats, state_labels


def aggregate_by_protocol(data):
    """Split sessions by protocol and compare."""
    state_labels = data['state_labels']
    K = len(state_labels)

    protocols = {'Meditation': [], 'PE': []}
    for sess in data['sessions']:
        proto = detect_protocol(sess)
        if proto in protocols:
            protocols[proto].append(sess['usage'])
        else:
            print(f"  WARNING: {sess['session']} has unknown protocol (baselines only?)")

    stats = {}
    for proto, usages in protocols.items():
        if not usages:
            continue
        arr = np.array(usages)
        stats[proto] = {
            'n': len(arr),
            'mean': arr.mean(axis=0).tolist(),
            'std': arr.std(axis=0, ddof=1).tolist() if len(arr) > 1 else [0.0] * K,
        }

    return stats


def plot_condition_usage(stats, state_labels, out_path):
    """Bar chart: state usage by condition (mean ± SEM)."""
    groups = [g for g in CONDITION_GROUPS if g in stats]
    K = len(state_labels)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(groups))
    width = 0.18

    for k in range(K):
        means = [stats[g]['mean'][k] for g in groups]
        sems = [stats[g]['std'][k] / np.sqrt(stats[g]['n'])
                if stats[g]['n'] > 1 else 0 for g in groups]
        ax.bar(x + k * width, means, width, yerr=sems,
               label=f'S{k}({state_labels[k]})',
               color=STATE_COLORS[k], alpha=0.85, capsize=3)

    ax.set_xticks(x + width * (K - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('State Usage (fraction)', fontsize=11)
    ax.set_title('V11 Grand Average: State Usage by Condition', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=9, ncol=K)
    ax.set_ylim(0, 1)

    # Add n labels
    for i, g in enumerate(groups):
        ax.text(i + width * (K - 1) / 2, -0.06, f'n={stats[g]["n"]}',
                ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_protocol_comparison(protocol_stats, state_labels, out_path):
    """Side-by-side protocol comparison."""
    K = len(state_labels)
    protocols = list(protocol_stats.keys())
    if len(protocols) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(K)
    width = 0.3

    for pi, proto in enumerate(protocols):
        means = protocol_stats[proto]['mean']
        sems = [protocol_stats[proto]['std'][k] / np.sqrt(protocol_stats[proto]['n'])
                if protocol_stats[proto]['n'] > 1 else 0 for k in range(K)]
        ax.bar(x + pi * width, means, width, yerr=sems,
               label=f'{proto} (n={protocol_stats[proto]["n"]})',
               alpha=0.85, capsize=4)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'S{k}({state_labels[k]})' for k in range(K)], fontsize=10)
    ax.set_ylabel('Overall State Usage', fontsize=11)
    ax.set_title('V11: Meditation vs PE Protocol', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_session_overview(data, state_labels, out_path):
    """Stacked bar: per-session state usage."""
    sessions = data['sessions']
    K = len(state_labels)

    fig, ax = plt.subplots(figsize=(16, 5))
    names = [s['session'] for s in sessions]
    x = np.arange(len(names))

    bottom = np.zeros(len(names))
    for k in range(K):
        vals = [s['usage'][k] for s in sessions]
        ax.bar(x, vals, bottom=bottom, label=f'S{k}({state_labels[k]})',
               color=STATE_COLORS[k], alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('State Usage', fontsize=11)
    ax.set_title('V11: Per-Session State Usage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=K, loc='upper right')
    ax.set_ylim(0, 1.05)

    # Mark protocol
    for i, s in enumerate(sessions):
        proto = detect_protocol(s)
        label = 'M' if proto == 'Meditation' else ('PE' if proto == 'PE' else '?')
        ax.text(i, -0.05, label, ha='center', fontsize=7, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transition_rates(data, out_path):
    """Transitions per minute by session."""
    fig, ax = plt.subplots(figsize=(12, 4))
    names = [s['session'] for s in data['sessions']]
    # Estimate duration from total periods
    rates = []
    for s in data['sessions']:
        total_dur = sum(p[1] for p in s['periods'])
        rate = s['n_transitions'] / (total_dur / 60) if total_dur > 0 else 0
        rates.append(rate)

    colors = ['#1565C0' if detect_protocol(s) == 'Meditation'
              else '#E65100' for s in data['sessions']]
    ax.bar(range(len(names)), rates, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Transitions / min', fontsize=11)
    ax.set_title('V11: State Transition Rates', fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1565C0', label='Meditation'),
                       Patch(color='#E65100', label='PE')], fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    results_path = 'results/v11/hierarchical/v11_hierarchical_results.json'
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run _run_v11_hierarchical.py first.")
        return

    data = load_results(results_path)
    out_dir = 'results/v11/hierarchical'
    state_labels = data['state_labels']

    print("=" * 70)
    print("  V11 Grand Average Aggregation")
    print("=" * 70)
    print(f"\n  Sessions: {data['n_sessions']}")
    print(f"  BIC: {data['bic']:.0f}")
    print(f"  States: {state_labels}")

    # ── 1. Per-condition aggregation ──────────────────────────────────────
    stats, _ = aggregate_by_condition(data)
    print(f"\n  Per-Condition State Usage (mean ± std):")
    print(f"  {'Condition':<18s} {'n':>3s}  ", end='')
    for k, sl in enumerate(state_labels):
        print(f"  S{k}({sl:>6s})", end='')
    print()
    print("  " + "-" * 80)

    for group in CONDITION_GROUPS:
        if group not in stats:
            continue
        s = stats[group]
        print(f"  {group:<18s} {s['n']:>3d}  ", end='')
        for k in range(len(state_labels)):
            print(f"  {s['mean'][k]:.3f}±{s['std'][k]:.3f}", end='')
        print()

    # ── 2. Protocol comparison ────────────────────────────────────────────
    protocol_stats = aggregate_by_protocol(data)
    print(f"\n  Protocol Comparison (overall session usage):")
    for proto, ps in protocol_stats.items():
        print(f"    {proto} (n={ps['n']}): ", end='')
        for k, sl in enumerate(state_labels):
            print(f"  {sl}={ps['mean'][k]:.3f}±{ps['std'][k]:.3f}", end='')
        print()

    # ── 3. Per-session summary ────────────────────────────────────────────
    print(f"\n  Per-Session Summary:")
    print(f"  {'Session':<20s} {'Trans':>5s}  ", end='')
    for sl in state_labels:
        print(f"  {sl:>7s}", end='')
    print(f"  {'Protocol':>10s}")
    print("  " + "-" * 85)

    for sess in data['sessions']:
        name = sess['session']
        proto = detect_protocol(sess)
        proto = 'Med' if proto == 'Meditation' else ('PE' if proto == 'PE' else '?')
        print(f"  {name:<20s} {sess['n_transitions']:>5d}  ", end='')
        for k in range(len(state_labels)):
            print(f"  {sess['usage'][k]:>7.3f}", end='')
        print(f"  {proto:>10s}")

    # ── 4. Covariate effects ──────────────────────────────────────────────
    if 'covariate_keys' in data:
        print(f"\n  Covariate keys: {data['covariate_keys']}")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print(f"\n  Generating plots...")
    plot_condition_usage(stats, state_labels,
                         os.path.join(out_dir, 'v11_grand_condition_usage.png'))
    plot_protocol_comparison(protocol_stats, state_labels,
                              os.path.join(out_dir, 'v11_grand_protocol_comparison.png'))
    plot_session_overview(data, state_labels,
                          os.path.join(out_dir, 'v11_grand_session_overview.png'))
    plot_transition_rates(data,
                          os.path.join(out_dir, 'v11_grand_transition_rates.png'))

    # ── 6. Save aggregate JSON ────────────────────────────────────────────
    aggregate = {
        'version': 'v11',
        'n_sessions': data['n_sessions'],
        'bic': data['bic'],
        'state_labels': state_labels,
        'condition_stats': {
            g: {k: v for k, v in s.items() if k != 'sessions'}
            for g, s in stats.items()
        },
        'protocol_stats': protocol_stats,
    }
    agg_path = os.path.join(out_dir, 'v11_grand_average_summary.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Saved: {agg_path}")

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
