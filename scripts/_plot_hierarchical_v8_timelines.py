"""Generate timeline plots for hierarchical rSLDS + constrained Viterbi results.

Produces per-session timeline plots showing:
  - 9D coupling z-timecourses
  - Constrained Viterbi state assignments (color-coded)
  - Condition shading (baseline, conversation, meditation, gaps)
  - Cross-session aligned state labels

Usage:
    python scripts/_plot_hierarchical_v8_timelines.py
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, fit_slds, fit_hierarchical_slds,
    _align_states_to_reference, build_observation_mask,
)
from scripts._run_rslds_phase2 import MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS, FS_OUT

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
STATE_NAMES = ['S0', 'S1', 'S2', 'S3']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
MIN_DWELL = 20
K = 4


def constrained_viterbi_from_gamma(gamma, min_dwell=MIN_DWELL):
    """Merge short segments into neighbors based on posterior strength."""
    T, K = gamma.shape
    path = np.argmax(gamma, axis=1)
    if min_dwell <= 1:
        return path
    for _ in range(10):
        changed = False
        seg_starts = [0]
        for t in range(1, T):
            if path[t] != path[t-1]:
                seg_starts.append(t)
        seg_starts.append(T)
        for i in range(len(seg_starts) - 1):
            s, e = seg_starts[i], seg_starts[i+1]
            if e - s < min_dwell:
                current_state = path[s]
                left_state = path[seg_starts[i] - 1] if i > 0 else current_state
                left_score = gamma[s:e, left_state].mean() if i > 0 else -np.inf
                right_state = path[seg_starts[i+1]] if i < len(seg_starts) - 2 else current_state
                right_score = gamma[s:e, right_state].mean() if i < len(seg_starts) - 2 else -np.inf
                best = left_state if left_score >= right_score else right_state
                if best != current_state:
                    path[s:e] = best
                    changed = True
        if not changed:
            break
    return path


def load_session(session_name):
    """Load V8 data + conditions."""
    npz = f'results/rslds/{session_name}/rslds_scaffold_v8_ztimecourses.npz'
    if not os.path.exists(npz):
        return None, None, None
    data = np.load(npz)
    t = data['t_common']
    Y = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)

    json_path = f'results/rslds/{session_name}/rslds_scaffold_v8_results.json'
    segments = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            info = json.load(f)
        segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]
    return Y, t, segments


def plot_session_timeline(session_name, Y, t, segments, path, gamma, out_path):
    """Plot timeline: z-timecourses + state coloring + conditions."""
    T, D = Y.shape
    n_panels = D + 2  # D modality traces + state posterior + Viterbi strip

    fig, axes = plt.subplots(n_panels, 1, figsize=(28, 2.2 * n_panels),
                             gridspec_kw={'height_ratios': [1]*D + [0.8, 0.4]},
                             sharex=True)

    # Condition shading on all panels
    for ax in axes:
        for seg_name, t0, t1 in segments:
            ax.axvspan(t0, t1, alpha=0.2,
                       color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)
        ax.set_xlim(t[0], t[-1])

    # Condition labels on top
    for seg_name, t0, t1 in segments:
        axes[0].text((t0 + t1) / 2, 1.08, seg_name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=7, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    # Z-timecourse panels with state background coloring
    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        z = Y[:, idx]

        # State background coloring
        for k in range(K):
            state_mask = path == k
            runs = np.diff(np.concatenate([[0], state_mask.astype(int), [0]]))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            for s, e in zip(starts, ends):
                if s < T and e <= T:
                    ax.axvspan(t[max(0, s)], t[min(e-1, T-1)],
                               alpha=0.12, color=STATE_COLORS[k], zorder=0)

        ax.plot(t, z, color=color, linewidth=0.5, alpha=0.8, zorder=2)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.3)
        ax.axhline(2, color='gray', linewidth=0.3, linestyle='--', alpha=0.2)
        ax.set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=6)

    # State posterior panel
    ax_gamma = axes[D]
    for k in range(K):
        ax_gamma.fill_between(t, 0, gamma[:, k], alpha=0.5,
                              color=STATE_COLORS[k], label=f'S{k}')
    ax_gamma.set_ylabel('State\nposterior', fontsize=7, rotation=0, ha='right', va='center')
    ax_gamma.set_ylim(0, 1)
    ax_gamma.legend(fontsize=6, loc='upper right', ncol=K)
    ax_gamma.tick_params(labelsize=6)

    # Viterbi strip
    ax_vit = axes[D + 1]
    for k in range(K):
        mask_k = path == k
        ax_vit.fill_between(t, 0, 1, where=mask_k,
                            color=STATE_COLORS[k], alpha=0.8)
    ax_vit.set_ylabel('State', fontsize=7, rotation=0, ha='right', va='center')
    ax_vit.set_ylim(0, 1)
    ax_vit.set_yticks([])
    ax_vit.set_xlabel('Time (s)', fontsize=8)
    ax_vit.tick_params(labelsize=6)

    # Per-state usage + dwell info
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

    usage = [float((path == k).mean()) for k in range(K)]
    title = (f'{session_name} — Hierarchical rSLDS + Constrained Viterbi '
             f'(K={K}, min_dwell={MIN_DWELL/FS_OUT:.0f}s)\n'
             f'Transitions: {n_trans}, Mean dwell: {np.mean(dwells):.0f}s, '
             f'Usage: ' + ' '.join(f'S{k}={u:.0%}' for k, u in enumerate(usage)))

    fig.suptitle(title, fontsize=9, fontfamily='monospace', ha='left', x=0.02, y=0.995)
    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print("=" * 70)
    print("  Generating Hierarchical rSLDS Timeline Plots")
    print("=" * 70)

    # Load all V8 sessions
    npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_v8_ztimecourses.npz'))
    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        Y, t, segments = load_session(name)
        if Y is not None:
            sessions.append((name, Y, t, segments))

    print(f"  {len(sessions)} sessions loaded")

    # Fit hierarchical rSLDS
    print(f"\n  Fitting hierarchical rSLDS...")
    session_tuples = [(Y, np.zeros((len(Y), 2)), np.ones((len(Y), len(MODALITY_KEYS)), dtype=bool))
                      for _, Y, _, _ in sessions]

    cfg = IOHMMConfig(
        K=K, D_obs=len(MODALITY_KEYS), D_input=2, D_latent=3,
        n_factors=2, recurrent=True,
        n_restarts=2, max_em_iter=80,
        sticky_strength=3.0,
    )
    result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)

    print(f"  BIC: {result['bic']:.0f}")

    # Generate per-session plots
    print(f"\n  Generating timeline plots...")
    for i, (name, Y, t, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path = constrained_viterbi_from_gamma(gamma, min_dwell=MIN_DWELL)

        out_dir = f'results/rslds/{name}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'hierarchical_rslds_timeline.png')
        plot_session_timeline(name, Y, t, segments, path, gamma, out_path)

    # Grand summary plot: state usage by condition across sessions
    fig, ax = plt.subplots(figsize=(10, 6))
    periods = ['conversation', 'baseline', 'meditation', 'gaps']
    bar_width = 0.18
    x = np.arange(len(periods))

    for k in range(K):
        means = []
        stds = []
        for period in periods:
            usages = []
            for i, (name, Y, t, segments) in enumerate(sessions):
                gamma = result['sessions'][i]['gamma']
                path = constrained_viterbi_from_gamma(gamma, min_dwell=MIN_DWELL)
                segs = segments

                conv_m = np.zeros(len(Y), dtype=bool)
                base_m = np.zeros(len(Y), dtype=bool)
                med_m = np.zeros(len(Y), dtype=bool)
                gap_m = np.ones(len(Y), dtype=bool)
                for sn, t0, t1 in segs:
                    seg = (t >= t0) & (t <= t1)
                    gap_m[seg] = False
                    if 'conv' in sn: conv_m |= seg
                    elif 'base' in sn: base_m |= seg
                    elif 'meditate' in sn: med_m |= seg

                pm = {'conversation': conv_m, 'baseline': base_m,
                      'meditation': med_m, 'gaps': gap_m}[period]
                if pm.sum() >= 5:
                    usages.append(float((path[pm] == k).mean()))

            means.append(np.mean(usages) if usages else 0)
            stds.append(np.std(usages) if usages else 0)

        ax.bar(x + k * bar_width, means, bar_width, yerr=stds,
               label=f'S{k}', color=STATE_COLORS[k], alpha=0.8, capsize=3)

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(periods, fontsize=10)
    ax.set_ylabel('State Usage Fraction', fontsize=11)
    ax.set_title('Hierarchical rSLDS State Usage by Condition (8 sessions)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.8)
    plt.tight_layout()
    fig.savefig('results/rslds/hierarchical_state_usage_by_condition.png', dpi=150)
    plt.close(fig)
    print(f"  Saved results/rslds/hierarchical_state_usage_by_condition.png")

    print(f"\n  Done!")


if __name__ == '__main__':
    main()
