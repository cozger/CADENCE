"""rSLDS Excursion Analysis: decompose large trajectory jumps by modality and condition.

Identifies timesteps where the observation velocity exceeds the 95th percentile,
then shows:
  1. Which modalities drove each large jump (bar decomposition)
  2. Where in the session they occurred (condition timeline)
  3. Annotated trajectory plot highlighting the top excursions
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'rslds')

MOD_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MOD_NAMES = [
    'ImCoh θ', 'ImCoh α', 'ImCoh β',
    'Conc θ', 'Conc α', 'Conc β',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]
# Group labels for summarizing which system drove a jump
MOD_GROUPS = {
    'EEG phase':  [0, 1, 2],       # ImCoh
    'EEG power':  [3, 4, 5],       # Concordance
    'Face':       [6, 7],           # BL
    'Autonomic':  [8, 9, 10],      # ECG + Resp
    'Body':       [11],             # Pose
}
GROUP_COLORS = {
    'EEG phase': '#1565C0', 'EEG power': '#E65100',
    'Face': '#E91E63', 'Autonomic': '#4CAF50', 'Body': '#795548',
}

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
STATE_LABELS = ['NULL', 'COUP', 'OTHER', 'SHARED']
CONDITION_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#9FA8DA', 'baseline': '#90CAF9',
    'conv_1': '#FFE0B2', 'conv_2': '#FFCC80',
    'PE': '#F8BBD0', 'PE_1': '#F8BBD0', 'PE_2': '#F8BBD0',
    'meditate_B': '#CE93D8', 'meditate_K': '#A5D6A7',
    'gap': '#E0E0E0', 'gap_pre': '#E0E0E0', 'gap_post': '#E0E0E0',
}


def load_session(session_id):
    """Load observations, Viterbi, timestamps, and condition segments."""
    scaffold_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_ztimecourses.npz')
    results_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_results.json')
    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in MOD_KEYS if f'z_{k}' in data])
    t = data['t_common']

    # Viterbi
    v8_path = os.path.join(RESULTS_DIR, session_id, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(RESULTS_DIR, session_id, 'rslds_phase2_results.npz')
    if os.path.exists(v8_path):
        viterbi = np.load(v8_path)['full_viterbi']
    else:
        viterbi = np.load(p2_path)['viterbi_path']

    T = min(len(Y), len(viterbi))
    Y, viterbi, t = Y[:T], viterbi[:T], t[:T]

    # Condition segments
    with open(results_path) as f:
        meta = json.load(f)
    segments = meta.get('segments', [])

    return Y, viterbi, t, segments


def find_excursions(Y, percentile=95):
    """Find timesteps with large velocity (above percentile threshold).

    Returns:
        velocities: (T-1, D) per-channel velocity
        speed: (T-1,) L2 norm of velocity
        excursion_idx: indices where speed > threshold
    """
    velocities = np.diff(Y, axis=0)  # (T-1, D)
    speed = np.linalg.norm(velocities, axis=1)  # (T-1,)
    threshold = np.percentile(speed, percentile)
    excursion_idx = np.where(speed > threshold)[0]
    return velocities, speed, excursion_idx, threshold


def get_condition_at_time(lsl_time, segments):
    """Map an LSL timestamp to its experimental condition."""
    for seg in segments:
        name, t_start, t_end = seg[0], seg[1], seg[2]
        if t_start <= lsl_time <= t_end:
            return name
    return 'gap'


def decompose_by_group(velocity_vec):
    """Decompose a velocity vector into modality group contributions.

    Returns dict of {group_name: fraction_of_total_energy}.
    Energy = sum of squared velocity per channel in the group.
    """
    total_energy = np.sum(velocity_vec**2)
    if total_energy < 1e-12:
        return {g: 0.0 for g in MOD_GROUPS}
    result = {}
    for group_name, indices in MOD_GROUPS.items():
        group_energy = np.sum(velocity_vec[indices]**2)
        result[group_name] = group_energy / total_energy
    return result


def main():
    session_id = 'y_06'
    out_dir = os.path.join(RESULTS_DIR, 'quiver_plots')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"rSLDS Excursion Analysis — {session_id}")
    print("=" * 70)

    Y, viterbi, t, segments = load_session(session_id)
    T, D = Y.shape
    fs = 2.0  # Hz
    t_rel = (t - t[0]) / 60.0  # minutes from session start

    velocities, speed, exc_idx, threshold = find_excursions(Y, percentile=95)
    print(f"\nT={T} samples ({T/fs/60:.1f} min), D={D} channels")
    print(f"Speed threshold (95th pctl): {threshold:.3f}")
    print(f"Excursions found: {len(exc_idx)} ({len(exc_idx)/(T-1)*100:.1f}%)")

    # ── Decompose each excursion by modality group ──
    group_fracs = []
    conditions = []
    states_from = []
    states_to = []

    for idx in exc_idx:
        group_fracs.append(decompose_by_group(velocities[idx]))
        conditions.append(get_condition_at_time(t[idx], segments))
        states_from.append(viterbi[idx])
        states_to.append(viterbi[min(idx + 1, T - 1)])

    group_fracs = {g: np.array([f[g] for f in group_fracs]) for g in MOD_GROUPS}

    # ── Summary statistics ──
    print(f"\n{'─' * 50}")
    print("Modality group driving excursions (mean energy %):")
    for g in MOD_GROUPS:
        pct = group_fracs[g].mean() * 100
        print(f"  {g:>12s}: {pct:5.1f}%")

    print(f"\nCondition distribution of excursions:")
    from collections import Counter
    cond_counts = Counter(conditions)
    for cond, count in sorted(cond_counts.items(), key=lambda x: -x[1]):
        pct = count / len(conditions) * 100
        print(f"  {cond:>15s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nState at excursion (from → to):")
    state_trans = Counter(zip(states_from, states_to))
    for (sf, st), count in sorted(state_trans.items(), key=lambda x: -x[1])[:10]:
        label = f"{STATE_LABELS[sf]}→{STATE_LABELS[st]}"
        same = "stay" if sf == st else "TRANSITION"
        print(f"  {label:>20s}: {count:4d}  ({same})")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Excursion decomposition — which modalities drive big jumps
    # ════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1.2], hspace=0.35, wspace=0.3)

    # ── Panel A: Timeline of excursion speed colored by dominant modality ──
    ax_timeline = fig.add_subplot(gs[0, :])

    # Background condition bands
    for seg in segments:
        name, t_start, t_end = seg[0], seg[1], seg[2]
        t_start_rel = (t_start - t[0]) / 60.0
        t_end_rel = (t_end - t[0]) / 60.0
        color = CONDITION_COLORS.get(name, '#F5F5F5')
        ax_timeline.axvspan(t_start_rel, t_end_rel, alpha=0.3, color=color, zorder=0)
        mid = (t_start_rel + t_end_rel) / 2
        ax_timeline.text(mid, speed.max() * 1.02, name.replace('_', '\n'),
                        ha='center', va='bottom', fontsize=7, fontweight='bold', alpha=0.7)

    # Plot full speed trace
    ax_timeline.plot(t_rel[1:], speed, color='#BDBDBD', lw=0.5, alpha=0.5, zorder=1)

    # Highlight excursions colored by dominant modality group
    for idx in exc_idx:
        decomp = decompose_by_group(velocities[idx])
        dominant = max(decomp, key=decomp.get)
        ax_timeline.scatter(t_rel[idx + 1], speed[idx],
                          c=GROUP_COLORS[dominant], s=15, alpha=0.8, zorder=3)

    # Threshold line
    ax_timeline.axhline(threshold, color='red', ls='--', lw=1, alpha=0.5,
                       label=f'95th pctl = {threshold:.2f}')

    ax_timeline.set_xlabel('Time (minutes)', fontsize=11)
    ax_timeline.set_ylabel('Speed (L2 norm of Δy)', fontsize=11)
    ax_timeline.set_title(f'Excursion Timeline — {session_id}  (colored by dominant modality)',
                         fontsize=13, fontweight='bold')

    # Legend for modality groups
    for g, c in GROUP_COLORS.items():
        ax_timeline.scatter([], [], c=c, s=40, label=g)
    ax_timeline.legend(fontsize=8, loc='upper right', ncol=3)

    # ── Panel B: Average modality breakdown of excursions vs non-excursions ──
    ax_bar = fig.add_subplot(gs[1, 0])

    groups = list(MOD_GROUPS.keys())
    exc_means = [group_fracs[g].mean() for g in groups]

    # Also compute for non-excursions as comparison
    non_exc_mask = np.ones(T - 1, dtype=bool)
    non_exc_mask[exc_idx] = False
    non_exc_decomps = {g: [] for g in MOD_GROUPS}
    for idx in np.where(non_exc_mask)[0][::5]:  # subsample for speed
        d = decompose_by_group(velocities[idx])
        for g in MOD_GROUPS:
            non_exc_decomps[g].append(d[g])
    non_exc_means = [np.mean(non_exc_decomps[g]) for g in groups]

    x = np.arange(len(groups))
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, exc_means, width,
                       color=[GROUP_COLORS[g] for g in groups], alpha=0.85,
                       label='Excursions (top 5%)')
    bars2 = ax_bar.bar(x + width/2, non_exc_means, width,
                       color=[GROUP_COLORS[g] for g in groups], alpha=0.3,
                       label='Normal (bottom 95%)')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(groups, fontsize=9, fontweight='bold')
    ax_bar.set_ylabel('Energy Fraction', fontsize=10)
    ax_bar.set_title('What Drives Large Jumps?', fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.15, axis='y')

    # ── Panel C: Excursions by condition (normalized by condition duration) ──
    ax_cond = fig.add_subplot(gs[1, 1])

    # Compute excursion rate per condition (excursions/minute)
    cond_durations = {}
    for seg in segments:
        name = seg[0]
        dur = (seg[2] - seg[1]) / 60.0  # minutes
        cond_durations[name] = cond_durations.get(name, 0) + dur

    cond_rates = {}
    for cond in set(conditions):
        count = cond_counts[cond]
        dur = cond_durations.get(cond, 1.0)
        cond_rates[cond] = count / dur if dur > 0 else 0

    # Also add gap rate
    gap_time = (T / fs / 60.0) - sum(cond_durations.values())
    if gap_time > 0 and 'gap' in cond_counts:
        cond_rates['gap'] = cond_counts['gap'] / gap_time

    sorted_conds = sorted(cond_rates.items(), key=lambda x: -x[1])
    cond_names = [c[0] for c in sorted_conds]
    cond_vals = [c[1] for c in sorted_conds]
    cond_colors = [CONDITION_COLORS.get(c, '#BDBDBD') for c in cond_names]

    bars = ax_cond.barh(range(len(cond_names)), cond_vals,
                       color=cond_colors, edgecolor='#666', linewidth=0.5)
    ax_cond.set_yticks(range(len(cond_names)))
    ax_cond.set_yticklabels(cond_names, fontsize=9, fontweight='bold')
    ax_cond.set_xlabel('Excursions / minute', fontsize=10)
    ax_cond.set_title('Excursion Rate by Condition', fontsize=12, fontweight='bold')
    ax_cond.invert_yaxis()
    ax_cond.grid(True, alpha=0.15, axis='x')

    for bar, val in zip(bars, cond_vals):
        ax_cond.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}/min', va='center', fontsize=8, fontweight='bold')

    # ── Panel D: Per-channel signed velocity for top 20 excursions ──
    ax_heat = fig.add_subplot(gs[2, :])

    # Get top 20 by speed
    top_n = 30
    top_idx = exc_idx[np.argsort(speed[exc_idx])[-top_n:]]
    top_idx = top_idx[np.argsort(top_idx)]  # sort by time

    vel_matrix = velocities[top_idx]  # (top_n, D)

    # Normalize per channel for visibility
    abs_max = np.abs(vel_matrix).max(axis=0, keepdims=True)
    abs_max[abs_max < 1e-6] = 1.0
    vel_norm = vel_matrix / abs_max

    im = ax_heat.imshow(vel_norm.T, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1, interpolation='nearest')

    # Y-axis: modality names
    ax_heat.set_yticks(range(D))
    ax_heat.set_yticklabels(MOD_NAMES[:D], fontsize=8)

    # X-axis: time + condition labels
    x_labels = []
    for idx in top_idx:
        cond = get_condition_at_time(t[idx], segments)
        minutes = t_rel[idx]
        state = STATE_LABELS[viterbi[idx]]
        x_labels.append(f'{minutes:.1f}m\n{cond}\n{state}')
    ax_heat.set_xticks(range(len(top_idx)))
    ax_heat.set_xticklabels(x_labels, fontsize=6.5, ha='center')

    ax_heat.set_title(f'Top {top_n} Excursions — Per-Channel Velocity (normalized)',
                     fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_heat, label='Normalized Δy (red=increase, blue=decrease)',
                shrink=0.6, pad=0.02)

    # Highlight dominant channel per excursion
    for col in range(len(top_idx)):
        dom_row = np.argmax(np.abs(vel_matrix[col]))
        ax_heat.scatter(col, dom_row, marker='*', c='black', s=60, zorder=5)

    plt.suptitle(f'rSLDS Excursion Decomposition — {session_id}',
                fontsize=15, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(out_dir, f'excursion_analysis_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"\n  Saved: excursion_analysis_{session_id}.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Annotated trajectory with top excursions labeled
    # ════════════════════════════════════════════════════════════════════
    from scripts._plot_rslds_quiver import compute_composite_axes

    comp_obs = compute_composite_axes(Y, MOD_KEYS)
    x_obs = comp_obs['phase'][0]
    y_obs = comp_obs['shared'][0]

    # d_emit for state centers
    with open(os.path.join(RESULTS_DIR, 'v82_rslds_full_results.json')) as f:
        all_data = json.load(f)
    y06_data = [d for d in all_data if d['session'] == session_id][0]
    d_emit = np.array(y06_data['d_emit'])
    comp_emit = compute_composite_axes(d_emit, MOD_KEYS)
    x_emit = comp_emit['phase'][0]
    y_emit = comp_emit['shared'][0]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axhline(0, color='#999', lw=0.6, ls='--', zorder=0)
    ax.axvline(0, color='#999', lw=0.6, ls='--', zorder=0)

    # Background scatter
    K = d_emit.shape[0]
    for k in range(K):
        mask = viterbi == k
        if mask.any():
            ax.scatter(x_obs[mask], y_obs[mask], c=STATE_COLORS[k],
                      s=3, alpha=0.08, label=STATE_LABELS[k], rasterized=True)

    # State centers
    for k in range(K):
        ax.scatter(x_emit[k], y_emit[k], c=STATE_COLORS[k], s=400, marker='*',
                  edgecolors='black', linewidth=1.5, zorder=6)

    # Draw top excursions as prominent arrows with annotations
    top_15 = exc_idx[np.argsort(speed[exc_idx])[-15:]]

    for rank, idx in enumerate(sorted(top_15, key=lambda i: -speed[i])):
        # Arrow showing the jump
        x0, y0 = x_obs[idx], y_obs[idx]
        x1, y1 = x_obs[idx + 1], y_obs[idx + 1]

        # Dominant modality group
        decomp = decompose_by_group(velocities[idx])
        dominant = max(decomp, key=decomp.get)
        dom_pct = decomp[dominant] * 100
        cond = get_condition_at_time(t[idx], segments)

        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle='->', color=GROUP_COLORS[dominant],
                                   lw=2.5, alpha=0.85, mutation_scale=18))

        # Label
        label = f'#{rank+1}: {dominant} ({dom_pct:.0f}%)\n{cond}, {t_rel[idx]:.1f}m'
        ax.annotate(label, xy=(x1, y1), fontsize=6.5,
                   xytext=(8, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                            ec=GROUP_COLORS[dominant]),
                   fontweight='bold', color=GROUP_COLORS[dominant])

    ax.set_xlabel('Phase Coupling (mean ImCoh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shared Power (mean Concordance)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Excursions Annotated — {session_id}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, markerscale=3, loc='upper right')
    ax.grid(True, alpha=0.15)

    # Add group legend
    for g, c in GROUP_COLORS.items():
        ax.plot([], [], color=c, lw=3, label=f'→ {g}')
    ax.legend(fontsize=8, loc='upper left', ncol=2)

    fig.savefig(os.path.join(out_dir, f'excursion_annotated_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: excursion_annotated_{session_id}.png")
    plt.close(fig)

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
