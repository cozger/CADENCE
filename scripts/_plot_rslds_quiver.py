"""rSLDS Quiver Plots: state-center flow, per-modality shifts, observation trajectories.

Three plot types from V8.2 rSLDS results:
  1. State-center quiver using domain-informed composite axes
  2. Per-modality flow arrows showing how each channel shifts between states
  3. Observation trajectory quiver colored by Viterbi state

Axes use interpretable composites rather than PCA:
  - "Phase coupling" = mean(ImCoh θ/α/β)  — genuine inter-brain phase locking
  - "Shared power"   = mean(Conc θ/α/β)   — both brains in similar power state
  - "Body coupling"  = mean(BL expr, BL act, Pose) — non-neural synchrony
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch  # noqa: F401

# ── Constants ──────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'rslds')

# Per-session analysis used 12D
MOD_KEYS_12 = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MOD_NAMES_12 = [
    'ImCoh θ', 'ImCoh α', 'ImCoh β',
    'Conc θ', 'Conc α', 'Conc β',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]

# Hierarchical used 15D (adds dynamics)
MOD_KEYS_15 = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_theta', 'dyn_alpha', 'dyn_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MOD_NAMES_15 = [
    'ImCoh θ', 'ImCoh α', 'ImCoh β',
    'Conc θ', 'Conc α', 'Conc β',
    'Dyn θ', 'Dyn α', 'Dyn β',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]

# Modality group colors
MOD_GROUP_COLORS_12 = [
    '#1565C0', '#2196F3', '#64B5F6',     # ImCoh
    '#E65100', '#FF9800', '#FFB74D',     # Conc
    '#E91E63', '#F48FB1',                # BL
    '#4CAF50', '#81C784',                # ECG
    '#9C27B0',                           # Resp
    '#795548',                           # Pose
]

MOD_GROUP_COLORS_15 = [
    '#1565C0', '#2196F3', '#64B5F6',     # ImCoh
    '#E65100', '#FF9800', '#FFB74D',     # Conc
    '#BF360C', '#E64A19', '#FF7043',     # Dyn
    '#E91E63', '#F48FB1',                # BL
    '#4CAF50', '#81C784',                # ECG
    '#9C27B0',                           # Resp
    '#795548',                           # Pose
]

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
STATE_LABELS = ['NULL', 'COUP', 'OTHER', 'SHARED']


# ── Domain-informed composite axes ───────────────────────────────────
def compute_composite_axes(data, mod_keys):
    """Compute interpretable composite scores from raw modality data.

    Returns dict of {axis_name: (values, label)} where values is (N,) or (K,).
    Works on both d_emit (K, D) and observation matrices (T, D).

    Composites:
      - Phase coupling:  mean(ImCoh θ/α/β)
      - Shared power:    mean(Conc θ/α/β)
      - EEG dynamics:    mean(Dyn θ/α/β)       [if present]
      - Body coupling:   mean(BL expr, BL act, Pose)
      - Autonomic:       mean(ECG LF, ECG HF, Resp)
    """
    def _idx(keys):
        return [mod_keys.index(k) for k in keys if k in mod_keys]

    imcoh_idx = _idx(['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'])
    conc_idx = _idx(['conc_theta', 'conc_alpha', 'conc_beta'])
    dyn_idx = _idx(['dyn_theta', 'dyn_alpha', 'dyn_beta'])
    body_idx = _idx(['bl_expr', 'bl_activity_conc', 'pose'])
    auto_idx = _idx(['ecg_lf', 'ecg_hf', 'resp'])

    composites = {}
    if imcoh_idx:
        composites['phase'] = (data[..., imcoh_idx].mean(axis=-1),
                               'Phase Coupling (mean ImCoh)')
    if conc_idx:
        composites['shared'] = (data[..., conc_idx].mean(axis=-1),
                                'Shared Power (mean Concordance)')
    if dyn_idx:
        composites['dynamics'] = (data[..., dyn_idx].mean(axis=-1),
                                  'EEG Dynamics (mean EWMAD)')
    if body_idx:
        composites['body'] = (data[..., body_idx].mean(axis=-1),
                              'Body Coupling (BL + Pose)')
    if auto_idx:
        composites['autonomic'] = (data[..., auto_idx].mean(axis=-1),
                                   'Autonomic (ECG + Resp)')
    return composites


# ── Plot 1: State-center quiver with domain axes ─────────────────────
def plot_state_center_quiver(d_emit, state_labels, trans_matrix, mod_keys,
                             ax_x='phase', ax_y='shared',
                             title='rSLDS State Flow',
                             out_path=None):
    """Plot states in domain-informed 2D space with transition arrows.

    Args:
        d_emit: (K, D) emission means per state
        state_labels: list of K state names
        trans_matrix: (K, K) empirical transition probabilities
        mod_keys: list of D modality key names (not display names)
        ax_x, ax_y: which composite axes to use ('phase', 'shared', 'dynamics', 'body', 'autonomic')
    """
    K, D = d_emit.shape
    composites = compute_composite_axes(d_emit, mod_keys)

    if ax_x not in composites or ax_y not in composites:
        avail = list(composites.keys())
        print(f"  Warning: requested axes ({ax_x}, {ax_y}) not all available. Have: {avail}")
        ax_x, ax_y = avail[0], avail[1] if len(avail) > 1 else avail[0]

    x_vals, x_label = composites[ax_x]
    y_vals, y_label = composites[ax_y]
    coords = np.column_stack([x_vals, y_vals])

    fig, ax = plt.subplots(figsize=(10, 9))

    # Light quadrant shading to aid interpretation
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=0)
    ax.axvline(0, color='#999', lw=0.8, ls='--', zorder=0)

    # Draw transition arrows (behind markers)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            p = trans_matrix[i, j]
            if p < 0.005:
                continue

            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                continue

            # Shorten to avoid overlapping state markers
            shrink = 0.15 * dist
            start = coords[i] + shrink * np.array([dx, dy]) / dist
            end = coords[j] - shrink * np.array([dx, dy]) / dist

            alpha = np.clip(p * 4, 0.2, 0.9)
            lw = np.clip(p * 12, 0.8, 6)

            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color=STATE_COLORS[i],
                                       lw=lw, alpha=alpha,
                                       mutation_scale=15 + p * 25,
                                       connectionstyle='arc3,rad=0.12'))

            # Transition label perpendicular to arrow
            mid = (start + end) / 2
            perp = np.array([-dy, dx]) / dist * 0.015
            if p > 0.01:
                ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{p:.1%}',
                       fontsize=8, ha='center', va='center',
                       color=STATE_COLORS[i], alpha=0.85, fontweight='bold')

    # State markers
    for k in range(K):
        ax.scatter(coords[k, 0], coords[k, 1], c=STATE_COLORS[k],
                  s=1200, zorder=5, edgecolors='white', linewidth=2.5)
        ax.text(coords[k, 0], coords[k, 1], state_labels[k],
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white', zorder=6)

    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.15)

    # Expand limits
    margin_x = max(0.05, (coords[:, 0].max() - coords[:, 0].min()) * 0.25)
    margin_y = max(0.05, (coords[:, 1].max() - coords[:, 1].min()) * 0.25)
    ax.set_xlim(coords[:, 0].min() - margin_x, coords[:, 0].max() + margin_x)
    ax.set_ylim(coords[:, 1].min() - margin_y, coords[:, 1].max() + margin_y)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close(fig)
    return composites


# ── Plot 2: Per-modality flow arrows ─────────────────────────────────
def plot_modality_flow(d_emit, state_labels, mod_names, mod_colors,
                       title='Per-Modality Emission Shifts by State',
                       out_path=None):
    """Show how each modality's emission mean differs across states.

    Horizontal grouped bar chart: for each modality, show d_emit[k] for each state.
    Arrows from NULL (d=0) to each coupling state show the shift direction.
    """
    K, D = d_emit.shape
    n_mod = min(D, len(mod_names))

    fig, ax = plt.subplots(figsize=(14, max(6, n_mod * 0.55)))

    y_pos = np.arange(n_mod)
    bar_height = 0.8 / K

    for k in range(K):
        offset = (k - K/2 + 0.5) * bar_height
        vals = d_emit[k, :n_mod]
        bars = ax.barh(y_pos + offset, vals, bar_height * 0.9,
                      color=STATE_COLORS[k], alpha=0.75,
                      label=state_labels[k], edgecolor='white', linewidth=0.5)

        # Add value labels for significant values
        for i, v in enumerate(vals):
            if abs(v) > 0.08:
                ax.text(v + 0.01 * np.sign(v), y_pos[i] + offset,
                       f'{v:+.2f}', va='center',
                       ha='left' if v > 0 else 'right',
                       fontsize=6.5, color=STATE_COLORS[k], fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names[:n_mod], fontsize=9)
    ax.set_xlabel('Emission Mean (z-scored)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', lw=1, zorder=0)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.15, axis='x')

    # Color-code y-axis labels by modality group
    for i, label in enumerate(ax.get_yticklabels()):
        if i < len(mod_colors):
            label.set_color(mod_colors[i])
            label.set_fontweight('bold')

    ax.invert_yaxis()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Plot 3: Observation trajectory quiver ────────────────────────────
def plot_trajectory_quiver(Y, viterbi, d_emit, state_labels, mod_names,
                           dim_x=0, dim_y=1, t=None,
                           arrow_stride=20, title=None, out_path=None):
    """2D scatter of observation trajectory colored by Viterbi state with velocity arrows.

    Args:
        Y: (T, D) observation matrix
        viterbi: (T,) integer state assignments
        d_emit: (K, D) emission means per state
        dim_x, dim_y: which dimensions to plot
        arrow_stride: plot velocity arrow every N samples
    """
    T, D = Y.shape
    K = d_emit.shape[0]

    if title is None:
        title = f'Observation Flow: {mod_names[dim_x]} vs {mod_names[dim_y]}'

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot trajectory points colored by state
    for k in range(K):
        mask = viterbi == k
        if not mask.any():
            continue
        ax.scatter(Y[mask, dim_x], Y[mask, dim_y],
                  c=STATE_COLORS[k], s=3, alpha=0.15, label=state_labels[k],
                  rasterized=True)

    # Velocity arrows at regular intervals
    arrow_idx = np.arange(arrow_stride, T - 1, arrow_stride)
    dx = Y[arrow_idx, dim_x] - Y[arrow_idx - 1, dim_x]
    dy = Y[arrow_idx, dim_y] - Y[arrow_idx - 1, dim_y]

    # Color arrows by current state
    arrow_colors = [STATE_COLORS[viterbi[i]] for i in arrow_idx]

    # Normalize arrow lengths for visibility
    magnitudes = np.sqrt(dx**2 + dy**2)
    med_mag = np.median(magnitudes[magnitudes > 0]) if (magnitudes > 0).any() else 1.0
    scale_factor = 0.3 / med_mag if med_mag > 0 else 1.0

    ax.quiver(Y[arrow_idx, dim_x], Y[arrow_idx, dim_y],
             dx * scale_factor, dy * scale_factor,
             color=arrow_colors, alpha=0.6, scale=1, scale_units='xy',
             width=0.003, headwidth=4, headlength=5, zorder=4)

    # Plot state centers as large stars
    for k in range(K):
        ax.scatter(d_emit[k, dim_x], d_emit[k, dim_y],
                  c=STATE_COLORS[k], s=400, marker='*',
                  edgecolors='black', linewidth=1.5, zorder=6)
        ax.text(d_emit[k, dim_x] + 0.05, d_emit[k, dim_y] + 0.05,
               state_labels[k], fontsize=10, fontweight='bold',
               color=STATE_COLORS[k], zorder=7,
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=STATE_COLORS[k], alpha=0.8))

    ax.set_xlabel(mod_names[dim_x], fontsize=12, fontweight='bold')
    ax.set_ylabel(mod_names[dim_y], fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, markerscale=3, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_trajectory_quiver_composite(Y, viterbi, d_emit, state_labels, mod_keys,
                                     ax_x='phase', ax_y='shared',
                                     arrow_stride=20, title=None, out_path=None):
    """Observation trajectory in domain-composite space with velocity arrows."""
    T, D = Y.shape
    K = d_emit.shape[0]

    comp_obs = compute_composite_axes(Y, mod_keys)
    comp_emit = compute_composite_axes(d_emit, mod_keys)

    x_obs, x_label = comp_obs[ax_x]
    y_obs, y_label = comp_obs[ax_y]
    x_emit = comp_emit[ax_x][0]
    y_emit = comp_emit[ax_y][0]

    if title is None:
        title = f'Observation Flow: {x_label} vs {y_label}'

    fig, ax = plt.subplots(figsize=(11, 10))

    # Crosshairs at origin
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=0)
    ax.axvline(0, color='#999', lw=0.8, ls='--', zorder=0)

    # Scatter by state
    for k in range(K):
        mask = viterbi == k
        if not mask.any():
            continue
        ax.scatter(x_obs[mask], y_obs[mask],
                  c=STATE_COLORS[k], s=4, alpha=0.12, label=state_labels[k],
                  rasterized=True)

    # Velocity arrows
    arrow_idx = np.arange(arrow_stride, T - 1, arrow_stride)
    dx = x_obs[arrow_idx] - x_obs[arrow_idx - 1]
    dy = y_obs[arrow_idx] - y_obs[arrow_idx - 1]
    arrow_colors = [STATE_COLORS[viterbi[i]] for i in arrow_idx]

    magnitudes = np.sqrt(dx**2 + dy**2)
    med_mag = np.median(magnitudes[magnitudes > 0]) if (magnitudes > 0).any() else 1.0
    scale_factor = 0.4 / med_mag if med_mag > 0 else 1.0

    ax.quiver(x_obs[arrow_idx], y_obs[arrow_idx],
             dx * scale_factor, dy * scale_factor,
             color=arrow_colors, alpha=0.5, scale=1, scale_units='xy',
             width=0.003, headwidth=4, headlength=5, zorder=4)

    # State emission centers
    for k in range(K):
        ax.scatter(x_emit[k], y_emit[k], c=STATE_COLORS[k], s=500, marker='*',
                  edgecolors='black', linewidth=1.5, zorder=6)
        ax.text(x_emit[k] + 0.03, y_emit[k] + 0.03,
               state_labels[k], fontsize=11, fontweight='bold',
               color=STATE_COLORS[k], zorder=7,
               bbox=dict(boxstyle='round,pad=0.2', fc='white',
                        ec=STATE_COLORS[k], alpha=0.85))

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, markerscale=3, loc='upper right')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Data loading ─────────────────────────────────────────────────────
def load_hierarchical():
    """Load hierarchical results (shared d_emit across sessions)."""
    path = os.path.join(RESULTS_DIR, 'v82_hierarchical_results.json')
    with open(path) as f:
        data = json.load(f)
    d_emit = np.array(data['d_emit_shared'])
    state_labels = data['state_labels']
    return d_emit, state_labels, data


def load_per_session():
    """Load per-session results."""
    path = os.path.join(RESULTS_DIR, 'v82_rslds_full_results.json')
    with open(path) as f:
        data = json.load(f)
    return data


def load_session_observations(session_id, mod_keys):
    """Load observation timecourses and Viterbi path for a session."""
    # Scaffold observations
    scaffold_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_ztimecourses.npz')
    if not os.path.exists(scaffold_path):
        raise FileNotFoundError(f"No scaffold NPZ for {session_id}")

    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in mod_keys if f'z_{k}' in data])
    t = data['t_common']

    # Viterbi path — prefer v8 full results, fall back to phase2
    v8_path = os.path.join(RESULTS_DIR, session_id, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(RESULTS_DIR, session_id, 'rslds_phase2_results.npz')

    if os.path.exists(v8_path):
        vd = np.load(v8_path)
        viterbi = vd['full_viterbi']
    elif os.path.exists(p2_path):
        vd = np.load(p2_path)
        viterbi = vd['viterbi_path']
    else:
        raise FileNotFoundError(f"No Viterbi path for {session_id}")

    # Align lengths (scaffold and viterbi should match, but be safe)
    T = min(len(Y), len(viterbi))
    return Y[:T], viterbi[:T], t[:T]


def estimate_transition_matrix(viterbi, K):
    """Estimate empirical transition matrix from Viterbi path."""
    trans = np.zeros((K, K))
    for t in range(len(viterbi) - 1):
        trans[viterbi[t], viterbi[t + 1]] += 1
    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return trans / row_sums


# ── Main ─────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(RESULTS_DIR, 'quiver_plots')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("rSLDS Quiver Plots — V8.2")
    print("=" * 70)

    # ── Load data ──
    d_emit_hier, state_labels_hier, hier_data = load_hierarchical()
    per_session_data = load_per_session()

    K = d_emit_hier.shape[0]
    print(f"\nHierarchical model: {K} states, {d_emit_hier.shape[1]}D")
    print(f"Per-session data: {len(per_session_data)} sessions")

    # ════════════════════════════════════════════════════════════════════
    # PLOT 1: State-center quiver (hierarchical, shared model)
    # ════════════════════════════════════════════════════════════════════
    print("\n─── Plot 1: State-center quiver (hierarchical) ───")

    # Estimate average transition matrix across all sessions using Viterbi paths
    trans_accum = np.zeros((K, K))
    n_sessions_with_viterbi = 0

    for sess_info in per_session_data:
        sid = sess_info['session']
        try:
            _, viterbi, _ = load_session_observations(sid, MOD_KEYS_12)
            trans = estimate_transition_matrix(viterbi, K)
            trans_accum += trans
            n_sessions_with_viterbi += 1
        except (FileNotFoundError, Exception) as e:
            print(f"  Skipping {sid}: {e}")

    avg_trans = trans_accum / max(n_sessions_with_viterbi, 1)
    print(f"  Averaged transitions over {n_sessions_with_viterbi} sessions")
    print("  Transition matrix (off-diagonal):")
    for i in range(K):
        off_diag = [f'{avg_trans[i,j]:.2f}' if i != j else '  - ' for j in range(K)]
        print(f"    {state_labels_hier[i]:>6s}: {' '.join(off_diag)}")

    # Generate state-center plots for multiple axis pairs
    axis_pairs = [
        ('phase', 'shared', 'Phase Coupling vs Shared Power'),
        ('phase', 'body',   'Phase Coupling vs Body Coupling'),
        ('shared', 'body',  'Shared Power vs Body Coupling'),
    ]
    # Add dynamics axis if available in hierarchical model
    if 'dyn_theta' in MOD_KEYS_15:
        axis_pairs.append(('shared', 'dynamics', 'Shared Power vs EEG Dynamics'))

    for ax_x, ax_y, pair_label in axis_pairs:
        plot_state_center_quiver(
            d_emit_hier, state_labels_hier, avg_trans, MOD_KEYS_15,
            ax_x=ax_x, ax_y=ax_y,
            title=f'V8.2 State Flow — {pair_label} (hierarchical)',
            out_path=os.path.join(out_dir, f'state_center_{ax_x}_vs_{ax_y}_hier.png')
        )

    # Also do per-session for y_06
    y06_data = [d for d in per_session_data if d['session'] == 'y_06'][0]
    d_emit_y06 = np.array(y06_data['d_emit'])
    try:
        _, vit_y06, _ = load_session_observations('y_06', MOD_KEYS_12)
        trans_y06 = estimate_transition_matrix(vit_y06, K)
        for ax_x, ax_y, pair_label in [('phase', 'shared', 'Phase vs Shared'),
                                        ('phase', 'body', 'Phase vs Body')]:
            plot_state_center_quiver(
                d_emit_y06, y06_data['sl'], trans_y06, MOD_KEYS_12,
                ax_x=ax_x, ax_y=ax_y,
                title=f'V8.2 State Flow — {pair_label} (y_06)',
                out_path=os.path.join(out_dir, f'state_center_{ax_x}_vs_{ax_y}_y06.png')
            )
    except Exception as e:
        print(f"  y_06 state-center plot failed: {e}")

    # ════════════════════════════════════════════════════════════════════
    # PLOT 2: Per-modality flow arrows
    # ════════════════════════════════════════════════════════════════════
    print("\n─── Plot 2: Per-modality emission profiles ───")

    # Hierarchical (15D)
    plot_modality_flow(
        d_emit_hier, state_labels_hier, MOD_NAMES_15, MOD_GROUP_COLORS_15,
        title='V8.2 Hierarchical rSLDS — Per-Modality Emission Means',
        out_path=os.path.join(out_dir, 'modality_flow_hierarchical.png')
    )

    # Per-session y_06 (12D)
    plot_modality_flow(
        d_emit_y06, y06_data['sl'], MOD_NAMES_12, MOD_GROUP_COLORS_12,
        title='V8.2 rSLDS — Per-Modality Emission Means (y_06)',
        out_path=os.path.join(out_dir, 'modality_flow_y06.png')
    )

    # ════════════════════════════════════════════════════════════════════
    # PLOT 3: Observation trajectory quiver (y_06)
    # ════════════════════════════════════════════════════════════════════
    print("\n─── Plot 3: Observation trajectory quiver (y_06) ───")

    try:
        Y_y06, vit_y06, t_y06 = load_session_observations('y_06', MOD_KEYS_12)

        # A) Composite-axis trajectory plots
        traj_pairs = [
            ('phase', 'shared', 'Phase Coupling vs Shared Power'),
            ('phase', 'body',   'Phase Coupling vs Body Coupling'),
            ('shared', 'body',  'Shared Power vs Body Coupling'),
        ]
        for ax_x, ax_y, pair_label in traj_pairs:
            plot_trajectory_quiver_composite(
                Y_y06, vit_y06, d_emit_y06, y06_data['sl'], MOD_KEYS_12,
                ax_x=ax_x, ax_y=ax_y,
                arrow_stride=25,
                title=f'V8.2 y_06: {pair_label}',
                out_path=os.path.join(out_dir, f'trajectory_{ax_x}_vs_{ax_y}_y06.png')
            )

        # B) Key single-channel pairs (most interpretable raw axes)
        pairs = [
            (0, 3, 'ImCoh_th_vs_Conc_th'),   # Phase vs shared power (theta)
            (1, 4, 'ImCoh_al_vs_Conc_al'),   # Same for alpha
            (3, 6, 'Conc_th_vs_BL_expr'),     # EEG shared power vs facial
        ]
        for dim_x, dim_y, safe_name in pairs:
            if dim_x < Y_y06.shape[1] and dim_y < Y_y06.shape[1]:
                plot_trajectory_quiver(
                    Y_y06, vit_y06, d_emit_y06, y06_data['sl'], MOD_NAMES_12,
                    dim_x=dim_x, dim_y=dim_y, arrow_stride=25,
                    out_path=os.path.join(out_dir, f'trajectory_{safe_name}_y06.png')
                )

        # C) Multi-session comparison (composite axes, not PCA)
        print("\n─── Multi-session composite trajectory ───")
        sessions_to_overlay = ['y_06', 'y_17', 'y_32_03132026']
        fig, axes = plt.subplots(1, len(sessions_to_overlay),
                                figsize=(7 * len(sessions_to_overlay), 7))

        for ax_idx, sid in enumerate(sessions_to_overlay):
            ax_plot = axes[ax_idx]
            try:
                sess_data = [d for d in per_session_data if d['session'] == sid]
                if not sess_data:
                    ax_plot.set_title(f'{sid} — not found', fontsize=11)
                    continue
                sess_data = sess_data[0]
                d_emit_s = np.array(sess_data['d_emit'])
                Y_s, vit_s, _ = load_session_observations(sid, MOD_KEYS_12)

                comp_obs = compute_composite_axes(Y_s, MOD_KEYS_12)
                comp_emit = compute_composite_axes(d_emit_s, MOD_KEYS_12)

                x_obs, x_label = comp_obs['phase']
                y_obs, y_label = comp_obs['shared']
                x_emit = comp_emit['phase'][0]
                y_emit = comp_emit['shared'][0]

                ax_plot.axhline(0, color='#999', lw=0.6, ls='--', zorder=0)
                ax_plot.axvline(0, color='#999', lw=0.6, ls='--', zorder=0)

                for k in range(K):
                    mask = vit_s == k
                    if mask.any():
                        ax_plot.scatter(x_obs[mask], y_obs[mask],
                                       c=STATE_COLORS[k], s=3, alpha=0.1,
                                       label=sess_data['sl'][k], rasterized=True)

                stride = 30
                idx = np.arange(stride, len(x_obs) - 1, stride)
                dx = x_obs[idx] - x_obs[idx - 1]
                dy = y_obs[idx] - y_obs[idx - 1]
                mags = np.sqrt(dx**2 + dy**2)
                med = np.median(mags[mags > 0]) if (mags > 0).any() else 1.0
                sf = 0.4 / med if med > 0 else 1.0
                colors = [STATE_COLORS[vit_s[i]] for i in idx]
                ax_plot.quiver(x_obs[idx], y_obs[idx], dx * sf, dy * sf,
                              color=colors, alpha=0.4, scale=1, scale_units='xy',
                              width=0.003, headwidth=4, zorder=4)

                for k in range(K):
                    ax_plot.scatter(x_emit[k], y_emit[k], c=STATE_COLORS[k],
                                   s=300, marker='*', edgecolors='black',
                                   linewidth=1, zorder=6)
                    ax_plot.text(x_emit[k] + 0.03, y_emit[k] + 0.03,
                                sess_data['sl'][k], fontsize=8, fontweight='bold',
                                color=STATE_COLORS[k], zorder=7)

                ax_plot.set_title(f'{sid}', fontsize=12, fontweight='bold')
                ax_plot.set_xlabel(x_label, fontsize=9)
                if ax_idx == 0:
                    ax_plot.set_ylabel(y_label, fontsize=9)
                ax_plot.grid(True, alpha=0.15)
                if ax_idx == 0:
                    ax_plot.legend(fontsize=7, markerscale=3, loc='upper right')
            except Exception as e:
                ax_plot.set_title(f'{sid} — error: {e}', fontsize=9)

        plt.suptitle('V8.2 Observation Flow — Phase vs Shared Power (3 sessions)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        multi_path = os.path.join(out_dir, 'trajectory_multi_session.png')
        fig.savefig(multi_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {multi_path}")
        plt.close(fig)

    except Exception as e:
        print(f"  Trajectory plots failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
