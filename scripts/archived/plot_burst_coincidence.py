"""Visualize V11 Burst Coincidence Results.

Generates three figures:
  1. y_06 timeline: coincidence z per band + condition shading + rSLDS states
  2. Per-condition bar chart: cross-session mean coincidence by condition
  3. Per-state comparison: SHARED vs NULL vs OTHER

Usage:
    python scripts/plot_burst_coincidence.py
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CADENCE plotting conventions ────────────────────────────────────
CONDITION_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#9FA8DA', 'baseline': '#90CAF9',
    'conv_1': '#FFE0B2', 'conv_2': '#FFCC80',
    'PE_1': '#F8BBD0', 'PE_2': '#F8BBD0',
    'meditate_B': '#CE93D8', 'meditate_K': '#A5D6A7',
}

BAND_COLORS = {
    'theta': '#1565C0',   # blue
    'alpha': '#E65100',   # orange
    'beta':  '#2E7D32',   # green
}

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']

# Display order used for plotting — held fixed regardless of the hierarchical
# fit's index permutation so figures stay visually consistent across refits.
DISPLAY_ORDER = ['NULL', 'COUP', 'SHARED', 'OTHER']

CONDS_MED = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
CONDS_PE = ['base_EO', 'base_EC', 'conv_1', 'PE_1', 'PE_2', 'conv_2']

SCAFFOLD_DIR = 'results/v11'
OUT_DIR = 'results/v11/burst_coincidence'
HIERARCHICAL_RESULTS_PATH = 'results/v11/hierarchical/v11_hierarchical_results.json'


def _load_state_labels(path=HIERARCHICAL_RESULTS_PATH):
    """Load index-ordered state labels from the V11 hierarchical fit.

    The hierarchical fit assigns labels by emission structure (COUP=argmax
    imcoh sum, SHARED=argmax conc sum), so the index→label permutation
    differs across refits. Always read it from the fit, not a hardcoded list.
    """
    with open(path) as f:
        labels = list(json.load(f)['state_labels'])
    assert sorted(labels) == ['COUP', 'NULL', 'OTHER', 'SHARED'], \
        f'Unexpected state labels in {path}: {labels}'
    return labels


def load_results():
    """Load all-session burst coincidence results."""
    path = os.path.join(OUT_DIR, 'burst_coincidence_results.json')
    with open(path) as f:
        return json.load(f)


# ── Figure 1: y_06 Timeline ────────────────────────────────────────

def plot_y06_timeline(state_labels):
    """Coincidence z timecourse for y_06 with condition shading + rSLDS states.

    Args:
        state_labels: index-ordered labels from the hierarchical fit;
            state_labels[si] names the V11 rSLDS state with index si.
    """

    # Load scaffold + coincidence for y_06 (V11)
    scaffold_npz = f'{SCAFFOLD_DIR}/y_06/scaffold_v11_ztimecourses.npz'
    scaffold_json = f'{SCAFFOLD_DIR}/y_06/scaffold_v11_results.json'
    rslds_npz = f'{SCAFFOLD_DIR}/y_06/v11_rslds_results.npz'

    data = np.load(scaffold_npz)
    t_common = data['t_common']
    t0_abs = t_common[0]

    with open(scaffold_json) as f:
        info = json.load(f)
    segments = [(s[0], s[1], s[2]) for s in info['segments']]

    rslds = np.load(rslds_npz)
    state_path = rslds['path']

    # Recompute coincidence z for y_06 (load from saved results)
    results = load_results()
    y06_data = None
    for sess in results['sessions']:
        if sess['session'] == 'y_06':
            y06_data = sess
            break

    if y06_data is None:
        print("No y_06 data found in results")
        return

    # We need the full z timecourse — recompute from raw
    # Load cache + XDF to get the raw coincidence timecourses
    from cadence.config import load_config
    from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
    from cadence.significance.burst_coincidence import eeg_burst_coincidence
    from scripts.run_session_v6 import load_xdf_session

    config = load_config()
    raw_dir = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
    xdf_files = glob.glob(os.path.join(raw_dir, 'y_06*.xdf'))
    session_data = load_xdf_session(xdf_files[0])
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if 'y_06' in n.lower()]
    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    coinc = eeg_burst_coincidence(cached, t_common, lsl_offset,
                                   tau_samples=1, n_surrogates=200, seed=42)

    # Time axis in minutes from session start
    t_min = (t_common - t0_abs) / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(24, 10),
                              gridspec_kw={'height_ratios': [1, 1, 1, 0.3]},
                              sharex=True)

    band_names = ['theta', 'alpha', 'beta']

    for bi, band in enumerate(band_names):
        ax = axes[bi]
        z = coinc[band]['z']
        ax.set_ylim(-5, 5)

        # Condition shading + labels
        for cond_name, t_start, t_end in segments:
            ts = (t_start - t0_abs) / 60.0
            te = (t_end - t0_abs) / 60.0
            color = CONDITION_COLORS.get(cond_name, '#E0E0E0')
            ax.axvspan(ts, te, alpha=0.2, color=color, zorder=0)
            if bi == 0:
                ax.text((ts + te) / 2, 4.5,
                        cond_name.replace('_', ' '), ha='center', va='top',
                        fontsize=7, fontweight='bold', alpha=0.8)

        # Plot z timecourse
        ax.plot(t_min, z, color=BAND_COLORS[band], linewidth=0.8, alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(2, color='red', ls=':', lw=0.5, alpha=0.3)
        ax.axhline(-2, color='red', ls=':', lw=0.5, alpha=0.3)

        # Fill positive/negative regions
        ax.fill_between(t_min, z, 0, where=z > 0,
                         color=BAND_COLORS[band], alpha=0.15)
        ax.fill_between(t_min, z, 0, where=z < 0,
                         color='gray', alpha=0.1)

        # Per-condition mean annotation
        for cond_name, t_start, t_end in segments:
            mask = (t_common >= t_start) & (t_common <= t_end)
            if mask.sum() > 10:
                cond_mean = z[mask].mean()
                ts = (t_start - t0_abs) / 60.0
                te = (t_end - t0_abs) / 60.0
                ax.text((ts + te) / 2, -4.5, f'z={cond_mean:+.2f}',
                        ha='center', va='bottom', fontsize=6,
                        fontweight='bold', color=BAND_COLORS[band], alpha=0.8)

        ax.set_ylabel(f'{band}\ncoincidence z', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # State bar
    ax_state = axes[3]
    for cond_name, t_start, t_end in segments:
        ts = (t_start - t0_abs) / 60.0
        te = (t_end - t0_abs) / 60.0
        ax_state.axvspan(ts, te, alpha=0.15,
                          color=CONDITION_COLORS.get(cond_name, '#E0E0E0'))

    # Plot state blocks
    changes = np.where(np.diff(state_path) != 0)[0] + 1
    block_starts = np.concatenate([[0], changes])
    block_ends = np.concatenate([changes, [len(state_path)]])
    for s, e in zip(block_starts, block_ends):
        state = state_path[s]
        if state < len(STATE_COLORS):
            ax_state.axvspan(t_min[s], t_min[min(e, len(t_min) - 1)],
                              color=STATE_COLORS[state], alpha=0.6)

    ax_state.set_ylabel('rSLDS\nstate', fontsize=8, fontweight='bold')
    ax_state.set_yticks([])
    ax_state.set_xlabel('Time (minutes)', fontsize=10, fontweight='bold')
    ax_state.tick_params(labelsize=7)

    # Legend for states (use loaded label permutation, not a hardcoded list)
    from matplotlib.patches import Patch
    state_patches = [Patch(facecolor=STATE_COLORS[i], alpha=0.6,
                            label=state_labels[i])
                     for i in sorted(set(state_path))]
    ax_state.legend(handles=state_patches, loc='upper right', fontsize=7,
                     ncol=len(state_patches))

    fig.suptitle('EEG Burst Coincidence Timeline -- y_06 (tau=500ms, 200 surrogates)',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, 'y06_coincidence_timeline.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Per-Condition Bar Chart ──────────────────────────────

def plot_condition_bars():
    """Cross-session mean coincidence z by condition, per band."""
    results = load_results()
    sessions = results['sessions']

    band_names = ['theta', 'alpha', 'beta']

    # Separate protocols
    med_sessions = []
    pe_sessions = []
    for s in sessions:
        conds = list(s['per_condition'].keys())
        if any('meditate' in c for c in conds):
            med_sessions.append(s)
        elif any('PE' in c for c in conds):
            pe_sessions.append(s)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for bi, band in enumerate(band_names):
        ax = axes[bi]

        # Collect per-condition means across sessions (all protocols)
        all_conds = CONDS_MED + ['PE_1', 'PE_2']
        all_conds = list(dict.fromkeys(all_conds))  # unique, ordered

        means = []
        ses = []
        colors = []
        labels = []

        for cond in all_conds:
            vals = []
            for s in sessions:
                if cond in s['per_condition'] and band in s['per_condition'][cond]:
                    vals.append(s['per_condition'][cond][band]['mean_z'])
            if len(vals) >= 1:
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                colors.append(CONDITION_COLORS.get(cond, '#E0E0E0'))
                labels.append(cond.replace('_', '\n'))

        if not means:
            continue

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Value labels on bars
        for xi, (m, s_val) in enumerate(zip(means, ses)):
            y_offset = 0.02 if m >= 0 else -0.02
            va = 'bottom' if m >= 0 else 'top'
            ax.text(xi, m + y_offset, f'{m:+.2f}', ha='center', va=va,
                    fontsize=7, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, fontweight='bold')
        ax.set_title(f'{band.upper()} Burst Coincidence', fontsize=11,
                      fontweight='bold', color=BAND_COLORS[band])

        if bi == 0:
            ax.set_ylabel('Coincidence z-score\n(surrogate-calibrated)',
                          fontsize=9, fontweight='bold')

    n_sess = len(sessions)
    fig.suptitle(f'Per-Condition EEG Burst Coincidence (n={n_sess} sessions, '
                 f'tau=500ms, 200 surrogates)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, 'condition_coincidence_bars.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: Per-State Comparison ─────────────────────────────────

def plot_state_comparison(state_labels):
    """Per-rSLDS-state coincidence: SHARED vs NULL vs OTHER.

    Args:
        state_labels: index-ordered labels from the hierarchical fit. Used
            to map the rSLDS state index back to a color via STATE_COLORS.
            Display order is held fixed at DISPLAY_ORDER for visual
            consistency across refits.
    """
    results = load_results()
    sessions = results['sessions']

    band_names = ['theta', 'alpha', 'beta']
    # Display order is fixed; only show states actually present in the fit.
    states_to_show = [s for s in DISPLAY_ORDER if s in state_labels]
    # label → state index in the fit (for color lookup against STATE_COLORS)
    label_to_idx = {lab: i for i, lab in enumerate(state_labels)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for bi, band in enumerate(band_names):
        ax = axes[bi]
        means = []
        ses = []
        colors = []
        labels = []

        for slabel in states_to_show:
            vals = []
            for s in sessions:
                if slabel in s.get('per_state', {}) and band in s['per_state'][slabel]:
                    vals.append(s['per_state'][slabel][band]['mean_z'])
            if vals:
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                # Color by the state's index in the fit so the color stays
                # tied to the rSLDS state regardless of display order.
                colors.append(STATE_COLORS[label_to_idx[slabel]])
                labels.append(slabel)

        if not means:
            continue

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        for xi, m in enumerate(means):
            y_offset = 0.01 if m >= 0 else -0.01
            va = 'bottom' if m >= 0 else 'top'
            ax.text(xi, m + y_offset, f'{m:+.3f}', ha='center', va=va,
                    fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
        ax.set_title(f'{band.upper()}', fontsize=11,
                      fontweight='bold', color=BAND_COLORS[band])

        if bi == 0:
            ax.set_ylabel('Coincidence z-score', fontsize=9, fontweight='bold')

    n_sess = len(sessions)
    fig.suptitle(f'Burst Coincidence by rSLDS State (n={n_sess} sessions)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, 'state_coincidence_bars.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Conversation vs Rest scatter ─────────────────────────

def plot_conv_vs_rest_scatter():
    """Per-session scatter: conversation coincidence vs rest, per band."""
    results = load_results()
    sessions = results['sessions']

    band_names = ['theta', 'alpha', 'beta']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for bi, band in enumerate(band_names):
        ax = axes[bi]

        conv_vals = []
        rest_vals = []
        labels = []

        for s in sessions:
            pc = s['per_condition']
            conv_z = []
            rest_z = []

            for cond in ['conv_1', 'conv_2']:
                if cond in pc and band in pc[cond]:
                    conv_z.append(pc[cond][band]['mean_z'])
            for cond in ['base_EO', 'base_EC', 'meditate_B', 'meditate_K']:
                if cond in pc and band in pc[cond]:
                    rest_z.append(pc[cond][band]['mean_z'])

            if conv_z and rest_z:
                conv_vals.append(np.mean(conv_z))
                rest_vals.append(np.mean(rest_z))
                labels.append(s['session'])

        if not conv_vals:
            continue

        conv_vals = np.array(conv_vals)
        rest_vals = np.array(rest_vals)

        ax.scatter(rest_vals, conv_vals, c=BAND_COLORS[band], s=60,
                   alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)

        # Label each point
        for i, lab in enumerate(labels):
            short = lab.replace('_0', '').replace('_', '')[:6]
            ax.annotate(short, (rest_vals[i], conv_vals[i]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.1,
                max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.1]
        ax.plot(lims, lims, 'k--', lw=0.5, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # How many above identity line
        n_above = int((conv_vals > rest_vals).sum())
        ax.set_title(f'{band.upper()} ({n_above}/{len(conv_vals)} conv > rest)',
                      fontsize=11, fontweight='bold', color=BAND_COLORS[band])
        ax.set_xlabel('Rest/Baseline z', fontsize=9, fontweight='bold')
        if bi == 0:
            ax.set_ylabel('Conversation z', fontsize=9, fontweight='bold')
        ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.3)
        ax.axvline(0, color='gray', ls=':', lw=0.5, alpha=0.3)
        ax.tick_params(labelsize=7)

    n_sess = len(sessions)
    fig.suptitle(f'Conversation vs Rest Burst Coincidence (n={n_sess} sessions)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, 'conv_vs_rest_scatter.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    state_labels = _load_state_labels()
    print(f"Loaded state labels (index order): {state_labels}")
    print()

    print("Generating burst coincidence visualizations...")
    print()

    print("[1/4] y_06 timeline...")
    plot_y06_timeline(state_labels)

    print("[2/4] Per-condition bars...")
    plot_condition_bars()

    print("[3/4] Per-state comparison...")
    plot_state_comparison(state_labels)

    print("[4/4] Conversation vs rest scatter...")
    plot_conv_vs_rest_scatter()

    print("\nDone.")
