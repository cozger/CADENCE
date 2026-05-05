"""V6 visualizations: facial event timelines + EEG coupling bars.

Functions:
  plot_shared_smile_timeline  — timeline of shared smiles with confidence
  plot_eeg_condition_bars     — per-condition volt_amp z-scores
  plot_session_dashboard      — combined EEG + BL summary
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_shared_smile_timeline(catalog, title='', save_path=None):
    """Plot timeline of facial events with shared smiles highlighted.

    Shows therapist and patient event timelines with connecting arcs
    for shared smiles, colored by confidence.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    dur = catalog.duration_s

    # Plot all events as dots
    for ev in catalog.events_p1:
        color = 'tab:orange' if ev.smile_composite > 0.3 else 'lightgray'
        ax.plot(ev.time, 1.0, 'o', color=color, markersize=3, alpha=0.5)

    for ev in catalog.events_p2:
        color = 'tab:blue' if ev.smile_composite > 0.3 else 'lightgray'
        ax.plot(ev.time, 0.0, 'o', color=color, markersize=3, alpha=0.5)

    # Highlight shared smiles with connecting lines
    for ss in catalog.shared_smiles:
        conf = ss.joint_smile_confidence
        alpha = min(1.0, conf * 3)  # scale confidence to visibility
        lw = 1 + conf * 4

        ax.plot([ss.event_a.time, ss.event_b.time], [1.0, 0.0],
                '-', color='red', alpha=alpha, linewidth=lw)

        # Mark the events
        ax.plot(ss.event_a.time, 1.0, 'o', color='red',
                markersize=4 + conf * 8, alpha=alpha)
        ax.plot(ss.event_b.time, 0.0, 'o', color='red',
                markersize=4 + conf * 8, alpha=alpha)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Therapist', 'Patient'])
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, dur)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title(title or f'{catalog.segment_name}: {catalog.n_shared_smiles} shared smiles')

    # Legend
    ax.plot([], [], 'o', color='tab:orange', label='Patient smile', markersize=5)
    ax.plot([], [], 'o', color='tab:blue', label='Therapist smile', markersize=5)
    ax.plot([], [], '-', color='red', label='Shared smile', linewidth=2)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_eeg_condition_bars(session_results, title='', save_path=None):
    """Bar chart of EEG volt_amp z-scores per condition.

    Args:
        session_results: dict from v6_results.json with 'segments' key.
    """
    segments = session_results.get('segments', {})

    conditions = []
    z_theta = []
    z_alpha = []
    z_beta = []
    z_combined = []

    for seg_name in ['conv_1', 'conv_2', 'meditate_K', 'meditate_B',
                     'base_EO', 'base_EC']:
        seg = segments.get(seg_name, {})
        eeg = seg.get('eeg')
        if eeg is None:
            continue
        conditions.append(seg_name)
        z_theta.append(eeg.get('theta_volt_amp_z', 0))
        z_alpha.append(eeg.get('alpha_volt_amp_z', 0))
        z_beta.append(eeg.get('beta_volt_amp_z', 0))
        z_combined.append(eeg.get('combined_volt_amp_z', 0))

    if not conditions:
        return None

    x = np.arange(len(conditions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, z_theta, width, label='Theta', color='#2196F3')
    ax.bar(x - 0.5 * width, z_alpha, width, label='Alpha', color='#4CAF50')
    ax.bar(x + 0.5 * width, z_beta, width, label='Beta', color='#FF9800')
    ax.bar(x + 1.5 * width, z_combined, width, label='Combined', color='#F44336')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha='right')
    ax.set_ylabel('Volt_amp z-score')
    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='z=2')
    ax.legend()
    ax.set_title(title or 'EEG Amplitude Coupling by Condition')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_session_dashboard(session_results, bl_catalogs=None,
                           title='', save_path=None):
    """Combined EEG + BL dashboard for one session.

    Args:
        session_results: dict from v6_results.json.
        bl_catalogs: dict[segment_name] -> FacialEventCatalog (optional,
            for timeline plots). If None, only shows summary stats.
    """
    segments = session_results.get('segments', {})
    n_segs = len(segments)
    if n_segs == 0:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1.5]})

    # Top: EEG bars
    ax_eeg = axes[0]
    conditions = []
    z_vals = []
    for seg_name in ['conv_1', 'conv_2', 'meditate_K', 'meditate_B',
                     'base_EO', 'base_EC']:
        seg = segments.get(seg_name, {})
        eeg = seg.get('eeg')
        if eeg is None:
            continue
        conditions.append(seg_name)
        z_vals.append(eeg.get('combined_volt_amp_z', 0))

    if conditions:
        colors = ['#F44336' if z > 2 else '#9E9E9E' for z in z_vals]
        ax_eeg.bar(conditions, z_vals, color=colors)
        ax_eeg.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
        ax_eeg.set_ylabel('EEG volt_amp z')
        ax_eeg.set_title('EEG Inter-Brain Amplitude Coupling')
    else:
        ax_eeg.text(0.5, 0.5, 'No EEG data', transform=ax_eeg.transAxes,
                    ha='center', va='center')

    # Bottom: BL summary
    ax_bl = axes[1]
    bl_conditions = []
    bl_smiles = []
    bl_events = []

    for seg_name in ['conv_1', 'conv_2', 'meditate_K', 'meditate_B',
                     'base_EO', 'base_EC']:
        seg = segments.get(seg_name, {})
        bl = seg.get('bl')
        if bl is None:
            continue
        bl_conditions.append(seg_name)
        bl_smiles.append(bl['n_shared_smiles'])
        bl_events.append(bl['n_events_p1'] + bl['n_events_p2'])

    if bl_conditions:
        x = np.arange(len(bl_conditions))
        ax_bl.bar(x, bl_smiles, color='#E91E63', label='Shared smiles')
        ax_bl.set_xticks(x)
        ax_bl.set_xticklabels(bl_conditions, rotation=30, ha='right')
        ax_bl.set_ylabel('Count')
        ax_bl.set_title('Shared Facial Events')
        ax_bl.legend()
    else:
        ax_bl.text(0.5, 0.5, 'No BL data', transform=ax_bl.transAxes,
                    ha='center', va='center')

    session = session_results.get('session', '')
    roles = f"{session_results.get('p1_role', '?')} / {session_results.get('p2_role', '?')}"
    fig.suptitle(title or f'{session} ({roles})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
