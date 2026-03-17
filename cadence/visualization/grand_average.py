"""Grand average visualizations: condition-level coupling across sessions.

Replicates MCCT's grand average plots:
1. Stacked classification bars (sync/T-leads/P-leads/none per condition)
2. dR2 effect size bars (T->P vs P->T per condition)
3. Grand coupling matrix (4x4 source->target heatmap)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.conditions import CONDITION_LABELS, CONDITION_COLORS


# Direction colors (matches MCCT)
DIR_COLORS = {
    'tp': '#2196F3',      # blue — therapist to patient
    'pt': '#FF9800',      # orange — patient to therapist
    'tp_edge': '#1565C0',
    'pt_edge': '#E65100',
}

# Classification colors
CLASS_COLORS = {
    'sync': '#4CAF50',      # green
    't_leads': '#2196F3',   # blue
    'p_leads': '#FF9800',   # orange
    'none': '#BDBDBD',      # gray
}

# Modality display order (V1 default, V2 detected from data)
MOD_DISPLAY_V1 = ['EEG', 'ECG', 'BL', 'Pose', 'Overall']
MOD_DISPLAY_V2 = ['EEGw', 'EEGib', 'ECG', 'BL', 'Pose', 'Overall']
MOD_FULL_NAMES = {
    'EEG': 'EEG Features',
    'ECG': 'ECG Features',
    'BL': 'Blendshapes',
    'Pose': 'Pose Features',
    'Overall': 'Overall',
    'EEGw': 'EEG Wavelet',
    'EEGib': 'EEG Inter-Brain',
}


def _detect_mod_display(grand_cond):
    """Detect which modality display order to use from the data."""
    all_mods = set()
    for cond_data in grand_cond.get('conditions', {}).values():
        all_mods.update(cond_data.get('modalities', {}).keys())
    # If V2 modalities present, use V2 order
    if 'EEGw' in all_mods or 'EEGib' in all_mods:
        return [m for m in MOD_DISPLAY_V2 if m in all_mods or m == 'Overall']
    return [m for m in MOD_DISPLAY_V1 if m in all_mods or m == 'Overall']


def plot_grand_classification_bars(grand_cond, save_path=None, dpi=150):
    """Stacked horizontal bars: sync/T-leads/P-leads/none per condition.

    One row of subplots per modality (EEG, ECG, BL, Pose, Overall).
    Each bar = one condition, stacked by coupling classification.

    Args:
        grand_cond: Grand condition summary dict from aggregate_condition_summaries().
        save_path: Path to save figure.
        dpi: Resolution.
    """
    conditions_data = grand_cond['conditions']

    # Filter to conditions with Overall data, skip 'uncategorized'
    cond_keys = [c for c in conditions_data
                 if c != 'uncategorized'
                 and 'Overall' in conditions_data[c].get('modalities', {})]
    if not cond_keys:
        return None

    # Determine which modalities are available
    mod_display = _detect_mod_display(grand_cond)
    mod_keys = [m for m in mod_display
                if any(m in conditions_data[c].get('modalities', {})
                       for c in cond_keys)]
    n_mods = len(mod_keys)
    n_conds = len(cond_keys)

    fig, axes = plt.subplots(n_mods, 1, figsize=(12, 1.8 * n_mods + 1.5),
                              sharex=True)
    if n_mods == 1:
        axes = [axes]

    for ax_idx, mod_key in enumerate(mod_keys):
        ax = axes[ax_idx]

        # Collect data per condition
        labels = []
        sync_vals, t_vals, p_vals, none_vals = [], [], [], []

        for cond in cond_keys:
            mods = conditions_data[cond].get('modalities', {})
            if mod_key not in mods:
                continue
            m = mods[mod_key]
            n = m['n_sessions']
            cond_label = conditions_data[cond]['label']
            labels.append(f"{cond_label}\n(n={n})")
            sync_vals.append(m['sync_pct_mean'])
            t_vals.append(m['t_leads_pct_mean'])
            p_vals.append(m['p_leads_pct_mean'])
            none_vals.append(m['none_pct_mean'])

        if not labels:
            ax.set_visible(False)
            continue

        y_pos = np.arange(len(labels))
        bar_h = 0.6

        # Stacked horizontal bars
        sync_arr = np.array(sync_vals)
        t_arr = np.array(t_vals)
        p_arr = np.array(p_vals)
        none_arr = np.array(none_vals)

        ax.barh(y_pos, sync_arr, bar_h, color=CLASS_COLORS['sync'],
                label='Synchrony' if ax_idx == 0 else '')
        ax.barh(y_pos, t_arr, bar_h, left=sync_arr,
                color=CLASS_COLORS['t_leads'],
                label='Therapist leads' if ax_idx == 0 else '')
        ax.barh(y_pos, p_arr, bar_h, left=sync_arr + t_arr,
                color=CLASS_COLORS['p_leads'],
                label='Patient leads' if ax_idx == 0 else '')
        ax.barh(y_pos, none_arr, bar_h, left=sync_arr + t_arr + p_arr,
                color=CLASS_COLORS['none'],
                label='No coupling' if ax_idx == 0 else '')

        # Annotate sync % inside bar if > 5%
        for i, sv in enumerate(sync_vals):
            if sv > 5:
                ax.text(sv / 2, i, f'{sv:.0f}%', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(MOD_FULL_NAMES.get(mod_key, mod_key), fontsize=10,
                     fontweight='bold', loc='left')

    axes[-1].set_xlabel('% of timepoints', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8, ncol=4)

    n_sess = grand_cond['n_sessions']
    fig.suptitle(f'Grand Average Coupling Classification (n={n_sess} sessions)\n'
                 f'Mean across sessions; dR2 > 0 threshold',
                 fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_grand_dr2_bars(grand_cond, save_path=None, dpi=150):
    """Grouped vertical bars: mean dR2 (T->P vs P->T) per condition.

    One row per modality. Side-by-side bars with error bars (between-session SD).

    Args:
        grand_cond: Grand condition summary dict.
        save_path: Path to save figure.
        dpi: Resolution.
    """
    conditions_data = grand_cond['conditions']

    cond_keys = [c for c in conditions_data
                 if c != 'uncategorized'
                 and 'Overall' in conditions_data[c].get('modalities', {})]
    if not cond_keys:
        return None

    mod_display = _detect_mod_display(grand_cond)
    mod_keys = [m for m in mod_display
                if any(m in conditions_data[c].get('modalities', {})
                       for c in cond_keys)]
    n_mods = len(mod_keys)
    n_conds = len(cond_keys)

    fig, axes = plt.subplots(n_mods, 1,
                              figsize=(max(10, 1.8 * n_conds + 2), 2.2 * n_mods + 2),
                              sharex=True)
    if n_mods == 1:
        axes = [axes]

    x = np.arange(n_conds)
    bar_w = 0.35

    for ax_idx, mod_key in enumerate(mod_keys):
        ax = axes[ax_idx]

        tp_means, tp_stds = [], []
        pt_means, pt_stds = [], []
        cond_labels = []

        for cond in cond_keys:
            mods = conditions_data[cond].get('modalities', {})
            cond_labels.append(conditions_data[cond]['label'])
            if mod_key in mods:
                m = mods[mod_key]
                tp_means.append(m['mean_dr2_tp_mean'])
                tp_stds.append(m['mean_dr2_tp_std'])
                pt_means.append(m['mean_dr2_pt_mean'])
                pt_stds.append(m['mean_dr2_pt_std'])
            else:
                tp_means.append(0)
                tp_stds.append(0)
                pt_means.append(0)
                pt_stds.append(0)

        tp_means = np.array(tp_means)
        tp_stds = np.array(tp_stds)
        pt_means = np.array(pt_means)
        pt_stds = np.array(pt_stds)

        # Condition background shading
        for i, cond in enumerate(cond_keys):
            color = CONDITION_COLORS.get(cond, '#F5F5F5')
            ax.axvspan(i - 0.45, i + 0.45, color=color, alpha=0.15, zorder=0)

        # Grouped bars
        ax.bar(x - bar_w / 2, tp_means, bar_w, yerr=tp_stds,
               color=DIR_COLORS['tp'], edgecolor=DIR_COLORS['tp_edge'],
               linewidth=0.5, alpha=0.85, capsize=3,
               error_kw={'linewidth': 0.8},
               label='T -> P' if ax_idx == 0 else '')
        ax.bar(x + bar_w / 2, pt_means, bar_w, yerr=pt_stds,
               color=DIR_COLORS['pt'], edgecolor=DIR_COLORS['pt_edge'],
               linewidth=0.5, alpha=0.85, capsize=3,
               error_kw={'linewidth': 0.8},
               label='P -> T' if ax_idx == 0 else '')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Mean dR2', fontsize=8)
        ax.set_title(MOD_FULL_NAMES.get(mod_key, mod_key), fontsize=9,
                     fontweight='bold', loc='left')

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cond_labels, fontsize=8, rotation=20, ha='right')
    axes[0].legend(loc='upper right', fontsize=8)

    n_sess = grand_cond['n_sessions']
    fig.suptitle(f'Grand Average dR2 by Condition (n={n_sess} sessions)\n'
                 f'Error bars = between-session SD',
                 fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_grand_coupling_matrix(grand_summary, save_path=None, dpi=150):
    """Side-by-side 4x4 coupling matrices: T->P and P->T.

    Rows = source modality, Cols = target modality.
    Cell values: mean dR2 +/- SD across sessions, significance rate.

    Args:
        grand_summary: List of per-direction summary dicts from run_all_sessions.
        save_path: Path to save figure.
        dpi: Resolution.
    """
    # Detect modality keys from actual data
    all_pathway_keys = set()
    for entry in grand_summary:
        all_pathway_keys.update(entry.get('pathways', {}).keys())
    # Extract unique short names in data
    data_mods = set()
    for pkey in all_pathway_keys:
        src, tgt = pkey.split('->')
        data_mods.add(src)
        data_mods.add(tgt)

    # V2 detected if wavelet modalities present
    if 'EEGw' in data_mods or 'EEGib' in data_mods:
        mod_keys = [m for m in ['EEGw', 'EEGib', 'ECG', 'BL', 'Pose'] if m in data_mods]
        mod_labels = [MOD_FULL_NAMES.get(m, m) for m in mod_keys]
    else:
        mod_keys = [m for m in ['EEG', 'ECG', 'BL', 'Pose'] if m in data_mods]
        mod_labels = [{'EEG': 'EEG', 'ECG': 'ECG', 'BL': 'Face', 'Pose': 'Pose'}.get(m, m) for m in mod_keys]
    n = len(mod_labels)

    # Collect per-direction matrices
    directions = {
        'therapist_to_patient': {'label': 'Therapist -> Patient', 'dr2': {}, 'sig': {}},
        'patient_to_therapist': {'label': 'Patient -> Therapist', 'dr2': {}, 'sig': {}},
    }

    for entry in grand_summary:
        d = entry['direction']
        if d not in directions:
            continue
        for pkey, pdata in entry['pathways'].items():
            if pkey not in directions[d]['dr2']:
                directions[d]['dr2'][pkey] = []
                directions[d]['sig'][pkey] = []
            directions[d]['dr2'][pkey].append(pdata['mean_dr2'])
            directions[d]['sig'][pkey].append(pdata['significant'])

    panels = [(k, v) for k, v in directions.items() if v['dr2']]
    n_panels = len(panels)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for ax_idx, (dir_key, dir_data) in enumerate(panels):
        ax = axes[ax_idx]
        matrix = np.zeros((n, n))
        sd_matrix = np.zeros((n, n))
        sig_matrix = np.zeros((n, n))
        count_matrix = np.zeros((n, n))

        for i, src in enumerate(mod_keys):
            for j, tgt in enumerate(mod_keys):
                pkey = f"{src}->{tgt}"
                if pkey in dir_data['dr2']:
                    vals = dir_data['dr2'][pkey]
                    sigs = dir_data['sig'][pkey]
                    matrix[i, j] = np.mean(vals)
                    sd_matrix[i, j] = np.std(vals)
                    sig_matrix[i, j] = np.mean(sigs)
                    count_matrix[i, j] = len(vals)

        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='equal', interpolation='nearest')

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                sd = sd_matrix[i, j]
                sig = sig_matrix[i, j]
                cnt = count_matrix[i, j]

                if cnt == 0:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=7, color='gray')
                    continue

                color = 'white' if abs(val) > 0.55 * vmax else 'black'
                weight = 'bold' if sig > 0.1 else 'normal'
                text = f'dR2={val:.3f}\nSD={sd:.3f}\nsig {sig*100:.0f}%'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=7, color=color, fontweight=weight)

        ax.set_xticks(range(n))
        ax.set_xticklabels(mod_labels, fontsize=10, fontweight='bold')
        ax.set_yticks(range(n))
        ax.set_yticklabels(mod_labels, fontsize=10, fontweight='bold')
        ax.set_xlabel('Target modality', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Source modality', fontsize=10)
        ax.set_title(dir_data['label'], fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Mean dR2', shrink=0.8)

    n_sess = grand_summary[0].get('n_sessions', len(set(s['session'] for s in grand_summary)))
    fig.suptitle(f'Grand Coupling Matrix (n={len(set(s["session"] for s in grand_summary))} sessions)',
                 fontsize=13, y=1.02)

    fig.text(0.5, -0.02,
             'dR2 = variance explained by cross-participant source above autoregressive baseline.\n'
             'Positive = source causally drives target. SD = between-session standard deviation.\n'
             'sig % = fraction of sessions where pathway is significant (p < 0.05 vs surrogates).',
             ha='center', fontsize=8, style='italic', color='#555555')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_grand_coupling_by_condition(grand_cond, save_path=None, dpi=150):
    """Per-condition coupling intensity comparison across all conditions.

    Single figure: one panel per condition, showing modality-level dR2
    for both directions as grouped bars.

    Args:
        grand_cond: Grand condition summary dict.
        save_path: Path to save figure.
        dpi: Resolution.
    """
    conditions_data = grand_cond['conditions']

    # Skip uncategorized, keep conditions with data
    cond_keys = [c for c in conditions_data
                 if c not in ('uncategorized', 'all')
                 and 'Overall' in conditions_data[c].get('modalities', {})]
    if not cond_keys:
        return None

    mod_display = _detect_mod_display(grand_cond)
    mod_display_no_overall = [m for m in mod_display if m != 'Overall']
    mod_keys = [m for m in mod_display_no_overall if any(
        m in conditions_data[c].get('modalities', {}) for c in cond_keys)]
    n_mods = len(mod_keys)
    n_conds = len(cond_keys)

    fig, axes = plt.subplots(1, n_conds, figsize=(3.5 * n_conds, 5), sharey=True)
    if n_conds == 1:
        axes = [axes]

    x = np.arange(n_mods)
    bar_w = 0.35

    for ax_idx, cond in enumerate(cond_keys):
        ax = axes[ax_idx]
        mods = conditions_data[cond].get('modalities', {})
        n_sess = mods.get('Overall', {}).get('n_sessions', 0)

        tp_vals, pt_vals = [], []
        tp_errs, pt_errs = [], []
        for mk in mod_keys:
            if mk in mods:
                tp_vals.append(mods[mk]['mean_dr2_tp_mean'])
                pt_vals.append(mods[mk]['mean_dr2_pt_mean'])
                tp_errs.append(mods[mk]['mean_dr2_tp_std'])
                pt_errs.append(mods[mk]['mean_dr2_pt_std'])
            else:
                tp_vals.append(0)
                pt_vals.append(0)
                tp_errs.append(0)
                pt_errs.append(0)

        ax.bar(x - bar_w / 2, tp_vals, bar_w, yerr=tp_errs,
               color=DIR_COLORS['tp'], edgecolor=DIR_COLORS['tp_edge'],
               linewidth=0.5, alpha=0.85, capsize=3,
               error_kw={'linewidth': 0.8},
               label='T->P' if ax_idx == 0 else '')
        ax.bar(x + bar_w / 2, pt_vals, bar_w, yerr=pt_errs,
               color=DIR_COLORS['pt'], edgecolor=DIR_COLORS['pt_edge'],
               linewidth=0.5, alpha=0.85, capsize=3,
               error_kw={'linewidth': 0.8},
               label='P->T' if ax_idx == 0 else '')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(mod_keys, fontsize=8, rotation=0)

        cond_color = CONDITION_COLORS.get(cond, '#F5F5F5')
        ax.set_facecolor(cond_color + '22')  # very light tint

        label = conditions_data[cond]['label']
        ax.set_title(f'{label}\n(n={n_sess})', fontsize=9, fontweight='bold')

    axes[0].set_ylabel('Mean dR2', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8)

    fig.suptitle('Coupling Intensity by Condition and Modality',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig
