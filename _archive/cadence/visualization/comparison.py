"""CADENCE vs MCCT side-by-side comparison visualization."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.constants import MODALITY_ORDER, MODALITY_COLORS, MOD_SHORT


def plot_cadence_vs_mcct(cadence_result, mcct_json_path, save_path=None,
                          figsize=(16, 10), dpi=150):
    """Side-by-side comparison of CADENCE and MCCT coupling timecourses.

    Args:
        cadence_result: CouplingResult from CADENCE.
        mcct_json_path: Path to MCCT timecourse JSON file.
        save_path: Output path.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        fig: matplotlib Figure.
    """
    # Load MCCT results
    with open(mcct_json_path) as f:
        mcct_data = json.load(f)

    mcct_windows = mcct_data if isinstance(mcct_data, list) else mcct_data.get('windows', [])

    # Extract MCCT timecourse per modality
    mcct_times = np.array([w['time'] for w in mcct_windows]) / 60.0
    mcct_dr2 = {}
    for mod in MODALITY_ORDER:
        dr2_vals = []
        for w in mcct_windows:
            r2_real = w.get('r2_real', {}).get(mod, 0)
            r2_surr = w.get('r2_surrogate', {}).get(mod, 0)
            dr2_vals.append(r2_real - r2_surr)
        if dr2_vals:
            mcct_dr2[mod] = np.array(dr2_vals)

    # CADENCE timecourse: average across targets for each source modality
    cadence_times = cadence_result.times / 60.0
    cadence_dr2 = {}
    for src_mod in MODALITY_ORDER:
        dr2_list = []
        for tgt_mod in MODALITY_ORDER:
            key = (src_mod, tgt_mod)
            if key in cadence_result.pathway_dr2 and src_mod != tgt_mod:
                dr2_list.append(np.nan_to_num(cadence_result.pathway_dr2[key], nan=0.0))
        if dr2_list:
            cadence_dr2[src_mod] = np.mean(dr2_list, axis=0)

    # Plot side by side
    mods_with_data = [m for m in MODALITY_ORDER if m in mcct_dr2 or m in cadence_dr2]
    n_mods = len(mods_with_data)

    if n_mods == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No comparable data', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(n_mods, 2, figsize=figsize, sharex=True)
    if n_mods == 1:
        axes = axes.reshape(1, 2)

    for i, mod in enumerate(mods_with_data):
        color = MODALITY_COLORS.get(mod, '#333333')
        label = MOD_SHORT.get(mod, mod)

        # MCCT (left)
        ax_mcct = axes[i, 0]
        if mod in mcct_dr2:
            ax_mcct.plot(mcct_times, mcct_dr2[mod], color=color, linewidth=1)
            ax_mcct.fill_between(mcct_times, 0, mcct_dr2[mod],
                                  where=mcct_dr2[mod] > 0, color=color, alpha=0.2)
        ax_mcct.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_mcct.set_ylabel(f'{label} dR2')
        if i == 0:
            ax_mcct.set_title('MCCT (Transformer)', fontsize=12)

        # CADENCE (right)
        ax_cad = axes[i, 1]
        if mod in cadence_dr2:
            ax_cad.plot(cadence_times, cadence_dr2[mod], color=color, linewidth=1)
            ax_cad.fill_between(cadence_times, 0, cadence_dr2[mod],
                                 where=cadence_dr2[mod] > 0, color=color, alpha=0.2)
        ax_cad.axhline(0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax_cad.set_title('CADENCE (Regression)', fontsize=12)

    axes[-1, 0].set_xlabel('Time (min)')
    axes[-1, 1].set_xlabel('Time (min)')

    fig.suptitle(f'CADENCE vs MCCT - {cadence_result.direction}', fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def compute_correlation(cadence_result, mcct_json_path):
    """Compute correlation between CADENCE and MCCT dR2 timecourses.

    Args:
        cadence_result: CouplingResult from CADENCE.
        mcct_json_path: Path to MCCT timecourse JSON file.

    Returns:
        correlations: dict mapping modality -> Pearson r.
    """
    with open(mcct_json_path) as f:
        mcct_data = json.load(f)

    mcct_windows = mcct_data if isinstance(mcct_data, list) else mcct_data.get('windows', [])
    mcct_times = np.array([w['time'] for w in mcct_windows])

    correlations = {}
    for mod in MODALITY_ORDER:
        # MCCT dR2
        mcct_dr2 = []
        for w in mcct_windows:
            r2_real = w.get('r2_real', {}).get(mod, 0)
            r2_surr = w.get('r2_surrogate', {}).get(mod, 0)
            mcct_dr2.append(r2_real - r2_surr)
        mcct_dr2 = np.array(mcct_dr2)

        # CADENCE dR2 (average across targets)
        dr2_list = []
        for tgt in MODALITY_ORDER:
            key = (mod, tgt)
            if key in cadence_result.pathway_dr2 and mod != tgt:
                dr2_list.append(np.nan_to_num(cadence_result.pathway_dr2[key], nan=0.0))
        if not dr2_list:
            continue
        cadence_dr2 = np.mean(dr2_list, axis=0)

        # Interpolate CADENCE to MCCT times for comparison
        cadence_interp = np.interp(mcct_times, cadence_result.times, cadence_dr2)

        # Pearson correlation
        if len(cadence_interp) > 5 and np.std(cadence_interp) > 1e-10 and np.std(mcct_dr2) > 1e-10:
            r = np.corrcoef(cadence_interp, mcct_dr2)[0, 1]
            correlations[mod] = float(r)

    return correlations
