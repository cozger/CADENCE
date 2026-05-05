"""Coupling matrix heatmaps: source modality -> target modality.

Supports both v1 (4x4) and v2 (expanded modality set including
eeg_wavelet, eeg_interbrain, blendshapes_v2, ecg_features_v2).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.constants import (
    MODALITY_ORDER, MOD_SHORT,
    MODALITY_ORDER_V2, MOD_SHORT_V2, INTERBRAIN_MODALITY,
)


def _detect_modality_set(result):
    """Auto-detect whether result uses v1 or v2 modality set."""
    pathway_keys = set(result.pathway_dr2.keys())
    # Only check V2-specific names (exclude pose_features which is in both)
    v2_only_mods = {m for m in MODALITY_ORDER_V2
                    if m not in MODALITY_ORDER} | {INTERBRAIN_MODALITY}
    for src, tgt in pathway_keys:
        if src in v2_only_mods or tgt in v2_only_mods:
            return 'v2'
    return 'v1'


def plot_coupling_matrix(result, save_path=None, figsize=(8, 7), dpi=150,
                          pipeline=None):
    """Plot source->target coupling matrix.

    Auto-detects v1 (4x4) vs v2 (5x4 with inter-brain source row).

    Args:
        result: CouplingResult from CouplingEstimator.
        save_path: Path to save figure.
        figsize: Figure size.
        dpi: Resolution.
        pipeline: 'v1' or 'v2' (auto-detected if None).

    Returns:
        fig: matplotlib Figure.
    """
    if pipeline is None:
        pipeline = _detect_modality_set(result)

    if pipeline == 'v2':
        # V2: participant modalities + inter-brain source row
        src_mods = MODALITY_ORDER_V2 + [INTERBRAIN_MODALITY]
        tgt_mods = MODALITY_ORDER_V2
        short = MOD_SHORT_V2
    else:
        src_mods = MODALITY_ORDER
        tgt_mods = MODALITY_ORDER
        short = MOD_SHORT

    n_src = len(src_mods)
    n_tgt = len(tgt_mods)
    matrix = np.zeros((n_src, n_tgt))
    sig_mask = np.zeros((n_src, n_tgt), dtype=bool)

    for i, src in enumerate(src_mods):
        for j, tgt in enumerate(tgt_mods):
            key = (src, tgt)
            if key in result.pathway_dr2:
                dr2 = result.pathway_dr2[key]
                matrix[i, j] = np.nanmean(dr2)
                sig_mask[i, j] = result.pathway_significant.get(key, False)

    labels_src = [short.get(m, m) for m in src_mods]
    labels_tgt = [short.get(m, m) for m in tgt_mods]

    # Use labels_tgt for x-axis, labels_src for y-axis below
    labels = labels_tgt  # keep backward compat for rest of function

    fig, ax = plt.subplots(figsize=figsize)

    # Diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal' if n_src == n_tgt else 'auto')

    # Add text annotations
    for i in range(n_src):
        for j in range(n_tgt):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            text = f'{val:.4f}'
            if sig_mask[i, j]:
                text += ' *'
            fontsize = 10 if max(n_src, n_tgt) <= 5 else 8
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize,
                    color=color, fontweight='bold' if sig_mask[i, j] else 'normal')

    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels(labels_tgt, fontsize=11)
    ax.set_yticks(range(n_src))
    ax.set_yticklabels(labels_src, fontsize=11)
    ax.set_xlabel('Target Modality', fontsize=12)
    ax.set_ylabel('Source Modality', fontsize=12)
    version_str = ' (v2)' if pipeline == 'v2' else ''
    ax.set_title(f'CADENCE Coupling Matrix{version_str} - {result.direction}',
                 fontsize=13)

    plt.colorbar(im, ax=ax, label='Mean dR2', shrink=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_feature_coupling_matrix(result, save_path=None, figsize=(12, 10), dpi=150):
    """Plot feature-level coupling matrix for Phase 2 decomposition.

    Only includes pathways with feature-level data (significant pathways
    that went through Phase 2 decomposition).
    """
    if not result.feature_dr2:
        return None

    # Collect unique feature names and target modalities
    features = sorted(set(k[0] for k in result.feature_dr2))
    targets = sorted(set(k[1] for k in result.feature_dr2))

    if not features or not targets:
        return None

    matrix = np.zeros((len(features), len(targets)))
    for i, feat in enumerate(features):
        for j, tgt in enumerate(targets):
            key = (feat, tgt)
            if key in result.feature_dr2:
                matrix[i, j] = np.nanmean(result.feature_dr2[key])

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.001)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    # Shorten labels
    feat_labels = [f.replace('eeg_', '').replace('ecg_', '').replace('pose_', '').replace('bl_', '')
                   for f in features]
    tgt_labels = [MOD_SHORT.get(t, t) for t in targets]

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(tgt_labels, fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_xlabel('Target Modality')
    ax.set_ylabel('Source Feature')
    ax.set_title(f'Feature-Level Coupling - {result.direction}')

    plt.colorbar(im, ax=ax, label='Mean dR2', shrink=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
