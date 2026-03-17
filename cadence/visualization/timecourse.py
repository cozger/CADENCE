"""Time-varying dR2 trajectory visualization — focused on detected coupling."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from cadence.constants import (
    MODALITY_ORDER, MODALITY_COLORS, MOD_SHORT,
    MODALITY_ORDER_V2, MODALITY_COLORS_V2, MOD_SHORT_V2, INTERBRAIN_MODALITY,
)


def _get_mod_info(result):
    """Detect V1/V2 and return (src_order, colors, short_names)."""
    v2_only = {m for m in MODALITY_ORDER_V2
               if m not in MODALITY_ORDER} | {INTERBRAIN_MODALITY}
    all_mods = set()
    for src, tgt in result.pathway_dr2:
        all_mods.add(src)
        all_mods.add(tgt)
    if all_mods & v2_only:
        return (MODALITY_ORDER_V2 + [INTERBRAIN_MODALITY],
                MODALITY_COLORS_V2, MOD_SHORT_V2)
    return MODALITY_ORDER, MODALITY_COLORS, MOD_SHORT


def _plot_sig_shaded(ax, times, dr2, pvalues, color, alpha_thresh=0.05):
    """Plot dR2 timecourse with significant timepoints bold, rest faded.

    Significant regions (p < alpha_thresh) are drawn in full opacity with
    filled area; non-significant regions are drawn as thin faded lines.
    """
    sig_mask = pvalues < alpha_thresh

    # Background: full timecourse faded
    ax.plot(times, dr2, color=color, linewidth=0.4, alpha=0.25)

    # Significant regions: bold line + filled area
    dr2_sig = np.where(sig_mask, dr2, np.nan)
    ax.plot(times, dr2_sig, color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(times, 0, dr2, where=sig_mask,
                    color=color, alpha=0.25, linewidth=0)

    # Compute sig fraction for annotation
    sig_frac = sig_mask.sum() / max(len(sig_mask), 1)
    return sig_frac


def plot_coupling_timecourse(result, save_path=None, figsize=(16, 10), dpi=150,
                              smooth_window=None):
    """Plot dR2 timecourse for significant pathways only.

    Each significant pathway gets its own subplot. Timepoints where
    p < 0.05 are drawn bold with filled area; non-significant timepoints
    are shown as faded thin lines.
    """
    src_order, mod_colors, mod_short = _get_mod_info(result)

    # Collect significant pathways
    sig_keys = [(src, tgt) for (src, tgt), sig
                in result.pathway_significant.items()
                if sig and (src, tgt) in result.pathway_dr2]

    if not sig_keys:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No significant pathways detected',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig

    n = len(sig_keys)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], 2.5 * n + 1),
                              sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (src, tgt) in zip(axes, sig_keys):
        key = (src, tgt)
        dr2 = result.pathway_dr2[key]
        dr2_clean = np.nan_to_num(dr2, nan=0.0)

        # Use per-pathway time grid when available
        if hasattr(result, 'pathway_times') and key in result.pathway_times:
            pw_times = result.pathway_times[key] / 60.0
        else:
            pw_times = result.times / 60.0

        # Get per-timepoint p-values
        pvalues = result.pathway_pvalues.get(key, np.zeros_like(dr2))

        if smooth_window and smooth_window > 0:
            window = max(int(smooth_window * 2), 2)
            kern = np.ones(window) / window
            dr2_clean = np.convolve(dr2_clean, kern, mode='same')

        color = mod_colors.get(src, '#333333')
        src_name = mod_short.get(src, src)
        tgt_name = mod_short.get(tgt, tgt)
        mean_dr2 = float(np.nanmean(dr2))

        # Plot with significant timepoints highlighted
        sig_frac = _plot_sig_shaded(ax, pw_times, dr2_clean, pvalues, color)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.set_ylabel('dR2', fontsize=9)
        ax.set_title(
            f'{src_name} \u2192 {tgt_name}   '
            f'(mean dR2 = {mean_dr2:.4f}, sig = {sig_frac:.0%})',
            fontsize=10, loc='left', fontweight='bold')

        # Show per-feature breakdown if available
        if hasattr(result, 'pathway_feature_dr2') and key in result.pathway_feature_dr2:
            feat_dict = result.pathway_feature_dr2[key]
            if len(feat_dict) > 1:
                items = sorted(feat_dict.items(),
                               key=lambda x: np.nanmean(x[1]), reverse=True)
                cmap = plt.cm.Set2
                for rank, (feat_idx, feat_dr2) in enumerate(items[:5]):
                    feat_dr2_clean = np.nan_to_num(feat_dr2, nan=0.0)
                    if smooth_window and smooth_window > 0:
                        feat_dr2_clean = np.convolve(feat_dr2_clean, kern, mode='same')
                    # Feature lines: bold where pathway is significant, faded elsewhere
                    c = cmap(rank / 5)
                    feat_sig = np.where(pvalues < 0.05, feat_dr2_clean, np.nan)
                    feat_nonsig = np.where(pvalues >= 0.05, feat_dr2_clean, np.nan)
                    ax.plot(pw_times, feat_nonsig, color=c,
                            linewidth=0.4, alpha=0.2)
                    ax.plot(pw_times, feat_sig, color=c,
                            linewidth=0.8, alpha=0.7, label=f'ch{feat_idx}')
                ax.legend(loc='upper right', fontsize=6, ncol=3,
                          title='top features', title_fontsize=6)

    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Detected Coupling Timecourses \u2014 {result.direction}',
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_overall_timecourse(result, save_path=None, figsize=(16, 4), dpi=150):
    """Plot overall (average cross-modal) dR2 timecourse."""
    if result.overall_dr2 is None:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    times = result.times / 60.0

    ax.plot(times, result.overall_dr2, color='#333333', linewidth=1.5)
    ax.fill_between(times, 0, result.overall_dr2,
                    where=result.overall_dr2 > 0, color='steelblue', alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Overall dR2')
    ax.set_title(f'CADENCE Overall Coupling - {result.direction}')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
