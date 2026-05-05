"""Coupling kernel h(s) visualization: impulse response over lag."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.constants import (
    MODALITY_COLORS, MOD_SHORT,
    MODALITY_COLORS_V2, MOD_SHORT_V2,
)


def plot_coupling_kernels(result, save_path=None, figsize=(16, 8), dpi=150):
    """Plot time-averaged coupling kernels h(s) for each significant pathway.

    Shows the impulse response function: how a unit change in source
    at time t propagates to target over the next few seconds.

    Args:
        result: CouplingResult from CouplingEstimator.
        save_path: Path to save figure (optional).
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        fig: matplotlib Figure.
    """
    sig_pathways = {k: v for k, v in result.pathway_kernels.items()
                    if result.pathway_significant.get(k, False)}

    if not sig_pathways:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No significant pathways detected',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig

    n_paths = len(sig_pathways)
    n_cols = min(3, n_paths)
    n_rows = (n_paths + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_paths == 1:
        axes = [axes]
    elif n_rows > 1:
        axes = axes.flatten()

    from cadence.basis.raised_cosine import raised_cosine_basis

    for i, ((src, tgt), kernel) in enumerate(sig_pathways.items()):
        if i >= len(axes):
            break
        ax = axes[i]

        # Time-averaged kernel
        mean_kernel = np.nanmean(kernel, axis=0)
        std_kernel = np.nanstd(kernel, axis=0)
        n_lags = kernel.shape[1]
        lags = np.linspace(0, 5.0, n_lags)

        color = MODALITY_COLORS.get(src, MODALITY_COLORS_V2.get(src, '#333333'))
        src_name = MOD_SHORT.get(src, MOD_SHORT_V2.get(src, src))
        tgt_name = MOD_SHORT.get(tgt, MOD_SHORT_V2.get(tgt, tgt))
        ax.plot(lags, mean_kernel, color=color, linewidth=2)
        ax.fill_between(lags, mean_kernel - std_kernel, mean_kernel + std_kernel,
                        color=color, alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('h(s)')
        ax.set_title(f'{src_name} \u2192 {tgt_name}')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Coupling Kernels - {result.direction}', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
