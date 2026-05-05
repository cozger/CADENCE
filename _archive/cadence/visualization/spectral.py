"""Spectral coupling visualizations: frequency x ROI maps and coupling spectra."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.constants import WAVELET_CENTER_FREQS, EEG_ROI_NAMES


def plot_spectral_coupling_map(freqs, rois, coupling_strength,
                                save_path=None, figsize=(10, 6), dpi=150,
                                title='Spectral Coupling Map'):
    """Plot frequency x ROI heatmap of coupling strength.

    The "coupling spectrum" — shows which frequencies and spatial locations
    carry interpersonal influence.

    Args:
        freqs: (n_freqs,) center frequencies
        rois: list of ROI names
        coupling_strength: (n_freqs, n_rois) coupling values (dR2 or coefficients)
        save_path: path to save figure (optional)
        figsize: figure dimensions
        dpi: resolution
        title: plot title

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(np.abs(coupling_strength).max(), 1e-4)
    im = ax.imshow(coupling_strength, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, origin='lower',
                   extent=[-0.5, len(rois) - 0.5, 0, len(freqs) - 1])

    # Y-axis: frequencies (log-spaced, show actual Hz values)
    n_ticks = min(10, len(freqs))
    tick_idx = np.linspace(0, len(freqs) - 1, n_ticks, dtype=int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f'{freqs[i]:.1f}' for i in tick_idx], fontsize=9)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)

    # X-axis: ROIs
    ax.set_xticks(range(len(rois)))
    roi_labels = [r.replace('_', '\n') for r in rois]
    ax.set_xticklabels(roi_labels, fontsize=10)
    ax.set_xlabel('ROI', fontsize=11)

    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label='Coupling Strength', shrink=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_coupling_spectrum(freqs, strength, roi_name='all',
                            confidence_lower=None, confidence_upper=None,
                            save_path=None, figsize=(8, 5), dpi=150,
                            title=None):
    """Line plot of coupling strength vs frequency for one ROI.

    Args:
        freqs: (n_freqs,) center frequencies in Hz
        strength: (n_freqs,) coupling strength values
        roi_name: name of ROI (for labeling)
        confidence_lower: (n_freqs,) lower confidence bound (optional)
        confidence_upper: (n_freqs,) upper confidence bound (optional)
        save_path: path to save figure
        figsize: figure dimensions
        dpi: resolution
        title: plot title (auto-generated if None)

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(freqs, strength, 'b-', linewidth=2, label=roi_name)

    if confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(freqs, confidence_lower, confidence_upper,
                        alpha=0.2, color='blue')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Coupling Strength (dR²)', fontsize=11)

    if title is None:
        title = f'Coupling Spectrum — {roi_name}'
    ax.set_title(title, fontsize=13)

    # Mark conventional band boundaries
    for f, label in [(4, 'θ'), (8, 'α'), (13, 'β'), (30, 'γ')]:
        if freqs[0] <= f <= freqs[-1]:
            ax.axvline(x=f, color='gray', linestyle=':', alpha=0.3)
            ax.text(f, ax.get_ylim()[1] * 0.95, label,
                    ha='center', fontsize=8, color='gray')

    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def extract_spectral_map(discovery_result, pathway, n_freqs=20, n_rois=4):
    """Extract frequency x ROI coupling map from discovery coefficients.

    Reshapes the group lasso coefficient norms into a (n_freqs, n_rois) array.

    Args:
        discovery_result: DiscoveryResult with coefficients
        pathway: (src_mod, tgt_mod) tuple
        n_freqs: number of frequency bins
        n_rois: number of ROIs

    Returns:
        coupling_map: (n_freqs, n_rois) array of coefficient norms
    """
    if pathway not in discovery_result.coefficients:
        return np.zeros((n_freqs, n_rois))

    beta = discovery_result.coefficients[pathway]
    # beta shape: (p, C_tgt) where p includes basis-convolved features
    # Source features are ordered: for each component (real/imag),
    #   for each freq, for each ROI -> 2 * n_freqs * n_rois source features

    n_source = 2 * n_freqs * n_rois  # 160 wavelet features
    if beta.shape[0] < n_source:
        return np.zeros((n_freqs, n_rois))

    # Group norm across basis coefficients and target channels
    # Assume groups of n_basis columns per source feature
    n_basis = beta.shape[0] // n_source if n_source > 0 else 1
    coupling_map = np.zeros((n_freqs, n_rois))

    for f_idx in range(n_freqs):
        for r_idx in range(n_rois):
            # Collect norms for real and imag components at this freq-ROI
            total_norm = 0.0
            for comp_idx in range(2):
                feat_idx = comp_idx * (n_freqs * n_rois) + f_idx * n_rois + r_idx
                start_col = feat_idx * n_basis
                end_col = start_col + n_basis
                if end_col <= beta.shape[0]:
                    group_beta = beta[start_col:end_col]
                    total_norm += np.linalg.norm(group_beta)
            coupling_map[f_idx, r_idx] = total_norm

    return coupling_map
