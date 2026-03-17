"""Doubly-sparse feature selection visualization.

Shows which source features were selected by the Stage 1 doubly-sparse
procedure: stability scores, block hit counts, and final selection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cadence.constants import MOD_SHORT, MOD_SHORT_V2, MODALITY_COLORS, MODALITY_COLORS_V2


def _short(mod, mod_short=None):
    if mod_short is None:
        mod_short = MOD_SHORT
    return mod_short.get(mod, MOD_SHORT_V2.get(mod, mod))


def plot_sparsity_summary(result, save_path=None, figsize=(16, 10), dpi=150):
    """Plot doubly-sparse selection summary for all pathways.

    For each pathway shows: total grouped features, prescreened, stable,
    block-selected, intersection, and final selected count.
    Color-codes by selection method and significance.
    """
    discovery = getattr(result, 'discovery', None)
    if discovery is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No discovery data (V1 pipeline)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig

    # Collect pathways with data
    pathways = []
    for key in sorted(discovery.selection_method.keys()):
        src, tgt = key
        method = discovery.selection_method.get(key, 'none')
        n_sel = discovery.n_selected.get(key, 0)
        sig = result.pathway_significant.get(key, False)
        pw_p = discovery.block_pathway_pvalue.get(key, 1.0)
        pathways.append((key, method, n_sel, sig, pw_p))

    if not pathways:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No doubly-sparse pathways',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig

    # --- Panel 1: Pathway selection overview ---
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={'width_ratios': [2, 3]})

    # Left panel: pathway-level bar chart
    ax = axes[0]
    labels = []
    n_selected = []
    colors = []
    edge_colors = []
    method_labels = []

    method_color = {
        'intersection': '#2196F3',
        'secondary': '#FF9800',
        'tertiary': '#9C27B0',
        'none': '#BDBDBD',
    }

    for key, method, n_sel, sig, pw_p in pathways:
        src, tgt = key
        label = f'{_short(src)}\u2192{_short(tgt)}'
        labels.append(label)
        n_selected.append(n_sel)
        c = method_color.get(method, '#BDBDBD')
        colors.append(c if sig else '#E0E0E0')
        edge_colors.append('black' if sig else '#BDBDBD')
        method_labels.append(method if n_sel > 0 else '')

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, n_selected, color=colors, edgecolor=edge_colors,
                   linewidth=0.8)

    # Annotate with method and p-value
    for i, (key, method, n_sel, sig, pw_p) in enumerate(pathways):
        if n_sel > 0:
            ax.text(n_sel + 0.5, i, f'{method} (p={pw_p:.3f})',
                    va='center', fontsize=7, style='italic',
                    color='black' if sig else '#999999')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Features Selected', fontsize=10)
    ax.set_title('Stage 1: Feature Selection', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # Legend for methods
    legend_items = [Patch(facecolor=c, edgecolor='black', label=m)
                    for m, c in method_color.items() if m != 'none']
    legend_items.append(Patch(facecolor='#E0E0E0', edgecolor='#BDBDBD',
                              label='not sig'))
    ax.legend(handles=legend_items, loc='lower right', fontsize=7,
              title='Selection method', title_fontsize=8)

    # --- Right panel: per-feature stability heatmap for significant pathways ---
    ax2 = axes[1]
    sig_pathways = [(key, method, n_sel) for key, method, n_sel, sig, pw_p
                    in pathways if sig and n_sel > 0]

    if sig_pathways:
        # Build heatmap: rows = pathways, cols = feature indices
        max_features = max(max(discovery.selected_features.get(k, [0]))
                          for k, _, _ in sig_pathways) + 1
        # Use stability scores if available, else binary
        heatmap_rows = []
        row_labels = []
        for key, method, n_sel in sig_pathways:
            src, tgt = key
            row_labels.append(f'{_short(src)}\u2192{_short(tgt)}')
            scores = discovery.stability_scores.get(key, np.array([]))
            if len(scores) > 0:
                heatmap_rows.append(scores)
            else:
                selected = discovery.selected_features.get(key, [])
                row = np.zeros(max_features)
                for idx in selected:
                    if idx < max_features:
                        row[idx] = 1.0
                heatmap_rows.append(row)

        if heatmap_rows:
            # Pad rows to same length
            max_len = max(len(r) for r in heatmap_rows)
            padded = np.zeros((len(heatmap_rows), max_len))
            for i, row in enumerate(heatmap_rows):
                padded[i, :len(row)] = row

            im = ax2.imshow(padded, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=1.0, interpolation='nearest')

            # Mark selected features
            for i, (key, method, n_sel) in enumerate(sig_pathways):
                selected = discovery.selected_features.get(key, [])
                for idx in selected:
                    if idx < max_len:
                        ax2.plot(idx, i, 'k.', markersize=3)

            ax2.set_yticks(range(len(row_labels)))
            ax2.set_yticklabels(row_labels, fontsize=8)
            ax2.set_xlabel('Grouped Feature Index', fontsize=10)
            ax2.set_title('Stability Scores (selected = dot)', fontsize=11,
                         fontweight='bold')
            plt.colorbar(im, ax=ax2, label='Stability Score', shrink=0.6)
    else:
        ax2.text(0.5, 0.5, 'No significant pathways with features',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Stability Scores', fontsize=11, fontweight='bold')

    fig.suptitle(f'Doubly-Sparse Selection \u2014 {result.direction}',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_block_detail(result, save_path=None, figsize=(14, 8), dpi=150):
    """Plot block-level detail for significant pathways.

    Shows block hit counts vs surrogate-calibrated threshold for each
    significant pathway.
    """
    discovery = getattr(result, 'discovery', None)
    if discovery is None:
        return None

    sig_keys = [k for k, sig in result.pathway_significant.items()
                if sig and k in discovery.selected_features
                and len(discovery.selected_features[k]) > 0]

    if not sig_keys:
        return None

    n = len(sig_keys)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i, key in enumerate(sig_keys):
        ax = axes[i]
        src, tgt = key
        method = discovery.selection_method.get(key, '?')
        n_blocks = discovery.n_blocks.get(key, 0)
        selected = discovery.selected_features.get(key, [])
        block_hits = discovery.block_hit_counts.get(key, np.array([]))

        if len(block_hits) == 0:
            ax.text(0.5, 0.5, 'No block data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{_short(src)}\u2192{_short(tgt)}')
            continue

        n_groups = len(block_hits)
        x = np.arange(n_groups)

        # Color selected features differently
        bar_colors = ['#2196F3' if idx in selected else '#E0E0E0'
                      for idx in range(n_groups)]

        ax.bar(x, block_hits, color=bar_colors, edgecolor='none', width=0.8)

        # Mark significance threshold
        if n_blocks > 0:
            from scipy.stats import binom
            thresh = binom.ppf(0.95, n_blocks, 0.5)
            ax.axhline(thresh, color='red', linestyle='--', linewidth=1,
                       alpha=0.7, label=f'p<0.05 thresh ({thresh:.0f}/{n_blocks})')

        ax.set_xlabel('Feature group', fontsize=9)
        ax.set_ylabel('Block hits', fontsize=9)
        ax.set_title(f'{_short(src)}\u2192{_short(tgt)}  [{method}]  '
                     f'({len(selected)} sel / {n_groups} groups)',
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')

    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Block-Level Feature Hits \u2014 {result.direction}',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig
