"""Discovery visualization: cross-session reports, lambda paths, selection heatmaps."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_discovery_report(consistency_result, save_path=None,
                           figsize=(14, 8), dpi=150):
    """Cross-session feature selection matrix.

    Rows = features, columns = sessions. Cell color = selected (green) or
    not (white). Rows sorted by consistency count (most consistent at top).

    Args:
        consistency_result: ConsistencyResult from cross_session_consistency()
        save_path: path to save figure
        figsize: figure dimensions
        dpi: resolution

    Returns:
        fig: matplotlib Figure (or None if no data)
    """
    if not consistency_result.feature_counts:
        return None

    # Pick the pathway with most features for display
    best_pathway = max(consistency_result.feature_counts.keys(),
                       key=lambda k: len(consistency_result.feature_counts[k]))
    counts = consistency_result.feature_counts[best_pathway]
    n_features = len(counts)
    n_sessions = consistency_result.n_sessions
    min_sessions = consistency_result.min_sessions

    # Sort features by count (descending)
    sorted_idx = np.argsort(-counts)
    sorted_counts = counts[sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={'width_ratios': [1, 4]})

    # Left panel: bar chart of selection counts
    ax_bar = axes[0]
    colors = ['#4CAF50' if c >= min_sessions else '#BDBDBD'
              for c in sorted_counts]
    y_pos = np.arange(min(n_features, 50))  # Show top 50
    n_show = len(y_pos)
    ax_bar.barh(y_pos, sorted_counts[:n_show], color=colors[:n_show])
    ax_bar.axvline(x=min_sessions, color='red', linestyle='--', alpha=0.7,
                   label=f'threshold={min_sessions}')
    ax_bar.set_xlabel('Sessions Selected')
    ax_bar.set_ylabel('Feature Index')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([str(sorted_idx[i]) for i in range(n_show)],
                           fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.legend(fontsize=8)

    # Right panel: text summary
    ax_text = axes[1]
    ax_text.axis('off')
    src, tgt = best_pathway
    summary_lines = [
        f'Pathway: {src} → {tgt}',
        f'Total features: {n_features}',
        f'Consistent features (≥{min_sessions}/{n_sessions} sessions): '
        f'{len(consistency_result.consistent_features.get(best_pathway, []))}',
        '',
        'All pathways:',
    ]
    for pw in sorted(consistency_result.consistent_features.keys()):
        s, t = pw
        n_cons = len(consistency_result.consistent_features[pw])
        n_tot = len(consistency_result.feature_counts.get(pw, []))
        summary_lines.append(f'  {s} → {t}: {n_cons}/{n_tot} consistent')

    ax_text.text(0.05, 0.95, '\n'.join(summary_lines),
                 transform=ax_text.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')

    fig.suptitle('Discovery Report — Cross-Session Feature Consistency',
                 fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_lambda_path(cv_scores, lambda_path, selected_counts=None,
                      save_path=None, figsize=(10, 5), dpi=150):
    """CV error vs lambda with optional feature count overlay.

    Args:
        cv_scores: (n_lambdas,) mean CV R² scores
        lambda_path: (n_lambdas,) lambda values
        selected_counts: (n_lambdas,) groups selected at each lambda (optional)
        save_path: path to save figure
        figsize: figure dimensions
        dpi: resolution

    Returns:
        fig: matplotlib Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # CV score
    ax1.plot(lambda_path, cv_scores, 'b-o', markersize=4, label='CV R²')
    best_idx = np.argmax(cv_scores)
    ax1.axvline(x=lambda_path[best_idx], color='red', linestyle='--',
                alpha=0.7, label=f'best λ={lambda_path[best_idx]:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('λ (regularization)', fontsize=11)
    ax1.set_ylabel('CV R²', fontsize=11, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Feature count on secondary axis
    if selected_counts is not None:
        ax2 = ax1.twinx()
        ax2.plot(lambda_path, selected_counts, 'g--s', markersize=3,
                 alpha=0.7, label='# groups selected')
        ax2.set_ylabel('Groups Selected', fontsize=11, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax1.legend(fontsize=9)

    ax1.set_title('Group Lasso Regularization Path', fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_feature_selection_heatmap(discovery_results, pathway,
                                    feature_names=None,
                                    save_path=None, figsize=(14, 8), dpi=150):
    """Heatmap showing which features were selected in which sessions.

    Args:
        discovery_results: list of DiscoveryResult
        pathway: (src_mod, tgt_mod) to visualize
        feature_names: list of feature names (optional)
        save_path: path to save figure
        figsize: figure dimensions
        dpi: resolution

    Returns:
        fig: matplotlib Figure (or None if no data)
    """
    # Find max feature index
    max_feat = 0
    for r in discovery_results:
        if pathway in r.selected_features:
            selected = r.selected_features[pathway]
            if selected:
                max_feat = max(max_feat, max(selected) + 1)

    if max_feat == 0:
        return None

    n_sessions = len(discovery_results)
    matrix = np.zeros((max_feat, n_sessions))

    for s_idx, r in enumerate(discovery_results):
        if pathway in r.selected_features:
            for feat_idx in r.selected_features[pathway]:
                if feat_idx < max_feat:
                    matrix[feat_idx, s_idx] = 1.0

    # Sort features by total count (most selected at top)
    counts = matrix.sum(axis=1)
    sorted_idx = np.argsort(-counts)
    matrix_sorted = matrix[sorted_idx]

    # Only show features selected at least once
    nonzero_mask = counts[sorted_idx] > 0
    if not nonzero_mask.any():
        return None
    matrix_show = matrix_sorted[nonzero_mask]
    show_idx = sorted_idx[nonzero_mask]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix_show, aspect='auto', cmap='Greens',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels([r.session_name or f'S{i}' for i, r in enumerate(discovery_results)],
                       fontsize=8, rotation=45, ha='right')

    if feature_names is not None:
        labels = [feature_names[i] if i < len(feature_names) else str(i)
                  for i in show_idx]
    else:
        labels = [str(i) for i in show_idx]

    n_show = min(len(labels), 60)
    if len(labels) > n_show:
        # Too many to label — show subset
        tick_idx = np.linspace(0, len(labels) - 1, n_show, dtype=int)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=7)
    else:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    src, tgt = pathway
    ax.set_title(f'Feature Selection: {src} → {tgt}', fontsize=13)
    plt.colorbar(im, ax=ax, label='Selected', shrink=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig
