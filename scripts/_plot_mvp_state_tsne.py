"""t-SNE companion to the UMAP plot — for honest comparison.

Same pooled subsample as the UMAP plot. Two perplexities (15 and 50) to span
local-vs-global emphasis. Identical color scheme so plots are stackable.

Usage:
    python scripts/_plot_mvp_state_tsne.py
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reuse the UMAP script's data loading helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_mvp_state_umap import (collect_pooled, stratified_subsample,
                                    plot_panel, STATE_COLOR_BY_LABEL,
                                    CONDITION_COLORS)

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'


def make_tsne(X, perplexity=30, seed=42):
    print(f'  Fitting t-SNE on shape={X.shape} (perplexity={perplexity})...')
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca',
                learning_rate='auto', random_state=seed, n_jobs=4, verbose=0)
    return tsne.fit_transform(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-points', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('=== Loading pooled data ===')
    P = collect_pooled()
    state_labels = P['state_labels']
    state_color_map = {lbl: STATE_COLOR_BY_LABEL.get(lbl, '#37474F') for lbl in state_labels}

    print('\n=== Stratified subsample ===')
    rng = np.random.default_rng(args.seed)
    keep_idx = stratified_subsample(P, args.max_points, rng)

    obs_sub  = P['obs'][keep_idx]
    lat_sub  = P['lat'][keep_idx]
    path_sub = P['path'][keep_idx]
    cond_sub = P['cond'][keep_idx]
    state_labels_per_pt = np.array([state_labels[k] for k in path_sub])

    print('\n=== Fitting t-SNE (4 variants) ===')
    print('[1/4] Observation-space, perplexity=15:')
    Y_obs_15 = make_tsne(obs_sub, perplexity=15, seed=args.seed)
    print('[2/4] Observation-space, perplexity=50:')
    Y_obs_50 = make_tsne(obs_sub, perplexity=50, seed=args.seed)
    print('[3/4] Latent-space, perplexity=15:')
    Y_lat_15 = make_tsne(lat_sub, perplexity=15, seed=args.seed)
    print('[4/4] Latent-space, perplexity=50:')
    Y_lat_50 = make_tsne(lat_sub, perplexity=50, seed=args.seed)

    print('\n=== Plotting ===')
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    plot_panel(axes[0, 0], Y_obs_15, state_labels_per_pt, state_color_map,
               'Obs-space t-SNE (perp=15) · by state', point_size=10, alpha=0.55)
    plot_panel(axes[0, 1], Y_obs_15, cond_sub, CONDITION_COLORS,
               'Obs-space t-SNE (perp=15) · by condition', point_size=10, alpha=0.55)
    plot_panel(axes[0, 2], Y_obs_50, state_labels_per_pt, state_color_map,
               'Obs-space t-SNE (perp=50) · by state', point_size=10, alpha=0.55)
    plot_panel(axes[0, 3], Y_obs_50, cond_sub, CONDITION_COLORS,
               'Obs-space t-SNE (perp=50) · by condition', point_size=10, alpha=0.55)

    plot_panel(axes[1, 0], Y_lat_15, state_labels_per_pt, state_color_map,
               'Latent-space t-SNE (perp=15) · by state', point_size=10, alpha=0.55)
    plot_panel(axes[1, 1], Y_lat_15, cond_sub, CONDITION_COLORS,
               'Latent-space t-SNE (perp=15) · by condition', point_size=10, alpha=0.55)
    plot_panel(axes[1, 2], Y_lat_50, state_labels_per_pt, state_color_map,
               'Latent-space t-SNE (perp=50) · by state', point_size=10, alpha=0.55)
    plot_panel(axes[1, 3], Y_lat_50, cond_sub, CONDITION_COLORS,
               'Latent-space t-SNE (perp=50) · by condition', point_size=10, alpha=0.55)

    fig.suptitle(f'MVP rSLDS state-space t-SNE — N={len(P["sids"])} canonical sessions, '
                 f'{len(keep_idx)} pts; left columns perp=15 (local), right columns perp=50 (global)',
                 fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05,
                        hspace=0.30, wspace=0.18)

    out = HIER / f'state_tsne_{_VARIANT}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
