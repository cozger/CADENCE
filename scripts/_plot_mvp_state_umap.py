"""UMAP of the MVP rSLDS state-space.

Pools all timepoints from canonical sessions in the production K=4 fit, runs
UMAP on (a) the 7-channel observation vectors and (b) the 3-channel latent
state vectors x_smooth, and produces side-by-side scatter plots colored by:
  - Viterbi state assignment
  - Experimental condition

The two coloring schemes side-by-side reveal whether the rSLDS states form
distinct clusters in the underlying behavioral space, and whether those
clusters align with conditions.

Usage:
    python scripts/_plot_mvp_state_umap.py
    python scripts/_plot_mvp_state_umap.py --max-points 8000
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import umap
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'
_SESS_NPZ = f'mvp_rslds_results{_SUFFIX}.npz'

STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',
    'OTHER':  '#9C27B0',
    'SHARED': '#2E7D32',
    'COUP':   '#FF6F00',
}

CONDITION_COLORS = {
    'base_EO':    '#1565C0',
    'base_EC':    '#283593',
    'conv_1':     '#E65100',
    'conv_2':     '#BF360C',
    'meditate_B': '#7B1FA2',
    'meditate_K': '#388E3C',
    'PE_1':       '#C62828',
    'PE_2':       '#B71C1C',
}


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


def parse_periods(markers):
    starts = {}
    out = []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    return sorted(out, key=lambda x: x[1])


def collect_pooled():
    with open(HIER / 'mvp_hierarchical_results.json') as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    sids = [s['session'] for s in d['sessions']]
    canonical = load_canonical_set()
    sids = [s for s in sids if s in canonical]

    with open(REPO / 'results/mvp' / sids[0] / 'mvp_scaffold.json') as f:
        meta0 = json.load(f)
    mod_keys = meta0['observation_channels']

    obs_list, valid_list, lat_list, path_list, cond_list, sess_list = [], [], [], [], [], []
    for i, sid in enumerate(sids):
        sd = REPO / 'results/mvp' / sid
        scaff = np.load(sd / 'mvp_scaffold.npz')
        rs = np.load(sd / _SESS_NPZ, allow_pickle=True)
        with open(REPO / 'data/digest/v1' / f'{sid}.json') as f:
            digest = json.load(f)

        obs   = scaff['obs']
        valid = scaff['obs_valid']
        t_lsl = scaff['t_common']
        path  = rs['path']
        x     = rs['x_smooth']
        # Map per-session state ordering -> production label ordering
        sl = list(rs['state_labels'])
        idx_map = {sl.index(lbl): k_disp for k_disp, lbl in enumerate(state_labels)}
        path_disp = np.array([idx_map[int(p)] for p in path], dtype=np.int8)

        # Condition assignment per timepoint
        periods = parse_periods(digest['markers'])
        ci = np.full(len(t_lsl), -1, dtype=int)
        for name, t0, t1 in periods:
            mask = (t_lsl >= t0) & (t_lsl < t1)
            ci[mask] = hash(name) % (10**9)  # placeholder; replaced after we know cond_names
        # Re-fill with deterministic indices below
        cond_list.append(np.full(len(t_lsl), '', dtype=object))
        for name, t0, t1 in periods:
            mask = (t_lsl >= t0) & (t_lsl < t1)
            cond_list[-1][mask] = name

        obs_list.append(obs)
        valid_list.append(valid)
        lat_list.append(x)
        path_list.append(path_disp)
        sess_list.append(np.full(len(t_lsl), i, dtype=np.int16))

    obs   = np.concatenate(obs_list, axis=0)
    valid = np.concatenate(valid_list, axis=0)
    lat   = np.concatenate(lat_list, axis=0)
    path  = np.concatenate(path_list, axis=0)
    cond  = np.concatenate(cond_list, axis=0)
    sess  = np.concatenate(sess_list, axis=0)

    return {
        'state_labels': state_labels, 'mod_keys': mod_keys,
        'sids': sids, 'obs': obs, 'valid': valid, 'lat': lat,
        'path': path, 'cond': cond, 'sess': sess,
    }


def stratified_subsample(P, n_target, rng):
    """Equal points per (state, condition) cell where possible. Drop NaN obs."""
    state_labels = P['state_labels']
    K = len(state_labels)

    # Mask out timepoints with any invalid channel (UMAP cant handle NaN)
    finite = np.isfinite(P['obs']).all(axis=1) & P['valid'].all(axis=1)
    has_cond = (P['cond'] != '')
    # Drop legacy single-occurrence labels
    legit_cond = ~np.isin(P['cond'], ['PE', 'baseline', ''])
    keep = finite & has_cond & legit_cond
    print(f'  total timepoints: {len(P["path"])}, valid+conditioned: {keep.sum()} '
          f'({keep.mean()*100:.1f}%)')

    # Stratify by state x condition
    cond_names = sorted(set(c for c in P['cond'][keep] if c))
    n_per_cell = max(1, n_target // (K * len(cond_names)))
    print(f'  conditions: {cond_names}; n_per (state,cond) cell ≈ {n_per_cell}')

    chosen = []
    for k in range(K):
        for c in cond_names:
            mask = keep & (P['path'] == k) & (P['cond'] == c)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            take = min(len(idx), n_per_cell)
            chosen.append(rng.choice(idx, take, replace=False))
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    print(f'  subsampled: {len(chosen)} points')
    return chosen


def make_umap(X, seed=42, n_neighbors=30, min_dist=0.1):
    print(f'  Fitting UMAP on shape={X.shape} (n_neighbors={n_neighbors}, min_dist={min_dist})...')
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, metric='euclidean', verbose=False)
    Y = reducer.fit_transform(X)
    return Y


def plot_panel(ax, Y, labels, color_map, title, point_size=8, alpha=0.5):
    for lbl, color in color_map.items():
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(Y[mask, 0], Y[mask, 1], s=point_size, c=color,
                   alpha=alpha, linewidths=0, label=f'{lbl} (N={int(mask.sum())})')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[k],
                       markersize=8, label=f'{k}', linestyle='')
               for k in color_map if (labels == k).sum() > 0]
    ax.legend(handles=handles, fontsize=8, loc='best',
              frameon=True, framealpha=0.85, ncol=2)


def plot_state_facets(fig, gs_block, Y, path, state_labels, state_color_map,
                       which='obs'):
    """4 small panels (one per state): grey background + that state highlighted."""
    K = len(state_labels)
    for k in range(K):
        ax = fig.add_subplot(gs_block[k // 2, k % 2])
        # Background: ALL points in light grey
        ax.scatter(Y[:, 0], Y[:, 1], s=4, c='#E0E0E0', alpha=0.4, linewidths=0, zorder=1)
        # Foreground: only this state
        mask = path == k
        ax.scatter(Y[mask, 0], Y[mask, 1], s=12, c=state_color_map[state_labels[k]],
                   alpha=0.7, linewidths=0, zorder=2)
        ax.set_title(f'{state_labels[k]}  (N={int(mask.sum())} of {len(path)} pts)',
                     fontsize=9, color=state_color_map[state_labels[k]],
                     fontweight='bold')
        ax.tick_params(labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(state_color_map[state_labels[k]])
            spine.set_linewidth(1.5)
        if which == 'obs':
            ax.set_xlabel('UMAP 1 (obs)', fontsize=7)
            ax.set_ylabel('UMAP 2 (obs)', fontsize=7)
        else:
            ax.set_xlabel('UMAP 1 (lat)', fontsize=7)
            ax.set_ylabel('UMAP 2 (lat)', fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-points', type=int, default=12000,
                    help='Subsample target across all (state×condition) cells')
    ap.add_argument('--n-neighbors', type=int, default=30)
    ap.add_argument('--min-dist', type=float, default=0.1)
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

    print('\n=== Fitting UMAPs ===')
    print('[1/2] Observation-space UMAP (7D -> 2D):')
    Y_obs = make_umap(obs_sub, seed=args.seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    print('[2/2] Latent-space UMAP (3D -> 2D):')
    Y_lat = make_umap(lat_sub, seed=args.seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    # Translate path indices to labels for plotting
    state_labels_per_pt = np.array([state_labels[k] for k in path_sub])

    print('\n=== Plotting ===')
    # Layout:
    #   row 0: 2 main panels — obs UMAP (state) | obs UMAP (condition)
    #   row 1: 2 main panels — lat UMAP (state) | lat UMAP (condition)
    #   row 2: 4 small panels (one per state) — obs UMAP, that state highlighted
    #   row 3: 4 small panels (one per state) — lat UMAP, that state highlighted
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 4, height_ratios=[2.5, 2.5, 1.4, 1.4],
                          hspace=0.38, wspace=0.25,
                          left=0.05, right=0.97, top=0.95, bottom=0.04)

    ax_obs_state = fig.add_subplot(gs[0, :2])
    ax_obs_cond  = fig.add_subplot(gs[0, 2:])
    ax_lat_state = fig.add_subplot(gs[1, :2])
    ax_lat_cond  = fig.add_subplot(gs[1, 2:])

    plot_panel(ax_obs_state, Y_obs, state_labels_per_pt, state_color_map,
               f'Observation-space UMAP ({obs_sub.shape[1]}D obs → 2D)  ·  by Viterbi state',
               point_size=10, alpha=0.55)
    plot_panel(ax_obs_cond, Y_obs, cond_sub, CONDITION_COLORS,
               f'Observation-space UMAP  ·  by experimental condition',
               point_size=10, alpha=0.55)
    plot_panel(ax_lat_state, Y_lat, state_labels_per_pt, state_color_map,
               f'Latent-space UMAP ({lat_sub.shape[1]}D x_smooth → 2D)  ·  by Viterbi state',
               point_size=10, alpha=0.55)
    plot_panel(ax_lat_cond, Y_lat, cond_sub, CONDITION_COLORS,
               f'Latent-space UMAP  ·  by experimental condition',
               point_size=10, alpha=0.55)

    # State-faceted small multiples: one row per modality (obs / lat)
    print('  rendering state-faceted small multiples...')
    K = len(state_labels)
    for k in range(K):
        ax = fig.add_subplot(gs[2, k])
        ax.scatter(Y_obs[:, 0], Y_obs[:, 1], s=3, c='#E0E0E0', alpha=0.5, linewidths=0)
        m = path_sub == k
        ax.scatter(Y_obs[m, 0], Y_obs[m, 1], s=8,
                   c=state_color_map[state_labels[k]], alpha=0.7, linewidths=0)
        ax.set_title(f'obs · {state_labels[k]} ({m.sum()})',
                     fontsize=9, color=state_color_map[state_labels[k]], fontweight='bold')
        ax.tick_params(labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(state_color_map[state_labels[k]])
            spine.set_linewidth(1.5)

        ax = fig.add_subplot(gs[3, k])
        ax.scatter(Y_lat[:, 0], Y_lat[:, 1], s=3, c='#E0E0E0', alpha=0.5, linewidths=0)
        ax.scatter(Y_lat[m, 0], Y_lat[m, 1], s=8,
                   c=state_color_map[state_labels[k]], alpha=0.7, linewidths=0)
        ax.set_title(f'lat · {state_labels[k]} ({m.sum()})',
                     fontsize=9, color=state_color_map[state_labels[k]], fontweight='bold')
        ax.tick_params(labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(state_color_map[state_labels[k]])
            spine.set_linewidth(1.5)

    fig.suptitle(f'MVP rSLDS state-space UMAP — N={len(P["sids"])} canonical sessions, '
                 f'{len(keep_idx)} stratified-sampled timepoints '
                 f'(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})',
                 fontsize=12, fontweight='bold')

    out = HIER / f'state_umap_{_VARIANT}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
