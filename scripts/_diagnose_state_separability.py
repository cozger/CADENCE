"""Immediate read: pairwise d_emit distances + shuffle-null silhouette.

Answers "are rSLDS states actually distinguishable in observation space, or is
the low silhouette (-0.003) from the Module 1 diagnostic just noise?"

Runs on existing 28D hierarchical results. Qualitative answer should survive
the 26D re-run since we're only dropping redundant channels.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def part1_pairwise_d_emit(params_path: str) -> dict:
    print('=' * 72)
    print('PART 1: Pairwise d_emit distances')
    print('=' * 72)
    hp = np.load(params_path, allow_pickle=True)
    mean_d = hp['mean_d_emit']
    labels = list(hp['state_labels'])
    keys = list(hp['modality_keys'])
    print(f'mean_d_emit shape: {mean_d.shape}')
    print(f'State labels: {labels}')
    print()

    K, D = mean_d.shape
    print('Per-state L2 norm of d_emit:')
    for k in range(K):
        n = float(np.linalg.norm(mean_d[k]))
        print(f'  {labels[k]:>6s} (k={k}): ||d||={n:.3f}')
    print()

    print('Pairwise L2 distance matrix:')
    print('          ' + '  '.join(f'{l:>8s}' for l in labels))
    dist = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dist[i, j] = np.linalg.norm(mean_d[i] - mean_d[j])
        print(f'  {labels[i]:>6s}  ' + '  '.join(f'{v:>8.3f}' for v in dist[i]))
    print()

    off_diag = dist[np.triu_indices(K, k=1)]
    print(f'Off-diagonal distance stats: '
          f'min={off_diag.min():.3f}, mean={off_diag.mean():.3f}, max={off_diag.max():.3f}')
    print(f'Ratio min/max: {off_diag.min() / off_diag.max():.3f}  '
          f'(near 1.0 = all pairs equidistant; near 0 = some pairs much closer)')
    print()

    print('Top 5 channels distinguishing each state (|d[k] - mean(d[others])|):')
    for k in range(K):
        others = np.mean(np.delete(mean_d, k, axis=0), axis=0)
        diff = np.abs(mean_d[k] - others)
        top = np.argsort(diff)[::-1][:5]
        pairs = [(keys[d], float(mean_d[k, d]), float(diff[d])) for d in top]
        print(f'  {labels[k]:>6s}: ' +
              ', '.join(f'{k_}={v:+.2f}(d={dv:.2f})' for k_, v, dv in pairs))
    print()

    return {'distances': dist, 'labels': labels, 'norms': np.linalg.norm(mean_d, axis=1)}


def part2_shuffle_null(results_dir: str, n_shuffles: int = 200, seed: int = 42) -> dict:
    print('=' * 72)
    print('PART 2: Shuffle-null silhouette')
    print('=' * 72)

    Y_list, path_list = [], []
    for sess_dir in sorted(glob.glob(os.path.join(results_dir, '*/'))):
        name = os.path.basename(os.path.dirname(sess_dir))
        if name == 'hierarchical':
            continue
        scaffold = os.path.join(sess_dir, 'scaffold_v11_ztimecourses.npz')
        rslds = os.path.join(sess_dir, 'v11_rslds_results.npz')
        json_path = os.path.join(sess_dir, 'scaffold_v11_results.json')
        if not (os.path.exists(scaffold) and os.path.exists(rslds) and os.path.exists(json_path)):
            continue
        with open(json_path) as f:
            meta = json.load(f)
        mod_keys = meta['modality_keys']
        npz = np.load(scaffold, allow_pickle=False)
        Y_pw = np.column_stack([npz[f'z_{k}'] for k in mod_keys])
        r = np.load(rslds, allow_pickle=True)
        path = r['path'] if 'path' in r else np.argmax(r['gamma'], axis=1)
        Y_list.append(Y_pw.astype(np.float32))
        path_list.append(path.astype(np.int32))

    print(f'Loaded {len(Y_list)} sessions')
    Y_all = np.vstack(Y_list)
    path_all = np.concatenate(path_list)
    print(f'Pooled: Y_all={Y_all.shape}, path_all={path_all.shape}')

    pca = PCA().fit(Y_all)
    n_80 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80)) + 1
    Y_r = PCA(n_components=n_80).fit_transform(Y_all)
    print(f'PCA-reduced to {n_80} components (80% variance)')

    rng = np.random.default_rng(seed)
    sub_n = min(5000, len(Y_r))
    sub_idx = rng.choice(len(Y_r), sub_n, replace=False)
    Y_sub = Y_r[sub_idx]
    p_sub = path_all[sub_idx]

    sil_real = float(silhouette_score(Y_sub, p_sub))
    db_real = float(davies_bouldin_score(Y_sub, p_sub))
    print(f'Real:    silhouette={sil_real:+.4f}, davies-bouldin={db_real:.3f}')
    print(f'Shuffling path labels (n={n_shuffles})...', flush=True)

    sils = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(p_sub)
        sils[i] = float(silhouette_score(Y_sub, shuffled))
    print(f'Shuffle: silhouette mean={sils.mean():+.4f}, std={sils.std():.4f}, '
          f'range=[{sils.min():+.4f}, {sils.max():+.4f}]')
    z = (sil_real - sils.mean()) / max(sils.std(), 1e-6)
    p_right = float((sils >= sil_real).mean())
    print(f'Real z-score vs shuffle: {z:+.2f}  (p_right={p_right:.3f})')
    print()

    if abs(z) < 2:
        verdict = 'Real silhouette INDISTINGUISHABLE from random labels.'
    elif z >= 2:
        verdict = f'Real silhouette {z:.1f}sigma ABOVE shuffle: some spatial structure.'
    else:
        verdict = f'Real silhouette {-z:.1f}sigma BELOW shuffle: states MORE scattered than random (suspicious).'
    print(f'Verdict: {verdict}')
    return {'real': sil_real, 'shuffle_mean': float(sils.mean()),
            'shuffle_std': float(sils.std()), 'z': float(z), 'p_right': p_right}


if __name__ == '__main__':
    params_path = 'results/v11/hierarchical/v11_hierarchical_params.npz'
    results_dir = 'results/v11'
    if os.path.exists(params_path):
        r1 = part1_pairwise_d_emit(params_path)
    else:
        print('=' * 72)
        print('PART 1 SKIPPED: hierarchical params npz not found')
        print(f'  Expected at: {params_path}')
        print('  Will run after 26D re-fit (which now saves mean_d_emit).')
        print('=' * 72)
        print()
    r2 = part2_shuffle_null(results_dir, n_shuffles=200, seed=42)
