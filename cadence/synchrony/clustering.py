"""Stages 5+6 — UMAP embedding + HDBSCAN clustering on cohort features.

Two UMAP fits on the same cohort feature matrix:
  - 2D for visualization (n_components=2, min_dist=0.05)
  - 10D for clustering   (n_components=10, min_dist=0.0)

HDBSCAN on the 10D embedding with EOM cluster selection. Multiple
resolution cuts (5, 8, 12, 20) saved alongside the production fit so
Stage 7 can inspect coarse vs fine taxonomies side by side.
"""
from __future__ import annotations

# Hoist torch via _gpu before numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np

from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG
from cadence.synchrony.io import cohort_dir


def _load_cohort_features():
    cf = cohort_dir() / 'cohort_features.npz'
    if not cf.exists():
        raise FileNotFoundError(f'{cf} not found — run Stage 4 cohort fan-in first.')
    return dict(np.load(cf, allow_pickle=True))


def _fit_umap(X: np.ndarray, *, n_components: int, min_dist: float,
                n_neighbors: int, random_state: int):
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                          min_dist=min_dist, metric='euclidean',
                          random_state=random_state, n_jobs=1)
    return reducer.fit_transform(X), reducer


def _hdbscan_fit(X10: np.ndarray, *, min_cluster_size: int, min_samples: int,
                   selection_method: str = 'eom'):
    import hdbscan
    h = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples,
                         cluster_selection_method=selection_method,
                         gen_min_span_tree=True, prediction_data=True)
    h.fit(X10)
    return h


def _medoids_per_cluster(X10: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """For each non-noise cluster, return the row-index closest to the centroid."""
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroid = X10[idx].mean(axis=0)
        d = np.linalg.norm(X10[idx] - centroid, axis=1)
        out[int(c)] = int(idx[np.argmin(d)])
    return out


def fit_cohort_clusters(config: SynchronyConfig = DEFAULT_CONFIG,
                          force: bool = False) -> dict:
    """Fit UMAP + HDBSCAN on the cohort feature matrix; cache results."""
    out_path = cohort_dir(ensure=True) / 'cohort_clusters.npz'
    sidecar_path = cohort_dir() / 'cohort_clusters.json'
    config_hash = config.hash()
    if not force and out_path.exists() and sidecar_path.exists():
        meta = json.loads(sidecar_path.read_text())
        if meta.get('synchrony_config_hash') == config_hash:
            arrays = dict(np.load(out_path, allow_pickle=True))
            return {**arrays, '_meta': meta, '_from_cache': True}

    t0 = time.perf_counter()
    coh = _load_cohort_features()
    X = np.asarray(coh['features'], dtype=np.float32)

    # Drop low-quality episodes from the cluster pool but keep their
    # indices for back-mapping.
    low_q = np.asarray(coh.get('low_quality', np.zeros(len(X), dtype=bool)))
    keep_mask = ~low_q
    keep_idx = np.where(keep_mask)[0]
    X_keep = X[keep_idx]
    n_keep = len(keep_idx)
    if n_keep < 30:
        raise RuntimeError(f'Only {n_keep} cluster-eligible episodes; need ≥30')

    # UMAP 2D + 10D
    embed_2d, _ = _fit_umap(X_keep, n_components=config.umap_n_components_2d,
                              min_dist=config.umap_min_dist_2d,
                              n_neighbors=config.umap_n_neighbors,
                              random_state=config.umap_random_state)
    embed_10d, _ = _fit_umap(X_keep, n_components=config.umap_n_components_10d,
                               min_dist=config.umap_min_dist_10d,
                               n_neighbors=config.umap_n_neighbors,
                               random_state=config.umap_random_state)

    # HDBSCAN on 10D
    min_cluster = max(config.hdbscan_min_cluster_floor,
                      int(config.hdbscan_min_cluster_frac * n_keep))
    h = _hdbscan_fit(embed_10d, min_cluster_size=min_cluster,
                       min_samples=config.hdbscan_min_samples,
                       selection_method=config.hdbscan_cluster_selection)
    labels = np.asarray(h.labels_, dtype=np.int32)
    probs  = np.asarray(h.probabilities_, dtype=np.float32)

    # Multi-resolution cuts: ask for K clusters via cluster_persistence cuts.
    # hdbscan's `condensed_tree_.get_clusters(...)` is the right API. Fall back
    # to varying min_cluster_size if persistence cuts fail.
    multi_res_labels = {}
    target_K = [5, 8, 12, 20]
    for K in target_K:
        # Try varying min_cluster_size to bracket K
        best = None
        for mcs in (max(2, n_keep // 10),
                     max(2, n_keep // 20),
                     max(2, n_keep // 30),
                     max(2, n_keep // 50)):
            try:
                hk = _hdbscan_fit(embed_10d, min_cluster_size=mcs,
                                    min_samples=config.hdbscan_min_samples,
                                    selection_method='eom')
                k_found = len(set(hk.labels_)) - (1 if -1 in hk.labels_ else 0)
                if best is None or abs(k_found - K) < abs(best[1] - K):
                    best = (hk.labels_, k_found, mcs)
            except Exception:
                continue
        if best is not None:
            multi_res_labels[f'k{K}'] = np.asarray(best[0], dtype=np.int32)

    medoids_main = _medoids_per_cluster(embed_10d, labels)

    # Map (kept-only) results back to full cohort indexing (-2 = low-quality
    # excluded; -1 = HDBSCAN noise).
    full_labels = np.full(len(X), -2, dtype=np.int32)
    full_labels[keep_idx] = labels
    full_probs = np.zeros(len(X), dtype=np.float32)
    full_probs[keep_idx] = probs
    full_embed_2d = np.full((len(X), 2), np.nan, dtype=np.float32)
    full_embed_2d[keep_idx] = embed_2d
    full_embed_10d = np.full((len(X), 10), np.nan, dtype=np.float32)
    full_embed_10d[keep_idx] = embed_10d
    full_multi = {}
    for K, lk in multi_res_labels.items():
        flk = np.full(len(X), -2, dtype=np.int32)
        flk[keep_idx] = lk
        full_multi[K] = flk

    elapsed = time.perf_counter() - t0
    clusters_present = sorted(set(int(c) for c in full_labels) - {-2, -1})
    n_clusters = len(clusters_present)
    n_noise = int((full_labels == -1).sum())

    arrays = {
        'embedding_2d':  full_embed_2d,
        'embedding_10d': full_embed_10d,
        'labels':        full_labels,
        'probabilities': full_probs,
        'medoid_episode_idx': np.array([medoids_main[c] for c in clusters_present],
                                          dtype=np.int32) if clusters_present else
                                np.zeros(0, dtype=np.int32),
        'medoid_cluster_id':  np.array(clusters_present, dtype=np.int32),
        **{f'labels_{K}': full_multi[K] for K in full_multi},
    }
    np.savez(out_path, **arrays)

    sidecar = {
        'synchrony_config_hash': config_hash,
        'n_episodes_total':      int(len(X)),
        'n_low_quality_excluded': int(low_q.sum()),
        'n_episodes_clustered':  int(n_keep),
        'min_cluster_size':      int(min_cluster),
        'min_samples':           int(config.hdbscan_min_samples),
        'cluster_selection':     config.hdbscan_cluster_selection,
        'n_clusters':            n_clusters,
        'n_noise':               n_noise,
        'cluster_sizes':         {int(c): int((labels == c).sum())
                                    for c in clusters_present},
        'multi_resolution_cluster_counts': {
            K: int(len(set(int(c) for c in full_multi[K]) - {-2, -1}))
            for K in full_multi
        },
        'umap_params': {
            'n_neighbors': config.umap_n_neighbors,
            'min_dist_2d': config.umap_min_dist_2d,
            'min_dist_10d': config.umap_min_dist_10d,
            'random_state': config.umap_random_state,
        },
        'wall_seconds':           round(elapsed, 3),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return {**arrays, '_meta': sidecar, '_from_cache': False}
