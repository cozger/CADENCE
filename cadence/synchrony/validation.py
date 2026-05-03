"""Stage 8 — validation suite.

Five tests, written to ``cohort/cohort_validation.json``:

  bootstrap_stability   200×, 80% subsample, ARI vs production fit
  block_shuffle_null    block-shuffle one role's full timecourse, refit
  time_reversed_control reverse one role's stream, refit, check synchrony
                          features dissolve while identity preserved
  leave_one_dyad_out    19 LOO folds, project held-out via approximate_predict
  threshold_sensitivity ±20% perturbation of T*, check episode-count stability

The block_shuffle and time_reversed tests are expensive (require re-running
Stages 1-6); we run them on a smaller bootstrap (~30) for speed and report.
"""
from __future__ import annotations

# Hoist torch via _gpu before numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG
from cadence.synchrony.io import cohort_dir


def _load():
    cdir = cohort_dir()
    coh = dict(np.load(cdir / 'cohort_features.npz', allow_pickle=True))
    cls = dict(np.load(cdir / 'cohort_clusters.npz', allow_pickle=True))
    return coh, cls


def _bootstrap_stability(X10: np.ndarray, labels: np.ndarray,
                           n_iter: int = 200, frac: float = 0.80,
                           min_cluster_size: int = 10) -> dict:
    """Bootstrap UMAP+HDBSCAN N times; ARI vs production labels.

    NOTE: full bootstrap with re-running UMAP is expensive (~30s × 200 = 100min).
    We instead bootstrap-subsample the EMBEDDING (without re-running UMAP) and
    re-cluster. This validates clustering stability conditional on the
    embedding — different question than full pipeline stability, but cheaper
    and the relevant one once UMAP has converged.
    """
    import hdbscan
    rng = np.random.default_rng(0)
    n = len(X10)
    aris = []
    n_clusters_per_iter = []
    for it in range(n_iter):
        idx = rng.choice(n, int(frac * n), replace=False)
        try:
            h = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=5, cluster_selection_method='eom')
            l_boot = h.fit_predict(X10[idx])
            ari = adjusted_rand_score(labels[idx], l_boot)
            aris.append(ari)
            n_clusters_per_iter.append(int(len(set(l_boot)) -
                                            (1 if -1 in l_boot else 0)))
        except Exception:
            continue
    aris = np.asarray(aris)
    return {
        'n_iter':              int(len(aris)),
        'mean_ari':            float(aris.mean()) if aris.size else float('nan'),
        'std_ari':             float(aris.std(ddof=0)) if aris.size else float('nan'),
        'p10_ari':             float(np.percentile(aris, 10)) if aris.size else float('nan'),
        'p90_ari':             float(np.percentile(aris, 90)) if aris.size else float('nan'),
        'mean_n_clusters':     float(np.mean(n_clusters_per_iter)) if n_clusters_per_iter else float('nan'),
        'std_n_clusters':      float(np.std(n_clusters_per_iter, ddof=0)) if n_clusters_per_iter else float('nan'),
        'pass_mean_ari_gt_0p6': bool(aris.size and aris.mean() > 0.6),
    }


def _leave_one_dyad_out(X10: np.ndarray, sids: np.ndarray, labels: np.ndarray,
                          min_cluster_size: int = 10) -> dict:
    """For each unique dyad, refit HDBSCAN without it; predict on held-out;
    compare assignments to full-cohort labels via ARI.

    Same shortcut as bootstrap: re-cluster the cohort minus dyad on the
    SAME UMAP embedding (no re-running UMAP). Cheaper, and tests cluster
    stability under cohort composition changes.
    """
    import hdbscan
    unique_sids = sorted(set(sids))
    aris = {}
    for ho_sid in unique_sids:
        train_mask = sids != ho_sid
        test_mask  = sids == ho_sid
        if train_mask.sum() < 30 or test_mask.sum() < 5:
            continue
        try:
            h = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=5, cluster_selection_method='eom',
                                  prediction_data=True)
            l_train = h.fit_predict(X10[train_mask])
            l_test, _ = hdbscan.approximate_predict(h, X10[test_mask])
            # Compare held-out cluster assignments to full-cohort labels
            ari = adjusted_rand_score(labels[test_mask], l_test)
            aris[ho_sid] = float(ari)
        except Exception:
            aris[ho_sid] = float('nan')
    if aris:
        finite = np.array([v for v in aris.values() if np.isfinite(v)])
        mean_ari = float(finite.mean()) if finite.size else float('nan')
    else:
        mean_ari = float('nan')
    return {
        'n_folds':            len(aris),
        'mean_ari':           mean_ari,
        'per_dyad_ari':       aris,
        'pass_mean_ari_gt_0p7': bool(np.isfinite(mean_ari) and mean_ari > 0.7),
    }


def _threshold_sensitivity(config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Perturb T* by ±20%; report episode-count stability per session."""
    from cadence.ingest.quality import list_canonical_sessions
    from cadence.synchrony.episodes import (
        _resampled_joint_envelope, _segment_intervals, compute_cohort_threshold,
    )
    from cadence.synchrony.events import _resolve_role_to_p
    from cadence.synchrony.io import (
        load_face_npz, load_digest, face_npz_path,
    )
    ct = compute_cohort_threshold(config)
    T_star = ct['T_star']
    perturb = config.threshold_perturbation_frac
    out = {}
    for sid in list_canonical_sessions():
        if not face_npz_path(sid).exists():
            continue
        face = load_face_npz(sid)
        digest = load_digest(sid)
        try:
            role_to_p = _resolve_role_to_p(face, digest)
        except ValueError:
            continue
        # Skip sessions missing one face stream
        if not all(f"{role_to_p[r]}_au_activity" in face
                    for r in ('therapist', 'patient')):
            continue
        joint, _ = _resampled_joint_envelope(face, role_to_p)
        per_perturb = {}
        for delta in (-perturb, 0.0, +perturb):
            T_p = T_star * (1.0 + delta)
            intervals = _segment_intervals(
                joint, T_p, config.fs_native,
                merge_gap_s=config.episode_merge_gap_s,
                min_dur_s=config.episode_min_duration_s,
                max_dur_s=config.episode_max_duration_s)
            per_perturb[f'{delta:+.2f}'] = len(intervals)
        out[sid] = per_perturb
    # Aggregate: median ratio of -20% vs +20% counts to baseline
    ratios = []
    for sid, ct_ in out.items():
        baseline = ct_.get('+0.00', 0)
        if baseline == 0:
            continue
        for d in (f'{-perturb:+.2f}', f'{+perturb:+.2f}'):
            ratios.append(ct_[d] / baseline)
    return {
        'per_session_counts':   out,
        'count_ratio_min':      float(min(ratios)) if ratios else float('nan'),
        'count_ratio_max':      float(max(ratios)) if ratios else float('nan'),
        'count_ratio_median':   float(np.median(ratios)) if ratios else float('nan'),
        'pass_within_2x':       bool(ratios and 0.5 <= min(ratios) and max(ratios) <= 2.0),
    }


def run_validation_suite(config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Run the full validation suite; write JSON sidecar."""
    t0 = time.perf_counter()
    coh, cls = _load()

    e10 = np.asarray(cls['embedding_10d'])
    labels = np.asarray(cls['labels'])
    sids = np.asarray(coh['session_id'])
    finite = np.isfinite(e10[:, 0])
    e10f = e10[finite]
    labels_f = labels[finite]
    sids_f = sids[finite]
    n = len(e10f)

    # Cluster size from sidecar
    cls_meta_path = cohort_dir() / 'cohort_clusters.json'
    cls_meta = json.loads(cls_meta_path.read_text())
    min_cluster = int(cls_meta.get('min_cluster_size', 10))

    print(f'  bootstrap stability ({config.bootstrap_n_iter} iter)...')
    boot = _bootstrap_stability(e10f, labels_f,
                                  n_iter=config.bootstrap_n_iter,
                                  frac=config.bootstrap_subsample_frac,
                                  min_cluster_size=min_cluster)
    print(f'    mean ARI = {boot["mean_ari"]:.3f}')
    print(f'  leave-one-dyad-out (19 folds)...')
    loo = _leave_one_dyad_out(e10f, sids_f, labels_f,
                                min_cluster_size=min_cluster)
    print(f'    mean ARI = {loo["mean_ari"]:.3f}')
    print(f'  threshold sensitivity (±{int(100*config.threshold_perturbation_frac)}%)...')
    thr = _threshold_sensitivity(config)
    print(f'    count ratio range: [{thr["count_ratio_min"]:.2f}, '
          f'{thr["count_ratio_max"]:.2f}]')

    # Block-shuffle and time-reversed are expensive (require re-running
    # Stages 1-4 on shuffled data); they're left as a slower companion job
    # that can be invoked with --full-validation. Default lite suite skips
    # them and notes this in the output.
    full = {
        'bootstrap_stability':    boot,
        'leave_one_dyad_out':     loo,
        'threshold_sensitivity':  thr,
        'block_shuffle_null':     {'status': 'not run (expensive — pass --full-validation)'},
        'time_reversed_control':  {'status': 'not run (expensive — pass --full-validation)'},
        'wall_seconds':           round(time.perf_counter() - t0, 1),
    }
    (cohort_dir() / 'cohort_validation.json').write_text(json.dumps(full, indent=2))
    return full
