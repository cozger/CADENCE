"""Centralized configuration for the synchrony repertoire pipeline.

All tunable parameters live here so a single config-hash can decide whether
a cached artifact is stale. Pass a custom ``SynchronyConfig`` to
``pipeline.run_session(...)`` to override any default; otherwise
``DEFAULT_CONFIG`` is used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SynchronyConfig:
    # ── Stage 0 preprocessing ─────────────────────────────────────────
    # Reused via cadence.significance.face_event_coincidence helpers.
    smooth_window:   int   = 5      # SavGol window (frames)
    smooth_poly:     int   = 2
    baseline_window_s: float = 5.0  # rolling-q10 baseline window (s)
    baseline_q:      int   = 10     # quantile percentile

    # ── Stage 1 multiscale events ─────────────────────────────────────
    fs_native:           float        = 30.0
    event_scales_seconds: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    event_noise_thresh_factor: float  = 3.0
    event_min_chain_frac: float       = 0.5
    # If True, estimate each AU's noise floor from base_EO frames only;
    # fall back to global MAD where base_EO marker is absent. base_EC is
    # NOT used as a baseline because eye-region AUs are pinned during
    # closed-eyes blocks and would miscalibrate.
    use_baseline_noise_floor: bool    = True

    # ── Stage 2 episodes ──────────────────────────────────────────────
    # Cohort-anchored absolute threshold (computed at cohort-pool time
    # from base_EO segments). Per-session quantile is used as a fallback
    # only when the cohort threshold isn't provided.
    cohort_threshold_quantile: float  = 0.90   # of base_EO joint envelope
    fallback_session_quantile: float  = 0.60
    # Empirical defaults from y_06 sweep (see _check_conv2_v2.py): the
    # plan's 0.5s merge_gap collapses conversational blocks (>5min of
    # >90% above-threshold frames) into single 300s episodes that exceed
    # max_dur. 0.2s merge_gap respects natural turn-taking pauses; 120s
    # max keeps even long conversational holds.
    episode_merge_gap_s:    float    = 0.2
    episode_min_duration_s: float    = 0.2
    episode_max_duration_s: float    = 120.0

    # ── Stage 3a coincidence ──────────────────────────────────────────
    coinc_max_lag_s:        float    = 2.0   # NN-lag matching window per AU
    coinc_region_lag_s:     float    = 1.0   # tighter window for region presence

    # ── Stage 3b DTW ──────────────────────────────────────────────────
    dtw_band_s:             float    = 0.5   # Sakoe-Chiba band half-width

    # ── Stage 3c sparse CCA ───────────────────────────────────────────
    cca_lag_min_s:          float    = -1.0
    cca_lag_max_s:          float    = +1.0
    cca_lag_step_s:         float    = 0.1
    cca_tau:                float    = 0.5   # cca-zoo SCCA_PMD L1 bound (0,1]
    cca_min_per_episode_samples: int = 200   # ≈ 6.7 s at 30 fps
    cca_min_pooled_samples:    int   = 600
    cca_pool_duration_tol:     float = 0.30  # ±30% duration band for pooling

    # ── Stage 3d coherence ────────────────────────────────────────────
    coh_band_lo_hz:         float    = 0.1
    coh_band_hi_hz:         float    = 2.0
    coh_smooth_s:           float    = 0.5

    # ── Stage 4 standardization ───────────────────────────────────────
    low_quality_imputation_threshold: float = 0.40

    # ── Stage 5+6 clustering ──────────────────────────────────────────
    umap_n_neighbors:           int  = 20
    umap_min_dist_2d:           float = 0.05
    umap_min_dist_10d:          float = 0.0
    umap_n_components_2d:       int  = 2
    umap_n_components_10d:      int  = 10
    umap_random_state:          int  = 42
    hdbscan_min_cluster_frac:   float = 0.015   # fraction of n_episodes
    hdbscan_min_cluster_floor:  int  = 8
    hdbscan_min_samples:        int  = 5
    hdbscan_cluster_selection:  str  = 'eom'

    # ── Stage 8 validation ────────────────────────────────────────────
    bootstrap_n_iter:           int  = 200
    bootstrap_subsample_frac:   float = 0.80
    null_block_size_s_min:      float = 5.0
    null_block_size_s_max:      float = 10.0
    threshold_perturbation_frac: float = 0.20

    # ── Acceleration ──────────────────────────────────────────────────
    use_gpu_for_events:    bool = True
    use_gpu_for_coherence: bool = True
    n_jobs_per_session:    int  = -1   # joblib threading; -1 = all cores

    def hash(self) -> str:
        """SHA-256 of the JSON-serialized config; used for cache freshness."""
        payload = json.dumps(asdict(self), sort_keys=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()[:16]

    def lag_grid_s(self):
        """Return CCA lag sweep as a numpy-friendly list."""
        import numpy as np
        return np.arange(self.cca_lag_min_s,
                         self.cca_lag_max_s + 0.5 * self.cca_lag_step_s,
                         self.cca_lag_step_s).tolist()


DEFAULT_CONFIG = SynchronyConfig()
