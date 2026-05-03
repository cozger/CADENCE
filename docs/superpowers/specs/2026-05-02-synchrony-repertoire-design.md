# Synchrony Repertoire Pipeline — Design Spec

**Date**: 2026-05-02
**Owner**: cozger
**Status**: Approved (brainstorm exit) — implementing
**Cohort**: 19 canonical MVP-eligible sessions
**Builds on**: `cadence/significance/face_event_coincidence.py` (Stage 0+1 substrate)
**Parallel artifact**: `results/mvp/<sid>/mvp_rslds_results.npz` (used only by Stage 9c cross-tab)

---

## 1. Problem statement

Quantify a dyad's *synchrony repertoire* — the typology of synchronous facial events that occur in dyadic interaction — by decomposing each co-active episode into multiple complementary feature views and clustering events to discover the taxonomy data-driven. Replaces single-scalar synchrony scores with a per-dyad distribution over discovered event types plus an orthogonal expressivity axis.

Two scientific axes are produced per dyad:

- **Repertoire signature** — distribution over the cohort-discovered cluster taxonomy ("what kinds of synchronous events").
- **Expressivity profile** — episode rate, intensity, AU diversity, dyadic balance ("how much activity").

Two dyads with similar repertoires can have very different expressivity, and vice versa. Both axes are first-class outputs.

## 2. Stage map

| Stage | Function | Output |
|---|---|---|
| 0 | Smooth (SavGol w=5, p=2) + rolling-p10 baseline (5–10 s window) | clean per-AU traces |
| 1 | Multiscale 1st-deriv-of-Gaussian event detection + chain linking + Hölder α | per-AU event lists w/ regularity |
| 2 | Joint-activity envelope + episode segmentation (cohort-anchored absolute threshold) | per-episode start/end + member events |
| 3a | Per-AU NN-lag stats within episode | 13 features |
| 3b | Sakoe-Chiba-banded DTW × (1 full 52-D + 7 region) | 32 features |
| 3c | Lag-swept sparse CCA on event-gated samples (with short-episode pooling) | 17 features |
| 3d | GPU Morlet wavelet coherence in 0.1–2 Hz | 2 features |
| 4 | Identity + dynamics + intensity + context features; cohort-pooled standardization | +43; total 107 dims |
| 5 | UMAP — 2D for viz, 10D for clustering | embeddings |
| 6 | HDBSCAN — `min_cluster_size = 1.5%`, `min_samples = 5`, EOM selection | cluster labels |
| 7 | Per-cluster identity / dynamics / synchrony profiles + paired-face schematic + top-3 medoids | per-cluster JSON + figures |
| 8 | Bootstrap stability + block-shuffle null + time-reversed control + leave-one-dyad-out + sanity dyads | validation JSON + summary fig |
| 9 | Per-dyad repertoire signature + expressivity profile + rSLDS-state cross-tab + REPORT.md | per-dyad CSVs + cross-tab fig |

## 3. Module + artifact layout

```
cadence/synchrony/
  __init__.py
  config.py                 # central defaults + hash for cache freshness
  preprocessing.py          # Stage 0 (re-exports from face_event_coincidence)
  events.py                 # Stage 1 wrapper; adds Hölder α to detect_au_events
  episodes.py               # Stage 2
  features/
    __init__.py
    coincidence.py          # 3a
    dtw.py                  # 3b (dtaidistance, threaded)
    cca.py                  # 3c (cca-zoo SCCA + pooling)
    coherence.py            # 3d (GPU CWT via bl_wavelet._cwt_gpu)
    assembly.py             # Stage 4
    _standardization.py     # feature classification (continuous / binary / sign / fraction)
  clustering.py             # Stages 5+6
  interpretation.py         # Stage 7
  validation.py             # Stage 8
  reporting.py              # Stage 9
  pipeline.py               # callable orchestrator
  io.py                     # cache I/O at well-defined boundaries
  viz.py                    # paired-face schematic + plotting utilities
  _gpu.py                   # batched_dog_pyramid (Stage 1 GPU)

scripts/
  _run_synchrony_per_session.py    # Stages 0-4 on one or all sessions
  _run_synchrony_cohort.py         # Stages 5-9 on cached episode features

tests/synchrony/
  test_events_alpha.py
  test_episodes.py
  test_cca_pooling.py
  test_coincidence_lag.py

results/synchrony/
  <sid>/
    01_events.npz, 01_events_overlay.png
    02_episodes.npz, 02_episodes_overlay.png
    03_features_per_episode.npz, 03_features_per_episode.json
    03_features_diagnostics.png
  cohort/
    cohort_features.npz
    cohort_clusters.npz
    cohort_validation.json
    cluster_summaries.json
    per_dyad_repertoire.csv
    per_dyad_expressivity.csv
    fig_cohort_embedding.png
    fig_cluster_<k>_profile.png
    fig_all_clusters_grid.png
    fig_per_dyad_repertoire.png
    fig_per_dyad_expressivity.png
    fig_repertoire_x_rslds_state.png
    fig_validation_summary.png
    REPORT.md
```

## 4. Key methodological decisions

### Reuse of `face_event_coincidence.detect_au_events`

Stage 1 augments the existing function (in-place, with `return_extra=True` keyword to preserve the MVP path's positional return) to compute Hölder α from the chain's `(log s, log |amp|)` slope. Single source of truth shared between MVP channel and synchrony repertoire.

### AU regions (Stage 4 identity + 3a/3c region presence)

Subdivided 7-region map added to `cadence/constants.py` alongside existing `AU_REGIONS`:

```python
AU_REGIONS_7 = {
  'brow': [1, 2, 3, 4, 5],
  'eye': [9, 10, 19, 20, 21, 22],
  'nose': [50, 51],
  'cheek': [6, 7, 8],
  'mouth_smile': [7, 8, 28, 29, 44, 45],          # smile-cluster: smile L/R + dimple L/R + cheek squint
  'mouth_frown': [30, 31, 39, 40, 42, 43, 47, 49], # frown / press / pucker / shrug
  'mouth_jaw':   [23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 41, 46, 48], # jaw + oromotor
}
```

(Final AU lists per region verified against `BLENDSHAPE_NAMES` in `docs/validate_blendshape_isolation.py` during implementation.)

### Episode segmentation threshold — absolute, cohort-anchored

`T*` = 90th percentile of joint envelope pooled across all base_EO segments in canonical sessions. Applied identically to every dyad. Quiet dyads correctly yield few episodes; expressivity becomes a first-class signal at three scales:

- **Segmentation level**: episode count varies with expressivity.
- **Per-episode features**: `int_peak_env`, `int_mean_env`, `int_peak_au_amp` z-scored cohort-wide.
- **Per-dyad reporting**: separate expressivity CSV + figure (Stage 9b).

### CCA pooling policy (Stage 3c)

```
T_min_per_episode_samples = 200   (~6.7 s at 30 fps)
T_min_pooled              = 600
```

Episodes ≥200 samples → per-episode CCA. Shorter → pool with same-condition same-dyad episodes whose duration is in `[0.7×, 1.3×]` of target episode's duration. Pooled CCA ≥600 samples → use; else NaN features + `_3c_valid=False`.

### Standardization classes

| Class | Treatment | Examples |
|---|---|---|
| Continuous | z-score cohort-wide | `dtw_distance_norm`, `id_*_mean`, `int_peak_env` |
| Binary presence | leave at 0/1 | `region_coinc_present[7]`, `cca_*_region_present[7]` |
| Sign-valued | leave at {−1, 0, +1} | `lead_follow` |
| Bounded fraction | leave at [0, 1] | `cca_*_sparsity`, `frac_aus_partnered` |

NaN handling: continuous → cohort-median imputation + `_imputed_mask`; binary/sign → 0; episodes with `imputation_fraction > 0.40` → flagged low-quality, excluded from cluster pool, retained for descriptive reporting.

### Clustering hyperparameters

| Stage | Parameter | Value |
|---|---|---|
| UMAP-2D | `n_components`, `n_neighbors`, `min_dist`, `random_state` | 2, 20, 0.05, 42 |
| UMAP-10D | `n_components`, `n_neighbors`, `min_dist`, `random_state` | 10, 20, 0.0, 42 |
| HDBSCAN | `min_cluster_size` | `max(8, int(0.015 × n_episodes))` |
| HDBSCAN | `min_samples`, `cluster_selection_method` | 5, `eom` |
| HDBSCAN | `gen_min_span_tree`, `prediction_data` | True, True |

Multiple resolution cuts (5, 8, 12, 20 clusters) saved for resolution-comparison figures.

## 5. Acceleration

| Stage | Strategy |
|---|---|
| 1 | New `_gpu.batched_dog_pyramid`: torch.conv1d batched across (participants × AUs × scales) — single launch for all 624 channels |
| 3a | Vectorized `np.searchsorted` for per-AU NN-lag, all 52 AUs at once |
| 3b | dtaidistance C-impl + joblib threading across episodes (16 cores) |
| 3c | joblib threading across episodes; cca-zoo PMD releases GIL on dense linalg |
| 3d | Reuse `bl_wavelet._cwt_gpu` + `_coherence_from_coeffs_gpu` — one full-session CWT, slice per episode |
| 5–6 | sklearn UMAP + hdbscan CPU (sub-second at 1000-episode scale) |
| 8 | Threaded surrogates (bootstrap/null/time-reversed/LOO) |

Per `project_win_torch_dll_fix.md`: entry scripts import torch BEFORE numpy; joblib uses threading backend on the Windows torch+numpy stack.

End-to-end on cohort: ~35 min including validation suite.

## 6. Validation suite (Stage 8)

| Test | Pass criterion |
|---|---|
| Bootstrap stability (200×, 80% subsample) | mean ARI > 0.6, ≥70% of clusters present in ≥80% of bootstraps |
| Block-shuffle null (5–10 s blocks of one role) | shuffled clusters have intra-cluster distance ≥1 SD looser than real |
| Time-reversed control (one role reversed) | synchrony features lose structure; identity features preserved |
| Leave-one-dyad-out reproducibility | mean ARI > 0.7 across 19 LOO folds |
| Sanity dyads (pseudo-dyad + self-vs-self) | pseudo-dyad collapses to noise / one degenerate cluster; self-vs-self collapses to one dominant zero-lag cluster |
| Threshold-perturbation sensitivity | episode count stable to ±20% perturbation of `T*` |

## 7. Reporting (Stage 9)

Three deliverables per dyad:

- **(a) Repertoire signature**: per-dyad distribution over cohort cluster IDs → `per_dyad_repertoire.csv` + 19-dyad stacked-bar figure.
- **(b) Expressivity profile**: 7 metrics (episode_rate_overall, episode_rate_per_condition, total_active_fraction, mean_episode_intensity, mean_episode_duration, au_diversity, dyadic_intensity_balance) → `per_dyad_expressivity.csv` + scatter figure.
- **(c) Repertoire × rSLDS-state cross-tab**: `P(cluster | rSLDS_state)` from `results/mvp/<sid>/mvp_rslds_results.npz` + cluster labels → heatmap figure.

Plus `cohort/REPORT.md` — auto-generated summary with embedded figures.

## 8. Implementation order

1. Install missing deps (`cca-zoo`, `hdbscan`, `pycwt`).
2. Augment `face_event_coincidence.detect_au_events` with Hölder α (`return_extra=True` keyword; backward-compatible).
3. Add `AU_REGIONS_7` to `cadence/constants.py`.
4. Build `cadence/synchrony/` package skeleton.
5. Stages 0+1 + sanity figure on y_06; eyeball events overlay before continuing.
6. Stage 2 + sanity figure on y_06.
7. Stages 3a–3d (in order); dimensions match Section 4 design.
8. Stage 4 assembly + diagnostics figure on y_06.
9. Cohort fan-out: run Stages 1–4 on all 19 sessions.
10. Stages 5–6 cohort clustering.
11. Stage 7 per-cluster figures (drives the paired-face schematic dev).
12. Stage 8 validation.
13. Stage 9 reporting (cross-tab waits for V11/MVP regen to finish).

Single-dyad y_06 verification at three checkpoints: after Stage 1 (events overlay), after Stage 2 (episodes overlay), after Stage 6 (cohort clustering — validate clusters look interpretable on y_06's contributions before fanning out cohort interpretation).

## 9. Out of scope (deferred)

- 3D rendered face (Option 3 from the cluster-face discussion) — schematic + medoid playback covers the scientific need.
- Per-dyad clustering with cross-dyad alignment (Approach B from the brainstorm) — pooled clustering chosen for cleaner signature comparison.
- Feeding cluster-occupancy timecourses back into rSLDS as new channels (Q6 option c) — separate workstream.
- Cross-cohort comparison (this dyad cohort only). Future cohorts can project onto saved standardization params.

## 10. Spec self-review

**Placeholders**: none.
**Internal consistency**: Section 4 standardization classes match Section 3 feature definitions. Section 5 hyperparameters match Section 3 acceleration story (sklearn UMAP/hdbscan CPU is fine at scale).
**Scope**: focused on the synchrony repertoire as a parallel scientific output of the MVP cohort. Does not modify rSLDS, MVP scaffold, or any production channel.
**Ambiguity**: AU lists per region in `AU_REGIONS_7` are tentative — final mapping verified against canonical `BLENDSHAPE_NAMES` during step 8.3 of implementation. CCA solver class (`SCCA_PMD` vs `SCCA`) resolved at install time depending on `cca-zoo` version.
