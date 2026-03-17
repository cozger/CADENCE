# CADENCE v2 Implementation TODO

## Phase A: Wavelet EEG Features
- [ ] A1. Create `cadence/data/wavelet_features.py` — Morlet CWT analytic signal extraction
  - [ ] `_morlet_wavelet_bank()` — complex Morlet wavelets at 20 log-spaced frequencies
  - [ ] `_cwt_gpu()` — grouped conv1d CWT (60s chunks, 5s overlap)
  - [ ] `_analytic_to_features()` — real/imag parts, z-scored
  - [ ] `extract_wavelet_features()` — main entry point (N,160) output at 5Hz
- [ ] A2. Create `cadence/data/interbrain_features.py` — inter-brain phase synchrony
  - [ ] `_compute_phase_diff()` — cos/sin of phase difference
  - [ ] `extract_interbrain_features()` — main entry (N,160) output at 5Hz
- [ ] A3. Update `cadence/constants.py` — ROI defs, wavelet names, v2 specs
  - [ ] `EEG_ROIS` dict
  - [ ] `WAVELET_CENTER_FREQS` array
  - [ ] `WAVELET_FEATURE_NAMES` (160 names)
  - [ ] `INTERBRAIN_FEATURE_NAMES` (160 names)
  - [ ] `MODALITY_SPECS_V2`, `MODALITY_ORDER_V2`
- [ ] A4. Update `cadence/data/alignment.py` — wavelet/interbrain extraction paths
  - [ ] V2 extraction path conditioned on `pipeline == 'v2'`
  - [ ] Inter-brain computation (both participants' raw EEG)
  - [ ] Cache upgrade for missing wavelet keys
- [ ] A-verify: Run wavelet extraction on y_06, inspect output shapes

## Phase B: Expand Other Modalities
- [ ] B1. Update `cadence/data/preprocessors.py` — Blendshape PCA + derivatives
  - [ ] `_compute_temporal_derivatives()` — smoothed finite differences
  - [ ] `extract_blendshapes_v2()` — PCA 52->15 + 15 derivatives + activity = 31
- [ ] B2. Pose features — no changes needed (keep current 41)
- [ ] B3. Update `cadence/data/preprocessors.py` — Cardiac expansion
  - [ ] Add RMSSD derivative = 7 features total
- [ ] B4. Update `cadence/constants.py` — v2 modality names
  - [ ] `BL_FEATURE_NAMES_V2` (31 names)
  - [ ] Update `MODALITY_SPECS_V2` for blendshapes_v2, ecg_features_v2
- [ ] B-verify: Run on y_06, verify PCA variance, derivative dynamics

## Phase C: Group Lasso Solver
- [ ] C1. Create `cadence/regression/group_lasso.py` — FISTA solver
  - [ ] `GroupLassoSolver.__init__()` — group definitions
  - [ ] `_compute_lipschitz()` — power iteration for step size
  - [ ] `_block_soft_threshold()` — proximal operator
  - [ ] `fit()` — FISTA proximal gradient descent
  - [ ] `selected_groups()` — extract surviving groups
- [ ] C2. Create `cadence/regression/time_blocked_cv.py` — contiguous time-block CV
  - [ ] `create_time_blocks()` — folds with 30s gaps
  - [ ] `cross_validate_lambda()` — lambda path + CV scoring
- [ ] C-verify: Test on synthetic sparse data (5 true groups out of 200)

## Phase D: Two-Stage Coupling Estimator
- [ ] D1. Update `cadence/coupling/estimator.py` — Stage 1 discovery
  - [ ] `_stage1_discovery()` method
  - [ ] Full-feature design matrix (no PCA)
  - [ ] Group lasso with time-blocked CV per pathway
  - [ ] Return DiscoveryResult
- [ ] D2. Update `cadence/coupling/estimator.py` — Stage 2 estimation
  - [ ] `_stage2_estimate()` method
  - [ ] Nonlinear channels (x²)
  - [ ] Moderation terms (HR, RMSSD * basis columns)
  - [ ] EWLS on reduced feature set
- [ ] D3. Update `cadence/coupling/pathways.py` — v2 modalities
  - [ ] `get_modality_pathways_v2()`
  - [ ] `get_pathway_category()` — fast/medium/slow
  - [ ] `get_feature_groups_v2()`
- [ ] D4. Pathway-specific temporal parameters
  - [ ] Update configs/default.yaml with pathway_temporal
  - [ ] Estimator resolves params per pathway category
- [ ] D-verify: Run two-stage on y_06, compare v1 vs v2, check VRAM

## Phase E: Multi-Session Discovery Script
- [ ] E1. Create `scripts/run_discovery.py`
  - [ ] Stage 1 on all sessions
  - [ ] Cross-session consistency aggregation
  - [ ] Stage 2 on each session with consistent feature set
  - [ ] Save discovery report
- [ ] E2. Update `scripts/run_session.py` — `--pipeline v2` flag

## Phase F: Synthetic Validation
- [ ] F1. Update `cadence/synthetic.py` — wavelet-aware generators
  - [ ] `build_synthetic_wavelet_session()` — sinusoidal EEG coupling
  - [ ] `build_synthetic_interbrain_session()` — phase sync coupling
- [ ] F2. Create `scripts/run_synthetic_v2.py` — test battery
  - [ ] Frequency-specific recovery test
  - [ ] Phase-driven recovery test
  - [ ] Inter-brain phase recovery test
  - [ ] False positive control test
  - [ ] Cross-session consistency test

## Phase G: Visualizations
- [ ] G1. Create `cadence/visualization/spectral.py`
  - [ ] `plot_spectral_coupling_map()` — freq × ROI heatmap
  - [ ] `plot_coupling_spectrum()` — coupling vs frequency line plot
- [ ] G2. Create `cadence/visualization/discovery.py`
  - [ ] `plot_discovery_report()` — cross-session selection matrix
  - [ ] `plot_lambda_path()` — CV error vs lambda
  - [ ] `plot_feature_selection_heatmap()`
- [ ] G3. Update `cadence/visualization/heatmaps.py` — expanded modality support

## Phase H: Configuration and Integration
- [ ] H1. Update `configs/default.yaml` — all v2 parameters
  - [ ] pipeline flag
  - [ ] wavelet parameters
  - [ ] blendshapes_v2 parameters
  - [ ] discovery (Stage 1) parameters
  - [ ] stage2 parameters (moderation, nonlinear)
  - [ ] pathway_temporal + pathway_category
- [ ] H2. Update `cadence/config.py` — v2 defaults + deep merge
