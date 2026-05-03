# CLAUDE.md — CADENCE

## Project Overview

**CADENCE** (Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation) is a fully interpretable, regression-based framework for quantifying directed, time-varying, cross-modal interpersonal coupling from continuous multimodal recordings. Replaces MCCT's transformer with basis-expanded distributed lag regression where every quantity is a measured signal or regression weight.

**Fully standalone** — all needed data pipeline code from MCCT is copied with internal imports. No runtime dependency on MCCT.

## Architecture

**MVP Pipeline** (current grant baseline — 2026-05-02): 6-channel observation set + 2-channel covariates, hierarchical rSLDS over 19 canonical sessions. **Production BL channel: `bl_event_coincidence`** — surrogate-z dyadic coincidence of facial-activity peaks (per-participant peaks of `au_activity` envelope above session p70, ±500ms tolerance, 200 circular shifts). Replaces obsolete `bl_expr` (Morlet wavelet coherence, fired on z-score noise during quiet periods) and `bl_activity_conc` (`(z_P1+z_P2)/2`, was shared activity LEVEL not coupling — confounded by who's talking). Implementation: `cadence/significance/face_event_coincidence.py`. Per-condition validation on y_06: conv_1=+0.34, conv_2=+0.36, meditation=−0.16/−0.17, baselines≈0 — clean discrimination between interactive coupling and quiescent periods.

**V12 Pipeline** (planned — pending stereo gaze hardware): Gaze coupling from stereo-calibrated gaze rays. 29-31D observation + 9D covariates. Extends V11 with gaze concordance (obs: mutual attention intensity), gaze asymmetry (cov: who attends), gaze approach rate (cov: convergence predicting state transitions). First modality where raw measurement IS the coupling — no surrogates needed. Design doc: `docs/v12_gaze_design.md`.

**V11 Pipeline** (current — production): Burst coincidence + transfer entropy scaffold. 26D observation + 7D transition covariates. Extends V10 with TE concordance (2ch obs: bidirectional flow), burst coincidence (3ch obs), and TE asymmetry (2ch cov: directionality as transition modulator). **Note:** dyn_theta/alpha/beta collapsed to `dyn_mean` (diagnostic suite confirmed r=0.99 collinearity; all three are EWMAD of concordance channels which share the same slow timescale). 28D→26D.

**rSLDS engine** (`cadence/significance/rslds_model.py`): numpy + Numba (Phase 0 patches + Phase 0.5 Tier A patches as of 2026-05-02). Phase 0 alone delivers 5.8× hierarchical-fit speedup vs unpatched (138 min → 23.6 min on 19-session production cohort); Tier A adds ~12% on top via vectorized K loop in `slds_e_step` and a `nogil=True` Numba kernel for `_log_transitions_recurrent`. JAX migration deferred to ~2027-2028 when cohort approaches ~100+ sessions; trigger criteria and decision gates documented in `docs/dynamax_migration_plan.md` §0.

**V10 Pipeline** (extended by V11): Graph-enhanced rSLDS with LZ complexity. 23D observation + 5D transition covariates. Extends V8.2 with LZ complexity channels, graph modularity, and coupling flexibility as transition covariate.

**V7 Pipeline** (feature extraction layer): CWT wavelet decomposition + wavelet coherence.

1. **EEG**: `fast_cycles.py` — GPU multi-band (theta/alpha/beta) cycle analysis. Extracts per-cycle volt_amp, cross-correlates between participants with 200 surrogates. Combined Stouffer z across bands. Primary metric: amplitude co-modulation (z=+10 on real data).
2. **BL**: `bl_wavelet.py` — CWT decomposition of all 52 AU timeseries into frequency bands (state <0.5 Hz, expression 0.5-2 Hz, speech 2-7 Hz). Wavelet coherence between participants gives multi-scale coupling profile. GPU-accelerated surrogate z-scoring (200 surrogates in 0.6s). Low-pass filter at 8 Hz removes tracker noise. Semisynthetic validated: AUC=0.78 at kappa=0.4 with realistic smile injection (d=0.97).
3. **Output**: Per-segment coherence spectrograms + band-averaged coupling z-scores + continuous band-power timecourses.

**V6 Pipeline** (legacy, kept for reference): Saliency-based facial event detection via `bl_coupling.py`. Affect-only AU saliency + phasic smile scoring + speech/blink gating. Replaced by V7 wavelet approach.

**V2 Pipeline** (legacy, kept for reference): EWLS regression on z-scored PCA features via `CouplingEstimator`. Not used in production.

## Environment

Uses the MCCT conda environment (Python 3.11, PyTorch, scipy, numpy, matplotlib, pyyaml).

```bash
conda activate MCCT
```

## Key Commands

```bash
# V11 Production pipeline (burst coincidence + TE scaffold)
python scripts/_run_scaffold_v11.py                                 # Single session (y_06), 26D scaffold
python scripts/_run_scaffold_v11.py --session y_17                  # Specific session
python scripts/_run_scaffold_v11.py --all                           # All sessions (n_jobs=4)
python scripts/_run_v11_hierarchical.py                             # Hierarchical rSLDS (26D + 7D covariates)

# V10 Production pipeline (graph-enhanced rSLDS, extended by V11)
python scripts/_run_scaffold_v10.py                                 # Single session (y_06), 23D scaffold
python scripts/_run_scaffold_v10.py --session y_17                  # Specific session
python scripts/_run_scaffold_v10.py --all                           # All sessions (n_jobs=8)
python scripts/_run_v10_hierarchical.py                             # Hierarchical rSLDS (23D + 5D covariates)

# V10 Validation
python scripts/_test_v10_semisynthetic_battery.py --all             # Full battery (LZ, regression, null)
python scripts/_test_v10_semisynthetic_battery.py --phase 1 --quick # LZ detection only (3 pairs)

# V11 Burst coincidence (post-hoc analysis on V10, also integrated into V11 scaffold)
python scripts/run_burst_coincidence_analysis.py                    # Single session (y_06)
python scripts/run_burst_coincidence_analysis.py --all              # All sessions (~90s)
python scripts/plot_burst_coincidence.py                            # 4 figures (timeline, bars, scatter)

# V8.2 Burst analysis (production)
python scripts/run_burst_analysis.py                                # All sessions: per-segment burst rates + asymmetry
python scripts/run_burst_analysis.py --session y_06                 # Single session
python scripts/run_burst_analysis.py --no-plots                    # JSON only (fast)

# V7 Production pipeline (wavelet)
python scripts/_run_v7_timeline.py                                  # Full session timeline (EEG + BL wavelet)
python scripts/_run_v7_individual_scalograms.py                     # Per-participant CWT scalograms

# V7 Validation
python scripts/_test_wavelet_validation.py                          # Ground truth checks (speech, coherence, null)
python scripts/_test_wavelet_semisynthetic_v3.py                    # Semisynthetic AUC (real smile injection, 42 pairs)

# V6 session runner (still used for EEG + BL segment extraction)
python scripts/run_session_v6.py --session y_06                    # Single session (EEG + BL)
python scripts/run_session_v6.py --session y_06 --bl-only          # BL only
python scripts/run_all_sessions_v6.py                               # All sessions

# EEG validation
python scripts/test_eeg_pipeline.py --quick     # EEG: fast_cycles multi-band (~30s)

# Legacy V2 pipeline (kept for reference)
python scripts/run_session.py --session y_06    # V2 CouplingEstimator
```

## Project Structure

```
CADENCE/
  cadence/
    __init__.py, config.py, constants.py, synthetic.py, surrogates.py
    ingest/        # Layer 1 (Data Pipeline v1): raw XDF -> data/digest/v1/
                   # roles, schema, quality, xdf_reader, digest, cli
    preprocess/    # Layer 2: per-modality cleaning -> data/preproc/<mod>/v1/
                   # eeg/ (MATLAB bridge), face/, ecg/, pose/ (+ pose_subset)
    data/          # [DEPRECATED] eager-loading legacy dict view shim only;
                   # all signal code moved to ingest/ + preprocess/
    basis/         # raised_cosine.py, design_matrix.py
    regression/    # ewls.py (core), ridge.py, ftest.py, group_lasso.py
    coupling/      # pathways.py, estimator.py (CouplingEstimator), discovery.py, serialization.py
                   # interbrain.py (cross-participant features)
    io/            # paths.py, cache.py
    significance/  # fast_cycles.py (GPU EEG, extract_burst_grids), burst_coincidence.py (V11 burst coincidence), bl_wavelet.py (V7 BL wavelet), burst_analysis.py (V8.2 bursts), lz_complexity.py (V10 LZ76), spectral_graph.py (V9/V10 graph theory), rslds_model.py (rSLDS/SLDS), bl_coupling.py (V6 legacy), coherence_localization.py (PLV)
    visualization/ # kernels.py, timecourse.py, heatmaps.py, comparison.py, sparsity.py
  configs/
    default.yaml
    session_quality.yaml  # canonical session registry (drives --all)
  data/
    digest/v1/<sid>.npz + .json     # Layer 1 outputs
    preproc/{eeg,face,ecg,pose}/v1/ # Layer 2 outputs
    matlab/<sid>_p{1,2}_clean.mat   # MATLAB-cleaned EEG with .mat.json sidecars
  scripts/         # run_session, run_all_sessions, _audit_xdf_inventory, _relocate_clean_mats, etc.
  results/
```

## Config

All parameters in `configs/default.yaml`. Key settings:
- `session_cache`: [LEGACY] Points to MCCT's session_cache directory.
  Post-2026-05-01 the canonical cache is `data/digest/v1/` + `data/preproc/<mod>/v1/`;
  this field is retained only for legacy script paths and ignored by the new shim.
- `ewls.tau_seconds`: 30s exponential decay (time locality)
- `basis.layer1.n_basis`: 8 raised cosine basis functions
- `basis.layer1.max_lag_seconds`: 5.0s maximum lag
- `autoregressive.order`: 3 AR lags
- `significance.surrogate.n_surrogates`: 100 circular shifts (session-level)
- `significance.max_pathway_p`: 0.7 (skip Stage 2 for high session-level p)
- `significance.timepoint.n_surrogates`: 20 (per-timepoint)
- `significance.timepoint.surrogate_eval_rate`: 1.0 Hz (surrogates at lower rate for speed)
- `significance.fdr_correction`: Benjamini-Hochberg
- `interbrain.min_freq_hz`: 4.0 (exclude delta band — artifact-prone on Emotiv EPOC)
- `interbrain.surrogate_method`: fourier_phase (stronger null for autocorrelated features)

## Data Pipeline v1 (2026-05-01 — production)

Replaces the MCCT-style session cache and the EEGLAB-wavelet pipeline cache as the
single source of truth for raw-data ingestion. **Two cleanly-versioned layers:**

```
raw sessions/<sid>.xdf
   |
   |-- Layer 1 (cadence.ingest)     -> data/digest/v1/<sid>.npz + .json
   |
   |-- Layer 2 (cadence.preprocess) -> data/preproc/{eeg,face,ecg,pose}/v1/<sid>.npz + .json
   |
   |-- Analytics (V11/V10/V8.2 scaffolds, rSLDS, wavelet)
```

`xdf_md5` keys flow downstream: a digest re-run invalidates all four preproc
artifacts via the staleness predicate. Independent versioning means bumping
`preproc/eeg/v1` doesn't force re-running pose/face/ecg.

**Single classification scheme:** `configs/session_quality.yaml` is hand-maintained
(drafted by `scripts/_audit_xdf_inventory.py`). Carries `canonical: bool`,
per-participant `excluded_modalities`, and per-session `marker_overrides`
(rename/drop with t_min/t_max windows for genuine recording errors —
mis-clicks, spurious labels).

**Three pose formats supported transparently:** mediapipe33 (132 ch),
mediapipe33_meta (133 ch), wholebody_133 (400 ch — RTMW). Format dispatch in
`cadence.preprocess.pose.pose_subset.normalize_pose_to_mp33` returns
`(N, 33, 4)` MediaPipe-shaped output regardless of source.

**Pose features are visibility-aware** (deliberate behavioral fix): groups
with no visible landmarks return NaN (filtered by NaN-safe statistics)
instead of biasing centroids toward origin. Frames with face occlusion
produce different head features post-migration — this is intended.

**Backward compatibility:** `cadence.data` is now an eager-loading dict-view
shim. Pre-existing scripts using `load_session_from_cache(sid)` keep
working unchanged; the shim materializes ~30 legacy keys per participant by
reading from digest + 4 preproc artifacts and write-protects via
`setflags(write=False)` for joblib safety.

### CLI

```bash
# Layer 1: digest
python -m cadence.ingest --session y_06
python -m cadence.ingest --all --n-jobs 4

# Layer 2: per-modality preprocessing (idempotent; honors digest_xdf_md5 staleness)
python -m cadence.preprocess.pose --all
python -m cadence.preprocess.face --all
python -m cadence.preprocess.ecg  --all
python -m cadence.preprocess.eeg  --all   # requires fresh data/matlab/<sid>_p{1,2}_clean.mat

# Step 0 audit — regenerates configs/session_quality.yaml preserving user-edited fields
python scripts/_audit_xdf_inventory.py

# Relocation of legacy MATLAB-cleaned EEG into data/matlab/ with provenance sidecars
python scripts/_relocate_clean_mats.py --canonical-only
```

See `docs/data_pipeline_v1.md` for the canonical schema, role resolution
algorithm, marker pipeline, migration record (2026-05-01), and the
"How to add a modality" recipe.

### Sessions still needing fresh MATLAB EEG cleaning

y04_020626, y11_022526, y24_022526, y_53_04302026 (no legacy clean.mat
existed). Until the user runs `analysis/eeglab_wavelet/matlab/preprocess_eeg.m`
on these, `cadence.preprocess.eeg.has_fresh_clean_mat` returns False and
the EEG submodule raises with a clear error.

## Reference Papers

`G:\My Drive\ARPA Shared Documents\Reference Papers` — shared Google Drive folder with all project literature PDFs.

## Reference Files

- `docs/validate_blendshape_isolation.py` — canonical 52-name `BLENDSHAPE_NAMES` list (MediaPipe FaceLandmarker blendshapes, indices 0–51, starting with `_neutral`) and `BLENDSHAPE_LANDMARK_INDICES`. Use as the source of truth when referring to face AUs by name rather than integer index. `AFFECT_AUS = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]` maps to cheekSquintL/R, mouthDimpleL/R, mouthFrownL/R, mouthSmileL/R, noseSneerL/R.

## Session Protocols

Two distinct experimental protocols in the dataset:

**Meditation protocol** (y_06, y_17, y_19, y_11, y_04, y_24):
- base_EO → base_EC → conv_1 → meditate_B → meditate_K → conv_2

**Psychoeducation (PE) protocol** (y_01, y_05, y_10, y_32, y_41):
- base_EO → base_EC → conv_1 → PE_1 → PE_2 → conv_2
- PE = therapist presents educational material (control for meditation)

Markers: `base_EO_start/stop`, `base_EC_start/stop`, `conv_1_start/stop`, `PE_start/stop` or `PE_1_start/stop` + `PE_2_start/stop`, `meditate_B_start/stop`, `meditate_K_start/stop`, `conv_2_start/stop`, `baseline_start/stop` (single block variant)

## Therapist/Patient Role Resolution

**Stream order (P1/P2) in the XDF is NOT consistent across sessions.** Always
resolve roles before any per-role analysis. Never label outputs with P1/P2.

Single source of truth: `cadence.ingest.roles.resolve_roles(streams)`.

- Therapist alias: `{"Sway", "RA"}` (case-insensitive). Sway = older sessions;
  RA = research-assistant alias used in early-2026 sessions onward.
- Patient pattern: `^(?:[Yy]_?\d+|SYNC_\d+)(?:[_-].*)?$` (suffix-tolerant for
  date-suffixed names like `Y01_020626`).
- Source: read from BOTH `P1_landmarks` and `P2_landmarks` `desc->participant_name`.
- Exactly-one-therapist + exactly-one-patient → resolved. Otherwise raise
  `UnresolvedRolesError`; the digest CLI logs and skips.

`role_source ∈ {"xdf_landmark", "xdf_ra_alias", "unresolved"}` for downstream
audit. After 2026-05-01 migration the legacy two-stage fallback is gone —
every digest ships with resolved roles or refuses.

Six previously-mislabelled sessions (Y_55, y_59, y_64, y_66, y26, y_65) now
correctly resolve as therapist/patient under the new resolver.

## V10 Graph-Enhanced rSLDS Pipeline (extended by V11)

23D observation vector at 2 Hz + 5D transition covariates. Extends V8.2 with LZ complexity, graph modularity, and graph-spectral transition covariates. Pure-numpy LZ76 (no Numba).

**23D observation vector** (V8.2 18D + 5 new channels):
- EEG imaginary coherence theta/alpha/beta (3ch)
- EEG concordance theta/alpha/beta (3ch)
- EEG dynamics theta/alpha/beta (3ch)
- EEG asymmetry theta/alpha/beta (3ch)
- BL expression (1ch) + BL activity concordance (1ch)
- ECG LF/HF (2ch), Resp (1ch), Pose (1ch)
- **LZ concordance theta/alpha (2ch)**: shared neural complexity state from Hilbert envelope LZ76 on frontal ROI
- **LZ asymmetry theta/alpha (2ch)**: therapist - patient complexity, role-corrected
- **Graph modularity (1ch)**: Louvain Q on windowed (90s) correlation graph of base 18D

**5D transition covariates** (modulate state transitions, not emissions):
- z_slow PCs (2D): slow behavioral drift
- **Coupling flexibility (1D)**: graph spectral energy ratio from V9 — STRONGEST covariate (max|S|=1.659)
- **Lambda-2 (1D)**: algebraic connectivity of modality graph
- **Graph change-point (1D)**: topology transition score (derivative of modularity + lambda-2)

**Design decisions**:
- Flexibility is a **covariate, not observation** — avoids circularity (computed from same 18D base)
- Graph metrics computed from base 18D (channels 0-17), not 23D — avoids second-order circularity
- graph_centrality_eeg dropped: near-constant timecourse → spurious AUC after prewhitening
- LZ on **Hilbert amplitude envelope**, not raw bandpass — amplitude modulation changes LZ; zero-crossings don't
- c_shrinkage=0.3: emission loading C[k] regularized toward shared C_mean across states (Phase 3.2)
- Pure-numpy LZ76 (~0.74ms/window) — antropy's Numba JIT takes 30-60s warmup for negligible speedup

**Model**: rSLDS K=4, D_latent=3, D_obs=23, D_input=5, recurrent=True, c_shrinkage=0.3, n_factors=2, null_state=True, sticky_strength=3.0, viterbi_min_dwell=20.

**Key results (n=12 sessions)**:
- Normalized BIC 4.8% better than V8.2 (1.4298 vs 1.5017 per-channel-per-timepoint)
- State structure preserved: NULL/COUP/SHARED/OTHER
- Coupling flexibility is strongest transition covariate (max|S|=1.659, 5x > z_slow PCs) — confirms Gordon 2025
- Lambda-2 second strongest (max|S|=0.992)
- LZ semi-synthetic: monotonic AUC 0.51→0.57 (alpha, kappa 0→0.4); null integrity 23/23 PASS
- V8.2 asymmetry reversal replicates in V10 (therapist-dominant PE, patient-dominant conv_2)

**V10 per-condition findings (2026-03-31)**:

Coupling flexibility by condition (strongest → weakest):
- conv_1 (0.40) > meditate_B (0.33) > meditate_K (0.27) > conv_2 post-meditation (0.18)
- PE has no effect on flexibility: conv_1 (0.39) ≈ PE_1 (0.39) ≈ PE_2 (0.42) ≈ conv_2 (0.34)
- **Meditation rigidifies coupling, and rigidity persists into post-meditation conversation** (conv_2 med=0.18 vs PE=0.34, p=0.177 at n=6/5)
- Baselines matched across protocols (base_EO: med=0.38, PE=0.41; base_EC: both 0.42)

Burst rates (per minute) by condition:
- Conversation has highest burst rates across all modalities (EEG phase 0.79, power 0.83, Face+Body 0.67, LZ 1.04)
- meditate_K maximally suppresses Face+Body (0.13/min) and EEG Power (0.26/min)
- LZ complexity bursts track same hierarchy as EEG/BL (novel modality, not redundant)

EEG asymmetry (therapist - patient) by condition:
- PE_1/PE_2: therapist-dominant (theta +0.06/+0.08, beta +0.11/+0.13) — teaching
- conv_2: patient-dominant (theta -0.10, beta -0.16) — post-intervention reversal
- Baselines null (correct control)

Coupling excess analysis (surrogate-calibrated, standard test):
- **Session-adaptive burst thresholds INVALID** — self-normalize via prewhitening z-scoring. Replaced by continuous coupling excess: `excess_z(t) = (real - surrogate_mean) / surrogate_std` using 200 circular-shift surrogates per session (<1s).
- **Conversation coupling is GENUINE** (above circular-shift null):
  - Postural: z=+0.85, Facial: z=+0.50, LZ Shared: z=+0.54, EEG Phase: z=+0.25
- **Meditation SUPPRESSES coupling below null**: EEG Phase z=-0.68, Postural z=-0.50, LZ z=-0.53 (correct: patient eyes-closed, no interaction)
- **Facial coherence is universally genuine** — positive excess in ALL conditions (z=+0.22 to +0.72)
- **LZ bidirectional**: conversation shows POSITIVE excess (shared complexity enhancement), meditation shows NEGATIVE (complexity divergence)
- Channel tiers: Tier 1 (coupling-specific: ImCoh, BL expr, LZ conc, Resp, Pose), Tier 2 (activity-confounded: Concordance, BL act, ECG), Tier 3 (direction: Asymmetry, Dynamics)
- Scripts: `cadence/significance/coupling_bursts.py` (module), `scripts/run_coupling_excess_analysis.py` (runner), `scripts/_validate_v10_pseudo_dyad.py` (raw-recompute validation)

Key scripts:
- `scripts/_run_scaffold_v10.py` — 23D scaffold + 5D covariates (~30s/session)
- `scripts/_run_v10_hierarchical.py` — hierarchical rSLDS (23D, 5D cov, ~19min for 12 sessions)
- `scripts/_test_v10_semisynthetic_battery.py` — LZ detection, V8.2 regression, null integrity
- `cadence/significance/lz_complexity.py` — pure-numpy LZ76 + concordance/asymmetry
- `cadence/significance/spectral_graph.py` — modularity, centrality, change-point, flexibility
- `cadence/synthetic_v10.py` — LZ injection for semi-synthetic validation

Output: `results/v10/` (per-session scaffolds + hierarchical results)

## V11 Scaffold: Burst Coincidence + Transfer Entropy (current — production)

26D observation vector at 2 Hz + 7D transition covariates. Extends V10 (21D after dyn collapse) with TE concordance (2ch obs), burst coincidence (3ch obs), and TE asymmetry (2ch cov). Promotes V10 post-hoc layers to first-class channels.

**Key design — TE observation/covariate decomposition:**
Transfer entropy is decomposed into concordance (observation) and asymmetry (covariate):
- `te_conc = (z_T→P + z_P→T) / 2` — total bidirectional information flow. This is a **state property**: high during interactive coupling, low during independent states. Belongs in observations because COUP state should load positively on it.
- `te_asym = z_T→P - z_P→T` — who leads the information flow. This is a **transition modulator**: when therapist leads → more likely to stay in teaching/NULL state; when patient leads → transitions to interactive COUP. Belongs in covariates (like coupling flexibility).

**Rationale:** Raw TE asymmetry conflates "bidirectional coupling" (both high, diff≈0) with "no coupling" (both low, diff≈0) — the rSLDS cannot distinguish these because both produce near-zero asymmetry. TE concordance separates them: high conc + low asym = interactive exchange; low conc + low asym = independent; moderate conc + high asym = didactic/teaching.

**26D observation vector** (V10 21D + 5 new channels; was 28D before dyn collapse):
- V10 channels (unchanged): ImCoh θ/α/β, Conc θ/α/β, **dyn_mean** (collapsed from dyn_θ/α/β — r=0.99 collinear, all are EWMAD of concordance at the same slow timescale), Asym θ/α/β, BL, ECG, Resp, Pose, LZ conc θ/α, LZ asym θ/α, Graph modularity
- **TE concordance theta**: bidirectional burst-level information flow, (z_P1→P2 + z_P2→P1)/2, surrogate z-scored per timepoint (60s windows, 200 surrogates, GPU)
- **TE concordance alpha**: same for alpha band
- **Burst coincidence theta**: "did both participants burst within ±500ms?" — z-scored against 200 surrogates, ROI-averaged across channels
- **Burst coincidence alpha**: same for alpha band
- **Burst coincidence beta**: same for beta band

**7D transition covariates** (V10 5D + 2 TE asymmetry):
- z_slow PCs (2D), coupling flexibility, lambda-2, graph change-point (V10, unchanged)
- **TE asymmetry theta (cov 5)**: therapist→patient - patient→therapist, sign-corrected, prewhitened
- **TE asymmetry alpha (cov 6)**: same for alpha band

**te_asym_beta omitted**: collinear with existing asym_beta (r=-0.74).

**Design decisions**:
- Single `extract_burst_grids()` call shared between TE and coincidence (avoids redundant GPU bandpass + cycle extraction)
- Prewhitening on full 26D jointly (not appended to V10 results) — AR(1) + standardization must be computed jointly
- TE covariates also prewhitened (AR(1) + standardization) before inclusion in U matrix
- TE sign convention: positive = therapist-leads (multiplied by asym_sign, consistent with existing asym_* channels)
- c_shrinkage=0.2 (reduced from V10's 0.3 to let new channels express with 26D)
- n_jobs=4 for --all (GPU-heavy: burst grid extraction + TE surrogate computation)

**General principle for rSLDS channel placement:** Symmetric/magnitude measures belong in emissions (they characterize the state). Signed/directional measures belong in transitions (they modulate state dynamics). This parallels coupling flexibility (unsigned graph topology measure → strongest covariate) vs ImCoh (unsigned coupling intensity → observation).

**Model**: rSLDS K=4, D_latent=3, D_obs=26, D_input=7, recurrent=True, c_shrinkage=0.2, n_factors=2, null_state=True, sticky_strength=3.0, viterbi_min_dwell=20.

**V11 vs V10 model comparison (n=12 sessions, 2026-04-02):**

BIC comparison across different D_obs is not apples-to-apples — modeling 5 extra channels inherently costs log-likelihood. Fair decomposition:
- V10 BIC/T/D=1.4298, V11 BIC/T/D=1.5058 (+5.3% raw gap)
- 97% of BIC gap is from -2*LL (more channels to model), 3% from parameter penalty (only 32 extra params)
- Expected -2*LL cost of 5 unit-variance Gaussian channels: 969,291. Actual cost: 614,619 (37% better than null model)
- The rSLDS explains significant variance in the new channels — the BIC gap is not a fit quality issue but the mathematical cost of additional observation dimensions

Key V11 results:
- State labels preserved: NULL/COUP/SHARED/OTHER (identical to V10)
- NULL state usage: 20.2% (healthy, vs V10's 23.2%). Early V11 design with TE asymmetry as observation inflated NULL to 47.7% — the concordance/covariate decomposition fixed this
- COUP state usage: 7.0% (matches V10's 6.9%)
- burst_coinc_theta loads +0.15-0.16 in COUP state (stable across all model configs)
- te_conc_theta/alpha: near-zero state loadings (TE concordance signal may need larger windows or more sessions)
- Coupling flexibility remains strongest covariate: max|S|=0.824 (lower than V10's 1.659, partially absorbed by new observation channels)
- Lambda-2 second strongest: max|S|=0.962
- TE asymmetry covariates: modest effects (theta max|S|=0.045, alpha 0.060) — directionality signal is weak per-timepoint but correctly placed as covariate

**Model capacity grid search (D_latent × n_factors, 2026-04-02):**

| D_latent | n_factors | BIC/T/D | vs best |
|----------|-----------|---------|---------|
| 2 | 0 (diagonal) | 1.5278 | +1.5% |
| 2 | 1 | 1.5710 | +4.3% |
| 2 | 2 | 1.5404 | +2.3% |
| 3 | 0 (diagonal) | 1.5525 | +3.1% |
| 3 | 1 | 1.6697 | +10.9% |
| **3** | **2** | **1.5058** | **best** |
| 4 | 2 | 1.5205 | +1.0% |
| 3 | 3 | 1.5354 | +2.0% |

**D_latent=3, n_factors=2 is BIC-optimal.** Key findings:
- D_latent=4: 4th latent axis poorly identified (~1,400 effective obs per state for 28×4 emission matrix). LL worse, not just BIC penalty.
- n_factors=3: 3rd residual factor overfits cross-channel noise. LL worse.
- D_latent=2: loses a load-bearing latent axis (likely separates EEG phase coupling from body/autonomic from burst/LZ).
- n_factors=0 (diagonal): ignores real cross-channel residual correlations (e.g., burst_coinc_theta and burst_coinc_alpha share burst grid extraction noise).
- n_factors=1: optimization trap — single noise axis gets misoriented, distorts C[k] estimation. Worse than both 0 and 2.
- Most configs cluster at BIC/T/D ≈ 1.50-1.55 because D_latent and n_factors are **partially fungible** — they're different ways to explain cross-channel structure, and the model compensates when one is reduced. The dominant signal is discrete state switching (d[k] + gamma), not within-state dynamics.
- The V10-optimal config (D=3, nf=2) being also V11-optimal (26D) suggests this captures a fundamental property of the data — ~3 independent coupling dynamics and ~2 noise correlation axes — rather than being an artifact of observation dimensionality.
- **Scientific conclusions (state structure, condition effects, covariate rankings) are robust to model capacity choices.** They emerge from the data regardless of D_latent/n_factors setting.

Key scripts:
- `scripts/_run_scaffold_v11.py` — 26D scaffold + 7D covariates
- `scripts/_run_v11_hierarchical.py` — hierarchical rSLDS (26D, 7D cov)
- `cadence/significance/burst_coincidence.py` — `compute_burst_coincidence()` (grid-level)
- `cadence/significance/directed_burst_coupling.py` — `gpu_sliding_te_surrogates()` (grid-level, returns p1p2_z + p2p1_z for concordance)
- `cadence/significance/fast_cycles.py` — `extract_burst_grids()` (shared extraction)

Output: `results/v11/` (per-session scaffolds + hierarchical results)

## V11 Burst Coincidence (post-hoc layer on V10)

Native-rate EEG burst coincidence — "did P1 and P2 both burst within ±500ms?" — as a temporally precise coupling metric that bypasses the coupling uncertainty principle of continuous metrics (2s+ windows). Post-hoc layer on V10 scaffold, not a scaffold change.

**Method**: Per-band per-channel burst detection via `extract_burst_grids()` (factored from `eeg_coupling_timecourse()`), resampled to 2 Hz, pointwise AND within ±1 sample (500ms), ROI-averaged across 14 channels, z-scored against 200 circular-shift surrogates. Per-channel rate gate (<1% → excluded) + z-clip [-10, 10] for numerical stability.

**Band-specific burst detection** (see `docs/burst_detection_literature.md` for literature review):
- Theta/alpha: original parameters (mono>0.7, min 2 consecutive cycles) — permissive, since surrogate z-scoring normalizes base rate
- Beta: tighter parameters (mono>0.8, min 3 consecutive cycles) — Cole & Voytek 2019 bycycle defaults, Sherman 2016 (~3 cycles = stereotypical beta event). Fixes beta burst rate from 56% (saturated) to 31%.
- Amplitude threshold p25 for all bands (surrogates normalize overcounting)
- Literature: Rayson 2022 showed alpha is sustained (lagged coherence to 7 cycles), beta is genuinely bursty (drops after 2 cycles). Beta burst coincidence is the most theoretically grounded.

**Key findings (n=11 sessions, 2026-04-01)**:

Per-condition burst coincidence z (surrogate-calibrated):
- **Theta/alpha**: base_EC dominates (theta +0.91, alpha +0.62) — shared eyes-closed resting-state oscillatory synchrony. Conversation null. Meditation modest positive. Distinct from ImCoh (which peaks during conversation).
- **Beta**: PE sessions highest (+0.21/+0.17) — active engagement during psychoeducation. base_EC most negative (−0.17) — no motor/cognitive beta during rest. Conversation variable across dyads (−0.58 to +0.87, mean null). Y_45 shows strong conversation beta coincidence (+0.87), suggesting individual differences in motor synchrony.
- **Beta raw coincidence** is higher during conversation (0.16-0.24) than rest (0.10-0.18), but z-score is null because both participants independently have more beta bursts during conversation — the excess above independent rates is not significant.

Per-rSLDS-state: COUP state has highest theta coincidence (+0.37). Beta near null across states — burst coincidence captures something orthogonal to rSLDS continuous coupling states.

**Complementarity with ImCoh**: ImCoh peaks during conversation (interaction-driven phase coupling). Burst coincidence peaks during rest for theta/alpha (state-driven oscillatory synchrony) and PE for beta (task-driven motor synchrony). The two metrics capture different coupling mechanisms.

Key scripts:
- `cadence/significance/fast_cycles.py` — `extract_burst_grids()` (shared by coupling + coincidence)
- `cadence/significance/burst_coincidence.py` — `compute_burst_coincidence()`, `eeg_burst_coincidence()`
- `scripts/run_burst_coincidence_analysis.py` — post-hoc runner (all sessions, ~90s)
- `scripts/plot_burst_coincidence.py` — 4 figures: timeline, condition bars, state bars, conv-vs-rest scatter

Output: `results/v10/burst_coincidence/`

### Directed Burst Coupling: ECA + Transfer Entropy (2026-04-01)

Directed ECA ("who leads?") and transfer entropy ("does P1's burst history predict P2's bursts?") on the same burst grids. GPU-accelerated per-timepoint surrogate z-scoring (200 surrogates, ~14s/session).

**ECA asymmetry is confounded by burst rate differences** — the participant with more bursts mechanically "leads" more often. ECA-TE anticorrelation (theta r=−0.76, alpha r=−0.90, beta r=−0.67) confirms. **TE is the trustworthy directed coupling metric.** ECA should not be used for directionality.

**Directed TE episode detection** (per-timepoint surrogate z > 2, min 3s, 5s merge):
- **Conversation**: therapist→patient dominates theta (18% vs 5%) and alpha (13-19% vs 2-7%). Beta reverses: patient→therapist (25%).
- **Meditation**: all directed coupling suppressed (1-7% theta/alpha). Patient has strong alpha but coupling is autonomous (eyes closed, no interaction).
- **PE_1**: beta therapist→patient (16%), theta/alpha suppressed. PE_2 reverses to patient→therapist theta/alpha (15%).
- Replicates V8.2 asymmetry reversal finding (therapist-dominant PE, patient-dominant conversation) from an independent metric.

**Beta TE collinearity warning**: beta TE asymmetry correlates r=−0.74 with existing power asymmetry channel. The participant with fewer bursts has more informative (higher Shannon surprise) events, creating inverted rate-driven TE. **Do NOT add beta TE to scaffold.** Theta/alpha TE are safe (|r| < 0.40 with all existing channels).

**V11 scaffold recommendation**: Add `te_asym_theta` + `te_asym_alpha` (2 channels → 25D). Skip `te_asym_beta`. Also add burst coincidence (3 channels → 28D). See `docs/burst_detection_literature.md` for full collinearity analysis.

Key scripts:
- `cadence/significance/directed_burst_coupling.py` — directed ECA, binary TE, GPU sliding-window TE, episode detection
- `scripts/run_directed_burst_coupling.py` — runner (all sessions, ~2.5 min on GPU)
- `scripts/plot_directed_burst_coupling.py` — condition bars, lag profile, ECA-vs-TE scatter

Output: `results/v10/directed_burst_coupling/`

## V8.2 RSLDS Pipeline (extended by V10)

18D observation vector at 2 Hz, prewhitened + standardized:
- EEG imaginary coherence theta/alpha/beta (3ch): Welch CSD phase coupling, 2s windows
- EEG concordance theta/alpha/beta (3ch): shared power state (z_P1+z_P2)/2
- EEG dynamics theta/alpha/beta (3ch): EWMAD of concordance (tau=3s), log-transformed
- EEG asymmetry theta/alpha/beta (3ch): z_therapist - z_patient, role-corrected
- BL expression (1ch): per-segment wavelet coherence z
- BL activity concordance (1ch): shared facial activity
- ECG LF/HF (2ch): Hilbert envelope cross-product of bandpass IBI
- Resp (1ch): phase coherence cos(phi1-phi2)
- Pose (1ch): multi-lag upper-body velocity cross-product z (±5s lag bank, max over lags)

Model: rSLDS with null-state constraint (d_emit[0]=0), sticky transitions (kappa=3), asymmetric transition prior, constrained Viterbi (10s min dwell). Hierarchical pooling across sessions. D_latent=3 (BIC-optimal).

Key findings:
- EEG cross-product coupling (volt_amp, envelope correlation) does NOT separate conditions (all p>0.29). Replaced by imaginary coherence (phase coupling, p=0.016) and concordance (shared state, p=0.031).
- Pose zero-lag cross-product cannot detect lagged coupling (velocity autocorrelation at lag>0 is negative for first-differenced signals). Replaced by multi-lag ±5s bank: conv vs meditation p=0.006.
- Role-corrected asymmetry validates ground truth: therapist-leading during meditation/PE, patient-leading during conversation.

Key scripts:
- `scripts/_run_scaffold_v82.py` — 18D scaffold (production), includes `run_from_raw()` for semi-synthetic
- `scripts/_run_v82_hierarchical.py` — hierarchical rSLDS across sessions
- `scripts/_run_v82_rslds_analysis.py` — per-session rSLDS
- `scripts/_test_eeg_metrics_h2h.py` — metric comparison
- `scripts/_test_v82_semisynthetic_battery.py` — comprehensive semi-synthetic validation

### V8.2 Semi-Synthetic Test Battery (2026-03-30)

Raw-level coupling injection validates the full 18D scaffold pipeline end-to-end.
Pseudo-dyad base (P1 from session A, P2 from session B) guarantees null AUC≈0.50.

**EEG (validated)**: Narrowband signal mixing at 256 Hz. 4 literature-grounded scenarios:
- E1 mutual gaze (alpha frontal): dyn_alpha AUC=0.67, imcoh_alpha=0.59
- E2 cooperative (theta frontal+temporal): imcoh_theta AUC=0.63
- E4 therapist-leading (alpha+asymmetry): dyn_alpha AUC=0.73, conc_alpha=0.69
- Band isolation PASS: alpha injection → theta/beta null

**BL (validated)**: Expression-band (0.5-2 Hz) signal mixing across all 10 AFFECT_AUS:
- B1 smile: bl_expr AUC=0.64→0.82→0.89→0.92 (kappa 0.10→0.20→0.30→0.40)

**Pose**: Semi-synthetic limited (circular-shift surrogates absorb continuous mixing). Validated on real data: multi-lag conv vs med p=0.006.

**Key injection findings**:
1. Kuramoto phase rotation → zero-lag → ImCoh rejects. Use signal mixing with temporal lag.
2. BL template events too sparse for CWT coherence. Use continuous expression-band mixing on structured data.
3. Pose zero-lag cross-product can't detect lagged coupling. Multi-lag ±5s bank required.
4. Cached BL is z-scored, not raw [0,1]. Never clip to [0,1].

Scripts: `cadence/synthetic_v82.py` (injection), `scripts/_test_v82_semisynthetic_battery.py` (runner), `scripts/_extract_bl_templates.py` (templates)
Output: `results/v82_semisynthetic/` (heatmaps, dose-response curves, timeline plots)

### V8.2 Burst Analysis (2026-03-30)

Continuous coupling intensity + burst event detection layered on top of rSLDS state assignments. Reveals per-segment directionality invisible in discrete state labels. Full synthesis: `docs/rslds_burst_analysis_synthesis.md`.

Validated findings (n=12 sessions, cross-session error bars):
1. **Therapist/patient asymmetry reverses by condition**: Patient drives conversation coupling bursts (theta -0.213±0.036 in conv_1); therapist drives meditation (+0.367 in meditate_K) and PE (+0.679 in PE_1). PE_2 collapses to null (habituation). Baselines are null (correct control).
2. **Burst rates differentiate conditions**: Conversation ~2/min Face+Body bursts; meditation suppresses all except EEG phase (~1/min). base_EC has 2.5x EEG phase bursts vs base_EO.
3. **Per-participant power validates protocol**: Patient alpha higher in meditation than PE (+0.236±0.052, p<0.05). Patient theta elevated in conversation (+0.275), suppressed below eyes-closed baseline during meditation (-0.12, p<0.01). meditate_B has more patient alpha than meditate_K.
4. **Cross-modal lead/lag does NOT validate**: No consistent timing survived cross-session replication, pseudo-dyad null, or permutation CIs. Underpowered at n=12.
5. **Per-segment stratification mandatory**: Pooling conditions masks opposite asymmetry profiles. Always analyze base_EO, base_EC, conv_1, conv_2, meditate_B, meditate_K, PE_1, PE_2 separately.

Domain-axis visualization (replaces PCA): Phase Coupling = mean(ImCoh θ/α/β), Shared Power = mean(Conc θ/α/β), Body Coupling = mean(BL, Pose), Autonomic = mean(ECG, Resp).

Key scripts:
- `scripts/_plot_rslds_quiver.py` — State-center flow, modality profiles, trajectory quiver
- `scripts/_plot_rslds_bursts.py` — Continuous intensity, burst detection, peri-burst averages
- `scripts/_validate_bursts_by_condition.py` — Per-segment asymmetry, burst rates (production)
- `scripts/_validate_rslds_bursts.py` — Cross-session, pseudo-dyad null, permutation CIs

Output: `results/rslds/quiver_plots/` (39 figures + 2 JSON results files)

## Modalities (V2, legacy)

- EEG wavelet: 160ch @ 10Hz (2 components × 20 freqs × 4 ROIs)
- EEG interbrain: 120ch @ 5Hz (cross-brain PLV, delta excluded by default)
- ECG features: 7ch @ 2Hz (HRV)
- Blendshapes v2: 31ch @ 30Hz (15 PCA + 15 derivatives + activity)
- Pose features: 41ch @ 12Hz (40 joint groups + activity)

## Literature-Informed Priors (from ~190 papers, 2026-03-25)

For full details invoke `/literature`. Key priors that should influence all CADENCE development:

- **EDA/SC is the strongest therapy synchrony signal** (r=0.32-0.47) — not yet captured. #1 hardware addition.
- **Sympathetic (SNS) and parasympathetic (PNS) synchrony have OPPOSITE relational valence** (SNS: ES=+0.19, PNS: ES=-0.21). RMSSD is PNS — its synchrony is negatively associated with outcomes. This explains weak ECG pathways.
- **Behavioral synchrony Granger-causes neural synchrony** (Koul 2023) — face/pose are leading indicators of EEG coupling. NOTE: burst analysis cross-modal lead/lag did NOT replicate this in CADENCE data (n=12, underpowered; see `docs/rslds_burst_analysis_synthesis.md` section 3d).
- **Vocal pitch synchrony is meta-analytically harmful** (r=-0.20). Prefer linguistic/semantic synchrony if adding speech.
- **Coupling flexibility > aggregate synchrony** (Gordon 2025). **CONFIRMED in V10**: flexibility is strongest transition covariate (max|S|=1.659, 5x > baseline drift). Operationalized via graph spectral energy ratio.
- **No EEG hyperscanning during psychedelic sessions exists** — MAP-Neuro is first.
- **LZ complexity** is trivially real-time (<1ms/channel), beats alpha for psychedelic state. **Now implemented in V10** as concordance/asymmetry channels. Semi-synthetic validated (monotonic AUC at kappa 0→0.4). First use as interpersonal coupling moderator.
- **Respiratory rate extractable from Polar H10** (FMRR, <2 bpm error) — no new hardware needed.
- **ECA > ES** for event-based coupling (ES confounds synchrony with serial dependency).
- **Hawkes + basis functions + group sparsity** (Xu 2016) = CADENCE architecture for point processes.

## Key Design Decisions

### V7 BL Wavelet Pipeline
- CWT (FFT-based Morlet, w=5) on all 52 AUs at 30 log-spaced frequencies 0.3-8 Hz
- GPU-accelerated: torch.fft for CWT, GPU conv1d for coherence smoothing
- Low-pass filter at 8 Hz (Butterworth 4th order) removes tracker noise before CWT
- Three frequency bands: state (<0.5 Hz), expression (0.5-2 Hz), speech (2-7 Hz)
- Wavelet coherence per AU group with Gaussian temporal smoothing (0.5s)
- Surrogate z-scoring: 200 circular shifts of P2 CWT coefficients, Welford accumulation on GPU (0.6s total)
- Literature basis: Jeganathan 2022 (eLife), Fujiwara 2016/2018/2020, Hale 2019, Likens 2021
- Semisynthetic validation: AUC=0.78 at kappa=0.4 (d=0.97) with real smile waveform injection, 42 pseudo-dyad pairs

### V2/V6 Legacy
- EWLS forward-backward with streaming backward pass (1.05x memory vs 3x previously)
- Raised cosine basis with log-spacing (denser at short lags)
- Circular shift surrogates (vectorized gather, preserves all signal statistics)
- Per-modality PCA channels and pathway-specific temporal parameters

## Critical Testing Rule

**Semi-synthetic tests MUST use pseudo-dyad (cross-session) as the base signal.** Real dyad data already has coupling — injecting on top of it means κ=0 is not null. Always use P1 from session A + P2 from session B. This guarantees κ=0 produces AUC≈0.50.

## Validation Status

### V7 BL Wavelet validation (y_06)
- **Speech detection**: Therapist speech in meditation correctly detected; patient silence = 0% (null)
- **Expression events**: 80-97% reduction vs V6 saliency; zero events in baselines (perfect null)
- **Coherence hierarchy**: conv_2 (0.354) > conv_1 (0.227) > meditate (0.178) > baseline (0.134)
- **Shared smile coherence**: 1.60x higher during shared smiles vs baseline (conv_1)
- **Pseudo-pair control**: Real pair > pseudo pair (1.19x, consistent with literature)
- **Semisynthetic AUC**: 42 pseudo-dyad pairs, real smile waveform injection
  - kappa=0.2: AUC=0.69, d=0.62
  - kappa=0.3: AUC=0.75, d=0.84
  - kappa=0.4: AUC=0.78, d=0.97
- **Frequency bands empirically grounded**: state <0.5 Hz, expression 0.5-2 Hz, speech 2-7 Hz, noise >8 Hz

### Synthetic validation (6/6 tests pass at 600s)
- EEG-only, ECG-only, BL-only, Pose-only, EEG+BL, Null — all pass
- Corpus (5 seeds × 6 categories): TP EEG=100%, ECG=100%, BL=60%, Pose=60%. FP=0%
- Results: `results/cadence_synthetic/`

### Real session (y_06)
- P1→P2: 6/16 sig (EEG-EEG, EEG→BL, BL-BL, BL→EEG, Pose→BL, Pose-Pose)
- P2→P1: 6/16 sig (similar pattern, bidirectional)
- ECG never significant (expected — slow timescale vs 5s max lag)
- Cross-modal pathways are CADENCE's unique contribution (not measurable by MCCT)
- Results: `results/cadence/y_06/`

### CADENCE vs MCCT comparison
- Both agree on modality ranking (BL, Pose strongest; ECG weakest)
- CADENCE dR2 5-10x larger than MCCT CSGI (different baselines)
- Results: `results/cadence_comparison/`
