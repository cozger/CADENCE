# CLAUDE.md — CADENCE

## Project Overview

**CADENCE** (Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation) is a fully interpretable, regression-based framework for quantifying directed, time-varying, cross-modal interpersonal coupling from continuous multimodal recordings. Replaces MCCT's transformer with basis-expanded distributed lag regression where every quantity is a measured signal or regression weight.

**Active arms:** MVP (production grant baseline) and V11 (production scaffold). V10/V8.2/V7/V6/V2 archived 2026-05-05; their scripts live in `scripts/archived/` and the V2 estimator cluster (`cadence/coupling`, `cadence/regression`, `cadence/basis`, `cadence/visualization`, `cadence/synthetic*.py`, plus V2-era `cadence/significance/*.py` modules) lives in `_archive/cadence/`. V11's scaffold script (`_run_scaffold_v11.py`) still imports utilities from `run_session_v6.py`, `_run_scaffold_v82.py`, `_run_rslds_scaffold_v8.py`, `_run_scaffold_v10.py`, and `_extract_respiratory.py` — those five files stay in `scripts/` as internal libraries even though their `_v6`/`_v8`/`_v10` filenames are now misleading.

## Architecture

**MVP Pipeline** (current grant baseline — 2026-05-02): 6-channel observation set + 2-channel covariates, hierarchical rSLDS over 19 canonical sessions. **Production fit: K=3 hierarchical** at `results/mvp/hierarchical_evtcoinc_smooth/` — separates **NULL** (quiescence), **SHARED** (behavioral coupling: bl_evt=+1.04, frontal theta concordance, peaks during conversation 50-64%), and **COUP** (neural alpha coupling: conc_α=+0.34, peaks during eyes-closed/meditation 46-58%). Sensitivities: K=2 (`hierarchical_evtcoinc_smooth_k2/`, BIC −131k vs K=3) collapses behavioral and neural mechanisms into a single binary state but is BIC-preferred; `hierarchical_evtcoinc_smooth_unstdcov/` is the K=3 fit before the covariate-standardization fix (covariates were inert, BIC=838,646 vs canonical 832,482). **Production BL channel: `bl_event_coincidence`** — surrogate-z dyadic coincidence of facial-activity peaks (per-participant peaks of `au_activity` envelope above session p70, ±500ms tolerance, 200 circular shifts), then σ=15s Gaussian smoothing + per-session standardize (without standardization the smoothed std collapses to 0.15-0.45 and the rSLDS cannot allocate a state to it). Replaces obsolete `bl_expr` (Morlet wavelet coherence, fired on z-score noise during quiet periods) and `bl_activity_conc` (`(z_P1+z_P2)/2`, was shared activity LEVEL not coupling — confounded by who's talking). **MVP transition covariates** (`coupling_flexibility`, `lambda2`) are also per-session standardized in `_run_mvp_scaffold.py` because V11 stores them on raw scale (std≈0.07) — this is a V11-side bug that the MVP scaffold builder works around; un-fixing this reverts the MVP fit to the inert-covariate behavior shown in the `_unstdcov` sensitivity. Implementation: `cadence/significance/face_event_coincidence.py`. Per-condition validation on y_06: conv_1=+0.34, conv_2=+0.36, meditation=−0.16/−0.17, baselines≈0 — clean discrimination between interactive coupling and quiescent periods.

**Pose channel — Phase 1 candidates (2026-09-09, code-complete; data validation pending):** four alternatives to the Phase 0 DDTW-on-PCA pose channel, all emitting the same `{sid}__z` / `{sid}__stride_ts` 2 Hz surrogate-z contract as `pose_ddtw_per_session.npz`: `angles` and `angle_speed` (DDTW on 12 torso-frame segment angles / angular speeds from `cadence/significance/pose_angles.py`, each column in measured noise-SD units, plus warping-path lag/asymmetry — `pose_ddtw.py` `feature_mode`; `feature_mode='pca'` is byte-identical to Phase 0), and `evt_landing` / `evt_peak` (`cadence/significance/pose_event_coincidence.py`: speed-envelope minima = both bodies coming to rest, or movement onsets, through the same circular-shift coincidence z + σ=15s smoothing as `bl_event_coincidence`). `cadence/significance/block_bootstrap.py` provides block-bootstrap CIs / paired tests / ranking for per-session summaries. `scripts/_validate_pose_channels.py` runs every mode through the Phase 0 gates (Tests 1–4) and writes `results/mvp/phase1_pose/phase1_report.md`; `_run_mvp_scaffold.py --pose-channel` gains `angles | angle_speed | evt_landing | evt_peak`. **The production pose channel is unchanged until `phase1_report.md` exists** — `--pose-channel auto` keeps reading `phase0_report.md` (or defaults to `pose_baseline`), and the K=3 fit above is untouched. Design spec: `docs/superpowers/specs/2026-09-09-pose-coupling-phase1-design.md`; plan: `docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md`. Lag sign everywhere: positive = P2 later than P1.

**V11 Pipeline** (scaffold — feeds MVP): Burst coincidence + transfer entropy scaffold. 26D observation + 7D transition covariates. TE concordance (2ch obs: bidirectional flow), burst coincidence (3ch obs), and TE asymmetry (2ch cov: directionality as transition modulator). MVP slices conc_θ/α, pose, resp, ecg_hf, plus the two transition covariates from V11 scaffold output at `results/v11/`. **Note:** dyn_theta/alpha/beta collapsed to `dyn_mean` (r=0.99 collinearity, all are EWMAD of concordance at the same slow timescale). 28D→26D. Full design rationale, BIC decomposition, capacity grid search, and per-session findings: `docs/v11_pipeline_report.md` and `docs/v11_identifiability_diagnostics.md`.

**V12 Pipeline** (planned — pending stereo gaze hardware): Gaze coupling from stereo-calibrated gaze rays. 29-31D observation + 9D covariates. Extends V11 with gaze concordance (obs: mutual attention intensity), gaze asymmetry (cov: who attends), gaze approach rate (cov: convergence predicting state transitions). First modality where raw measurement IS the coupling — no surrogates needed. Design doc: `docs/v12_gaze_design.md`.

**rSLDS engine** (`cadence/significance/rslds_model.py`): numpy + Numba (Phase 0 patches + Phase 0.5 Tier A patches as of 2026-05-02). Phase 0 alone delivers 5.8× hierarchical-fit speedup vs unpatched (138 min → 23.6 min on 19-session production cohort); Tier A adds ~12% on top via vectorized K loop in `slds_e_step` and a `nogil=True` Numba kernel for `_log_transitions_recurrent`. JAX migration deferred to ~2027-2028 when cohort approaches ~100+ sessions; trigger criteria and decision gates documented in `docs/dynamax_migration_plan.md` §0.

## Environment

Uses the MCCT conda environment (Python 3.11, PyTorch, scipy, numpy, matplotlib, pyyaml).

```bash
conda activate MCCT
```

## Key Commands

```bash
# MVP pipeline (production grant baseline)
python scripts/_run_mvp_scaffold.py                                 # Slice 6+2D MVP scaffold from V11 output
python scripts/_run_mvp_scaffold.py --all                           # All canonical sessions
python scripts/_run_mvp_hierarchical.py                             # Hierarchical K=3 rSLDS fit
python scripts/_run_mvp_dlatent_study.py                            # D_latent identifiability study
python scripts/_run_mvp_hier_multi_init.py                          # Multi-init basin study
python scripts/_run_mvp_ppc.py                                      # Posterior predictive check
python scripts/_run_mvp_diagnostics_queue.py                        # Diagnostic queue runner
python scripts/_make_mvp_figures.py                                 # Production figures
python scripts/_validate_pose_channels.py --all                     # Phase 1 pose candidates -> phase1_report.md

# V11 scaffold (feeds MVP)
python scripts/_run_scaffold_v11.py                                 # Single session (y_06), 26D scaffold
python scripts/_run_scaffold_v11.py --session y_17                  # Specific session
python scripts/_run_scaffold_v11.py --all                           # All sessions (n_jobs=4)
python scripts/_run_v11_hierarchical.py                             # Hierarchical rSLDS (26D + 7D covariates)

# Synchrony repertoire pipeline (per-cohort clustering of synchrony episodes)
python scripts/_run_synchrony_per_session.py --session y_06
python scripts/_run_synchrony_cohort.py
```

## Project Structure

```
CADENCE/
  cadence/
    __init__.py, config.py, constants.py
    ingest/        # Layer 1 (Data Pipeline v1): raw XDF -> data/digest/v1/
                   # roles, schema, quality, xdf_reader, digest, cli
    preprocess/    # Layer 2: per-modality cleaning -> data/preproc/<mod>/v1/
                   # eeg/ (MATLAB bridge), face/, ecg/, pose/ (+ pose_subset)
    data/          # Eager-loading legacy dict-view shim (V11 still imports this).
                   # New code should use cadence.ingest / cadence.preprocess directly.
    io/            # paths.py, cache.py, resources.py
    significance/  # rslds_model.py, rslds_validation.py,
                   # fast_cycles.py (GPU EEG, extract_burst_grids),
                   # burst_coincidence.py, directed_burst_coupling.py (V11 burst layers),
                   # bl_wavelet.py, lz_complexity.py, spectral_graph.py (V11 obs/cov),
                   # face_event_coincidence.py (MVP BL channel),
                   # pose_ddtw.py (MVP pose channel; Phase 1 feature modes),
                   # pose_angles.py, pose_event_coincidence.py,
                   # block_bootstrap.py (Phase 1 pose candidates — see
                   # docs/superpowers/specs/2026-09-09-pose-coupling-phase1-design.md),
                   # bl_coupling.py + distributional_stats.py (run_session_v6 utility deps),
                   # smap.py
    synchrony/     # Per-cohort synchrony repertoire pipeline (Stages 0-9)
  configs/
    default.yaml
    session_quality.yaml  # canonical session registry (drives --all)
  data/
    digest/v1/<sid>.npz + .json     # Layer 1 outputs
    preproc/{eeg,face,ecg,pose}/v1/ # Layer 2 outputs
    matlab/<sid>_p{1,2}_clean.mat   # MATLAB-cleaned EEG with .mat.json sidecars
  scripts/
    _run_mvp_*.py, _plot_mvp_*.py, _make_mvp_figures.py    # MVP entry points
    _validate_pose_ddtw.py, _validate_pose_channels.py     # Pose channel gates (Phase 0 / Phase 1)
    _run_scaffold_v11.py, _run_v11_hierarchical.py         # V11 entry points
    _run_synchrony_*.py                                    # Synchrony pipeline
    run_session_v6.py, _run_scaffold_v82.py,               # Legacy filenames retained
    _run_rslds_scaffold_v8.py, _run_scaffold_v10.py,       # — V11 scaffold imports
    _extract_respiratory.py                                # utilities from these
    archived/                                              # All pre-V11 scripts (~180)
  _archive/
    cadence/                                               # V2 estimator cluster
      coupling/, regression/, basis/, visualization/
      synthetic.py, synthetic_v82.py, synthetic_v10.py
      significance/                                        # V2-era helpers
  results/
  docs/
    v11_pipeline_report.md, v11_identifiability_diagnostics.md
    v12_gaze_design.md, dynamax_migration_plan.md
    data_pipeline_v1.md, burst_detection_literature.md
    archive/                                               # V8/V10 legacy docs
```

## Config

All parameters in `configs/default.yaml`. Key settings:
- `significance.surrogate.n_surrogates`: 100 circular shifts (session-level)
- `significance.timepoint.n_surrogates`: 20 (per-timepoint)
- `significance.timepoint.surrogate_eval_rate`: 1.0 Hz
- `significance.fdr_correction`: Benjamini-Hochberg
- `interbrain.min_freq_hz`: 4.0 (exclude delta — Emotiv EPOC artifact-prone)
- `interbrain.surrogate_method`: fourier_phase

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
   |-- Analytics (V11 scaffold, MVP slice, hierarchical rSLDS, synchrony pipeline)
```

`xdf_md5` keys flow downstream: a digest re-run invalidates all four preproc
artifacts via the staleness predicate. Independent versioning means bumping
`preproc/eeg/v1` doesn't force re-running pose/face/ecg.

**Single classification scheme:** `configs/session_quality.yaml` is hand-maintained
(drafted by `scripts/_audit_xdf_inventory.py`). Carries `canonical: bool`,
per-participant `excluded_modalities`, and per-session `marker_overrides`
(rename/drop with t_min/t_max windows for genuine recording errors).

**Three pose formats supported transparently:** mediapipe33 (132 ch),
mediapipe33_meta (133 ch), wholebody_133 (400 ch — RTMW). Format dispatch in
`cadence.preprocess.pose.pose_subset.normalize_pose_to_mp33` returns
`(N, 33, 4)` MediaPipe-shaped output regardless of source.

**Pose features are visibility-aware** (deliberate behavioral fix): groups
with no visible landmarks return NaN (filtered by NaN-safe statistics)
instead of biasing centroids toward origin.

**Backward compatibility:** `cadence.data` is an eager-loading dict-view shim.
Pre-existing scripts (V11 scaffold, run_session_v6, etc.) using
`load_session_from_cache(sid)` keep working unchanged; the shim materializes
~30 legacy keys per participant by reading from digest + 4 preproc artifacts
and write-protects via `setflags(write=False)` for joblib safety.

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
algorithm, marker pipeline, migration record, and the
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

## Critical Testing Rule

**Semi-synthetic tests MUST use pseudo-dyad (cross-session) as the base signal.** Real dyad data already has coupling — injecting on top of it means κ=0 is not null. Always use P1 from session A + P2 from session B. This guarantees κ=0 produces AUC≈0.50.

## Literature-Informed Priors

For full details invoke `/literature` (skill backed by ~190+ paper review). Headline priors that influence CADENCE design:

- **EDA/SC is the strongest therapy synchrony signal** (r=0.32–0.47) — not yet captured. #1 hardware addition.
- **SNS and PNS synchrony have OPPOSITE relational valence** (SNS: ES=+0.19, PNS: ES=−0.21). RMSSD is PNS — its synchrony is negatively associated with outcomes.
- **Coupling flexibility > aggregate synchrony** (Gordon 2025). Confirmed in V10/V11 as strongest transition covariate.
- **No EEG hyperscanning during psychedelic sessions exists** — MAP-Neuro is first.
- **LZ complexity** beats alpha for psychedelic state; first use as interpersonal coupling moderator (V11 LZ conc/asym channels).
- **Vocal pitch synchrony is meta-analytically harmful** (r=−0.20). Prefer linguistic/semantic synchrony if adding speech.
- **NO Emotiv hyperscanning study uses wPLI/ImCoh** — CADENCE wPLI null is expected hardware ceiling. Use envelope/concordance metrics instead. 32+ ch saline minimum for true phase coupling.
