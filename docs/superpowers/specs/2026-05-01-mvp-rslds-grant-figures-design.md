# MVP rSLDS for Grant Figures — Design Spec

**Date:** 2026-05-01 (data-pipeline integration update 2026-05-01)
**Status:** Approved for implementation
**Goal:** Produce two grant-application figures (cohort dyad-level variability + protocol comparison) from a minimum-viable rSLDS trained only on literature-validated and diagnostically-confirmed channels. Justify funding for larger exploration of the framework.

**Data sources (2026-05-01 update).** The MVP consumes Data Pipeline v1
artifacts (see `docs/data_pipeline_v1.md`):

- **Layer 1 digests** at `data/digest/v1/<sid>.{npz,json}` — raw streams,
  resolved roles, normalized markers, pose-format dispatch metadata.
- **Layer 2 preproc** at `data/preproc/{eeg,face,ecg,pose}/v1/<sid>.{npz,json}` —
  per-modality cleaned arrays + features. Visibility-aware pose features +
  3-format pose dispatch are first-class.
- **Cohort:** the 22 canonical sessions per `configs/session_quality.yaml`
  (the audit-driven registry; `canonical: true` flag). The historical
  "14 sessions matching the V11 K=4 no-null cohort" set is a subset of this;
  protocol classification per the registry, with explicit per-session
  table emitted to `results/mvp/cohort_protocol_assignment.csv` as the first
  step of scaffold-slice.
- **V11 scaffold derived channels:** `results/v11/<sid>/scaffold_v11_ztimecourses.npz`
  remains the source of the 7 MVP observation channels and 2 covariates by
  column slice. V11's own scaffold extraction now reads from the new pipeline
  through the `cadence.data` legacy dict-view shim, so the chain back to
  raw XDF goes: raw -> digest -> preproc -> V11 scaffold -> MVP scaffold.
- **Surrogate-calibrated raw concordance z** (Figure 1 caption agreement-check):
  computed on demand from `data/preproc/eeg/v1/<sid>.npz` + the standard
  surrogate pipeline. The legacy `results/v10/coupling_excess/` cache is
  consulted only as a faster path; if absent or stale (digest_xdf_md5
  mismatch), recompute from preproc.
- **Pose for DDTW (Phase 0):** `data/preproc/pose/v1/<sid>.npz` provides
  `{p}_pose33` as `(N, 33, 4)` MediaPipe-shaped data — uniform across the
  three source formats (mediapipe33 / mediapipe33_meta / wholebody_133).
  Per-frame validity mask is `{p}_pose_features_valid`.

---

## Context

The V11 production rSLDS uses 26 observation channels + 7 transition covariates. The Tier 1/2 diagnostic suite (`docs/v11_identifiability_diagnostics.md`) established that ~5 channels carry most of the state-discriminating signal and that several novel channels (LZ concordance, TE concordance, graph modularity) sit on the literature-validation side without internal-diagnostic backing or vice versa. The MVP fits a smaller, defensible model where every channel can be cited and every state has a non-trivial emission-mean loading.

The MVP is **not a replacement for V11**. It is a defensible parallel baseline whose role is to demonstrate that a K=4 (or K=3, empirically determined) state structure and condition-level state usage findings emerge from aggressive channel pruning to literature-validated channels. Every novel channel that earns its way back into a future production model becomes a hypothesis test against this MVP baseline.

---

## Scope rule (grant-application constraint)

**The grant application makes no reference to V11, V10, V8.2, or any prior CADENCE version.** Every claim, validation, or evidence statement in the grant text is sourced from MVP-internal results. This spec retains references to prior versions in development-context sections (Architecture, Implementation outline, Critical files) so the implementer knows where data and code live. Sections that feed into grant-facing methods text (Channel justifications, State-structure claims, K-selection, Phase 0 outcomes, Verification results) report MVP-internal evidence only.

**Practical consequences for this design:**
- K=3 vs K=4 is decided by an MVP-internal BIC + held-out LL comparison, not by reference to V11's 51k margin at 26D.
- Identifiability, protocol-pooling validity, and state-structure stability are tested by MVP-internal sensitivity fits (shared-`d_emit`, protocol-stratified), not by citing V11's diagnostic suite.
- The "multi-lag cross-correlation pose-coupling baseline" against which DDTW is compared in Phase 0 Test 4 is described methodologically; the V8.2 lineage is not named in grant text.
- `viterbi_min_dwell=20` (10s, 20 frames at 2 Hz) is the MVP production parameter, with unconstrained Viterbi run as a sensitivity check. The MVP does not cite V11 Module 3 results to defend a different choice.
- Channel-set justifications cite literature; emission-mean profiles in figures are reported from the MVP fit itself.

---

## Architecture

```
results/mvp/
  <session>/
    mvp_scaffold.npz       ← 7D obs + 2D cov sliced from V11 scaffold
    mvp_rslds_results.npz  ← per-session gamma, path, x_smooth, d_emit, ...
  hierarchical/
    mvp_hierarchical_params.npz  ← shared transition / dynamics / mean_d_emit
    mvp_hierarchical_results.json
  figures/
    figure1_dyad_variability.png
    figure2_protocol_comparison.png
  diagnostics/
    module3_dwell_report.md   ← post-fit Viterbi dwell verification
scripts/
  _run_mvp_scaffold.py     ← NEW: slice 7D obs + 2D cov from V11 scaffold
  _run_mvp_hierarchical.py ← NEW: fit hierarchical rSLDS K=4 on MVP scaffold
  _make_mvp_figures.py     ← NEW: generate Figure 1 + Figure 2
```

The MVP scaffold is a **slice** of the existing V11 scaffold, not a re-extraction. The 7 observation channels and 2 covariates are already present in `results/v11/<session>/scaffold_v11_ztimecourses.npz`; the MVP scaffold runner reads that file, selects the right columns, and writes the reduced scaffold. This preserves all V11 prewhitening, validity masking, and surrogate calibration without re-auditing.

**Note (2026-05-01 data-pipeline integration).** V11 scaffold construction
now reads from Data Pipeline v1 (`data/digest/v1/` + `data/preproc/<mod>/v1/`)
via the `cadence.data` dict-view shim. This is transparent to the MVP — V11
scaffold output retains the same NPZ schema (`z_<modality>` /
`u_<covariate>` keys) and the same channel names. What changes is provenance:

1. The MVP scaffold runner additionally reads each session's
   `data/digest/v1/<sid>.json` to record `digest_xdf_md5` in the MVP
   sidecar, so freshness chains all the way back to raw XDF md5.
2. Pose-channel sourcing differs by Phase 0 outcome (see §Phase 0): the
   multi-lag baseline column comes from V11 scaffold; the DDTW alternative
   is computed from `data/preproc/pose/v1/<sid>.npz` directly.
3. The 6 previously-mislabelled sessions (Y_55, y_59, y_64, y_66, y26, y_65)
   now resolve as therapist/patient correctly, fixing role-asymmetry
   computations for those dyads in V11 -> MVP.

If a digest's `xdf_md5` no longer matches the V11 scaffold's recorded
`digest_xdf_md5` (i.e., the source XDF was re-recorded), the V11 scaffold
must be regenerated before MVP slice. The MVP runner must check this and
fail-loud rather than producing a stale slice.

---

## Channel set (7D observations + 2D covariates)

### Observations

| Channel | Literature anchor | Diagnostic role (from K=4 no-null fit) |
|---|---|---|
| `conc_theta` | Theta hyperscanning during cooperation/joint attention; project research priority | COUP loader +0.52, SHARED loader +0.49 |
| `conc_alpha` | Alpha hyperscanning gold standard (Babiloni / Astolfi / Dumas tradition) | COUP loader +0.43 |
| `bl_expr` | Facial behavioral synchrony meta-analyses; Koul 2023 (Granger-leads neural) | SHARED loader +0.18; V7-validated AUC=0.78 at κ=0.4 |
| `bl_activity_conc` | Facial activity coupling; same lineage | SHARED loader +0.44 (primary COUP-vs-SHARED discriminator) |
| `pose` | Body-movement synchrony meta-analyses; Shokoohi-Yekta et al. 2017 (multi-dimensional DTW formalization, dependent vs independent distinction) if Phase 0 swap; Fujiwara, Otmar, Dunbar, & Hansia (2022) (DTW for body-movement dyadic synchrony, segmental design); behavioral-mimicry literature (Chartrand & Bargh 1999) | Defines autonomic-quiescence in pilot diagnostic data; **operationalization conditional on Phase 0 validation outcome** — multi-lag cross-correlation pose-coupling baseline (the inherited V11-cached column) or DDTW (replacement, if Phase 0 Tests 1–3 pass). See §Phase 0. Grant text describes this as "multi-lag cross-correlation pose baseline vs DDTW alternative" without naming V8.2. |
| `resp` | Respiratory phase coherence (Codrons 2014, McFarland 2001) | Autonomic-quiescence (peaks meditate_K) |
| `ecg_hf` | RSA / vagal-tone synchrony (Palumbo 2017 meta) | Autonomic-quiescence |

### Transition covariates

| Covariate | Literature anchor | Diagnostic role |
|---|---|---|
| `coupling_flexibility` | Gordon 2025 (coupling flexibility > aggregate synchrony) | Strong covariate effect in V11 pilot diagnostic (max\|S\|≈0.82); MVP-internal `max\|S\|` reported in `verification_report.md` as the grant-text effect size |
| `lambda_2` | Algebraic connectivity of modality graph (graph theory) | Strong covariate effect in V11 pilot diagnostic (max\|S\|≈0.96); MVP-internal `max\|S\|` reported in `verification_report.md` as the grant-text effect size |

### Channels explicitly not included (and why)

- **`conc_beta`** — not called out as research-relevant; theta + alpha span EEG-band signal with full diagnostic backing.
- **Imaginary coherence (θ/α/β)** — collinear with concordance (r > 0.83); block PCA compresses three channels to one PC. Skipped to avoid axis duplication.
- **Asymmetry (θ/α/β)** — signed/directional; per `feedback_rslds_obs_vs_cov.md`, signed measures belong in covariates, not observations. Modest covariate effect (max\|S\| < 0.06) so dropped from MVP covariate set.
- **LZ concordance (θ/α)** — load **0.00** on Complexity block PCs after surrogate z-scoring + burst-rate gating. Defer to future scope.
- **LZ asymmetry (θ/α)** — signed; modest effect; defer to future scope.
- **TE concordance (θ/α)** — flagged `LOW_INFO` (median N_eff < 50/session). Defer to future scope.
- **TE asymmetry (θ/α)** — max\|S\| < 0.06 in V11 covariate slot. Defer.
- **Graph modularity** — flagged `LOW_INFO`; solo Block PC; no state discrimination.
- **Burst coincidence (θ/α/β)** — modest signal (load +0.15 on COUP); partly absorbed by concordance channels.
- **`ecg_lf`** — orthogonal to `ecg_hf` in block PCA, but PNS-only synchrony (HF) is the more citable construct for the MVP.

---

## Phase 0 — Pose channel validation via DDTW

**Purpose.** Determine whether dependent dynamic time warping is a more informative pose-coupling operationalization than the multi-lag cross-correlation baseline (the inherited V11-cached pose column derived from a ±5s lag-bank cross-product on velocity features). Outcome is a single channel decision: the MVP scaffold uses `pose_baseline` (default, the multi-lag baseline) or `pose_ddtw` (if Phase 0 Tests 1–3 pass). The MVP runs at 7D regardless.

**Why DDTW is the right method to evaluate for pose specifically.** Body movement is non-periodic and co-articulated across joints. CWT-coherence (the V7 face approach) is poorly suited because pose lacks characteristic frequencies. Cross-correlation at fixed lag (V8.2 multi-lag) captures instantaneous co-movement at predetermined offsets. DDTW captures the gap between these — variable-lag mimicry of whole-body patterns — and the *dependent* warping path respects joint co-articulation by aligning all dimensions simultaneously rather than independently.

### DDTW configuration

- **Input.** Per-participant 33-keypoint × 3-axis = 99D position stream from
  `data/preproc/pose/v1/<sid>.npz`'s `{p}_pose33` array (shape `(N, 33, 4)` —
  drop the visibility column to get `(N, 33, 3)` -> flatten to 99D). Native
  pose rate is preserved; the digest pose-format dispatcher already
  normalized mediapipe33 (12 Hz, 132 ch raw) / mediapipe33_meta (60 Hz, 133
  ch raw) / wholebody_133 (60 Hz, 400 ch raw) into the same `(N, 33, 4)`
  shape. **Important**: native rates differ across sessions (12 Hz on older
  Sway-tag sessions vs 60 Hz on Y_55-era recordings); the DDTW pre-window
  step must resample per-session to a common 12 Hz target so the 4s window /
  500ms stride math (T=48 frames) holds uniformly. Resample after the shared
  PCA projection (smaller D = faster).
- **Validity masking.** `{p}_pose_features_valid` from the same NPZ is the
  per-frame mask (≥ 10 keypoints with visibility > 0.5). Surrogate windows
  containing < 50% valid frames are dropped from per-stride DDTW; this
  preserves the surrogate distribution's null structure when face occlusion
  or off-camera frames are present.
- **Pre-DDTW dimensionality reduction.** Shared cross-session PCA: fit on the
  pooled 99D position stream (visibility-zeroed coordinates -> NaN-fill via
  per-keypoint median imputation -> PCA) from all canonical sessions in the
  cohort, retain the top 10 components capturing ≥80% pooled variance.
  Project each session's stream into the shared subspace before DDTW.
  **Per-session PCA is explicitly avoided** — it produces session-specific
  axes, which would make DDTW cost magnitudes incomparable across sessions
  even after surrogate z-scoring (the within-session magnitude is corrected,
  but the geometric meaning of "warp distance" differs per session).
  Document the shared loadings; verify in Phase 0 output that no session
  needs > 15 components for 80% variance under the shared fit.

  **Note on wholebody_133 sessions.** y_53 (and any future RTMW-format
  sessions) has 11 of 33 MediaPipe slots structurally zeroed (mouth /
  eye_inner / eye_outer / hand-tip slots — see
  `cadence/preprocess/pose/pose_subset.RTMW_TO_MP33`). The shared PCA must
  be fit on the union of all sessions; the structurally-zeroed slots will
  still load on a few PCs but with reduced magnitude on RTMW sessions.
  Verify in Phase 0 output that the per-session reconstruction error from
  the shared 10D fit is comparable across formats (no session > 1.5x the
  cohort median); if RTMW sessions show pathologically higher reconstruction
  error, fit the shared PCA only on the populated 22-of-33 slots common to
  all formats.
- **DTW windowing constraint.** No Sakoe-Chiba band (`window.type = "none"`), matching the R `dtw` package default and the convention in Fujiwara, Otmar, Dunbar, & Hansia (2022). The 4s segment length itself bounds the maximum allowable warp; an additional band would be redundant at this scale.
- **Sliding window.** 4s window, 500ms stride. Per-window: T=48 frames × D=10. Stride matches rSLDS rate (2 Hz) exactly — no resampling needed downstream.
- **Per-stride output.** DDTW coupling score = −1 × normalized cumulative warp cost (higher = more coupled). One scalar per stride position.
- **Feature choice.** Position (PCA of keypoints), not velocity. Postural mimicry is itself coupling; derivative-domain DTW is harder to interpret because warping paths on velocity signals are unstable.
- **Surrogate calibration.** 200 condition-block circular shifts of P2 pose: shifts are integer multiples of full-condition durations (`base_EO`, `base_EC`, `conv_1`, etc.) rather than within-session circular wraps, eliminating cross-condition-boundary blending in surrogate windows. Per-stride z-score against the surrogate distribution.
- **Prewhitening.** AR(1) + standardize, validity-mask aware (V11 standard pipeline). Applied to the surrogate-z timecourse only.

### Why a 4s window specifically

| Window | T frames @ 12 Hz | DDTW path stability | What it captures |
|---|---|---|---|
| 1s | 12 | Marginal — warping path has few choices | Fast micro-mimicry only |
| 2s | 24 | OK | Hand gestures, head nods |
| **4s** | **48** | **Good** | Gestures + posture transitions + conversational rhythms |
| 8s | 96 | Excellent | All of above + slow postural matching; smooths coupling onset/offset |
| 30s | 360 | Heavy smoothing | Session-level aggregate only |

4s sits at the sweet spot for behavioral-synchrony timescales in conversational and meditation contexts. Three independent justifications support this choice:

1. **Behavioral-mimicry timescale**: behavioral synchrony literature documents nonconscious interpersonal mimicry (Chartrand & Bargh 1999) operating on sub-second-to-few-second lead/lag offsets; a 4s window is long enough to capture mimicry events together with their offsets without being so long that it smooths over coupling onset/offset.
2. **Consistency with the multi-lag cross-correlation baseline scope**: the inherited baseline uses a ±5s lag bank as its temporal flexibility scope; a 4s DDTW window stays within the same temporal scope so the two methods are evaluated at comparable timescales.
3. **DDTW path stability**: at 48 frames, the cost matrix has 2,304 cells — sufficient warping degrees of freedom for stable alignment without the dominant-diagonal degeneracy that affects shorter windows.

**Note on prior DTW-on-pose work.** The most directly comparable study, Fujiwara, Otmar, Dunbar, & Hansia (2022), uses 20-second *segments* (one DDTW score per segment) rather than sliding windows. Their parameters are not directly transferable to our 2 Hz time-resolved design, where each rSLDS timepoint requires a coupling estimate. The 4s/500ms-stride configuration deviates from their segmental approach for time-resolution reasons and is justified on the three independent grounds above rather than by direct citation.

### Why a 500ms stride matches the rSLDS

The rSLDS operates at 2 Hz (= 500ms timestep). Setting the DDTW stride to 500ms emits one DDTW score per rSLDS timepoint with zero resampling — eliminating any interpolation artifact between Phase 0 output and scaffold ingestion. Window overlap is 87.5% (= 1 − 0.5/4), heavy but standard for sliding-window coupling at this rate. Adjacent-window correlation will be high; AR(1) prewhitening downstream handles the resulting autocorrelation per V11 standard.

### Compute strategy

**Default: CPU prototype.** Use `dtaidistance` (OpenMP parallel) for both validation and one-off production extraction. Per-DDTW cost at T=48, D=10 ≈ ~1ms; per-stride with 200 surrogates ≈ ~200ms; per session (3,000 stride positions × 200ms) ≈ 10 min sequential. With `joblib(threading)` over 8 cores: ~1.5 min/session. Full 14-session pipeline: **~20–25 min**.

**Forward-looking GPU port (deferred).** If subsequent work needs more granular stride or longer windows, follow the chunking pattern of `gpu_sliding_te_surrogates`: batch 200 surrogates per stride position on the GPU. Per-batch memory at T=48, D=10 ≈ ~10 MB; even at 1000-position batches the footprint is well under the 16 GB cap. Document the chunk size in `pose_ddtw.py`'s docstring at port time.

### Validation criteria

| Test | Method | Pass threshold |
|---|---|---|
| 1. Semi-synthetic dose response | Pseudo-dyad base (cross-session P1+P2), ~30 pseudo-dyad pairs. Inject DDTW-detectable coupling on a 60s injection segment: take P1 pose, apply piecewise-linear time warps, mix with P2: `P2_inj = κ · warped_P1 + (1−κ) · P2`. Sweep κ ∈ {0.0, 0.1, 0.2, 0.3, 0.4}. Compute AUC per κ. **Kendall τ permutation test** for monotonic AUC trend across κ levels (1000 permutations). | Kendall τ permutation p < 0.05 AND AUC ≥ 0.65 at κ=0.4 |
| 2. Real-data condition contrast | Mean DDTW excess z per condition per session; paired comparison `(conv_1+conv_2)` vs `(meditate_K+meditate_B)` across the meditation-protocol sessions. Both **paired t-test** and **Wilcoxon signed-rank** (n is small — at meditation-only n≈6 the paired t has ~50% power for Δz=0.5; report both, gate on the more conservative). | Δz ≥ 0.5 AND (paired t p < 0.05 OR Wilcoxon p < 0.05) |
| 3. Pseudo-dyad null | Mean real-pair DDTW excess z vs cross-session pseudo-pair excess z, across same session set | Δ ≥ 0.5 z (real > pseudo) |
| 4. Redundancy descriptive (not gated) | Pearson r between DDTW timecourse and the multi-lag cross-correlation pose-coupling baseline, mean across 14 sessions | **No pass/fail threshold.** Report `r` and 95% bootstrap CI in `phase0_report.md`; described in methods as "DDTW timecourse correlated r=X.XX with the multi-lag baseline, indicating [partial overlap / largely independent / largely redundant signal]." |

### Decision rule

The redundancy comparison (Test 4) is reported descriptively, not as a gate. The pose channel decision is determined by Tests 1–3 only.

| Outcome | MVP pose channel |
|---|---|
| Tests 1, 2, AND 3 all pass | **Swap to `pose_ddtw`** (7D MVP scaffold). Document validation evidence + Test 4 redundancy r in methods. |
| Any of Tests 1–3 fails | Keep multi-lag cross-correlation baseline as the pose channel. Document failure outcome and defer DDTW to follow-up. |
| Phase 0 cannot run at all (e.g., `dtaidistance` install failure, pose data corrupt for too many sessions) | Default to multi-lag cross-correlation baseline. Phase 0 is non-blocking for downstream MVP — the worst case is "we don't get the DDTW upgrade in this MVP." |

The decision is binary: one pose channel either way. The MVP scaffold runs at 7D regardless. Multiplicity correction across Tests 1–3 is unnecessary — the conjunction is intentionally strict (we want all three to pass for a swap), and the joint Type I error rate at α=0.05 per test is bounded above by 0.05 (most stringent test only).

### Output

`results/mvp/phase0/`:
- `pose_ddtw_per_session.npz` — per-session DDTW timecourses (real + surrogate-calibrated z) at 2 Hz
- `validation_semisynthetic.csv` — AUC per κ per pseudo-dyad pair
- `validation_condition_contrast.csv` — per-session per-condition mean DDTW excess z
- `validation_pseudo_null.csv` — real vs pseudo-pair excess z
- `redundancy_vs_baseline.csv` — per-session correlation between DDTW and multi-lag cross-correlation baseline timecourses (descriptive, not gated)
- `phase0_report.md` — pass/fail per criterion + decision rule outcome + selected pose channel

---

## Model specification

| Component | Value | Justification |
|---|---|---|
| K | 4 | NULL, COUP, SHARED, autonomic-quiescence; 26D K=4 BIC win of 51k over K=3 |
| `null_state` | False | Module 6 verdict — constrained NULL/OTHER bookkeeping; K=4 no-null wins BIC by 13k on 26D |
| `D_latent` | 3 | V11 BIC grid optimum, robust across D_obs |
| `n_factors` | 2 | V11 BIC grid optimum |
| `sticky_strength` | 3.0 | Carry-over default; principled prior |
| `viterbi_min_dwell` | 20 (10s at 2 Hz) | Production parameter; an unconstrained Viterbi (`viterbi_min_dwell=0`) is run as a sensitivity check and reported in the Phase 0 / Verification output |
| Whitening | Per-channel AR(1) + standardize on V11 prewhitening pipeline | Validity-mask aware; preserved by scaffold slice |
| Sessions | up to 22 (canonical-flagged) | Defined by `configs/session_quality.yaml` `canonical: true` registry. After 2026-05-01 migration the canonical pool grew from 14 (V11 K=4 no-null cohort) to 22 by including 6 previously-mislabelled RA-tag sessions (now correctly resolving) and the post-V11 sessions. The MVP defaults to the full 22 if `data/preproc/eeg/v1/<sid>.npz` exists; sessions waiting on a fresh MATLAB EEG run (y04, y11, y24, y_53 as of 2026-05-01) are excluded from the production fit and reported in the cohort table. The first scaffold-slice step emits `results/mvp/cohort_protocol_assignment.csv` with one row per session: `session_id, protocol, n_meditation_phases_present, n_pe_phases_present, has_eeg_preproc, included_in_production_fit`. The grant text reports exact n per protocol from this CSV. |

### Why `viterbi_min_dwell = 20` and not unconstrained

A 10s minimum-dwell constraint is the V11-default carried forward as the MVP production parameter. The MVP grant text describes it as "10s minimum-dwell Viterbi decoding to suppress sub-resolution state oscillations." Production figures use the constrained path; a single sensitivity panel (or supplementary figure) reports the unconstrained Viterbi to confirm robustness — if state usage in conditions is qualitatively similar with and without the constraint, the constraint is doing only its intended job.

Note on `sticky_strength=3` and recurrent covariates: at the per-session data sample sizes (T ≈ 3000 timepoints, hundreds of on-diagonal transitions per state), a Dirichlet pseudocount of 3 is empirically dominated by the data and contributes minimal regularization. The Viterbi minimum, not sticky, is what prevents flicker. The recurrent covariate term `S_trans · u_t` modulates transition logits and can swing state probabilities materially when flexibility/lambda_2 covariates change rapidly — this is signal modulation, not noise.

### Fallback ladder (if MVP-internal dwell verification fails)

If MVP-internal Module 3 metrics show pathological flicker — `empirical/predicted dwell ratio < 0.5` per state OR `> 50% of dwells < 10s` per state — at the production setting, escalate in this order:

1. Raise `viterbi_min_dwell`: 20 → 30 (15s) → 40 (20s). The principled first move; tightens the dwell floor without changing model architecture.
2. Reduce K. If K=4 is straining the data and producing transient states, K=3 is the parsimonious response. (The K=3 sensitivity fit is run unconditionally in §Verification protocol — its output is available without an additional fit.)
3. Channel-set review. If neither (1) nor (2) produces stable dwells, the channels themselves are the issue: insufficient state-discrimination signal in the 7D MVP set. This is a return-to-design step, not a parameter tweak.

Sticky-strength escalation is *not* in the ladder — at the data scales in this fit, sticky pseudocounts of 3 → 10 are mathematically inert and do not constrain dwell.

---

## Verification protocol

The MVP fit is verified by MVP-internal sensitivity tests run alongside the production fit. All thresholds and decisions are documented in `results/mvp/diagnostics/verification_report.md` from MVP outputs only — no V11 results are cited.

### V1. Dwell verification (Module 3 adapted to MVP)

Run a Module-3-style transition analysis on the MVP K=4 production fit. Adapted from `diagnostics/tier1_screening/module3_transition_analysis.py` to point at `results/mvp/<session>/mvp_rslds_results.npz`. Output `results/mvp/diagnostics/dwell_report.md`:

1. **Dwell ratio per state**: empirical / model-predicted mean dwell ≥ 0.5 per state. If any state fails, advance the §Fallback ladder.
2. **Fraction of dwells < 10s per state**: ≤ 50% per state. If any state fails, advance the §Fallback ladder.
3. **Transition-event coincidence test**: for each transition, compute distance to nearest condition boundary; KS test against a uniform-random null (1000 bootstrap replicates). Report KS statistic, p-value, mean-distance shift with 95% CI. **Note: the test must be implemented against an explicit uniform-random null distribution of transition placements; do not compare empirical CDF against a delta-at-zero (which mechanically gives KS=1.0 and p=0).** Sanity check: typical KS values for genuine event-locked transitions are 0.1–0.4, not 1.0.

### V2. K=3 vs K=4 sensitivity comparison (unconditional)

Run a K=3 hierarchical fit on the same 7-channel MVP scaffold (`scripts/_run_mvp_hierarchical.py --K 3 --suffix _k3`), in parallel with the K=4 production fit. Compare on MVP-internal evidence only:

- **BIC** (K=3 vs K=4): report Δ BIC. The decision rule is whichever K wins BIC; do not cite V11's 26D margin.
- **Held-out log-likelihood**: 5-fold leave-some-sessions-out CV on each K. Whichever K has higher mean held-out LL/frame/dim is preferred.
- **Pairwise emission-mean separation** at each K: report `min/max` ratio of emission-mean L2 distances; closer to 1.0 indicates equidistant, well-separated states; closer to 0 indicates near-duplicate states.
- **Merger pattern at K=3**: identify which K=4 states are absorbed in K=3 and report the loadings. If the merger is COUP↔SHARED (multimodal distinction lost), document the trade-off. If it's NULL↔autonomic-quiescence (low-engagement merger), reconsider whether K=4's distinction is meaningful.

The methods section reports the comparison and selects the empirically-winning K. K=3 may turn out to be the production choice; this is acceptable.

### V3. Protocol-stratified sensitivity fits

Hierarchical pooling assumes that meditation and PE protocols share enough transition structure to be fit jointly. Test this assumption by running two additional fits:

- **Meditation-only fit**: hierarchical rSLDS on the meditation-protocol sessions only (n≈6).
- **PE-only fit**: hierarchical rSLDS on the PE-protocol sessions only (n≈5).

For each, fit at the K-value selected in V2, with otherwise-identical hyperparameters. Compare to the pooled production fit:

- **Ranked-dyad selection**: do the highest- and lowest-coupling sessions identified in each protocol-stratified fit match the selections from the pooled fit? If yes, pooling is robust. If no, the pooled fit's W_trans is biased by the protocol asymmetry and Figure 1's selections should be re-run on within-protocol rankings.
- **Per-state usage by condition**: do meditation-protocol per-condition state usages from the meditation-only fit qualitatively agree with the pooled fit? Same for PE.

Document the comparison in `results/mvp/diagnostics/protocol_stratified_report.md`. Compute cost: 2 additional hierarchical fits × ~50 min = ~100 min.

### V4. Shared-`d_emit` identifiability sensitivity

Per-session emission means are estimated with ~125 observations per parameter, which is adequate but not generous given gamma-weighted assignment. Test whether figures depend on per-session `d_emit` deviations by refitting with `d_emit` shared across sessions (no per-session deviation) and comparing:

- **Per-session-state usage trajectories**: same time courses, same condition-aggregate state usage, or different? Report both Figure 1 and Figure 2 panels rendered from this constrained fit.
- **BIC**: shared-`d_emit` will have higher (worse) BIC by definition; the question is the magnitude — small Δ implies the shared-emission assumption is approximately what the data supports anyway, large Δ implies per-session deviations are doing real work.

Document in `results/mvp/diagnostics/shared_demit_report.md`. Compute cost: 1 additional hierarchical fit × ~50 min.

### V5. Coupling-flexibility partial-circularity diagnostic

The `coupling_flexibility` covariate is computed from a graph-spectral analysis of multimodal channels that includes some of the 7 MVP observation channels. This is not full circularity (the construct is a meta-property) but partial. Quantify with: report the **partial correlation** between `coupling_flexibility` (the input covariate) and the leading 3 PCs of the 7-channel emission residual (`y_t − C[k_t] x_t − d_emit[k_t]`, the part of `y_t` not explained by the inferred state). If partial r > 0.5, document this and report `max|S|` for flexibility with a caveat; if r < 0.3, the covariate is genuinely external to the emission residuals.

This is computed once after the production fit completes; no additional fit needed.

### Failure handling

- If V1 fails despite the §Fallback ladder, the channel set is the issue, not the model — return to channel-set design rather than continuing to escalate parameters.
- If V2 selects K=3, all downstream figures, condition-aggregate panels, and the §Verification documentation use the K=3 fit. The "NULL / COUP / SHARED / autonomic-quiescence" labeling becomes whatever the K=3 fit's emission profiles support (potentially "NULL / merged-active-coupling / autonomic-quiescence").
- If V3 shows protocol-asymmetry biases the pooled selections, switch to within-protocol selection for Figure 1 dyads (high-meditation + low-meditation, OR high-PE + low-PE) and document.
- If V4 shows per-session `d_emit` deviations are doing critical work, defend the n=14 hierarchical pooling explicitly in methods (sample size adequate for the per-session emission parameters at this complexity).
- If V5 shows partial r > 0.5, document the caveat and report a sensitivity figure where flexibility is replaced by `lambda_2` only as the transition covariate.

---

## Figure 1 — Dyad-level variability (cohort)

### Panels

```
A. Highest (COUP+SHARED)-during-conv dyad — full session timecourse
B. Lowest  (COUP+SHARED)-during-conv dyad — full session timecourse
C. Cohort condition aggregate — shared conditions only (n=14 per bar)
```

### Selection criterion

Rank all 14 sessions by mean `(COUP + SHARED) gamma` averaged across timepoints labeled `conv_1` and `conv_2`. Select the highest and lowest. Conversation phases are common to both protocols, so this ranking is protocol-neutral and isolates dyad-level coupling capacity from protocol-induced floor effects (per CLAUDE.md, meditation phases drag down whole-session COUP+SHARED averages).

### Caption agreement check (not independent validation)

Report the raw `(conc_theta + conc_alpha)` mean surrogate-excess z during `conv_1+conv_2` for both selected dyads. Note that `conc_theta` and `conc_alpha` are 2 of the 7 MVP observation channels and are the highest-loading channels on COUP — the agreement between the rSLDS state-usage ranking and the raw concordance excess z is therefore expected by construction and is not an independent validator.

The caption text frames this honestly: "Selected dyads rank highest/lowest by both rSLDS state-usage and raw EEG-concordance surrogate-excess z (z = X.XX vs Y.YY); the two rankings agree as expected since concordance is a model emission channel, providing a within-channel sanity check rather than independent validation." This sentence preempts the "isn't this circular?" reviewer question by acknowledging it explicitly. Stronger ranking validation (held-out window, channel-disjoint validator) is deferred to follow-up work.

### Layout details

- **Panels A, B**: discrete-state color strip as the headline. Same color palette: NULL=gray, COUP=blue, SHARED=green, autonomic-quiescence=orange. Same time normalization (rescale x-axis to fraction-of-session [0, 1] to handle unequal session durations). Phase-boundary dashed vertical lines with condition labels above the strip. Show full session including all phases regardless of protocol — this is the "continuous state assignment visual" the figure exists to demonstrate.

- **Panel C**: stacked bars per condition. X-axis = {`base_EO`, `base_EC`, `conv_1`, `conv_2`}. Bar height = 1.0 (normalized state usage). Stacking order = NULL bottom, COUP, SHARED, autonomic-quiescence top. Error bars = ±1 SEM across the 14 sessions per stacked segment. Drop protocol-specific middle phases (`meditate_B/K`, `PE_1/PE_2`) — those move to Figure 2 to keep cohort claims commensurable.

---

## Figure 2 — Protocol comparison

### Panels

```
A. Highest (COUP+SHARED)-during-conv MEDITATION session — full timecourse
B. Highest (COUP+SHARED)-during-conv PSYCHOEDUCATION session — full timecourse
C. Per-protocol condition aggregate — meditation (n≈6) | PE (n≈5)
```

### Selection criterion

Within each protocol, select the session with the highest mean `(COUP + SHARED) gamma` during `conv_1+conv_2`. Sessions selected for Figure 2 may overlap with Figure 1 (e.g., if the cohort highest is also the meditation highest); this is acceptable.

### Layout details

- **Panels A, B**: same conventions as Figure 1 panels A/B (color palette, fraction-of-session x-axis, dashed phase boundaries). Panel A includes `meditate_B` and `meditate_K` as middle phases; panel B includes `PE_1` and `PE_2`. Visual narrative: similar baselines (`base_EO`, `base_EC`, `conv_1`), divergent middles (autonomic-quiescence dominance in `meditate_K` vs sustained engagement in PE), differential `conv_2` (per CLAUDE.md, meditation rigidifies coupling into post-meditation conv_2; PE shows no equivalent).

- **Panel C**: side-by-side stacked bar chart with two protocol groups. Left half = meditation columns (`base_EO`, `base_EC`, `conv_1`, `meditate_B`, `meditate_K`, `conv_2`, n≈6). Right half = PE columns (`base_EO`, `base_EC`, `conv_1`, `PE_1`, `PE_2`, `conv_2`, n≈5). A vertical separator and protocol labels above the column groups. Same stacking and color conventions as Figure 1 panel C. Error bars = ±1 SEM within protocol.

### Caption framing

"Matched baselines (`base_EO`, `base_EC`) confirm both protocols start from comparable cohort-level state-usage profiles. Divergent middle phases (`meditate_B/K` vs `PE_1/PE_2`) reflect the experimental manipulation: meditation drives autonomic-quiescence dominance peaking at `meditate_K` (38.5% in K=4 no-null V11 fit), while psychoeducation maintains active-coupling engagement. Post-intervention `conv_2` shows differential carry-over: meditation-protocol dyads show suppressed coupling flexibility persisting into conversation, consistent with meditation-induced rigidification reported in V11 analysis."

---

## Implementation outline (for plan-writing phase)

0. **Phase 0: DDTW pose validation** (`cadence/significance/pose_ddtw.py` + `scripts/_validate_pose_ddtw.py`):
   - Implement `compute_pose_ddtw()` — sliding window (4s window, 500ms stride at 12 Hz native pose rate) + per-session PCA to 10D + surrogate calibration (200 circular shifts) + AR(1) prewhitening. Stride matches rSLDS rate, no resampling. CPU implementation via `dtaidistance`; joblib threading over sessions.
   - Run the four validation tests (semi-synthetic dose response with Kendall τ permutation; condition contrast with paired t + Wilcoxon; pseudo-dyad null; descriptive redundancy vs multi-lag cross-correlation baseline).
   - Write `results/mvp/phase0/phase0_report.md` with decision rule outcome.
   - Selected pose channel (multi-lag baseline or DDTW) is read by Step 1 below.

1. **Slice scaffold** (`scripts/_run_mvp_scaffold.py`): for each session in the
   canonical cohort (`cadence.ingest.quality.list_canonical_sessions()`),
   first verify pipeline freshness:

   a. Read `data/digest/v1/<sid>.json` -> `xdf_md5`.
   b. Read `results/v11/<sid>/scaffold_v11_ztimecourses.npz` (NPZ keys are
      `z_<modality>` and `u_<covariate>` per `cadence/constants.py:V11_MODALITY_KEYS`
      / `V11_COVARIATE_KEYS`) AND its sidecar
      `results/v11/<sid>/scaffold_v11_results.json`.
   c. **Staleness check:** the V11 scaffold sidecar must record the same
      `xdf_md5` as the digest. If absent (older V11 runs predate the
      provenance field) the runner falls back to a content-hash on the
      preproc artifacts (`data/preproc/{eeg,face,ecg,pose}/v1/<sid>.json`
      `digest_xdf_md5` fields, all four expected to match). If the V11
      scaffold's recorded `digest_xdf_md5` disagrees with the digest, fail-loud
      with a clear directive: re-run V11 scaffold for that session before
      MVP slice. Sessions with no V11 scaffold present at all are skipped
      with a warning entry in `cohort_protocol_assignment.csv`.
   d. Sessions in the canonical registry but lacking
      `data/preproc/eeg/v1/<sid>.npz` (i.e. no fresh MATLAB clean.mat) are
      flagged `included_in_production_fit=False` in the cohort table; they
      can still be sliced for sensitivity analysis if their V11 scaffold
      exists from a prior run, but the production K=4 fit excludes them.

   Once freshness is verified, select the 7 observation channels by name +
   2 covariates, write `results/mvp/<sid>/mvp_scaffold.npz` (arrays:
   `obs (T, 7)`, `cov (T, 2)`, `obs_valid (T, 7)`, plus passthrough of the
   V11 prewhitening masks). Write `mvp_scaffold.json` with `digest_xdf_md5`
   and `v11_scaffold_md5` (sha256 of the V11 NPZ). The pose column is
   sourced from `results/mvp/phase0/pose_ddtw_per_session.npz` if Phase 0
   selected DDTW; otherwise from the multi-lag cross-correlation column
   already in the V11-cached scaffold. Preserves prewhitening + validity
   masks.

2. **Fit production hierarchical rSLDS** (`scripts/_run_mvp_hierarchical.py`): adapted from `scripts/_run_v11_hierarchical.py`. Same model class, same training loop, K=4, `null_state=False`, D_latent=3, n_factors=2, sticky=3.0, `viterbi_min_dwell=20`. Per-session NPZ outputs match the V11 result schema (gamma, path, x_smooth, d_emit, C_emit, R_emit + shared transition/dynamics tensors). Threading backend per the Windows DLL handling pattern. Estimated wall-clock: **50–90 min for 14 sessions at 7D**. An additional `viterbi_min_dwell=0` unconstrained-Viterbi pass is run as a sensitivity decode (this is post-fit decoding, not a re-fit — milliseconds).

3. **K=3 sensitivity fit** (`scripts/_run_mvp_hierarchical.py --K 3 --suffix _k3`): unconditional, run in parallel with K=4 production fit. Same scaffold, same hyperparameters except `K=3`.

4. **Protocol-stratified sensitivity fits** (`scripts/_run_mvp_hierarchical.py --protocol meditation --suffix _med` and `--protocol pe --suffix _pe`): two additional fits at the empirically-winning K, on meditation-only (n≈6) and PE-only (n≈5) subsets.

5. **Shared-`d_emit` sensitivity fit** (`scripts/_run_mvp_hierarchical.py --share-demit --suffix _shared_demit`): one additional fit with per-session `d_emit` deviation disabled.

6. **Verification** (`scripts/_run_mvp_verification.py`, extends `diagnostics/tier1_screening/module3_transition_analysis.py`):
   - V1: dwell-ratio + < 10s fraction + transition-event coincidence KS test (with explicit uniform-random null per §Verification protocol) on the K=4 fit. Output `results/mvp/diagnostics/dwell_report.md`.
   - V2: K=3 vs K=4 BIC + held-out LL + emission-mean separation comparison. Output `results/mvp/diagnostics/k_comparison_report.md`. Select K_winner.
   - V3: protocol-stratified comparison vs pooled. Output `results/mvp/diagnostics/protocol_stratified_report.md`. Switch to within-protocol Figure 1 selection if pooling is biased.
   - V4: shared-`d_emit` figure-stability comparison. Output `results/mvp/diagnostics/shared_demit_report.md`.
   - V5: coupling-flexibility partial-correlation diagnostic. Output `results/mvp/diagnostics/flexibility_circularity_report.md`.
   - Top-level `verification_report.md` summarizes V1–V5 outcomes and final K choice.

7. **Figure generation** (`scripts/_make_mvp_figures.py`):
   - Use the K_winner fit (K=3 or K=4) and the pooled-or-stratified production fit per V3 outcome.
   - Compute per-session ranking metric `mean((COUP + SHARED) gamma)` (or merged-active-coupling at K=3) over `conv_1`+`conv_2` timepoints.
   - Compute per-session raw `(conc_theta + conc_alpha)` excess z from existing surrogate-calibrated outputs for the caption agreement-check (with the honest framing per §Figure 1 caption).
   - Render Figure 1 (3 panels) and Figure 2 (3 panels) using matplotlib with shared style configuration. Output PNG + PDF at 300 DPI.

---

## Performance and parallelization

### CPU parallelism (in-scope for MVP)

The rSLDS implementation in `cadence/significance/rslds_model.py` is pure numpy/scipy (no torch, no GPU backend). Existing patterns to retain in the MVP runners:

- **Per-session E-step** (`fit_hierarchical_slds`, line ~1801 of `rslds_model.py`): `joblib.Parallel(n_jobs=-1, prefer='threads')` over the N sessions. Each session's E-step is independent given the shared params and benefits from numpy's GIL-releasing BLAS calls.
- **BLAS thread limiting** inside each E-step worker (line ~1795): `threadpoolctl.threadpool_limits(limits=1, user_api='blas')`. Without this, OpenBLAS's default 24 threads contend with joblib's per-session threads. The pattern must be preserved.
- **Top-level threading backend** in the hierarchical runner (`scripts/_run_v11_hierarchical.py`, line ~197): `with parallel_backend('threading'): result = fit_hierarchical_slds(...)`. This is required on Windows because loky workers re-import numpy before any user code runs, triggering the torch 2.10 + numpy 2.4 DLL-load bug. The MVP hierarchical runner must replicate this pattern. Reference: `project_win_torch_dll_fix.md`.
- **Per-session scaffold slice** (`_run_mvp_scaffold.py`): use `joblib.Parallel(n_jobs=-1, prefer='threads')` over the 14 sessions. Each iteration is small NPZ I/O (read 26-channel scaffold, slice 7 columns + 2 cov columns, write reduced NPZ). Total wall-clock < 30 s parallel vs ~5 min sequential — trivial but free.
- **Module 3 verification** (`module3_transition_analysis.py` extension): joblib threading over the 14 sessions. Per-session work is dwell-statistic computation + KS test against bootstrap null — already vectorized in numpy. Parallel wall-clock < 1 min.

### Structural speedup from D_obs reduction

The dominant MVP speedup over V11 comes from `D_obs` 26 → 7, not from parallelization. Per-iteration cost in the rSLDS E-step scales linearly to quadratically with D_obs (emission means O(D·K·T), Kalman smoother filter O(D·T·D_latent²), emission covariance update O(D²·T)). Expected hierarchical-fit wall-clock at 7D: **50–90 min** vs V11's 215 min at 26D, on the same threading-backend infrastructure.

### GPU usage

**The MVP introduces no new GPU compute.** Every GPU operation already done during V11 scaffold construction is reused without re-running:

- `cadence/significance/fast_cycles.py` `extract_burst_grids()` — burst coincidence base; not used in MVP
- `cadence/significance/bl_wavelet.py` — facial wavelet coherence; produces `bl_expr` and `bl_activity_conc` that the MVP reads
- `cadence/significance/directed_burst_coupling.py` `gpu_sliding_te_surrogates()` — TE concordance/asymmetry; not used in MVP
- Coupling flexibility and `lambda_2` — produced from V11 18D base via numpy graph operations; the MVP reads the existing `cov` array slice

The 7D observation array, 2D covariate array, and per-session prewhitening masks are all already in `results/v11/<session>/scaffold_v11_ztimecourses.npz`. The MVP scaffold runner is column slicing + NPZ rewrite.

**Forward-looking 16 GB guardrail (for future MVP-extension work that adds GPU operations):**

Any GPU operation in CADENCE that the MVP design does not cover (e.g., a future GPU port of the rSLDS Kalman smoother, or re-extraction of MVP channels under a new pipeline variant) must chunk so peak GPU memory stays under 16 GB. Existing modules to follow as reference:

- `extract_burst_grids()` chunks per session and per band; per-session GPU footprint < 1 GB at 256 Hz × 14 channels × 60 min.
- `bl_wavelet.py` chunks by frequency band, holds CWT coefficients on GPU only during coherence computation, transfers to CPU between bands.
- `gpu_sliding_te_surrogates()` chunks by surrogate batch (typically 50 surrogates / batch).

If future work introduces a new GPU operation, document the chunk size and per-chunk GPU memory budget in the module's docstring before the change is merged.

### Wall-clock estimate (full MVP pipeline)

| Step | Time | Notes |
|---|---|---|
| Phase 0: DDTW pose computation (14 sessions, 4s window, 500ms stride, T=48, D=10 shared PCA) | ~20–25 min | `dtaidistance` CPU + joblib threading over sessions |
| Phase 0: Semi-synthetic dose response (~30 pseudo-dyad pairs × 5 κ levels, ~60s injection segments only) | ~10–15 min | CPU; injection script + DDTW recompute per κ on short segments only |
| Phase 0: Condition contrast + pseudo-dyad null + redundancy descriptive | ~3 min | Statistics on already-computed timecourses |
| **Phase 0 total** | **~35–45 min** | Plus ~2–3 days dev time for the DDTW module + validation tests |
| Scaffold slice (14 sessions) | < 30 s | joblib threading over sessions |
| Production hierarchical rSLDS fit (K=4, 7D, 14 sessions) | 50–90 min | threading backend |
| K=3 unconditional sensitivity fit (7D, 14 sessions) | 40–80 min | run in parallel with K=4 if compute permits |
| Protocol-stratified meditation fit (n≈6) | 20–40 min | smaller n |
| Protocol-stratified PE fit (n≈5) | 20–40 min | smaller n |
| Shared-`d_emit` sensitivity fit (14 sessions) | 50–90 min | full cohort, slightly faster than production due to reduced parameters |
| Verification (V1–V5) | ~5 min | mostly statistics on already-computed outputs |
| Figure 1 + Figure 2 rendering | ~2 min | sequential matplotlib |
| **Total runtime** | **~5–7 hours** | All sensitivity fits run unconditionally; can be parallelized across CPU cores if RAM permits |

### Tunable parameters that trade speed for thoroughness

- `n_restarts`: number of EM restarts at initialization. V11 default is in `IOHMMConfig`. Reducing from default to `n_restarts=1` would halve init time but increase risk of local-optimum traps. Recommendation: keep V11 default; the cleaner channel set should make the EM landscape less rugged anyway.
- `max_em_iter`: V11 default is in `IOHMMConfig`. Same trade-off. Recommendation: keep default; use the convergence criterion (`em_tol`) to terminate early if possible.
- `em_tol`: relative log-likelihood improvement threshold. Looser tolerance = earlier termination. Recommendation: keep V11 default unless wall-clock becomes a blocker.

---

## Critical files

### NEW
- `cadence/significance/pose_ddtw.py` — DDTW pose-coupling module: shared cross-session PCA, sliding-window dependent DTW (`dtaidistance.dtw_ndim`), condition-block circular-shift surrogate calibration, AR(1) prewhitening
- `scripts/_validate_pose_ddtw.py` — Phase 0 validation runner: semi-synthetic injection (Kendall τ permutation), condition contrast (paired t + Wilcoxon), pseudo-dyad null, descriptive redundancy vs multi-lag cross-correlation baseline; writes `phase0_report.md` with channel decision
- `scripts/_run_mvp_scaffold.py` — slice 7D obs + 2D cov from V11-cached scaffold per session; pose column sourced per Phase 0 outcome
- `scripts/_run_mvp_hierarchical.py` — hierarchical rSLDS fit on MVP scaffold; argparse flags: `--K {3,4}`, `--protocol {meditation,pe,all}`, `--share-demit`, `--suffix <str>` for sensitivity variants
- `scripts/_run_mvp_verification.py` — V1–V5 verification runner: dwell + transition-event coincidence (with explicit uniform-random null), K-comparison BIC + held-out LL, protocol-stratified figure-stability, shared-`d_emit` figure-stability, coupling-flexibility partial-correlation diagnostic
- `scripts/_make_mvp_figures.py` — Figure 1 + Figure 2 generators
- `results/mvp/phase0/` — DDTW timecourses + validation CSVs + report
- `results/mvp/` — output tree (per-session, hierarchical, sensitivity variants, figures, diagnostics)

### Prerequisite installation
- `pip install dtaidistance` — verify import succeeds AND `from dtaidistance.dtw_ndim import warping_paths` works (this is the dependent multivariate variant). Document the pinned version in `cadence/requirements_mvp.txt` (new file). If install fails on Windows, fall back to the multi-lag cross-correlation baseline per Phase 0 failure handler.

### MODIFY
- None. The MVP is read-only over existing V11 scaffold and rSLDS infrastructure.

### READ ONLY
- `cadence/significance/rslds_model.py` — model class, do not modify
- `cadence/constants.py` — channel-name conventions (`V11_MODALITY_KEYS`, `V11_COVARIATE_KEYS`)
- `cadence/ingest/` — Layer 1 ingestion module (digest, roles, quality, schema). Read for provenance
  semantics; do not modify in MVP scope.
- `cadence/preprocess/{eeg,face,ecg,pose}/` — Layer 2 preproc submodules. The MVP reads outputs only;
  do not modify pipelines.
- `cadence/data/__init__.py` — eager-loading legacy dict-view shim (the V11 scaffold reads through it).
- `configs/session_quality.yaml` — canonical cohort registry. Read by
  `cadence.ingest.quality.list_canonical_sessions()` for the MVP cohort.
- `scripts/_run_scaffold_v11.py` — scaffold extraction lineage (NPZ writer is at `_run_scaffold_v11.py:624`)
- `scripts/_run_v11_hierarchical.py` — fit-loop reference for adapted hierarchical runner
- `diagnostics/tier1_screening/module3_transition_analysis.py` — verification module template
- `data/digest/v1/<sid>.{npz,json}` — Layer 1 outputs (raw streams + roles + markers + xdf_md5).
- `data/preproc/{eeg,face,ecg,pose}/v1/<sid>.{npz,json}` — Layer 2 outputs.
  Phase 0 DDTW reads `data/preproc/pose/v1/<sid>.npz` `{p}_pose33` directly.
- `data/matlab/<sid>_p{1,2}_clean.mat` — MATLAB-cleaned EEG with `.mat.json` provenance sidecars.
- `results/v11/<session>/scaffold_v11_ztimecourses.npz` — source for the 7D obs + 2D cov MVP slice.
- `results/v10/coupling_excess/` — legacy surrogate-calibrated raw concordance z-scores for caption
  agreement-check. Falls back to recomputation from `data/preproc/eeg/v1/<sid>.npz` if absent or stale.
- `docs/data_pipeline_v1.md` — canonical Layer 1 + Layer 2 schema reference.

---

## Verification: how to test the MVP end-to-end

-1. **Verify pipeline freshness** (one-time, before first MVP run):
   - `python -m cadence.ingest --all` — ensure all canonical digests exist
     and are up-to-date (idempotent; skips fresh ones).
   - `python -m cadence.preprocess.pose --all`,
     `python -m cadence.preprocess.face --all`,
     `python -m cadence.preprocess.ecg --all`,
     `python -m cadence.preprocess.eeg --all` — ensure all four preproc
     artifacts are fresh per session.
   - Sessions waiting on a fresh MATLAB EEG run (y04, y11, y24, y_53 as of
     2026-05-01) will be flagged by the EEG submodule with a clear error;
     they will appear as `included_in_production_fit=False` in
     `results/mvp/cohort_protocol_assignment.csv` until MATLAB runs.
   - `python scripts/_run_scaffold_v11.py --all` — regenerate any V11
     scaffolds whose `digest_xdf_md5` is missing or stale. (V11 scaffold
     itself reads from the new pipeline via the `cadence.data` shim, so
     this single command propagates digest -> preproc -> V11 scaffold
     freshness in one pass.)

0. **Run Phase 0 validation**: `python scripts/_validate_pose_ddtw.py --all`
   - Expected: `results/mvp/phase0/phase0_report.md` with explicit pass/fail per Test 1–3, descriptive Test 4 r value, and a final pose-channel decision line. Pose channel for downstream steps is determined by this report.
1. **Slice a single session** (`y_06`, the workhorse): `python scripts/_run_mvp_scaffold.py --session y_06`
   - Expected: `results/mvp/y_06/mvp_scaffold.npz` with `obs.shape == (T, 7)`, `cov.shape == (T, 2)`, channel names `["conc_theta", "conc_alpha", "bl_expr", "bl_activity_conc", "pose", "resp", "ecg_hf"]`.
   - Expected: `results/mvp/y_06/mvp_scaffold.json` with `digest_xdf_md5`
     matching `data/digest/v1/y_06.json`'s `xdf_md5` and a `v11_scaffold_md5`
     content hash, so MVP outputs are provenance-traceable to raw XDF.
2. **Fit single-session rSLDS** (sanity check before hierarchical): K=4, `null_state=False`. Verify state usage is roughly balanced (no state at < 5% or > 60%).
3. **Run K=4 hierarchical fit + sensitivity fits in parallel**: K=3 unconditional, protocol-stratified meditation, protocol-stratified PE, shared-`d_emit`. All five fits run from the same scaffold.
4. **Run V1–V5 verification**. Inspect `verification_report.md` and the V1/V2/V3/V4/V5 sub-reports.
5. **Resolve K choice**: per V2, select K_winner. If K_winner = 3, all downstream uses the K=3 fit and the labeling becomes whatever K=3's emission profiles support.
6. **Resolve dyad-selection scope**: per V3, decide whether Figure 1 uses cross-cohort or within-protocol ranking.
7. **Generate Figure 1 + Figure 2**. Visually inspect for: clear color contrast between high/low dyads in Figure 1, distinct autonomic-quiescence dominance in `meditate_K` of Figure 2 panel A (if K=4 won), comparable baselines in Figure 2 panels A/B.

---

## Out of scope (deferred to future grant work)

- Re-running the full V11 production pipeline with the MVP channel set as a proposal (the K=4 no-null V11 fit at 26D is the production model; MVP is parallel).
- Re-running V11 burst coincidence, directed burst coupling, or any post-hoc V11 analyses on the MVP fit.
- Adding novel channels (LZ, TE, gaze) — these are hypotheses to test against the MVP baseline in future work.
- Method comparisons against HMM/GMM beyond what Module 4 already established at 26D.
- Cross-validation beyond Tier 1 Module 3. The MVP relies on the existing V11 LOO-CV (Module 4) result that rSLDS K=4 > HMM K=4 by >1 SEM; if reviewers ask for re-verification at 7D, that's a follow-up.

---

*Spec self-review: no placeholders, no contradictions identified. Channel
justifications match the K=4 no-null diagnostic loadings. Figure scopes are
non-overlapping (Fig 1 = dyad variability across cohort; Fig 2 = protocol
comparison). Verification protocol is concrete with explicit thresholds and
a fallback ladder. K=3 comparison is conditional on K=4 fit symptoms, not
blocking.*

*Data-pipeline integration update (2026-05-01): the MVP now consumes Data
Pipeline v1 artifacts (digest + preproc) per `docs/data_pipeline_v1.md`.
Provenance flows: raw XDF md5 -> digest_xdf_md5 -> preproc sidecars ->
V11 scaffold -> mvp_scaffold.json. Cohort grew from 14 (V11 K=4 no-null
set) to up-to-22 canonical sessions; sessions waiting on a fresh MATLAB
EEG run are excluded from the production K=4 fit but tracked in the
cohort table. DDTW (Phase 0) reads `data/preproc/pose/v1/<sid>.npz`
`{p}_pose33` directly — uniform `(N, 33, 4)` shape across the three
supported pose formats (mediapipe33 / mediapipe33_meta / wholebody_133).
The 6 previously-mislabelled RA-tag sessions now resolve correctly as
therapist/patient, fixing role-asymmetry computations for those dyads.*
