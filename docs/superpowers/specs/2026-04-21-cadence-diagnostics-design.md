# CADENCE rSLDS Diagnostic Suite — Design Spec
**Date:** 2026-04-21  
**Status:** Approved for implementation

---

## Context

The CADENCE rSLDS pipeline fits a hierarchical recurrent switching linear dynamical system (K=4 states) to a 28-dimensional 2 Hz feature matrix derived from EEG, ECG/HRV, facial AUs, body pose, and burst/TE coincidence channels across ~15–17 sessions (mixed meditation + psychoeducation protocols). Three active concerns motivate this diagnostic pass:

1. **State flicker** — without a 10s Viterbi minimum constraint, state assignments switch rapidly. Unclear whether this reflects real fast dynamics, slow-drift feature confounds, or transition-matrix misfit.
2. **No cluster-structure validation** — the observation space has never been inspected to verify it supports discrete states rather than a continuous manifold.
3. **Feature dimensionality** — 28D at n=15 sessions is potentially overparameterized; no systematic collinearity or redundancy audit has been done.
4. **Model architecture** — rSLDS has not been benchmarked against simpler alternatives (HMM, GMM) at this sample size.

This diagnostic suite is a **read-only pass on existing fits** — it does not modify the rSLDS implementation. Findings are presented as reports; pipeline changes are decided separately.

---

## Architecture

```
diagnostics/
  tier1_screening/
    module2_feature_diagnostics.py   ← raw features, all sessions
    module3_transition_analysis.py   ← prewhitened features + Viterbi paths
    run_tier1.py                     ← orchestrator; writes screening_report.md
  tier2_deepdive/
    module1_obs_space.py             ← prewhitened features + rSLDS states
    module4_model_comparison.py      ← raw features, LOO-CV
    module5_block_pca.py             ← raw features + Module 2 flags
    module6_null_ablation.py         ← prewhitened features, re-fits model
    run_tier2.py                     ← reads screening_report; warns if flags unaddressed
  shared/
    data_loader.py                   ← single source of truth for loading raw/prewhitened/fits
    report_utils.py                  ← timestamped output dirs, Markdown writers
  outputs/                           ← all plots + CSVs + reports, timestamped subdirs
```

### Data representation per module

| Module | Input | Rationale |
|--------|-------|-----------|
| 2 | Raw surrogate-corrected (pre-whitening) | Diagnoses what whitening corrects for |
| 3 | Prewhitened + standardized | What the model actually sees |
| 1 | Prewhitened + standardized | Cluster structure in model input space |
| 4 | Raw (re-fits own preprocessing) | Baseline models handle their own whitening |
| 5 | Raw → block-PCA → whiten | PCA captures natural variance, then whitened before rSLDS |
| 6 | Prewhitened + standardized | Re-fits unconstrained rSLDS from same inputs |

### Protocol handling

- **Module 1 (UMAP/PCA)**: pool all sessions — UMAP needs full n for reliable neighbor structure
- **Module 3 (transition-event coincidence)**: run per-protocol — meditation and PE have non-overlapping phase labels; conflating them invalidates the event-coincidence null
- All other modules: pool across protocols, annotate plots with protocol membership

---

## Tier 1 — Screening

### Module 2: Feature Diagnostics

**Purpose:** Identify channels that are slow-drift dominated, underinformative, or collinear before any model comparison.

**Per-channel analysis:**
- ACF out to 60s lag with 95% CI bands (Bartlett formula)
- Power spectrum (Welch, 0–1 Hz range emphasized since features are at 2 Hz)
- N_eff per session via Geyer's monotone truncation (NOT naive `N / (1 + 2*sum(ACF))` — that formula goes negative for over-whitened signals)

**Cross-channel analysis:**
- 28×28 correlation heatmap at lag 0, channels sorted by modality group
- VIF table: for each channel, regress it on all others, report VIF = 1/(1 − R²)

**Flags (written to screening_report.md):**
- `SLOW_DRIFT`: ACF(10s) > 0.3 — channel not adequately handled by AR(1) whitening
- `LOW_INFO`: median N_eff < 50 across sessions — channel oversampled relative to independent content
- `COLLINEAR_PAIR`: pairwise |r| > 0.8 at lag 0 — report both channel names + r value; do not prescribe which to drop (domain decision)
- `HIGH_VIF`: VIF > 10 — multicollinear with the channel set collectively

**Output files:**
```
outputs/<timestamp>/module2/
  acf_all_channels.png         ← grid of 28 ACF plots
  power_spectra.png            ← grid of 28 power spectra
  correlation_heatmap.png      ← 28×28 cross-channel correlation
  vif_table.csv                ← channel, VIF, flag
  n_eff_per_session.csv        ← channel × session N_eff matrix
  module2_report.md
```

---

### Module 3: State Transition Analysis

**Purpose:** Determine whether rapid state switching reflects real dynamics (event-locked transitions) or model artifacts (transitions indistinguishable from random).

**Dwell-time analysis:**
- Compute unconstrained Viterbi path (re-run if only constrained path is saved)
- Empirical CDF of dwell times per state vs. geometric(1 − T_kk) — discrete-time, NOT continuous exponential
- Empirical mean dwell (seconds) vs. model-predicted mean = 0.5 / (1 − T_kk); flag ratio < 0.5
- Fraction of dwells < 10s per state (quantifies Viterbi minimum suppression)

**Per-session timeline:**
- Color strip row 1: unconstrained Viterbi state
- Color strip row 2: 10s-minimum Viterbi state
- Overlaid vertical lines: session-phase boundaries (protocol-specific labels, not conflated)
- Separate figure per session; saved as `session_<id>_timeline.png`

**Transition-event coincidence test (run per-protocol):**
- For each unconstrained transition, compute distance (seconds) to nearest session-phase boundary
- Null: same number of transitions, placed uniformly at random within session, 1000 bootstrap replicates
- Report: KS statistic, p-value, bootstrap 95% CI on mean distance shift
- Interpretation: p < 0.05 → transitions cluster near events (real signal); p > 0.1 → indistinguishable from random (model noise)

**Flags (written to screening_report.md):**
- `DWELL_RATIO_LOW`: per-state empirical/predicted dwell ratio < 0.5 — model transition matrix misfit
- `HIGH_FLICKER_PCT`: >50% of unconstrained dwells < 10s — Viterbi minimum is suppressing most transitions
- `TRANSITIONS_EVENT_LOCKED`: KS p < 0.05 — flicker is detecting real dynamics
- `TRANSITIONS_RANDOM`: KS p > 0.1 — flicker is model noise

**Output files:**
```
outputs/<timestamp>/module3/
  dwell_time_distributions.png   ← empirical CDF vs geometric null, per state
  session_<id>_timeline.png      ← one per session
  coincidence_test_results.csv   ← per-protocol KS stat, p-value, CI
  module3_report.md
```

---

## Tier 2 — Deep Dive

### Module 1: Observation Space Structure

**Purpose:** Assess whether the 28D prewhitened observation space has genuine discrete cluster structure or is continuous.

**PCA:**
- Scree plot: variance explained per component + cumulative
- Report number of PCs for 80% and 95% variance thresholds
- 2D scatter of PC1 vs PC2, PC1 vs PC3, PC2 vs PC3

**UMAP (n_neighbors=30, min_dist=0.1, exposed as CLI args):**
Run once, produce four colored 2D embeddings:
1. rSLDS-assigned state (post-Viterbi, 4 discrete colors)
2. Coarse protocol phase: {baseline, conversation, task} (task = meditate_B/K or PE_1/PE_2)
3. Session ID (detects if session effects dominate over coupling structure)
4. Time-within-session normalized 0→1 (detects temporal drift — if UMAP shows a gradient, AR(1) whitening is insufficient)

**Cluster quality (computed in PCA-reduced space, not full 28D):**
- Silhouette score for rSLDS state assignments
- Silhouette score for coarse protocol-phase labels
- Davies-Bouldin index for rSLDS states
- Key comparison: if phase silhouette > state silhouette, the states are primarily tracking experimental condition rather than coupling dynamics

**Interpretation logic:**
> "If rSLDS-state silhouette < protocol-phase silhouette: states track condition more than coupling. If both silhouettes are near zero in PCA space and UMAP coloring 4 (time) shows a gradient: prewhitening is insufficient. If silhouettes are near zero and no temporal gradient: observation space is genuinely continuous — consider reporting z_t latent trajectory as the primary object rather than state labels."

**Output files:**
```
outputs/<timestamp>/module1/
  pca_scree.png
  pca_scatter_2d.png          ← PC1/2/3 pairs
  umap_by_state.png
  umap_by_phase.png
  umap_by_session.png
  umap_by_time.png
  cluster_quality.csv         ← silhouette, DB index
  module1_report.md
```

---

### Module 4: Alternative-Model Baselines

**Purpose:** Determine whether rSLDS is justified over simpler models at n=15.

**Models:**
- Gaussian HMM (diagonal emissions), K ∈ {2, 3, 4, 5}
- GMM, K ∈ {2, 3, 4, 5} (no temporal structure)
- rSLDS K=4 (existing fit, or re-fit on LOO train sets)

Note: sticky HDP-HMM deferred — pyhsmm is unmaintained and ssm has environment conflicts. Flag for future addition.

**Evaluation:** leave-one-session-out CV (15 folds)

**Fair comparison protocol:**
- Metric: held-out log-likelihood per frame per dimension (`LL / (T × D)`) — normalizes for dimensionality differences
- For rSLDS: use emission LL conditioned on MAP z (Viterbi path), NOT marginal LL. This is a lower bound on rSLDS's true advantage; report it as such.
- Primary comparison: rSLDS K=4 vs. HMM K=4 — identical K and emissions, differs only in recurrent covariate modulation of transitions

**Per-model outputs:**
- Mean held-out LL/frame/dim ± SEM across sessions
- Mean state dwell time under Viterbi
- Silhouette score in PCA-reduced space (reusing Module 1's PCA)
- Sign test p-value: fraction of sessions where rSLDS > HMM K=4

**Interpretation logic:**
> "If HMM K=4 held-out LL is within 1 SEM of rSLDS K=4: recurrent covariate structure is not paying off at n=15. If GMM K=4 is within 1 SEM of HMM K=4: temporal dynamics are not contributing — observation structure is the primary signal source."

**Output files:**
```
outputs/<timestamp>/module4/
  loo_cv_results.csv              ← model × session LL/frame/dim
  model_comparison_summary.png    ← bar chart with SEMs
  dwell_time_by_model.png
  module4_report.md
```

---

### Module 5: Block PCA Preprocessing Variant

**Purpose:** Test whether per-modality PCA compression improves or preserves model quality while reducing dimensionality.

**Conditioned on Module 2 flags**: channels flagged as SLOW_DRIFT or HIGH_VIF are excluded from their group PCA before compression. The loadings dictionary names excluded channels explicitly.

**Modality groups (28D V11 scaffold):**

| Group | Channels |
|-------|----------|
| EEG_phase | imcoh_theta, imcoh_alpha, imcoh_beta |
| EEG_shared | conc_theta, conc_alpha, conc_beta |
| EEG_dynamics | dyn_theta, dyn_alpha, dyn_beta |
| EEG_asymmetry | asym_theta, asym_alpha, asym_beta |
| Facial | bl_expr, bl_act_conc |
| Autonomic | ecg_lf, ecg_hf, resp |
| Body | pose |
| Complexity | lz_conc_theta, lz_conc_alpha, lz_asym_theta, lz_asym_alpha |
| Graph | graph_mod |
| Burst_TE | te_conc_theta, te_conc_alpha, burst_coinc_theta, burst_coinc_alpha, burst_coinc_beta |

**PCA per group:** retain ≥80% cumulative variance, capped at 2 PCs per group. Report cap hit rate — if most groups hit the cap at 2, compression is doing real work.

**Output:** reduced feature matrix + loadings dict naming each PC (e.g., `EEG_phase_PC1: 0.71*imcoh_theta + 0.62*imcoh_alpha + 0.32*imcoh_beta`)

**rSLDS re-fit** (identical hyperparameters: K=4, same sticky strength, null-state constraint):

Side-by-side comparison vs. full 28D fit:
- State trajectories for 3 representative sessions: one meditation, one PE, one with high flicker rate from Module 3
- State dwell-time distributions
- Held-out LL/frame/dim (same LOO-CV folds as Module 4 for direct comparability)
- Silhouette score in PCA-reduced space

**Output files:**
```
outputs/<timestamp>/module5/
  block_pca_loadings.md        ← human-readable loadings dict
  reduced_feature_matrix.npz
  state_trajectory_comparison_<session>.png   ← 3 sessions
  dwell_time_comparison.png
  loo_cv_comparison.csv
  module5_report.md
```

---

### Module 6: Null-State Ablation

**Purpose:** Determine whether the forced null-coupling state is encoding real structure or absorbing variance that should go to data-driven states.

**Re-fit:** rSLDS K=4, no null-state constraint (all four states data-driven). Use identical hyperparameters otherwise.

**State alignment:** Hungarian matching on cosine similarity of emission mean vectors between constrained and unconstrained fits. Flag if no permutation achieves mean cosine similarity > 0.5 — indicates genuinely different structure discovered.

**Comparison metrics:**
- Emission mean L2 norm per state (sorted ascending) — does unconstrained model discover a near-zero state?
- Emission covariance Frobenius norm per state — does it also have low variance?
- KL divergence between matched Gaussian emissions (constrained vs. unconstrained)
- Held-out LL/frame/dim vs. constrained model (same LOO-CV)
- State occupancy per session phase: does the unconstrained "quiet" state (lowest-norm emissions) occur in the same phases as the constrained null state?

**Interpretation logic:**
> "If unconstrained lowest-norm state has cosine similarity > 0.7 with constrained null state and same phase occupancy: constraint is encoding a real regime, keep it. If four states have similar norms and no near-zero state emerges: constraint is forcing structure absent from data."

**Output files:**
```
outputs/<timestamp>/module6/
  emission_means_comparison.png    ← constrained vs unconstrained, sorted by norm
  state_occupancy_by_phase.png
  loo_cv_comparison.csv
  hungarian_alignment.csv          ← cosine similarities of matched states
  module6_report.md
```

---

## User Guide: What Does Each Output Mean?

This guide explains every plot type and metric so any lab member can interpret outputs without reading the code.

### Tier 1 outputs

---

#### ACF plot (Module 2: `acf_all_channels.png`)

**What it shows:** How correlated a channel's value at time t is with its own value at time t+lag, for lags from 0 to 60 seconds.

**How to read it:** The y-axis is correlation (−1 to 1); x-axis is lag in seconds. A flat drop to near zero within 5–10 seconds means the channel carries independent information at each timepoint. A slow decay that stays above 0.3 past 10 seconds means the channel is dominated by slow drift — its "signal" at any given moment is largely predictable from its value 10–60 seconds ago.

**Red flag:** Any channel where the ACF line stays above the dashed 0.3 line past the 10s mark is flagged as SLOW_DRIFT. These channels may cause the rSLDS to track drift as coupling states rather than real coupling dynamics.

**Gray bands** are 95% confidence intervals — values inside the band are not significantly different from zero.

---

#### Power spectrum (Module 2: `power_spectra.png`)

**What it shows:** How much variance in a feature channel exists at each frequency, from 0 (DC/constant) to 1 Hz (the Nyquist of a 2 Hz signal).

**How to read it:** A spike near 0 Hz means most of the channel's energy is in very slow trends (minutes-long drifts). A flat spectrum means the channel's variance is spread evenly across timescales. Most coupling features should peak in the 0.01–0.1 Hz range (10–100 second coupling fluctuations) and fall off toward DC.

**Red flag:** If a channel's spectrum rises sharply toward 0 Hz with most energy in the lowest bin, that channel is essentially a slow drift and will confound the rSLDS.

---

#### Cross-channel correlation heatmap (Module 2: `correlation_heatmap.png`)

**What it shows:** The Pearson correlation between every pair of the 28 feature channels at lag 0, arranged by modality group along both axes.

**How to read it:** Dark red = strong positive correlation (~1.0); dark blue = strong negative correlation (~−1.0); white = no correlation. Blocks of dark red along the diagonal correspond to modality groups whose channels co-vary — this is expected (e.g., theta and alpha burst coincidence both go up during EEG coupling bursts).

**Red flag:** Off-diagonal dark red cells (high correlation between channels from *different* modality groups) indicate redundancy. Two channels that correlate >0.8 are essentially the same signal measured twice — including both inflates the apparent dimensionality without adding information.

---

#### VIF table (Module 2: `vif_table.csv`)

**What it shows:** Variance Inflation Factor for each channel. VIF measures how well a channel can be predicted from all the other channels. VIF = 1 means completely independent; VIF = 10 means 90% of that channel's variance is explained by the others.

**How to read it:** Columns are: channel name, VIF value, flagged (yes/no). Values above 10 are flagged as HIGH_VIF.

**Why it matters:** Pairwise correlation catches two-way redundancy. VIF catches the case where three or more channels collectively explain each other even if no two channels are highly correlated pairwise. High VIF means the rSLDS emission matrix has near-collinear columns, making it harder to identify what each channel is contributing.

---

#### N_eff per session table (Module 2: `n_eff_per_session.csv`)

**What it shows:** The effective sample size (N_eff) for each channel in each session. N_eff is the number of truly independent observations, accounting for autocorrelation. A 20-minute session at 2 Hz has 2400 nominal timepoints, but if the ACF decays slowly, the effective number of independent samples may be much lower.

**How to read it:** Rows are channels; columns are sessions. High values (e.g., 200–800) are healthy. Values below 50 (flagged) mean the channel has very few independent observations per session, and statistical tests based on that channel are underpowered.

---

#### State timeline (Module 3: `session_<id>_timeline.png`)

**What it shows:** Two horizontal color strips across the session duration, stacked vertically. Top strip = state sequence from unconstrained Viterbi (no minimum dwell). Bottom strip = state sequence with 10-second minimum dwell constraint. Vertical dashed lines mark session-phase boundaries (baseline end, meditation start, etc.).

**How to read it:** Each color represents a state (null=gray, COUP=blue, SHARED=green, OTHER=orange). Compare the two strips: where they agree, the 10s minimum is not changing anything. Where the bottom strip shows a long solid block but the top strip shows rapid alternation, the minimum constraint is suppressing fast transitions.

**What to look for:** Do the long blocks in the bottom strip line up with session phases? Do the rapid transitions in the top strip cluster near the dashed phase-boundary lines? If yes, fast switching is detecting real transitions between conditions.

---

#### Dwell-time distributions (Module 3: `dwell_time_distributions.png`)

**What it shows:** For each of the 4 states, the distribution of how long the model stays in that state before switching, plotted as an empirical CDF (solid line) vs. the theoretical geometric distribution predicted by the fitted transition matrix (dashed line).

**How to read it:** X-axis = dwell duration in seconds; Y-axis = fraction of episodes shorter than that duration. If the solid line closely follows the dashed line, the model's transition matrix accurately predicts actual dwell behavior. If the solid line rises much faster than the dashed line (lots of very short episodes), transitions are happening faster than the model thinks they should — this is the "flicker" signature.

**Key numbers:** The ratio (empirical mean dwell) / (model-predicted mean dwell) is reported. Ratio near 1.0 = well-specified. Ratio < 0.5 = flickering twice as fast as the transition matrix expects.

---

#### Transition-event coincidence test (Module 3: `coincidence_test_results.csv`)

**What it shows:** Whether state transitions tend to happen near session-phase boundaries more than chance. For each transition, the distance to the nearest event is measured. This is compared against a null distribution of randomly-placed transitions.

**How to read it:** Key columns: `KS_statistic`, `p_value`, `mean_distance_real_s`, `mean_distance_null_s`. A low p-value (< 0.05) means transitions cluster near events — the flickering is tracking real dynamics. A high p-value (> 0.1) means transitions are placed randomly relative to events — the flickering is model noise.

---

### Tier 2 outputs

---

#### PCA scree plot (Module 1: `pca_scree.png`)

**What it shows:** How much variance each principal component explains (bar chart) and the cumulative variance (line). X-axis = component number (1 = most important); Y-axis = fraction of total variance.

**How to read it:** A sharp "elbow" at component k means the first k components capture most of the structure and the rest is noise. If the cumulative line reaches 80% at component 5, the 28 feature dimensions effectively carry ~5 dimensions of independent information. If 80% requires 15+ components, the features are genuinely diverse and compression will be lossy.

---

#### UMAP embeddings (Module 1: `umap_by_*.png`)

**What it shows:** Each timepoint (2 Hz × all sessions ≈ 36,000 points) is projected into 2D using UMAP, which preserves local neighborhood structure. Four versions, each colored differently.

**How to read each version:**

- **By state** (`umap_by_state.png`): If the rSLDS states form distinct islands in UMAP space, the model has found discrete regimes. If the four colors are interleaved or form a gradient, the states are carving up a continuous manifold arbitrarily.

- **By phase** (`umap_by_phase.png`): If baseline / conversation / task form distinct regions, the feature space primarily reflects experimental condition rather than moment-to-moment coupling variation. This is useful context but suggests the states are condition-tracking.

- **By session** (`umap_by_session.png`): If each session forms its own island, session-to-session variability dominates — the hierarchical pooling assumption may be strained. If sessions overlap substantially, pooling is valid.

- **By time-within-session** (`umap_by_time.png`): If there's a visible gradient from blue (early) to red (late) within each session, slow temporal drift is present despite AR(1) whitening. This is a red flag for Module 2's slow-drift findings.

---

#### Cluster quality metrics (Module 1: `cluster_quality.csv`)

**What it shows:** Silhouette score and Davies-Bouldin index for (a) rSLDS state assignments and (b) protocol-phase labels, computed in PCA-reduced space.

**Silhouette score** ranges from −1 to 1. Near 1 = each point is much closer to its own cluster than any other. Near 0 = overlapping clusters. Near −1 = points assigned to the wrong cluster.

**Davies-Bouldin index**: lower is better. DB = 0 means perfectly separated clusters; higher values mean clusters overlap.

**The critical comparison:** If the state silhouette is lower than the phase silhouette, the rSLDS states are less well-separated in observation space than the experimental conditions are. This doesn't invalidate the model (states may be capturing dynamics the conditions don't), but it means phase labels explain more geometric structure than the inferred states do.

---

#### LOO-CV comparison table (Module 4: `loo_cv_results.csv`, Module 5/6: similar)

**What it shows:** For each model and each left-out session, the held-out log-likelihood per frame per dimension. Negative values — more negative = worse fit.

**How to read it:** Compare the rSLDS column to HMM K=4. If rSLDS values are consistently less negative (higher), the recurrent covariate structure is earning its complexity. If they're within each other's SEM bars, the extra complexity isn't helping at n=15.

**Normalizing by frame × dimension** makes different-sized models comparable: a model that fits 28 channels and one that fits 12 channels are on the same scale.

---

#### Model comparison summary (Module 4: `model_comparison_summary.png`)

**What it shows:** Bar chart with one bar per model (GMM K=2..5, HMM K=2..5, rSLDS K=4). Height = mean held-out LL/frame/dim across sessions. Error bars = ±1 SEM.

**How to read it:** The rightmost rSLDS bar should ideally be the tallest (least negative). If the HMM K=4 bar overlaps with rSLDS's error bar, the recurrent input is not justified. If GMM K=4 overlaps with HMM K=4, temporal modeling isn't contributing.

---

#### Emission means comparison (Module 6: `emission_means_comparison.png`)

**What it shows:** For each of the 4 states, a horizontal bar showing the L2 norm of the emission mean vector (how "active" the state is on average across all features), for both the constrained model (left panel) and the unconstrained re-fit (right panel). States sorted ascending by norm.

**How to read it:** In the constrained model, the lowest bar should be near-zero (the forced null state). In the unconstrained model, look at whether the lowest bar is also near-zero or whether all four bars have similar heights.

**What to look for:** If the unconstrained lowest-norm state has a similar height to the constrained null bar, the model is self-discovering the null state without the constraint — the constraint is redundant but not harmful. If all four unconstrained bars are similar height, the constraint was preventing the model from finding four equally-active states.

---

#### Hungarian alignment table (Module 6: `hungarian_alignment.csv`)

**What it shows:** The best matching between constrained states (rows) and unconstrained states (columns), with cosine similarity scores showing how similar each matched pair's emission means are.

**How to read it:** High diagonal values (> 0.7) = unconstrained model found the same states as the constrained model. Low values = the unconstrained fit found different structure. A cosine similarity near 0 for the matched null-state pair means the unconstrained model's "equivalent" state has a very different mean emission profile.

---

## Dependencies

- `umap-learn` — for Module 1 UMAP (install via `pip install umap-learn`)
- `scikit-learn` — PCA, GMM, silhouette, Davies-Bouldin (already in MCCT environment)
- `hmmlearn` — Gaussian HMM baselines for Module 4 (`pip install hmmlearn`)
- `scipy` — KS test, exponential fit (already present)
- `matplotlib`, `numpy`, `pandas` — already present

Note: sticky HDP-HMM deferred due to pyhsmm/ssm environment conflicts. Flag for future addition.

---

## Verification: How to Test the Suite End-to-End

1. **Run Tier 1** on session y_06 (fastest session): `python diagnostics/tier1_screening/run_tier1.py --session y_06`
   - Expected: `outputs/<timestamp>/` directory created with module2/ and module3/ subdirs
   - Verify: `screening_report.md` exists and lists flags for at least one channel (te_conc channels are expected to show LOW_INFO based on prior model analysis)

2. **Inspect Module 3 timeline** for y_06: open `session_y_06_timeline.png`
   - Expected: unconstrained strip shows rapid switching in conversation phases; constrained strip shows longer blocks

3. **Run Tier 2 Module 1** on all sessions: `python diagnostics/tier2_deepdive/module1_obs_space.py --session all`
   - Expected: UMAP by-state shows partial separation; by-session shows substantial overlap (pooling is valid)

4. **Run Module 4** (expensive, ~30 min): `python diagnostics/tier2_deepdive/module4_model_comparison.py --session all`
   - Expected: rSLDS K=4 held-out LL ≥ HMM K=4; if not, flag for review

5. **Run Module 6**: `python diagnostics/tier2_deepdive/module6_null_ablation.py --session all`
   - Expected: unconstrained fit lowest-norm state cosine similarity > 0.5 with constrained null state

6. **Review all `*_report.md` files** in output directory for flag summaries and interpretation text before making any pipeline changes.

---

## Critical Files to Modify / Create

- `diagnostics/shared/data_loader.py` — **NEW**: must handle the case where scaffold saves only prewhitened data (reconstruct raw by unwhitening, or re-run scaffold extraction)
- `diagnostics/tier1_screening/module2_feature_diagnostics.py` — **NEW**
- `diagnostics/tier1_screening/module3_transition_analysis.py` — **NEW**: must re-run unconstrained Viterbi if only constrained path saved
- `diagnostics/tier2_deepdive/module1_obs_space.py` — **NEW**
- `diagnostics/tier2_deepdive/module4_model_comparison.py` — **NEW**
- `diagnostics/tier2_deepdive/module5_block_pca.py` — **NEW**
- `diagnostics/tier2_deepdive/module6_null_ablation.py` — **NEW**
- `diagnostics/shared/report_utils.py` — **NEW**
- `cadence/significance/rslds_model.py` — **READ ONLY** (diagnostics load, do not modify)

---

*Spec self-review: no placeholders, no contradictions, no ambiguity in thresholds. Scope is focused on diagnostics only — no pipeline changes included. User guide covers every output file produced.*
