# rSLDS Burst Analysis Synthesis — 2026-03-30

## Overview

Starting from quiver plots of the V8.2 rSLDS results, we progressively uncovered a layer of continuous coupling dynamics that the discrete state assignment compresses away. The session produced five scripts, 39 output files, and several validated findings about condition-specific coupling directionality.

---

## 1. What We Built

### Scripts

| Script | Purpose |
|---|---|
| `_plot_rslds_quiver.py` | State-center flow, per-modality emission profiles, observation trajectory quiver plots |
| `_plot_rslds_excursions.py` | Identify and decompose large trajectory jumps by modality group and condition |
| `_plot_rslds_bursts.py` | Continuous coupling intensity timecourses, burst detection, peri-burst averaging, cross-modal timing |
| `_validate_rslds_bursts.py` | Cross-session replication (12 sessions), pseudo-dyad null (30 pairs), permutation CIs (200 perms) |
| `_validate_bursts_by_condition.py` | Per-segment burst rates, lead/lag, and therapist/patient asymmetry (8 segments, no pooling) |

### Outputs

All saved to `results/rslds/quiver_plots/`. Key files:

- **State-center quiver plots** (`state_center_*`): Domain-axis projections (Phase Coupling vs Shared Power, etc.) showing where each rSLDS state sits in interpretable coupling space
- **Modality flow** (`modality_flow_*`): Per-channel emission means showing what each state "looks like" across all 12-15 observation channels
- **Trajectory quiver** (`trajectory_*`): Observation timecourses colored by Viterbi state with velocity arrows
- **Burst timeline** (`burst_timeline_y_06.png`): Full-session continuous coupling intensity per modality group with detected bursts marked
- **Peri-burst averages** (`peri_burst_by_segment.png`): Cross-modal timing triggered on each modality, broken out by experimental segment
- **Validation matrices** (`burst_validation_*`): Cross-session lead/lag, pseudo-dyad null comparison, permutation CIs
- **Asymmetry by segment** (`burst_asymmetry_by_segment.png`): Therapist/patient EEG power balance during coupling bursts, per protocol segment
- **Burst rate by segment** (`burst_rate_by_segment.png`): Coupling burst frequency per modality group per segment
- **JSON results** (`burst_validation_results.json`, `burst_by_condition_results.json`): Machine-readable numeric outputs

---

## 2. Methodological Progression

### 2a. From PCA to Domain Axes

Initial quiver plots used PCA to project the 15D emission space to 2D. This was replaced with domain-informed composite axes:

- **Phase Coupling** = mean(ImCoh theta/alpha/beta) — genuine inter-brain phase locking
- **Shared Power** = mean(Concordance theta/alpha/beta) — both brains in similar power state
- **EEG Dynamics** = mean(EWMAD theta/alpha/beta) — fluctuation rate of shared power
- **Body Coupling** = mean(BL expr, BL activity, Pose) — non-neural synchrony
- **Autonomic** = mean(ECG LF, ECG HF, Resp) — cardiovascular/respiratory coupling

**Rationale:** PCA maximizes total variance, which is dominated by concordance magnitude. The resulting axes are uninterpretable without decoding loadings. Domain composites preserve neuroscience meaning — every axis directly maps to a sentence in a paper.

### 2b. From State Assignments to Continuous Intensity

The rSLDS assigns one of 4 discrete states (NULL, COUP, OTHER, SHARED) per 0.5s timestep. This is a lossy compression: within-state variance is enormous, and brief intense coupling bursts get flattened to the same label as moderate sustained coupling.

We added four layers of increasing resolution:

1. **State assignment** (existing): categorical, ~100s resolution
2. **Coupling intensity** (new): continuous RMS per modality group, 3s smoothing
3. **Burst events** (new): discrete events with onset/peak/offset/amplitude/duration, detected at 90th percentile threshold with 2s minimum duration and 3s merge gap
4. **Peri-burst averages** (new): average all modality groups in a +/-10s window around each burst peak, revealing cross-modal timing

### 2c. Excursion Decomposition

Large trajectory arrows in the quiver plots (observations moving far beyond state centers) were decomposed into modality contributions using energy fractions. Key finding: excursions are driven by Face (27%), Autonomic (25%), and EEG power (25%) — not EEG phase (12%). The bursty modalities (face, heart rate) produce the largest transient jumps.

### 2d. Validation Layers

Three validation approaches were applied to the cross-modal lead/lag estimates:

1. **Cross-session replication** (12 sessions): aggregate lead/lag matrices with standard errors and sign consistency
2. **Pseudo-dyad null** (30 pairs): EEG channels from session A paired with non-EEG from session B, breaking real coupling while preserving marginal statistics
3. **Permutation CIs** (200 circular shifts on y_06): null distribution for peak latency under the assumption of no cross-modal temporal structure

**Result:** When pooled across conditions, no cross-modal lead/lag survived validation. All real peak latencies fell within the permutation null 95% CI. The pseudo-dyad null produced similar-magnitude lags. The y_06 finding of "Face leads EEG phase by 8s" was a single-session artifact.

### 2e. Condition Stratification

Pooling across conditions masked real effects because different conditions have opposite asymmetry profiles. The final analysis separated all 8 experimental segments without pooling:

- base_EO, base_EC (baselines)
- conv_1, conv_2 (pre/post-intervention conversations)
- meditate_B, meditate_K (body scan, loving-kindness)
- PE_1, PE_2 (psychoeducation blocks)

This revealed condition-specific patterns invisible in pooled analysis.

---

## 3. Validated Findings

### 3a. Therapist/Patient Asymmetry Reverses by Condition

**The direction of EEG coupling asymmetry during burst events depends on who is the active cognitive agent.**

Asymmetry = (therapist power - patient power) during coupling bursts. Positive = therapist-dominated. Values are z-scored concordance units.

| Segment | EEG Power theta | EEG Power alpha | EEG Power beta | EEG Phase beta |
|---|---|---|---|---|
| **conv_1** | **-0.213 +/- 0.036*** | -0.058 | -0.194 | -0.052 |
| **conv_2** | -0.091 | -0.094 | **-0.265 +/- 0.079*** | **-0.325 +/- 0.081*** |
| **meditate_K** | **+0.367 +/- 0.034*** | +0.139 | +0.034 | -0.041 |
| **meditate_B** | +0.027 | **+0.084 +/- 0.004*** | **+0.117 +/- 0.025*** | -0.045 |
| **PE_1** | **+0.679 +/- 0.177*** | **+0.810 +/- 0.017*** | **+0.634 +/- 0.028*** | +0.129 |
| **PE_2** | -0.058 | -0.098 | -0.073 | **+0.165 +/- 0.045*** |
| base_EC | +0.009 | +0.235 | +0.027 | +0.013 |

\* = |mean| > 2*SE

**Interpretation:** During conversation, the patient drives coupling bursts (negative asymmetry in theta). During meditation and PE, the therapist drives them (positive). PE_1 has the largest therapist dominance in the dataset (+0.679 in theta), collapsing to near-zero in PE_2 (habituation or patient engagement shift).

### 3b. Burst Rates Differentiate Conditions

| Segment | EEG phase | EEG power | Face | Autonomic | Body |
|---|---|---|---|---|---|
| base_EO | 0.8 | 0.8 | 1.6 | 1.4 | 0.6 |
| **base_EC** | **2.0** | 1.7 | 1.6 | 1.1 | 1.1 |
| **conv_1** | 1.8 | **2.2** | 1.9 | 1.0 | **2.1** |
| **conv_2** | 1.0 | 1.6 | **2.0** | 0.8 | 1.6 |
| meditate_B | 1.1 | 0.5 | 0.7 | 0.5 | 0.2 |
| meditate_K | 0.9 | 0.5 | 0.6 | 0.3 | 0.5 |
| PE_1 | 1.3 | 0.5 | 1.3 | 1.3 | 0.8 |
| PE_2 | 0.9 | 0.6 | 1.4 | 1.1 | 0.9 |

Values are bursts/minute (mean across sessions).

Key patterns:
- **base_EC has 2.5x the EEG phase bursts of base_EO** (2.0 vs 0.8/min). Eyes-closed baseline enables more phase coupling.
- **Conversation maximizes EEG power and Body bursts**. conv_1 peaks at 2.2 EEG power bursts/min and 2.1 Body bursts/min.
- **Meditation suppresses everything except EEG phase.** Face/Body/Autonomic drop to 0.2-0.7/min while EEG phase stays at ~1.0/min. Neural phase coupling persists during silent meditation.
- **conv_2 has more Face bursts than conv_1** (2.0 vs 1.9), but fewer EEG power bursts (1.6 vs 2.2). Post-intervention conversation is more facially expressive but less neurally power-coupled.

### 3c. Per-Participant Band Power Validates Protocol Design

Reconstructed from concordance + asymmetry: z_therapist = conc + asym/2, z_patient = conc - asym/2.

**Patient alpha — meditation vs PE:**
- meditate_B: +0.119 (eyes closed, internal focus)
- meditate_K: +0.035 (eyes closed, generating emotional states)
- PE_1: **-0.118** (eyes open, listening)
- PE_2: -0.065

meditate_B - PE_1 = **+0.236 +/- 0.052** (p<0.05). Patient alpha is significantly higher during meditation than psychoeducation, consistent with eyes-closed alpha enhancement.

**Patient theta — conversation vs meditation:**
- conv_1: +0.275 (active cognitive engagement)
- conv_2: +0.220
- base_EC: +0.118
- meditate_B: -0.119
- meditate_K: -0.091

Meditation vs conv_1 = **-0.39** (p<0.01). Conversations have ~0.4 SD more patient theta than meditation. Both meditation types suppress theta significantly below eyes-closed baseline (p<0.01), suggesting active suppression of default-mode wandering rather than simple disengagement.

**meditate_B vs meditate_K:**
- Alpha: +0.119 vs +0.035 (body scan has more alpha — deeper internal focus)
- Theta: -0.119 vs -0.091 (no significant difference)

### 3d. What Did NOT Validate

- **Cross-modal lead/lag timing.** No consistent lead/lag between modality groups survived cross-session replication, pseudo-dyad null comparison, or permutation testing. The y_06 finding of "Face leads EEG phase by 8s" was noise. At n=12 sessions with ~50 bursts per condition, there is insufficient power to detect sub-10-second cross-modal timing.
- **Burst coincidence beyond chance.** EEG phase and power bursts co-occur ~35% of the time within a 5s window. This is moderate but not dramatically above the chance rate given burst density.
- **Baseline asymmetry.** No significant therapist/patient asymmetry during baseline conditions (as expected — correct null).

---

## 4. Conceptual Framework

The rSLDS state assignment and the burst analysis capture different aspects of coupling dynamics:

| Property | State Assignment | Burst Analysis |
|---|---|---|
| Resolution | ~100s regimes | ~5s events |
| Output | Categorical (4 states) | Continuous intensity + discrete events |
| Amplitude information | Lost (COUP = COUP regardless of intensity) | Preserved (peak amplitude per burst) |
| Cross-modal structure | All modalities contribute to one state | Per-modality decomposition |
| Directionality | Symmetric (no therapist/patient) | Asymmetry channels reveal who drives each burst |
| Temporal structure | State usage fractions per condition | Burst rate, duration, peri-burst timing per condition |
| Validated at group level | Yes (hierarchical model, condition effects) | Burst rates and asymmetry replicate; lead/lag does not |

The two are complementary: the state assignment provides the macro-level "coupling regime" context, and the burst analysis characterizes the micro-level events within each regime.

---

## 5. What This Enables

### For the paper
- Report burst rates by condition with cross-session error bars (Figure: `burst_rate_by_segment.png`)
- Report therapist/patient asymmetry reversal as primary finding (Figure: `burst_asymmetry_by_segment.png`)
- Show continuous coupling intensity timeline as supplementary (Figure: `burst_timeline_y_06.png`)
- Patient alpha/theta validation confirms protocol fidelity

### For future analysis
- Burst events can serve as anchors for ERP-style cross-modal averaging once sample size increases
- The asymmetry decomposition (concordance +/- asymmetry) generalizes to any new modality with per-participant channels
- Burst rate per condition is a robust outcome measure for clinical comparisons (meditation protocol vs PE protocol)

### Known limitations
- Cross-modal lead/lag is underpowered at n=12 sessions. Need ~50+ sessions or ~500+ bursts per condition for reliable peri-burst timing.
- PE_2 asymmetry collapse needs replication — could be habituation, patient engagement, or small n (4 sessions).
- Burst detection threshold (90th percentile) is arbitrary. Sensitivity analysis across thresholds not yet performed.

---

## 6. File Inventory

### Scripts (5)
```
scripts/_plot_rslds_quiver.py          # State-center, modality flow, trajectory quiver
scripts/_plot_rslds_excursions.py      # Excursion decomposition and annotation
scripts/_plot_rslds_bursts.py          # Continuous intensity, burst detection, peri-burst
scripts/_validate_rslds_bursts.py      # Cross-session, pseudo-dyad, permutation validation
scripts/_validate_bursts_by_condition.py # Per-segment stratified validation
```

### Key Output Figures (12 primary)
```
results/rslds/quiver_plots/
  state_center_phase_vs_shared_hier.png    # State flow in interpretable 2D
  state_center_shared_vs_dynamics_hier.png # Shared power vs dynamics axis
  modality_flow_hierarchical.png           # Per-channel emission means (all states)
  modality_flow_y06.png                    # Same for y_06
  burst_timeline_y_06.png                  # Full-session coupling intensity
  excursion_analysis_y_06.png              # What drives large jumps
  excursion_annotated_y_06.png             # Top 15 excursions labeled on trajectory
  peri_burst_by_segment.png                # Cross-modal timing per condition
  burst_rate_by_segment.png                # Burst frequency per segment
  burst_asymmetry_by_segment.png           # Therapist/patient directionality
  burst_validation_lag_matrices.png        # Real vs pseudo-dyad null
  burst_validation_permutation_ci.png      # Permutation CIs for y_06
```

### JSON Results (2)
```
results/rslds/quiver_plots/
  burst_validation_results.json            # Cross-session lag matrices, null, permutation p-values
  burst_by_condition_results.json          # Per-segment burst rates, asymmetry, lag
```
