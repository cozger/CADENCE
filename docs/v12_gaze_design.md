# V12 Design: Stereo-Calibrated Gaze Coupling

## Overview

V12 extends V11 (28D obs + 7D cov) with gaze coupling channels derived from stereo-calibrated geometric gaze rays. Each participant's eye gaze is represented as a 3D ray; coupling is computed from the geometric relationship between rays and face positions.

This is the first modality where the raw measurement IS the coupling — no statistical inference (coherence, surrogates) needed to establish that coupling exists. The gaze-face distance is a direct, geometric, unambiguous measure of attentional coupling.

## Hardware

Stereo-calibrated camera system providing per-participant gaze rays at 30-60 Hz. Each gaze ray is a 3D origin + direction vector in a shared (calibrated) coordinate frame. Face positions available from the same system or from existing MediaPipe pose estimation.

## Primary Measures

### Directed gaze-face distances (the fundamental primitives)

Two directed timeseries per timepoint:
- **d(P1→P2)**: distance from P1's gaze ray to P2's face center (how far is P1 from looking at P2?)
- **d(P2→P1)**: distance from P2's gaze ray to P1's face center (how far is P2 from looking at P1?)

These are NOT symmetric — one participant can stare at the other while the other looks away.

### Closest-point distance (reserve for joint attention)

The minimum distance between the two 3D gaze rays in space. In face-to-face therapy, this is nearly identical to gaze concordance (r > 0.95) because joint attention at a third point is rare. However, if protocols include shared stimuli (screen-based tasks, PE materials), the closest-point distance captures "both looking at the same thing" which the face-directed distances miss.

**Use case separation:**
- Face-to-face therapy: concordance of face-directed distances is sufficient
- Shared-stimulus protocols: add closest-point as a separate channel to distinguish mutual gaze (at each other) from joint attention (at same stimulus)

## Scaffold Integration (observation/covariate rule)

### Observations (coupling state — magnitude measures)

| Channel | Formula | Interpretation |
|---------|---------|----------------|
| `gaze_conc` | (d_T→P + d_P→T) / 2 | Mutual attention intensity. Low = both looking at each other. High = both looking away. |

### Covariates (transition modulators — directional/derivative measures)

| Channel | Formula | Interpretation |
|---------|---------|----------------|
| `gaze_asym` | d_T→P - d_P→T | Who is attending to whom. Negative = therapist watching patient who looks away. |
| `gaze_approach` | -d/dt(gaze_conc) | Approach rate (positive = converging). Predicts imminent coupling state transitions. |

### Rationale for placement

Follows the V11-established principle: symmetric/magnitude → observations, signed/directional → covariates.

- `gaze_conc` is a state property: COUP state should have low gaze distance (mutual gaze), NULL state should have high gaze distance (independent attention). Different mean per state → observation.
- `gaze_asym` is directional metadata: who is watching whom modulates which state transitions are likely (therapist watching disengaged patient → different transition dynamics than mutual withdrawal). → covariate.
- `gaze_approach` is a temporal derivative: rate of gaze convergence/divergence predicts state transitions. Could be the strongest transition covariate — "they're about to look at each other" directly precedes coupling onset. → covariate.

## CWT Frequency Decomposition (second pass)

If raw gaze_conc shows condition structure at multiple timescales (likely), decompose via CWT into frequency bands:

| Band | Frequency | What it captures |
|------|-----------|------------------|
| Sustained | <0.5 Hz | Prolonged eye contact vs prolonged aversion (state-level attention) |
| Interaction | 0.5-2 Hz | Gaze-turn dynamics — natural rhythm of looking/looking away (~2-5s cycles) |
| Saccadic | 2-5 Hz | Fine gaze shifts — do they shift gaze at the same moments? |

This mirrors the BL wavelet pipeline (state/expression/speech bands). Would produce 3 observation channels instead of 1, capturing multi-scale gaze coupling.

**Implementation: start simple.** Raw gaze_conc at 2Hz first. Add CWT decomposition only if the raw signal shows multi-scale condition structure.

## Expected Condition Structure

| Condition | gaze_conc | gaze_asym | gaze_approach |
|-----------|-----------|-----------|---------------|
| base_EO | Moderate (polite looking) | ~0 (symmetric) | Low variance |
| base_EC | Very high (patient eyes closed) | Large (therapist watches, patient can't) | ~0 |
| conv_1 | Low (active eye contact) | Variable (turn-taking) | High variance |
| conv_2 | Low (active eye contact) | Variable | High variance |
| meditate_B | Very high (patient eyes closed) | Large | ~0 |
| meditate_K | Very high (patient eyes closed) | Large | ~0 |
| PE_1 | Moderate (patient may look at materials) | Therapist watches patient | Variable |
| PE_2 | Moderate | Variable (habituation) | Variable |

Key predictions:
- Gaze_conc should be the **crispest condition discriminator** in the scaffold — zero ambiguity between eyes-open conversation and eyes-closed meditation.
- Gaze_approach may be the **strongest transition covariate** — surpassing coupling flexibility — because gaze convergence directly precedes coupling onset.

## Scientific Value

### 1. Ground truth anchor for coupling states
Every other channel requires statistical inference to establish coupling. Gaze distance < 5cm is unambiguously "they're looking at each other." If COUP state doesn't align with low gaze distance, that's a model validity problem. This is the first channel that could serve as a hard constraint or validation target for state assignment.

### 2. Cross-modal Granger-causal anchor
Literature (Koul 2023) says behavioral synchrony Granger-causes neural synchrony. Gaze onset → neural coupling onset is the most plausible lead-lag pathway. The cross-modal timing that didn't validate in V8.2 burst analysis (n=12, underpowered, see docs/rslds_burst_analysis_synthesis.md section 3d) may validate with gaze as the behavioral anchor because it's a cleaner, crisper event than facial expression.

### 3. Continuous gradient reveals dynamics invisible to binary detection
The approach from 100cm → 5cm → 0cm before eye contact is an anticipatory coupling signal. Binary mutual gaze detection (threshold at e.g. 5cm) discards this. The continuous distance preserves the full approach/withdrawal dynamics.

### 4. Fills the attentional modality gap

| Modality | What it measures | Scaffold channels |
|----------|-----------------|-------------------|
| EEG | Neural synchrony | ImCoh, Conc, Dyn, Asym, LZ, Burst, TE |
| Face | Expressive synchrony | BL expr, BL activity |
| Body | Motor synchrony | Pose velocity |
| Autonomic | Physiological synchrony | ECG LF/HF, Resp |
| **Gaze** | **Attentional coupling** | **gaze_conc (+ CWT bands)** |

## V12 Scaffold Dimensions (projected)

### Minimal (raw gaze only)
- 29D observations (28 V11 + gaze_conc)
- 9D covariates (7 V11 + gaze_asym + gaze_approach)

### With CWT decomposition
- 31D observations (28 V11 + gaze_sustained + gaze_interaction + gaze_saccadic)
- 9D covariates (7 V11 + gaze_asym + gaze_approach)

### With joint attention (shared-stimulus protocols)
- 32D observations (31 + gaze_joint_attention via closest-point)
- 9D covariates (unchanged)

Model hyperparameters: start with V11-optimal D_latent=3, n_factors=2. Grid search if V12 BIC/T/D degrades.

## Implementation Plan

### Phase 1: Validate signal (one session)
1. Extract gaze rays from stereo system
2. Compute d(P1→P2) and d(P2→P1) at native rate
3. Resample to 2Hz, compute concordance
4. Plot gaze_conc timeline against conditions — verify expected structure (low conv, high med)
5. Check autocorrelation — expect moderate rho (like burst_coinc), minimal prewhitening needed

### Phase 2: Scaffold integration
1. Add gaze extraction function to V12 scaffold (same pattern as compute_burst_features)
2. Add gaze_conc to observations, gaze_asym + gaze_approach to covariates
3. Prewhiten jointly with all other channels
4. Run hierarchical rSLDS, compare BIC to V11

### Phase 3: CWT decomposition (if warranted)
1. CWT on gaze_conc at native rate (30-60 Hz)
2. Band-average into sustained/interaction/saccadic
3. Replace single gaze_conc with 3 band channels
4. Re-run hierarchical, compare

### Phase 4: Cross-modal validation
1. Gaze onset → EEG coupling onset lag analysis
2. Peri-gaze-onset burst rate analysis (do EEG bursts increase after eye contact onset?)
3. Granger causality: gaze_conc → ImCoh / burst_coinc

## Existing Data Note

Current recordings have MediaPipe blendshape eye gaze AUs (eyeLookDown/Up/In/Out, indices 10-17 in the 52-AU vector) at 30 Hz. These provide a rough proxy for gaze direction but lack the geometric precision of stereo-calibrated rays. They could be used for preliminary validation of the pipeline structure before stereo data is available, with the caveat that mutual gaze detection from blendshapes requires assumptions about seating geometry and is binary rather than continuous.
