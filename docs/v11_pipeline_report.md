# CADENCE V11 Pipeline: End-to-End Technical Report

> **Note on rSLDS state specification (2026-04-23):** the recommended production configuration is now **K=4, `null_state=False`**. The prior K=4 / `null_state=True` config is superseded: Tier 2 diagnostics showed OTHER was a phantom produced by the null constraint being slightly stricter than the data supports. The K=4/no-null refit has lower BIC (Δ = −13k) and recovers OTHER as a biologically meaningful autonomic-quiescence regime (peaks at 38.5% during Kundalini meditation). A K=3/no-null comparison was also run and rejected (BIC +51k vs K=4/no-null; collapses the two coupling regimes rather than dropping a dominated baseline). Full investigation and next steps in `docs/v11_identifiability_diagnostics.md`. Dimension numbers below reflect the post-`dyn_mean`-collapse scaffold (26D obs, not the earlier 28D).

## Overview

The V11 pipeline transforms raw multi-modal physiological recordings of two people interacting into a probabilistic model of their moment-to-moment coupling dynamics. It produces a **26-dimensional observation vector** and a **7-dimensional transition covariate vector** at 2 Hz, then fits a **hierarchical recurrent Switching Linear Dynamical System (rSLDS)** across all sessions to discover discrete coupling states and the forces that drive transitions between them.

The pipeline has three major phases:
1. **Data acquisition and preprocessing** — raw signals to clean, aligned timeseries
2. **Scaffold construction** — clean signals to 26D + 7D feature representation
3. **Hierarchical rSLDS** — feature representation to coupling state model and outputs

---

## Phase 1: Data Acquisition and Preprocessing

### 1.1 Raw Data Loading

Sessions are recorded using LabRecorder in XDF format. Each recording contains simultaneous streams from both participants:

- **EEG** (2 streams): Emotiv EPOC, 14 channels each, 256 Hz. Channels cover frontal (AF3/AF4, F3/F4), central (FC5/FC6, C3/C4), parietal (CP5/CP6, P3/P4), and occipital (O1/O2) regions.
- **ECG** (2 streams): Polar H10, single-lead chest strap, 130 Hz. Regular sampling — pyxdf dejitters Bluetooth timestamps.
- **Facial blendshapes** (2 streams): MediaPipe face mesh, 52 action unit coefficients, 30 Hz.
- **Body pose** (2 streams): MediaPipe pose, 33 keypoints with x/y/z/visibility, 30 Hz.
- **Markers**: Event timestamps encoding condition boundaries (e.g., `conv_1_start`, `meditate_B_stop`).

All streams are timestamped in LSL (Lab Streaming Layer) clock. The loader identifies which participant is therapist vs. patient from metadata, and assigns an **asymmetry sign** (+1 if P1 is therapist, -1 if P1 is patient) used throughout the pipeline to ensure consistent directionality.

### 1.2 EEG Preprocessing

Raw 14-channel Emotiv data passes through:

1. **Notch filtering** at 50 Hz and 60 Hz (removes mains interference from both EU and US power grids, Q=30).
2. **Bandpass filtering** at 1–45 Hz (4th-order Butterworth) — removes DC drift and high-frequency noise while preserving all standard EEG bands.
3. **Artifact masking** — samples exceeding 100 µV are marked invalid. These are physiologically implausible on Emotiv hardware and indicate movement artifacts or electrode contact loss.
4. **Per-channel z-scoring** — each channel is independently normalized to zero mean and unit variance using only valid samples. This removes gain/offset differences across channels and participants.
5. **Clipping** to [-10, 10] — prevents extreme outliers from dominating downstream computations.

### 1.3 ECG Preprocessing

Polar H10 single-lead ECG passes through:

1. **Bandpass filtering** at 0.5–40 Hz — removes baseline wander (respiratory artifact, below 0.5 Hz) and high-frequency noise.
2. **Z-score normalization** — standardizes across participants with different electrode impedance.

No resampling is applied because pyxdf already produces a uniform 130 Hz grid from the Bluetooth timestamps.

### 1.4 Blendshape Preprocessing

52 facial action unit coefficients (in [0,1] from MediaPipe) pass through:

1. **Face detection gating** — frames where the face tracker produced no detection (all AUs near zero) are marked invalid.
2. **Short gap interpolation** — gaps shorter than 0.5 seconds (15 frames at 30 Hz) are linearly interpolated. Longer gaps remain masked.
3. **Per-AU z-scoring** — each of the 52 AUs is independently z-scored using valid frames only. This converts the native [0,1] range to a distribution-centered representation.

### 1.5 Pose Preprocessing

33 body keypoints with 3D coordinates pass through:

1. **Visibility filtering** — keypoints with visibility below 0.5 are zeroed out (occluded or extrapolated).
2. **Frame validity** — a frame is valid only if at least 10 keypoints are visible.
3. **Per-coordinate z-scoring** on valid samples only.
4. **Clipping** to [-10, 10].

### 1.6 Temporal Alignment

Each modality starts and stops at different times due to hardware initialization delays. The alignment step:

1. Finds the **overlap interval** — the time range where ALL modalities have valid data (max of all start times to min of all end times).
2. **Zero-references** all timestamps by subtracting the start time, so the session begins at t=0.
3. Establishes a **common time grid** at 2 Hz (one sample every 0.5 seconds) spanning the full overlap duration. This is the scaffold's temporal resolution.

All subsequent resampling targets this 2 Hz grid via linear interpolation. The choice of 2 Hz balances temporal resolution against the coupling uncertainty principle — most interpersonal coupling signals evolve on timescales of seconds, not milliseconds.

The LSL offset between EEG timestamps and the common scaffold time grid is computed from the blendshape stream's timestamps, which serve as the reference clock. This offset is critical: raw EEG is at 256 Hz in EEG-local time, and every downstream feature that operates on raw EEG (burst grids, TE, LZ complexity) must apply this offset before resampling to the 2 Hz scaffold grid.

---

## Phase 2: Scaffold Construction (26D Observations + 7D Covariates)

The scaffold transforms preprocessed multi-modal streams into a unified representation suitable for state-space modeling. Each channel captures a specific coupling phenomenon between the two participants.

### 2.1 EEG Phase Coupling — Imaginary Coherence (3 channels)

**What it measures:** Phase-locked oscillatory coupling between participants' brains, resistant to volume conduction artifacts.

**How it works:**
1. Sliding 2-second windows at 0.25s stride are applied to both participants' 14-channel EEG.
2. Within each window, FFTs are computed via Welch's method (sub-segmented with Hann windows and 50% overlap).
3. Cross-spectra are computed for all ROI-pair combinations (4 ROIs × 4 ROIs = 16 pairs). ROIs are: Anterior (AF3/AF4/F3/F4), Central (FC5/FC6/C3/C4), Parietal (CP5/CP6/P3/P4), Occipital (O1/O2/POz/Oz).
4. **Imaginary coherency** is extracted: `ImCoh = |imag(Sxy)| / sqrt(Sxx · Syy)`. By taking only the imaginary part of the cross-spectrum, this metric rejects zero-lag coupling (which is dominated by volume conduction and shared reference artifacts) while preserving genuine phase-lagged neural coupling.
5. ImCoh is averaged across ROI pairs and frequency bins within each band: **theta** (4–8 Hz), **alpha** (8–13 Hz), **beta** (13–30 Hz).
6. Resampled to the 2 Hz scaffold grid.

This produces 3 channels: `imcoh_theta`, `imcoh_alpha`, `imcoh_beta`.

### 2.2 EEG Shared Power State — Concordance (3 channels)

**What it measures:** Whether both participants' brains are in similar power states (e.g., both showing high alpha power simultaneously).

**How it works:**
1. Per-participant band power is extracted from the same Welch windows used for ImCoh.
2. Each participant's band power is z-scored independently across the session.
3. Concordance is computed as `(z_P1 + z_P2) / 2` — the mean of both participants' standardized power. When both are high (or both low), concordance is high; when they differ, it's near zero.

This produces 3 channels: `conc_theta`, `conc_alpha`, `conc_beta`.

### 2.3 EEG Power Dynamics (3 channels)

**What it measures:** The rate of change in shared power — how volatile the concordance signal is.

**How it works:**
1. An exponentially weighted moving average deviation (EWMAD) with a 3-second time constant is applied to the concordance timecourse.
2. The result is log-transformed to compress the heavy-tailed distribution.

This captures moments of rapid co-fluctuation (e.g., simultaneous alpha desynchronization at the start of an interaction). Produces 3 channels: `dyn_theta`, `dyn_alpha`, `dyn_beta`.

### 2.4 EEG Asymmetry (3 channels)

**What it measures:** Which participant has higher band power — a proxy for who is more "active" in each frequency band.

**How it works:**
1. Per-participant z-scored band power is differenced: `asymmetry = asym_sign × (z_P1 - z_P2)`.
2. The `asym_sign` ensures positive values always mean "therapist has higher power" regardless of which physical participant was P1.

This produces 3 channels: `asym_theta`, `asym_alpha`, `asym_beta`. These are placed in **observations** (not covariates) because they characterize the current state — but their directional nature makes them borderline. They stayed in observations because they have been validated through semi-synthetic battery testing since V8.2.

### 2.5 Facial Expression Coupling (1 channel)

**What it measures:** Synchrony in facial expression dynamics between participants.

**How it works:**
1. Per-condition segments are extracted from the 52 AU blendshape timeseries (at 30 Hz).
2. A **Continuous Wavelet Transform** (Morlet wavelet, FFT-based) decomposes each AU into time-frequency representation across 30 log-spaced frequencies from 0.3–8 Hz.
3. **Wavelet coherence** is computed between participants on the "affect AUs" (smile, brow, cheek muscles) — specifically the expression band (0.5–2 Hz), which captures facial expression onset/offset dynamics.
4. The coherence is **surrogate z-scored**: 200 circular shifts of P2's CWT coefficients produce a null distribution, and the real coherence is expressed as a z-score above this null. The Welford online algorithm accumulates null statistics on GPU without storing all 200 surrogate matrices.
5. The maximum z-score across expression-band frequencies is extracted per timepoint and resampled to the scaffold grid.

This produces 1 channel: `bl_expr`.

Three empirically-grounded facial frequency bands are used: state (<0.5 Hz) captures slow emotional drift, expression (0.5–2 Hz) captures expression transitions like smiles, and speech (2–7 Hz) captures articulatory movements. The 8 Hz low-pass filter removes MediaPipe tracking noise before CWT.

### 2.6 Facial Activity Concordance (1 channel)

**What it measures:** Whether both participants have similar overall levels of facial movement (not expression-specific, just "how much is happening").

**How it works:**
1. A causal RMS-based activity channel is computed for each participant's blendshape data: instantaneous RMS across all AUs minus a trailing 30-second mean.
2. Activity concordance = `(z_P1_activity + z_P2_activity) / 2`.

This produces 1 channel: `bl_activity_conc`.

### 2.7 Cardiac Coupling (2 channels)

**What it measures:** Sympathetic and parasympathetic autonomic coupling between participants.

**How it works:**
1. Inter-beat intervals (IBI) are extracted from each participant's ECG via R-peak detection (threshold at 0.5× std, minimum 0.4s gap). Ectopic beats are removed by median filtering.
2. IBI timeseries are bandpass filtered into **LF** (0.04–0.15 Hz, mixed sympathetic/parasympathetic) and **HF** (0.15–0.4 Hz, parasympathetic) bands.
3. Hilbert amplitude envelopes are extracted from each band.
4. **Cross-products** of the envelopes (P1 × P2) at each timepoint give instantaneous coupling intensity.
5. Resampled to the scaffold grid.

This produces 2 channels: `ecg_lf`, `ecg_hf`.

### 2.8 Respiratory Phase Coherence (1 channel)

**What it measures:** Whether participants are breathing in phase with each other.

**How it works:**
1. Respiratory rate is extracted from each participant's ECG signal (the Polar H10 chest strap captures respiratory modulation of the R-wave amplitude).
2. Phase coherence is computed as `cos(phi_P1 - phi_P2)`, where phi is the instantaneous respiratory phase from the Hilbert transform.

This produces 1 channel: `resp`.

### 2.9 Postural Coupling (1 channel)

**What it measures:** Coordinated body movement between participants.

**How it works:**
1. Upper-body joint velocities (first temporal derivative of keypoint positions) are computed for each participant.
2. **Multi-lag cross-products** are computed at lags spanning ±5 seconds. This is critical because postural mimicry is often delayed — one person leans forward, and the other follows seconds later.
3. The maximum absolute cross-product across the lag bank is taken as the coupling estimate at each timepoint.
4. Surrogate z-scored against 200 circular shifts.

The multi-lag design solved a long-standing problem: zero-lag velocity cross-products produce negative autocorrelation at lag>0 for first-differenced signals, making single-lag approaches systematically unable to detect lagged postural coupling.

This produces 1 channel: `pose`.

### 2.10 LZ Complexity — Concordance and Asymmetry (4 channels)

**What it measures:** Shared neural complexity state (concordance) and complexity asymmetry between participants.

**How it works:**
1. EEG from the **frontal ROI** (6 channels: AF3/F3/F4/AF4 and neighbors) is bandpass-filtered per band (theta, alpha).
2. The **Hilbert amplitude envelope** is extracted. This is deliberately chosen over the raw signal: amplitude modulation changes carry psychologically meaningful information (e.g., burst intensity), while zero-crossings do not change LZ complexity.
3. The envelope is averaged across frontal channels to produce a single ROI-level timecourse.
4. **Sliding windows** (4 seconds, 0.5s stride → 2 Hz output) are applied.
5. Within each window, the signal is **binarized** at its median value, then **Lempel-Ziv complexity (LZ76)** is computed. LZ76 counts the number of distinct substrings needed to describe the binary sequence, normalized by the theoretical maximum (n/log2(n)). Higher values indicate more complex, less predictable neural dynamics.
6. Per-participant LZ timecourses are z-scored independently, then:
   - **Concordance** = `(z_P1 + z_P2) / 2` — shared complexity state
   - **Asymmetry** = `asym_sign × (z_P1 - z_P2)` — who has more complex dynamics

The pure-numpy LZ76 implementation runs in ~0.74ms/window, avoiding the 30–60 second JIT warmup cost of Numba-based alternatives for negligible speed difference.

This produces 4 channels: `lz_conc_theta`, `lz_conc_alpha`, `lz_asym_theta`, `lz_asym_alpha`.

### 2.11 Graph Modularity (1 channel)

**What it measures:** The community structure of the multimodal coupling network — whether modalities form distinct clusters or operate as a unified system.

**How it works:**
1. The **base 18 channels** (V8.2 channels: 12 EEG + 2 BL + 2 ECG + 1 Resp + 1 Pose) are assembled.
2. In sliding 90-second windows with 15-second stride, a **correlation graph** is built: each of the 18 channels is a node, and edges are the absolute Pearson correlations between channels within the window, thresholded to keep only significant edges.
3. **Louvain community detection** finds the modular structure, and the **modularity Q** score quantifies how strongly the network decomposes into communities.
4. Resampled to 2 Hz.

Importantly, the graph is built from the **base 18D only**, not the full 26D — this prevents circularity, since some of the added channels (LZ, burst coincidence) are themselves derived from the base channels.

This produces 1 channel: `graph_modularity`.

### 2.12 Transfer Entropy Concordance (2 channels)

**What it measures:** Total bidirectional information flow between participants' burst patterns — does knowing one person's recent burst history help predict the other's next burst?

**How it works:**
1. A single call to `extract_burst_grids()` processes both participants' raw EEG through GPU FFT bandpass filtering and CPU per-cycle feature extraction. This produces boolean burst grids (channels × timepoints @ 2 Hz) per band per participant.
2. For **transfer entropy**, burst sequences are encoded into k-bit histories (k=3, so 8 possible patterns like "burst, no-burst, burst"). At each timepoint, a joint pattern encodes: P1's 3-bit history, P2's 3-bit history, and P2's current state → a single integer indexing into 2^(2k+1) bins.
3. These patterns are one-hot encoded and accumulated over sliding 60-second windows using a **cumulative-sum trick** — this gives O(N) computation instead of O(N×W) for window-based counting.
4. TE is computed as the log-ratio: how much does knowing P1's past reduce uncertainty about P2's future, beyond what P2's own past already tells us?
5. Both directions are computed (P1→P2 and P2→P1), each z-scored against 200 circular-shift surrogates using Welford online accumulation.
6. **Concordance** = `(z_P1→P2 + z_P2→P1) / 2` — total bidirectional flow. This is a state property: high during interactive coupling, low during independence.

This produces 2 channels: `te_conc_theta`, `te_conc_alpha`.

The decomposition of TE into concordance (observation) and asymmetry (covariate) is a key V11 design decision. Raw TE asymmetry conflates "bidirectional coupling" (both high, difference ≈ 0) with "no coupling" (both low, difference ≈ 0) — the rSLDS cannot distinguish these because both produce near-zero values. By placing concordance in observations and asymmetry in covariates, the model can separately represent "how much coupling" (via state emissions on concordance) and "who leads" (via state transition modulation from asymmetry).

### 2.13 Burst Coincidence (3 channels)

**What it measures:** Whether both participants exhibit EEG bursts within a narrow temporal window — a temporally precise coupling metric that bypasses the 2+ second uncertainty window of continuous metrics.

**How it works:**
1. Uses the same burst grids from step 2.12 (shared extraction, no redundant computation).
2. P1's burst grid is **dilated** by ±1 sample (±500ms at 2 Hz) using convolution with a ones kernel — this allows for slight temporal jitter in coupling.
3. **Pointwise AND** between dilated P1 and raw P2 gives the coincidence rate per channel.
4. Channels are averaged across the ROI.
5. A **rate gate** excludes channels where either participant has <1% burst rate — sparse grids would produce surrogate std ≈ 0, causing z-score overflow.
6. Z-scored against 200 circular-shift surrogates (shift P2, preserving burst autocorrelation and rate), clipped to [-10, 10].

This produces 3 channels: `burst_coinc_theta`, `burst_coinc_alpha`, `burst_coinc_beta`.

Band-specific burst detection criteria matter here: theta/alpha use permissive settings (monotonicity > 0.7, ≥2 consecutive cycles) while beta uses stricter criteria (monotonicity > 0.8, ≥3 consecutive cycles) based on Cole & Voytek 2019 and Sherman 2016, reflecting that beta events are genuinely shorter and burstier than alpha oscillations.

### 2.14 Prewhitening and Standardization

Before the 26 channels enter the rSLDS, they must be made suitable for a model that assumes conditionally Gaussian emissions:

1. **Conditional AR(1) prewhitening**: For each channel, the lag-1 autocorrelation (rho) is computed. If rho exceeds the threshold, the channel is prewhitened: `z[t] = z[t] - rho × z[t-1]`. This is applied iteratively (up to 3 rounds) until rho falls below threshold.
   
   **Why conditional:** Not all channels need prewhitening. Some (like burst coincidence) are already approximately white noise. Unnecessary prewhitening would amplify high-frequency noise. The threshold gates this per-channel.

2. **Standardization**: Every channel is shifted to mean=0 and scaled to std=1. This ensures all channels contribute equally to the rSLDS likelihood, preventing high-variance channels from dominating state assignments.

The prewhitening diagnostics are logged (rho before/after, whether prewhitening was applied, std before/after) for each channel, enabling post-hoc quality control.

### 2.15 Observation Mask

Not every channel has valid data at every timepoint. The observation mask is a boolean matrix (T × 26) that marks:

- **Dead channels**: Any channel with max absolute value < 1e-8 is masked entirely (e.g., missing ECG stream).
- **Face validity**: BL expression and activity channels are masked during periods when either participant's face was not detected by MediaPipe.
- **Pose validity**: Pose channel is masked when either participant's body tracking was insufficient.

The rSLDS uses this mask during the E-step to compute emission likelihoods only from valid observations, preventing missing data from corrupting state estimates.

### 2.16 Transition Covariates (7D)

These do not characterize the current state — instead, they **modulate the probability of transitioning between states**. The 7D covariate vector `U` at each timepoint contains:

**From V10 (5D):**

1. **z_slow PC1 and PC2** (2D): The base 18D observation matrix is heavily Gaussian-smoothed (long time constant), then PCA-reduced to 2 components and standardized. These capture slow behavioral drift — gradual changes in the overall coupling landscape over minutes. They let the model know whether the interaction is trending warmer or cooler, independent of rapid fluctuations.

2. **Coupling flexibility** (1D): From the windowed graph Laplacian eigendecomposition. The signal is projected onto graph eigenvectors, and flexibility = high-frequency energy / total energy. Low flexibility means all modalities are in sync (rigid coupling); high flexibility means modalities are operating independently. This has proven to be the **strongest transition covariate** in the model.

3. **Lambda-2** (1D): The algebraic connectivity — the second-smallest eigenvalue of the graph Laplacian. This measures the "bottleneck" of the coupling network: how easy it is to disconnect the network by removing one edge. High lambda-2 = tightly connected; low lambda-2 = fragile coupling.

4. **Graph change-point score** (1D): The derivative of modularity + lambda-2 over time, Gaussian-smoothed. Detects moments when the coupling topology itself is changing — not just the intensity of coupling, but which modalities are coupled to which.

**From V11 (2D):**

5. **TE asymmetry theta** (1D): `z_therapist→patient - z_patient→therapist`, sign-corrected via asym_sign. Positive when therapist's neural bursts predict patient's bursts (therapist leads). This modulates transitions because leadership direction affects which state the dyad moves toward.

6. **TE asymmetry alpha** (1D): Same for alpha band. Beta TE asymmetry was deliberately excluded due to high collinearity (r = -0.74) with the existing power asymmetry observation channel.

All 7 covariates are independently prewhitened (AR(1)) and standardized before inclusion in the U matrix.

The observation vs. covariate placement follows a principled rule: **symmetric/magnitude measures → observations**, **signed/directional measures → covariates**. Observations characterize what state the system is in (e.g., high TE concordance = interactive state). Covariates modulate what state the system transitions to (e.g., therapist-leading TE asymmetry promotes transition toward teaching state). Violating this rule — putting TE asymmetry in observations — was experimentally tested and caused the NULL state to absorb 47.7% of the data, because the model couldn't distinguish "balanced bidirectional coupling" from "no coupling" (both have asymmetry ≈ 0).

---

## Phase 3: Hierarchical rSLDS

### 3.1 Model Architecture

The rSLDS models the 26D observation timeseries as generated by a system that switches between K=4 discrete states, each with its own emission characteristics:

- **Discrete state z_t** ∈ {0, 1, 2, 3}: Which coupling regime the dyad is in at time t.
- **Emissions**: Each state k has emission mean μ[k] (26D) and diagonal variance σ²[k] (26D). The observation y_t at time t is drawn from: `y_t ~ N(μ[z_t], σ²[z_t])`.
- **Transitions**: The probability of moving from state j to state k is a softmax over logits: `P(z_t=k | z_{t-1}=j, u_t) = softmax(W[j,k] + S[j,k,:] · u_t)`, where W is the base transition matrix and S modulates transitions based on covariates.
- **Null state** (k=0): A designated "no coupling" state with fixed mean μ[0] = 0 and capped variance. This prevents any state from becoming a catch-all for low-signal periods.
- **Sticky transitions**: Self-transition logits are boosted, encouraging the model to stay in a state rather than oscillating rapidly (minimum dwell enforcement via constrained Viterbi post-hoc).
- **Low-rank noise factors** (n_factors=2): The emission covariance includes 2 low-rank factors that capture cross-channel residual correlations (e.g., burst coincidence theta and alpha share extraction noise).

### 3.2 Hierarchical Fitting: 4 Phases

The model is fit across all sessions simultaneously to discover shared coupling dynamics while respecting session-specific physiology.

#### Phase 1: Per-Session Initialization

Each session is fit independently with a reduced configuration. This produces session-specific initial estimates of emission means, variances, and transition matrices. Multiple random restarts guard against local optima.

#### Phase 2: State Alignment (Hungarian Algorithm)

A fundamental challenge in fitting mixture models independently is **label switching** — state 1 in session A might correspond to state 3 in session B. To resolve this:

1. The session with the highest log-likelihood is chosen as the **reference**.
2. For every other session, the Hungarian algorithm finds the permutation of state labels that minimizes the distance between that session's emission means and the reference's emission means.
3. All session parameters are reordered accordingly.

After this step, state 0 = NULL, state 1 = COUP, etc., consistently across all sessions.

#### Phase 3: Pool Shared Parameters

Transition parameters (W, S) and initial-state probabilities are averaged across sessions to create a shared starting point. Emission parameters remain session-specific (each dyad has different baseline physiology).

#### Phase 4: Hierarchical EM (Alternating)

The core fitting loop alternates between:

**E-step** (per session, sequential): Forward-backward algorithm in log-space computes:
- γ[t,k] = P(z_t = k | all observations) — marginal state posteriors
- ξ[t,j,k] = P(z_{t-1}=j, z_t=k | all observations) — pairwise transition posteriors
- Log-likelihood for convergence monitoring

**M-step shared transitions**: W and S are optimized on pooled transition posteriors from all sessions via L-BFGS-B with L2 regularization on S. This learns transition dynamics that generalize across dyads.

**M-step session-specific emissions**: Each session's emission means and variances are updated using the state posteriors as soft weights. The null state's mean is locked at 0. Emission loadings are regularized toward the cross-session mean (c_shrinkage=0.2) — this prevents session-specific emissions from drifting too far from the shared structure while still capturing individual differences.

**Staged fitting**: In early iterations, emission means are frozen to let the transition structure stabilize first. This prevents the well-known d-C tradeoff where emission means and transition probabilities co-adapt in a degenerate way.

Convergence is declared when the relative change in total log-likelihood falls below the tolerance threshold.

### 3.3 Viterbi Decoding with Minimum Dwell

The soft state posteriors γ are converted to a hard state sequence via constrained Viterbi:

1. Take the MAP (maximum a posteriori) state at each timepoint: `path[t] = argmax_k γ[t,k]`.
2. Iteratively enforce a minimum dwell time of 10 seconds: any segment shorter than 20 samples (at 2 Hz) is merged with the neighboring state that has higher mean posterior probability.
3. Repeat until all segments meet the minimum dwell constraint.

This prevents physiologically implausible micro-oscillations between states — real coupling regimes don't switch every second.

### 3.4 State Labeling

After fitting, the four states are assigned human-readable labels based on their emission profiles:

1. **NULL** (k=0): Fixed at zero mean, captures periods of no significant coupling.
2. **COUP**: The non-null state with the highest loading on imaginary coherence channels — represents active phase-locked neural coupling.
3. **SHARED**: The remaining state with the highest loading on concordance channels — represents shared power/activity states without phase coupling.
4. **OTHER**: The remaining state — captures coupling patterns that don't fit the primary two categories.

---

## Phase 4: Output Analysis

### 4.1 Per-Session Outputs

Each session produces:
- **State posterior timecourse** γ (T × 4): soft probability of each state at each timepoint.
- **Viterbi path** (T,): hard state sequence.
- **State usage**: fraction of time spent in each state.
- **Transition count**: number of state switches.
- **Per-condition state usage**: how state occupancy differs between conversation, meditation, psychoeducation, etc.

### 4.2 Shared Model Outputs

Across all sessions:
- **Emission profiles** μ[k] (26D per state): which channels are elevated or suppressed in each state. These are the "fingerprint" of each coupling regime.
- **Transition matrix** W (4×4): base transition probabilities between states.
- **Covariate effects** S (4 × 4 × 7): how each covariate modulates each possible transition. The magnitude |S[j,k,d]| indicates how strongly covariate d influences the j→k transition.
- **BIC** (Bayesian Information Criterion): model selection score.

### 4.3 Post-Hoc Coupling Excess Analysis

Operating on the 23D (V10) scaffold features directly, this analysis asks: "Is the real coupling genuinely above what random chance produces?"

1. Channels are grouped into 8 modality groups (EEG Phase, Facial, LZ Shared, Respiratory, Postural, EEG Power, Facial Activity, Autonomic).
2. Per-group RMS intensity is computed with Gaussian smoothing.
3. 200 circular-shift surrogates (independent per-channel shifts) generate a null distribution.
4. **Excess z-score** = `(real_intensity - null_mean) / null_std` per group per timepoint.
5. Discrete coupling events are detected (z > 2.0, minimum 2 seconds, merge gaps < 3 seconds).
6. **Cross-modal event coincidence analysis** tests whether events in one modality predict events in another (e.g., does a facial coupling burst predict an EEG coupling burst within 5 seconds?), with analytic binomial p-values under the independence null.

Groups are classified into tiers: **Tier 1** (coupling-specific: ImCoh, BL expr, LZ conc, Resp, Pose), **Tier 2** (activity-confounded: Concordance, BL activity, ECG). Tier 1 metrics genuinely index inter-person coupling; Tier 2 metrics partially reflect individual activity levels that happen to co-occur.

### 4.4 Post-Hoc Burst Coincidence Analysis

A separate analysis path that revisits the native-rate EEG burst grids after rSLDS fitting:

1. Per-condition mean coincidence z-scores reveal which conditions produce genuine burst synchrony (e.g., theta/alpha burst coincidence is strongest during eyes-closed rest, not conversation — a distinct coupling mechanism from ImCoh).
2. Per-rSLDS-state mean coincidence z-scores test whether the discrete states the model discovered align with burst-level coupling events.
3. Cross-band and cross-modal ECA tests which coupling modalities temporally co-occur.

### 4.5 Post-Hoc Directed Coupling Analysis

Tests leadership/directionality in the coupling:

1. **TE asymmetry** per condition reveals who leads information flow (e.g., therapist→patient dominates during psychoeducation; patient→therapist dominates during conversation).
2. **Directed episodes** (sustained TE asymmetry z > 2.0, minimum 3 seconds, 5 second merge window) are detected and classified as "therapist-leads" or "patient-leads".
3. Per-condition and per-state fractions of therapist-leading vs. patient-leading time are computed.
4. ECA asymmetry is also computed but noted as **confounded by burst rate differences** — the participant with more bursts mechanically "leads" more often. TE is the trustworthy directed metric.

### 4.6 V7 Timeline Visualization

A comprehensive per-session visualization combining all modalities:

1. EEG per-band z-score timecourses (from `fast_cycles` coupling analysis)
2. Blendshape expression-band power (participant 1 plotted upward, participant 2 downward)
3. Blendshape speech-band power (same dual-axis layout)
4. Z-scored wavelet coherence spectrogram (time × frequency heatmap, non-significant regions muted, significant regions colored)
5. Band-averaged z-score timecourses for state, expression, and speech bands

Condition boundaries are overlaid as colored background regions, providing a single visual overview of the entire session's multi-modal coupling dynamics.

---

## Summary: Data Flow

```
Raw XDF (EEG 256Hz, ECG 130Hz, BL 30Hz, Pose 30Hz, Markers)
  │
  ├─ Preprocessing (notch, bandpass, z-score, artifact mask)
  │
  ├─ Temporal alignment → common 2 Hz grid
  │
  ├─ Feature extraction:
  │   ├─ EEG: ImCoh + Concordance + Dynamics + Asymmetry (12ch)
  │   ├─ BL: Wavelet coherence + Activity concordance (2ch)
  │   ├─ ECG: Hilbert envelope cross-product LF/HF (2ch)
  │   ├─ Resp: Phase coherence (1ch)
  │   ├─ Pose: Multi-lag velocity cross-product (1ch)
  │   ├─ LZ: Hilbert amplitude LZ76 concordance + asymmetry (4ch)
  │   ├─ Graph: Louvain modularity on base 18D (1ch)
  │   ├─ TE: Concordance from GPU sliding-window TE (2ch)
  │   └─ Burst: Coincidence on shared burst grids (3ch)
  │
  ├─ Prewhitening (conditional AR(1)) + Standardization → 26D observations
  │
  ├─ Transition covariates: z_slow PCs + flexibility + λ₂ + change-point + TE asymmetry → 7D
  │
  ├─ Observation mask (face/pose validity, dead channels)
  │
  └─ Hierarchical rSLDS:
      ├─ Per-session init → Hungarian alignment → pooled shared params
      ├─ Alternating EM (shared transitions, session-specific emissions)
      ├─ Constrained Viterbi (10s minimum dwell)
      └─ State labeling: NULL / COUP / SHARED / OTHER
          │
          └─ Post-hoc analyses:
              ├─ Coupling excess (surrogate-calibrated, per-group, per-condition)
              ├─ Burst coincidence (per-band, per-condition, per-state)
              ├─ Directed coupling (TE leadership episodes)
              └─ V7 timeline visualization
```
