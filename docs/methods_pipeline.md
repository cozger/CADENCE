# CADENCE Methods: From Raw Recordings to Hierarchical Coupling States

**Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation**

This document describes the complete CADENCE data processing and modeling pipeline, from raw multi-modal physiological recordings through to the hierarchical rSLDS that characterizes coupling state dynamics across therapy sessions.

---

## 1. Data Acquisition

### 1.1 Recording Setup

Each session records a patient-therapist dyad during a ~50-minute protocol consisting of structured conditions (baseline, conversation, guided meditation). Streams are synchronized via Lab Streaming Layer (LSL) and recorded to XDF format with LabRecorder.

**Recorded streams per participant:**

| Stream | Hardware | Channels | Native Rate |
|--------|----------|----------|-------------|
| EEG | Emotiv EPOC X | 14 (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) | 256 Hz |
| ECG | Polar H10 | 1 | 130 Hz |
| Facial Action Units | MediaPipe FaceLandmarker | 52 blendshapes | ~30 Hz |
| Body Pose | MediaPipe Pose | 33 landmarks x 4 (x,y,z,visibility) | ~30 Hz |

LSL handles clock synchronization across devices. The XDF loader (`xdf_loader.py`) extracts each stream, identifies participant roles (therapist vs patient) from stream metadata, and preserves LSL timestamps. The pyxdf library dejitters BLE-transmitted ECG timestamps during loading.

### 1.2 Why These Modalities

Each modality captures a distinct physiological channel of interpersonal coupling:

- **EEG**: Neural synchrony (inter-brain phase coupling, amplitude co-modulation). Literature shows theta-alpha hyperscanning correlates with shared attention (Dikker 2017, Goldstein 2018).
- **ECG**: Autonomic co-regulation. Sympathetic (HR, acceleration) and parasympathetic (RMSSD) synchrony have *opposite* relational valence (Palumbo 2017).
- **Facial AUs**: Behavioral mimicry and emotional contagion. Wavelet analysis captures multi-timescale expression dynamics (Jeganathan 2022).
- **Pose**: Postural synchrony and movement coordination. Captures gross motor mirroring that predicts rapport (Ramseyer 2011).

---

## 2. Preprocessing

Each modality undergoes signal-appropriate preprocessing (`preprocessors.py`) designed to preserve coupling-relevant dynamics while removing artifacts.

### 2.1 EEG

1. **Notch filtering**: Remove 50 Hz and 60 Hz powerline artifacts (IIR notch, Q=30)
2. **Bandpass**: 1-45 Hz (Butterworth 4th order, zero-phase `sosfiltfilt`)
3. **Artifact rejection**: Samples exceeding |100 muV| are marked invalid per channel
4. **Z-scoring**: Per-channel, computed on valid samples only
5. **Clipping**: Hard bounds at +/-10 sigma to contain residual artifacts

**Rationale**: The 1-45 Hz passband covers all EEG rhythms of interest (theta through gamma) while removing DC drift and high-frequency EMG contamination. Per-channel z-scoring normalizes across the 14 EPOC channels which have variable impedances.

### 2.2 ECG

1. **Bandpass**: 0.5-40 Hz (Butterworth 4th order)
2. **Z-score + clip**: Global normalization, +/-10 sigma

**Rationale**: The Polar H10 delivers clean single-lead ECG at 130 Hz. The 0.5 Hz high-pass removes respiratory baseline wander; 40 Hz low-pass retains the QRS complex while removing EMG noise. No resampling is applied -- the 130 Hz rate is regular (BLE transmission jitter is corrected by pyxdf).

### 2.3 Facial Blendshapes

1. **AU extraction**: First 52 columns from the 1489-dimensional landmark stream (MediaPipe blendshape coefficients, range [0,1])
2. **Face detection gating**: Frames where all AUs ~ 0 are flagged as face-not-detected
3. **Gap interpolation**: Linear interpolation across gaps shorter than 0.5s (~15 frames)
4. **Z-scoring**: Per-AU using valid frames only
5. **Activity channel**: Causal RMS deviation from a 30-second trailing mean, appended as channel 53

**Rationale**: The 52 MediaPipe blendshapes provide a comprehensive representation of facial expression without requiring custom AU coding. The activity channel captures overall expressiveness magnitude, useful for gating coupling analyses during periods of low facial movement.

### 2.4 Body Pose

1. **Reshape**: (N, 132) -> (N, 33, 4) separating (x, y, z, visibility) per landmark
2. **Visibility masking**: Coordinates with visibility < 0.5 are zeroed
3. **Frame validity**: Require >= 10 visible keypoints per frame
4. **Z-scoring**: Per-coordinate on visible values only
5. **Clipping**: +/-10 sigma
6. **Activity channel**: Causal RMS deviation (appended)

---

## 3. Feature Extraction

Preprocessed signals are transformed into interpretable, coupling-relevant features at standardized output rates.

### 3.1 EEG Features (8 channels @ 2 Hz)

Computed from 2-second FFT windows (512 samples at 256 Hz), stepped at 2 Hz:

| Channel | Feature | Description | Physiological Meaning |
|---------|---------|-------------|----------------------|
| 0 | Engagement index | beta / (alpha + theta) from frontal ROI | Attentional engagement |
| 1 | Aperiodic exponent | Log-log PSD slope (2-40 Hz) | Excitation/inhibition balance |
| 2 | Theta burst fraction | BOSC detection (chi2 threshold on Hilbert envelope) | Working memory load |
| 3-4 | Theta phase | cos(phi), sin(phi) from 4-8 Hz Hilbert | Theta rhythm timing |
| 5-6 | Alpha phase | cos(phi), sin(phi) from 8-13 Hz Hilbert | Alpha rhythm timing |
| 7 | Activity | Causal RMS deviation from 30s trailing mean | Signal quality / arousal |

**Frontal ROI**: Average of AF3, F3, F4, AF4 (anterior electrodes most sensitive to cognitive/emotional state).

### 3.2 ECG / Autonomic Features (7 channels @ 2 Hz)

Derived from R-peak detection (peak finding with minimum 0.4s inter-beat interval) followed by ectopic beat rejection (outlier IBIs > 40% from local median):

| Channel | Feature | ANS Branch | Interpretation |
|---------|---------|-----------|----------------|
| 0 | Heart rate (bpm) | Mixed | Global arousal |
| 1 | IBI deviation | Mixed | Beat-to-beat variability |
| 2 | RMSSD (10s rolling) | Parasympathetic | Vagal tone |
| 3 | HR acceleration | Sympathetic | Fight-or-flight activation |
| 4 | QRS amplitude | Sympathetic | Cardiac contractility |
| 5 | HR trend (10s slope) | Mixed | Sustained arousal shifts |
| 6 | RMSSD derivative | Parasympathetic | Autonomic state transitions |

**Design decision**: SNS and PNS features are separated into distinct coupling channels (ECG_SNS: channels 1,3,4; ECG_PNS: channel 2) because their synchrony has *opposite* relational valence in the literature (Palumbo 2017: SNS ES=+0.19, PNS ES=-0.21).

### 3.3 Pose Features (41 channels @ 12 Hz)

33 MediaPipe landmarks are grouped into 7 body segments, each yielding interpretable kinematic features:

| Segment | Features | Total |
|---------|----------|-------|
| Head | centroid(3), extent, tilt(3), rotation | 8 |
| Left arm | centroid(3), elbow angle, extension | 5 |
| Right arm | centroid(3), elbow angle, extension | 5 |
| Torso | centroid(3), lateral lean, forward lean, rotation | 6 |
| Left leg | centroid(3), knee angle, extension | 5 |
| Right leg | centroid(3), knee angle, extension | 5 |
| Global | center of mass(3), L-R symmetry, velocity, openness | 6 |

Joint angles computed via arccos of normalized dot products. Activity channel (causal RMS) appended as channel 41.

### 3.4 Respiratory Rate (extracted from ECG)

Respiratory rate is extracted from the Polar H10 ECG signal using spectral peak detection in the 0.1-0.5 Hz band of the R-R interval series. This exploits respiratory sinus arrhythmia (RSA) -- heart rate naturally modulates with breathing.

**Rationale**: No additional hardware is needed (the Polar H10 already records ECG). Error is <2 bpm compared to dedicated respiratory sensors (Schaffarczyk 2022).

---

## 4. Temporal Alignment

All modality streams are aligned to a common time base (`alignment.py`):

1. **Find overlap**: The intersection of all stream time ranges: `[max(all_starts), min(all_ends)]`
2. **Trim**: Each stream is cropped to the overlap window
3. **Zero-reference**: All timestamps shifted so t=0 corresponds to recording start

This produces streams of different lengths (due to different sampling rates) but with a shared time reference. Downstream processing resamples to a common grid as needed.

---

## 5. Coupling Z-Timecourse Construction

This is the core innovation of CADENCE: converting multi-modal feature streams into **7 interpretable coupling z-timecourses** that quantify how strongly two participants are synchronized in each modality at each moment.

### 5.1 Cross-Product Coupling Statistic

For modalities represented as multi-channel feature vectors (EEG, ECG, Pose), the instantaneous coupling is measured via the cross-product:

```
c(t) = (1/D) * sum_d [ p1(t,d) * p2(t,d) ]
```

where `p1`, `p2` are the z-scored feature vectors of participant 1 and 2, and `D` is the number of channels. This captures moment-to-moment co-variation: when both participants' features move in the same direction simultaneously, `c(t)` is positive.

The cross-product is then Gaussian-smoothed (sigma = 3 samples at 2 Hz = 1.5s) to reduce noise.

### 5.2 Surrogate Null Distribution

Raw cross-products are confounded by autocorrelation (correlated signals produce spurious cross-products). CADENCE uses **circular-shift surrogates** to construct a proper null:

```
For k = 1, ..., 200:
    shift_k ~ Uniform[0.1*T, 0.9*T]
    c_surr(k)(t) = (1/D) * sum_d [ p1(t,d) * p2((t + shift_k) mod T, d) ]
```

Circular shifts preserve all temporal statistics (autocorrelation, spectral content, distributional properties) of each participant's signal while destroying the inter-participant coupling. The shift range [10%-90%] avoids trivial near-zero shifts.

The coupling z-score is then:

```
z(t) = [ c(t) - mean(c_surr) ] / std(c_surr)
```

**GPU acceleration**: All 200 surrogate shifts are computed in a single batched operation using `torch.gather` on GPU, achieving ~0.6s for a full 50-minute session.

### 5.3 Wavelet Coherence (for Facial AUs)

Blendshape coupling uses **wavelet coherence** instead of the cross-product, because facial expressions operate across multiple timescales simultaneously:

1. **CWT decomposition**: Morlet wavelets (omega_0 = 5) at 30 log-spaced frequencies from 0.3-8 Hz, applied to all 52 AUs via FFT-based convolution
2. **Low-pass**: 4th-order Butterworth at 8 Hz removes face tracker digitization noise before CWT
3. **Frequency bands**: state (<0.5 Hz tonic expressions), expression (0.5-2 Hz transitions), speech (2-7 Hz articulatory)
4. **Wavelet coherence**: Per-AU cross-spectral coherence between participants, smoothed with Gaussian kernel (sigma = 0.5s)
5. **Stouffer z-combination**: Per-frequency z-scores combined across AU groups via Stouffer's method: `z_combined = sum(z_i) / sqrt(N)`

### 5.4 The 7 Coupling Dimensions

| Dimension | Modality | Method | Physiological Interpretation |
|-----------|----------|--------|------------------------------|
| 0: `eeg` | EEG engagement + phase | Cross-product z | Neural synchrony (attention, shared processing) |
| 1: `bl_expr` | Facial AUs 0.5-2 Hz | Wavelet coherence z | Expression mimicry, emotional contagion |
| 2: `bl_state` | Facial AUs <0.5 Hz | Wavelet coherence z | Tonic affect alignment |
| 3: `ecg_sns` | HR accel + QRS + IBI | Cross-product z | Sympathetic co-activation |
| 4: `ecg_pns` | RMSSD | Cross-product z | Parasympathetic co-regulation |
| 5: `resp` | Respiratory rate | Cross-product z | Breathing synchrony |
| 6: `pose` | Joint kinematics | Cross-product z | Postural mirroring, movement coordination |

All dimensions are output at **2 Hz** on a common time grid (the lowest common rate across modalities).

---

## 6. Spectral Decomposition

The 7D coupling z-timecourses contain both sustained condition-level trends and transient coupling dynamics. CADENCE separates these via spectral decomposition:

### 6.1 Slow/Fast Split

Each z-timecourse is decomposed:

```
z_slow(t) = LPF[z(t)]       (Butterworth 4th order, cutoff 0.01 Hz, zero-phase)
z_fast(t) = z(t) - z_slow(t) (residual)
```

- **z_slow** (<0.01 Hz, period >100s): Captures sustained condition-level state. For example, EEG coupling that is consistently higher during conversation than baseline. This reflects shared behavioral context, not moment-to-moment coupling dynamics.
- **z_fast** (>0.01 Hz): Captures the coupling dynamics of interest -- transient episodes of synchronization and desynchronization within conditions.

**Empirical slow fractions** (from y_06 session):
- EEG: 30% slow (moderate condition dependence)
- BL expression: 22% slow
- ECG SNS/PNS: 7-8% slow (mostly fast dynamics)
- Respiratory: 62% slow (strong condition-driven breathing patterns)
- Pose: 47% slow (posture is condition-dependent)

### 6.2 Shared State Extraction via PCA

The 7 z_slow traces are stacked into a (T x 7) matrix and reduced via PCA to 2 principal components:

```
z_slow_pcs = PCA(z_slow_matrix, n_components=2)   ->   (T, 2)
```

These 2 PCs serve as **transition covariates** for the RSLDS: they summarize the shared behavioral state (which condition, general engagement level) that modulates *how* the dyad transitions between coupling states.

**Typical variance explained**: PC1 ~ 36%, PC2 ~ 29% (65% total). The slow PCA loadings are heterogeneous across modalities -- they do NOT simply track a single condition variable.

---

## 7. The RSLDS Model Hierarchy

CADENCE fits a hierarchy of increasingly expressive models. Each level subsumes the previous:

### 7.1 Level 1: IOHMM (Input-Output Hidden Markov Model)

The base model. K discrete coupling states with input-driven transitions:

**Emission model** (what each state looks like):
```
y_t | z_t = k  ~  N(mu_k, diag(sigma^2_k))
```
Each state k has a characteristic mean coupling profile `mu_k` across the 7 modalities. For example, State 0 might have high EEG + BL coupling but low ECG, while State 1 has high ECG + Pose coupling.

**Transition model** (how states switch):
```
P(z_t = k | z_{t-1} = j, u_t) = softmax_k( W_jk + S_jk^T u_t )
```
- `W_jk`: Base transition logits (K x K matrix). Diagonal entries encode state persistence (stickiness).
- `S_jk`: Input modulation (K x K x 2 tensor). The z_slow PCs `u_t` modulate transition probabilities -- for example, during conversation (high PC1), transitions to the "engaged coupling" state become more likely.

**Fitting**: Expectation-Maximization with:
- E-step: Forward-backward algorithm (vectorized, log-space)
- M-step: Closed-form for emissions, L-BFGS-B for transition logits
- Multiple random restarts (best by log-likelihood)
- Annealing: Temperature > 1 on transitions for first 20 iterations to avoid local optima

**Observation masking**: Sessions with missing modalities (e.g., no ECG in y24) have those dimensions masked -- they contribute 0 to the emission log-likelihood. This allows all sessions to be fit with the same K-state model.

### 7.2 Level 2: SLDS (Switching Linear Dynamical System)

Adds a **D-dimensional continuous latent** `x_t` that captures within-state dynamics (autocorrelation, cross-modal lead-lag structure):

```
x_t | z_t = k, x_{t-1}  ~  N(A_k x_{t-1} + b_k, Q_k)     [latent dynamics]
y_t | z_t = k, x_t       ~  N(C_k x_t + d_k, diag(R_k))    [emissions]
```

- `A_k` (D x D): Per-state dynamics matrix. Encodes how the latent evolves within each coupling state. Eigenvalues < 1 ensure stability.
- `C_k` (m x D): Emission matrix mapping latent to observations.
- `d_k` (m,): Between-state mean offsets (what makes states different -- analogous to mu_k in IOHMM).

**Why a continuous latent?** The z_fast timecourses have lag-1 autocorrelation rho ~ 0.93. A naive fix (AR(1) on observations) was attempted and **rejected**: with phi = 0.95, the innovation `y_t - mu_k - phi*(y_{t-1} - mu_k)` shrinks between-state mean differences by factor (1-phi) = 0.05, collapsing states to ARI = 0.001. The continuous latent correctly separates autocorrelation modeling (in A_k) from state identity (in d_k).

**Inference**: Structured Mean-Field (SMF) variational EM:
1. Fix q(x), update q(z) via forward-backward using expected emission log-likelihoods
2. Fix q(z), update q(x) via gamma-weighted Kalman filter-smoother
3. Repeat 3 times per EM iteration

**Staged fitting**: To prevent the latent from absorbing between-state signal (the d-C tradeoff), CADENCE uses staged optimization:
- Stage 1 (0-70% of EM): d frozen to IOHMM warm-start values, C initialized near zero
- Stage 2 (70-100%): d unfrozen, full parameter refinement

**Configuration**: D_latent = 3, K = 3-4, 3 inner SMF iterations, 3-5 random restarts.

### 7.3 Level 3: Factor-Analyzed Emissions (Phase 3.2)

Replaces diagonal emission noise with **low-rank + diagonal** structure to capture cross-modal noise correlations:

```
y_t | z_t = k, x_t  ~  N(C_k x_t + d_k,  F_k F_k^T + diag(R_k))
```

- `F_k` (m x r): Factor loading matrix, rank r = 2. Captures shared noise structure across modalities that isn't explained by the latent dynamics -- for example, correlated measurement artifacts or shared physiological arousal that affects multiple channels simultaneously.

**M-step**: After estimating C, d via regression, the residual covariance is factor-analyzed using PPCA-style eigendecomposition:
1. Compute gamma-weighted residual covariance: `Sigma_resid = E[(y - Cx - d)(y - Cx - d)^T]`
2. Eigendecompose, take top r eigenvectors scaled by sqrt(eigenvalue - noise_floor) as F
3. Remaining diagonal as R

**Initialization**: F initialized to small random values (std = 0.05), NOT from PCA of raw residuals (which confounds the C@x component with factor noise). Factor analysis is delayed until Stage 2 of fitting.

**Emission LL computation**: The expected log-likelihood under q(x) with full noise covariance is:
```
E_q[log p(y|z=k, x)] = -m/2 log(2pi) - 1/2 log|Sigma_k| - 1/2 [r^T Sigma_k^{-1} r + tr(Sigma_k^{-1} C P C^T)]
```
where `Sigma_k = F_k F_k^T + diag(R_k)` and `r = y - C_k x_smooth - d_k`. Vectorized over T using batch `np.linalg.slogdet` and `np.einsum`.

### 7.4 Level 4: Recurrent Transitions / rSLDS (Phase 3.3)

Adds feedback from the continuous latent to the discrete state transitions:

```
P(z_t = k | z_{t-1} = j, u_t, x_{t-1}) = softmax_k( W_jk + S_jk^T u_t + R_jk^T x_{t-1} )
```

- `R_jk` (K x K x D): Recurrence weights. The continuous latent state `x_{t-1}` directly influences which discrete state the model transitions to. This creates a feedback loop: x evolves according to A[z], and x in turn influences the next z.

**Interpretation**: If the latent trajectory moves toward a particular region of the state space (e.g., high EEG + low ECG coupling dynamics), this can trigger a transition to the state that best describes that regime.

**Regularization**: Stronger L2 penalty on R_recur (lambda = 0.05) compared to S (lambda = 0.01) to prevent the recurrence from overfitting to noise in x_smooth.

### 7.5 Level 5: Hierarchical Pooling (Phase 3.4)

Fits a **shared dynamics model across all sessions** while allowing session-specific emission parameters:

**Shared across sessions** (pooled during M-step):
- `A_k, b_k, Q_k`: Per-state latent dynamics -- *how* coupling states evolve is universal
- `W, S`: Transition structure -- *when* states switch follows the same rules
- `R_recur`: Recurrence weights (if enabled)

**Session-specific** (estimated per session):
- `C_n, d_n, R_n, F_n`: How coupling states manifest in the observations varies per session due to sensor placement, individual physiology, and baseline coupling levels

**Why hierarchical?** Per-session fitting suffers from two problems:
1. **State non-identifiability**: State 0 in session A might correspond to State 2 in session B. There is no way to compare states across sessions.
2. **Small-sample estimation**: Each session has T ~ 6000 samples. Pooling N sessions gives N*T ~ 72,000 samples for shared parameter estimation, dramatically reducing variance.

**Implementation**:
1. **Per-session initialization**: Fit per-session SLDS (parallelized via joblib)
2. **State alignment**: Hungarian matching on emission means d_n against a reference session. This permutes each session's state labels to match the reference.
3. **Pooling**: Average aligned parameters to initialize shared params
4. **Hierarchical EM**: Alternate between per-session E-steps and pooled M-steps for shared parameters

**Dynamics M-step (pooled)**: Sufficient statistics (weighted second moments of x_smooth) are accumulated across all sessions before solving the regression. This gives a single set of A_k, b_k, Q_k that best explains the dynamics across all sessions.

**Transition M-step (pooled)**: L-BFGS-B optimizes the pooled log-likelihood across all sessions' xi (pairwise state posteriors) simultaneously.

**Convergence**: Hierarchical EM converges much faster than per-session fitting (typically 15-20 iterations vs 80-100) because pooling provides much stronger constraints on the shared parameters.

**BIC**: `BIC = -2 * sum_n(LL_n) + (n_shared + N * n_per_session) * log(sum_n(T_n))`

---

## 8. Model Selection and Outputs

### 8.1 Model Selection

BIC (Bayesian Information Criterion) guides model selection at each level:

| Comparison | Decision Rule |
|-----------|--------------|
| K (number of states) | Sweep K = 2, 3, 4, 5; select lowest BIC |
| IOHMM vs SLDS | SLDS if BIC improves by > 2% |
| Diagonal vs Factor-analyzed | FA if BIC improves |
| Per-session vs Hierarchical | Hierarchical if total BIC < sum of per-session BICs |

### 8.2 Primary Outputs

**Per-session:**
- **State posteriors** gamma(t, k): Probability of being in each coupling state at each timepoint
- **Viterbi path**: Most likely state sequence
- **Latent trajectory** x(t): Continuous within-state dynamics
- **Flexibility metrics**:
  - Transition rate (Hz): How frequently coupling states switch
  - Shannon entropy: Evenness of state usage
  - Mean dwell time per state: Average duration in each state
  - State usage proportions: Fraction of session in each state

**Cross-session (hierarchical):**
- **Shared dynamics** A_k: Universal coupling state evolution rules
- **Shared transitions** W, S: Universal state-switching patterns
- **Per-session emission profiles** d_n: How each session's coupling states manifest
- **Cross-session state correspondence**: States are aligned and comparable

### 8.3 Interpretation Framework

Each discrete state represents a **coupling regime** -- a characteristic pattern of multi-modal interpersonal synchrony. For example, a 3-state model might identify:

- **State 0** (high EEG + BL, low ECG): "Engaged exchange" -- neural and facial synchrony without autonomic stress
- **State 1** (high ECG SNS, low everything else): "Sympathetic co-activation" -- shared arousal without behavioral coupling
- **State 2** (low everything): "Uncoupled" -- independent activity

The transition dynamics reveal *when* and *why* the dyad shifts between these regimes. The z_slow PCs as transition covariates link these shifts to broader session context (conversation topics, meditation phases).

The continuous latent x(t) captures *how* coupling evolves within a state -- for example, a gradual build-up of EEG synchrony within an "engaged exchange" episode, or an oscillation between high and moderate facial mimicry within a conversation block.

---

## 9. Validation

### 9.1 Synthetic Recovery

Data is generated from known model parameters and fit to verify parameter recovery:

- **IOHMM**: ARI = 0.959, k-means baseline = 0.632 (validates temporal model adds value)
- **SLDS variants**: Full model ARI = 0.638, BIC 31k vs IOHMM BIC 36k
- **Hierarchical**: ARI = 0.804, BIC 97,969 vs per-session sum 99,373

All tests use realistic difficulty calibration (mean_scale = 0.5, k-means ARI < 0.7) to ensure the model is tested on non-trivial data.

### 9.2 Semi-Synthetic Coupling Detection

Pseudo-dyad z-timecourses (P1 from session A + P2 from session B) serve as null-coupling base. Synthetic coupling episodes (kappa = 0.0 to 0.4) are injected at known times. AUC measures detection performance:

- Null (kappa = 0): AUC = 0.500 (correct null for all models)
- kappa = 0.4: AUC = 0.54-0.57 (models detect elevated coupling)

The modest AUC reflects the nature of the task: these models characterize *state dynamics*, not detect individual coupling episodes. The value is in the flexibility metrics and state profiles, not binary event detection.

### 9.3 Pseudo-Dyad Null Control

Real dyads show 4x more state transitions than pseudo-dyads (d = +2.52), confirming that the detected state dynamics reflect genuine interpersonal coupling rather than individual-level dynamics.

---

## 10. Computational Pipeline

```
XDF File
  |
  v
[xdf_loader] -----> Raw streams (EEG 256Hz, ECG 130Hz, BL 30Hz, Pose 30Hz)
  |
  v
[preprocessors] ---> Filtered, z-scored, artifact-rejected streams
  |
  v
[feature extraction] -> Per-modality features (EEG 8ch@2Hz, ECG 7ch@2Hz,
  |                      BL 52ch@30Hz, Pose 41ch@12Hz)
  v
[alignment] --------> Common time reference
  |
  v
[rslds_scaffold] ---> 7D coupling z-timecourses @ 2Hz
  |                    (200 circular-shift surrogates, GPU-accelerated)
  v
[spectral decompose] -> z_fast (7D observations) + z_slow_pcs (2D covariates)
  |
  v
[fit_hierarchical_slds] -> Shared dynamics + per-session emissions
  |                         (IOHMM warm-start -> SLDS -> FA -> rSLDS)
  v
[outputs] ----------> State posteriors, flexibility metrics,
                       latent trajectories, cross-session comparison
```

**Runtime** (12 sessions, 50 min each):
- Scaffold (per session): ~30s (GPU-accelerated surrogates)
- Per-session SLDS fit: ~60s (3 restarts, 100 EM iterations)
- Hierarchical EM: ~120s (17 iterations, pooling across 12 sessions)
- Total pipeline: ~15 minutes

---

## 11. V8 Preprocessing and Temporal Localization (2026-03-29)

### 11.1 V8 Preprocessing Fixes

Critical review revealed V7 z-timecourses had autocorrelation rho=0.95 (from Gaussian smoothing) and EEG variance dominance (39.8% between-state R²). V8 fixes:

- **EEG**: `smooth_samples=0` removes Gaussian smoothing (rho 0.95 → 0.26), per-band z-scores (theta/alpha/beta) replace Stouffer combined
- **BL**: Per-segment wavelet coherence (not full-session diluted surrogates)
- **ECG**: Hilbert envelope cross-product on bandpass-filtered IBI (LF=SNS, HF=PNS) replaces pre-computed HRV features
- **Resp**: Phase coherence cos(phi1-phi2) from EDR
- **Pose**: Upper-body velocity (first-difference) cross-product z
- **Iterative prewhitening**: Only channels with rho > 0.3, up to 3 rounds
- **Variance standardization**: All channels to mean=0, std=1

**Result**: All 9 channels rho < 0.3 after prewhitening. Between-state R² distributed across EEG, BL, ECG (not EEG-only).

### 11.2 Observation Space (9D)

| Dim | Modality | Method | Prewhitened? |
|-----|----------|--------|:---:|
| 0 | EEG theta | Per-band surrogate z cross-product (smooth=0) | No (rho=0.26) |
| 1 | EEG alpha | Per-band surrogate z cross-product (smooth=0) | Yes (rho 0.31→-0.04) |
| 2 | EEG beta | Per-band surrogate z cross-product (smooth=0) | No (rho=0.26) |
| 3 | BL expression | Per-segment wavelet coherence z (0.5-2 Hz) | Yes (rho 0.86→0.18) |
| 4 | BL state | Per-segment wavelet coherence z (<0.5 Hz) | Yes (rho 0.95→0.27) |
| 5 | ECG LF (SNS) | Hilbert envelope cross-product (0.04-0.15 Hz) | Yes (rho 0.94→0.08) |
| 6 | ECG HF (PNS) | Hilbert envelope cross-product (0.15-0.4 Hz) | Yes (rho 0.83→0.08) |
| 7 | Resp | Phase coherence cos(phi1-phi2) (0.1-0.5 Hz) | Yes (rho 0.99→-0.23) |
| 8 | Pose | Upper-body velocity cross-product z (11 features) | No (rho=0.05) |

### 11.3 Temporal Localization

Standard IOHMM/rSLDS on V8 data found condition-differentiated states (p=0.0000) but switched every ~3s (1,015 transitions / 3,094s) — observation-level clusters, not temporal regimes.

**Approaches tested** (on y_06):

| Method | Transitions | Mean Dwell | cond p | Notes |
|--------|------------|-----------|--------|-------|
| Baseline IOHMM | 1,011 | 3.1s | 0.0000 | Rapid flickering |
| Sticky k=5 | 369 | 8.4s | 0.0000 | Modest improvement |
| Constrained Viterbi min=20 (10s) | 164 | 18.8s | 0.0000 | **Best practical balance** |
| Constrained Viterbi min=40 (20s) | 82 | 37.3s | 0.0000 | Strongest localization |
| Block aggregation 20 | 98 | 31.2s | 0.0000 | Implicit min dwell |
| TICC b=500 | 38 | 79.3s | 0.0000 | Cluster imbalance (87% in one) |

**Constrained Viterbi min_dwell=20 (10s)** selected as production decoder.

### 11.4 Hierarchical rSLDS on V8 (8 sessions)

Hierarchical rSLDS (K=4, D_latent=3, FA rank 2, recurrent, sticky k=3) with constrained Viterbi (min_dwell=10s):

- **BIC**: 18,515 (vs per-session IOHMM ~90,000 per session)
- **Mean dwell times**: 47-127s across sessions (genuinely localized regimes)
- **Cross-session condition usage** (mean +/- std across 8 sessions):
  - Conversation: S1 dominant (41%+/-33)
  - Baseline: S1 (42%+/-41)
  - Meditation: S0 (37%+/-34)
  - Inter-condition gaps: S2 (30%+/-31), S3 (31%+/-24), S1 reduced (8%+/-13)
- **Gap detection**: Inter-condition gaps show reduced S1 usage (8%) vs conditions (31-42%), partially consistent across sessions
- **Cross-session variance**: High (+/-30-40%) — states capture session-specific variance alongside condition-universal patterns

### 11.5 Validation Results

| Test | Result | Key Metric |
|------|--------|-----------|
| AR(1) null | **PASS** | Real BIC=89,812 vs Null BIC=158,578 |
| Pseudo-dyad contrast | **PASS** | Real: 17% fewer transitions, lower entropy |
| Condition alignment | **PASS** | p=0.0000 (permutation test) |
| Semi-synthetic AUC | **FAIL** | kappa=0.4 AUC=0.526 (expected — model characterizes regimes, not detects 5s episodes) |

### 11.6 Important Caveat: Gap State is a Data Availability Artifact

The "gap detection" finding (a state dominating inter-condition periods) is **not** detecting a coupling regime change. During inter-condition gaps, both participants look down to fill surveys — the face tracker returns zeros and pose visibility drops. This creates a distinctive observation pattern (near-zero BL expression, BL state, and pose channels) that the model learns as a separate state.

**Implication**: The gap-dominant state (S1 in y_06) is driven by **missing face/pose data**, not by absence of interpersonal coupling. The V8 obs_mask only flags entirely-missing modalities (whole session), not per-timepoint dropouts. A proper fix would require:
- Per-timepoint face detection validity masking in the scaffold
- Observation masking in the IOHMM likelihood for timepoints where face/pose is unavailable
- Or: excluding inter-condition gaps from the analysis entirely

The condition differentiation (p=0.0000) is partly inflated by this artifact. The meaningful comparison is between conversation, baseline, and meditation — all conditions where face+pose data is available.

### 11.7 State Interpretation (y_06, K=4, per-timepoint masking + constrained Viterbi 10s)

With per-timepoint obs_mask for BL/pose validity and constrained Viterbi (min_dwell=20 = 10s), the K=4 IOHMM on y_06 produces four interpretable coupling states:

| State | Usage | Profile | Condition Mapping |
|-------|-------|---------|-------------------|
| **S3 "Baseline"** | 41% | Near-zero across all modalities | Dominant everywhere; default uncoupled state |
| **S0 "Moderate"** | 36% | Mild BL state (+0.06), ECG HF (+0.07) | Elevated in baseline (56%) and meditation (44%) |
| **S2 "BL disengaged"** | 19% | Strongly negative bl_expr (-0.21) | 80% in inter-condition gaps; facial anti-coupling |
| **S1 "Deep coupling"** | 3% | EEG theta/alpha/beta all +0.3-0.6, BL expr +0.3, ECG HF -0.3 | **10% of conversation, 0% baseline, 1% meditation** |

**Key finding**: S1 (deep coupling) is a rare (~90s per session) multi-modal burst state — high neural synchrony + facial engagement + parasympathetic withdrawal — that occurs **exclusively during conversation**. The model discovers this without being told which condition is which. This matches the "Deep engagement" state from the original RSLDS vision.

**Per-timepoint masking effect**: 176 timepoints (2.8% = ~88s) are masked for BL/pose during inter-condition survey gaps where face detection drops out. S2 now captures genuine low-BL-coupling periods (valid face data, negative coupling) rather than a zero-data artifact.

### 11.8 PE Protocol Discovery and Cross-Session Analysis (2026-03-29)

The V8 scaffold originally only recognized meditation protocol conditions (base_EO, base_EC, conv_1, meditate_B, meditate_K, conv_2). Marker analysis of all XDF files revealed two distinct protocols across the dataset:

**Meditation protocol** (6 sessions: y_06, y_17, y_19, y_11, y_04, y_24):
- base_EO → base_EC → conv_1 → meditate_B → meditate_K → conv_2

**Psychoeducation (PE) protocol** (5 sessions: y_01, y_05, y_10, y_32, y_41):
- base_EO → base_EC → conv_1 → PE_1 → PE_2 → conv_2
- PE = therapist presents educational material; control for meditation (passive listening, no active coupling expected)

Sessions y_01, y_05, and y_32 previously had 29-38 minute "gaps" that were actually unlabeled PE blocks. Adding PE_1/PE_2/PE markers to CONDITION_ORDER resolved this entirely.

**Additional marker findings:**
- `baseline` marker (single block, no EO/EC split) used in y_04
- Case-insensitive cache matching needed (Y_10 vs y_10)
- y_03 has negative-duration marker (conv_1 start > stop) — session excluded
- y_04 has broken conv_1 (start but no stop) — only meditate_K usable

### 11.9 Null-State rSLDS (K=4, null_state=True)

Added principled null-state constraint to the IOHMM/rSLDS:
- **State 0 (null)**: mu[0]/d_emit[0] = 0 exactly (fixed, never updated in M-step)
- **Sigma2[0]**: learned freely (cap=5.0, effectively unconstrained)
- **Asymmetric transitions**: leaving null is costly (W[0,k]=-1.0), returning is easy (W[k,0]=+1.5)
- **Transition prior regularization**: L2 penalty toward asymmetric W structure (lambda=0.1)
- **States 1-3**: unconstrained coupling states, free to specialize by modality

**Rationale**: Z-score input space means z=0 is chance-level coupling. The null state captures "nothing special happening." Coupling states must have genuine signal to pull timepoints away from null.

### 11.10 Cross-Session Results (11 sessions, null-state IOHMM + constrained Viterbi 10s)

**Per-session state interpretation (y_06, the cleanest session):**

| State | Usage | Profile | Condition Mapping |
|-------|-------|---------|-------------------|
| S0 null | 38% | mu=0 exactly | base_EO 63%, base_EC 47%, conv 38%, med 47% |
| S1 mild | 40% | bl_expr +0.05 | conv 51%, med 52%, base 45% |
| S2 coupling | 3% | EEG +0.37-0.56, BL +0.17 | **conv_1 10%, conv_2 10%, base 0%, med 1.5%** |
| S3 BL_disengaged | 19% | bl_expr -0.21 | **gaps 68-95%, conditions 0%** |

**PE protocol validation (y_41, cleanest PE session):**
- conv_1: MILD=56%, NULL=44% — active dialogue, moderate coupling
- PE_1: **NULL=85%**, MILD=13% — passive listening, minimal coupling
- PE_2: **NULL=81%**, MILD=19% — same pattern
- conv_2: MILD=81%, NULL=4% — back to active coupling

This matches the expected coupling hierarchy: conversation > PE > baseline.

**Aggregate condition-type analysis (11 sessions, 109 condition segments):**

| Condition Type | n | Coupling State | Null State | Mean Duration |
|---------------|---|---------------|-----------|---------------|
| Conversation | 16 | 21% +/- 26% | 24% +/- 24% | 312s |
| Psychoeducation | 7 | 22% +/- 28% | **40% +/- 30%** | 600s |
| Meditation | 10 | 15% +/- 25% | **35% +/- 27%** | 686s |
| Baseline | 16 | 23% +/- 22% | **37% +/- 30%** | 162s |
| Gaps (surveys) | 60 | 30% +/- 41% | 10% +/- 23% | 188s |

**Key pattern**: Null state usage is lowest in conversation (24%) and highest in PE/meditation/baseline (35-40%). The coupling state shows high cross-session variance (+/-26%) but the null-state gradient is consistent with the expected coupling hierarchy.

**BL_disengaged state**: Consistently identifies inter-condition gaps across all well-marked sessions (68-99% of gap time), validating the per-timepoint face/pose observation masking.

### 11.11 Limitations and Honest Assessment

**What works within individual sessions:**
- y_06: Coupling state (3%) appears exclusively in conversation (10%), never in baseline/meditation
- y_41: Null state correctly dominates PE (81-85%) vs conversation (4-44%)
- BL_disengaged state reliably identifies survey/transition gaps across all sessions

**What doesn't work cross-session:**
- Coupling state identity varies across sessions (state permutation despite Hungarian alignment)
- Wilcoxon conv vs base: p=0.25-1.0 (not significant across sessions)
- Sessions without strong facial coupling (BL_sig < 0.5) show no condition differentiation
- Cohen's d for "deep coupling" state ranges from 0.05 (noise) to 0.33 (genuine) across sessions

**Root causes:**
1. **Facial expression coupling is the key discriminator** — sessions with BL_sig > 0.6 (y_06, y_17) show clean separation; others don't
2. **EEG coupling z-scores don't reliably differ between conditions** — mean |z| is 0.7-0.95 across ALL conditions including baselines
3. **Per-session IOHMM fitting**: without shared transitions, "coupling state" means different things in different sessions
4. **Missing EDA modality**: literature's strongest coupling signal (r=0.32-0.47) not yet captured

### 11.12 V8.2: Revised Observation Vector (15D) — Production Version

Critical analysis (2026-03-30) revealed the V8 cross-product EEG features fail to separate conditions (all p>0.29). Root cause: the volt_amp cross-product measures amplitude co-fluctuation, which is NOT condition-dependent. Two new feature types capture condition-dependent information:

**Feature set revision:**
- **Dropped**: EEG volt_amp cross-product (3ch) — does not separate any condition pair
- **Dropped**: BL state (<0.5 Hz wavelet coherence) — never showed signal
- **Added**: EEG imaginary coherence (3ch) — phase coupling, rejects amplitude confounds (alpha p=0.016, beta p=0.047)
- **Added**: EEG concordance (3ch) — shared neural state level (z_P1+z_P2)/2 (alpha p=0.031, beta p=0.031)
- **Added**: EEG concordance dynamics (3ch) — EWMAD (Exponentially-Weighted Mean Absolute Deviation) of concordance, captures turn-taking rate (alpha p=0.016)
- **Added**: BL activity concordance (1ch) — shared facial activity (p=0.023)
- **Kept**: ECG LF/HF, Resp, Pose (4ch)

**V8.2 15D Observation Space:**

| Ch | Feature | Type | What it captures | p (conv vs base) |
|----|---------|------|------------------|:-:|
| 0-2 | ImCoh theta/alpha/beta | Phase coupling | Inter-brain oscillation coordination | 0.69/0.016/0.047 |
| 3-5 | Conc theta/alpha/beta | Shared state | Both participants in similar power state | 0.047/0.031/0.031 |
| 6-8 | Dyn theta/alpha/beta | Dynamics | Rate of change of shared state (EWMAD (Exponentially-Weighted Mean Absolute Deviation)) | 0.30/0.016/0.30 |
| 9 | BL expression | Coupling | Per-segment wavelet coherence z | existing |
| 10 | BL activity concordance | Shared state | Both facially active simultaneously | 0.023 |
| 11-12 | ECG LF/HF | Coupling | Hilbert envelope cross-product | existing |
| 13 | Resp phase coherence | Phase coupling | cos(phi1-phi2) | 0.023 |
| 14 | Pose velocity | Coupling | Upper-body cross-product z | existing |

**Key methodological findings:**

1. **Concordance vs coupling**: The cross-product (co-fluctuation) and concordance (shared level) are orthogonal signals. During eyes-closed baseline, both participants increase alpha power (high concordance) but their fluctuations are independent (zero coupling). During conversation, concordance averages near zero (asymmetric: one talks, one listens) while coupling varies.

2. **Imaginary coherence vs cross-product**: ImCoh measures phase consistency in 2s Welch windows. Unlike the cross-product, it separates conditions because phase locking is higher during stable baseline than dynamic conversation. ImCoh is NOT equivalent to smoothed cross-product — head-to-head comparison confirmed cross-product fails at all smoothing levels (p>0.22) while ImCoh succeeds (p=0.016).

3. **Dynamics (EWMAD (Exponentially-Weighted Mean Absolute Deviation))**: Captures the rate of change of concordance via exponentially-weighted mean absolute deviation (tau=3s). Log-transformed for Gaussian-like distribution. At 2 Hz, this measures sub-second volatility rather than 10-20s turn-taking dynamics.

4. **Envelope correlation was tested and FAILED** (all p>0.58). Amplitude co-fluctuation is not condition-dependent in our data, regardless of the metric used (cross-product, envelope correlation).

**Hierarchical rSLDS state profiles (15D, 12 sessions):**

| State | ImCoh | Concordance | Dynamics | BL | Interpretation |
|-------|:---:|:---:|:---:|:---:|:---|
| S0 NULL | 0 | 0 | 0 | 0 | Baseline / no coupling |
| S1 COUP | +0.01 | +0.15 | +0.15 | +0.15 | Moderate concordance + BL expression |
| S2 OTHER | -0.05/+0.06 | ~0 | **+0.45/+0.29** | -0.06 | High dynamics, zero concordance (turn-taking) |
| S3 SHARED | -0.02 | **+0.49** | **+0.46** | -0.07 | High concordance + high dynamics (active shared state) |

**Cross-session condition mapping** (12 sessions, hierarchical rSLDS):
- Conversation: S0=22%, S1=28%, S2=28%, S3=22% (+/-29-33%)
- PE: S0=16%, S1=50%, S2=29%, S3=5%
- Meditation: S0=12%, S1=14%, S2=52%, S3=22%
- Baseline: S0=23%, S1=31%, S2=30%, S3=17%
- Cross-session variance: +/-29-42% (high — session-specific patterns dominate)

**Scripts:**
- `scripts/_run_scaffold_v82.py` — 15D scaffold (ImCoh + concordance + dynamics + BL/ECG/Resp/Pose)
- `scripts/_run_v82_hierarchical.py` — hierarchical rSLDS on 15D
- `scripts/_run_v82_rslds_analysis.py` — per-session rSLDS on 15D
- `scripts/_test_eeg_metrics_h2h.py` — head-to-head metric comparison

### 11.13 Current Status and Remaining Gaps

**What works (V8.2):**
- 15D observation vector with empirically-validated features (8/15 significantly separate conditions)
- Three orthogonal EEG feature types: phase coupling (ImCoh), shared state (concordance), dynamics (EWMAD (Exponentially-Weighted Mean Absolute Deviation))
- Per-timepoint face/pose observation masking
- Null-state rSLDS constraint (d_emit[0]=0, asymmetric transitions)
- Constrained Viterbi (10s min dwell) produces temporally localized states
- Two session protocols identified: meditation (6 sessions) + PE/psychoeducation (5 sessions)
- State profiles are more interpretable with dynamics channels (S2 captures "high dynamics, zero concordance" turn-taking pattern)

**What doesn't work:**
- Cross-session state-condition mapping has +/-30% variance — states don't consistently map to the same conditions across sessions
- The null state doesn't consistently dominate baseline as expected (conversation features average near zero after standardization, making conversation look "null-like")
- EEG coupling (amplitude co-fluctuation) does not differentiate conditions by any metric tested

**Remaining gaps:**
1. **Ground truth validation** — the single most important next step. Need controlled coupling ON/OFF blocks.
2. **EDA modality** — literature's strongest coupling signal (r=0.32-0.47), not yet in hardware
3. **Cross-session state alignment** — hierarchical model helps but variance remains high
4. **Drug-state covariates** — LZc/alpha power as transition modulators not yet implemented
5. **Outcome prediction** — no link yet between coupling flexibility and therapy outcomes
6. **Turn-taking dynamics timescale** — EWMAD (Exponentially-Weighted Mean Absolute Deviation) captures sub-second volatility, which is consistently elevated during conversation (high turn-taking activity) and low during baseline (stable). The rSLDS with constrained Viterbi (10s min dwell) groups these into temporally coherent state blocks — the model handles the temporal integration, not the feature.

### 11.14 Burst Analysis Layer (2026-03-30)

Continuous coupling intensity + burst event detection on top of V8.2 state assignments. Full synthesis: `docs/rslds_burst_analysis_synthesis.md`.

**Methodology:**
- Continuous per-modality-group RMS intensity (3s smoothing) replaces categorical state labels for fine-grained dynamics
- Burst events: 90th percentile threshold, 2s min duration, 3s merge gap
- Per-segment stratification (8 segments: base_EO, base_EC, conv_1, conv_2, meditate_B, meditate_K, PE_1, PE_2) — never pool conditions
- Therapist/patient decomposition: z_therapist = concordance + asymmetry/2, z_patient = concordance - asymmetry/2
- Domain-axis visualization replaces PCA for state-center quiver plots: Phase Coupling = mean(ImCoh), Shared Power = mean(Concordance), Body Coupling = mean(BL, Pose), Autonomic = mean(ECG, Resp)

**Validated findings (n=12 sessions, cross-session error bars):**
1. Therapist/patient asymmetry reverses by condition: patient drives conversation bursts (theta -0.213±0.036), therapist drives meditation (+0.367) and PE_1 (+0.679). PE_2 collapses to null.
2. Burst rates: conversation ~2/min Face+Body; meditation suppresses all except EEG phase (~1/min); base_EC 2.5× EEG phase vs base_EO.
3. Protocol validation: patient alpha meditation > PE (+0.236, p<0.05); patient theta conversation >> meditation (p<0.01); meditate_B more alpha than meditate_K.
4. Cross-modal lead/lag does NOT validate at n=12 (all within permutation null 95% CI).

**Scripts:** `_plot_rslds_quiver.py`, `_plot_rslds_excursions.py`, `_plot_rslds_bursts.py`, `_validate_rslds_bursts.py`, `_validate_bursts_by_condition.py`
**Output:** `results/rslds/quiver_plots/` (39 figures + 2 JSON)
