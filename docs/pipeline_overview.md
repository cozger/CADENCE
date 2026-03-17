# CADENCE Pipeline: From Raw Data to Final Results

**Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation**

This document describes every step of the CADENCE analysis pipeline, from raw XDF recordings through to statistical inference and visualization of interpersonal coupling.

---

## Architecture Overview

CADENCE is a three-layer distributed lag regression framework that detects time-varying, directed, cross-modal coupling between two participants in a social interaction. Unlike black-box neural approaches (e.g., MCCT), every quantity in CADENCE is either a measured signal or a regression weight -- fully interpretable.

```
Raw XDF Recording
    |
    v
[1. Data Ingestion] -----> Per-modality streams (variable rates)
    |
    v
[2. Preprocessing] ------> Cleaned, z-scored signals with validity masks
    |
    v
[3. Feature Extraction] -> Compact feature representations (EEG 8ch, ECG 6ch, Pose 41ch, BL 53ch)
    |
    v
[4. Temporal Alignment] -> All streams trimmed to common time range
    |
    v
[5. Basis Construction] -> Raised cosine lag kernels (per-modality rates)
    |
    v
[6. Design Matrix] ------> X = [basis-convolved source | AR target terms]
    |
    v
[7. Static Screening] ---> Ridge regression dR2 + surrogate significance
    |
    v
[8. Time-Varying EWLS] --> Forward-backward exponentially weighted regression
    |
    v
[9. Significance] -------> Surrogate p-values + EWLS consistency check
    |
    v
[10. Feature Decomposition] -> Within significant pathways: per-feature dR2
    |
    v
[11. Visualization] ------> Timecourses, coupling matrices, kernels, comparisons
```

---

## Step 1: Data Ingestion

**Files**: `cadence/data/xdf_loader.py`, `cadence/data/alignment.py`

### 1.1 XDF File Loading

Raw data comes from Lab Streaming Layer (LSL) recordings in XDF format, captured by YouQuantiPy (YQP) LabRecorder. Each recording contains two participants with four modality streams each:

| Stream | Channels | Native Rate | Content |
|--------|----------|-------------|---------|
| EEG (Emotiv EPOC) | 19 (use cols 3-16 = 14 channels) | 256 Hz | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 |
| ECG | 1 | 130 Hz | Single-lead cardiac signal |
| Face Landmarks | 1489 | ~30 Hz | ARKit blendshape coefficients (use cols 0-51 = 52 channels) |
| Body Pose | 132 = 33 landmarks x 4 (x,y,z,vis) | ~12-14 Hz actual | MediaPipe pose landmarks |

### 1.2 Session Cache

Preprocessed sessions are cached as `.npz` (arrays) + `.json` (metadata) files. The cache version is `v7`. On load, EEG features are auto-upgraded if the cached shape doesn't match the current `EEG_N_FEATURES` (8 channels: 7 base + 1 activity).

**Cache location**: `C:/Users/optilab/desktop/MCCT/session_cache` (shared with MCCT project).

---

## Step 2: Preprocessing

**File**: `cadence/data/preprocessors.py`

Each modality undergoes modality-specific preprocessing. The goal is to produce clean, z-scored signals with per-sample validity masks.

### 2.1 EEG Preprocessing

```
Raw (N, 19) @ 256 Hz
  -> Extract channels 3-16                    -> (N, 14)
  -> Notch filter at 50 Hz and 60 Hz (Q=30)  -> Remove powerline interference
  -> Bandpass 1-45 Hz (4th-order Butterworth) -> Isolate neural band
  -> Per-channel z-score (valid samples only) -> Mean=0, std=1
  -> Clip to [-10, 10]                        -> Prevent extreme outliers
  -> Validity: per-channel boolean mask
```

### 2.2 ECG Preprocessing

```
Raw (N, 1) @ irregular ~130 Hz
  -> Interpolate to uniform 130 Hz grid       -> Regular time series
  -> Detect gaps (dt > 3x sample period)      -> Mark as invalid
  -> Bandpass 0.5-40 Hz (4th-order Butter.)   -> Cardiac frequency range
  -> Z-score (valid samples only)             -> Standardized
  -> Validity: per-sample boolean mask
```

### 2.3 Blendshape Preprocessing

```
Raw (N, 1489) landmarks @ ~30 Hz
  -> Extract channels 0-51                    -> (N, 52) FACS-compatible AUs
  -> Face detection (all channels ~0?)        -> Mark absent frames invalid
  -> Linear interpolation of gaps <= 0.5s     -> Fill brief tracking drops
  -> Per-blendshape z-score                   -> Standardized
  -> Clip to [-10, 10]
  -> Append activity channel (ch 52)          -> (N, 53) total
```

### 2.4 Pose Preprocessing

```
Raw (N, 132) = 33 landmarks x 4 @ ~12-14 Hz
  -> Reshape to (N, 33, 4) = (x, y, z, visibility)
  -> Zero landmarks with visibility < 0.5
  -> Frame validity = >= 10 landmarks visible
  -> Flatten to (N, 99) = 33 x 3 (drop visibility column)
  -> Z-score per-coordinate (visible samples only, zeros excluded)
  -> Clip to [-10, 10]
```

### 2.5 Activity Channel

For modalities that receive an activity channel (blendshapes, pose features, EEG features), the channel is a **causal** RMS deviation from a trailing 30-second mean:

```python
activity(t) = RMS_across_channels(features(t)) - mean(RMS[t-30s : t])
```

This captures temporal changes in signal energy (e.g., a burst of facial movement) without using future data. The activity channel is clipped to [-10, 10].

---

## Step 3: Feature Extraction

**Files**: `cadence/data/eeg_features.py`, `cadence/data/preprocessors.py`

Three modalities undergo further feature extraction to produce compact, coupling-relevant representations:

### 3.1 EEG Features (8 channels at 2 Hz)

The 14-channel raw EEG is transformed into 7 neuroscience-motivated features plus an activity channel:

| Channel | Feature | Description | Method |
|---------|---------|-------------|--------|
| 0 | Engagement Index | beta / (alpha + theta), frontal ROI | Band power ratio from Welch PSD |
| 1 | Frontal Aperiodic Exponent | 1/f slope (E/I balance proxy) | Log-log linear regression on [2-3.5] u [30-40] Hz |
| 2 | Frontal Theta Burst Fraction | % time with above-threshold theta power | BOSC detection (chi2 p=0.95 threshold) |
| 3-4 | Frontal Theta Phase (cos, sin) | Circular mean of instantaneous phase | Hilbert transform on bandpass-filtered theta |
| 5-6 | Frontal Alpha Phase (cos, sin) | Circular mean of instantaneous phase | Hilbert transform on bandpass-filtered alpha |
| 7 | Activity | Causal RMS deviation | Trailing 30s mean subtracted |

**Key parameters**:
- FFT window: 2.0s (constant for spectral quality)
- Output rate: configurable via `output_hz` (default 2 Hz, can be set to 5 Hz)
- Frontal ROI: AF3, F3, F4, AF4 (Emotiv EPOC indices 0, 2, 11, 13)
- Selective z-scoring: only engagement_index and aperiodic_exponent (unbounded) are z-scored; phase and burst features are naturally bounded

### 3.2 ECG Features (6 channels at 2 Hz)

Heart rate variability metrics derived from R-peak detection:

| Channel | Feature | Description |
|---------|---------|-------------|
| 0 | HR | Instantaneous heart rate (60/IBI) |
| 1 | IBI Dev | Inter-beat interval deviation from local median |
| 2 | RMSSD | Root mean square of successive IBI differences (vagal tone) |
| 3 | HR Accel | First derivative of HR (sympathetic activation) |
| 4 | QRS Amp | R-peak amplitude (signal quality proxy) |
| 5 | HR Trend | Detrended HR (30s trailing mean subtracted) |

### 3.3 Pose Features (41 channels at ~12-14 Hz)

The 99-channel raw pose is compressed into 7 body segment groups (40 features + 1 activity):

| Segment | Channels | Features |
|---------|----------|----------|
| Head | 0-7 | Centroid(3) + angle(1) + extent(1) + lean(2) + rotation(1) |
| Left Arm | 8-12 | Centroid(3) + angle(1) + extent(1) |
| Right Arm | 13-17 | Centroid(3) + angle(1) + extent(1) |
| Torso | 18-23 | Centroid(3) + angle(1) + extent(1) + lean(1) |
| Left Leg | 24-28 | Centroid(3) + angle(1) + extent(1) |
| Right Leg | 29-33 | Centroid(3) + angle(1) + extent(1) |
| Global | 34-39 | Center of mass(3) + height(1) + spread(1) + symmetry(1) |
| Activity | 40 | Causal RMS deviation |

---

## Step 4: Temporal Alignment

**File**: `cadence/data/alignment.py`

All modality streams are trimmed to the common overlapping time range:

```
t_start = max(earliest timestamp across all streams)
t_end   = min(latest timestamp across all streams)

For each stream:
  - Trim to [t_start, t_end]
  - Zero-reference timestamps: ts -= t_start
```

Each modality retains its native sampling rate -- alignment only clips the time range, it does not resample. This is critical: CADENCE preserves native temporal resolution (e.g., blendshapes at 30 Hz, EEG features at 2 Hz) and handles cross-rate alignment later in the design matrix stage.

**Known exclusions** (applied via `apply_modality_exclusions`):
- Session y24: participant 2 pose_features fully masked (anomalous 71 Hz capture rate)
- Session y26: excluded entirely (non-overlapping time ranges)

---

## Step 5: Basis Function Construction

**File**: `cadence/basis/raised_cosine.py`

CADENCE models coupling as a distributed lag: the effect of source signal x(t) on target signal y(t) is spread over a range of time lags. Rather than estimating a weight at every lag, we use a smooth basis expansion.

### 5.1 Raised Cosine Basis

Each basis function is a raised cosine bump:

```
phi_j(s) = 0.5 * (1 + cos(pi * clamp((s - c_j) / w_j, -1, 1)))
```

where `c_j` is the center lag and `w_j` is the half-width.

**Log-spacing** (default): Centers are placed on a logarithmic scale, giving denser coverage at short lags (where coupling is typically strongest) and sparser coverage at long lags.

### 5.2 Per-Modality Basis Configuration

| Modality | Basis Type | n_basis | Lag Range | Rate | Kernel Samples |
|----------|-----------|---------|-----------|------|---------------|
| EEG | Single-band | 8 | 0-5s | 5 Hz | 25 |
| BL | Single-band | 8 | 0-5s | 5 Hz | 25 |
| Pose | Single-band | 8 | 0-5s | 2 Hz (internal) | 10 |
| ECG | Multi-band | 8+5=13 | 0-5s + 20-60s | 2 Hz (internal) | 10 + 80 |

ECG uses a multi-band basis: a dense short-lag band (0-5s, sensorimotor coupling) plus a sparse long-lag band (20-60s, autonomic coupling). The gap at 5-20s means no coupling is modeled in that range -- intentional, as ECG coupling at those timescales is physiologically implausible.

### 5.3 Output

The basis is a matrix B of shape `(n_lag_samples, n_basis)`. Each column is one basis function evaluated at every lag sample. This matrix becomes the convolution kernel in the next step.

---

## Step 6: Design Matrix Construction

**File**: `cadence/basis/design_matrix.py`

For each pathway (source_modality -> target_modality), we build a regression design matrix X that combines basis-convolved source signals with autoregressive target terms.

### 6.1 Source Convolution (GPU conv1d)

Each source channel is convolved with all basis functions simultaneously using PyTorch grouped conv1d:

```
For source signal x(t) with C_src channels and n_basis basis functions:
  Input:  (1, C_src, T_src)
  Kernel: (n_basis, 1, n_lag_samples), flipped for causality
  Output: (1, C_src * n_basis, T_src)  via groups=C_src
```

**Causal padding**: The kernel is applied with left-side zero-padding so that the convolved output at time t only uses source values at times <= t (no future leakage).

### 6.2 Cross-Rate Resampling

Source signals are pre-resampled to the pathway's internal rate before convolution. This ensures the conv1d kernel samples correspond to the correct lag durations:

```
Example: EEG at internal_rate=5 Hz, max_lag=5s
  -> Kernel has 25 samples covering 0-5s (correct)

Without resampling: EEG native at 2 Hz, max_lag=5s
  -> Kernel has 10 samples, but conv1d operates at 2 Hz = correct

But: BL at native 30 Hz with a 5 Hz basis
  -> Must resample BL to 5 Hz first, then convolve with 25-sample kernel
```

Target signals are similarly resampled to the evaluation time grid via linear interpolation.

### 6.3 Autoregressive Terms

The target signal's own past values are included as predictors (AR model). With `ar_order=3`:

```
AR terms = [y(t-1), y(t-2), y(t-3)]  for each target channel
```

These capture the target's autocorrelation, ensuring that any dR2 attributed to the source is above and beyond what the target predicts from its own history.

### 6.4 Full Design Matrix Structure

```
X_full = [ phi_0(x_ch0) | phi_1(x_ch0) | ... | phi_0(x_ch1) | ... | y(t-1) | y(t-2) | y(t-3) ]
         |<------------- C_src * n_basis source terms ----------->|<-- ar_order * C_tgt -->|

X_restricted = [ y(t-1) | y(t-2) | y(t-3) ]  (AR-only baseline)

Shape: X_full is (T_eval, p_full), X_restricted is (T_eval, p_restricted)
```

**Typical predictor counts** (Phase 1, with PCA-reduced channels):

| Pathway | C_src (PCA) | n_basis | AR terms | p_full | p_restricted |
|---------|-------------|---------|----------|--------|-------------|
| EEG -> EEG | 8 | 8 | 3*4=12 | 76 | 12 |
| ECG -> ECG | 4 | 13 | 3*4=12 | 64 | 12 |
| BL -> BL | 4 | 8 | 3*4=12 | 44 | 12 |
| Pose -> Pose | 6 | 8 | 3*4=12 | 60 | 12 |

---

## Step 7: Static Screening (Ridge Regression)

**Files**: `cadence/regression/ridge.py`, `cadence/coupling/estimator.py`

Before running the expensive time-varying analysis, each pathway is screened for evidence of coupling using global (static) ridge regression.

### 7.1 Ridge Regression

For each pathway, two models are fit to the entire session:

```
Full model:       y = X_full @ beta_full + eps
Restricted model: y = X_restricted @ beta_restricted + eps

Solution: beta = (X'X + lambda*I)^{-1} X'y   (via Cholesky decomposition on GPU)
```

**Parameters**: `lambda_ridge = 1e-3` (prevents overfitting when p is large relative to T).

### 7.2 Delta-R-squared

The coupling effect is measured as the *incremental variance explained* by the source:

```
dR2 = R2_full - R2_restricted
```

If `dR2 > 0`, the source signal explains variance in the target above and beyond the target's own autoregressive structure. This is the Granger causality principle.

### 7.3 Surrogate Significance Testing

To determine whether the observed dR2 is statistically significant, we generate 100 surrogate datasets by circular-shifting the convolved source columns:

```
For k = 1, ..., n_surrogates:
  1. Randomly shift convolved source columns: X_source_k = roll(X_source, shift_k)
     - shift_k drawn uniformly from [10%, 90%] of T (preserves autocorrelation)
  2. Build surrogate design matrix: X_surr_k = [X_source_k | X_restricted]
  3. Fit ridge regression, compute R2_surr_k
  4. Compute dR2_surr_k = R2_surr_k - R2_restricted

p-value = fraction of surrogates with dR2_surr >= dR2_real
         (floored at 1/(n_surrogates + 1))
```

**Key optimization**: The surrogate test shifts the already-convolved source columns, not the raw source signal. This is valid because `shift(phi * x) = phi * shift(x)` for circular shifts, and avoids re-computing the expensive conv1d for each surrogate.

**GPU batching**: All `n_surrogates` systems are solved in a single batched `torch.linalg.solve` call via `batched_ridge_multi()`, with VRAM-aware batch sizing.

### 7.4 PCA Reduction

For Phase 1 (modality-level screening), high-dimensional modalities are reduced via PCA to keep the predictor count manageable:

| Modality | Raw Channels | PCA Channels (Phase 1) |
|----------|-------------|----------------------|
| EEG | 8 | 8 (no reduction) |
| ECG | 6 | 4 |
| BL | 53 | 4 |
| Pose | 41 | 6 |

The PCA-projected signals are standardized to unit variance so that `lambda_ridge` is scale-independent.

---

## Step 8: Time-Varying EWLS Analysis

**File**: `cadence/regression/ewls.py`

For all pathways (regardless of screening result), CADENCE computes a time-varying coupling trajectory using Exponentially Weighted Least Squares (EWLS).

### 8.1 Forward-Backward EWLS

The EWLS solver fits a separate regression at every timepoint, using exponentially decaying weights for past and future observations:

```
Forward accumulator:
  S_fwd(t) = gamma * S_fwd(t-1) + x(t) x(t)'
  where gamma = exp(-dt / tau)

Backward accumulator (carry-based, memory-efficient):
  S_bwd(t) = gamma * S_bwd(t+1) + x(t+1) x(t+1)'

Combined:
  S_xx(t) = S_fwd(t) + S_bwd(t)
  S_xy(t) = S_xy_fwd(t) + S_xy_bwd(t)

Solve at each t:
  beta(t) = solve(S_xx(t) + lambda*I, S_xy(t))
```

**Time constant tau** controls the effective window width:
- EEG, ECG: `tau = 30s` (slow modalities, smooth coupling trajectory)
- BL, Pose: `tau = 8s` (fast modalities, track brief coupling episodes)

### 8.2 Performance Optimizations

The solver uses several GPU-friendly optimizations:

1. **Batch outer products**: `S_xx = torch.bmm(X.unsqueeze(2), X.unsqueeze(1))` computes all T outer products in one GPU kernel
2. **Pre-zeroing invalids**: Invalid timepoints are zeroed before the scan (not checked per-step)
3. **Branch-free forward scan**: `S_xx[t].add_(S_xx[t-1], alpha=gamma)` -- no conditionals
4. **Carry-based backward**: Only one `(T, p, p)` buffer allocated (forward result); backward carry adds directly to it
5. **CPU-side validity**: `valid_cpu = valid.cpu().numpy()` -- avoids CUDA synchronization in the backward loop

**Complexity**: O(T * p^2) -- two passes over data. For the largest pathway (EEG at 5 Hz, 3000s, p=76): ~4 seconds, ~1.6 GB VRAM.

### 8.3 Per-Modality Eval Rates

Different modalities run EWLS at different temporal resolutions:

| Modality | Internal Rate | Output Rate | Rationale |
|----------|--------------|-------------|-----------|
| EEG | 5 Hz | 5 Hz | Fast neural transients (0.2s resolution) |
| BL | 5 Hz | 5 Hz | Brief facial expressions |
| Pose | 2 Hz | 1 Hz | Slow postural changes |
| ECG | 2 Hz | 0.033 Hz | Autonomic timescale (~30s windows) |

**Internal rate** is always >= the default 2 Hz to maintain numerical stability (sufficient effective_n). For slow output rates (ECG at 0.033 Hz), EWLS runs at 2 Hz internally and the output is downsampled via linear interpolation.

### 8.4 Output

For each pathway at each eval timepoint:
- `dR2(t)` = R2_full(t) - R2_restricted(t): time-varying coupling strength
- `beta(t)` = (p,) regression coefficients: time-varying coupling kernel weights
- `n_eff(t)` = effective sample count: data quality indicator

---

## Step 9: Significance Decision

**File**: `cadence/coupling/estimator.py`

After all pathways are processed, significance is determined via a three-gate test:

```
is_significant = (
    surrogate_p < alpha          # Gate 1: Surrogate test (p < 0.05)
    AND static_dR2 > min_dR2    # Gate 2: Effect size threshold (dR2 > 0.001)
    AND ewls_mean_dR2 > 0       # Gate 3: EWLS consistency check
)
```

### 9.1 Gate 1: Surrogate p-value

The surrogate test controls for false positives from autocorrelation. With 100 circular-shift surrogates and alpha=0.05, a pathway must have dR2 exceeding at least 95% of surrogates.

**Optional BH-FDR correction**: Can be enabled via `fdr_correction: 'bh'` in config, but requires >= 500 surrogates with 16 pathways to be practical. Disabled by default.

### 9.2 Gate 2: Effect Size

A minimum dR2 threshold (default 0.001) prevents reporting of statistically significant but practically negligible coupling.

### 9.3 Gate 3: EWLS Consistency

The EWLS mean dR2 (averaged over the full session) must be positive. This catches cases where the static (global) ridge estimate is borderline positive but the time-varying estimate disagrees -- indicating a spurious result. This gate was added to eliminate ECG false positives where strong autocorrelation created inconsistent static vs. time-varying estimates.

---

## Step 10: Feature-Level Decomposition (Phase 2)

**File**: `cadence/coupling/estimator.py`

For pathways that pass significance screening, CADENCE decomposes the coupling into per-feature contributions. This reveals *which specific features* drive the modality-level coupling.

### 10.1 Feature Groups

Each modality is decomposed into interpretable feature groups:

**EEG Features** (8 groups, 1 channel each):
- engagement_index, frontal_aperiodic_exponent, frontal_theta_burst_frac
- 4 phase features (theta cos/sin, alpha cos/sin)
- activity

**ECG Features** (6 groups, 1 channel each):
- HR, IBI_dev, RMSSD, HR_accel, QRS_amp, HR_trend

**Pose Features** (7 groups, 5-6 channels each):
- Head, Left Arm, Right Arm, Torso, Left Leg, Right Leg, Global

**Blendshape Features** (11 groups, 1-12 channels each):
- Brow, Cheek/Nose, Eye Blink, Eye Gaze, Eye Lid, Jaw
- Mouth Affect, Mouth Form, Mouth Move, Neutral, Activity

### 10.2 Per-Feature dR2

For each feature group within a significant pathway:
1. Extract the subset of source channels belonging to that group
2. Pre-resample to internal rate
3. Build a subset design matrix (group channels x basis + AR)
4. Run EWLS `solve_restricted` to get time-varying dR2

The per-feature dR2 values indicate how much coupling each feature group contributes. These can be visualized as stacked timecourses or feature-level heatmaps.

---

## Step 11: Visualization and Reporting

**Files**: `cadence/visualization/{timecourse,heatmaps,kernels,comparison}.py`, `scripts/run_session.py`

### 11.1 Coupling Timecourse

A multi-panel time series showing dR2(t) for each pathway, grouped by target modality. Significant pathways shown with full opacity; non-significant shown faded. Optional rolling-mean smoothing to reduce high-frequency noise.

### 11.2 4x4 Coupling Matrix

A heatmap showing session-average dR2 for all 16 source->target modality pairs. Significant pathways annotated with `*`. Uses a diverging colormap centered at zero (blue = negative, red = positive coupling).

### 11.3 Coupling Kernels

For significant pathways, the impulse response function h(s) is reconstructed:

```
h(s) = sum_j alpha_j * phi_j(s)
```

This shows how a unit change in the source modality propagates to the target over lag time (0-5 seconds). The kernel shape reveals the lag structure of coupling -- e.g., immediate (peak at 0s) vs. delayed (peak at 2s).

### 11.4 CADENCE vs MCCT Comparison

Side-by-side timecourse plots comparing CADENCE dR2 with MCCT CSGI for the same session. Pearson correlation computed between the two methods per modality.

### 11.5 Output Files

Per session, per direction:
```
{direction}-timecourse.png     # Time-varying dR2 per pathway
{direction}-matrix.png         # 4x4 coupling matrix heatmap
{direction}-kernels.png        # Impulse response h(s) for significant pathways
{direction}_results.json       # Summary: dR2 means, significance, timing
```

---

## Synthetic Validation

**File**: `scripts/run_synthetic_timing.py`, `cadence/synthetic.py`

CADENCE is validated on synthetic sessions with known ground-truth coupling.

### Synthetic Session Generation

1. Two independent Lorenz attractors generate chaotic base dynamics for each participant
2. Features are created via random projection + nonlinear transformations (xy product, tanh, trig)
3. Per-modality coupling gates define when coupling is active (episodic, with smooth ramps)
4. During active coupling: `p2(t) = kappa * gate(t) * p1(t-lag) + sqrt(1-kappa^2) * noise(t)`
5. Activity envelopes create independent amplitude modulations per participant

### Test Battery

| Test | Coupled Modality | Expected |
|------|-----------------|----------|
| A | EEG only (kappa=0.5) | Detect EEG, not others |
| B | BL only (kappa=0.5) | Detect BL, not others |
| C | ECG only (kappa=0.5) | Detect ECG, not others |
| D | Pose only (kappa=0.5) | Detect Pose, not others |
| E | EEG + BL (kappa=0.5) | Detect both, not others |
| F | Null (all kappa=0) | Detect nothing (< 5% FP rate) |

### Timing Accuracy Metrics

- **AUC-ROC**: How well does smoothed dR2(t) discriminate coupled vs. uncoupled timepoints
- **Pearson r**: Correlation between smoothed dR2(t) and ground-truth coupling gate

### Validated Results (94% pass rate, 17/18 test-seeds)

| Modality | Detection Rate | Timing AUC-ROC | False Positive Rate |
|----------|---------------|----------------|-------------------|
| EEG | 100% | 0.810 +/- 0.077 | 0% |
| BL | 100% | 0.635 +/- 0.045 | 0% |
| ECG | 67% | 0.586 +/- 0.045 | 0% |
| Pose | 100% | 0.555 +/- 0.029 | 0% |

---

## Configuration Reference

**File**: `configs/default.yaml`

```yaml
# Data
session_cache: "C:/Users/optilab/desktop/MCCT/session_cache"
device: cuda

# Basis functions (default single band)
basis:
  layer1:
    n_basis: 8
    max_lag_seconds: 5.0
    min_lag_seconds: 0.0
    log_spacing: true

# ECG multi-band lag structure
lag_bands:
  ecg_features:
    - {n_basis: 8, min_lag_seconds: 0.0, max_lag_seconds: 5.0, log_spacing: true}
    - {n_basis: 5, min_lag_seconds: 20.0, max_lag_seconds: 60.0, log_spacing: false}

# EWLS solver
ewls:
  tau_seconds: 30.0       # Exponential decay time constant
  lambda_ridge: 1.0e-3    # Ridge regularization
  eval_rate: 2.0           # Default evaluation rate (Hz)
  min_effective_n: 20      # Minimum effective samples

# Per-modality tau overrides (fast modalities)
ewls_tau_overrides:
  blendshapes: 8.0
  pose_features: 8.0

# Per-modality eval rate overrides
eval_rate_overrides:
  eeg_features: 5.0       # 0.2s resolution
  blendshapes: 5.0         # 0.2s resolution
  pose_features: 1.0       # 1.0s resolution
  ecg_features: 0.0333     # ~30s windows

# Autoregressive baseline
autoregressive:
  order: 3
  include: true

# Phase 1 PCA reduction
phase1_max_channels:
  eeg_features: 8
  ecg_features: 4
  blendshapes: 4
  pose_features: 6

# Significance testing
significance:
  f_test_alpha: 0.05
  surrogate:
    n_surrogates: 200
    n_screen_surrogates: 100
    method: circular_shift
    min_shift_frac: 0.1
  fdr_correction: false
  session_level:
    min_dr2: 0.001
```

---

## Computational Performance

For a typical 45-minute session (T ~ 2700s at 2 Hz base rate):

| Stage | Time | VRAM |
|-------|------|------|
| Session loading (cached) | ~1s | N/A |
| Design matrix (16 pathways) | ~2s | ~0.5 GB |
| Static screening (16 x 100 surrogates) | ~15s | ~1 GB peak |
| EWLS (16 pathways) | ~20s | ~2 GB peak |
| Phase 2 decomposition (~4 significant) | ~15s | ~1.5 GB |
| Visualization | ~3s | N/A |
| **Total per direction** | **~45s** | **~2 GB peak** |

For 3000s synthetic sessions, total analysis time is ~45s per test (including generation).
