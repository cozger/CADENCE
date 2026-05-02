# The Normalization Chain: Every Transformation from Raw to rSLDS

This document traces every z-scoring, standardization, and prewhitening step in the V11 pipeline, in order. This is critical for your modeler audience to confirm: they need to know exactly what statistical properties the data has by the time the model sees it.

---

## The Big Picture

Data passes through **four normalization layers** before the rSLDS sees it:

```
Layer 1: Raw preprocessing      — per-channel z-score (removes hardware gain/offset)
Layer 2: Metric computation      — surrogate z-score OR within-metric normalization
Layer 3: Prewhitening            — conditional AR(1) (removes temporal autocorrelation)
Layer 4: Final standardization   — mean=0, std=1 (equalizes scale across channels)
```

**Important:** Not every channel passes through all four layers. Some metrics are inherently normalized (e.g., phase coherence lives in [-1, 1]). The table below shows exactly which transformations each channel receives.

---

## Layer 1: Raw Signal Preprocessing

These happen once, on the raw sensor data, before any coupling computation.

| Signal | Normalization | What it does | Why |
|--------|--------------|--------------|-----|
| EEG (14ch × 2 people) | Per-channel z-score (valid samples only) | Centers each electrode at 0, scales to std=1 | Emotiv channels have different gains/impedance; makes channels comparable |
| ECG (1ch × 2) | Z-score after bandpass | Same centering/scaling | Different electrode contact across participants |
| Blendshapes (52 AU × 2) | Per-AU z-score (valid frames only) | Converts [0,1] native range to z-distribution | AU coefficients have different dynamic ranges; z-scoring makes them comparable across participants |
| Pose (99 coords × 2) | Per-coordinate z-score (visible frames only) | Centers body coordinates | Different body sizes/positions relative to camera |

**Key point for the room:** After Layer 1, all raw signals are zero-mean, unit-variance, per-channel. This is the input to all coupling computations.

---

## Layer 2: Per-Metric Normalization

Each of the 28 scaffold channels is produced by a different computation, and each has its own normalization. Here's what happens to each:

### EEG Channels (12 channels)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **imcoh_θ/α/β** (3ch) | `\|imag(Sxy)\| / sqrt(Sxx·Syy)` averaged over ROI pairs | **None** — ImCoh is bounded [0, 1] by construction | Raw imaginary coherence [0, 1] |
| **conc_θ/α/β** (3ch) | Hilbert amplitude envelope → per-participant z-score → `(z_P1 + z_P2) / 2` | **Per-participant z-score** on the Hilbert envelope before combining | z-scored concordance, roughly N(0, 0.5) if independent |
| **dyn_θ/α/β** (3ch) | EWMAD of concordance, log-transformed | **Log transform** of the EMA deviation | Log-scale volatility (unbounded, right-skewed → approximately symmetric) |
| **asym_θ/α/β** (3ch) | `asym_sign × (z_P1 - z_P2)` | **Inherits** per-participant z-scores from concordance computation | z-scored difference, roughly N(0, 1) if independent |

**What to tell the room about EEG concordance:** "Each participant's band power envelope is z-scored independently, then we average their z-scores. This means concordance is NOT the coherence of raw power — it's the alignment of standardized power states. A participant with 10× more alpha power doesn't dominate; both are on the same scale."

### Facial Channels (2 channels)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **bl_expr** (1ch) | Wavelet coherence in expression band (0.5–2 Hz) | **Surrogate z-score**: `(real_coh - null_mean) / null_std` against 200 circular-shift surrogates | z above circular-shift null |
| **bl_activity_conc** (1ch) | `(z_P1_activity + z_P2_activity) / 2` where activity = RMS(AUs) - trailing 30s mean | **Causal activity z-score** (per-participant), then averaged | z-scored concordance |

### Autonomic/Body Channels (4 channels)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **ecg_lf** (1ch) | Hilbert envelope cross-product of bandpass IBI, LF band | **Surrogate z-score**: cross-product vs. 200 circular-shift surrogates | z above circular-shift null |
| **ecg_hf** (1ch) | Same for HF band | **Surrogate z-score** | z above circular-shift null |
| **resp** (1ch) | `cos(phi_P1 - phi_P2)` where phi = Hilbert instantaneous phase | **None** — phase coherence is bounded [-1, 1] | Raw phase coherence [-1, 1] |
| **pose** (1ch) | Max-lag velocity cross-product over ±5s bank | **Surrogate z-score**: best-lag cross-product vs. 200 circular-shift surrogates | z above circular-shift null |

### LZ Complexity Channels (4 channels)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **lz_conc_θ/α** (2ch) | LZ76 on Hilbert amplitude envelope → `(z_P1 + z_P2) / 2` | **Per-participant z-score** on LZ timecourse before combining | z-scored concordance |
| **lz_asym_θ/α** (2ch) | `asym_sign × (z_P1 - z_P2)` | **Inherits** per-participant z-scores | z-scored difference |

### Graph Channel (1 channel)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **graph_modularity** (1ch) | Louvain Q on windowed correlation graph of base 18D | **None** — modularity Q is bounded [0, 1] | Raw modularity [0, 1] |

### V11 Burst Channels (5 channels)

| Channel | Raw metric | Layer 2 normalization | Units entering scaffold |
|---------|-----------|----------------------|------------------------|
| **te_conc_θ/α** (2ch) | `(z_P1→P2 + z_P2→P1) / 2` where each z is TE vs. 200 circular-shift surrogates | **Surrogate z-score** per direction, then averaged | z above circular-shift null |
| **burst_coinc_θ/α/β** (3ch) | Dilated AND co-occurrence rate | **Surrogate z-score**: coincidence rate vs. 200 circular-shift surrogates, clipped [-10, 10] | z above circular-shift null |

---

## Summary of Layer 2 Output: What Units Are the 28 Channels In?

Before prewhitening, the 28 channels are a **heterogeneous mix**:

| Type | Channels | Range | Count |
|------|----------|-------|-------|
| Surrogate z-scores | ecg_lf, ecg_hf, pose, bl_expr, te_conc_θ/α, burst_coinc_θ/α/β | Unbounded, typically [-3, +5] | 9 |
| Per-participant z-scored concordance/asymmetry | conc_θ/α/β, asym_θ/α/β, lz_conc_θ/α, lz_asym_θ/α, bl_activity_conc | Roughly N(0, ~0.7) | 11 |
| Raw bounded metrics | imcoh_θ/α/β, resp, graph_modularity | [0,1] or [-1,1] | 5 |
| Log-transformed volatility | dyn_θ/α/β | Unbounded, roughly symmetric | 3 |

**This heterogeneity is why Layers 3 and 4 are essential.** Without them, the rSLDS would weight channels by their variance, not by their information content.

---

## Layer 3: Conditional AR(1) Prewhitening

Applied per-channel to the assembled 28D matrix.

**Algorithm:**
```
For each channel d in 0..27:
    rho = lag-1 autocorrelation of z_matrix[:, d]
    if |rho| > threshold:
        z[t] = z[t] - rho * z[t-1]    (repeat up to 3 rounds)
```

**What it does:** Removes first-order temporal dependence. After prewhitening, each channel's value at time t is the *innovation* — the part not predicted by the previous timepoint.

**Which channels get prewhitened:** In practice, most channels have significant autocorrelation at 2 Hz. Slowly-varying channels (modularity, LZ concordance, concordance) have rho > 0.8 and get heavy prewhitening. Burst coincidence and TE concordance tend to have lower rho and may pass through unchanged.

**What to tell the room:** "Prewhitening is conditional — we only filter channels that actually need it. The threshold prevents us from amplifying noise in channels that are already approximately white. The diagnostics log rho before and after for every channel."

---

## Layer 4: Final Standardization

After prewhitening, every channel is standardized:

```
For each channel d:
    z_out[:, d] = (z_out[:, d] - mean) / std
```

**What it does:** Centers all 28 channels at exactly 0 and scales to exactly std=1. This is why the null state's mean can be fixed at zero — after standardization, zero *is* the session-wide average.

**What to tell the room:** "After this step, all 28 channels are mean-zero, unit-variance, and approximately temporally white. The rSLDS sees a (T × 28) matrix where each column is an innovation sequence with standardized scale."

---

## Transition Covariates (7D) — Same Treatment

The 7D covariate matrix U goes through the same Layers 3 and 4:

| Covariate | Layer 2 source | Prewhitened? | Standardized? |
|-----------|---------------|-------------|--------------|
| z_slow PC1 | Gaussian-smoothed PCA of base 18D | Yes (AR(1)) | Yes (std=1) |
| z_slow PC2 | Same | Yes | Yes |
| Coupling flexibility | Graph spectral energy ratio [0,1] | Yes | Yes |
| Lambda-2 | Algebraic connectivity [0, 2] | Yes | Yes |
| Graph change-point | Derivative of modularity + λ₂ | Yes | Yes |
| TE asym theta | Surrogate z-score difference | Yes | Yes |
| TE asym alpha | Surrogate z-score difference | Yes | Yes |

---

## What the rSLDS Actually Sees

After all four layers, the model receives:

- **Y**: (T × 28) matrix — every column is approximately zero-mean, unit-variance, temporally white (low autocorrelation)
- **U**: (T × 7) matrix — same properties
- **mask**: (T × 28) boolean — which observations are valid

The emission model `y_t ~ N(μ[z_t], σ²[z_t])` assumes conditionally iid Gaussian observations given the state. The prewhitening + standardization make this assumption approximately correct.

### Critical caveat: what session-wide standardization erases

Because Layer 4 standardizes to the **session-wide** mean, the model sees deviations from the session average — not absolute coupling levels. The NULL state (mean fixed at zero) means "coupling at the session average," not "no coupling."

This has a concrete consequence: if an intervention (meditation) elevates the coupling floor for the rest of the session, post-intervention conversation at the new, higher baseline registers as NULL. Clinically, conv_2 post-meditation shows signs of increased rapport, but the model assigns 50% NULL because that rapport sits at the (now-elevated) session mean.

**Evidence this is a normalization artifact, not real absence:**
- COUP state usage is identical in conv_1 and conv_2 (~25%)
- NULL replaced OTHER (unstructured activity), not COUP (phase coupling)
- Coupling flexibility is the lowest of any condition (0.18 = maximally organized)

**Possible alternatives:**
- Baseline-relative normalization (z-score relative to base_EO/base_EC only)
- Including raw condition-level means as additional covariates
- Two-stage analysis: rSLDS on relative dynamics + separate condition-mean analysis

---

## The Question for the Room

> "We normalize in four layers: raw hardware calibration, per-metric surrogate z-scoring, AR(1) prewhitening, and final standardization. Each layer has a clear purpose, but the interaction between layers could be non-trivial. For example:
>
> - Surrogate z-scoring already partially removes autocorrelation (the null distribution absorbs slow drift). Does subsequent AR(1) prewhitening double-correct?
> - The per-participant z-scoring in concordance happens BEFORE cross-participant combination. Should it happen after?
> - ImCoh enters raw [0,1] while pose enters as a surrogate z-score. After final standardization they're both unit-variance, but the *information content per unit of variance* is different. Does this matter?
>
> I'd welcome your take on whether this chain is sound or if we're over-processing."

---

## Quick Reference: Draw This on the Board

```
RAW → [z-score per channel] → [coupling metric] → [surrogate z OR bounded] → [AR(1) if rho>thresh] → [mean=0, std=1] → rSLDS
       Layer 1                  Layer 2                                         Layer 3                  Layer 4
       hardware calibration     statistical calibration                         temporal whitening       scale equalization
```

This is a one-line summary you can draw at the top of the whiteboard and point to when questions arise.
