# CADENCE V5: Literature-Informed Pipeline Upgrades

## Context

CADENCE has two validated temporal localization (TL) detectors with known detection thresholds:

| Pipeline | Method | Best Performance | Detection Threshold |
|----------|--------|-----------------|-------------------|
| **EEG** | PLV (30f broadband, 20s win, 15s smooth, pooled, avg-ref) | κ=0.2: 80% hit / 5% FA | κ ≈ 0.20 (spatial decay, 14ch EPOC) |
| **BL** | Cross-product multi-lag bank (52 AU, AR3 res, 0.5-5s, 3s σ) | κ=0.118: 46.5% hit / 6.5% FA (oracle lag) | κ ≈ 0.10 (18 heterogeneous AUs) |

Four literature searches identified methods that could push these thresholds lower. The goal is to upgrade both pipelines with the most impactful findings before building the multimodal fusion framework.

### Key Bottlenecks to Address

**EEG at κ=0.1** (currently 46% hit): Limited by ~4 effective independent channels (spatial correlation on 14-ch EPOC). z scales linearly with κ — need to extract more signal per channel or add complementary metrics.

**BL at κ=0.118 mixed scenario** (currently 37% hit with multi-lag bank): Limited by AU decorrelation time (~0.2s) and 10% coupling duty cycle. The cross-product captures continuous amplitude co-modulation but may miss event-level coupling (coordinated activation timing).

### Literature Methods Ranked by Expected Impact

**Tier 1 — High impact, implement now:**

1. **Event Synchronization** (Quian Quiroga 2002, Phys Rev E) — Detects co-occurrence of discrete events across signals. For BL, tests whether AU activation onsets in P1 trigger AU onsets in P2. For EEG, tests whether power burst onsets co-occur. Captures a DIFFERENT signal than the continuous cross-product/PLV — event timing vs amplitude co-modulation. Handles different rates naturally. O(N_events²), very fast. **If event sync captures partially independent coupling signal, within-modality fusion of cross-product + event sync could push thresholds lower.**

2. **Within-modality metric fusion** — Combine z-scores from two complementary metrics (cross-product z + event sync z for BL; PLV z + event sync z for EEG) before thresholding. If the metrics are partially independent, Stouffer combination gives √2 improvement in effective z. Literature supports: Likens & Wiltshire (2020) showed multi-metric approaches outperform single metrics for capturing different timescales of synchrony.

3. **IAAFT surrogates** — Iterative Amplitude-Adjusted Fourier Transform surrogates preserve both power spectrum AND amplitude distribution (vs circular shift which only preserves autocorrelation). Provides a stronger null, reducing false alarms from spectral artifacts. Drop-in replacement for existing circular shift. Literature: standard in the nonlinear dynamics community for significance testing (Theiler et al. 1992, Schreiber & Schmitz 2000).

**Tier 2 — Medium impact, implement after Tier 1:**

4. **Robust BOCPD** (Altamirano et al., ICML 2023) — Bayesian Online Change Point Detection for coupling regime detection on z-score timecourses. Replaces simple threshold + min-event filter with principled probabilistic onset/offset detection. Provably robust under model misspecification. 10x faster than standard BOCPD. Provides posterior probability of being in "coupled" state at each timepoint. **Key advantage: adapts to local z-score statistics instead of using a global threshold.**

5. **Multi-resolution PLV for EEG** — Compute PLV at multiple window sizes (2s, 5s, 10s, 20s) simultaneously. Short windows detect transient coupling events; long windows accumulate evidence for weak coupling. Combine via max-over-scales (analogous to max-over-lags in BL). Inspired by Wavelet Multiscale Synchrony (Likens & Wiltshire 2020, SCAN). **Could improve EEG detection at κ=0.1 by capturing transient coupling that the 20s-only window averages out.**

6. **Transfer Entropy as validation** — Use IDTxl (Wollstadt 2019) to compute windowed multivariate TE on the same semi-synthetic data. Provides ground-truth reference for whether our cross-product/PLV approach captures all available coupling information. If TE detects coupling that our methods miss, it reveals what we're leaving on the table. **Not for production (too slow), but invaluable for understanding detection limits.**

**Tier 3 — Investigate later (deferred to multimodal phase):**

7. STOK adaptive Kalman (Pascucci 2020) — If EWLS is kept for session preprocessing
8. Copula-based cross-modal dependence — For the multimodal fusion phase
9. PCMCI+ / Regime-PCMCI (Tigramite) — Complementary causal discovery
10. MdCRQA — Nonlinear coupling as complementary measure

---

## Implementation Plan

### Phase 1: Event Synchronization Module

**Goal**: New detection metric that captures event-level coupling (AU onset co-occurrence, EEG burst co-occurrence).

**New file**: `cadence/significance/event_sync.py`

**Algorithm** (Quian Quiroga et al. 2002):

```
For each pair of event sequences (P1 events, P2 events):
  For each P1 event at t_i and P2 event at t_j:
    tau_ij = min(t_{i+1}-t_i, t_i-t_{i-1}, t_{j+1}-t_j, t_j-t_{j-1}) / 2
    If 0 < t_j - t_i < tau_ij:
      Q_forward += 1  (P1 preceded P2)
    If 0 < t_i - t_j < tau_ij:
      Q_backward += 1  (P2 preceded P1)

  Strength: S = (Q_forward + Q_backward) / sqrt(N1 * N2)
  Direction: D = (Q_backward - Q_forward) / sqrt(N1 * N2)
```

**Windowed version for temporal localization**:
- Sliding window (30s, 5s stride)
- Compute S and D within each window
- Surrogate calibration: circular-shift P1 event times, recompute
- Z-score against K=100 surrogates
- Output: z_event_sync(t) at 0.2 Hz (5s stride)

**Event detection per modality**:

| Modality | Event definition | Parameters |
|----------|-----------------|------------|
| BL (AUs) | Activation onset: signal crosses above mean+1σ AND derivative > 0 | Min inter-event: 0.3s. Per-channel. |
| EEG | Power burst onset: Hilbert envelope of bandpass (theta/alpha/beta) exceeds 90th percentile | Min inter-event: 0.5s. Per-channel per-band. |
| Pose | Movement onset: velocity magnitude crosses above 75th percentile | Min inter-event: 0.5s. Per-joint-group. |

**Key files to reference**:
- `cadence/significance/coherence_localization.py` — Existing TL infrastructure (surrogate calibration, min-event filter, gap filling)
- `cadence/constants.py` — BLENDSHAPE_COUPLING_GROUPS, EPOC_CHANNEL_NAMES

**Verification**:
- Semi-synthetic BL: inject coupling where P1 AU onsets trigger P2 AU onsets at 2s lag with κ=0.10-0.40
- Semi-synthetic EEG: inject coupling where P1 power bursts trigger P2 power bursts at 30ms lag
- Compare event sync z-scores with cross-product/PLV z-scores — are they correlated or complementary?
- Measure hit/FA rates independently and combined

### Phase 2: Within-Modality Metric Fusion

**Goal**: Combine cross-product z-scores with event sync z-scores for improved detection.

**Modify**: `cadence/significance/coherence_localization.py` or new `cadence/significance/metric_fusion.py`

**Algorithm**:
1. Run existing detector → z_continuous(t) at 2 Hz (cross-product for BL, PLV for EEG)
2. Run event sync detector → z_events(t) at 0.2 Hz
3. Resample both to common rate (0.2 Hz — limited by event sync window stride)
4. Combine: `z_fused(t) = (z_continuous(t) + z_events(t)) / sqrt(2)` (Stouffer, equal weights)
5. Joint surrogate calibration:
   - For each surrogate: circular-shift P1 signals by SAME offset
   - Run BOTH metrics → z_continuous_null(t), z_events_null(t)
   - Combine → z_fused_null(t)
   - Threshold from null distribution of z_fused
6. Output: z_fused(t), mask, per-metric attribution

**Key question to resolve empirically**: Are z_continuous and z_events independent under the null? If yes, Stouffer gives √2 improvement. If correlated (both triggered by same artifacts), improvement is less. The joint surrogate handles this automatically.

**Verification**:
- Compare detection performance: z_continuous only vs z_events only vs z_fused
- At κ=0.10 (BL threshold): does fusion push hit rate above 50%?
- At κ=0.10 (EEG near-threshold): does fusion push hit rate above 50%?
- FA rate should remain ≤ 5% (joint surrogates ensure this)

### Phase 3: IAAFT Surrogates

**Goal**: Stronger null model that preserves both power spectrum and amplitude distribution.

**New file**: `cadence/surrogates.py` — Add `iaaft_surrogate()` alongside existing `circular_shift_surrogate()`

**Algorithm** (Schreiber & Schmitz 2000):
```
1. Compute FFT of original signal
2. Rank-order a Gaussian white noise to match original's amplitude distribution
3. Iterate:
   a. Replace phases with original's (Fourier step)
   b. Rank-reorder to match original's amplitude distribution (amplitude step)
   c. Repeat until convergence (typically 10-50 iterations)
```

**Integration**:
- Add `surrogate_method` parameter to `wpli_temporal_localization()` and `xcorr_temporal_localization()`
- Options: `'circular'` (current default), `'iaaft'` (new), `'fourier_phase'` (existing)
- For within-modality TL: default to IAAFT (preserves more signal structure → harder-to-beat null → fewer FA)
- For cross-modal: keep circular shift (IAAFT is per-channel, doesn't preserve cross-channel structure needed for cross-modal nulls)

**Verification**:
- Run existing semi-synthetic tests with IAAFT vs circular shift
- Expect: similar or slightly lower hit rate, but lower FA rate
- Net effect: threshold can be lowered to recover hit rate, with cleaner detections
- Null test: IAAFT surrogate z-scores should be closer to N(0,1) than circular shift z-scores

### Phase 4: Robust BOCPD for Regime Detection

**Goal**: Replace threshold + min-event-filter with principled change-point detection on z-score timecourses.

**New file**: `cadence/significance/bocpd.py`

**Algorithm** (Adams & MacKay 2007, upgraded per Altamirano et al. 2023):
```
At each time step t:
  For each possible run length r:
    p(r_t = r | z_{1:t}) ∝ p(z_t | r_t = r) × p(r_t | r_{t-1})

  Underlying model: Gaussian with unknown mean and variance (conjugate Normal-Inverse-Gamma)
  Hazard function: constant (geometric prior on run length)

  Posterior: probability of being in a "new regime" at each timepoint
  Coupling mask: p(coupled) = sum of run-length posteriors where mean > threshold
```

**Integration**:
- Add as optional post-processing step after z-score computation
- Replace: `mask = z_agg > threshold; mask = _min_event_filter(mask, min_samples)`
- With: `mask, posterior = bocpd_segment(z_agg, hazard_rate, prior_params)`
- The BOCPD posterior IS the coupling probability — no need for separate smoothing/filtering

**Key advantage over current approach**: BOCPD adapts to local statistics. In high-variance periods, it requires larger z-scores to declare coupling. In stable periods, it detects smaller changes. This should reduce both FA (fewer noise-triggered detections) and improve hit rate (faster onset detection when coupling starts).

**Verification**:
- Run on semi-synthetic z-score timecourses with known coupling windows
- Compare hit/FA with current threshold approach
- The BOCPD posterior should track the true coupling gate more closely than the binary mask

### Phase 5: Multi-Resolution PLV for EEG

**Goal**: Detect coupling at multiple temporal scales simultaneously.

**Modify**: `cadence/significance/coherence_localization.py`

**Algorithm**:
1. Compute PLV at multiple window sizes: 2s, 5s, 10s, 20s (all with 0.5s stride)
2. For each window size, run full surrogate pipeline → z_w(t) per window size
3. Max-over-scales: `z_multires(t) = max(z_2s(t), z_5s(t), z_10s(t), z_20s(t))`
4. Surrogate calibration: run same multi-resolution pipeline on surrogates, take max-over-scales under null
5. Threshold from multi-resolution null distribution

**Rationale**: Same principle as multi-lag bank for BL. Short windows (2s) capture transient coupling bursts that 20s windows average out. Long windows (20s) accumulate weak but sustained coupling that short windows can't detect. The max-over-scales penalty is small because PLV at different windows is correlated (like the ~5 effective lags in BL's multi-lag bank).

**Computational cost**: ~4× current (4 window sizes). Mitigated by:
- P2 CWT computed once, reused across all window sizes
- Surrogates also reuse P2 CWT
- Per-frequency processing already handles VRAM

**Verification**:
- At κ=0.1 (current 46% hit): does multi-resolution push above 50%?
- At κ=0.2 (current 80% hit): should maintain or improve
- FA rate with multi-resolution surrogates should remain ≤ 5%

---

## Critical Files

| File | Role | Modifications |
|------|------|---------------|
| `cadence/significance/event_sync.py` | **NEW** | Event synchronization TL pipeline |
| `cadence/significance/metric_fusion.py` | **NEW** | Within-modality metric fusion |
| `cadence/significance/bocpd.py` | **NEW** | Robust BOCPD regime detection |
| `cadence/surrogates.py` | MODIFY | Add IAAFT surrogate generation |
| `cadence/significance/coherence_localization.py` | MODIFY | Add multi-resolution PLV; add surrogate_method param |
| `scripts/_test_event_sync.py` | **NEW** | Event sync validation |
| `scripts/_test_metric_fusion.py` | **NEW** | Fusion validation |
| `scripts/_test_multires_plv.py` | **NEW** | Multi-resolution PLV validation |

## Existing Code to Reuse

| Function | File | Reuse For |
|----------|------|-----------|
| `xcorr_temporal_localization()` | `coherence_localization.py:1122` | Reference architecture for event sync TL |
| `wpli_temporal_localization()` | `coherence_localization.py:896` | Base for multi-resolution PLV |
| `_coherence_surrogates()` | `coherence_localization.py:822` | Surrogate pattern for event sync |
| `_min_event_filter()` | `coherence_localization.py:447` | Post-processing (replaced by BOCPD in Phase 4) |
| `_fill_gaps()` | `coherence_localization.py:459` | Post-processing |
| `generate_coupling_gate()` | `synthetic.py` | Semi-synthetic test data |
| `inject_eeg_coupling_spatial()` | `synthetic.py:1272` | EEG semi-synthetic |
| `_estimate_ar()` | `kim_filter.py` | AR residualization for BL/Pose |
| `circular_shift_surrogate()` | `surrogates.py` | Base for IAAFT comparison |

## Verification Strategy

Each phase has its own semi-synthetic validation. The overall verification is:

1. **Run existing baseline tests** with current methods → record hit/FA at κ=0.10, 0.20, 0.40
2. **Add each upgrade incrementally** and re-run → measure delta in hit/FA
3. **Expected outcomes**:
   - Event sync alone: comparable or slightly worse than cross-product/PLV (different signal, not necessarily stronger)
   - Metric fusion: +5-15 pp hit rate at κ=0.10-0.20 (if metrics are partially independent)
   - IAAFT surrogates: -1-3 pp FA at same threshold, allowing lower threshold → +2-5 pp hit
   - BOCPD: similar hit/FA but better temporal precision (onset/offset timing)
   - Multi-resolution PLV: +5-10 pp hit at κ=0.10 for EEG
4. **Cumulative effect**: push BL detection threshold from κ=0.10 to κ=0.07-0.08; push EEG threshold from κ=0.20 to κ=0.15

## Key Literature References

| Method | Paper | Key Finding |
|--------|-------|-------------|
| Event Synchronization | Quian Quiroga et al. 2002 (Phys Rev E) | Handles different rates, directional, O(N²_events) |
| IAAFT Surrogates | Schreiber & Schmitz 2000 (Physica D) | Preserves power spectrum + amplitude distribution |
| Robust BOCPD | Altamirano et al. 2023 (ICML) | 10× faster, provably robust, closed-form conjugate posteriors |
| Multi-scale synchrony | Likens & Wiltshire 2020 (SCAN) | Multi-resolution captures different coupling timescales |
| Behavioral → Neural causality | Koul et al. 2023 (NeuroImage) | BL/Pose sync Granger-causes neural sync |
| Cross-modal effect sizes | Ohayon & Gordon 2025 (Behav Brain Res) | r=0.18-0.32 cross-modal correlations |
| IDTxl (validation) | Wollstadt et al. 2019 (JOSS) | Gold standard multivariate TE |
| STOK (future) | Pascucci et al. 2020 (PLOS Comp Bio) | Adaptive Kalman for TV-VAR, ~25ms resolution |
| MMHP (future) | Wang et al. 2022 (Ann Appl Stat) | Hawkes process for sporadic social interaction |
| Flexible synchrony theory | Gordon et al. 2025 (Psych Review) | Dynamics matter more than magnitude |

---

# Implementation Results & Current Status (2026-03-25)

## Architecture Evolution

The original plan focused on pushing detection thresholds via literature methods. Through implementation and testing on real data, the architecture evolved into an **event-anchored multimodal synchrony** framework:

```
Stage 1: Cross-product multi-lag bank (z-scored AUs)
  → continuous coupling mask + data-driven lag estimate
  → hierarchical Bayesian shrinkage (prior 2.5±1.0s)

Stage 2: Per-event co-occurrence detection (RAW AU composites)
  → individual expression catalogs per person
  → co-occurrence detection with who-led analysis
  → causal attribution via onset analysis (mimicry / shared_stimulus / coincidence)
  → LSL timestamps for cross-modal anchoring
```

### Two Use Cases
1. **Session-level outcome prediction**: synchrony metrics → predict therapy outcomes
2. **Cross-modal temporal coupling**: BL co-occurrence events as anchors → query EEG/ECG/Pose at those moments (event-triggered synchrony analysis)

## Phase Results

### Phase 1: Event Synchronization — ✅ Implemented, ❌ Not viable for dense AUs
- Quian Quiroga algorithm implemented and tested
- At AU event rates (~0.5/s/ch), random coincidences dominate — SNR ≈ 0.002
- Works only for very sparse events (<0.05/s)
- **Superseded by co-occurrence approach** in `bl_coupling.py` which uses raw AU composites + prominence-based event detection

### Phase 2: Within-Modality Metric Fusion — ❌ Dropped
- Event sync doesn't provide a useful second metric to fuse with cross-product
- The co-occurrence approach replaces this entirely

### Phase 3: IAAFT Surrogates — ✅ Implemented, ❌ Not useful for BL
- IAAFT preserves amplitude distribution perfectly (KS=0.000) but spectrum match poor for zero-inflated AUs (PSD diff up to 3.85)
- Null z-distribution wider than circular shift (std=1.277 vs 1.177)
- **Circular shift is already optimal for zero-inflated AU signals**
- EEG (near-Gaussian) untested at scale — may still help there

### Phase 4: BOCPD — 🔲 Pending
### Phase 5: Multi-Resolution PLV — 🔲 Pending

## What Was Built Instead

### Two-Stage BL Pipeline (`cadence/significance/bl_coupling.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Raw AU composites | ✅ | Z-scored composites detect noise, not smiles. Prominence ≥ 0.3 in raw composite detects visible expressions |
| Multi-composite | ✅ | smile (AU43+44+17), brow (AU2+3+4), frown (AU25+26), speech (AU17+22+23), general |
| Hierarchical lag shrinkage | ✅ | Normal-Normal conjugate toward population prior (2.5±1.0s) |
| Iterative lag refinement | ✅ | 3-pass: wide → narrow from matches → re-test |
| Co-occurrence detection | ✅ | Symmetric (not source→target), who-led analysis |
| Causal attribution | ✅ | Onset analysis: mimicry vs shared_stimulus vs coincidence |
| Role-aware analysis | ✅ | Therapist/patient from XDF metadata |
| Event-mimicry coupling model | ✅ | `inject_bl_event_coupling()` for semi-synthetic |

### Critical Bug Found: Z-Score Inflation
- Z-scored composites made tiny AU fluctuations (0.05 raw) appear as 4σ events
- Previous corpus significance (Fisher p=0.044 for T→P) was detecting speech artifacts, not smiles
- **Fix**: use raw AU values with absolute prominence thresholds

## Real Data Results (y_06)

| Metric | conv_1 (397s) | conv_2 (308s) |
|--------|--------------|--------------|
| Patient smiles | 29 | 27 |
| Therapist smiles | 10 | 23 |
| Co-occurrences | 5 (p=0.070) | 9 (p=0.595) |
| Patient led | 3 | 7 |
| Therapist led | 2 | 2 |
| Mimicry events | 1 (conf=0.93) | 0 |
| Shared stimulus | 4 | 5 |

### Real Expression Characteristics
- Major smile events (prominence ≥ 0.3 raw): every ~26 seconds
- Smile composite = AU43 (mouthSmileL) + AU44 (mouthSmileR) + AU17 (jawOpen)
- Response lag: 2.9s mean, 3.4s median
- Most co-occurrences are **shared_stimulus** — both responding to conversation content
- Genuine causal mimicry is rare but identifiable via onset analysis

## Pending Work

| Item | Notes |
|------|-------|
| Corpus rerun with raw composites | Debug zero-event issue in corpus test |
| BOCPD regime detection | Still promising for Stage 1 z-score segmentation |
| Multi-resolution PLV for EEG | Still promising for EEG at κ=0.1 |
| Cross-modal event-triggered analysis | Use BL co-occurrence LSL timestamps to query EEG/ECG |
| Session-level summary metrics | Clean output table for outcome prediction |
| Pseudo-dyad null | Cross-session pairing for stronger null |

## Key Design Decisions

1. **Raw AU values, not z-scored**: Prominence ≥ 0.3 in raw composite ensures visible expressions
2. **Co-occurrence framing**: Most synchrony is shared_stimulus. Symmetric detection with who-led is more honest than forcing source→target
3. **Hierarchical lag shrinkage**: Population prior (2.5±1.0s) regularizes without overriding strong data
4. **Causal attribution as bonus**: ALL co-occurrences matter for outcome prediction; mimicry is the rare strong signal
5. **LSL timestamps**: Every co-occurrence anchors cross-modal queries

## Files

| File | Purpose |
|------|---------|
| `cadence/significance/bl_coupling.py` | Production two-stage pipeline |
| `cadence/surrogates.py` | IAAFT + circular shift + Fourier surrogates |
| `cadence/synthetic.py` | Event-mimicry coupling injection model |
| `cadence/data/xdf_loader.py` | Role detection (therapist/patient) |
| `scripts/_test_bl_corpus.py` | Corpus-level analysis |
| `scripts/_test_bl_event_catalog.py` | LSL timestamp catalog for video verification |
| `scripts/_test_bl_two_stage.py` | Single-session pipeline test |
| `scripts/_test_event_sync.py` | Event sync experiments (archived) |
| `scripts/_test_iaaft.py` | IAAFT validation (archived) |
