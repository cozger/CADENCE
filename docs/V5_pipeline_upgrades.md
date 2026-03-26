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

### Phase 3b: IAAFT for EEG — ❌ OOM / not needed
- 100 IAAFT surrogates × 460K samples exhausted memory (joblib parallelization)
- Even if fixed, circular shift already preserves more within-P1 structure (stronger null)
- IAAFT destroys cross-channel correlations → weaker null for pooled PLV

### Phase 4: BOCPD — 🔲 Pending

### Phase 5: Multi-Resolution PLV — ✅ Implemented, ❌ Neutral
- `multires_plv_temporal_localization()` added to `coherence_localization.py`
- Windows (2s, 5s, 10s, 20s) with auto scale-proportional smoothing (0.75×w)
- Max-over-scales with full surrogate calibration
- **Result**: Neutral for sustained coupling (−3 to +2 pp vs baseline)
- **Result**: Neutral for burst coupling (+1 pp at best)
- The 20s window dominates (62% of detections at κ=0.1)
- **Root cause**: Max-over-scales penalty (~0.3z) exactly offsets z_coupled gain
- Short windows (2s) have too few samples for meaningful PLV

### Phase 6: CCorr Metric — ✅ Implemented, ❌ Worse than PLV
- Circular correlation coefficient (Burgess 2013) added to `_coherence_windowed()`
- CCorr = correlation of sin(φ−μ) deviations, resistant to rhythmicity artifacts
- **Result**: 15-19 pp WORSE than PLV at all κ (26.7% vs 41.0% at κ=0.1 mixing)
- 4× slower (42s vs 10s) due to per-window circular mean computation
- Burgess's lower spurious rate doesn't compensate for ~40% sensitivity loss

### Phase 7: Burst-Triggered ITC — ✅ Implemented, ❌ Much worse than PLV
- `burst_itc_temporal_localization()` in `cadence/significance/burst_itc.py`
- Hilbert phase at P1 theta burst peaks, ITC across events in sliding windows
- **Result**: 30 pp worse than continuous PLV (8.3% vs 39.4% at κ=0.2)
- **Root cause**: Continuous PLV uses 5120 samples/window; burst ITC uses ~10 events
- √N advantage (√512 ≈ 22×) overwhelms 2× amplitude amplification at burst times

### Phase 8: Dual Coupling Model Comparison — ✅ Critical finding
- Three models tested: mixing (phase+amp), Kuramoto (phase-only), amplitude (amp-only)
- Three metrics: PLV, CCorr, envelope correlation
- **Critical finding**: Broadband mixing model was 2× too optimistic
  - Old baseline (broadband mixing + broadband PLV): κ=0.2 → 76.3% hit
  - Fair test (narrowband mixing + theta PLV): κ=0.2 → 44.4% hit
  - Kuramoto + theta PLV: κ=0.2 → 29.5% hit
- **Revised detection thresholds** (realistic narrowband theta, 14-ch EPOC):
  - Narrowband mixing: ~50% hit at κ≈0.25, ~80% at κ≈0.40
  - Kuramoto phase attractor: ~50% hit at κ≈0.35, ~80% at κ>0.50
- Literature confirms: "reliable estimation only at coupling > 0.3" (Yang et al. 2024)

### Phase 9: Real Data Theta Scan (y_06) — ✅ Reveals amplitude coupling
- Per-condition theta PLV (4-8 Hz, 20s window, avg-ref, all 14 channels)
- **meditate_K**: 8.3% coupling, z_max=2.48 (strongest theta PLV)
- Conversations: near-zero theta PLV (z_mean negative)
- base_EC: z_mean=+0.44 (possible alpha contamination)

### Phase 10: GPU Cycle Analysis (bycycle → fast_cycles) — ✅ Game-changing
- Bycycle (Voytek lab) provides per-cycle features: amplitude, period, symmetry, burst
- numpy 2.x bug fixed (read-only array in `detect_bursts_cycles`)
- Default burst thresholds too strict for consumer EEG; tuned: monotonicity=0.7, amp_consistency=0.4
- **Wrote `cadence/significance/fast_cycles.py`**: GPU-accelerated replacement, 50× faster
  - FFT bandpass on GPU (all channels simultaneously)
  - Per-cycle feature extraction (CPU, fast numpy)
  - GPU-vectorized cross-correlation + 200 surrogates
  - ~2-4s per condition vs bycycle's 4-10s+ per condition

**KEY FINDING — Amplitude co-modulation is the dominant inter-brain EEG signal:**

| Condition | volt_amp z | period z | symmetry z | burst z |
|-----------|-----------|----------|------------|---------|
| **conv_2** | **+7.24** | −1.46 | −0.12 | −0.95 |
| **meditate_K** | **+6.56** | −0.20 | −1.02 | +0.51 |
| base_EO | +3.55 | +0.48 | +0.79 | +1.58 |
| conv_1 | +3.10 | −0.59 | −0.93 | +1.51 |
| meditate_B | +0.69 | −0.03 | −0.11 | +0.90 |

- Amplitude coupling (z=6-7) is **2× stronger** than phase coupling (PLV z=2.5)
- Period, symmetry, burst co-occurrence show NO significant coupling
- Signal is genuine (not artifact): frontal-weighted spatial pattern, pseudo-dyad null is clean (z≈0), persists without avg-ref

**Spatial pattern (meditate_K):**
- Frontal R: mean r=+0.076 (AF4 r=0.10, F4 r=0.076)
- Frontal L: mean r=+0.044 (AF3 r=0.072, F3 r=0.055)
- Temporal: mean r=+0.036
- Parietal: mean r=+0.039
- Occipital: mean r=+0.033
- Right frontal 2× occipital — consistent with social/emotional processing

### Phase 11: Arousal Coupling Injection Model — ✅ Validated
- `inject_eeg_coupling_arousal()` added to `synthetic.py`
- Slow modulator (LP 0.5Hz of P1 global theta envelope) → lag → modulate P2 broadband
- Frontal-weighted spatial pattern (based on y_06 real distribution)
- Band-specific spatial modes: `'frontal'` (theta), `'occipital'` (alpha), `'centroparietal'` (beta)
- **Validated**: volt_amp z grows linearly with κ (0.16→3.85)
- **Selective**: period z≈0, symmetry z≈−0.85 (constant), burst z≈−0.8 at all κ
- **PLV cross-check**: PLV hit≈8% at all κ — no phase coupling created
- Detection threshold: volt_amp z>2 at κ≈0.20
- Calibration: y_06 real data (z≈6.5 at 688s) corresponds to κ≈0.5-0.6

### Phase 12: Multi-Band Analysis (θ + α + β) — ✅ Powerful
- `analyze_interbrain_cycles_multiband()` added to `fast_cycles.py`
- GPU-batched bandpass for all bands in one pass, joblib-parallel cycle extraction
- Stouffer combination across bands: `z_combined = mean(z_per_band) × √n_bands`
- ~0.3-0.5s per condition (vectorized cycle features + joblib + GPU surrogates)

**Multi-band y_06 real data (volt_amp z):**

| Condition | θ (4-8) | α (8-13) | β (13-30) | **Combined** |
|-----------|---------|----------|-----------|-------------|
| **conv_2** | +7.24 | +7.48 | **+9.97** | **+14.25** |
| **meditate_K** | **+6.56** | +3.89 | +2.94 | **+7.74** |
| conv_1 | +3.10 | +3.45 | +1.85 | +4.85 |
| base_EO | +3.55 | −0.26 | −1.39 | +1.10 |
| base_EC | +1.81 | +0.56 | −0.19 | +1.26 |
| meditate_B | +0.69 | −1.97 | −0.67 | −1.13 |

**Band profile differs by condition type:**
- **conv_2**: ALL bands coupled, beta strongest (z=9.97) — speech motor / turn-taking
- **meditate_K**: theta dominant (z=6.56), alpha moderate (z=3.89) — emotional attunement
- **conv_1**: theta + alpha — shared attention
- **Baselines**: weak/absent — no shared task
- **meditate_B (body scan)**: nothing — individual introspective activity

**Multi-band Stouffer is very powerful**: conv_2 goes from z=7.24 (theta alone) to z=14.25 (combined) — genuine independent information across bands.

### Validity Diagnostic — ✅ Signal is genuine

**Concern investigated**: Eyes-closed conditions (base_EC, meditate_B, meditate_K) don't all show alpha coupling. Suspicious?

**Findings:**
1. **Alpha power is asymmetric**: P1 (patient) has strong alpha in eyes-closed (1.02 base_EC, 0.70 meditate_K), but P2 (therapist) consistently has low alpha (0.09-0.19 all conditions). For amplitude co-modulation to register, BOTH participants need power fluctuations in the band. P2's flat alpha means nothing to correlate.
2. **Detrending has zero effect**: 3rd-order polynomial detrend changes z by ±0.01 — not a slow drift artifact. Circular-shift surrogates already handle non-stationarity.
3. **Pseudo-dyad null is clean**: All bands z≈0 (θ=−0.49, α=−0.72, β=−0.03) when pairing y_06 P1 with a different session's P2.

**Key insight**: Amplitude co-modulation requires BOTH participants to have power fluctuations in the detected band. The coupling is in the *modulation dynamics*, not the *presence of power*. meditate_K shows alpha coupling (z=3.89) because shared emotional attunement creates correlated alpha modulations that are absent during individual body scan (meditate_B) or passive eyes-closed rest (base_EC).

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

| Item | Priority | Notes |
|------|----------|-------|
| **Corpus-wide scan** | **✅ Done** | 8 sessions × 36 segments, 102s. meditate_K z=+3.5 avg, conv_2 variable |
| **Integrate fast_cycles into production pipeline** | **High** | Wire into main CouplingEstimator for session output |
| BL semi-synthetic calibration | Medium | Clustered pseudo-dyad works; co-occ scales with κ; attribution needs work |
| Cross-modal event-triggered analysis | Medium | BL co-occurrence LSL timestamps → query EEG amplitude coupling |
| Session-level summary metrics | Medium | Clean output table for outcome prediction |
| Event-triggered ITC with Hilbert phase | Low | For EEG characterization, not detection (burst ITC 30pp worse) |

## Key Design Decisions

1. **Raw AU values, not z-scored**: Prominence ≈ 0.3 in raw composite ensures visible expressions
2. **Co-occurrence framing**: Most synchrony is shared_stimulus. Symmetric detection with who-led is more honest than forcing source→target
3. **Hierarchical lag shrinkage**: Population prior (2.5±1.0s) regularizes without overriding strong data
4. **Causal attribution as bonus**: ALL co-occurrences matter for outcome prediction; mimicry is the rare strong signal
5. **LSL timestamps**: Every co-occurrence anchors cross-modal queries
6. **Amplitude co-modulation > phase coupling for EEG**: Real inter-brain EEG signal is shared theta power dynamics (z=6-7), not phase locking (z=2.5). PLV detects a secondary signal; volt_amp cross-correlation is the primary metric.
7. **Narrowband detection**: Broadband mixing model was 2× too optimistic. Realistic theta coupling threshold is κ≈0.25-0.30 on 14-ch EPOC.
8. **Continuous > event-triggered for EEG**: Continuous PLV/amplitude correlation always beats burst-triggered approaches due to √N advantage (5120 samples vs ~10 events per window).
9. **fast_cycles over bycycle**: GPU-native cycle analysis (50× faster) extracts the same features with identical results.

## What Did NOT Work for EEG Detection

| Approach | Result | Why |
|----------|--------|-----|
| Multi-resolution PLV | Neutral | Max-over-scales penalty = z gain |
| IAAFT surrogates | OOM / not needed | Circular shift already optimal |
| CCorr metric | −15 pp worse | Subtracting circular mean removes signal |
| Burst-triggered ITC | −30 pp worse | √N loss (10 events vs 5120 samples) |
| Envelope correlation (in mixing model) | 3.8× worse | κ² sensitivity vs PLV's κ |
| SNR-weighted aggregation | Risky | Can amplify spurious channels |
| ROI averaging | Hurts | Dilutes focal coupling |
| CaCoh/CCA | Overfits | C ≈ DOF on 14-ch |

## Files

| File | Purpose |
|------|---------|
| `cadence/significance/bl_coupling.py` | Production BL two-stage pipeline |
| `cadence/significance/fast_cycles.py` | **GPU cycle analysis — primary EEG coupling metric** |
| `cadence/significance/coherence_localization.py` | PLV/CCorr/envelope TL (CCorr added, multi-res added) |
| `cadence/significance/burst_itc.py` | Burst-triggered ITC (archived — worse than continuous) |
| `cadence/surrogates.py` | IAAFT + circular shift + Fourier surrogates |
| `cadence/synthetic.py` | All coupling injection models (mixing, Kuramoto, amplitude, **arousal**) |
| `cadence/data/xdf_loader.py` | Role detection (therapist/patient) |
| `scripts/_test_fast_cycles_y06.py` | GPU cycle analysis on real data |
| `scripts/_test_arousal_semisynthetic.py` | Arousal injection validation |
| `scripts/_test_metric_model_matrix.py` | 3×3 metric × model comparison |
| `scripts/_test_narrowband_fair.py` | Fair narrowband PLV comparison |
| `scripts/_test_multires_burst_eeg.py` | Multi-res + burst pattern tests |
| `scripts/_diag_spatial_amplitude.py` | Spatial pattern diagnostic (genuine vs artifact) |
| `scripts/_scan_y06_theta.py` | Per-condition theta PLV scan |
| `scripts/_test_bl_corpus.py` | Corpus-level BL analysis |
| `scripts/_test_bl_event_catalog.py` | LSL timestamp catalog |
| `scripts/_test_bl_two_stage.py` | Single-session BL pipeline test |
| `scripts/_test_event_sync.py` | Event sync experiments (archived) |
| `scripts/_test_iaaft.py` | IAAFT validation (archived) |
