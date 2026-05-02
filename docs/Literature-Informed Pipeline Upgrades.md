# CADENCE V5: Literature-Informed Pipeline Upgrades

## Context

CADENCE has two production temporal localization (TL) pipelines:

| Pipeline | Method | Key Metric | Primary File |
|----------|--------|------------|-------------|
| **EEG** | GPU cycle analysis — multi-band (θ/α/β) volt_amp cross-correlation | z=+3.6 avg (meditate_K), z=+14.25 peak (conv_2 combined) | `cadence/significance/fast_cycles.py` |
| **BL** | Two-stage: cross-product detection + raw AU co-occurrence characterization | Per-event mimicry catalogs, session p-values | `cadence/significance/bl_coupling.py` |

The original plan (below) attempted to push detection thresholds via literature methods. Most were tested and found unproductive. The game-changing finding was that **amplitude co-modulation (volt_amp) is 2× stronger than phase coupling (PLV z=2.5)** on real consumer EEG — a result not in any of the four literature searches, discovered empirically via bycycle-style cycle analysis.

### Original Bottlenecks (Framed Around PLV — Now Superseded)

**EEG at κ=0.1** (PLV: 46% hit): Limited by ~4 effective independent channels (spatial correlation on 14-ch EPOC). z scales linearly with κ. *Resolution*: fast_cycles volt_amp extracts a stronger signal (amplitude co-modulation) that PLV cannot access. The bottleneck was the metric, not the hardware.

**BL at κ=0.118 mixed scenario** (cross-product: 37% hit with multi-lag bank): Limited by AU decorrelation time (~0.2s) and 10% coupling duty cycle. *Resolution*: Two-stage pipeline separates continuous detection (Stage 1) from event characterization (Stage 2) using raw AU composites.

---

## Implementation Results

### Phase 1: Event Synchronization Module — SUPERSEDED

**Status**: `event_sync.py` was built and then DELETED. Superseded by the two-stage co-occurrence architecture in `bl_coupling.py`.

**What happened**:
- For **BL**: The formal Q&Q 2002 algorithm was replaced by a simpler, more interpretable co-occurrence pipeline: detect expression peaks (`find_peaks` on raw AU composites), match co-occurrences within a Bayesian-estimated lag window, attribute causality (mimicry vs shared_stimulus vs coincidence). This gives session-level p-values and per-event characterization — more useful than a temporal z-score for the V5 event-anchored architecture.
- For **EEG**: Burst-triggered ITC (`burst_itc.py`) was the analogous approach — restrict analysis to P1 burst events. Result: **30 pp worse than continuous PLV**. The √N loss (10 burst events vs 5120 continuous samples per window) overwhelms any SNR gain from conditioning on bursts.
- **Event matching for lag estimation** also failed: coincidental short-lag pairs overwhelm coupling pairs at 10% duty cycle.

**Lesson**: Event-level coupling detection loses to continuous methods when event rates are low relative to window sizes. The co-occurrence approach works for *characterization* (Stage 2) but not for *detection* (Stage 1).

**Files**: `cadence/significance/bl_coupling.py` (production), `cadence/significance/burst_itc.py` (EEG, inferior)

### Phase 2: Within-Modality Metric Fusion — TRIED, HURTS

**Status**: Tested for EEG; combining metrics dilutes the stronger signal.

**What happened**:
- **EEG envelope + PLV**: Envelope correlation z=0.32 vs PLV z=1.22. Amplitude envelopes scale as κ² (quadratic), PLV phase scales as κ (linear). Combined metrics **HURT** by diluting the phase signal with a weaker amplitude signal.
- **EEG cross-correlation + PLV**: Cross-correlation z=1.09 vs PLV z=0.88 at κ=0.1. Similar sensitivity, so combination adds nothing (correlated under null).
- **BL**: The two-stage architecture IS the fusion — cross-product for temporal detection (Stage 1), co-occurrence for event characterization (Stage 2). They serve different purposes rather than being combined into a single z-score.

**Lesson**: Stouffer combination only helps when metrics capture independent information. For EEG, PLV dominates all alternatives; for BL, cross-product dominates. The √2 Stouffer improvement is theoretical — in practice the weaker metric introduces noise that offsets the gain.

### Phase 3: IAAFT Surrogates — IMPLEMENTED, NOT USEFUL FOR EEG

**Status**: Code complete in `cadence/surrogates.py`. Not viable for EEG production use.

**What happened**:
- `iaaft_surrogate()` and `iaaft_surrogate_batched()` implemented with joblib parallelization.
- Integrated into `multires_plv_temporal_localization()` via `surrogate_method='iaaft'`.
- **EEG**: OOM for 460K-sample signals (iterative FFT/iFFT × 50 iterations × 100 surrogates). Even if memory were fixed, circular shift already preserves MORE signal structure (cross-channel correlations, amplitude distribution, AND power spectrum), making it a STRONGER null than IAAFT (which only preserves power spectrum + amplitude distribution per channel independently).
- **BL**: Works (shorter signals at 30 Hz) but circular shift is already adequate — BL false alarm rates are well-controlled.

**Lesson**: IAAFT is designed for univariate nonlinear time series. For multivariate EEG with cross-channel correlations, circular shift is the correct surrogate because it preserves the full multivariate structure. IAAFT destroys cross-channel relationships, making it a WEAKER null (easier to beat → more false alarms).

**Files**: `cadence/surrogates.py` (iaaft_surrogate, iaaft_surrogate_batched)

### Phase 4: Robust BOCPD for Regime Detection — NOT IMPLEMENTED, LIKELY UNPRODUCTIVE

**Status**: Not implemented. Strong evidence from related approaches that adaptive regime detection is counterproductive.

**Why it was deprioritized**:
- Every adaptive regime detection method tried (Gaussian HMM, SLDS, Kim filter EM, HSMM) either collapsed to degenerate solutions or was strictly worse than direct thresholding.
- **Direct threshold beats HMM by 40 pp** (85% hit / 8% FA vs 45% hit / 1% FA — V4 finding).
- The V3.5 comprehensive report documents 10 failed improvement attempts on regime detection. The fundamental issue: adaptive methods (Q>0, EM on sigma2, learned transitions) overfit to noise when the per-timepoint coupling signal is only 0.035 nats.
- BOCPD's "advantage" (adapting to local statistics) is exactly the property that causes overfitting in this domain.

**Lesson**: When the per-timepoint signal is very weak, simple global thresholds with temporal smoothing outperform all adaptive approaches. The coupling signal is too weak for Bayesian regime inference to distinguish from noise fluctuations.

### Phase 5: Multi-Resolution PLV for EEG — IMPLEMENTED, NEUTRAL

**Status**: Code complete in `coherence_localization.py:1152`. Does not improve detection.

**What happened**:
- `multires_plv_temporal_localization()` implemented with 4 window sizes (2s, 5s, 10s, 20s), max-over-scales, per-scale surrogate calibration, and IAAFT support.
- **Result**: Neutral. Max-over-scales penalty exactly offsets the z_coupled gain from shorter windows. The 20s window dominates because PLV scales with total number of coupled samples in the window — short windows have fewer samples and therefore lower PLV.
- The analogy to BL multi-lag bank was flawed: BL signals decorrelate in 0.2s (so different lags are nearly independent), but PLV at 5s and 20s windows are highly correlated (both measuring the same underlying phase coupling, just with different averaging).

**Lesson**: Multi-resolution is helpful when different scales capture genuinely independent information (like different lags for sparse BL signals). For PLV, the signal is the same at all scales — only the averaging duration differs.

**Files**: `cadence/significance/coherence_localization.py` (multires_plv_temporal_localization)

---

## What Actually Worked (Not in Original Plan)

### fast_cycles Multi-Band Amplitude Coupling — GAME-CHANGING

**Discovery**: Amplitude co-modulation (volt_amp) between participants' EEG is **2× stronger** than phase coupling (PLV) on real data. volt_amp z=6-7 vs PLV z=2.5 on y_06.

**Implementation**: `cadence/significance/fast_cycles.py`
- GPU-accelerated multi-band (theta/alpha/beta) cycle analysis
- Per-cycle features: amplitude, period, rise-decay symmetry, burst status
- Burst co-occurrence detection
- Stouffer combination across bands
- 50-100× faster than bycycle

**Corpus validation** (8 sessions, 36 segments):
- meditate_K (n=5): z=+3.5 — most consistent positive signal
- conv_2 (n=7): z=+1.0 — high variance (−6.7 to +11.2)
- Genuine signal confirmed: frontal-weighted, pseudo-dyad null clean

**Key insight**: The literature (and our initial focus) emphasized phase coupling (PLV/wPLI), but real inter-brain coupling on consumer EEG is dominated by amplitude co-modulation — likely shared arousal driving power fluctuations in both participants simultaneously.

**Files**: `cadence/significance/fast_cycles.py`, `scripts/run_corpus_scan.py`

### Cycle-PLV and Full Feature Characterization (2026-03-25)

**Question**: The bycycle decomposition yields per-cycle amplitude, period, rise-decay symmetry, and burst status. Only volt_amp was being used. Could cycle-derived phase reconstruction capture phase synchrony that Hilbert PLV misses?

**Implementation**: Piecewise-linear phase reconstruction from detected trough/peak landmarks (trough→0, peak→π, next_trough→2π), resampled to 2 Hz grid, PLV computed as |mean(exp(i·Δφ))| with circular-shift surrogates. Validated on synthetic coupled signals (z=+15 for phase-coupled, z≈0 for null).

**Corpus scan results (8 sessions, 36 segments, all features)**:

| Condition | n | **volt_amp** | **cycle_plv** | **period** | **rdsym** | **burst** |
|-----------|---|-------------|--------------|-----------|----------|----------|
| baseline | 1 | **+12.1** | -0.7 | +2.8 | -1.0 | **+8.4** |
| meditate_K | 5 | **+3.6** | +0.2 | -0.4 | +0.2 | +0.3 |
| conv_1 | 5 | +1.1 | +0.6 | +0.5 | -0.5 | +1.5 |
| conv_2 | 7 | +0.9 | -0.0 | -0.0 | -0.2 | +0.1 |
| meditate_B | 3 | -2.5 | +0.2 | +1.2 | +0.4 | +0.8 |
| base_EC | 5 | +0.7 | +0.6 | -1.1 | -0.2 | -0.1 |
| base_EO | 4 | +0.6 | -0.8 | +1.1 | +0.0 | +0.8 |

**Findings**:
- **cycle_plv**: No consistent signal. Condition averages all in [-0.8, +1.1]. The measurement is correct (synthetic validation), but there is no detectable phase coupling on real consumer EEG. This confirms the coupling is genuinely amplitude-based, not phase-based.
- **period** (frequency co-modulation): No consistent signal across conditions. Baseline z=+2.8 and meditate_B z=+1.2 are intriguing but don't replicate.
- **time_rdsym** (waveform shape): Near zero everywhere. Not informative.
- **burst_cooc**: baseline z=+8.4 is striking but driven by a single session (y04, which also had volt_amp z=+12.1). Not consistent across conditions.
- **volt_amp remains the only reliable EEG inter-brain signal** on this hardware. It's the only feature that consistently differentiates conditions and replicates across sessions.

**Why cycle-PLV doesn't help despite cleaner phase estimation**: The phase reconstruction IS cleaner than Hilbert (exact at landmarks, naturally gated to real oscillatory cycles). But the underlying coupling mechanism in real dyads is shared arousal driving power co-modulation — there is simply no phase coupling to detect. This rules out the entire class of "better phase estimation" approaches as a productive direction.

**All features are retained in the pipeline** (`fast_cycles.py` computes all 5 + burst_cooc, `run_corpus_scan.py` saves all to JSON). They may emerge in other paradigms (e.g., auditory entrainment, guided meditation with shared audio) where phase coupling is theoretically expected.

**Files**: `cadence/significance/fast_cycles.py` (cycle_plv, _reconstruct_cycle_phase), `results/corpus_scan_all_features.json`

### BL Two-Stage Pipeline — PRODUCTION ARCHITECTURE

**Implementation**: `cadence/significance/bl_coupling.py`
- Stage 1: Cross-product multi-lag bank on z-scored AUs → coupling mask + lag estimation
- Stage 2: Per-event characterization on RAW AU composites → co-occurrence detection, causal attribution (mimicry / shared_stimulus / coincidence), Bayesian hierarchical lag shrinkage
- Expression composites: smile, brow, frown, speech, general
- Session-level significance via surrogate co-occurrence rates

**Key findings from real data** (y_06):
- Real smile mimicry: 42% rate, 2.9s lag, events every ~26s
- Most co-occurrences are shared_stimulus — both responding to conversation
- Z-scored composites detect NOISE; must use RAW AU values with prominence ≥ 0.3

**Files**: `cadence/significance/bl_coupling.py`, `scripts/_test_bl_two_stage.py`

---

## Current State & Next Steps

### Critical Bug Fixed: AU Index Mapping (2026-03-25)

**All hand-defined BL composites had wrong AU indices** — used ARKit ordering instead of MediaPipe FaceLandmarker ordering. Verified against `YQP/core/face_processing/mediapipe_process.py`. The "smile" composite `(43, 44, 17)` was actually `mouthShrugUpper + mouthSmileLeft + eyeLookUpLeft`. The "frown" composite was detecting jaw movements. The "speech" composite was detecting eye movements.

**Fixed to correct MediaPipe indices:**
```python
'smile':  (44, 45),        # mouthSmileLeft + mouthSmileRight
'brow':   (3, 4, 5),       # browInnerUp + browOuterUpLeft + browOuterUpRight
'frown':  (30, 31),        # mouthFrownLeft + mouthFrownRight
'speech': (25, 33, 39),    # jawOpen + mouthLeft + mouthRight
```

**Impact**: Stage 1 (all 52 AUs) was NOT affected. All historical Stage 2 event results are unreliable. Corpus re-run with correct indices: **zero significant BL event coupling** across 11 segments (all composites, hand-defined and NMF-discovered). Previous "marginal" results were artifacts.

**Discovery method**: Corpus-level NMF independently found smile = `(44, 45)`. Discrepancy with hand-defined `(43, 44, 17)` triggered investigation. Confirmed against YQP source code.

### NMF Corpus Expression Discovery (2026-03-25)

Joint NMF on 1.47M samples (9 sessions, 18 recordings, 816 minutes of raw [0,1] blendshapes). k=6 components, 74.8% variance explained:

| Component | Top AUs | Interpretation |
|-----------|---------|----------------|
| Lip shrug | mouthShrugLower(74), mouthPressRight(33) | Lip compression / thought |
| Brow furrow | browDownR(25), browDownL(22), eyeSquintL(19) | Concentration / frown |
| Gaze shift | eyeSquintR(34), eyeLookOutL(33) | Looking left / thinking |
| Brow raise | browOuterUpL(23), browInnerUp(10) | Surprise / engagement |
| **Smile** | **mouthSmileL(11.5), mouthSmileR(11.1)** | **Clean bilateral smile — 10× gap to next AU** |
| Blink/gaze | eyeBlinkL(14), eyeLookDownL(13) | Blink + downward gaze |

**Key finding**: Corpus NMF produces much cleaner components than per-segment NMF. The smile component is purely bilateral mouth corners — no jaw/eye contamination. Validated the corrected AU indices.

Results cached: `results/corpus_bl_raw_cache.npz`, `results/corpus_nmf_k6.json`, `results/corpus_nmf_k8.json`.

### Stage 1 BL Cross-Product: Not Suitable for Expression Event Detection

**Problem identified**: Stage 1 cross-product over all 52 AUs detects coupling in **eye movement channels** (squint, blink, lookDown), not expressive channels (smile, brow, frown). At the highest z-score peaks, both people are just squinting/blinking with no expressive content. Confirmed by inspecting raw AU values at peak coupling timepoints.

**Attempted fixes and results:**
- Derivative (velocity instead of level): Still eye-dominated — blinks have the largest derivatives
- Expressiveness weighting (1/std per channel): Partially worked — surfaced a real smile coupling event (conv_1 t=155s, P1 smileL=0.48 → P2 smileR=0.63 with 2s lag, z=6.06). But Peak #1 (z=26) was still eye-dominated.
- Both fixes combined still missed a confirmed smile mimicry event at conv_2 t=279s (z=-0.04 at that timepoint despite clear P1 smile → P2 smile with 1s lag)

**Root cause**: Stage 1's 3s smoothing + 52-channel average buries discrete 1-2s expression events. It was designed for continuous co-modulation (like EEG amplitude coupling), not discrete facial events. Eye AUs (~20 channels) dominate the channel average regardless of weighting.

**Conclusion**: Stage 1 cross-product is the wrong tool for BL expression coupling. Stage 2 event detection + co-occurrence matching (which IS essentially ECA) is the correct approach. The problem was never the method — it was the wrong AU indices.

### What's Working Now

| Pipeline | Method | Status | Key Metric |
|----------|--------|--------|------------|
| **EEG amplitude** (PRIMARY) | fast_cycles multi-band volt_amp | Production, corpus-validated | z=+3.6 avg (meditate_K), z=+14.25 peak (conv_2) |
| **EEG full characterization** | fast_cycles (all 6 features) | Implemented, retained | volt_amp only signal so far |
| **EEG phase** (secondary) | PLV broadband 20s window | Available, not primary | 2× weaker than volt_amp on real data |
| **BL event detection** | Stage 2 co-occurrence (corrected AU indices) | Needs re-validation | Correct indices as of 2026-03-25; previous results unreliable |
| **BL Stage 1 cross-product** | 52-AU cross-product mask | **Deprecated for expression coupling** | Detects eye co-movement, not expression coupling |
| **NMF expression discovery** | Corpus-level joint NMF | Working, cached | 6 components, 74.8% explained |

### V6 Production Pipeline (2026-03-25) — DONE

Production pipeline implemented in `scripts/run_session_v6.py` and `scripts/run_all_sessions_v6.py`.

- **EEG**: `fast_cycles.py` multiband (theta/alpha/beta) cycle analysis with full per-band output (volt_amp, period, symmetry, burst co-occurrence, cycle_plv). GPU-accelerated, 200 surrogates.
- **BL**: `bl_coupling.py` saliency-based facial event detection (velocity norm on non-eye AUs) + smile confidence scoring (amplitude x dominance x bilaterality). No significance testing — shared smiles are facts with confidence scores.
- **Visualizations**: `cadence/visualization/v6_plots.py` — full-session timelines with condition markers, per-band EEG bar charts, shared smile timelines.

**y_06 validation results:**

| Condition | EEG combined z | Shared smiles | Rate (/min) |
|-----------|---------------|---------------|-------------|
| conv_2 | **+10.3** | **61** | **11.9** |
| conv_1 | **+8.3** | 12 | 1.8 |
| meditate_K | +4.3 | 7 | 0.6 |
| meditate_B | -7.6 | 12 | 1.1 |
| base_EO | +1.6 | 0 | 0 |
| base_EC | -4.9 | 0 | 0 |

### Alpha PLV During Body Scan Meditation — New Finding

meditate_B shows alpha cycle_plv z=+4.1 — the only feature x band combination above z=4 outside of volt_amp in conversations. All other features (period, symmetry, burst) and all other conditions show near-zero PLV.

**Literature confirms the mechanism (Hsu 2020, Balconi 2023, Zelano 2016):**
- Respiration modulates alpha phase through nasal airflow entrainment and phase-amplitude coupling
- Slow/guided breathing regularizes alpha phase dynamics (inspiration resets alpha phase)
- Breath-focused meditation produces greater inter-brain alpha coherence (Balconi 2023, N=15 dyads)

**Critical caveat (Burgess 2013):** Shared breathing during co-located meditation can produce **spurious interbrain PLV** — both people's alpha phases independently lock to their own respiratory rhythms, producing above-chance PLV without genuine brain-to-brain coupling. This is a respiratory confound.

**Control needed:** Extract respiratory rate from Polar H10 ECG (FMRR method). If respiratory phase predicts alpha PLV, it's respiratory-mediated. If alpha PLV persists after partialling out respiratory phase, it's genuine neural coupling.

**No body scan hyperscanning studies exist** — CADENCE would be first, but needs respiratory control.

**Key references:**
- Balconi et al. 2023 (Sci Rep): Dyadic inter-brain EEG coherence during interoceptive hyperscanning — alpha coherence increased during breath focus
- Hsu et al. 2020 (J Neurophysiol): Slow-paced inspiration regularizes alpha phase dynamics
- Zelano et al. 2016 (J Neurosci): Nasal respiration entrains human limbic oscillations
- Vieten et al. 2021 (Consciousness & Cognition): Joint meditation increases inter-subject alpha coherence
- Burgess 2013 (Front Hum Neurosci): Cautionary note — shared respiration can produce spurious interbrain PLV

### What's Next

1. **Cross-modal event-anchored analysis**: Use shared smile LSL timestamps → query EEG volt_amp at those moments. The core question: is EEG amplitude coupling elevated specifically during shared smiles?

2. **Respiratory control for meditation PLV**: Extract respiratory rate from Polar H10 (FMRR). Partial out respiratory phase from alpha PLV to disambiguate genuine neural coupling from respiratory-mediated spurious synchrony.

3. **Corpus-wide V6 analysis**: Run `run_all_sessions_v6.py` on all 9 sessions. Validate that conv > baseline pattern replicates.

4. **Coupling flexibility metrics**: Per Gordon et al. 2025 — entropy, DFA exponent, state transition counts on EEG volt_amp timecourse.

5. **Pose/backchannel detection**: Extend saliency-based detection to head pose (nods, turns) from MediaPipe pose data.

---

## Hawkes Process / Event-Based Detection Exploration (2026-03-25)

### Motivation
Attempted to improve BL temporal localization and event quantification using point-process methods from the literature: group-sparse multivariate Hawkes (Xu 2016), MMHP latent-state model (Wu 2022), and Event Coincidence Analysis (Donges 2016, Odenweller 2020).

### What Was Built
- `cadence/significance/hawkes_coupling.py`: NMF expression discovery, EM-based group-sparse Hawkes (Xu 2016 Algorithm 1), MMHP forward/Viterbi, per-pathway mutual Hawkes fitting
- NMF joint decomposition correctly discovers shared expression vocabulary (smile, squint+press, blink, etc.) from raw [0,1] blendshapes — 76% variance explained at k=6, stable across segments

### What Was Learned

**NMF expression discovery works well** — discovered a therapist-specific "squint+press" concentration face (eyeSquintR+L + mouthPressR) that hand-defined composites would miss. The NMF components are interpretable and stable.

**All event-based detection methods fail for BL because events are too sparse:**
- ~30-50 smile events per 400s segment. Continuous cross-product uses 52 AUs × 12,000 timepoints = 624,000 data points. Event-based methods use ~50 data points. The continuous approach has **12,000× more information**.
- Group-sparse Hawkes: within-person self-excitation dominates cross-person triggering. BIC prefers explaining P2 events via P2's own clustering rather than P1 triggering. Tested at all lambda values — either too many or zero cross-person pathways.
- MMHP: Same self-excitation dominance. Modified MMHP (self-excitation in both states) designed but not fully validated because the fundamental data sparsity issue remains.
- ECA: Analytical test gives p≈0.07 for 5 true mimicry events on top of 15 chance coincidences — insufficient power. Waiting-time surrogates (stronger null) would reduce power further.
- 2D per-component Hawkes (smile P1 → smile P2) correctly detects injected coupling in semi-synthetic (kernel=0.13 for P1→P2, 0 for P2→P1) — the model works, but requires pre-specifying which component to test, defeating the discovery purpose.

**The existing continuous cross-product already provides excellent BL temporal localization:**

| κ | TL AUC | Hit rate | FA rate |
|---|--------|----------|---------|
| 0.00 | 0.54 | 8.6% | 12.3% |
| 0.10 | **0.82** | 61% | 13% |
| 0.15 | **0.90** | 76% | 15% |
| 0.20 | **0.94** | 88% | 18% |
| 0.30 | **0.97** | 99% | 22% |

At κ=0.10 (detection threshold), AUC=0.82 — no event-based method with 50 events could match this. The 3s Gaussian smoothing + 5s min-event filter gives ~5-10s temporal resolution.

**FA rate creep at high κ (18-26%)** is caused by smoothing bleeding coupled windows into adjacent uncoupled periods. Potential improvements:
- BOCPD on the z-timecourse (now plausible — z_mean=2.44 at κ=0.20 is much stronger than the 0.035 nats that killed previous BOCPD attempts)
- Sharper smoothing kernel (rectangular instead of Gaussian)
- Adaptive threshold from local z-score statistics

### Key Insights from Literature Search

Three parallel literature searches (Hawkes self-excitation separation, residual analysis, ECA) found:

1. **The self-excitation dominance problem is well-known** (Aubrun et al. 2025: "cross-effects are usually one order of magnitude smaller than self-effects"). Solutions include score tests (Richards et al. 2024), nested LRT (Kim et al. 2011), and sequential calibration — all assume you only have event times. With continuous signals, these are unnecessary.

2. **ECA > ES for serial dependency** (Odenweller & Donner 2020): ECA's fixed window is robust to event clustering, while ES's adaptive window confounds synchrony with serial dependency. But ECA still has poor power with ~50 events.

3. **The fundamental issue**: Hawkes/MMHP/ECA methods assume you ONLY have event times (earthquake catalogs, neural spikes). CADENCE has full continuous blendshape signals at 30Hz. Reducing 624,000 data points to 50 discrete events discards 99.6% of the information. The continuous cross-product will always dominate for detection and TL when continuous data is available.

4. **NMF's value is discovery, not detection**: NMF finds expression types. The existing `bl_coupling.py` with physical prominence thresholds (≥0.3 raw) handles detection better because it gates on visible expressions rather than statistical fluctuations.

### What Remains Useful

| Component | Status | Future use |
|-----------|--------|-----------|
| `hawkes_coupling.py` NMF functions | Working | Corpus-level expression vocabulary discovery |
| `hawkes_coupling.py` Hawkes EM | Working | Event-sparse modalities (EDA peaks, respiratory events) where continuous cross-product is unavailable |
| `hawkes_coupling.py` MMHP forward/Viterbi | Working | Same — event-sparse modalities |
| ECA concept | Not implemented (unnecessary) | Could be useful for EDA SCR peak co-occurrence (truly event-sparse) |
| NMF-discovered "squint+press" composite | Discovered | Add to `EXPRESSION_COMPOSITES` for future runs |
| BOCPD on BL z-timecourse | Not implemented | Promising for reducing FA rate at high κ |

### Literature References from This Exploration

| Paper | Key Finding | Relevance |
|-------|------------|-----------|
| Xu, Farajtabar, Zha 2016 (ICML) | Group-sparse Hawkes with basis functions; EM + SGL | Implemented; works but dominated by continuous cross-product |
| Wu, Ward, Curley, Zheng 2022 (Ann Appl Stat) | MMHP separates active/inactive coupling states | Designed for dyadic interaction; useful for event-sparse modalities |
| Donges et al. 2016 (EPJST) | Event Coincidence Analysis — fixed window, binomial test | Robust to serial dependency; insufficient power for 50 events |
| Odenweller & Donner 2020 (Phys Rev E) | ECA > ES because ES confounds synchrony with serial dependency | Validates CADENCE's fixed-window co-occurrence approach |
| Kim et al. 2011 (PLoS Comp Bio) | Point-process Granger causality via nested GLM deviance test | Clean approach for testing cross-excitation controlling for self-excitation |
| Aubrun et al. 2025 (arXiv) | Sequential calibration: fit self first, then cross on residuals | Confirms two-stage approach; designed for weak cross-effects |
| Linderman & Adams 2014 (ICML) | Bayesian branching structure with spike-and-slab priors | Posterior attribution of events to self vs cross triggering |
| Lotz 2024 (arXiv) | Sparsity test with chi-bar-squared for boundary parameters | Correct asymptotics for testing α_cross = 0 |
| Chen et al. 2021 (Front. Neuroergon.) | N=236: envelope correlation > phase metrics for social closeness | External validation of amplitude coupling as primary EEG signal |

---

## Approaches Confirmed NOT Worth Pursuing

These have been empirically tested and found unproductive for this data:

| Approach | Why It Failed | Reference |
|----------|--------------|-----------|
| Any adaptive regime detection (HMM, SLDS, BOCPD, Kim EM) | Overfits when per-timepoint signal is 0.035 nats | V3.5 comprehensive report |
| Multi-resolution PLV | Penalty offsets gain; 20s window dominates | Phase 5 above |
| IAAFT surrogates for EEG | OOM; circular shift is stronger null for multivariate | Phase 3 above |
| Metric fusion (PLV + envelope, PLV + xcorr) | Weaker metric dilutes stronger one | Phase 2 above |
| Burst-triggered ITC | 30pp worse than continuous PLV (√N loss) | EEG upgrades memory |
| CCorr metric | 15-19 pp worse than PLV at all κ | EEG upgrades memory |
| Envelope correlation alone | z scales as κ² vs PLV's κ; 4× weaker | wPLI/PLV TL report |
| Event matching for lag estimation | Coincidental pairs overwhelm coupling at 10% duty | V4 TL report |
| SNR-weighted channel aggregation | Amplifies spurious channels; equal weights safer | V4 TL report |
| Iterative coefficient refinement | Overfits to false detections | V3.5 report |
| Coherence-based TL on 30 Hz behavioral | Per-window spectral DOF too low | V3.5 report |
| Direct dR2 thresholding | dR2 inverted at short tau | V3.5 report |
| Cycle-PLV (cycle-derived phase locking) | No phase coupling on real consumer EEG; coupling is amplitude-only | Corpus scan 2026-03-25 (retained for future paradigms) |
| Period correlation (frequency co-modulation) | No consistent signal across conditions | Corpus scan 2026-03-25 (retained) |
| Waveform symmetry correlation | Near zero everywhere | Corpus scan 2026-03-25 (retained) |
| Group-sparse Hawkes for BL detection | Self-excitation absorbs cross-excitation; BIC prefers self-only model | Hawkes exploration 2026-03-25 |
| MMHP for BL temporal localization | 50 events can't compete with 624,000 continuous samples; continuous cross-product AUC=0.82 at κ=0.10 | Hawkes exploration 2026-03-25 |
| ECA for BL detection | Insufficient power: 5 mimicry events on 15 chance coincidences → p≈0.07 | Hawkes exploration 2026-03-25 |
| NMF as runtime feature extractor for detection | IQR-based thresholds detect noise; hand-defined composites with prominence≥0.3 raw perform better | Hawkes exploration 2026-03-25 |
| Any event-based method for BL when continuous signal available | 50 events vs 624,000 samples = 12,000× less information; continuous always wins | Hawkes exploration 2026-03-25 |
| Stage 1 52-AU cross-product for BL expression coupling | Detects eye co-movement (squint/blink), not expressions. 3s smoothing buries 1-2s expression events. Derivative + expressiveness weighting partially helps but doesn't solve the fundamental issue. | BL investigation 2026-03-25 |

---

## Deferred Methods

| Method | Paper | Potential Use | Status |
|--------|-------|---------------|--------|
| **Copula-based cross-modal dependence** | — | Nonlinear dependence between EEG amplitude and BL coupling | Not tried. For multimodal fusion phase. |
| **PCMCI+ / Regime-PCMCI** (Tigramite) | Runge et al. 2019 | Causal discovery between modalities | Not tried. Could validate cross-modal directionality. |
| **IDTxl Transfer Entropy** | Wollstadt et al. 2019 (JOSS) | Gold-standard multivariate TE | Not tried. Offline validation only. |
| **BOCPD on BL z-timecourse** | Altamirano et al. 2023 (ICML) | Reduce FA rate in BL temporal localization | Not tried. **Now plausible** — z_mean=2.44 at κ=0.20 is 70× stronger than the signal that killed earlier BOCPD attempts (0.035 nats). Could sharpen coupling onset/offset detection. |
| **Hawkes/MMHP for event-sparse modalities** | Wu 2022, Xu 2016 | EDA SCR peaks, respiratory events — modalities where only event times are available, no continuous signal | Implemented in `hawkes_coupling.py`. Tested on BL (dominated by continuous cross-product). **Reserved for future EDA/respiratory integration** where continuous cross-product is unavailable. |
| **ECA for event-sparse modalities** | Donges 2016, Odenweller 2020 | Same as above — EDA/respiratory event co-occurrence | Not implemented. Concept validated; use waiting-time surrogates to control for clustering. |
| **MdCRQA** | — | Multidimensional cross-recurrence | Not tried. |
| **STOK adaptive Kalman** | Pascucci et al. 2020 | Time-varying VAR for EEG | All adaptive regime methods failed. Likely same fate. |
| **NMF corpus-level expression discovery** | — | Run NMF across corpus to discover dyad-specific expression types, then add to `EXPRESSION_COMPOSITES` | NMF works (discovered squint+press). Use as one-time discovery tool, not runtime extractor. |

---

## Key Literature References

| Method | Paper | Key Finding | CADENCE Outcome |
|--------|-------|-------------|-----------------|
| Event Synchronization | Quian Quiroga et al. 2002 (Phys Rev E) | Handles different rates, directional | Superseded by co-occurrence pipeline |
| IAAFT Surrogates | Schreiber & Schmitz 2000 (Physica D) | Preserves spectrum + distribution | Implemented; not useful for multivariate EEG |
| Robust BOCPD | Altamirano et al. 2023 (ICML) | Adaptive regime detection | Not implemented; all adaptive methods fail |
| Multi-scale synchrony | Likens & Wiltshire 2020 (SCAN) | Multi-resolution captures timescales | Implemented; neutral for PLV |
| Behavioral → Neural causality | Koul et al. 2023 (NeuroImage) | BL/Pose sync Granger-causes EEG | Motivates event-anchored cross-modal |
| Cross-modal effect sizes | Ohayon & Gordon 2025 (Behav Brain Res) | r=0.18-0.32 cross-modal | Realistic expectations for V5 |
| Flexible synchrony theory | Gordon et al. 2025 (Psych Review) | Dynamics > magnitude | Motivates coupling flexibility metrics |
| IDTxl (validation) | Wollstadt et al. 2019 (JOSS) | Gold standard multivariate TE | Deferred — offline validation only |
| Amplitude co-modulation | Yang et al. 2024 | Reliable IBS estimation only at coupling > 0.3 | Confirmed: narrowband threshold κ≈0.25-0.30 |
| Envelope correlation validation | Chen et al. 2021 (Front. Neuroergon.) | N=236: envelope corr > phase metrics for social closeness (alpha r=0.264, beta r=0.210) | External validation: amplitude coupling is most socially relevant EEG synchrony signal |
