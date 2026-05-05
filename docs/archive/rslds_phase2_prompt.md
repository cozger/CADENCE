# RSLDS Phase 2 Implementation — Session Start Prompt

## Task

Begin implementation planning for the RSLDS (Recurrent Switching Linear Dynamical System) Phase 2: fitting the multi-population RSLDS model to the 7D observation matrix produced by the Phase 1 scaffold. Before implementing, critically review the surrogate methodology decision (full-session vs per-condition surrogates) using the literature priors and empirical data.

## Current State

### Phase 1 Scaffold: Complete
`scripts/_run_rslds_scaffold.py` produces per-session 7D z-timecourses at 2 Hz:
1. **EEG volt_amp** — Stouffer across theta/alpha/beta, per-band breakdown available
2. **BL expression** — wavelet coherence (0.5-2 Hz, affect AUs), smile/speech sub-groups
3. **BL state** — wavelet coherence (<0.5 Hz, affect AUs)
4. **ECG SNS** — multi-channel [IBI_dev, HR_accel, QRS_amp] cross-product
5. **ECG PNS** — RMSSD cross-product
6. **Respiratory rate** — EDR-extracted rate cross-product (3-method fusion)
7. **Pose** — 40-channel joint feature cross-product

All modalities use circular-shift surrogates for z-scoring. Cross-modal lag analysis and flexibility metrics (transition count, dwell time, Shannon entropy, kurtosis, skewness) are computed.

### Outputs per session
- `results/rslds/{session}/rslds_scaffold_timeline.png` — 8-panel visualization (7 modalities + coupling state strip)
- `results/rslds/{session}/rslds_scaffold_ztimecourses.npz` — z-traces + masks on common 2 Hz grid
- `results/rslds/{session}/rslds_scaffold_results.json` — flexibility metrics, cross-modal lags, per-condition breakdowns

### Sessions available: 12 cached sessions (13 XDFs, y26 broken)
Y_10, Y_41, y01, y04, y05, y11, y24, y_03, y_06, y_17, y_19, y_32

### Pseudo-dyad FPR calibration: Complete (132 pairs from 12 sessions)
Per-condition pseudo-dyad p95 thresholds:

| Condition | EEG | theta | alpha | beta | BL expr | ECG SNS | Resp | Pose |
|---|---|---|---|---|---|---|---|---|
| conv_2 | **39.1%** | 28.7% | 23.9% | 26.3% | 5.5% | 14.9% | 6.4% | 18.0% |
| conv_1 | 27.9% | 20.3% | 13.4% | 13.7% | 10.1% | 5.6% | 8.8% | 2.2% |
| base_EO | 60.0% | 29.0% | 15.2% | 11.2% | 22.3% | 6.2% | 6.5% | 30.3% |
| base_EC | 42.3% | 13.0% | 44.1% | 11.8% | 9.3% | 6.3% | 3.8% | 7.7% |
| meditate_B | 11.3% | 3.0% | 4.5% | 3.0% | 10.2% | 3.7% | 5.2% | 9.1% |
| meditate_K | 9.5% | 6.4% | 6.9% | 4.6% | 3.3% | 4.1% | 5.7% | 22.3% |

Session-wide pseudo-dyad p95: EEG 12.9%, BL expr 5.6%, ECG SNS 4.3%, Resp 3.7%, Pose 10.8%

Real-dyad sessions exceeding session-wide p95: Resp (y01 7.0%*, y_06 4.0%*, y_17 7.3%*), ECG SNS (y04 6.1%*), ECG PNS (y04 9.3%*), Pose (y_32 17.7%*)

**Critical finding: No session's EEG coupling exceeds the pseudo-dyad null at session-wide level.** The per-condition FPR shows pseudo-dyads produce 39% coupling in conv_2 windows — meaning EEG "coupling" measured by volt_amp cross-product with full-session circular-shift surrogates largely reflects shared task-driven amplitude changes, not genuine inter-brain coupling.

## The Surrogate Methodology Question (MUST REVIEW BEFORE IMPLEMENTING)

### The debate: full-session vs per-condition surrogates

**Full-session surrogates** (current implementation): Circular shifts across the entire session. A shift might move conv_2 data into meditate_B's time window. This creates a lenient null — easy to beat because different conditions have different EEG characteristics. Result: inflated z-scores that pseudo-dyad pairs also show.

**Per-condition surrogates** (implemented but reverted): Circular shifts within each condition window only. The null for conv_2 is built only from conv_2 data. This controls for shared task effects — but it also removes the condition-dependent signal that the RSLDS is designed to capture.

### Arguments FOR full-session surrogates + RSLDS
1. The RSLDS models coupling as condition-dependent via transition covariates. It WANTS the z-timecourse to have condition-dependent structure.
2. Per-condition surrogates would flatten the z-timecourse, removing the very signal the RSLDS models.
3. The pseudo-dyad FPR is the proper null — it shows what coupling fractions unrelated pairs produce. The RSLDS should find that real dyads have different TEMPORAL DYNAMICS (when coupling activates, dwell times, state transitions) than pseudo-dyads, even if the overall coupling fraction is similar.
4. Cohen 2021: State-like (time-varying) synchrony predicts alliance; trait-like (aggregate) does not. The RSLDS captures state-like dynamics.

### Arguments FOR per-condition surrogates
1. If the z-timecourse is inflated by shared task effects, the RSLDS input is contaminated. It would learn "conversations have coupling" as an artifact, not a signal.
2. The cross-product metric conflates "both people's alpha suppresses during talking" with "P1's specific alpha fluctuation at time t predicts P2's at time t+lag."
3. Per-condition surrogates give a cleaner input where z>2 means "coupling above what two independent people in the same condition would show."

### Arguments from the literature
- **Pseudo-dyad controls are the standard null** (Sened 2025, Kojovic 2024, HyPyP). They control for shared task effects, shared environment, shared hardware artifacts. CADENCE already has this.
- **Cross-brain GLM** (Hakim 2023, 215 studies): Uses participant B's neural data as a PREDICTOR for participant A, with time-lag. This is fundamentally different from cross-product — it asks "does B predict A above autoregressive baseline?" The EWLS regression in CADENCE V2 did this but was replaced by the simpler volt_amp cross-product.
- **Amplitude co-modulation avoids phase inflation pitfalls** (Zimmermann 2024) but is still susceptible to shared task-driven amplitude changes.
- **Context-dependent interpretation required** (Uhl 2025): The condition is not just a confound — it's part of the signal.
- **Flexibility > aggregate** (Gordon 2025): The temporal dynamics of coupling (transitions, dwell times, entropy) predict outcomes better than mean coupling. This is what the RSLDS captures.

### What to decide
Should the RSLDS receive:
(a) Full-session z-timecourses (current) with condition effects baked in, relying on pseudo-dyad FPR and RSLDS condition covariates to separate signal from artifact?
(b) Per-condition z-timecourses where each segment is independently z-scored against within-condition surrogates?
(c) Something else — e.g., return to the V2 cross-brain GLM approach where B's signal predicts A's above autoregressive baseline, which naturally controls for shared task effects?
(d) **Two-track approach**: output BOTH z_shared_state (slow component, 8% of variance) AND z_coupling (residual, 92% of variance) as separate RSLDS inputs?

## Literature Synthesis: Shared States vs Genuine Coupling

### The three-way distinction (Hasson 2012, Burgess 2013, Schilbach & Redcay 2025)
1. **Stimulus-driven shared responses (ISC)**: Both participants receive the same sensory input (therapist's voice, room acoustics). Brains respond similarly independently — no interaction needed.
2. **Shared neural state**: Both are in the same behavioral/cognitive state (both relaxed, both conversing). Same task demands → similar EEG configuration. NOT interaction-dependent.
3. **Genuine inter-brain coupling**: P1's specific neural fluctuation at time t influences P2's at time t+lag. Removing the interaction (pseudo-dyad) removes this coupling.

### Key paper: Burgess 2013 (Frontiers in Human Neuroscience)
- PLV and coherence are 77% sensitive to shared rhythmicity (task-driven phase variance), only 18% to actual coupling
- CCorr is 99% sensitive to genuine coupling, 0.3% to shared rhythmicity
- PLV detected 145 spurious theta connections in pseudo-pairs; CCorr had far fewer false positives
- CADENCE's volt_amp avoids the phase trap but is susceptible to shared AMPLITUDE trends across conditions

### Cross-brain GLM = CADENCE V2 architecture (Hakim 2023)
The cross-brain GLM uses participant B's neural data as a PREDICTOR for participant A, controlling for A's autoregressive dynamics. AR terms absorb condition-driven amplitude changes predictable from A's own history. Only the unique predictive information from B survives. **This is exactly CADENCE's V2 EWLS distributed-lag regression**, which was replaced by the simpler cross-product for the V6/V7/scaffold pipeline.

### Does shared state have clinical value?
- **Partially**: shared state is a PRECONDITION for coupling (In-Sync model, Koole & Tschacher 2016)
- **But NOT the active ingredient**: coupling is dyad-specific (Sened 2025), state-like dynamics predict alliance while aggregate does not (Cohen 2021), flexibility predicts outcomes better than mean coupling (Gordon 2025)
- **Clinical use as process marker**: "Are therapist and patient both engaged?" — useful for session monitoring but not for outcome prediction

### The recommendation from the literature
1. **Pseudo-dyad controls remain the gold standard** (Sened 2025, Kojovic 2024, HyPyP)
2. **Cross-brain GLM with AR baseline** naturally separates shared state from coupling at the measurement level
3. **Lagged analysis** preferred over lag-0 (Schilbach & Redcay 2025): shared state produces simultaneous correlation; genuine coupling produces directional time-lagged effects
4. **No published study decomposes all three components** (shared response, shared state, genuine coupling) from the same data — CADENCE could be first

### Decided approach: Post-hoc spectral decomposition (Option 1)

**Keep the scaffold pipeline exactly as-is** (full-session surrogates, volt_amp cross-product). After computing each z-timecourse, apply a 2-step spectral decomposition:

1. Low-pass the z-timecourse at 0.01 Hz (100s cutoff) → `z_slow` (shared state, ~8% of variance)
2. Subtract: `z_fast = z_original - z_slow` (coupling dynamics, ~92% of variance)

The RSLDS receives BOTH as separate observation dimensions per modality. `z_slow` tells the model what behavioral state the dyad is in (data-driven, replaces explicit condition labels). `z_fast` captures genuine temporal coupling dynamics — engagement episodes on the 10-100s timescale.

**Why this works**: Empirical spectral analysis of y_06 showed the condition-level drift is confined to <0.01 Hz (8% of power). The dominant coupling signal peaks at 0.027 Hz (~37s period) and lives entirely in the 0.01-0.1 Hz band (75% of power). A 0.01 Hz cutoff cleanly separates these.

**No changes to surrogate computation**: Full-session circular shifts remain. The z-timecourse is computed identically to current. The decomposition is a post-processing step — two lines of scipy `butter` + `sosfiltfilt`.

**Validation**: Compare `z_fast` coupling fractions against pseudo-dyad `z_fast` coupling fractions. The shared-state inflation should be eliminated, making real-dyad coupling fractions distinguishable from pseudo-dyad.

### Fallback: Condition-mean demeaning (Option 2)

If the spectral decomposition doesn't cleanly separate (e.g., some condition effects have medium-frequency structure that bleeds into the coupling band), the alternative is:

Before computing the cross-product, subtract each participant's per-condition mean volt_amp from their z-scored channel within each condition window. This removes the ~0.94 z-unit condition swing while preserving all within-condition fluctuations. The surrogates then stay full-session (they work correctly because the condition trend is gone). The removed per-condition means become the shared-state channel.

This is more surgical than spectral filtering — it removes exactly the condition-level mean shift — but it requires knowing the condition boundaries at computation time, which the spectral approach does not.

### What the RSLDS sees (with Option 1)

Per modality, two observation channels:
- `z_slow`: smooth, condition-dependent, ~0.01 Hz bandwidth. High during conversation, low during meditation. Same for real and pseudo-dyads. Tells the model WHAT state they're in.
- `z_fast`: fluctuating, 10-100s dynamics. Contains genuine coupling episodes. Should differ between real and pseudo-dyads. Tells the model HOW WELL they're coupling.

For 7 modalities × 2 channels = 14D observation vector (or keep slow channels only for EEG where the inflation is worst, giving 8D).

### References
- Burgess 2013 — On the interpretation of synchronization in EEG hyperscanning studies (Frontiers)
- Hasson et al. 2012 — Brain-to-brain coupling: mechanism for creating and sharing a social world (Trends Cogn Sci)
- Schilbach & Redcay 2025 — Synchrony Across Brains (Annual Review of Psychology)
- Zimmermann/Ayrolles 2024 — Arbitrary methodological decisions skew IBS estimates
- Hakim et al. 2023 — Quantification of inter-brain coupling: A review (NeuroImage)
- Hamilton 2020 — Hyperscanning: Beyond the Hype (Neuron)
- Sened et al. 2025 — Inter-brain plasticity in psychotherapy
- Koole & Tschacher 2016 — Synchrony in Psychotherapy: In-Sync Model

## Key Files

### Source code
- `scripts/_run_rslds_scaffold.py` — Main scaffold script (1130 lines). 7D z-timecourses, per-condition breakdowns, pseudo-dyad FPR, GPU-accelerated.
- `cadence/significance/fast_cycles.py` — EEG coupling: cycle extraction, volt_amp cross-product, GPU-batched surrogates, `extract_all_volt_amp()`, `eeg_coupling_from_precomputed()` (1037 lines)
- `cadence/significance/bl_wavelet.py` — BL wavelet coupling: CWT, wavelet coherence, GPU surrogate z-scoring
- `cadence/data/preprocessors.py` — ECG feature extraction (v8 cache: reduced smoothing windows — IBI_dev 5s, RMSSD 5s, HR_accel 2s, HR_trend 10s)
- `cadence/data/alignment.py` — Session caching (v8), `discover_cached_sessions()`, `load_session_from_cache()`
- `scripts/_extract_respiratory.py` — 3-method EDR fusion (FMRR + AM + QRS slope), rate estimation
- `cadence/coupling/estimator.py` — V2 CouplingEstimator (EWLS regression, not currently used in scaffold)

### Documentation
- `docs/rslds_vision.md` — RSLDS architecture: multi-population, condition covariates, hierarchical across sessions
- `CLAUDE.md` — Project overview, pipeline architecture, validation status
- `/literature` skill — 235-paper distilled literature priors (invoke for specific questions)

### Results
- `results/rslds/` — Per-session timelines, z-timecourses, results JSON (12 sessions)
- `results/rslds/cross_session_summary.json` — Cross-session comparison + pseudo-dyad FPR

### Memory files (auto-loaded)
- `project_rslds_vision.md` — RSLDS design decision and implementation path
- `project_rslds_phase1_results.md` — Phase 1 scaffold results on y_06
- `project_eeg_upgrades_plan.md` — EEG: volt_amp is production metric, PLV/CCorr/envelope rejected
- `project_v6_pipeline.md` — V6: fast_cycles EEG + saliency BL + full-session timelines
- `project_v7_wavelet_pipeline.md` — V7: CWT wavelet BL coupling, AUC=0.78 at k=0.4
- `project_bl_wavelet_v7.md` — BL wavelet coherence architecture
- `project_ecg_derived_respiration.md` — EDR: rate only, no phase. Use rate cross-product.
- `feedback_empirical_not_theoretical.md` — Don't assume FPR from theory; measure empirically
- `feedback_semisynthetic_null.md` — Semi-synthetic MUST use pseudo-dyad base
- `feedback_meditation_protocol.md` — Both meditation blocks: patient silent eyes-closed, therapist speaks lightly
- `feedback_fix_at_source.md` — Fix preprocessing at source + re-cache; don't patch downstream
- `feedback_roles.md` — Always use therapist/patient roles, never P1/P2

## EEG Pipeline Details (for surrogate decision)

The EEG signal chain:
```
Raw EPOC (256 Hz, 14ch) → notch 50/60 Hz → bandpass 1-45 Hz → artifact mask (>100µV)
→ z-score (cache) → avg-reference per participant → z-score → per-band FFT bandpass
→ cycle extraction (find_peaks → peak-trough amplitude) → resample 2 Hz → per-channel z-score
→ cross-product(P1 × P2) → smooth σ=1.5s → circular-shift surrogates → z-score → Stouffer
```

Notable: No spatial filtering (Laplacian, CSD, ICA). No volume conduction mitigation beyond average reference. Two independent Emotiv EPOC headsets with separate hardware references (advantage: no shared reference artifact).

Validated: volt_amp z=+14.3 on y_06 conv_2 (per-segment surrogates). Phase coupling (PLV, cycle-PLV) null on real data. Semi-synthetic detection threshold κ≈0.20. Pseudo-dyad pairs show z≈0 per-segment.

## RSLDS Architecture (from vision document)

- **Observation**: 7D z-timecourses at 2 Hz (expandable to 9D with prosodic + linguistic)
- **Discrete state**: K=3-4 shared states (Disengaged, Emotional_alignment, Active_exchange, Deep_engagement)
- **Cross-population B matrices**: Directed cross-modal coupling per regime
- **Condition covariates**: conv_1, meditate_K, etc. modulate transition probabilities
- **Missing data**: Masked likelihood (y11, y24 have no ECG/Resp)
- **Hierarchical**: Population-level A, B, transition (shared); session-level emissions (unique)
- **Library**: ssm (lindermanlab/ssm) has multi-population RSLDS. dynamax lacks RSLDS.
- **Stress test risks**: Autocorrelation (fix: AR(1) emissions), non-Gaussian (fix: Student-t), edge effects

## Empirical Comparison: Full-Session vs Per-Condition Surrogates

Both surrogate methods were run on all 12 sessions + 132 pseudo-dyad pairs. Key comparison:

### Real-dyad coupling fractions (y_06 example)
| Modality | Full-session surrogates | Per-condition surrogates |
|---|---|---|
| EEG | 11.7% | 7.8% |
| BL expr | 0.7% | 0.7% |
| ECG SNS | 2.4% | 2.0% |
| Resp | 4.0% | 3.6% |
| Pose | 0.9% | 3.7% |

### Pseudo-dyad session-wide p95 thresholds
| Modality | Full-session surr | Per-condition surr |
|---|---|---|
| EEG | 12.9% | 13.2% |
| BL expr | 5.6% | 5.6% |
| ECG SNS | 4.3% | 4.1% |
| Resp | 3.7% | 4.0% |
| Pose | 10.8% | 10.4% |

The pseudo-dyad p95 thresholds are nearly identical between methods — the null is stable. But real-dyad coupling fractions drop with per-condition surrogates, especially EEG (11.7% → 7.8%). This means per-condition surrogates make the real signal HARDER to detect against the same null, because they remove condition-dependent amplitude that is genuinely shared between participants.

The RSLDS question: is that condition-dependent shared amplitude signal, or artifact? Per-condition surrogates assume artifact. Full-session surrogates + RSLDS condition covariates treat it as signal to be modeled.

NOTE: The scaffold currently has per-condition surrogates implemented (last code edit of 2026-03-28). The FIRST task in Phase 2 should be to revert to full-session surrogates (restore the single `eeg_coupling_timecourse()` call for the full session, and the full-session `extract_bl_segment()` + `surrogate_coherence_z()` calls), then add the spectral decomposition as a post-processing step. The git working tree has both versions — the full-session version from 2026-03-27 and the per-condition version from 2026-03-28.

## Shared-State vs Coupling Decomposition (Key Finding)

The cross-product can be mathematically decomposed:
```
CP(t) = μ_P1(cond) × μ_P2(cond)                    [shared state term]
      + μ_P1(cond) × δ_P2(t) + δ_P1(t) × μ_P2(cond) [condition-modulated noise]
      + Σ_ch δ_P1(t) × δ_P2(t) / n_ch              [genuine coupling term]
```

Where μ_P(cond) is the participant's mean volt_amp z-score in the current condition, and δ_P(t) is the fluctuation around that mean.

**The shared state term** is positive whenever both participants are in a condition that drives volt_amp in the same direction (both high during conversation, both low during meditation). This term is IDENTICAL for real and pseudo-dyads — it reflects universal neurophysiological responses (alpha suppression during eyes-open, beta activation during speech, etc.).

**The genuine coupling term** is the only one that should differ between real and pseudo-dyads. It requires P1's specific fluctuation at time t to co-vary with P2's.

### Proposed two-signal approach for RSLDS
Instead of choosing between full-session or per-condition surrogates, output BOTH:
1. **z_shared_state**: Low-pass filtered full-session z (<0.01 Hz). Captures condition-level co-modulation. Tells the RSLDS what behavioral state the dyad is in. Real and pseudo-dyads look similar.
2. **z_coupling**: Per-condition demeaned cross-product with per-condition surrogates. Captures genuine temporal coupling above shared state. Should be higher in real dyads than pseudo-dyads.

This gives the RSLDS two complementary inputs per modality: the shared state replaces explicit condition labels (data-driven), while the coupling channel is the properly-calibrated inter-brain signal.

### Per-band physiological basis for shared-state inflation
| Condition | Dominant band | Mechanism | Pseudo-dyad p95 |
|---|---|---|---|
| base_EO | theta (34%) | Eyes-open frontal theta | 60% |
| base_EC | alpha (41%) | Berger effect (eyes-closed alpha increase) | 42% |
| conv_2 | all bands (~25% each) | Speech motor, attention, arousal | 39% |
| meditate_B | none (~3-5%) | Individually variable meditation state | 11% |

Meditation conditions show low pseudo-dyad FPR because meditation is individually variable — there's no universal amplitude template, so unrelated participants don't co-modulate.

### Empirical decomposition (y_06 data)

**Condition-mean z-scores** span 0.94 units (conv_1: +0.60, meditate_B: -0.34). This means both participants' volt_amp rises ~0.6 SD during conversation and drops ~0.3 SD during meditation, creating the shared-state cross-product.

**Spectral decomposition of the z-timecourse itself:**
- Slow (<0.01 Hz, >100s periods): **7.9% of variance** — the condition-level shared state
- Medium (0.01-0.1 Hz, 10-100s): **75.0%** — engagement episodes (peak at 37s period)
- Fast (>0.1 Hz, <10s): **11.3%** — moment-to-moment fluctuations

The shared-state artifact is only 8-15% of the total signal. The dominant spectral band (75%) is 10-100s engagement episodes — NOT condition-level drift.

**After condition-mean subtraction:**
- 14% of coupling events (68/482) are net artifacts of condition means
- conv_2 coupling **survives** de-meaning (permutation p=0.0001) — genuine signal
- conv_1 is the most inflated (49% spurious), drops from rank 3 to rank 5
- meditate_B has 35 genuine coupling events **hidden** by negative condition mean
- Ranking of conditions changes — interpretation of per-condition coupling is unreliable without de-meaning

**Autocorrelation**: Both original and residual z-traces decay below 1/e at lag 8 (4.0s). The original has a slow pedestal (ACF ~0.04 at 15s lag) from condition trends; the residual drops to zero by 9s.

**Conclusion**: The two-signal decomposition is empirically justified. The slow component (8% of variance) is the shared state. The residual (92%) contains genuine coupling dynamics on the 10-100s timescale that the RSLDS should model.

## Implementation Constraints

- Environment: `conda activate MCCT` (Python 3.11, PyTorch, scipy, numpy, matplotlib, pyyaml)
- GPU: NVIDIA with CUDA, ~8 GB VRAM
- CPU: 32 logical cores (16 physical)
- DRY, SOLID, KISS. Parallelize, vectorize, GPU accelerate when appropriate.
- Fix preprocessing at source; don't patch downstream.
- Don't assume FPR from theory; measure empirically from pseudo-dyad pairs.
- Always use therapist/patient roles, never P1/P2.
