# CADENCE-Lite — Design Spec

**Date:** 2026-04-13
**Status:** Draft for review (user + comp-neuro collaborator)

## 1. Motivation

CADENCE V11 has grown to 28 observation channels + 7 transition covariates feeding an rSLDS state-inference model with per-modality burst extraction, transfer entropy, LZ complexity, graph-spectral covariates, and four-state switching dynamics. This complexity is justified for the production pipeline but obstructs the simpler scientific question:

> *Is there detectable, time-varying, condition-dependent dyadic coupling between therapist and patient using the most literature-validated metrics for the four modalities we have, evaluated by the most conventional statistics?*

CADENCE-Lite answers that question with a stripped-down, per-modality, no-state-inference pipeline. It is **not a successor** to V11 — it is a parallel reference implementation intended for collaborator review and as a clean baseline against which V11's added complexity can be argued.

## 2. Design Principles

1. **Literature-grounded** — every channel cites a published method; every statistical choice cites a published convention.
2. **Per-modality only** — no cross-modal coupling, no graph operations, no shared latent factors.
3. **No state inference** — no rSLDS, no HMM, no clustering. Outputs are continuous coupling timecourses + per-condition statistics.
4. **Inspectable in isolation** — each channel can be evaluated, debated, kept, or replaced without affecting any other channel.
5. **Raw-data semi-synthetic validation** — sensitivity is measured by injecting known coupling at the most upstream signal level the pipeline consumes, then recovering it.
6. **Conservative defaults** — when in doubt, use the convention from a published hyperscanning paper rather than a CADENCE-internal innovation.

## 3. Architecture

```
Raw multimodal session (EEG, ECG, AUs, pose)
        │
        ▼
[Per-modality feature extraction] ── reuses existing CADENCE code
   ├─ EEG @ 256 Hz → Morlet CWT (per-electrode, θ + α bands)
   ├─ ECG → RR intervals @ 4 Hz → bandpass (LF + HF) → Hilbert envelope
   ├─ Face AUs @ 30 Hz → Morlet CWT (per-AU, expression + speech bands)
   └─ Pose joints @ 30 Hz → upper-body velocity scalar
        │
        ▼
[Per-modality dyadic coupling]
   ├─ EEG: per-electrode wavelet coherence (P1↔P2 homotopic)
   ├─ ECG: HF/LF envelope cross-correlation (60s windows)
   ├─ Face: per-AU wavelet coherence (expression + speech bands)
   └─ Pose: multi-lag (±5s) velocity cross-correlation
        │
        ▼
[Per-bin surrogate z-scoring] ── 200 circular shifts of P2 per channel
        │
        ▼
[Output rate aligned to common 2 Hz; per-condition segment z-tracks]
        │
        ├─→ Per-condition statistics (mixed-effects, permutation, FDR)
        ├─→ Pseudo-dyad null check (full pipeline rerun)
        └─→ κ-detection battery (raw-data injection, AUC vs κ)
```

Single linear pipeline; no closed-loop, no cross-modal feedback, no state-inference module.

## 4. Channel Definitions (7 channels)

All 7 channels output at common **2 Hz** (500 ms bins, 0.5 s hop) to allow alignment for visualization. Each channel is a coupling-strength z-score (real coupling vs 200 circular-shift surrogates) at each bin within each session segment.

### 4.1 EEG α wavelet coherence (per-electrode)

- **Input:** 14-channel EEG @ 256 Hz (Emotiv EPOC montage).
- **Preprocessing:** Morlet CWT (`w=5`) at 30 log-spaced frequencies covering 4–30 Hz on the GPU, reusing `cadence/significance/bl_wavelet.py`'s CWT machinery (matches the V7 convention of 30 log-spaced bins per scalogram, with band edges adjusted from 0.3–8 Hz for face to 4–30 Hz for EEG to span θ + α + low β margin).
- **Coupling math:** Wavelet coherence per electrode pair (P1 electrode *e* ↔ P2 electrode *e*, homotopic), averaged across frequencies in the **8–12 Hz** band. Smoothed with 0.5 s Gaussian temporal kernel (matching `bl_wavelet.py` convention).
- **Output:** 14 per-electrode coupling timecourses at 2 Hz, plus a Stouffer-z-combined single-channel summary timecourse.
- **Citation:** Davidesco 2023 (per-electrode CCorr with frontal hub finding); Hakim 2023 review (wavelet coherence as the dominant EEG hyperscanning method); Grinsted 2004 (canonical wavelet coherence implementation).

### 4.2 EEG θ wavelet coherence (per-electrode)

- Identical pipeline to §4.1, but coherence averaged in the **4–7 Hz** band.
- **Citation:** Chen 2022 (theta interbrain coupling in mindfulness/social context).

### 4.3 ECG HF envelope coupling (RSA / PNS proxy)

- **Input:** Polar H10 RR intervals → resampled to 4 Hz IBI series.
- **Preprocessing:** Bandpass 0.15–0.4 Hz (HF/RSA band, respiratory-cardiac coupling timescale), Hilbert transform → instantaneous amplitude envelope.
- **Coupling math:** Pearson correlation between P1 and P2 envelopes, sliding 60 s window, 0.5 s hop.
- **Output:** Single-channel coupling timecourse at 2 Hz.
- **Important interpretation flag:** PNS/RSA synchrony is **negatively** valenced in the therapy-outcome literature (Mayo 2021 ES = −0.21; Wilson/Kiecolt-Glaser 2018: HRV synchrony during conflict predicts higher inflammation). Sign of effect must be reported explicitly; do not collapse with LF.
- **Citation:** Mayo 2021; Behrens 2020; Wilson/Kiecolt-Glaser 2018.

### 4.4 ECG LF envelope coupling (SNS-leaning)

- Identical pipeline to §4.3, bandpass **0.04–0.15 Hz** (LF band).
- LF reflects mixed SNS+PNS but leans sympathetic. Closer to the "positive valence" branch of cardiac synchrony literature than HF.
- **Citation:** Behrens 2020 (SCL / sympathetic synchrony predicts cooperation; HR did not).

### 4.5 Face expression-band wavelet coherence (0.5–2 Hz)

- **Input:** 52 facial AUs @ 30 Hz, restricted to AFFECT_AUS subset (10 AUs covering smile/frown/brow muscles).
- **Preprocessing:** Low-pass 8 Hz (Butterworth 4th order, removes tracker noise above face-tracker information bandwidth — Jeganathan 2022 finding); Morlet CWT (`w=5`) at 30 log-spaced frequencies 0.3–8 Hz on GPU.
- **Coupling math:** Wavelet coherence per AU pair (P1 AU *i* ↔ P2 AU *i*), averaged across AUs and across 0.5–2 Hz frequency band. 0.5 s Gaussian temporal smoothing.
- **Output:** Single-channel coupling timecourse at 2 Hz.
- **Citation:** Jeganathan 2022 eLife (CWT on facial AUs); Fujiwara 2018/2020 (expression-band coupling in dyadic interaction); validated in CADENCE V7 (AUC=0.78 at κ=0.4 with real smile injection, 42 pseudo-dyad pairs).

### 4.6 Face speech-band wavelet coherence (2–7 Hz)

- Identical CWT pipeline to §4.5, restricted to **jaw/lip AU subset** (~6 AUs covering speech articulators), coherence averaged in **2–7 Hz** band.
- **Citation:** Audiovisual speech literature (syllabic timescale); Jeganathan 2022 (facial frequency-band decomposition).

### 4.7 Pose multi-lag velocity coupling

- **Input:** 40 joint groups @ 30 Hz.
- **Preprocessing:** First derivative → joint velocities; mean across upper-body joints → single upper-body-velocity scalar per participant.
- **Coupling math:** Per 60 s window (0.5 s hop), compute cross-correlation at lags ±5 s in 0.5 s steps, take **max\|r\|** over the lag bank.
- **Output:** Single-channel coupling timecourse at 2 Hz; max-over-lag is the per-bin coupling magnitude. The lag at which the max occurs is also retained as an auxiliary directionality summary (not a separate channel — used only in per-condition descriptive reporting).
- **Citation:** Ramseyer 2011 (MEA convention); validated in CADENCE V8.2 (conv vs med p=0.006).

### 4.8 Window summary

| # | Channel | Internal window | Per-element resolution preserved |
|---|---|---|---|
| 1 | EEG α wavelet coherence | CWT (w=5, ~5-cycle window) | 14 electrodes |
| 2 | EEG θ wavelet coherence | CWT (w=5) | 14 electrodes |
| 3 | ECG HF envelope coupling | 60 s sliding | single |
| 4 | ECG LF envelope coupling | 60 s sliding | single |
| 5 | Face expression coherence | CWT + 0.5 s Gaussian smooth | 10 AUs |
| 6 | Face speech coherence | CWT + 0.5 s Gaussian smooth | ~6 AUs |
| 7 | Pose multi-lag velocity | 60 s sliding, ±5 s lag bank | single |

## 5. Per-Bin Surrogate Normalization

**Method (uniform across all 7 channels):**

1. Compute the real coupling value `r_real(t)` at each 2 Hz bin.
2. Apply 200 random circular shifts to P2's pre-coupling signal (CWT coefficients for §4.1, 4.2, 4.5, 4.6; envelope for §4.3, 4.4; velocity scalar for §4.7).
3. For each shift, recompute the coupling value at each bin.
4. Per bin, compute `z(t) = (r_real(t) − mean_null(t)) / std_null(t)`.

**Notes:**

- Circular shifts preserve the marginal distribution and autocorrelation structure of P2's signal, so the null isolates *between-participant* dependency from any *within-participant* structure.
- Per-bin (rather than session-level) surrogate normalization gives a non-stationary null appropriate for per-condition stratification.
- For multi-element channels (§4.1, 4.2, 4.5, 4.6), surrogate z is computed per-element first, then the Stouffer-z aggregator is applied at the per-element-z level to produce the channel summary timecourse.
- GPU acceleration (PyTorch) for the surrogate loop, modeled on `bl_wavelet.py` (~0.6 s for 200 surrogates on a single segment in the existing implementation).

## 6. Statistical Layer

### 6.1 Two protocol-specific mixed-effects models

The dataset comprises two protocols sharing 4 of 6 conditions but differing in the intervention block:

- **Meditation protocol:** `base_EO, base_EC, conv_1, meditate_B, meditate_K, conv_2`
- **PE protocol:** `base_EO, base_EC, conv_1, PE_1, PE_2, conv_2`

Splitting along protocol yields two internally **balanced** designs (every session of a protocol has all 6 of its conditions). This avoids the missing-cells problem of any single global model.

**Model M (Meditation)** — meditation-protocol sessions only:
```
coupling_z ~ condition + (1 | dyad)                     # ECG/Face/Pose channels
coupling_z ~ condition + (1 | dyad) + (1 | electrode)   # EEG channels
condition ∈ {base_EO, base_EC, conv_1, meditate_B, meditate_K, conv_2}
```

**Model P (Psychoeducation)** — PE-protocol sessions only, identical structure:
```
condition ∈ {base_EO, base_EC, conv_1, PE_1, PE_2, conv_2}
```

### 6.2 Pre-registered contrasts (5 per model, identical set)

Each model is evaluated on the same 5 contrasts, all referenced against eyes-closed rest:

| ID | Contrast | Scientific question |
|---|---|---|
| C1 | conv_1 vs base_EC | Does conversation elevate coupling above resting? |
| C2 | intervention_1 vs base_EC | Does the first intervention block differ from rest? (meditate_B or PE_1) |
| C3 | intervention_2 vs base_EC | Does the second intervention block differ from rest? (meditate_K or PE_2) |
| C4 | conv_2 vs base_EC | Does post-intervention conversation differ from rest? |
| C5 | conv_2 vs conv_1 | Does conversation change pre→post intervention? |

Total: 5 contrasts × 2 models × 7 channels = **70 tests**. FDR (Benjamini-Hochberg q < 0.05) applied across the entire 70-test family.

### 6.3 Permutation tests

For every contrast in every model:

1. Shuffle condition labels **within session** 1000 times (each session's labels are permuted only over the 6 conditions of its protocol).
2. Refit the mixed-effects model on each shuffle.
3. Recompute the contrast's test statistic.
4. The null distribution is built from those 1000 statistics; report the exact one-sided permutation p-value.

Within-session label permutation preserves session-level random effects (per-dyad variability, per-electrode variability) — the null is "if condition labels were random within this dyad/electrode, what would we see?".

Parallelization via `joblib` over the 1000 shuffles per contrast.

### 6.4 Reported per contrast

- Estimate of the contrast (mixed-model coefficient, in coupling-z units)
- 95% confidence interval (parametric)
- Cohen's d (standardized effect size)
- Parametric p-value (mixed model)
- Permutation p-value (exact)
- FDR-adjusted q-value (across the 70-test family)

### 6.5 Multiple-comparisons strategy summary

| Level | Adjustment |
|---|---|
| Within model × channel | Pre-registered contrasts only (5); no fishing |
| Across the 70-test family | Benjamini-Hochberg FDR, q < 0.05 |
| (Optional) within channel narratives | Additional within-channel FDR if collaborator wants per-channel framing |

### 6.6 Software

- **Mixed-effects models:** `pymer4` (lme4 via rpy2). Requires R installation in MCCT conda env. Backup: `statsmodels.MixedLM` (pure Python). Wrapped behind a thin `cadence/lite/stats/models.py` interface so swapping is one line.
- **Permutation testing:** custom code on top of the model wrapper, parallelized via `joblib` over 1000 shuffles per contrast.
- **FDR:** `statsmodels.stats.multitest.multipletests`.

## 7. Validation (Tier 2)

### 7.1 Pseudo-dyad null check

Run the entire pipeline (timecourses → mixed models → permutation tests → contrast tests) on **cross-session pseudo-dyads**:

- For each meditation session, pair P1 with P2 from a different meditation session (round-robin: session i's P1 with session ((i+1) mod n_med)'s P2).
- Same for PE.

**Expected result:** every per-condition contrast (all 70 tests) is null after FDR.

Any contrast that survives signals one of: (a) a pipeline bug, (b) a session-level confound (e.g., condition-locked artifact in a single participant), or (c) an indexing error in segment alignment.

Compute cost ≈ one full pipeline run.

### 7.2 Semi-synthetic sensitivity battery — raw-data injection

For each channel, take pseudo-dyad bases (guaranteed null at κ=0), inject known coupling at **κ ∈ {0.0, 0.1, 0.2, 0.3, 0.4}** at the **most upstream signal level the pipeline consumes**, run the entire pipeline, measure detection AUC.

| Channel | Raw injection level | Injection method | Code source |
|---|---|---|---|
| EEG α/θ wavelet coherence | 256 Hz EEG signal, per-electrode | Narrowband signal mixing | `cadence/synthetic_v82.py` (exists) |
| ECG HF/LF envelope | RR interval series @ 4 Hz | Narrowband signal mixing in LF/HF bands on RR series | **NEW** `cadence/lite/validation/synth_ecg.py` |
| Face expression coherence | 30 Hz AU timeseries (AFFECT_AUS) | Expression-band signal mixing | `cadence/synthetic_v82.py` (exists) |
| Face speech coherence | 30 Hz AU timeseries (jaw/lip AUs) | Speech-band signal mixing | **NEW** extension of expression injection (same module pattern, restricted AU subset, different band) |
| Pose multi-lag velocity | 30 Hz joint positions | Position mixing | `cadence/synthetic_v82.py` (exists; documented limitation — surrogates absorb continuous mixing) |

**Output per channel:** κ-detection curve (AUC vs κ) with 95 % bootstrapped CI.

## 8. Code Structure

```
cadence/lite/
    __init__.py                     # re-export pipeline entry
    config.py                       # 7-channel definitions, bands, ROIs, windows
    coupling/
        eeg_wavelet.py              # per-electrode wavelet coherence (NEW)
        ecg_envelope.py             # HF/LF bandpass + Hilbert + cross-corr (NEW)
        face_wavelet.py             # thin wrapper around bl_wavelet.py
        pose_multilag.py            # extracted from _run_scaffold_v82.py
    surrogates.py                   # uniform per-bin circular-shift z-scoring
    timecourses.py                  # 7-channel pipeline orchestration → 2 Hz output
    stats/
        models.py                   # pymer4 wrapper (statsmodels backup)
        contrasts.py                # 5 pre-registered contrasts × 2 models
        permutation.py              # within-session label-shuffle permutation
        fdr.py                      # Benjamini-Hochberg
    validation/
        pseudo_dyad.py              # cross-session pairing + full-pipeline rerun
        semisynthetic.py            # κ injection driver + AUC computation
        synth_ecg.py                # NEW — RR-series narrowband injection
    visualization/
        timeline.py                 # per-channel per-condition coupling z-plots
        contrast_summary.py         # forest plot of all 70 contrasts
        kappa_curves.py             # AUC-vs-κ per channel

scripts/
    _run_lite_pipeline.py           # main driver: --session, --all
    _run_lite_pseudo_dyad.py        # null check
    _run_lite_semisynthetic.py      # κ-detection battery

results/lite/
    timecourses/<session>/<channel>.npz       # 2 Hz coupling z per condition
    contrasts.json                            # all 70 contrast results
    pseudo_dyad_contrasts.json                # null check results
    semisynthetic/<channel>_kappa_auc.csv     # detection curves
    figures/                                  # all PDF/PNG outputs
```

**Imports from existing cadence/:** `cadence.significance.bl_wavelet` (CWT GPU machinery), `cadence.data.xdf_loader`, `cadence.data.preprocessors`, `cadence.data.alignment`, `cadence.constants`. No imports from `cadence.significance.rslds_*`, `cadence.significance.lz_complexity`, `cadence.significance.spectral_graph`, or `cadence.significance.directed_burst_coupling`.

## 9. Deliverables

1. **`results/lite/contrasts.json`** — flat machine-readable table: for each (channel, model, contrast): estimate, parametric p, permutation p, FDR q, Cohen's d, 95 % CI.
2. **Forest plot** of all 70 contrasts grouped by channel — single figure.
3. **κ-detection curves** — one figure per channel, AUC vs κ.
4. **Pseudo-dyad null table** — every contrast should be null after FDR.
5. **Per-session per-condition coupling-z plots** — 7 × 6 grid per session showing each channel's coupling timecourse over each condition segment.
6. **EEG topographic maps** — per condition, per band, mean coupling z across the 14 electrodes (since per-electrode resolution is preserved, this is free).

## 10. What's Deferred (and why)

These are intentionally *not* in CADENCE-Lite. Each can be added back if the lite pipeline reveals a question that needs the extra machinery.

| Deferred component | Why deferred from lite | When to add back |
|---|---|---|
| Per-modality asymmetry / "who leads" channels | Adds 7 channels and a directionality concept; doubles complexity | After lite findings replicate; literature says directionality is clinically meaningful (V11 finding: patient-led conv vs therapist-led PE) |
| rSLDS state inference | Lite's whole point is to *not* commit to a state model | After collaborator validates the channel set; state inference is V11's contribution |
| Burst event detection (V8.2/V11) | Continuous z-scores answer the same questions more interpretably for a baseline | If collaborator wants peri-event analyses |
| LZ complexity (V10) | Not a coupling metric — it's a per-participant complexity that *modulates* coupling | When extending to psychedelic context (REBUS — complexity is the most validated psychedelic biomarker) |
| Graph theory (V10) | Excluded per user's explicit request | Likely never for the collaborator-review version |
| Transfer entropy (V11) | Powerful but heavy compute and complex to interpret | After symmetric coupling story is established |
| EDA, respiratory | User-specified modality set is EEG/ECG/Face/Pose only | EDA is the strongest literature signal (#1 hardware addition); resp is extractable from Polar H10 (FMRR) |
| Speech / voice | Not in modality list; literature shows pitch synchrony is *negatively* valenced (r = −0.20) | If linguistic / semantic synchrony added later (Lord 2015 d = 0.62 for empathy) |
| Cross-modal coupling pathways | Per-modality only in lite | Core extension once channel-level findings are validated |
| Cross-protocol comparison (Model D / interaction) | Adds a fourth model and a missing-cell statistical concern | If C4 or C5 results look meaningfully different between Models M and P |

## 11. Sample Sizes (to confirm with current data)

- Current cohort: ~17 sessions total (per `project_v11_post_whitening_findings.md`); exact protocol breakdown to confirm at implementation time.
- Per CLAUDE.md: meditation sessions named in `y_06, y_17, y_19, y_11, y_04, y_24` (6); PE in `y_01, y_05, y_10, y_32, y_41` (5). Additional sessions added since.
- Power consideration: with n_med ≈ 9–10 and n_pe ≈ 7–8, mixed-effects with single random intercept is well-powered for medium-large effects (d ≥ 0.6) on within-session contrasts. Smaller effects may require extending the cohort.
- Permutation null space: each session has 6 conditions, so per-session label permutations = 6! = 720; for n sessions the global within-session permutation space is 720^n (e.g., 720^8 ≈ 7 × 10^22). The 1000-shuffle sample is well below this combinatorial ceiling for all realistic n, so the permutation null is fine-grained at all sample sizes we care about.

## 12. Open Questions for the Collaborator

The following are defensible defaults the collaborator might want to revisit:

1. **EEG ROI vs per-electrode aggregation.** Lite goes per-electrode (14 electrodes per band) with `(1 | electrode)` random factor and Stouffer-z aggregation for visualization. Alternatives: fixed frontal ROI (Davidesco-style), data-driven hub electrode identification.

2. **Pose at 60 s windows.** Matches V8.2 validated default. Some MEA literature uses 30 s windows. Would need re-validation if changed.

3. **AU subsets** (AFFECT_AUS for expression, jaw/lip for speech). The expression subset is a published convention; the speech subset is a CADENCE choice based on biomechanics. Collaborator may want to discuss data-driven AU selection or per-AU reporting.

4. **HF / LF band boundaries.** Standard HRV convention (Task Force 1996) but slightly different cutoffs are sometimes seen. Worth confirming.

5. **Surrogate count = 200.** Convention from V11 / V8.2; trades off precision (z-score noise floor ~ 1/√200 ≈ 0.07) vs compute. Higher counts (1000–10 000) are sometimes seen for tight tail estimates.

6. **Pseudo-dyad pairing scheme** (round-robin within protocol). Alternatives: random pairing across protocols, all-pairs.

7. **Contrast set.** 5 contrasts is a deliberate floor. Collaborator may want to add (e.g., conv_1 vs base_EO, intervention_2 vs intervention_1).

8. **`pymer4` vs `statsmodels.MixedLM`.** Functionally equivalent for these models; lme4 syntax familiarity drives the recommendation. Switching is a one-line change in the wrapper.

## 13. Implementation Notes

### 13.1 New code components

- Per-electrode EEG wavelet coherence (`cadence/lite/coupling/eeg_wavelet.py`) — wraps `bl_wavelet.py`'s GPU CWT for EEG signals at 256 Hz; computes per-electrode-pair (homotopic) wavelet coherence; produces θ-band and α-band per-electrode timecourses + Stouffer-aggregated channel summaries.
- ECG envelope coupling (`cadence/lite/coupling/ecg_envelope.py`) — bandpass + Hilbert + sliding cross-correlation in HF (0.15–0.4 Hz) and LF (0.04–0.15 Hz) bands.
- Face speech-band wavelet coherence — extension of the existing expression-band path in `bl_wavelet.py`, restricted to jaw/lip AU subset, 2–7 Hz band.
- Pose multi-lag — extracted from `_run_scaffold_v82.py` into `cadence/lite/coupling/pose_multilag.py`.
- Stats wrapper (`cadence/lite/stats/`) — `pymer4` mixed-model fitter, within-session permutation, BH FDR.
- Validation (`cadence/lite/validation/`) — pseudo-dyad full-pipeline rerun, semi-synthetic κ-injection driver, **new** RR-series narrowband injection module.
- Visualization (`cadence/lite/visualization/`) — per-channel timeline plots, all-contrast forest plot, AUC-vs-κ curves, EEG topographic maps.
- Driver scripts (`scripts/_run_lite_pipeline.py`, `_run_lite_pseudo_dyad.py`, `_run_lite_semisynthetic.py`).

### 13.2 Environment

- MCCT conda env (Python 3.11).
- `pymer4` is currently NOT installed (verified at spec-writing time). Installation requires R: `conda install -c conda-forge r-base r-lme4`, then `pip install pymer4`. Windows installation should be validated as part of the implementation.
- Backup: `statsmodels.MixedLM` (pure Python, no R dependency) — wrapper interface is one-line swap.

### 13.3 Compute

- PyTorch FFT-based CWT in `bl_wavelet.py` is GPU-accelerated. Per-electrode EEG wavelet coherence scales linearly in number of electrodes (14 vs the existing per-AU face implementation). GPU resources of the existing pipeline are sufficient.
- Surrogate compute is the bottleneck (200 shifts per channel per bin); the existing `bl_wavelet.py` uses Welford accumulation on GPU and is fast enough for full-session per-band processing.

### 13.4 Test data

- Primary single-session test target: `y_06` (also CADENCE's V11/V10/V7 reference session).
- Pseudo-dyad pairs use the cross-session round-robin scheme described in §7.1.
