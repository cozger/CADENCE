# EEG Burst Detection Parameters: Literature Review

**Date**: 2026-04-01
**Purpose**: Principled calibration of burst detection in `cadence/significance/fast_cycles.py`
**Problem**: Original CADENCE burst detector used permissive parameters (p25 amplitude, 0.7 monotonicity, 2 consecutive cycles), producing 56-61% beta burst occupancy — far above the literature-expected 15-25%.

## Key References (verified from full text)

### 1. Cole & Voytek 2019 — Bycycle cycle-by-cycle analysis
**Citation**: Cole SR & Voytek B (2019). "Cycle-by-cycle analysis of neural oscillations." *J Neurophysiology* 122(2):849-861. doi:10.1152/jn.00273.2019
**Source code**: https://github.com/bycycle-tools/bycycle

Bycycle code defaults (from `bycycle/objs/fit.py`):
```python
thresholds = {
    'amp_fraction_threshold': 0.,      # code default (tutorial recommends 0.3)
    'amp_consistency_threshold': .5,
    'period_consistency_threshold': .5,
    'monotonicity_threshold': .8,
    'min_n_cycles': 3
}
```

The code warns: *"No burst detection thresholds are provided. This is not recommended. Please inspect your data and choose appropriate parameters."*

Tutorial "more appropriate" parameter set (`plot_2_bycycle_algorithm.html`):
- amp_fraction=0.3, amp_consistency=0.4, period_consistency=0.5, monotonicity=0.8, min_n_cycles=3
- *"Adding a small amplitude fraction threshold (e.g. 0.3) helps remove some false positives."*

### 2. Tinkhauser et al. 2017 — 75th percentile standard
**Citation**: Tinkhauser G et al. (2017). "Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication." *Brain* 140(11):2968-2981. PMC5667742.

Methodology (exact quotes):
- *"Thresholds were defined in terms of percentiles of the DC corrected signal amplitude distribution."*
- The 75th percentile is the primary threshold shown in figures, with the threshold computed as *"the mean of the 75th percentile amplitudes across conditions"* applied as a common absolute threshold.
- *"The selection of a given percentile amplitude threshold to determine bursts is somewhat arbitrary, although previous work has shown that relative differences in burst properties during different conditions were preserved across various amplitude thresholds."* Tested 55th-90th.
- *"We did not consider bursts with durations shorter than 100 ms (less that about two cycles in duration) to limit the contribution of spontaneous fluctuations in amplitude due to noise."*
- Burst occupancy: *"27.0 ± 0.5% versus 16.1 ± 1.7%"* (OFF vs ON levodopa, p<0.001).

Signal processing: Wavelet transform (Morlet, width=10), 1-40 Hz at 1 Hz resolution, single 1 Hz bin at beta peak frequency. **Not** bandpass filtering at peak ± 3 Hz (this was erroneously reported in initial literature search).

### 3. Sherman et al. 2016 — Beta events are ~3 cycles
**Citation**: Sherman MA et al. (2016). "Neural mechanisms of transient neocortical beta rhythms." *PNAS* 113(33):E4885-E4894. PMC4995995.

Key findings (exact quotes):
- Threshold: *"We chose the 98th percentile of this power distribution as the threshold."*
- Duration: *"The duration of the high-power beta events was approximately three beta periods in each area (SI: 3.37 ± 0.12; IFC: 3.15 ± 0.13)."*
- Intermittent nature: *"In nonaveraged data, beta shows brief periods of high power falling off sharply to background activity levels."*
- *"Continuous bands of beta activity appeared only when the spectrograms were averaged across many trials."*
- *"Burst-like or intermittent periods of high beta power occurring stochastically within the time-averaged period could appear as continuous rhythms in averaged spectrograms, despite not ever actually being sustained."*

### 4. Shin et al. 2017 — Beta event rate and behavior
**Citation**: Shin H et al. (2017). "The rate of transient beta frequency events predicts behavior across tasks and species." *eLife* 6:e29086. PMC5683757.

Methodology (exact quotes):
- Threshold: *"Beta events were defined as local maxima in the trial-by-trial TFR matrix for which the frequency value at the maxima fell within the beta band (15-29 Hz) and the power exceeded a set cutoff"* — *"the power cutoff was set to be 6X the median power."*
- Duration (FWHM): *"event duration was confined to a restricted range around a stereotypical value (mean ± SEM across events: human detection 167 ± 1.6 ms; mouse detection 145 ± 1.3 ms; human attention 153 ± 1.4 ms)."*
- Spectrogram: 7-cycle Morlet wavelet, per-frequency median normalization from concatenated prestimulus windows.

### 5. Rayson et al. 2022 — Alpha is sustained, beta is bursty
**Citation**: Rayson H et al. (2022). "Detection and analysis of cortical beta bursts in developmental EEG data." *Dev Cog Neuroscience* 54:101069. PMC8816670.

Key findings (exact quotes):
- *"In both the adults and infants, lagged coherence in the alpha frequency range remained relatively high over a range of lags up to 7 cycles, but rapidly diminished after 2 cycles in the beta range."*
- *"Activity in the alpha band, therefore, appears to be rhythmic and oscillatory, whereas lagged coherence confirms that in both adults and infants, beta activity occurs predominantly as burst events."*
- Beta burst duration: C3: M=153.13ms (3.45 cycles), C4: M=162.48ms (3.49 cycles).
- **Alpha was excluded from burst analysis** because the lagged coherence evidence showed it is not burst-like.

### 6. BOSC/eBOSC/fBOSC — Principled cross-band detection
**Citations**:
- Kosciessa JQ et al. (2020). "Single-trial characterization of neural rhythms: Potential and challenges." *NeuroImage* 206:116331.
- Seymour RA et al. (2022). "Using OPMs to measure neural activity in standing, mobile participants." *NeuroImage* 263:119663. PMC9828710.

Parameters (from `BOSC_thresholds.m` source code):
- Duration: *"numcyclesthresh - duration threshold. A typical value is 3 cycles."* Both eBOSC examples use `threshold.duration = 3`.
- Power: *"percentilethresh - power threshold expressed as a percentile/100 of the estimated chi-square(2) probability distribution of power values. A typical value is 0.95."* **95th percentile is the BOSC/eBOSC default** (not 99th). fBOSC (Seymour 2022) used the 99th percentile as their specific choice.

## Parameter Comparison

| Parameter | CADENCE (original) | bycycle default | bycycle recommended | Tinkhauser 2017 | BOSC/eBOSC |
|---|---|---|---|---|---|
| **Amplitude** | >p25 (75% pass) | amp_frac=0.0 | amp_frac=0.3 | **>p75** | >p95 of chi2(2) |
| **Monotonicity** | >0.7 | **>0.8** | **>0.8** | N/A (wavelet) | N/A |
| **Min consecutive** | 2 | **3** | **3** | 100ms (~2 cycles) | **3 cycles** |
| **Period consistency** | >0.5 | >0.5 | >0.5 | N/A | N/A |

## Expected Burst Occupancy

| Band | CADENCE (original) | Literature expected |
|---|---|---|
| Theta (4-8 Hz) | 5-13% | 1-15% (low at rest, higher during cognitive tasks) |
| Alpha (8-13 Hz) | 19-36% | 15-30% EO, 30-50% EC (more sustained than bursty — Rayson 2022) |
| Beta (13-30 Hz) | **56-61%** | **15-25%** at p75 (Tinkhauser 2017) |

## Root Cause of Beta Over-Detection

Traced per-criterion pass rates for beta (ch0, y_06):
- `amp_ok >p25`: 75.0% pass (by definition — threshold too low)
- `period_ok >0.5`: 92.5% pass
- `in_band`: 100.0% pass
- `mono_ok >0.7`: **83.2%** pass (10-sample cycles too short for 0.7 to discriminate)
- All criteria: 63.1%
- After consecutive filter (2): **55.8%**

For comparison, theta (36-sample cycles): mono_ok passes only 34.9%, giving final 15.9%. The monotonicity criterion is selective for long cycles but useless for short ones.

## Calibrated Parameters for CADENCE

Applied to `extract_cycle_features()` in `cadence/significance/fast_cycles.py`:

### Final parameters: band-specific

Tighter criteria applied **only to beta** (>13 Hz) where short cycles (10 samples) made the original monotonicity threshold non-selective. Theta/alpha keep original permissive parameters — surrogate z-scoring normalizes base rate, so overcounting is harmless for relative comparisons.

| Parameter | Theta/Alpha | Beta | Rationale |
|---|---|---|---|
| Amplitude | >p25 (original) | >p25 (original) | Surrogates normalize base rate |
| Monotonicity | **>0.7** (original) | **>0.8** (bycycle default) | 10-sample beta cycles: 83%→52% pass at 0.8 |
| Min consecutive | **2** (original) | **3** (bycycle/BOSC/Sherman) | Beta events are ~3 cycles (Sherman 2016) |

### Design rationale

The amplitude threshold is kept at p25 (permissive) for all bands because:
1. Surrogate z-scoring normalizes out the absolute burst rate — overcounting affects real and surrogate equally
2. Condition comparisons are relative — a permissive threshold gives more events and more statistical power
3. The Tinkhauser p75 standard was designed for clinical STN LFPs with much higher SNR than consumer scalp EEG

Only beta gets tighter monotonicity and consecutive-cycle criteria because:
- At 10 samples/cycle, mono>0.7 passes 83% of cycles (non-selective). Mono>0.8 passes 52% (meaningful discrimination).
- Sherman 2016 showed stereotypical beta events are ~3 cycles. Doublets (2 cycles) are too short to be genuine beta bursts.
- Theta (36 samples/cycle) and alpha (24 samples/cycle) have enough samples per half-cycle for mono>0.7 to be selective (35% and 54% pass respectively).

### Parameter sweep results (y_06, ch0, P1)

| Parameters | Theta | Alpha | Beta |
|---|---|---|---|
| p25 mono>0.7 consec≥2 (original) | 15.9% | 32.9% | **55.8%** |
| **p25 band-specific (final)** | **15.9%** | **32.9%** | **31.1%** |
| p25 mono>0.8 consec≥3 (uniform strict) | 1.1% | 4.8% | 31.1% |
| p50 mono>0.8 consec≥3 | 0.9% | 3.5% | 15.9% |
| p75 mono>0.8 consec≥3 (Tinkhauser) | 0.4% | 1.5% | 5.3% |

### Note on Alpha and Theta

Rayson et al. 2022 excluded alpha from burst analysis entirely because lagged coherence evidence showed it is sustained oscillatory activity, not bursty (lagged coherence remains high to 7 cycles for alpha, drops after 2 for beta). Alpha "burst coincidence" is better interpreted as "co-occurrence of high-amplitude alpha epochs" rather than discrete event co-occurrence. **Beta is the most theoretically grounded band for burst coincidence** as genuinely transient events (Sherman 2016, Rayson 2022).

The per-channel rate gate in `eeg_burst_coincidence()` filters channels where either participant has <1% burst rate, and z-scores are clipped to [-10, 10] to prevent numerical overflow in sparse regimes.

## V11 Burst Coincidence Results (n=11 sessions, 2026-04-01)

Per-condition burst coincidence z (surrogate-calibrated, tau=500ms, 200 surrogates):

| Condition | Theta | Alpha | Beta | n |
|---|---|---|---|---|
| base_EO | +0.15 | +0.11 | −0.04 | 10 |
| **base_EC** | **+0.91** | **+0.62** | −0.17 | 10 |
| conv_1 | −0.01 | −0.02 | −0.05 | 9 |
| conv_2 | −0.02 | −0.07 | +0.00 | 10 |
| meditate_B | +0.24 | +0.18 | −0.03 | 5 |
| meditate_K | +0.11 | +0.09 | −0.13 | 5 |
| **PE_1** | −0.16 | −0.10 | **+0.21** | 4 |
| **PE_2** | −0.19 | −0.11 | **+0.17** | 4 |

Key findings:
- **Theta/alpha coincidence is rest-driven**: base_EC dominates. Shared eyes-closed resting-state oscillatory synchrony. Distinct from ImCoh (which peaks during conversation).
- **Beta coincidence is task-driven**: PE sessions highest (+0.21). Conversation variable across dyads (−0.58 to +0.87, mean null). Raw conversation beta coincidence IS higher (0.16-0.24) than rest (0.10-0.18), but both participants independently have more beta activity during conversation, so the z-score excess above independence is null.
- **Complementary to ImCoh**: ImCoh captures interaction-driven phase coupling (peaks during conversation). Burst coincidence captures state-driven (theta/alpha) and task-driven (beta) coupling — different mechanisms.
- **Individual differences in beta**: Y_45 shows conv_1 beta z=+0.87 (strong motor synchrony), most other sessions near null. Suggests beta burst coincidence captures dyad-specific engagement quality.

Per-rSLDS-state: COUP state has highest theta coincidence (+0.37) — rSLDS coupling states (from continuous features) predict discrete burst co-occurrence (non-circular validation).

Output: `results/v10/burst_coincidence/burst_coincidence_results.json`

## Directed Burst Coupling: ECA vs Transfer Entropy (2026-04-01)

### ECA asymmetry is confounded by burst rate differences

**Critical finding**: Directed ECA and Transfer Entropy produce **anticorrelated** asymmetry estimates (theta r=−0.76, alpha r=−0.90, beta r=−0.67). The cause: ECA counts "did P1 onset precede P2 onset?" without conditioning on either participant's burst rate. The participant with more bursts mechanically "leads" more often, creating a spurious asymmetry.

TE conditions on the target's own burst history (`H(y_t | y_past, x_past) - H(y_t | y_past)`), removing the rate confound. This makes TE the correct metric for directed coupling.

**Recommendation**: Use TE for directionality. Do not interpret ECA asymmetry as directional coupling.

### Session-averaged lag is uninformative

Averaging directed coupling across a full 50-min session dilutes transient directed episodes. The lag profile (directed ECA rates at 0.5-2s lags) is flat when session-averaged — no preferred lag emerges, consistent with the V8.2 finding that cross-participant timing didn't replicate at the session level.

**Solution**: Use sliding-window TE timecourses (60s windows) + threshold for directed episodes (z > 2), then condition-segment the episodes to find when directed coupling is active.

### TE results (n=11 sessions, + = therapist leads)

| Condition | Theta TE | Alpha TE | Beta TE |
|---|---|---|---|
| base_EO | +0.0054 | +0.0047 | −0.0069 |
| base_EC | +0.0049 | +0.0034 | −0.0061 |
| conv_1 | −0.0009 | +0.0010 | −0.0011 |
| conv_2 | +0.0023 | +0.0001 | +0.0033 |
| meditate_B | +0.0067 | +0.0078 | −0.0101 |
| meditate_K | +0.0049 | +0.0026 | −0.0070 |
| PE_1 | +0.0051 | +0.0067 | −0.0051 |
| PE_2 | +0.0079 | +0.0047 | −0.0073 |

Pattern: Therapist→patient information flow in theta/alpha (positive); patient→therapist in beta (negative). Conversation is near null. Meditation/PE show strongest asymmetry.

Output: `results/v10/directed_burst_coupling/directed_coupling_results.json`

## Collinearity Analysis: Directed TE vs Existing Scaffold Channels (2026-04-01)

Before promoting directed TE to the scaffold, checked per-condition correlation between TE asymmetry and existing channels (n=10 sessions):

| Existing Channel | Theta r | Alpha r | Beta r |
|---|---|---|---|
| Concordance | −0.06 | +0.10 | +0.38 |
| **Asymmetry** | −0.36 | −0.31 | **−0.74** |
| ImCoh | +0.40 | +0.12 | −0.41 |
| Dynamics | +0.05 | −0.02 | −0.32 |
| Burst coincidence z | +0.23 | −0.04 | −0.11 |

**Theta/alpha TE: low collinearity (|r| < 0.40)** — safe to add as new scaffold channels.

**Beta TE: highly collinear with power asymmetry (r=−0.74)** — DO NOT add to scaffold.

### Why beta TE is collinear (inverted) with power asymmetry

The correlation is **negative**: when the therapist has more beta power (positive asymmetry), TE says the patient drives (negative TE asymmetry). This is not TE recapitulating power differences — it's an **inverse confound from rate imbalance**.

The participant with **fewer** beta bursts has more informative bursts in the Shannon sense — each event carries more surprise against a sparse background. TE measures predictive information, so the low-rate participant's sparse bursts are more "predictive" of the high-rate participant's dense stream. This creates TE asymmetry that tracks the inverse of rate asymmetry.

This is a known limitation of TE on sequences with very different event rates. For theta/alpha, the overall rates are lower and the between-participant imbalance is less extreme, so the confound is weaker.

### V11 Scaffold Recommendation

- **Add**: `te_asym_theta`, `te_asym_alpha` (2 channels, 25D total) — directed burst coupling orthogonal to existing channels
- **Do NOT add**: `te_asym_beta` — 74% redundant with existing power asymmetry channel
- **Add**: `burst_coinc_theta`, `burst_coinc_alpha`, `burst_coinc_beta` (3 channels, 28D) — symmetric burst coincidence, low collinearity with all existing channels

### Concordance vs Directed TE: Two Coupling Modes

The 2D space (concordance × |TE asymmetry|) separates conditions:
- **Conversation**: low concordance, low TE (desynchronized, independent)
- **Baseline (EC)**: high alpha concordance, moderate TE (shared resting state)
- **Meditation**: moderate concordance, high TE (patient's strong alpha creates rate-driven TE)
- **PE**: negative concordance, high TE (actively different states, strong directional flow)

Caveat: meditation's high TE is partially a rate artifact — patient has much more alpha (eyes closed meditating) than therapist (eyes open, reading scripts). The TE picks up the rate difference, not necessarily genuine directed coupling. The rSLDS can use this as a condition discriminator regardless of mechanism, but it should not be interpreted as "patient's alpha drives therapist's alpha" during meditation.
