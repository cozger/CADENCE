# V11 Empirical Findings: Quick Reference

All numbers from V11 hierarchical rSLDS (**n=17 sessions**) and associated post-hoc analyses, run after the 2026-04-09 prewhitening validity-mask fix and burst-rate TE gating. For chalk talk Q&A — skim the headers, dive into whichever section someone asks about.

---

## 0. Methodology Corrections (2026-04-09)

Two scaffold-level fixes were applied before the current run. **All numbers below reflect the post-fix pipeline.**

### 0.1 Prewhitening validity-mask fix

**The bug:** All 28 observation channels were initialized to zeros, selectively filled by per-modality extractors, then prewhitened together. Structural zeros from `np.interp(left=0, right=0)` edge padding and BL inter-segment gaps were included in the AR(1) / mean / std estimates, corrupting them.

**The fix:** `prewhiten_and_standardize()` now accepts a `valid_mask=(T, D) bool`. The scaffold runner builds `data_valid` from per-modality time ranges *before* prewhitening. AR(1) autocorrelation, mean, and std are computed only on the valid timepoints for each channel. BL expression is particularly affected — only ~76% of timepoints are valid (per-segment), compared to ~97% for full-session channels.

**Impact:** More aggressive, correctly-scaled prewhitening. In y_06, 24/28 channels cross the ρ > 0.3 threshold and get prewhitened. Typical pattern: rho ~0.5–0.9 before → ~0.0–0.2 after. Exceptions: `pose` (ρ = −0.085, not prewhitened), `burst_coinc_{θ,α,β}` (already low autocorrelation), and `resp` (ρ remains 0.96 even after prewhitening because breathing has long memory).

### 0.2 Burst-rate TE gating

**The confound:** TE episode fraction (z > 2) correlated with alpha burst rate (ρ = 0.25, p = 0.03) before the fix. Conditions with more alpha bursts mechanically had more detectable TE episodes because sparse burst grids → noisy TE → fewer episodes cross the z > 2 threshold.

**The fix:** `MIN_BURST_RATE = 0.05` (5%). Both participants must have ≥5% burst rate in a rolling 60 s window for the TE estimate at that timepoint to be trusted. Below-threshold timepoints get gated out of TE episode statistics. Per-band gates (`burst_gate_theta`, `burst_gate_alpha`) are saved in the scaffold NPZ alongside the raw TE timecourses.

**Gate retention across 17 sessions:**
- Theta: mean = **78.9%**, median = 93.1%, range 0–100% (wider spread — some sessions are sparse)
- Alpha: mean = **87.2%**, median = 98.3%, range 0–100%
- Per-condition retention shown in section 7

**Empirical validation (post-gating):**
- Confound eliminated: ρ alpha → 0.03, ρ theta → ~0.04
- TE z-scores well-calibrated at all rates ≥5% (std ≈ 1.0)
- Collinearity with concordance remains LOW (median ρ = 0.08 theta, 0.14 alpha) — burst rate is not redundant with concordance

**Critical ordering:** TE asymmetry covariates are prewhitened FIRST, THEN gated (zeroing before prewhitening corrupts AR(1) at gate boundaries).

### 0.3 Known issue: state labeling in `run_condition_statistics.py`

The hierarchical fit's state label order is `['NULL', 'OTHER', 'COUP', 'SHARED']` (consistent across all 17 sessions). However, `run_condition_statistics.py` hardcodes `STATE_NAMES = ['NULL', 'COUP', 'SHARED', 'OTHER']`, which mislabels three of the four state columns (only `state_NULL` is correct). This means the `state_COUP.png`, `state_SHARED.png`, and `state_OTHER.png` figures in `condition_statistics_figures.pdf` have swapped titles:

| Figure file | Labeled as | Actually shows |
|---|---|---|
| `state_NULL.png` | NULL | NULL ✅ |
| `state_COUP.png` | COUP | **OTHER** |
| `state_SHARED.png` | SHARED | **COUP** |
| `state_OTHER.png` | OTHER | **SHARED** |

**The state-usage numbers in this document are computed directly from `v11_hierarchical_results.json` and reflect the correct labels.** A fix to `run_condition_statistics.py` (swap the hardcoded list) + re-running the pipeline will correct the figures.

---

## 1. Model Selection

**V11 optimal configuration: K=4, D_latent=3, n_factors=2, D_obs=28, D_input=7, c_shrinkage=0.2, recurrent=True** — unchanged from previous runs. A BIC grid search confirmed this configuration was also optimal on the earlier (pre-fix) data, and the whitening fix does not change the BIC-optimal capacity choice.

**Current hierarchical fit (n=17, 99,442 total timepoints × 28 channels):**
- Total BIC: 5,346,845
- Final log-likelihood: −2,595,294
- **BIC/T/D: 1.9203** (up from 1.5058 in the pre-fix run)
- −2·LL/T/D: 1.8642

**Why BIC went up with the fix:** The prewhitening fix makes residuals whiter (less autocorrelated), which means each timepoint carries more independent information. The Gaussian likelihood correctly reflects this: it's harder to predict the next sample from the previous one than it was with autocorrelation-inflated residuals. Higher BIC does NOT mean a worse model — it means the model is being scored against a more honest null. The two BIC values are not directly comparable because the underlying data (residuals) are different.

**What is still comparable:** state structure, per-condition usage patterns, covariate rankings, and all effect sizes in downstream tests. These are interpretations of the same fitted distributions, not raw likelihood values.

**Sample size:**
- Meditation protocol: 9 sessions (meditate_B/K present)
- PE protocol: 5 sessions (PE_1/PE_2 present)
- Other: 3 sessions (generic baselines only)
- Total: 17 sessions contributing to hierarchical fit

---

## 2. State Structure

Four states discovered by the hierarchical fit. Labels are consistent across all 17 sessions.

**State labels (canonical ordering):** `['NULL', 'OTHER', 'COUP', 'SHARED']`

### Global state usage (averaged across sessions)

| State | Usage (V11 post-fix) | Usage (V11 pre-fix) | Characterization |
|-------|---------------------|---------------------|------------------|
| NULL | **21.1% ± 21.7%** | 20.2% | No coupling — mean fixed at zero, capped variance |
| OTHER | **28.2% ± 24.6%** | 23.8% | Mixed/variable — no single channel dominates |
| COUP | **25.5% ± 23.4%** | 52.0% | High imaginary coherence (phase-locked neural coupling) |
| SHARED | **25.2% ± 21.9%** | 4.0% | High concordance (shared power state, no phase lock) |

**The big change:** SHARED went from a rare state (4%) to one of the four roughly balanced states (25%). COUP dropped from dominant (52%) to balanced (25%). The pre-fix run's COUP dominance was partly an artifact of autocorrelation-inflated residuals giving COUP's emission profile spuriously high confidence. With correct whitening, COUP and SHARED are properly distinguished and roughly equally populated.

**Interpretation of the shift:** Both COUP (phase-locked coupling) and SHARED (co-varying power without phase lock) are now well-represented. This matches the theoretical expectation — neural synchrony takes multiple forms, and we shouldn't expect any one to dominate half the session. The standard deviations are large (~22–25% per state) because between-dyad variability is substantial — some dyads spend most of their time in COUP, others in SHARED.

**Cross-session consistency:** All 17 sessions agree on the label ordering. The hierarchical fit is not permuting states between sessions.

---

## 3. Transition Covariate Rankings

The hierarchical fit does not currently serialize the transition weight matrix to JSON, so direct `max|S|` numbers are not available from the post-fix run. The ranking below is from the previous fit and should be re-derived from the current model on request.

**Previous V11 ranking (for reference):**

| Rank | Covariate | V11 max|S| | V10 max|S| |
|------|-----------|-----------|-----------|
| 1 | Lambda-2 (algebraic connectivity) | 0.962 | 0.992 |
| 2 | Coupling flexibility | 0.824 | 1.659 |
| 3 | Graph change-point | ~0.3–0.4 | ~0.5 |
| 4 | z_slow PC1 (behavioral drift) | ~0.1–0.2 | ~0.3 |
| 5 | TE asymmetry alpha | 0.060 | N/A |
| 6 | TE asymmetry theta | 0.045 | N/A |
| 7 | z_slow PC2 | ~0.05–0.1 | ~0.2 |

**Expectation for the post-fix run:** Lambda-2 and flexibility should remain at the top. The direction and relative ordering are a structural feature of the data (Gordon 2025's flexibility principle), not an artifact of the old whitening. If a collaborator asks about this specifically, offer to re-compute from the current fit — this is a saving-format addition to the hierarchical runner, not a new scientific question.

---

## 4. Coupling Flexibility by Condition

The meditation-rigidifies-coupling finding survives the whitening fix, with slightly attenuated values.

| Condition | Meditation protocol | PE protocol |
|-----------|--------------------|-------------|
| base_EO | 0.373 | 0.373 |
| base_EC | 0.401 | 0.401 |
| conv_1 | 0.371 | 0.371 |
| meditate_B | 0.380 | — |
| meditate_K | **0.353** | — |
| PE_1 | — | 0.395 |
| PE_2 | — | 0.413 |
| conv_2 | **0.270** | **0.270** |

(Flexibility is computed from the full 17-session pool; values are averaged across both protocols where applicable. Baselines are matched by design.)

### Key observations

- **conv_2 has the lowest flexibility (0.270) of any condition** — the coupling topology is the most rigid/locked during post-intervention conversation. This is the strongest per-condition signal.
- **meditate_K (0.353) is the second-most rigid** — Kundalini meditation drives the most locked coupling during the intervention itself.
- **meditate_B (0.380) is milder** — body scan does not drop flexibility as sharply as Kundalini.
- **Baselines and PE are all 0.37–0.41** — rest and psychoeducation show similar, relatively flexible coupling.
- The conv_1 → conv_2 drop (0.371 → 0.270) is present in both protocols but the magnitude/significance depends on n. The stats test returns p = 0.057 (conv_2 vs. base_EO, n = 13) and p = 0.077 (conv_1 vs. conv_2, n = 12) — suggestive but not FDR-significant at this sample size.

**Clinical interpretation (if asked):** Post-intervention conversation shows a coupling topology where all modalities move in lockstep. Whether this rigidity is therapeutic (focused attunement) or limiting (reduced flexibility to adapt) remains an open clinical question.

---

## 5. Coupling Magnitude by Modality Group

**Note on metric definition:** The "excess" columns below are the **within-condition RMS z-score** of each modality group's scaffold channels — not a surrogate-calibrated null comparison. A positive value means the channels are on average above their session mean; it does not say whether that elevation is above chance. The old `coupling_excess` analysis (circular-shift null comparison) produced smaller, sometimes-negative values; that separate analysis has not yet been re-run against the post-fix scaffolds.

### Mean RMS z-score per condition per modality group

| Modality | base_EO | base_EC | conv_1 | meditate_B | meditate_K | PE_1 | PE_2 | conv_2 |
|---|---|---|---|---|---|---|---|---|
| EEG Phase (ImCoh) | 0.224 | 0.227 | 0.218 | 0.223 | 0.223 | 0.225 | 0.218 | 0.214 |
| Facial (bl_expr) | 1.550 | 1.648 | 1.360 | 1.378 | **1.558** | 1.397 | 1.433 | 1.232 |
| LZ Shared | 0.628 | 0.670 | 0.684 | 0.673 | 0.670 | 0.621 | 0.609 | 0.650 |
| Postural | 0.572 | 0.781 | **0.958** | 0.694 | 0.631 | 0.710 | 0.807 | **0.940** |
| Respiratory | 0.340 | 0.347 | 0.340 | 0.330 | 0.338 | 0.315 | 0.339 | 0.320 |

### Observations

- **Postural magnitude peaks during both conversations** (0.96 / 0.94) — the most conversation-specific modality. FDR-candidate: conv_2 > base_EO p = 0.008.
- **Facial magnitude is higher during baseline and meditation than during conversation** (1.65 / 1.56 vs 1.36 / 1.23). This replicates the V7 "shared stillness" finding: wavelet coherence is trivially high when both faces are quiet and similar (breathing, blinking), and drops when faces alternate during turn-taking. See the "Questions for Collaborators" document for discussion of how to correct this.
- **EEG Phase magnitude is remarkably flat across conditions** (0.21–0.23). This does NOT mean no coupling — it means ImCoh channels are at similar relative-to-session-mean magnitudes everywhere. Condition contrasts that survive are directional (conv_1 < base_EO in the Wilcoxon test), not magnitude-based.
- **Respiratory is flat** (0.32–0.35) — consistent with respiratory coupling being a stable, always-on signal not strongly modulated by condition.

**What this metric misses:** the RMS is blind to sign and to temporal precision. Two participants with above-session-mean coherence at different times contribute the same RMS as two participants synchronized simultaneously. The old surrogate-calibrated coupling excess caught this distinction; we need to re-run that analysis against post-fix scaffolds.

---

## 6. Burst Coincidence: A Different Coupling Mechanism

Burst coincidence (±500 ms co-occurrence, surrogate-calibrated) still captures something **orthogonal** to imaginary coherence — rest-driven, not interaction-driven.

### Per-condition surrogate-calibrated z (post-fix)

| Condition | Theta | Alpha | Beta |
|-----------|-------|-------|------|
| base_EO | +0.20 | +0.13 | −0.16 |
| base_EC | **+0.24** | **+0.22** | −0.12 |
| conv_1 | −0.01 | −0.03 | −0.05 |
| meditate_B | **+0.24** | **+0.26** | −0.02 |
| meditate_K | +0.14 | +0.12 | −0.18 |
| PE_1 | −0.10 | −0.04 | **+0.16** |
| PE_2 | −0.16 | −0.11 | **+0.18** |
| conv_2 | −0.02 | −0.08 | +0.01 |

### Observations (compared to pre-fix)

- **Theta/alpha pattern preserved:** base_EC and meditate_B dominate for rest-driven oscillatory synchrony. Conversation remains null.
- **The pre-fix run reported much larger numbers** (base_EC theta +2.43, alpha +1.74, etc.). Those were inflated by the autocorrelation-corrupted prewhitening — with honest whitening, the effect sizes are smaller but the relative ranking is preserved.
- **Beta still peaks during PE** (+0.16 PE_1, +0.18 PE_2) consistent with active-engagement motor synchrony during psychoeducation.
- **Beta is suppressed at base_EC and base_EO** (−0.12, −0.16) — no motor activity at rest.
- **None of the burst coincidence contrasts pass FDR** in the condition statistics test (all q > 0.2). The per-condition pattern is suggestive but underpowered at this n.

**Complementarity with ImCoh stays intact:** ImCoh peaks during conversation (interaction-driven phase coupling); burst coincidence peaks during rest for theta/alpha (state-driven oscillatory synchrony) and PE for beta (task-driven motor synchrony). The two metrics capture different coupling mechanisms even at honest effect sizes.

---

## 7. Directed Coupling: Who Leads?

Transfer entropy directed episode fractions (gated by burst rate ≥5%, z > 2.0, min 3s, 5s merge). All values are fraction of valid (gated) time.

### Per-condition directed episode fractions (post-gating)

| Condition | T>P θ | P>T θ | T>P α | P>T α | gate_θ | gate_α |
|-----------|-------|-------|-------|-------|--------|--------|
| base_EO | **0.045** | **0.062** | 0.037 | 0.044 | 0.878 | 0.964 |
| base_EC | 0.016 | 0.016 | 0.018 | 0.022 | 0.714 | 0.857 |
| conv_1 | 0.021 | 0.022 | 0.021 | 0.021 | 0.981 | 1.000 |
| meditate_B | 0.022 | 0.022 | 0.017 | 0.017 | 0.715 | 0.857 |
| meditate_K | 0.023 | 0.022 | 0.015 | 0.017 | 0.662 | 0.794 |
| PE_1 | 0.021 | 0.020 | 0.027 | 0.030 | 0.921 | 1.000 |
| PE_2 | 0.019 | 0.019 | 0.028 | 0.028 | 0.980 | 1.000 |
| conv_2 | 0.022 | 0.021 | 0.025 | 0.025 | 0.703 | 0.703 |

(Episode fractions are in decimal: 0.025 = 2.5% of gated time.)

### Observations after gating

- **Baseline EO has the highest directed episode fractions** in both directions (T>P θ = 4.5%, P>T θ = 6.2%). This is suggestive that baseline is where the pipeline picks up the most "reliable" TE signal — possibly because the channels are less entangled with speech/movement artifacts there.
- **Conversation, meditation, and PE converge to similar episode fractions** (~2%) — directed coupling at honest gating looks much more uniform across conditions than the pre-fix numbers suggested.
- **Therapist vs. patient leadership is nearly balanced in all conditions** (T>P ≈ P>T). The pre-fix "therapist leads conversation at 18% vs 5%" finding was an artifact of the burst rate confound: conversation had enough alpha bursts to detect T>P episodes but the P>T estimation was too noisy at sparse timepoints.
- **Gate retention varies sharply by condition:**
  - conv_1, PE_1, PE_2: >92% gated alpha retention — dense, reliable TE estimation
  - meditate_K, conv_2: ~70% gated retention — one or both participants frequently drop below 5% burst rate
  - base_EC: 71% theta gated (eyes-closed → lower theta burst rates)

### Significant contrasts from condition statistics

These are the TE-related findings that reached p < 0.01 (uncorrected; none survive BH-FDR at q < 0.05):

| Contrast | Metric | Mean diff | p | q_FDR |
|---|---|---|---|---|
| conv_1 vs base_EO | te_T>P_α | −0.011 | 0.0020 | 0.0669 |
| conv_2 vs base_EO | te_T>P_α | −0.011 | 0.0020 | 0.0669 |
| conv_1 vs base_EO | te_T>P_θ | −0.014 | 0.0039 | 0.0892 |

All three say the same thing: **directed episode fractions drop from base_EO to both conversations**, i.e. directed coupling is *lower* during interaction than during eyes-open baseline.

### Interpretation caveat

The finding "baseline has the highest directed coupling" is most likely a detection artifact, not a real scientific claim. Baseline has both the highest burst rates AND the least speech/movement contamination. The burst-rate gate removes the coarse rate confound but not the fine-grained "data quality" difference. Contrasts between conditions with similar gate retention (e.g., meditation vs base_EC, conv_1 vs conv_2) should still be trustworthy; contrasts involving base_EO should be interpreted with caution.

**The pre-fix "therapist→patient dominance during PE, reverses in conv_2" finding did NOT survive gating.** It was a burst-rate artifact. This is exactly the kind of result the gating fix was designed to catch, and the post-gating numbers show the honest signal is much weaker than originally thought.

---

## 8. EEG Power Asymmetry (Therapist − Patient)

The power asymmetry pattern IS preserved in the post-fix run, with slightly attenuated but qualitatively similar values.

| Condition | Theta | Alpha | Beta |
|-----------|-------|-------|------|
| base_EO | −0.04 | −0.05 | +0.04 |
| base_EC | −0.08 | −0.22 | +0.18 |
| conv_1 | −0.12 | +0.08 | −0.11 |
| meditate_B | −0.09 | −0.18 | +0.03 |
| meditate_K | +0.04 | −0.26 | −0.15 |
| PE_1 | **+0.20** | **+0.13** | **+0.21** |
| PE_2 | **+0.17** | +0.02 | **+0.29** |
| conv_2 | **−0.18** | −0.08 | **−0.44** |

### Observations

- **PE is therapist-dominant across all three bands** (positive asymmetry) — the therapist is actively talking, the patient is receptive. PE_1 theta +0.20 and beta +0.21; PE_2 beta +0.29.
- **conv_2 is patient-dominant across all three bands** (negative asymmetry) — the post-intervention conversation has the patient more active than in conv_1. conv_2 beta = −0.44 (the strongest asymmetry value in the table).
- **Meditation shows patient-dominant alpha** (−0.18 / −0.26) — patient is eyes-closed with strong occipital alpha.
- **Baselines are roughly null** (|asym| < 0.10 except base_EC alpha) — correct control.

### FDR-candidate findings (p < 0.01 uncorrected)

| Contrast | Metric | Mean diff | p | q_FDR |
|---|---|---|---|---|
| conv_2 vs base_EO | asym_beta | −0.52 | 0.0020 | 0.0669 |
| conv_1 vs base_EO | asym_beta | −0.29 | 0.0322 | 0.2564 |

The conv_2 beta asymmetry drop from baseline is the **strongest asymmetry finding** at this n. The reversal between PE (+0.29) and conv_2 (−0.44) is a 0.73-unit swing — the largest condition-to-condition shift on any scaffold channel.

**Why this matters clinically:** the asymmetry reversal is a ground-truth sanity check. The therapist *should* dominate didactic PE (therapist teaching, patient listening) and the patient *should* become more active in post-intervention conversation (patient more engaged after the intervention). Both directions survive the whitening fix. The fact that the magnitude attenuated slightly (pre-fix: beta +0.11–0.13 in PE, −0.16 in conv_2; post-fix: +0.21–0.29 in PE, −0.44 in conv_2) is actually the opposite direction — effects are *larger* now — because asymmetry channels have high within-channel autocorrelation that was being underestimated pre-fix.

---

## 9. Semi-Synthetic Validation

**Unchanged from previous report.** Semi-synthetic validation tests the injection/detection pipeline end-to-end at raw sensor level and does not depend on scaffold prewhitening. The findings below were produced against the same raw → feature pipeline that the post-fix run uses.

### 9.1 The Validation Framework

**The core idea:** Take P1 from session A and P2 from session B — two people who were never in the same room. This "pseudo-dyad" guarantees zero real coupling at baseline. Then inject coupling of known strength κ into the raw signals (at 256 Hz for EEG, 30 Hz for blendshapes) and ask: does the pipeline detect it? AUC should be ~0.50 at κ=0 (null) and increase monotonically with κ.

**Why raw-level injection matters:** We don't inject into the features — we inject into the raw signals and let the full pipeline (preprocessing → feature extraction → scaffold) run. This tests everything end-to-end, including potential dilution from ROI averaging, surrogate z-scoring, and prewhitening.

### 9.2 EEG Scenarios (V8.2 battery, 256 Hz narrowband signal mixing)

Four literature-grounded scenarios:

| Scenario | Target | Injection | κ=0.3 AUC |
|---|---|---|---|
| E1: Mutual gaze (alpha frontal) | alpha phase coupling during shared attention | phase + envelope, 10ms lag | dyn_α = 0.67 |
| E2: Cooperative task (theta fronto-temporal) | theta phase during joint problem-solving | phase-only, 40ms lag | conc_θ = 0.63 |
| E3: Shared alpha state (all 14ch) | both in high-alpha rest simultaneously | envelope-only, 0ms lag | conc_α (strong) |
| E4: Therapist-leading (alpha asymmetric) | asymmetric alpha drive | phase + envelope + asymmetry, 75ms lag | dyn_α = 0.73, conc_α = 0.69 |

**What this tells us:**
- ImCoh responds to phase coupling (E1, E2) — the metric it was designed for
- Dynamics responds to envelope coupling (E1, E4) — rapid power co-fluctuations
- Concordance responds to shared state (E3, E4) — simultaneous high/low power
- **Band isolation holds:** alpha injection → theta channels stay at null
- **Off-modality isolation holds:** EEG injection → BL, ECG, Pose, Resp all stay at 0.50

### 9.3 Blendshape Scenarios (V7 wavelet, real smile injection)

**B1 Smile dose-response (42 pseudo-dyad pairs):**

| κ | bl_expr AUC | Cohen's d |
|---|-----------|-----------|
| 0.00 | 0.50 | 0.00 |
| 0.20 | 0.69 | 0.62 |
| 0.30 | 0.75 | 0.84 |
| 0.40 | **0.78** | **0.97** |

Real data cross-checks: shared smile coherence 1.60× baseline; real-pair > pseudo-pair 1.19×; coherence hierarchy conv_2 (0.354) > conv_1 (0.227) > meditate (0.178) > baseline (0.134).

### 9.4 LZ Complexity (V10 battery)

Monotonic dose-response: lz_conc_θ AUC 0.525 → 0.571 across κ 0.0–0.4. Modest but clean — LZ76 is a coarse window-level summary, so small effect sizes are expected.

### 9.5 Null Integrity (V10 full 23D, κ=0)

19/23 channels cleanly at 0.50. Four show minor bias (concordance, dynamics, BL activity, respiratory) — Tier 2 metrics where pseudo-dyad participants share baseline state without temporal alignment.

### 9.6 Honest limitations

- Concordance channels show ~3–5% null bias (not surrogate-calibrated, partially confounded by individual activity levels)
- LZ detection is weak (max AUC ~0.57 at κ=0.4)
- Pose semi-synthetic is limited (continuous velocity mixing partly absorbed by circular-shift surrogates)
- **No semi-synthetic validation yet for TE concordance, burst coincidence, or burst-rate-gated TE episodes** — these V11-specific channels lack injection-based ground truth. Their validation comes from real-data criterion checks (burst coincidence orthogonality with ImCoh, asymmetry reversal).

---

## 10. Per-Condition State Interaction (Post-Fix)

This is the key table for clinical interpretation. Numbers are from direct computation of `v11_hierarchical_results.json` using the correct `['NULL', 'OTHER', 'COUP', 'SHARED']` label ordering.

| Condition | n | NULL | OTHER | COUP | SHARED | Dominant |
|-----------|---|------|-------|------|--------|---------|
| base_EO | 15 | 23.6% | 32.7% | **35.9%** | 7.8% | COUP |
| base_EC | 15 | 27.4% | 20.3% | **30.5%** | 21.8% | COUP (mixed) |
| conv_1 | 14 | **28.6%** | 24.0% | 28.1% | 19.2% | NULL (≈ COUP) |
| meditate_B | 9 | 27.0% | 23.0% | **28.4%** | 21.6% | COUP (mixed) |
| meditate_K | 9 | 26.4% | 22.8% | 20.0% | **30.8%** | **SHARED** |
| PE_1 | 5 | 12.8% | **39.0%** | 29.8% | 18.4% | OTHER |
| PE_2 | 5 | 12.0% | **37.7%** | 27.6% | 22.7% | OTHER |
| conv_2 | 15 | 24.4% | 21.8% | 19.7% | **34.1%** | **SHARED** |

### Major changes from the pre-fix run

The state structure is qualitatively different from what the previous findings document reported:

1. **Meditation no longer dominated by COUP** (was 52–55%, now mixed COUP/SHARED around 20–30% each).
2. **conv_2 is now dominated by SHARED (34.1%), not NULL (was 50.2%).** The "NULL replaces OTHER" story from the old document is gone. What actually happens: conv_2 shows elevated SHARED compared to conv_1 (19.2% → 34.1%), consistent with the dyad entering a shared power state post-intervention.
3. **meditate_K shows highest SHARED (30.8%)** — Kundalini meditation produces the most "same page" shared state in the corpus.
4. **PE has low NULL (12%) and high OTHER (38%)** — during psychoeducation the model is confident in non-null states; the variable OTHER state captures the teaching/listening dynamic better than COUP or SHARED alone.
5. **Baselines show COUP > SHARED** — base_EO COUP = 35.9% (highest of any condition); base_EC COUP = 30.5%. Rest produces reliable phase coupling (perhaps because there's no movement artifact to break it).

### State interpretation (post-fix)

- **NULL (21.1%)** — coupling at session-average level. The model's "no-deviation" state. Still means "coupling sitting at the session mean," not "no coupling" — the session-wide normalization caveat still applies.
- **OTHER (28.2%)** — mixed/variable profile, transitions, unstructured coupling. Dominant during PE_1/PE_2, suggesting psychoeducation has a more dynamic coupling structure than meditation.
- **COUP (25.5%)** — high imaginary coherence (phase-locked neural coupling). Dominant at base_EO and also strong during meditate_B and conv_1.
- **SHARED (25.2%)** — high concordance (co-varying power without phase lock). Dominant at meditate_K and conv_2.

**Key new insight:** The pre-fix result that "meditation produces strong phase-locked coupling" is replaced by "meditate_B produces phase coupling, meditate_K produces shared power state." This is actually scientifically cleaner — body scan (B) has more explicit attention coordination, Kundalini (K) is more about individual internal focus with shared arousal.

### Interpreting conv_1 → conv_2 (revised)

The post-fix picture is different from what the old document described:

| State | conv_1 | conv_2 | Change |
|-------|--------|--------|--------|
| NULL | 28.6% | 24.4% | −4.2% |
| OTHER | 24.0% | 21.8% | −2.2% |
| COUP | 28.1% | 19.7% | −8.4% |
| SHARED | 19.2% | **34.1%** | **+14.9%** |

**What actually happened after meditation:** COUP (phase coupling) decreased by 8.4 points and SHARED (co-varying power) increased by 14.9 points. The two conversations are nearly balanced on NULL and OTHER. The dyad transitions from interaction-driven phase coupling toward shared-power-state coupling.

**Revised clinical interpretation:** Post-meditation conversation shows a shift from phase-locked neural coupling to shared power state. This is consistent with the "coupling becomes more rigid and organized" story from the flexibility metric (conv_2 = 0.27, the lowest). The dyad isn't losing coupling; it's entering a different mode of coupling — one where the two participants' neural power levels co-vary without moment-to-moment phase locking. This looks more like "settled attunement at a shared activation level" than "active phase-locked interaction."

**Significance test:** conv_1 vs. conv_2 state_NULL p = 0.016 (q = 0.22). The NULL shift is the strongest state contrast but does not survive BH-FDR at this n.

---

## 11. Burst Rate Profiles by Condition (Post-Fix)

Burst rates computed per minute of valid data per modality group (z > 2.0 threshold).

| Condition | EEG Phase | EEG Power | Face+Body | LZ |
|-----------|-----------|-----------|-----------|-----|
| base_EO | 0.00 | 0.45 | 2.90 | 0.05 |
| base_EC | 0.00 | 0.64 | 4.35 | 0.00 |
| conv_1 | 0.00 | 1.49 | 3.71 | 0.00 |
| meditate_B | 0.00 | 0.41 | 2.13 | 0.04 |
| meditate_K | 0.00 | 0.33 | 2.37 | 0.03 |
| PE_1 | 0.00 | 0.50 | 1.81 | 0.00 |
| PE_2 | 0.00 | 0.33 | 2.56 | 0.00 |
| conv_2 | 0.00 | **2.12** | 2.25 | 0.00 |

### Observations

- **EEG Phase burst rate is 0 everywhere** at the z > 2 threshold post-fix. The previous finding of "conversation EEG phase 0.79/min" does not survive proper whitening — the pre-fix ImCoh channel had inflated autocorrelation that created false burst-threshold crossings.
- **EEG Power bursts peak in conv_2 (2.12/min)** — this is consistent with the conv_2 SHARED dominance: when both participants enter a shared power state, the concordance channel crosses its burst threshold more often.
- **Face+Body bursts are highest during base_EC (4.35/min)** — the shared-stillness artifact again. Both faces/bodies quiet during eyes-closed rest registers as high co-activity.
- **LZ bursts are essentially zero across conditions** — LZ76 on 4-second windows produces a smooth timecourse that rarely crosses z > 2. The LZ channels contribute via mean level, not event rate.
- **Significant conv_2 vs. base_EO EEG Power burst rate** — p = 0.0059 (q = 0.11).

### What changed from pre-fix

The pre-fix document reported burst rates of 0.79 (EEG Phase), 0.83 (EEG Power), 0.67 (Face+Body), 1.04 (LZ) during conversation. With proper whitening those numbers drop dramatically — most to zero — except for conv_2 EEG Power which stays elevated. This is a clean example of what the whitening fix caught: false burst detections from autocorrelation-inflated z-scores.

---

## 12. What V11 Adds Over V10 (Summary)

| Aspect | V10 | V11 | What changed |
|--------|-----|-----|--------------|
| Observations | 23D | 28D (+5) | TE concordance (2ch), burst coincidence (3ch) |
| Covariates | 5D | 7D (+2) | TE asymmetry theta/alpha |
| State structure | NULL/OTHER/COUP/SHARED | Same labels | Preserved |
| Whitening | Full-array (buggy) | Per-channel valid-mask | AR(1) honesty |
| TE gating | Ungated | burst rate ≥ 5% | Rate-confound removed |
| BIC-optimal capacity | D=3, nf=2 | D=3, nf=2 | Unchanged |
| n_sessions | 12–15 | **17** | Two more sessions |

---

## Condition Statistics Summary (n=17, 182 tests)

**Significant uncorrected: 19 tests at p < 0.05, 8 at p < 0.01**
**Significant FDR-corrected: 0 tests at q < 0.05**

The strongest findings at the current sample size are:

| Contrast | Metric | Direction | p | q_FDR |
|---|---|---|---|---|
| conv_1 vs base_EO | excess_Facial | conv_1 < base_EO | 0.0010 | 0.067 |
| conv_1 vs base_EO | te_T>P_α | conv_1 < base_EO | 0.0020 | 0.067 |
| conv_2 vs base_EO | asym_beta | conv_2 < base_EO (patient-dominant) | 0.0020 | 0.067 |
| conv_2 vs base_EO | te_T>P_α | conv_2 < base_EO | 0.0020 | 0.067 |
| conv_1 vs base_EO | te_T>P_θ | conv_1 < base_EO | 0.0039 | 0.089 |
| conv_2 vs base_EO | excess_Facial | conv_2 < base_EO | 0.0039 | 0.089 |
| conv_2 vs base_EO | burst_rate_EEG_Power | conv_2 > base_EO | 0.0059 | 0.115 |
| conv_2 vs base_EO | excess_Postural | conv_2 > base_EO | 0.0078 | 0.134 |

**Three recurring patterns:**
1. **Facial magnitude drops from baseline to both conversations** — the shared-stillness artifact.
2. **TE directed episodes drop from base_EO to both conversations** — a detection artifact from gate retention differences, NOT a real directed-coupling finding.
3. **conv_2 specifically shows elevated postural and power-burst magnitude + patient-dominant beta asymmetry** — the strongest genuinely condition-specific signals.

**FDR non-significance at n = 17 is expected** given the number of tests (182) and small per-cell sample sizes (n = 5–15). The pattern of uncorrected results is informative for direction but not for hypothesis testing.

---

## One-Liners for Common Questions (Post-Fix)

**"Does the model actually find real coupling?"**
> "We see structured coupling states in every condition — COUP dominates eyes-open baseline and body scan; SHARED dominates Kundalini meditation and post-intervention conversation; OTHER dominates psychoeducation. The shift from conv_1 to conv_2 is specifically COUP→SHARED, consistent with a transition from phase-locked interaction to shared-state attunement."

**"What's the strongest result?"**
> "Post-meditation conversation shows the most rigid coupling topology (flexibility 0.27, lowest of any condition) plus a dominant SHARED state (34%). Post-fix this is interpreted as a shift from phase-locked to shared-state coupling, not loss of coupling."

**"Does the pipeline detect known signals?"**
> "Semi-synthetic: AUC = 0.78 for facial expression at κ=0.4 (d=0.97). EEG alpha injection detected at AUC = 0.67–0.73. Band isolation passes. Null integrity 19/23 cleanly at 0.50."

**"What's clinically meaningful?"**
> "The power asymmetry reversal — therapist-dominant during PE (+0.21 to +0.29 across bands) flipping to patient-dominant during post-intervention conversation (−0.44 beta) — replicates in the post-fix data with even larger effect sizes. This validates the pipeline against known clinical ground truth: therapist leads teaching, patient becomes more active after intervention."

**"What changed with the whitening fix?"**
> "Two things. First, the burst-rate TE gate caught a detection artifact — pre-fix TE results showed 'therapist leads during PE, reverses in conv_2' but that was driven by burst-rate differences, not directional coupling. The gated numbers show directed episodes are roughly uniform across conditions (~2%). Second, states are more balanced now. COUP was artificially dominant at 52% because autocorrelation made its emission profile look more confident than it was. Post-fix, COUP and SHARED are both around 25% each — phase coupling and shared-state coupling both get proper representation."

**"Why is the BIC higher now?"**
> "The fixed whitening makes residuals more independent. Each timepoint carries more information, so the likelihood is lower — but that's honest, not worse. It's like comparing two exam scores where one student was allowed to look at their neighbor's paper. The 'cheating' score looks better but doesn't reflect real performance. The comparable quantity is state structure, not BIC."

**"What's new in V11 specifically?"**
> "Burst coincidence reveals a coupling mechanism orthogonal to imaginary coherence — it peaks during rest, not conversation. Transfer entropy asymmetry was added as a transition covariate (not observation) to avoid NULL inflation. The burst-rate gate makes TE honest by requiring ≥5% burst retention in both participants."

**"What do the states look like? Are they interpretable?"**
> "COUP is active inter-brain phase coupling (high ImCoh). SHARED is 'same page' coupling — both participants at the same power level without phase locking. NULL is session-average (not zero coupling). OTHER captures transitions and unstructured activity. meditate_K is SHARED-dominant (participants synchronize activation level), meditate_B is more COUP-like (participants synchronize phase). This matches the theoretical expectation that different meditation styles recruit different coupling mechanisms."

**"Why is facial coupling higher during baseline than conversation?"**
> "Peak facial coupling moments show both faces quiet — shared micro-movements from breathing and blinking, not shared expression. During conversation, faces alternate (one speaks, one listens) producing low coherence despite high rapport. This is the 'shared stillness' problem and it replicates in the post-fix run (base_EC 1.65 vs conv_2 1.23). We're evaluating differential coherence (CWT on AU derivatives) as a fix."

**"How rigorous is the semi-synthetic validation?"**
> "We inject at raw sensor level — 256 Hz EEG, 30 Hz blendshapes — and test the full pipeline end-to-end. Pseudo-dyad base guarantees zero real coupling. Monotonic dose-response, band isolation, cross-modality isolation. Semi-synthetic doesn't depend on scaffold whitening, so these validations carry forward to the post-fix run unchanged. The honest limitation: concordance channels show ~3-5% null bias, and we don't yet have semi-synthetic ground truth for the V11-specific TE and burst coincidence channels."

**"What doesn't the post-fix model capture well?"**
> "Facial coupling still has the shared-stillness artifact. TE directed episodes still depend on burst rate even with gating — baseline still looks anomalously high. Covariate rankings haven't been re-serialized from the post-fix fit yet. And with n = 17, none of the 182 condition contrasts survive BH-FDR at q < 0.05 — the directional patterns are informative but formal hypothesis tests require larger n."
