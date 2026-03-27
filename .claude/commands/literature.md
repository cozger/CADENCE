# /literature — Synchrony Literature Priors for CADENCE Development

When this skill is invoked, use the following distilled knowledge from ~235 papers (83 from reference folder + ~150 from deep web search, compiled 2026-03-26) to guide CADENCE development decisions. This is the evidence base for what signals to measure, how to measure them, and what to expect.

---

## MAP-Neuro Project Context

CADENCE serves the MAP-Neuro project: multimodal dyadic synchrony during neuroplastogen-assisted psychotherapy (ketamine, psilocybin) for opioid use disorder, depression, PTSD, and burnout. Two clinical trials use ketamine + MORE (Mindfulness-Oriented Recovery Enhancement); two use psilocybin-assisted therapy. 60 participants across 4 trials. Modalities: EEG, ECG, facial expression, body pose, speech. Outcomes: opioid use, craving, pain, depression, PTSD, burnout. Process measures: attunement (0-10 NRS), State Empathy Scale, Working Alliance Inventory.

**Facilitators account for 13.6% of variance in clinical psychedelic sessions** (Goldy 2026) — the foundational justification for measuring dyadic synchrony.

**No one has ever done EEG hyperscanning during a psychedelic session** — MAP-Neuro would be genuinely first.

**Substance use disorder has ZERO synchrony-outcome studies** — MAP-Neuro fills a critical gap.

**Therapeutic alliance is 4x more important in psychedelic therapy than conventional therapy** (Levin 2024: post-psilocybin alliance predicts depression at r=-.85 vs typical r~-.20). If CADENCE can capture even a fraction of this variance objectively, it would be transformative.

**Dikker (NYU)** is in direct communication with the project team (Appelbaum, UCSD) — her group has done most real-world EEG hyperscanning work.

---

## Signal Hierarchy: What Predicts Therapy Outcomes

Two meta-analyses (Jennissen 2025: 23 pubs/13 trials/516 effect sizes; Gregorini 2025: 11 studies/N=1423) reveal that **the overall association between synchrony and therapy outcomes is NULL (r=0.03, p=.644)** — but this masks a critical modality dissociation:

**Jennissen 2025 modality breakdown:**

| Synchrony Modality | r | p | Interpretation |
|---|---|---|---|
| **Peripheral physiology (EDA, HR, resp)** | **0.32** | **0.006** | Only significant positive modality |
| **Vocal pitch** | **-0.20** | **0.011** | Negative — matching distress = codysregulation |
| Body movement | ~0.00-0.05 | ns | Near zero when aggregated |
| Overall (all combined) | 0.03 | 0.644 | Null — modalities cancel out |

**Gregorini 2025** (movement + vocal only, excluded physiological): Alliance r=0.19 (p=.02), Outcome r=0.22 (p=.09, ns).

**Strongest individual EDA studies:**

| Study | N | Effect | Key Detail |
|---|---|---|---|
| Gernert 2024 | 14 dyads, CBT | R²=0.43 (p=.011) | First-session SCR concordance predicts full symptom trajectory; alliance ratings were NOT significant predictors |
| Marci 2007 | 20 dyads | r=0.47, 26% var | SC slope concordance (5s windows, 15s correlations) predicts perceived empathy |
| Tschacher 2025 | 21 dyads, 299 sessions | Significant | Patient-leading EDA synchrony predicts WAI bond; SUSY algorithm (30s segments, +/-5s lag) |
| Behrens 2020 | 76 dyads | SCL beta=0.86, p=.001 | SCL predicts cooperation but HR does NOT (p=.389) — functional dissociation |
| Wynn/Heyn 2023 | 45 dyads | F=4.62, p=.032 | Parent-child SCR synchrony during vicarious extinction -> better recall; EDA synchrony is safety learning mechanism |
| Bar-Kalifa 2019 | -- | Significant | EDA synchrony related to alliance in imagery-based treatment |
| Prinz 2021 | -- | Significant | SC synchrony predicted next-session outcomes |

**Why vocal pitch is negative**: matching pitch = mirroring distress escalation ("codysregulation"), not co-regulation. Confirmed in clinical SAD population: Schoenherr 2021 found vocal f0 synchrony predicted WORSE outcomes (beta=.24-.37) — capturing pathological accommodation. Different modalities influence different regulatory competencies: vocal -> emotion regulation (attachment); movement -> behavioral (interpersonal problems).

### Signal Hierarchy Ranked by Effect Size

| Signal | Association with Alliance/Outcome | Direction | Priority for CADENCE |
|--------|----------------------------------|-----------|---------------------|
| **EDA/SC concordance** | r=0.32-0.47; R2=0.43 first-session | Positive | **NOT YET CAPTURED -- #1 addition** |
| **Language style synchrony** | d=0.62 (empathy); r=0.44-0.52 (WAI) | Positive | Not in CADENCE -- high value if NLP added |
| **Sympathetic (SNS) synchrony** | ES=0.19 (p=.02) | Positive | Need EDA hardware (Shimmer3 GSR+) |
| **Body movement (MEA)** | r=0.19-0.35 (alliance/outcome) | Positive | In CADENCE (pose) |
| **Combined SNS+PNS (IBI/HR)** | ES=0.16 (p=.02) | Positive | In CADENCE (ECG) |
| **EEG interbrain (alpha CCorr)** | r=0.46 (learning at 300ms lag) | Positive | In CADENCE (volt_amp) |
| **Parasympathetic (PNS/RMSSD) synchrony** | ES=-0.21 (p=.03) | **NEGATIVE** | In CADENCE -- reinterpret |
| **Vocal pitch synchrony** | r=-0.20 (p=.011); beta=.24-.37 in SAD | **NEGATIVE** | Not in CADENCE -- deprioritize |

### Critical Design Implications

**1. Modalities are ANTI-correlated and have OPPOSITE valence** (Uhl 2025: 90 clients, 22 therapists, 990 segments). Movement and EDA synchrony correlate at r=-.34. High movement synchrony + low EDA synchrony = best outcomes. The optimal state is behavioral engagement with emotional regulation (therapist is physically attuned but not physiologically enmeshed). Collapsing modalities into a single index would cancel signal.

**2. SNS vs PNS dissociation** (Mayo 2021, Behrens 2020, Wilson/Kiecolt-Glaser 2018). RMSSD synchrony is negatively associated with relationship quality. HRV synchrony during marital conflict predicts higher inflammation (Wilson 2018). HR synchrony is mixed SNS+PNS, diluting the signal. SCL is purely sympathetic — the branch with the positive association.

**3. Within-therapist variability > between-therapist variability** (Uhl 2025). A therapist's coupling with THIS patient relative to their baseline predicts outcome; their average coupling level across patients does not. CADENCE should track relative, not absolute, coupling levels.

**4. Context-dependent interpretation required** (Uhl 2025, Wynn 2023). EDA synchrony during cognitive work predicts WORSE outcomes; during emotional/extinction work predicts BETTER outcomes. CADENCE's segment-level analysis is essential.

---

## Modality-Specific Priors

### EEG Interbrain
- **Alpha (8-12 Hz) CCorr** is the most consistently significant band for learning/attention coupling (Davidesco 2023: r=0.46 at ~300ms teacher->student lag)
- **Beta (13-30 Hz) IC** relates to shared action representations and social closeness (Dikker 2021: r=0.16)
- **Theta (4-8 Hz)** appears in mindfulness/emotional processing (Chen 2022: r=0.37)
- **Oxytocin enhances alpha-band IBS** specifically during coordination (Mu 2016: 30 male dyads, double-blind OT vs placebo); gender differences are substantial (female dyads show stronger IBS)
- **rTPJ is the robust, task-domain-general hub of INS** (Lotter 2023: meta-analysis of 22 fMRI + 60 fNIRS experiments). GABA-mediated E/I balance is the neurochemical basis -- psychedelics dramatically alter this.
- **INS reliably present in all 11 clinical hyperscanning studies** (Adel 2025: 160 dyads). Attachment style moderates the INS-outcome link.
- **Amplitude > phase for practical coupling measurement** (Zimmermann 2024): epoch <5s inflates phase-based IBS; adjusted CCorr is essentially identical to PLV (r>.99). CADENCE's volt_amp approach avoids these pitfalls.
- **Phase coherence does NOT index psychedelic state** (Schartner 2017) -- stick with amplitude and complexity metrics.
- **27 IBC methods exist** (Hakim 2023: 215 studies reviewed); cross-brain GLM (= CADENCE architecture) is explicitly endorsed as the sophisticated approach.
- **Massive heterogeneity across studies** (Czeszumski 2022: meta-analysis g=1.98 for cooperation IBS, but I2=98.6%). Expect high between-session variability.

### EDA (NOT YET IN CADENCE)
- **Most validated therapy synchrony signal** — 9/11 interoceptive synchrony studies used EDA (Lucherini 2025)
- **EDA synchrony detects emotionally relevant events better than EEG or HR** (Stuldreher 2020: AUC=.658 for EDA vs .642 EEG vs .592 HR); EEG best for top-down attention
- **EDA synchrony is a mechanism of vicarious safety learning** (Wynn 2023): higher parent-child SCR synchrony during extinction -> lower arousal at recall
- **Two validated algorithms**:
  - SSI / Marci protocol: 5s slope windows -> 15s Pearson correlations -> log-ratio (Marci 2007, Gernert 2024)
  - SUSY: 30s segments, +/-5s lag cross-correlations, Fisher Z, 500 segment-shuffle surrogates (Tschacher 2025, CRAN R package)
- **Hardware**: Shimmer3 GSR+ (~EUR 430-514) — proven LSL integration, direct BLE-to-computer, finger electrodes. **NOT Empatica EmbracePlus** (no real-time streaming, cloud-only, enterprise SDK required).
- Cannot be derived from ECG/PPG — physiologically distinct pathways (Behrens 2020)
- **Patient-leading** EDA synchrony predicts therapeutic bond (Tschacher 2025: 21 dyads, 299 sessions)
- Positive concordance -> symptom reduction; negative concordance -> symptom aggravation (Gernert 2024)
- Preprocessing: cvxEDA or Ledalab for tonic/phasic decomposition; EDA Explorer for artifact detection; use SCR (phasic) not SCL (tonic)

### ECG/Cardiac
- **IBI synchrony**: <1s to 3s lags, 3-8s windows (CADENCE's 5s max lag is adequate)
- **RMSSD**: 30-60s epochs, PNS — **negatively associated with relationship quality**
- **HRV synchrony during conflict is MALADAPTIVE** — predicts inflammation (Wilson/Kiecolt-Glaser 2018). Interpret cardiac coupling by context.
- **HR synchrony predicts group decisions** with 70% accuracy via MdRQA features (Sharika 2024, using Polar H10)
- **Vocal + affect synchrony drives cardiac synchrony; gaze alone does not** (Feldman 2011: mother-infant, near-zero lag)
- **Touch amplifies cardio-respiratory coupling** during pain, moderated by empathy (Goldstein 2017: delta-R2=0.18-0.25)
- **RSA/HF-HRV**: 2.5-6.7s cycles, 30-120s epochs, PNS — respiratory coupling
- **Recommendation**: Decompose into SNS proxy (EDA) and PNS (RSA) as separate channels
- **Extract respiratory rate** from Polar H10 via FMRR method (<2 bpm error)
- **MdRQA** (Multidimensional Recurrence Quantification Analysis) captures nonlinear HR dynamics that linear methods miss (Sharika 2024)

### Facial Expression / Blendshapes
- Behavioral synchrony **Granger-causes** neural synchrony (Koul 2023) — face/pose are leading indicators of EEG coupling
- **Specific interactive behaviors drive coupling bursts** (Pan 2020: scaffolding eta2>.65, d=0.78). Not passive observation but active therapeutic techniques.
- **Event Coincidence Analysis (ECA)** preferred over Event Synchronization (ES) — ES confounds synchrony with serial dependency (Odenweller 2020)
- **Hawkes processes** with basis functions + group sparsity (Xu 2016) — architecturally identical to CADENCE's raised-cosine + group lasso, but for point processes. Halpin & De Boeck 2013 formalized the Dyadic Response (DR) model: EM algorithm with gamma kernels classifies each event as spontaneous vs. mimicry response.
- **Role-specific synchronization templates** predict outcomes better than symmetric measures (Li 2018: coupled HMMs discover therapist-smile-while-patient-speaks vs mutual-smile as distinct templates with different outcome associations)
- Real mimicry from y_06: smile composite, 42% rate, 2.9s lag, events every ~26s
- **Movement synchrony in session 3 predicts dropout**: 1% increase in sync frequency -> 5% reduction in dropout rate (Schoenherr 2019: 267 SAD dyads). Patient-led synchrony is the strongest predictor.

### Body Pose
- Leader-follower asymmetry is clinically meaningful: patient-led in first 3 sessions -> higher dropout (Mende 2021)
- Movement synchrony predicts therapy outcome (Ramseyer & Tschacher 2011: 70 patients, r=.33-.35 with alliance/outcome)
- Motion Energy Analysis synchrony predicts alliance (15 studies, Lucherini 2025)
- **Distributional statistics (kurtosis, skewness) outperform the mean** as features for clinical classification (Kojovic 2024: ASD classification BAC=63.4% from body sync kurtosis)
- Maps directly to CADENCE's directed pathways (therapist->patient vs patient->therapist)
- MEA parameters converge with CADENCE: 60s windows, +/-5s lags, Fisher-Z transformed (Kojovic 2024, Ramseyer 2011)

### Respiratory (NOT YET IN CADENCE — derivable from existing hardware)
- **Respiratory synchrony is the "first responder"** — emerges spontaneously without emotional engagement, independent of cardiac synchrony (Codrons 2014: joint action task, respiratory sync p<.005 while cardiac ns)
- Respiratory synchrony is stronger and more robust than cardiac synchrony across contexts (Mueller & Lindenberger 2011: choir singing, eta2=0.91 for respiration)
- Causally upstream of behavioral synchrony (Yi 2026, perturbation proof)
- Combined EEG + respiratory biofeedback outperforms either alone for empathy (Salminen & Jarvela 2019)
- **Polar H10 respiratory rate validated**: r=0.85, ~1 bpm error at rest (Rogers 2022: 21 participants, exercise ramp). Accuracy is BETTER at resting rates typical during therapy.
- **FMRR is the optimal extraction method** for Polar H10 at 130 Hz (Dong 2021: <2 bpm error at ALL sampling rates down to 50 Hz). Only needs RR intervals (already computed by CADENCE). Band-pass 0.15-0.4 Hz, 32s minimum windows.
- **QRS slope-based EDR** (Varon 2020: downslope dw or slope range sr) gives better waveform morphology if full respiratory coupling analysis is needed (not just rate).
- **314 respiratory algorithms benchmarked** (Charlton 2016): feature-based extraction + time-domain estimation + fusion = best pipeline

### Speech/Voice (NOT YET IN CADENCE — mixed evidence for prosody, strong for language)
- **Vocal pitch synchrony is meta-analytically harmful** (r=-0.20, Jennissen 2025; beta=.24-.37 in SAD, Schoenherr 2021)
- BUT f0 arousal synchrony r=0.71-0.80 in high-empathy MI sessions (Imel 2014)
- **Language style synchrony is STRONGLY positive**: d=0.62 for empathy prediction (Lord 2015: 122 MI sessions); 1 SD increase in LSS = 2.4x odds of high empathy rating, beyond therapist reflections
- **Lexical entrainment** predicts WAI: Task (r=.44-.52), Goal (r=.46), Bond (r=.46) (Bayerl 2022: CBT sessions)
- **Automated empathy scoring from transcripts**: r=0.56 with expert ratings (Xiao 2012, 2015: MI sessions). Empathy perceived as salient events, not continuous stream — aligns with CADENCE's event-driven architecture.
- **12 prosodic entrainment methods disagree substantially** (Kruyt 2023: same data, different results). If implementing prosodic features, use multiple methods (CRQA + windowed cross-correlation most robust).
- **If implemented, prefer linguistic/semantic synchrony (NLP) over raw prosody**
- COMPASS (Lin 2025) for alliance from transcripts; OpenSMILE for audio features

---

## Clinical Synchrony Dynamics

### Inter-Brain Plasticity Across Sessions
**Sened et al. 2025** (8 patients, 1 therapist, 6-session treatment): Inter-brain synchrony significantly **increased across therapy sessions** (d=1.34 — very large effect). Synchrony was above chance (true dyad > 994/1000 permutations) and tracked symptoms, NOT alliance. Synchrony did NOT generalize to a new person — coupling is relationship-specific, not a patient trait. **CADENCE should track session-over-session coupling trajectories, not just within-session coupling.**

### Alliance in Psychedelic Therapy
**Levin et al. 2024** (N=24, psilocybin for MDD): Post-psilocybin alliance predicts depression at r=-.85 (4 weeks), r=-.77 (6 months), r=-.61 (12 months). Pre-dosing alliance predicts mystical experience quality (r=.49). **Reciprocal relationship**: alliance enhances psychedelic experience, and experience enhances alliance. The Task subscale dominates — collaborative/goal-oriented behavioral signals matter most. CADENCE coupling in prep sessions should predict acute experience quality.

### Dropout Prediction
**Schoenherr et al. 2019** (267 SAD dyads): Session 3 movement synchrony predicts premature termination. 1% increase in sync -> 5% dropout reduction. **Patient-led** synchrony is the strongest predictor of retention; **therapist-led** synchrony predicts clinically significant change. Gender-matching has marginal moderating effects.

### Synchrony Profiles Are Therapist-Specific
**Uhl et al. 2025** (90 clients, 22 therapists): Within-therapist synchrony variability predicts outcome (b=-3.57, p=.048); between-therapist average does not. 20% of variance in symptom change attributable to therapist effects. Movement synchrony decreased during emotion-focused work but increased during cognitive work.

### Complementary Co-Regulation
**Koole & Tschacher 2016** (In-Sync model): Co-regulation is NOT just matching — when patient gets upset, therapist should respond complementarily, not mimetically. The three-level model: (1) movement synchrony -> (2) alliance (common language + I-sharing + co-regulation) -> (3) improved patient emotion regulation. **CADENCE should look for both synchronous AND complementary patterns.**

---

## Drug-State Considerations for Neuroplastogen Sessions

### The REBUS Model (Carhart-Harris & Friston 2019)
- **RElaxed Beliefs Under pSychedelics**: 5-HT2A agonism relaxes precision weighting of high-level priors, particularly in DMN
- Psychedelics cause decreased alpha/beta power + increased brain entropy/complexity
- Therapeutic mechanism = **simulated annealing**: temporarily increasing system entropy to escape local minima (rigid negative self-beliefs), followed by settling into revised attractors
- Creates a **window of plasticity** — context sensitivity is amplified, making therapist-patient coupling more important during dosing than in any other therapy context
- The "afterglow" period (24-48h post-dosing) reflects reintegration with revised priors — **integration sessions leverage heightened plasticity** (Muscat 2021: ketamine spinogenesis peaks at 24-48h)

### EEG Signatures Across Psychedelics
- **Alpha suppression** is the most reliable biomarker across ALL psychedelics (Godfrey 2025, Schartner 2017, Timmermann 2019)
- **LZ complexity** reliably increases; beats alpha for predicting subjective experience (Mediano 2024). Validated across psilocybin, ketamine, LSD (Schartner 2017: 86-100% of participants showed increase)
- **DMT/psilocybin-specific**: Emergent theta oscillations replace alpha at peak effects (Timmermann 2019: peak frequency shifts 9.3 -> 7.4 Hz). Theta emergence = "breakthrough" marker.
- **Ketamine**: suppresses theta/alpha, enhances beta/gamma; LZc peaks at 30min post-infusion. Rapidly normalizes PFC global brain connectivity (Muscat 2021).
- **Distinct signatures** (Schifano 2025): ketamine vs psilocybin must be handled separately.
- **Oscillatory vs fractal decomposition matters** (Timmermann 2019): theta increases only visible in oscillatory component. CADENCE's focus on oscillatory amplitude (volt_amp) is well-justified.

### LZ Complexity as Drug-State Moderator
- O(n), <1ms per channel, trivially real-time on EPOC (all 14 channels in ~14ms)
- Lower baseline LZc predicts favorable ketamine response (AUC=0.75)
- Temporal signal diversity (per-channel) is a stronger hallmark than spatial diversity (Schartner 2017)
- **No one has used neural complexity as a coupling moderator** — literature gap
- REBUS model: loosened priors -> increased susceptibility to social influence -> natural coupling moderator
- Libraries: `antropy`, `neurokit2`

### Pharmacological Effects on Coupling Measures
- **Linear and nonlinear coupling change in OPPOSITE directions under drugs** (Alonso 2010: alprazolam decreased linear EEG coupling while increasing nonlinear coupling across entire scalp). Critical for psychedelic sessions — CADENCE's cross-correlation (linear) may miss drug-induced nonlinear coupling changes. Adding mutual information could capture what linear methods miss.

### Oxytocin Context
- **Oxytocin does NOT simply enhance prosocial behavior** — effects are context-dependent (Shamay-Tsoory 2016: social salience hypothesis). OT can increase envy, in-group bias, and aggression in competitive contexts. It increases salience of social cues via dopamine modulation, not uniformly prosocial responses.
- If psilocybin modulates OT release, coupling patterns during dosing may reflect amplified social salience rather than simple prosociality.

### Session Design Implications
- **Eyes-closed** maximizes drug entropy effects; external visual stimulation competes with endogenous drug activity
- During eyes-closed phases: EEG, ECG, EDA, respiratory, and audio/voice become primary coupling channels; blendshapes/pose carry less signal
- Music supports psychedelic experience better than visual stimulation
- Therapist's **voice** may be the primary interpersonal coupling channel during dosing

---

## Coupling Flexibility — Not Just Aggregate Synchrony

**Gordon et al. 2025 (Psych Review)**: Flexibility of synchrony (moving in and out of coupled states) predicts outcomes better than average synchrony.

Metrics CADENCE should compute from dR2 timecourses:
- **Shannon entropy** of dR2 over sliding windows
- **DFA scaling exponent** of dR2 timecourse
- **State transition counts** (coupled -> uncoupled transitions)
- **Dwell/escape time ratio** from metastable dynamics (Tognoli & Kelso 2014)
- **Complexity matching** between partners' DFA exponents (Marmelat 2012: synchronization occurs through global embedding, NOT local error correction; DFA exponent matching captures coordination invisible to cross-correlation)
- **Higher-order distributional statistics** (kurtosis, skewness) of coupling timecourses outperform means for clinical classification (Kojovic 2024)

**Cohen et al. 2021**: State-like (time-varying) synchrony predicts alliance; trait-like (aggregate) does not. Validates CADENCE's EWLS time-varying approach.

**Zilcha-Mano 2025**: Three synchrony signatures — normative (healthy oscillation), hyperactivating (excessive synchrony), deactivating (insufficient). Optimal treatment is personalized.

---

## Methodological Priors

### Amplitude > Phase for EEG Inter-Brain Coupling
**Zimmermann/Ayrolles 2024** (18 dyads + simulations): Unadjusted circular correlations inflate IBS by 36-49%. Epochs <5s inflate alpha-band IBS. Adjusted CCorr is essentially identical to PLV (r>.99). Low SNR (e.g., alpha suppression under psychedelics) degrades phase estimates. **CADENCE's volt_amp amplitude co-modulation approach avoids ALL of these pitfalls.**

### Cross-Brain GLM = CADENCE Architecture
**Hakim et al. 2023** (review of 215 studies): Categorized 27 IBC methods across correlation, regression, coherence, phase synchrony, and causality families. Cross-brain GLM (using neural data from participant B as predictor for participant A, with time-lag) is endorsed as the sophisticated approach. **This is exactly CADENCE's EWLS distributed-lag regression.**

### Multimodal Measurement Is the Consensus
- Different modalities detect different events: EDA = emotional arousal, EEG = cognitive attention, HR = moderate emotional (Stuldreher 2020). Simple average of z-scored modalities is the most robust composite.
- Matching temporal windows across modalities nearly doubles cross-modal effect size (Ohayon & Gordon 2025: 0.42 vs 0.23)
- Behavioral data is essential for interpreting neural coupling (Hakim 2023, Adel 2025)
- **Multimodal fusion provides robustness, not peak accuracy** — validates CADENCE's cross-modal architecture

### Pseudo-Dyad Controls Are Standard
Confirmed as the standard null hypothesis approach by Sened 2025 (1000 permutations), Kojovic 2024 (500 pseudodyads), Ramseyer 2011 (100 segment-shuffled surrogates), HyPyP toolbox (randomize pairings). CADENCE's cross-session pseudo-dyad approach is validated.

### Information Theory as Future Framework
**Chidichimo et al. 2025 (Nature Reviews Neuroscience)**: Proposes mutual information (MI), transfer entropy (TE), and partial information decomposition (PID) as the unified framework for interpersonal coordination. TE is the nonlinear generalization of CADENCE's distributed lag regression. PID can decompose shared vs. unique vs. synergistic information between partners. Standard linear measures miss nonlinear transformations (cross-frequency mapping) common between brains.

---

## Signal Processing Methods to Consider

| Method | Use Case | Priority |
|--------|----------|----------|
| **ECA** (Event Coincidence Analysis) | BL event coupling (replaces ES) | Very High |
| **Hawkes DR model** (Halpin 2013, Xu 2016) | Point-process coupling with EM branching structure; classifies events as spontaneous vs. mimicry | Very High |
| **FMRR respiratory extraction** (Dong 2021) | Extract resp rate from Polar H10 RR intervals; <2 bpm; 32s windows | Very High |
| **Robust BOCPD** (Altamirano 2023) | Temporal localization of coupling onset/offset in dR2 | High |
| **STOK Kalman** (Pascucci 2020) | Adaptive tau + frequency-resolved directed connectivity | High |
| **SPIKE-distance** (Kreuz 2013) | Parameter-free, time-resolved, causal event synchrony; alternative to cross-correlation for point processes | High |
| **Synchronization templates** (Li 2018) | Coupled HMMs discover role-specific facial coordination patterns; predict outcomes from template frequencies | High |
| **MMHP** (Wu 2022) | Latent active/inactive states for bursty mimicry events | Moderate |
| **MdRQA** (Sharika 2024) | Nonlinear recurrence features for cardiac/physiological coupling | Moderate |
| **Mutual Information / TE** (Chidichimo 2025) | Nonlinear coupling; unified framework | Moderate |
| **Multivariate IAAFT** | Surrogates preserving cross-channel structure within participant | Moderate |
| **IDTxl Transfer Entropy** | Model-free validation metric for CADENCE coupling estimates | Moderate |
| **QRS slope EDR** (Varon 2020) | Full respiratory waveform from ECG (better than FMRR for morphology) | Low |
| **Language style matching** (Lord 2015) | Function word synchrony for empathy; d=0.62 | Low (needs NLP) |

---

## Wavelet Methods for Facial and Behavioral Coupling

### Frequency Bands in Facial Blendshape Data (empirically grounded)

| Band | Behavior | Source |
|------|----------|--------|
| <0.1 Hz | Emotional state, postural drift | Fujiwara & Daibo 2018, 2020 |
| 0.1-0.5 Hz | Sustained expressions (tonic smile/frown) | Schmidt 2003, Jeganathan 2022 |
| 0.5-2 Hz | Expression transitions (onset/offset) | Jeganathan 2022, Kawulok 2021 |
| 2-7 Hz | Speech articulation (syllable rate) | Audiovisual speech literature |
| 2.6-6.5 Hz | Listener backchannels (fast nods) | Hale & Ward 2019 |
| >8 Hz | Tracker noise (CNN inference jitter) | MediaPipe GitHub #825; no published PSD characterization exists |

### Key Methods

- **CWT + HMM** (Jeganathan 2022, eLife): Analytic Morse wavelet on AU timeseries, 0-5 Hz, 10 freq bins x 14 AUs -> HMM discovers discrete facial states with unique spectral fingerprints. Complexity of dynamic expressions captured by a small number of simple spatiotemporal states. **THE anchor paper for CADENCE BL wavelet pipeline.**
- **Wavelet Transform Coherence (WTC)**: Standard for fNIRS hyperscanning (Zhang 2020); Morlet wavelet w=6; real pairs > pseudo pairs at specific frequency bands. Dominant method in 27+ IBC studies (Hakim 2023).
- **Windowed Multiscale Synchrony (WMS)** (Likens & Wiltshire 2021, SCAN): Time-varying, scale-localized coupling dynamics. Tracks how synchrony changes across both time and frequency. Code: github.com/aaronlikens/wms. Python: multiSyncPy package.
- **Cross-Wavelet Transform (XWT)** (Issartel 2014, tutorial): Phase relationship between two signals at each time-frequency point. Arrow orientation encodes relative phase (right=in-phase, left=anti-phase).
- **Convolutional NMF on scalograms** (Mackevicius 2019, eLife): Discovers recurring time-frequency motifs without labels. Applied to neural data; directly applicable to AU scalograms.

### Key Findings for CADENCE

- **Different frequency bands = different social functions** (Hale & Ward 2019): Low-frequency coherence (0.2-1.1 Hz) = mimicry with ~600ms lag. High-frequency content (2.6-6.5 Hz) shows systematic ANTI-synchrony from listener backchannels. The same signal carries opposite social meaning at different frequencies.
- **Rapport associates with synchrony at TWO distinct bands**: <0.025 Hz (emotional state level, >40s cycles) AND 0.5-1.5 Hz (gestural/sub-second level) (Fujiwara & Daibo 2020). Maps to Koole & Tschacher's three temporal levels.
- **Wavelet coherence on AU timeseries between dyad members is UNEXPLORED** — all existing wavelet synchrony work uses gross body movement (MEA, motion capture), NOT AU-level facial signals. This is a genuine gap.
- **Low-pass at 5-8 Hz preserves all facial expression signal, removes tracker noise** (Jeganathan 2022 downsampled to 10 Hz without signal loss). CADENCE does not currently apply temporal filtering to blendshapes.
- **Different synchrony methods measure different facets, not one construct** (Schoenherr 2019: 7 methods, 84 therapy dyads, only partially correlated). Cannot substitute wavelet coherence for cross-correlation or vice versa.
- **CRQA may outperform WTC for naturalistic turn-taking interaction** (Schiavo 2025): WTC misses nonlinear, time-lagged coordination in reciprocal exchange.

### Face Tracker Noise Characteristics

- **MediaPipe applies no anti-aliasing filter** — users must implement their own (1-Euro filter recommended). CNN inference noise is broadband 0-15 Hz at 30 fps.
- **Physiological jaw tremor (6-8 Hz) is below tracker noise floor** — 0.6mm amplitude vs 1-2 pixel tracker precision. Not visible in blendshape data.
- **50 Hz fluorescent light flicker aliases to 10 Hz at 30 fps** — potential concern for lab recordings.
- **Effective information bandwidth of face tracker output: ~5-8 Hz** (all meaningful dynamics below 5 Hz for expression, up to 7-8 Hz for speech).

### Novel Opportunities for CADENCE

1. **Wavelet coherence on dyadic AU timeseries** — no one has done this; all existing work uses gross movement
2. **CWT scalogram clustering** for unsupervised facial behavior discovery (Jeganathan CWT + Mackevicius convNMF)
3. **Multi-scale coupling profile**: speech sync at 3-6 Hz, expression sync at 0.5-2 Hz, state sync at <0.1 Hz — all from a single computation
4. **No published PSD of any face tracker** — CADENCE could characterize MediaPipe/YQP noise spectrum
5. **Wavelet-native BL pipeline** replacing hand-crafted speech/blink/smile detectors with principled frequency-domain decomposition

---

## Key Theoretical Frameworks

1. **REBUS** (Carhart-Harris & Friston 2019): Psychedelics = simulated annealing of high-level priors. Window of plasticity for belief revision. Context sensitivity amplified.

2. **In-Sync Model** (Koole & Tschacher 2016): Movement synchrony -> alliance (common language + I-sharing + co-regulation) -> improved emotion regulation. Foundational theory for CADENCE.

3. **Flexible Multimodal Synchrony** (Gordon 2025): Flexibility > aggregate. DFA exponents, entropy, state transitions.

4. **Social Salience Hypothesis** (Shamay-Tsoory 2016): Oxytocin/neuromodulators increase salience of social cues, not prosociality per se. Context-dependent effects.

5. **Information-Theoretic Coordination** (Chidichimo 2025): MI, TE, PID as the unified mathematical framework. Nonlinear, directed, decomposable.

6. **Mutual Prediction** (Schilbach 2025): INS = encoding own behavior + predicting partner's. Cross-brain GLM (= CADENCE) is the natural implementation.

7. **Neurobiology of INS** (Lotter 2023): rTPJ hub, GABA-mediated E/I balance, linked to ventral attention + DMN networks. Psychedelics alter E/I balance -> fundamentally modulate INS capacity.

---

## Key Gaps CADENCE/MAP-Neuro Can Fill

1. **First EEG hyperscanning during psychedelic session** — genuinely novel
2. **First synchrony-outcome study in substance use disorder** — zero exist
3. **First use of neural complexity as coupling moderator** — literature gap
4. **First Hawkes process applied to facial mimicry events** — open niche
5. **No standard composite multimodal synchrony index exists** — CADENCE could define one, but must preserve modality-specific effects (Uhl 2025)
6. **Real-time multimodal synchrony feedback in clinical therapy** — Kleinbub 2020 proposed this concept; CADENCE builds the measurement backbone
7. **First cross-session coupling trajectory tracking in psychedelic therapy** — Sened 2025 showed inter-brain plasticity (d=1.34); no one has tracked this during neuroplastogen treatment
8. **First nonlinear (MI/TE) inter-brain coupling under psychedelics** — Alonso 2010 showed drugs increase nonlinear while decreasing linear coupling; no inter-brain study exists

---

## Opioid Misuse Detection (Garland Lab)

**Gullapalli et al. 2025** (169 chronic pain patients): TFT deep learning model detects opioid misuse from multimodal sensor + cognitive task data (AUC=0.81). **Behavioral signals >> physiological** (error rates > reaction times > ECG >> respiration). 45s optimal window (close to CADENCE's 30s EWLS tau). Opioid cues most discriminative. Same research group (Garland, UCSD) as MAP-Neuro.

---

## Reference Documents

For full paper details, effect sizes, and citations:
- `docs/literature_synthesis.md` — 83 papers from reference folder (updated 2026-03-26)
- `docs/literature_search_deep.md` — ~148 papers from web search, organized by topic
- `docs/cardiac_synchrony_detail.md` — cardiac features, timescales, SNS/PNS dissociation
- `docs/eda_synchrony_literature_review.md` — 20 EDA papers with full methods
- `docs/eda_key_studies.csv` — key EDA studies table (CSV)
- `docs/ground_truth_paradigm.md` — validation paradigm design (6 paradigms, 63 min, 12-16 dyads)
- PDFs in `G:\My Drive\ARPA Shared Documents\Reference Papers\` — all project literature including wavelet/facial dynamics papers (updated 2026-03-26)
