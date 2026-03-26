# EDA/Skin Conductance Synchrony in Psychotherapy and Dyadic Interactions
## Comprehensive Literature Review for CADENCE Integration

**Date:** 2026-03-25
**Purpose:** Evaluate EDA synchrony as a candidate modality for the CADENCE pipeline

---

## Executive Summary

Electrodermal activity (EDA) synchrony is the most validated physiological synchrony signal in psychotherapy research. It has been studied since the 1950s, with Marci et al. (2007) establishing the modern paradigm. The literature consistently shows that EDA concordance between therapist and patient predicts perceived empathy (r = 0.47), therapeutic alliance quality, and treatment outcomes (R^2 = 0.43 for symptom change). A 2025 meta-analysis found that peripheral physiological synchrony (primarily EDA) correlates with alliance/outcome at r = 0.32, significantly stronger than other nonverbal modalities. EDA cannot be derived from ECG/PPG signals, requiring dedicated hardware (e.g., Empatica EmbracePlus wristband, 4 Hz EDA sampling). Standard analysis uses windowed cross-correlation with 8-30s windows and lags up to 4-5s, which maps naturally onto CADENCE's existing lag-regression framework.

---

## Paper-by-Paper Detailed Review

### 1. Marci, Ham, Moran, & Orr (2007) -- LANDMARK STUDY

- **Citation:** Marci, C.D., Ham, J., Moran, E., & Orr, S.P. (2007). Physiologic correlates of perceived therapist empathy and social-emotional process during psychotherapy. *Journal of Nervous and Mental Disease*, 195(2), 103-111.
- **Sample:** 20 established patient-therapist dyads, 1 session each
- **Population:** Clinical (established psychodynamic psychotherapy patients)
- **EDA Metric:** Moving window correlations of skin conductance (SC); ratio of positive to total concordance windows ("physiologic concordance index")
- **What it predicted:** Patient-perceived therapist empathy; quality of social-emotional interactions
- **Effect size:** r = 0.47 (p = 0.03) for SC concordance vs. perceived empathy; p = 0.01 for positive social-emotional interactions during high vs. low concordance moments
- **Hardware:** Laboratory SC measurement equipment (specific model not reported)
- **Temporal parameters:** 30-second windows, 10-second increments (the "Marci protocol" -- became the de facto standard)
- **Key finding:** First study of social psychophysiology during live psychotherapy. During moments of high SC concordance, both patients and therapists exhibited significantly more positive social-emotional interactions.

### 2. Marci & Orr (2006)

- **Citation:** Marci, C.D. & Orr, S.P. (2006). The effect of emotional distance on psychophysiologic concordance and perceived empathy between patient and interviewer. *Applied Psychophysiology and Biofeedback*, 31(2), 115-128.
- **Sample:** 20 dyads, 1 session each
- **Population:** Clinical (adult outpatients from a mental health clinic)
- **EDA Metric:** Moving window correlations of SC; positive/negative concordance ratio
- **What it predicted:** Effect of emotional distance on concordance and perceived empathy
- **Effect size:** Significant reduction in concordance in emotionally distant condition (p < 0.05)
- **Hardware:** SC measurement equipment (not specified)
- **Temporal parameters:** 30-second windows, 10-second increments
- **Key finding:** Increased emotional distance → decreased psychophysiologic concordance → reduced perceived empathy. Established SC concordance as a potential biomarker of empathy.

### 3. Messina, Palmieri, Sambin, Kleinbub, Voci, & Calvo (2013)

- **Citation:** Messina, I., Palmieri, A., Sambin, M., Kleinbub, J.R., Voci, A., & Calvo, V. (2013). Somatic underpinnings of perceived empathy: The importance of psychotherapy training. *Psychotherapy Research*, 23(2), 169-177.
- **Sample:** 39 dyads (pseudo-patient + listener at 3 training levels)
- **Population:** Clinical simulations with trained therapists, psychologists, and non-therapists
- **EDA Metric:** Moving window correlations of SC; lag analysis
- **What it predicted:** Perceived empathy; effect of therapist training level
- **Effect size:** Significant positive correlation between concordance and perceived empathy; therapists showed highest concordance
- **Hardware:** Not specified
- **Temporal parameters:** Lag analysis revealed psychologists show higher synchrony at 0-s lag; psychotherapists at 3-s lag (reflecting more deliberate empathic processing)
- **Key finding:** Training level matters. Therapists showed higher concordance AND empathy. The 3-s lag for trained therapists suggests more effortful, controlled empathic responses.

### 4. Palmieri, Kleinbub, Calvo, Benelli, Messina, Sambin, & Voci (2018)

- **Citation:** Palmieri, A., Kleinbub, J.R., Calvo, V., Benelli, E., Messina, I., Sambin, M., & Voci, A. (2018). Attachment-security prime effect on skin-conductance synchronization in psychotherapists: An empirical study. *Journal of Counseling Psychology*, 65(4), 490-499.
- **Sample:** 18 psychodynamic therapists + 18 healthy volunteers (36 dyads)
- **Population:** Mixed (therapists + healthy volunteers in simulated clinical interviews)
- **EDA Metric:** Moving window correlations; lag analysis
- **What it predicted:** Effect of attachment-security priming on SC synchronization dynamics
- **Effect size:** Significant effect on lag dynamics (but not overall PS amount)
- **Hardware:** Not specified
- **Temporal parameters:** Lag analysis; priming increased synchrony specifically at negative lags (therapist-leading)
- **Key finding:** Attachment-security priming caused therapists to assume a leading role in the physiological coupling. Demonstrates that SC synchrony is modifiable and sensitive to internal states.

### 5. Gernert, Nelson, Falkai, & Falter-Wagner (2024)

- **Citation:** Gernert, C.C., Nelson, A., Falkai, P., & Falter-Wagner, C.M. (2024). Synchrony in psychotherapy: High physiological positive concordance predicts symptom reduction and negative concordance predicts symptom aggravation. *International Journal of Methods in Psychiatric Research*, 33(1), e1978.
- **Sample:** 14 patient-therapist dyads (from 25 recruited)
- **Population:** Clinical (mixed psychiatric diagnoses per ICD-10; >50% affective disorders)
- **EDA Metric:** Single Session Index (SSI) -- correlations of window-wise slopes of SCR between dyad members
- **What it predicted:** Change in Global Severity Index (GSI) over treatment
- **Effect size:** R^2 = 0.429 (p = 0.011); F(1,12) = 9.009. SSI > 0.11 predicted symptom reduction; SSI < 0.11 predicted symptom aggravation
- **Hardware:** Empatica E4 wristband (non-dominant arm)
- **Temporal parameters:** 4 Hz sampling; 5-second averaged slopes at 1-second increments; 15-second sliding correlation windows at lag-zero; 10-minute analysis window after 5-min acclimation; 5 Hz Butterworth low-pass filter
- **Key finding:** First-session SCR concordance predicted entire treatment trajectory. Objective physiological marker outperformed subjective therapeutic alliance ratings. Validated pseudo-dyad controls (real > pseudo, p < 0.001).

### 6. Tschacher, Ribeiro, Goncalves, Sampaio, Moreira, & Coutinho (2025)

- **Citation:** Tschacher, W., Ribeiro, E., Goncalves, A., Sampaio, A., Moreira, P., & Coutinho, J. (2025). Electrodermal synchrony of patient and therapist as a predictor of alliance and outcome in psychotherapy. *Frontiers in Psychology*, 16, 1545719.
- **Sample:** 21 therapeutic dyads (21 patients, 6 therapists)
- **Population:** Clinical (major depressive disorder or social anxiety disorder; CBT treatment)
- **EDA Metric:** In-phase and lagged electrodermal synchrony (patient-leading vs. therapist-leading)
- **What it predicted:** Therapeutic bond quality (therapist-rated); patient distress
- **Effect size:** Patient-leading synchrony positively linked with therapeutic bond; negatively linked with patient distress (specific coefficients in full paper)
- **Hardware:** EDA recording devices (specific model in full paper)
- **Temporal parameters:** All 16 sessions of CBT recorded per dyad
- **Key finding:** Patient-leading synchrony was significantly more pronounced than therapist-leading synchrony. The leading role of patients in sympathetic interactions was the key predictor, not overall synchrony magnitude.

### 7. Tschacher & Meier (2020)

- **Citation:** Tschacher, W. & Meier, D. (2020). Physiological synchrony in psychotherapy sessions. *Psychotherapy Research*, 30(5), 558-573.
- **Sample:** 55 sessions (4 clients, 1 therapist)
- **Population:** Clinical (naturalistic psychotherapy)
- **EDA Metric:** Not EDA specifically -- used ECG, heart rate, HRV, and respiration; two methods: windowed cross-correlation and correlation of local slopes (concordance), with surrogate controls using segment-wise shuffling
- **What it predicted:** Therapeutic alliance ratings; session self-report variables
- **Effect size:** Significant associations between synchrony and alliance in regression models
- **Hardware:** ECG and respiration monitors
- **Temporal parameters:** 15-second intervals; average session 51 minutes
- **Key finding:** Three of four physiological measures showed significant synchrony (not raw ECG). Demonstrated surrogate-controlled analysis for physiological synchrony in therapy.

### 8. Karvonen, Kykyri, Kaartinen, Penttonen, & Seikkula (2016)

- **Citation:** Karvonen, A., Kykyri, V.L., Kaartinen, J., Penttonen, M., & Seikkula, J. (2016). Sympathetic nervous system synchrony in couple therapy. *Journal of Marital and Family Therapy*, 42(3), 383-395.
- **Sample:** 10 tetrads (couple + 2 co-therapists), 1 session each
- **Population:** Clinical (couple therapy, dialogical approach)
- **EDA Metric:** EDA concordance via pairwise moving window correlations
- **What it predicted:** Therapeutic alliance (Session Rating Scale)
- **Effect size:** Co-therapists showed highest synchrony; couples showed lowest
- **Hardware:** Not specified
- **Temporal parameters:** Lag analysis up to 1 second
- **Key finding:** Different dyadic pairings within the therapy room show different synchrony levels. Therapist-therapist synchrony > client-therapist > couple.

### 9. Kykyri, Karvonen, Wahlstrom, Kaartinen, Penttonen, & Seikkula (2019)

- **Citation:** Kykyri, V.L., Karvonen, A., Wahlstrom, J., Kaartinen, J., Penttonen, M., & Seikkula, J. (2019). Sympathetic nervous system synchrony: An exploratory study of its relationship with the therapeutic alliance and outcome in couple therapy. *Psychotherapy Research*, 29(6), 766-785.
- **Sample:** 12 couple therapy processes (24 clients, 10 therapists)
- **Population:** Clinical (couple therapy)
- **EDA Metric:** EDA concordance indices (moving window correlations)
- **What it predicted:** Therapeutic alliance; therapy outcome; well-being changes
- **Effect size:** EDA synchrony increase from beginning to end of therapy correlated with positive trend in female clients' well-being
- **Hardware:** EDA recording equipment
- **Temporal parameters:** Multiple sessions per process analyzed longitudinally
- **Key finding:** Couple therapy brings spouses closer on a physiological level. EDA synchrony increase predicted better outcomes, particularly for women. However, in one case, decreasing synchrony was more beneficial.

### 10. Stratford, Lal, & Meara (2012)

- **Citation:** Stratford, T., Lal, S., & Meara, A. (2012). Neuroanalysis of therapeutic alliance in the symptomatically anxious: The physiological connection revealed between therapist and client. *American Journal of Psychotherapy*, 66(1), 1-21.
- **Sample:** 30 dyads (15M/15F clients), 6 weekly sessions each
- **Population:** Clinical (symptomatic anxiety)
- **EDA Metric:** Skin conductance resonance (SCR) as measure of therapeutic alliance
- **What it predicted:** Therapeutic alliance; EEG brain patterns during high alliance
- **Effect size:** Prefrontal, parietal, and occipital EEG sites associated with alliance periods identified by SC resonance
- **Hardware:** EEG + SC recording equipment
- **Temporal parameters:** 6 x 1-hour sessions
- **Key finding:** Used SC resonance to identify high-alliance moments, then linked these to specific EEG patterns. Multi-modal physiological validation of SC synchrony.

### 11. Behrens, Snijdewint, Moulder, Prochazkova, Sjak-Shie, Boker, & Kret (2020)

- **Citation:** Behrens, F., Snijdewint, J.A., Moulder, R.G., Prochazkova, E., Sjak-Shie, E.E., Boker, S.M., & Kret, M.E. (2020). Physiological synchrony is associated with cooperative success in real-life interactions. *Scientific Reports*, 10, 19609.
- **Sample:** 152 participants (76 dyads; 50 dyads for SCL analysis after exclusions)
- **Population:** Healthy (Prisoner's Dilemma game)
- **EDA Metric:** Windowed cross-correlation of skin conductance level (SCL)
- **What it predicted:** Cooperative success in Prisoner's Dilemma game
- **Effect size:** f^2 = 0.013; interaction effect (SCL synchrony x face-to-face) p = 0.001; face-to-face beta = 0.86
- **Hardware:** MP150 BIOPAC data acquisition system (wireless EDA)
- **Temporal parameters:** Recorded at 2000 Hz, downsampled to 20 Hz; 8-second windows; lag range up to 4 seconds (100 ms steps)
- **Key finding:** Only skin conductance (not heart rate) predicted cooperative success. Effect was strengthened by face-to-face contact. Specificity to sympathetic nervous system.

### 12. Prochazkova, Sjak-Shie, Behrens, Wieling, & Kret (2022)

- **Citation:** Prochazkova, E., Sjak-Shie, E.E., Behrens, F., Wieling, M., & Kret, M.E. (2022). Physiological synchrony is associated with attraction in a blind date setting. *Nature Human Behaviour*, 6(2), 269-278.
- **Sample:** 140 participants (70 women, 70 men; ages 18-38)
- **Population:** Healthy (blind date setting at Dutch festivals)
- **EDA Metric:** Skin conductance synchrony (cross-correlation)
- **What it predicted:** Romantic attraction (desire for second date)
- **Effect size:** Significant predictor of attraction; overt signals (smiles, eye gaze) were NOT significant predictors
- **Hardware:** Eye-tracking glasses + physiological devices for HR and SC
- **Temporal parameters:** Real-time dating interactions
- **Key finding:** Published in Nature Human Behaviour. Attraction was predicted by covert physiological synchrony (HR + SC), not by overt behavioral signals. SC synchrony reflects unconscious arousal alignment.

### 13. Slovak, Tennent, Reeves, & Fitzpatrick (2014)

- **Citation:** Slovak, P., Tennent, P., Reeves, S., & Fitzpatrick, G. (2014). Exploring skin conductance synchronisation in everyday interactions. *Proceedings of NordiCHI 2014*.
- **Sample:** Multiple dyads in everyday settings
- **Population:** Healthy (everyday real-world interactions)
- **EDA Metric:** EDA synchronization via wearable sensors
- **What it predicted:** Quality of interpersonal interaction; emotional engagement
- **Effect size:** Significant synchrony linked to mutual emotional engagement
- **Hardware:** Wearable sensors (Bluetooth) measuring EDA, fingertip temperature, HRV
- **Temporal parameters:** Real-time in-the-wild recording
- **Key finding:** Demonstrated feasibility of wearable EDA synchrony measurement outside the lab. EDA synchrony indicated meaningful social aspects in everyday settings.

### 14. Kleinbub (2017)

- **Citation:** Kleinbub, J.R. (2017). State of the art of interpersonal physiology in psychotherapy: A systematic review. *Frontiers in Psychology*, 8, 2053.
- **Sample:** Systematic review of all published studies (15+ studies with SC)
- **Population:** Mixed (psychotherapy, clinical interviews, simulations)
- **EDA Metric:** Review of multiple methods: moving window correlations, skin conductance resonance, graphical comparison, cross-recurrence quantification
- **What it predicted:** N/A (review)
- **Key findings:** (a) Almost all studies use SC as the primary physiological measure; (b) The field lacks specific theory-informed hypotheses and sound analytical procedures; (c) The "Marci protocol" (30s windows, 10s steps) is used without empirical justification for parameters; (d) No studies directly assessed potential violations of stationarity assumptions.

### 15. Palumbo, Marraccini, Weyandt, Wilder-Smith, McGee, Liu, & Goodwin (2017)

- **Citation:** Palumbo, R.V., Marraccini, M.E., Weyandt, L.L., Wilder-Smith, O., McGee, H.A., Liu, S., & Goodwin, M.S. (2017). Interpersonal autonomic physiology: A systematic review of the literature. *Personality and Social Psychology Review*, 21(2), 99-141.
- **Sample:** Comprehensive systematic review
- **Population:** Mixed (all dyadic/group autonomic physiology studies)
- **EDA Metric:** Review of all interpersonal autonomic methods including EDA/SC
- **What it predicted:** N/A (review)
- **Key findings:** (a) EDA is the most commonly studied autonomic measure in interpersonal physiology; (b) Physiological synchrony is a robust phenomenon across methods; (c) Multiple novel metrics exist; (d) Major methodological heterogeneity across studies.

### 16. Gregorini, Lutz, Tschacher, Meier, & Ramseyer (2025)

- **Citation:** Gregorini, S., Lutz, W., Tschacher, W., Meier, D., & Ramseyer, F.T. (2025). Potential role of nonverbal synchrony in psychotherapy: A meta-analysis. *Counselling and Psychotherapy Research*.
- **Sample:** Meta-analysis of 23 publications from 13 trials
- **Population:** Clinical (psychotherapy)
- **EDA Metric:** Meta-analysis including physiological synchrony (EDA/SC) as a category
- **What it predicted:** Alliance and therapy outcome
- **Effect size:** Overall NVS-outcome r = 0.03 (n.s.); BUT peripheral physiological synchrony specifically: r = 0.32 (p = 0.006); vocal pitch synchrony: r = -0.20 (p = 0.011)
- **Key finding:** CRITICAL. Peripheral physiological synchrony (primarily EDA) is the ONLY modality that shows a significant positive effect on therapy outcomes. Movement synchrony and vocal synchrony show null or negative effects. This makes EDA synchrony the premier biomarker.

### 17. Milstein & Gordon (2020)

- **Citation:** Milstein, N. & Gordon, I. (2020). Validating measures of electrodermal activity and heart rate variability derived from the Empatica E4 utilized in research settings that involve interactive dyadic states. *Frontiers in Behavioral Neuroscience*, 14, 148.
- **Sample:** 30 participants (15 dyads)
- **Population:** Healthy (psychology undergraduates)
- **EDA Metric:** SCL validation between E4 wristband and MindWare reference
- **What it predicted:** N/A (validation study)
- **Effect size:** EDA correlation: r = 0.606 (rest), r = 0.298 (conversation, n.s.); IBI: r > 0.995; ~73% of E4 EDA recordings appeared noisy
- **Hardware:** Empatica E4 vs. MindWare mobile impedance cardiograph
- **Temporal parameters:** 4 Hz sampling (E4); three 5-minute segments
- **Key finding:** WARNING: E4 wristband EDA data had poor reliability (~73% noise). HR/IBI data were excellent. This raises concerns about wrist-based EDA measurement for synchrony research.

### 18. Di Mascio, Boyd, & Greenblatt (1955)

- **Citation:** Di Mascio, A., Boyd, R.W., & Greenblatt, M. (1955). Physiological correlates of tension and antagonism during psychotherapy. *Diseases of the Nervous System*, 16(1).
- **Sample:** 3 dyads
- **Population:** Clinical (psychoanalytic psychotherapy)
- **EDA Metric:** Correlation analysis of SC
- **What it predicted:** Concordance/discordance patterns
- **Hardware:** Not detailed
- **Key finding:** The FIRST study of interpersonal SC in psychotherapy. Established the basic paradigm.

### 19. Robinson, Herman, & Kaplan (1982)

- **Citation:** Robinson, J.W., Herman, A., & Kaplan, B.J. (1982). Autonomic responses correlate with counselor-client empathy. *Journal of Counseling Psychology*, 29(2), 195-198.
- **Sample:** 21 dyads, 2 sessions each
- **Population:** Clinical (psychological counseling)
- **EDA Metric:** Manual peak matching within +-7 second window
- **What it predicted:** Perceived empathy (Barrett-Lennard Relationship Inventory)
- **Effect size:** Strong correlation between SC peak matching and empathy
- **Temporal parameters:** 7-second matching window
- **Key finding:** Early validation that autonomic concordance relates to empathy. The 7-second temporal window is notable as it aligns with CADENCE's lag parameters.

### 20. Kleinbub, Mannarini, & Palmieri (2020)

- **Citation:** Kleinbub, J.R., Mannarini, S., & Palmieri, A. (2020). Interpersonal biofeedback in psychodynamic psychotherapy. *Frontiers in Psychology*, 11, 1655.
- **Sample:** Proof-of-concept / theoretical paper
- **Population:** Clinical (psychodynamic psychotherapy)
- **EDA Metric:** Real-time SC synchrony displayed as interpersonal biofeedback
- **What it predicted:** Proposed application for enhancing therapist awareness
- **Key finding:** Proposed using real-time physiological synchronization (PS) displays as a clinical tool for therapists. Demonstrated the concept of feeding SC concordance information back during sessions.

---

## Methodological Summary: How EDA Synchrony Is Measured

### Signal Acquisition
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Sampling rate | 4-2000 Hz | Wearables: 4 Hz (E4); Lab: 2000 Hz (BIOPAC); 4 Hz sufficient for SCL |
| Electrode placement | Fingers (palmar), wrist (dorsal) | Fingers are gold standard; wrist is noisier but unobtrusive |
| Hardware | Empatica E4/EmbracePlus, BIOPAC MP150, Shimmer GSR3+ | E4 is most popular in therapy research |
| Filter | 5 Hz low-pass Butterworth | Remove high-frequency noise |

### Preprocessing
1. **Tonic/phasic decomposition**: Separate slow-varying SCL (skin conductance level) from fast SCR (skin conductance responses) using Ledalab (nonneg deconvolution) or cvxEDA (convex optimization)
2. **Artifact removal**: EDA Explorer, visual inspection, or motion artifact detection
3. **Downsampling**: 2000 Hz lab data typically downsampled to 10-20 Hz for analysis

### Synchrony Computation Methods
| Method | Description | Parameters | Reference |
|--------|-------------|------------|-----------|
| **Moving window correlation** | Pearson r in sliding windows | Window: 30s, step: 10s | Marci et al. (2007) |
| **Single Session Index (SSI)** | Correlation of window-wise slopes | 5s slope windows, 15s correlation windows | Gernert et al. (2024) |
| **Windowed cross-correlation (WCC)** | Peak cross-correlation across lags per window | Window: 8s, lag: +-4s | Boker et al. (2002) |
| **Normalized Symbolic Transfer Entropy (NSTE)** | Directed information flow | 60s sliding windows | Slovak et al. (2014) |
| **Cross-Recurrence Quantification (CRQA)** | Nonlinear dynamics | Embedding parameters | Orsucci et al. (2016) |
| **Concordance (slope correlation)** | Correlation of local slopes | 15s windows, 1s steps | Tschacher & Meier (2020) |

### Surrogate/Null Controls
- **Pseudo-dyad shuffling**: Pair recordings from non-interacting individuals (Gernert et al., 2024)
- **Segment-wise shuffling**: Shuffle time segments within a session (Tschacher & Meier, 2020)
- **Circular shift**: Circular permutation surrogates (analogous to CADENCE approach)

---

## Can EDA Be Derived from ECG/PPG?

**Short answer: No.**

EDA and cardiac measures reflect different branches of the sympathetic nervous system:
- **EDA** → sudomotor nerve activity (sweat glands) → purely sympathetic
- **ECG/HRV** → cardiac sympathetic + parasympathetic (vagal)
- **PPG** → peripheral blood flow → mixed sympathetic/parasympathetic

Key evidence:
- Behrens et al. (2020): Only SCL (not HR) predicted cooperative success, demonstrating functional dissociation
- A review found that "sympathetic sudomotor nerve activity may be different from the cardiac sympathetic dynamics" (Nature Scientific Reports, 2020)
- Gregorini et al. (2025) meta-analysis: Peripheral physiological synchrony (EDA) had r = 0.32 with outcomes, while other measures did not reach significance

**EDA requires dedicated hardware.** It cannot be estimated from the ECG (Polar H10) or PPG signals currently in the CADENCE pipeline.

---

## Wearable EDA Sensors for Therapy Settings

### Empatica EmbracePlus (Recommended)
- **FDA-cleared** medical-grade wristband
- **Sensors:** EDA (4 Hz), PPG (64 Hz), accelerometer, temperature
- **Form factor:** Minimalistic smartwatch design -- suitable for therapy
- **Data access:** Raw data export, no black-box algorithms
- **Battery:** Multi-day continuous recording
- **Cost:** Research license required
- **Validation:** Validated for clinical research, used by NASA, DoD

### Empatica E4 (Legacy)
- **EDA:** 4 Hz sampling, 0.01-100 uS range, 900 pS resolution
- **Concern:** Milstein & Gordon (2020) found ~73% of EDA recordings were noisy; wrist-based EDA is less reliable than finger-based
- **Status:** Being phased out in favor of EmbracePlus

### BIOPAC MP150 + BN-PPGED (Gold Standard)
- **Wireless:** BioNomadix transmitter
- **EDA:** Up to 2000 Hz, finger or palm electrodes
- **Advantage:** Research gold standard, highest signal quality
- **Disadvantage:** More obtrusive (finger electrodes), expensive

### Shimmer GSR3+
- **EDA + PPG** combined sensor
- **Concern:** Two finger electrodes restrict hand movement
- **Cost:** Lower than BIOPAC

### Fitbit Sense / Sense 2
- **Consumer-grade** EDA sensor
- **Concern:** Not validated for dyadic synchrony research; limited raw data access

### Recommendation for CADENCE
The **Empatica EmbracePlus** is the best option for therapy research:
- Unobtrusive (looks like a normal watch)
- FDA-cleared, research-grade
- Raw data access at 4 Hz -- sufficient for EDA synchrony
- Already the standard in recent psychotherapy synchrony research
- Complementary to Polar H10 (ECG) already in pipeline

---

## Real-Time EDA Synchrony Computation

### Streaming Algorithm Design
For integration with CADENCE's streaming EWLS pipeline:

1. **Input:** Two EDA streams at 4 Hz (one per participant)
2. **Preprocessing (online):**
   - 4th-order Butterworth low-pass at 1 Hz (remove artifacts)
   - Optional: online tonic/phasic decomposition via recursive filter
   - Z-score normalization per participant (running mean/std)
3. **Synchrony features (per window):**
   - Windowed cross-correlation (8-30s window, +-5s lag)
   - Concordance (slope correlation in 15s windows)
   - Peak cross-correlation magnitude and lag
   - Optional: Transfer entropy for directed coupling
4. **Output rate:** 1 Hz (one synchrony estimate per second)

### Mapping to CADENCE Architecture
EDA synchrony maps naturally as a new source modality:
- **Raw EDA features** (2 channels: SCL + SCR rate per person) → can be treated like ECG features
- **EDA synchrony features** (cross-correlation, concordance, lag) → new "interbrain-like" cross-person features
- **Temporal parameters:** 5s max lag aligns with current CADENCE basis function settings
- **Expected rate:** 1-4 Hz → similar to ECG features (2 Hz)

---

## Key Effect Sizes Across Studies

| Study | Metric | Outcome | Effect Size |
|-------|--------|---------|-------------|
| Marci et al. (2007) | SC concordance | Perceived empathy | r = 0.47 |
| Gernert et al. (2024) | SCR SSI | Symptom change (GSI) | R^2 = 0.43 |
| Gregorini et al. (2025) | Peripheral physiology NVS | Alliance/outcome (meta) | r = 0.32 |
| Behrens et al. (2020) | SCL synchrony | Cooperative success | f^2 = 0.013 |
| Tschacher et al. (2025) | EDA synchrony | Therapeutic bond | Significant (patient-leading) |
| Messina et al. (2013) | SC concordance | Perceived empathy | Significant positive |
| Kykyri et al. (2019) | EDA concordance | Well-being change | Significant (for women) |

---

## Synthesis: Why EDA Synchrony Should Be in CADENCE

1. **Strongest physiological biomarker:** Meta-analysis shows r = 0.32 for peripheral physiological synchrony vs. outcomes -- the ONLY modality with a significant positive effect.

2. **Predicts clinically meaningful outcomes:** Symptom change (R^2 = 0.43), therapeutic alliance, perceived empathy.

3. **Directional information available:** Patient-leading vs. therapist-leading synchrony have different clinical meanings (Tschacher et al., 2025) -- maps directly to CADENCE's directional coupling framework.

4. **Temporal dynamics match CADENCE:** Standard EDA synchrony lags of 3-7 seconds align with CADENCE's 5s maximum lag window. The 4 Hz sampling rate is similar to existing ECG (2 Hz) features.

5. **Complementary to existing modalities:** EDA is purely sympathetic; ECG/HRV is mixed sympathetic-parasympathetic. They provide non-redundant information about autonomic coupling.

6. **Feasible hardware:** Empatica EmbracePlus is unobtrusive, FDA-cleared, and provides raw data. Would add one wristband per participant.

7. **Well-established signal processing:** cvxEDA and Ledalab provide robust tonic/phasic decomposition. Windowed cross-correlation is standard.

8. **Cannot be derived from existing signals:** EDA requires dedicated sudomotor measurement hardware -- it is not estimable from ECG or PPG.

---

## References (Alphabetical)

1. Behrens, F., Snijdewint, J.A., Moulder, R.G., Prochazkova, E., Sjak-Shie, E.E., Boker, S.M., & Kret, M.E. (2020). Physiological synchrony is associated with cooperative success in real-life interactions. *Scientific Reports*, 10, 19609.

2. Di Mascio, A., Boyd, R.W., & Greenblatt, M. (1955). Physiological correlates of tension and antagonism during psychotherapy. *Diseases of the Nervous System*, 16(1).

3. Gernert, C.C., Nelson, A., Falkai, P., & Falter-Wagner, C.M. (2024). Synchrony in psychotherapy: High physiological positive concordance predicts symptom reduction and negative concordance predicts symptom aggravation. *International Journal of Methods in Psychiatric Research*, 33(1), e1978.

4. Gregorini, S., Lutz, W., Tschacher, W., Meier, D., & Ramseyer, F.T. (2025). Potential role of nonverbal synchrony in psychotherapy: A meta-analysis. *Counselling and Psychotherapy Research*.

5. Karvonen, A., Kykyri, V.L., Kaartinen, J., Penttonen, M., & Seikkula, J. (2016). Sympathetic nervous system synchrony in couple therapy. *Journal of Marital and Family Therapy*, 42(3), 383-395.

6. Kleinbub, J.R. (2017). State of the art of interpersonal physiology in psychotherapy: A systematic review. *Frontiers in Psychology*, 8, 2053.

7. Kleinbub, J.R., Mannarini, S., & Palmieri, A. (2020). Interpersonal biofeedback in psychodynamic psychotherapy. *Frontiers in Psychology*, 11, 1655.

8. Kykyri, V.L., Karvonen, A., Wahlstrom, J., Kaartinen, J., Penttonen, M., & Seikkula, J. (2019). Sympathetic nervous system synchrony: An exploratory study of its relationship with the therapeutic alliance and outcome in couple therapy. *Psychotherapy Research*, 29(6), 766-785.

9. Marci, C.D. & Orr, S.P. (2006). The effect of emotional distance on psychophysiologic concordance and perceived empathy between patient and interviewer. *Applied Psychophysiology and Biofeedback*, 31(2), 115-128.

10. Marci, C.D., Ham, J., Moran, E., & Orr, S.P. (2007). Physiologic correlates of perceived therapist empathy and social-emotional process during psychotherapy. *Journal of Nervous and Mental Disease*, 195(2), 103-111.

11. Messina, I., Palmieri, A., Sambin, M., Kleinbub, J.R., Voci, A., & Calvo, V. (2013). Somatic underpinnings of perceived empathy: The importance of psychotherapy training. *Psychotherapy Research*, 23(2), 169-177.

12. Milstein, N. & Gordon, I. (2020). Validating measures of electrodermal activity and heart rate variability derived from the Empatica E4 utilized in research settings that involve interactive dyadic states. *Frontiers in Behavioral Neuroscience*, 14, 148.

13. Palmieri, A., Kleinbub, J.R., Calvo, V., Benelli, E., Messina, I., Sambin, M., & Voci, A. (2018). Attachment-security prime effect on skin-conductance synchronization in psychotherapists: An empirical study. *Journal of Counseling Psychology*, 65(4), 490-499.

14. Palumbo, R.V., Marraccini, M.E., Weyandt, L.L., Wilder-Smith, O., McGee, H.A., Liu, S., & Goodwin, M.S. (2017). Interpersonal autonomic physiology: A systematic review of the literature. *Personality and Social Psychology Review*, 21(2), 99-141.

15. Prochazkova, E., Sjak-Shie, E.E., Behrens, F., Wieling, M., & Kret, M.E. (2022). Physiological synchrony is associated with attraction in a blind date setting. *Nature Human Behaviour*, 6(2), 269-278.

16. Robinson, J.W., Herman, A., & Kaplan, B.J. (1982). Autonomic responses correlate with counselor-client empathy. *Journal of Counseling Psychology*, 29(2), 195-198.

17. Slovak, P., Tennent, P., Reeves, S., & Fitzpatrick, G. (2014). Exploring skin conductance synchronisation in everyday interactions. *Proceedings of NordiCHI 2014*.

18. Stratford, T., Lal, S., & Meara, A. (2012). Neuroanalysis of therapeutic alliance in the symptomatically anxious: The physiological connection revealed between therapist and client. *American Journal of Psychotherapy*, 66(1), 1-21.

19. Tschacher, W. & Meier, D. (2020). Physiological synchrony in psychotherapy sessions. *Psychotherapy Research*, 30(5), 558-573.

20. Tschacher, W., Ribeiro, E., Goncalves, A., Sampaio, A., Moreira, P., & Coutinho, J. (2025). Electrodermal synchrony of patient and therapist as a predictor of alliance and outcome in psychotherapy. *Frontiers in Psychology*, 16, 1545719.
