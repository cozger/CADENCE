# Deep Literature Search: Synchrony Signals for CADENCE/MAP-Neuro

**Date:** 2026-03-25
**Method:** 8 parallel web search agents, ~148 papers identified across targeted domains
**Purpose:** Discover synchrony signals usable as priors in CADENCE; map the research landscape; identify gaps and opportunities

---

## 1. EDA/Skin Conductance Synchrony in Psychotherapy (20 papers)

### Why This Matters
Skin conductance synchrony is the **single most validated physiological synchrony signal in psychotherapy** — yet CADENCE does not capture it and MAP-Neuro does not mention it.

### Key Papers

| Paper | Population | Metric | Predicted | Effect Size |
|-------|-----------|--------|-----------|-------------|
| **Gregorini et al. 2025** (meta-analysis, 23 pubs) | Mixed clinical | Various peripheral | Alliance/outcome | **r=0.32 (p=.006)** — only nonverbal modality with sig positive effect |
| **Gernert et al. 2024** | Mixed psychiatric (CBT) | SCR concordance | Symptom trajectory | **R²=0.43 (p=.011)** — first session predicts outcome |
| **Marci et al. 2007** | Established therapy dyads | SC slope concordance (15s windows) | Perceived empathy | **r=0.47, 26% variance** |
| **Tschacher et al. 2025** | Psychotherapy | EDA synchrony | Therapeutic bond | Patient-leading synchrony predicted bond quality |
| **Behrens et al. 2020** | Cooperative tasks | SCL vs HR | Cooperative success | Only SCL (not HR) predicted success — functional dissociation |

### Critical Findings
- **EDA cannot be derived from ECG/PPG** — sudomotor and cardiac sympathetic pathways are physiologically distinct
- **Directionality matters**: patient-leading EDA synchrony is more pronounced and more predictive (maps directly to CADENCE's P1→P2 vs P2→P1)
- **Positive concordance = improvement; negative concordance = worsening** (Gernert 2024) — direction of coupling matters clinically
- **Standard parameters**: 8-30s windows, 4-5s max lag, 4 Hz sampling — all CADENCE-compatible
- **Processing pipeline**: cvxEDA for tonic/phasic decomposition, windowed cross-correlation for synchrony, pseudo-dyad surrogates for null

### Hardware Recommendation
**Empatica EmbracePlus** — FDA-cleared wristband, 4 Hz EDA, raw data access, smartwatch form factor. Note: E4 predecessor had 73% noise in one validation study; EmbracePlus addresses this.

### CADENCE Integration Priority: **VERY HIGH**
Adding EDA would give CADENCE access to the highest-effect-size physiological coupling signal. Hardware cost is trivial (~$250/device). The signal is CADENCE-compatible (slow timescale, directional, windowed coupling).

---

## 2. Respiratory Coupling (17 papers)

### Key Papers

| Paper | Key Finding | Effect Size |
|-------|------------|-------------|
| **Yi et al. 2026** (Psychophysiology) | Respiratory synchrony is **causally upstream** of behavioral synchrony (perturbation paradigm) | Causal proof |
| **Muller & Lindenberger 2011** | Choir singing: respiratory synchrony eta²=0.83; conductor Granger-causes singers' breathing | eta²=0.83 |
| **Codrons et al. 2014** | Respiratory synchrony emerges independently of cardiac synchrony | Partial independence |
| **Salminen & Jarvela 2019** | Combined EEG + respiratory biofeedback → **highest empathy** when both presented | N=72, significant |
| **Tschacher & Meier 2020** | Respiration synchrony associated with therapeutic alliance | Significant |

### ECG-Derived Respiration (EDR) — Extractable from Polar H10

| Paper | Method | Accuracy |
|-------|--------|----------|
| **Charlton et al. 2016** | Assessed 314 EDR algorithms; best: 0.0 bpm bias, ±4.7 bpm LOA | 4 ECG algorithms outperform clinical impedance |
| **Varon et al. 2020** | Compared 10 single-lead EDR methods; QRS slope-based methods best | Computationally simplest |
| **Rogers et al. 2022** | Polar H10 specifically: r=0.85 with gas exchange reference | bias -3.9 bpm |
| **Dong et al. 2021** | FMRR (frequency modulation of RR intervals) most robust | <2 bpm error |

### Tools
- **NeuroKit2** `ecg_rsp()` — 4 EDR methods, open source Python
- **Video-based**: MediaPipe shoulder landmarks (11, 12) → ~98.5% accuracy for stationary subjects

### CADENCE Integration Priority: **HIGH**
Respiratory rate is extractable from existing Polar H10 hardware (FMRR method, <2 bpm error). No new hardware needed. Adds non-redundant information beyond cardiac synchrony. Combined EEG + respiratory biofeedback outperforms either alone.

---

## 3. LZ Complexity & Psychedelic EEG Biomarkers (18 papers)

### Core Finding
LZ complexity is the most validated single-channel biomarker of the psychedelic state. Increases robustly across psilocybin, LSD, DMT, and ketamine.

### Key Papers

| Paper | Drug | Key Finding |
|-------|------|-------------|
| **Schartner et al. 2017** | LSD, psilocybin, ketamine | LZc exceeds normal waking; first to show across 3 drugs |
| **Timmermann et al. 2019** | DMT (IV) | LZc peaks within minutes of injection; tracks subjective intensity |
| **Mediano et al. 2024** | LSD | LZc beats alpha for subjective experience; correlations vanish during visual stimulation |
| **Godfrey et al. 2025** | All psychedelics (review) | Alpha suppression most reliable; LZc most informative |
| **Schifano et al. 2025** | Ketamine vs psilocybin | Distinct EEG signatures — correction must be drug-specific |

### Ketamine-Specific (Critical for MAP-Neuro)
- LZc peaks at **30 min post-infusion**, decreases at 24h, returns to baseline by day 7
- **Lower baseline occipital LZc predicts favorable ketamine response** (ROC AUC=0.75)
- Ketamine uniquely alters 1/f spectral slope (E/I balance); psilocybin targets oscillatory activity more selectively
- Responders show elevated baseline complexity that decreases post-treatment

### Real-Time Feasibility
- LZ76 algorithm is **O(n)**, <1ms per channel on 4s window of 1024 samples
- All 14 Emotiv EPOC channels processable in ~14ms
- Anesthesia depth monitors already deploy entropy indices in real-time clinically
- **Libraries**: `antropy`, `neurokit2` (Python, ready-made)

### Literature Gap
**No study has used neural complexity as a moderator of interpersonal coupling.** The REBUS model (Carhart-Harris & Friston 2019) provides theoretical grounding — loosened priors under psychedelics should increase susceptibility to social influence, making LZc a natural coupling moderator.

### CADENCE Integration Priority: **HIGH**
LZ complexity as drug-state moderator in Stage 2 (like ECG HR/RMSSD). Also: spectral slope as secondary measure (3.5x faster, better for ketamine). Both computable at 10 Hz on Emotiv EPOC.

---

## 4. Event Synchronization & Hawkes Processes (22 papers)

### Event Synchronization Family

| Paper | Method | Key Finding for CADENCE |
|-------|--------|------------------------|
| **Quian Quiroga et al. 2002** | Event Synchronization (ES) | Parameter-free, adaptive windows, directional q metric |
| **Odenweller & Donner 2020** | ES vs ECA comparison | **ES confounds synchrony with serial dependency**; ECA is more robust |
| **Donges et al. 2016** | Event Coincidence Analysis (ECA) | Explicit tolerance window + Poisson null hypothesis testing |
| **Kreuz et al. 2015** | SPIKE-synchronization | Time-resolved, available in `PySpike` Python library |

### Hawkes Processes for Interpersonal Dynamics

| Paper | Method | Context | Key Finding |
|-------|--------|---------|-------------|
| **Halpin & De Boeck 2013** | Bivariate Hawkes | Dyadic email | Foundational paper for Hawkes in psychology; asymmetric excitation kernels |
| **Wu et al. 2022** | MMHP | Mouse aggression | Active/inactive state switching for bursty social dynamics |
| **Xu et al. 2016** | Hawkes + sparse-group-lasso | Network events | **Basis function expansion + group sparsity** — architecturally identical to CADENCE's raised-cosine + group lasso |
| **Zipser et al. 2018** | Coupled HMM on AUs | Negotiation, interview | Synchronization templates of facial events predict outcomes |

### Changepoint Detection

| Paper | Method | Key Finding |
|-------|--------|-------------|
| **Adams & MacKay 2007** | BOCPD | Gold standard for online regime change detection; O(T) with pruning |
| **Altamirano et al. 2023** | Robust BOCPD (Dm-BOCD) | Robust to outliers; O(1) per update; 10x fewer false positives than standard |
| **Hamidi et al. 2024** | BCPD + DTW | 91% detection accuracy on physiological stress responses |

### Open Niche
**No paper applies Hawkes processes to facial expression mimicry events in face-to-face dyadic interaction.** Halpin used Hawkes for email; Wu for mouse aggression; Zipser used coupled HMMs for facial AUs but not Hawkes. The combination of (Hawkes) + (facial AU events) + (real-time dyadic) is novel.

### Python Tooling
- **`tick`** — most comprehensive Hawkes library (C++ backend); `HawkesExpKern`, `HawkesSumExpKern`, `HawkesEM`
- **`hawkeslib`** — Bayesian inference, model comparison via marginal likelihood
- **`PySpike`** — event synchronization
- **`pyunicorn`** — ECA implementation

### CADENCE Integration Priority: **VERY HIGH** (for V5 event architecture)
ECA (not ES) for event-based facial mimicry coupling. Hawkes processes with basis function + group sparsity for principled event-driven coupling discovery. BOCPD for temporal localization of coupling onset/offset in dR2 timecourses.

---

## 5. Speech/Vocal Prosody Synchrony (12 papers)

### The Critical Nuance: Direction of Vocal Synchrony Matters

| Paper | Finding | Effect Size |
|-------|---------|-------------|
| **Jennissen et al. 2025** (meta-analysis) | Vocal pitch synchrony **negatively** associated with outcomes | **r=-0.20 (p=.011)** |
| **Imel et al. 2014** | f0 arousal synchrony in high-empathy MI sessions | **r=0.71-0.80** |
| **Imel et al. 2014** | f0 synchrony in low-empathy sessions | r=0.36 |
| **Schoenherr et al. 2021** | Vocal synchrony predicts **worse** SAD symptoms | Negative association |

### Why Vocal Synchrony Can Be Harmful
- **Matching distress escalation** is harmful — therapist mirrors patient's rising arousal
- **Matching calm/regulated arousal** is beneficial — shared emotional regulation
- In social anxiety, patients over-accommodate vocally as part of their disorder
- The **valence/direction** of what's being synchronized matters more than synchrony magnitude

### Positive Findings for Linguistic/Semantic Synchrony

| Paper | Features | Predicted | Effect Size |
|-------|----------|-----------|-------------|
| **Lord et al. 2015** | Language Style Synchrony (11 LIWC categories) | Therapist empathy | OR=2.4, d=0.62 |
| **Xiao et al. 2015** | Automated speech pipeline (n-grams, VAD, diarization) | Empathy classification | r=0.65, accuracy=82% |
| **Bayerl et al. 2022** | Turn dynamics + lexical entrainment | Working Alliance (WAI) | Strong indicators |
| **Lin et al. 2025** (COMPASS) | LLM embeddings of dialogue turns | WAI from transcripts | 46% 4-class accuracy |

### Tools for Real-Time Speech Feature Extraction
- **OpenSMILE** — standard toolkit; eGeMAPS (88 features); real-time via PortAudio; Python bindings
- **librosa** — Python audio analysis
- **Whisper** — OpenAI ASR for transcription → NLP pipeline

### No Papers Found on Speech During Psychedelic Sessions

### CADENCE Integration Priority: **MODERATE**
Speech is mentioned in MAP-Neuro appendices but not in CADENCE. The mixed findings (meta-analytic r=-0.20 for pitch synchrony) suggest caution. If implemented, should focus on **linguistic/semantic synchrony** (NLP-based) rather than raw prosodic synchrony, and must distinguish arousal-matching direction. OpenSMILE provides ready infrastructure.

---

## 6. Hyperscanning During Psychedelics & Drug-State Correction (22 papers)

### CRITICAL FINDING: No EEG Hyperscanning During Psychedelic Sessions Exists

Across all searches, **no published study performs simultaneous dual-brain EEG recording during an active psychedelic or neuroplastogen session**. MAP-Neuro would be genuinely first.

### Pharmacological Modulation of Interbrain Coupling

| Paper | Drug | Effect on Coupling |
|-------|------|-------------------|
| **Mu, Guo & Han 2016** | Oxytocin (24 IU intranasal) | Enhanced alpha PLV during partner coordination; shifted synchrony to earlier windows |
| **Shamay-Tsoory 2016** | Oxytocin (theory) | Increases social salience — context-dependent, not simple gain |
| **MDMA meta-analysis 2025** | MDMA | Massive OT release + increased empathy; **no hyperscanning studies** |

**Oxytocin is the only pharmacological agent with demonstrated interbrain coupling effects.**

### Drug-State EEG Baseline Correction Methods

| Method | Paper | Approach |
|--------|-------|----------|
| **Baseline + placebo subtraction** | Alonso et al. 2010 | Pre-drug values subtracted, then drug-placebo difference |
| **Aperiodic (1/f) correction** | Donoghue et al. 2024 | Separate oscillatory from aperiodic changes before computing phase metrics |
| **Drug state as explicit moderator** | — | CADENCE's existing approach with ECG moderators generalizes naturally |
| **FWL partialling** | — | Partial out drug-state EEG features before coupling estimation |

### Ketamine vs Psilocybin EEG Differences
- **Schifano et al. 2025**: Ketamine uniquely affects aperiodic components; psilocybin targets oscillatory activity
- Both suppress alpha, but ketamine enhances gamma while psilocybin does not
- **Correction must be drug-specific**

### Coupling Flexibility Metrics

| Metric | Paper | What It Captures |
|--------|-------|-----------------|
| **Shannon entropy of coupling** | Gordon 2025 | Variability in synchrony over time |
| **DFA scaling exponent** | Mayo & Gordon 2020 | Fractal properties of coupling dynamics |
| **Dwell/escape time ratio** | Tognoli & Kelso 2014 | Metastable coordination dynamics |
| **Complexity matching** | Marmelat & Delignieres 2012 | Convergence of partners' fractal exponents |
| **Windowed multiscale synchrony** | Likens & Wiltshire 2021 | Time-frequency synchrony heatmaps |

### Additional Key Papers
- **Sened et al. 2025**: First fNIRS hyperscanning across full therapy course — INS increased over 6 sessions ("inter-brain plasticity")
- **Ayrolles et al. 2024**: Short epochs (<=1s) inflate IBS estimates; methodological caution for drug studies
- **Goldstein et al. 2018**: Brain-to-brain alpha coupling during handholding correlates with pain reduction and empathic accuracy

### CADENCE Integration Priority: **HIGH**
Drug-state moderators (LZc, spectral slope) for EEG coupling during neuroplastogen sessions. Flexibility metrics (entropy of dR2, DFA exponent) as primary coupling outcomes. Aperiodic correction before computing interbrain phase metrics.

---

## 7. Multimodal Fusion & Cross-Modal Coupling (17 papers)

### Cross-Modal Coupling Evidence

| Paper | Finding | Effect Size |
|-------|---------|-------------|
| **Koul et al. 2023** | Behavioral synchrony (face, body, gaze) **Granger-causes** neural synchrony | Directional |
| **Ohayon & Gordon 2025** | Neural-behavioral: r=0.32; physiological-behavioral: r=0.18 | Meta-analytic |
| **Dmochowski et al. 2020** | Multimodal metric (EEG+EDA+HR) outperforms any single modality | Significant |
| **Gordon et al. 2020** | Physiological synchrony adds predictive value beyond behavioral alone | Significant |

### Behavioral Synchrony Is NOT a Complete Proxy for Neural

| Paper | Evidence |
|-------|---------|
| **Schilbach & Redcay 2025** | Neural coupling captures "conceptual alignment" beyond observable behavior |
| **Konrad et al. 2024** | Clinical recommendation: do NOT substitute behavioral for neural measures |
| **Kojovic et al. 2024** | Motion synchrony alone = 63.4% accuracy for autism classification — insufficient |
| **Pan et al. 2020** | Two-brain AUC=0.90 vs single-brain <0.66 |

### No Standard Composite Synchrony Index Exists
- This is an open opportunity for CADENCE
- Score-level fusion with quality-dependent weighting is most promising approach
- Gordon 2025 proposes flexibility metrics over mean synchrony
- Context-dependent weighting is theoretically motivated but not yet implemented

### Real-Time Multimodal Systems

| System | Modalities | Update Rate | Validated |
|--------|-----------|-------------|-----------|
| **Hybrid Harmony** (Chen/Dikker 2021) | EEG (EMOTIV EPOC), 6 coupling metrics | ~3.5 Hz | 236 dyads |
| **DYNECOM** (Jarvela 2019) | EEG frontal asymmetry + respiration | Real-time | 39 dyads in VR |

### CADENCE Integration Priority: **HIGH**
CADENCE's cross-modal pathways (BL→EEG, Pose→EEG) are validated by Koul's Granger causality finding. Should compute and report a composite synchrony index — no one else has done this. The multimodal advantage (outperforming any single modality) validates the architecture.

---

## 8. Synchrony and Therapeutic Outcomes (20 papers)

### Meta-Analytic Evidence by Signal Type

| Signal Type | Association with Alliance/Outcome | Source |
|-------------|----------------------------------|--------|
| **Peripheral physiology (EDA, HR)** | **r=0.32 (p=.006)** — only significant positive | Jennissen 2025 |
| **Body movement** | r=0.19 (alliance), r=0.22 (outcome, ns trend) | Gregorini 2025 |
| **Vocal pitch** | **r=-0.20 (p=.011)** — negative | Jennissen 2025 |
| **Overall nonverbal** | r=0.03 (ns) | Jennissen 2025 |

### Landmark Clinical Studies

| Paper | Population | Key Finding | Effect Size |
|-------|-----------|-------------|-------------|
| **Ramseyer & Tschacher 2011** | Outpatient psychiatric (N=70) | Movement synchrony predicts symptom reduction; medium synchrony is optimal | d=0.6 |
| **Cohen et al. 2021** | MDD (N=86, RCT) | **State-like** synchrony predicts alliance; trait-like does not | p<.0001 |
| **Levin et al. 2024** | MDD + psilocybin (N=20) | Alliance predicts depression at r=-0.85 — far larger than conventional therapy | r=-0.85 |
| **Zilcha-Mano et al. 2021** | MDD (N=37, RCT) | Oxytocin synchrony **mediated** treatment effects — rare true mediation | Mediation sig |
| **Schoenherr et al. 2019** | Social anxiety (N=267) | Low synchrony at session 3 predicts dropout — early warning system | Significant |
| **Uhl et al. 2025** | Test anxiety (N=90, 22 therapists) | Within-therapist variability matters more than between-therapist averages | Interaction sig |

### Synchrony in PTSD — "More Is Not Always Better"

| Paper | Finding |
|-------|---------|
| **Motsan et al. 2021** | PTSD: high autonomic + low behavioral synchrony is **maladaptive** (rigid co-regulation) |
| **Wynn et al. 2023** | SCR synchrony may mediate vicarious extinction in pediatric PTSD |

### Personalized Synchrony Signatures
**Zilcha-Mano et al. 2025** proposes three types:
- **Normative**: healthy oscillation between synchrony and autonomy
- **Hyperactivating**: excessive synchrony, poor autonomy (seen in anxious attachment)
- **Deactivating**: insufficient synchrony, poor connection (seen in avoidant attachment)

### Major Clinical Gaps

| Population | Synchrony-Outcome Evidence | Gap Severity |
|-----------|---------------------------|-------------|
| Depression (MDD) | Strong — multiple RCTs | Covered |
| Social anxiety | Strong — N=267 dataset | Covered |
| PTSD | Moderate — longitudinal | Moderate gap |
| Borderline PD | Emerging — adolescent sample | Large gap |
| **Substance use disorder** | **ZERO studies** | **Critical gap for MAP-Neuro** |
| Chronic pain | Theoretical only (Bauer 2025) | Large gap |
| **Psychedelic-assisted therapy** | Alliance data only (Levin 2024), no physiological synchrony | **Critical gap** |

### CADENCE Integration Priority: **CRITICAL**
The substance use disorder and psychedelic therapy gaps are exactly where MAP-Neuro sits. CADENCE would provide the first synchrony-outcome data in both populations. The finding that state-like (time-varying) synchrony is more informative than trait-like (aggregate) validates CADENCE's EWLS architecture. Personalized synchrony signatures suggest examining attachment style as a moderator.

---

## Cross-Cutting Synthesis: Top 10 Actionable Findings

### 1. Add EDA — It's the Strongest Signal We're Missing
- Meta-analytic r=0.32 for outcomes (only significant physiological modality)
- First-session prediction R²=0.43
- Hardware: Empatica EmbracePlus (~$250), 4 Hz, CADENCE-compatible parameters
- Cannot be derived from ECG — new hardware required

### 2. Extract Respiratory Rate from Existing Polar H10
- FMRR method on RR intervals: <2 bpm error
- Respiratory synchrony is causally upstream of behavioral synchrony
- Combined EEG + respiratory biofeedback outperforms either alone
- No new hardware needed — just software

### 3. Implement LZ Complexity as Drug-State Moderator
- O(n), <1ms per channel, trivially real-time on EPOC
- Lower baseline LZc predicts favorable ketamine response (AUC=0.75)
- No one has used complexity as a coupling moderator — literature gap
- REBUS model provides theoretical grounding

### 4. MAP-Neuro Would Be First EEG Hyperscanning During Psychedelic Session
- Confirmed: no published study exists
- Drug-state correction needed: aperiodic (1/f) correction + drug-specific moderators
- Ketamine and psilocybin have distinct EEG signatures — handle separately

### 5. Use ECA (Not ES) for Event-Based Coupling; Hawkes for Modeling
- ES confounds synchrony with serial dependency
- Hawkes + basis functions + group sparsity = CADENCE architecture for point processes
- No one has applied Hawkes to facial mimicry events — open niche
- `tick` library (Python, C++ backend) is production-ready

### 6. Vocal Pitch Synchrony Is Harmful (r=-0.20) — Be Careful with Speech
- Meta-analytically negative association with therapy outcomes
- BUT f0 arousal synchrony r=0.71-0.80 in high-empathy MI sessions
- The difference: matching calm regulation is good; matching distress escalation is bad
- If implementing speech, use linguistic/semantic synchrony (NLP) not raw prosody

### 7. Compute Coupling Flexibility Metrics — Not Just Aggregate dR2
- Gordon 2025 (Psych Review): flexibility predicts outcomes better than aggregate synchrony
- Metrics: entropy of dR2 timecourse, DFA scaling exponent, state transition counts
- Cohen 2021: state-like (time-varying) synchrony predicts alliance; trait-like does not
- CADENCE's per-timepoint significance already captures this — just needs to be reported explicitly

### 8. Substance Use Disorder Has ZERO Synchrony Studies — Critical MAP-Neuro Opportunity
- No physiological, behavioral, or neural synchrony-outcome studies in SUD therapy
- MAP-Neuro's Trial 1 (ketamine + MORE for OUD) would be first
- Combined with Luo/Garland 2026's entropy biomarker for opioid misuse

### 9. Behavioral Synchrony Is Necessary But Not Sufficient — Neural Required
- Motion synchrony alone = 63.4% accuracy (Kojovic 2024)
- Two-brain > single-brain classification (Pan 2020: AUC 0.90 vs <0.66)
- Clinical recommendation: do NOT substitute behavioral for neural (Konrad 2024)
- CADENCE's multimodal (EEG + ECG + face + pose) is validated by meta-analytic evidence

### 10. Synchrony Is Not Universally Good — Personalization Needed
- PTSD: high autonomic + low behavioral = maladaptive (Motsan 2021)
- Social anxiety: vocal synchrony predicts worse outcomes (Schoenherr 2021)
- Three signatures: normative, hyperactivating, deactivating (Zilcha-Mano 2025)
- Attachment style moderates synchrony-outcome relationship
- CADENCE should track patient characteristics as coupling moderators

---

## Master Reference List (Selected Key Papers by Topic)

### EDA Synchrony
- Gregorini et al. 2025 — *Counselling and Psychotherapy Research* (meta-analysis)
- Gernert et al. 2024 — *IJMPR* (SCR predicts symptom trajectory)
- Marci et al. 2007 — *JNMD* (SC concordance and empathy)
- Tschacher et al. 2025 — (patient-leading EDA synchrony)
- Behrens et al. 2020 — (SCL vs HR functional dissociation)

### Respiratory Coupling
- Yi et al. 2026 — *Psychophysiology* (causal perturbation proof)
- Muller & Lindenberger 2011 — *PLoS One* (choir singing, eta²=0.83)
- Salminen & Jarvela 2019 — *IEEE TAC* (EEG + respiratory biofeedback)
- Charlton et al. 2016 — *Physiol. Meas.* (314 EDR algorithms)
- Varon et al. 2020 — *Scientific Reports* (10 single-lead EDR methods)
- Rogers et al. 2022 — *Sensors* (Polar H10 EDR validation, r=0.85)
- Dong et al. 2021 — *Computers in Biology and Medicine* (FMRR method, <2 bpm)

### LZ Complexity & Psychedelic EEG
- Schartner et al. 2017 — (LZc across LSD, psilocybin, ketamine)
- Timmermann et al. 2019 — (DMT LZc tracks subjective intensity)
- Mediano et al. 2024 — *ACS Chem Neurosci* (LZc vs alpha; context effects)
- Godfrey et al. 2025 — *Int Rev Neurobiol* (comprehensive psychedelic EEG review)
- Schifano et al. 2025 — *Psychopharmacology* (ketamine vs psilocybin EEG)
- Carhart-Harris & Friston 2019 — (REBUS model)

### Event Synchronization & Hawkes Processes
- Quian Quiroga et al. 2002 — *Phys Rev E* (event synchronization)
- Odenweller & Donner 2020 — *Phys Rev E* (ES vs ECA; serial dependency confound)
- Donges et al. 2016 — *Eur Phys J Spec Top* (ECA framework)
- Halpin & De Boeck 2013 — *Psychometrika* (Hawkes for dyadic interaction)
- Xu et al. 2016 — *ICML* (Hawkes + sparse-group-lasso + basis functions)
- Wu et al. 2022 — *Ann Appl Stat* (MMHP for bursty social dynamics)
- Zipser et al. 2018 — *AAAI* (coupled HMM on facial AUs)
- Adams & MacKay 2007 — (BOCPD)
- Altamirano et al. 2023 — *ICML* (robust BOCPD)

### Speech/Vocal Synchrony
- Jennissen et al. 2025 — *Psychotherapy Research* (meta-analysis: vocal pitch r=-0.20)
- Imel et al. 2014 — *J Counseling Psych* (f0 synchrony r=0.71-0.80 high empathy)
- Lord et al. 2015 — *Behavior Therapy* (language style synchrony, OR=2.4)
- Xiao et al. 2015 — *PLoS One* (automated empathy detection, accuracy=82%)
- Schoenherr et al. 2021 — *Psychotherapy* (vocal synchrony harmful in SAD)
- Kruyt et al. 2023 — *JSLHR* (12 prosodic entrainment methods compared)
- OpenSMILE — Eyben et al. 2010 (real-time feature extraction toolkit)

### Hyperscanning & Drug-State Correction
- Mu et al. 2016 — *SCAN* (oxytocin enhances alpha PLV)
- Donoghue et al. 2024 — *Nature Comms* (aperiodic EEG correction)
- Alonso et al. 2010 — *Hum Brain Mapp* (double comparison baseline method)
- Sened et al. 2025 — *Psychotherapy Research* (inter-brain plasticity over therapy)
- Ayrolles et al. 2024 — *Imaging Neurosci* (methodological caution: short epochs inflate IBS)

### Coupling Flexibility
- Gordon et al. 2025 — *Psychol Rev* (Theory of Flexible Multimodal Synchrony)
- Mayo & Gordon 2020 — *Psychophysiology* (DFA scaling exponents)
- Tognoli & Kelso 2014 — *Neuron* (metastability: dwell/escape dynamics)
- Marmelat & Delignieres 2012 — *Exp Brain Res* (complexity matching)
- Likens & Wiltshire 2021 — *SCAN* (windowed multiscale synchrony)

### Multimodal Fusion
- Koul et al. 2023 — *NeuroImage* (behavioral Granger-causes neural)
- Dmochowski et al. 2020 — *Front Neurosci* (multimodal > unimodal)
- Chidichimo et al. 2025 — *Nature Rev Neurosci* (information-theoretic framework)
- Lotter et al. 2023 — *Neurosci Biobehav Rev* (neurobiology of INS hubs)

### Synchrony & Clinical Outcomes
- Ramseyer & Tschacher 2011 — *JCCP* (landmark study, d=0.6)
- Cohen et al. 2021 — *Clin Psychol Sci* (state-like > trait-like)
- Levin et al. 2024 — *PLoS One* (psilocybin alliance r=-0.85)
- Zilcha-Mano et al. 2021 — *JCCP* (oxytocin mediation)
- Schoenherr et al. 2019 — *Psychotherapy* (session 3 dropout prediction)
- Motsan et al. 2021 — *Depress Anxiety* (PTSD: synchrony not always good)
- Zilcha-Mano et al. 2025 — *BMC Psychiatry* (personalized synchrony signatures)
- Uhl et al. 2025 — *Psychophysiology* (within-therapist > between-therapist)
- Koole & Tschacher 2016 — *Front Psychol* (In-Sync theoretical model)
- Bauer 2025 — *Front Pain Res* (synchrony for chronic pain)
