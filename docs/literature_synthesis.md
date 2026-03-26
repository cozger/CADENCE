## Comprehensive Literature Synthesis: Synchrony Signals for CADENCE/MAP-Neuro

### I. Available Synchrony Signals by Modality

These are the signals the literature identifies as carrying interpersonal coupling information, organized by what CADENCE already captures vs. what's new.

#### A. EEG Interbrain — Already in CADENCE (PLV, wPLI wavelet features)

| Signal | Band | What It Indexes | Key Evidence | Effect Size |
|--------|------|----------------|-------------|-------------|
| **Alpha CCorr/PLV** | 8-12 Hz | Shared attention, engagement | Davidesco 2023: predicts learning at ~300ms lag | r=0.46 |
| **Theta CCorr** | 4-8 Hz | Emotional processing, mindfulness | Chen 2022: r=-0.37 with mindful awareness | r=0.37 |
| **Beta IC/PLV** | 13-30 Hz | Shared action representations, social closeness | Dikker 2021: r=0.16; Reinero 2021 | r=0.16-0.21 |
| **Imaginary Coherence** | broadband | Noise-robust phase coupling (removes volume conduction) | Dikker 2021, Chen 2021 (Hybrid Harmony) | validated N=726 |
| **Projected Power Correlation** | broadband | Envelope co-fluctuation (hundreds-of-ms scale) | Dikker 2021: captures different info than IC | complementary to phase |
| **Envelope Correlation** | per-band | Amplitude co-modulation | Koul 2023: superior to phase for INS; HyPyP standard | Granger-caused by behavior |

**NEW signals CADENCE should consider:**
- **CCorr** (circular correlation coefficient) — used in Davidesco 2023, Chen 2022; argued more robust than PLV to spurious synchronization
- **Projected Power Correlation (PPC)** — Dikker 2021; removes shared instantaneous projection, captures different coupling timescale than phase metrics
- **Amplitude envelope correlation** — Koul 2023 argues superior to phase-based INS; captures "similarity of cognitive states" at larger timescales

#### B. EEG Individual (Within-Person) — Partially in CADENCE (wavelet power)

| Signal | What It Indexes | Key Evidence |
|--------|----------------|-------------|
| **Lempel-Ziv complexity** | Neural entropy, psychedelic state intensity | Mediano 2024: beats alpha for subjective experience; Godfrey 2025 review |
| **Alpha suppression** | Most reliable psychedelic biomarker | Godfrey 2025: consistent across all psychedelics |
| **Frontal midline theta** | Executive function, engagement | Enriquez-Geppert 2025: NF target for psilocybin-assisted sessions |
| **Aperiodic 1/f slope** | Neural excitability | Godfrey 2025: flattens under psychedelics |

**Key insight for MAP-Neuro**: During neuroplastogen sessions, alpha power drops and LZ complexity rises. The interbrain synchrony metrics CADENCE computes (PLV, wPLI) will shift baselines. LZ complexity could serve as a **drug-state moderator** (like ECG HR/RMSSD currently moderate coupling).

#### C. ECG/Autonomic — Already in CADENCE (HR, RMSSD)

| Signal | What It Indexes | Key Evidence | Effect Size |
|--------|----------------|-------------|-------------|
| **HF-HRV cross-correlation** | Autonomic co-regulation | MAP-Neuro appendices: primary ECG outcome | — |
| **Anti-phase HR synchrony** | Therapist alliance | Lucherini 2025 review | sig in 1 study |
| **In-phase HRV synchrony** | Therapist alliance | Lucherini 2025 review | sig in 1 study |
| **Persistence entropy of affective trajectories** | Opioid misuse vs. flexibility | Luo/Garland 2026: AUC=0.86 | AUC=0.86 |

**NEW signal**: **Persistence entropy** (via Takens embedding + persistent homology) on HRV-derived stress/craving trajectories. This is intrapersonal, not interpersonal, but directly relevant as a treatment outcome biomarker for the OUD population.

#### D. Skin Conductance — NOT in CADENCE

| Signal | What It Indexes | Key Evidence | Effect Size |
|--------|----------------|-------------|-------------|
| **SC concordance** (windowed cross-correlation of SC derivatives) | Empathic attunement | Marci 2007: r=0.47 with perceived empathy | r=0.47, 26% variance |
| **Therapist-led SC synchrony** | Positive patient emotional states | Lucherini 2025: contentment, vigor, calmness | multiple studies |

**This is the single most validated physiological synchrony signal in psychotherapy research** (9 of 11 interoceptive synchrony studies used EDA). CADENCE does not currently capture this. The MAP-Neuro appendices don't mention it either, but the hardware (wearable EDA sensors) is trivial.

#### E. Facial Expression/Blendshapes — Already in CADENCE (31 PCA channels)

| Signal | What It Indexes | Key Evidence | Effect Size |
|--------|----------------|-------------|-------------|
| **Facial mimicry event co-occurrence** | Emotional contagion, rapport | Koul 2023: Granger-causes neural synchrony | directional |
| **Smile composite synchrony** | Affiliation | CADENCE V5: ~42% rate, 2.9s lag, events every ~26s | from y_06 data |
| **Motion Energy Analysis (MEA) synchrony** | Therapeutic alliance, dropout prediction | Mende 2021, Lucherini 2025: 15 studies | consistent |

**NEW signal**: **Event Synchronization** (Quian Quiroga 2002) is a parameter-free method specifically designed for the kind of discrete facial expression events CADENCE V5 uses. It gives direction (q metric) and time-resolved coupling with adaptive windows. Direct drop-in for V5.

#### F. Body Pose — Already in CADENCE (41 channels)

| Signal | What It Indexes | Key Evidence |
|--------|----------------|-------------|
| **Postural alignment** | Rapport, coordination | MAP-Neuro: primary pose outcome |
| **Leader-follower movement dynamics** | Power dynamics, engagement | Mende 2021: followers → higher beta; patient-led → dropout |
| **Body movement MEA** | Therapist alliance | Lucherini 2025: most studied behavioral signal (15 studies) |

**Key insight**: The leader-follower asymmetry in body movement is directly what CADENCE's directed pathways (P1→P2 vs P2→P1) capture. If the patient leads in the first 3 sessions, there's a higher chance of early dropout (Mende 2021).

#### G. Respiratory — NOT in CADENCE

| Signal | What It Indexes | Key Evidence |
|--------|----------------|-------------|
| **Respiratory synchrony** | Alliance, enhanced empathy when combined with EEG NF | Konrad 2024: combined EEG+respiratory biofeedback outperforms EEG alone |
| **In-phase respiration** | Patient alliance, therapist progress | Lucherini 2025: 3 studies |

**Opportunity**: Respiratory rate can potentially be derived from ECG (Polar H10) or from pose (chest movement). Combining respiratory biofeedback with EEG neurofeedback enhanced both synchrony and subjective empathy (Jarvela 2019, 2021).

#### H. Speech/Voice — Mentioned in MAP-Neuro, NOT in CADENCE

| Signal | What It Indexes | Key Evidence |
|--------|----------------|-------------|
| **Vocal prosody synchrony** | Mixed — therapist-led pitch sync → *worse* alliance in one study | Lucherini 2025: 3 studies, mixed |
| **Semantic alignment (NLP)** | Therapeutic working alliance | Lin 2025 COMPASS: WAI from transcripts, 46% 4-class accuracy |
| **Shared voice affect categories** | Emotional concordance | MAP-Neuro appendices: planned measure |

**Note**: Speech synchrony results are inconsistent in the literature. The NLP-based approach (COMPASS) is more promising as a complementary modality to physiological synchrony for validating MAP-Neuro.

---

### II. Signal Processing Methods Relevant to CADENCE

| Method | What It Does | How It Relates to CADENCE | Priority |
|--------|-------------|--------------------------|----------|
| **Event Synchronization** (Quian Quiroga 2002) | Parameter-free directed event coupling | Direct fit for V5 blendshape mimicry events | **Very High** |
| **IAAFT surrogates** (Schreiber 2000) | Preserve spectrum+distribution, destroy nonlinear coupling | Already used for interbrain EEG; multivariate extension available | **Critical (already in use)** |
| **Robust BOCPD** (Altamirano 2023) | Online changepoint detection in dR2 timecourses | Could replace Kim HMM for temporal localization; O(1) per update, robust to outliers | **High** |
| **STOK Kalman** (Pascucci 2020) | Self-tuning time-varying MVAR connectivity | Adaptive tau (vs CADENCE's fixed 30s); frequency-resolved directed coupling | **High** |
| **MMHP Hawkes** (Wu 2022) | Latent active/inactive states for bursty event dynamics | Models mimicry clustering; principled alternative to binomial test | **High** |
| **IDTxl Transfer Entropy** (Wollstadt 2019) | Model-free directed information transfer | Validation metric for CADENCE; captures nonlinear coupling EWLS misses | **Moderate** |
| **Windowed Multiscale Synchrony** (Likens 2021) | Time-frequency synchrony heatmaps | Extends wavelet approach to all modalities (not just EEG) | **Moderate** |

---

### III. Key Theoretical Frameworks

1. **Theory of Flexible Multimodal Synchrony** (Gordon et al. 2025, Psych Review)
   - **Flexibility > aggregate synchrony**: Entropy of coupling timecourse, transitions in/out of synchrony, DFA scaling exponents predict outcomes better than mean coupling
   - CADENCE already measures this via per-timepoint significance — but should explicitly compute flexibility metrics (coupling entropy, state transition counts)
   - Psychedelic therapy = **metastable context** (high pulls to both synchronize AND segregate) → expect dynamic, fluctuating, multimodal coupling

2. **Mutual Prediction Framework** (Schilbach 2025)
   - INS = encoding own behavior + predicting partner's behavior
   - Directly supports CADENCE's regression approach (predicting one brain from another's signals)
   - Cross-brain GLM from animal studies (Kingsbury 2019) is conceptually identical to CADENCE

3. **Interoexteroceptive / Exteroproprioceptive Self** (Lucherini 2025)
   - Physiological synchrony (ECG, EDA) → moment-to-moment emotional attunement
   - Behavioral synchrony (face, body) → therapeutic alliance and relational engagement
   - Neural synchrony (EEG) → shared mental states
   - These are linked but serve different therapeutic functions

4. **Entropic Brain Hypothesis** (Mediano 2024, Godfrey 2025)
   - Psychedelics increase brain entropy → disrupts rigid negative patterns → enables reorganization
   - Eyes-closed maximizes this effect; external stimulation *competes* with endogenous drug activity
   - Implication: during neuroplastogen sessions, therapist's **voice** may be the primary interpersonal coupling channel

---

### IV. Issues and Opportunities Not Previously Considered

#### Critical Issues

1. **Drug-altered EEG baselines**: Psychedelics reduce alpha power and interbrain connectivity (Godfrey 2025). CADENCE's interbrain PLV/wPLI features will have fundamentally different distributions during neuroplastogen sessions vs. behavioral therapy sessions. **Need**: Drug-state-aware baselines or LZ complexity as a state moderator.

2. **Eyes-closed sessions**: During psilocybin/ketamine dosing, participants typically have eyes closed. This eliminates visually-mediated behavioral synchrony (gaze, much facial expression). CADENCE's blendshape and pose channels may carry less signal during acute drug phases. **Audio/prosodic and physiological channels become primary**.

3. **Missing EDA modality**: Skin conductance is the most validated physiological synchrony signal in therapy (Marci 2007, 9/11 studies), yet neither CADENCE nor MAP-Neuro currently capture it. Adding simple wrist-worn EDA would give access to the highest-effect-size physiological coupling signal.

4. **Neurofeedback mechanism is motivational, not calibrational**: Dikker 2019's finding that sham feedback works as well as real feedback means the feedback system need not be perfectly accurate in real-time — **the awareness of being monitored is the active ingredient**. This relaxes real-time accuracy requirements but raises questions about what "real" feedback adds beyond placebo.

5. **Epoch size consistency across modalities**: Ohayon & Gordon 2025 meta-analysis shows that matching temporal windows across modalities **nearly doubles** the effect size for cross-modal synchrony (0.42 vs 0.23). CADENCE's alignment module resamples to common timebase, but the coupling estimation window should also be consistent.

#### Novel Opportunities

1. **Event Synchronization for V5**: Quian Quiroga's method is a natural fit for CADENCE's event-driven blendshape architecture. The adaptive local coincidence window handles variable inter-event intervals (~26s), the q metric gives direction, and it's computationally trivial. Could combine: ES for event detection + lag estimation, then basis-expanded regression for continuous coupling.

2. **MMHP Hawkes Process for mimicry dynamics**: Model facial mimicry events as a Markov-modulated Hawkes process — mimicry episodes cluster (self-exciting) during active coupling, then become sporadic (Poisson) during non-coupled periods. This gives principled latent state recovery instead of ad-hoc thresholding.

3. **Robust BOCPD for temporal localization**: After 10 failed improvement attempts on temporal localization (V3.5 tracking doc), Altamirano's Dm-BOCPD offers a fresh approach: detect changepoints in dR2 timecourses online, with robustness to the outlier spikes that plagued the Kim HMM.

4. **LZ Complexity as drug-state moderator**: Like ECG HR/RMSSD currently moderate coupling in Stage 2, LZ complexity could moderate EEG coupling estimates — accounting for the drug-altered neural state during neuroplastogen sessions.

5. **Respiratory synchrony from existing hardware**: Respiratory rate may be derivable from ECG (respiratory sinus arrhythmia from Polar H10) or from pose (chest/shoulder movement). This adds a modality the literature says enhances neurofeedback effects when combined with EEG.

6. **Coupling flexibility metrics**: Gordon's theory suggests computing entropy of the dR2 timecourse, DFA scaling exponents, and coupling state transition counts as primary outcomes — not just mean dR2 or detection proportion.

7. **NLP-based alliance validation**: Lin 2025's COMPASS framework could run on session transcripts to provide independent validation that CADENCE's physiological synchrony metrics correlate with computationally-derived therapeutic alliance measures.

8. **Cross-brain GLM / STOK Kalman for frequency-resolved directed coupling**: CADENCE currently gets directionality from source→target regression but lacks frequency resolution in the coupling estimate. STOK's self-tuning adaptive Kalman could provide both, with the added benefit of adaptive temporal smoothing (vs. CADENCE's fixed 30s tau).

9. **Strangers vs. established dyads**: Ohayon meta-analysis finds strangers show *higher* neural-behavioral correlations than familiar dyads (0.40 vs 0.15). First therapy sessions may show different coupling patterns than established therapeutic relationships — CADENCE should track session number as a covariate.

10. **Facilitator selection via coupling profiles**: Goldy 2026 shows facilitators account for 13.6% of psychedelic session variance. CADENCE could generate per-facilitator coupling profiles that predict which facilitator-patient pairings produce the strongest synchrony — enabling data-driven facilitator matching.

---

### V. New Papers Added 2026-03-26 (~45 papers)

#### A. Inter-Brain Synchrony — Clinical & Methodological

| Paper | N | Key Finding | Effect Size | CADENCE Relevance |
|-------|---|-------------|-------------|-------------------|
| **Adel et al. NBR 2025** | 160 dyads (11 studies) | INS present in ALL clinical hyperscanning studies; attachment moderates INS-outcome link | Mixed (systematic review) | Validates premise; add attachment as moderator |
| **Zimmermann/Ayrolles 2024** | 18 dyads + sim | Epoch <5s inflates phase-based IBS; adjusted CCorr = PLV (r>.99); low SNR degrades phase | 36-49% inflation | Validates CADENCE volt_amp over phase methods |
| **Bevilacqua et al. 2019** | 12 students + 1 teacher | Social closeness predicts teacher-student synchrony; synchrony tracks engagement NOT outcomes | Closeness-TI r=.38 | Relationship quality moderates coupling magnitude |
| **Czeszumski et al. 2022** | 890 (13 fNIRS studies) | Cooperation IBS in PFC+TPJ; massive heterogeneity | g=1.98, I2=98.6% | Expect high between-session variability |
| **Hakim et al. 2023** | 215 studies | 27 IBC methods; cross-brain GLM endorsed; multimodal data essential | Review | CADENCE architecture (EWLS distributed lag) validated |
| **Mu et al. 2016** | 65 + 30 dyads | Oxytocin enhances alpha-band IBS during coordination; gender differences | F(1,28)=10.18, p<.005 | Gender composition as moderator; alpha band key |
| **Pan et al. 2020** | 24 dyads fNIRS | Scaffolding behaviors drive coupling bursts; not passive observation | eta2>.65, d=0.78 | Validates event-anchored architecture |
| **Sened et al. 2025** | 8 patients, 1 therapist | IBS increases across therapy sessions (inter-brain plasticity); tracks symptoms not alliance; coupling is dyad-specific | d=1.34 | Track session-over-session coupling trajectories |

#### B. Neurofeedback & Intervention — Psychedelic Context

| Paper | Key Finding | CADENCE Relevance |
|-------|-------------|-------------------|
| **Carhart-Harris & Friston 2019 (REBUS)** | Psychedelics = simulated annealing; relax precision of high-level priors via 5-HT2A; window of plasticity | Foundational theory for drug-state coupling |
| **Levin et al. 2024** (N=24, psilocybin MDD) | Post-psilocybin alliance predicts depression r=-.85 (4wk), r=-.77 (6mo). Pre-dosing alliance predicts mystical experience r=.49. Task subscale dominates. | CADENCE coupling in prep sessions should predict acute experience; alliance 4x more important than conventional therapy |
| **Kleinbub et al. 2020** | Proposes "interpersonal biofeedback" — real-time physiological coupling feedback to therapists via haptic signals. SC synchrony tracks attachment processes. | CADENCE builds exactly this measurement backbone |
| **Muscat et al. 2021** | Ketamine: 5 mechanisms (NMDA, neuroplasticity, anti-inflammatory, high-entropy, psychedelic). Spinogenesis peaks 24-48h. | Integration sessions in plasticity window; track pre/post resting-state EEG |
| **Schartner et al. 2017** (N=54, psilocybin+ketamine+LSD) | All 3 psychedelics increase LZ complexity above waking consciousness (86-100% of participants). Temporal > spatial diversity. Phase coherence does NOT index drug state. | LZ complexity per-channel per-epoch (2s sufficient); phase metrics won't capture drug effects |
| **Shamay-Tsoory & Abu-Akel 2016** | Oxytocin social salience hypothesis: context-dependent effects via dopamine modulation. Can increase envy, aggression in threatening contexts. | Interpret coupling in context (cooperative vs rupture) |
| **Timmermann et al. 2019** (N=13, DMT) | Alpha suppression + theta emergence (9.3->7.4 Hz) + LZs increase. Oscillatory component shows functionally relevant changes; fractal less relevant. Minute-by-minute resolution sufficient. | Track alpha/theta/LZs in real-time; oscillatory > fractal; CADENCE's volt_amp focus validated |
| **De Jong et al. 2025** | Multi-dimensional progress feedback (symptoms + process measures) outperforms simple monitoring. NOT cases benefit most. Up to 50% of clinicians don't use feedback. | CADENCE must be designed for minimal cognitive load |

#### C. Signal Processing Methods — New

| Paper | Method | Key Technical Detail | CADENCE Application |
|-------|--------|---------------------|---------------------|
| **Dong et al. 2021** | FMRR (ECG-derived respiration) | <2 bpm error at ALL sampling rates down to 50 Hz; needs only RR intervals; 32s min windows; band-pass 0.15-0.4 Hz | Trivially add respiratory synchrony from Polar H10 |
| **Varon et al. 2020** | QRS slope EDR | Downslope/slope-range methods best for full waveform; validated on 59,482 segments, 156 subjects | Better respiratory waveform if needed beyond rate |
| **Charlton et al. 2016** | 314 respiratory algorithms | ECG-based outperforms impedance pneumography; public RRest toolbox | Reference pipeline for respiratory extraction |
| **Halpin & De Boeck 2013** | Dyadic Response Hawkes model | EM algorithm with gamma kernels; classifies events as spontaneous/self-response/other-response; 14 parameters | Proper framework for BL event co-occurrence |
| **Kreuz et al. 2013** | SPIKE-distance | Parameter-free, time-resolved, causal variant for point-process synchrony | Alternative to cross-correlation for BL events |
| **Li et al. 2018** | Synchronization templates | Coupled HMMs + Beta-Bernoulli discover role-specific facial coordination patterns; templates predict negotiation outcomes | Role-specific therapist/patient templates for BL |
| **Ayrolles/HyPyP 2021** | HyPyP toolbox | 12 inter-brain connectivity measures; pseudo-dyad permutation standard; envelope correlation = volt_amp concept | Confirms CADENCE design choices |

#### D. Clinical Synchrony Dynamics — Therapy-Specific

| Paper | N | Key Finding | Effect Size | CADENCE Relevance |
|-------|---|-------------|-------------|-------------------|
| **Schoenherr et al. 2019** | 267 SAD dyads | Session 3 movement synchrony predicts dropout. Patient-led sync strongest. | 1% sync -> 5% dropout reduction | Early-session dropout risk indicator |
| **Schoenherr et al. 2021** | 64 SAD patients | Vocal f0 synchrony predicts WORSE outcomes. Different modalities = different regulatory competencies. | beta=.24-.37 | Confirms vocal sync is negative; keep modalities separate |
| **Uhl et al. 2025** | 90 clients, 22 therapists | Movement and EDA sync NEGATIVELY correlated (r=-.34). High move + low EDA = best outcomes. Within-therapist variability predicts, not between. | b=-3.57, p=.048 | Critical: don't average modalities; track relative sync |
| **Stuldreher et al. 2020** | ~52 participants | EDA detects emotional events (AUC=.658), EEG detects attentional (AUC=.642), HR moderate (.592). Multimodal average most robust. | AUC .58-.66 | Validates multimodal approach; each modality serves different function |
| **Wynn/Heyn et al. 2023** | 45 mother-child dyads | SCR synchrony during vicarious extinction -> better recall. EDA sync = safety learning mechanism. | F=4.62, p=.032 | EDA synchrony positive in emotional contexts; context-dependent |
| **Ramseyer & Tschacher 2011** | 70 patients, 42 therapists | Movement synchrony predicts alliance + outcome. 15-minute segment sufficient (r=.70 reliability with full session). | r=.33-.35 | Foundational validation for automated movement synchrony |
| **Koole & Tschacher 2016** | Review | In-Sync model: movement sync -> alliance -> emotion regulation. Co-regulation = complementary responses, not just matching. | Theory | Look for both synchronous AND complementary patterns |
| **Swidrak 2025** | Review | Interpersonal synchrony as "reconnection" for chronic pain (fibromyalgia). Dance synchrony shows large effects. | Theory | Relevant if extending to chronic pain populations |

#### E. Cardiac & Respiratory Synchrony — New Evidence

| Paper | N | Key Finding | CADENCE Relevance |
|-------|---|-------------|-------------------|
| **Feldman et al. 2011** | 40 mother-infant | Vocal + affect sync drives cardiac sync; gaze alone does NOT. Near-zero lag. | Behavioral events anchor cardiac coupling |
| **Goldstein et al. 2017** | 22 couples | Touch amplifies cardio-respiratory coupling during pain (delta-R2=.18-.25); empathy moderates. Pain WITHOUT touch disrupts coupling. | Empathy is critical moderator |
| **Wilson/Kiecolt-Glaser 2018** | Review | HRV synchrony during marital CONFLICT predicts HIGHER inflammation. | PNS sync maladaptive in distress; interpret by context |
| **Sharika et al. 2024** | 204 in 44 groups | HR synchrony predicts group decisions >70% accuracy (Polar H10). MdRQA features outperform questionnaires. | Validates Polar H10; MdRQA captures nonlinear dynamics |
| **Codrons et al. 2014** | 60 participants | Respiratory sync spontaneous and INDEPENDENT of cardiac. Cardiac sync absent in emotionally neutral context. | Respiratory = "first responder"; add to CADENCE |
| **Mueller & Lindenberger 2011** | 12 singers | Respiratory sync >> cardiac in choir. Granger causality reveals director->singer directionality. Graph analysis partitions voice groups. | Respiratory more robust; directed coupling measurable |
| **Rogers et al. 2022** | 21 participants | Polar H10 respiratory rate: r=0.85, ~1 bpm error. Better at rest than exercise. | Hardware validation: no new equipment needed |

#### F. Complexity, Language & Theoretical

| Paper | Key Finding | CADENCE Relevance |
|-------|-------------|-------------------|
| **Chidichimo et al. 2025** (Nature Rev Neurosci) | Information theory (MI, TE, PID) as unified framework for interpersonal coordination. Nonlinear transformations common between brains. | Future upgrade: TE generalizes CADENCE regression to nonlinear |
| **Lotter et al. 2023** (meta-analysis) | rTPJ is robust INS hub; GABA/E/I balance is neurochemical basis. INS linked to ventral attention + DMN. | Prioritize temporal-parietal EEG channels; E/I balance altered by psychedelics |
| **Alonso et al. 2010** | Drug (alprazolam) DECREASES linear coupling while INCREASING nonlinear coupling across scalp. | CADENCE cross-correlation (linear) may miss drug effects; add MI |
| **Marmelat & Delignieres 2012** | Complexity matching (DFA exponents) captures coordination invisible to cross-correlation. Strong anticipation, not local error correction. | Compute DFA exponents on coupling timecourses |
| **Kojovic/Koehler 2024** | MEA synchrony classifies autism (BAC=63.4%). Kurtosis/skewness > mean as features. | Use distributional statistics of coupling timecourses |
| **Lord et al. 2015** | Language style synchrony d=0.62 for empathy; 2.4x odds high empathy per 1 SD LSS increase. Beyond reflections. | Highest-value NLP feature if speech added |
| **Bayerl et al. 2022** | Lexical entrainment predicts WAI (r=.44-.52); participation equality predicts alliance. | Convergent evidence for language synchrony |
| **Kruyt et al. 2023** | 12 prosodic entrainment methods applied to same data DISAGREE. CRQA + windowed cross-correlation most robust. | If adding prosody, use multiple methods |
| **Xiao et al. 2012, 2015** | NLP empathy detection r=0.56 with expert ratings. Empathy perceived as salient events (thresholded features > cumulative). Automated pipeline scalable. | Event-based empathy detection aligns with CADENCE architecture |
| **Gullapalli et al. 2025** | TFT opioid misuse detection AUC=0.81. Behavioral >> physiological. 45s optimal window. | Same research group; behavioral signals most discriminative |
