# CADENCE V11 Chalk Talk Outline

**Duration:** ~20 minutes + discussion
**Format:** Whiteboard only
**Audience:** Computational neuroscientists, ML researchers (familiar with electrophysiology/behavior in animals, not human clinical EEG)
**Goal:** Get feedback on model choice, statistical framework, and overall approach

---

## Minute 0–3: The Clinical Problem (WHY)

**Draw on board:** Therapist ←→ Patient, with arrows labeled "attunement"

> "We're part of MAP-Neuro, an ARPA-H project testing whether therapist-patient synchrony *mediates* clinical outcomes in neuroplastogen-assisted psychotherapy — ketamine for opioid use disorder, psilocybin for depression. The idea is that neuroplastogens open a psychoplastic window where the patient is unusually receptive, and what happens *between* therapist and patient during that window may drive durable clinical change.
>
> The problem: nobody has a principled way to quantify moment-to-moment coupling across modalities in a dyad. We have EEG, heart rate, facial expression, body movement — all at different timescales and rates. We need a framework that can tell us: *what kind of coupling is happening right now, and what drives transitions between coupling regimes?*
>
> Eventually this feeds a real-time neurofeedback dashboard for the therapist. But first we need the offline analysis pipeline to be right."

**Key point to land:** This isn't exploratory. It feeds mediation analyses and eventually an FDA-relevant feedback system. The modeling choices matter.

---

## Minute 3–8: The Data and Feature Scaffold (WHAT)

**Draw on board:** A vertical stack of 5 boxes (modalities), each with an arrow pointing right to "2 Hz scaffold"

```
EEG (256 Hz, 14ch × 2 people) ──→ ┐
ECG (130 Hz, 1ch × 2)         ──→ │
Face (30 Hz, 52 AUs × 2)      ──→ ├──→ 28D @ 2 Hz
Pose (30 Hz, 33 joints × 2)   ──→ │
Markers (conditions)           ──→ ┘
```

> "Each session is ~60 minutes, 6 conditions: eyes-open rest, eyes-closed rest, two conversations, and either meditation or psychoeducation blocks. We record five modalities simultaneously from both participants."

**Then sketch the 28D decomposition as a simple table on the board:**

> "Everything gets reduced to 28 coupling timecourses at 2 Hz. The key design question is *what goes in*. We group them by what they measure:"

| What | Channels | How |
|------|----------|-----|
| "Are their brains phase-locked?" | ImCoh θ/α/β (3) | Imaginary coherence — rejects volume conduction |
| "Are they in the same neural state?" | Concordance + Dynamics + Asymmetry (9) | Shared power, its volatility, who's higher |
| "Are their faces in sync?" | Expression + Activity (2) | Wavelet coherence on facial AUs |
| "Are their bodies in sync?" | ECG LF/HF + Resp + Pose (4) | Hilbert envelopes, respiratory phase, multi-lag velocity |
| "Shared neural complexity?" | LZ concordance + asymmetry (4) | Lempel-Ziv on Hilbert amplitude envelope |
| "Network structure?" | Graph modularity (1) | Louvain on windowed correlation graph of base 18D |
| "Predictive information flow?" | TE concordance (2) | Transfer entropy on burst grids |
| "Simultaneous bursting?" | Burst coincidence θ/α/β (3) | ±500ms co-occurrence, surrogate-calibrated |

**Don't belabor the feature extraction.** If someone asks "how exactly does the wavelet coherence work?" — say "CWT Morlet at 30 Hz, coherence in the expression band, z-scored against 200 circular-shift surrogates. Happy to go deeper offline." Move on.

**Key point to land:** Every channel is surrogate-calibrated. Nothing enters the model raw — everything is a z-score above a circular-shift null that preserves autocorrelation.

---

## Minute 8–10: Observation vs. Covariate Split (KEY DESIGN DECISION)

**Draw on board:** Two boxes — "Observations (28D)" and "Covariates (7D)" — with an arrow from covariates to "transition probabilities"

> "Not everything goes into the same place. We split the 28 channels from 7 transition covariates. The rule:"

**Write on board:**
- **Symmetric / magnitude → observations** (what state are we in?)
- **Signed / directional → covariates** (what state do we transition to?)

> "For example: transfer entropy concordance — the *total* bidirectional flow — goes into observations. Transfer entropy *asymmetry* — who leads — goes into covariates. We tested putting asymmetry in observations and the null state absorbed 48% of the data. The model couldn't distinguish 'balanced coupling' from 'no coupling' since both have asymmetry ≈ 0."

**The 7 covariates** (just list, don't explain deeply):
1. Slow behavioral drift (2 PCs)
2. Coupling flexibility (graph spectral — strongest covariate)
3. Algebraic connectivity (λ₂)
4. Graph topology change-point
5. TE asymmetry θ
6. TE asymmetry α

**Key point to land:** This split is principled but empirically validated — we tried the alternative and it failed in a specific, interpretable way.

---

## Minute 10–15: The Model (rSLDS) — WHERE FEEDBACK IS MOST WANTED

**Draw the graphical model on the board:**

```
z_{t-1} ──→ z_t ──→ z_{t+1}     (discrete states, K=4)
  ↓           ↓
 y_{t-1}     y_t               (28D observations)
              ↑
             u_t               (7D covariates modulate transitions)
```

> "We fit a recurrent Switching Linear Dynamical System. Four discrete states — one is a constrained null state with mean fixed at zero and capped variance. The other three discover themselves from data. Transitions between states are modulated by the 7 covariates via softmax logits."

**Then explain the hierarchical structure:**

> "We have ~12 sessions. Each session is a different therapist-patient pair. The challenge: different dyads have different baseline physiology, but we want shared coupling dynamics. So:"

**Write on board:**
- **Shared across sessions:** Transition matrix (W, S) — how states switch
- **Session-specific:** Emission means and variances — what each state looks like in this dyad

> "Fitting: per-session initialization, Hungarian alignment to solve label switching, then hierarchical EM alternating shared transitions and session-specific emissions. Constrained Viterbi with 10-second minimum dwell post-hoc."

**The four states that emerge:**
- **NULL:** No coupling (fixed at zero)
- **COUP:** High imaginary coherence (active phase-locked neural coupling)
- **SHARED:** High concordance (shared neural/behavioral state, not phase-locked)
- **OTHER:** Everything else

**Key point to land:** "This is where I most want your feedback. Is rSLDS the right model? Should we be considering alternatives? Is K=4 justified?"

---

## Minute 15–18: What We've Found So Far (EVIDENCE IT WORKS)

**Quick hits — don't draw, just state:**

1. **Coupling flexibility is the strongest transition covariate** (5× stronger than baseline drift) — confirms Gordon 2025 (flexibility > aggregate synchrony). Meditation rigidifies coupling, and that rigidity persists into post-meditation conversation.

2. **Asymmetry reversal validates ground truth:** Therapist leads during psychoeducation (they're teaching). Patient leads during conversation (they're sharing). Baselines are null. This replicates from two independent metrics (EEG power asymmetry AND transfer entropy).

3. **Semi-synthetic validation:** We can inject known coupling into pseudo-dyad data (P1 from session A, P2 from session B — guarantees null baseline). AUC scales monotonically with injection strength. Null integrity passes.

4. **Burst coincidence captures something orthogonal to ImCoh:** ImCoh peaks during conversation (interaction-driven). Burst coincidence peaks during eyes-closed rest (state-driven). They're measuring different coupling mechanisms.

5. **Important caveat — session-wide normalization:** Everything is z-scored to the session mean. The NULL state means "at session average," not "no coupling." Post-meditation conversation is NULL-dominated but has identical COUP usage to pre-meditation conversation and the most rigid (organized) coupling topology of any condition. Clinically, increased rapport in conv_2 is consistent with stable, session-average-level coupling — the model just can't distinguish that from genuine absence of coupling. This is an open question for the room.

**Key point to land:** "The model produces interpretable, replicable results that align with what we know clinically. But the session-wide normalization creates a blind spot for absolute coupling levels, and I want to know if there are statistical pitfalls I'm not seeing."

---

## Minute 18–20: Open Questions for the Room

**Write these on the board and leave them there for discussion:**

1. **Is rSLDS overkill or underkill?** Would a simpler HMM with diagonal emissions do just as well? Or do we need the continuous latent dynamics?

2. **K=4 states — how do we justify this?** We used BIC, but BIC with this many observation dimensions is noisy. Is there a better model selection approach?

3. **Prewhitening before fitting vs. letting the model handle AR:** We AR(1)-prewhiten each channel before it enters the model. Is it better to let the state-space model's own dynamics absorb the autocorrelation?

4. **Session-wide normalization blinds us to absolute coupling levels.** The NULL state means "at session average," not "no coupling." Post-meditation conversation registers as NULL even though clinical observation suggests increased rapport — because meditation may have raised the session average. Should we use condition-relative baselines, or give the model access to absolute levels alongside relative deviations?

5. **Hierarchical structure with n=12 sessions:** Is this enough to estimate shared transitions reliably? Are we overpooling or underpooling?

6. **Circular-shift surrogates everywhere:** We use the same null model for every metric. Is there a case where this is too conservative or too liberal?

> "These are genuine open questions. I'm not asking rhetorically — I want your opinions."

---

## DELIVERY NOTES

**What to practice:**
- Drawing the graphical model cleanly (z → z → z with y hanging below and u feeding into transitions). Practice this 3 times on paper.
- The obs/cov split explanation — this is your strongest "I thought carefully about this" moment.
- The null state inflation story (48% absorption) — this is concrete, memorable, and shows empirical rigor.

**What NOT to do:**
- Don't explain wavelet coherence, Morlet wavelets, or CWT unless asked.
- Don't explain Lempel-Ziv. If asked, say "it measures how compressible the amplitude envelope is — low LZ means repetitive, high means complex."
- Don't explain the burst detection criteria (monotonicity, consecutive cycles). Nobody will ask.
- Don't apologize for not understanding the model deeply. You understand the architecture and the design logic — that's what matters for this talk.

**If you get lost in a question:**
- "That's exactly the kind of thing I'm hoping to get feedback on. Let me write it down."
- "The computational details are in a design doc I can share — can we take that offline?"

**Your strongest card:** You have semi-synthetic validation with known ground truth. Most synchrony papers don't. If anyone questions whether the pipeline detects real coupling, you can point to dose-response AUC curves from injected pseudo-dyads.
