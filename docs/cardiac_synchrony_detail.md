## Cardiac Synchrony: Features, Timescales, Mechanisms

### The Core Dissociation (Mayo et al. 2021 meta-analysis)

| Branch | Feature | Association with Relationship | Effect Size |
|--------|---------|-------------------------------|-------------|
| **Sympathetic (SNS)** | SCL, SCR, PEP | **Positive** (arousal matching → engagement) | ES=0.19 (p=.02) |
| **Parasympathetic (PNS)** | RSA, HF-HRV, RMSSD | **Negative** (vagal coupling → less positive) | ES=-0.21 (p=.03) |
| **Combined SNS+PNS** | IBI, HR | Positive but weaker | ES=0.16 (p=.02) |
| **Performance outcomes** | All | Positive | ES=0.26 (p<.01) |

This is why CADENCE's ECG pathways have shown weak or null results — **RMSSD is parasympathetic, and parasympathetic synchrony is *negatively* associated with relationship quality**. Meanwhile HR is mixed SNS+PNS, diluting the signal.

### Feature-by-Feature Breakdown

| Feature | Timescale | Window | Lag | Branch | What It Captures |
|---------|-----------|--------|-----|--------|-----------------|
| **IBI (raw interbeat interval)** | Beat-to-beat, <1s | 3-8s | ±3s | Mixed | Broadest signal; sub-second coupling (Feldman 2011: mother-infant <1s) |
| **HR (bpm)** | Seconds to 10s of seconds | 15-30s | ±3-5s | Mixed | In-phase arousal matching |
| **RMSSD** | 10-60s epochs | 30-60s | ±3s | PNS (vagal) | Parasympathetic co-regulation |
| **RSA / HF-HRV** (0.15-0.40 Hz) | 2.5-6.7s per cycle | 30-120s epochs | ±3s | PNS (vagal) | Vagal tone, respiratory coupling |
| **LF-HRV** (0.04-0.15 Hz) | 6.7-25s per cycle | 120s+ minimum | N/A | Mixed/baroreceptor | Slow autonomic oscillations |
| **SCL/SCR** | 1-5s | 5-15s sliding | 0-5s | SNS | Arousal concordance (r=0.47 for empathy) |

### Coupling Mechanisms (ranked by evidence strength)

1. **Arousal matching / shared attention** — context-dependent HR synchrony; strongest in conflict/pain/novel tasks
2. **Respiratory entrainment** — drives RSA/HF-HRV synchrony when people are in proximity; causally upstream (Yi 2026)
3. **Autonomic co-regulation** — bidirectional ANS regulation in parent-infant and therapeutic dyads
4. **Emotional contagion / empathic resonance** — touch and empathy enhance coupling (Goldstein 2017: touch → HR coupling specifically under pain)
5. **Mechanical/ballistocardiographic entrainment** — physical proximity, within ~3% frequency deviation

### Why CADENCE's Current ECG Setup May Be Suboptimal

**Problem 1: RMSSD as moderation term is backwards**
CADENCE uses ECG HR/RMSSD as **moderators** of coupling. But RMSSD captures parasympathetic tone, and parasympathetic *synchrony* is negatively associated with relationship quality. Using RMSSD as a coupling moderator could be suppressing rather than enhancing detection.

**Problem 2: The 5s max lag is fine for IBI but wrong for HRV**
- IBI synchrony operates at <1s to 3s lags — well within CADENCE's 5s window
- But RMSSD requires 30-60s epochs to compute, so it changes on a 30-60s timescale
- RSA cycles are 2.5-6.7s — coupling at this timescale needs longer observation windows
- LF-HRV needs 120s+ of data — essentially inaccessible at CADENCE's current temporal resolution

**Problem 3: Mixed-branch IBI dilutes signal**
IBI contains both SNS and PNS contributions. The meta-analysis shows these have *opposite* relational valence. Decomposing into sympathetic (via SCR/EDA) vs parasympathetic (via RSA) would double the effective signal.

### What This Means for CADENCE

**Immediate changes:**
1. **Add EDA** — captures the sympathetic branch that has the *positive* association with outcomes (ES=0.19 for relationship, r=0.47 for empathy)
2. **Extract respiratory rate from Polar H10** (FMRR, <2 bpm error) — enables RSA computation and respiratory synchrony as a separate channel
3. **Decompose IBI into sympathetic (pre-ejection period proxy) and parasympathetic (RMSSD/RSA)** and treat them as separate coupling channels with potentially opposite interpretive valence

**Methodological recommendations from the literature:**
- IBI series must be ARIMA-preprocessed to remove autocorrelation before cross-correlation
- Smaller windows (3-8s for IBI, 15-30s for HR) discriminate real from spurious synchrony better
- In-phase HR vs anti-phase HRV carry different relational meaning
- Coupling is situational/dynamic — CADENCE's time-varying EWLS approach is correct

### Key Effect Sizes to Calibrate Expectations

| Signal | Context | Effect Size | Source |
|--------|---------|-------------|--------|
| SC concordance → empathy | Established therapy | r=0.47 | Marci 2007 |
| SCR concordance → symptom change | First session CBT | R²=0.43 | Gernert 2024 |
| HR synchrony → group decision accuracy | Naturalistic groups | >70% CV accuracy | Elkins 2024 |
| IBI synchrony → mother-infant coordination | Parent-infant | <1s lag coupling | Feldman 2011 |
| HRV synchrony → inflammation | Couple conflict | Predicts IL-6, TNF-α | Wilson 2018 |
| SNS synchrony → relationship | Meta-analysis | ES=0.19 | Mayo 2021 |
| PNS synchrony → relationship | Meta-analysis | ES=-0.21 (negative!) | Mayo 2021 |

The bottom line: **the sympathetic branch (EDA, arousal) carries the positive therapy signal; the parasympathetic branch (RMSSD, RSA) carries a paradoxically negative relationship signal**. CADENCE needs both, separately, not mixed in IBI. And EDA — which we don't currently capture — is where the largest effect sizes live.
