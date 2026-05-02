# Decision Log: Open Questions for Feedback

These are genuine design decisions where external input would change what we build. Each is framed as: what we chose, why, what we're unsure about, and the specific question for the room.

---

## 1. rSLDS vs. Simpler Alternatives

**What we chose:** Recurrent Switching Linear Dynamical System with K=4 states, D_latent=3 continuous dimensions, 28D observations, 7D transition covariates.

**Why:** We need discrete coupling regimes (not just continuous tracking) because the clinical question is "what state is the dyad in?" — therapist feedback needs categorical signals, not continuous ones. We need covariate-driven transitions because we want to know *what drives* state changes (flexibility? leadership asymmetry?). The rSLDS gives us both.

**What we're unsure about:** The continuous latent state x_t may not be doing much work. A grid search showed D_latent and n_factors are partially fungible — the model compensates when one is reduced. Scientific conclusions are robust to model capacity. This suggests the discrete states carry most of the information.

**Question for the room:** Given that the discrete states seem to do the heavy lifting, would we be better served by a covariate-driven HMM (IOHMM) with richer emission structure (e.g., mixture-of-Gaussians emissions) instead of the continuous latent dynamics? Or is there a principled way to test whether the latent dynamics contribute beyond what discrete states capture?

---

## 2. Prewhitening Strategy

**What we chose:** Conditional AR(1) prewhitening per channel before model fitting. Channels with lag-1 autocorrelation above a threshold are iteratively filtered (up to 3 rounds). Channels already near white noise are left alone.

**Why:** Without prewhitening, slowly-drifting channels (rho > 0.9) dominate the emission likelihood and the model tracks drift rather than coupling regimes. An early version with AR in the emission model caused state collapse (ARI=0.001) — the AR coefficients absorbed all variance.

**What we're unsure about:** We're removing temporal structure that might be informative. The prewhitening threshold is somewhat arbitrary. And it creates a disconnect: we prewhiten the data, but the model's emissions assume iid Gaussian within each state — which is only approximately true even after prewhitening.

**Question for the room:** Is there a more principled approach? Possibilities we've considered: (a) AR emissions per state (failed empirically), (b) letting D_latent absorb the autocorrelation (partially works but increases capacity demands), (c) differencing instead of AR filtering. What would you recommend?

---

## 3. The Observation/Covariate Split

**What we chose:** 28D observations (symmetric/magnitude measures) + 7D transition covariates (signed/directional measures + slow drift + graph topology). The split follows the principle that observations characterize "what state are we in?" while covariates modulate "what state do we move to?"

**Why:** Empirically validated — placing TE asymmetry in observations caused NULL state inflation to 48% (up from ~20%). The model couldn't distinguish "balanced bidirectional coupling" from "no coupling" since both have asymmetry near zero.

**What we're unsure about:** The EEG asymmetry channels (asym_theta/alpha/beta) are directional but remain in observations because they've worked well since V8.2. Are they an exception to the rule, or would moving them to covariates improve the model? We haven't tested this yet.

**Question for the room:** Is there a formal criterion for this split? Something like mutual information with state labels vs. transition labels? Or is the current empirical approach (try both, see which inflates NULL) sufficient?

---

## 4. Hierarchical Pooling with Small N

**What we chose:** Shared transition parameters (W, S) across all sessions, session-specific emission parameters (μ, σ²), with emission shrinkage toward group mean (c_shrinkage=0.2). Fitted via alternating EM with staged parameter unfreezing.

**Why:** Different dyads have different baseline physiology (session-specific emissions needed), but we hypothesize that the *dynamics* of coupling — how states transition — are universal (shared transitions). This also provides enough data to estimate the 7D covariate effects.

**What we're unsure about:** With n=12 sessions, the shared S matrix (4×4×7 = 112 parameters, L2-regularized) may be overfitting despite regularization. The emission shrinkage factor (0.2) was chosen by feel, not by cross-validation. And we haven't tested whether the "shared transitions" assumption is even correct — maybe meditation-protocol dyads transition differently from psychoeducation-protocol dyads.

**Question for the room:** (a) Should we consider protocol-specific transition matrices (meditation vs. PE) even at n=6/5? (b) Is there a principled way to set the shrinkage factor? (c) With 60 sessions coming in MAP-Neuro, should we plan a different hierarchical architecture now (e.g., nested random effects: site → protocol → session)?

---

## 5. Surrogate Framework Assumptions

**What we chose:** Circular-shift surrogates (200 per metric) as the universal null model. Each metric is expressed as a z-score above this null. The shift preserves each participant's autocorrelation and marginal distribution while destroying inter-participant temporal alignment.

**Why:** It's the standard hyperscanning null. It's conservative (preserves more structure than phase-randomization or block-shuffle). It's computationally efficient (single FFT bandpass → shift → recompute). And it's universal — the same null logic applies to coherence, burst coincidence, wavelet coherence, transfer entropy, and pose coupling.

**What we're unsure about:** Two specific concerns: (a) Shared environmental input (both participants hear the same therapist instruction) could create coupling that circular shift incorrectly attributes to interpersonal synchrony. (b) For very slow signals (graph modularity on 90-second windows), a circular shift of 10% of the session may not displace enough to destroy coupling structure — the null might be too easy to beat.

**Question for the room:** Should we use different null models for different timescales? Phase-randomization (IAAFT) for fast signals, block-permutation for slow ones? Or is the consistency of a single null framework more valuable than per-metric optimality?

---

## 6. Session-Wide Normalization Masks Absolute Coupling Levels

**What we chose:** All 28 observation channels are z-scored to session-wide mean=0, std=1 before entering the rSLDS. The NULL state has its mean fixed at zero, meaning it represents "coupling at the session-wide average level."

**Why:** Session-wide standardization equalizes scale across channels and dyads, makes the NULL state's zero-mean constraint meaningful, and ensures the model fits deviations from baseline rather than absolute levels that vary with hardware, electrode impedance, and participant physiology.

**What we're unsure about:** This creates a blind spot. If an intervention (meditation) raises the coupling floor for the rest of the session, post-intervention conversation at the new elevated baseline registers as NULL — the model can't distinguish "genuinely no coupling" from "elevated but stable coupling that matches the session mean."

Concretely: conv_2 post-meditation is 50% NULL, which initially seems to contradict clinical observation of increased rapport. But COUP usage is identical between conv_1 and conv_2 (~25%), NULL replaced OTHER (unstructured activity, not coupling), and flexibility is the lowest of any condition (0.18 = maximally organized topology). The "NULL-dominated" conv_2 is actually a settled, organized interaction state — not an absent one.

**Question for the room:**
- (a) Should we z-score relative to baseline conditions only (base_EO, base_EC) rather than session-wide? This would preserve condition-level mean shifts, but baselines are short (~2-3 min) and noisy.
- (b) Should we include the pre-standardized session-segment means as additional covariates, giving the model access to absolute levels?
- (c) Should we use a two-stage approach — session-wide normalization for the rSLDS, then a separate analysis on condition-level raw means to capture absolute shifts?
- (d) Is there a Bayesian approach that naturally handles both relative dynamics and absolute levels?

---

## 7. Facial Coupling Metric Captures Shared Stillness, Not Shared Expression

**What we chose:** CWT Morlet wavelet coherence in the expression band (0.5-2 Hz) across the 10 affect AUs, surrogate z-scored against 200 circular shifts.

**Why:** Wavelet coherence captures time-varying synchrony at the timescale of expression transitions. It's continuous, well-suited to the CWT framework, and validated semi-synthetically (AUC=0.78 at kappa=0.4 for injected smile mimicry).

**What we found:** Facial coupling excess is statistically *higher* during eyes-open baseline than during conversation (p=0.001, FDR-surviving, every session same direction). Inspection of the peak coupling timepoints reveals the reason: during baseline, both participants sit quietly with minimal expression, and their faces co-vary trivially from shared micro-movements (breathing, blinking, environmental responses). During conversation, faces are asynchronous by design — one person speaks (mouth/jaw active) while the other listens (face neutral), alternating in turns. This produces low coherence even when rapport is high.

**The fundamental problem:** The metric conflates "both faces doing nothing synchronously" with "both faces expressing the same emotion." Clinically meaningful facial coupling — shared smiles, empathic mirroring, coordinated expression transitions — is event-like and specific to certain AUs, not a broadband frequency-domain signal.

**Candidate solutions:**

**(a) Differential coherence** — Run the CWT on the temporal *derivative* of the AU timeseries. A still face has derivative ≈ 0, contributing no signal. Only expression *transitions* produce power. Coherence on derivatives captures co-transition rather than co-state. Eliminates shared stillness at the signal level. Minimal code change (one `np.diff` before CWT). But derivatives amplify tracker noise.

**(b) Product-weighted coherence** — Weight coherence at each timepoint by `sqrt(power_P1 × power_P2)`. When both are quiet, weight → 0. When both are expressively active, weight is large. One line change to the existing pipeline. But still treats any shared activity as coupling, even unrelated simultaneous expressions.

**(c) Amplitude-gated coherence** — Only compute coherence on timepoints where at least one participant has above-threshold expression-band power. Same approach as the TE burst-rate gating. Simple but throws away temporal coverage.

**(d) AU-state concordance** — Classify each participant's face into discrete states (neutral, positive, negative, speech, mixed) at each timepoint using the affect AU clusters, then compute state concordance excluding the neutral-neutral cell. Most clinically interpretable — directly answers "are they in the same emotional state?" Independent of the frequency-domain framework.

**(e) Expression-event coincidence** — Detect discrete expression events per participant (smile onsets, brow raises) using the existing `detect_expression_events()`, count co-occurrences within ±2-3 seconds, surrogate z-scored. Interpretable ("shared smile rate per minute") but sparse and noisy. V6 used a version of this (saliency events) and it was abandoned for low sensitivity.

**Question for the room:**
- Is there a principled way to separate "coherent expression" from "coherent stillness" in the frequency domain, or is this fundamentally an event-detection problem?
- Should we go with (a) differential coherence as a drop-in replacement for bl_expr in the scaffold, and add (d) AU-state concordance as a complementary clinical metric?
- Are there approaches from animal behavior analysis (e.g., syllable detection in vocalization, grooming bout coincidence) that map onto this problem?

---

## 8. Scaling to 60 Sessions and Real-Time

**What we chose:** The current pipeline is offline: full-session scaffold computation, then hierarchical rSLDS fitting across all sessions. Computation is ~30 seconds per session for the scaffold, ~19 minutes for hierarchical fitting across 12 sessions.

**Why:** Offline is fine for the mechanistic discovery phase (Task 1 of MAP-Neuro). But Task 2 requires a real-time neurofeedback dashboard that gives the therapist moment-to-moment coupling state estimates during the session.

**What we're unsure about:** The rSLDS requires forward-backward inference, which is inherently non-causal (the backward pass uses future data). For real-time use, we'd need forward-only filtering, which gives noisier state estimates. And the hierarchical structure assumes we've already seen all sessions — a new session would need to use the pre-fitted shared parameters with online emission adaptation.

**Question for the room:** (a) How much do we lose going from forward-backward to forward-only filtering for state estimation? (b) Is there a better online adaptation strategy than "fix shared W/S, adapt session-specific μ/σ² on the fly"? (c) Should we be thinking about a fundamentally different model for real-time (e.g., particle filtering, online changepoint detection) and only use the rSLDS for offline analysis?
