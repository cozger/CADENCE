# Questions for Collaborators

Things I specifically want input on during the chalk talk. These are not rhetorical — I don't have answers and would genuinely benefit from your expertise.

---

## 1. The Shared Stillness Problem in Facial Coupling

Our facial coupling metric (CWT wavelet coherence in the 0.5-2 Hz expression band) is statistically *higher* during quiet baseline than during conversation. When we inspected the peak coupling moments, both faces are quiet and similar — shared micro-movements from breathing and blinking, not shared emotional expression.

During conversation, faces are asynchronous by design: one person speaks (jaw/mouth active) while the other listens (face neutral), then they switch. This produces low coherence despite high rapport.

**What I need:** A method that captures "both people are expressing the same emotion" while ignoring "both people are doing nothing." The distinction is between coherent expression and coherent stillness.

**Ideas I'm considering:**
- CWT on AU *derivatives* instead of raw values (still face → zero derivative → no signal)
- Classifying each face into emotional states and computing concordance, excluding neutral-neutral
- Expression-event coincidence (detect smile onsets, count co-occurrences)
- Amplitude-weighting the coherence by both participants' expression power

**Has anyone solved this for animal behavior?** The same problem exists for grooming synchrony, vocalization timing, or social approach — "both doing nothing" is trivially synchronous. How do you handle that?

---

## 2. Transfer Entropy Is Confounded by Burst Rate

Our TE (transfer entropy) directed coupling metric correlates with the underlying alpha oscillation rate (rho=0.36). Conditions with more alpha bursts show more TE episodes — partly because the estimator is more reliable with more events, not because coupling is stronger.

We confirmed this: gating TE by alpha power drops the correlation to rho=0.04, and our strongest TE finding (therapist→patient alpha, FDR p=0.0005) collapsed to p=0.85. It was almost entirely a detectability artifact.

**What I need:** How should we handle directed coupling estimation when the carrier signal varies across conditions? Rate-conditional analysis (restrict to high-burst windows) works but loses temporal coverage. Is there a better approach?

**Specific question:** Is there a TE estimator that's inherently normalized for event rate, or do we always need external gating?

---

## 3. Session-Wide Normalization Erases Absolute Coupling Levels

All 28 scaffold channels are z-scored to the session-wide mean. The rSLDS null state has mean fixed at zero. So "null state" means "at session average," not "no coupling."

This creates a real problem: post-meditation conversation shows increased rapport clinically, but the model assigns 50% null because meditation raised the session average — conv_2's elevated coupling sits at the new mean.

**What I need:** How to give the model access to both relative dynamics (deviations from baseline) AND absolute coupling levels. Options:
- Z-score relative to baseline conditions only (short, potentially noisy reference)
- Include raw condition-level means as additional covariates
- Two-stage: rSLDS on relative dynamics + separate condition-mean analysis
- Bayesian approach that handles both?

**Has anyone dealt with this in animal electrophysiology?** Anytime you z-score across a session that contains an intervention, the intervention shifts the reference frame.

---

## 4. Is rSLDS the Right Model, or Is It Overkill?

We use a recurrent Switching Linear Dynamical System with 4 states, 28D observations, 7D transition covariates, 3 latent dimensions, and 2 noise factors. A grid search shows the scientific conclusions are robust to model capacity — most configurations cluster within 5% BIC.

The continuous latent dynamics (D_latent=3) may not be doing much. The discrete state switching carries most of the signal.

**What I need:** Would a simpler model (covariate-driven HMM with richer emissions) give us the same answers more robustly? Or is there something the latent dynamics capture that we'd lose?

**Also:** With n=15 sessions and 28D observations, are we in a regime where the hierarchical EM is well-identified, or could we be overfitting shared parameters?

---

## 5. Surrogate Framework: One Null to Rule Them All?

We use circular-shift surrogates (200 per metric) as the universal null model. This tests whether inter-participant temporal alignment matters above and beyond individual signal structure.

**What I need:** Is circular shift the right null for every metric, or should we use different nulls for different timescales?
- Fast signals (burst coincidence, ImCoh): circular shift seems appropriate
- Slow signals (graph modularity on 90s windows): a shift of 10% of the session may not displace enough
- Continuous signals vs. event-based signals: different sensitivity profiles?

**Specific concern:** Shared environmental input (both participants hear the same instruction) creates coupling that circular shift would incorrectly attribute to interpersonal synchrony. How do others handle this?

---

## 6. Scaling to Real-Time Neurofeedback

The current pipeline is offline: full-session scaffold, then hierarchical rSLDS. But the MAP-Neuro project needs a real-time therapist-facing dashboard during psilocybin-assisted therapy.

**What I need:**
- How much do we lose going from forward-backward to forward-only filtering for state estimation?
- Is there a better online model (particle filter, online changepoint detection) for real-time use, while keeping rSLDS for offline analysis?
- How should we handle a brand-new session that the model has never seen — fix shared W/S from the hierarchical fit and adapt emissions online?

---

## What I'm NOT Asking For

- I don't need help with the EEG signal processing (ImCoh, burst detection, LZ complexity) — those are validated
- I don't need alternative state-space models suggested without understanding the constraint that outputs must be interpretable for FDA mediation analyses
- I don't need "have you tried deep learning" — with n=15 sessions and clinical interpretability requirements, that's not the right tool

---

## How You Can Help Most

The single most valuable thing would be: **point out a confound or assumption violation I haven't noticed.** We've already found and addressed two (TE rate confound, facial shared-stillness). There may be more. If something in the pipeline design seems fragile, biased, or under-tested, I want to hear it now — before we run 60 sessions through it.
