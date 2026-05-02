# What They'll Ask: Anticipated Questions & Answers

Questions ordered from most likely to least likely. For each: the question as they'd phrase it, why they're asking, your honest answer, and the evidence.

---

## 1. "Why not just an HMM? Why do you need the switching dynamics?"

**Why they're asking:** rSLDS is complex. Occam's razor says use the simplest model that works. They want to know if the continuous latent state x_t is actually doing anything.

**Your answer:**
> "Honestly, that's an open question. In V11 we actually run with D_latent=3 and n_factors=2, but we did a grid search over model capacity. The BIC difference between a plain IOHMM (D_latent=0) and the full rSLDS is modest — most of the signal is in discrete state switching, not within-state dynamics. The latent axes help separate EEG phase coupling from body/autonomic from burst/LZ, but a diagonal HMM gets you 80% of the way there. I'd genuinely welcome an argument for going simpler."

**Evidence:** BIC grid search showed D_latent=2,3,4 and n_factors=0,1,2,3 all cluster within ~5% BIC. D_latent and n_factors are partially fungible. Scientific conclusions (state structure, condition effects, covariate rankings) are robust to model capacity choices.

---

## 2. "How do you choose K=4? Have you tried K=3 or K=5?"

**Why they're asking:** Number of states is the most consequential free parameter in any HMM-family model. They'll want to see principled selection, not just "it worked."

**Your answer:**
> "BIC over K=3,4,5,6. K=4 wins, but the gap to K=3 is larger than K=4 to K=5. The null state is constrained (mean=0, capped variance), so really we're choosing among 2, 3, or 4 *active* coupling states. With K=3, the COUP and SHARED states merge — you lose the distinction between phase-locked coupling and shared-state coupling, which are empirically different (ImCoh peaks during conversation, concordance peaks during rest). K=5 splits OTHER into two poorly-identified substates."

**Evidence:** K=4 is BIC-optimal. The three active states have distinct, interpretable emission profiles and condition-specific usage patterns.

---

## 3. "Why prewhiten before the model? Shouldn't the state-space dynamics handle autocorrelation?"

**Why they're asking:** This is a legitimate modeling concern. AR(1) prewhitening removes temporal structure that the model's latent dynamics *should* capture. If you prewhiten too aggressively, you're removing signal.

**Your answer:**
> "Fair concern. The issue is practical: without prewhitening, channels with rho=0.95 (like slowly-drifting pose coupling) dominate the likelihood, and the model spends its state budget tracking slow drift rather than coupling regime changes. The conditional prewhitening only fires when rho exceeds a threshold, and it's iterative — at most 3 rounds. Channels that are already close to white noise are untouched. But I agree this is a place where we might be leaving information on the table, and I'd welcome suggestions for a better approach."

**If they push:** "One alternative would be AR emissions within each state — but we tried AR on observations in an earlier version and it caused state collapse (ARI=0.001). The AR coefficients absorbed all the variance and left nothing for state switching. AR in the *latent* dynamics works, which is what the rSLDS's A[k] matrices do."

---

## 4. "Circular-shift surrogates everywhere — isn't that one-size-fits-all? When does it fail?"

**Why they're asking:** Sophisticated audience knows that different null hypotheses test different things. Circular shift preserves autocorrelation and marginal distribution but destroys phase alignment. They'll wonder if this is too conservative for some metrics and too liberal for others.

**Your answer:**
> "Circular shift is our universal null because it preserves exactly what we want to preserve — each participant's own temporal structure — while destroying what we want to test — inter-participant temporal alignment. For phase coupling metrics like ImCoh, this is the standard hyperscanning null. For burst coincidence, it tests whether co-occurrence exceeds what you'd expect from independent burst processes with the same rate and autocorrelation.
>
> Where it could be too liberal: if there's a shared external driver (e.g., both participants hear the same therapist instruction and both react). The circular shift would destroy this shared-input coupling and make it look like spurious synchrony. We partially address this by having the experimental conditions (conversation vs. meditation) as analysis strata rather than pooling.
>
> Where it could be too conservative: for very slow signals (LZ complexity with 4-second windows), a circular shift might not disrupt enough structure. We enforce a minimum 10% shift offset to help, but it's not perfect."

---

## 5. "12 sessions for hierarchical estimation — is that enough?"

**Why they're asking:** Hierarchical models need enough groups to estimate group-level parameters. N=12 is small.

**Your answer:**
> "It's on the edge. The shared parameters are the transition matrices W (4×4 = 16 free parameters) and S (4×4×7 = 112 parameters, L2-regularized). With 12 sessions averaging ~1200 timepoints each, we have ~14,400 total transitions to estimate from, which is adequate for W but tight for S. The L2 regularization on S is doing real work — without it, the covariate effects overfit.
>
> This is the pilot phase. The full MAP-Neuro protocol collects 60 dyads across two trial sites. The current 12 sessions are proof-of-concept from the MORE pilot. We're specifically looking for feedback on whether the hierarchical structure will scale gracefully to 60 sessions, or if we need architectural changes now."

---

## 6. "How do you handle the multiple comparisons problem with 28 channels?"

**Why they're asking:** 28 channels × 4 states × multiple conditions = lots of tests.

**Your answer:**
> "At the scaffold level, we don't do per-channel hypothesis testing — everything feeds into the rSLDS as a single joint observation. The model handles the dimensionality internally through its emission parameters. Where we do post-hoc testing (coupling excess, per-condition comparisons), we use Benjamini-Hochberg FDR correction. For the semi-synthetic validation, we use AUC (which doesn't require a threshold) rather than binary significance tests.
>
> The bigger concern is probably not multiple comparisons but multicollinearity — some of these 28 channels are correlated by construction (e.g., burst coincidence theta and alpha share burst grid extraction). The low-rank noise factors (n_factors=2) in the emission covariance are meant to capture this shared noise structure."

---

## 7. "What's the emission covariance structure? Are you assuming independence across channels?"

**Why they're asking:** Diagonal covariance is a strong assumption for 28 correlated channels.

**Your answer:**
> "Diagonal plus low-rank. Each state has a diagonal variance σ²[k] (28 independent variances) plus 2 shared noise factors that capture the dominant cross-channel correlations. We tried diagonal-only and full covariance — diagonal underfits the cross-channel structure, full covariance is wildly overparameterized at 28×28 per state with only ~300 effective observations per state. The low-rank compromise (2 factors = 56 extra parameters per state) is BIC-optimal."

---

## 8. "How sensitive are the results to the 2 Hz scaffold rate?"

**Why they're asking:** Temporal resolution is a design choice that constrains what the model can see.

**Your answer:**
> "2 Hz is a compromise. Most interpersonal coupling signals evolve on timescales of seconds — the fastest meaningful thing we measure is burst coincidence at ±500ms, which is exactly the Nyquist limit at 2 Hz. Going to 4 Hz would double the data and roughly double computation, but the additional timepoints would mostly be interpolated (ECG coupling, wavelet coherence, and graph modularity are all computed in windows >> 0.5s). Going to 1 Hz would lose burst coincidence resolution. We haven't formally tested sensitivity to scaffold rate — that could be worth doing."

---

## 9. "The null state has mean fixed at zero — what if the true baseline isn't zero?"

**Why they're asking:** The prewhitened, standardized data has mean=0 by construction, but within conditions the mean could shift.

**Your answer:**
> "This is actually a real limitation we've identified. The null state's zero mean is relative to session-wide standardization, so NULL means 'at the session average,' not 'no coupling.' This creates a concrete problem: post-meditation conversation clinically shows increased rapport, but registers as NULL-dominated (50%) because meditation may have raised the session-wide coupling average — conv_2's genuinely elevated coupling sits right at the new mean.
>
> The evidence that this is a normalization artifact rather than real absence of coupling: COUP state usage is identical between conv_1 and conv_2 (~25%). NULL replaced OTHER (unstructured activity), not COUP (phase coupling). And coupling flexibility is the lowest of any condition (0.18) — the coupling *structure* is maximally organized.
>
> This is an open question for us: should we use condition-relative baselines, separate pre/post normalization windows, or include absolute coupling levels as additional channels alongside the relative deviations?"

**If they push:** "One approach would be to include the raw (pre-standardized) session mean as an additional covariate or as a session-level random effect. Another would be to z-score relative to the baseline conditions only (base_EO, base_EC) rather than session-wide. Both have trade-offs we haven't explored yet."

---

## 10. "Isn't your transfer entropy confounded with burst rate?"

**Why they're asking:** Anyone who works with point processes will immediately wonder whether differences in event rate drive the TE results rather than genuine coupling differences.

**Your answer:**
> "Partially, yes — and we've quantified it. TE episode fraction correlates with alpha burst rate at rho=0.36 (p=0.001). Conditions with more alpha bursts produce more detectable TE episodes. But it's not purely rate-driven: meditation has MORE alpha than conversation but LESS TE, consistent with autonomous oscillations without directed coupling. PE has the LEAST alpha but more TE than meditation, consistent with active therapist-led teaching.
>
> The baseline comparisons are the most suspect — baseline has both the highest alpha and highest TE, so we can't cleanly separate the two. We're considering a rate-conditional TE analysis that restricts estimation to timepoints where both participants have sufficient burst rate, which would remove the power confound at the cost of temporal coverage."

---

## 11. "Your facial coupling just measures shared stillness — how do you fix that?"

**Why they're asking:** If they look at the condition statistics, they'll see facial excess is higher at baseline than conversation. That's suspicious.

**Your answer:**
> "You're right, and we've confirmed it. Peak facial coupling timepoints show both faces quiet — co-variation from micro-movements, not shared expression. During conversation, turn-taking produces asynchronous faces even when rapport is high.
>
> We're evaluating five approaches. The most promising: differential coherence — running the CWT on AU time-derivatives instead of raw values. A still face has zero derivative, so it contributes nothing. Only expression transitions produce signal. Alternatively, AU-state concordance bypasses the frequency domain entirely: classify each face into a discrete emotional state and compute concordance excluding the neutral-neutral cell.
>
> This is actually a general problem in behavioral synchrony — how do you separate 'both doing the same thing' from 'both doing nothing.' If anyone here has solved this for grooming, vocalization, or social behavior in animals, I'd love to hear about it."

---

## 12. "Have you considered deep learning approaches — VAEs, neural ODEs, transformers?"

**Why they're asking:** ML researchers will instinctively compare to modern deep generative models.

**Your answer:**
> "Intentionally not. This project has two constraints that push us toward interpretable models. First, it feeds FDA-relevant mediation analyses — we need to be able to say 'state X is characterized by high phase coupling and low postural coupling,' not 'latent dimension 3 is high.' Second, with 12 sessions of ~20 minutes each, we have maybe 15,000 total timepoints. A transformer would memorize this dataset. The rSLDS gives us interpretable states, interpretable transition drivers, and a principled generative model we can do posterior inference on. If we had 1000 sessions, the calculus might change."
