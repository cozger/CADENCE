# Clinical ↔ Computational Glossary

A translation table for real-time use during Q&A. Left column is how you think about it; right column is what your audience calls it.

---

## Core Model Concepts

| When you say... | They think... | Bridge phrase |
|-----------------|--------------|---------------|
| "Coupling state" / "coupling regime" | Emission distribution — a specific mean vector μ[k] and covariance that generates observations when the system is in state k | "Each coupling state has a characteristic emission profile across our 28 channels" |
| "The model discovers states" | Unsupervised clustering in emission space, segmented in time by the Markov transition structure | "The states are learned, not predefined — the model finds clusters in the 28D observation space that persist over time" |
| "Null state" | A constrained component in the mixture — zero mean, capped variance. Acts as a regularizer preventing a catch-all state | "State 0 is an explicit 'nothing happening' baseline with its mean pinned at zero" |
| "State switching" / "transition" | A change in the discrete latent variable z_t. Governed by the transition matrix and covariates | "The probability of switching from state j to state k is a softmax over learned logits plus covariate effects" |
| "Sticky transitions" | Diagonal boost in the transition matrix — a prior that favors self-transitions (persistence). Equivalent to a Dirichlet prior with elevated diagonal | "We add a pseudo-count to self-transitions so states don't flicker" |
| "Minimum dwell" | Constrained Viterbi post-processing — short segments are merged. Not part of the generative model, just decoding | "After decoding, we enforce 10-second minimum segments by merging short runs" |

## Hierarchical Structure

| When you say... | They think... | Bridge phrase |
|-----------------|--------------|---------------|
| "Shared transitions" | Group-level parameters in the hierarchical model — W and S are estimated from pooled data across sessions | "The transition dynamics are shared — all dyads switch states the same way, just with different emission profiles" |
| "Session-specific emissions" | Random effects on emission parameters — each session has its own μ[k] and σ²[k] | "Each dyad has different baseline physiology, so what 'high coupling' looks like differs per pair" |
| "Label switching" | The identifiability problem in mixture models — permuting state labels gives identical likelihood | "State 1 in session A might be state 3 in session B — we align with the Hungarian algorithm" |
| "Emission shrinkage" | Regularization of session-specific parameters toward the group mean (c_shrinkage=0.2). Partial pooling | "Session-specific emission means are pulled toward the cross-session average to prevent overfitting" |

## Feature Engineering

| When you say... | They think... | Bridge phrase |
|-----------------|--------------|---------------|
| "Imaginary coherence" | The imaginary part of the cross-spectral density, normalized. Rejects zero-lag (volume-conducted) coupling | "It only measures coupling with a non-zero phase lag — instantaneous artifacts can't produce it" |
| "Concordance" | The average of two z-scored signals: (z₁+z₂)/2. High when both are high or both are low | "Are both people's brains in the same power state right now?" |
| "Surrogate z-scoring" | Empirical null via circular shift → z = (real - null_mean) / null_std. A non-parametric test | "Every metric is expressed as standard deviations above a shuffled null that preserves autocorrelation" |
| "Circular-shift surrogates" | Time-domain resampling that preserves marginal distribution and autocorrelation but destroys cross-signal phase relationships | "We shift one person's data in time, preserving their signal structure, and recompute the coupling metric" |
| "Prewhitening" | AR(1) residual filtering: x'[t] = x[t] - ρ·x[t-1]. Removes first-order temporal dependence | "We subtract the predicted-from-previous-sample component so the model sees innovations, not drift" |
| "Coupling flexibility" | Spectral energy ratio on a graph Laplacian — fraction of signal energy in high-frequency graph modes. From Gordon 2025 | "How independently are the different coupling modalities behaving? High flexibility = modalities decoupled" |

## Statistical Framework

| When you say... | They think... | Bridge phrase |
|-----------------|--------------|---------------|
| "Semi-synthetic validation" | Injection of known signal into real noise backgrounds. Ground truth AUC evaluation | "We inject coupling of known strength into pseudo-dyad data where no real coupling exists, and ask if the pipeline detects it" |
| "Pseudo-dyad" | Surrogate pair: P1 from session A, P2 from session B. Guarantees no real coupling at κ=0 | "We pair people who were never in the same room — this gives us a true null baseline" |
| "Dose-response" | AUC as a function of injection strength κ. Should be monotonically increasing | "As we inject more coupling, detection should improve smoothly — and it does" |
| "Coupling excess" | Real metric minus surrogate mean, divided by surrogate std. Continuous z-score of how much coupling exceeds chance | "How many standard deviations above the random baseline is the real coupling?" |
| "Transfer entropy" | Information-theoretic measure: how much does knowing X's past reduce uncertainty about Y's future, beyond Y's own past? | "Does knowing what one brain did help predict what the other will do next?" |
| "Burst coincidence" | Temporal co-occurrence of discrete neural events within a narrow window (±500ms), above chance | "Did both brains burst at the same time, more often than random?" |

## Key Interpretation Pitfalls

| If you catch yourself saying... | Stop and say instead... |
|-------------------------------|----------------------|
| "conv_2 shows no coupling — it's all NULL" | "conv_2 shows coupling at the session-average level. NULL means 'at the mean,' not 'absent.' COUP is identical to conv_1. NULL replaced OTHER (unstructured activity), not coupling." |
| "Meditation coupling doesn't transfer into conversation" | "The absolute coupling level may be elevated, but session-wide normalization can't see that. The low flexibility (0.18) shows the coupling *structure* is maximally organized in conv_2." |
| "The model says nothing is happening" | "The model says nothing is deviating from the session baseline. That's different from nothing happening — especially if the intervention shifted the baseline." |

## Things They Might Say That You Should Recognize

| If they say... | They mean... | Your move |
|----------------|-------------|-----------|
| "What's the generative model?" | Write down the full probabilistic model: p(z,x,y|u) | Point to the graphical model on the board: z→z, z→y, u→transitions |
| "Is this identifiable?" | Can you uniquely recover parameters from data? (label switching, emission/transition tradeoffs) | "We handle label switching via Hungarian alignment, and use staged fitting to prevent emission-transition co-adaptation" |
| "What are your sufficient statistics?" | What summary of the data does the model actually use? | "Weighted means and covariances (from γ posteriors) for emissions, pairwise transition counts (ξ) for transitions" |
| "How do you initialize?" | Starting point for EM matters — local optima are a real concern | "Per-session fits with multiple random restarts, then Hungarian alignment, then pooled initialization for hierarchical EM" |
| "What's the mixing time?" | How long does it take for the Markov chain to forget its initial state? | "With sticky transitions (κ=3), states persist for 50-130 seconds on average. The chain doesn't 'mix' in the MCMC sense — it's fitted via EM, not sampled" |
| "Are you doing inference or learning?" | Inference = computing posteriors given parameters. Learning = estimating parameters from data. EM does both alternately | "EM: E-step is inference (forward-backward), M-step is learning (parameter updates)" |
| "What's your held-out performance?" | They want cross-validation or test-set evaluation | "We don't have held-out sessions yet — with n=12, we can't afford to hold out. The 60-session MAP-Neuro dataset will enable leave-one-session-out CV. For now, we have semi-synthetic validation as the closest substitute" |
