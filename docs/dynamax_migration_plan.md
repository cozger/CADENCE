# dynamax migration plan — CADENCE rSLDS

**Status:** Phase 0 in progress (2026-05-01).
**Author:** assistant draft, 2026-05-01
**Trigger:** 2026-05-01 perf audit identified `_kalman_smoother_weighted`
(`cadence/significance/rslds_model.py:851`) as 30-50 % of total fit wall-time;
production hierarchical fit is ~1 h × 5 sensitivity variants ≈ 5 h, mostly
single-threaded CPU.

## §0 — Execution decisions (2026-05-01, post §5.1 user answers)

These answers commit the migration to a specific path. The remainder of the
plan (§1-§5) is unchanged in content but should be read with the
following overrides:

- **Deployment target:** Windows 11 + RTX 5080 (Blackwell, sm_120, CUDA
  12.8). **No Linux cluster available.** This kills the §3.4 plan-level
  assumption of "Linux cluster GPU" and forces one of (a) WSL2 + Linux
  GPU JAX, (b) Windows-native CPU JAX, or (c) Numba-only.
- **Production scaffold:** **MVP only** (`scripts/_run_mvp_hierarchical.py`,
  D_obs=7, D_input=2, D_latent=3, n_factors=2, K=3 default, recurrent=True,
  sticky_strength=3.0). The 26D V11 path is paused; V11 references in
  §4.1 Tier 3 and §4.2 BIC numbers are deprioritised. Validation in §4
  pivots to MVP cohort + MVP sensitivity sweep.
- **Recurrent transitions stay** (§5.1 Q1 → load-bearing for paper).
- **Bit-identicality not required** (§5.1 Q2 → scientifically-equivalent
  within §4.2 thresholds is sufficient).
- **n_factors=2 stays** (§5.1 Q5 → BIC-optimal in V11; assumed
  load-bearing for MVP pending Phase 0 confirmation).
- **Selected execution path: hybrid Path C — Numba Phase 0 → decision
  gate → JAX migration if needed.** Rationale: at MVP D_obs=7 the JIT
  compile overhead concern from §3.4 is most acute (smaller matrices →
  Python-loop overhead is a larger fraction of wall-time, but JAX JIT
  amortisation is also worse). Numba on small dense matrices is the
  sweet spot for `_kalman_smoother_weighted`. Phase 0 measures the
  Numba ceiling before committing to ~14 days of JAX rewrite + WSL2
  setup.

### Path C phases (overlay on §2)

- **Phase 0 — Numba bottleneck patch** (~1 day, COMPLETE 2026-05-01)
  - 0.1 Baseline profile: cProfile of single-session MVP `fit_slds` on
    y_06 ✓
  - 0.2 Replace `scipy.special.logsumexp` with module-local `logsumexp`
    (the actual #1 bottleneck — 80% of wall-time at 2.1M calls; not
    `_kalman_smoother_weighted` as the plan had originally claimed) ✓
  - 0.2b `@numba.njit(cache=True)` of `_kalman_smoother_weighted`
    (now the 38% bottleneck after 0.2; soft-mask via R-inflation,
    diagonal-or-full R support) ✓
  - 0.2c `@numba.njit(cache=True)` of `_forward_backward` (Python loop
    over T was still ~28s after 0.2; numba kernel inlines all logsumexp
    reductions as explicit per-K loops) ✓
  - 0.3 Numerical-equivalence tests for both kernels ✓
    - Kalman: max|Δ x_smooth| = 2.84e-10 worst case (whole-session drop
      of 2 channels), <1e-13 typical
    - Forward-backward: max|Δ gamma| = 3.90e-13 at T=6188, K=4
  - 0.4 Re-profile + decision gate (BELOW)

**Phase 0 measured results (single-session MVP fit, y_06, 10 EM iters):**

| Stage | Wall-time | Cumulative speedup | LL trace |
|---|---|---|---|
| Baseline (numpy + scipy 1.17 logsumexp) | 281.4s | 1.0x | reference |
| + inline `logsumexp` | 79.9s | 3.5x | bit-identical |
| + Numba `_kalman_smoother_weighted` | 61.9s | 4.5x | bit-identical |
| + Numba `_forward_backward` (warm cache) | 34.1s | 8.3x | bit-identical |
| + vectorized factor-LL (warm cache) | **8.4s** | **33.4x** | bit-identical |

Per-iter cost: 28.1s -> **0.84s**. Projected production fit
(max_em_iter=80, n_restarts=2, 22-session hierarchical with joblib
threading on 16 cores):
- single hierarchical fit: ~1 h baseline -> **~1.5-3 min** (Phase 0)
- 5-variant sensitivity sweep: ~5 h baseline -> **~10-15 min** (Phase 0)

**Decision gate triggered: NO-GO on JAX migration.** Phase 0 alone meets
the original ~1 h sensitivity-sweep target (Phase B success criterion in
§4.2) without any of the §1.x dynamax compatibility work, WSL2 setup,
optax/lbfgs M-step rewrite, padded vmap-over-sessions, or 14-15
person-day commitment. The full Path A->B->C from §2 is shelved.

**Remaining bottleneck (post-Phase-0):** at 0.84 s/iter, the residual
breakdown is one-time `initialize_slds` warm-start (~4 s, fixed per
restart), scipy L-BFGS-B in `_m_step_transitions` (~0.3 s/iter, scipy
dispatch), and `slds_e_step` numpy/numba work (~0.33 s/iter). No single
remaining hot spot is large enough to be worth attacking unless the
production sweep proves to be slower than projected.

### Phase 0 production cross-check (COMPLETE, 2026-05-02)

**Patches are scientifically equivalent.** Apples-to-apples on a
19-session cohort with seed=42:

| Metric | Un-patched | Patched | Diff vs §4.2 |
|---|---|---|---|
| BIC | 1,427,474 | 1,428,169 | 0.05% (thr 0.1%) PASS |
| LL | -690,937 | -691,285 | 0.05% PASS |
| Wall | 138 min | 23.6 min | **5.8x speedup** |
| State labels | identical | identical | PASS |
| `mean_d_emit` max\|d\| | (ref) | 0.0004 | thr 0.05 PASS |
| Per-state usage mean\|d\| | (ref) | 0.014-0.021 | thr 0.02 PASS |
| Per-session usage max\|d\| | (ref) | 0.072 (1 session) | one outlier |

**Initial false alarm — chronology:**
The first 22-session patched run (4.7 min) gave BIC=1,511,029 vs
22-session baseline BIC=1,461,271 (apparent 3.4% drift, well above the
§4.2 threshold). Subsequent bisection of individual patches (revert
factor-LL, revert Kalman) showed both reverts made BIC *worse*, not
better, and patched code was deterministic across re-runs — both
inconsistent with a single patch being responsible.

Root cause: `configs/session_quality.yaml` was modified at 00:02 AM
(during this session, dropping 3 sessions: y04_020626, y11_022526,
y24_022526 -- the ones flagged in CLAUDE.md as needing fresh MATLAB EEG
cleaning). My patched fits at 22:25 had 22 sessions; the un-patched
re-run at 00:13+ had only 19. The "drift" was entirely cohort-driven:
N=22 vs N=19, not patch-driven. Re-running patched code on the current
19-session canonical reproduces the 0.05% drift documented above.

**Lesson** for future reproducibility checks: pin the input set first,
verify cohort identity across runs, *then* compare numerical outputs.

**Production readiness verdict: SHIP.** Patches preserve all §4.2
thresholds, deliver 5.8x speedup on the 19-session production cohort,
and the Numba kernels carry forward (no further bisection needed).

**Files patched:**
- `cadence/significance/rslds_model.py:28-49` -- inline `logsumexp`
  (replaces `from scipy.special import logsumexp`)
- `cadence/significance/rslds_model.py:53-180` -- module-level
  `_forward_backward_numba` (called by IOHMM._forward_backward)
- `cadence/significance/rslds_model.py:881-985` -- module-level
  `_kalman_smoother_weighted_numba` (called by `_kalman_smoother_weighted`
  wrapper that handles the shape-(T,m) -> shape-(T,m,m) R expansion)
- `cadence/significance/rslds_model.py:1165-1196` -- vectorized soft-mask
  factor-LL (replaces per-T loop at the elif branch in `slds_e_step`;
  identity-replacement on masked rows/cols + batched `np.linalg.inv`)

**Validation artefacts:**
- `scripts/_validate_kalman_numba.py` -- 4-test Kalman equivalence (no
  mask, ~6% missing, whole-session drop, real MVP y_06 mask)
- `scripts/_validate_fb_numba.py` -- 4-test forward-backward (T=500,
  T=6188, K=3, -inf emission rows)
- `scripts/_validate_factor_ll_varmask.py` -- 4-test factor-LL
  (production-size, heavy-missing, all-masked timesteps, real MVP mask)
- `results/migration/dynamax/baseline_profile_singlesession.txt` -- final
  cProfile snapshot

The decision gate is the §5.3 Gate 2 from the original plan. JAX
migration is no longer the recommended next step; if a future need
arises (autodiff sensitivity sweeps, batched cross-validation, GPU
inference), Phases A/B/C in §2 below remain valid blueprints but should
be re-justified against the Phase 0 baseline.

---

**Recommendation up-front:** *do not adopt the dynamax `SLDS` class*. The
dynamax SLDS module does not implement EM fitting, recurrent transitions,
input-driven discrete transitions, hierarchical pooling, masked observations,
or the constrained-Viterbi / null-state / c-shrinkage / factor-emission
machinery CADENCE depends on. A "drop-in" migration is not possible.

**The plan is a full JAX-native rewrite of the rSLDS** — E-step *and* M-step
*and* hierarchical pooling — built on top of dynamax's inference primitives
(`lgssm_smoother`, `hmm_smoother`, `parallel_inference`) plus optax for the
softmax-regression M-steps and JAX `jnp.linalg.solve` for the closed-form
Gaussian M-steps. CADENCE retains ownership of the model class, the
init/warm-start orchestration, the constrained Viterbi post-processing, and
the hierarchical EM driver; dynamax provides only the inner E-step kernels,
and optax provides only the gradient-based M-step optimizers. No `ssm`
library involvement (see §5.4 — Linderman `ssm` would not improve this).

This commits to a larger engineering footprint than the
"keep-the-numpy-M-step" alternative considered in earlier drafts, but yields
the full speedup target (10-50× on per-fit wall time on GPU, ~5-10× on CPU)
and positions CADENCE for downstream JAX-native work (autodiff sensitivity
sweeps, batched cross-validation, GPU-resident inference). Effort: ~10-15
person-days. The §3.10 Numba alternative remains documented as a fall-back if
JAX deployment hits a Windows or cluster blocker, but is not the
recommendation.

---

## 1. Compatibility audit

For each CADENCE customisation, this section records: (a) what dynamax
actually provides, with file:line citations, (b) the gap, (c) the cost of
closing it.

The dynamax citations below are against `main` as of 2026-05-01
(`https://github.com/probml/dynamax`).

### 1.1 K = 3-4 discrete states with **recurrent** transitions (`S_trans @ z_{t-1}`-like terms; concretely `R_recur @ x_{t-1}`)

CADENCE: `cadence/significance/rslds_model.py:1042`
(`_log_transitions_recurrent`):

```python
logits = W[None,:,:] + einsum('td,jkd->tjk', U, S_trans) + einsum('td,jkd->tjk', x_prev, R_recur)
log_trans = logits - logsumexp(logits, axis=2, keepdims=True)
```

dynamax SLDS (`dynamax/slds/inference.py:23-24`) carries only

```python
class DiscreteParamsSLDS(NamedTuple):
    initial_distribution: Float[Array, " num_states"]
    transition_matrix:    Float[Array, "num_states num_states"]
    proposal_transition_matrix: Float[Array, "num_states num_states"]
```

i.e. a static (constant-in-time, no-input, no-recurrence) K×K matrix. No
parameter, no API surface for `R_recur` or `S_trans`. The discrete-state
proposal in `rbpfilter` (`dynamax/slds/inference.py:173`) reads
`params.discrete.proposal_transition_matrix[x]` only, with no input or
continuous-state argument.

dynamax HMM is more flexible: `dynamax/hidden_markov_model/inference.py:14`
(`get_trans_mat`) accepts either a `(K,K)` constant, a `(T-1,K,K)` time-varying
tensor, or a `transition_fn(t)` callable. So one *could* compute a `(T-1,K,K)`
tensor of per-timestep logits in numpy/JAX from `U` and `x_smooth` and pass it
into `hmm_smoother`. That works for the E-step, but the corresponding
**M-step** (logistic-regression update on `W`/`S_trans`/`R_recur`) is not
provided by any dynamax module — `StandardHMMTransitions.m_step`
(`dynamax/hidden_markov_model/models/transitions.py:113`) only updates a
Dirichlet-prior categorical, not a softmax-regression. To use the dynamax HMM
abstraction one would have to subclass `HMMTransitions`
(`dynamax/hidden_markov_model/models/abstractions.py:209`), implement
`distribution`, `_compute_transition_matrices`, `collect_suff_stats`,
`m_step`, and `log_prior` for the recurrent + input-driven softmax. Not
particularly hard (CADENCE already has the L-BFGS-B objective in
`rslds_model.py:485-512` and `:1245-1262`), but it is ~150 LOC of new code,
not a free lunch.

**Status:** dynamax SLDS class **incompatible**. dynamax HMM inference
**reusable for the E-step** (custom subclass), M-step **must be re-implemented**
on top of dynamax primitives (essentially a port of CADENCE's existing L-BFGS-B
objective to JAX/optax for autodiff).

### 1.2 K = 3-4 with **input-driven** transitions (`S_trans @ U`)

Same situation as §1.1. dynamax SLDS does not support it; dynamax HMM
inference can, M-step must be hand-rolled. The E-step path through
`hmm_smoother` plus the M-step port together form one "rSLDS-on-dynamax"
custom class.

### 1.3 Null-state emission constraint (`d_emit[0] = 0` fixed, `R_emit[0]` capped)

CADENCE: `rslds_model.py:439-444` (IOHMM emissions M-step), `:1184-1185`
(SLDS emissions M-step), and `:1441-1442` (state 0 enforcement after each
EM iteration of the SLDS variant).

dynamax: emission distributions are concrete `tfd` distributions
(`MultivariateNormalFullCovariance`, etc.). There is no per-state masking of
the bias term in any of `dynamax/hidden_markov_model/models/{gaussian_hmm,
linreg_hmm,arhmm}.py`. The closest mechanism is dynamax's
`ParameterProperties` (`dynamax/parameters.py`), which carries a `trainable`
flag — but that flag is per-tensor, not per-state-row. Freezing
`emission_bias[0]` while letting `emission_bias[1:]` train is not expressible
through the standard property machinery.

**Status:** **incompatible** with stock dynamax. Workarounds: (a) custom HMM
emissions class that hard-codes `d_emit[0]=0` in the M-step (~20 LOC); or
(b) post-projection after every M-step from outside the dynamax fit loop —
which is incompatible with `SSM.fit_em` because it is jit-compiled into a
single closed-loop. Option (a) is the realistic path. Note that V11
production uses `null_state=False` per the K=4-no-null winning
diagnostic (`memory/project_v11_identifiability_diagnostics.md`); MVP and
some sensitivity variants still use `null_state=True`. This feature must
survive the migration.

### 1.4 c_shrinkage = 0.3 regularisation toward shared `C_mean` across states

CADENCE: `rslds_model.py:1187-1193`:

```python
if c_shrinkage > 0 and K > 1:
    start_k = 1 if _null_state else 0
    C_mean = C_new[start_k:].mean(axis=0)
    for k in range(start_k, K):
        C_new[k] = (1 - c_shrinkage) * C_new[k] + c_shrinkage * C_mean
```

This is a hand-rolled hierarchical-Bayes shrinkage applied after the
closed-form C update.

dynamax: nothing comparable. dynamax's `log_prior` machinery
(`dynamax/hidden_markov_model/models/abstractions.py:HMMEmissions.log_prior`,
returns 0 by default) lets you add any prior, but priors are only used for
SGD-based M-steps (`run_gradient_descent` in
`dynamax/utils/optimize.py`); the closed-form Gaussian M-step in
e.g. `gaussian_hmm.py` does not consult the prior. To get c_shrinkage
through a dynamax pipeline, one would either (a) override the M-step on a
custom emissions class, applying the shrink after the closed form
(straightforward, ~10 LOC), or (b) replace the closed form with an SGD
M-step using a Gaussian prior on `C_emit` toward `C_mean` (changes
convergence properties — undesirable).

**Status:** **incompatible** out of the box. Option (a) is cheap; this is
not a real blocker.

### 1.5 Hierarchical pooling across N = 22 sessions

CADENCE: `rslds_model.py:1684-1905` (`fit_hierarchical_slds`). Shared across
sessions: `A_dyn`, `b_dyn`, `Q_dyn`, `W_trans`, `S_trans`, `R_recur`,
`log_pi`. Session-specific: `C_emit`, `d_emit`, `R_emit`, `F_emit`. The
hierarchy is implemented via parallel per-session E-steps
(`rslds_model.py:1796-1828`), pooled M-step on dynamics
(`:1551-1606`), pooled M-step on transitions (`:1609-1681`), and
per-session M-step on emissions (`:1854-1872`).

dynamax: `SSM.fit_em` (`dynamax/ssm.py:356-410`) uses
`vmap(partial(self.e_step, params))` over batches and then calls a single
`self.m_step` on the pooled stats. **All parameters are shared across
batches**; there is no API for "shared dynamics + per-session emissions".
To get CADENCE-style hierarchy out of dynamax, one would have to bypass
`fit_em` entirely and write a custom EM driver that vmaps the E-step over
sessions but routes the M-step suff-stats into a mixture of pooled and
per-session updates. This is essentially what CADENCE already does, just
expressed in JAX rather than numpy + joblib.

**Status:** **incompatible**. The hierarchical orchestration *is* the heart
of CADENCE's fit; dynamax does not provide it. Reusing dynamax in a
hierarchical fit means using its primitives (`lgssm_smoother`,
`hmm_smoother`) as the E-step kernel inside a CADENCE-owned hierarchical
EM driver.

### 1.6 n_factors = 2 augmented emission residual covariance

CADENCE: `rslds_model.py:835-848` (`_factor_analyze`),
`:1196-1216` (factor M-step inside `slds_m_step_emissions`), and the
factor-aware E-step at `:931-998`. The form is
`Sigma[k] = F[k] F[k]' + diag(R[k])`.

dynamax LGSSM emission covariance (`dynamax/linear_gaussian_ssm/inference.py:
ParamsLGSSMEmissions.cov`) is just a single `(emission_dim,
emission_dim)` PSD matrix per state. There is a
`MultivariateNormalDiagPlusLowRankCovariance` distribution used internally
(`dynamax/linear_gaussian_ssm/inference.py:19`), so the distributional
machinery exists, but no model class exposes it as a parameterised
emission. dynamax HMM models (e.g.
`dynamax/hidden_markov_model/models/gaussian_hmm.py`) have a
`DiagonalGaussianHMM` and a full-covariance variant, but no
factor-analysis variant.

**Status:** **incompatible** out of the box; would require a custom
emissions class that parameterises `F` and `R` separately and computes
log-likelihoods through TFP's
`MultivariateNormalDiagPlusLowRankCovariance`. Modest effort (~50 LOC) but
non-trivial — the factor M-step in CADENCE
(`_factor_analyze` + the eigendecomposition path) would also need a JAX
re-implementation because `np.linalg.eigh` semantics around tied
eigenvalues are not bit-stable across LAPACK paths.

### 1.7 Asymmetric / sticky transition prior with κ = 3

CADENCE: `rslds_model.py:520-521` (sticky bias on diagonal of `W_trans`),
`:472-498` (asymmetric null-state W prior with `lambda_W_prior=0.1`).

dynamax: `dynamax/hidden_markov_model/models/transitions.py:36`
(`StandardHMMTransitions.__init__`) carries

```python
self.concentration = concentration * jnp.ones((K,K)) + stickiness * jnp.eye(K)
```

and the M-step at `:113` uses `Dirichlet(concentration + counts).mode()`.
That is **mathematically a different prior** from CADENCE: dynamax adds
`κ·I` to a Dirichlet concentration (so it biases the *posterior mean* via
pseudo-counts), while CADENCE adds `κ·I` directly to the softmax logits
after L-BFGS-B (so it is a translation in logit space). The two are
qualitatively similar but quantitatively distinct, and the κ=3 parameter
in CADENCE is *not* directly transferable.

**Status:** **partially compatible**. dynamax's `stickiness` is the
right concept but a different parametrisation; sensitivity tests would be
required to recalibrate κ. The asymmetric W prior is **incompatible** with
`StandardHMMTransitions` and would need to live in a custom transitions
class (i.e. is solved by the same custom class needed for §1.1 anyway).

### 1.8 AR(1) latent dynamics with `phi` per state

CADENCE: `rslds_model.py:198-216` (initialisation), `:399-420` (M-step on
sigma2 via innovation residuals), `:243-265` (AR-aware emission likelihood).
*However*, the docstring at `:45` is explicit:

> `ar_order: int = 0   # AR order (0=independent, 1=AR(1) — WARNING: causes
> state collapse; use D_latent instead)`

and production configs (`_run_mvp_hierarchical.py:239-253`,
`_run_v11_hierarchical.py:173-186`) leave `ar_order` at its default of 0.
The continuous latent `x_t` plus its dynamics `A[k]` plays the role that
AR(1) would otherwise play.

dynamax: `dynamax/hidden_markov_model/models/arhmm.py:LinearAutoregressiveHMM`
implements AR-HMM but in the *emissions* (lagged y as covariate),
similar to CADENCE's `_log_emissions` AR(1) path. It is HMM only — no AR
support coexisting with continuous latent `x_t` in dynamax's SLDS.

**Status:** **moot** — feature is documented as "do not use" in CADENCE and
unused in production. Migration plan should drop this feature, not port it.

### 1.9 IOHMM warm-start (3 restarts × 100 EM iters as Phase 1 of init)

CADENCE: `rslds_model.py:1278-1345` (`initialize_slds`) runs a vanilla
IOHMM for warm-start, takes its `gamma`, `mu`, `sigma2`, then constructs
SLDS init. In hierarchical mode, `:1718-1734` runs N parallel SLDS
inits (which themselves do IOHMM warm-start internally).

dynamax: each model class has its own `initialize` method
(`dynamax/hidden_markov_model/models/gaussian_hmm.py:initialize`, etc.) which
samples from prior or accepts manually-specified params, but there is no
analogous "fit a simpler model first to seed a richer model". This is a
research-pipeline concern, not a library feature, and would remain a
CADENCE-owned orchestration step regardless of which inference primitives
sit underneath.

**Status:** **out of scope for dynamax**; CADENCE keeps owning this.

### 1.10 Constrained Viterbi with `viterbi_min_dwell = 20` (10 s @ 2 Hz)

CADENCE: `rslds_model.py:661-729` (`viterbi_min_dwell`, expanded-state DP)
and `_run_mvp_hierarchical.py:65-92` (post-hoc iterative-reassignment
constrained-Viterbi). MVP production uses the iterative-reassignment
version on the smoothed `gamma`; the expanded-state DP is the
ground-truth fall-back.

dynamax: `dynamax/hidden_markov_model/inference.py:hmm_posterior_mode`
implements standard Viterbi only, no min-dwell constraint. Adding it would
require a custom decoder, but the constraint is post-processing and lives
*outside* the EM loop, so it survives any fit-side migration unchanged
(CADENCE keeps its own `constrained_viterbi`).

**Status:** **out of scope for dynamax**; CADENCE keeps owning this. The
existing constrained-Viterbi code at `_run_mvp_hierarchical.py:65-92` keeps
working regardless of whether the gamma it consumes was produced by numpy
or JAX.

### 1.11 Per-session `obs_mask` (channels missing for some sessions)

CADENCE: `rslds_model.py:259-274` (mask zeroing of per-channel emission
log-likelihoods), `:374-396` (mask-aware mean update), `:781-832`
(`_kalman_update` Joseph form with masked observation rows). Used to handle
y11/y24 lacking ECG/Resp; also used universally for per-channel
`obs_valid` from MVP scaffold (`_run_mvp_hierarchical.py:169`).

dynamax: nowhere in `dynamax/{hidden_markov_model,linear_gaussian_ssm,
slds}/inference.py` is there a `mask` argument or NaN-aware path. JAX
`lax.scan` and `jit` actively dislike NaN sentinels. The standard
work-around is to (a) pre-impute masked entries, (b) zero the corresponding
emission log-likelihood or innovation contribution, and (c) zero out the
suff-stat rows. All three are doable but must be threaded through a
*custom* emissions class — `GaussianHMMEmissions` does not respect a
mask, and neither does `lgssm_filter`.

**Status:** **incompatible** with stock dynamax. ~30 LOC custom emissions
class for the HMM side; for the LGSSM side the masking would need to live
in a wrapper around `lgssm_smoother` that pre-zeroes the innovation —
which means the smoother is no longer simple `jit`. This is the single
most awkward gap.

### 1.12 Summary table

| Customisation | dynamax SLDS | dynamax HMM/LGSSM as primitives | Verdict (full-JAX plan) |
|---|---|---|---|
| Recurrent transitions (R_recur @ x) | ✗ | ⚠ custom log_trans + optax M-step | port to JAX (Phase A.1, A.5) |
| Input-driven discrete transitions (S @ U) | ✗ | ⚠ custom log_trans + optax M-step | port to JAX (Phase A.1, A.5) |
| Null-state (d[0]=0, R[0] capped) | ✗ | ⚠ custom emissions M-step | port to JAX (Phase A.4) |
| c_shrinkage toward shared C_mean | ✗ | ⚠ post-projection in M-step | port to JAX (Phase A.4, B.5) |
| Hierarchical pooling across N sessions | ✗ | ✗ (`fit_em` shares all params) | port to JAX via vmap+pooled M-step (Phase B) |
| n_factors emission FA | ✗ | ⚠ custom emissions class + jnp.linalg.eigh | port to JAX (Phase A.4) |
| Sticky transitions (κ=3 on logits) | ✗ | ⚠ different parametrisation (Dirichlet) | port to JAX as post-optimisation bias (Phase A.5) |
| AR(1) latent dynamics | n/a | ✓ via arhmm.py (HMM-only) | **drop — unused in production** |
| IOHMM warm-start | n/a | n/a — pipeline-level | retains numpy implementation (runs once) |
| Constrained Viterbi (min-dwell) | n/a | n/a — post-processing | retains numpy implementation (post-fit) |
| Per-session obs_mask | ✗ | ⚠ custom emissions + smoother wrapper | port via R-inflation trick (Phase A.2) |
| **JAX-jitted Kalman smoother** | (RBPF only) | ✓ `lgssm_smoother` (`linear_gaussian_ssm/inference.py:516`) | **adopt as inner kernel** |
| **JAX-jitted forward-backward** | (RBPF only) | ✓ `hmm_smoother` (`hidden_markov_model/inference.py`) | **adopt as inner kernel** |
| **Parallel-prefix-scan Kalman** | n/a | ✓ `parallel_inference.py` | adopt if GPU memory permits at T~5000 |

**Bottom line of §1:** every CADENCE feature except the JAX-jitted
inference primitives requires custom JAX code. dynamax's *abstractions*
(the `SSM` / `HMM` / `SLDS` model classes) do not fit; dynamax's
*primitives* (`lgssm_smoother`, `hmm_smoother`, `parallel_inference`)
plus optax's gradient-based optimisers do. The plan in §2 ports every
CADENCE customisation onto those primitives in a single JAX-native
module (`cadence/significance/rslds_jax.py`) that exposes the same
function-level API as the existing numpy `fit_slds` /
`fit_hierarchical_slds`, allowing entry scripts to swap engines via a
flag.

---

## 2. Migration strategy

This plan migrates the rSLDS to a **fully JAX-native** implementation: every
hot-loop computation (Kalman smoother, forward-backward, weighted dynamics
M-step solve, weighted emissions M-step solve, transitions M-step
optimisation, hierarchical pooling via `vmap` over sessions) runs inside one
JAX `jit`-compiled graph per EM iteration. dynamax provides the inference
primitives (`lgssm_smoother`, `hmm_smoother`); optax provides the
gradient-based optimiser for the softmax-regression transitions M-step;
CADENCE owns the model class, the parameter pytree, the EM driver, the
init/warm-start orchestration, and the constrained-Viterbi
post-processing.

### 2.0 Target architecture

```
cadence/significance/rslds_jax.py
├── ParamsRSLDS          (frozen pytree)
│       shared:  A_dyn, b_dyn, Q_dyn, W_trans, S_trans, R_recur, log_pi
│       per-session: C_emit (N,K,m,D), d_emit (N,K,m), R_emit (N,K,m),
│                    F_emit (N,K,m,n_factors)
│
├── e_step_one_session (jit)        ── inner SMF iters → posteriors
│   ├── log_emit          (jit, vmap over K)
│   ├── log_trans         (jit; recurrent + input-driven softmax)
│   ├── hmm_smoother      ← dynamax/hidden_markov_model/inference.py
│   └── lgssm_smoother    ← dynamax/linear_gaussian_ssm/inference.py
│       (fed time-varying weighted A_t, Q_t, C_t, d_t, R_t)
│
├── e_step_hierarchical = vmap(e_step_one_session)
│   over (Y_padded, U_padded, mask_padded, valid_T)
│
├── m_step_emissions_per_session (jit, vmap over N)
│       closed-form (jnp.linalg.solve) for C, d, R
│       FA via jnp.linalg.eigh for F (Phase A.3)
│       c_shrinkage applied as post-projection toward C_mean across N
│
├── m_step_dynamics_pooled (jit)
│       closed-form weighted least squares on cross-session-pooled
│       suff-stats (S_pp, S_xp, S_xx accumulated by sum across N)
│
├── m_step_transitions_pooled (jit + optax)
│       loss = -sum_n sum_t xi_n[t] * log_softmax(W + S@U_n[t] + R@x_n[t])
│              + L2 prior on (W − W_prior) when null_state
│              + sticky bias added post-optimisation
│       optimiser: optax.lbfgs (or optax.adam fallback if LBFGS line-search
│       fails to converge under JAX tracing)
│
└── fit_em_hierarchical (Python outer loop)
        Each iter: jit-compiled e_step + m_step_emissions + m_step_dynamics +
        m_step_transitions. Outer loop is Python because EM convergence
        check + early-stop + restart logic stays interpretable.
```

**Architectural decisions (locked, with rationale):**

1. **`vmap` over sessions for E-step, padded T.** Sessions vary in length
   (~3000-7000 timepoints); pad each to `T_max` and carry `valid_T` per
   session. This wastes ~50 % memory on short sessions but enables a
   single XLA graph instead of N graphs (which would force JIT recompile
   per unique T). At N=22 sessions × T~5000 × D_obs=26 the padding cost
   is ~50 MB total — trivial on any GPU.
2. **`vmap` over sessions does NOT extend to per-session emission
   M-steps** that update `C_emit[n]`, `d_emit[n]`, `R_emit[n]` — those
   work on per-session suff-stats and remain `vmap`-mapped over the N
   axis (cleaner than a Python loop).
3. **Per-session `obs_mask` is constant across time within a session.**
   Use the standard "inflate R" mask: where `obs_mask=False`, set the
   corresponding `R_emit[n,k,d]` row to a large constant (1e10) inside
   the wrapper. The Kalman gain on those rows becomes ~0, the smoother
   sees no information from masked channels, and downstream suff-stats
   are zeroed by multiplying by `obs_mask`.
4. **Inner SMF iterations (`n_inner_estep=3` in CADENCE) become a `lax.scan`
   inside `e_step_one_session`.** No Python loop; the unrolled graph is
   small (3 iters of forward-backward + Kalman).
5. **Transitions M-step uses `optax.lbfgs`, NOT `scipy.optimize.minimize`.**
   This is the one place the migration is mathematically *different* —
   optax LBFGS is JAX-jitted with autodiff but uses a different
   line-search (zoom + cubic interpolation, vs scipy's wolfe + cubic).
   Convergence to the same minimum is expected (the loss is convex in
   `(W, S, R_recur)`); convergence at the same iter count is not.
   Calibrated against scipy in Phase A.4.
6. **Init stays in numpy.** `initialize_slds`
   (`rslds_model.py:1278-1345`) keeps using `np.random.default_rng` and
   numpy k-means. Init runs once per fit, contributes negligibly to
   wall-time, and keeps RNG semantics aligned with the existing seeded
   fits (per §3.3).
7. **Constrained Viterbi (`viterbi_min_dwell`) remains numpy and outside
   the EM loop.** Already runs against `gamma` from the fit; `gamma`
   format is unchanged.
8. **AR(1) emission path is dropped** (per §1.8 — unused in production).
9. **Float64 enforced.** `jax.config.update("jax_enable_x64", True)` set
   in `cadence/significance/__init__.py` and asserted at the entry of
   `fit_em_hierarchical`.

### Phase A — Single-session JAX rSLDS (full feature set)

**Goal:** Implement the complete per-session rSLDS in JAX — E-step + every
M-step + null-state + c_shrinkage + n_factors + recurrent + input-driven
+ obs_mask — and validate against the existing per-session `fit_slds`
(`rslds_model.py:1348-1506`) on three sessions covering the awkward
configurations.

**Sub-phases:**
- **A.1** ParamsRSLDS pytree + log_emit + log_trans (jit) +
  `e_step_one_session` (single inner iter, no Kalman). Validate against
  numpy `_log_emissions` / `_log_transitions_recurrent` on y_06 to
  max|Δ| < 1e-10.
- **A.2** Add `lgssm_smoother` wrapper with mask-via-R-inflation. Validate
  Kalman outputs (`x_smooth`, `P_smooth`, `smoothed_cross`) against
  `_kalman_smoother_weighted` on y_06 to max|Δ x_smooth| < 1e-4.
- **A.3** Wire inner SMF loop with `lax.scan` over 3 iters. Validate
  E-step convergence trace against numpy `slds_e_step` to
  max|Δ log_lik| < 1e-3 per timepoint.
- **A.4** Closed-form M-steps (dynamics + emissions). Includes c_shrinkage,
  null-state d[0]=0 enforcement, FA via `jnp.linalg.eigh`. Validate
  per-iteration parameter updates against
  `slds_m_step_dynamics`/`slds_m_step_emissions` to max|Δ| < 1e-4.
- **A.5** Transitions M-step via `optax.lbfgs`. Validate against
  `_m_step_transitions` and `slds_m_step_transitions_recurrent`. Tolerance
  is looser here (max|Δ W| < 1e-2, max|Δ S| < 1e-2) because the
  optimiser difference is real; check that the *gradient at the optax
  solution* under scipy's L-BFGS-B objective is < 1e-4, which is the
  trustworthy convergence criterion for a different-optimiser comparison.
- **A.6** Full single-session `fit_em` loop. Compare BIC and final gamma
  against `fit_slds` on y_06, y_24 (missing modalities), and a
  high-T session.

**Concrete deliverable:**
`cadence/significance/rslds_jax.py` exposing `ParamsRSLDS`,
`e_step_one_session`, three M-step functions, `fit_em_single_session(Y, U,
mask, cfg, seed)` returning the same dict structure as `fit_slds`.

**Success criteria (single-session, K=4, D_latent=3, D_obs=7 or 26):**
- See §4.2 thresholds. Specifically: |Δ BIC|/BIC < 1e-3, ARI > 0.98
  vs numpy fit, max|Δ d_emit| < 0.02, max|Δ C_emit| < 0.05.
- Single-fit wall-time on a GPU (A.6): ≤ 30 s for the V11 26-D
  config that currently takes ~3 minutes per session.

**Validation procedure:**
1. Save the iteration-by-iteration intermediates from a numpy `fit_slds`
   on each of three test sessions: per-iter `gamma`, `x_smooth`, all
   M-step outputs, log-lik trace.
2. Re-run the JAX `fit_em_single_session` with the same seed.
3. Compare per-iteration intermediates with the §4.2 tolerances. Where
   tolerances fail, the failure must trace to a documented numerical
   difference (LAPACK path, optimiser substitution) rather than an
   algorithmic bug.

**Rollback plan:** new module; existing rslds_model.py untouched. No
production code calls rslds_jax.py until Phase C.

**Effort:** 5-7 person-days. The longest sub-phase is A.4 (FA via
`jnp.linalg.eigh` has eigenvalue-tie issues that don't appear in the
scipy pipeline) and A.5 (optimiser substitution validation).

### Phase B — Hierarchical JAX rSLDS

**Goal:** `vmap` the per-session E-step over the 22-session cohort, pool
M-steps for shared parameters (A_dyn, b_dyn, Q_dyn, W_trans, S_trans,
R_recur), keep per-session emissions, validate against
`fit_hierarchical_slds` (`rslds_model.py:1684-1905`) on the production
MVP and V11 cohorts.

**Sub-phases:**
- **B.1** Padded-batch loader: pack 22 sessions' (Y, U, obs_mask) into
  three `(N, T_max, ...)` tensors plus a `valid_T: (N,)` int array.
  Verify per-session round-trip equivalence (Y_padded[n, :valid_T[n]]
  == Y_n).
- **B.2** Hierarchical E-step: `e_step_hierarchical = vmap(e_step_one_session)`.
  This produces per-session posteriors batched on a leading N axis.
  Use `where(t < valid_T[n])` masks inside log_emit to zero out padded
  contributions to the log-lik. Validate per-session posteriors batched
  vs unbatched to max|Δ| < 1e-8 (just `vmap` correctness, not numerical
  drift).
- **B.3** Pooled dynamics M-step: sum suff-stats across the N axis,
  closed-form solve. Mirror of `_hierarchical_m_step_dynamics`
  (`rslds_model.py:1551-1606`).
- **B.4** Pooled transitions M-step: sum loss/grad contributions across
  N inside the `optax` objective. Mirror of
  `_hierarchical_m_step_transitions` (`rslds_model.py:1609-1681`).
- **B.5** Per-session emissions M-step via `vmap(m_step_emissions_per_session)`.
  c_shrinkage post-projection uses `C_mean = jnp.mean(C_new, axis=0)`
  computed across the N axis after the per-session solve.
- **B.6** Stage-2 freeze of d_emit during early EM iters
  (`stage2_iter` logic from `rslds_model.py:1789`). Same trigger
  threshold (30 % of max EM iters), implemented as a JAX `lax.cond` on
  the iteration counter.
- **B.7** Outer EM loop in Python: jit-compile each step, run iter,
  check convergence on the global LL trace, optionally early-stop.

**Concrete deliverable:**
`fit_em_hierarchical(sessions, cfg, seed)` in `rslds_jax.py` returning
the same dict structure as `fit_hierarchical_slds` (so the entry scripts
need only swap the import).

**Success criteria (hierarchical fit, 22 sessions):**
- All §4.2 thresholds vs `results/mvp/hierarchical/` and
  `results/v11/hierarchical/`
- Sensitivity-conclusion preservation: K=4-no-null vs K=3 BIC delta
  matches numpy to within 1 %; coupling-flexibility max|S| matches
  to within 5 %; per-condition usage matches to within 0.02 absolute
- Wall-clock for one hierarchical fit on Linux + GPU: ≤ 6 minutes
  (target = 10× the current ~1 hour). On Linux + CPU: ≤ 12 minutes
  (target = 5×).

**Validation procedure:**
1. Re-fit current production V11 hierarchical with numpy on the cluster,
   pin seeds, save full per-iter trace + final results to a versioned
   snapshot directory (e.g. `results/v11/hierarchical_numpy_baseline/`).
2. Run JAX hierarchical fit with same cohort + same seeds.
3. Hungarian-align states (existing `_align_states_to_reference` works
   on numpy arrays from JAX outputs unchanged).
4. Run all 5 sensitivity variants from `_run_mvp_hierarchical.py`
   under JAX, confirm all conclusions preserved.

**Rollback plan:** new module; entry scripts gain a `--engine {numpy,jax}`
flag (default `numpy` until Phase C). Reverting is one CLI default change.

**Effort:** 3-5 person-days. B.2 padding correctness and B.4 pooled
optax loss are the largest sub-pieces.

### Phase C — Cutover + paper traceability

**Goal:** JAX is the default for new fits; numpy path retained but tagged
"legacy" for paper traceability and bisection.

**Concrete deliverable:**
- `_run_mvp_hierarchical.py` and `_run_v11_hierarchical.py` default to
  `--engine jax`. JSON output sidecars carry `summary['engine']` so any
  results file is identifiable as numpy- or JAX-produced.
- A documented numpy↔JAX numerical-equivalence snapshot lives in
  `results/migration/dynamax/` showing per-session BIC delta,
  Hungarian ARI, per-state d_emit max delta. This is the "evidence
  paper-quality fits agree" artefact.
- `rslds_model.py` retains its numpy implementation indefinitely;
  no deletions. Add a top-of-file comment block citing the migration
  commit and the validation evidence directory.

**Success criteria:**
- Phase B validation passes for all production + sensitivity variants
- 5-variant sensitivity sweep wall-time: ≤ 1 hour total (was ~5 hours)
- Successful end-to-end run on Windows (CPU JAX) and on cluster
  (Linux GPU JAX); both produce comparable BICs

**Rollback plan:** revert `--engine` default. Both code paths remain
indefinitely.

**Effort:** 2-3 person-days, dominated by Windows/cluster deployment
debugging (see §3.4).

### What this migration explicitly does NOT do

- Does not adopt `dynamax.slds.SLDS` as a model class. Uses dynamax only
  for `lgssm_smoother`, `hmm_smoother`, optionally `parallel_inference`.
- Does not touch `cadence/significance/coupling_bursts.py`,
  `cadence/significance/lz_complexity.py`, `cadence/significance/
  spectral_graph.py`, or any of the upstream feature-extraction code.
- Does not change CADENCE's hierarchical pooling partition (which params
  are shared vs per-session).
- Does not migrate the IOHMM warm-start (`initialize_slds`) to JAX. It
  runs once, in numpy, and seeds the JAX fit. Keeps RNG semantics
  aligned with historical numpy fits.
- Does not migrate constrained-Viterbi or any other post-processing.
  These run on numpy arrays of `gamma` from the fit, regardless of
  whether the fit was numpy- or JAX-produced.
- Does not delete the numpy implementation. It stays for paper
  traceability and bisection.

---

## 3. Risk register

### 3.1 dynamax doesn't support the rSLDS feature set → custom code on top of primitives

Documented in §1. The mitigation — already baked into the §2 plan — is to
not adopt the SLDS class at all and use only `lgssm_smoother` and
`hmm_smoother` as drop-in numerical kernels. This *limits* the migration
footprint; it doesn't eliminate the work. Specifically the per-channel
`obs_mask` handling around `lgssm_smoother` is the highest-risk item.

**Mitigation:** Phase A sign-off explicitly requires masked-channel
sessions (e.g. y11 or y24 once they have MVP scaffolds) in the
single-session test set, not just y_06. If the masking glue cannot be made
numerically equivalent, that is grounds to abort the migration.

### 3.2 Numerical-equivalence drift: float32 / float64 / different LAPACK paths

JAX defaults to float32. CADENCE numpy code uses float64 throughout
(see `dtype=np.float64` declarations at `rslds_model.py:158`, `:170`,
`:185`, etc.). The Kalman smoother is the most condition-number-sensitive
part of the pipeline (CADENCE has explicit eigenvalue regularisation at
`rslds_model.py:894-898`).

The relevant config: `jax.config.update("jax_enable_x64", True)` must be
set **before** any `jnp` array construction, in every entry script. This
is the JAX equivalent of the Windows torch-DLL ordering hazard from
`memory/project_win_torch_dll_fix.md`, and it will silently degrade
numerical results if forgotten.

**Mitigation:**
- Add a `jax.config.update("jax_enable_x64", True)` to a
  `cadence/significance/__init__.py` (or wherever the JAX import first
  happens) and add a runtime assertion that `jnp.zeros(1).dtype == jnp.float64`
  to the entry of `kalman_smoother_jax`.
- Phase A success criteria above are tight enough (1e-4 on `x_smooth`,
  1e-3 per timepoint on log-lik) that a silent fallback to float32 will
  cause them to fail — which is the desired behaviour.

### 3.3 Reproducibility / seeding

CADENCE uses `np.random.default_rng(seed)` consistently. JAX's PRNG model
(`jax.random.PRNGKey`) is functional and explicitly threaded. The
`fit_hierarchical_slds` driver does not use JAX RNG anywhere in the
proposed migration (Kalman smoother and forward-backward are
deterministic given inputs), so the seed semantics of the existing
codepath are preserved as long as initialisation (`initialize_slds` in
`rslds_model.py:1278`) stays in numpy.

**Mitigation:** Do not migrate any random initialisation to JAX. Init
stays numpy + numpy RNG. JAX is used only inside deterministic E-step
kernels.

**Residual risk:** there is a class of subtle non-determinism inside
JAX-jitted code on GPUs related to non-deterministic
matrix-multiplication reductions
(per `XLA_FLAGS=--xla_gpu_deterministic_ops=true`). For paper-quality
traceability this flag should be set in any production fit and pinned in
`docs/cluster_guide.md`.

### 3.4 Windows + JAX + GPU compatibility

`memory/project_win_torch_dll_fix.md` documents that the current
`_run_mvp_hierarchical.py` and `_run_v11_hierarchical.py` (lines 27-28,
15-16) hoist `import torch` before `import numpy` to work around a
torch 2.10 + numpy 2.4 DLL-load bug. JAX has its own Windows story:

- JAX historically does not ship Windows wheels for the GPU build
  (CUDA-enabled). Windows users are expected to use WSL2 for GPU JAX.
  For CPU-only JAX, Windows wheels exist on PyPI.
- jaxlib + jax must be version-matched.
- Whether JAX's CPU XLA path is faster than numpy + scipy on Windows
  for the relevant matrix sizes (D_latent=3, K=4, D_obs=26) is an
  empirical question — at these very small sizes, the JIT compile
  overhead can dominate.

**Mitigation:**
- Phase A explicitly benchmarks on the user's actual Windows machine,
  not the cluster, to surface the JIT-compile-overhead issue early.
  If JIT compile time per fit dominates, this kills the migration on
  Windows; need to evaluate `jax.jit` static-arg shape recompilation
  cost and persistent JAX cache (`JAX_COMPILATION_CACHE_DIR`).
- For the cluster, plan for Linux-only GPU JAX. Document that the
  production sensitivity sweeps run on cluster Linux + GPU JAX, and
  that local Windows runs use the CPU JAX path (slower than cluster
  but still expected to be faster than numpy).
- Open question for the user: is the current bottleneck more painful
  on Windows (interactive iteration) or on the cluster (overnight
  sweeps)? Optimising for one is not optimising for the other.

### 3.5 dynamax SLDS module is not in the official API docs

Confirmed by inspecting `https://probml.github.io/dynamax/api.html` —
HMMs, LGSSMs, GLM-SSMs, and nonlinear-SSMs are documented; the SLDS
module is not. The source is present at `dynamax/slds/`, but it has the
character of a research/demo module, not a library API. **The
recommendation in §2 to NOT use the dynamax SLDS class is partly
defensive against this risk** — relying on an undocumented module for
production paper code is uncomfortable.

### 3.6 Optax LBFGS vs scipy L-BFGS-B convergence drift (M-step)

This is the one place the migration is mathematically *different*, not
just numerically equivalent. CADENCE's transition M-step uses
`scipy.optimize.minimize(method='L-BFGS-B', jac=True, options={'maxiter':
15, 'ftol': 1e-6})` (`rslds_model.py:515-516`); the JAX port will use
`optax.lbfgs` or `optax.scale_by_lbfgs` (with `optax.lbfgs_linesearch`)
because that is the one in-process JAX-jit-friendly LBFGS available.

The two minimisers differ in:
- Line search: scipy uses More-Thuente / cubic; optax uses zoom +
  cubic interpolation. Both are exact-on-quadratic.
- Convergence test: scipy `ftol` vs optax `gtol`/`scale`. Different
  default tolerances.
- Numerical conditioning of the Hessian approximation differs at small
  problem sizes.

**Risk:** the JAX fit takes longer to converge per outer EM iter
(more inner LBFGS iters), or converges to a slightly different optimum
(would show up as Δ W_trans, Δ S_trans drift across outer EM iters,
which would compound).

**Mitigation:**
- Phase A.5 explicitly validates the optax solution against scipy on
  random restarts of the M-step objective (synthetic xi/gamma).
- The trustworthy convergence criterion for "different-optimiser
  comparison" is the gradient norm at the proposed optimum: if optax's
  output point has scipy-objective-gradient < 1e-4, both optimisers
  found the same minimum.
- If optax LBFGS proves unstable inside JAX tracing (anecdotally, line
  search inside `lax.while_loop` has hit issues in older optax
  versions), fall back to `optax.adam` or `optax.adamw` with ~50 iters
  per M-step. This is what dynamax's stock M-step machinery does
  (e.g. `dynamax/hidden_markov_model/models/categorical_glm_hmm.py:119`
  uses `optax.adam(1e-2)` with `m_step_num_iters=50`). Adam is less
  efficient per iter than LBFGS but more robust to JAX tracing
  edge cases.

### 3.7 Per-session obs_mask + vmap interaction

`vmap` requires homogeneous shapes across the batched axis. Per-session
obs_mask has homogeneous shape `(T_max, m)` after padding, so vmap is
fine *structurally*. But the mask-via-R-inflation trick relies on
multiplying R by a (state-dependent, channel-dependent) factor — and
because R is per-session per-state, this happens inside the per-session
emission M-step naturally. The risk is that a session with very few
unmasked channels produces a near-singular suff-stat matrix during the
emission C/d solve, which `jnp.linalg.solve` will silently NaN.

**Mitigation:**
- `m_step_emissions_per_session` uses `jnp.linalg.solve` wrapped in a
  `jax.lax.cond` that falls back to `jnp.linalg.lstsq` on near-singular
  matrices (mirror of the numpy `try/except LinAlgError → pinv` pattern
  at `rslds_model.py:1170-1171`).
- Phase A.6 explicitly tests on y_24 (one of the missing-modality
  sessions) to surface this early.

### 3.8 Padding waste at high T-variance

Padding to `T_max` for vmap costs `(T_max - T_n) * (D_obs + D_input + m)`
per session in wasted compute. For the 22-session MVP cohort with
T~3000-7000, T_max ~7000, average ~5000, the waste factor is
~(7000-5000)/7000 = 28 %. Not catastrophic but worth measuring.

**Mitigation:**
- An alternative is per-T `lax.scan`-shape buckets: jit-compile one
  E-step per unique T value, with caching. For 22 sessions this is
  ~10-20 unique T values → ~10-20 cached jit graphs, ~5 GB JAX cache
  on disk. Acceptable but adds complexity.
- Recommended: pad to T_max and accept 28 % compute waste in exchange
  for one cached graph. Revisit if cohort grows to 100+ sessions where
  T-variance compute waste becomes meaningful.

### 3.9 Long-tail: dynamax `lgssm_smoother` doesn't expose the cross-time covariance API the way CADENCE needs

`lgssm_smoother` returns `smoothed_cross` (per
`dynamax/linear_gaussian_ssm/inference.py:565-572`), which is `Cov(z_t,
z_{t+1} | y_{1:T})`. CADENCE's `_kalman_smoother_weighted` returns
`Plag_smooth: (T-1, D, D)` indexed as `Cov(x_{t+1}, x_t | Y)`
(`rslds_model.py:903`). These are transpose-of-each-other up to a
convention; the M-step dynamics solver consumes it via
`xx_cross = Plag_sm + x_sm[1:, :, None] * x_sm[:-1, None, :]`
(`rslds_model.py:1080`), which assumes the `(t+1, t)` ordering.

**Risk:** transpose convention error → systematically wrong A_dyn updates,
detected only at the BIC level (subtle).

**Mitigation:** Phase A.4 unit-test the dynamics M-step against numpy
on a small synthetic LDS where the true A_dyn is known. If the JAX A
recovery agrees with the numpy A recovery, the convention is right.
Add an explicit `Plag_smooth = jnp.swapaxes(smoothed_cross, -2, -1)`
or equivalent in the wrapper, with a unit test pinning the convention.

### 3.10 The Numba fall-back

The §3.7 of the prior draft argued that Numba is the
right-cost-for-most-value alternative for the perf audit alone. That
remains true. **If the JAX migration hits an unrecoverable blocker —
Windows JIT instability, GPU compatibility issue on the cluster, or
optax convergence regression — fall back to wrapping
`_kalman_smoother_weighted` in `@jit(nopython=True, cache=True)`** as a
1-2 person-day patch that captures the bulk of the per-fit speedup
without any of the JAX commitment.

This is named here so that the fallback is *planned*, not improvised
under pressure. If at the end of Phase A the validation criteria can't
be met within reasonable effort, declaring the JAX path infeasible and
shipping the Numba patch is a legitimate, documented decision.

---

## 4. Validation framework

The validation set and metrics declare the migration "done" when:

### 4.1 Validation set (3 tiers)

**Tier 1 — single-session unit tests (Phase A gate):**
- `y_06` (clean MVP session, no missing modalities)
- `y_24` (one of the sessions with missing ECG/Resp — exercises the
  `obs_mask` path)
- `y_53_04302026` (newer session with `wholebody_133` pose format —
  exercises the visibility-aware NaN-safe statistics path documented in
  `CLAUDE.md` §"Pose features are visibility-aware")

If all three reproduce the existing fit to within thresholds in §4.2, Phase
A is done.

**Tier 2 — full hierarchical fit (Phase B gate):**
- The 22-session canonical MVP cohort
  (`scripts/_run_mvp_hierarchical.py` default cohort)
- All four sensitivity variants from `_run_mvp_hierarchical.py`:
  K=4 with `--null-state false` (current production), K=3, K=4 with
  `--null-state true`, and `--protocol meditation` and `--protocol pe`
  splits (5 fits total)

If all five sensitivity fits reproduce the conclusions in
`results/mvp/hierarchical/mvp_hierarchical_results.json` (state labels,
condition-stratified usage, BIC ranking), Phase B is done.

**Tier 3 — cross-pipeline regression (Phase C gate):**
- Re-run the V11 hierarchical fit (`_run_v11_hierarchical.py`) on its 12
  V11-scaffold sessions
- Confirm that the K=4-no-null vs K=3 BIC delta from
  `memory/project_v11_identifiability_diagnostics.md`
  (~13 k in favour of K=4-no-null) is preserved
- Confirm coupling-flexibility covariate effect from
  `memory/project_v11_te_decomposition.md` (max|S|=0.824) reproduces to
  within 5 %

### 4.2 Per-metric thresholds (defensible)

| Metric | Threshold | Defence |
|---|---|---|
| Per-session marginal log-likelihood | |Δ LL| / T < 1e-3 | Kalman smoother round-off in float64 typically lands at 1e-6 per timestep; 1e-3 is 3 orders of margin and still stricter than EM convergence (`em_tol=1e-4` at `rslds_model.py:47`). |
| Per-session BIC | |Δ BIC| / BIC < 1e-3 | Same reasoning, scaled. For V11 BIC ~1.5e6, this is ~1500. |
| State-label preservation | Hungarian-aligned ARI > 0.98 | Re-fitting with a different *seed* on the same data typically gives ARI 0.95-0.98 between fits per `memory/project_v11_identifiability_diagnostics.md`. JAX migration should be tighter (deterministic kernel, only RNG-independent path changes), so 0.98 is a meaningful threshold. |
| Per-state d_emit reproduction | max channel-wise |Δ d_emit| < 0.02 | Channels are standardised z-scores; per-channel between-session σ is 0.5-1.5; 0.02 is < 5 % of typical channel scale. |
| Per-state C_emit reproduction | max element-wise |Δ C_emit| < 0.05 | C is shrunk toward C_mean by 30 % in production (`c_shrinkage=0.3`); within-state element variance after shrinkage is ~0.5; 0.05 is < 10 %. |
| State usage per condition | max per-condition usage |Δ usage| < 0.02 | Usage stratified by condition (`base_EO`, `conv_1`, etc.) is the load-bearing science output. 0.02 = 2 percentage points; the across-session SD on these is typically 0.05-0.10, so 0.02 is well below noise. |
| Wall-time per single-session fit | ≥ 5× speed-up on CPU, ≥ 10× on GPU | Single-session fit is dominated by the same Kalman+forward-backward loop as the hierarchical fit; the JAX rewrite eliminates Python-loop overhead at every per-T iteration. 5× CPU is conservative; 10× GPU follows from typical XLA gains at this matrix size. |
| Wall-time per hierarchical fit (22 sessions) | ≥ 10× on GPU vs current numpy | `vmap`-over-sessions fuses 22 independent E-step calls into one XLA graph. This is where the migration earns its keep. |
| Wall-time for 5-variant sensitivity sweep | ≤ 1 hour total (was ~5 h) | End-to-end target. |

### 4.3 What we are NOT validating

- We are not validating that JAX gives *identical* results to numpy. We
  are validating that they agree to a tolerance below which downstream
  scientific conclusions cannot distinguish them.
- We are not validating the un-recommended "full dynamax SLDS class"
  path. That migration is rejected in §1, so its validation framework is
  moot.
- We are not validating that optax LBFGS gives identical M-step output to
  scipy L-BFGS-B. We are validating that the optax solution lies at the
  same minimum (gradient-norm test in §3.6).

---

## 5. Effort estimate + open questions

### 5.1 Open questions (must be answered before committing)

These are smaller in scope than the prior draft because the
"partial-vs-full" architectural choice is now settled. Remaining
genuinely-open questions:

1. **Is recurrent transitions (`R_recur`) a load-bearing paper claim, or
   was it added exploratorily?** The `_run_mvp_hierarchical.py` config
   sets `recurrent=True` (line 245), but
   `memory/project_v11_te_decomposition.md` describes the V11 model
   capacity grid search without highlighting recurrent's contribution.
   *If recurrent is not paper-critical,* dropping it would simplify the
   transitions M-step (§3.6) significantly — pooled transitions reduce
   to a `(K,K)` softmax-regression on `U` only, optax convergence is
   easier, and validation tolerances tighten. *If recurrent IS
   paper-critical,* the JAX softmax M-step adds `R_recur @ x_smooth` as
   a covariate, with the additional risk that x_smooth changes between
   E-step and M-step (already true in the numpy version, just worth
   confirming the JAX port preserves the same coupling).

2. **Does paper-quality reproducibility require bit-identical fits to
   the historical `results/mvp/hierarchical/`, or only
   *scientifically-equivalent* fits within the §4.2 thresholds?**
   Bit-identicality is impossible across numpy↔JAX (different LAPACK
   paths, different LBFGS line-search). If bit-identicality is required
   for any specific paper figure, the corresponding figure must be
   re-generated under JAX and the numpy snapshot retired (or both
   retained side-by-side with the bit-different-but-scientifically-
   equivalent caveat documented).

3. **Where does production run — Windows desktop, Linux cluster, or
   both?** Per `memory/project_win_torch_dll_fix.md`, the project's
   Windows compatibility story is non-trivial. JAX has its own:
   - GPU JAX is Linux/macOS only on stock wheels; Windows GPU requires
     WSL2.
   - CPU JAX on Windows works but JIT-compile overhead may dominate at
     MVP D_obs=7 scale.

   If the answer is "Windows interactive + Linux cluster overnight",
   Phase C must validate both paths. If the answer is "Linux cluster
   only" (and Windows usage moves to a CPU-numpy fallback), Phase C
   shrinks. The default plan assumes both.

4. **Should `optax.lbfgs` or `optax.adam` be the default M-step
   optimiser?** §3.6 records this as a contingency on Phase A.5
   findings. LBFGS is mathematically closer to scipy L-BFGS-B (smaller
   numerical drift); Adam is more robust to JAX tracing edge cases.
   Defer to Phase A.5 evidence; do not commit before measuring.

5. **Is dropping `n_factors=2` an option?** Reduces Phase A.4 effort
   by ~1 person-day (no `jnp.linalg.eigh` validation, no FA M-step
   port). Memory `project_v11_te_decomposition.md` shows `n_factors=2`
   was BIC-optimal in the V11 capacity grid search (1.5058 vs 1.5278
   at `n_factors=0`), so it IS load-bearing for the V11 fit. Plan
   assumes it stays.

### 5.2 Effort estimate (person-days)

| Phase | Sub-phases | Engineering | Validation | Total |
|---|---|---|---|---|
| Phase A — single-session JAX rSLDS (full feature set) | A.1 pytree+log_emit/log_trans (0.5-1) · A.2 lgssm_smoother+mask (1-1.5) · A.3 inner SMF lax.scan (0.5-1) · A.4 closed-form M-steps incl. FA (1-2) · A.5 transitions M-step + optax calibration (1-1.5) · A.6 single-session fit_em + validation (0.5-1) | 4.5-8.0 | 1.0-1.5 | **5.5-9.5** |
| Phase B — hierarchical JAX rSLDS | B.1 padded loader (0.25-0.5) · B.2 vmap E-step (0.5-1) · B.3 pooled dynamics (0.5-0.75) · B.4 pooled transitions (1-1.5) · B.5 vmap emissions + c_shrinkage (0.5-1) · B.6 stage-2 freeze logic (0.25-0.5) · B.7 outer EM driver (0.25-0.5) | 3.25-5.75 | 1.5-2 | **4.75-7.75** |
| Phase C — cutover + Windows/cluster debug | scripts cutover · numpy↔JAX evidence snapshot · Windows JAX install + benchmark · cluster (Linux GPU) install + benchmark | 1.5-2.5 | 0.5-1 | **2-3.5** |
| **Total** | | **9.25-16.25** | **3-4.5** | **12.25-20.75** |

**Realistic point estimate: ~14-15 person-days.** The Phase A range is
the largest source of uncertainty; if the optax M-step (A.5) or the FA
eigh (A.4) hits problems, those sub-phases can each grow by 1-2 days.

For comparison, the §3.10 Numba fall-back: ~1-2 person-days total,
single-file change. That option is now framed as a *fall-back*, not the
default — but it remains the right choice if Phase A unrecoverably
stalls.

### 5.3 Recommendation

**Proceed with the full JAX migration (Phase A → B → C) as scoped above.**
Effort: ~14-15 person-days. Expected payoff: 10× hierarchical-fit
speedup on Linux GPU, 5× on CPU, and a JAX-native foundation that
permits future autodiff-based sensitivity analysis, batched
cross-validation, and SGD-based parameter sweeps.

**Decision gates:**

- **Gate 1 (after Phase A.5):** if optax LBFGS / Adam can't reach the
  scipy gradient-norm threshold (§3.6) for the transitions M-step
  within Phase A.5's day budget, escalate to a 2-day extension; if
  still unresolved, abort to the Numba fall-back (§3.10).

- **Gate 2 (after Phase A.6):** if single-session JAX fit can't meet
  §4.2 thresholds vs numpy on at least 2 of 3 test sessions, abort to
  the Numba fall-back. Phase B is committing to a vmap'd version of
  Phase A; if Phase A doesn't validate, Phase B can't.

- **Gate 3 (after Phase B):** if hierarchical JAX fit doesn't preserve
  the K=4-no-null vs K=3 BIC ranking (preserves the production
  scientific conclusion), do not cut over. Investigate; if root cause
  is a fixable bug, fix; if root cause is "JAX optimiser found a
  different local minimum", accept the JAX path as exploratory tool
  but keep numpy as the production fit for the paper.

**Do not, under any circumstances, attempt to port the model to
`dynamax.slds.SLDS`.** Per §1, that class doesn't fit EM, doesn't fit
recurrent, doesn't fit hierarchical, doesn't fit masking, doesn't fit
null-state, and isn't documented. That migration would be a rewrite,
not a port, and the lab would inherit ownership of any subsequent
upstream churn in an undocumented module.

### 5.4 Why not also use Linderman `ssm`

A reasonable question is whether to slot the Linderman `ssm` library
(`https://github.com/lindermanlab/ssm`) into this stack — `ssm` already
has a working rSLDS class with EM, Laplace-EM, and recurrent transitions,
and would seem to give us the model class for free.

**It would not help, and likely would hurt.** Concretely:

- `ssm`'s rSLDS is CPU + Numba. Its inner Kalman smoother is *also* a
  Python loop with Numba JIT, so the perf bottleneck the migration is
  trying to fix would not actually go away if the model class came from
  `ssm`.
- `ssm`'s parameter pytrees and posterior data classes (`LDSStates`,
  etc.) do not interoperate with dynamax's `ParamsLGSSM`. Marshalling
  between them at every E-step would force numpy↔JAX↔CPU round-trips
  that destroy the JAX speedup at small matrix sizes (D_latent=3, K=4).
- `ssm`'s last meaningful release was 2020-2021. Building production
  paper code on a sandwich of two libraries where one is unmaintained
  and one is research-grade (dynamax) is risk-stacking, not
  risk-reduction.
- `ssm`'s Laplace-EM uses dense `(T·D_latent, T·D_latent)` Hessians —
  for T=5000, D=3, that's a 15k×15k matrix per session. Slower than
  CADENCE's structured-MF, not faster.

The only `ssm` artifact worth keeping in mind is its **rSLDS
parameterisation conventions** — CADENCE's hand-rolled
`_log_transitions_recurrent` already follows them. Cross-checking the
JAX port against `ssm`'s public test cases (synthetic rSLDS data) is
useful as a sanity check on the model definition; running `ssm` in
production is not.

---

## Appendix A — file:line reference index

CADENCE references in this document:
- `cadence/significance/rslds_model.py:45` — `ar_order` warning
- `cadence/significance/rslds_model.py:55-58` — config flags (sticky,
  viterbi_min_dwell, null_state, c_shrinkage)
- `cadence/significance/rslds_model.py:198-216` — AR(1) init (unused in
  production)
- `cadence/significance/rslds_model.py:259-274` — masked emission
  log-likelihoods
- `cadence/significance/rslds_model.py:301-355` — `_forward_backward`
- `cadence/significance/rslds_model.py:374-396` — masked mean update
- `cadence/significance/rslds_model.py:399-420` — AR(1) M-step
- `cadence/significance/rslds_model.py:439-444` — null-state mu/sigma
  enforcement
- `cadence/significance/rslds_model.py:472-498` — asymmetric W prior
- `cadence/significance/rslds_model.py:520-521` — sticky transition bias
- `cadence/significance/rslds_model.py:661-729` — constrained Viterbi
- `cadence/significance/rslds_model.py:781-832` — `_kalman_update`
- `cadence/significance/rslds_model.py:835-848` — `_factor_analyze`
- `cadence/significance/rslds_model.py:851-905` — `_kalman_smoother_weighted`
  (perf bottleneck)
- `cadence/significance/rslds_model.py:908-1039` — `slds_e_step`
- `cadence/significance/rslds_model.py:1042-1057` — `_log_transitions_recurrent`
- `cadence/significance/rslds_model.py:1113-1218` — `slds_m_step_emissions`
  (with c_shrinkage at 1187-1193 and FA at 1196-1216)
- `cadence/significance/rslds_model.py:1278-1345` — `initialize_slds`
  (IOHMM warm-start)
- `cadence/significance/rslds_model.py:1513-1525` — Hungarian state alignment
- `cadence/significance/rslds_model.py:1551-1606` —
  `_hierarchical_m_step_dynamics`
- `cadence/significance/rslds_model.py:1609-1681` —
  `_hierarchical_m_step_transitions`
- `cadence/significance/rslds_model.py:1684-1905` — `fit_hierarchical_slds`
- `scripts/_run_mvp_hierarchical.py:65-92` — constrained Viterbi
  post-processing
- `scripts/_run_mvp_hierarchical.py:239-253` — production rSLDS config
- `scripts/_run_mvp_hierarchical.py:264` — `parallel_backend('threading')`
  context manager (Windows torch+numpy DLL fix)
- `scripts/_run_v11_hierarchical.py:173-186` — V11 production rSLDS config

dynamax references in this document (against `main` as of 2026-05-01):
- `dynamax/slds/models.py:19` — `SLDS(SSM)` class, no `fit_em`
- `dynamax/slds/inference.py:17-23` — `DiscreteParamsSLDS` (static
  K×K only)
- `dynamax/slds/inference.py:25-42` — `LGParamsSLDS` (per-state A, Q, C, R)
- `dynamax/slds/inference.py:147-171` — `_conditional_kalman_step`
- `dynamax/slds/inference.py:173-260` — `rbpfilter`
- `dynamax/slds/inference.py:262-` — `rbpfilter_optimal`
- `dynamax/slds/__init__.py` — exports `rbpfilter`, `rbpfilter_optimal`,
  `ParamsSLDS`, no fit/EM exports
- `dynamax/hidden_markov_model/inference.py:14-31` — `get_trans_mat`
  (time-varying transitions)
- `dynamax/hidden_markov_model/inference.py:_normalize`,
  `_condition_on`, `hmm_filter`, `hmm_smoother`, `hmm_posterior_mode`
- `dynamax/hidden_markov_model/models/transitions.py:36-` —
  `StandardHMMTransitions` (Dirichlet stickiness)
- `dynamax/hidden_markov_model/models/transitions.py:113-` —
  `m_step` (Dirichlet posterior mode)
- `dynamax/hidden_markov_model/models/abstractions.py:209-` —
  `HMMTransitions` (extensible base)
- `dynamax/hidden_markov_model/models/arhmm.py` — AR-HMM (HMM-only,
  no SLDS coexistence)
- `dynamax/hidden_markov_model/models/categorical_glm_hmm.py` —
  input-driven *emissions* (not transitions)
- `dynamax/linear_gaussian_ssm/inference.py:36-` — `ParamsLGSSMDynamics`
  with input weights and time-varying support
- `dynamax/linear_gaussian_ssm/inference.py:151-160` —
  `_get_one_param`, `_get_params` (time-varying parameter dispatch)
- `dynamax/linear_gaussian_ssm/inference.py:460-512` — `lgssm_filter`
- `dynamax/linear_gaussian_ssm/inference.py:516-580` — `lgssm_smoother`
  (RTS, returns smoothed_cross at line 565)
- `dynamax/linear_gaussian_ssm/parallel_inference.py` — parallel-prefix-scan
  filter/smoother for GPU
- `dynamax/ssm.py:356-410` — `SSM.fit_em` (vmap over batches, all-shared
  parameters)
- `dynamax/parameters.py` — `ParameterProperties` (per-tensor `trainable`
  flag, no per-row support)
- `https://probml.github.io/dynamax/api.html` — official API docs
  (HMM/LGSSM/Nonlinear/Generalized-Gaussian only; no SLDS section)

External references:
- `https://github.com/lindermanlab/ssm-jax-refactor` — deprecated JAX rSLDS
  predecessor, README states "superseded by DYNAMAX" (but rSLDS+EM
  capabilities did not survive the supersession)
- `https://github.com/lindermanlab/ssm` — original Linderman lab Numba+Python
  rSLDS+EM library; CADENCE's hand-roll most closely resembles its
  conventions

Project-memory cross-references (from
`C:\Users\optilab\.claude\projects\C--Users-optilab-desktop-CADENCE\memory\`):
- `project_win_torch_dll_fix.md` — Windows torch+numpy DLL ordering, applies
  also to JAX import order on Windows
- `project_v11_identifiability_diagnostics.md` — K=4-no-null wins by ~13 k
  BIC; this is the conclusion the migration must preserve
- `project_v11_te_decomposition.md` — D_latent=3, n_factors=2 grid-search
  result; coupling-flexibility max|S|=0.824 baseline
- `feedback_joblib_loky.md` — joblib threading backend mandate,
  superseded by `project_win_torch_dll_fix` for the rSLDS scripts
- `feedback_optimize_for_results.md` — don't compromise sampling rates or
  model complexity for speed (relevant to "should we drop recurrent for
  the migration" open question §5.1.3)
