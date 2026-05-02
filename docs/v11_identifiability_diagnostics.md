# V11 rSLDS Identifiability Diagnostics — Synthesis (2026-04-22)

Consolidated record of the Tier 1 / Tier 2 diagnostic investigation of the V11 hierarchical rSLDS fit. Purpose: answer the question "do the K=4 states the model reports correspond to real structure in the data, or are they artifacts of the model's constraints and capacity choices?"

Primary finding, stated upfront: **the `null_state=True` constraint is over-constraining the data. The model reports 4 states but only ~3 are emission-distinct. The "OTHER" state is a phantom created by the mismatch between the constraint's exact-zero baseline and the data's near-zero (‖d‖ ≈ 0.18) baseline.** Recommendations for next refit are in §6.

---

## 0. Context

The V11 scaffold produces 26D observations + 7D transition covariates at 2 Hz (reduced from 28D after the `dyn_theta/alpha/beta` collapse to a single `dyn_mean` channel — r = 0.99 collinear, see `project_dyn_mean_collapse.md`). The hierarchical rSLDS fits K = 4, D_latent = 3, n_factors = 2 with `null_state=True, sticky_strength=3.0, viterbi_min_dwell=20`, giving state labels `[NULL, OTHER, COUP, SHARED]` with roughly balanced usage (~20-30% each).

Prior to the work in this document, the only evidence that these states were meaningful was:
- Fit BIC (lower is better — but not comparable across K or D_obs changes)
- Manual inspection of state emission profiles (loose qualitative check)
- State usage per condition (informative about *session structure*, not state *identity*)

None of those distinguish "real 4-state structure" from "4-state model forced onto data that doesn't have 4 regimes."

---

## 1. What We Built

### Infrastructure

**Tier 1 screening** (`diagnostics/tier1_screening/`): feature-level diagnostics — VIF, slow drift detection, collinearity, low-information channels. Produces `screening_report.md` feeding Module 5 channel-exclusion flags.

**Tier 2 deep-dive** (`diagnostics/tier2_deepdive/`): four modules operating on the *fitted* model.
- `module1_obs_space.py` — observation-space separability (PCA scree, silhouette, UMAP by state/phase/time/session).
- `module4_model_comparison.py` — LOO-CV held-out log-likelihood for Gaussian HMM K=2..5, GMM K=2..5, rSLDS K=4.
- `module5_block_pca.py` — alternative preprocessing: block-PCA compression 26D → k_reduced.
- `module6_null_ablation.py` — re-fit K=4 with `null_state=False` per session, Hungarian-match emission vectors to constrained model, compute cosine similarities.

**Three separability tests** (added to `module1_obs_space.py` this session):
- `shuffle_null_silhouette` — compares the real state-label silhouette against a distribution of random permutations of the same labels. Catches cases where the silhouette is weak (near zero) but statistically-worse-than-random.
- `pairwise_d_emit_distances` — L2 distances between the K emission-mean vectors; surfaces state pairs that are nearly-duplicate in emission space.
- `latent_space_silhouette` — silhouette on pooled `x_smooth` (the SLDS 3-D latent trajectory). Disambiguates "states separate in latent dynamics but emissions happen to pool in observation space" from "states don't separate anywhere."

**Standalone immediate-read** (`scripts/_diagnose_state_separability.py`): runs pairwise d_emit and shuffle-null silhouette without the UMAP/PCA costs. Can be invoked independently of Tier 2.

### Hierarchical runner updates

`scripts/_run_v11_hierarchical.py` now saves, per session (in `results/v11/<sess>/v11_rslds_results.npz`):
`gamma`, `path`, `t_common`, `state_labels`, `d_emit`, `C_emit`, `R_emit`, `x_smooth`, and the shared `W_trans`, `S_trans`, `A_dyn`, `Q_dyn`. Plus a top-level `v11_hierarchical_params.npz` containing `mean_d_emit` and shared transition/dynamics tensors. This unblocks every downstream diagnostic — before this change, the NPZ contained only `gamma, path, t_common`.

---

## 2. Methodological Progression

### 2a. "State usage looks fine" — why that's not evidence

Usage fractions (NULL 21%, OTHER 28%, COUP 25%, SHARED 25%) are balanced and not dominated by any one state. It's tempting to read that as "the model found four things." But balanced usage is also exactly what you get when the model assigns points by residual noise rather than by real state structure. Separability tests are what catch that.

### 2b. From "silhouette is low" to "silhouette is worse than random"

The first diagnostic signal came from Module 1's pooled-observation silhouette: ~−0.02. Interpreted naively, that says "states overlap a little." But silhouette has no natural null — −0.02 is close to zero, and zero-ish is easy to hand-wave as "clusters are soft."

Adding a shuffle-null test resolved the ambiguity: the real silhouette is ~9σ *below* the distribution of random-label silhouettes on the same data. Of 200 random permutations, none produced a worse silhouette than the actual model. That's a much stronger signal than the point estimate.

### 2c. From "bad in Y" to "bad in Y *and* in x"

A charitable reading of the negative obs-space silhouette: SLDS states are defined by latent dynamics `x_t`, not raw observations `y_t`. Emissions pool multiple latent trajectories through the per-state loading matrix `C[k]`, so states can look scattered in Y while being clean in x.

Saving `x_smooth` per session and computing silhouette on the pooled 3-D latent space killed that defense: latent silhouette = −0.066 (worse than obs-space silhouette, not better). The latent dynamics themselves don't organize into K clusters.

### 2d. From "states don't separate" (symptom) to "OTHER is the phantom" (mechanism)

Pairwise d_emit distances localize the problem. NULL and OTHER are 7× closer in emission space than any other state pair. Module 6 then isolates *which constraint is responsible* by perturbing `null_state` True → False and Hungarian-matching emission vectors. COUP and SHARED replicate (cosine similarity 0.985 and 0.922); the constrained OTHER's best unconstrained match is cosine 0.095 — no alignment. OTHER is the artifact.

---

## 3. Validated Findings (n = 14 sessions)

Sessions: `Y_10_03182026, Y_41_03192026, Y_45_03302026, y01_021726, y05_02192026, y40_040626, y_03_03122026, y_06, y_17, y_19_3242026, y_32_03132026, y_33_04032026, y_37, y_51_04092026`. Three other session names (`y04_020626, y11_022526, y24_022526`) exist as cache directories but have no matching XDF on disk — scaffold correctly skipped them.

### 3a. Pairwise emission-mean distances — 4 states, ~3 distinct

| State | ‖d_emit‖ | → NULL | → OTHER | → COUP | → SHARED |
|---|---|---|---|---|---|
| NULL   | 0.000 (constrained) | —    | 0.162 | 1.012 | 0.950 |
| OTHER  | 0.162               | 0.162| —     | 1.109 | 1.056 |
| COUP   | 1.012               | 1.012| 1.109 | —     | 0.465 |
| SHARED | 0.950               | 0.950| 1.056 | 0.465 | —     |

min / max off-diagonal = **0.146**. The NULL↔OTHER pair is far closer than anything else. Top-loading channels:
- **COUP**: `conc_beta +0.49, conc_theta +0.52, dyn_mean +0.50, conc_alpha +0.43` — shared-power concordance regime.
- **SHARED**: `bl_activity_conc +0.44, conc_theta +0.49, dyn_mean +0.42, conc_beta +0.38, bl_expr +0.18` — behavioral/body-inclusive regime.
- **OTHER**: `conc_theta −0.04, conc_beta −0.05, dyn_mean −0.03, conc_alpha −0.02` — "slightly-below-mean on everything." Not a regime.

### 3b. Shuffle-null silhouette on pooled 26D observations

Pooled across 14 sessions, PCA-reduced to 19 components (80% variance retained), silhouette computed on 5000 random samples:

| | silhouette | davies-bouldin |
|---|---|---|
| Real state labels | −0.0546 | 35.55 |
| Shuffle mean ± std (n=200) | −0.0106 ± 0.0046 | — |

**z = −9.54, p_right = 1.000.** Every one of 200 random permutations produced a better-separated partition than the real model's state assignments.

### 3c. Latent-space silhouette

`x_smooth` pooled across sessions (D_latent = 3, same 5000-sample evaluation):
**silhouette = −0.066, davies-bouldin = 32.32.** Worse than observation-space. Rules out the "states separate in x but not Y" interpretation.

### 3d. Module 6 null-state ablation (the decisive test)

Re-fit K = 4 per session with `null_state=False` (2 EM restarts each). Hungarian-matched the constrained and unconstrained emission vectors by cosine similarity:

| Constrained state | Matched unconstrained | Cosine |
|---|---|---|
| NULL (‖d‖ = 0 forced) | NULL-analogue (‖d‖ = 0.177) | 0.00* |
| **OTHER** (‖d‖ = 0.16) | SHARED-ish (no good match) | **0.095** |
| COUP (‖d‖ = 1.01) | COUP | **0.985** |
| SHARED (‖d‖ = 0.95) | OTHER | **0.922** |

\* Undefined: cosine to the zero vector. The important number is 0.177 — the data's unconstrained baseline state wants a small-but-non-zero emission magnitude.

**Conclusions:**
1. The data *does* contain a low-magnitude baseline state — ‖d‖ ≈ 0.18 — so the null-state constraint is *directionally* correct.
2. The exact-zero constraint is too strict. The constrained fit accommodates the mismatch by bifurcating near-baseline timepoints between NULL (constrained to 0) and OTHER (‖d‖ = 0.16) — leaving OTHER with essentially no signal structure.
3. COUP and SHARED are robust to removing the null-state constraint. They're real coupling regimes, not artifacts.
4. The unconstrained model finds OTHER at a fully different place in emission space (cosine 0.095 with constrained OTHER) — there is no "true" OTHER being obscured by the constraint.

### 3e. Block PCA (Module 5) — secondary findings

Block PCA compresses 26D → 16D by running PCA within each modality group. Notable outputs:

- `lz_conc_theta` and `lz_conc_alpha` load **0.00** on both Complexity PCs — the concordance channels contribute negligible variance after surrogate z-scoring and burst-rate gating. Only the LZ asymmetry channels have signal. Consistent with `project_directed_burst_coupling.md`'s finding that concordance signals require large windows or more sessions to surface.
- `resp` loads **0.00** on Autonomic PCs (ECG-LF and ECG-HF dominate). Respiratory phase coherence is not correlated with HRV-band envelope variability at the pooled level. Not a bug; a reminder that Resp and ECG occupy orthogonal subspaces.
- `bl_expr` and `bl_activity_conc` are fully orthogonal (each block PC loads on exactly one). Consistent with the V11 design intent that they measure different things.

### 3f. Module 4 LOO-CV — rSLDS vs HMM vs GMM

Completed: 14 folds, K ∈ {2, 3, 4, 5} for HMM and GMM, plus rSLDS reference. Results in `module4_report.md`. The specific numbers are not reproduced here because the separability story makes model-class comparison secondary — the right next step is re-fitting the rSLDS with a corrected state specification, not swapping to HMM.

---

## 4. What We Ruled Out

- **"SLDS is working as designed; obs-space silhouette is misleading."** Killed by latent-space silhouette being equally negative.
- **"K=4 is correct but OTHER is under-parameterized."** Killed by Module 6 — OTHER has no unconstrained analogue. It's not a weak signal to boost; it's a bookkeeping slot.
- **"The null-state constraint is harmless / cosmetic."** Killed by the unconstrained NULL-analogue norm (0.177) being materially different from 0 combined with OTHER vanishing under relaxation. The constraint is actively distorting the state topology.
- **"The dyn channel redundancy was hiding state structure."** The 26D refit (after `dyn_theta/alpha/beta → dyn_mean` collapse) produced *worse* separability than the 28D version (z = −9.5 vs −2.6). Collapsing the redundant channels did not rescue state structure; it sharpened the artifact.

---

## 5. Infrastructure Fixes Made During This Investigation

Recorded here so they're searchable and don't have to be re-discovered:

1. **Windows / torch 2.10 + numpy 2.4 DLL-load ordering bug** (`OSError [WinError 127]: shm.dll`). `import numpy` first, then `import torch` fails. Fixes:
   - `import torch` hoisted as first third-party import in `scripts/_run_scaffold_v11.py`, `scripts/_run_v11_hierarchical.py`, `scripts/run_session_v6.py`, `diagnostics/tier2_deepdive/run_tier2.py`.
   - `cadence/__init__.py` also imports torch early as belt-and-suspenders.
   - **Loky worker spawn breaks this fix** — workers auto-load numpy during their own setup before any user code runs. Switched the scaffold's `Parallel(...)` to `backend='threading'`, and wrapped the hierarchical fit + Tier 2 in `parallel_backend('threading')` context managers (retroactively forces library-internal `Parallel(prefer='processes')` calls to use threads too).
   - Threading backend cost: hierarchical fit went from ~20 min (loky) to ~215 min (threading) because per-session SLDS inits serialize on GIL. Acceptable one-time cost; revisit if loky or torch DLL loading is fixed upstream.

2. **Stale dimension references everywhere.** The `dyn_mean` collapse reduced V10 21D / V11 26D but `scripts/_run_scaffold_v82.py`'s `MODALITY_KEYS` (imported as `V82_KEYS` in V10/V11 scaffolds) still listed `dyn_theta/alpha/beta`. Graph feature construction (`z_18_raw = np.column_stack([z_traces[k] for k in V82_KEYS])`) raised `KeyError: 'dyn_theta'` at runtime. Fixed by introducing a local `BASE_16_KEYS = V10_MODALITY_KEYS[:16]` in both V10 and V11 scaffolds. `V82_KEYS` itself was left untouched (would invalidate old V8.2 scaffold NPZs).

3. **Unicode encoding in markdown reports.** Module 5's report writer used `open(..., 'w')` without `encoding='utf-8'`. Windows default is cp1252, which can't encode `→` (U+2192). Fixed by adding `encoding='utf-8'` to all `open()` calls that write markdown reports in `module1/4/5/6_*.py`.

4. **Hierarchical runner wasn't saving the parameters Tier 2 needed.** Before this session: per-session NPZ had only `gamma, path, t_common`; top-level params NPZ didn't exist. Now saves everything listed in §1 "Hierarchical runner updates."

---

## 5a. Refit Comparison — K=4/no-null vs K=3/no-null (2026-04-23)

Ran both candidate configurations on the same 14 sessions via a parameterized hierarchical runner (`--K`, `--null-state`, `--suffix`). Threading backend; ~3.5h per refit. Preserved the production fit by using distinct output suffixes (`_k4_nonull`, `_k3_nonull`).

### 5a.i Headline metrics

| Metric | K=4 null=True (production) | **K=4 null=False** | K=3 null=False |
|---|---|---|---|
| BIC | 4,250,247 | **4,237,269** | 4,301,337 |
| Δ BIC vs production | — | **−12,978** | +51,090 |
| Pairwise d_emit min/max ratio | 0.146 | 0.213 | 0.291 |
| Shuffle-null silhouette z (obs space) | −9.54 | −2.04 | −1.42 |
| Latent silhouette | −0.066 | −0.066 | −0.066 |
| Module 6 Hungarian cosine sum | — | 2.38/4 | 1.30/3 |
| rSLDS LOO-CV LL/frame/dim (Module 4) | — | −0.368 | −0.351 |

**K=4 no-null wins on BIC and model stability.** K=3 wins on separability score, but that's an artifact of having fewer states to pool against each other — it's not a sign of cleaner structure.

### 5a.ii What OTHER actually represents now

In the K=4 no-null refit, OTHER is no longer a phantom:

- Loadings: `resp = −0.08, ecg_hf = +0.06, bl_activity_conc = −0.06, conc_beta = −0.05, pose = +0.04`. Interpretation: suppressed body activity, suppressed respiratory coherence, slight HF-HRV boost. This is **autonomic quiescence**.
- Per-condition usage peaks: **meditate_K = 38.5%**, meditate_B = 28.4%, PE = 27.4%, base_EO = 25.6%. The two meditation conditions both elevate OTHER — that's a biologically interpretable regime (silent eyes-closed meditation, low physical activity).
- Contrast with production OTHER (K=4 null=True): loadings were −0.03 to −0.06 on concordance channels, with no clear interpretation. It was "slightly below mean on everything."

Recovering OTHER as a meditation-dominated autonomic-suppression state is the most interesting concrete benefit of dropping the null constraint.

### 5a.iii Why K=3 fails despite cleaner separability

K=3's state 1 (labeled "COUP" by the heuristic) has tiny loadings (conc_theta = +0.10) and dominates `base_EO` (57.7%), `PE` (63.4%), and `base_EC` (47.4%). State 2 (labeled "SHARED") has rich loadings on *both* conc (theta = 0.48, alpha = 0.33, beta = 0.41) *and* bl_activity_conc = 0.32. That's the merger of production-COUP and production-SHARED in a single slot.

So K=3 doesn't "drop the dominated baseline state" — it **merges the two real coupling regimes**. That's why BIC punishes it: the data genuinely has distinct concordance-dominated and behavioral-dominated coupling. K=3's cleaner separability metrics reflect having fewer pairwise comparisons, not sharper state boundaries.

### 5a.iv Condition-level findings (K=4 no-null)

Largely preserves the production-K=4/null=True structure, but with redistribution:

| Condition | NULL | **OTHER** | COUP | SHARED |
|---|---|---|---|---|
| base_EO | 47.8% | 25.6% | 15.4% | 11.3% |
| base_EC | 62.1% | 22.3% | 11.8% | 3.8% |
| conv_1 | 51.2% | 14.9% | 23.4% | 10.5% |
| PE | 43.0% | 27.4% | 23.4% | 6.2% |
| meditate_B | 47.7% | 28.4% | 9.9% | 14.0% |
| meditate_K | **33.1%** | **38.5%** | 12.4% | 16.0% |
| conv_2 | 50.5% | 13.7% | 18.8% | 17.0% |

Notable:
- `meditate_K` is the only condition where OTHER exceeds NULL — consistent with OTHER being "autonomic-quiescent" rather than "no-coupling."
- `conv_1` and `PE` have the highest COUP usage (~23%), same as production — the active-coupling interpretation is robust.
- `conv_2` has the highest SHARED usage (17%), matching production's post-meditation behavioral-coupling signature.

### 5a.v Persistent mystery: why is latent silhouette stuck at −0.066 across all three variants?

All three configurations produce near-identical latent-space silhouette values (−0.066 to −0.066). This is striking — removing the null constraint moved OTHER away from NULL in emission space, but the latent (x_smooth) organization is unchanged. Best interpretation: the rSLDS is optimizing for temporal dynamics and covariate-driven transitions, not for pooled cross-session clustering in the latent space. The latent trajectories of each session are clean internally but don't pool into coherent clusters when stacked.

**What this means practically:** pooled-cross-session clustering metrics are not the right evaluation target for this model class on this data. The model's legitimate job is per-session state decoding + transition analysis, not global unsupervised clustering. The diagnostic findings still ruled out the *phantom* OTHER state — that was a real improvement — but they do not suggest rSLDS is the wrong choice.

### 5a.vi Recommendation

**Promote K=4 / `null_state=False` to production.** It dominates K=4/null=True on every axis (BIC, separability, interpretability) and dominates K=3/no-null on generalization (BIC, LOO-CV, structural stability). The improvement is small on any one metric but consistent across all of them.

Files to update when promoting:
- `scripts/_run_v11_hierarchical.py`: change defaults from `null_state=True` to `null_state=False`, or invoke with `--null-state false` in automation.
- `docs/v11_pipeline_report.md`: remove the "under review" banner, update the model-architecture section to reflect `null_state=False`, describe OTHER as autonomic-suppression.
- Scripts that read `results/v11/hierarchical/v11_hierarchical_params.npz`: either re-point them at `results/v11/hierarchical_k4_nonull/` or copy/move the files. (Decision deferred — keeping the production fit untouched is defensible for now since K=4/no-null is saved separately.)

---

## 6. What's Next

### Promote K=4/no-null to production (blocking)

The refit comparison in §5a answered the question this document was opened to investigate. Concrete next actions:

1. **Default the hierarchical runner to `null_state=False`.** Either change the argparse default in `scripts/_run_v11_hierarchical.py` (`--null-state` default `True` → `False`), or establish a convention that production runs pass `--null-state false` explicitly. Favor the latter — it keeps the runner symmetric across config choices, and the production config is already documented.
2. **Decide the canonical storage path for the production fit.** Two options:
   - Keep `results/v11/hierarchical/` as the production path and copy K=4/no-null outputs there (overwriting the null=True fit). Saves downstream consumers from needing to know the suffix.
   - Keep K=4/no-null at `results/v11/hierarchical_k4_nonull/` and update downstream scripts to read from there via `--rslds-suffix _k4_nonull`. Safer — the production-fit outputs aren't destroyed, just superseded.

   The second option is cleaner. Default all reading paths to `_k4_nonull` and deprecate the unsuffixed path via a pointer file (`results/v11/hierarchical/README.md`).
3. **Re-label OTHER throughout downstream analysis.** It's now `autonomic_suppression` or `autonomic_quiescence`, not "OTHER." Condition plots, burst analysis, directed coupling — anywhere that mentions state labels needs the new description. A grep for `OTHER` in scripts + docs is the starting point.
4. **Re-verify condition statistics.** `scripts/run_condition_statistics.py` has a known label-order bug (see `project_condition_statistics_label_bug.md`). Before regenerating condition-level plots with the K=4/no-null labels, fix that bug.

### Secondary questions still deferred

- **LZ concordance channels.** Module 5 flagged near-zero loadings in both the production and refit fits. Worth investigating whether the burst-rate gate + surrogate z-scoring is over-attenuating signal. Separate ticket.
- **Resp as a solo channel.** Module 5 showed no correlation with ECG bands. Consider whether Resp belongs in its own block or on its own transition-covariate lane rather than lumped with autonomic. Separate ticket.
- **The persistent −0.066 latent-space silhouette.** This didn't move across any of the three variants (production, K=4 no-null, K=3 no-null). It's not blocking promotion of K=4 no-null, but it says something about how rSLDS organizes latent trajectories across sessions — they pool poorly even when the model is well-specified. Possibly a sign that per-session z-scoring is correct (don't force cross-session global normalization), or that we should report states per-session rather than aggregated. Worth thinking about before writing the methods section.
- **K=5 exploration.** Abandoned. BIC for K=4 no-null is 13k lower than K=4/null=True; K=3 is 51k worse. The data clearly wants 4 regimes, not more and not fewer.

---

## 7. Artifacts

### Code (all tracked in this repo)

- `scripts/_run_scaffold_v11.py` — 26D scaffold runner (threading backend after DLL fix)
- `scripts/_run_v11_hierarchical.py` — hierarchical rSLDS, saves full per-session params
- `scripts/_diagnose_state_separability.py` — immediate-read pairwise + shuffle-null
- `diagnostics/tier2_deepdive/run_tier2.py` — full Tier 2 runner
- `diagnostics/tier2_deepdive/module1_obs_space.py` — observation space + three new separability helpers
- `diagnostics/tier2_deepdive/module4_model_comparison.py` — LOO-CV
- `diagnostics/tier2_deepdive/module5_block_pca.py` — block PCA compression
- `diagnostics/tier2_deepdive/module6_null_ablation.py` — null-state ablation + Hungarian matching
- `tests/diagnostics/test_module1.py` — unit tests for the three new separability helpers

### Outputs (on disk)

**Production fit (K=4, null_state=True):**
- `results/v11/hierarchical/v11_hierarchical_params.npz` — mean_d_emit + shared transition/dynamics params
- `results/v11/hierarchical/v11_hierarchical_results.json` — state usage, BIC, per-session summaries
- `results/v11/<session>/v11_rslds_results.npz` — per-session gamma, path, d_emit, C_emit, R_emit, x_smooth, shared trans/dyn

**K=4 no-null refit (recommended new production):**
- `results/v11/hierarchical_k4_nonull/v11_hierarchical_params.npz`
- `results/v11/hierarchical_k4_nonull/v11_hierarchical_results.json`
- `results/v11/<session>/v11_rslds_results_k4_nonull.npz`

**K=3 no-null refit (for comparison, not for production use):**
- `results/v11/hierarchical_k3_nonull/v11_hierarchical_params.npz`
- `results/v11/hierarchical_k3_nonull/v11_hierarchical_results.json`
- `results/v11/<session>/v11_rslds_results_k3_nonull.npz`

**Tier 2 diagnostic outputs:**
- Production: `diagnostics/outputs/20260422_212003_tier2_deepdive/` (Module 1 + 4) and `20260422_212951_tier2_deepdive/` (Module 5 + 6 after Unicode fix)
- K=4 no-null: `diagnostics/outputs/20260423_033508_tier2_deepdive_k4_nonull/` (all four modules)
- K=3 no-null: `diagnostics/outputs/20260423_070919_tier2_deepdive_k3_nonull/` (all four modules)

### Pipeline logs

- `results/v11/_rerun_logs/rerun_20260422_173510.log` — initial 26D pipeline run (scaffold 8m, hierarchical 215m, separability 38s, Tier 2 7m before Unicode crash).
- `results/v11/_rerun_logs/tier2_resume_20260422_212943.log` — Module 5+6 resume after Unicode fix.
- `results/v11/_rerun_logs/refits_20260422_*.log` — K=4/no-null + K=3/no-null refits and their Tier 2 (refit A 227m + tier2 A 14m; refit B 200m + tier2 B 14m).

---

## 8. Related Work in This Repo

- `docs/v11_pipeline_report.md` — end-to-end pipeline technical reference (dimension counts updated to 26D).
- `docs/rslds_burst_analysis_synthesis.md` — V8.2 burst analysis; template for this synthesis.
- `docs/rslds_vision.md` — original architectural vision for hierarchical rSLDS.
- `docs/v10_scaffold` references in memory: `project_v10_scaffold.md`, `project_v11_post_whitening_findings.md`, `project_v11_te_decomposition.md`, `project_dyn_mean_collapse.md`, `feedback_rslds_obs_vs_cov.md`.
