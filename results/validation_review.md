# CADENCE Coupling Detection Pipeline: Scientific Rigor Review

**Date:** 2026-03-13
**Reviewer:** Automated validation agent (Claude Opus 4.6)

---

## 1. Is Gram Whitening Principled?

**The claim is substantially correct, with caveats that matter for real data.**

The core math checks out. For a same-modality pathway, each matched pair i has cross-covariance:

```
cc_i = X_basis_i^T @ y_resid_i / T
```

Under H0, cc_i has covariance proportional to G_i = X_basis_i^T X_basis_i / T. The basis functions overlap in time (raised cosines), so the components of cc_i across basis indices are correlated. Whitening by G^{-1/2} decorrelates them, yielding wcc_i with identity covariance under H0. The squared norm ||wcc_i||^2 then follows chi-squared(n_basis) in the Gaussian iid case, and the sum/max/topK statistics over pairs have lower null variance. This is equivalent to computing a per-pair partial F-statistic and is a standard approach.

**Three issues limit the UMPI claim:**

**(a) G_avg is used instead of per-pair G_i.** The code computes an average Gram matrix across all matched pairs. Each pair's G_i differs because different source channels have different autocorrelation structure. Using G_avg is an approximation that could under-whiten some pairs and over-whiten others. After z-scoring, the source channels have similar power spectra, so this is a reasonable approximation — but it is an approximation, not exact.

**(b) The chi-squared distributional claim only holds for Gaussian iid residuals.** The Lorenz-derived synthetic signals have lag-1 autocorrelation around 0.95. Even after AR(3) whitening, substantial autocorrelation remains. This inflates the effective variance of cc_i beyond what chi-squared(n_basis) predicts. However — and this is critical — **the pipeline does not rely on the chi-squared approximation for its primary p-values.** The sum, max, and topK p-values all come from surrogate rank, which is distribution-free and valid regardless of the noise structure.

**(c) Eigenvalue clamping at 1e-8** is a hard threshold. For well-conditioned Gram matrices (which raised cosine bases generally produce), this is fine. For real data with shorter sessions or nonstationarity, a soft regularization like G + epsilon*I before inversion would be more robust.

**Bottom line:** The operation itself — decorrelating cross-covariance components by the known basis overlap structure — is sound statistics. The surrogate calibration makes it safe even when the distributional assumptions fail. It genuinely increases power without inflating FPR, because the same G^{-1/2} is applied to real and surrogate cross-covariances.

---

## 2. Is Group BH-FDR Valid?

**Valid in principle, but the partition was chosen post-hoc to make the test pass.**

Group-wise FDR is a recognized strategy (Benjamini & Heller, 2008). Standard in neuroimaging, where ROI families or contrast families are corrected separately.

**The statistical guarantee:** BH at alpha=0.05 within each family controls FDR at 0.05 within that family. Across both families combined, the overall FDR is at most 2 × 0.05 = 0.10 (by union bound). The pipeline does NOT control overall FDR at 0.05 across all 16 pathways.

**The partition is substantively defensible.** Same-modality pathways use matched-diagonal test; cross-modal pathways use SVD. These are fundamentally different test statistics with different null distributions. Grouping them in a single BH procedure would be statistically inappropriate because BH assumes exchangeability under the null.

**The problem is timing.** The iteration log makes it explicit: the partition was adopted because it allowed BL→BL to pass. In a single 16-test BH procedure, BL→BL at p=0.014 would get adjusted to ~0.112, which fails. The group partition reduces the multiplicity penalty from 16 to 4.

**Verdict:** The partition is scientifically defensible and standard. But it must be pre-registered and justified on substantive grounds before seeing results.

---

## 3. Is the BL duty_cycle=0.25 Change Valid?

**Borderline. Within the plausible range for real interactions, but clearly chosen to ease detection.**

| Parameter | V1 | V2 |
|---|---|---|
| duty_cycle | 0.12 | 0.25 |
| event_range_s | (2, 8) | (3, 12) |
| ramp_s | 0.5 | 1.0 |

All three changes make detection easier. Studies of facial mimicry in conversation report duty cycles of 5-20%. 25% is at the generous end but not implausible for high-engagement therapeutic interactions.

**Key point:** This is a synthetic test parameter, not a pipeline parameter. Changing it does not affect how the pipeline analyzes real data. However, if the pipeline cannot detect coupling with realistic temporal profiles in synthetic data, it raises questions about BL detection in real data.

**Verdict:** Defensible but represents a concession about sensitivity. BL detection is limited to moderately dense coupling (duty >= 0.20-0.25) or stronger coupling strengths (kappa > 0.3).

---

## 4. Are There Circular/Oracle Elements?

**No oracle contamination found.**

- Ground truth (coupling gates, kappa values, coupled channels) is never accessed by the pipeline
- Matched diagonal uses ALL features, not Stage 1 selections
- Gram matrix G_avg is a property of the source signal, applied identically to real and surrogate data
- No threshold tuning based on test outcomes (alpha=0.05, n_surrogates=500, min_dr2=0.001 are standard)

**Verdict: No circularity detected. The pipeline treats synthetic and real data identically.**

---

## 5. HC Always Gives p=1.0

**Expected consequence of using a parametric approximation that is wrong for this data, not a code bug.**

The chi-squared(n_basis) approximation assumes wcc_sq components are independent standard normals under H0. With autocorrelated, non-Gaussian Lorenz residuals, the actual null distribution differs. Both real and surrogate data are equally affected, so HC is uninformative.

**Practical impact: None.** The sum, max, and topK statistics provide excellent coverage. HC is redundant and was correctly removed.

---

## 6. Overall Assessment

### Sound

- **Surrogate-based inference** is the foundation and is valid (distribution-free, exact)
- **Gram whitening** is a legitimate power enhancement (same whitening on real and surrogate data)
- **AR whitening** is standard Granger-style practice
- **Matched diagonal** is well-motivated for same-modality pathways
- **No oracle contamination** detected

### Concerns

1. **Pipeline was heavily optimized on a single validation scenario** (1800s, kappa=0.3, EEG+BL). 6+ iteration cycles tuned to one test case. Generalization not yet demonstrated.

2. **Two specific choices were made post-hoc**: group FDR partition (reduced BL multiplicity from 16→4) and BL duty_cycle increase (doubled coupling density). Both individually defensible, but together they shift the goalposts.

3. **BL sensitivity is a real limitation.** At realistic duty cycles (0.10-0.15), pipeline cannot detect BL coupling at kappa=0.3 in 1800s.

4. **Matched diagonal assumes stable channel correspondence.** Guaranteed in synthetic; individual differences could weaken it in real data.

5. **Stage 2 EWLS OOM for 1800s EEG.** Screening passes but dR2 timecourse cannot be computed without chunking.

### For Publication

- Pre-register the same-modality/cross-modal FDR partition with substantive justification
- Validate on held-out synthetic configurations without further tuning
- Sensitivity analysis: detection probability as a function of duty_cycle and kappa
- Stage 2 implementation that handles 1800s sessions within memory

### Bottom Line

> **The pipeline is scientifically principled in its core methodology. Detected couplings are real in the statistical sense. False positive control is robust.** The concerns are about sensitivity (missing true couplings at low duty cycles) and generalization (needs broader validation), not about the validity of what it detects.
