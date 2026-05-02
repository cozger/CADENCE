# Design: Within-Session Condition Statistics for V11

**Date:** 2026-04-09
**Purpose:** Test whether significant effects between experimental conditions are emerging in the V11 pipeline, using within-session repeated measures.

---

## Contrasts (7 total)

### Family 1 — Intervention vs. baseline (4 contrasts)
- meditate_B vs. base_EC (meditation protocol sessions, n=6-8)
- meditate_K vs. base_EC (meditation protocol sessions, n=6-8)
- PE_1 vs. base_EO (PE protocol sessions, n=5-7)
- PE_2 vs. base_EO (PE protocol sessions, n=5-7)

### Family 2 — Pre vs. post conversation (1 contrast)
- conv_1 vs. conv_2 (all sessions, n=13-15)

### Family 3 — Conversation vs. eyes-open baseline (2 contrasts)
- conv_1 vs. base_EO (all sessions, n=13-15)
- conv_2 vs. base_EO (all sessions, n=13-15)

Each session contributes one paired observation per metric. Sessions missing a condition are excluded from that contrast.

## Metrics (24 total)

### Coupling dynamics (1)
- Coupling flexibility (mean over condition timepoints)

### State usage (4)
- Fraction of time in NULL, COUP, SHARED, OTHER

### Coupling excess z — Tier 1 groups (5)
- EEG Phase, Facial, LZ Shared, Respiratory, Postural

### EEG asymmetry (3)
- asym_theta, asym_alpha, asym_beta (mean over condition)

### TE directed episode fractions (4)
- Therapist→Patient theta, Patient→Therapist theta, T→P alpha, P→T alpha

### Burst coincidence z (3)
- burst_coinc_theta, burst_coinc_alpha, burst_coinc_beta

### Burst rates per modality group (4)
- EEG Phase, EEG Power, Face+Body, LZ

## Statistical Approach

- **Test:** Wilcoxon signed-rank (non-parametric paired, no normality assumption, appropriate for n=5-15)
- **Effect size:** Matched-pairs rank-biserial correlation r
- **Also report:** Mean difference ± SEM, median difference, number of valid pairs
- **FDR:** Benjamini-Hochberg across all 168 tests (7 contrasts × 24 metrics)
- **Both p_uncorrected and q_FDR reported** for every test
- Note: FDR across 168 tests at n=5-15 will be conservative. Uncorrected p-values and effect sizes are the primary exploratory output at this sample size.

## Data Sources

Per-session V11 scaffold outputs provide:
- `scaffold_v11_ztimecourses.npz` — 28D observations, condition segments, obs_mask
- `scaffold_v11_results.json` — condition boundaries, session metadata
- `v11_rslds_results.npz` — state posteriors (gamma), Viterbi path
- Post-hoc results: burst coincidence, directed coupling, coupling excess JSONs
- Windowed graph results for coupling flexibility

Extraction: for each session × condition, compute the mean of each metric over the timepoints belonging to that condition segment.

## Output

### Script
`scripts/run_condition_statistics.py`
- Loads all V11 scaffold NPZs, rSLDS results, and post-hoc analysis JSONs
- Extracts per-session per-condition summary statistics
- Runs all 168 Wilcoxon signed-rank tests
- Applies BH-FDR correction
- Saves all three output formats

### Results JSON
`results/v11/condition_statistics.json`
- Structured: every contrast × metric with p_uncorrected, q_fdr, effect_size_r, mean_a, mean_b, mean_diff, sem_diff, median_diff, n_pairs, direction

### Summary Table
`results/v11/condition_statistics_summary.md`
- Human-readable markdown table sorted by p_uncorrected
- Columns: contrast, metric, n, mean_diff, effect_size_r, p_uncorrected, q_fdr, significance markers (* p<0.05, ** p<0.01, *** p<0.001 uncorrected; + q<0.05 FDR)

### Figures
`results/v11/condition_statistics/`
- One figure per metric (24 figures)
- Each figure shows all 7 contrasts as paired dot plots
- Lines connect same-session pairs
- Colored by significance (uncorrected): gray (ns), blue (p<0.05), red (p<0.01)
- Title shows metric name, subtitle shows overall pattern
