# CLAUDE.md — CADENCE

## Project Overview

**CADENCE** (Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation) is a fully interpretable, regression-based framework for quantifying directed, time-varying, cross-modal interpersonal coupling from continuous multimodal recordings. Replaces MCCT's transformer with basis-expanded distributed lag regression where every quantity is a measured signal or regression weight.

**Fully standalone** — all needed data pipeline code from MCCT is copied with internal imports. No runtime dependency on MCCT.

## Architecture

**V2 Pipeline**: Two-stage doubly-sparse discovery + time-varying estimation.

1. **Stage 1 — Discovery**: Group lasso with stability selection discovers which source features couple to each target modality. Doubly-sparse: feature sparsity (group lasso) + temporal sparsity (block selection).
2. **Stage 1.5 — Surrogate Screening**: Static ridge surrogates for session-level significance, with BH-FDR correction.
3. **Stage 2 — Estimation**: EWLS forward-backward solve on selected features, with optional moderation (ECG HR/RMSSD moderators) and nonlinear (squared) terms. Produces per-timepoint dR2 timecourse, per-feature dR2 decomposition, and source×target feature-level coupling breakdown.
4. **Stage 2 — Per-Timepoint Significance**: Circular-shift surrogates (K=20) through the same EWLS solver produce per-timepoint p-values. Both real and null dR2 are smoothed identically (30s) before p-value computation.

**Full Pipeline**: Source signals -> wavelet/PCA features -> group lasso discovery -> surrogate screening -> EWLS streaming solve (selected features + moderation + nonlinear) -> dR2 timecourse + per-timepoint p-values -> session-level detection.

## Environment

Uses the MCCT conda environment (Python 3.11, PyTorch, scipy, numpy, matplotlib, pyyaml).

```bash
conda activate MCCT
```

## Key Commands

```bash
# Single session analysis
python scripts/run_session.py --session y_06

# All sessions
python scripts/run_all_sessions.py

# Synthetic validation
python scripts/run_synthetic.py --duration 300

# CADENCE vs MCCT comparison
python scripts/compare_mcct.py --session y_06

# Generate synthetic corpus
python scripts/generate_synthetic.py --n-coupled 50 --n-null 20

# Plot coupling activity (per-timepoint significance)
python scripts/plot_coupling_activity.py --results results/cluster_session/y_06
```

## Project Structure

```
CADENCE/
  cadence/
    __init__.py, config.py, constants.py, synthetic.py, surrogates.py
    data/          # Copied from MCCT: xdf_loader, preprocessors, eeg_features, alignment
    basis/         # raised_cosine.py, design_matrix.py
    regression/    # ewls.py (core), ridge.py, ftest.py, group_lasso.py
    coupling/      # pathways.py, estimator.py (CouplingEstimator), discovery.py, serialization.py
    significance/  # surrogate.py (per-timepoint + session-level), detection.py
    visualization/ # kernels.py, timecourse.py, heatmaps.py, comparison.py, sparsity.py
  configs/default.yaml
  scripts/         # run_session, run_all_sessions, run_synthetic, generate_synthetic, compare_mcct, plot_coupling_activity
  results/
```

## Config

All parameters in `configs/default.yaml`. Key settings:
- `session_cache`: Points to MCCT's session_cache directory
- `ewls.tau_seconds`: 30s exponential decay (time locality)
- `basis.layer1.n_basis`: 8 raised cosine basis functions
- `basis.layer1.max_lag_seconds`: 5.0s maximum lag
- `autoregressive.order`: 3 AR lags
- `significance.surrogate.n_surrogates`: 100 circular shifts (session-level)
- `significance.max_pathway_p`: 0.7 (skip Stage 2 for high session-level p)
- `significance.timepoint.n_surrogates`: 20 (per-timepoint)
- `significance.timepoint.surrogate_eval_rate`: 1.0 Hz (surrogates at lower rate for speed)
- `significance.fdr_correction`: Benjamini-Hochberg
- `interbrain.min_freq_hz`: 4.0 (exclude delta band — artifact-prone on Emotiv EPOC)
- `interbrain.surrogate_method`: fourier_phase (stronger null for autocorrelated features)

## Session Cache

Shares MCCT's `session_cache/` directory. Reads cached `.npz` + `.json` session files.

## Modalities (V2)

- EEG wavelet: 160ch @ 5Hz (2 components × 20 freqs × 4 ROIs)
- EEG interbrain: 120ch @ 5Hz (cross-brain PLV, delta excluded by default)
- ECG features: 7ch @ 2Hz (HRV)
- Blendshapes v2: 31ch @ 30Hz (15 PCA + 15 derivatives + activity)
- Pose features: 41ch @ 12Hz (40 joint groups + activity)

## Key Design Decisions

- EWLS forward-backward with streaming backward pass (1.05x memory vs 3x previously)
- Raised cosine basis with log-spacing (denser at short lags)
- Circular shift surrogates (vectorized gather, preserves all signal statistics)
- Doubly-sparse discovery (group lasso + stability selection + block selection)
- Moderation terms (ECG HR/RMSSD) and nonlinear terms (squared) in Stage 2
- Per-timepoint significance via K=20 EWLS surrogates with matched smoothing
- Source×target feature decomposition: beta energy per (src_feat, tgt_channel) pair
- BH-FDR correction across pathways
- Session-level detection: binomial test + t-test (matches MCCT for comparison)
- Per-modality PCA channels and pathway-specific temporal parameters
- Surrogate-based screening with static ridge for significance
- min_dr2 threshold gates significance (prevents tiny but spurious dR2)

## Validation Status

### Synthetic validation (6/6 tests pass at 600s)
- EEG-only, ECG-only, BL-only, Pose-only, EEG+BL, Null — all pass
- Corpus (5 seeds × 6 categories): TP EEG=100%, ECG=100%, BL=60%, Pose=60%. FP=0%
- Results: `results/cadence_synthetic/`

### Real session (y_06)
- P1→P2: 6/16 sig (EEG-EEG, EEG→BL, BL-BL, BL→EEG, Pose→BL, Pose-Pose)
- P2→P1: 6/16 sig (similar pattern, bidirectional)
- ECG never significant (expected — slow timescale vs 5s max lag)
- Cross-modal pathways are CADENCE's unique contribution (not measurable by MCCT)
- Results: `results/cadence/y_06/`

### CADENCE vs MCCT comparison
- Both agree on modality ranking (BL, Pose strongest; ECG weakest)
- CADENCE dR2 5-10x larger than MCCT CSGI (different baselines)
- Results: `results/cadence_comparison/`
