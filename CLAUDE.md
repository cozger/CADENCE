# CLAUDE.md — CADENCE

## Project Overview

**CADENCE** (Continuous Analysis of Dyadic Exchange via Native-rate Coupling Estimation) is a fully interpretable, regression-based framework for quantifying directed, time-varying, cross-modal interpersonal coupling from continuous multimodal recordings. Replaces MCCT's transformer with basis-expanded distributed lag regression where every quantity is a measured signal or regression weight.

**Fully standalone** — all needed data pipeline code from MCCT is copied with internal imports. No runtime dependency on MCCT.

## Architecture

**V7 Pipeline** (production): CWT wavelet decomposition + wavelet coherence.

1. **EEG**: `fast_cycles.py` — GPU multi-band (theta/alpha/beta) cycle analysis. Extracts per-cycle volt_amp, cross-correlates between participants with 200 surrogates. Combined Stouffer z across bands. Primary metric: amplitude co-modulation (z=+10 on real data).
2. **BL**: `bl_wavelet.py` — CWT decomposition of all 52 AU timeseries into frequency bands (state <0.5 Hz, expression 0.5-2 Hz, speech 2-7 Hz). Wavelet coherence between participants gives multi-scale coupling profile. GPU-accelerated surrogate z-scoring (200 surrogates in 0.6s). Low-pass filter at 8 Hz removes tracker noise. Semisynthetic validated: AUC=0.78 at kappa=0.4 with realistic smile injection (d=0.97).
3. **Output**: Per-segment coherence spectrograms + band-averaged coupling z-scores + continuous band-power timecourses.

**V6 Pipeline** (legacy, kept for reference): Saliency-based facial event detection via `bl_coupling.py`. Affect-only AU saliency + phasic smile scoring + speech/blink gating. Replaced by V7 wavelet approach.

**V2 Pipeline** (legacy, kept for reference): EWLS regression on z-scored PCA features via `CouplingEstimator`. Not used in production.

## Environment

Uses the MCCT conda environment (Python 3.11, PyTorch, scipy, numpy, matplotlib, pyyaml).

```bash
conda activate MCCT
```

## Key Commands

```bash
# V7 Production pipeline (wavelet)
python scripts/_run_v7_timeline.py                                  # Full session timeline (EEG + BL wavelet)
python scripts/_run_v7_individual_scalograms.py                     # Per-participant CWT scalograms

# V7 Validation
python scripts/_test_wavelet_validation.py                          # Ground truth checks (speech, coherence, null)
python scripts/_test_wavelet_semisynthetic_v3.py                    # Semisynthetic AUC (real smile injection, 42 pairs)

# V6 session runner (still used for EEG + BL segment extraction)
python scripts/run_session_v6.py --session y_06                    # Single session (EEG + BL)
python scripts/run_session_v6.py --session y_06 --bl-only          # BL only
python scripts/run_all_sessions_v6.py                               # All sessions

# EEG validation
python scripts/test_eeg_pipeline.py --quick     # EEG: fast_cycles multi-band (~30s)

# Legacy V2 pipeline (kept for reference)
python scripts/run_session.py --session y_06    # V2 CouplingEstimator
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
    significance/  # fast_cycles.py (GPU EEG), bl_wavelet.py (V7 BL wavelet production), bl_coupling.py (V6 legacy), coherence_localization.py (PLV), surrogate.py, detection.py
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

## Reference Papers

`G:\My Drive\ARPA Shared Documents\Reference Papers` — shared Google Drive folder with all project literature PDFs.

## Modalities (V2)

- EEG wavelet: 160ch @ 10Hz (2 components × 20 freqs × 4 ROIs)
- EEG interbrain: 120ch @ 5Hz (cross-brain PLV, delta excluded by default)
- ECG features: 7ch @ 2Hz (HRV)
- Blendshapes v2: 31ch @ 30Hz (15 PCA + 15 derivatives + activity)
- Pose features: 41ch @ 12Hz (40 joint groups + activity)

## Literature-Informed Priors (from ~190 papers, 2026-03-25)

For full details invoke `/literature`. Key priors that should influence all CADENCE development:

- **EDA/SC is the strongest therapy synchrony signal** (r=0.32-0.47) — not yet captured. #1 hardware addition.
- **Sympathetic (SNS) and parasympathetic (PNS) synchrony have OPPOSITE relational valence** (SNS: ES=+0.19, PNS: ES=-0.21). RMSSD is PNS — its synchrony is negatively associated with outcomes. This explains weak ECG pathways.
- **Behavioral synchrony Granger-causes neural synchrony** (Koul 2023) — face/pose are leading indicators of EEG coupling.
- **Vocal pitch synchrony is meta-analytically harmful** (r=-0.20). Prefer linguistic/semantic synchrony if adding speech.
- **Coupling flexibility > aggregate synchrony** (Gordon 2025). Compute entropy of dR2, DFA exponent, state transition counts.
- **No EEG hyperscanning during psychedelic sessions exists** — MAP-Neuro is first.
- **LZ complexity** is trivially real-time (<1ms/channel), beats alpha for psychedelic state, and has never been used as a coupling moderator.
- **Respiratory rate extractable from Polar H10** (FMRR, <2 bpm error) — no new hardware needed.
- **ECA > ES** for event-based coupling (ES confounds synchrony with serial dependency).
- **Hawkes + basis functions + group sparsity** (Xu 2016) = CADENCE architecture for point processes.

## Key Design Decisions

### V7 BL Wavelet Pipeline
- CWT (FFT-based Morlet, w=5) on all 52 AUs at 30 log-spaced frequencies 0.3-8 Hz
- GPU-accelerated: torch.fft for CWT, GPU conv1d for coherence smoothing
- Low-pass filter at 8 Hz (Butterworth 4th order) removes tracker noise before CWT
- Three frequency bands: state (<0.5 Hz), expression (0.5-2 Hz), speech (2-7 Hz)
- Wavelet coherence per AU group with Gaussian temporal smoothing (0.5s)
- Surrogate z-scoring: 200 circular shifts of P2 CWT coefficients, Welford accumulation on GPU (0.6s total)
- Literature basis: Jeganathan 2022 (eLife), Fujiwara 2016/2018/2020, Hale 2019, Likens 2021
- Semisynthetic validation: AUC=0.78 at kappa=0.4 (d=0.97) with real smile waveform injection, 42 pseudo-dyad pairs

### V2/V6 Legacy
- EWLS forward-backward with streaming backward pass (1.05x memory vs 3x previously)
- Raised cosine basis with log-spacing (denser at short lags)
- Circular shift surrogates (vectorized gather, preserves all signal statistics)
- Per-modality PCA channels and pathway-specific temporal parameters

## Critical Testing Rule

**Semi-synthetic tests MUST use pseudo-dyad (cross-session) as the base signal.** Real dyad data already has coupling — injecting on top of it means κ=0 is not null. Always use P1 from session A + P2 from session B. This guarantees κ=0 produces AUC≈0.50.

## Validation Status

### V7 BL Wavelet validation (y_06)
- **Speech detection**: Therapist speech in meditation correctly detected; patient silence = 0% (null)
- **Expression events**: 80-97% reduction vs V6 saliency; zero events in baselines (perfect null)
- **Coherence hierarchy**: conv_2 (0.354) > conv_1 (0.227) > meditate (0.178) > baseline (0.134)
- **Shared smile coherence**: 1.60x higher during shared smiles vs baseline (conv_1)
- **Pseudo-pair control**: Real pair > pseudo pair (1.19x, consistent with literature)
- **Semisynthetic AUC**: 42 pseudo-dyad pairs, real smile waveform injection
  - kappa=0.2: AUC=0.69, d=0.62
  - kappa=0.3: AUC=0.75, d=0.84
  - kappa=0.4: AUC=0.78, d=0.97
- **Frequency bands empirically grounded**: state <0.5 Hz, expression 0.5-2 Hz, speech 2-7 Hz, noise >8 Hz

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
