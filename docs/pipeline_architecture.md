# CADENCE V5 Pipeline Architecture

## Overview

CADENCE detects interpersonal coupling from dual-participant recordings through two independent, parallel analysis layers:

1. **BL (Blendshape) Layer** — Behavioral synchrony via facial expression co-occurrence
2. **EEG Layer** — Neural synchrony via multi-band amplitude co-modulation

Each layer has its own detection pipeline, semi-synthetic validation, and detection thresholds. Results are combined at the session level for outcome prediction.

```
Raw XDF → Session Cache (256 Hz EEG, 30 Hz BL, 130 Hz ECG)
              │
              ├── BL Layer ──→ Co-occurrence + causal attribution
              │                  (therapist/patient role-aware)
              │
              └── EEG Layer ─→ Multi-band amplitude coupling
                                 (θ + α + β cycle analysis, GPU)
              │
              └── Session-level integration → outcome prediction
```

---

## BL Layer: Two-Stage Behavioral Coupling

### Architecture
```
Stage 1: Cross-product multi-lag bank
  Input: Raw 52 AUs × 2 participants @ 30 Hz
  → AR(3) residualize → cross-product at all lags (0–5s)
  → Gaussian smooth (3s) → max over lags → surrogate z-score
  → Coupling mask + data-driven lag estimate
  → Hierarchical Bayesian lag shrinkage (prior 2.5±1.0s)

Stage 2: Per-event co-occurrence on RAW AU composites
  Composites: smile (AU43+44+17), brow (AU2+3+4), frown, speech, general
  → Prominence-based event detection (≥0.3 raw)
  → Co-occurrence matching within ±lag_window
  → Causal attribution: mimicry / shared_stimulus / coincidence
  → Session-level p-value via temporal circulant null
```

### Detection Thresholds
- Semi-synthetic: κ≈0.10 (18 heterogeneous AUs, 46.5% hit @ FA 5%)
- Real data (y_06): brow T→P p=0.04, most co-occurrences are shared_stimulus

### Key Files
- `cadence/significance/bl_coupling.py` — Production pipeline
- `cadence/synthetic.py` → `inject_bl_event_coupling()` — Semi-synthetic
- `scripts/test_bl_pipeline.py` — Validation suite

---

## EEG Layer: Multi-Band Amplitude Coupling

### Architecture
```
Input: Raw 14-ch EPOC EEG × 2 participants @ 256 Hz
  → Average re-reference → z-normalize per channel

Per band (θ 4-8 Hz, α 8-13 Hz, β 13-30 Hz):
  → GPU FFT bandpass (all channels simultaneously)
  → Per-channel cycle extraction (trough detection → peak finding)
  → Per-cycle features: volt_amp, period, rise-decay symmetry, burst status
  → Resample to 2 Hz regular grid
  → GPU cross-correlation (therapist vs patient) + 200 surrogates
  → Per-channel z-score → Stouffer pooling across channels

Cross-band combination:
  → Stouffer across θ + α + β → combined z-score
```

### Key Finding
**Amplitude co-modulation is the dominant inter-brain EEG signal** — 2× stronger than phase coupling (PLV). The signal reflects shared arousal dynamics, not phase locking.

### Detection Profile (y_06 real data)

| Condition | θ | α | β | Combined |
|-----------|---|---|---|----------|
| conv_2 | +7.2 | +7.5 | +10.0 | **+14.3** |
| meditate_K | +6.6 | +3.9 | +2.9 | **+7.7** |
| conv_1 | +3.1 | +3.5 | +1.9 | +4.9 |
| base_EO | +3.6 | −0.3 | −1.4 | +1.1 |
| meditate_B | +0.7 | −2.0 | −0.7 | −1.1 |

Band profile is condition-specific: β dominates conversation (speech motor), θ dominates meditation (emotional attunement).

### Validity Evidence
- **Spatial specificity**: Frontal 2× occipital, right hemisphere dominant
- **Pseudo-dyad null**: z≈0 (no cross-session artifact)
- **Avg-ref independent**: Signal persists without re-referencing
- **Detrending invariant**: 3rd-order polynomial detrend has zero effect
- **Asymmetric alpha explained**: Coupling requires BOTH participants to have power fluctuations

### Semi-Synthetic Model
`inject_eeg_coupling_arousal()`: Slow modulator (LP 0.5 Hz) from therapist's global theta envelope → lags → modulates patient's broadband amplitude. Frontal-weighted spatial pattern.
- Selectivity validated: volt_amp detects, period/symmetry/PLV do not
- Detection threshold: κ≈0.20 (z>2)

### Key Files
- `cadence/significance/fast_cycles.py` — GPU cycle analysis (production)
- `cadence/synthetic.py` → `inject_eeg_coupling_arousal()` — Semi-synthetic
- `scripts/test_eeg_pipeline.py` — Validation suite

---

## What Doesn't Work (Tested and Rejected)

| Approach | Why it fails |
|----------|-------------|
| Multi-resolution PLV | Max-over-scales penalty = z gain (neutral) |
| CCorr metric | 15 pp worse than PLV; removes signal with circular mean |
| Burst-triggered ITC | √N loss: 10 events vs 5120 samples per window |
| IAAFT surrogates | OOM at 460K samples; circular shift already optimal |
| Envelope correlation (mixing model) | κ² sensitivity vs PLV's κ |
| Broadband mixing semi-synthetic | 2× too optimistic; unrealistic for θ coupling |
| CaCoh/CCA | Overfits when channels ≈ DOF |
| HMM on z-scores | 40 pp worse than direct threshold |

---

## Running the Test Suites

```bash
conda activate MCCT

# EEG pipeline (smoke test ~30s, full ~2min)
python scripts/test_eeg_pipeline.py --quick
python scripts/test_eeg_pipeline.py

# BL pipeline (smoke test ~5s, full ~30s)
python scripts/test_bl_pipeline.py --quick
python scripts/test_bl_pipeline.py

# Specific session
python scripts/test_eeg_pipeline.py --session y_32
python scripts/test_bl_pipeline.py --session y_32
```

Both suites print PASS/FAIL summary and exit with code 0 (pass) or 1 (fail).
