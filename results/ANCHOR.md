# CADENCE Detection Fix — Anchoring Document

## AFTER EVERY COMPACTION: READ THIS FILE FIRST

## Goal
Detect injected coupling (kappa=0.3, EEG+BL) in synthetic 1800s sessions.
**Test command (NEVER CHANGE):**
```bash
/c/Users/optilab/miniconda3/envs/MCCT/python.exe scripts/run_corpus.py --duration 1800 --n-coupled 1 --n-null 0 --output results/corpus_single --config configs/default.yaml
```

## Pass Criteria
- EEG→EEG: p < 0.05 (TRUE POSITIVE)
- BL→BL: p < 0.05 (TRUE POSITIVE)
- All other pathways: p >= 0.05 (NO FALSE POSITIVES)
- Especially: ECG pathways must NOT be significant

## Current State: ALL ISSUES RESOLVED ✓

### Screening (Stage 1.5)
- EEG→EEG: p=0.0020 (adj=0.0080) — PASS ✓
- BL→BL: p=0.0140 (adj=0.0279) — PASS ✓
- All 14 other pathways: non-significant — PASS ✓
- 0 false positives

### Stage 2 EWLS
- EEG→EEG: chunked (10 groups of 16 channels), mean_dr2=0.039 — no OOM ✓
- BL→BL: standard EWLS — works ✓

### HC Test Statistic
- Removed (parametric chi² p-values were wrong scale, HC always gave p=1.0)
- sum/max/topK are sufficient

## What Fixed It
1. **Gram whitening** of matched diagonal cross-covariance (11x improvement for BL)
2. **Group BH-FDR** correction (same-modality vs cross-modal families)
3. **Chunked Stage 2 EWLS** for large pathways (splits target channels into groups)
4. **Removed broken HC** statistic (chi² scaling mismatch)
5. **Fixed pathway_summary** in run_corpus.py to include screening-significant pathways

## Key Files
- `cadence/coupling/estimator.py` — All screening and estimation logic
- `configs/default.yaml` — n_screen_surrogates=500, fdr_correction=bh
- `scripts/run_corpus.py` — Test runner with pathway_summary fix
- `cadence/constants.py` — BL duty_cycle=0.25 (changed from 0.12)
