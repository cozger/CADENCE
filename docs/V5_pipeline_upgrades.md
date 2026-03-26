# CADENCE V5: Event-Anchored Multimodal Synchrony

## Status: Active Development (2026-03-25)

## Architecture

Two-stage BL coupling pipeline producing event catalogs that serve as anchors for cross-modal analysis and session-level outcome prediction.

```
Stage 1: Cross-product multi-lag bank (z-scored AUs)
  → continuous coupling mask + data-driven lag estimate
  → hierarchical Bayesian shrinkage (prior 2.5±1.0s)

Stage 2: Per-event co-occurrence detection (RAW AU composites)
  → individual expression catalogs per person
  → co-occurrence detection with who-led analysis
  → causal attribution via onset analysis (mimicry / shared_stimulus / coincidence)
  → LSL timestamps for cross-modal anchoring
```

## Two Use Cases

1. **Session-level outcome prediction**: synchrony metrics → predict therapy outcomes
2. **Cross-modal temporal coupling**: BL co-occurrence events as anchors → query EEG/ECG/Pose at those moments (event-triggered synchrony analysis)

## Implementation Status

### Completed

| Component | Status | Key Finding |
|-----------|--------|-------------|
| **Two-stage BL pipeline** | ✅ Production | `cadence/significance/bl_coupling.py` |
| **Raw AU composites** | ✅ Fixed | Z-scored composites detect noise, not smiles. Raw AU values with prominence ≥ 0.3 detect real visible expressions |
| **Multi-composite support** | ✅ | smile (AU43+44+17), brow (AU2+3+4), frown (AU25+26), speech (AU17+22+23), general (all AUs) |
| **Hierarchical lag shrinkage** | ✅ | Normal-Normal conjugate: session estimate shrunk toward population prior (2.5±1.0s). Data-driven performs best (Fisher p=0.044 for T→P on z-scored — needs revalidation with raw) |
| **Iterative lag refinement** | ✅ | Pass 1 wide window → Pass 2 narrow from matched lags → Pass 3 re-test |
| **Causal attribution** | ✅ | Onset analysis: follower's pre-slope and baseline determine mimicry vs shared_stimulus vs coincidence |
| **Co-occurrence framing** | ✅ | Symmetric detection (not source→target), who-led analysis, LSL timestamps |
| **Role-aware corpus analysis** | ✅ | Therapist/patient from XDF stream metadata (Sway = therapist) |
| **IAAFT surrogates** | ✅ Implemented, ❌ not useful for BL | Zero-inflated AU distributions break IAAFT convergence. Circular shift is already optimal for AUs. EEG untested at scale. |
| **Event synchronization (Quian Quiroga)** | ✅ Implemented, ❌ not viable for dense AUs | Random coincidences dominate at AU event rates (~0.5/s/ch). Works only for very sparse events. Superseded by co-occurrence approach. |
| **Event-mimicry coupling model** | ✅ | `inject_bl_event_coupling()` in synthetic.py for semi-synthetic testing |

### Pending

| Component | Status | Notes |
|-----------|--------|-------|
| **Corpus rerun with raw composites** | 🔲 | Previous corpus significance (Fisher p=0.044) was on z-scored composites (inflated). Needs rerun with raw AU values + debug zero-event issue in corpus test |
| **BOCPD regime detection** | 🔲 | Replace threshold + min-event-filter with Bayesian change-point detection on z-score timecourses |
| **Multi-resolution PLV for EEG** | 🔲 | Compute PLV at 2s/5s/10s/20s windows, max-over-scales. Could improve EEG detection at κ=0.1 |
| **Cross-modal event-triggered analysis** | 🔲 | Use BL co-occurrence timestamps to query EEG PLV, ECG coherence at those moments |
| **Session-level summary metrics** | 🔲 | Clean per-session output table for outcome prediction modeling |
| **Pseudo-dyad null** | 🔲 | Pair person A from session X with person B from session Y for stronger null that controls for shared conversational context |

## Key Findings

### Real Data (y_06)

| Metric | conv_1 (397s) | conv_2 (308s) |
|--------|--------------|--------------|
| Patient smiles | 29 | 27 |
| Therapist smiles | 10 | 23 |
| Co-occurrences | 5 (p=0.070) | 9 (p=0.595) |
| Patient led | 3 | 7 |
| Therapist led | 2 | 2 |
| Mimicry events | 1 (conf=0.93) | 0 |
| Shared stimulus | 4 | 5 |

### Real Mimicry Characteristics (from y_06 conv_2)
- Major smile events (prominence ≥ 0.3 raw): every ~26 seconds
- Smile composite = AU43 (mouthSmileL) + AU44 (mouthSmileR) + AU17 (jawOpen)
- Response lag: 2.9s mean, 3.4s median (when co-occurring)
- Most co-occurrences are shared_stimulus — both responding to conversation content
- Genuine causal mimicry is rare but identifiable via onset analysis

### Literature Synthesis (4 parallel searches)
- **No existing method combines TL + directionality + cross-modal + interpretability** — CADENCE is novel
- **Behavioral sync Granger-causes neural sync** (Koul et al. 2023 NeuroImage) — BL→EEG dominant direction
- **Cross-modal correlations modest** (r=0.18-0.32, Ohayon & Gordon 2025 meta-analysis)
- **Field moving toward information theory** (Chidichimo et al. 2025 Nat Rev Neurosci)
- **Event synchronization** not viable for dense signals; **IAAFT** not better than circular shift for zero-inflated AUs
- **STOK adaptive Kalman**, **Robust BOCPD**, **Multi-resolution PLV** still promising for EEG

### Methods Evaluated but Not Selected
| Method | Why Not |
|--------|---------|
| Transfer Entropy (IDTxl) | Too expensive for production. Validation tool only. |
| Neural Granger (GC-xLSTM) | Loses interpretability |
| CRQA/MdRQA | O(N²), no directionality |
| Hawkes Processes | Event sync captures same signal more simply |
| Deep learning (TACI, transformers) | Black box, needs training data |
| MEMD | Significant infrastructure for uncertain benefit |

## Files

| File | Purpose |
|------|---------|
| `cadence/significance/bl_coupling.py` | Production two-stage pipeline |
| `cadence/surrogates.py` | IAAFT + circular shift + Fourier surrogates |
| `cadence/synthetic.py` | Event-mimicry coupling injection model |
| `cadence/data/xdf_loader.py` | Role detection (therapist/patient) |
| `scripts/_test_bl_corpus.py` | Corpus-level analysis across all sessions |
| `scripts/_test_bl_event_catalog.py` | LSL timestamp event catalog for video verification |
| `scripts/_test_bl_two_stage.py` | Single-session pipeline test |

## Key Design Decisions

1. **Raw AU values, not z-scored**: Z-scoring inflates tiny fluctuations into "events". Prominence ≥ 0.3 in raw composite ensures visible expressions.
2. **Co-occurrence framing, not source→target**: Most synchrony is shared_stimulus (both responding to conversation). Symmetric detection with who-led is more honest than forcing a causal direction.
3. **Hierarchical lag shrinkage**: Population prior (2.5±1.0s) regularizes sessions with poor lag estimates without overriding sessions with strong data.
4. **Causal attribution as bonus**: Onset analysis distinguishes mimicry from shared_stimulus from coincidence, but ALL co-occurrences are interesting for outcome prediction.
5. **LSL timestamps for cross-modal**: Every co-occurrence has precise timestamps for querying EEG/ECG/Pose at that moment.
