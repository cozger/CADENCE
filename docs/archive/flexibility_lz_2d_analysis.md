# Flexibility × LZ Concordance: 2D Coupling Regime Analysis

## Overview

Coupling flexibility and LZ concordance are complementary dimensions of interpersonal coupling:

- **Flexibility** measures *how modalities relate to each other* (rigid vs independent). Computed as the graph spectral energy ratio of the 18D modality correlation graph (90s windows). Range [0,1]: 0 = all modalities lock together, 1 = each modality independent.

- **LZ concordance** measures *how similar two people's neural dynamics are*. Computed as Lempel-Ziv complexity on the Hilbert amplitude envelope of frontal EEG (theta/alpha), then concordance = (z_therapist + z_patient) / 2. Coupling excess = real concordance minus circular-shift surrogate null. Positive = shared complexity, negative = divergent complexity.

You can have high LZ concordance (brains similarly complex) with low flexibility (all modalities locked together) — that's early meditation. Or high LZ concordance with high flexibility (brains match but modalities couple independently) — that's conversation.

## The Four Quadrants

| Quadrant | Flexibility | LZ Concordance | Conditions | Interpretation |
|----------|------------|----------------|------------|----------------|
| **Flexible + Shared Complexity** | HIGH (0.39+) | POSITIVE (+0.54) | **conv_1** | Active conversation: modalities couple independently, but brains are in similar complex states |
| **Rigid + Shared Complexity** | LOW (0.33) | POSITIVE (+0.30) | **meditate_B** | Early meditation: everything locks together AND brains share similar (calm) complexity |
| **Rigid + Divergent Complexity** | LOW (0.25-0.27) | NEGATIVE (-0.44 to -1.40) | **meditate_K, conv_2** | Deep meditation / post-meditation: rigid coupling regime, but brains are in *different* complexity states |
| **Mixed** | HIGH (0.39-0.42) | ~ZERO | **PE_1, PE_2, base_EO** | Psychoeducation and baseline: flexible but no complexity coupling |

## Per-Condition Values (n=12 sessions)

| Condition | Flexibility | LZ Excess | Quadrant |
|-----------|------------|-----------|----------|
| base_EO | +0.394 | +0.074 | MIXED |
| base_EC | +0.417 | +0.361 | Flexible + Shared Complexity |
| conv_1 | +0.392 | +0.543 | Flexible + Shared Complexity |
| conv_2 | +0.249 | -1.398 | Rigid + Divergent Complexity |
| meditate_B | +0.327 | +0.302 | Rigid + Shared Complexity |
| meditate_K | +0.269 | -0.440 | Rigid + Divergent Complexity |
| PE_1 | +0.388 | +0.017 | MIXED |
| PE_2 | +0.416 | -0.089 | MIXED |

## Session Trajectory

The trajectory through the flexibility × LZ space tells the meditation story:

```
base_EO (0.39, +0.07)  →  base_EC (0.42, +0.36)  →  conv_1 (0.39, +0.54)
     [neutral]              [eyes closed:              [conversation:
                              shared calm =              flexible coupling +
                              shared complexity]          shared complexity]

                                    ↓

                            meditate_B (0.33, +0.30)  →  meditate_K (0.27, -0.44)
                                 [early meditation:        [deep meditation:
                                  rigidifying but            rigid + brains diverge
                                  still shared]              (patient inward)]

                                                                ↓

                                                        conv_2 (0.25, -1.40)
                                                           [POST-meditation:
                                                            still rigid from meditation
                                                            + brains maximally divergent]
```

Meditation first rigidifies coupling while maintaining shared complexity (meditate_B), then drives complexity divergence while maintaining rigidity (meditate_K). The post-meditation conversation (conv_2) is maximally rigid AND maximally divergent — the meditation effect is still unresolved.

PE conditions stay in the MIXED zone (high flexibility, near-zero LZ) throughout. Psychoeducation doesn't change coupling architecture at all — the expected control result.

## Cross-Condition Correlations

Spearman correlations across all 60 session × condition points:

| Pair | rho | p |
|------|-----|---|
| **Flexibility vs Facial coupling excess** | **+0.393** | **0.002** |
| Flexibility vs LZ excess | +0.204 | 0.118 |
| Flexibility vs Postural excess | +0.131 | 0.317 |
| Flexibility vs EEG Phase excess | +0.058 | 0.660 |

Facial synchrony specifically requires flexible coupling (rho=+0.39, p=0.002). When the system locks into a rigid regime, face-to-face coupling is suppressed. Mechanistically: facial mimicry requires rapid, modality-specific coupling, not global synchronization.

## Coupling Excess by Condition Type

| Condition type | Flexibility | LZ excess | EEG Phase | Facial | Postural | n |
|---------------|------------|-----------|-----------|--------|----------|---|
| baseline | +0.406 | +0.217 | +0.216 | +0.50* | +0.087 | 20 |
| conversation | +0.313 | -0.524 | -0.065 | +0.410 | +0.224 | 20 |
| meditation | +0.298 | -0.069 | -0.179 | +0.254 | -0.352 | 12 |
| PE | +0.402 | -0.036 | -0.277 | +0.601 | -0.170 | 8 |

*Facial excess during baseline may include numerical artifacts from single-session outliers.

## Key Findings

1. **Meditation creates a lasting coupling regime change** visible only in the 2D space: rigidification (flexibility drops 0.39→0.25) + complexity divergence (LZ excess drops +0.54→-1.40). Neither dimension alone tells the full story.

2. **PE has no effect on coupling architecture**: flexibility and LZ stay in the MIXED zone throughout. This is the correct control result.

3. **Facial coupling requires flexibility**: the significant correlation (rho=+0.39, p=0.002) suggests rigid coupling regimes suppress face-to-face synchrony. This is consistent with the burst analysis finding that meditation suppresses Face+Body coupling bursts.

4. **The post-meditation conversation (conv_2) is the most extreme point**: maximally rigid (0.25) AND maximally divergent (-1.40). The dyad has not yet "recovered" from the meditation-induced regime change during the post-meditation conversation.

## Methods

- **Flexibility**: Coupling flexibility index from `cadence/significance/spectral_graph.py`. Graph spectral energy ratio on windowed (90s, 15s stride) correlation graph of 18D V8.2 base features. Interpolated to 2 Hz as V10 transition covariate.
- **LZ concordance excess**: From `cadence/significance/coupling_bursts.py`. 200 circular-shift surrogates per session. Excess z-score of LZ Shared group (channels 18-19: lz_conc_theta, lz_conc_alpha).
- **Data**: 12 sessions (6 meditation protocol, 5 PE protocol, 1 excluded for unknown protocol). V10 scaffold (23D observations + 5D covariates). Hierarchical rSLDS (K=4, D_latent=3).

---

*Generated 2026-03-31. CADENCE V10 pipeline.*
