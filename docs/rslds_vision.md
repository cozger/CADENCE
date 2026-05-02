## Design Decision: Multi-Population RSLDS for Cross-Modal Integration

**Architecture**: Three-population RSLDS on full 50-min sessions at 2 Hz (6000 timepoints):
- Population 1: EEG volt_amp z-timecourse (from fast_cycles.py)
- Population 2: BL expression-band coherence z (from bl_wavelet.py, 0.5-2 Hz)
- Population 3: BL state-band coherence z (from bl_wavelet.py, <0.5 Hz)

Shared discrete state z(t) ∈ {Disengaged, Emotional_alignment, Active_exchange, Deep_engagement} (K=3-4).
Cross-population B matrices capture directed cross-modal coupling per regime.
Condition labels (conv_1, meditate_K, etc.) enter as exogenous covariates modulating transition probabilities.
Drug-state variables (LZc, alpha power) will enter the same way for dosing sessions.

**Why full session, not per-segment**: Coupling doesn't reset at condition boundaries. Transitions between conditions are the most informative data. 6000 timepoints (full session) vs 600 (per-segment) eliminates all identifiability concerns. Condition effects become testable S coefficients.

**Why RSLDS now (after V3.3 SLDS failed)**: Previous attempts ran on raw dR2 (0.035 nats SNR). V6/V7 z-timecourses have 70× stronger signal (0.5-2+ nats). RSLDS characterizes dynamics of already-detected signal, not detects coupling from noise.

**Novelty**: Literature search confirms NO published work applying multi-pop RSLDS to hyperscanning, dyadic coupling, or therapy outcome prediction. All component pieces exist separately but integration is genuinely novel.

**Library**: ssm (lindermanlab/ssm) has multi-population RSLDS implemented. dynamax lacks RSLDS. Input-driven transitions may need extension.

**Implementation path**: Simple scaffold first (threshold + flexibility metrics) → 2-pop RSLDS (EEG + BL expr) → 3-pop + K=4 → drug-state covariates → outcome prediction.

**Why:** Gordon 2025 shows flexibility > aggregate. Cohen 2021 shows state-like predicts alliance, trait-like does not. Uhl 2025 shows within-therapist variability is the signal. The RSLDS formalization captures all of this.

**Expanded to 9D observations**: EEG volt_amp z, BL expression z, BL state z, ECG SNS z, ECG PNS z, Pose z, Prosodic z, Linguistic z, + future EDA SCR z. Data/param ratio: 300:1 at 9D.

**Missing data**: Handled via masked likelihood. Temporal drops → other channels constrain state. Entire modality absent → hierarchical borrowing. No session harms other sessions.

**Hierarchical across sessions**: Population-level A, B, transition (shared); session-level emissions (unique). Partial pooling = mixed-effects for state-space models. Therapist-baseline deviation is directly computed.

**Deliverable**: Per-session significance-masked timeline with RSLDS state coloring, cross-checkable with video via LSL timestamps. Clinical report with engagement episodes, flexibility metrics, condition effects, and therapist-baseline deviation. Longitudinal trajectory across treatment course.

**Stress test risks**: Autocorrelation (fix: AR(1) emissions), non-Gaussian (fix: Student-t), edge effects (fix: transition condition labels). 2 Hz sampling is appropriate for coupling dynamics.

**How to apply:** Always run on full sessions, not per-segment. Conditions are covariates, not boundaries. Build simple scaffold first to inform RSLDS priors.

---

## Implementation Status (2026-03-30)

### V8.2 Production Pipeline (Current)

**18D observation vector** with three orthogonal EEG feature types:
- **Phase coupling** (ImCoh): 3ch — imaginary coherence per band from Welch CSD (2s windows)
- **Shared state** (Concordance): 3ch — (z_power_P1 + z_power_P2)/2 per band
- **Dynamics** (EWMAD): 3ch — exponentially-weighted mean absolute deviation of concordance (tau=3s)
- Plus BL expression coupling, BL activity concordance, ECG LF/HF, Resp, Pose (6ch)

**Key empirical finding**: The original V8 cross-product EEG features (volt_amp from bycycle) fail to separate ANY condition pair (all p>0.29). Replaced by ImCoh + concordance which together give 5 significant features (p<0.05). The cross-product measures amplitude co-fluctuation which is NOT condition-dependent. ImCoh measures phase consistency which IS. Concordance measures shared power state which IS.

**Model architecture**:
- Null-state rSLDS: d_emit[0]=0 fixed, asymmetric transitions, sticky prior (kappa=3)
- Constrained Viterbi: 10s minimum dwell time
- Hierarchical pooling: shared dynamics + transitions across 12 sessions
- Per-timepoint face/pose observation masking

**State profiles (hierarchical rSLDS, 12 sessions)**:
- S0 NULL: everything zero (baseline/uncoupled)
- S1 COUP: moderate concordance + BL expression (engaged interaction)
- S2 OTHER: high dynamics + zero concordance (turn-taking asymmetric coupling)
- S3 SHARED: high concordance + high dynamics (active shared neural state)

### Dataset
- 12 sessions with V8.2 scaffold data (+ y_26 missing cache, y_03 broken markers)
- 6 meditation protocol (y_06, y_17, y_19, y_11, y_04, y_24)
- 5 PE protocol (y_01, y_05, y_10, y_32, y_41)
- 1 baselines-only (y_03)
- PE markers (PE, PE_1, PE_2) and baseline marker included in CONDITION_ORDER

### Key Limitations
- Cross-session state-condition variance: +/-30% (session-specific patterns dominate)
- Null state captures conversation profile (mixed features near zero after standardization)
- EWMAD dynamics captures sub-second volatility, not 10-20s turn-taking dynamics

### Resolved Issues
- **EEG amplitude coupling** does not differentiate conditions → replaced by ImCoh + concordance
- **Pose zero-lag cross-product** can't detect lagged coupling → replaced by multi-lag ±5s bank (p=0.006)
- **Semi-synthetic validation** confirms EEG + BL pipelines work end-to-end (AUC 0.6-0.9)

### Remaining Issues
1. **EDA modality** — literature's strongest coupling signal, not yet in hardware
2. **Turn-taking dynamics at correct timescale** — need feature targeting 10-20s period
3. **ECG/Resp coupling** empirically weak (1-2% coupling fraction) — may not contribute to rSLDS

### Production Scripts
- `scripts/_run_scaffold_v82.py` — 18D scaffold (production), `run_from_raw()` for semi-synthetic
- `scripts/_run_v82_hierarchical.py` — hierarchical rSLDS
- `scripts/_run_v82_rslds_analysis.py` — per-session rSLDS
- `scripts/_test_v82_semisynthetic_battery.py` — semi-synthetic validation battery
- `scripts/_test_eeg_metrics_h2h.py` — metric comparison
- `scripts/_run_v82_analysis.py` — concordance + coherence condition analysis

### Burst Analysis Scripts (2026-03-30)
- `scripts/_plot_rslds_quiver.py` — State-center flow (domain axes, not PCA), modality profiles, trajectory quiver
- `scripts/_plot_rslds_excursions.py` — Decompose large trajectory jumps by modality/condition
- `scripts/_plot_rslds_bursts.py` — Continuous intensity timecourse, burst detection, peri-burst averages
- `scripts/_validate_rslds_bursts.py` — Cross-session validation, pseudo-dyad null, permutation CIs
- `scripts/_validate_bursts_by_condition.py` — Per-segment asymmetry and burst rates (production)

### Burst Analysis: Validated Findings
Coupling burst events (90th pctl, 2s min, 3s merge) layered on state assignments. Full synthesis: `docs/rslds_burst_analysis_synthesis.md`.

1. **Therapist/patient asymmetry reverses by condition**: Patient drives conversation (theta -0.213±0.036); therapist drives meditate_K (+0.367±0.034) and PE_1 (+0.679±0.177). PE_2 collapses. Baselines null.
2. **Burst rates**: Conv ~2/min Face+Body; meditation suppresses all except EEG phase (~1/min); base_EC 2.5× EEG phase vs base_EO.
3. **Protocol validation**: Patient alpha meditation > PE (+0.236, p<0.05); patient theta conversation >> meditation (p<0.01); meditate_B more alpha than meditate_K.
4. **Cross-modal lead/lag does NOT validate** at n=12 (needs ~50+ sessions).
5. **Per-segment stratification mandatory**: 8 segments (base_EO, base_EC, conv_1, conv_2, meditate_B, meditate_K, PE_1, PE_2), never pool.

### Domain-Axis Visualization (replaces PCA)
State-center quiver plots use interpretable composites: Phase Coupling = mean(ImCoh θ/α/β), Shared Power = mean(Conc θ/α/β), Body Coupling = mean(BL, Pose), Autonomic = mean(ECG, Resp). PCA dominated by concordance variance and produced uninterpretable axes.

### Timeline Plots
- Per-session: `results/rslds/*/v82_hierarchical_timeline.png`
- Results: `results/rslds/v82_hierarchical_results.json`
- Burst analysis: `results/rslds/quiver_plots/` (39 figures + 2 JSON)
