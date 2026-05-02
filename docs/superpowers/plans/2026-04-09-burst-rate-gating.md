# Burst-Rate Gating for TE Estimation — Implementation Plan (SUPERSEDED)

> **SUPERSEDED** by comprehensive prewhitening fix + burst-rate gating.
> See `C:\Users\optilab\.claude\plans\snoopy-snuggling-noodle.md` for the executed plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate TE (transfer entropy) channels by per-participant burst rate to remove the condition-level confound where TE episode detection correlates with alpha/theta burst rate (rho=0.25, p=0.03 ungated; rho=0.03 after gating at 5%).

**Architecture:** Compute per-participant rolling burst rates inside `compute_burst_features()` alongside existing TE/coincidence extraction. After prewhitening (which can't handle NaN), apply the burst gate by updating `obs_mask` for TE observation channels and zeroing TE asymmetry covariates at gated timepoints. Save burst rates in the scaffold NPZ for downstream analysis.

**Tech Stack:** numpy, existing V11 scaffold pipeline

**Key constraint:** `prewhiten_and_standardize()` uses `np.corrcoef`/`np.mean`/`np.std` which propagate NaN. Gating MUST happen AFTER prewhitening, at the obs_mask stage — not before. The rSLDS already handles obs_mask correctly (zeroes log-emissions for masked dimensions).

**Empirically validated threshold:** `MIN_BURST_RATE = 0.05` (5%) for both bands. At this threshold:
- Confound eliminated: theta rho +0.21→-0.04, alpha rho +0.25→+0.03
- TE z-scores well-calibrated at all rates ≥5% (std≈1.0, tails≈2.5%)
- Data retention: ≥67% for all conditions (theta PE lowest at 90%, alpha all ≥100%)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/_run_scaffold_v11.py` | Modify | Add burst rate computation to `compute_burst_features()`, apply gate to obs_mask + covariates, save rates in NPZ |
| `scripts/run_condition_statistics.py` | Modify | Gate TE episode fractions by burst rate, report gate fraction |
| `scripts/_run_v11_hierarchical.py` | Read-only | Verify obs_mask propagation (already correct) |
| `cadence/significance/fast_cycles.py` | Read-only | `extract_burst_grids()` unchanged |
| `cadence/significance/directed_burst_coupling.py` | Read-only | `gpu_sliding_te_surrogates()` unchanged |
| `scripts/_run_rslds_scaffold_v8.py` | Read-only | `prewhiten_and_standardize()` unchanged |

---

### Task 1: Extend `compute_burst_features()` to return burst rates

**Files:**
- Modify: `scripts/_run_scaffold_v11.py:80-188` (`compute_burst_features` function)

The function currently extracts burst grids, computes TE and coincidence, then discards the grids. We need to also compute per-participant rolling burst rates and return them.

- [ ] **Step 1: Add `MIN_BURST_RATE` constant at module level**

Add after the existing imports (around line 75, before the function definition at line 80):

```python
# ── Burst-rate gating for TE reliability ────────────────────────────
# Minimum rolling burst rate (60s window) for TE to be estimable.
# Empirically validated: eliminates burst-rate/TE-episode confound
# (rho: +0.25 -> +0.03) while retaining >=67% of all conditions.
# See scripts/_analyze_burst_rate_thresholds.py for derivation.
MIN_BURST_RATE = 0.05
```

- [ ] **Step 2: Add rolling rate helper inside `compute_burst_features()`**

Add inside the function body, after the `obs`/`cov` dict initialization (after line 112), before the EEG guard clause (line 114):

```python
    # Burst rate outputs (per-participant rolling rates + gate)
    rate_keys = ['p1_burst_rate_theta', 'p2_burst_rate_theta',
                 'p1_burst_rate_alpha', 'p2_burst_rate_alpha',
                 'burst_gate_theta', 'burst_gate_alpha']
    rates = {k: np.zeros(N, dtype=np.float32) for k in rate_keys[:4]}
    rates['burst_gate_theta'] = np.zeros(N, dtype=bool)
    rates['burst_gate_alpha'] = np.zeros(N, dtype=bool)
```

- [ ] **Step 3: Compute rolling burst rates inside the per-band loop**

Inside the `for band_name, bg in grids.items():` loop (line 142), after `p1b` and `p2b` are defined (after line 155), add burst rate computation for theta and alpha:

```python
        # ── Per-participant rolling burst rate (theta + alpha) ──────
        if band_name in ('theta', 'alpha'):
            window = 120  # 60s at 2 Hz, matching TE window
            C_valid = p1b.shape[0]
            N_local = p1b.shape[1]

            # Channel-averaged rolling mean via cumsum
            p1_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
            p2_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
            for c in range(C_valid):
                cs1 = np.cumsum(p1b[c].astype(np.float32))
                cs1 = np.insert(cs1, 0, 0.0)
                cs2 = np.cumsum(p2b[c].astype(np.float32))
                cs2 = np.insert(cs2, 0, 0.0)
                for t in range(N_local):
                    t0 = max(0, t - window // 2)
                    t1 = min(N_local, t + window // 2)
                    span = t1 - t0
                    p1_rates_ch[c, t] = (cs1[t1] - cs1[t0]) / span
                    p2_rates_ch[c, t] = (cs2[t1] - cs2[t0]) / span

            p1_rate_local = p1_rates_ch.mean(axis=0)
            p2_rate_local = p2_rates_ch.mean(axis=0)

            # Interp to t_common
            p1_rate = np.interp(t_common, t_grid_lsl, p1_rate_local,
                                left=0, right=0).astype(np.float32)
            p2_rate = np.interp(t_common, t_grid_lsl, p2_rate_local,
                                left=0, right=0).astype(np.float32)
            rates[f'p1_burst_rate_{band_name}'] = p1_rate
            rates[f'p2_burst_rate_{band_name}'] = p2_rate
            rates[f'burst_gate_{band_name}'] = (
                (p1_rate >= MIN_BURST_RATE) & (p2_rate >= MIN_BURST_RATE))
```

- [ ] **Step 4: Update the return signature**

Change the return statement at line 188 from:

```python
    return obs, cov
```

to:

```python
    return obs, cov, rates
```

- [ ] **Step 5: Update the docstring**

Update the Returns section of the docstring (lines 99-103) to:

```python
    Returns:
        obs: dict with te_conc_theta, te_conc_alpha,
             burst_coinc_theta, burst_coinc_alpha, burst_coinc_beta.
        cov: dict with te_asym_theta, te_asym_alpha.
        rates: dict with p1_burst_rate_theta, p2_burst_rate_theta,
               p1_burst_rate_alpha, p2_burst_rate_alpha,
               burst_gate_theta, burst_gate_alpha.
        All (N,) arrays aligned to t_common.
```

---

### Task 2: Apply burst gate in `run_session()` — obs_mask + covariates

**Files:**
- Modify: `scripts/_run_scaffold_v11.py:357-439` (steps [9/13] through [12/13] in `run_session()`)

The burst gate modifies two things:
1. `obs_mask` — set False for TE observation channels at gated timepoints
2. TE asymmetry covariates — set to 0 at gated timepoints (before they enter U matrix)

- [ ] **Step 1: Update the `compute_burst_features` call site to unpack 3 return values**

Change lines 360-362 from:

```python
    burst_obs, burst_cov = compute_burst_features(
        cached, t_common, lsl_offset, asym_sign=asym_sign,
        n_surrogates=200, seed=42)
```

to:

```python
    burst_obs, burst_cov, burst_rates = compute_burst_features(
        cached, t_common, lsl_offset, asym_sign=asym_sign,
        n_surrogates=200, seed=42)
```

- [ ] **Step 2: Add burst rate diagnostics after existing TE/coincidence diagnostics**

After line 374 (`print(f"  Burst features: {time.time() - t0:.1f}s")`), add:

```python
    # Burst rate gate diagnostics
    for bn in ['theta', 'alpha']:
        gate = burst_rates[f'burst_gate_{bn}']
        p1r = burst_rates[f'p1_burst_rate_{bn}']
        p2r = burst_rates[f'p2_burst_rate_{bn}']
        print(f"    Burst rate {bn}: P1 mean={p1r.mean():.3f}, P2 mean={p2r.mean():.3f}, "
              f"gate={gate.mean():.1%} (min_rate={MIN_BURST_RATE})")
```

- [ ] **Step 3: Apply burst gate to obs_mask for TE observation channels**

After the existing obs_mask construction (after line 416, where pose_valid is applied), add:

```python
    # Burst-rate gate: mask TE observation channels at low-rate timepoints
    te_obs_keys = ['te_conc_theta', 'te_conc_alpha']
    for te_key in te_obs_keys:
        band = te_key.split('_')[-1]  # 'theta' or 'alpha'
        gate_key = f'burst_gate_{band}'
        if gate_key in burst_rates:
            te_idx = V11_MODALITY_KEYS.index(te_key)
            gate = burst_rates[gate_key]
            n_gated = (~gate).sum()
            obs_mask[:, te_idx] &= gate
            print(f"    TE gate {band}: {n_gated} timepoints masked "
                  f"({n_gated/N_common:.1%} of session)")
```

- [ ] **Step 4: Apply burst gate to TE asymmetry covariates AFTER prewhitening**

In the transition covariate section (around lines 428-437), the gate must be applied AFTER `prewhiten_and_standardize` — zeroing before would corrupt the AR(1) estimate (gate boundaries create artifactual residuals). Change:

```python
    # Append TE asymmetry covariates (prewhiten individually)
    te_asym_covs = np.column_stack([
        burst_cov['te_asym_theta'],
        burst_cov['te_asym_alpha'],
    ])
    # Prewhiten TE asymmetry covariates (same AR(1) as observations)
    _pw_fn = prewhiten_and_standardize  # avoid re-import shadowing
    te_asym_pw, _ = _pw_fn(te_asym_covs, ['te_asym_theta', 'te_asym_alpha'])
    U = np.column_stack([U_v10, te_asym_pw])
```

to:

```python
    # Append TE asymmetry covariates (prewhiten individually)
    te_asym_covs = np.column_stack([
        burst_cov['te_asym_theta'],
        burst_cov['te_asym_alpha'],
    ])
    # Prewhiten TE asymmetry covariates (same AR(1) as observations)
    _pw_fn = prewhiten_and_standardize  # avoid re-import shadowing
    te_asym_pw, _ = _pw_fn(te_asym_covs, ['te_asym_theta', 'te_asym_alpha'])
    # Zero out AFTER prewhitening — gating before would corrupt AR(1).
    # Zero covariates contribute nothing to transition probabilities.
    te_asym_pw[~burst_rates['burst_gate_theta'], 0] = 0.0
    te_asym_pw[~burst_rates['burst_gate_alpha'], 1] = 0.0
    U = np.column_stack([U_v10, te_asym_pw])
```

---

### Task 3: Save burst rates in scaffold NPZ

**Files:**
- Modify: `scripts/_run_scaffold_v11.py:444-455` (save_dict in step [13/13])

- [ ] **Step 1: Add burst rate channels to save_dict**

After line 453 (`save_dict[f'u_{key}'] = U[:, i]`), add:

```python
    # Burst rate channels for downstream gating
    for rk in ['p1_burst_rate_theta', 'p2_burst_rate_theta',
               'p1_burst_rate_alpha', 'p2_burst_rate_alpha']:
        save_dict[rk] = burst_rates[rk]
    save_dict['burst_gate_theta'] = burst_rates['burst_gate_theta']
    save_dict['burst_gate_alpha'] = burst_rates['burst_gate_alpha']
```

- [ ] **Step 2: Add gating metadata to results JSON**

After line 472 (the prewhitening diagnostics in the results dict), add:

```python
        'burst_rate_gating': {
            'min_burst_rate': MIN_BURST_RATE,
            'gate_frac_theta': float(burst_rates['burst_gate_theta'].mean()),
            'gate_frac_alpha': float(burst_rates['burst_gate_alpha'].mean()),
        },
```

---

### Task 4: Update `run_from_raw_v11` for semi-synthetic path

**Files:**
- Modify: `scripts/_run_scaffold_v11.py:518-635` (`run_from_raw_v11` function)

This function processes pre-injected raw arrays for semi-synthetic validation. It needs the same gating logic.

- [ ] **Step 1: Compute burst rates from grids in the semi-synthetic path**

Inside the `for band_name, bg in grids.items():` loop (line 574), after `p1b` and `p2b` are defined (line 582), add the same rolling rate computation as Task 1 Step 3. But since this path doesn't save NPZ files, we only need the gate for obs_mask and covariates.

Add a `rates` dict before the loop (after line 570):

```python
        burst_rates = {
            'burst_gate_theta': np.zeros(N, dtype=bool),
            'burst_gate_alpha': np.zeros(N, dtype=bool),
        }
```

Then inside the loop, after `p2b = bg['p2_burst'][ch_mask]` (line 582), add:

```python
            if band_name in ('theta', 'alpha'):
                window = 120
                C_valid, N_local = p1b.shape
                p1_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
                p2_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
                for c in range(C_valid):
                    cs1 = np.cumsum(p1b[c].astype(np.float32))
                    cs1 = np.insert(cs1, 0, 0.0)
                    cs2 = np.cumsum(p2b[c].astype(np.float32))
                    cs2 = np.insert(cs2, 0, 0.0)
                    for t in range(N_local):
                        t0_ = max(0, t - window // 2)
                        t1_ = min(N_local, t + window // 2)
                        span = t1_ - t0_
                        p1_rates_ch[c, t] = (cs1[t1_] - cs1[t0_]) / span
                        p2_rates_ch[c, t] = (cs2[t1_] - cs2[t0_]) / span
                p1_rate = np.interp(t_common, t_grid_lsl,
                                    p1_rates_ch.mean(axis=0), left=0, right=0)
                p2_rate = np.interp(t_common, t_grid_lsl,
                                    p2_rates_ch.mean(axis=0), left=0, right=0)
                burst_rates[f'burst_gate_{band_name}'] = (
                    (p1_rate >= MIN_BURST_RATE) & (p2_rate >= MIN_BURST_RATE))
```

- [ ] **Step 2: Apply gate to obs_mask in semi-synthetic path**

After line 624 (`mask_v11[:, d] = False`), add:

```python
    # Burst-rate gate for TE observation channels
    for te_key in ['te_conc_theta', 'te_conc_alpha']:
        band = te_key.split('_')[-1]
        te_idx = V11_MODALITY_KEYS.index(te_key)
        mask_v11[:, te_idx] &= burst_rates[f'burst_gate_{band}']
```

- [ ] **Step 3: Apply gate to TE asymmetry covariates AFTER prewhitening in semi-synthetic path**

Change lines 627-633 from:

```python
    te_asym_raw = np.column_stack([
        cov_traces['te_asym_theta'],
        cov_traces['te_asym_alpha'],
    ])
    te_asym_pw, _ = prewhiten_and_standardize(
        te_asym_raw, ['te_asym_theta', 'te_asym_alpha'])
    U = np.column_stack([U_v10, te_asym_pw])
```

to:

```python
    te_asym_raw = np.column_stack([
        cov_traces['te_asym_theta'],
        cov_traces['te_asym_alpha'],
    ])
    te_asym_pw, _ = prewhiten_and_standardize(
        te_asym_raw, ['te_asym_theta', 'te_asym_alpha'])
    # Zero out AFTER prewhitening — same as run_session path
    te_asym_pw[~burst_rates['burst_gate_theta'], 0] = 0.0
    te_asym_pw[~burst_rates['burst_gate_alpha'], 1] = 0.0
    U = np.column_stack([U_v10, te_asym_pw])
```

---

### Task 5: Update condition statistics for gated TE

**Files:**
- Modify: `scripts/run_condition_statistics.py:220-231` (TE episode extraction in `extract_metrics()`)

- [ ] **Step 1: Replace the ungated TE episode fraction computation**

Change lines 220-230 from:

```python
        # 6. TE directed episode fractions
        te_theta_key = 'u_te_asym_theta'
        te_alpha_key = 'u_te_asym_alpha'
        if te_theta_key in npz:
            te_th = npz[te_theta_key][mask]
            metrics['te_T>P_theta'] = float(np.mean(te_th > TE_EPISODE_Z_THRESH))
            metrics['te_P>T_theta'] = float(np.mean(te_th < -TE_EPISODE_Z_THRESH))
        if te_alpha_key in npz:
            te_al = npz[te_alpha_key][mask]
            metrics['te_T>P_alpha'] = float(np.mean(te_al > TE_EPISODE_Z_THRESH))
            metrics['te_P>T_alpha'] = float(np.mean(te_al < -TE_EPISODE_Z_THRESH))
```

to:

```python
        # 6. TE directed episode fractions (burst-rate gated)
        for band in ['theta', 'alpha']:
            te_key = f'u_te_asym_{band}'
            gate_key = f'burst_gate_{band}'
            if te_key not in npz:
                continue
            te_vals = npz[te_key][mask]
            if gate_key in npz:
                gate = npz[gate_key][mask]
                n_gated = int(gate.sum())
                metrics[f'te_gate_frac_{band}'] = float(gate.mean())
                if n_gated >= 10:
                    te_gated = te_vals[gate]
                    metrics[f'te_T>P_{band}'] = float(np.mean(te_gated > TE_EPISODE_Z_THRESH))
                    metrics[f'te_P>T_{band}'] = float(np.mean(te_gated < -TE_EPISODE_Z_THRESH))
            else:
                # Fallback: ungated (pre-gating scaffold)
                metrics[f'te_T>P_{band}'] = float(np.mean(te_vals > TE_EPISODE_Z_THRESH))
                metrics[f'te_P>T_{band}'] = float(np.mean(te_vals < -TE_EPISODE_Z_THRESH))
```

---

### Task 6: Re-run all sessions and condition statistics

**Files:**
- Run: `scripts/_run_scaffold_v11.py --all`
- Run: `scripts/run_condition_statistics.py`

- [ ] **Step 1: Delete existing V11 scaffold results to force re-run**

The scaffold script skips sessions that already have results. We need to regenerate all of them with the new burst rate channels.

```bash
# Remove only the scaffold NPZ files (keep rSLDS results — they'll be regenerated by hierarchical)
find results/v11 -name 'scaffold_v11_ztimecourses.npz' -delete
find results/v11 -name 'scaffold_v11_results.json' -delete
```

- [ ] **Step 2: Re-run V11 scaffold on all sessions**

```bash
conda activate MCCT
python scripts/_run_scaffold_v11.py --all
```

Expected: 14 sessions complete (y_19 has no EEG). Each session should print burst rate gate diagnostics showing gate fraction per band.

- [ ] **Step 3: Re-run condition statistics**

```bash
python scripts/run_condition_statistics.py
```

Expected: TE metrics now use gated episode fractions. `te_gate_frac_*` metrics appear in output.

---

### Task 7: Validate — confirm confound eliminated

**Files:**
- Run: `scripts/_analyze_rate_conditional_te.py`

- [ ] **Step 1: Run the rate-conditional TE analysis**

```bash
python scripts/_analyze_rate_conditional_te.py
```

Check:
1. Correlation between alpha power and TE episode fraction should be near zero (rho < 0.10)
2. Meditation dissociation holds: meditation has more alpha than conversation but less TE
3. Null conditions (base_EO, base_EC) don't show inflated TE relative to active conditions

- [ ] **Step 2: Compare gated vs ungated condition statistics**

Diff the new `results/v11/condition_statistics/condition_statistics_summary.md` against the pre-gating version. Key checks:
- Which TE p-values survived gating (genuine directed coupling)?
- Which collapsed (were rate artifacts)?
- Gate fractions should appear for each condition

- [ ] **Step 3: Re-run hierarchical rSLDS**

```bash
python scripts/_run_v11_hierarchical.py
```

Check that state structure is preserved (NULL/COUP/SHARED/OTHER) and that TE observation channels still contribute to state differentiation despite gating.
