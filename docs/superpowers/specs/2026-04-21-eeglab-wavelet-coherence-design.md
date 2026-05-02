# EEGLAB Wavelet Coherence Analysis Module — Design Spec

**Date:** 2026-04-21
**Status:** Proposed
**Author:** cozger + Claude

---

## 1. Context

CADENCE's main pipelines (V7–V11) are rSLDS/regression-based and treat EEG and face signals as inputs to a learned dynamical system. They use CADENCE's own Python preprocessing chain (notch + 1–45 Hz bandpass + z-score + ±10σ clip) and collapse the 52 face AUs into small task-motivated groups for wavelet coherence.

For the current work we want a **simpler analysis module** that:

1. Uses the **EEGLAB ecosystem** (Artifact Subspace Reconstruction + bad-channel detection/interpolation + FIR bandpass) for EEG preprocessing — a widely accepted, reproducible reference pipeline.
2. Produces **per-channel wavelet coherograms** for a dyad, one per EEG electrode and one per face AU, across the canonical session conditions (baseline EO/EC, conversations, meditation/PE blocks).
3. Keeps EEG and AU treated symmetrically: each of the 14 EEG electrodes and each of the 52 face blendshapes is analyzed independently, rather than grouped into ROIs or affect sets.
4. Is **self-contained** — does not modify any existing CADENCE pipeline (V7/V10/V11), does not touch MCCT's session cache, and can be removed cleanly.

The intended use case is descriptive: generate a visual reference for dyadic coupling structure in each modality, per condition, with EEGLAB-standard preprocessing so results are directly comparable to the broader hyperscanning literature.

Output is **raw wavelet coherence magnitude (0–1)** — no surrogate z-scoring, no thresholding, no significance test. Figures plus `.npz` arrays.

---

## 2. Architecture

```
analysis/
  eeglab_wavelet/
    # Entry-point scripts (each independently runnable)
    run_all.py                  # master — calls steps 1 → 2 → 3 with shared args
    run_matlab_preprocess.py    # step 1 — parallel MATLAB over all sessions
    run_eeg_coherence.py        # step 2 — EEG wavelet coherograms per session
    run_au_coherence.py         # step 3 — AU wavelet coherograms per session

    # Support modules (imported by scripts)
    session_io.py               # XDF → raw EEG, .mat export/import,
                                # condition slicing, AU loading from cache
    eeg_wavelet.py              # CWT + per-electrode coherence (reuses bl_wavelet)
    au_wavelet.py               # CWT + per-AU coherence (reuses bl_wavelet)
    plotting.py                 # coherogram PNG renderer

    # MATLAB pipeline (single script, three steps inline)
    matlab/
      preprocess_eeg.m          # ASR → pop_interp → pop_eegfiltnew (1-40 Hz)

    # Auto-created at runtime
    cache/                      # MATLAB I/O (not checked into git)
      {session_id}_p1_raw.mat   # Python → MATLAB input
      {session_id}_p1_clean.mat # MATLAB → Python output
      # same for p2
    results/
      {session_id}/
        {condition}/
          eeg_{electrode}.npz   # (freqs, times, coherence[F×T])
          eeg_{electrode}.png
          au_{blendshape}.npz
          au_{blendshape}.png
```

**Separation of concerns:**
- `matlab/preprocess_eeg.m` is the **preprocessing boundary**. Python never runs ASR; MATLAB never computes coherence. Data crosses via `.mat` files.
- Python support modules are narrow: `session_io` handles I/O only, `eeg_wavelet` and `au_wavelet` are pure compute, `plotting` is pure rendering.
- Scripts are thin (< ~80 lines each) — orchestration only.

**Reuse vs. new:**

| Reused | Source | Role |
|---|---|---|
| `xdf_loader.load_session()` | `cadence/data/xdf_loader.py:14` | Load raw EEG (bypasses main cache, which deletes raw) |
| `parse_condition_intervals()` | `cadence/conditions.py:39` | Segment session into conditions |
| `_cwt_gpu()` | `cadence/significance/bl_wavelet.py:127` | GPU Morlet CWT, multi-channel vectorized |
| `_morlet_wavelet_bank()` | `cadence/significance/bl_wavelet.py:189` | Wavelet bank generator |
| `EPOC_CHANNEL_NAMES` | `cadence/constants.py:222` | 14-channel labels |
| `BLENDSHAPE_NAMES` | `docs/validate_blendshape_isolation.py:44` | 52 AU names |
| Cached 52 AUs at 30 Hz | MCCT `session_cache/` | AU side — no re-preprocessing |

| New | Location | Role |
|---|---|---|
| `preprocess_eeg.m` | `matlab/` | Single MATLAB script: ASR → interpolate → bandpass |
| Per-channel coherence (no AU summing) | `eeg_wavelet.py`, `au_wavelet.py` | Adapt `_coherence_from_coeffs_gpu` to keep channel dim |
| `.mat` export/import helpers | `session_io.py` | `scipy.io.savemat`/`loadmat` boilerplate |
| Coherogram renderer | `plotting.py` | Matplotlib PNG per coherogram |

---

## 3. Data Flow

### Step 1 — MATLAB preprocessing (`run_matlab_preprocess.py`)

```
For each session (joblib.Parallel, backend='loky', n_jobs=-1):
    1. session = xdf_loader.load_session(xdf_path)
       → session['p1_eeg_raw'] (N × 19), session['p2_eeg_raw']
       → session['markers'] (list of (ts, label))
    2. p1_eeg = session['p1_eeg_raw'][:, 3:17]     # 14 EEG channels
       p2_eeg = session['p2_eeg_raw'][:, 3:17]
       srate = estimate_srate(session['p1_eeg_ts']) # 256 Hz
    3. session_io.export_raw_mat(session_id, p1_eeg, p2_eeg, srate,
                                  EPOC_CHANNEL_NAMES, cache_dir)
       → writes cache/{session_id}_p1_raw.mat, cache/{session_id}_p2_raw.mat
       Also writes cache/{session_id}_markers.json (markers + t_start_absolute)
    4. subprocess.run([
           MATLAB_EXE, '-batch',
           f"addpath('{matlab_dir}'); preprocess_eeg('{session_id}', '{cache_dir}')"
       ], check=True, timeout=1800)
       MATLAB produces: cache/{session_id}_p1_clean.mat, cache/{session_id}_p2_clean.mat
    5. session_io.verify_clean_mat(session_id, cache_dir)
       → assert file exists, shape matches expected (N_samples × 14)
```

**Parallelism:** Each `matlab -batch` invocation is independent. `joblib.Parallel(n_jobs=-1)` launches them simultaneously. MATLAB startup is ~15–30 s; with 11 sessions running in parallel this is paid once, not 11 times. If startup overhead is a concern, `n_jobs` can be clamped to ≤ number of physical cores.

**Idempotency:** If `cache/{session_id}_p1_clean.mat` already exists and is newer than the corresponding raw `.mat`, step 1 skips that session (`--force` flag to override).

### Step 2 — EEG wavelet coherence (`run_eeg_coherence.py`)

```
For each session (joblib.Parallel, n_jobs=4 — GPU-bound):
    1. p1_clean, p2_clean, srate, ch_labels =
           session_io.load_clean_eeg(session_id, cache_dir)
       # Shapes: (N × 14), scalar, list of 14 strings
    2. intervals = session_io.load_condition_intervals(session_id, cache_dir)
       # From cached markers.json → parse_condition_intervals()
    3. For each (start_s, end_s, cond) in intervals:
        a. i0, i1 = int(start_s * srate), int(end_s * srate)
           p1_seg = p1_clean[i0:i1, :]    # (T_cond × 14)
           p2_seg = p2_clean[i0:i1, :]
        b. coh, freqs, times = eeg_wavelet.compute_coherence(
               p1_seg, p2_seg, fs=srate,
               f_lo=1.0, f_hi=40.0, n_freqs=50,
               smooth_s=0.5,                # Gaussian temporal smoothing
               device='auto')
           # coh: (14, F, T_cond)   — raw coherence magnitude 0-1
        c. For ch_idx, ch_name in enumerate(ch_labels):
               out_dir = results/{session_id}/{cond}/
               np.savez(out_dir / f'eeg_{ch_name}.npz',
                        freqs=freqs, times=times, coherence=coh[ch_idx])
               plotting.plot_coherogram(coh[ch_idx], freqs, times,
                                         title=f'{session_id} | {cond} | {ch_name}',
                                         out_path=out_dir / f'eeg_{ch_name}.png')
```

### Step 3 — AU wavelet coherence (`run_au_coherence.py`)

```
For each session (joblib.Parallel, n_jobs=4):
    1. p1_aus, p2_aus, srate = session_io.load_au_from_cache(session_id)
       # From MCCT session_cache: shape (N × 52) at 30 Hz
       # Uses cadence.data.alignment.load_session_from_cache()
    2. intervals = session_io.load_condition_intervals(session_id, cache_dir)
    3. For each (start_s, end_s, cond) in intervals:
        a. i0, i1 = int(start_s * srate), int(end_s * srate)
           p1_seg = p1_aus[i0:i1, :]      # (T_cond × 52)
           p2_seg = p2_aus[i0:i1, :]
        b. coh, freqs, times = au_wavelet.compute_coherence(
               p1_seg, p2_seg, fs=30.0,
               f_lo=0.3, f_hi=8.0, n_freqs=30,
               smooth_s=0.5,
               device='auto')
           # coh: (52, F, T_cond)
        c. For au_idx, au_name in enumerate(BLENDSHAPE_NAMES):
               np.savez(out_dir / f'au_{au_name}.npz',
                        freqs=freqs, times=times, coherence=coh[au_idx])
               plotting.plot_coherogram(coh[au_idx], freqs, times,
                                         title=f'{session_id} | {cond} | {au_name}',
                                         out_path=out_dir / f'au_{au_name}.png')
```

### `run_all.py`

```
parser: --session SESSION_ID | --all
        --force (re-run MATLAB even if clean .mat exists)
        --skip-matlab (reuse cached clean .mat)
        --skip-eeg / --skip-au

Calls in order:
    run_matlab_preprocess.main(args)
    run_eeg_coherence.main(args)
    run_au_coherence.main(args)
```

---

## 4. MATLAB Script

Single file: `analysis/eeglab_wavelet/matlab/preprocess_eeg.m`.

```matlab
function preprocess_eeg(session_id, cache_dir)
    % Master EEG preprocessing: ASR → bad channel interpolation → 1-40 Hz bandpass.
    % Expects cache/{session_id}_p{1,2}_raw.mat to exist.
    % Writes cache/{session_id}_p{1,2}_clean.mat.

    for p = 1:2
        raw_path   = fullfile(cache_dir, sprintf('%s_p%d_raw.mat',   session_id, p));
        clean_path = fullfile(cache_dir, sprintf('%s_p%d_clean.mat', session_id, p));

        s = load(raw_path);
        % s.data: (N × 14), s.srate: 256, s.ch_labels: cellstr of 14 electrode names

        % Build minimal EEGLAB struct. EEGLAB's 10-20 lookup resolves labels.
        EEG = eeg_emptyset();
        EEG.data     = single(s.data');        % EEGLAB wants (n_ch × n_samp)
        EEG.srate    = double(s.srate);
        EEG.nbchan   = size(EEG.data, 1);
        EEG.pnts     = size(EEG.data, 2);
        EEG.trials   = 1;
        EEG.xmin     = 0;
        EEG.xmax     = (EEG.pnts - 1) / EEG.srate;
        for k = 1:numel(s.ch_labels)
            EEG.chanlocs(k).labels = char(s.ch_labels{k});
        end
        EEG = pop_chanedit(EEG, 'lookup', ...
            'standard-10-5-cap385.elp');  % EEGLAB-bundled standard locations
        orig_chanlocs = EEG.chanlocs;
        EEG = eeg_checkset(EEG);

        % ---- Step 1: clean_rawdata (ASR + bad channel detection) ----
        % Burst reconstruction on (no rejection of time windows).
        % High-pass off here; we apply bandpass as step 3.
        EEG = clean_rawdata(EEG, ...
            'FlatlineCriterion',  5, ...
            'ChannelCriterion',   0.8, ...
            'LineNoiseCriterion', 4, ...
            'Highpass',           'off', ...
            'BurstCriterion',     20, ...
            'WindowCriterion',    'off', ...
            'BurstRejection',     'off');

        % ---- Step 2: Bad channel interpolation (restore removed channels) ----
        EEG = pop_interp(EEG, orig_chanlocs, 'spherical');

        % ---- Step 3: Bandpass 1-40 Hz (FIR, EEGLAB default) ----
        EEG = pop_eegfiltnew(EEG, 1, 40);

        % Save cleaned EEG. Transpose back to (N × 14) for Python convenience.
        data   = double(EEG.data');   %#ok<NASGU>
        srate  = EEG.srate;           %#ok<NASGU>
        ch_labels = s.ch_labels;      %#ok<NASGU>
        save(clean_path, 'data', 'srate', 'ch_labels', '-v7');
    end
end
```

**EEGLAB dependency:** The script assumes EEGLAB (with `clean_rawdata` plugin) is on MATLAB's path. The master Python script will pass `addpath('{EEGLAB_ROOT}')` before invoking `preprocess_eeg` — configurable via env var `CADENCE_EEGLAB_ROOT` (default prompts user to set it, fails fast with a clear message).

**No notch filter:** 1–40 Hz bandpass is below typical mains frequencies (50/60 Hz), so notch is unnecessary. This matches the spec exactly and avoids the double-filtering artifact concern.

**ASR parameters:** Defaults from Chang et al. 2018 / EEGLAB tutorial. `BurstCriterion=20` keeps ASR reconstruction mild (the current Python pipeline uses a hard 100 µV cutoff which is more aggressive). `BurstRejection='off'` + `WindowCriterion='off'` means ASR reconstructs artifacts in place rather than cutting time windows, so the sample count is preserved — critical for downstream condition slicing.

---

## 5. CWT + Coherence Implementation

Two tiny modules — mostly a wrapper around reused `bl_wavelet.py` utilities, with **one** modified coherence kernel.

**`eeg_wavelet.compute_coherence(p1, p2, fs, f_lo, f_hi, n_freqs, smooth_s, device)`:**

```
1. freqs = np.logspace(log10(f_lo), log10(f_hi), n_freqs)
2. w1 = _cwt_gpu(p1, fs, freqs, MORLET_OMEGA)   # (F, T, C)
   w2 = _cwt_gpu(p2, fs, freqs, MORLET_OMEGA)
3. coh = _coherence_per_channel_gpu(w1, w2, sigma_samples=smooth_s*fs)
   # (C, F, T) — NEW function: like _coherence_from_coeffs_gpu but without
   # the AU-sum; keeps channel as leading dim.
4. times = np.arange(T) / fs
5. return coh, freqs, times
```

`au_wavelet.compute_coherence` is identical — just different `fs` and frequency range defaults.

**`_coherence_per_channel_gpu(w1, w2, sigma_samples)`:**

```python
# w1, w2: (F, T, C) complex CWT coefficients
c1 = torch.as_tensor(w1, device='cuda', dtype=torch.complex64)
c2 = torch.as_tensor(w2, device='cuda', dtype=torch.complex64)

cross = c1 * c2.conj()       # (F, T, C)
auto1 = (c1.abs() ** 2)      # (F, T, C)
auto2 = (c2.abs() ** 2)      # (F, T, C)

# Gaussian smooth in time, broadcast over (F, C)
cross_s = smooth_time(cross, sigma_samples)  # (F, T, C)
auto1_s = smooth_time(auto1, sigma_samples)
auto2_s = smooth_time(auto2, sigma_samples)

coherence = cross_s.abs() ** 2 / (auto1_s * auto2_s + 1e-10)
# Return (C, F, T) for convenient per-channel indexing
return coherence.permute(2, 0, 1).cpu().numpy().astype(np.float32)
```

Smoothing is the same Gaussian conv1d as `_coherence_from_coeffs_gpu`, applied per (freq, channel) row over time.

**Frequency grids:**
- EEG: 50 log-spaced freqs from 1–40 Hz (covers delta 1–4, theta 4–8, alpha 8–13, beta 13–30, low gamma 30–40).
- AU: 30 log-spaced freqs from 0.3–8 Hz (matches V7 bl_wavelet — expression band 0.5–2 Hz, speech 2–7 Hz).

**Morlet ω:** Reuse `MORLET_OMEGA` from `bl_wavelet.py` (5.0, standard).

**GPU chunking:** If a condition is long (e.g. 10-min conversation × 256 Hz = 153,600 samples × 50 freqs × 14 ch × complex64 = ~8.2 GB), use existing `_cwt_chunked` (in `bl_wavelet.py`) which already handles this case.

---

## 6. Coherogram Plot Format

`plotting.plot_coherogram(coherence, freqs, times, title, out_path)`:
- Figure: 12 × 4 in, 120 dpi
- `pcolormesh(times, freqs, coherence, vmin=0, vmax=1, cmap='viridis')`
- Y axis: log scale, labeled freqs (e.g. 1, 2, 4, 8, 16, 32 for EEG; 0.3, 1, 2, 4, 8 for AU)
- X axis: seconds within condition (0 to condition_duration)
- Title as `{session_id} | {condition} | {channel_or_au_name}`
- Colorbar with label "Wavelet coherence"
- `tight_layout()` + `savefig(out_path, bbox_inches='tight')` + `close()`

Each condition folder contains 14 EEG PNGs + 52 AU PNGs = 66 coherograms. For 11 sessions × ~6 conditions/session that's ~4,350 PNG files. Each ~50–200 KB → ~500 MB total. Acceptable.

`.npz` files are small (F × T × float32): a 10-min conversation coherogram at 50 freqs × 30720 time samples ~ 6 MB uncompressed, ~2 MB compressed.

---

## 7. Parameters and Defaults

| Parameter | EEG | AU |
|---|---|---|
| Source sample rate | 256 Hz | 30 Hz |
| Morlet ω | 5.0 | 5.0 |
| Frequency range | 1–40 Hz | 0.3–8 Hz |
| n_freqs | 50 | 30 |
| Temporal smoothing σ | 0.5 s | 0.5 s |
| Device | auto (GPU if available) | auto |

MATLAB parameters follow EEGLAB tutorial defaults for `clean_rawdata`. All values are constants at the top of each module (no hidden magic).

---

## 8. Dependencies

**Python (already installed in MCCT env):**
- numpy, scipy (for `scipy.io.savemat/loadmat`), torch, matplotlib, joblib, pyxdf

**Python (new):**
- None. All dependencies already present.

**MATLAB (user must have installed):**
- MATLAB (any recent version; script uses `save(..., '-v7')` for maximum compatibility)
- EEGLAB (https://eeglab.org/)
- `clean_rawdata` plugin (EEGLAB Extension Manager)

**Environment:**
- `CADENCE_EEGLAB_ROOT` env var → path to EEGLAB installation (e.g. `C:/eeglab_current`).
- `CADENCE_MATLAB_EXE` env var → optional override for matlab executable path (default: `matlab` on PATH).

---

## 9. Verification Plan

### Smoke test — single session

```bash
python analysis/eeglab_wavelet/run_all.py --session y_06
```

Expected outputs in `analysis/eeglab_wavelet/results/y_06/`:
- One subdirectory per condition present in y_06 (meditation protocol: base_EO, base_EC, conv_1, meditate_B, meditate_K, conv_2 — 6 conditions)
- Each condition dir has 14 `eeg_*.png` + 14 `eeg_*.npz` + 52 `au_*.png` + 52 `au_*.npz` = 132 files per condition
- Check: open `eeg_AF3.png` for conv_1 and meditate_B — conversation should show higher broadband coherence than meditation in frontal channels (consistent with V11 findings).

### Cross-check with V7 BL wavelet results

- The new `au_mouthSmileLeft.png` and `au_mouthSmileRight.png` should qualitatively match the AFFECT_AUS coherogram from V7 (`bl_wavelet.py` summed affect group). Mean coherence across smile AUs ≈ per-AU coherence averaged.

### Cache invariance

- Run the script, delete one `eeg_*.png` (not its `.npz`), re-run: the PNG is regenerated from the cached `.mat`, not from MATLAB re-preprocessing.
- Run with `--force` and confirm `.mat` files are regenerated (mtime updated).

### MATLAB sanity

- After step 1 completes, manually `load('cache/y_06_p1_clean.mat')` in MATLAB. Confirm: `size(data)` = `(N, 14)`, no NaNs, `srate == 256`, amplitude range looks physiological (< ±100 µV for most samples).
- Spot-check a channel in the cleaned output vs. the raw `_raw.mat`: large transient artifacts (e.g. blink spikes >200 µV) should be reduced but the underlying neural rhythm should be preserved.

### Pseudo-dyad null (optional, follow-up)

Running the pipeline with P1 from session A + P2 from session B should yield low, unstructured coherograms (matches CADENCE's pseudo-dyad null principle). This is a future validation — not part of initial implementation.

### Figures to sanity-check by eye

- `results/y_06/base_EC/eeg_O1.png`: should show a visible alpha (~10 Hz) coherence ridge — shared eyes-closed alpha.
- `results/y_06/meditate_B/eeg_AF3.png`: should show LOW frontal coherence (patient eyes closed, no interaction).
- `results/y_06/conv_1/au_mouthSmileLeft.png`: should show bursts of coherence in expression band (0.5–2 Hz) aligned with shared laughter/smiles.

---

## 10. Out of Scope / Non-Goals

- **No significance testing.** No surrogates, no FDR. Raw coherence magnitude only. A future follow-up module can layer surrogates on top of the saved `.npz` files without touching this module.
- **No rSLDS integration.** This module does not feed into V10/V11 scaffolds. It is standalone.
- **No ECG / respiratory / pose.** EEG and face only.
- **No new CADENCE preprocessing.** The MATLAB step is the only new preprocessing; existing Python pipelines are untouched.
- **No cross-session aggregation.** Per-session, per-condition outputs only. Grouping or averaging across sessions is a separate analysis.

---

## 11. Open Questions (for implementation-time resolution, not blocking design approval)

1. Should `run_all.py --all` run MATLAB preprocessing for all sessions in one shot (max parallelism) or chunk it (e.g. 4 at a time to avoid exhausting RAM)? Default: `n_jobs=-1` (all); chunking via `--n-jobs N` flag.
2. For conditions longer than the GPU chunk size, does `_cwt_chunked` preserve enough context at chunk boundaries that coherence smoothing doesn't have seams? Verify empirically; if seams appear, use overlap-add.
3. MATLAB path resolution on Windows: `matlab -batch` picks up the first `matlab.exe` on PATH. If the user has multiple MATLAB versions installed, they should set `CADENCE_MATLAB_EXE`. Document in the module's README.
