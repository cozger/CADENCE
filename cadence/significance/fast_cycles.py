"""Fast GPU-accelerated cycle analysis for inter-brain coupling.

Replaces bycycle with a lean torch-native implementation that processes
all channels simultaneously on GPU.  Extracts per-cycle: amplitude,
period, rise-decay symmetry, burst status, and instantaneous phase.
Cross-correlates features between participants with vectorized surrogates.
Cycle-PLV computes phase-locking from cycle-derived phase landmarks,
avoiding Hilbert artifacts and naturally gating to real oscillatory cycles.

Typical speedup: 50-100× over bycycle (seconds vs minutes).
"""

import numpy as np
import torch
from scipy.signal import find_peaks

from cadence.io.resources import limit_blas_threads, pick_n_jobs


def _fft_bandpass(sigs, fs, band, device):
    """FFT bandpass filter — all channels simultaneously on GPU.

    Args:
        sigs: (C, T) float tensor on device.
        fs: sampling rate.
        band: (lo, hi) Hz.

    Returns:
        (C, T) bandpass-filtered tensor.
    """
    T = sigs.shape[1]
    X = torch.fft.rfft(sigs, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=device)

    # Smooth bandpass (raised cosine edges, 1 Hz transition)
    lo, hi = band
    trans = 1.0  # Hz transition width
    mask = torch.ones(freqs.shape, device=device)
    mask[freqs < lo - trans] = 0
    mask[freqs > hi + trans] = 0
    # Smooth edges
    rise = (freqs >= lo - trans) & (freqs < lo)
    fall = (freqs > hi) & (freqs <= hi + trans)
    mask[rise] = 0.5 * (1 + torch.cos(np.pi * (lo - freqs[rise]) / trans))
    mask[fall] = 0.5 * (1 + torch.cos(np.pi * (freqs[fall] - hi) / trans))

    X_filt = X * mask.unsqueeze(0)
    return torch.fft.irfft(X_filt, n=T, dim=1)


def extract_cycle_features(sig_raw, sig_filt, fs, band):
    """Extract per-cycle features from one channel (CPU, fast numpy).

    Uses filtered signal for cycle boundary detection, raw signal for
    feature extraction (following bycycle methodology).

    Returns:
        dict with arrays: peak_sample, period, volt_amp, time_rdsym,
        is_burst. One entry per cycle.
    """
    sig_raw = np.asarray(sig_raw)
    sig_filt = np.asarray(sig_filt)

    min_period = int(fs / band[1] * 0.5)
    max_period = int(fs / band[0] * 2.0)

    # Find troughs of filtered signal
    troughs, _ = find_peaks(-sig_filt, distance=min_period)

    if len(troughs) < 3:
        return None

    # Find peak between consecutive troughs (in RAW signal) — vectorized
    n_cycles = len(troughs) - 1
    # Pad raw signal to allow uniform-length slicing
    max_seg = int(np.diff(troughs).max()) + 1
    # Build index matrix: each row is the samples for one cycle
    starts = troughs[:-1]
    lengths = np.diff(troughs)
    # Use broadcasting: indices[i, j] = starts[i] + j, clipped to valid range
    col_idx = np.arange(max_seg)
    seg_idx = np.minimum(starts[:, None] + col_idx[None, :],
                          len(sig_raw) - 1)  # (n_cycles, max_seg)
    # Mask out-of-segment positions
    seg_vals = sig_raw[seg_idx]  # (n_cycles, max_seg)
    seg_mask = col_idx[None, :] < lengths[:, None]
    seg_vals[~seg_mask] = -np.inf  # won't be argmax
    peaks = starts + seg_vals.argmax(axis=1)

    # Period (trough-to-trough, in samples)
    periods = lengths.astype(np.float32)

    # Amplitude (peak - mean of flanking troughs, in raw signal)
    v_peak = sig_raw[peaks]
    v_trough_l = sig_raw[troughs[:-1]]
    v_trough_r = sig_raw[troughs[1:]]
    volt_amp = ((v_peak - v_trough_l) + (v_peak - v_trough_r)) / 2

    # Rise-decay symmetry (fraction of cycle in rise phase)
    rise_samp = peaks - troughs[:-1]
    time_rdsym = rise_samp.astype(np.float32) / periods

    # ── Burst detection ───────────────────────────────────────────────────
    # Band-specific criteria. Beta (>13 Hz) uses tighter thresholds because
    # short cycles (~10 samples) make monotonicity non-selective at 0.7.
    # Theta/alpha keep permissive thresholds — surrogate z-scoring
    # normalizes base rate for relative comparisons.
    # See docs/burst_detection_literature.md for literature review.
    is_beta = band[0] >= 13.0
    mono_thresh = 0.8 if is_beta else 0.7   # Cole & Voytek 2019 default for beta
    min_consec = 3 if is_beta else 2         # Sherman 2016: beta bursts ~3 cycles

    # Amplitude: above 25th percentile
    amp_ok = volt_amp > np.percentile(volt_amp, 25)

    # Period consistency: ratio of adjacent periods > 0.5
    period_ratio = np.minimum(periods[:-1], periods[1:]) / np.maximum(
        periods[:-1], periods[1:])
    period_ok = np.ones(n_cycles, dtype=bool)
    period_ok[1:] &= period_ratio > 0.5
    period_ok[:-1] &= period_ratio > 0.5

    # Period in band range
    in_band = (periods >= min_period) & (periods <= max_period)

    # Monotonicity — vectorized via the same segment matrix
    sig_diff = np.diff(sig_raw)  # (T-1,)
    # Rise: from trough to peak
    rise_starts = troughs[:-1]
    rise_lengths = (peaks - troughs[:-1])
    max_rise = int(rise_lengths.max()) + 1 if n_cycles > 0 else 1
    r_idx = np.minimum(rise_starts[:, None] + np.arange(max_rise)[None, :],
                        len(sig_diff) - 1)
    r_vals = sig_diff[r_idx]  # (n_cycles, max_rise)
    r_mask = np.arange(max_rise)[None, :] < rise_lengths[:, None]
    r_pos = (r_vals > 0) & r_mask
    mono_rise = r_pos.sum(axis=1) / np.maximum(r_mask.sum(axis=1), 1)

    # Decay: from peak to next trough
    decay_starts = peaks
    decay_lengths = (troughs[1:] - peaks)
    max_decay = int(decay_lengths.max()) + 1 if n_cycles > 0 else 1
    d_idx = np.minimum(decay_starts[:, None] + np.arange(max_decay)[None, :],
                        len(sig_diff) - 1)
    d_vals = sig_diff[d_idx]
    d_mask = np.arange(max_decay)[None, :] < decay_lengths[:, None]
    d_neg = (d_vals < 0) & d_mask
    mono_decay = d_neg.sum(axis=1) / np.maximum(d_mask.sum(axis=1), 1)

    mono = (mono_rise + mono_decay) / 2
    mono_ok = mono > mono_thresh

    is_burst = amp_ok & period_ok & in_band & mono_ok

    # Consecutive cycle filter (band-specific minimum)
    if min_consec >= 3:
        # Remove runs shorter than min_consec cycles
        padded = np.concatenate([[False], is_burst, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0]
        is_burst[:] = False
        for s, e in zip(starts, ends):
            if e - s >= min_consec:
                is_burst[s:e] = True
    else:
        # Original: remove isolated cycles (need >=1 neighbor)
        is_burst[0] = False
        is_burst[-1] = False
        isolated = is_burst.copy()
        isolated[1:] &= ~is_burst[:-1]
        isolated[:-1] &= ~is_burst[1:]
        is_burst[isolated] = False

    return {
        'peak_sample': peaks,
        'trough_sample': troughs,
        'period': periods,
        'volt_amp': volt_amp.astype(np.float32),
        'time_rdsym': time_rdsym,
        'is_burst': is_burst.astype(np.float32),
        'n_cycles': n_cycles,
    }


def _reconstruct_cycle_phase(troughs, peaks, n_cycles, fs, t_grid):
    """Reconstruct piecewise-linear phase from cycle landmarks, resampled to grid.

    Phase convention (uniform, matching Hilbert for sinusoids):
      trough[k]  → 2πk        (cycle start)
      peak[k]    → 2πk + π    (cycle midpoint)
      trough[k+1]→ 2π(k+1)   (cycle end)

    Between landmarks, phase is linearly interpolated.  This means phase
    velocity is faster during the shorter half-cycle (rise or decay),
    which correctly captures waveform asymmetry.

    Args:
        troughs: (n_cycles+1,) sample indices of troughs.
        peaks: (n_cycles,) sample indices of peaks.
        n_cycles: number of complete cycles.
        fs: sampling rate.
        t_grid: (n_grid,) time points to interpolate to.

    Returns:
        phase_grid: (n_grid,) unwrapped phase at grid points.
            exp(i * (phase_P1 - phase_P2)) wraps automatically for PLV.
    """
    # Build interleaved landmark arrays: trough, peak, trough, peak, ...
    n_landmarks = 2 * n_cycles + 1
    t_landmarks = np.empty(n_landmarks, dtype=np.float64)
    phi_landmarks = np.empty(n_landmarks, dtype=np.float64)

    for k in range(n_cycles):
        t_landmarks[2 * k] = troughs[k] / fs
        phi_landmarks[2 * k] = 2.0 * np.pi * k
        t_landmarks[2 * k + 1] = peaks[k] / fs
        phi_landmarks[2 * k + 1] = 2.0 * np.pi * k + np.pi
    # Final trough
    t_landmarks[2 * n_cycles] = troughs[n_cycles] / fs
    phi_landmarks[2 * n_cycles] = 2.0 * np.pi * n_cycles

    # Interpolate to regular grid (unwrapped — exp(i*phi) wraps automatically)
    phase_grid = np.interp(t_grid, t_landmarks, phi_landmarks).astype(np.float32)
    return phase_grid


def analyze_interbrain_cycles(p1_eeg, p2_eeg, fs, band=(4.0, 8.0),
                               n_surrogates=200, seed=42, device=None):
    """Full inter-brain theta cycle analysis.

    1. GPU bandpass all channels
    2. CPU extract per-cycle features
    3. Resample to regular grid
    4. GPU-vectorized cross-correlation + surrogates

    Args:
        p1_eeg, p2_eeg: (T, C) numpy arrays, avg-ref + z-scored.
        fs: sampling rate.
        band: (lo, hi) Hz.
        n_surrogates: number of circular-shift surrogates.
        seed: random seed.
        device: torch device.

    Returns:
        results: dict with per-channel and pooled coupling statistics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, C = p1_eeg.shape
    rate = 2.0  # resampled output rate

    # ── GPU bandpass ─────────────────────────────────────────────────────
    # Stack both participants: (2C, T)
    both = np.vstack([p1_eeg.T, p2_eeg.T])  # (2C, T)
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)
    both_filt = _fft_bandpass(both_t, fs, band, device)
    both_filt_np = both_filt.cpu().numpy()  # (2C, T)

    p1_filt = both_filt_np[:C].T   # (T, C)
    p2_filt = both_filt_np[C:].T

    # ── Per-channel cycle features (CPU) ─────────────────────────────────
    dur = T / fs
    t_grid = np.arange(0, dur, 1.0 / rate)
    n_grid = len(t_grid)

    # Preallocate resampled feature grids: (C, n_grid)
    feat_names = ['volt_amp', 'period', 'time_rdsym', 'is_burst']
    p1_feats = {f: np.zeros((C, n_grid), dtype=np.float32) for f in feat_names}
    p2_feats = {f: np.zeros((C, n_grid), dtype=np.float32) for f in feat_names}
    # Phase grids for cycle-PLV
    p1_phase = np.zeros((C, n_grid), dtype=np.float32)
    p2_phase = np.zeros((C, n_grid), dtype=np.float32)
    p1_burst_frac = np.zeros(C)
    p2_burst_frac = np.zeros(C)
    valid_ch = np.zeros(C, dtype=bool)

    for ch in range(C):
        cyc1 = extract_cycle_features(p1_eeg[:, ch], p1_filt[:, ch], fs, band)
        cyc2 = extract_cycle_features(p2_eeg[:, ch], p2_filt[:, ch], fs, band)

        if cyc1 is None or cyc2 is None:
            continue
        if cyc1['n_cycles'] < 10 or cyc2['n_cycles'] < 10:
            continue

        valid_ch[ch] = True
        p1_burst_frac[ch] = cyc1['is_burst'].mean()
        p2_burst_frac[ch] = cyc2['is_burst'].mean()

        # Resample to regular grid
        t1 = cyc1['peak_sample'] / fs
        t2 = cyc2['peak_sample'] / fs
        for f in feat_names:
            p1_feats[f][ch] = np.interp(t_grid, t1, cyc1[f])
            p2_feats[f][ch] = np.interp(t_grid, t2, cyc2[f])

        # Reconstruct cycle-derived phase
        p1_phase[ch] = _reconstruct_cycle_phase(
            cyc1['trough_sample'], cyc1['peak_sample'],
            cyc1['n_cycles'], fs, t_grid)
        p2_phase[ch] = _reconstruct_cycle_phase(
            cyc2['trough_sample'], cyc2['peak_sample'],
            cyc2['n_cycles'], fs, t_grid)

    n_valid = int(valid_ch.sum())
    if n_valid == 0:
        return {'error': 'no valid channels'}

    # ── GPU cross-correlation + surrogates ───────────────────────────────
    # Move resampled features to GPU: (C_valid, n_grid)
    valid_idx = np.where(valid_ch)[0]
    rng = np.random.default_rng(seed)
    K = n_surrogates

    results = {
        'n_valid_channels': n_valid,
        'p1_burst_frac': {int(ch): float(p1_burst_frac[ch]) for ch in valid_idx},
        'p2_burst_frac': {int(ch): float(p2_burst_frac[ch]) for ch in valid_idx},
    }

    for feat in feat_names:
        x1 = torch.as_tensor(p1_feats[feat][valid_idx],
                              dtype=torch.float32, device=device)  # (C_valid, n_grid)
        x2 = torch.as_tensor(p2_feats[feat][valid_idx],
                              dtype=torch.float32, device=device)

        # Per-channel Pearson correlation (vectorized)
        x1c = x1 - x1.mean(dim=1, keepdim=True)
        x2c = x2 - x2.mean(dim=1, keepdim=True)
        r_real = (x1c * x2c).sum(dim=1) / (
            x1c.norm(dim=1) * x2c.norm(dim=1) + 1e-10)  # (C_valid,)

        # Surrogate correlations: shift x1, compute r (all shifts at once)
        shifts = rng.integers(int(0.1 * n_grid), int(0.9 * n_grid), size=K)
        r_surr = torch.zeros(K, n_valid, device=device)
        for k in range(K):
            x1_shifted = torch.roll(x1, int(shifts[k]), dims=1)
            x1sc = x1_shifted - x1_shifted.mean(dim=1, keepdim=True)
            r_surr[k] = (x1sc * x2c).sum(dim=1) / (
                x1sc.norm(dim=1) * x2c.norm(dim=1) + 1e-10)

        # Per-channel z-scores
        surr_mean = r_surr.mean(dim=0)
        surr_std = r_surr.std(dim=0).clamp(min=1e-10)
        z_per_ch = ((r_real - surr_mean) / surr_std).cpu().numpy()
        r_per_ch = r_real.cpu().numpy()

        # Stouffer pooled z
        pooled_z = float(z_per_ch.mean() * np.sqrt(n_valid))
        mean_r = float(r_per_ch.mean())

        results[feat] = {
            'pooled_z': pooled_z,
            'mean_r': mean_r,
            'per_channel_z': {int(valid_idx[i]): float(z_per_ch[i])
                              for i in range(n_valid)},
            'per_channel_r': {int(valid_idx[i]): float(r_per_ch[i])
                              for i in range(n_valid)},
        }

    # ── Burst co-occurrence (per-channel, GPU) ───────────────────────────
    b1 = torch.as_tensor(p1_feats['is_burst'][valid_idx] > 0.5,
                          dtype=torch.float32, device=device)  # (C_valid, n_grid)
    b2 = torch.as_tensor(p2_feats['is_burst'][valid_idx] > 0.5,
                          dtype=torch.float32, device=device)

    cooc_real = (b1 * b2).mean(dim=1)  # (C_valid,)
    expected = b1.mean(dim=1) * b2.mean(dim=1)

    cooc_surr = torch.zeros(K, n_valid, device=device)
    for k in range(K):
        b1_shifted = torch.roll(b1, int(shifts[k]), dims=1)
        cooc_surr[k] = (b1_shifted * b2).mean(dim=1)

    surr_mean_b = cooc_surr.mean(dim=0)
    surr_std_b = cooc_surr.std(dim=0).clamp(min=1e-10)
    z_burst = ((cooc_real - surr_mean_b) / surr_std_b).cpu().numpy()

    # Only pool channels where both have bursts
    has_bursts = ((b1.mean(dim=1) > 0.01) & (b2.mean(dim=1) > 0.01)).cpu().numpy()
    if has_bursts.any():
        pooled_burst_z = float(z_burst[has_bursts].mean() *
                                np.sqrt(has_bursts.sum()))
    else:
        pooled_burst_z = 0.0

    results['burst_cooc'] = {
        'pooled_z': pooled_burst_z,
        'n_channels_with_bursts': int(has_bursts.sum()),
        'per_channel_z': {int(valid_idx[i]): float(z_burst[i])
                          for i in range(n_valid)},
    }

    # ── Cycle-PLV (phase-locking from cycle-derived phase) ────────────
    phi1 = torch.as_tensor(p1_phase[valid_idx],
                            dtype=torch.float32, device=device)  # (C_valid, n_grid)
    phi2 = torch.as_tensor(p2_phase[valid_idx],
                            dtype=torch.float32, device=device)
    delta = phi1 - phi2
    plv_real = (torch.cos(delta).mean(dim=1) ** 2 +
                torch.sin(delta).mean(dim=1) ** 2).sqrt()  # (C_valid,)

    plv_surr = torch.zeros(K, n_valid, device=device)
    for k in range(K):
        phi1_s = torch.roll(phi1, int(shifts[k]), dims=1)
        d_s = phi1_s - phi2
        plv_surr[k] = (torch.cos(d_s).mean(dim=1) ** 2 +
                        torch.sin(d_s).mean(dim=1) ** 2).sqrt()

    sm_p = plv_surr.mean(dim=0)
    ss_p = plv_surr.std(dim=0).clamp(min=1e-10)
    z_plv = ((plv_real - sm_p) / ss_p).cpu().numpy()
    plv_vals = plv_real.cpu().numpy()

    pooled_plv_z = float(z_plv.mean() * np.sqrt(n_valid))
    results['cycle_plv'] = {
        'pooled_z': pooled_plv_z,
        'mean_plv': float(plv_vals.mean()),
        'per_channel_z': {int(valid_idx[i]): float(z_plv[i])
                          for i in range(n_valid)},
        'per_channel_plv': {int(valid_idx[i]): float(plv_vals[i])
                            for i in range(n_valid)},
    }

    return results


# Standard EEG bands for multi-band analysis
EEG_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}


def analyze_interbrain_cycles_multiband(p1_eeg, p2_eeg, fs,
                                         bands=None,
                                         n_surrogates=200, seed=42,
                                         device=None):
    """Multi-band inter-brain cycle analysis (theta + alpha + beta).

    GPU-batched: bandpass filters for ALL bands are computed in one FFT
    pass.  Per-cycle feature extraction runs on CPU (parallelized via
    joblib across band×channel).  Cross-correlation + surrogates run on
    GPU for all bands simultaneously.

    Args:
        p1_eeg, p2_eeg: (T, C) numpy arrays, avg-ref + z-scored.
        fs: sampling rate.
        bands: dict of {name: (lo, hi)} or None for default (theta/alpha/beta).
        n_surrogates: surrogates per band.
        seed: random seed.
        device: torch device.

    Returns:
        results: dict with:
            per_band: {band_name: single-band results dict}
            combined: {feature: pooled_z across bands} via Stouffer
    """
    if bands is None:
        bands = EEG_BANDS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, C = p1_eeg.shape
    band_names = list(bands.keys())
    band_ranges = list(bands.values())
    n_bands = len(band_names)
    rate = 2.0

    # ── GPU: bandpass ALL bands at once ──────────────────────────────────
    # Stack: (n_bands * 2C, T) — each band gets its own copy of both participants
    both = np.vstack([p1_eeg.T, p2_eeg.T])  # (2C, T)
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)

    # Compute all bandpass filters simultaneously
    all_filt = {}
    for bi, (bname, brange) in enumerate(zip(band_names, band_ranges)):
        filt = _fft_bandpass(both_t, fs, brange, device)
        filt_np = filt.cpu().numpy()
        all_filt[bname] = (filt_np[:C].T, filt_np[C:].T)  # (T,C) each
    del both_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── CPU: per-band per-channel cycle features (joblib parallel) ──────
    dur = T / fs
    t_grid = np.arange(0, dur, 1.0 / rate)
    n_grid = len(t_grid)
    feat_names = ['volt_amp', 'period', 'time_rdsym', 'is_burst']

    def _extract_one(sig_raw_ch, sig_filt_ch, brange):
        """Extract + resample for one channel. Returns (feats_dict, bf, phase) or None."""
        # Resource-discipline guard: pin BLAS threads inside every joblib worker
        # to prevent the OOM mechanism documented in docs/resource_audit_2026_05_01.md
        with limit_blas_threads(1):
            cyc = extract_cycle_features(sig_raw_ch, sig_filt_ch, fs, brange)
            if cyc is None or cyc['n_cycles'] < 10:
                return None
            tc = cyc['peak_sample'] / fs
            resampled = {f: np.interp(t_grid, tc, cyc[f]) for f in feat_names}
            phase = _reconstruct_cycle_phase(
                cyc['trough_sample'], cyc['peak_sample'],
                cyc['n_cycles'], fs, t_grid)
            return resampled, float(cyc['is_burst'].mean()), phase

    # Build all (band, participant, channel) jobs
    from joblib import Parallel, delayed
    jobs = []
    job_keys = []  # (bname, 'p1'|'p2', ch)
    for bname in band_names:
        p1_filt, p2_filt = all_filt[bname]
        brange = bands[bname]
        for ch in range(C):
            jobs.append(delayed(_extract_one)(p1_eeg[:, ch], p1_filt[:, ch], brange))
            job_keys.append((bname, 'p1', ch))
            jobs.append(delayed(_extract_one)(p2_eeg[:, ch], p2_filt[:, ch], brange))
            job_keys.append((bname, 'p2', ch))

    # Adaptive cap (extract_cycle_features peaks ~50-80 MB per worker on a
    # 30-min EEG channel; 0.1 GB is the conservative budget per audit P0.1).
    n_jobs = pick_n_jobs(per_worker_ram_gb=0.1, requested=-1,
                          max_jobs_hard_cap=len(jobs))
    results_list = Parallel(n_jobs=n_jobs, prefer='threads')(jobs)

    # Unpack into band_data structure
    band_data = {}
    for bname in band_names:
        p1_f = {f: np.zeros((C, n_grid), dtype=np.float32) for f in feat_names}
        p2_f = {f: np.zeros((C, n_grid), dtype=np.float32) for f in feat_names}
        valid = np.zeros(C, dtype=bool)
        p1_bf = np.zeros(C)
        p2_bf = np.zeros(C)
        p1_ph = np.zeros((C, n_grid), dtype=np.float32)
        p2_ph = np.zeros((C, n_grid), dtype=np.float32)
        band_data[bname] = {
            'p1': p1_f, 'p2': p2_f, 'valid': valid,
            'p1_bf': p1_bf, 'p2_bf': p2_bf,
            'p1_phase': p1_ph, 'p2_phase': p2_ph,
        }

    for idx, key in enumerate(job_keys):
        bname, participant, ch = key
        res = results_list[idx]
        if res is None:
            continue
        resampled, bf, phase = res
        bd = band_data[bname]
        for f in feat_names:
            bd[participant][f][ch] = resampled[f]
        if participant == 'p1':
            bd['p1_bf'][ch] = bf
            bd['p1_phase'][ch] = phase
        else:
            bd['p2_bf'][ch] = bf
            bd['p2_phase'][ch] = phase

    # Mark channels valid only if BOTH participants have data
    for bname in band_names:
        bd = band_data[bname]
        for ch in range(C):
            has_p1 = np.any(bd['p1']['volt_amp'][ch] != 0)
            has_p2 = np.any(bd['p2']['volt_amp'][ch] != 0)
            bd['valid'][ch] = has_p1 and has_p2

    del all_filt

    # ── GPU: cross-correlation + surrogates for ALL bands at once ────────
    rng = np.random.default_rng(seed)
    K = n_surrogates
    shifts = rng.integers(int(0.1 * n_grid), int(0.9 * n_grid), size=K)

    per_band = {}
    for bname in band_names:
        bd = band_data[bname]
        valid = bd['valid']
        valid_idx = np.where(valid)[0]
        n_valid = len(valid_idx)

        result = {
            'n_valid_channels': n_valid,
            'p1_burst_frac': {int(ch): float(bd['p1_bf'][ch]) for ch in valid_idx},
            'p2_burst_frac': {int(ch): float(bd['p2_bf'][ch]) for ch in valid_idx},
        }

        if n_valid == 0:
            for f in feat_names:
                result[f] = {'pooled_z': 0, 'mean_r': 0,
                             'per_channel_z': {}, 'per_channel_r': {}}
            result['burst_cooc'] = {'pooled_z': 0, 'n_channels_with_bursts': 0,
                                    'per_channel_z': {}}
            result['cycle_plv'] = {'pooled_z': 0, 'mean_plv': 0,
                                   'per_channel_z': {}, 'per_channel_plv': {}}
            per_band[bname] = result
            continue

        for feat in feat_names:
            x1 = torch.as_tensor(bd['p1'][feat][valid_idx],
                                  dtype=torch.float32, device=device)
            x2 = torch.as_tensor(bd['p2'][feat][valid_idx],
                                  dtype=torch.float32, device=device)

            x1c = x1 - x1.mean(dim=1, keepdim=True)
            x2c = x2 - x2.mean(dim=1, keepdim=True)
            r_real = (x1c * x2c).sum(dim=1) / (
                x1c.norm(dim=1) * x2c.norm(dim=1) + 1e-10)

            r_surr = torch.zeros(K, n_valid, device=device)
            for k in range(K):
                x1s = torch.roll(x1, int(shifts[k]), dims=1)
                x1sc = x1s - x1s.mean(dim=1, keepdim=True)
                r_surr[k] = (x1sc * x2c).sum(dim=1) / (
                    x1sc.norm(dim=1) * x2c.norm(dim=1) + 1e-10)

            sm = r_surr.mean(dim=0)
            ss = r_surr.std(dim=0).clamp(min=1e-10)
            z_ch = ((r_real - sm) / ss).cpu().numpy()
            r_ch = r_real.cpu().numpy()

            result[feat] = {
                'pooled_z': float(z_ch.mean() * np.sqrt(n_valid)),
                'mean_r': float(r_ch.mean()),
                'per_channel_z': {int(valid_idx[i]): float(z_ch[i])
                                  for i in range(n_valid)},
                'per_channel_r': {int(valid_idx[i]): float(r_ch[i])
                                  for i in range(n_valid)},
            }

        # Burst co-occurrence
        b1 = torch.as_tensor(bd['p1']['is_burst'][valid_idx] > 0.5,
                              dtype=torch.float32, device=device)
        b2 = torch.as_tensor(bd['p2']['is_burst'][valid_idx] > 0.5,
                              dtype=torch.float32, device=device)
        cooc_real = (b1 * b2).mean(dim=1)
        cooc_surr = torch.zeros(K, n_valid, device=device)
        for k in range(K):
            cooc_surr[k] = (torch.roll(b1, int(shifts[k]), dims=1) * b2).mean(dim=1)
        sm_b = cooc_surr.mean(dim=0)
        ss_b = cooc_surr.std(dim=0).clamp(min=1e-10)
        z_burst = ((cooc_real - sm_b) / ss_b).cpu().numpy()
        has_bursts = ((b1.mean(dim=1) > 0.01) & (b2.mean(dim=1) > 0.01)).cpu().numpy()
        pooled_bz = (float(z_burst[has_bursts].mean() * np.sqrt(has_bursts.sum()))
                     if has_bursts.any() else 0.0)
        result['burst_cooc'] = {
            'pooled_z': pooled_bz,
            'n_channels_with_bursts': int(has_bursts.sum()),
            'per_channel_z': {int(valid_idx[i]): float(z_burst[i])
                              for i in range(n_valid)},
        }

        # Cycle-PLV
        phi1 = torch.as_tensor(bd['p1_phase'][valid_idx],
                                dtype=torch.float32, device=device)
        phi2 = torch.as_tensor(bd['p2_phase'][valid_idx],
                                dtype=torch.float32, device=device)
        delta = phi1 - phi2
        plv_real = (torch.cos(delta).mean(dim=1) ** 2 +
                    torch.sin(delta).mean(dim=1) ** 2).sqrt()

        plv_surr = torch.zeros(K, n_valid, device=device)
        for k in range(K):
            phi1_s = torch.roll(phi1, int(shifts[k]), dims=1)
            d_s = phi1_s - phi2
            plv_surr[k] = (torch.cos(d_s).mean(dim=1) ** 2 +
                            torch.sin(d_s).mean(dim=1) ** 2).sqrt()

        sm_p = plv_surr.mean(dim=0)
        ss_p = plv_surr.std(dim=0).clamp(min=1e-10)
        z_plv = ((plv_real - sm_p) / ss_p).cpu().numpy()
        plv_vals = plv_real.cpu().numpy()

        result['cycle_plv'] = {
            'pooled_z': float(z_plv.mean() * np.sqrt(n_valid)),
            'mean_plv': float(plv_vals.mean()),
            'per_channel_z': {int(valid_idx[i]): float(z_plv[i])
                              for i in range(n_valid)},
            'per_channel_plv': {int(valid_idx[i]): float(plv_vals[i])
                                for i in range(n_valid)},
        }

        per_band[bname] = result
        # P2.1 — release per-band GPU tensors before next band starts so the
        # peak VRAM stays bounded across theta/alpha/beta accumulation.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Stouffer combination across bands ────────────────────────────────
    combined = {}
    for feat in feat_names:
        band_zs = [per_band[b][feat].get('pooled_z', 0) for b in band_names]
        band_rs = [per_band[b][feat].get('mean_r', 0) for b in band_names]
        combined[feat] = {
            'stouffer_z': float(np.mean(band_zs) * np.sqrt(n_bands)),
            'mean_r': float(np.mean(band_rs)),
            'per_band_z': {b: per_band[b][feat].get('pooled_z', 0) for b in band_names},
        }

    burst_zs = [per_band[b].get('burst_cooc', {}).get('pooled_z', 0) for b in band_names]
    combined['burst_cooc'] = {
        'stouffer_z': float(np.mean(burst_zs) * np.sqrt(n_bands)),
        'per_band_z': {b: per_band[b].get('burst_cooc', {}).get('pooled_z', 0)
                       for b in band_names},
    }

    plv_zs = [per_band[b].get('cycle_plv', {}).get('pooled_z', 0) for b in band_names]
    plv_vals = [per_band[b].get('cycle_plv', {}).get('mean_plv', 0) for b in band_names]
    combined['cycle_plv'] = {
        'stouffer_z': float(np.mean(plv_zs) * np.sqrt(n_bands)),
        'mean_plv': float(np.mean(plv_vals)),
        'per_band_z': {b: per_band[b].get('cycle_plv', {}).get('pooled_z', 0)
                       for b in band_names},
    }

    return {'per_band': per_band, 'combined': combined}


# ── GPU-batched cross-product surrogates ──────────────────────────────

def _gpu_cross_product_surrogates(p1v, p2v, smooth_samples, n_surrogates, rng, device):
    """GPU-batched cross-product z-score for pre-computed volt_amp arrays.

    Args:
        p1v, p2v: (C_v, N) z-scored volt_amp arrays.
        smooth_samples: Gaussian sigma in samples.
        n_surrogates: number of circular-shift surrogates.
        rng: numpy random generator.
        device: torch device.

    Returns:
        z: (N,) z-scored cross-product timecourse.
    """
    C_v, N = p1v.shape
    shifts = rng.integers(int(0.1 * N), int(0.9 * N), size=n_surrogates)

    use_gpu = (device is not None and
               str(device) != 'cpu' and torch.cuda.is_available())

    if use_gpu:
        p1_t = torch.as_tensor(p1v, dtype=torch.float32, device=device)  # (C_v, N)
        p2_t = torch.as_tensor(p2v, dtype=torch.float32, device=device)

        # Cross-product: mean across channels
        cp_t = (p1_t * p2_t).mean(dim=0)  # (N,)

        # Gaussian smoothing kernel
        if smooth_samples > 0:
            ks = int(6 * smooth_samples) | 1
            t_k = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
            kernel = torch.exp(-0.5 * (t_k / smooth_samples) ** 2)
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            pad = ks // 2

            def _sm(x):
                if x.dim() == 1:
                    return torch.nn.functional.conv1d(
                        x.view(1, 1, -1), kernel, padding=pad).view(-1)
                return torch.nn.functional.conv1d(
                    x.unsqueeze(1), kernel, padding=pad).squeeze(1)

            cp_t = _sm(cp_t)

        # Batched surrogates via index gathering
        shifts_t = torch.as_tensor(shifts, dtype=torch.long, device=device)
        base_idx = torch.arange(N, device=device)
        shifted_idx = (base_idx[None, :] - shifts_t[:, None]) % N  # (n_surr, N)

        # p1_t is (C_v, N), index along dim=1 for each surrogate
        # p1_t[:, shifted_idx] -> (C_v, n_surr, N)
        surr = (p1_t[:, shifted_idx] * p2_t[:, None, :]).mean(dim=0)  # (n_surr, N)

        if smooth_samples > 0:
            surr = _sm(surr)

        sm = surr.mean(dim=0)
        ss = torch.clamp(surr.std(dim=0), min=1e-10)
        z = ((cp_t - sm) / ss).cpu().numpy()
    else:
        # CPU fallback
        from scipy.ndimage import gaussian_filter1d as gf1d
        cp = (p1v * p2v).mean(axis=0)
        if smooth_samples > 0:
            cp = gf1d(cp, sigma=smooth_samples)

        surr = np.zeros((n_surrogates, N))
        for k in range(n_surrogates):
            s = (np.roll(p1v, int(shifts[k]), axis=1) * p2v).mean(axis=0)
            if smooth_samples > 0:
                s = gf1d(s, sigma=smooth_samples)
            surr[k] = s

        sm = surr.mean(axis=0)
        ss = np.maximum(surr.std(axis=0), 1e-10)
        z = (cp - sm) / ss

    return z


def extract_all_volt_amp(eeg, fs, bands=None, feature_rate=2.0, device=None):
    """Pre-compute per-band per-channel volt_amp timecourses for one participant.

    Returns dict of {band_name: {'va': (C_v, N), 'valid': list_of_ch_indices}}.
    Used by pseudo-dyad FPR to avoid redundant cycle extraction.
    """
    if bands is None:
        bands = EEG_BANDS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, C = eeg.shape
    dur = T / fs
    t_grid = np.arange(0, dur, 1.0 / feature_rate)
    N = len(t_grid)

    # joblib (repo convention per feedback_joblib_loky) replaces the legacy
    # ThreadPoolExecutor here so the resource discipline is consistent.
    from joblib import Parallel, delayed

    # GPU bandpass all bands
    eeg_t = torch.as_tensor(eeg.T, dtype=torch.float32, device=device)  # (C, T)
    band_filt = {}
    for band_name, (lo, hi) in bands.items():
        filt = _fft_bandpass(eeg_t, fs, (lo, hi), device).cpu().numpy().T  # (T, C)
        band_filt[band_name] = filt
        # P2.1 — release per-band GPU tensors before next band starts.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    del eeg_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Parallel cycle extraction across all bands × channels
    def _extract(band_name, ch):
        # BLAS pin per audit P0.2 — extract_cycle_features uses np.percentile,
        # np.diff, scipy.find_peaks; without pinning each thread spawns ~16
        # BLAS threads.
        with limit_blas_threads(1):
            filt = band_filt[band_name]
            cyc = extract_cycle_features(eeg[:, ch], filt[:, ch], fs, bands[band_name])
            if cyc is not None and cyc['n_cycles'] >= 10:
                return (band_name, ch, np.interp(t_grid, cyc['peak_sample'] / fs, cyc['volt_amp']))
            return None

    tasks = [(bn, ch) for bn in bands for ch in range(C)]
    n_jobs = pick_n_jobs(per_worker_ram_gb=0.15, requested=8,
                          max_jobs_hard_cap=len(tasks))
    raw_results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_extract)(bn, ch) for bn, ch in tasks)
    ch_results = {}  # {band: {ch: va}}
    for r in raw_results:
        if r is not None:
            bn, ch, va = r
            ch_results.setdefault(bn, {})[ch] = va

    result = {}
    for band_name in bands:
        ch_va = ch_results.get(band_name, {})
        valid = sorted(ch_va.keys())
        if len(valid) >= 3:
            va_valid = np.stack([ch_va[ch] for ch in valid])
            for c in range(len(valid)):
                mu, sd = va_valid[c].mean(), max(va_valid[c].std(), 1e-8)
                va_valid[c] = (va_valid[c] - mu) / sd
            result[band_name] = {'va': va_valid, 'valid': valid}
        else:
            result[band_name] = {'va': np.zeros((0, N), dtype=np.float32), 'valid': []}

    result['times'] = t_grid
    result['feature_rate'] = feature_rate
    return result


def eeg_coupling_from_precomputed(p1_precomp, p2_precomp,
                                   smooth_samples=3, n_surrogates=100,
                                   seed=None, device=None):
    """Compute EEG coupling from pre-computed volt_amp (skips cycle extraction).

    Args:
        p1_precomp, p2_precomp: dicts from extract_all_volt_amp().
        smooth_samples, n_surrogates, seed: surrogate parameters.
        device: torch device for GPU surrogates.

    Returns:
        Same structure as eeg_coupling_timecourse().
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rng = np.random.default_rng(seed)
    t_grid = p1_precomp['times']
    N = len(t_grid)
    bands = {k: v for k, v in p1_precomp.items()
             if k not in ('times', 'feature_rate') and isinstance(v, dict)}

    band_results = {}
    for band_name, p1_bd in bands.items():
        p2_bd = p2_precomp.get(band_name, {'va': np.zeros((0, N)), 'valid': []})
        p1v = p1_bd['va']
        p2v = p2_bd['va']

        # Trim to min channels AND min time (pseudo-dyad sessions differ in length)
        n_ch_min = min(p1v.shape[0], p2v.shape[0])
        n_t_min = min(p1v.shape[1], p2v.shape[1])
        if n_ch_min < 3 or n_t_min < 20:
            band_results[band_name] = {'z': np.zeros(N), 'mask': np.zeros(N, dtype=bool),
                                        'mean_z': 0.0, 'n_valid': 0}
            continue

        p1v = p1v[:n_ch_min, :n_t_min]
        p2v = p2v[:n_ch_min, :n_t_min]
        n_min = n_ch_min

        z = _gpu_cross_product_surrogates(
            p1v, p2v, smooth_samples, n_surrogates, rng, device)
        mask = z > 2.0

        band_results[band_name] = {
            'z': z, 'mask': mask, 'mean_z': float(z.mean()),
            'n_valid': n_min,
        }
        # P2.1 — release per-band GPU tensors before next band starts.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Combined: Stouffer across bands
    band_zs = [band_results[b]['z'] for b in bands if band_results[b]['n_valid'] > 0]
    n_bands_valid = len(band_zs)
    if n_bands_valid > 0:
        combined_z = np.mean(band_zs, axis=0) * np.sqrt(n_bands_valid)
        combined_mask = combined_z > 2.0
    else:
        combined_z = np.zeros(N)
        combined_mask = np.zeros(N, dtype=bool)

    return {
        'times': t_grid,
        'feature_rate': p1_precomp.get('feature_rate', 2.0),
        'smooth_samples': smooth_samples,
        'per_band': band_results,
        'combined': {
            'z': combined_z, 'mask': combined_mask,
            'mean_z': float(combined_z.mean()),
            'coupling_fraction': float(combined_mask.mean()),
        },
    }


# ── Burst grid extraction (shared by coupling + coincidence) ─────────

def extract_burst_grids(p1_eeg, p2_eeg, fs, t_grid, bands=None, device=None):
    """Extract per-band per-channel burst and volt_amp grids for both participants.

    GPU bandpass + threaded cycle extraction.  Resamples is_burst and volt_amp
    to ``t_grid`` (in EEG-local seconds, i.e. 0-based).

    Args:
        p1_eeg, p2_eeg: (T, C) numpy arrays, avg-ref + z-scored.
        fs: EEG sampling rate (Hz).
        t_grid: (N,) target time grid in seconds (EEG-local, 0-based).
        bands: dict of {name: (lo, hi)} or None for default (theta/alpha/beta).
        device: torch device.

    Returns:
        dict of {band_name: {
            'p1_burst': (C_valid, N) bool ndarray,
            'p2_burst': (C_valid, N) bool ndarray,
            'p1_va':    (C_valid, N) float32 ndarray,
            'p2_va':    (C_valid, N) float32 ndarray,
            'valid_channels': list of int (channel indices valid for both),
        }}
    """
    if bands is None:
        bands = EEG_BANDS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # joblib (repo convention) replaces ThreadPoolExecutor here. The audit
    # P0.2 noted this site is the V11 scaffold's hottest CPU stage, called
    # 4-wide from _run_scaffold_v11.py --all; without resource discipline the
    # compounded thread count was ~4 × 8 × 16 BLAS = 512 — the OOM trigger.
    from joblib import Parallel, delayed

    T, C = p1_eeg.shape
    N = len(t_grid)

    # GPU bandpass ALL bands at once
    both = np.vstack([p1_eeg.T, p2_eeg.T])  # (2C, T)
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)

    band_filtered = {}
    for band_name, (lo, hi) in bands.items():
        filt = _fft_bandpass(both_t, fs, (lo, hi), device).cpu().numpy()
        band_filtered[band_name] = (filt[:C].T, filt[C:].T)  # (T, C) each
        # P2.1 — release per-band GPU tensors before next band starts.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    del both_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Parallel cycle extraction: 3 bands × 14 channels × 2 participants = 84 tasks
    def _extract_one(band_name, ch, person_eeg, person_filt, band, person):
        with limit_blas_threads(1):
            cyc = extract_cycle_features(person_eeg[:, ch], person_filt[:, ch], fs, band)
            if cyc is not None and cyc['n_cycles'] >= 10:
                tc = cyc['peak_sample'] / fs
                va = np.interp(t_grid, tc, cyc['volt_amp'])
                burst = np.interp(t_grid, tc, cyc['is_burst']) > 0.5
                return (person, band_name, ch, va, burst)
            return (person, None, None, None, None)

    tasks = []
    for band_name, (lo, hi) in bands.items():
        p1_filt, p2_filt = band_filtered[band_name]
        for ch in range(C):
            tasks.append((band_name, ch, p1_eeg, p1_filt, (lo, hi), 'p1'))
            tasks.append((band_name, ch, p2_eeg, p2_filt, (lo, hi), 'p2'))

    n_jobs = pick_n_jobs(per_worker_ram_gb=0.15, requested=8,
                          max_jobs_hard_cap=len(tasks))
    raw = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_extract_one)(band_name, ch, eeg, filt, band, person)
        for band_name, ch, eeg, filt, band, person in tasks)

    # {person: {band: {ch: (va, burst)}}}
    extracted = {'p1': {}, 'p2': {}}
    for person, bn, ch, va, burst in raw:
        if bn is not None:
            extracted[person].setdefault(bn, {})[ch] = (va, burst)

    # Assemble per-band grids (only channels valid for BOTH participants)
    grids = {}
    for band_name in bands:
        p1_ch = extracted['p1'].get(band_name, {})
        p2_ch = extracted['p2'].get(band_name, {})
        valid = sorted(set(p1_ch.keys()) & set(p2_ch.keys()))

        if not valid:
            grids[band_name] = {
                'p1_burst': np.zeros((0, N), dtype=bool),
                'p2_burst': np.zeros((0, N), dtype=bool),
                'p1_va': np.zeros((0, N), dtype=np.float32),
                'p2_va': np.zeros((0, N), dtype=np.float32),
                'valid_channels': [],
            }
            continue

        grids[band_name] = {
            'p1_burst': np.stack([p1_ch[ch][1] for ch in valid]),
            'p2_burst': np.stack([p2_ch[ch][1] for ch in valid]),
            'p1_va': np.stack([p1_ch[ch][0] for ch in valid]).astype(np.float32),
            'p2_va': np.stack([p2_ch[ch][0] for ch in valid]).astype(np.float32),
            'valid_channels': valid,
        }

    return grids


# ── Time-resolved coupling ────────────────────────────────────────────

def eeg_coupling_timecourse(p1_eeg, p2_eeg, fs, bands=None,
                             smooth_samples=3, n_surrogates=100,
                             feature_rate=2.0, seed=42, device=None):
    """Time-resolved inter-brain coupling from cycle volt_amp timecourses.

    Extracts per-cycle volt_amp, resamples to feature_rate, computes
    cross-product with light smoothing and surrogate-calibrated z-scores.

    Args:
        p1_eeg, p2_eeg: (T, C) numpy arrays, avg-ref + z-scored.
        fs: EEG sampling rate.
        bands: dict of {name: (lo, hi)} or None for default.
        smooth_samples: Gaussian sigma in samples at feature_rate.
            3 samples at 2 Hz = 1.5s sigma ~ 3s resolution.
        n_surrogates: circular-shift surrogates.
        feature_rate: Hz for resampled feature timecourse.
        seed: random seed.
        device: torch device.

    Returns:
        dict with:
            times: (N,) seconds.
            per_band: {band: {z: (N,), mask: (N,), mean_z: float}}
            combined: {z: (N,), mask: (N,), mean_z: float}
    """
    if bands is None:
        bands = EEG_BANDS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, C = p1_eeg.shape
    dur = T / fs
    t_grid = np.arange(0, dur, 1.0 / feature_rate)
    N = len(t_grid)
    rng = np.random.default_rng(seed)

    # Extract burst grids + volt_amp via shared function
    grids = extract_burst_grids(p1_eeg, p2_eeg, fs, t_grid, bands, device)

    band_results = {}
    for band_name in bands:
        bg = grids[band_name]
        valid = bg['valid_channels']

        if len(valid) < 3:
            band_results[band_name] = {'z': np.zeros(N), 'mask': np.zeros(N, dtype=bool),
                                        'mean_z': 0.0, 'n_valid': 0}
            continue

        p1v = bg['p1_va'].copy()
        p2v = bg['p2_va'].copy()
        C_v = len(valid)

        # Z-score per channel
        for c in range(C_v):
            for arr in [p1v, p2v]:
                mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
                arr[c] = (arr[c] - mu) / sd

        # Cross-product + surrogates (GPU-batched)
        z = _gpu_cross_product_surrogates(
            p1v, p2v, smooth_samples, n_surrogates, rng, device)

        # Significance mask: z > 2 (one-tailed)
        mask = z > 2.0

        band_results[band_name] = {
            'z': z,
            'mask': mask,
            'mean_z': float(z.mean()),
            'n_valid': C_v,
            'per_channel_valid': valid,
        }
        # P2.1 — release per-band GPU tensors before next band starts.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Combined: Stouffer across bands
    band_zs = [band_results[b]['z'] for b in bands if band_results[b]['n_valid'] > 0]
    n_bands_valid = len(band_zs)

    if n_bands_valid > 0:
        combined_z = np.mean(band_zs, axis=0) * np.sqrt(n_bands_valid)
        combined_mask = combined_z > 2.0
    else:
        combined_z = np.zeros(N)
        combined_mask = np.zeros(N, dtype=bool)

    return {
        'times': t_grid,
        'feature_rate': feature_rate,
        'smooth_samples': smooth_samples,
        'per_band': band_results,
        'combined': {
            'z': combined_z,
            'mask': combined_mask,
            'mean_z': float(combined_z.mean()),
            'coupling_fraction': float(combined_mask.mean()),
        },
    }
