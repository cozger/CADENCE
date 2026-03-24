"""Coherence-based temporal localization (V3, GPU-accelerated).

Uses wavelet coherence + Canonical Coherence (CaCoh) for maximum
sensitivity to sparse, transient coupling between multivariate signals.

Pipeline:
  1. Morlet CWT on both participants' matched channels (GPU conv1d)
  2. Time-smoothed cross-spectral density matrices (Gaussian conv1d)
  3. CaCoh: SVD of whitened cross-spectral density → scalar coherence
  4. Surrogate calibration via circular-shift surrogates
  5. Threshold → binary coupling mask

Key advantages over Welch MSC:
  - Wavelet: frequency-adaptive windows (long at low f, short at high f)
  - CaCoh: optimal spatial filtering via SVD, no dilution from null channels
  - GPU-batched: all channels × frequencies × surrogates in parallel
"""

import numpy as np
import torch
import torch.nn.functional as F


def _get_device(cfg):
    """Resolve torch device from config, with CUDA fallback to CPU."""
    device_str = cfg.get('device', 'cuda') if cfg else 'cuda'
    if isinstance(device_str, torch.device):
        return device_str
    if device_str == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Morlet CWT (torch-native, complex output on device)
# ---------------------------------------------------------------------------

def _morlet_cwt(x, fs, center_freqs, n_cycles=5, device=None):
    """Morlet continuous wavelet transform, all channels batched on GPU.

    Args:
        x: (C, N) float32 tensor on device — multichannel signal.
        fs: sampling rate in Hz.
        center_freqs: (F,) array of center frequencies in Hz.
        n_cycles: number of cycles (scalar or per-frequency array).
        device: torch device.

    Returns:
        W: (C, N, F) complex64 tensor on device — wavelet coefficients.
    """
    if device is None:
        device = x.device
    C, N = x.shape
    F_n = len(center_freqs)

    if np.isscalar(n_cycles):
        n_cycles_arr = np.full(F_n, float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=np.float64)

    W_real = torch.zeros(C, N, F_n, device=device)
    W_imag = torch.zeros(C, N, F_n, device=device)

    # x reshaped for conv1d: (C, 1, N) — each channel is a separate batch
    x_3d = x.unsqueeze(1)  # (C, 1, N)

    for fi, freq in enumerate(center_freqs):
        sigma = n_cycles_arr[fi] / (2.0 * np.pi * freq)
        half_len = int(np.ceil(4.0 * sigma * fs))
        t = np.arange(-half_len, half_len + 1) / fs

        # Complex Morlet wavelet, unit energy normalized
        gaussian = np.exp(-t ** 2 / (2.0 * sigma ** 2))
        w_real = (gaussian * np.cos(2.0 * np.pi * freq * t)).astype(np.float32)
        w_imag = (gaussian * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        energy = np.sqrt(np.sum(w_real ** 2 + w_imag ** 2))
        if energy > 0:
            w_real /= energy
            w_imag /= energy

        L = len(t)
        # conv1d weight: (1, 1, L) — same kernel for all C channels
        kr = torch.tensor(w_real[::-1].copy(), device=device).reshape(1, 1, L)
        ki = torch.tensor(w_imag[::-1].copy(), device=device).reshape(1, 1, L)

        # Symmetric padding
        pad_left = (L - 1) // 2
        pad_right = L - 1 - pad_left
        x_pad = F.pad(x_3d, (pad_left, pad_right), mode='reflect')

        # Conv1d with batch=C channels (each channel convolved independently)
        W_real[:, :, fi] = F.conv1d(x_pad, kr).squeeze(1)  # (C, N)
        W_imag[:, :, fi] = F.conv1d(x_pad, ki).squeeze(1)

    # Combine to complex
    W = torch.complex(W_real, W_imag)  # (C, N, F)
    return W


# ---------------------------------------------------------------------------
# Time-smoothed cross-spectral density
# ---------------------------------------------------------------------------

def _gaussian_smooth_1d(x, sigma_samples, dim=-2):
    """1D Gaussian smoothing along a specified dimension.

    Args:
        x: tensor of any shape.
        sigma_samples: Gaussian sigma in samples (can be fractional).
        dim: dimension to smooth along.

    Returns:
        Smoothed tensor (same shape).
    """
    if sigma_samples < 0.5:
        return x

    # Kernel: ±3σ (always real-valued)
    half = int(np.ceil(3.0 * sigma_samples))
    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    k = torch.arange(-half, half + 1, device=x.device, dtype=real_dtype)
    kernel = torch.exp(-k ** 2 / (2.0 * sigma_samples ** 2))
    kernel = kernel / kernel.sum()

    # Move target dim to last position for conv1d
    x_moved = x.movedim(dim, -1)
    orig_shape = x_moved.shape
    # Flatten all dims except last into batch
    x_flat = x_moved.reshape(-1, 1, orig_shape[-1])

    L = len(kernel)
    pad = L // 2
    kernel_w = kernel.reshape(1, 1, L)

    if x.is_complex():
        # Smooth real and imag separately
        x_r = F.pad(x_flat.real, (pad, pad), mode='reflect')
        x_i = F.pad(x_flat.imag, (pad, pad), mode='reflect')
        out_r = F.conv1d(x_r, kernel_w)
        out_i = F.conv1d(x_i, kernel_w)
        out_flat = torch.complex(out_r, out_i)
    else:
        x_pad = F.pad(x_flat, (pad, pad), mode='reflect')
        out_flat = F.conv1d(x_pad, kernel_w)

    out = out_flat.reshape(orig_shape)
    return out.movedim(-1, dim)


def _smoothed_cross_spectra(W_p1, W_p2, fs, center_freqs, n_smooth_cycles=5):
    """Compute time-smoothed cross-spectral density matrices.

    Args:
        W_p1, W_p2: (C, T, F) complex tensors — wavelet coefficients.
        fs: sampling rate in Hz.
        center_freqs: (F,) array — frequencies for σ_t scaling.
        n_smooth_cycles: smoothing width in wavelet cycles.

    Returns:
        S_xy: (F, T, C, C) complex — cross-spectral density.
        S_xx: (F, T, C, C) complex — P1 auto-spectral density.
        S_yy: (F, T, C, C) complex — P2 auto-spectral density.
    """
    C, T, F_n = W_p1.shape
    device = W_p1.device

    S_xy = torch.zeros(F_n, T, C, C, device=device, dtype=torch.complex64)
    S_xx = torch.zeros(F_n, T, C, C, device=device, dtype=torch.complex64)
    S_yy = torch.zeros(F_n, T, C, C, device=device, dtype=torch.complex64)

    for fi in range(F_n):
        freq = center_freqs[fi]
        sigma_t = n_smooth_cycles / (2.0 * np.pi * freq)  # seconds
        sigma_samp = sigma_t * fs  # samples

        # Instantaneous outer products: (C, T) → (T, C, C)
        w1 = W_p1[:, :, fi]  # (C, T)
        w2 = W_p2[:, :, fi]  # (C, T)

        # Outer product per timepoint: w1[:, t] ⊗ w2[:, t]* → (T, C, C)
        # Using einsum: 'ct,dt->tcd' (instantaneous CSD matrices)
        xy_inst = torch.einsum('ct,dt->tcd', w1, w2.conj())  # (T, C, C)
        xx_inst = torch.einsum('ct,dt->tcd', w1, w1.conj())
        yy_inst = torch.einsum('ct,dt->tcd', w2, w2.conj())

        # Gaussian smoothing along time (dim=0)
        S_xy[fi] = _gaussian_smooth_1d(xy_inst, sigma_samp, dim=0)
        S_xx[fi] = _gaussian_smooth_1d(xx_inst, sigma_samp, dim=0)
        S_yy[fi] = _gaussian_smooth_1d(yy_inst, sigma_samp, dim=0)

    return S_xy, S_xx, S_yy


# ---------------------------------------------------------------------------
# Canonical Coherence (CaCoh) via batched SVD
# ---------------------------------------------------------------------------

def _cacoh_from_spectra(S_xy, S_xx, S_yy, regularization=1e-4):
    """Compute CaCoh from cross-spectral density matrices.

    CaCoh(t,f) = max singular value of S_xx^{-1/2} S_xy S_yy^{-1/2}

    Args:
        S_xy: (F, T, C, C) complex — cross-spectral density.
        S_xx: (F, T, C, C) complex — P1 auto-spectral density.
        S_yy: (F, T, C, C) complex — P2 auto-spectral density.
        regularization: ridge for matrix inversion stability.

    Returns:
        cacoh: (F, T) float — canonical coherence per time-frequency bin.
    """
    F_n, T, C, _ = S_xy.shape
    device = S_xy.device

    # Process per-frequency to keep batch sizes manageable for cusolver
    cacoh = torch.zeros(F_n, T, device=device)

    for fi in range(F_n):
        S_xx_f = S_xx[fi]  # (T, C, C)
        S_yy_f = S_yy[fi]
        S_xy_f = S_xy[fi]

        # Add regularization to diagonal for numerical stability
        eye_C = torch.eye(C, device=device, dtype=S_xx_f.dtype) * regularization
        S_xx_f = S_xx_f + eye_C
        S_yy_f = S_yy_f + eye_C

        # Regularized inverse square root via eigendecomposition
        def _inv_sqrt(S_batch):
            eigvals, eigvecs = torch.linalg.eigh(S_batch)
            eigvals = torch.clamp(eigvals.real, min=regularization)
            inv_sqrt_vals = (1.0 / torch.sqrt(eigvals)).to(eigvecs.dtype)
            return eigvecs * inv_sqrt_vals.unsqueeze(-2) @ eigvecs.conj().transpose(-2, -1)

        try:
            S_xx_isq = _inv_sqrt(S_xx_f)  # (T, C, C)
            S_yy_isq = _inv_sqrt(S_yy_f)

            # M = S_xx^{-1/2} S_xy S_yy^{-1/2}
            M = S_xx_isq @ S_xy_f @ S_yy_isq  # (T, C, C)

            # CaCoh = largest singular value of M
            sv = torch.linalg.svdvals(M)  # (T, C)
            cacoh[fi] = sv[:, 0].real.clamp(0.0, 1.0)
        except Exception:
            # Fallback: simple MSC trace (sum of diagonal coherence)
            denom = torch.clamp(
                S_xx_f.diagonal(dim1=-2, dim2=-1).real *
                S_yy_f.diagonal(dim1=-2, dim2=-1).real, min=1e-20)
            msc_diag = S_xy_f.diagonal(dim1=-2, dim2=-1).abs().square() / denom
            cacoh[fi] = msc_diag.mean(dim=-1).clamp(0.0, 1.0)

    return cacoh


# ---------------------------------------------------------------------------
# Full wavelet CaCoh pipeline
# ---------------------------------------------------------------------------

def _wavelet_cacoh(p1_signal, p2_signal, fs, matched_channels,
                   center_freqs, n_cycles=5, n_smooth_cycles=5,
                   regularization=1e-4, device=None):
    """Compute wavelet canonical coherence timecourse.

    Args:
        p1_signal, p2_signal: (T_raw, C_all) numpy arrays at native rate.
        fs: sampling rate in Hz.
        matched_channels: list of channel indices.
        center_freqs: (F,) array of frequencies in Hz.
        n_cycles: Morlet wavelet cycles.
        n_smooth_cycles: Gaussian smoothing width in cycles.
        regularization: ridge for SVD stability.
        device: torch device.

    Returns:
        cacoh_t: (T_raw,) numpy array — band-averaged CaCoh timecourse.
        cacoh_tf: (F, T_raw) numpy array — per-frequency CaCoh.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract matched channels
    x1 = p1_signal[:, matched_channels].T  # (C, T)
    x2 = p2_signal[:, matched_channels].T

    x1_t = torch.as_tensor(np.ascontiguousarray(x1), dtype=torch.float32,
                           device=device)
    x2_t = torch.as_tensor(np.ascontiguousarray(x2), dtype=torch.float32,
                           device=device)

    # Step 1: Morlet CWT
    W_p1 = _morlet_cwt(x1_t, fs, center_freqs, n_cycles=n_cycles,
                       device=device)  # (C, T, F)
    W_p2 = _morlet_cwt(x2_t, fs, center_freqs, n_cycles=n_cycles,
                       device=device)

    # Step 2: Time-smoothed cross-spectral density
    S_xy, S_xx, S_yy = _smoothed_cross_spectra(
        W_p1, W_p2, fs, center_freqs, n_smooth_cycles=n_smooth_cycles)

    del W_p1, W_p2  # free CWT memory

    # Step 3: CaCoh via batched SVD
    cacoh_tf = _cacoh_from_spectra(S_xy, S_xx, S_yy,
                                    regularization=regularization)

    del S_xy, S_xx, S_yy

    # Band-average across frequencies
    cacoh_t = cacoh_tf.mean(dim=0)  # (T,)

    return cacoh_t.cpu().numpy(), cacoh_tf.cpu().numpy()


# ---------------------------------------------------------------------------
# Surrogate pipeline (wavelet CaCoh with P2 CWT reuse)
# ---------------------------------------------------------------------------

def _wavelet_cacoh_surrogates(p1_signal, p2_signal, fs, matched_channels,
                               center_freqs, n_surrogates=100,
                               n_cycles=5, n_smooth_cycles=5,
                               regularization=1e-4,
                               min_shift_frac=0.1, seed=42,
                               device=None):
    """Compute real + K surrogate wavelet CaCoh in GPU-batched passes.

    P2 CWT is computed once and reused across all surrogates.

    Returns:
        cacoh_real: (T,) real CaCoh timecourse.
        cacoh_surr: (K, T) surrogate CaCoh timecourses.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    K = n_surrogates
    C = len(matched_channels)

    x1_np = p1_signal[:, matched_channels].T  # (C, N)
    x2_np = p2_signal[:, matched_channels].T
    N = x1_np.shape[1]

    x1_t = torch.as_tensor(np.ascontiguousarray(x1_np), dtype=torch.float32,
                           device=device)
    x2_t = torch.as_tensor(np.ascontiguousarray(x2_np), dtype=torch.float32,
                           device=device)

    # --- P2 CWT: compute once ---
    W_p2 = _morlet_cwt(x2_t, fs, center_freqs, n_cycles=n_cycles,
                       device=device)  # (C, N, F)

    # --- Real CaCoh ---
    W_p1_real = _morlet_cwt(x1_t, fs, center_freqs, n_cycles=n_cycles,
                            device=device)
    S_xy, S_xx, S_yy = _smoothed_cross_spectra(
        W_p1_real, W_p2, fs, center_freqs, n_smooth_cycles=n_smooth_cycles)
    cacoh_tf_real = _cacoh_from_spectra(S_xy, S_xx, S_yy,
                                         regularization=regularization)
    cacoh_real = cacoh_tf_real.mean(dim=0).cpu().numpy()  # (N,)
    del W_p1_real, S_xy, S_xx, S_yy, cacoh_tf_real

    # --- Surrogate CaCoh ---
    min_shift = max(1, int(min_shift_frac * N))
    max_shift = N - min_shift
    if min_shift >= max_shift:
        min_shift, max_shift = 1, N - 1

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    shifts = torch.randint(min_shift, max_shift + 1, (K,),
                           generator=gen, device=device, dtype=torch.int32)

    # Precompute P2 cross-spectral pieces that don't change
    # S_yy only depends on P2
    F_n = len(center_freqs)
    # We need to recompute S_xy and S_xx per surrogate, but S_yy is fixed
    # Compute S_yy once
    S_yy_fixed = torch.zeros(F_n, N, C, C, device=device,
                             dtype=torch.complex64)
    for fi in range(F_n):
        freq = center_freqs[fi]
        sigma_samp = n_smooth_cycles / (2.0 * np.pi * freq) * fs
        w2 = W_p2[:, :, fi]
        yy_inst = torch.einsum('ct,dt->tcd', w2, w2.conj())
        S_yy_fixed[fi] = _gaussian_smooth_1d(yy_inst, sigma_samp, dim=0)

    cacoh_surr = np.zeros((K, N))

    # Process surrogates one at a time (CWT is the expensive part)
    for k in range(K):
        shift = int(shifts[k].item())
        x1_shifted = torch.roll(x1_t, shift, dims=1)  # (C, N)

        W_p1_k = _morlet_cwt(x1_shifted, fs, center_freqs,
                             n_cycles=n_cycles, device=device)

        # Cross-spectral density with precomputed P2
        S_xy_k = torch.zeros(F_n, N, C, C, device=device,
                             dtype=torch.complex64)
        S_xx_k = torch.zeros(F_n, N, C, C, device=device,
                             dtype=torch.complex64)
        for fi in range(F_n):
            freq = center_freqs[fi]
            sigma_samp = n_smooth_cycles / (2.0 * np.pi * freq) * fs
            w1 = W_p1_k[:, :, fi]
            w2 = W_p2[:, :, fi]
            xy_inst = torch.einsum('ct,dt->tcd', w1, w2.conj())
            xx_inst = torch.einsum('ct,dt->tcd', w1, w1.conj())
            S_xy_k[fi] = _gaussian_smooth_1d(xy_inst, sigma_samp, dim=0)
            S_xx_k[fi] = _gaussian_smooth_1d(xx_inst, sigma_samp, dim=0)

        cacoh_tf_k = _cacoh_from_spectra(S_xy_k, S_xx_k, S_yy_fixed,
                                          regularization=regularization)
        cacoh_surr[k] = cacoh_tf_k.mean(dim=0).cpu().numpy()

        del W_p1_k, S_xy_k, S_xx_k, cacoh_tf_k

    del W_p2, S_yy_fixed
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return cacoh_real, cacoh_surr


# ---------------------------------------------------------------------------
# Coupling mask generation
# ---------------------------------------------------------------------------

def coherence_coupling_mask(z_agg, z_null, target_fa=0.10,
                            min_event_s=2.0, output_rate=1.0):
    """Convert z-score to binary coupling mask with calibrated threshold."""
    from scipy.special import expit

    surr_maxes = np.max(z_null, axis=1)
    threshold = max(float(np.percentile(surr_maxes, 100 * (1 - target_fa))),
                    1.0)
    mask = z_agg > threshold

    min_samples = max(1, int(min_event_s * output_rate))
    if min_samples > 1:
        mask = _min_event_filter(mask, min_samples)

    posterior = expit(2.0 * (z_agg - threshold))
    return mask, threshold, posterior


def _min_event_filter(mask, min_samples):
    """Remove contiguous True runs shorter than min_samples."""
    result = mask.copy()
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < min_samples:
            result[s:e] = False
    return result


def _fill_gaps(mask, max_gap_samples):
    """Fill False gaps shorter than max_gap_samples between True runs."""
    result = mask.copy()
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    # Gaps are between end[i] and start[i+1]
    for i in range(len(ends) - 1):
        gap = starts[i + 1] - ends[i]
        if gap < max_gap_samples:
            result[ends[i]:starts[i + 1]] = True
    return result


def _roi_average_signals(signals, roi_map):
    """Average EEG channels within ROIs to create virtual super-channels.

    Args:
        signals: (T, C) numpy array — raw EEG at native rate.
        roi_map: dict mapping ROI names to lists of channel indices.

    Returns:
        roi_signals: (T, n_rois) numpy array — ROI-averaged signals.
    """
    T = signals.shape[0]
    roi_names = list(roi_map.keys())
    n_rois = len(roi_names)
    roi_signals = np.zeros((T, n_rois), dtype=np.float32)
    for ri, name in enumerate(roi_names):
        ch_idx = roi_map[name]
        roi_signals[:, ri] = signals[:, ch_idx].mean(axis=1)
    return roi_signals


def _spatial_cluster_filter(mask, per_channel_z, adjacency,
                            min_cluster=2, z_min=0.5):
    """Remove detections without spatially contiguous channel support.

    For each flagged window, checks that at least `min_cluster` spatially
    adjacent channels have z > z_min. Eliminates false alarms from
    isolated noise channels.

    Args:
        mask: (n_win,) boolean detection mask.
        per_channel_z: (C, n_win) per-channel z-scores.
        adjacency: (C, C) boolean adjacency matrix.
        min_cluster: minimum number of adjacent above-threshold channels.
        z_min: minimum per-channel z to count as active.

    Returns:
        filtered_mask: (n_win,) boolean with isolated detections removed.
    """
    C, n_win = per_channel_z.shape
    result = mask.copy()

    for w in np.where(mask)[0]:
        active = per_channel_z[:, w] > z_min  # channels above threshold
        if active.sum() < min_cluster:
            result[w] = False
            continue

        # Check spatial contiguity via BFS on adjacency subgraph
        active_idx = np.where(active)[0]
        visited = set()
        max_cluster = 0
        for start in active_idx:
            if start in visited:
                continue
            # BFS from this node
            queue = [start]
            cluster_size = 0
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster_size += 1
                for neighbor in active_idx:
                    if neighbor not in visited and adjacency[node, neighbor]:
                        queue.append(neighbor)
            max_cluster = max(max_cluster, cluster_size)

        if max_cluster < min_cluster:
            result[w] = False

    return result


# ---------------------------------------------------------------------------
# Full temporal localization pipeline
# ---------------------------------------------------------------------------

def coherence_temporal_localization(p1_signal, p2_signal, p1_ts, p2_ts,
                                    matched_channels, fs_native,
                                    cfg=None):
    """Wavelet CaCoh temporal localization pipeline (GPU-accelerated).

    Uses Morlet wavelet transform + canonical coherence for maximum
    sensitivity to sparse, transient coupling.

    Args:
        p1_signal, p2_signal: (T, C) numpy arrays at native rate.
        p1_ts, p2_ts: (T,) timestamps.
        matched_channels: list of channel indices.
        fs_native: native sampling rate.
        cfg: dict with config + optional 'device'.

    Returns:
        mask: (n_out,) boolean coupling mask.
        posterior: (n_out,) soft posterior [0, 1].
        out_times: (n_out,) center times in seconds.
        diagnostics: dict with pipeline metadata.
    """
    if cfg is None:
        cfg = {}

    device = _get_device(cfg)
    n_surrogates = cfg.get('n_surrogates', 100)
    target_fa = cfg.get('target_false_alarm', 0.10)
    min_event_s = cfg.get('min_event_s', 2.0)
    regularization = cfg.get('whitening_regularization', 1e-4)
    seed = cfg.get('seed', 42)
    n_cycles = cfg.get('n_wavelet_cycles', 5)
    n_smooth_cycles = cfg.get('n_smooth_cycles', 5)
    output_rate = cfg.get('output_rate', 2.0)  # Hz for decimated output

    # Frequency range: up to Nyquist/2 for the modality
    max_freq = min(fs_native / 2.0 - 0.5, 15.0)
    min_freq = max(0.5, 1.0)
    n_freqs = cfg.get('n_freqs', 10)
    center_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq),
                               n_freqs)

    C_match = len(matched_channels)

    # Align signals to common time range
    t_start = max(float(p1_ts[0]), float(p2_ts[0]))
    t_end = min(float(p1_ts[-1]), float(p2_ts[-1]))
    duration = t_end - t_start
    if duration < 10.0:
        return (np.zeros(1, dtype=bool), np.zeros(1), np.array([t_start]),
                {'method': 'skipped', 'reason': 'insufficient duration'})

    # Resample to uniform grid
    N = int(duration * fs_native)
    t_uniform = np.linspace(t_start, t_end, N)

    p1_uniform = np.stack([np.interp(t_uniform, p1_ts, p1_signal[:, ch])
                           for ch in range(p1_signal.shape[1])]).T  # (N, C_all)
    p2_uniform = np.stack([np.interp(t_uniform, p2_ts, p2_signal[:, ch])
                           for ch in range(p2_signal.shape[1])]).T

    # Compute real + surrogate CaCoh
    cacoh_real, cacoh_surr = _wavelet_cacoh_surrogates(
        p1_uniform, p2_uniform, fs_native, matched_channels,
        center_freqs, n_surrogates=n_surrogates,
        n_cycles=n_cycles, n_smooth_cycles=n_smooth_cycles,
        regularization=regularization,
        min_shift_frac=0.1, seed=seed, device=device)

    # Decimate to output rate
    dec_factor = max(1, int(fs_native / output_rate))
    cacoh_real_dec = cacoh_real[::dec_factor]
    cacoh_surr_dec = cacoh_surr[:, ::dec_factor]
    T_out = len(cacoh_real_dec)
    out_times = t_uniform[::dec_factor][:T_out]

    # Z-score against surrogates
    surr_mean = cacoh_surr_dec.mean(axis=0)  # (T_out,)
    surr_std = np.maximum(cacoh_surr_dec.std(axis=0), 1e-8)
    z_real = (cacoh_real_dec - surr_mean) / surr_std
    z_surr = (cacoh_surr_dec - surr_mean[None]) / surr_std[None]  # (K, T_out)

    # Coupling mask
    mask, threshold, posterior = coherence_coupling_mask(
        z_real, z_surr, target_fa=target_fa,
        min_event_s=min_event_s, output_rate=output_rate)

    coupling_frac = float(np.mean(mask))

    diagnostics = {
        'method': 'wavelet_cacoh_v3',
        'device': str(device),
        'n_channels': C_match,
        'n_freqs': n_freqs,
        'n_windows': T_out,
        'n_surrogates': n_surrogates,
        'fs_native': fs_native,
        'output_rate': output_rate,
        'center_freqs': center_freqs.tolist(),
        'threshold': threshold,
        'coupling_fraction': coupling_frac,
        'z_agg_mean': float(np.mean(z_real)),
        'z_agg_max': float(np.max(z_real)),
        'z_agg_p95': float(np.percentile(z_real, 95)),
        'n_significant_channels': C_match,  # CaCoh uses all channels
        'significant_channels': list(range(C_match)),
    }

    return mask, posterior, out_times, diagnostics


# ---------------------------------------------------------------------------
# Masked feature breakdown (unchanged from previous version)
# ---------------------------------------------------------------------------

def masked_feature_breakdown(dr2_perchannel, coherence_mask,
                             dr2_eval_rate, window_times,
                             feature_names=None):
    """Average per-channel dR2 within coherence-active windows."""
    C, T_dr2 = dr2_perchannel.shape
    dr2_times = np.arange(T_dr2) / dr2_eval_rate

    if len(window_times) == 0 or len(coherence_mask) == 0:
        return np.zeros(C), np.zeros(C), np.zeros(C)

    mask_interp = np.interp(dr2_times, window_times,
                            coherence_mask.astype(float))
    mask_bool = mask_interp > 0.5

    if mask_bool.sum() == 0:
        dr2_active = np.zeros(C)
    else:
        dr2_active = np.nanmean(dr2_perchannel[:, mask_bool], axis=1)

    if (~mask_bool).sum() == 0:
        dr2_inactive = np.zeros(C)
    else:
        dr2_inactive = np.nanmean(dr2_perchannel[:, ~mask_bool], axis=1)

    return dr2_active, dr2_inactive, dr2_active - dr2_inactive


# ---------------------------------------------------------------------------
# wPLI (weighted Phase Lag Index) temporal localization
# ---------------------------------------------------------------------------

def _coherence_windowed(W_p1, W_p2, win_samp, stride_samp, metric='csd'):
    """Compute windowed coherence per channel per frequency from CWT.

    Processes frequencies in chunks to stay within VRAM. The unfold
    operation on (C, T) with large T and win_samp can exceed 16 GB
    for broadband (30+ freq) analysis at 256 Hz.

    Supports metrics: 'wpli', 'csd', 'plv', 'envelope', 'combined'.

    Args:
        W_p1: (C, T, F) complex tensor — P1 wavelet coefficients.
        W_p2: (C, T, F) complex tensor — P2 wavelet coefficients.
        win_samp: window length in samples.
        stride_samp: stride between windows in samples.
        metric: coherence metric string.

    Returns:
        coh: (C, F, n_win) float tensor — coherence per channel per frequency.
        centers: (n_win,) long tensor — window center indices.
    """
    C, T, F_n = W_p1.shape
    device = W_p1.device

    # Estimate n_win for pre-allocation
    n_win = max(1, (T - win_samp) // stride_samp + 1)
    coh = torch.zeros(C, F_n, n_win, device=device)

    # Process per-frequency to control VRAM (unfold of (C, T) is manageable)
    for fi in range(F_n):
        w1 = W_p1[:, :, fi]  # (C, T) complex
        w2 = W_p2[:, :, fi]

        if metric == 'wpli':
            im = (w1 * w2.conj()).imag  # (C, T)
            im_w = im.unfold(1, win_samp, stride_samp)  # (C, n_win, W)
            num = im_w.mean(dim=-1).abs()
            den = im_w.abs().mean(dim=-1).clamp(min=1e-10)
            coh[:, fi, :] = num / den

        elif metric == 'csd':
            sxy = w1 * w2.conj()
            sr = sxy.real.unfold(1, win_samp, stride_samp)
            si = sxy.imag.unfold(1, win_samp, stride_samp)
            coh[:, fi, :] = torch.sqrt(sr.mean(-1)**2 + si.mean(-1)**2)

        elif metric == 'plv':
            sxy = w1 * w2.conj()
            sxy_n = sxy / sxy.abs().clamp(min=1e-10)
            sr = sxy_n.real.unfold(1, win_samp, stride_samp)
            si = sxy_n.imag.unfold(1, win_samp, stride_samp)
            coh[:, fi, :] = torch.sqrt(sr.mean(-1)**2 + si.mean(-1)**2)

        elif metric == 'envelope':
            e1 = w1.abs().unfold(1, win_samp, stride_samp)  # (C, n_win, W)
            e2 = w2.abs().unfold(1, win_samp, stride_samp)
            e1c = e1 - e1.mean(-1, keepdim=True)
            e2c = e2 - e2.mean(-1, keepdim=True)
            cov = (e1c * e2c).mean(-1)
            std_p = (e1c.pow(2).mean(-1) * e2c.pow(2).mean(-1)
                     ).sqrt().clamp(min=1e-10)
            coh[:, fi, :] = cov / std_p

        elif metric == 'combined':
            # PLV
            sxy = w1 * w2.conj()
            sxy_n = sxy / sxy.abs().clamp(min=1e-10)
            sr = sxy_n.real.unfold(1, win_samp, stride_samp)
            si = sxy_n.imag.unfold(1, win_samp, stride_samp)
            plv = torch.sqrt(sr.mean(-1)**2 + si.mean(-1)**2)
            del sr, si, sxy_n

            # Envelope correlation
            e1 = w1.abs().unfold(1, win_samp, stride_samp)
            e2 = w2.abs().unfold(1, win_samp, stride_samp)
            e1c = e1 - e1.mean(-1, keepdim=True)
            e2c = e2 - e2.mean(-1, keepdim=True)
            cov = (e1c * e2c).mean(-1)
            std_p = (e1c.pow(2).mean(-1) * e2c.pow(2).mean(-1)
                     ).sqrt().clamp(min=1e-10)
            env = cov / std_p
            del e1, e2, e1c, e2c

            coh[:, fi, :] = (plv + (env + 1) / 2) / 2

        elif metric == 'power_event':
            # Co-activation: both participants have elevated power simultaneously
            p1_pow = w1.abs().pow(2).unfold(1, win_samp, stride_samp)  # (C, n_win, W)
            p2_pow = w2.abs().pow(2).unfold(1, win_samp, stride_samp)
            # Per-window 75th percentile threshold (adaptive to non-stationarity)
            thr1 = p1_pow.quantile(0.75, dim=-1, keepdim=True)
            thr2 = p2_pow.quantile(0.75, dim=-1, keepdim=True)
            co_act = ((p1_pow > thr1) & (p2_pow > thr2)).float().mean(dim=-1)
            coh[:, fi, :] = co_act
            del p1_pow, p2_pow

        elif metric == 'plv_power':
            # Joint: PLV (phase) + power co-activation (amplitude events)
            # PLV component
            sxy = w1 * w2.conj()
            sxy_n = sxy / sxy.abs().clamp(min=1e-10)
            sr = sxy_n.real.unfold(1, win_samp, stride_samp)
            si = sxy_n.imag.unfold(1, win_samp, stride_samp)
            plv = torch.sqrt(sr.mean(-1)**2 + si.mean(-1)**2)
            del sr, si, sxy_n

            # Power co-activation component
            p1_pow = w1.abs().pow(2).unfold(1, win_samp, stride_samp)
            p2_pow = w2.abs().pow(2).unfold(1, win_samp, stride_samp)
            thr1 = p1_pow.quantile(0.75, dim=-1, keepdim=True)
            thr2 = p2_pow.quantile(0.75, dim=-1, keepdim=True)
            co_act = ((p1_pow > thr1) & (p2_pow > thr2)).float().mean(dim=-1)
            del p1_pow, p2_pow

            # Average (both in [0,1]): surrogate z handles correlation
            coh[:, fi, :] = (plv + co_act) / 2

        else:
            raise ValueError(f"Unknown metric: {metric}")

    starts = torch.arange(0, T - win_samp + 1, stride_samp,
                          device=device, dtype=torch.long)
    centers = starts + win_samp // 2

    return coh, centers


def _coherence_surrogates(x1, x2, fs, center_freqs, n_surrogates=100,
                          win_samp=512, stride_samp=128,
                          n_cycles=5, min_shift_frac=0.1, seed=42,
                          metric='csd', device=None):
    """Compute real + surrogate per-channel coherence via circular-shift surrogates.

    P2's CWT is computed once and reused across all surrogates.

    Args:
        x1, x2: (C, N) float32 tensors on device — P1 and P2 signals.
        fs: sampling rate in Hz.
        center_freqs: (F,) array of center frequencies in Hz.
        n_surrogates: number of circular-shift surrogates.
        win_samp: window length in samples.
        stride_samp: stride in samples.
        n_cycles: Morlet wavelet cycles.
        min_shift_frac: minimum circular shift as fraction of N.
        seed: random seed.
        metric: 'wpli', 'csd', or 'plv'.
        device: torch device.

    Returns:
        coh_real: (C, F, n_win) real coherence.
        coh_surr: (K, C, F, n_win) surrogate coherence.
        centers: (n_win,) window center indices.
    """
    if device is None:
        device = x1.device

    N = x1.shape[1]
    K = n_surrogates

    # P2 CWT: computed once
    W_p2 = _morlet_cwt(x2, fs, center_freqs, n_cycles=n_cycles,
                       device=device)  # (C, N, F)

    # Real coherence
    W_p1 = _morlet_cwt(x1, fs, center_freqs, n_cycles=n_cycles,
                       device=device)  # (C, N, F)
    coh_real, centers = _coherence_windowed(W_p1, W_p2, win_samp, stride_samp,
                                            metric=metric)
    del W_p1

    C, F_n, n_win = coh_real.shape

    # Generate circular shifts
    min_shift = max(1, int(min_shift_frac * N))
    max_shift = N - min_shift
    if min_shift >= max_shift:
        min_shift, max_shift = 1, N - 1

    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    shifts = torch.randint(min_shift, max_shift + 1, (K,), generator=gen)

    # Surrogate coherence
    coh_surr = torch.zeros(K, C, F_n, n_win, device=device)

    for k in range(K):
        x1_shifted = torch.roll(x1, int(shifts[k].item()), dims=1)
        W_p1_k = _morlet_cwt(x1_shifted, fs, center_freqs,
                             n_cycles=n_cycles, device=device)
        coh_surr[k], _ = _coherence_windowed(W_p1_k, W_p2,
                                              win_samp, stride_samp,
                                              metric=metric)
        del W_p1_k

    del W_p2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return coh_real, coh_surr, centers


def wpli_temporal_localization(p1_signal, p2_signal, fs,
                               channels=None, center_freqs=None,
                               n_surrogates=100,
                               window_s=2.0, stride_s=0.5,
                               n_cycles=5, z_threshold=None,
                               target_fa=0.05, min_event_s=2.0,
                               metric='csd', seed=42, device=None,
                               **kwargs):
    """Full wavelet coherence temporal localization pipeline.

    Pipeline:
      1. Morlet CWT at specified frequencies for both participants
      2. Per-channel windowed coherence (metric selectable)
      3. Surrogate calibration via K circular shifts
      4. Per-channel z-scoring against surrogate distribution
      5. Aggregate across channels and frequencies → z_agg timecourse
      6. Threshold → binary coupling mask

    Args:
        p1_signal, p2_signal: (T, C) numpy arrays at native rate.
        fs: sampling rate in Hz.
        channels: list of channel indices to use (None = all).
        center_freqs: (F,) array of center frequencies in Hz.
            Default: 5 log-spaced frequencies from 4-8 Hz (theta).
        n_surrogates: number of circular-shift surrogates (default 100).
        window_s: window length in seconds (default 2.0).
        stride_s: stride between windows in seconds (default 0.5).
        n_cycles: Morlet wavelet cycles. Scalar for fixed, or [min, max]
            for frequency-scaled cycles (linearly interpolated across
            center_freqs — standard for EEG time-frequency analysis).
            Default 5 (fixed).
        z_threshold: fixed z-score threshold (None = calibrate from surrogates).
        target_fa: target false alarm rate for threshold calibration (default 0.05).
        min_event_s: minimum event duration in seconds (default 2.0).
        metric: coherence metric — 'csd' (default), 'wpli', or 'plv'.
        seed: random seed.
        device: torch device.

    Returns:
        mask: (n_win,) boolean coupling mask.
        z_agg: (n_win,) aggregated z-score across channels.
        per_channel_z: (C_sel, n_win) per-channel z-scores.
        diagnostics: dict with pipeline metadata.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T_raw, C_all = p1_signal.shape

    # Default: theta band (4-8 Hz)
    if center_freqs is None:
        center_freqs = np.logspace(np.log10(4.0), np.log10(8.0), 5)

    # Channel selection
    if channels is None:
        channels = list(range(C_all))
    C_sel = len(channels)

    # Convert window/stride to samples
    win_samp = int(window_s * fs)
    stride_samp = int(stride_s * fs)

    # Resolve n_cycles: [min, max] → per-frequency linear scaling
    if isinstance(n_cycles, (list, tuple)) and len(n_cycles) == 2:
        n_cycles_arr = np.linspace(n_cycles[0], n_cycles[1],
                                    len(center_freqs))
    else:
        n_cycles_arr = n_cycles  # scalar, passed through to _morlet_cwt

    # Optional ROI averaging: reduces 14 correlated channels to 4 independent ROIs
    roi_map = kwargs.get('roi_map', None)
    if roi_map is not None:
        p1_signal = _roi_average_signals(p1_signal, roi_map)
        p2_signal = _roi_average_signals(p2_signal, roi_map)
        channels = list(range(p1_signal.shape[1]))
        C_sel = len(channels)

    # Extract selected channels and transpose to (C, T)
    x1 = torch.as_tensor(
        np.ascontiguousarray(p1_signal[:, channels].T),
        dtype=torch.float32, device=device)  # (C_sel, T)
    x2 = torch.as_tensor(
        np.ascontiguousarray(p2_signal[:, channels].T),
        dtype=torch.float32, device=device)

    # Compute real + surrogate coherence
    coh_real, coh_surr, centers = _coherence_surrogates(
        x1, x2, fs, center_freqs,
        n_surrogates=n_surrogates,
        win_samp=win_samp, stride_samp=stride_samp,
        n_cycles=n_cycles_arr, metric=metric, seed=seed, device=device)
    # coh_real: (C_sel, F, n_win)
    # coh_surr: (K, C_sel, F, n_win)

    # Z-score per (channel, frequency) then aggregate across both
    coh_real_np = coh_real.cpu().numpy()    # (C_sel, F, n_win)
    coh_surr_np = coh_surr.cpu().numpy()    # (K, C_sel, F, n_win)
    centers_np = centers.cpu().numpy()
    n_win = coh_real_np.shape[2]
    F_n = coh_real_np.shape[1]

    del coh_real, coh_surr, x1, x2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Aggregation mode: 'pooled' (aggregate first, z-score once) or
    # 'stouffer' (z-score per pair, then Stouffer combination)
    aggregation = kwargs.get('aggregation', 'pooled')

    if aggregation == 'pooled':
        # Pool coherence across channels and frequencies first,
        # then z-score the aggregate against surrogate aggregates.
        # More powerful because the single z-score uses K surrogate
        # samples of the SAME aggregate statistic.
        agg_weights = kwargs.get('aggregation_weights', 'equal')
        if agg_weights == 'snr':
            # SNR-weighted: upweight channels with stronger surrogate contrast
            ch_real = coh_real_np.mean(axis=(1, 2))         # (C,) session-avg per ch
            ch_surr = coh_surr_np.mean(axis=(0, 2, 3))     # (C,) null avg per ch
            w = np.maximum(ch_real - ch_surr, 0.0)          # (C,) positive contrast
            w /= (w.sum() + 1e-10)                          # normalize
            coh_agg_real = (coh_real_np * w[:, None, None]).sum(0).mean(0)
            coh_agg_surr = (coh_surr_np * w[None, :, None, None]).sum(1).mean(1)
        else:
            coh_agg_real = coh_real_np.mean(axis=(0, 1))    # (n_win,)
            coh_agg_surr = coh_surr_np.mean(axis=(1, 2))    # (K, n_win)

        surr_mean_agg = coh_agg_surr.mean(axis=0)       # (n_win,)
        surr_std_agg = np.maximum(coh_agg_surr.std(axis=0), 1e-10)

        z_agg = (coh_agg_real - surr_mean_agg) / surr_std_agg  # (n_win,)
        z_agg_surr = (coh_agg_surr - surr_mean_agg[None]) / surr_std_agg[None]

        # Per-channel z for diagnostics (still compute per-cf)
        surr_mean_cf = coh_surr_np.mean(axis=0)
        surr_std_cf = np.maximum(coh_surr_np.std(axis=0), 1e-8)
        per_channel_z = ((coh_real_np - surr_mean_cf) / surr_std_cf).mean(axis=1)

    else:  # stouffer
        surr_mean = coh_surr_np.mean(axis=0)
        surr_std = np.maximum(coh_surr_np.std(axis=0), 1e-8)

        z_per_cf = (coh_real_np - surr_mean) / surr_std
        surr_z_cf = (coh_surr_np - surr_mean[None]) / surr_std[None]

        per_channel_z = z_per_cf.mean(axis=1)
        surr_z_ch = surr_z_cf.mean(axis=2)

        z_agg = per_channel_z.mean(axis=0) * np.sqrt(C_sel)
        z_agg_surr = surr_z_ch.mean(axis=1) * np.sqrt(C_sel)

    # Temporal smoothing of z-scores (accumulates evidence across windows)
    smooth_s = kwargs.get('smooth_s', 0.0)
    if smooth_s > 0:
        from scipy.ndimage import uniform_filter1d
        smooth_win = max(1, int(smooth_s / stride_s))
        z_agg = uniform_filter1d(z_agg, smooth_win)
        z_agg_surr = uniform_filter1d(z_agg_surr, smooth_win, axis=1)

    # Threshold calibration from surrogates
    adaptive = kwargs.get('adaptive_threshold', False)
    if z_threshold is None:
        per_win_thresh = np.percentile(z_agg_surr, 100 * (1 - target_fa),
                                        axis=0)  # (n_win,)
        if adaptive:
            # Per-window threshold: adapts to local noise level
            z_threshold_arr = np.maximum(per_win_thresh, 1.0)
            z_threshold = float(np.median(z_threshold_arr))  # for diagnostics
        else:
            z_threshold = max(float(np.median(per_win_thresh)), 1.0)

    # Binary mask with minimum event filter
    if adaptive and isinstance(z_threshold_arr, np.ndarray):
        mask = z_agg > z_threshold_arr
    else:
        mask = z_agg > z_threshold
    # Morphological post-processing: fill gaps then remove short events
    max_gap_s = kwargs.get('max_gap_s', 0.0)
    if max_gap_s > 0:
        gap_samples = max(1, int(max_gap_s / stride_s))
        mask = _fill_gaps(mask, gap_samples)
    min_samples = max(1, int(min_event_s / stride_s))
    if min_samples > 1:
        mask = _min_event_filter(mask, min_samples)

    # Spatial cluster filter: require spatially contiguous channel support
    spatial_adj = kwargs.get('spatial_adjacency', None)
    min_spatial = kwargs.get('min_spatial_cluster', 0)
    if spatial_adj is not None and min_spatial >= 2:
        mask = _spatial_cluster_filter(
            mask, per_channel_z, spatial_adj,
            min_cluster=min_spatial, z_min=0.5)

    # Window center times in seconds
    win_times = centers_np / fs

    coupling_frac = float(np.mean(mask))

    diagnostics = {
        'method': 'wpli_v1',
        'device': str(device),
        'n_channels': C_sel,
        'n_freqs': len(center_freqs),
        'n_windows': n_win,
        'n_surrogates': n_surrogates,
        'fs': fs,
        'window_s': window_s,
        'stride_s': stride_s,
        'n_cycles': n_cycles,
        'center_freqs': center_freqs.tolist(),
        'z_threshold': float(z_threshold),
        'coupling_fraction': coupling_frac,
        'z_agg_mean': float(np.mean(z_agg)),
        'z_agg_max': float(np.max(z_agg)),
        'z_agg_p95': float(np.percentile(z_agg, 95)),
        'per_channel_z_mean': float(np.mean(per_channel_z)),
        'win_times': win_times,
    }

    return mask, z_agg, per_channel_z, diagnostics


# ---------------------------------------------------------------------------
# Cross-correlation temporal localization (time-domain, no spectral decomp)
# ---------------------------------------------------------------------------

def xcorr_temporal_localization(p1_signal, p2_signal, fs,
                                channels=None, max_lag_s=0.1,
                                lag_step_s=None,
                                smooth_s=1.0, n_surrogates=100,
                                target_fa=0.05, min_event_s=5.0,
                                seed=42, device=None):
    """Multi-lag bank cross-correlation temporal localization.

    Tests coupling at ALL lags simultaneously (0 to max_lag_s), takes the
    max over lags at each timepoint. Surrogate calibration automatically
    accounts for the max-over-lags penalty.

    Pipeline: cross-products at all lags → smooth each → average across
    channels → max over lags → surrogate-calibrated threshold.

    Args:
        p1_signal, p2_signal: (T, C) numpy arrays at native rate.
        fs: sampling rate in Hz.
        channels: list of channel indices (None = all).
        max_lag_s: maximum lag in seconds (default 0.1).
        lag_step_s: lag step size in seconds (default None = 1 sample).
            Set to e.g. 0.01 for 10ms steps to reduce computation.
        smooth_s: Gaussian smoothing sigma in seconds (default 1.0).
        n_surrogates: circular-shift surrogates (default 100).
        target_fa: target false alarm rate (default 0.05).
        min_event_s: minimum detection duration in seconds (default 5.0).
        seed: random seed.
        device: torch device.

    Returns:
        mask: (T_out,) boolean coupling mask at output rate.
        cc_agg: (T_out,) max-over-lags aggregated timecourse.
        best_lag: (T_out,) best lag in samples at each timepoint.
        diagnostics: dict with metadata.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, C_all = p1_signal.shape
    if channels is None:
        channels = list(range(C_all))
    C_sel = len(channels)
    max_lag_samp = max(1, int(max_lag_s * fs))
    lag_step = max(1, int(lag_step_s * fs)) if lag_step_s else 1
    lags = list(range(0, max_lag_samp + 1, lag_step))
    n_lags = len(lags)

    # Decimate output to ~4 Hz for memory (native rate is too large)
    dec = max(1, int(fs / 4))
    T_out = (T + dec - 1) // dec

    # Move to GPU
    x1 = torch.as_tensor(
        np.ascontiguousarray(p1_signal[:, channels].T),
        dtype=torch.float32, device=device)  # (C, T)
    x2 = torch.as_tensor(
        np.ascontiguousarray(p2_signal[:, channels].T),
        dtype=torch.float32, device=device)

    # Build Gaussian smoothing kernel
    if smooth_s > 0:
        sigma_samp = smooth_s * fs
        half = int(np.ceil(3.0 * sigma_samp))
        k = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-k ** 2 / (2 * sigma_samp ** 2))
        kernel = kernel / kernel.sum()
        smooth_kernel = kernel.reshape(1, 1, len(kernel))
        smooth_pad = len(kernel) // 2
    else:
        smooth_kernel = None

    def _smooth_1d(x):
        """Smooth (N,) tensor."""
        if smooth_kernel is None:
            return x
        xp = F.pad(x.unsqueeze(0).unsqueeze(0),
                    (smooth_pad, smooth_pad), mode='reflect')
        return F.conv1d(xp, smooth_kernel).squeeze()

    def _multi_lag_bank(s1, s2):
        """Compute smoothed cross-product at each lag, avg across channels,
        max over lags. Returns (T_out,) max-cc and (T_out,) best_lag."""
        C, N = s1.shape
        best_cc = torch.full((N,), -1e10, device=device)
        best_lag_arr = torch.zeros(N, device=device, dtype=torch.long)

        for lag in lags:
            # Cross-product at this lag
            if lag == 0:
                cp = s1 * s2
            else:
                cp = torch.zeros(C, N, device=device)
                cp[:, lag:] = s1[:, :-lag] * s2[:, lag:]

            # Average across channels → (N,)
            cp_avg = cp.mean(dim=0)

            # Smooth
            cp_smooth = _smooth_1d(cp_avg)

            # Update max
            better = cp_smooth > best_cc
            best_cc = torch.where(better, cp_smooth, best_cc)
            best_lag_arr = torch.where(better, torch.tensor(lag, device=device),
                                       best_lag_arr)

        # Decimate to output rate
        return best_cc[::dec][:T_out], best_lag_arr[::dec][:T_out]

    # Real multi-lag bank
    cc_real, lag_real = _multi_lag_bank(x1, x2)

    # Surrogate multi-lag bank
    min_shift = max(1, int(0.1 * T))
    max_shift = T - min_shift
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    shifts = torch.randint(min_shift, max_shift + 1, (n_surrogates,),
                           generator=gen)

    cc_surr = torch.zeros(n_surrogates, T_out, device=device)
    for k in range(n_surrogates):
        x1_shifted = torch.roll(x1, int(shifts[k].item()), dims=1)
        cc_surr[k], _ = _multi_lag_bank(x1_shifted, x2)

    # Move to CPU
    cc_real_np = cc_real.cpu().numpy()
    cc_surr_np = cc_surr.cpu().numpy()
    lag_np = lag_real.cpu().numpy()

    del x1, x2, cc_real, cc_surr, lag_real
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Z-score against surrogates
    surr_mean = cc_surr_np.mean(axis=0)
    surr_std = np.maximum(cc_surr_np.std(axis=0), 1e-10)
    z_agg = (cc_real_np - surr_mean) / surr_std

    z_surr = (cc_surr_np - surr_mean[None]) / surr_std[None]
    per_win_thresh = np.percentile(z_surr, 100 * (1 - target_fa), axis=0)
    z_threshold = max(float(np.median(per_win_thresh)), 1.0)

    # Threshold + event filter at output rate (4 Hz)
    out_rate = fs / dec
    mask = z_agg > z_threshold
    mask = _min_event_filter(mask, max(1, int(min_event_s * out_rate)))

    diagnostics = {
        'method': 'xcorr_multilag_v2',
        'device': str(device),
        'n_channels': C_sel,
        'n_lags': n_lags,
        'max_lag_samp': max_lag_samp,
        'lag_step': lag_step,
        'smooth_s': smooth_s,
        'n_surrogates': n_surrogates,
        'fs': fs,
        'output_rate': out_rate,
        'z_threshold': float(z_threshold),
        'coupling_fraction': float(mask.mean()),
        'z_agg_mean': float(z_agg.mean()),
        'z_agg_max': float(z_agg.max()),
    }

    return mask, z_agg, lag_np, diagnostics
