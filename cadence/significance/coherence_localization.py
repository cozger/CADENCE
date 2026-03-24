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
