"""GPU acceleration helpers for the synchrony pipeline.

Per ``project_win_torch_dll_fix.md``: torch must be imported BEFORE numpy in
entry scripts. This module imports torch first; consumers should import
``cadence.synchrony._gpu`` early in their stack.
"""
from __future__ import annotations

import torch  # noqa: F401  -- precede numpy on Windows
import numpy as np

# CUDA detection at import time so callers can decide before doing work.
_HAS_CUDA = torch.cuda.is_available()


def _dog_kernel_1d(sigma: float, support_factor: float = 4.0,
                    dtype=torch.float32):
    """1D 1st-derivative-of-Gaussian kernel with Mallat normalization.

    ``g'(x; σ) = -x / σ² · (1/(σ √(2π))) · exp(-x²/(2σ²))``

    Multiplied by σ to make peak amplitudes comparable across scales
    (Mallat normalization). Returns a 1D torch tensor of odd length.
    """
    half = max(3, int(np.ceil(support_factor * sigma)))
    x = torch.arange(-half, half + 1, dtype=dtype)
    g = (-x / (sigma ** 2)) * (1.0 / (sigma * np.sqrt(2 * np.pi))) \
        * torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g * sigma  # Mallat normalization


def batched_dog_pyramid(signal_np: np.ndarray, scales_seconds, fs: float,
                         device: str = 'auto'):
    """Batched 1st-derivative-of-Gaussian pyramid for a (n_ch, T) array.

    Convolves every channel with every scale's DoG kernel in a single
    ``torch.conv1d`` call. Falls back to scipy.ndimage.gaussian_filter1d
    when CUDA is unavailable (the per-channel scipy path inside
    ``cadence.significance.face_event_coincidence.detect_au_events``).

    Args:
        signal_np: (n_ch, T) float32/float64 input.
        scales_seconds: list of σ values (s).
        fs: sampling rate (Hz).
        device: 'cuda' / 'cpu' / 'auto'.

    Returns:
        (n_scales, n_ch, T) numpy array of σ-normalized DoG responses.
    """
    if signal_np.ndim != 2:
        raise ValueError(f'signal_np must be 2D (n_ch, T); got {signal_np.shape}')
    n_ch, T = signal_np.shape

    sigmas_frames = [max(1.0, s * fs) for s in scales_seconds]
    n_scales = len(sigmas_frames)

    if device == 'auto':
        device = 'cuda' if _HAS_CUDA else 'cpu'

    if device == 'cpu':
        # Fallback to scipy per-channel — slower but no GPU dependency.
        from scipy.ndimage import gaussian_filter1d
        out = np.empty((n_scales, n_ch, T), dtype=np.float64)
        for j, sigma in enumerate(sigmas_frames):
            for c in range(n_ch):
                out[j, c] = gaussian_filter1d(
                    signal_np[c].astype(np.float64),
                    sigma=sigma, order=1, mode='reflect') * sigma
        return out

    # GPU path: build all kernels, conv1d once per scale (kernel sizes differ).
    sig = torch.from_numpy(signal_np.astype(np.float32)).to('cuda')
    sig_b = sig.unsqueeze(0)  # (1, n_ch, T)

    # Output container on GPU; one conv1d per scale (kernels of different sizes).
    out_gpu = torch.empty((n_scales, n_ch, T), dtype=torch.float32, device='cuda')
    for j, sigma in enumerate(sigmas_frames):
        kernel = _dog_kernel_1d(sigma).to('cuda')
        K = kernel.numel()
        pad = K // 2
        # Apply same kernel to all n_ch channels via groups=n_ch.
        # Kernel shape: (n_ch, 1, K) — same kernel replicated for each channel.
        kernel_g = kernel.view(1, 1, K).expand(n_ch, 1, K).contiguous()
        # 'reflect' padding via manual pad to mirror scipy 'reflect'
        sig_pad = torch.nn.functional.pad(sig_b, (pad, pad), mode='reflect')
        resp = torch.nn.functional.conv1d(sig_pad, kernel_g, groups=n_ch)
        # Crop to (1, n_ch, T) — conv after reflect-pad recovers same-length
        out_gpu[j] = resp.squeeze(0)
    return out_gpu.cpu().numpy().astype(np.float64)


def has_cuda() -> bool:
    return _HAS_CUDA
