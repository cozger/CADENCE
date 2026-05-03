"""Stage 3d — Morlet wavelet coherence in 0.1–2 Hz on the joint envelope.

GPU implementation reuses ``cadence.significance.bl_wavelet._cwt_gpu`` and
``_coherence_from_coeffs_gpu`` (FFT Morlet + cross-spectrum smoothing).
One CWT pair per session up front; then per-episode features are
extracted by slicing the (n_freqs, T) coherence/phase arrays — the heavy
work happens once.

Output (2 features per episode):
  coh_mean_band      mean coherence within 0.1–2 Hz, masked by COI
  coh_phase_lag_s    phase difference / 2πf at the peak coherence freq, in s
"""
from __future__ import annotations

# Hoist torch via _gpu before numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import numpy as np

from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG


# Use a finer frequency grid than the V7 default since we're explicitly
# looking at 0.1–2 Hz (not 0.3–8 Hz).
DEFAULT_COH_FREQS = np.logspace(np.log10(0.1), np.log10(2.5), 25)
MORLET_OMEGA = 5.0


def compute_session_coherence_arrays(p1_env: np.ndarray, p2_env: np.ndarray,
                                       fs: float = 30.0,
                                       config: SynchronyConfig = DEFAULT_CONFIG):
    """Compute (n_freqs, T) coherence + phase for one session's joint envelopes.

    Returns:
        freqs:     (n_freqs,)
        coherence: (n_freqs, T) float32
        phase:     (n_freqs, T) float32
    """
    from cadence.significance.bl_wavelet import (
        _cwt_gpu, _cwt_cpu, _coherence_from_coeffs_gpu, _coherence_from_coeffs_cpu,
    )

    # Align lengths
    L = min(len(p1_env), len(p2_env))
    p1 = np.asarray(p1_env[:L], dtype=np.float64).reshape(-1, 1)
    p2 = np.asarray(p2_env[:L], dtype=np.float64).reshape(-1, 1)

    use_gpu = _gpu.has_cuda() and config.use_gpu_for_coherence
    cwt_fn = _cwt_gpu if use_gpu else _cwt_cpu
    coh_fn = _coherence_from_coeffs_gpu if use_gpu else _coherence_from_coeffs_cpu

    w1 = cwt_fn(p1, fs, DEFAULT_COH_FREQS, MORLET_OMEGA)
    w2 = cwt_fn(p2, fs, DEFAULT_COH_FREQS, MORLET_OMEGA)
    sigma_samples = max(1, int(round(config.coh_smooth_s * fs)))
    coh, phase = coh_fn(w1, w2, aus=[0], sigma_samples=sigma_samples)
    return DEFAULT_COH_FREQS, coh, phase


def compute_episode_features_from_arrays(coh: np.ndarray, phase: np.ndarray,
                                          freqs: np.ndarray, ep_start_idx: int,
                                          ep_end_idx: int,
                                          config: SynchronyConfig = DEFAULT_CONFIG
                                          ) -> dict:
    """Slice per-session coherence arrays and extract per-episode features."""
    band = (freqs >= config.coh_band_lo_hz) & (freqs <= config.coh_band_hi_hz)
    if not band.any() or ep_end_idx <= ep_start_idx:
        return {'coh_mean_band': np.nan, 'coh_phase_lag_s': np.nan,
                '_3d_valid': False}
    coh_slice = coh[band][:, ep_start_idx:ep_end_idx + 1]
    phase_slice = phase[band][:, ep_start_idx:ep_end_idx + 1]
    if coh_slice.size == 0:
        return {'coh_mean_band': np.nan, 'coh_phase_lag_s': np.nan,
                '_3d_valid': False}
    mean_coh = float(np.nanmean(coh_slice))
    # Peak-coherence frequency for this episode → time-mean phase there → lag (s)
    mean_coh_per_freq = np.nanmean(coh_slice, axis=1)  # (n_band_freqs,)
    if not np.isfinite(mean_coh_per_freq).any():
        return {'coh_mean_band': mean_coh, 'coh_phase_lag_s': np.nan,
                '_3d_valid': True}
    peak_f_idx = int(np.nanargmax(mean_coh_per_freq))
    peak_freq = freqs[band][peak_f_idx]
    mean_phase = float(np.angle(np.exp(1j * phase_slice[peak_f_idx]).mean()))
    phase_lag_s = mean_phase / (2 * np.pi * peak_freq)
    return {'coh_mean_band':    mean_coh,
            'coh_phase_lag_s':  float(phase_lag_s),
            '_3d_valid':        True}


def feature_names() -> list[str]:
    return ['coh_mean_band', 'coh_phase_lag_s']
