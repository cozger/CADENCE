"""Inter-brain phase synchrony features for CADENCE v2.

Computes inter-brain phase coherence from both participants' EEG analytic
signals (Morlet CWT).  For each frequency-ROI pair the instantaneous phase
difference delta_theta = angle(z_p1) - angle(z_p2) is computed and
represented as (cos(delta_theta), sin(delta_theta)), yielding 160 features
(2 components x 20 frequencies x 4 ROIs) sampled at output_hz (default 5 Hz).

Features are naturally bounded in [-1, 1] so no z-scoring is applied.
"""

import numpy as np

from cadence.constants import (
    EEG_ROIS,
    EEG_ROI_NAMES,
    WAVELET_CENTER_FREQS,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_interbrain_features(eeg_p1, eeg_p2, valid_p1, valid_p2,
                                ts_p1, ts_p2, srate=256, output_hz=5.0,
                                config=None):
    """Compute inter-brain phase synchrony features from both participants' EEG.

    Process:
      1. For each ROI, average channels and compute Morlet CWT (reusing
         wavelet_features internals).
      2. Time-align both analytic signal grids to a common output grid.
      3. Compute phase difference via cross-spectrum.
      4. Output: cos(delta_theta), sin(delta_theta) at each freq-ROI pair.

    Args:
        eeg_p1:    (N1, 14) preprocessed EEG, participant 1.
        eeg_p2:    (N2, 14) preprocessed EEG, participant 2.
        valid_p1:  (N1, 14) per-channel validity mask, participant 1.
        valid_p2:  (N2, 14) per-channel validity mask, participant 2.
        ts_p1:     (N1,) timestamps in seconds, participant 1.
        ts_p2:     (N2,) timestamps in seconds, participant 2.
        srate:     input EEG sampling rate (default 256 Hz).
        output_hz: output feature rate (default 5.0 Hz).
        config:    optional config dict (wavelet.n_cycles forwarded to CWT).

    Returns:
        features:  (N_out, 160) inter-brain phase synchrony features.
        valid_out: (N_out,) boolean validity.
        t_out:     (N_out,) timestamps at output_hz.
    """
    from cadence.data.wavelet_features import (
        _morlet_wavelet_bank,
        _cwt_gpu,
        _build_roi_signals,
    )

    # Resolve wavelet parameters
    n_cycles = 6
    device_str = 'cuda'
    if config is not None:
        wav_cfg = config.get('wavelet', {})
        n_cycles = wav_cfg.get('n_cycles', n_cycles)
        device_str = config.get('device', device_str)

    n_freqs = len(WAVELET_CENTER_FREQS)
    n_rois = len(EEG_ROI_NAMES)
    n_features = 2 * n_freqs * n_rois  # 160

    # Common output time grid (overlap of both participants)
    t_start = max(ts_p1[0], ts_p2[0])
    t_end = min(ts_p1[-1], ts_p2[-1])

    if t_end <= t_start:
        t_out = np.array([], dtype=np.float64)
        return (np.zeros((0, n_features), dtype=np.float32),
                np.zeros(0, dtype=bool), t_out)

    out_dt = 1.0 / output_hz
    t_out = np.arange(t_start + 1.0, t_end, out_dt)
    n_out = len(t_out)

    if n_out == 0:
        return (np.zeros((0, n_features), dtype=np.float32),
                np.zeros(0, dtype=bool), t_out)

    # Build wavelet bank (shared for both participants)
    wavelets_real, wavelets_imag, wavelet_lengths = _morlet_wavelet_bank(
        WAVELET_CENTER_FREQS, srate, n_cycles=n_cycles)

    # Build ROI signals for each participant
    roi_p1, valid_roi_p1 = _build_roi_signals(eeg_p1, valid_p1)  # (N1, 4)
    roi_p2, valid_roi_p2 = _build_roi_signals(eeg_p2, valid_p2)  # (N2, 4)

    # CWT for each participant -> (N, n_freqs, n_rois) real + imag
    real_p1, imag_p1 = _cwt_gpu(
        roi_p1, wavelets_real, wavelets_imag, wavelet_lengths, device=device_str)
    real_p2, imag_p2 = _cwt_gpu(
        roi_p2, wavelets_real, wavelets_imag, wavelet_lengths, device=device_str)

    # Interpolate both to common output grid
    real_p1_out = _interp_to_grid(real_p1, ts_p1, t_out)
    imag_p1_out = _interp_to_grid(imag_p1, ts_p1, t_out)
    real_p2_out = _interp_to_grid(real_p2, ts_p2, t_out)
    imag_p2_out = _interp_to_grid(imag_p2, ts_p2, t_out)

    # Compute phase difference via cross-spectrum
    # z1 * conj(z2) = (r1*r2 + i1*i2) + j*(i1*r2 - r1*i2)
    cross_real = real_p1_out * real_p2_out + imag_p1_out * imag_p2_out
    cross_imag = imag_p1_out * real_p2_out - real_p1_out * imag_p2_out

    # Normalize to unit circle
    magnitude = np.sqrt(cross_real**2 + cross_imag**2)
    magnitude = np.maximum(magnitude, 1e-30)

    cos_diff = (cross_real / magnitude).astype(np.float32)  # (n_out, n_freqs, n_rois)
    sin_diff = (cross_imag / magnitude).astype(np.float32)

    # Pack into output array matching INTERBRAIN_FEATURE_NAMES ordering:
    #   for comp in [cos, sin]:
    #     for freq in range(n_freqs):
    #       for roi in range(n_rois):
    features = np.zeros((n_out, n_features), dtype=np.float32)

    # cos_diff/sin_diff are (n_out, n_freqs, n_rois)
    # Flatten: freq-major, roi-minor -> (n_out, n_freqs * n_rois)
    cos_flat = cos_diff.reshape(n_out, n_freqs * n_rois)
    sin_flat = sin_diff.reshape(n_out, n_freqs * n_rois)
    features = np.concatenate([cos_flat, sin_flat], axis=1)  # (n_out, 160)

    # Validity: both participants must have valid data
    valid_out = _interp_validity(valid_roi_p1, ts_p1, t_out) & \
                _interp_validity(valid_roi_p2, ts_p2, t_out)

    features[~valid_out] = 0.0
    return features, valid_out, t_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interp_to_grid(data_3d, ts_src, t_out):
    """Interpolate (N_src, n_freqs, n_rois) array to output timestamps.

    Uses linear interpolation per channel.

    Returns:
        (N_out, n_freqs, n_rois) float32 array
    """
    N_src, n_freqs, n_rois = data_3d.shape
    n_out = len(t_out)
    result = np.zeros((n_out, n_freqs, n_rois), dtype=np.float32)

    for fi in range(n_freqs):
        for ri in range(n_rois):
            result[:, fi, ri] = np.interp(t_out, ts_src, data_3d[:, fi, ri])

    return result


def _interp_validity(valid_1d, ts_src, t_out):
    """Nearest-neighbor validity interpolation.

    Args:
        valid_1d: (N_src,) boolean
        ts_src: (N_src,) timestamps
        t_out: (N_out,) output timestamps

    Returns:
        (N_out,) boolean
    """
    idx = np.clip(np.searchsorted(ts_src, t_out), 0, len(valid_1d) - 1)
    return valid_1d[idx]
