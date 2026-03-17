"""Vectorized EEG feature extraction: 7 channels at 2 Hz.

Feature groups (7 total):
  [0]   Engagement index: beta / (alpha + theta), frontal channels
  [1]   Frontal aperiodic exponent (1/f slope, E/I balance proxy)
  [2]   Frontal theta burst fraction (BOSC detection)
  [3-4] Frontal theta phase (cos, sin)
  [5-6] Frontal alpha phase (cos, sin)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, hilbert
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EEG_FEATURES_SRATE = 2       # default; overridden by output_hz parameter
EEG_N_FEATURES = 8           # 7 base features + 1 activity channel
WIN_DURATION = 2.0           # FFT window size (seconds) — kept constant for spectral quality

# Electrode ROIs (14-ch Emotiv EPOC, indices after preprocess_eeg cols 3-16)
FRONTAL = np.array([0, 2, 11, 13])         # AF3, F3, F4, AF4

BANDS = {'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}

EEG_FEATURE_NAMES = [
    'engagement_index',
    'frontal_aperiodic_exponent',
    'frontal_theta_burst_frac',
    'phase_frontal_theta_cos', 'phase_frontal_theta_sin',
    'phase_frontal_alpha_cos', 'phase_frontal_alpha_sin',
    'eeg_activity',
]

# Features to z-score (unbounded); phase and burst are naturally bounded
_ZSCORE_NAMES = {'engagement_index', 'frontal_aperiodic_exponent'}
_ZSCORE_IDX = np.array([i for i, n in enumerate(EEG_FEATURE_NAMES)
                         if n in _ZSCORE_NAMES])

# Module-level bandpass filters (precomputed once)
_BAND_SOS = {
    'theta': butter(4, [4.0, 8.0], btype='bandpass', fs=256, output='sos'),
    'alpha': butter(4, [8.0, 13.0], btype='bandpass', fs=256, output='sos'),
    'beta':  butter(4, [13.0, 30.0], btype='bandpass', fs=256, output='sos'),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_eeg_features(eeg, eeg_valid, eeg_ts, srate=256, output_hz=None):
    """Extract 8 EEG feature channels from preprocessed EEG.

    7 base features + 1 activity channel (causal RMS deviation from trailing mean).
    The FFT window is always WIN_DURATION (2s) for spectral quality. The output_hz
    parameter controls the STEP between windows (stride), not the window size.

    Parameters:
        eeg: (N, 14) preprocessed EEG
        eeg_valid: (N, 14) per-channel validity mask
        eeg_ts: (N,) timestamps
        srate: sampling rate (256 Hz)
        output_hz: output rate in Hz (default: EEG_FEATURES_SRATE=2).
                   E.g., output_hz=5 gives 0.2s steps with 2s FFT windows.

    Returns:
        features: (N_out, 8) at output_hz
        valid_out: (N_out,) boolean
        t_out: (N_out,) timestamps
    """
    if output_hz is None:
        output_hz = EEG_FEATURES_SRATE
    win_samples = int(WIN_DURATION * srate)  # 512 for 2s @ 256Hz
    out_dt = 1.0 / output_hz

    t_out = np.arange(eeg_ts[0] + 1.0, eeg_ts[-1], out_dt)
    n_out = len(t_out)
    features = np.zeros((n_out, EEG_N_FEATURES))
    valid_out = np.zeros(n_out, dtype=bool)

    if n_out == 0 or len(eeg) < win_samples:
        return features, valid_out, t_out

    # Step 1: extract all 2s windows
    segments, seg_valid = _extract_segments(
        eeg, eeg_valid, eeg_ts, t_out, srate, win_samples)
    valid_out = seg_valid

    if not seg_valid.any():
        return features, valid_out, t_out

    # Step 2: Welch PSD (needed for engagement index + aperiodic fit)
    nperseg = min(256, win_samples)
    freqs, psds = _batch_welch_psd(segments, srate, nperseg=nperseg)

    # [0] Engagement index: beta / (alpha + theta), frontal channels
    features[:, 0] = _compute_engagement_index(psds, freqs)

    # Frontal aperiodic fit (needed for [1] exponent and [2] burst)
    frontal_ap = _fit_frontal_aperiodic(freqs, psds)

    # [1] Frontal aperiodic exponent
    features[:, 1] = np.nan_to_num(frontal_ap[:, 0], nan=0.0)

    # [2] Frontal theta burst fraction
    features[:, 2] = _compute_frontal_theta_burst(segments, srate, frontal_ap)

    # [3-6] Phase: frontal theta cos/sin, frontal alpha cos/sin
    features[:, 3:7] = _compute_frontal_phase(segments, srate)

    # Normalize + clip (on base 7 features)
    features = _normalize_features(features, valid_out)
    features[~valid_out] = 0.0

    # [7] Activity channel: causal RMS deviation from trailing 30s mean
    from cadence.data.preprocessors import compute_activity_channel
    activity = compute_activity_channel(
        features[:, :7], output_hz, trailing_seconds=30.0)
    features[:, 7] = activity[:, 0]

    return features, valid_out, t_out


# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------

def _extract_segments(eeg, eeg_valid, eeg_ts, t_out, srate, win_samples):
    """Extract all 2s windows via batch searchsorted."""
    W = len(t_out)

    t_starts = t_out - WIN_DURATION / 2
    i_starts = np.searchsorted(eeg_ts, t_starts, side='left')

    offsets = np.arange(win_samples)[None, :]  # (1, S)
    all_idx = np.clip(i_starts[:, None] + offsets, 0, len(eeg) - 1)  # (W, S)

    segments = eeg[all_idx]  # (W, S, 14)
    segments = segments.transpose(0, 2, 1)  # (W, 14, S)

    valid_samples = eeg_valid[all_idx]  # (W, S, 14)
    valid_frac = valid_samples.mean(axis=(1, 2))  # (W,)
    seg_valid = valid_frac >= 0.5

    segments[~seg_valid] = 0.0
    return segments, seg_valid


def _batch_welch_psd(segments, srate, nperseg=256):
    """Batch Welch PSD for all windows and channels."""
    W, C, S = segments.shape
    freqs, psds_flat = welch(
        segments.reshape(W * C, S), fs=srate, nperseg=nperseg, axis=-1)
    return freqs, psds_flat.reshape(W, C, -1)


def _compute_engagement_index(psds, freqs):
    """Engagement index: beta / (alpha + theta) from frontal channels."""
    theta_mask = (freqs >= BANDS['theta'][0]) & (freqs <= BANDS['theta'][1])
    alpha_mask = (freqs >= BANDS['alpha'][0]) & (freqs <= BANDS['alpha'][1])
    beta_mask = (freqs >= BANDS['beta'][0]) & (freqs <= BANDS['beta'][1])

    frontal_psds = psds[:, FRONTAL, :].mean(axis=1)  # (W, F)
    theta = frontal_psds[:, theta_mask].sum(axis=-1) + 1e-20
    alpha = frontal_psds[:, alpha_mask].sum(axis=-1) + 1e-20
    beta = frontal_psds[:, beta_mask].sum(axis=-1) + 1e-20

    return beta / (alpha + theta)


def _fit_frontal_aperiodic(freqs, psds, min_r2=0.70):
    """Frontal-ROI aperiodic fit via vectorized log-log regression."""
    fit_mask = ((freqs >= 2) & (freqs <= 3.5)) | ((freqs >= 30) & (freqs <= 40))
    if fit_mask.sum() < 3:
        fit_mask = (freqs >= 2) & (freqs <= 40)

    fit_freqs = freqs[fit_mask]
    log_f = np.log10(np.maximum(fit_freqs, 1e-10))

    x_mean = log_f.mean()
    x_var = ((log_f - x_mean) ** 2).sum()

    W = psds.shape[0]
    roi_psds = psds[:, FRONTAL, :].mean(axis=1)  # (W, F)
    roi_fit = roi_psds[:, fit_mask]  # (W, F_fit)

    log_p = np.log10(roi_fit + 1e-20)

    y_mean = log_p.mean(axis=1)
    xy_cov = ((log_f[None, :] - x_mean) * (log_p - y_mean[:, None])).sum(axis=1)

    b = xy_cov / (x_var + 1e-20)
    a = y_mean - b * x_mean

    exponents = -b
    offsets = a

    y_pred = a[:, None] + b[:, None] * log_f[None, :]
    ss_res = ((log_p - y_pred) ** 2).sum(axis=1)
    ss_tot = ((log_p - y_mean[:, None]) ** 2).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2s = np.where(ss_tot > 1e-20, 1.0 - ss_res / ss_tot, 0.0)

    bad = r2s < min_r2
    exponents[bad] = np.nan
    offsets[bad] = np.nan

    return np.column_stack([exponents, offsets])


def _compute_frontal_theta_burst(segments, srate, frontal_ap):
    """Frontal theta burst fraction via BOSC detection. Returns (W,)."""
    exponents, offsets = frontal_ap[:, 0], frontal_ap[:, 1]
    frontal_avg = segments[:, FRONTAL, :].mean(axis=1)  # (W, S)

    theta_expected = 10 ** (offsets - exponents * np.log10(6.0))
    theta_thresh = theta_expected * chi2.ppf(0.95, 2)
    theta_filt = sosfiltfilt(_BAND_SOS['theta'], frontal_avg, axis=-1)
    theta_env = np.abs(hilbert(theta_filt, axis=-1)) ** 2
    theta_burst = (theta_env > theta_thresh[:, None]).mean(axis=-1)

    theta_burst[np.isnan(exponents)] = 0.0
    return theta_burst


def _compute_frontal_phase(segments, srate):
    """Frontal theta and alpha phase (cos, sin). Returns (W, 4)."""
    cols = []
    for band_key in ['theta', 'alpha']:
        roi_avg = segments[:, FRONTAL, :].mean(axis=1)  # (W, S)
        filt = sosfiltfilt(_BAND_SOS[band_key], roi_avg, axis=-1)
        analytic = hilbert(filt, axis=-1)
        inst_phase = np.angle(analytic)
        cos_mean = np.cos(inst_phase).mean(axis=-1)
        sin_mean = np.sin(inst_phase).mean(axis=-1)
        phi = np.arctan2(sin_mean, cos_mean)
        cols.extend([np.cos(phi), np.sin(phi)])
    return np.column_stack(cols)  # (W, 4)


def _normalize_features(features, valid_mask):
    """Selective z-score + clip. Only z-scores unbounded features."""
    features = features.copy()

    if valid_mask.sum() < 10:
        return np.clip(features, -10, 10)

    for idx in _ZSCORE_IDX:
        vals = features[valid_mask, idx]
        if len(vals) > 10:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 1e-8:
                features[:, idx] = (features[:, idx] - mu) / sigma

    return np.clip(features, -10, 10)
