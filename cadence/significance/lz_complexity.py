"""LZ Complexity Extraction for CADENCE V10.

Computes per-participant, per-band Lempel-Ziv complexity timecourses
from raw EEG, then derives concordance and asymmetry features for
the V10 scaffold observation vector.

Pure-numpy LZ76 implementation — no Numba JIT overhead.
"""

import numpy as np
import torch
from scipy.signal import hilbert as sci_hilbert
from scipy.stats import zscore

from cadence.significance.fast_cycles import _fft_bandpass

# ── Constants ────────────────────────────────────────────────────────────

FRONTAL_ROI = [0, 1, 2, 11, 12, 13]  # AF3, F3, F7, F4, AF4, F8 on Emotiv EPOC
LZ_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
}
LZ_WINDOW_S = 4.0     # 4-second window (1024 samples at 256 Hz)
LZ_STRIDE_S = 0.5     # 0.5s stride → 2 Hz output rate


# ── Pure-numpy LZ76 ─────────────────────────────────────────────────────

def lz76(binary_seq):
    """Lempel-Ziv complexity (LZ76) of a binary sequence.

    Counts the number of distinct substrings encountered when scanning
    left to right. Normalized by n/log2(n) for length invariance.

    Standard algorithm: at each position, extend the current word until
    it is NOT a substring of the history seen so far, then increment
    complexity and start a new word.

    Args:
        binary_seq: (n,) array of 0s and 1s (uint8).

    Returns:
        Normalized LZ complexity in [0, 1].
    """
    n = len(binary_seq)
    if n <= 1:
        return 0.0

    # Convert to string for fast substring search
    s = ''.join(chr(b + 48) for b in binary_seq)  # '0' and '1' chars

    complexity = 1
    i = 0  # start of current word
    l = 1  # current word length

    while i + l <= n:
        # Current word: s[i:i+l]
        # History: s[0:i+l-1] (everything before the last char of current word)
        word = s[i:i + l]
        history = s[:i + l - 1]

        if word in history:
            l += 1
        else:
            complexity += 1
            i += l
            l = 1

    # Normalize: asymptotic complexity of random binary is n/log2(n)
    norm = n / np.log2(n) if n > 1 else 1.0
    return complexity / norm


def _lz76_vectorized(windows):
    """Compute LZ76 for a batch of windows. Vectorized binarization, loop for LZ.

    Args:
        windows: (n_windows, win_samp) float array.

    Returns:
        (n_windows,) normalized LZ complexity values.
    """
    n_windows = windows.shape[0]

    # Vectorized binarization: median threshold per window
    medians = np.median(windows, axis=1, keepdims=True)
    binary = (windows >= medians).astype(np.uint8)

    lz_vals = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        lz_vals[i] = lz76(binary[i])

    return lz_vals


# ── Core LZ extraction ──────────────────────────────────────────────────

def extract_lz_timecourse(eeg_raw, fs_eeg, bands=None, window_s=LZ_WINDOW_S,
                          stride_s=LZ_STRIDE_S, roi_channels=None):
    """Per-band frontal-ROI LZ complexity timecourse at 2 Hz.

    Args:
        eeg_raw: (T, n_ch) raw EEG array.
        fs_eeg: sampling rate (e.g. 256 Hz).
        bands: dict of band_name -> (lo, hi) Hz. Default: theta + alpha.
        window_s: window length in seconds (default 4.0).
        stride_s: stride in seconds (default 0.5 → 2 Hz).
        roi_channels: channel indices for ROI averaging. Default: FRONTAL_ROI.

    Returns:
        dict of band_name -> (N_out,) normalized LZ complexity timecourse.
        Also returns 't_out' key with (N_out,) relative timestamps.
    """
    if bands is None:
        bands = LZ_BANDS
    if roi_channels is None:
        roi_channels = FRONTAL_ROI

    T, n_ch = eeg_raw.shape
    win_samp = int(window_s * fs_eeg)
    stride_samp = int(stride_s * fs_eeg)

    # Validate ROI channels exist
    roi_channels = [c for c in roi_channels if c < n_ch]
    if not roi_channels:
        roi_channels = list(range(min(n_ch, 6)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract ROI channels: (n_roi, T)
    eeg_roi = eeg_raw[:, roi_channels].T.astype(np.float64)
    eeg_t = torch.from_numpy(eeg_roi).float().to(device)

    result = {}

    for band_name, (lo, hi) in bands.items():
        # Bandpass filter all ROI channels at once (GPU)
        filtered = _fft_bandpass(eeg_t, fs_eeg, (lo, hi), device)
        filtered_np = filtered.cpu().numpy()  # (n_roi, T)

        # Hilbert amplitude envelope per channel, then average across ROI
        # LZ on envelope captures amplitude structure (not zero-crossing patterns)
        envelopes = np.abs(sci_hilbert(filtered_np, axis=1))  # (n_roi, T)
        roi_mean = envelopes.mean(axis=0)  # (T,)

        # Sliding windows via stride_tricks (vectorized window extraction)
        n_windows = max(1, (len(roi_mean) - win_samp) // stride_samp + 1)
        shape = (n_windows, win_samp)
        strides = (roi_mean.strides[0] * stride_samp, roi_mean.strides[0])
        windows = np.lib.stride_tricks.as_strided(roi_mean, shape=shape, strides=strides)

        # Batch LZ computation (vectorized binarize, loop for LZ76)
        result[band_name] = _lz76_vectorized(windows)

    # Output timestamps (center of each window, relative to start)
    half_win = window_s / 2
    t_out = np.arange(len(result[list(bands.keys())[0]])) * stride_s + half_win
    result['t_out'] = t_out

    return result


def lz_concordance_asymmetry(lz_p1, lz_p2, asym_sign=1.0):
    """Compute LZ concordance and asymmetry from per-participant timecourses.

    Follows the same pattern as EEG concordance (ch 3-5) and asymmetry (ch 9-11):
      concordance = (z_P1 + z_P2) / 2   (shared complexity state)
      asymmetry   = sign * (z_P1 - z_P2) (positive = therapist higher)

    Args:
        lz_p1: (N,) LZ timecourse for participant 1.
        lz_p2: (N,) LZ timecourse for participant 2.
        asym_sign: +1.0 if P1=therapist, -1.0 if P2=therapist.

    Returns:
        (concordance, asymmetry) — each (N,) z-scored arrays.
    """
    # Z-score each participant independently (per-session normalization)
    z1 = zscore(lz_p1, nan_policy='omit')
    z2 = zscore(lz_p2, nan_policy='omit')

    # Handle NaN from zscore (constant input)
    z1 = np.nan_to_num(z1, nan=0.0)
    z2 = np.nan_to_num(z2, nan=0.0)

    concordance = (z1 + z2) / 2.0
    asymmetry = asym_sign * (z1 - z2)

    return concordance, asymmetry
