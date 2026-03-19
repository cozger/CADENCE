"""Morlet CWT analytic signal extraction for CADENCE v2 EEG features.

Extracts wavelet-based EEG features using Morlet continuous wavelet transform
(CWT) computed via GPU conv1d, following the same grouped convolution pattern
as DesignMatrixBuilder.

Feature output: 160 channels = 2 components (real, imag) x 20 frequencies x 4 ROIs
at 5 Hz (configurable).

Process:
  1. Average raw EEG channels within each ROI (frontal, left_temp, right_temp, posterior)
  2. Apply Morlet CWT at 20 log-spaced frequencies (2-45 Hz) via GPU conv1d
  3. Extract real and imaginary parts of the analytic signal
  4. Decimate to output_hz with anti-aliasing
  5. Z-score per session
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import decimate as scipy_decimate

from cadence.constants import EEG_ROIS, EEG_ROI_NAMES, WAVELET_CENTER_FREQS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ROIS = len(EEG_ROI_NAMES)        # 4
_N_FREQS = len(WAVELET_CENTER_FREQS)  # 20
_N_FEATURES = 2 * _N_FREQS * _N_ROIS  # 160

# Chunk parameters for GPU memory management
_CHUNK_SECONDS = 60.0
_OVERLAP_SECONDS = 5.0

# Validity threshold: fraction of channels in an ROI that must be valid
_VALIDITY_FRAC = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_wavelet_features(eeg, eeg_valid, eeg_ts, srate=256, output_hz=5.0,
                             config=None):
    """Extract wavelet-based EEG features via Morlet CWT.

    Process:
    1. For each ROI, average channels to get ROI signal
    2. Apply Morlet CWT at 20 log-spaced frequencies (2-45 Hz) via GPU conv1d
    3. Extract real and imaginary parts of analytic signal
    4. Decimate to output_hz with anti-aliasing
    5. Z-score per session

    Args:
        eeg: (N, 14) preprocessed EEG
        eeg_valid: (N, 14) per-channel validity mask
        eeg_ts: (N,) timestamps
        srate: input sampling rate (256 Hz)
        output_hz: output rate (default 5.0 Hz)
        config: optional config dict (for wavelet params)

    Returns:
        features: (N_out, 160) wavelet features (2 components x 20 freqs x 4 ROIs)
        valid_out: (N_out,) boolean validity
        t_out: (N_out,) timestamps at output_hz
    """
    # Parse config overrides
    n_cycles = 6
    center_freqs = WAVELET_CENTER_FREQS
    if config is not None:
        wavelet_cfg = config.get('wavelet', {})
        n_cycles = wavelet_cfg.get('n_cycles', n_cycles)
        output_hz = wavelet_cfg.get('output_hz', output_hz)
        n_freqs = wavelet_cfg.get('n_frequencies', len(center_freqs))
        freq_range = wavelet_cfg.get('freq_range', [2.0, 45.0])
        if n_freqs != len(center_freqs) or freq_range != [2.0, 45.0]:
            center_freqs = np.logspace(
                np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs
            ).astype(np.float64)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config is not None:
        device_str = config.get('device', device_str)

    N = len(eeg_ts)
    decim_factor = int(round(srate / output_hz))
    if decim_factor < 1:
        decim_factor = 1

    # Output time grid
    out_dt = 1.0 / output_hz
    # Start after half a wavelet period of the lowest frequency to avoid edge
    t_start = eeg_ts[0] + 1.0
    t_end = eeg_ts[-1]
    t_out = np.arange(t_start, t_end, out_dt)
    n_out = len(t_out)

    if n_out == 0 or N < int(srate * 2):
        features = np.zeros((max(n_out, 0), _N_FEATURES), dtype=np.float32)
        valid_out = np.zeros(max(n_out, 0), dtype=bool)
        return features, valid_out, t_out

    # Step 1: Build ROI signals and validity
    roi_signals, roi_valid = _build_roi_signals(eeg, eeg_valid)
    # roi_signals: (N, n_rois), roi_valid: (N,)

    # Step 2: Build wavelet bank
    wavelets_real, wavelets_imag, wavelet_lengths = _morlet_wavelet_bank(
        center_freqs, srate, n_cycles=n_cycles)

    # Step 3: CWT via GPU conv1d, processing in chunks
    # Result shapes: (N, n_freqs, n_rois) for both real and imag
    analytic_real, analytic_imag = _cwt_chunked(
        roi_signals, wavelets_real, wavelets_imag, wavelet_lengths,
        srate, device=device_str)

    # Step 4: Decimate to output_hz
    feat_real_dec, feat_imag_dec, valid_dec, ts_dec = _decimate_features(
        analytic_real, analytic_imag, roi_valid, eeg_ts,
        srate, output_hz, decim_factor)

    # Step 5: Align to output time grid via interpolation
    features, valid_out = _align_to_output_grid(
        feat_real_dec, feat_imag_dec, valid_dec, ts_dec, t_out)

    # Step 6: Clip only (no second z-scoring — input EEG is already z-scored,
    # wavelet transform preserves relative scaling)
    features = np.clip(features, -10, 10).astype(np.float32)
    features[~valid_out] = 0.0

    return features, valid_out, t_out


# ---------------------------------------------------------------------------
# ROI averaging
# ---------------------------------------------------------------------------

def _build_roi_signals(eeg, eeg_valid):
    """Average EEG channels within each ROI.

    Args:
        eeg: (N, 14) preprocessed EEG
        eeg_valid: (N, 14) validity mask

    Returns:
        roi_signals: (N, n_rois) ROI-averaged signals
        valid: (N,) boolean — True if at least one ROI has >= 50% valid channels
    """
    N = eeg.shape[0]
    n_rois = len(EEG_ROI_NAMES)
    roi_signals = np.zeros((N, n_rois), dtype=np.float32)
    roi_valid = np.zeros((N, n_rois), dtype=bool)

    for r, roi_name in enumerate(EEG_ROI_NAMES):
        ch_idx = EEG_ROIS[roi_name]
        n_ch = len(ch_idx)

        # Average valid channels, zero-filling invalid ones
        ch_data = eeg[:, ch_idx].astype(np.float32)       # (N, n_ch)
        ch_valid = eeg_valid[:, ch_idx]                     # (N, n_ch)

        valid_count = ch_valid.sum(axis=1)                  # (N,)
        roi_valid[:, r] = valid_count >= max(1, int(np.ceil(n_ch * _VALIDITY_FRAC)))

        # Safe mean: only over valid channels
        ch_data_masked = np.where(ch_valid, ch_data, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            safe_count = np.maximum(valid_count, 1).astype(np.float32)
            roi_signals[:, r] = ch_data_masked.sum(axis=1) / safe_count

        # Zero out fully invalid timepoints
        roi_signals[~roi_valid[:, r], r] = 0.0

    # A timepoint is valid if at least one ROI meets the validity threshold
    valid = roi_valid.any(axis=1)

    return roi_signals, valid


# ---------------------------------------------------------------------------
# Morlet wavelet bank
# ---------------------------------------------------------------------------

def _morlet_wavelet_bank(center_freqs, srate, n_cycles=6):
    """Create Morlet wavelet bank as separate real and imaginary kernels.

    Each Morlet wavelet: w(t) = exp(2*pi*i*f*t) * exp(-t^2 / (2*sigma^2))
    where sigma = n_cycles / (2*pi*f). Normalized to unit energy.

    Args:
        center_freqs: (n_freqs,) array of center frequencies in Hz
        srate: sampling rate in Hz
        n_cycles: number of cycles (controls frequency resolution vs time resolution)

    Returns:
        wavelets_real: list of (L_k,) float32 arrays — real parts
        wavelets_imag: list of (L_k,) float32 arrays — imaginary parts
        wavelet_lengths: list of int — kernel lengths
    """
    # Resolve n_cycles: [min, max] for linear scaling across frequencies, or scalar
    if isinstance(n_cycles, (list, tuple)) and len(n_cycles) == 2:
        n_cycles = np.linspace(n_cycles[0], n_cycles[1], len(center_freqs))
    else:
        n_cycles = np.full(len(center_freqs), float(n_cycles))

    wavelets_real = []
    wavelets_imag = []
    wavelet_lengths = []

    for fi, freq in enumerate(center_freqs):
        sigma = n_cycles[fi] / (2.0 * np.pi * freq)
        # Wavelet duration: +-4 sigma (captures >99.99% of energy)
        half_len = int(np.ceil(4.0 * sigma * srate))
        t = np.arange(-half_len, half_len + 1) / srate  # symmetric time vector

        # Complex Morlet wavelet
        gaussian = np.exp(-t ** 2 / (2.0 * sigma ** 2))
        sinusoid_real = np.cos(2.0 * np.pi * freq * t)
        sinusoid_imag = np.sin(2.0 * np.pi * freq * t)

        w_real = gaussian * sinusoid_real
        w_imag = gaussian * sinusoid_imag

        # Normalize to unit energy: ||w||_2 = 1
        energy = np.sqrt(np.sum(w_real ** 2 + w_imag ** 2))
        if energy > 0:
            w_real /= energy
            w_imag /= energy

        wavelets_real.append(w_real.astype(np.float32))
        wavelets_imag.append(w_imag.astype(np.float32))
        wavelet_lengths.append(len(t))

    return wavelets_real, wavelets_imag, wavelet_lengths


# ---------------------------------------------------------------------------
# GPU CWT via conv1d
# ---------------------------------------------------------------------------

def _cwt_gpu(roi_signal, wavelets_real, wavelets_imag, wavelet_lengths,
             device='cuda'):
    """Compute CWT of ROI signals via GPU conv1d.

    Uses the same grouped conv1d pattern as DesignMatrixBuilder: one convolution
    pass per frequency (since wavelet lengths vary), with groups=n_rois.

    Args:
        roi_signal: (N, n_rois) numpy array — ROI-averaged EEG
        wavelets_real: list of (L_k,) real parts of wavelets
        wavelets_imag: list of (L_k,) imaginary parts of wavelets
        wavelet_lengths: list of int — kernel lengths per frequency
        device: torch device string

    Returns:
        out_real: (N, n_freqs, n_rois) numpy float32 — real part of CWT
        out_imag: (N, n_freqs, n_rois) numpy float32 — imaginary part of CWT
    """
    N, n_rois = roi_signal.shape
    n_freqs = len(wavelets_real)

    # Move input to device: (1, n_rois, N)
    x = torch.tensor(roi_signal.T, dtype=torch.float32, device=device).unsqueeze(0)

    out_real = np.zeros((N, n_freqs, n_rois), dtype=np.float32)
    out_imag = np.zeros((N, n_freqs, n_rois), dtype=np.float32)

    for fi in range(n_freqs):
        L = wavelet_lengths[fi]

        # Build grouped conv1d weight: (n_rois, 1, L) — same filter for each ROI
        # Flip for convolution (conv1d computes cross-correlation)
        wr = torch.tensor(
            wavelets_real[fi][::-1].copy(), dtype=torch.float32, device=device
        )
        wi = torch.tensor(
            wavelets_imag[fi][::-1].copy(), dtype=torch.float32, device=device
        )

        # Expand to (n_rois, 1, L) for grouped conv
        w_real = wr.unsqueeze(0).unsqueeze(0).expand(n_rois, 1, L)
        w_imag = wi.unsqueeze(0).unsqueeze(0).expand(n_rois, 1, L)

        # Causal-ish padding: center the wavelet (symmetric padding)
        # For CWT we want the wavelet centered at each time point
        pad_left = (L - 1) // 2
        pad_right = L - 1 - pad_left
        x_padded = F.pad(x, (pad_left, pad_right), mode='reflect')

        # Grouped conv1d: groups=n_rois
        conv_real = F.conv1d(x_padded, w_real, groups=n_rois)  # (1, n_rois, N)
        conv_imag = F.conv1d(x_padded, w_imag, groups=n_rois)  # (1, n_rois, N)

        # Store: transpose to (N, n_rois)
        out_real[:, fi, :] = conv_real.squeeze(0).T.cpu().numpy()
        out_imag[:, fi, :] = conv_imag.squeeze(0).T.cpu().numpy()

    return out_real, out_imag


def _cwt_chunked(roi_signals, wavelets_real, wavelets_imag, wavelet_lengths,
                 srate, device='cuda'):
    """Process CWT in chunks to avoid GPU OOM on long sessions.

    Uses 60s chunks with 5s overlap. Stitches results by keeping only the
    non-overlapping center of each chunk (discarding edge artifacts).

    Args:
        roi_signals: (N, n_rois) ROI-averaged EEG
        wavelets_real: list of wavelet real parts
        wavelets_imag: list of wavelet imaginary parts
        wavelet_lengths: list of kernel lengths
        srate: sampling rate
        device: torch device string

    Returns:
        out_real: (N, n_freqs, n_rois) float32
        out_imag: (N, n_freqs, n_rois) float32
    """
    N, n_rois = roi_signals.shape
    n_freqs = len(wavelets_real)

    chunk_samples = int(_CHUNK_SECONDS * srate)
    overlap_samples = int(_OVERLAP_SECONDS * srate)

    # If signal fits in one chunk, process directly
    if N <= chunk_samples + 2 * overlap_samples:
        return _cwt_gpu(roi_signals, wavelets_real, wavelets_imag,
                        wavelet_lengths, device=device)

    out_real = np.zeros((N, n_freqs, n_rois), dtype=np.float32)
    out_imag = np.zeros((N, n_freqs, n_rois), dtype=np.float32)

    # Step through the signal in non-overlapping strides
    stride = chunk_samples
    n_chunks = int(np.ceil(N / stride))

    for ci in range(n_chunks):
        # Define the region of interest (what we want to keep)
        keep_start = ci * stride
        keep_end = min(keep_start + stride, N)

        # Define the chunk to process (with overlap for edge handling)
        proc_start = max(keep_start - overlap_samples, 0)
        proc_end = min(keep_end + overlap_samples, N)

        chunk = roi_signals[proc_start:proc_end]

        # CWT on this chunk
        cr, ci_arr = _cwt_gpu(chunk, wavelets_real, wavelets_imag,
                              wavelet_lengths, device=device)

        # Extract the center (non-overlapped) region
        offset_start = keep_start - proc_start
        offset_end = offset_start + (keep_end - keep_start)

        out_real[keep_start:keep_end] = cr[offset_start:offset_end]
        out_imag[keep_start:keep_end] = ci_arr[offset_start:offset_end]

    return out_real, out_imag


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

def _decimate_features(analytic_real, analytic_imag, valid, eeg_ts,
                       srate, output_hz, decim_factor):
    """Decimate CWT output from srate to output_hz with anti-aliasing.

    Args:
        analytic_real: (N, n_freqs, n_rois)
        analytic_imag: (N, n_freqs, n_rois)
        valid: (N,) boolean
        eeg_ts: (N,) timestamps
        srate: input sample rate
        output_hz: target rate
        decim_factor: int decimation ratio

    Returns:
        real_dec: (N_dec, n_freqs, n_rois)
        imag_dec: (N_dec, n_freqs, n_rois)
        valid_dec: (N_dec,) boolean
        ts_dec: (N_dec,) timestamps
    """
    N, n_freqs, n_rois = analytic_real.shape

    if decim_factor <= 1:
        return analytic_real, analytic_imag, valid, eeg_ts

    # scipy.signal.decimate applies a Chebyshev type I anti-aliasing filter
    # then downsamples. Process each (freq, roi) channel independently.
    # Determine output length from scipy's decimate behavior
    n_dec = len(np.arange(0, N, decim_factor))

    real_dec = np.zeros((n_dec, n_freqs, n_rois), dtype=np.float32)
    imag_dec = np.zeros((n_dec, n_freqs, n_rois), dtype=np.float32)

    # Limit decimate order to avoid instability with large decimation factors
    # scipy default is 8, but for large factors we use fir
    use_fir = decim_factor > 13

    for fi in range(n_freqs):
        for ri in range(n_rois):
            try:
                real_dec[:, fi, ri] = scipy_decimate(
                    analytic_real[:, fi, ri], decim_factor,
                    ftype='fir' if use_fir else 'iir',
                    zero_phase=True)[:n_dec]
                imag_dec[:, fi, ri] = scipy_decimate(
                    analytic_imag[:, fi, ri], decim_factor,
                    ftype='fir' if use_fir else 'iir',
                    zero_phase=True)[:n_dec]
            except Exception:
                # Fallback: simple subsample (no anti-aliasing)
                real_dec[:, fi, ri] = analytic_real[::decim_factor, fi, ri][:n_dec]
                imag_dec[:, fi, ri] = analytic_imag[::decim_factor, fi, ri][:n_dec]

    # Decimate validity: a decimated point is valid if any sample in its
    # neighborhood was valid (conservative: use minimum over the window)
    # Simple approach: subsample the validity flag
    valid_dec = valid[::decim_factor][:n_dec]

    # Timestamps: subsample
    ts_dec = eeg_ts[::decim_factor][:n_dec]

    return real_dec, imag_dec, valid_dec, ts_dec


# ---------------------------------------------------------------------------
# Alignment to output grid
# ---------------------------------------------------------------------------

def _align_to_output_grid(feat_real, feat_imag, valid, ts_dec, t_out):
    """Align decimated features to the uniform output time grid via nearest-neighbor.

    Also reshapes from (N_dec, n_freqs, n_rois) to the flat feature ordering
    specified by WAVELET_FEATURE_NAMES:
        [real_f0_roi0, real_f0_roi1, ..., real_f0_roi3,
         real_f1_roi0, ..., real_f19_roi3,
         imag_f0_roi0, ..., imag_f19_roi3]

    Args:
        feat_real: (N_dec, n_freqs, n_rois)
        feat_imag: (N_dec, n_freqs, n_rois)
        valid: (N_dec,) boolean
        ts_dec: (N_dec,) timestamps
        t_out: (N_out,) target timestamps

    Returns:
        features: (N_out, 160) float32
        valid_out: (N_out,) boolean
    """
    n_out = len(t_out)
    n_dec = len(ts_dec)
    n_freqs = feat_real.shape[1]
    n_rois = feat_real.shape[2]

    if n_dec == 0:
        return (np.zeros((n_out, _N_FEATURES), dtype=np.float32),
                np.zeros(n_out, dtype=bool))

    # Nearest-neighbor lookup
    idx = np.searchsorted(ts_dec, t_out, side='left')
    idx = np.clip(idx, 0, n_dec - 1)

    # Also check the previous index for true nearest neighbor
    idx_prev = np.clip(idx - 1, 0, n_dec - 1)
    dist_curr = np.abs(ts_dec[idx] - t_out)
    dist_prev = np.abs(ts_dec[idx_prev] - t_out)
    use_prev = dist_prev < dist_curr
    idx[use_prev] = idx_prev[use_prev]

    # Resample
    real_out = feat_real[idx]  # (N_out, n_freqs, n_rois)
    imag_out = feat_imag[idx]  # (N_out, n_freqs, n_rois)
    valid_out = valid[idx]

    # Flatten to match WAVELET_FEATURE_NAMES ordering:
    # for comp in ['real', 'imag']:
    #     for freq in center_freqs:
    #         for roi in EEG_ROI_NAMES:
    # real_out is (N_out, n_freqs, n_rois) — freq varies along axis 1, roi along axis 2
    # Flatten: real first (n_freqs * n_rois), then imag (n_freqs * n_rois)
    real_flat = real_out.reshape(n_out, n_freqs * n_rois)  # freq-major, roi-minor
    imag_flat = imag_out.reshape(n_out, n_freqs * n_rois)

    features = np.concatenate([real_flat, imag_flat], axis=1)  # (N_out, 160)

    # Mark timepoints too far from any source sample as invalid
    max_gap = 2.0 / 5.0  # allow up to 2x the output sample period
    nearest_dist = np.abs(ts_dec[idx] - t_out)
    valid_out = valid_out & (nearest_dist < max_gap)

    return features.astype(np.float32), valid_out


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------

def _zscore_features(features, valid, clip=10.0):
    """Z-score each feature channel across the session (valid timepoints only).

    Args:
        features: (N_out, n_features) float32
        valid: (N_out,) boolean
        clip: clip range after z-scoring

    Returns:
        features: (N_out, n_features) z-scored and clipped
    """
    features = features.copy()
    n_valid = valid.sum()

    if n_valid < 10:
        logger.warning("Too few valid timepoints (%d) for z-scoring; "
                        "returning clipped raw features.", n_valid)
        return np.clip(features, -clip, clip)

    for c in range(features.shape[1]):
        vals = features[valid, c]
        mu = vals.mean()
        sigma = vals.std()
        if sigma > 1e-8:
            features[:, c] = (features[:, c] - mu) / sigma
        else:
            features[:, c] = 0.0

    features[~valid] = 0.0
    return np.clip(features, -clip, clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: analytic-to-features (per-ROI)
# ---------------------------------------------------------------------------

def _analytic_to_features(analytic_real, analytic_imag, roi_name):
    """Convert CWT analytic signal components to feature columns for one ROI.

    This is a helper used when processing ROIs individually. In the main
    pipeline, the full (N, n_freqs, n_rois) tensors are processed together.

    Args:
        analytic_real: (N, n_freqs) real part of CWT for this ROI
        analytic_imag: (N, n_freqs) imaginary part of CWT for this ROI
        roi_name: string name of the ROI (for logging/debugging)

    Returns:
        features: (N, 2 * n_freqs) — [real_f0, real_f1, ..., imag_f0, imag_f1, ...]
    """
    return np.concatenate([analytic_real, analytic_imag], axis=1).astype(np.float32)
