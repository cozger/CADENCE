"""EEG Burst Coincidence — temporally precise inter-brain coupling.

Post-hoc analysis layer on V10 scaffold.  Computes whether P1 and P2 both
have EEG bursts within ±tau (500ms at 2 Hz = ±1 sample), z-scored against
circular-shift surrogates.

Bypasses the coupling uncertainty principle: per-participant bursts exist
at 256 Hz native rate, resampled to 2 Hz for alignment with scaffold timing.
"""

import numpy as np
from cadence.significance.fast_cycles import (
    extract_burst_grids, EEG_BANDS,
)


def compute_burst_coincidence(p1_grid, p2_grid, tau_samples=1,
                               n_surrogates=200, seed=42):
    """Surrogate-calibrated burst coincidence timecourse.

    Core operation: for each timepoint, check if both participants have a
    burst within ±tau samples.  ROI-averaged across channels, then z-scored
    against circular-shift surrogates.

    Args:
        p1_grid: (C, N) boolean burst grid (channels × timepoints at 2 Hz).
        p2_grid: (C, N) boolean burst grid.
        tau_samples: coincidence window in samples (1 = ±500ms at 2 Hz).
        n_surrogates: circular-shift surrogates for z-scoring.
        seed: random seed.

    Returns:
        coincidence_z: (N,) z-scored coincidence timecourse.
        coincidence_raw: (N,) raw coincidence rate (0–1, fraction of channels).
    """
    C, N = p1_grid.shape
    if C == 0 or N == 0:
        return np.zeros(N), np.zeros(N)

    # Gate: skip if either participant has <1% burst rate (too sparse
    # for meaningful coincidence — surr_std ≈ 0 causes z overflow)
    p1_rate = p1_grid.mean()
    p2_rate = p2_grid.mean()
    if p1_rate < 0.01 or p2_rate < 0.01:
        return np.zeros(N), np.zeros(N)

    # Dilate burst grids by ±tau using convolution with ones kernel
    kernel = np.ones(2 * tau_samples + 1)

    def _dilate(grid):
        """(C, N) bool -> (C, N) bool with ±tau dilation."""
        out = np.zeros_like(grid)
        for c in range(grid.shape[0]):
            out[c] = np.convolve(grid[c].astype(np.float32),
                                  kernel, mode='same') > 0
        return out

    p1_dilated = _dilate(p1_grid)

    # Real coincidence: pointwise AND, averaged across channels
    real_coinc = (p1_dilated & p2_grid).mean(axis=0).astype(np.float64)

    # Surrogate distribution (Welford online accumulation)
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * N))
    max_shift = N - min_shift

    surr_mean = np.zeros(N, dtype=np.float64)
    surr_m2 = np.zeros(N, dtype=np.float64)

    for si in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        p2_shifted = np.roll(p2_grid, shift, axis=1)
        surr_coinc = (p1_dilated & p2_shifted).mean(axis=0).astype(np.float64)

        # Welford update
        delta = surr_coinc - surr_mean
        surr_mean += delta / (si + 1)
        delta2 = surr_coinc - surr_mean
        surr_m2 += delta * delta2

    surr_std = np.sqrt(surr_m2 / max(n_surrogates - 1, 1))
    surr_std = np.maximum(surr_std, 1e-10)

    coincidence_z = (real_coinc - surr_mean) / surr_std
    # Clip to prevent overflow when surr_std is near-zero at sparse timepoints
    coincidence_z = np.clip(coincidence_z, -10, 10)
    return coincidence_z, real_coinc


def eeg_burst_coincidence(cached, t_common, lsl_offset, bands=None,
                           tau_samples=1, n_surrogates=200, seed=42):
    """Full pipeline: load EEG → extract burst grids → compute coincidence.

    Args:
        cached: session cache dict (must contain p1_eeg, p2_eeg, p1_eeg_ts).
        t_common: (N,) scaffold time grid in LSL seconds.
        lsl_offset: float, LSL time offset for EEG alignment.
        bands: dict of {name: (lo, hi)} or None for default.
        tau_samples: coincidence window (1 sample = 500ms at 2 Hz).
        n_surrogates: surrogates for z-scoring.
        seed: random seed.

    Returns:
        dict of {band_name: {'z': (N,), 'raw': (N,), 'n_valid_channels': int}}
        or None if EEG data is missing.
    """
    import torch

    if bands is None:
        bands = EEG_BANDS

    # Check required data
    if 'p1_eeg' not in cached or 'p2_eeg' not in cached or 'p1_eeg_ts' not in cached:
        return None

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    fs_eeg = 256.0

    # Trim to common length and 14 channels
    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg = p1_eeg[:, :n_ch]
    p2_eeg = p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg = p1_eeg[:mlen]
    p2_eeg = p2_eeg[:mlen]

    # EEG-local time grid (0-based) at 2 Hz
    dur = mlen / fs_eeg
    feature_rate = 2.0
    t_grid_local = np.arange(0, dur, 1.0 / feature_rate)

    # Extract burst grids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                 bands=bands, device=device)

    # Map EEG-local time to LSL time for resampling to t_common
    t_grid_lsl = t_grid_local + (p1_ts[0] + lsl_offset)

    N = len(t_common)
    result = {}

    for band_name, bg in grids.items():
        valid = bg['valid_channels']
        if len(valid) < 3:
            result[band_name] = {
                'z': np.zeros(N), 'raw': np.zeros(N),
                'n_valid_channels': len(valid),
            }
            continue

        # Filter to channels where BOTH participants have >1% burst rate
        # (sparse channels produce near-zero surrogate std → z overflow)
        p1_ch_rates = bg['p1_burst'].mean(axis=1)
        p2_ch_rates = bg['p2_burst'].mean(axis=1)
        ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)

        if ch_mask.sum() < 3:
            result[band_name] = {
                'z': np.zeros(N), 'raw': np.zeros(N),
                'n_valid_channels': int(ch_mask.sum()),
            }
            continue

        p1_burst_filt = bg['p1_burst'][ch_mask]
        p2_burst_filt = bg['p2_burst'][ch_mask]

        # Compute coincidence in EEG-local time
        coinc_z_local, coinc_raw_local = compute_burst_coincidence(
            p1_burst_filt, p2_burst_filt,
            tau_samples=tau_samples, n_surrogates=n_surrogates, seed=seed)

        # Resample to scaffold's t_common (LSL time)
        coinc_z = np.interp(t_common, t_grid_lsl, coinc_z_local,
                             left=0, right=0).astype(np.float32)
        coinc_raw = np.interp(t_common, t_grid_lsl, coinc_raw_local,
                               left=0, right=0).astype(np.float32)

        result[band_name] = {
            'z': coinc_z,
            'raw': coinc_raw,
            'n_valid_channels': int(ch_mask.sum()),
        }

    return result
