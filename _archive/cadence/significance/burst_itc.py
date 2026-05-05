"""Event-triggered inter-trial coherence (ITC) for EEG coupling detection.

Uses Hilbert-based instantaneous phase (no CWT smearing) and restricts
analysis to P1 theta burst events where coupling signal is amplified
by the burst amplitude.

Pipeline:
  1. Bandpass P1 & P2 to theta → Hilbert → analytic signal
  2. Detect P1 bursts from channel-averaged envelope
  3. At each burst: extract cross-channel-averaged phase difference
  4. Sliding-window ITC across burst events
  5. Surrogate calibration via circular-shift of P1 analytic signal
  6. Z-score → coupling mask

Key advantage over continuous PLV: at P1 burst times, amplitude is ~2σ,
amplifying the coupling-induced phase perturbation in P2 by ~2×.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


def burst_itc_temporal_localization(p1_signal, p2_signal, fs,
                                    channels=None,
                                    band=(4.0, 8.0),
                                    burst_percentile=90,
                                    min_burst_distance_s=0.15,
                                    n_surrogates=100,
                                    window_s=20.0,
                                    stride_s=0.5,
                                    smooth_s=15.0,
                                    min_bursts_per_window=3,
                                    target_fa=0.05,
                                    min_event_s=5.0,
                                    seed=42):
    """Burst-triggered ITC temporal localization.

    Args:
        p1_signal, p2_signal: (T, C) numpy arrays at native rate.
        fs: sampling rate in Hz.
        channels: channel indices (None = all).
        band: (lo, hi) Hz for bandpass (default theta 4-8).
        burst_percentile: envelope percentile for burst threshold (default 90).
        min_burst_distance_s: minimum inter-burst interval in seconds.
        n_surrogates: number of circular-shift surrogates (default 100).
        window_s: sliding window size in seconds (default 20).
        stride_s: stride in seconds (default 0.5).
        smooth_s: temporal smoothing on ITC timecourse (default 15).
        min_bursts_per_window: minimum bursts for valid ITC (default 3).
        target_fa: target false alarm rate (default 0.05).
        min_event_s: minimum event duration in seconds (default 5).
        seed: random seed.

    Returns:
        mask: (n_win,) boolean coupling mask.
        z_agg: (n_win,) z-scored ITC timecourse.
        diagnostics: dict with metadata.
    """
    T, C_all = p1_signal.shape
    if channels is None:
        channels = list(range(C_all))
    C_sel = len(channels)

    # ── Bandpass + Hilbert ────────────────────────────────────────────────
    sos = butter(4, [band[0], band[1]], btype='band', fs=fs, output='sos')

    # Per-channel analytic signals
    p1_analytic = np.zeros((T, C_sel), dtype=np.complex128)
    p2_analytic = np.zeros((T, C_sel), dtype=np.complex128)

    for i, ch in enumerate(channels):
        p1_filt = sosfiltfilt(sos, p1_signal[:, ch])
        p2_filt = sosfiltfilt(sos, p2_signal[:, ch])
        p1_analytic[:, i] = hilbert(p1_filt)
        p2_analytic[:, i] = hilbert(p2_filt)

    # ── Channel-averaged quantities ──────────────────────────────────────
    # Cross-spectral phase difference (averaged across channels as unit vectors)
    cross_spec = p2_analytic * np.conj(p1_analytic)  # (T, C_sel)
    cross_spec_norm = cross_spec / np.maximum(np.abs(cross_spec), 1e-10)
    # Average unit vectors across channels → pooled phase difference
    dphi_pooled = cross_spec_norm.mean(axis=1)  # (T,) complex, |.| ≤ 1

    # Channel-averaged P1 envelope for burst detection
    p1_env = np.abs(p1_analytic).mean(axis=1)  # (T,)

    # ── Burst detection ──────────────────────────────────────────────────
    min_dist_samp = max(1, int(min_burst_distance_s * fs))
    threshold = np.percentile(p1_env, burst_percentile)

    burst_peaks, properties = find_peaks(
        p1_env, height=threshold, distance=min_dist_samp)

    n_bursts = len(burst_peaks)
    burst_rate = n_bursts / (T / fs)

    # ── Sliding window ITC ───────────────────────────────────────────────
    win_samp = int(window_s * fs)
    stride_samp = int(stride_s * fs)
    n_win = max(1, (T - win_samp) // stride_samp + 1)

    win_starts = np.arange(n_win) * stride_samp
    win_ends = win_starts + win_samp
    win_centers = win_starts + win_samp // 2

    def compute_itc_timecourse(burst_idx, dphi):
        """Compute ITC per window from burst indices and pooled phase diff."""
        itc = np.zeros(n_win)
        n_per_win = np.zeros(n_win, dtype=int)

        for w in range(n_win):
            mask_w = (burst_idx >= win_starts[w]) & (burst_idx < win_ends[w])
            idx_in_win = burst_idx[mask_w]
            n_b = len(idx_in_win)
            n_per_win[w] = n_b
            if n_b >= min_bursts_per_window:
                # ITC = magnitude of mean unit phase vector at burst times
                itc[w] = np.abs(dphi[idx_in_win].mean())
        return itc, n_per_win

    itc_real, n_bursts_per_win = compute_itc_timecourse(burst_peaks, dphi_pooled)

    # ── Surrogates ───────────────────────────────────────────────────────
    K = n_surrogates
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * T))
    max_shift = T - min_shift
    shifts = rng.integers(min_shift, max_shift, size=K)

    itc_surr = np.zeros((K, n_win))

    for k in range(K):
        s = int(shifts[k])
        # Shift P1 analytic → shifted envelope → shifted burst detection
        p1_env_shifted = np.roll(p1_env, s)
        burst_shifted, _ = find_peaks(
            p1_env_shifted, height=threshold, distance=min_dist_samp)

        # Shifted cross-spectral phase
        dphi_shifted = np.roll(dphi_pooled, s)

        itc_surr[k], _ = compute_itc_timecourse(burst_shifted, dphi_shifted)

    # ── Z-score ──────────────────────────────────────────────────────────
    surr_mean = itc_surr.mean(axis=0)
    surr_std = np.maximum(itc_surr.std(axis=0), 1e-10)

    z_agg = (itc_real - surr_mean) / surr_std
    z_surr = (itc_surr - surr_mean[None]) / surr_std[None]

    # Temporal smoothing
    if smooth_s > 0:
        from scipy.ndimage import uniform_filter1d
        smooth_win = max(1, int(smooth_s / stride_s))
        z_agg = uniform_filter1d(z_agg, smooth_win)
        z_surr = uniform_filter1d(z_surr, smooth_win, axis=1)

    # Threshold calibration
    per_win_thresh = np.percentile(z_surr, 100 * (1 - target_fa), axis=0)
    z_threshold = max(float(np.median(per_win_thresh)), 1.0)

    mask = z_agg > z_threshold

    # Post-processing
    from cadence.significance.coherence_localization import _min_event_filter
    min_samples = max(1, int(min_event_s / stride_s))
    if min_samples > 1:
        mask = _min_event_filter(mask, min_samples)

    win_times = win_centers / fs

    diagnostics = {
        'method': 'burst_itc',
        'n_channels': C_sel,
        'band': list(band),
        'burst_percentile': burst_percentile,
        'burst_threshold': float(threshold),
        'n_bursts_total': n_bursts,
        'burst_rate_hz': float(burst_rate),
        'mean_bursts_per_window': float(n_bursts_per_win.mean()),
        'n_windows': n_win,
        'n_surrogates': K,
        'z_threshold': float(z_threshold),
        'coupling_fraction': float(mask.mean()),
        'z_agg_mean': float(z_agg.mean()),
        'z_agg_max': float(z_agg.max()),
        'win_times': win_times,
    }

    return mask, z_agg, diagnostics
