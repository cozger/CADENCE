"""Directed EEG Burst Coupling: ECA + Transfer Entropy.

Post-hoc analysis on V10 scaffold.  Answers "who leads?" using the same
burst grids from ``extract_burst_grids()`` that symmetric coincidence uses.

Two complementary metrics:
  - **Directed ECA**: at each lag, what fraction of P2 onsets were preceded
    by a P1 onset?  Asymmetry = therapist-leads minus patient-leads.
  - **Transfer entropy**: does knowing P1's burst history reduce uncertainty
    about P2's next burst?  Model-free, conditions on target history.

Both use circular-shift surrogates (shift P2, preserve autocorrelation).
"""

import numpy as np


# ── Onset extraction ────────────────────────────────────────────────

def burst_onsets(grid):
    """Convert boolean burst grid to onset grid (rising edges).

    A rising edge at sample t means grid[c, t] is True and grid[c, t-1]
    is False (or t == 0 and grid[c, 0] is True).

    Args:
        grid: (C, N) boolean burst grid.

    Returns:
        (C, N) boolean onset grid.
    """
    onsets = np.zeros_like(grid)
    onsets[:, 0] = grid[:, 0]
    onsets[:, 1:] = grid[:, 1:] & ~grid[:, :-1]
    return onsets


# ── Directed ECA ────────────────────────────────────────────────────

def directed_eca_burst(p1_grid, p2_grid, max_lag_samples=4, fs=2.0):
    """Session-level directed Event Coincidence Analysis on burst grids.

    For each lag L in [1, max_lag_samples]:
      - P1-leads rate: fraction of P2 onsets that have a P1 onset exactly
        L samples before them, averaged across channels.
      - P2-leads rate: symmetric (P2 onset L samples before P1 onset).

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids.
        max_lag_samples: max lag (4 @ 2 Hz = 2s).
        fs: sampling rate (for metadata).

    Returns:
        dict with p1_leads_rate, p2_leads_rate (arrays over lags),
        lags_s, asymmetry, n_p1_onsets, n_p2_onsets.
    """
    C, N = p1_grid.shape
    p1_on = burst_onsets(p1_grid)
    p2_on = burst_onsets(p2_grid)

    n_p1 = p1_on.sum()
    n_p2 = p2_on.sum()

    lags = np.arange(1, max_lag_samples + 1)
    p1_leads = np.zeros(len(lags))
    p2_leads = np.zeros(len(lags))

    for li, lag in enumerate(lags):
        # P1 leads: P1 onset at t, P2 onset at t+lag
        # Shift P1 onsets forward by lag, AND with P2 onsets
        p1_shifted = np.zeros_like(p1_on)
        p1_shifted[:, lag:] = p1_on[:, :-lag]
        # Fraction of P2 onsets preceded by a P1 onset at this lag
        if n_p2 > 0:
            p1_leads[li] = (p1_shifted & p2_on).sum() / n_p2

        # P2 leads: P2 onset at t, P1 onset at t+lag
        p2_shifted = np.zeros_like(p2_on)
        p2_shifted[:, lag:] = p2_on[:, :-lag]
        if n_p1 > 0:
            p2_leads[li] = (p2_shifted & p1_on).sum() / n_p1

    asym = float(p1_leads.sum() - p2_leads.sum())

    return {
        'p1_leads_rate': p1_leads,
        'p2_leads_rate': p2_leads,
        'lags_s': lags / fs,
        'asymmetry': asym,
        'n_p1_onsets': int(n_p1),
        'n_p2_onsets': int(n_p2),
    }


def directed_eca_sliding(p1_grid, p2_grid, window_samples=120,
                          stride_samples=1, max_lag_samples=4, fs=2.0):
    """Sliding-window directed ECA asymmetry timecourse.

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids.
        window_samples: window size (120 @ 2 Hz = 60s).
        stride_samples: stride in samples.
        max_lag_samples: max lag for directed ECA.
        fs: sampling rate.

    Returns:
        dict with t_centers (sample indices), eca_asym timecourse,
        p1_leads / p2_leads timecourses.
    """
    C, N = p1_grid.shape
    centers = np.arange(window_samples // 2, N - window_samples // 2,
                        stride_samples)

    eca_asym = np.zeros(len(centers))
    p1_leads_tc = np.zeros(len(centers))
    p2_leads_tc = np.zeros(len(centers))

    for i, c in enumerate(centers):
        s = c - window_samples // 2
        e = c + window_samples // 2
        res = directed_eca_burst(p1_grid[:, s:e], p2_grid[:, s:e],
                                  max_lag_samples=max_lag_samples, fs=fs)
        eca_asym[i] = res['asymmetry']
        p1_leads_tc[i] = res['p1_leads_rate'].sum()
        p2_leads_tc[i] = res['p2_leads_rate'].sum()

    return {
        't_centers': centers,
        'eca_asym': eca_asym,
        'p1_leads': p1_leads_tc,
        'p2_leads': p2_leads_tc,
    }


# ── Binary Transfer Entropy ────────────────────────────────────────

def binary_transfer_entropy(x, y, k=3):
    """Transfer entropy TE(x → y) from binary sequences.

    TE(x→y) = H(y_t | y_past) - H(y_t | y_past, x_past)

    Computed via frequency counting on k-bit packed history patterns.
    For k=3: 8 y-history patterns, 64 joint patterns — trivial.

    Args:
        x: (N,) binary array (source, 0/1 or bool).
        y: (N,) binary array (target).
        k: history length.

    Returns:
        te: float, transfer entropy in nats (≥0).
    """
    x = np.asarray(x, dtype=np.int8)
    y = np.asarray(y, dtype=np.int8)
    N = len(x)
    if N <= k:
        return 0.0

    # Pack k-bit histories into integers via bit-shifting
    # y_hist[t] encodes y[t-1], y[t-2], ..., y[t-k] as a k-bit integer
    # x_hist[t] encodes x[t-1], x[t-2], ..., x[t-k] as a k-bit integer
    y_hist = np.zeros(N - k, dtype=np.int32)
    x_hist = np.zeros(N - k, dtype=np.int32)
    for lag in range(k):
        y_hist |= y[k - 1 - lag: N - 1 - lag].astype(np.int32) << lag
        x_hist |= x[k - 1 - lag: N - 1 - lag].astype(np.int32) << lag

    y_future = y[k:].astype(np.int32)  # y_t
    M = len(y_future)

    # Joint pattern: (y_hist, x_hist, y_future) → single integer
    # y_hist: k bits, x_hist: k bits, y_future: 1 bit
    joint = (y_hist << (k + 1)) | (x_hist << 1) | y_future
    n_joint_bins = 2 ** (2 * k + 1)

    # Count frequencies
    joint_counts = np.bincount(joint, minlength=n_joint_bins).astype(np.float64)

    # Marginal: (y_hist, y_future) — sum over x_hist
    yx_key = (y_hist << 1) | y_future
    yx_counts = np.bincount(yx_key, minlength=2 ** (k + 1)).astype(np.float64)

    # Marginal: (y_hist, x_hist) — sum over y_future
    yxh_key = (y_hist << k) | x_hist
    yxh_counts = np.bincount(yxh_key, minlength=2 ** (2 * k)).astype(np.float64)

    # Marginal: y_hist — sum over x_hist and y_future
    yh_counts = np.bincount(y_hist, minlength=2 ** k).astype(np.float64)

    # TE = sum p(y_t, y_past, x_past) * log[ p(y_t|y_past,x_past) / p(y_t|y_past) ]
    # = sum joint * [ log(joint / yxh) - log(yx / yh) ]
    te = 0.0
    for idx in range(n_joint_bins):
        if joint_counts[idx] == 0:
            continue
        # Decode indices
        yf = idx & 1
        xh = (idx >> 1) & ((1 << k) - 1)
        yh = idx >> (k + 1)

        yx_idx = (yh << 1) | yf
        yxh_idx = (yh << k) | xh

        p_joint = joint_counts[idx] / M
        p_yt_given_yx = joint_counts[idx] / max(yxh_counts[yxh_idx], 1e-30)
        p_yt_given_y = yx_counts[yx_idx] / max(yh_counts[yh], 1e-30)

        if p_yt_given_y > 0:
            te += p_joint * np.log(p_yt_given_yx / p_yt_given_y)

    return max(te, 0.0)  # TE is non-negative in expectation


def directed_te_burst(p1_grid, p2_grid, k=3):
    """Session-level directed transfer entropy on burst grids.

    Computes TE(P1→P2) and TE(P2→P1) per channel, then averages.

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids.
        k: history length.

    Returns:
        dict with te_p1_to_p2, te_p2_to_p1, te_asymmetry,
        per_channel arrays.
    """
    C, N = p1_grid.shape
    te_p1p2 = np.zeros(C)
    te_p2p1 = np.zeros(C)

    for c in range(C):
        te_p1p2[c] = binary_transfer_entropy(p1_grid[c], p2_grid[c], k=k)
        te_p2p1[c] = binary_transfer_entropy(p2_grid[c], p1_grid[c], k=k)

    return {
        'te_p1_to_p2': float(te_p1p2.mean()),
        'te_p2_to_p1': float(te_p2p1.mean()),
        'te_asymmetry': float((te_p1p2 - te_p2p1).mean()),
        'per_channel_te_p1p2': te_p1p2,
        'per_channel_te_p2p1': te_p2p1,
    }


def directed_te_sliding(p1_grid, p2_grid, window_samples=120,
                         stride_samples=1, k=3, fs=2.0):
    """Sliding-window directed TE asymmetry timecourse (CPU fallback).

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids.
        window_samples: window size (120 @ 2 Hz = 60s).
        stride_samples: stride.
        k: TE history length.
        fs: sampling rate.

    Returns:
        dict with t_centers, te_asym, te_p1p2, te_p2p1 timecourses.
    """
    C, N = p1_grid.shape
    centers = np.arange(window_samples // 2, N - window_samples // 2,
                        stride_samples)

    te_asym = np.zeros(len(centers))
    te_p1p2_tc = np.zeros(len(centers))
    te_p2p1_tc = np.zeros(len(centers))

    for i, c in enumerate(centers):
        s = c - window_samples // 2
        e = c + window_samples // 2
        res = directed_te_burst(p1_grid[:, s:e], p2_grid[:, s:e], k=k)
        te_asym[i] = res['te_asymmetry']
        te_p1p2_tc[i] = res['te_p1_to_p2']
        te_p2p1_tc[i] = res['te_p2_to_p1']

    return {
        't_centers': centers,
        'te_asym': te_asym,
        'te_p1p2': te_p1p2_tc,
        'te_p2p1': te_p2p1_tc,
    }


# ── GPU-accelerated sliding-window TE with surrogates ──────────────

def _pack_histories_torch(x, k, device):
    """Pack k-bit binary histories into integers. (C, N) → (C, N-k) int."""
    import torch
    C, N = x.shape
    x_t = torch.as_tensor(x.astype(np.int32), device=device)  # (C, N)
    hist = torch.zeros(C, N - k, dtype=torch.int32, device=device)
    for lag in range(k):
        hist |= x_t[:, k - 1 - lag: N - 1 - lag] << lag
    return hist


def _te_from_windowed_counts(joint_counts, n_joint_bins, k):
    """Compute TE from windowed bin counts. (B, n_windows, n_bins) → (B, n_windows).

    Uses vectorized log computation — no Python loops over bins.
    """
    import torch
    B, W, _ = joint_counts.shape
    M = joint_counts.sum(dim=2, keepdim=True).clamp(min=1)  # (B, W, 1)

    # Decode bin indices to get marginal indices
    # joint index = (y_hist << (k+1)) | (x_hist << 1) | y_future
    idx = torch.arange(n_joint_bins, device=joint_counts.device)
    yf = idx & 1                          # (n_bins,)
    xh = (idx >> 1) & ((1 << k) - 1)
    yh = idx >> (k + 1)

    # Marginal indices
    yx_idx = (yh << 1) | yf              # (y_hist, y_future) → index into 2^(k+1)
    yxh_idx = (yh << k) | xh             # (y_hist, x_hist) → index into 2^(2k)
    yh_idx = yh                            # y_hist → index into 2^k

    n_yx = 2 ** (k + 1)
    n_yxh = 2 ** (2 * k)
    n_yh = 2 ** k

    # Compute marginals via scatter_add (equivalent to summing joint over subsets)
    # yx_counts[b, w, yx_idx[j]] += joint_counts[b, w, j]
    yx_counts = torch.zeros(B, W, n_yx, device=joint_counts.device)
    yx_counts.scatter_add_(2, yx_idx.expand(B, W, -1), joint_counts)

    yxh_counts = torch.zeros(B, W, n_yxh, device=joint_counts.device)
    yxh_counts.scatter_add_(2, yxh_idx.expand(B, W, -1), joint_counts)

    yh_counts = torch.zeros(B, W, n_yh, device=joint_counts.device)
    yh_counts.scatter_add_(2, yh_idx.expand(B, W, -1), joint_counts)

    # Gather marginals back to joint shape for vectorized TE
    yx_at_joint = yx_counts.gather(2, yx_idx.expand(B, W, -1))    # (B, W, n_bins)
    yxh_at_joint = yxh_counts.gather(2, yxh_idx.expand(B, W, -1))
    yh_at_joint = yh_counts.gather(2, yh_idx.expand(B, W, -1))

    # TE = sum_j p(j) * log[ p(y|y_hist,x_hist) / p(y|y_hist) ]
    #    = sum_j (c_j/M) * log[ (c_j / c_yxh) / (c_yx / c_yh) ]
    #    = sum_j (c_j/M) * [ log(c_j * c_yh) - log(c_yxh * c_yx) ]
    # Only compute log where all counts are positive (avoid log(0))
    valid = ((joint_counts > 0) & (yxh_at_joint > 0)
             & (yx_at_joint > 0) & (yh_at_joint > 0))
    log_ratio = torch.zeros_like(joint_counts)
    log_ratio[valid] = torch.log(
        joint_counts[valid] * yh_at_joint[valid]
        / (yxh_at_joint[valid] * yx_at_joint[valid]))

    te = (joint_counts / M * log_ratio).sum(dim=2)  # (B, W)

    return te.clamp(min=0)


def gpu_sliding_te_surrogates(p1_grid, p2_grid, window_samples=120,
                               stride_samples=2, k=3, n_surrogates=200,
                               seed=42, device=None):
    """GPU-accelerated sliding-window TE with per-timepoint surrogate z.

    Uses cumsum trick for O(N) sliding-window bin counting. Surrogates
    are batched on GPU (mini-batches of 20 to fit VRAM).

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids (numpy).
        window_samples: window size (120 @ 2 Hz = 60s).
        stride_samples: stride.
        k: TE history length.
        n_surrogates: circular-shift surrogates.
        seed: random seed.
        device: torch device.

    Returns:
        dict with:
            te_p1p2_z: (n_windows,) z-scored TE(P1→P2) timecourse
            te_p2p1_z: (n_windows,) z-scored TE(P2→P1) timecourse
            te_asym_z: (n_windows,) z-scored asymmetry timecourse
            te_p1p2_raw: (n_windows,) raw TE(P1→P2)
            te_p2p1_raw: (n_windows,) raw TE(P2→P1)
            t_centers: (n_windows,) center sample indices
    """
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    C, N = p1_grid.shape
    n_joint_bins = 2 ** (2 * k + 1)
    W = window_samples
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * (N - k)))
    max_shift = (N - k) - min_shift

    # Window positions
    starts = np.arange(0, N - k - W, stride_samples)
    n_win = len(starts)
    if n_win == 0:
        return {
            'te_p1p2_z': np.zeros(0), 'te_p2p1_z': np.zeros(0),
            'te_asym_z': np.zeros(0), 'te_p1p2_raw': np.zeros(0),
            'te_p2p1_raw': np.zeros(0), 't_centers': np.zeros(0),
        }

    # Pack histories for all channels (once)
    p1_hist = _pack_histories_torch(p1_grid, k, device)  # (C, N-k)
    p2_hist = _pack_histories_torch(p2_grid, k, device)
    p1_future = torch.as_tensor(p1_grid[:, k:].astype(np.int32), device=device)
    p2_future = torch.as_tensor(p2_grid[:, k:].astype(np.int32), device=device)

    def _compute_te_timecourse(src_hist, tgt_hist, tgt_future):
        """Compute sliding-window TE(src→tgt) for all windows. Returns (n_win,)."""
        # Joint pattern: (tgt_hist << (k+1)) | (src_hist << 1) | tgt_future
        joint_idx = (tgt_hist << (k + 1)) | (src_hist << 1) | tgt_future  # (C, N-k)

        # One-hot encode → (C, N-k, n_bins)
        one_hot = torch.zeros(C, N - k, n_joint_bins, device=device)
        one_hot.scatter_(2, joint_idx.unsqueeze(2), 1.0)

        # Cumsum for sliding window counts
        cumsum = torch.cumsum(one_hot, dim=1)  # (C, N-k, n_bins)

        # Window counts: cumsum[start+W] - cumsum[start] for each window
        # starts: (n_win,) indices
        ends = starts + W  # (n_win,)
        starts_t = torch.as_tensor(starts, device=device, dtype=torch.long)
        ends_t = torch.as_tensor(ends, device=device, dtype=torch.long)

        # Gather: (C, n_win, n_bins)
        cs_end = cumsum[:, ends_t - 1, :]    # (C, n_win, n_bins)
        cs_start = torch.zeros_like(cs_end)
        mask_s = starts_t > 0
        if mask_s.any():
            cs_start[:, mask_s, :] = cumsum[:, starts_t[mask_s] - 1, :]
        win_counts = cs_end - cs_start  # (C, n_win, n_bins)

        # TE from windowed counts — batch across channels
        te_per_ch = _te_from_windowed_counts(win_counts, n_joint_bins, k)  # (C, n_win)
        return te_per_ch.mean(dim=0)  # (n_win,) averaged across channels

    # Real TE timecourses
    te_p1p2_real = _compute_te_timecourse(p1_hist, p2_hist, p2_future)  # (n_win,)
    te_p2p1_real = _compute_te_timecourse(p2_hist, p1_hist, p1_future)

    # Surrogate accumulation (Welford, mini-batched)
    surr_mean_p1p2 = torch.zeros(n_win, device=device)
    surr_m2_p1p2 = torch.zeros(n_win, device=device)
    surr_mean_p2p1 = torch.zeros(n_win, device=device)
    surr_m2_p2p1 = torch.zeros(n_win, device=device)

    for si in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        # Circular-shift source histories (P2 for p1→p2 direction)
        p2_hist_s = torch.roll(p2_hist, int(shift), dims=1)
        p2_future_s = torch.roll(p2_future, int(shift), dims=1)
        p1_hist_s = torch.roll(p1_hist, int(shift), dims=1)
        p1_future_s = torch.roll(p1_future, int(shift), dims=1)

        s_p1p2 = _compute_te_timecourse(p1_hist, p2_hist_s, p2_future_s)
        s_p2p1 = _compute_te_timecourse(p2_hist, p1_hist_s, p1_future_s)

        # Welford update
        n = si + 1
        d1 = s_p1p2 - surr_mean_p1p2
        surr_mean_p1p2 += d1 / n
        surr_m2_p1p2 += d1 * (s_p1p2 - surr_mean_p1p2)

        d2 = s_p2p1 - surr_mean_p2p1
        surr_mean_p2p1 += d2 / n
        surr_m2_p2p1 += d2 * (s_p2p1 - surr_mean_p2p1)

    # Z-scores
    std_p1p2 = (surr_m2_p1p2 / max(n_surrogates - 1, 1)).sqrt().clamp(min=1e-10)
    std_p2p1 = (surr_m2_p2p1 / max(n_surrogates - 1, 1)).sqrt().clamp(min=1e-10)

    z_p1p2 = ((te_p1p2_real - surr_mean_p1p2) / std_p1p2).clamp(-10, 10)
    z_p2p1 = ((te_p2p1_real - surr_mean_p2p1) / std_p2p1).clamp(-10, 10)
    z_asym = z_p1p2 - z_p2p1

    centers = starts + W // 2

    return {
        'te_p1p2_z': z_p1p2.cpu().numpy().astype(np.float32),
        'te_p2p1_z': z_p2p1.cpu().numpy().astype(np.float32),
        'te_asym_z': z_asym.cpu().numpy().astype(np.float32),
        'te_p1p2_raw': te_p1p2_real.cpu().numpy().astype(np.float32),
        'te_p2p1_raw': te_p2p1_real.cpu().numpy().astype(np.float32),
        't_centers': centers,
    }


# ── Surrogate significance ─────────────────────────────────────────

def surrogate_significance_directed(p1_grid, p2_grid, metric_fn,
                                     n_surrogates=200, seed=42):
    """Circular-shift surrogate z-scoring for directed metrics.

    Shifts P2 grid circularly (preserves autocorrelation and burst rate),
    recomputes metric. Reports z-score of real vs surrogate distribution.

    Args:
        p1_grid, p2_grid: (C, N) boolean burst grids.
        metric_fn: callable(p1_grid, p2_grid) -> dict with float values.
        n_surrogates: number of surrogates.
        seed: random seed.

    Returns:
        z_scores: dict matching metric_fn output, z-scored.
        raw_values: dict with real metric values.
    """
    C, N = p1_grid.shape
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * N))
    max_shift = N - min_shift

    # Real values
    real = metric_fn(p1_grid, p2_grid)

    # Extract scalar keys for z-scoring
    scalar_keys = [k for k, v in real.items()
                   if isinstance(v, (int, float, np.floating))]

    # Welford accumulation
    surr_mean = {k: 0.0 for k in scalar_keys}
    surr_m2 = {k: 0.0 for k in scalar_keys}

    for si in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        p2_shifted = np.roll(p2_grid, shift, axis=1)
        surr = metric_fn(p1_grid, p2_shifted)

        for k in scalar_keys:
            val = float(surr[k])
            delta = val - surr_mean[k]
            surr_mean[k] += delta / (si + 1)
            delta2 = val - surr_mean[k]
            surr_m2[k] += delta * delta2

    z_scores = {}
    for k in scalar_keys:
        std = np.sqrt(surr_m2[k] / max(n_surrogates - 1, 1))
        std = max(std, 1e-10)
        z_scores[k] = np.clip((float(real[k]) - surr_mean[k]) / std, -10, 10)

    return z_scores, {k: float(real[k]) for k in scalar_keys}


# ── Directed episode detection ──────────────────────────────────────

def detect_directed_episodes(te_asym_z_tc, threshold_z=2.0, min_dur_s=3.0,
                              merge_gap_s=5.0, fs=2.0):
    """Detect directed coupling episodes from surrogate-z TE asymmetry.

    Detects sustained excursions above +threshold (therapist-leads episodes)
    or below -threshold (patient-leads episodes) in the per-timepoint
    surrogate z-scored TE asymmetry timecourse.

    Args:
        te_asym_z_tc: (N,) surrogate z-scored TE asymmetry timecourse.
        threshold_z: z-score threshold for episode detection.
        min_dur_s: minimum episode duration in seconds.
        merge_gap_s: merge episodes within this gap.
        fs: sampling rate.

    Returns:
        dict with:
            therapist_leads: list of episode dicts (onset, offset, duration_s, mean_z)
            patient_leads: list of episode dicts
            frac_therapist: fraction of time in therapist-leads episodes
            frac_patient: fraction of time in patient-leads episodes
    """
    N = len(te_asym_z_tc)
    z_tc = te_asym_z_tc
    min_samp = int(min_dur_s * fs)
    merge_samp = int(merge_gap_s * fs)

    def _detect_runs(signal, thresh):
        above = signal > thresh
        events = []
        in_ev = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_ev:
                start = i
                in_ev = True
            elif not above[i] and in_ev:
                events.append((start, i))
                in_ev = False
        if in_ev:
            events.append((start, len(above)))

        # Merge nearby
        merged = []
        for s, e in events:
            if merged and s - merged[-1][1] <= merge_samp:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Filter by duration
        result = []
        for s, e in merged:
            dur = (e - s) / fs
            if dur >= min_dur_s:
                result.append({
                    'onset': int(s), 'offset': int(e),
                    'duration_s': float(dur),
                    'mean_z': float(signal[s:e].mean()),
                })
        return result

    therapist_eps = _detect_runs(z_tc, threshold_z)
    patient_eps = _detect_runs(-z_tc, threshold_z)
    # Fix sign for patient episodes
    for ep in patient_eps:
        ep['mean_z'] = -ep['mean_z']

    t_samp = sum(e['offset'] - e['onset'] for e in therapist_eps)
    p_samp = sum(e['offset'] - e['onset'] for e in patient_eps)

    return {
        'therapist_leads': therapist_eps,
        'patient_leads': patient_eps,
        'frac_therapist': float(t_samp / N) if N > 0 else 0.0,
        'frac_patient': float(p_samp / N) if N > 0 else 0.0,
    }


# ── Pipeline wrapper ────────────────────────────────────────────────

def eeg_directed_burst_coupling(cached, t_common, lsl_offset, bands=None,
                                 max_lag_samples=4, k_te=3, n_surrogates=200,
                                 seed=42):
    """Full pipeline: extract burst grids → directed ECA + TE per band.

    Args:
        cached: session cache dict.
        t_common: (N,) scaffold time grid in LSL seconds.
        lsl_offset: LSL time offset.
        bands: dict of {name: (lo, hi)} or None for default.
        max_lag_samples: max lag for ECA (4 = 2s at 2 Hz).
        k_te: TE history length.
        n_surrogates: surrogates for z-scoring.
        seed: random seed.

    Returns:
        dict of {band: {eca_*, te_*, z_*}} or None if no EEG.
    """
    import torch
    from cadence.significance.fast_cycles import extract_burst_grids, EEG_BANDS

    if bands is None:
        bands = EEG_BANDS

    if 'p1_eeg' not in cached or 'p2_eeg' not in cached or 'p1_eeg_ts' not in cached:
        return None

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    fs_eeg = 256.0

    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg = p1_eeg[:, :n_ch]
    p2_eeg = p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg = p1_eeg[:mlen]
    p2_eeg = p2_eeg[:mlen]

    dur = mlen / fs_eeg
    feature_rate = 2.0
    t_grid_local = np.arange(0, dur, 1.0 / feature_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                 bands=bands, device=device)

    t_grid_lsl = t_grid_local + (p1_ts[0] + lsl_offset)
    N = len(t_common)
    result = {}

    for band_name, bg in grids.items():
        valid = bg['valid_channels']

        # Channel filter: both participants >1% burst rate
        p1_ch_rates = bg['p1_burst'].mean(axis=1)
        p2_ch_rates = bg['p2_burst'].mean(axis=1)
        ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)

        if ch_mask.sum() < 3:
            result[band_name] = {
                'eca_asymmetry': 0.0, 'te_asymmetry': 0.0, 'te_z': 0.0,
                'n_valid_channels': int(ch_mask.sum()),
                'eca_asym_tc': np.zeros(N, dtype=np.float32),
                'te_asym_tc': np.zeros(N, dtype=np.float32),
                'te_asym_z_tc': np.zeros(N, dtype=np.float32),
                'te_p1p2_z_tc': np.zeros(N, dtype=np.float32),
                'te_p2p1_z_tc': np.zeros(N, dtype=np.float32),
                'therapist_leads_episodes': [],
                'patient_leads_episodes': [],
                'frac_therapist_leads': 0.0, 'frac_patient_leads': 0.0,
            }
            continue

        p1b = bg['p1_burst'][ch_mask]
        p2b = bg['p2_burst'][ch_mask]

        # Session-level directed ECA (fast, no surrogates needed)
        eca = directed_eca_burst(p1b, p2b, max_lag_samples=max_lag_samples)

        # Session-level directed TE
        te = directed_te_burst(p1b, p2b, k=k_te)

        # GPU per-timepoint surrogate z-scored TE timecourses
        gpu_te = gpu_sliding_te_surrogates(
            p1b, p2b, window_samples=120, stride_samples=2,
            k=k_te, n_surrogates=n_surrogates, seed=seed, device=device)

        # ECA sliding window (no surrogates — just raw timecourse)
        eca_tc = directed_eca_sliding(p1b, p2b, window_samples=120,
                                       stride_samples=2,
                                       max_lag_samples=max_lag_samples)

        # Resample timecourses to t_common
        tc_local_eca = eca_tc['t_centers'] / feature_rate
        tc_lsl_eca = tc_local_eca + (p1_ts[0] + lsl_offset)
        eca_asym_common = np.interp(t_common, tc_lsl_eca, eca_tc['eca_asym'],
                                     left=0, right=0).astype(np.float32)

        tc_local_te = gpu_te['t_centers'] / feature_rate
        tc_lsl_te = tc_local_te + (p1_ts[0] + lsl_offset)
        te_asym_z_common = np.interp(t_common, tc_lsl_te, gpu_te['te_asym_z'],
                                      left=0, right=0).astype(np.float32)
        te_p1p2_z_common = np.interp(t_common, tc_lsl_te, gpu_te['te_p1p2_z'],
                                      left=0, right=0).astype(np.float32)
        te_p2p1_z_common = np.interp(t_common, tc_lsl_te, gpu_te['te_p2p1_z'],
                                      left=0, right=0).astype(np.float32)
        te_asym_raw_common = np.interp(
            t_common, tc_lsl_te,
            gpu_te['te_p1p2_raw'] - gpu_te['te_p2p1_raw'],
            left=0, right=0).astype(np.float32)

        # Directed episode detection on surrogate-z TE asymmetry
        episodes = detect_directed_episodes(
            te_asym_z_common, threshold_z=2.0,
            min_dur_s=3.0, merge_gap_s=5.0, fs=feature_rate)

        # Session-level surrogate z (mean of per-timepoint z)
        te_session_z = float(te_asym_z_common.mean())

        result[band_name] = {
            'eca_asymmetry': eca['asymmetry'],
            'eca_p1_leads': eca['p1_leads_rate'].tolist(),
            'eca_p2_leads': eca['p2_leads_rate'].tolist(),
            'eca_lags_s': eca['lags_s'].tolist(),
            'te_p1_to_p2': te['te_p1_to_p2'],
            'te_p2_to_p1': te['te_p2_to_p1'],
            'te_asymmetry': te['te_asymmetry'],
            'te_z': te_session_z,
            'n_valid_channels': int(ch_mask.sum()),
            'n_p1_onsets': eca['n_p1_onsets'],
            'n_p2_onsets': eca['n_p2_onsets'],
            'eca_asym_tc': eca_asym_common,
            'te_asym_tc': te_asym_raw_common,
            'te_asym_z_tc': te_asym_z_common,
            'te_p1p2_z_tc': te_p1p2_z_common,
            'te_p2p1_z_tc': te_p2p1_z_common,
            'therapist_leads_episodes': episodes['therapist_leads'],
            'patient_leads_episodes': episodes['patient_leads'],
            'frac_therapist_leads': episodes['frac_therapist'],
            'frac_patient_leads': episodes['frac_patient'],
        }

    return result
