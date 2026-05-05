"""Rank-product coupling detection for temporal localization (Track B, V4).

Threshold-free, model-free approach to detecting episodic interpersonal
coupling in zero-inflated AU signals. Replaces event detection + matching
with a continuous rank-product metric:

1. Rank-normalize each AU to [0, 1] percentiles (handles different scales)
2. Compute lagged rank-products across participants (high when same AUs active)
3. Calibrate against circular-shift null (CADENCE's standard surrogate)
4. HMM on z-scored coupling metric for temporal smoothing

Under independence: E[rank_p1 × rank_p2] = 0.25 (product of two U[0,1])
Under coupling:     E[rank_p1 × rank_p2] > 0.25 (high ranks co-occur)
"""

import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d


def rank_normalize(signal):
    """Rank-transform each channel to [0, 1] percentiles within the session.

    Args:
        signal: (C, T) raw AU values.

    Returns:
        ranked: (C, T) rank-normalized values in [0, 1].
    """
    C, T = signal.shape
    ranked = np.zeros_like(signal)
    for c in range(C):
        ranked[c] = rankdata(signal[c], method='average') / T
    return ranked


def rank_product_coupling(ranked_p1, ranked_p2, fs,
                          lag_range_s=(0.5, 5.0), smooth_s=0.5,
                          au_indices=None, n_lags=10):
    """Compute continuous coupling metric from lagged rank-products.

    For each AU and lag, computes rank_p1[t-lag] × rank_p2[t].
    Takes max over lags (best-matching lag), averages across AUs,
    then Gaussian-smooths.

    Args:
        ranked_p1: (C, T) rank-normalized P1 AUs.
        ranked_p2: (C, T) rank-normalized P2 AUs.
        fs: Sampling rate (Hz).
        lag_range_s: (min, max) lag in seconds.
        smooth_s: Gaussian smoothing sigma in seconds.
        au_indices: Which AUs to include (default: all).
        n_lags: Number of lag values to test within range.

    Returns:
        coupling: (T,) continuous coupling metric.
    """
    C, T = ranked_p1.shape

    if au_indices is not None:
        ranked_p1 = ranked_p1[au_indices]
        ranked_p2 = ranked_p2[au_indices]
        C = len(au_indices)

    lag_min = max(1, int(lag_range_s[0] * fs))
    lag_max = int(lag_range_s[1] * fs)
    lags = np.linspace(lag_min, lag_max, n_lags, dtype=int)
    lags = np.unique(lags)

    # (n_lags, C, T) rank-product at each lag
    rp_best = np.full((C, T), -np.inf)
    for lag in lags:
        # rp[c, t] = ranked_p1[c, t - lag] × ranked_p2[c, t]
        rp = ranked_p1[:, :T - lag] * ranked_p2[:, lag:]
        # Pad front with 0.25 (null expectation)
        rp_padded = np.full((C, T), 0.25)
        rp_padded[:, lag:] = rp
        rp_best = np.maximum(rp_best, rp_padded)

    # Average across AUs
    coupling = rp_best.mean(axis=0)  # (T,)

    # Gaussian smooth
    if smooth_s > 0:
        sigma_samples = smooth_s * fs
        coupling = gaussian_filter1d(coupling, sigma=sigma_samples)

    return coupling


def calibrate_null(ranked_p1, ranked_p2, fs, n_shifts=50,
                   au_indices=None, **kwargs):
    """Estimate null distribution of rank-product metric via circular shifts.

    Args:
        ranked_p1: (C, T) rank-normalized P1 AUs.
        ranked_p2: (C, T) rank-normalized P2 AUs.
        fs: Sampling rate (Hz).
        n_shifts: Number of circular shifts for null estimation.
        au_indices: Which AUs to include.
        **kwargs: Passed to rank_product_coupling.

    Returns:
        null_mean: (T,) per-timepoint null mean.
        null_std: (T,) per-timepoint null std.
    """
    T = ranked_p1.shape[1]
    rng = np.random.RandomState(42)

    # Minimum shift: 30s worth of samples
    min_shift = int(30 * fs)
    max_shift = T - min_shift

    null_stack = np.zeros((n_shifts, T))
    for i in range(n_shifts):
        shift = rng.randint(min_shift, max_shift)
        ranked_p2_shifted = np.roll(ranked_p2, shift, axis=1)
        null_stack[i] = rank_product_coupling(
            ranked_p1, ranked_p2_shifted, fs,
            au_indices=au_indices, **kwargs)

    null_mean = null_stack.mean(axis=0)
    null_std = np.maximum(null_stack.std(axis=0), 1e-8)
    return null_mean, null_std


def rank_coupling_hmm(coupling, null_mean, null_std, p_stay=0.95):
    """2-state HMM on z-scored rank-product coupling metric.

    State 0: z ≈ N(0, 1) — null (rank products at chance level)
    State 1: z ≈ N(mu1, sigma1) — coupled (elevated rank products)

    mu1 and sigma1 are estimated from the top 10% of z-scores
    (warm-start for the coupled state).

    Args:
        coupling: (T,) continuous coupling metric.
        null_mean: (T,) per-timepoint null mean.
        null_std: (T,) per-timepoint null std.
        p_stay: Probability of staying in current regime.

    Returns:
        posterior: (T,) P(coupled | data) in [0, 1].
        params: dict with diagnostics.
    """
    z = (coupling - null_mean) / np.maximum(null_std, 1e-8)
    T = len(z)

    # Estimate coupled state parameters from top 10%
    z_sorted = np.sort(z)
    top_10pct = z_sorted[int(0.9 * T):]
    mu1 = max(float(np.mean(top_10pct)), 0.5)
    sigma1 = max(float(np.std(top_10pct)), 0.5)

    # Transition matrix
    A = np.array([[p_stay, 1 - p_stay],
                  [1 - p_stay, p_stay]])

    # Forward pass (Hamilton filter)
    xi_filt = np.zeros((T, 2))
    xi_filt[0] = [0.9, 0.1]  # Prior: mostly uncoupled

    for t in range(1, T):
        xi_pred = A.T @ xi_filt[t - 1]
        xi_pred = np.maximum(xi_pred, 1e-10)

        # Log-likelihoods under each state
        ll0 = -0.5 * (np.log(2 * np.pi) + z[t] ** 2)
        ll1 = -0.5 * (np.log(2 * np.pi * sigma1 ** 2)
                       + (z[t] - mu1) ** 2 / sigma1 ** 2)

        log_joint = np.array([ll0, ll1]) + np.log(xi_pred)
        log_joint -= log_joint.max()
        joint = np.exp(log_joint)
        xi_filt[t] = joint / max(joint.sum(), 1e-20)

    # Backward smoother
    xi_smooth = np.zeros((T, 2))
    xi_smooth[T - 1] = xi_filt[T - 1]
    for t in range(T - 2, -1, -1):
        xp = A.T @ xi_filt[t]
        xp = np.maximum(xp, 1e-10)
        ratio = xi_smooth[t + 1] / xp
        for j in range(2):
            xi_smooth[t, j] = xi_filt[t, j] * (
                A[j, 0] * ratio[0] + A[j, 1] * ratio[1])
        s = xi_smooth[t].sum()
        if s > 0:
            xi_smooth[t] /= s
        else:
            xi_smooth[t] = xi_filt[t]

    posterior = np.clip(xi_smooth[:, 1], 0.0, 1.0)

    params = {
        'coupling_fraction': float(np.mean(posterior > 0.5)),
        'z_mean': float(z.mean()),
        'z_std': float(z.std()),
        'mu1': mu1,
        'sigma1': sigma1,
        'z': z,
    }
    return posterior, params


def detect_rank_coupling(p1_signal, p2_signal, fs,
                         au_indices=None,
                         lag_range_s=(0.5, 5.0),
                         smooth_s=0.5,
                         n_lags=10,
                         n_shifts=50,
                         p_stay=0.95):
    """Full rank-product coupling detection pipeline.

    Args:
        p1_signal: (C, T) raw P1 AU values.
        p2_signal: (C, T) raw P2 AU values.
        fs: Sampling rate (Hz).
        au_indices: Which AUs to analyze (default: all).
        lag_range_s: Coupling lag range.
        smooth_s: Smoothing kernel width.
        n_lags: Number of lag values.
        n_shifts: Circular shifts for null.
        p_stay: HMM transition parameter.

    Returns:
        posterior: (T,) P(coupled | data) in [0, 1].
        params: dict with all intermediate results.
    """
    # Step 1: Rank normalize
    ranked_p1 = rank_normalize(p1_signal)
    ranked_p2 = rank_normalize(p2_signal)

    rpc_kwargs = dict(lag_range_s=lag_range_s, smooth_s=smooth_s,
                      au_indices=au_indices, n_lags=n_lags)

    # Step 2: Compute coupling metric
    coupling = rank_product_coupling(ranked_p1, ranked_p2, fs, **rpc_kwargs)

    # Step 3: Null calibration
    null_mean, null_std = calibrate_null(
        ranked_p1, ranked_p2, fs, n_shifts=n_shifts, **rpc_kwargs)

    # Step 4: HMM
    posterior, hmm_params = rank_coupling_hmm(
        coupling, null_mean, null_std, p_stay=p_stay)

    params = {
        **hmm_params,
        'coupling_raw': coupling,
        'null_mean': null_mean,
        'null_std': null_std,
    }
    return posterior, params
