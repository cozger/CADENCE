"""GPD (Generalized Pareto Distribution) parametric null for continuous p-values.

With K=20 surrogates, rank-based p-values are coarse (0.048 or >=0.095).
GPD tail fitting produces continuous p-values by modelling the upper tail
of the pooled surrogate distribution parametrically.

Uses temporal pooling: surrogates from t +/- pool_half are combined,
giving ~2000 null samples from K=20 surrogates with pool_half=50.
"""

import numpy as np
from scipy.stats import genpareto


def _fit_gpd_safe(exceedances):
    """Fit GPD to exceedances with shape bounds and validation.

    Returns (shape, loc, scale) or None if fit fails validation.
    """
    if len(exceedances) < 10:
        return None

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception:
        return None

    # Validate: reject degenerate fits
    if scale <= 0 or abs(shape) > 0.5:
        return None

    return shape, loc, scale


def gpd_tail_pvalues(dr2_real, dr2_surr, pool_half=50,
                     threshold_quantile=0.9, p_floor=None):
    """Compute continuous p-values using GPD tail fitting on pooled surrogates.

    Args:
        dr2_real: (T,) smoothed real dR2.
        dr2_surr: (K, T) smoothed surrogate dR2.
        pool_half: Half-width of temporal pooling window (in samples).
        threshold_quantile: Quantile for GPD threshold (e.g. 0.9).
        p_floor: Minimum p-value. Default: 1 / (10 * K).

    Returns:
        p_values: (T,) continuous p-values.
    """
    K, T = dr2_surr.shape
    if p_floor is None:
        p_floor = 1.0 / (10 * K)

    # Edge case: very short timeseries
    if T < 10:
        pool_half = max(pool_half, T // 4)
        pool_half = max(pool_half, 2)

    p_values = np.ones(T)

    # --- Pre-compute pooled statistics for all timepoints ---
    # Sliding window median and rank-based p via cumsum trick.
    # For each t, pool = dr2_surr[:, max(0,t-ph):min(T,t+ph+1)].ravel()
    # Pre-compute cumulative sum for rank-based p-values.
    # Vectorize the easy cases (~90% of timepoints), loop only for GPD tail.

    # Pre-compute per-timepoint pool median and quantile using sliding window
    # (avoids redundant slicing in the loop for the ~10% GPD cases)
    pool_medians = np.empty(T)
    pool_thresholds = np.empty(T)
    pool_rank_p = np.empty(T)  # rank-based p for all t
    valid_pool = np.ones(T, dtype=bool)

    for t in range(T):
        t_lo = max(0, t - pool_half)
        t_hi = min(T, t + pool_half + 1)
        pool = dr2_surr[:, t_lo:t_hi].ravel()
        pool = pool[np.isfinite(pool)]
        if len(pool) < 20:
            valid_pool[t] = False
            continue
        pool_medians[t] = np.median(pool)
        pool_thresholds[t] = np.quantile(pool, threshold_quantile)
        pool_rank_p[t] = np.mean(pool >= dr2_real[t])

    # Classify timepoints into fast paths
    bad_real = np.isnan(dr2_real) | (dr2_real < 0) | ~valid_pool
    p_values[bad_real] = 1.0

    # Below median: rank-based (fast path)
    below_median = ~bad_real & (dr2_real <= pool_medians)
    p_values[below_median] = np.maximum(pool_rank_p[below_median], p_floor)

    # Between median and threshold: rank-based (fast path)
    between = ~bad_real & ~below_median & (dr2_real <= pool_thresholds)
    p_values[between] = np.maximum(pool_rank_p[between], p_floor)

    # Above threshold: need GPD tail fitting (loop only over ~10%)
    needs_gpd = ~bad_real & ~below_median & ~between
    gpd_indices = np.where(needs_gpd)[0]

    for t in gpd_indices:
        real_val = dr2_real[t]
        t_lo = max(0, t - pool_half)
        t_hi = min(T, t + pool_half + 1)
        pool = dr2_surr[:, t_lo:t_hi].ravel()
        pool = pool[np.isfinite(pool)]

        u = pool_thresholds[t]
        exceedances = pool[pool > u] - u

        if len(exceedances) >= 10:
            fit = _fit_gpd_safe(exceedances)
            if fit is not None:
                shape, _, scale = fit
                tail_p = genpareto.sf(real_val - u, shape, scale=scale)
                p_val = (1.0 - threshold_quantile) * tail_p
                p_values[t] = max(float(p_val), p_floor)
                continue

        # GPD fit failed: fall through to rank-based
        p_values[t] = max(float(pool_rank_p[t]), p_floor)

    return p_values
