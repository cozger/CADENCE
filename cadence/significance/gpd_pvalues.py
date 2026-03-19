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

    for t in range(T):
        real_val = dr2_real[t]

        # Edge case: negative or NaN dR2
        if np.isnan(real_val) or real_val < 0:
            p_values[t] = 1.0
            continue

        # Pool surrogates from t +/- pool_half
        t_lo = max(0, t - pool_half)
        t_hi = min(T, t + pool_half + 1)
        pool = dr2_surr[:, t_lo:t_hi].ravel()

        # Remove NaN
        pool = pool[np.isfinite(pool)]
        if len(pool) < 20:
            p_values[t] = 1.0
            continue

        # Zero-variance null
        pool_std = np.std(pool)
        if pool_std < 1e-12:
            p_values[t] = 0.5 if abs(real_val - np.mean(pool)) < 1e-12 else (
                1.0 if real_val <= np.mean(pool) else p_floor)
            continue

        pool_median = np.median(pool)

        # Below median: clearly not significant
        if real_val <= pool_median:
            # Rank-based p from pooled distribution
            p_values[t] = max(np.mean(pool >= real_val), p_floor)
            continue

        # Above threshold quantile: GPD tail fit
        u = np.quantile(pool, threshold_quantile)

        if real_val > u:
            exceedances = pool[pool > u] - u

            if len(exceedances) >= 10:
                fit = _fit_gpd_safe(exceedances)
                if fit is not None:
                    shape, _, scale = fit
                    # P(X > real | X > u) via GPD survival function
                    tail_p = genpareto.sf(real_val - u, shape, scale=scale)
                    # Scale by probability of exceeding threshold
                    p_val = (1.0 - threshold_quantile) * tail_p
                    p_values[t] = max(float(p_val), p_floor)
                    continue

            # GPD fit failed: fall through to rank-based
            p_values[t] = max(np.mean(pool >= real_val), p_floor)
        else:
            # Between median and threshold: rank-based
            p_values[t] = max(np.mean(pool >= real_val), p_floor)

    return p_values
