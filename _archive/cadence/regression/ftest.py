"""Analytical F-tests for fast significance screening.

Compares full model (source + AR) vs restricted model (AR only)
at each timepoint. Used for initial pathway screening before
expensive surrogate testing.
"""

import numpy as np
import torch
from scipy.stats import f as f_dist


def f_test_timecourse(r2_full, r2_restricted, p_full, p_restricted, n_effective,
                       alpha=0.05):
    """Compute F-statistic and p-value at each timepoint.

    F = ((R2_full - R2_restr) / (p_full - p_restr)) /
        ((1 - R2_full) / (n - p_full))

    Args:
        r2_full: (T,) full model R-squared (numpy or tensor).
        r2_restricted: (T,) restricted model R-squared.
        p_full: Number of parameters in full model.
        p_restricted: Number of parameters in restricted model.
        n_effective: (T,) effective sample count per timepoint.
        alpha: Significance level for binary significance mask.

    Returns:
        f_stat: (T,) F-statistic per timepoint.
        p_values: (T,) p-values per timepoint.
        significant: (T,) boolean mask at given alpha.
    """
    if isinstance(r2_full, torch.Tensor):
        r2_full = r2_full.cpu().numpy()
    if isinstance(r2_restricted, torch.Tensor):
        r2_restricted = r2_restricted.cpu().numpy()
    if isinstance(n_effective, torch.Tensor):
        n_effective = n_effective.cpu().numpy()

    T = len(r2_full)
    df1 = p_full - p_restricted  # numerator df
    df2 = n_effective - p_full   # denominator df

    # Delta R-squared
    dr2 = np.maximum(r2_full - r2_restricted, 0.0)

    # F-statistic
    numerator = dr2 / max(df1, 1)
    denominator = np.maximum(1.0 - r2_full, 1e-10) / np.maximum(df2, 1.0)

    f_stat = numerator / denominator

    # P-values from F distribution
    p_values = np.ones(T)
    valid = (df2 > 1) & (~np.isnan(f_stat)) & (f_stat >= 0)
    if valid.any():
        p_values[valid] = 1.0 - f_dist.cdf(f_stat[valid], df1, df2[valid])

    # NaN -> p=1
    p_values[np.isnan(p_values)] = 1.0

    significant = p_values < alpha

    return f_stat, p_values, significant


def f_test_static(r2_full, r2_restricted, p_full, p_restricted, n):
    """Single F-test comparing two nested models (static, whole-session).

    Args:
        r2_full: Full model R-squared (scalar).
        r2_restricted: Restricted model R-squared (scalar).
        p_full: Number of parameters in full model.
        p_restricted: Number of parameters in restricted model.
        n: Total number of observations.

    Returns:
        f_stat: F-statistic.
        p_value: p-value.
    """
    df1 = p_full - p_restricted
    df2 = n - p_full

    if df2 <= 0 or df1 <= 0:
        return 0.0, 1.0

    dr2 = max(r2_full - r2_restricted, 0.0)
    f_stat = (dr2 / df1) / (max(1.0 - r2_full, 1e-10) / df2)
    p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)

    return float(f_stat), float(p_value)
