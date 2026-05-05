"""Analytical significance: F-test + Benjamini-Hochberg FDR correction."""

import numpy as np
from cadence.regression.ftest import f_test_timecourse


def bh_fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: (T,) array of p-values.
        alpha: Target FDR level.

    Returns:
        significant: (T,) boolean mask after FDR correction.
        adjusted_p: (T,) BH-adjusted p-values.
    """
    T = len(p_values)
    if T == 0:
        return np.array([], dtype=bool), np.array([])

    # Handle NaN p-values
    nan_mask = np.isnan(p_values)
    p_clean = np.where(nan_mask, 1.0, p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_clean)
    sorted_p = p_clean[sorted_idx]

    # BH procedure: p_adj[i] = p[i] * T / (rank[i])
    ranks = np.arange(1, T + 1)
    adjusted = sorted_p * T / ranks

    # Enforce monotonicity (cumulative minimum from the right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Map back to original order
    adjusted_p = np.empty(T)
    adjusted_p[sorted_idx] = adjusted

    # NaN positions stay p=1
    adjusted_p[nan_mask] = 1.0

    significant = adjusted_p < alpha

    return significant, adjusted_p


def fdr_corrected_f_test(r2_full, r2_restricted, p_full, p_restricted,
                          n_effective, alpha=0.05):
    """F-test with BH-FDR correction across timepoints.

    Args:
        r2_full: (T,) full model R-squared.
        r2_restricted: (T,) restricted model R-squared.
        p_full: Number of parameters in full model.
        p_restricted: Number of parameters in restricted model.
        n_effective: (T,) effective sample count.
        alpha: FDR level.

    Returns:
        significant: (T,) boolean mask after FDR correction.
        p_values_raw: (T,) raw p-values.
        p_values_adj: (T,) BH-adjusted p-values.
        f_stat: (T,) F-statistics.
    """
    f_stat, p_values_raw, _ = f_test_timecourse(
        r2_full, r2_restricted, p_full, p_restricted, n_effective, alpha=alpha)

    significant, p_values_adj = bh_fdr_correction(p_values_raw, alpha=alpha)

    return significant, p_values_raw, p_values_adj, f_stat
