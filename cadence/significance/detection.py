"""Session-level coupling detection: binomial test + t-test.

Matches MCCT's dual-test approach for direct comparison.
"""

import numpy as np
from scipy.stats import binomtest, ttest_1samp


def detect_coupling_session(p_values, dr2_values, alpha_binomial=0.01,
                             alpha_ttest=0.01, min_dr2=0.001,
                             chance_level=0.05):
    """Session-level coupling detection via dual statistical test.

    Uses two complementary tests:
    1. Binomial test: is the fraction of significant timepoints > chance level?
    2. One-sample t-test: is the mean dR2 significantly > 0?

    Both must pass to declare coupling detected. This prevents:
    - Binomial alone: triggered by tiny but consistent dR2
    - T-test alone: triggered by a few outlier dR2 values

    Args:
        p_values: (T,) per-timepoint p-values.
        dr2_values: (T,) per-timepoint delta-R2 values.
        alpha_binomial: Significance threshold for binomial test.
        alpha_ttest: Significance threshold for t-test.
        min_dr2: Minimum mean dR2 to be considered meaningful.
        chance_level: Expected false positive rate under null.

    Returns:
        detected: bool, whether coupling was detected.
        details: dict with test statistics.
    """
    # Ensure matching shapes (session-level p may be size-1 if timepoint failed)
    if p_values.shape != dr2_values.shape:
        if p_values.size == 1:
            p_values = np.full_like(dr2_values, p_values.flat[0])
        elif dr2_values.size == 1:
            dr2_values = np.full_like(p_values, dr2_values.flat[0])

    # Clean NaN
    valid = ~np.isnan(dr2_values) & ~np.isnan(p_values)
    p_clean = p_values[valid]
    dr2_clean = dr2_values[valid]

    if len(dr2_clean) < 5:
        return False, {'reason': 'insufficient_data', 'n_valid': len(dr2_clean)}

    n_sig = np.sum(p_clean < 0.05)
    n_total = len(p_clean)
    sig_rate = n_sig / n_total

    # Test 1: Binomial - sig_rate > chance level?
    binom_result = binomtest(n_sig, n_total, chance_level, alternative='greater')
    binom_p = binom_result.pvalue

    # Test 2: t-test - mean dR2 > 0?
    mean_dr2 = np.mean(dr2_clean)
    if np.std(dr2_clean) > 1e-12 and len(dr2_clean) > 1:
        t_stat, t_p = ttest_1samp(dr2_clean, 0, alternative='greater')
    else:
        t_stat, t_p = 0.0, 1.0

    # Both tests + minimum effect size
    detected = (binom_p < alpha_binomial and
                t_p < alpha_ttest and
                mean_dr2 > min_dr2)

    details = {
        'detected': detected,
        'sig_rate': sig_rate,
        'n_sig': int(n_sig),
        'n_total': n_total,
        'binom_p': float(binom_p),
        'mean_dr2': float(mean_dr2),
        't_stat': float(t_stat),
        't_p': float(t_p),
    }

    return detected, details


def detection_summary(result, config=None):
    """Summarize detection results across all pathways.

    Args:
        result: CouplingResult from CouplingEstimator.
        config: Config dict (for significance thresholds).

    Returns:
        summary: dict mapping (src_mod, tgt_mod) -> detection details.
    """
    if config is None:
        from cadence.config import load_config
        config = load_config()

    sig_cfg = config['significance']['session_level']

    summary = {}
    for key in result.pathway_dr2:
        dr2 = result.pathway_dr2[key]
        p_values = result.pathway_pvalues.get(key, np.ones_like(dr2))

        detected, details = detect_coupling_session(
            p_values, dr2,
            alpha_binomial=sig_cfg['binomial_alpha'],
            alpha_ttest=sig_cfg['ttest_alpha'],
            min_dr2=sig_cfg['min_dr2'],
        )
        summary[key] = details

    return summary
