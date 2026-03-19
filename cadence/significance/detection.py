"""Session-level coupling detection via 2-state Gaussian HMM.

Replaces V2's binomial + t-test AND-gate with HMM coupling-state model.
The HMM naturally detects episodic coupling by modelling transitions
between uncoupled (state 0) and coupled (state 1) states.
"""

import numpy as np


def _forward_backward(log_emissions, log_A, log_pi):
    """Standard 2-state forward-backward in log-space.

    Args:
        log_emissions: (T, 2) log emission probabilities.
        log_A: (2, 2) log transition matrix.
        log_pi: (2,) log initial state probabilities.

    Returns:
        gamma: (T, 2) posterior state probabilities.
        log_likelihood: scalar total log-likelihood.
    """
    T = log_emissions.shape[0]

    # Forward pass (log-space with normalization)
    log_alpha = np.zeros((T, 2))
    log_alpha[0] = log_pi + log_emissions[0]
    # Normalize
    log_norm = np.logaddexp(log_alpha[0, 0], log_alpha[0, 1])
    log_alpha[0] -= log_norm
    log_evidence = log_norm

    for t in range(1, T):
        for j in range(2):
            log_alpha[t, j] = np.logaddexp(
                log_alpha[t-1, 0] + log_A[0, j],
                log_alpha[t-1, 1] + log_A[1, j]
            ) + log_emissions[t, j]
        log_norm = np.logaddexp(log_alpha[t, 0], log_alpha[t, 1])
        log_alpha[t] -= log_norm
        log_evidence += log_norm

    # Backward pass
    log_beta = np.zeros((T, 2))
    for t in range(T - 2, -1, -1):
        for i in range(2):
            log_beta[t, i] = np.logaddexp(
                log_A[i, 0] + log_emissions[t+1, 0] + log_beta[t+1, 0],
                log_A[i, 1] + log_emissions[t+1, 1] + log_beta[t+1, 1]
            )
        log_norm = np.logaddexp(log_beta[t, 0], log_beta[t, 1])
        if np.isfinite(log_norm):
            log_beta[t] -= log_norm

    # Posterior
    log_gamma = log_alpha + log_beta
    log_gamma_norm = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])[:, None]
    log_gamma -= log_gamma_norm
    gamma = np.exp(log_gamma)

    return gamma, log_evidence


def _fit_hmm(dr2, null_mu, null_sigma, eval_rate, hmm_cfg):
    """Fit 2-state Gaussian HMM via Baum-Welch EM.

    State 0 = uncoupled (null distribution).
    State 1 = coupled (higher dR2).

    Args:
        dr2: (T,) dR2 timecourse.
        null_mu: Mean of null (surrogate) dR2 distribution.
        null_sigma: Std of null dR2 distribution.
        eval_rate: Sampling rate of dr2 in Hz.
        hmm_cfg: Dict with episode_duration_s, max_em_iter, em_tol.

    Returns:
        gamma: (T, 2) posterior probabilities.
        params: dict with mu_0, sigma_0, mu_1, sigma_1.
        log_likelihood: scalar.
    """
    T = len(dr2)
    episode_s = hmm_cfg.get('episode_duration_s', 60.0)
    max_iter = hmm_cfg.get('max_em_iter', 20)
    em_tol = hmm_cfg.get('em_tol', 1e-6)

    # Initialize emission parameters
    mu_0 = null_mu
    sigma_0 = max(null_sigma, 1e-8)

    # State 1: initialized from upper quartile of dR2
    upper = dr2[dr2 > np.percentile(dr2, 75)]
    if len(upper) < 5:
        upper = dr2[dr2 > np.median(dr2)]
    mu_1 = max(float(np.mean(upper)), mu_0 + sigma_0)
    sigma_1 = max(float(np.std(upper)), sigma_0 * 0.5)

    # Transition probabilities (per-step, scaled by eval_rate)
    # alpha: P(0->1) — rare transitions into coupled state
    # beta: P(1->0) — episodes end after ~episode_duration_s
    alpha = min(0.3, max(1e-6, 1.0 / (episode_s * eval_rate) * 0.01))
    beta = min(0.3, max(1e-6, 1.0 / (episode_s * eval_rate)))

    # Initial state: mostly uncoupled
    log_pi = np.log(np.array([0.95, 0.05]))

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # Build transition matrix
        log_A = np.log(np.array([
            [1.0 - alpha, alpha],
            [beta, 1.0 - beta],
        ]))

        # Emission log-probabilities
        log_emissions = np.zeros((T, 2))
        for t in range(T):
            x = dr2[t]
            log_emissions[t, 0] = _log_normal(x, mu_0, sigma_0)
            log_emissions[t, 1] = _log_normal(x, mu_1, sigma_1)

        # E-step
        gamma, ll = _forward_backward(log_emissions, log_A, log_pi)

        # Check convergence
        if abs(ll - prev_ll) < em_tol * max(1.0, abs(ll)):
            break
        prev_ll = ll

        # M-step
        g0 = gamma[:, 0]
        g1 = gamma[:, 1]
        w0 = np.sum(g0)
        w1 = np.sum(g1)

        # State 0: anchor mu_0 to surrogate estimate (fixed)
        if w0 > 1e-10:
            sigma_0 = max(
                np.sqrt(np.sum(g0 * (dr2 - mu_0)**2) / w0),
                1e-8)

        # State 1: update freely
        if w1 > 1e-10:
            mu_1 = float(np.sum(g1 * dr2) / w1)
            sigma_1 = max(
                np.sqrt(np.sum(g1 * (dr2 - mu_1)**2) / w1),
                1e-8)

        # Enforce separation: mu_1 > mu_0 + sigma_0
        if mu_1 <= mu_0 + sigma_0:
            mu_1 = mu_0 + sigma_0 * 1.01

        # Update transitions from expected counts
        # Sum of expected transitions i->j
        n_00 = n_01 = n_10 = n_11 = 0.0
        for t in range(T - 1):
            for i in range(2):
                for j in range(2):
                    xi_ij = (gamma[t, i] *
                             np.exp(log_A[i, j] + log_emissions[t+1, j]) *
                             gamma[t+1, j])
                    if i == 0 and j == 0:
                        n_00 += xi_ij
                    elif i == 0 and j == 1:
                        n_01 += xi_ij
                    elif i == 1 and j == 0:
                        n_10 += xi_ij
                    else:
                        n_11 += xi_ij

        if n_00 + n_01 > 1e-10:
            alpha = np.clip(n_01 / (n_00 + n_01), 1e-6, 0.3)
        if n_10 + n_11 > 1e-10:
            beta = np.clip(n_10 / (n_10 + n_11), 1e-6, 0.3)

    params = {
        'mu_0': mu_0, 'sigma_0': sigma_0,
        'mu_1': mu_1, 'sigma_1': sigma_1,
    }
    return gamma, params, ll


def _log_normal(x, mu, sigma):
    """Log of normal PDF, safe for small sigma."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def detect_coupling_hmm(dr2, null_stats, eval_rate, hmm_cfg):
    """Detect coupling using 2-state Gaussian HMM.

    Args:
        dr2: (T,) dR2 timecourse.
        null_stats: Dict with 'mu_0', 'sigma_0' from surrogate distribution.
            If None, estimated from lower 50% of dR2.
        eval_rate: Sampling rate of dR2 in Hz.
        hmm_cfg: Dict with episode_duration_s, min_coupling_fraction, min_dr2.

    Returns:
        detected: bool.
        details: dict with coupling_fraction, coupling_posterior, etc.
    """
    min_coupling_frac = hmm_cfg.get('min_coupling_fraction', 0.03)
    min_dr2 = hmm_cfg.get('min_dr2', 0.001)

    # Clean NaN
    valid = np.isfinite(dr2)
    if np.sum(valid) < 20:
        return False, {
            'detected': False,
            'reason': 'insufficient_data',
            'n_valid': int(np.sum(valid)),
            'p_value': 1.0,
            'coupling_fraction': 0.0,
            'mean_dr2': 0.0,
            'sig_rate': 0.0,
        }

    dr2_clean = np.where(valid, dr2, 0.0)
    mean_dr2 = float(np.nanmean(dr2[valid]))

    # Constant dR2
    if np.std(dr2_clean[valid]) < 1e-12:
        det = mean_dr2 > min_dr2
        return det, {
            'detected': det,
            'method': 'constant',
            'mean_dr2': mean_dr2,
            'p_value': 0.0 if det else 1.0,
            'coupling_fraction': 1.0 if det else 0.0,
            'sig_rate': 0.0,
        }

    # All negative
    if np.all(dr2_clean[valid] <= 0):
        return False, {
            'detected': False,
            'reason': 'all_negative',
            'mean_dr2': mean_dr2,
            'p_value': 1.0,
            'coupling_fraction': 0.0,
            'sig_rate': 0.0,
        }

    # Null distribution statistics
    if null_stats is not None:
        null_mu = null_stats.get('mu_0', 0.0)
        null_sigma = null_stats.get('sigma_0', 1.0)
    else:
        # Estimate from lower 50% of dR2
        lower_half = dr2_clean[valid]
        lower_half = lower_half[lower_half <= np.median(lower_half)]
        null_mu = float(np.mean(lower_half)) if len(lower_half) > 0 else 0.0
        null_sigma = float(np.std(lower_half)) if len(lower_half) > 1 else 1.0

    null_sigma = max(null_sigma, 1e-8)

    # Fit HMM
    gamma, params, ll = _fit_hmm(dr2_clean, null_mu, null_sigma,
                                  eval_rate, hmm_cfg)

    coupling_posterior = gamma[:, 1]
    coupling_fraction = float(np.mean(coupling_posterior[valid]))

    # Mean dR2 during coupled state
    coupled_mask = coupling_posterior > 0.5
    if np.any(coupled_mask & valid):
        mean_coupled_dr2 = float(np.mean(dr2_clean[coupled_mask & valid]))
    else:
        mean_coupled_dr2 = 0.0

    mu_0 = params['mu_0']
    sigma_0 = params['sigma_0']
    mu_1 = params['mu_1']
    sigma_1 = params['sigma_1']

    # States separable?
    separable = mu_1 > mu_0 + sigma_0

    # Above-null criterion: coupled state must exceed surrogate null by 2*sigma
    # Prevents false positives from feature selection bias (double-dipping)
    if null_stats is not None:
        above_null = mean_coupled_dr2 > null_mu + 2 * null_sigma
    else:
        above_null = mean_coupled_dr2 > min_dr2

    # Detection criteria
    detected = (
        coupling_fraction > min_coupling_frac
        and above_null
        and separable
    )

    # Continuous score: 1 - coupling_fraction
    # Lower = more likely coupled (same semantics as binom_p for ROC)
    p_value = 1.0 - coupling_fraction

    # Compute sig_rate for backward compatibility logging
    # (fraction of timepoints with coupling posterior > 0.5)
    sig_rate = float(np.mean(coupled_mask[valid]))

    details = {
        'detected': detected,
        'coupling_fraction': coupling_fraction,
        'coupling_posterior': coupling_posterior,
        'p_value': p_value,
        'mean_dr2': mean_dr2,
        'mean_coupled_dr2': mean_coupled_dr2,
        'mu_0': mu_0, 'sigma_0': sigma_0,
        'mu_1': mu_1, 'sigma_1': sigma_1,
        'log_likelihood': float(ll),
        'sig_rate': sig_rate,
    }

    if not separable:
        details['reason'] = 'no_coupled_state'

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
    hmm_cfg = sig_cfg.get('hmm', {})
    hmm_cfg['min_dr2'] = sig_cfg.get('min_dr2', 0.001)

    # Get eval_rate overrides
    eval_rate_overrides = config.get('eval_rate_overrides', {})
    default_eval_rate = config.get('ewls', {}).get('eval_rate', 2.0)

    # Get null stats from result if available
    null_stats_dict = getattr(result, 'pathway_null_stats', {})

    summary = {}
    for key in result.pathway_dr2:
        dr2 = result.pathway_dr2[key]

        # Determine eval rate for this pathway
        src_mod = key[0]
        eval_rate = eval_rate_overrides.get(src_mod, default_eval_rate)

        # Get null stats for this pathway
        null_stats = null_stats_dict.get(key, None)

        detected, details = detect_coupling_hmm(
            dr2, null_stats, eval_rate, hmm_cfg)

        # Remove coupling_posterior from summary (large array)
        summary_details = {k: v for k, v in details.items()
                           if k != 'coupling_posterior'}
        summary[key] = summary_details

    return summary
