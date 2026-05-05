"""Temporal localization via z-score + Hidden Semi-Markov Model at native rate.

V2: Short-tau EWLS (tau=1.0s) produces sharp dR2 timecourses. HSMM runs at
native eval_rate (no downsampling) for maximum temporal fidelity.

V3: Adds zscore_stouffer_whitened() with Mahalanobis decorrelation and
coherence-based temporal localization dispatch.

Pipeline:
  Stage 1: Per-channel z-score + Stouffer aggregation (whitened in V3)
  Stage 2: Hidden Semi-Markov Model (duration-aware segmentation)
  Stage 3: Surrogate null calibration (false alarm control)

Key properties:
  - No downsampling — HSMM operates at native eval_rate
  - No smoothing — short tau already provides temporal structure
  - Z-score preserves magnitude information for sensitivity
  - HSMM duration constraints prevent runaway state occupancy
  - Dynamic batch sizing keeps calibration memory bounded
  - Whitened Stouffer restores null_sigma ≈ 1.0 despite inter-channel correlation
"""

import numpy as np
from scipy.stats import rankdata, norm
from scipy.ndimage import uniform_filter1d
from scipy.special import logsumexp as _logsumexp


# ---------------------------------------------------------------------------
# Stage 1: Z-score front-end
# ---------------------------------------------------------------------------

def zscore_stouffer(dr2_real_pc, dr2_surr_pc, eval_rate, smooth_s=3.0):
    """Per-channel z-score + Stouffer aggregation.

    Preserves magnitude information (unlike CDF rank logit) for better
    sensitivity to weak coupling. Under null, z_agg ~ N(0, 1).

    Args:
        dr2_real_pc: (C, T) real per-channel dR2.
        dr2_surr_pc: (K, C, T) surrogate per-channel dR2.
        eval_rate: Hz.
        smooth_s: smoothing window for null stats in seconds.

    Returns:
        z_agg: (T,) Stouffer-aggregated z-score.
        z_agg_null: (K, T) null z-scores (for calibration).
        null_mu: mean of null z_agg distribution.
        null_sigma: std of null z_agg distribution.
    """
    K, C, T = dr2_surr_pc.shape

    real_clean = np.nan_to_num(dr2_real_pc, nan=0.0)
    surr_clean = np.nan_to_num(dr2_surr_pc, nan=0.0)

    # Null mean/std per channel, smoothed over time for stability
    null_smooth = max(1, int(smooth_s * eval_rate))
    surr_mean = np.nanmean(surr_clean, axis=0)  # (C, T)
    surr_std = np.nanstd(surr_clean, axis=0)    # (C, T)

    mu_ct = uniform_filter1d(surr_mean, null_smooth, axis=1, mode='nearest')
    sig_ct = np.maximum(
        uniform_filter1d(surr_std, null_smooth, axis=1, mode='nearest'), 1e-8)

    # Per-channel z-scores for real data
    z_real = (real_clean - mu_ct) / sig_ct  # (C, T)

    # Stouffer aggregation
    sqrt_C = np.sqrt(max(C, 1))
    z_agg = np.nansum(z_real, axis=0) / sqrt_C  # (T,)

    # Null z-scores for each surrogate
    z_agg_null = np.zeros((K, T))
    for k in range(K):
        z_k = (surr_clean[k] - mu_ct) / sig_ct  # (C, T)
        z_agg_null[k] = np.nansum(z_k, axis=0) / sqrt_C

    null_mu = float(z_agg_null.mean())
    null_sigma = max(float(z_agg_null.std()), 1e-6)

    return z_agg, z_agg_null, null_mu, null_sigma


def zscore_stouffer_whitened(dr2_real_pc, dr2_surr_pc, eval_rate,
                              smooth_s=3.0, regularization=1e-4):
    """Per-channel z-score + covariance-corrected Stouffer aggregation.

    Standard Stouffer: z_agg = Σz_c / √C, assumes independence.
    Corrected:         z_agg = Σz_c / √(1'Σ1), accounts for correlation.

    The key issue: standard Stouffer assumes independent channels, but
    EEG/blendshape channels are correlated. With C=15 and inter-channel
    ρ≈0.4, 1'Σ1 ≈ 99 vs C=15, so the standard denominator is 2.5x too
    small, inflating null_sigma from ~1.0 to ~2.5.

    Args:
        dr2_real_pc: (C, T) real per-channel dR2.
        dr2_surr_pc: (K, C, T) surrogate per-channel dR2.
        eval_rate: Hz.
        smooth_s: smoothing window for null stats in seconds.
        regularization: ridge for covariance matrix stability.

    Returns:
        z_agg: (T,) covariance-corrected Stouffer z-score.
        z_agg_null: (K, T) null z-scores (for calibration).
        null_mu: mean of null z_agg distribution.
        null_sigma: std of null z_agg distribution.
        whiten_meta: dict with diagnostics (C_eff, eigvals).
    """
    K, C, T = dr2_surr_pc.shape

    real_clean = np.nan_to_num(dr2_real_pc, nan=0.0)
    surr_clean = np.nan_to_num(dr2_surr_pc, nan=0.0)

    # Null mean/std per channel, smoothed over time for stability
    null_smooth = max(1, int(smooth_s * eval_rate))
    surr_mean = np.nanmean(surr_clean, axis=0)  # (C, T)
    surr_std = np.nanstd(surr_clean, axis=0)    # (C, T)

    mu_ct = uniform_filter1d(surr_mean, null_smooth, axis=1, mode='nearest')
    sig_ct = np.maximum(
        uniform_filter1d(surr_std, null_smooth, axis=1, mode='nearest'), 1e-8)

    # Per-channel z-scores for real data
    z_real = (real_clean - mu_ct) / sig_ct  # (C, T)

    # Per-channel z-scores for all surrogates (vectorized broadcast)
    z_surr = (surr_clean - mu_ct[None]) / sig_ct[None]  # (K, C, T)

    # Estimate C×C covariance from surrogate z-scores (pool K×T)
    # z_surr is (K, C, T) — transpose to (C, K, T) then flatten to (C, K*T)
    z_pool = z_surr.transpose(1, 0, 2).reshape(C, K * T)  # (C, K*T)
    Sigma = np.cov(z_pool)  # (C, C)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])

    # Regularize covariance
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, regularization)
    Sigma_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Normalization: sqrt(1'Σ1) — the true variance of sum(z_c)
    ones = np.ones(C)
    sum_var = float(ones @ Sigma_reg @ ones)  # 1'Σ1
    sum_var = max(sum_var, 1.0)
    norm_factor = np.sqrt(sum_var)

    # Effective channel count: C_eff = C² / (1'Σ1)
    # Measures how many independent channels the correlated set is worth
    C_eff = C ** 2 / sum_var

    # Covariance-corrected aggregation
    z_agg = np.sum(z_real, axis=0) / norm_factor  # (T,)

    # Null z-scores with same normalization (vectorized)
    z_agg_null = z_surr.sum(axis=1) / norm_factor  # (K, T)

    null_mu = float(z_agg_null.mean())
    null_sigma = max(float(z_agg_null.std()), 1e-6)

    whiten_meta = {
        'C_eff': C_eff,
        'sum_var': sum_var,
        'null_mu': null_mu,
        'null_sigma': null_sigma,
        'eigvals': eigvals.tolist(),
        'C_raw': C,
    }

    return z_agg, z_agg_null, null_mu, null_sigma, whiten_meta


# ---------------------------------------------------------------------------
# Stage 2: Hidden Semi-Markov Model
# ---------------------------------------------------------------------------

def _truncated_normal_log_pmf(D_max, d_min, mu_d, sigma_d):
    """Log-PMF of truncated normal over integer durations.

    Index i in the returned array corresponds to total duration (i+1) samples.
    Entries outside [d_min, D_max] are -inf.
    """
    log_pmf = np.full(D_max, -np.inf)
    if d_min > D_max:
        return log_pmf

    d_vals = np.arange(d_min, D_max + 1, dtype=np.float64)
    log_p = norm.logpdf(d_vals, loc=mu_d, scale=max(sigma_d, 0.5))
    log_p -= _logsumexp(log_p)

    log_pmf[d_min - 1:D_max] = log_p
    return log_pmf


def _hsmm_forward_backward(obs, null_mu, null_sigma,
                            coupled_mu, coupled_sigma,
                            log_p_exit, log_1mp_exit,
                            log_dur_pmf, D_max):
    """2-state HSMM forward-backward in log-space.

    State 0: geometric duration (standard HMM recursion), emission fixed.
    State 1: explicit duration from log_dur_pmf, emission learned.

    Forward variable:
      f0[t]    = log P(z_t=0, y_{1:t})
      f1[t, d] = log P(z_t=1, remaining=d+1, y_{1:t})  for d=0..D_max-1

    Duration convention:
      d=0      -> remaining=1 (last timestep of state 1 segment)
      d=D_max-1 -> remaining=D_max (just entered state 1)

    Returns:
        gamma_1: (T,) posterior P(z_t = 1 | y).
        log_lik: scalar log-likelihood.
    """
    T = len(obs)
    if T < 2:
        return np.zeros(T), -np.inf

    # Emission log-probabilities
    log_b0 = (-0.5 * np.log(2 * np.pi) - np.log(null_sigma)
              - 0.5 * ((obs - null_mu) / null_sigma) ** 2)
    log_b1 = (-0.5 * np.log(2 * np.pi) - np.log(coupled_sigma)
              - 0.5 * ((obs - coupled_mu) / coupled_sigma) ** 2)

    # ---- Forward pass ----
    log_f0 = np.full(T, -np.inf)
    log_f1 = np.full((T, D_max), -np.inf)

    # t=0: prior (favour null state)
    log_f0[0] = np.log(0.9) + log_b0[0]
    valid_d = log_dur_pmf > -1e10
    if valid_d.any():
        log_f1[0, valid_d] = np.log(0.1) + log_dur_pmf[valid_d] + log_b1[0]

    for t in range(1, T):
        # State 0: from state 0 (stay) or state 1 ending (d_idx=0)
        log_f0[t] = log_b0[t] + np.logaddexp(
            log_f0[t - 1] + log_1mp_exit,
            log_f1[t - 1, 0])

        # State 1 continuing: f1[t-1, d+1] -> f1[t, d]
        cont = np.full(D_max, -np.inf)
        if D_max > 1:
            cont[:D_max - 1] = log_f1[t - 1, 1:] + log_b1[t]

        # State 1 entry from state 0
        entry = log_f0[t - 1] + log_p_exit + log_dur_pmf + log_b1[t]

        log_f1[t] = np.logaddexp(cont, entry)

    log_lik = float(np.logaddexp(log_f0[-1], _logsumexp(log_f1[-1])))

    # ---- Backward pass ----
    log_a0 = np.zeros(T)
    log_a1 = np.zeros((T, D_max))

    for t in range(T - 2, -1, -1):
        # State 0 backward
        log_stay = log_1mp_exit + log_b0[t + 1] + log_a0[t + 1]
        log_exit_arr = (log_p_exit + log_dur_pmf
                        + log_b1[t + 1] + log_a1[t + 1])
        log_a0[t] = np.logaddexp(log_stay, _logsumexp(log_exit_arr))

        # State 1 backward
        # d=0: segment ends -> must go to state 0
        log_a1[t, 0] = log_b0[t + 1] + log_a0[t + 1]
        # d>0: continue in state 1, duration decrements
        if D_max > 1:
            log_a1[t, 1:] = log_b1[t + 1] + log_a1[t + 1, :D_max - 1]

    # ---- Posterior ----
    log_g0 = log_f0 + log_a0
    log_g1 = _logsumexp(log_f1 + log_a1, axis=1)
    log_norm = np.logaddexp(log_g0, log_g1)

    gamma_1 = np.exp(np.clip(log_g1 - log_norm, -50, 0))
    return np.clip(gamma_1, 0.0, 1.0), log_lik


def hsmm_em(obs, null_mu, null_sigma, obs_rate,
            d_min, D_max, mu_d, sigma_d, max_iter=20):
    """EM for 2-state HSMM. State 0 emission is fixed; state 1 is learned.

    Args:
        obs: (T,) observation timecourse (z-score).
        null_mu, null_sigma: state 0 emission parameters (NEVER updated).
        obs_rate: sample rate of obs in Hz.
        d_min, D_max: duration bounds for state 1 in samples.
        mu_d, sigma_d: initial duration distribution parameters in samples.
        max_iter: maximum EM iterations.

    Returns:
        posterior: (T,) P(z_t = 1 | y).
        params: dict with learned parameters.
    """
    T = len(obs)

    # Initialize state 1 emission from upper quartile
    # Identifiability floor: coupled_mu > null_mu + 0.3*null_sigma
    mu_floor = null_mu + 0.3 * null_sigma
    upper = obs[obs > np.percentile(obs, 75)]
    if len(upper) < 5:
        upper = obs[obs > np.median(obs)]
    if len(upper) >= 3:
        coupled_mu = max(float(np.mean(upper)), mu_floor)
        coupled_sigma = max(float(np.std(upper)), null_sigma * 0.5)
    else:
        coupled_mu = null_mu + 2 * null_sigma
        coupled_sigma = null_sigma

    # Initialize p_exit: expected null segment ~10 seconds
    p_exit = np.clip(1.0 / (10.0 * obs_rate), 0.001, 0.3)

    log_dur_pmf = _truncated_normal_log_pmf(D_max, d_min, mu_d, sigma_d)

    prev_ll = -np.inf
    posterior = np.zeros(T)
    ll = -np.inf

    for it in range(max_iter):
        lpe = np.log(max(p_exit, 1e-10))
        l1mpe = np.log(max(1 - p_exit, 1e-10))

        posterior, ll = _hsmm_forward_backward(
            obs, null_mu, null_sigma, coupled_mu, coupled_sigma,
            lpe, l1mpe, log_dur_pmf, D_max)

        if abs(ll - prev_ll) < 1e-6 * max(1.0, abs(ll)):
            break
        prev_ll = ll

        # M-step: update state 1 emission and p_exit
        g1 = posterior
        g0 = 1.0 - g1
        w1 = g1.sum()

        # State 1 emission (enforce identifiability: mu_1 > mu_0 + 0.3*sigma_0)
        if w1 > 5:
            new_mu = float(np.sum(g1 * obs) / w1)
            coupled_mu = max(new_mu, mu_floor)
            residual = (obs - coupled_mu) ** 2
            coupled_sigma = max(
                float(np.sqrt(np.sum(g1 * residual) / w1)),
                null_sigma * 0.3)

        # p_exit from approximate transition counts
        if T > 2:
            n_01 = float(np.sum(g0[:-1] * g1[1:]))
            n_00 = float(np.sum(g0[:-1] * g0[1:]))
            denom = n_00 + n_01
            if denom > 1:
                p_exit = float(np.clip(n_01 / denom, 0.001, 0.3))

    params = {
        'coupled_mu': coupled_mu,
        'coupled_sigma': coupled_sigma,
        'p_exit': p_exit,
        'null_mu': null_mu,
        'null_sigma': null_sigma,
        'log_dur_pmf': log_dur_pmf,
        'D_max': D_max,
        'd_min': d_min,
        'iterations': it + 1,
        'log_likelihood': ll,
    }

    return posterior, params


def _hsmm_fwd_bwd_batch(obs_batch, null_mu, null_sigma,
                         coupled_mu, coupled_sigma,
                         log_p_exit, log_1mp_exit,
                         log_dur_pmf, D_max):
    """Batched HSMM forward-backward (fixed params, no EM).

    Dynamic batch sizing based on T and D_max to keep memory bounded.

    Args:
        obs_batch: (B, T) observation timecourses.

    Returns:
        gamma_1: (B, T) posterior P(z_t = 1 | y) for each timecourse.
    """
    B, T = obs_batch.shape
    if T < 2:
        return np.zeros((B, T))

    # Emission log-probs: (B, T)
    log_b0 = (-0.5 * np.log(2 * np.pi) - np.log(null_sigma)
              - 0.5 * ((obs_batch - null_mu) / null_sigma) ** 2)
    log_b1 = (-0.5 * np.log(2 * np.pi) - np.log(coupled_sigma)
              - 0.5 * ((obs_batch - coupled_mu) / coupled_sigma) ** 2)

    # Forward
    log_f0 = np.full((B, T), -np.inf)
    log_f1 = np.full((B, T, D_max), -np.inf)

    log_f0[:, 0] = np.log(0.9) + log_b0[:, 0]
    valid_d = log_dur_pmf > -1e10
    if valid_d.any():
        log_f1[:, 0, valid_d] = (np.log(0.1)
                                 + log_dur_pmf[valid_d][np.newaxis, :]
                                 + log_b1[:, 0:1])

    for t in range(1, T):
        log_f0[:, t] = log_b0[:, t] + np.logaddexp(
            log_f0[:, t - 1] + log_1mp_exit,
            log_f1[:, t - 1, 0])

        cont = np.full((B, D_max), -np.inf)
        if D_max > 1:
            cont[:, :D_max - 1] = log_f1[:, t - 1, 1:] + log_b1[:, t:t + 1]

        entry = (log_f0[:, t - 1:t] + log_p_exit
                 + log_dur_pmf[np.newaxis, :] + log_b1[:, t:t + 1])

        log_f1[:, t] = np.logaddexp(cont, entry)

    # Backward
    log_a0 = np.zeros((B, T))
    log_a1 = np.zeros((B, T, D_max))

    for t in range(T - 2, -1, -1):
        log_stay = log_1mp_exit + log_b0[:, t + 1] + log_a0[:, t + 1]
        log_exit_arr = (log_p_exit + log_dur_pmf[np.newaxis, :]
                        + log_b1[:, t + 1:t + 2] + log_a1[:, t + 1])
        log_a0[:, t] = np.logaddexp(log_stay,
                                    _logsumexp(log_exit_arr, axis=1))

        log_a1[:, t, 0] = log_b0[:, t + 1] + log_a0[:, t + 1]
        if D_max > 1:
            log_a1[:, t, 1:] = (log_b1[:, t + 1:t + 2]
                                + log_a1[:, t + 1, :D_max - 1])

    # Posterior
    log_g0 = log_f0 + log_a0
    log_g1 = _logsumexp(log_f1 + log_a1, axis=2)
    log_norm = np.logaddexp(log_g0, log_g1)
    gamma_1 = np.exp(np.clip(log_g1 - log_norm, -50, 0))

    return np.clip(gamma_1, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 3: Surrogate null calibration
# ---------------------------------------------------------------------------

def _calibrate_threshold(z_agg_null, params, eval_rate, target_fa=0.10):
    """Run HSMM on null z-score timecourses, return coupling-fraction threshold.

    Processes surrogates in small batches to manage memory at native rate
    (T can be ~27K at 15Hz for 30-min sessions, D_max ~450).

    Args:
        z_agg_null: (K, T) null z-score timecourses at native eval_rate.
        params: dict from hsmm_em (learned parameters).
        eval_rate: Hz (for logging only; T already at native rate).
        target_fa: target false alarm rate.

    Returns:
        threshold: coupling fraction above which detection is significant.
        null_fracs: (K,) coupling fractions under null.
    """
    K, T = z_agg_null.shape
    D_max = params['D_max']

    lpe = np.log(max(params['p_exit'], 1e-10))
    l1mpe = np.log(max(1 - params['p_exit'], 1e-10))

    # Dynamic batch size: keep memory <= 500 MB
    # Main arrays: log_f1 (B,T,D) + log_a1 (B,T,D) + log_f0 (B,T) + log_a0 (B,T)
    # ~= 2 * B * T * D * 8 bytes + 2 * B * T * 8 bytes
    mem_per_item = T * D_max * 8 * 2 + T * 8 * 2
    BATCH = max(1, min(5, int(500e6 / max(mem_per_item, 1))))

    null_fracs = np.zeros(K)

    for b0 in range(0, K, BATCH):
        b1 = min(b0 + BATCH, K)
        batch = z_agg_null[b0:b1]

        g1 = _hsmm_fwd_bwd_batch(
            batch,
            params['null_mu'], params['null_sigma'],
            params['coupled_mu'], params['coupled_sigma'],
            lpe, l1mpe, params['log_dur_pmf'], D_max)

        null_fracs[b0:b1] = (g1 > 0.5).mean(axis=1)

    threshold = float(np.percentile(null_fracs, 100 * (1 - target_fa)))
    return threshold, null_fracs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def temporal_localization_pipeline(dr2_real_pc, dr2_surr_pc, eval_rate, cfg):
    """Z-score + HSMM + calibration for matched-diagonal pathways.

    Operates at native eval_rate (no downsampling). Short-tau EWLS (tau=1.0s)
    provides sharp dR2 timecourses; HSMM segments at full temporal resolution.

    Three-stage pipeline:
      1) Per-channel z-score + Stouffer aggregation
      2) HSMM EM with fixed null emission and learned coupled emission
      3) Surrogate calibration to control false alarm rate

    Args:
        dr2_real_pc: (C, T) real per-channel dR2.
        dr2_surr_pc: (K, C, T) surrogate per-channel dR2.
        eval_rate: evaluation rate in Hz.
        cfg: temporal_localization config dict.

    Returns:
        posterior: (T,) coupling posterior in [0, 1].
        dr2_smooth: (T,) smoothed aggregate dR2 (for display).
        diagnostics: dict with model parameters and calibration results.
    """
    C, T = dr2_real_pc.shape
    K = dr2_surr_pc.shape[0]

    d_min_s = cfg.get('hsmm_min_event_s', 0.5)
    d_max_s = cfg.get('hsmm_max_event_s', 30.0)
    mu_d_s = cfg.get('hsmm_mean_event_s', 8.0)
    sigma_d_s = cfg.get('hsmm_std_event_s', 5.0)
    max_iter = cfg.get('hsmm_max_iter', 20)
    do_calibrate = cfg.get('calibrate_on_surrogates', True)
    target_fa = cfg.get('target_false_alarm', 0.10)

    # Smoothed dR2 for display only
    dr2_agg = np.nanmean(dr2_real_pc, axis=0)
    smooth_w = max(1, int(5.0 * eval_rate))
    dr2_smooth = uniform_filter1d(
        np.nan_to_num(dr2_agg, nan=0.0), smooth_w, mode='nearest')

    # Guard: insufficient data
    if T < 10 or K < 5:
        return (np.zeros(T), dr2_smooth,
                {'method': 'skipped', 'reason': 'insufficient data'})

    # --- Stage 1: Per-channel z-score + Stouffer aggregation ---
    z_agg, z_agg_null, null_mu, null_sigma = zscore_stouffer(
        dr2_real_pc, dr2_surr_pc, eval_rate)

    # --- Duration params at native eval_rate ---
    d_min = max(1, int(round(d_min_s * eval_rate)))
    D_max_samp = max(d_min + 1, int(round(d_max_s * eval_rate)))
    if T < 2 * D_max_samp:
        D_max_samp = max(d_min + 1, T // 3)

    mu_d_samp = mu_d_s * eval_rate
    sigma_d_samp = sigma_d_s * eval_rate

    # --- Stage 2: HSMM EM (at native rate, no smoothing) ---
    posterior, params = hsmm_em(
        z_agg, null_mu, null_sigma, eval_rate,
        d_min, D_max_samp, mu_d_samp, sigma_d_samp, max_iter)

    # --- Stage 3: Calibration ---
    detected_cal = True
    null_fracs = None
    if do_calibrate and K >= 20:
        threshold, null_fracs = _calibrate_threshold(
            z_agg_null, params, eval_rate, target_fa)

        real_frac = float(np.mean(posterior > 0.5))
        detected_cal = real_frac > threshold

    # Z-score distribution diagnostics
    z_p95 = float(np.percentile(z_agg, 95))
    z_p99 = float(np.percentile(z_agg, 99))
    z_std = float(np.std(z_agg))

    # Per-channel dR2 diagnostics
    dr2_pc_mean = float(np.nanmean(dr2_real_pc))
    surr_pc_mean = float(np.nanmean(dr2_surr_pc))
    dr2_pc_std = float(np.nanstd(np.nanmean(dr2_real_pc, axis=0)))
    surr_pc_std = float(np.nanstd(np.nanmean(dr2_surr_pc, axis=(0, 1))))

    diag = {
        'method': 'zscore_hsmm_v2',
        'eval_rate': eval_rate,
        'T': T,
        'C': C,
        'K': K,
        'D_max': D_max_samp,
        'null_mu': null_mu,
        'null_sigma': null_sigma,
        'coupled_mu': params['coupled_mu'],
        'coupled_sigma': params['coupled_sigma'],
        'p_exit': params['p_exit'],
        'hsmm_iterations': params['iterations'],
        'log_likelihood': params['log_likelihood'],
        'z_agg_mean': float(np.mean(z_agg)),
        'z_agg_max': float(np.max(z_agg)),
        'z_agg_std': z_std,
        'z_agg_p95': z_p95,
        'z_agg_p99': z_p99,
        'dr2_pc_mean': dr2_pc_mean,
        'surr_pc_mean': surr_pc_mean,
        'dr2_pc_temporal_std': dr2_pc_std,
        'detected_calibrated': detected_cal,
    }
    if null_fracs is not None:
        diag['null_frac_95'] = float(np.percentile(null_fracs, 95))
        diag['real_frac'] = float(np.mean(posterior > 0.5))

    return posterior, dr2_smooth, diag


def hsmm_aggregate_localization(dr2_arr, null_stats, eval_rate, cfg):
    """Cross-modal temporal localization: z-score -> HSMM at native rate.

    For cross-modal pathways without per-channel surrogate arrays.
    Z-scores the aggregate dR2 against null stats, then runs the same
    HSMM segmentation with state 0 emission N(0, 1).

    Args:
        dr2_arr: (T,) aggregate dR2 timecourse.
        null_stats: dict with 'mu_0' and 'sigma_0'.
        eval_rate: evaluation rate in Hz.
        cfg: temporal_localization config dict.

    Returns:
        posterior: (T,) coupling posterior in [0, 1].
    """
    T = len(dr2_arr)
    mu_0 = null_stats.get('mu_0', 0.0)
    sigma_0 = max(null_stats.get('sigma_0', 1e-8), 1e-8)

    dr2_clean = np.nan_to_num(dr2_arr, nan=mu_0)
    z = (dr2_clean - mu_0) / sigma_0

    d_min_s = cfg.get('hsmm_min_event_s', 0.5)
    d_max_s = cfg.get('hsmm_max_event_s', 30.0)
    mu_d_s = cfg.get('hsmm_mean_event_s', 8.0)
    sigma_d_s = cfg.get('hsmm_std_event_s', 5.0)
    max_iter = cfg.get('hsmm_max_iter', 20)

    if T < 20:
        return np.clip(1.0 - np.exp(-np.maximum(z, 0)), 0.0, 1.0)

    d_min = max(1, int(round(d_min_s * eval_rate)))
    D_max_samp = max(d_min + 1, int(round(d_max_s * eval_rate)))
    if T < 2 * D_max_samp:
        D_max_samp = max(d_min + 1, T // 3)

    # State 0 = N(0, 1) for z-scored observations
    posterior, _ = hsmm_em(
        z, 0.0, 1.0, eval_rate,
        d_min, D_max_samp,
        mu_d_s * eval_rate, sigma_d_s * eval_rate, max_iter)

    return np.clip(posterior, 0.0, 1.0)


# ---------------------------------------------------------------------------
# V3.4: LLR + MRC + Bilateral CUSUM temporal localization
# ---------------------------------------------------------------------------

def _bilateral_cusum(x, mu0, h=0.0):
    """Forward + backward CUSUM on a 1D timecourse.

    Forward CUSUM detects upward shifts (coupling onset).
    Backward CUSUM detects downward shifts (coupling offset).
    Their intersection gives coupling episodes.

    Args:
        x: (T,) timecourse (pooled LLR).
        mu0: null reference level.
        h: allowance (half expected shift). 0 = accumulate everything.

    Returns:
        cusum_fwd: (T,) forward CUSUM statistic.
        cusum_bwd: (T,) backward CUSUM statistic (reversed).
    """
    T = len(x)

    # Forward: detect upward shifts
    cusum_fwd = np.zeros(T)
    for t in range(1, T):
        cusum_fwd[t] = max(0.0, cusum_fwd[t - 1] + x[t] - mu0 - h)

    # Backward: detect when coupling ends (run reversed)
    cusum_bwd = np.zeros(T)
    for t in range(T - 2, -1, -1):
        cusum_bwd[t] = max(0.0, cusum_bwd[t + 1] + x[t] - mu0 - h)

    return cusum_fwd, cusum_bwd


def llr_cusum_localization(dr2_real_pc, dr2_surr_pc, eval_rate,
                            tau_seconds, target_fa=0.10, min_event_s=2.0,
                            smooth_scales_s=(5.0, 15.0, 30.0)):
    """LLR + MRC + multi-scale smoothed threshold temporal localization.

    Converts per-channel dR2 to log-likelihood ratio, pools channels
    with SNR-optimal Maximum Ratio Combining weights, smooths at multiple
    timescales, takes max-envelope, and thresholds against surrogates.

    Args:
        dr2_real_pc: (C, T) real per-channel dR2 from EWLS.
        dr2_surr_pc: (K, C, T) surrogate per-channel dR2.
        eval_rate: EWLS eval rate in Hz.
        tau_seconds: EWLS tau (for T_eff computation).
        target_fa: false alarm rate for threshold.
        min_event_s: minimum coupling event duration.
        smooth_scales_s: tuple of smoothing scales in seconds.

    Returns:
        mask: (T,) boolean coupling mask.
        posterior: (T,) soft posterior in [0, 1].
        diagnostics: dict with pipeline metadata.
    """
    from scipy.special import expit

    C, T = dr2_real_pc.shape
    K = dr2_surr_pc.shape[0]

    # Guard: insufficient data
    if T < 20 or K < 5 or C < 1:
        return (np.zeros(T, dtype=bool), np.zeros(T),
                {'method': 'llr_mrc', 'reason': 'insufficient data'})

    # --- Step 1: dR2 → LLR ---
    T_eff = 2.0 * tau_seconds * eval_rate
    dr2_clip = np.clip(dr2_real_pc, -0.99, 0.99)
    llr_real_pc = (T_eff / 2.0) * np.log(1.0 / (1.0 - dr2_clip))

    dr2_surr_clip = np.clip(dr2_surr_pc, -0.99, 0.99)
    llr_surr_pc = (T_eff / 2.0) * np.log(1.0 / (1.0 - dr2_surr_clip))

    # --- Step 2: Per-channel z-score against surrogates FIRST ---
    # This removes session-level mean before pooling, avoiding circularity
    surr_mean_pc = llr_surr_pc.mean(axis=0)                # (C, T)
    surr_std_pc = np.maximum(llr_surr_pc.std(axis=0), 1e-10)
    z_real_pc = (llr_real_pc - surr_mean_pc) / surr_std_pc  # (C, T)
    z_surr_pc = (llr_surr_pc - surr_mean_pc[None]) / surr_std_pc[None]  # (K, C, T)

    # --- Step 3: MRC-weighted channel pooling on z-scores ---
    # Weight by session-level mean z (positive = coupled channel)
    mu_z_c = np.nanmean(z_real_pc, axis=1)                  # (C,)
    var_z_c = np.maximum(np.nanvar(z_real_pc, axis=1), 1e-10)
    w = np.maximum(mu_z_c, 0.0) / var_z_c
    w_sum = w.sum()
    if w_sum < 1e-10:
        return (np.zeros(T, dtype=bool), np.zeros(T),
                {'method': 'llr_mrc', 'coupling_fraction': 0.0,
                 'n_positive_channels': 0, 'C': C})
    w = w / w_sum

    z_real = w @ z_real_pc                                   # (T,)
    z_surr = np.einsum('c,kct->kt', w, z_surr_pc)           # (K, T)

    # --- Step 4: Multi-scale smoothing + max-envelope ---
    z_scales = []
    z_surr_scales = []
    for scale_s in smooth_scales_s:
        w_samp = max(1, int(scale_s * eval_rate))
        if w_samp > 1:
            z_s = uniform_filter1d(z_real, w_samp, mode='nearest')
            z_surr_s = np.array([
                uniform_filter1d(z_surr[k], w_samp, mode='nearest')
                for k in range(K)])
        else:
            z_s = z_real.copy()
            z_surr_s = z_surr.copy()
        z_scales.append(z_s)
        z_surr_scales.append(z_surr_s)

    # Max-envelope across scales
    z_multi = np.maximum.reduce(z_scales)                  # (T,)
    z_surr_multi = np.maximum.reduce(z_surr_scales)        # (K, T)

    # --- Step 5: Surrogate-calibrated threshold ---
    surr_maxes = z_surr_multi.max(axis=1)                  # (K,)
    threshold = max(float(np.percentile(surr_maxes,
                                         100 * (1 - target_fa))), 1.0)

    mask = z_multi > threshold

    # Min-event filter
    min_samples = max(1, int(min_event_s * eval_rate))
    if min_samples > 1:
        diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_samples:
                mask[s:e] = False

    # Posterior = mask (binary) with soft edges
    posterior = mask.astype(np.float64)

    coupling_frac = float(np.mean(mask))
    n_pos_ch = int(np.sum(mu_z_c > 0))

    diagnostics = {
        'method': 'llr_mrc',
        'C': C, 'T': T, 'K': K,
        'T_eff': T_eff,
        'eval_rate': eval_rate,
        'tau_seconds': tau_seconds,
        'n_positive_channels': n_pos_ch,
        'mrc_weights': w,
        'threshold': threshold,
        'z_multi_max': float(z_multi.max()),
        'z_multi_mean': float(z_multi.mean()),
        'z_multi_p95': float(np.percentile(z_multi, 95)),
        'smooth_scales_s': list(smooth_scales_s),
        'coupling_fraction': coupling_frac,
    }

    return mask, posterior, diagnostics
