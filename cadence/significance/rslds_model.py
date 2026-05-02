"""CADENCE IOHMM/rSLDS: Input-Output Hidden Markov Model for coupling state dynamics.

Fits a K-state HMM with:
  - Per-state AR(1) diagonal Gaussian emissions on 7D z_fast observations
  - Input-driven transitions: P(z_t=k | z_{t-1}=j, u_t) = softmax(W_jk + S_jk @ u_t)
    where u_t are z_slow PCs (shared-state summary)
  - Masked observations for missing modalities (y11, y24 lack ECG/Resp)
  - Multiple random restarts with k-means initialization

Phase 3 extensions (incremental):
  - AR(1) emissions: P(y_t | z_t=k, y_{t-1}) accounts for rho~0.93 autocorrelation
  - (TODO) Continuous latent x_t with Kalman filter-smoother
  - Factor-analyzed emission noise: Sigma[k] = F[k]F[k]' + diag(R[k])
  - Recurrent transitions (rSLDS): x_{t-1} influences z_t via R_recur
  - (TODO) Hierarchical pooling across sessions

EM algorithm:
  E-step: forward-backward (CPU numpy, log-space with logaddexp)
  M-step: closed-form for emissions + AR, L-BFGS-B for transition logits

Usage:
    config = IOHMMConfig(K=3, D_obs=7, D_input=2, ar_order=1)
    model = IOHMM(config)
    params, history = model.fit(Y, U, obs_mask)
    posteriors = model.state_posteriors(Y, U, obs_mask, params)
"""

import numpy as np
import numba
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.optimize import minimize


# Local logsumexp — scipy 1.17's logsumexp dispatches through the array-API
# machinery (xp_promote, is_torch_array, etc.) which adds ~30 us per call
# regardless of input size. With 200k+ calls per E-step on (K,) and (K,K)
# arrays in _forward_backward, that dispatch overhead dominates fit
# wall-time. Bench: 4-8x faster than scipy on typical shapes; numerically
# equivalent to <1e-15. Edge cases (-inf rows) handled identically.
def logsumexp(x, axis=None, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    if not keepdims:
        if axis is None:
            out = out.reshape(())
        else:
            out = np.squeeze(out, axis=axis)
    return out


# Numba-jitted forward-backward kernel. Phase 0.2c: Python-loop overhead in
# IOHMM._forward_backward called fast inline logsumexp 2.1M times per fit
# (still ~28s of post-Phase-0.2b 62s wall-time on MVP 10-iter profile). This
# kernel inlines all logsumexp reductions as explicit per-K loops, eliminating
# both Python iteration overhead AND function-call overhead. Returns
# (gamma: (T,K), xi: (T-1,K,K), log_lik: float).
#
# Convention (matches numpy version):
#   log_trans[t][j,k] = log P(z_t=k | z_{t-1}=j, U_t)  for t in 1..T-1
#   log_trans[0] is computed but unused
#   Forward update:  alpha[t,k] = lse_j(alpha[t-1,j] + log_trans[t,j,k]) + emit[t,k]
#   Backward update: beta[t,j]  = lse_k(log_trans[t+1,j,k] + emit[t+1,k] + beta[t+1,k])
#   xi[t-1, j, k]   = alpha[t-1,j] + log_trans[t,j,k] + emit[t,k] + beta[t,k]  (normalized)
@numba.njit(cache=True, fastmath=False)
def _forward_backward_numba(log_emit, log_trans, log_pi):
    T = log_emit.shape[0]
    K = log_emit.shape[1]

    log_alpha = np.empty((T, K))
    log_scale = np.empty(T)

    # Initial alpha = log_pi + log_emit[0], rescaled by its own logsumexp
    for k in range(K):
        log_alpha[0, k] = log_pi[k] + log_emit[0, k]
    mx = log_alpha[0, 0]
    for k in range(1, K):
        if log_alpha[0, k] > mx:
            mx = log_alpha[0, k]
    if not np.isfinite(mx):
        mx = 0.0
    s = 0.0
    for k in range(K):
        s += np.exp(log_alpha[0, k] - mx)
    log_scale[0] = mx + np.log(s)
    for k in range(K):
        log_alpha[0, k] -= log_scale[0]

    # Forward pass
    for t in range(1, T):
        for k in range(K):
            mx = log_alpha[t - 1, 0] + log_trans[t, 0, k]
            for j in range(1, K):
                v = log_alpha[t - 1, j] + log_trans[t, j, k]
                if v > mx:
                    mx = v
            if not np.isfinite(mx):
                mx = 0.0
            s = 0.0
            for j in range(K):
                s += np.exp(log_alpha[t - 1, j] + log_trans[t, j, k] - mx)
            log_alpha[t, k] = mx + np.log(s) + log_emit[t, k]

        mx = log_alpha[t, 0]
        for k in range(1, K):
            if log_alpha[t, k] > mx:
                mx = log_alpha[t, k]
        if not np.isfinite(mx):
            mx = 0.0
        s = 0.0
        for k in range(K):
            s += np.exp(log_alpha[t, k] - mx)
        log_scale[t] = mx + np.log(s)
        for k in range(K):
            log_alpha[t, k] -= log_scale[t]

    log_lik = 0.0
    for t in range(T):
        log_lik += log_scale[t]

    # Backward pass
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for j in range(K):
            mx = log_trans[t + 1, j, 0] + log_emit[t + 1, 0] + log_beta[t + 1, 0]
            for k in range(1, K):
                v = log_trans[t + 1, j, k] + log_emit[t + 1, k] + log_beta[t + 1, k]
                if v > mx:
                    mx = v
            if not np.isfinite(mx):
                mx = 0.0
            s = 0.0
            for k in range(K):
                s += np.exp(log_trans[t + 1, j, k] + log_emit[t + 1, k]
                            + log_beta[t + 1, k] - mx)
            log_beta[t, j] = mx + np.log(s)

    # Posterior gamma per timestep
    gamma = np.empty((T, K))
    for t in range(T):
        mx = log_alpha[t, 0] + log_beta[t, 0]
        for k in range(1, K):
            v = log_alpha[t, k] + log_beta[t, k]
            if v > mx:
                mx = v
        if not np.isfinite(mx):
            mx = 0.0
        s = 0.0
        for k in range(K):
            s += np.exp(log_alpha[t, k] + log_beta[t, k] - mx)
        norm = mx + np.log(s)
        for k in range(K):
            gamma[t, k] = np.exp(log_alpha[t, k] + log_beta[t, k] - norm)

    # Pairwise xi per timestep
    xi = np.empty((T - 1, K, K))
    for t in range(T - 1):
        mx = log_alpha[t, 0] + log_trans[t + 1, 0, 0] + log_emit[t + 1, 0] + log_beta[t + 1, 0]
        for j in range(K):
            for k in range(K):
                v = (log_alpha[t, j] + log_trans[t + 1, j, k]
                     + log_emit[t + 1, k] + log_beta[t + 1, k])
                if v > mx:
                    mx = v
        if not np.isfinite(mx):
            mx = 0.0
        s = 0.0
        for j in range(K):
            for k in range(K):
                s += np.exp(log_alpha[t, j] + log_trans[t + 1, j, k]
                            + log_emit[t + 1, k] + log_beta[t + 1, k] - mx)
        norm = mx + np.log(s)
        for j in range(K):
            for k in range(K):
                xi[t, j, k] = np.exp(log_alpha[t, j] + log_trans[t + 1, j, k]
                                      + log_emit[t + 1, k] + log_beta[t + 1, k] - norm)

    return gamma, xi, log_lik


# ─────────────────────────────────────────────────────────────────────
#  Configuration and Parameters
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IOHMMConfig:
    K: int = 3               # number of discrete states
    D_obs: int = 7           # observation dimensionality (z_fast channels)
    D_input: int = 2         # transition covariate dims (z_slow PCs)
    D_latent: int = 0        # continuous latent dim (0=IOHMM, >0=SLDS)
    ar_order: int = 0        # AR order (0=independent, 1=AR(1) — WARNING: causes state collapse; use D_latent instead)
    max_em_iter: int = 200   # max EM iterations
    em_tol: float = 1e-4     # relative LL improvement tolerance
    n_restarts: int = 5      # random restarts (keep best by LL)
    anneal_iters: int = 20   # number of iterations with annealed transitions
    anneal_temp: float = 2.0 # initial temperature for transition softmax
    n_inner_estep: int = 3   # inner SMF iterations per E-step (SLDS only)
    kalman_init_P0: float = 1.0  # initial latent covariance scale (SLDS only)
    n_factors: int = 0           # emission noise factor rank (0=diagonal, >0=factor-analyzed)
    recurrent: bool = False      # recurrent transitions x_{t-1} -> z_t (rSLDS)
    sticky_strength: float = 0.0 # κ pseudo-count for self-transitions (0=disabled, 1-5 typical)
    viterbi_min_dwell: int = 0   # minimum samples per state in Viterbi (0=disabled)
    null_state: bool = False     # if True, state 0 is null: mu[0]/d[0]=0 fixed, sigma2[0] bounded
    null_sigma2_cap: float = 0.5 # upper bound on null state variance (prevents catch-all)
    c_shrinkage: float = 0.0    # emission loading shrinkage toward shared C (0=none, 0.3 typical for V10)


@dataclass
class IOHMMParams:
    mu: np.ndarray           # (K, D_obs) emission means
    log_sigma2: np.ndarray   # (K, D_obs) log emission variances (diagonal)
    W_trans: np.ndarray      # (K, K) base transition logits
    S_trans: np.ndarray      # (K, K, D_input) input-modulated transition logits
    log_pi: np.ndarray       # (K,) initial state log-probabilities
    phi: np.ndarray = None   # (K, D_obs) AR(1) coefficients (None = no AR)

    @property
    def sigma2(self):
        return np.exp(self.log_sigma2)

    def copy(self):
        return IOHMMParams(
            mu=self.mu.copy(),
            log_sigma2=self.log_sigma2.copy(),
            W_trans=self.W_trans.copy(),
            S_trans=self.S_trans.copy(),
            log_pi=self.log_pi.copy(),
            phi=self.phi.copy() if self.phi is not None else None,
        )

    def to_dict(self):
        d = {
            'mu': self.mu.tolist(),
            'sigma2': self.sigma2.tolist(),
            'W_trans': self.W_trans.tolist(),
            'S_trans': self.S_trans.tolist(),
            'log_pi': self.log_pi.tolist(),
        }
        if self.phi is not None:
            d['phi'] = self.phi.tolist()
        return d


# ─────────────────────────────────────────────────────────────────────
#  Core IOHMM
# ─────────────────────────────────────────────────────────────────────

class IOHMM:
    """Input-Output Hidden Markov Model with EM fitting."""

    def __init__(self, config: IOHMMConfig):
        self.cfg = config

    # ── Initialization ───────────────────────────────────────────────

    def initialize(self, Y: np.ndarray, U: np.ndarray,
                   obs_mask: Optional[np.ndarray] = None,
                   seed: int = 42) -> IOHMMParams:
        """Data-driven initialization using k-means on observations.

        Args:
            Y: (T, D_obs) observations (z_fast)
            U: (T, D_input) transition covariates (z_slow PCs)
            obs_mask: (T, D_obs) boolean, True = observed
            seed: random seed
        """
        K, D = self.cfg.K, self.cfg.D_obs
        rng = np.random.default_rng(seed)

        # Fill masked values with 0 for k-means
        Y_filled = Y.copy()
        if obs_mask is not None:
            Y_filled[~obs_mask] = 0.0

        # k-means initialization (simple: random centroids + Lloyd's)
        idx = rng.choice(len(Y_filled), K, replace=False)
        centroids = Y_filled[idx].copy()
        for _ in range(20):
            dists = np.sum((Y_filled[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask_k = labels == k
                if mask_k.sum() > 0:
                    centroids[k] = Y_filled[mask_k].mean(axis=0)

        # Null state: swap closest-to-zero centroid to position 0
        if self.cfg.null_state:
            norms = np.linalg.norm(centroids, axis=1)
            null_idx = np.argmin(norms)
            centroids[[0, null_idx]] = centroids[[null_idx, 0]]
            centroids[0] = 0.0  # enforce exactly zero
            # Sort remaining centroids (1..K-1) by first dimension
            remaining_order = np.argsort(centroids[1:, 0]) + 1
            centroids[1:] = centroids[remaining_order]
            order = np.concatenate([[null_idx], np.delete(np.argsort(centroids[:, 0]), 0)])
            # Rebuild order mapping for label assignment
            order = np.arange(K)  # already reordered above
        else:
            # Sort centroids by first dimension (consistent ordering)
            order = np.argsort(centroids[:, 0])
            centroids = centroids[order]

        # Emission variances: per-cluster variance or data variance fallback
        sigma2 = np.ones((K, D), dtype=np.float64)
        labels_sorted = np.zeros_like(labels)
        for i, k in enumerate(order):
            labels_sorted[labels == k] = i
        for k in range(K):
            mask_k = labels_sorted == k
            if mask_k.sum() > 1:
                sigma2[k] = np.var(Y_filled[mask_k], axis=0).clip(min=1e-4)
            else:
                sigma2[k] = np.var(Y_filled, axis=0).clip(min=1e-4)

        # Transition: self-transition bias
        W = np.zeros((K, K), dtype=np.float64)
        for k in range(K):
            W[k, k] = 1.0  # self-transition logit bias

        if self.cfg.null_state:
            # Asymmetric transitions: leaving null (S0) is costly,
            # returning to null is easy. This makes coupling states
            # "expensive" — they need genuine signal to justify activation.
            W[0, 0] = 3.0    # S0→S0: strong self-loop (stay in null)
            for k in range(1, K):
                W[0, k] = -1.0   # S0→Sk: costly to leave null
                W[k, 0] = 1.5    # Sk→S0: easy to return to null
                W[k, k] = 1.0    # Sk→Sk: moderate self-loop

        # Input modulation: small random
        S = rng.normal(0, 0.01, size=(K, K, self.cfg.D_input))

        # Initial state
        if self.cfg.null_state:
            # Strong prior: start in null state
            log_pi = np.full(K, -5.0, dtype=np.float64)  # very low for coupling states
            log_pi[0] = 0.0  # high for null
            log_pi -= np.log(np.exp(log_pi).sum())  # normalize
        else:
            log_pi = np.full(K, -np.log(K), dtype=np.float64)

        # AR(1) coefficients: fixed from empirical lag-1 autocorrelation (shared across states)
        # NOT learned per-state — per-state phi causes state collapse (AR absorbs mean differences)
        phi = None
        if self.cfg.ar_order >= 1:
            phi_shared = np.zeros(D, dtype=np.float64)
            for d in range(D):
                yd = Y_filled[:, d]
                if len(yd) > 2 and np.std(yd) > 1e-8:
                    r1 = np.corrcoef(yd[:-1], yd[1:])[0, 1]
                    phi_shared[d] = np.clip(r1, 0.0, 0.99)
            # Broadcast to (K, D) — same phi for all states
            phi = np.tile(phi_shared, (K, 1))

        return IOHMMParams(
            mu=centroids.astype(np.float64),
            log_sigma2=np.log(sigma2),
            W_trans=W,
            S_trans=S,
            log_pi=log_pi,
            phi=phi,
        )

    # ── Log-emissions ────────────────────────────────────────────────

    def _log_emissions(self, Y: np.ndarray, obs_mask: Optional[np.ndarray],
                       params: IOHMMParams) -> np.ndarray:
        """Compute (T, K) log emission probabilities.

        If AR(1) enabled (params.phi is not None):
          t=0: marginal p(y_0 | z_0=k) = N(mu_k, sigma2_k)
          t>=1: conditional p(y_t | z_t=k, y_{t-1}) = N(mu_k + phi_k*(y_{t-1}-mu_k), sigma2_k*(1-phi_k^2))

        AR(1) conditional innovation: epsilon_t = y_t - mu_k - phi_k*(y_{t-1} - mu_k)
        This is white noise by construction, preventing over-counting autocorrelated samples.

        Masked dimensions contribute 0 to log-likelihood.
        """
        T, D = Y.shape
        K = self.cfg.K
        sigma2 = params.sigma2  # (K, D)
        use_ar = params.phi is not None and self.cfg.ar_order >= 1

        # Vectorized over K (was per-k Python loop). At T~4000, K=4, D=7 the
        # inner numpy ops are tiny but interpreter overhead dominated the K
        # loop body. Memory: (T, K, D) = ~0.9 MB at MVP shape — trivial.
        # See perf-audit S6-3.
        mu = params.mu                                           # (K, D)
        if use_ar:
            phi = params.phi                                     # (K, D)
            innov_var = sigma2 * np.maximum(1.0 - phi ** 2, 1e-6)  # (K, D)

            # t=0: marginal emission. resid_0: (K, D)
            resid_0 = Y[0][None, :] - mu
            ll_0 = (-0.5 * np.log(2 * np.pi * sigma2)
                    - 0.5 * resid_0 ** 2 / sigma2)              # (K, D)

            # t>=1: AR(1) conditional. innovation: (T-1, K, D)
            Y_prev = Y[:-1][:, None, :]                          # (T-1, 1, D)
            Y_curr = Y[1:][:, None, :]                           # (T-1, 1, D)
            innovation = Y_curr - mu[None, :, :] - phi[None, :, :] * (Y_prev - mu[None, :, :])
            ll_ar = (-0.5 * np.log(2 * np.pi * innov_var)[None, :, :]
                     - 0.5 * innovation ** 2 / innov_var[None, :, :])  # (T-1, K, D)

            if obs_mask is not None:
                ll_0 = ll_0 * obs_mask[0][None, :]
                ll_ar = ll_ar * obs_mask[1:, None, :]

            log_emit = np.empty((T, K), dtype=np.float64)
            log_emit[0] = ll_0.sum(axis=1)                       # (K,)
            log_emit[1:] = ll_ar.sum(axis=2)                     # (T-1, K)
        else:
            # Standard (no AR): marginal emission at all timepoints.
            # residual: (T, K, D)
            residual = Y[:, None, :] - mu[None, :, :]
            ll_per_dim = (-0.5 * np.log(2 * np.pi * sigma2)[None, :, :]
                          - 0.5 * residual ** 2 / sigma2[None, :, :])
            if obs_mask is not None:
                ll_per_dim = ll_per_dim * obs_mask[:, None, :]
            log_emit = ll_per_dim.sum(axis=2)                    # (T, K)

        return log_emit

    # ── Log-transitions ──────────────────────────────────────────────

    def _log_transitions(self, U: np.ndarray, params: IOHMMParams,
                         temperature: float = 1.0) -> np.ndarray:
        """Compute (T, K, K) log transition matrices.

        log A_t[j, k] = softmax_k(W[j, k] + S[j, k, :] @ u_t) / temperature
        """
        T = U.shape[0]
        K = self.cfg.K

        # Compute logits: (T, K, K)
        # W[j,k] + sum_d S[j,k,d] * U[t,d]
        logits = params.W_trans[None, :, :] + np.einsum('td,jkd->tjk', U, params.S_trans)
        logits /= temperature

        # Softmax per row (j): normalize over k
        log_trans = logits - logsumexp(logits, axis=2, keepdims=True)

        return log_trans  # (T, K, K)

    # ── Forward-Backward ─────────────────────────────────────────────

    def _forward_backward(self, log_emit: np.ndarray, log_trans: np.ndarray,
                          log_pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm in log-space; thin wrapper around
        _forward_backward_numba (defined at module top). Convention preserved
        from the original numpy implementation.

        Args:
            log_emit: (T, K) log emission probabilities
            log_trans: (T, K, K) log transition matrices [j -> k]
            log_pi: (K,) log initial state probabilities

        Returns:
            gamma: (T, K) posterior state probabilities
            xi: (T-1, K, K) pairwise posteriors P(z_{t-1}=j, z_t=k | Y)
            log_lik: total log-likelihood
        """
        return _forward_backward_numba(
            np.ascontiguousarray(log_emit),
            np.ascontiguousarray(log_trans),
            np.ascontiguousarray(log_pi))

    # ── M-step: emissions ────────────────────────────────────────────

    def _m_step_emissions(self, Y: np.ndarray, gamma: np.ndarray,
                          obs_mask: Optional[np.ndarray],
                          params: IOHMMParams) -> IOHMMParams:
        """Update emission means, variances, and AR(1) coefficients (closed form).

        With AR(1): uses innovation residuals (t>=1) for sigma2 estimation.
        mu is estimated from all timepoints. phi from weighted regression of
        y_t - mu on y_{t-1} - mu.
        """
        K, D = self.cfg.K, self.cfg.D_obs
        T = Y.shape[0]
        params = params.copy()
        use_ar = params.phi is not None and self.cfg.ar_order >= 1

        # Variance floor: 1e-4 × data variance per dimension
        if obs_mask is not None:
            var_floor = np.array([max(np.var(Y[obs_mask[:, d], d]) if obs_mask[:, d].sum() > 1 else 1e-4,
                                      1e-4) * 1e-4
                                  for d in range(D)])
        else:
            var_floor = np.var(Y, axis=0).clip(min=1e-4) * 1e-4

        for k in range(K):
            w = gamma[:, k]  # (T,)
            w_sum = w.sum()
            if w_sum < 1e-8:
                continue

            # Step 1: Update mu (weighted mean of all observations)
            if obs_mask is not None:
                for d in range(D):
                    m = obs_mask[:, d]
                    w_d = w * m
                    w_d_sum = w_d.sum()
                    if w_d_sum < 1e-8:
                        continue
                    params.mu[k, d] = (w_d * Y[:, d]).sum() / w_d_sum
            else:
                params.mu[k] = (w[:, None] * Y).sum(axis=0) / w_sum

            if use_ar and T > 1:
                # phi is FIXED (shared across states, estimated from data at init)
                # Only update sigma2 from innovation residuals

                w_ar = w[1:]  # weights for t=1,...,T-1
                phi_k = params.phi[k]  # fixed, not updated
                innovation = Y[1:] - params.mu[k] - phi_k * (Y[:-1] - params.mu[k])  # (T-1, D)
                innov_scale = np.maximum(1.0 - phi_k ** 2, 1e-6)

                for d in range(D):
                    if obs_mask is not None:
                        m_ar = obs_mask[1:, d] & obs_mask[:-1, d]
                        w_d = w_ar * m_ar
                    else:
                        w_d = w_ar
                    w_d_sum = w_d.sum()
                    if w_d_sum < 1e-8:
                        continue
                    innov_var = (w_d * innovation[:, d] ** 2).sum() / w_d_sum
                    # sigma2 = innovation_variance / (1 - phi^2)
                    params.log_sigma2[k, d] = np.log(
                        max(innov_var / innov_scale[d], var_floor[d]))
            else:
                # No AR: standard residual-based sigma2
                if obs_mask is not None:
                    for d in range(D):
                        m = obs_mask[:, d]
                        w_d = w * m
                        w_d_sum = w_d.sum()
                        if w_d_sum < 1e-8:
                            continue
                        params.log_sigma2[k, d] = np.log(
                            max((w_d * (Y[:, d] - params.mu[k, d]) ** 2).sum() / w_d_sum,
                                var_floor[d]))
                else:
                    resid = Y - params.mu[k]
                    params.log_sigma2[k] = np.log(
                        np.maximum((w[:, None] * resid ** 2).sum(axis=0) / w_sum, var_floor))

        # Null state constraint: mu[0]=0, sigma2[0] capped
        if self.cfg.null_state:
            params.mu[0, :] = 0.0
            sigma2_0 = np.exp(params.log_sigma2[0])
            sigma2_0 = np.clip(sigma2_0, var_floor, self.cfg.null_sigma2_cap)
            params.log_sigma2[0] = np.log(sigma2_0)

        return params

    # ── M-step: transitions (L-BFGS-B) ──────────────────────────────

    def _m_step_transitions(self, U: np.ndarray, xi: np.ndarray,
                            gamma: np.ndarray,
                            params: IOHMMParams) -> IOHMMParams:
        """Update transition logits W, S via L-BFGS-B on multinomial logistic loss."""
        K, D_in = self.cfg.K, self.cfg.D_input
        params = params.copy()

        # Pack W and S into a flat vector for optimization
        # W: (K, K), S: (K, K, D_in)
        def _pack(W, S):
            return np.concatenate([W.ravel(), S.ravel()])

        def _unpack(x):
            n_W = K * K
            W = x[:n_W].reshape(K, K)
            S = x[n_W:].reshape(K, K, D_in)
            return W, S

        # Objective: negative expected log-transition probability
        # L = -sum_t sum_j sum_k xi[t,j,k] * log_A[t+1, j, k]
        # where log_A[t, j, k] = softmax_k(W[j,k] + S[j,k,:] @ U[t,:])
        T_minus_1 = xi.shape[0]

        # Null-state transition prior: regularize W toward asymmetric structure
        # that makes leaving the null state costly
        W_prior = None
        lambda_W_prior = 0.0
        if self.cfg.null_state:
            W_prior = np.zeros((K, K), dtype=np.float64)
            W_prior[0, 0] = 3.0    # stay in null
            for k in range(1, K):
                W_prior[0, k] = -1.0  # costly to leave null
                W_prior[k, 0] = 1.5   # easy to return
                W_prior[k, k] = 1.0
            lambda_W_prior = 0.1  # regularization strength

        def _objective(x):
            W, S = _unpack(x)
            # Logits: (T-1, K, K) using U[1:] (transition at t uses u_t)
            logits = W[None, :, :] + np.einsum('td,jkd->tjk', U[1:], S)
            log_A = logits - logsumexp(logits, axis=2, keepdims=True)

            # Negative expected log-likelihood
            loss = -np.sum(xi * log_A)
            # L2 regularization on S to prevent overfitting
            loss += 0.01 * np.sum(S ** 2)

            # Null-state transition prior: penalize W departing from asymmetric structure
            if W_prior is not None:
                loss += lambda_W_prior * np.sum((W - W_prior) ** 2)

            # Gradient
            A = np.exp(log_A)  # (T-1, K, K)
            # Expected counts - predicted: xi[t,j,k] - gamma[t,j] * A[t,j,k]
            gamma_from = gamma[:-1]  # (T-1, K)
            err = xi - gamma_from[:, :, None] * A  # (T-1, K, K)

            grad_W = -err.sum(axis=0)  # (K, K)
            if W_prior is not None:
                grad_W += 2 * lambda_W_prior * (W - W_prior)
            grad_S = -np.einsum('tjk,td->jkd', err, U[1:])  # (K, K, D_in)
            grad_S += 0.02 * S  # L2 gradient

            return loss, _pack(grad_W, grad_S)

        x0 = _pack(params.W_trans, params.S_trans)
        result = minimize(_objective, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': 15, 'ftol': 1e-6})
        params.W_trans, params.S_trans = _unpack(result.x)

        # Sticky prior: bias self-transition logits by κ
        if self.cfg.sticky_strength > 0:
            params.W_trans[np.arange(K), np.arange(K)] += self.cfg.sticky_strength

        # Update initial state from gamma[0]
        g0 = gamma[0].clip(min=1e-8)
        params.log_pi = np.log(g0 / g0.sum())

        return params

    # ── Full EM fit ──────────────────────────────────────────────────

    def fit(self, Y: np.ndarray, U: np.ndarray,
            obs_mask: Optional[np.ndarray] = None,
            seed: int = 42, verbose: bool = True) -> Tuple[IOHMMParams, dict]:
        """Fit IOHMM via EM with multiple random restarts.

        Args:
            Y: (T, D_obs) observations (z_fast channels)
            U: (T, D_input) transition covariates (z_slow PCs)
            obs_mask: (T, D_obs) boolean mask (True = observed)
            seed: base random seed
            verbose: print progress

        Returns:
            best_params: IOHMMParams with highest log-likelihood
            history: dict with log-likelihood traces, BIC, etc.
        """
        if len(Y) < 2:
            raise ValueError("Need at least T=2 observations for IOHMM fitting")

        best_ll = -np.inf
        best_params = None
        best_history = None

        for restart in range(self.cfg.n_restarts):
            rseed = seed + restart * 1000
            params = self.initialize(Y, U, obs_mask, seed=rseed)
            ll_trace = []
            prev_ll = -np.inf

            for it in range(self.cfg.max_em_iter):
                # Annealing: higher temperature early on for better mixing
                if it < self.cfg.anneal_iters:
                    temp = self.cfg.anneal_temp - (self.cfg.anneal_temp - 1.0) * it / self.cfg.anneal_iters
                else:
                    temp = 1.0

                # E-step
                log_emit = self._log_emissions(Y, obs_mask, params)
                log_trans = self._log_transitions(U, params, temperature=temp)
                gamma, xi, log_lik = self._forward_backward(log_emit, log_trans, params.log_pi)
                ll_trace.append(log_lik)

                # Convergence check
                if it > 0:
                    rel_change = (log_lik - prev_ll) / max(abs(prev_ll), 1.0)
                    if rel_change < self.cfg.em_tol and temp == 1.0:
                        if verbose:
                            print(f"    Restart {restart}: converged at iter {it}, "
                                  f"LL={log_lik:.1f}")
                        break
                prev_ll = log_lik

                # M-step
                params = self._m_step_emissions(Y, gamma, obs_mask, params)
                params = self._m_step_transitions(U, xi, gamma, params)

            final_ll = ll_trace[-1] if ll_trace else -np.inf
            if verbose:
                state_usage = gamma.mean(axis=0)
                usage_str = ' '.join(f'{u:.1%}' for u in state_usage)
                print(f"    Restart {restart}: LL={final_ll:.1f}, "
                      f"iters={len(ll_trace)}, usage=[{usage_str}]")

            if final_ll > best_ll:
                best_ll = final_ll
                best_params = params.copy()
                best_history = {
                    'll_trace': ll_trace,
                    'restart': restart,
                    'n_iters': len(ll_trace),
                }

        # Final E-step with best params (no annealing)
        log_emit = self._log_emissions(Y, obs_mask, best_params)
        log_trans = self._log_transitions(U, best_params)
        gamma, xi, log_lik = self._forward_backward(log_emit, log_trans, best_params.log_pi)

        # BIC
        n_params = self._count_params()
        bic = -2 * log_lik + n_params * np.log(len(Y))

        best_history['final_ll'] = log_lik
        best_history['bic'] = bic
        best_history['n_params'] = n_params
        best_history['gamma'] = gamma
        best_history['xi'] = xi

        if verbose:
            print(f"  Best: restart {best_history['restart']}, "
                  f"LL={log_lik:.1f}, BIC={bic:.1f}, "
                  f"params={n_params}")

        return best_params, best_history

    # ── Inference ────────────────────────────────────────────────────

    def state_posteriors(self, Y: np.ndarray, U: np.ndarray,
                         obs_mask: Optional[np.ndarray],
                         params: IOHMMParams) -> np.ndarray:
        """Compute (T, K) posterior state probabilities."""
        log_emit = self._log_emissions(Y, obs_mask, params)
        log_trans = self._log_transitions(U, params)
        gamma, _, _ = self._forward_backward(log_emit, log_trans, params.log_pi)
        return gamma

    def viterbi(self, Y: np.ndarray, U: np.ndarray,
                obs_mask: Optional[np.ndarray],
                params: IOHMMParams) -> np.ndarray:
        """Viterbi decoding: MAP state sequence (vectorized over K)."""
        T, K = Y.shape[0], self.cfg.K
        log_emit = self._log_emissions(Y, obs_mask, params)
        log_trans = self._log_transitions(U, params)

        delta = np.empty((T, K), dtype=np.float64)
        psi = np.zeros((T, K), dtype=np.int32)
        delta[0] = params.log_pi + log_emit[0]

        for t in range(1, T):
            # (K,1) + (K,K) -> (K,K), then max over j (axis=0)
            scores = delta[t - 1, :, None] + log_trans[t]  # (K, K)
            psi[t] = np.argmax(scores, axis=0)  # (K,)
            delta[t] = scores[psi[t], np.arange(K)] + log_emit[t]

        path = np.empty(T, dtype=np.int32)
        path[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    def viterbi_min_dwell(self, Y: np.ndarray, U: np.ndarray,
                          obs_mask: Optional[np.ndarray],
                          params: IOHMMParams,
                          min_dwell: int = 0) -> np.ndarray:
        """Viterbi with minimum dwell time constraint.

        Uses expanded-state DP: each (state, dwell_counter) pair is a node.
        Transitions to a new state only allowed when dwell_counter >= min_dwell.

        Args:
            min_dwell: minimum samples in a state before transition allowed.
                       0 = standard Viterbi.
        """
        if min_dwell <= 1:
            return self.viterbi(Y, U, obs_mask, params)

        T, K = Y.shape[0], self.cfg.K
        log_emit = self._log_emissions(Y, obs_mask, params)
        log_trans = self._log_transitions(U, params)

        # Expanded state space: (K, min_dwell) — state k with d samples of dwell
        # d=0..min_dwell-1: must stay in state k
        # d=min_dwell-1: can transition
        D = min_dwell
        n_exp = K * D

        # delta[t, k*D+d] = best log-prob ending at time t in state k with dwell d
        delta = np.full((T, n_exp), -np.inf)
        psi = np.zeros((T, n_exp), dtype=np.int32)

        # Initialize: all states start at dwell=0
        for k in range(K):
            delta[0, k * D] = params.log_pi[k] + log_emit[0, k]

        for t in range(1, T):
            for k in range(K):
                # Self-transition: increment dwell counter
                for d in range(1, D):
                    prev = k * D + (d - 1)
                    score = delta[t-1, prev] + log_trans[t, k, k] + log_emit[t, k]
                    if score > delta[t, k * D + d]:
                        delta[t, k * D + d] = score
                        psi[t, k * D + d] = prev
                # Stay at max dwell (d=D-1 -> d=D-1)
                prev = k * D + (D - 1)
                score = delta[t-1, prev] + log_trans[t, k, k] + log_emit[t, k]
                if score > delta[t, k * D + (D-1)]:
                    delta[t, k * D + (D-1)] = score
                    psi[t, k * D + (D-1)] = prev

                # Transition from OTHER states (only from d=D-1)
                for j in range(K):
                    if j == k:
                        continue
                    prev = j * D + (D - 1)
                    score = delta[t-1, prev] + log_trans[t, j, k] + log_emit[t, k]
                    if score > delta[t, k * D]:  # new state starts at d=0
                        delta[t, k * D] = score
                        psi[t, k * D] = prev

        # Backtrack
        exp_path = np.empty(T, dtype=np.int32)
        exp_path[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            exp_path[t] = psi[t+1, exp_path[t+1]]

        # Convert expanded state to original state
        path = exp_path // D
        return path

    def log_likelihood(self, Y: np.ndarray, U: np.ndarray,
                       obs_mask: Optional[np.ndarray],
                       params: IOHMMParams) -> float:
        """Marginal log-likelihood from forward pass."""
        log_emit = self._log_emissions(Y, obs_mask, params)
        log_trans = self._log_transitions(U, params)
        _, _, ll = self._forward_backward(log_emit, log_trans, params.log_pi)
        return ll

    def bic(self, Y: np.ndarray, U: np.ndarray,
            obs_mask: Optional[np.ndarray],
            params: IOHMMParams) -> float:
        """Bayesian Information Criterion."""
        ll = self.log_likelihood(Y, U, obs_mask, params)
        n_params = self._count_params()
        return -2 * ll + n_params * np.log(len(Y))

    def _count_params(self) -> int:
        K, D, D_in = self.cfg.K, self.cfg.D_obs, self.cfg.D_input
        n_emission = K * D * 2      # mu + sigma2
        if self.cfg.ar_order >= 1:
            n_emission += K * D     # phi (AR coefficients)
        n_trans = K * K * (1 + D_in)  # W + S
        n_pi = K - 1                 # initial (one free)
        return n_emission + n_trans + n_pi


# ─────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────

def build_observation_mask(z_traces: dict, modality_keys: list,
                           T: int) -> np.ndarray:
    """Build (T, D) boolean mask: True = modality has data.

    A modality is considered missing if its entire z-trace is zero.
    """
    D = len(modality_keys)
    mask = np.ones((T, D), dtype=bool)
    for i, key in enumerate(modality_keys):
        z = z_traces.get(key)
        if z is None or np.abs(z).max() < 1e-8:
            mask[:, i] = False
    return mask


# ─────────────────────────────────────────────────────────────────────
#  SLDS Extension: Continuous Latent x_t with Kalman Filter-Smoother
# ─────────────────────────────────────────────────────────────────────

def _kalman_update(x_pred, P_pred, y, C, d, R_noise, mask=None):
    """Single Kalman measurement update with masked dims (Joseph form).

    Args:
        x_pred: (D,) predicted mean
        P_pred: (D, D) predicted covariance
        y: (m,) observation
        C: (m, D) emission matrix
        d: (m,) emission offset
        R_noise: (m,) diagonal or (m, m) full noise covariance
        mask: (m,) boolean or None

    Returns:
        x_filt, P_filt, log_lik_t
    """
    D = len(x_pred)
    if mask is not None:
        obs_idx = np.where(mask)[0]
        if len(obs_idx) == 0:
            return x_pred.copy(), P_pred.copy(), 0.0
        y_o, C_o, d_o = y[obs_idx], C[obs_idx], d[obs_idx]
        if R_noise.ndim == 1:
            R_mat = np.diag(R_noise[obs_idx])
        else:
            R_mat = R_noise[np.ix_(obs_idx, obs_idx)]
    else:
        y_o, C_o, d_o = y, C, d
        R_mat = np.diag(R_noise) if R_noise.ndim == 1 else R_noise.copy()

    innov = y_o - C_o @ x_pred - d_o
    S = C_o @ P_pred @ C_o.T + R_mat

    # Solve for Kalman gain via Cholesky (numerically stable)
    try:
        L = np.linalg.cholesky(S)
        K_gain = P_pred @ C_o.T @ np.linalg.solve(S, np.eye(len(y_o)))
    except np.linalg.LinAlgError:
        S += 1e-6 * np.eye(len(y_o))
        K_gain = P_pred @ C_o.T @ np.linalg.inv(S)

    x_filt = x_pred + K_gain @ innov

    # Joseph form for numerical stability
    I_KC = np.eye(D) - K_gain @ C_o
    P_filt = I_KC @ P_pred @ I_KC.T + K_gain @ R_mat @ K_gain.T
    P_filt = 0.5 * (P_filt + P_filt.T)

    # Log-likelihood contribution
    sign, logdet = np.linalg.slogdet(S)
    ll_t = -0.5 * (len(y_o) * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov))

    return x_filt, P_filt, float(ll_t)


def _factor_analyze(Sigma, r):
    """Factor analyze (m, m) covariance into F (m, r) and R (m,) diagonal.

    Sigma ~ F F' + diag(R)   via PPCA-style eigendecomposition.
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    m = Sigma.shape[0]
    noise_floor = max(np.mean(eigvals[r:]), 1e-6) if r < m else 1e-6
    F = eigvecs[:, :r] * np.sqrt(np.maximum(eigvals[:r] - noise_floor, 1e-8))
    R = np.diag(Sigma) - np.sum(F ** 2, axis=1)
    R = np.maximum(R, 1e-6)
    return F, R


# Numba-jitted Kalman smoother kernel. Profile (Phase 0.2b, 2026-05-01) showed
# the numpy version was 30s of 80s wall-time after the logsumexp swap;
# numba-jitted version eliminates the per-timestep Python loop overhead and
# uses np.linalg.solve directly. Two structural divergences from the numpy
# version are documented + validated:
#   1. Soft-mask via R-inflation: masked rows of R_full are zeroed except
#      the diagonal which gets BIG_R (1e10) so the Kalman gain on those rows
#      is ~0. Innovations on masked rows are explicitly zeroed to avoid
#      0*NaN=NaN if Y[t,masked_d] is garbage. Equivalent to the variable-
#      shape extraction up to ~1e-10.
#   2. Always-on ridge of 1e-10*I added to S (innovation covariance) before
#      solve; numpy version only added 1e-6*I on Cholesky failure. Ridge is
#      below typical conditioning levels in MVP/V11 fits.
# R_full has shape (T, m, m) always — wrapper expands diagonal R to full.
@numba.njit(cache=True, fastmath=False)
def _kalman_smoother_weighted_numba(Y, obs_mask, A_t, b_t, Q_t, C_t, d_t, R_full,
                                      x0_mean, P0):
    T = Y.shape[0]
    m = Y.shape[1]
    D = A_t.shape[1]
    BIG_R = 1e10
    eye_D = np.eye(D)
    eye_m = np.eye(m)

    x_pred = np.zeros((T, D))
    P_pred = np.zeros((T, D, D))
    x_filt = np.zeros((T, D))
    P_filt = np.zeros((T, D, D))

    # ── Forward filter ──
    for t in range(T):
        if t == 0:
            x_pred[0] = x0_mean
            P_pred[0] = P0
        else:
            x_pred[t] = A_t[t] @ x_filt[t - 1] + b_t[t]
            P_pred[t] = A_t[t] @ P_filt[t - 1] @ A_t[t].T + Q_t[t]
            P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T)

        # Build effective R matrix: inflate masked rows AND columns
        # (zero off-diagonals + BIG diagonal preserves Joseph form correctness)
        R_eff = R_full[t].copy()
        for d in range(m):
            if not obs_mask[t, d]:
                for j in range(m):
                    R_eff[d, j] = 0.0
                    R_eff[j, d] = 0.0
                R_eff[d, d] = BIG_R

        # Innovation: zero garbage on masked rows
        innov = Y[t] - C_t[t] @ x_pred[t] - d_t[t]
        for d in range(m):
            if not obs_mask[t, d]:
                innov[d] = 0.0

        # Innovation covariance + Kalman gain via solve
        S = C_t[t] @ P_pred[t] @ C_t[t].T + R_eff
        for d in range(m):
            S[d, d] += 1e-10
        K_gain = P_pred[t] @ C_t[t].T @ np.linalg.solve(S, eye_m)

        x_filt[t] = x_pred[t] + K_gain @ innov
        I_KC = eye_D - K_gain @ C_t[t]
        # Joseph form: P = (I-KC) P_pred (I-KC)' + K R_eff K'
        P_filt[t] = I_KC @ P_pred[t] @ I_KC.T + K_gain @ R_eff @ K_gain.T
        P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T)

    # ── RTS backward smoother ──
    x_smooth = np.zeros((T, D))
    P_smooth = np.zeros((T, D, D))
    Plag_smooth = np.zeros((T - 1, D, D))

    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred_tp1 = P_pred[t + 1].copy()
        eigvals = np.linalg.eigvalsh(P_pred_tp1)
        ev_min = eigvals[0]  # eigvalsh returns ascending
        for i in range(1, D):
            if eigvals[i] < ev_min:
                ev_min = eigvals[i]
        if ev_min < 1e-10:
            shift = 1e-10 - ev_min
            for i in range(D):
                P_pred_tp1[i, i] += shift
        G = P_filt[t] @ A_t[t + 1].T @ np.linalg.solve(P_pred_tp1, eye_D)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_tp1) @ G.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
        Plag_smooth[t] = P_smooth[t + 1] @ G.T

    return x_smooth, P_smooth, Plag_smooth


def _kalman_smoother_weighted(Y, obs_mask, A_t, b_t, Q_t, C_t, d_t, R_t,
                               x0_mean, P0):
    """Gamma-weighted Kalman filter-smoother.

    TEMPORARILY REVERTED to original numpy implementation for bisection
    (Phase 0 cross-check). The Numba kernel is suspect — soft-mask BIG_R
    inflation could compound 1e-10 errors across many EM iterations.
    """
    T, m = Y.shape
    D = A_t.shape[1]

    x_pred = np.zeros((T, D))
    P_pred = np.zeros((T, D, D))
    x_filt = np.zeros((T, D))
    P_filt = np.zeros((T, D, D))

    x_pred[0] = x0_mean
    P_pred[0] = P0.copy()
    mask_0 = obs_mask[0] if obs_mask is not None else None
    x_filt[0], P_filt[0], _ = _kalman_update(x_pred[0], P_pred[0], Y[0],
                                              C_t[0], d_t[0], R_t[0], mask_0)

    for t in range(1, T):
        x_pred[t] = A_t[t] @ x_filt[t - 1] + b_t[t]
        P_pred[t] = A_t[t] @ P_filt[t - 1] @ A_t[t].T + Q_t[t]
        P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T)
        mask_t = obs_mask[t] if obs_mask is not None else None
        x_filt[t], P_filt[t], _ = _kalman_update(x_pred[t], P_pred[t], Y[t],
                                                   C_t[t], d_t[t], R_t[t], mask_t)

    x_smooth = np.zeros((T, D))
    P_smooth = np.zeros((T, D, D))
    Plag_smooth = np.zeros((T - 1, D, D))

    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred_tp1 = P_pred[t + 1]
        eigvals = np.linalg.eigvalsh(P_pred_tp1)
        if eigvals.min() < 1e-10:
            P_pred_tp1 = P_pred_tp1 + (1e-10 - eigvals.min()) * np.eye(D)
        G = P_filt[t] @ A_t[t + 1].T @ np.linalg.inv(P_pred_tp1)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_tp1) @ G.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
        Plag_smooth[t] = P_smooth[t + 1] @ G.T

    return x_smooth, P_smooth, Plag_smooth


# ---------------------------------------------------------------------------
# Emissions log-likelihood helpers (Task 1: vectorized K loop)
# ---------------------------------------------------------------------------

def _mask_is_constant(obs_mask):
    """True iff every row of obs_mask is identical to the first row."""
    return np.all(obs_mask == obs_mask[0:1], axis=0).all()


def _emit_ll_per_k_loop_ref(Y, x_sm, P_sm, C_emit, d_emit, F_emit, R_emit, obs_mask):
    """Reference implementation: per-K Python loop with three branches.

    Extracted verbatim from the original slds_e_step inner loop so that the
    validation script can import and diff against _emit_ll_vectorized.

    Args:
        Y        : (T, m) observations
        x_sm     : (T, D) latent mean (smoothed)
        P_sm     : (T, D, D) latent covariance (smoothed)
        C_emit   : (K, m, D) emission loading matrices
        d_emit   : (K, m) emission offsets
        F_emit   : (K, m, n_factors) factor loadings, or None for diagonal
        R_emit   : (K, m) diagonal residual variances
        obs_mask : (T, m) bool mask or None

    Returns:
        log_emit : (T, K) float64 emission log-likelihoods
    """
    T, m = Y.shape
    K = C_emit.shape[0]
    use_factors = F_emit is not None

    log_emit = np.zeros((T, K), dtype=np.float64)

    # Precompute noise covariances for factor-analyzed case
    if use_factors:
        Sigma_noise = np.zeros((K, m, m))
        Sigma_inv = np.zeros((K, m, m))
        logdet_noise = np.zeros(K)
        for k in range(K):
            Sigma_noise[k] = F_emit[k] @ F_emit[k].T + np.diag(R_emit[k])
            eigv = np.linalg.eigvalsh(Sigma_noise[k])
            if eigv.min() < 1e-8:
                Sigma_noise[k] += (1e-8 - eigv.min()) * np.eye(m)
            Sigma_inv[k] = np.linalg.inv(Sigma_noise[k])
            logdet_noise[k] = np.linalg.slogdet(Sigma_noise[k])[1]

    # Determine mask structure once
    mask_const = True
    obs_idx_const = None
    m_obs_const = m
    if obs_mask is not None:
        mask_const = _mask_is_constant(obs_mask)
        if mask_const:
            obs_idx_const = np.where(obs_mask[0])[0]
            m_obs_const = len(obs_idx_const)

    if use_factors:
        for k in range(K):
            C_k = C_emit[k]
            d_k = d_emit[k]
            pred_mean = x_sm @ C_k.T + d_k
            resid = Y - pred_mean

            if obs_mask is not None and mask_const and m_obs_const < m:
                oi = obs_idx_const
                Sig_o = Sigma_noise[k][np.ix_(oi, oi)]
                Sig_inv_o = np.linalg.inv(Sig_o)
                logdet_o = np.linalg.slogdet(Sig_o)[1]
                C_o = C_k[oi]
                resid_o = resid[:, oi]
                CPC = np.einsum('di,tij,ej->tde', C_o, P_sm, C_o)
                quad = np.einsum('ti,ij,tj->t', resid_o, Sig_inv_o, resid_o)
                trace = np.einsum('ij,tij->t', Sig_inv_o, CPC)
                log_emit[:, k] = -0.5 * (m_obs_const * np.log(2 * np.pi)
                                          + logdet_o + quad + trace)
            elif obs_mask is not None and not mask_const:
                pair_mask = obs_mask[:, :, None] & obs_mask[:, None, :]
                pair_mask_f = pair_mask.astype(np.float64)
                Sigma_eff = Sigma_noise[k][None] * pair_mask_f
                diag_idx = np.arange(m)
                diag_add = (~obs_mask).astype(np.float64)
                Sigma_eff[:, diag_idx, diag_idx] += diag_add
                Sigma_inv_eff = np.linalg.inv(Sigma_eff)
                logdet_eff = np.linalg.slogdet(Sigma_eff)[1]
                n_obs_t = obs_mask.sum(axis=1).astype(np.float64)
                resid_eff = resid * obs_mask
                quad = np.einsum('ti,tij,tj->t', resid_eff, Sigma_inv_eff, resid_eff)
                CPC = np.einsum('di,tij,ej->tde', C_k, P_sm, C_k)
                CPC_eff = CPC * pair_mask_f
                trace = np.einsum('tij,tji->t', Sigma_inv_eff, CPC_eff)
                log_emit[:, k] = -0.5 * (n_obs_t * np.log(2 * np.pi)
                                          + logdet_eff + quad + trace)
            else:
                CPC = np.einsum('di,tij,ej->tde', C_k, P_sm, C_k)
                quad = np.einsum('ti,ij,tj->t', resid, Sigma_inv[k], resid)
                trace = np.einsum('ij,tij->t', Sigma_inv[k], CPC)
                log_emit[:, k] = -0.5 * (m * np.log(2 * np.pi)
                                          + logdet_noise[k] + quad + trace)
    else:
        for k in range(K):
            pred_mean = x_sm @ C_emit[k].T + d_emit[k]
            resid = Y - pred_mean
            var_x = np.einsum('di,tij,dj->td', C_emit[k], P_sm, C_emit[k])
            R_k = R_emit[k]
            ll = (-0.5 * np.log(2 * np.pi * R_k)
                  - 0.5 * (resid ** 2 + var_x) / R_k)
            if obs_mask is not None:
                ll = ll * obs_mask
            log_emit[:, k] = ll.sum(axis=1)

    return log_emit


def _emit_ll_vectorized(Y, x_sm, P_sm, C_emit, d_emit, F_emit, R_emit, obs_mask):
    """Vectorized K-loop emission log-likelihood via batched einsums.

    Numerically equivalent to _emit_ll_per_k_loop_ref (max|delta| < 1e-9).
    Replaces the per-K Python loop in slds_e_step with a leading K axis,
    saving ~K-1 einsum dispatch round-trips per inner SMF iteration.

    Signature and return value identical to _emit_ll_per_k_loop_ref.
    """
    T, m = Y.shape
    K = C_emit.shape[0]
    use_factors = F_emit is not None

    # pred_mean[k, t, :] = C_emit[k] @ x_sm[t] + d_emit[k]
    # C_emit shape: (K, m, D); x_sm: (T, D) -> einsum gives (K, T, m)
    pred_mean = np.einsum('kmd,td->ktm', C_emit, x_sm) + d_emit[:, np.newaxis, :]
    resid = Y[np.newaxis, :, :] - pred_mean   # (K, T, m)

    if use_factors:
        # Build Sigma_noise[k] = F[k] F[k]' + diag(R[k])  for all k at once
        # F_emit: (K, m, n_factors)
        Sigma = np.einsum('kmi,kni->kmn', F_emit, F_emit)     # (K, m, m)
        Sigma += np.eye(m) * R_emit[:, :, np.newaxis]          # broadcast diag

        # Per-state eigenvalue regularization (preserves per-k logic)
        eigv = np.linalg.eigvalsh(Sigma)          # (K, m)
        eigv_min = eigv.min(axis=1)               # (K,)
        adj = np.maximum(1e-8 - eigv_min, 0.0)   # (K,)
        Sigma += adj[:, np.newaxis, np.newaxis] * np.eye(m)

        Sigma_inv = np.linalg.inv(Sigma)                         # (K, m, m)
        logdet = np.linalg.slogdet(Sigma)[1]                     # (K,)

        # CPC[k, t, d, e] = sum_{ij} C[k,d,i] P[t,i,j] C[k,e,j]
        CPC = np.einsum('kdi,tij,kej->ktde', C_emit, P_sm, C_emit)   # (K, T, m, m)

        if obs_mask is not None and not _mask_is_constant(obs_mask):
            # Variable mask: identity-replacement trick, batched over K
            pair_mask = obs_mask[:, :, np.newaxis] & obs_mask[:, np.newaxis, :]  # (T, m, m)
            pair_mask_f = pair_mask.astype(np.float64)
            # Sigma_eff[k, t] = Sigma[k] * pair_mask[t] + I * (~obs_mask[t])
            Sigma_eff = Sigma[:, np.newaxis, :, :] * pair_mask_f[np.newaxis, :, :, :]  # (K, T, m, m)
            diag_idx = np.arange(m)
            diag_add = (~obs_mask).astype(np.float64)   # (T, m)
            Sigma_eff[:, :, diag_idx, diag_idx] += diag_add[np.newaxis, :, :]
            Sigma_inv_eff = np.linalg.inv(Sigma_eff)    # (K, T, m, m)
            logdet_eff = np.linalg.slogdet(Sigma_eff)[1]  # (K, T)
            n_obs_t = obs_mask.sum(axis=1).astype(np.float64)  # (T,)
            resid_eff = resid * obs_mask[np.newaxis, :, :]     # (K, T, m)
            quad = np.einsum('kti,ktij,ktj->kt', resid_eff, Sigma_inv_eff, resid_eff)
            CPC_eff = CPC * pair_mask_f[np.newaxis, :, :, :]   # (K, T, m, m)
            trace = np.einsum('ktij,ktji->kt', Sigma_inv_eff, CPC_eff)
            log_emit = -0.5 * (n_obs_t[np.newaxis, :] * np.log(2 * np.pi)
                               + logdet_eff + quad + trace)
            return log_emit.T   # (T, K)

        # Constant or no mask
        if obs_mask is not None:
            oi = np.where(obs_mask[0])[0]
            m_obs = len(oi)
            if m_obs < m:
                # Subselect observed channels
                Sigma_o = Sigma[:, oi[:, np.newaxis], oi[np.newaxis, :]]   # (K, m_obs, m_obs)
                Sigma_inv_o = np.linalg.inv(Sigma_o)
                logdet_o = np.linalg.slogdet(Sigma_o)[1]   # (K,)
                C_o = C_emit[:, oi, :]                      # (K, m_obs, D)
                resid_o = resid[:, :, oi]                   # (K, T, m_obs)
                CPC_o = np.einsum('kdi,tij,kej->ktde', C_o, P_sm, C_o)   # (K, T, m_obs, m_obs)
                quad = np.einsum('kti,kij,ktj->kt', resid_o, Sigma_inv_o, resid_o)
                trace = np.einsum('kij,ktij->kt', Sigma_inv_o, CPC_o)
                log_emit = -0.5 * (m_obs * np.log(2 * np.pi)
                                   + logdet_o[:, np.newaxis] + quad + trace)
                return log_emit.T   # (T, K)
            # All channels observed (all-true constant mask) -> fall through to no-mask path

        # No mask (or all-obs constant mask)
        quad = np.einsum('kti,kij,ktj->kt', resid, Sigma_inv, resid)
        trace = np.einsum('kij,ktij->kt', Sigma_inv, CPC)
        log_emit = -0.5 * (m * np.log(2 * np.pi) + logdet[:, np.newaxis] + quad + trace)
        return log_emit.T   # (T, K)

    # Diagonal path (n_factors == 0)
    # var_x[k, t, d] = sum_{ij} C[k,d,i] P[t,i,j] C[k,d,j]
    var_x = np.einsum('kdi,tij,kdj->ktd', C_emit, P_sm, C_emit)   # (K, T, m)
    R = R_emit[:, np.newaxis, :]   # (K, 1, m)
    ll_per_dim = -0.5 * np.log(2 * np.pi * R) - 0.5 * (resid ** 2 + var_x) / R
    if obs_mask is not None:
        ll_per_dim = ll_per_dim * obs_mask[np.newaxis, :, :]
    log_emit = ll_per_dim.sum(axis=2)   # (K, T)
    return log_emit.T   # (T, K)


def slds_e_step(Y, U, obs_mask, params, cfg, iohmm):
    """Structured Mean-Field E-step for SLDS.

    Alternates between q(z) (forward-backward) and q(x) (Kalman smoother).
    Supports factor-analyzed emissions (n_factors>0) and recurrent transitions.
    """
    T, m = Y.shape
    K, D = cfg.K, cfg.D_latent
    use_factors = (cfg.n_factors > 0 and hasattr(params, 'F_emit')
                   and params.F_emit is not None)
    use_recurrent = (cfg.recurrent and hasattr(params, 'R_recur')
                     and params.R_recur is not None)

    # Initialize x_smooth from zero
    x_sm = np.zeros((T, D))
    P_sm = np.tile(params.P0, (T, 1, 1))
    Plag_sm = np.zeros((T - 1, D, D))

    gamma = None
    xi = None
    log_lik = -np.inf

    # Precompute per-state noise covariances if factor-analyzed
    if use_factors:
        Sigma_noise = np.zeros((K, m, m))
        for k in range(K):
            Sigma_noise[k] = (params.F_emit[k] @ params.F_emit[k].T
                              + np.diag(params.R_emit[k]))
            # Regularize for invertibility
            eigv = np.linalg.eigvalsh(Sigma_noise[k])
            if eigv.min() < 1e-8:
                Sigma_noise[k] += (1e-8 - eigv.min()) * np.eye(m)

    for inner in range(cfg.n_inner_estep):
        # Step 1: q(z) — compute expected emission LL under q(x)
        # Vectorized over K states (replaces per-K Python loop; see
        # _emit_ll_vectorized for branch logic and _emit_ll_per_k_loop_ref
        # for the numerically equivalent reference implementation).
        F_emit = params.F_emit if use_factors else None
        log_emit = _emit_ll_vectorized(
            Y, x_sm, P_sm,
            params.C_emit, params.d_emit, F_emit, params.R_emit,
            obs_mask,
        )

        # Transitions (with optional recurrence from x_{t-1})
        if use_recurrent:
            log_trans = _log_transitions_recurrent(U, x_sm, params, cfg)
        else:
            log_trans = iohmm._log_transitions(U, params)

        gamma, xi, log_lik = iohmm._forward_backward(log_emit, log_trans,
                                                       params.log_pi)

        # Step 2: q(x) — gamma-weighted Kalman smoother
        A_t = np.einsum('tk,kij->tij', gamma, params.A_dyn)
        b_t = np.einsum('tk,ki->ti', gamma, params.b_dyn)
        Q_t = np.einsum('tk,kij->tij', gamma, params.Q_dyn)
        C_t = np.einsum('tk,kdi->tdi', gamma, params.C_emit)
        d_t = np.einsum('tk,kd->td', gamma, params.d_emit)

        if use_factors:
            R_t = np.einsum('tk,kij->tij', gamma, Sigma_noise)
        else:
            R_t = np.einsum('tk,kd->td', gamma, params.R_emit)

        x_sm, P_sm, Plag_sm = _kalman_smoother_weighted(
            Y, obs_mask, A_t, b_t, Q_t, C_t, d_t, R_t,
            params.x0_mean, params.P0)

    return gamma, xi, log_lik, x_sm, P_sm, Plag_sm


def _log_transitions_recurrent(U, x, params, cfg):
    """Log transitions with recurrent x_{t-1} -> z_t term.

    logits[t,j,k] = W[j,k] + S[j,k,:] @ u_t + R_recur[j,k,:] @ x_{t-1}
    """
    T = U.shape[0]
    K = cfg.K
    logits = params.W_trans[None, :, :] + np.einsum('td,jkd->tjk', U, params.S_trans)

    x_prev = np.empty((T, cfg.D_latent))
    x_prev[0] = params.x0_mean
    x_prev[1:] = x[:-1]
    logits += np.einsum('td,jkd->tjk', x_prev, params.R_recur)

    log_trans = logits - logsumexp(logits, axis=2, keepdims=True)
    return log_trans


def slds_m_step_dynamics(gamma, x_sm, P_sm, Plag_sm, K, D):
    """M-step for dynamics: A[k], b[k], Q[k]."""
    T = x_sm.shape[0]
    A_new = np.zeros((K, D, D))
    b_new = np.zeros((K, D))
    Q_new = np.zeros((K, D, D))

    for k in range(K):
        w = gamma[1:, k]
        N_k = w.sum()
        if N_k < D + 2:
            A_new[k] = 0.9 * np.eye(D)
            Q_new[k] = 0.1 * np.eye(D)
            continue

        # Sufficient statistics
        xx_prev = P_sm[:-1] + x_sm[:-1, :, None] * x_sm[:-1, None, :]
        S_pp = np.einsum('t,tij->ij', w, xx_prev)
        S_p = np.einsum('t,ti->i', w, x_sm[:-1])

        xx_cross = Plag_sm + x_sm[1:, :, None] * x_sm[:-1, None, :]
        S_xp = np.einsum('t,tij->ij', w, xx_cross)
        S_x = np.einsum('t,ti->i', w, x_sm[1:])

        # Augmented system: [A | b] = S_xp_aug @ inv(S_pp_aug)
        S_pp_aug = np.zeros((D + 1, D + 1))
        S_pp_aug[:D, :D] = S_pp
        S_pp_aug[:D, D] = S_p
        S_pp_aug[D, :D] = S_p
        S_pp_aug[D, D] = N_k

        S_xp_aug = np.zeros((D, D + 1))
        S_xp_aug[:, :D] = S_xp
        S_xp_aug[:, D] = S_x

        try:
            Ab = np.linalg.solve(S_pp_aug.T, S_xp_aug.T).T
        except np.linalg.LinAlgError:
            Ab = S_xp_aug @ np.linalg.pinv(S_pp_aug)
        A_new[k] = Ab[:, :D]
        b_new[k] = Ab[:, D]

        xx_curr = P_sm[1:] + x_sm[1:, :, None] * x_sm[1:, None, :]
        S_xx = np.einsum('t,tij->ij', w, xx_curr)
        Q_new[k] = (S_xx - Ab @ S_xp_aug.T) / N_k
        Q_new[k] = 0.5 * (Q_new[k] + Q_new[k].T)
        eigvals = np.linalg.eigvalsh(Q_new[k])
        if eigvals.min() < 1e-6:
            Q_new[k] += (1e-6 - eigvals.min()) * np.eye(D)

    return A_new, b_new, Q_new


def slds_m_step_emissions(Y, obs_mask, gamma, x_sm, P_sm, K, D, m,
                          n_factors=0, c_shrinkage=0.0):
    """M-step for emissions: C[k], d[k], R[k], optionally F[k].

    When n_factors > 0, residual covariance is factor-analyzed into
    F[k] (m, r) loadings + R[k] (m,) diagonal, capturing cross-modal
    correlations in the emission noise.

    When c_shrinkage > 0, C[k] is regularized toward C_mean across states
    (Phase 3.2 shared loading). Typical value: 0.3.
    """
    T = Y.shape[0]
    C_new = np.zeros((K, m, D))
    d_new = np.zeros((K, m))
    R_new = np.ones((K, m)) * 0.1  # default

    # null_state flag passed via closure or global; check if available
    _null_state = getattr(slds_m_step_emissions, '_null_state', False)
    _null_sigma2_cap = getattr(slds_m_step_emissions, '_null_sigma2_cap', 0.5)

    for k in range(K):
        w = gamma[:, k]
        # Null state (k=0): solve for C only (d[0]=0 fixed)
        is_null = _null_state and k == 0

        if is_null:
            x_aug = x_sm  # no intercept column
            n_aug = D
        else:
            x_aug = np.concatenate([x_sm, np.ones((T, 1))], axis=1)
            n_aug = D + 1

        for d_obs in range(m):
            w_d = w.copy()
            if obs_mask is not None:
                w_d = w * obs_mask[:, d_obs]
            N_kd = w_d.sum()
            if N_kd < D + 2:
                continue

            # S_xx_aug with P uncertainty
            xx = P_sm + x_sm[:, :, None] * x_sm[:, None, :]
            if is_null:
                S_xx_aug = np.einsum('t,tij->ij', w_d, xx)
                S_yx_aug = np.einsum('t,t,ti->i', w_d, Y[:, d_obs], x_sm)
            else:
                S_xx_aug = np.zeros((D + 1, D + 1))
                S_xx_aug[:D, :D] = np.einsum('t,tij->ij', w_d, xx)
                S_xx_aug[:D, D] = np.einsum('t,ti->i', w_d, x_sm)
                S_xx_aug[D, :D] = S_xx_aug[:D, D]
                S_xx_aug[D, D] = N_kd
                S_yx_aug = np.zeros(D + 1)
                S_yx_aug[:D] = np.einsum('t,t,ti->i', w_d, Y[:, d_obs], x_sm)
                S_yx_aug[D] = (w_d * Y[:, d_obs]).sum()

            try:
                Cd = np.linalg.solve(S_xx_aug, S_yx_aug)
            except np.linalg.LinAlgError:
                Cd = np.linalg.pinv(S_xx_aug) @ S_yx_aug

            if is_null:
                C_new[k, d_obs] = Cd[:D]
                d_new[k, d_obs] = 0.0  # fixed at zero
            else:
                C_new[k, d_obs] = Cd[:D]
                d_new[k, d_obs] = Cd[D]

            S_yy = (w_d * Y[:, d_obs] ** 2).sum()
            R_new[k, d_obs] = max((S_yy - Cd @ S_yx_aug) / N_kd, 1e-6)

    # Cap null state emission variance
    if _null_state:
        R_new[0] = np.clip(R_new[0], 1e-6, _null_sigma2_cap)

    # Phase 3.2: C shrinkage toward shared loading
    if c_shrinkage > 0 and K > 1:
        # Exclude null state from mean computation
        start_k = 1 if _null_state else 0
        C_mean = C_new[start_k:].mean(axis=0)
        for k in range(start_k, K):
            C_new[k] = (1 - c_shrinkage) * C_new[k] + c_shrinkage * C_mean

    # Factor analysis of residual covariance (Phase 3.2)
    F_new = None
    if n_factors > 0:
        F_new = np.zeros((K, m, n_factors))
        for k in range(K):
            w = gamma[:, k]
            N_k = w.sum()
            if N_k < m + 2:
                continue
            pred = x_sm @ C_new[k].T + d_new[k]
            resid = Y - pred
            if obs_mask is not None:
                resid = resid * obs_mask
                w_f = w * obs_mask.all(axis=1).astype(float)
            else:
                w_f = w
            N_kf = w_f.sum()
            if N_kf < m + 2:
                continue
            w_n = w_f / N_kf
            Sigma_resid = np.einsum('t,ti,tj->ij', w_n, resid, resid)
            F_new[k], R_new[k] = _factor_analyze(Sigma_resid, n_factors)

    return C_new, d_new, R_new, F_new


def slds_m_step_transitions_recurrent(U, xi, gamma, x_smooth, params, cfg):
    """M-step for transitions with recurrence: W, S, R_recur.

    Optimizes P(z_t=k | z_{t-1}=j, u_t, x_{t-1}) = softmax(W + S@u + R@x)
    via L-BFGS-B with stronger regularization on R_recur to prevent overfitting.
    """
    K = cfg.K
    D_in = cfg.D_input
    D_lat = cfg.D_latent
    T_minus_1 = xi.shape[0]

    x_prev = x_smooth[:-1].copy()  # (T-1, D_lat)

    def _pack(W, S, R):
        return np.concatenate([W.ravel(), S.ravel(), R.ravel()])

    def _unpack(x):
        n_W = K * K
        n_S = K * K * D_in
        W = x[:n_W].reshape(K, K)
        S = x[n_W:n_W + n_S].reshape(K, K, D_in)
        R = x[n_W + n_S:].reshape(K, K, D_lat)
        return W, S, R

    def _objective(x):
        W, S, R = _unpack(x)
        logits = (W[None, :, :] + np.einsum('td,jkd->tjk', U[1:], S)
                  + np.einsum('td,jkd->tjk', x_prev, R))
        log_A = logits - logsumexp(logits, axis=2, keepdims=True)

        loss = -np.sum(xi * log_A)
        loss += 0.01 * np.sum(S ** 2) + 0.05 * np.sum(R ** 2)

        A = np.exp(log_A)
        gamma_from = gamma[:-1]
        err = xi - gamma_from[:, :, None] * A

        grad_W = -err.sum(axis=0)
        grad_S = -np.einsum('tjk,td->jkd', err, U[1:]) + 0.02 * S
        grad_R = -np.einsum('tjk,td->jkd', err, x_prev) + 0.10 * R

        return loss, _pack(grad_W, grad_S, grad_R)

    x0 = _pack(params.W_trans, params.S_trans, params.R_recur)
    result = minimize(_objective, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': 20, 'ftol': 1e-6})
    W_new, S_new, R_new = _unpack(result.x)

    params.W_trans = W_new
    params.S_trans = S_new
    params.R_recur = R_new

    g0 = gamma[0].clip(min=1e-8)
    params.log_pi = np.log(g0 / g0.sum())
    return params


def initialize_slds(Y, U, obs_mask, cfg, seed=42):
    """Initialize SLDS from IOHMM warm-start."""
    K, D, m = cfg.K, cfg.D_latent, cfg.D_obs
    rng = np.random.default_rng(seed)

    # Run vanilla IOHMM for warm-start (strong init for good d, gamma)
    iohmm_cfg = IOHMMConfig(K=K, D_obs=m, D_input=cfg.D_input,
                            max_em_iter=100, n_restarts=3)
    iohmm = IOHMM(iohmm_cfg)
    iohmm_params, iohmm_hist = iohmm.fit(Y, U, obs_mask, seed=seed, verbose=False)
    gamma_init = iohmm_hist['gamma']

    d_emit = iohmm_params.mu.copy()
    R_emit = iohmm_params.sigma2.copy()

    # C from global PCA of pooled residuals (shared across states initially).
    # Per-state PCA is noisy and can diverge; global PCA gives stable C init.
    C_emit = np.zeros((K, m, D))
    Y_filled = Y.copy()
    if obs_mask is not None:
        Y_filled[~obs_mask] = 0.0
    # Pooled residual PCA
    resid_pooled = Y_filled - np.einsum('tk,kd->td', gamma_init, d_emit)
    cov_pooled = np.cov(resid_pooled, rowvar=False)
    eigvals_g, eigvecs_g = np.linalg.eigh(cov_pooled)
    C_global = eigvecs_g[:, -D:]  # top D eigenvectors
    for k in range(K):
        C_emit[k] = C_global * 0.01  # start very small — acts like IOHMM initially

    A_dyn = np.zeros((K, D, D))
    for k in range(K):
        A_dyn[k] = 0.9 * np.eye(D)
    b_dyn = np.zeros((K, D))
    data_var = max(np.var(Y_filled), 1e-4)
    Q_dyn = np.zeros((K, D, D))
    for k in range(K):
        Q_dyn[k] = 0.1 * data_var * np.eye(D)

    x0_mean = np.zeros(D)
    P0 = cfg.kalman_init_P0 * np.eye(D)

    # Factor loadings: small random init (Phase 3.2)
    # Don't use PCA of residuals — it confounds C@x with F@w, causing
    # identifiability issues. Let EM discover factor structure from residuals
    # after C has converged.
    F_emit = None
    if cfg.n_factors > 0:
        r = cfg.n_factors
        F_emit = rng.normal(0, 0.05, (K, m, r))
        # R_emit stays from IOHMM sigma2 (already set above)

    # Recurrence weights: initialize to zero (Phase 3.3)
    R_recur = None
    if cfg.recurrent and cfg.D_latent > 0:
        R_recur = np.zeros((K, K, cfg.D_latent))

    return IOHMMParams(
        mu=d_emit,
        log_sigma2=np.log(R_emit.clip(min=1e-6)),
        W_trans=iohmm_params.W_trans.copy(),
        S_trans=iohmm_params.S_trans.copy(),
        log_pi=iohmm_params.log_pi.copy(),
    ), {
        'A_dyn': A_dyn, 'b_dyn': b_dyn, 'Q_dyn': Q_dyn,
        'C_emit': C_emit, 'd_emit': d_emit, 'R_emit': R_emit,
        'x0_mean': x0_mean, 'P0': P0,
        'F_emit': F_emit, 'R_recur': R_recur,
    }, iohmm


def fit_slds(Y, U, obs_mask, cfg, seed=42, verbose=True):
    """Fit SLDS via EM with structured mean-field E-step.

    Returns:
        params: IOHMMParams (with slds_extras dict attached)
        history: dict with gamma, x_smooth, ll_trace, etc.
    """
    K, D, m = cfg.K, cfg.D_latent, cfg.D_obs
    best_ll = -np.inf
    best_result = None
    iohmm_ref = IOHMM(cfg)  # for forward-backward and transitions

    # Pass null_state config to emission M-step via function attributes
    slds_m_step_emissions._null_state = cfg.null_state
    slds_m_step_emissions._null_sigma2_cap = cfg.null_sigma2_cap

    for restart in range(cfg.n_restarts):
        rseed = seed + restart * 1000
        params, slds_ext, _ = initialize_slds(Y, U, obs_mask, cfg, seed=rseed)

        # Attach SLDS params as attributes on a simple namespace
        class SLDSState:
            pass
        sp = SLDSState()
        sp.A_dyn = slds_ext['A_dyn']
        sp.b_dyn = slds_ext['b_dyn']
        sp.Q_dyn = slds_ext['Q_dyn']
        sp.C_emit = slds_ext['C_emit']
        sp.d_emit = slds_ext['d_emit']
        sp.R_emit = slds_ext['R_emit']
        sp.x0_mean = slds_ext['x0_mean']
        sp.P0 = slds_ext['P0']
        # Factor loadings (Phase 3.2)
        sp.F_emit = slds_ext.get('F_emit', None)
        # Recurrence weights (Phase 3.3)
        sp.R_recur = slds_ext.get('R_recur', None)
        # Copy transition params from IOHMM params
        sp.W_trans = params.W_trans
        sp.S_trans = params.S_trans
        sp.log_pi = params.log_pi

        ll_trace = []
        prev_ll = -np.inf

        # Staged fitting: freeze d (between-state means) for most of EM
        # to prevent C@x from absorbing between-state signal. Also delay
        # factor analysis until the base model has stabilized.
        # Stage 1 (0 to 70%): d frozen, diagonal R, C/A/Q learn from residuals
        # Stage 2 (70% to end): d unfrozen, FA enabled, full refinement
        d_init = sp.d_emit.copy()
        stage2_iter = max(cfg.anneal_iters, int(0.7 * cfg.max_em_iter))

        for it in range(cfg.max_em_iter):
            temp = cfg.anneal_temp - (cfg.anneal_temp - 1.0) * min(it, cfg.anneal_iters) / max(cfg.anneal_iters, 1)
            temp = max(temp, 1.0)

            # Stage control: freeze d and delay FA during early iterations
            freeze_d = (it < stage2_iter)
            use_fa_this_iter = (cfg.n_factors > 0 and it >= stage2_iter)

            # E-step: during stage 1, F_emit should be near-zero (from init)
            # so the emission LL uses diagonal path. After stage 2, F grows.
            gamma, xi, log_lik, x_sm, P_sm, Plag_sm = slds_e_step(
                Y, U, obs_mask, sp, cfg, iohmm_ref)
            ll_trace.append(log_lik)

            if it > 0:
                rel_change = (log_lik - prev_ll) / max(abs(prev_ll), 1.0)
                if rel_change < cfg.em_tol and temp == 1.0:
                    if verbose:
                        print(f"    Restart {restart}: converged at iter {it}, LL={log_lik:.1f}")
                    break
            prev_ll = log_lik

            # M-step: dynamics
            sp.A_dyn, sp.b_dyn, sp.Q_dyn = slds_m_step_dynamics(
                gamma, x_sm, P_sm, Plag_sm, K, D)

            # M-step: emissions (FA only after stage2)
            n_fac = cfg.n_factors if use_fa_this_iter else 0
            C_new, d_new, R_new, F_new = slds_m_step_emissions(
                Y, obs_mask, gamma, x_sm, P_sm, K, D, m,
                n_factors=n_fac,
                c_shrinkage=cfg.c_shrinkage if use_fa_this_iter else 0.0)
            sp.C_emit = C_new
            sp.R_emit = R_new
            if F_new is not None:
                sp.F_emit = F_new
            if freeze_d:
                sp.d_emit = d_init  # lock d to IOHMM values
            else:
                sp.d_emit = d_new
            # Null state: always enforce d[0]=0
            if cfg.null_state:
                sp.d_emit[0, :] = 0.0

            # M-step: transitions
            if cfg.recurrent and sp.R_recur is not None:
                sp = slds_m_step_transitions_recurrent(
                    U, xi, gamma, x_sm, sp, cfg)
            else:
                trans_params = IOHMMParams(
                    mu=sp.d_emit, log_sigma2=np.log(sp.R_emit.clip(min=1e-6)),
                    W_trans=sp.W_trans, S_trans=sp.S_trans, log_pi=sp.log_pi)
                trans_params = iohmm_ref._m_step_transitions(
                    U, xi, gamma, trans_params)
                sp.W_trans = trans_params.W_trans
                sp.S_trans = trans_params.S_trans
                sp.log_pi = trans_params.log_pi

        final_ll = ll_trace[-1] if ll_trace else -np.inf
        if verbose:
            usage = gamma.mean(axis=0)
            usage_str = ' '.join(f'{u:.1%}' for u in usage)
            print(f"    Restart {restart}: LL={final_ll:.1f}, iters={len(ll_trace)}, "
                  f"usage=[{usage_str}]")

        if final_ll > best_ll:
            best_ll = final_ll
            slds_dict = {
                'A_dyn': sp.A_dyn.copy(), 'b_dyn': sp.b_dyn.copy(),
                'Q_dyn': sp.Q_dyn.copy(), 'C_emit': sp.C_emit.copy(),
                'd_emit': sp.d_emit.copy(), 'R_emit': sp.R_emit.copy(),
                'x0_mean': sp.x0_mean.copy(), 'P0': sp.P0.copy(),
            }
            if sp.F_emit is not None:
                slds_dict['F_emit'] = sp.F_emit.copy()
            if sp.R_recur is not None:
                slds_dict['R_recur'] = sp.R_recur.copy()
            best_result = {
                'params': IOHMMParams(
                    mu=sp.d_emit.copy(), log_sigma2=np.log(sp.R_emit.clip(min=1e-6)),
                    W_trans=sp.W_trans.copy(), S_trans=sp.S_trans.copy(),
                    log_pi=sp.log_pi.copy()),
                'slds': slds_dict,
                'gamma': gamma, 'x_smooth': x_sm, 'P_smooth': P_sm,
                'll_trace': ll_trace, 'restart': restart,
            }

    # BIC
    n_dynamics = K * (D * D + D + D * (D + 1) // 2)  # A, b, Q
    n_emission = K * (m * D + m + m)  # C, d, R
    if cfg.n_factors > 0:
        n_emission += K * m * cfg.n_factors  # F factor loadings
    n_trans = K * K * (1 + cfg.D_input) + K - 1
    if cfg.recurrent:
        n_trans += K * K * D  # R_recur
    n_params = n_dynamics + n_emission + n_trans
    bic = -2 * best_ll + n_params * np.log(len(Y))

    best_result['bic'] = bic
    best_result['final_ll'] = best_ll
    best_result['n_params'] = n_params

    if verbose:
        print(f"  Best: restart {best_result['restart']}, LL={best_ll:.1f}, "
              f"BIC={bic:.1f}, params={n_params}")

    return best_result['params'], best_result


# ─────────────────────────────────────────────────────────────────────
#  Phase 3.4: Hierarchical SLDS (shared dynamics, session-specific emissions)
# ─────────────────────────────────────────────────────────────────────

def _align_states_to_reference(ref_mu, session_mus):
    """Align state orderings using Hungarian matching on emission means."""
    from scipy.optimize import linear_sum_assignment
    K = ref_mu.shape[0]
    perms = []
    for mu in session_mus:
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.sum((ref_mu[i] - mu[j]) ** 2)
        _, col_ind = linear_sum_assignment(cost)
        perms.append(col_ind)
    return perms


def _apply_state_permutation(d, perm):
    """Apply state permutation to a parameter dict."""
    out = {}
    for key, val in d.items():
        if val is None:
            out[key] = None
        elif key in ('A_dyn', 'Q_dyn', 'C_emit', 'b_dyn', 'd_emit', 'R_emit'):
            out[key] = val[perm]
        elif key == 'F_emit':
            out[key] = val[perm] if val is not None else None
        elif key == 'W_trans':
            out[key] = val[perm][:, perm]
        elif key in ('S_trans', 'R_recur'):
            out[key] = val[perm][:, perm, :] if val is not None else None
        elif key == 'gamma':
            out[key] = val[:, perm]
        elif key == 'log_pi':
            out[key] = val[perm]
        else:
            out[key] = val
    return out


def _hierarchical_m_step_dynamics(all_gamma, all_x_sm, all_P_sm, all_Plag,
                                   K, D):
    """Pool dynamics M-step across sessions: shared A[k], b[k], Q[k]."""
    A_new = np.zeros((K, D, D))
    b_new = np.zeros((K, D))
    Q_new = np.zeros((K, D, D))

    for k in range(K):
        S_pp = np.zeros((D + 1, D + 1))
        S_xp = np.zeros((D, D + 1))
        S_xx = np.zeros((D, D))
        total_N = 0.0

        for n in range(len(all_gamma)):
            w = all_gamma[n][1:, k]
            N_k = w.sum()
            if N_k < 1e-8:
                continue
            total_N += N_k
            x = all_x_sm[n]
            P = all_P_sm[n]
            Pl = all_Plag[n]

            xx_prev = P[:-1] + x[:-1, :, None] * x[:-1, None, :]
            S_pp[:D, :D] += np.einsum('t,tij->ij', w, xx_prev)
            S_pp[:D, D] += np.einsum('t,ti->i', w, x[:-1])
            S_pp[D, D] += N_k

            xx_cross = Pl + x[1:, :, None] * x[:-1, None, :]
            S_xp[:, :D] += np.einsum('t,tij->ij', w, xx_cross)
            S_xp[:, D] += np.einsum('t,ti->i', w, x[1:])

            xx_curr = P[1:] + x[1:, :, None] * x[1:, None, :]
            S_xx += np.einsum('t,tij->ij', w, xx_curr)

        S_pp[D, :D] = S_pp[:D, D]  # symmetrize after accumulation

        if total_N < D + 2:
            A_new[k] = 0.9 * np.eye(D)
            Q_new[k] = 0.1 * np.eye(D)
            continue

        try:
            Ab = np.linalg.solve(S_pp.T, S_xp.T).T
        except np.linalg.LinAlgError:
            Ab = S_xp @ np.linalg.pinv(S_pp)
        A_new[k] = Ab[:, :D]
        b_new[k] = Ab[:, D]

        Q_new[k] = (S_xx - Ab @ S_xp.T) / total_N
        Q_new[k] = 0.5 * (Q_new[k] + Q_new[k].T)
        eigvals = np.linalg.eigvalsh(Q_new[k])
        if eigvals.min() < 1e-6:
            Q_new[k] += (1e-6 - eigvals.min()) * np.eye(D)

    return A_new, b_new, Q_new


def _hierarchical_m_step_transitions(sessions, estep_results, shared, cfg):
    """Pool transition M-step across sessions: shared W, S, optionally R_recur."""
    K, D_in = cfg.K, cfg.D_input
    use_rec = cfg.recurrent and shared.get('R_recur') is not None
    D_lat = cfg.D_latent if use_rec else 0

    def _pack(*args):
        return np.concatenate([a.ravel() for a in args])

    def _unpack_ws(x):
        n_W = K * K
        return x[:n_W].reshape(K, K), x[n_W:].reshape(K, K, D_in)

    def _unpack_wsr(x):
        n_W, n_S = K * K, K * K * D_in
        return (x[:n_W].reshape(K, K), x[n_W:n_W + n_S].reshape(K, K, D_in),
                x[n_W + n_S:].reshape(K, K, D_lat))

    def _objective(x):
        if use_rec:
            W, S, R = _unpack_wsr(x)
        else:
            W, S = _unpack_ws(x)

        loss = 0.0
        gW = np.zeros((K, K))
        gS = np.zeros((K, K, D_in))
        gR = np.zeros((K, K, D_lat)) if use_rec else None

        for n in range(len(sessions)):
            U_n = sessions[n][1]
            xi_n = estep_results[n]['xi']
            gamma_n = estep_results[n]['gamma']

            logits = W[None, :, :] + np.einsum('td,jkd->tjk', U_n[1:], S)
            if use_rec:
                xp = estep_results[n]['x_smooth'][:-1]
                logits += np.einsum('td,jkd->tjk', xp, R)

            log_A = logits - logsumexp(logits, axis=2, keepdims=True)
            loss += -np.sum(xi_n * log_A)

            err = xi_n - gamma_n[:-1, :, None] * np.exp(log_A)
            gW += -err.sum(axis=0)
            gS += -np.einsum('tjk,td->jkd', err, U_n[1:])
            if use_rec:
                gR += -np.einsum('tjk,td->jkd', err, xp)

        loss += 0.01 * np.sum(S ** 2)
        gS += 0.02 * S
        if use_rec:
            loss += 0.05 * np.sum(R ** 2)
            gR += 0.10 * R
            return loss, _pack(gW, gS, gR)
        return loss, _pack(gW, gS)

    if use_rec:
        x0 = _pack(shared['W_trans'], shared['S_trans'], shared['R_recur'])
    else:
        x0 = _pack(shared['W_trans'], shared['S_trans'])

    res = minimize(_objective, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 20, 'ftol': 1e-6})

    if use_rec:
        shared['W_trans'], shared['S_trans'], shared['R_recur'] = _unpack_wsr(res.x)
    else:
        shared['W_trans'], shared['S_trans'] = _unpack_ws(res.x)

    g0 = np.mean([er['gamma'][0] for er in estep_results], axis=0)
    g0 = g0.clip(min=1e-8)
    shared['log_pi'] = np.log(g0 / g0.sum())
    return shared


def fit_hierarchical_slds(sessions, cfg, seed=42, verbose=True):
    """Fit hierarchical SLDS: shared dynamics + transitions, session-specific emissions.

    Shared across sessions: A_dyn, b_dyn, Q_dyn, W_trans, S_trans (+ R_recur)
    Session-specific: C_emit, d_emit, R_emit, F_emit

    This solves the state non-identifiability problem: shared transitions ensure
    State k means the same thing across sessions.

    Args:
        sessions: list of (Y, U, obs_mask) tuples
        cfg: IOHMMConfig (K, D_latent, D_obs, D_input, etc.)

    Returns:
        result: dict with 'shared', 'sessions', 'll_trace', 'bic'
    """
    from joblib import Parallel, delayed
    N = len(sessions)
    K, D, m = cfg.K, cfg.D_latent, cfg.D_obs

    if verbose:
        print(f"  Hierarchical SLDS: {N} sessions, K={K}, D_latent={D}, m={m}")

    # ── Phase 1: Per-session SLDS init (parallel) ────────────────────
    if verbose:
        print("  Phase 1: Per-session initialization...")

    # Resource-discipline guard per audit P0.4: this used to fan out with
    # prefer='processes', which forks the full session list per worker. With
    # large D_obs and many sessions that is the worst case for memory under
    # the 60 GB cap. Switch to threading + per-worker BLAS pin so workers
    # share the session arrays in-place.
    from cadence.io.resources import limit_blas_threads, pick_n_jobs

    def _init_one(args):
        with limit_blas_threads(1):
            Y, U, mask, s_seed = args
            from cadence.significance.rslds_model import IOHMMConfig, fit_slds
            sp_cfg = IOHMMConfig(K=K, D_obs=m, D_input=cfg.D_input, D_latent=D,
                                  n_factors=cfg.n_factors, recurrent=cfg.recurrent,
                                  n_restarts=2, max_em_iter=60)
            params, hist = fit_slds(Y, U, mask, sp_cfg, seed=s_seed, verbose=False)
            return params, hist

    init_args = [(Y, U, mask, seed + i * 1000) for i, (Y, U, mask) in enumerate(sessions)]
    # Each SLDS init carries one session's Y/U/mask + an SLDS workspace; ~1 GB
    # peak per session at D_obs=26 (~0.3 GB at MVP D_obs=7, but stay conservative).
    init_n_jobs = pick_n_jobs(per_worker_ram_gb=1.0, requested=-1,
                                max_jobs_hard_cap=len(init_args))
    init_results = Parallel(n_jobs=init_n_jobs, prefer='threads')(
        delayed(_init_one)(a) for a in init_args)

    # ── Phase 2: Align states across sessions ────────────────────────
    if verbose:
        print("  Phase 2: Aligning states...")

    lls = [r[1]['final_ll'] for r in init_results]
    ref_idx = int(np.argmax(lls))
    ref_mu = init_results[ref_idx][1]['slds']['d_emit']
    session_mus = [r[1]['slds']['d_emit'] for r in init_results]
    perms = _align_states_to_reference(ref_mu, session_mus)

    aligned = []
    for i, (params, hist) in enumerate(init_results):
        d = dict(hist['slds'])
        d['gamma'] = hist['gamma']
        d['x_smooth'] = hist['x_smooth']
        d['log_pi'] = params.log_pi
        d['W_trans'] = params.W_trans
        d['S_trans'] = params.S_trans
        aligned.append(_apply_state_permutation(d, perms[i]))

    # ── Phase 3: Pool initial shared parameters ──────────────────────
    if verbose:
        print("  Phase 3: Pooling shared parameters...")

    shared = {
        'A_dyn': np.mean([s['A_dyn'] for s in aligned], axis=0),
        'b_dyn': np.mean([s['b_dyn'] for s in aligned], axis=0),
        'Q_dyn': np.mean([s['Q_dyn'] for s in aligned], axis=0),
        'W_trans': np.mean([s['W_trans'] for s in aligned], axis=0),
        'S_trans': np.mean([s['S_trans'] for s in aligned], axis=0),
        'log_pi': np.mean([s['log_pi'] for s in aligned], axis=0),
    }
    if cfg.recurrent:
        R_recs = [s['R_recur'] for s in aligned if s.get('R_recur') is not None]
        shared['R_recur'] = np.mean(R_recs, axis=0) if R_recs else np.zeros((K, K, D))

    sess_params = []
    for s in aligned:
        sp = {k: s.get(k, np.array([])).copy() if s.get(k) is not None else None
              for k in ('C_emit', 'd_emit', 'R_emit', 'F_emit')}
        sess_params.append(sp)

    # ── Phase 4: Hierarchical EM ─────────────────────────────────────
    if verbose:
        print("  Phase 4: Hierarchical EM...")

    iohmm_ref = IOHMM(cfg)
    slds_m_step_emissions._null_state = cfg.null_state
    slds_m_step_emissions._null_sigma2_cap = cfg.null_sigma2_cap
    x0_mean = np.zeros(D)
    P0 = cfg.kalman_init_P0 * np.eye(D)
    ll_trace = []
    d_inits = [sp['d_emit'].copy() for sp in sess_params]
    stage2_iter = max(cfg.anneal_iters, int(0.3 * cfg.max_em_iter))

    for it in range(cfg.max_em_iter):
        freeze_d = (it < stage2_iter)
        use_fa = (cfg.n_factors > 0 and it >= stage2_iter)

        # E-step: per-session (parallel — each is independent given shared params)
        # Limit BLAS threads to 1 so joblib threading provides parallelism
        # (OpenBLAS defaults to 24 threads, causing contention with joblib)
        import threadpoolctl
        def _estep_one(n):
            Y_n, U_n, mask_n = sessions[n]
            class SP:
                pass
            sp = SP()
            sp.A_dyn, sp.b_dyn, sp.Q_dyn = shared['A_dyn'], shared['b_dyn'], shared['Q_dyn']
            sp.C_emit = sess_params[n]['C_emit']
            sp.d_emit = sess_params[n]['d_emit']
            sp.R_emit = sess_params[n]['R_emit']
            sp.F_emit = sess_params[n].get('F_emit')
            sp.W_trans, sp.S_trans = shared['W_trans'], shared['S_trans']
            sp.log_pi = shared['log_pi']
            sp.R_recur = shared.get('R_recur')
            sp.x0_mean, sp.P0 = x0_mean, P0

            with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
                gamma, xi, ll, x_sm, P_sm, Plag_sm = slds_e_step(
                    Y_n, U_n, mask_n, sp, cfg, iohmm_ref)
            return {'gamma': gamma, 'xi': xi, 'x_smooth': x_sm,
                    'P_smooth': P_sm, 'Plag_smooth': Plag_sm}, ll

        # E-step is called per EM iteration; cap n_jobs by per-worker RAM
        # (per-session Kalman smoother + accumulators ~ 0.3 GB at MVP D_obs=7,
        # ~1 GB at V11 D_obs=26). The manual threadpool_limits guard above
        # already pins BLAS, so per_worker_ram_gb here only needs to cover
        # session data + workspace.
        estep_n_jobs = pick_n_jobs(per_worker_ram_gb=0.5, requested=-1,
                                    max_jobs_hard_cap=N)
        results = Parallel(n_jobs=estep_n_jobs, prefer='threads')(
            delayed(_estep_one)(n) for n in range(N))
        estep_results = [r[0] for r in results]
        total_ll = sum(r[1] for r in results)

        ll_trace.append(total_ll)
        if it > 0:
            rel = (total_ll - ll_trace[-2]) / max(abs(ll_trace[-2]), 1.0)
            if rel < cfg.em_tol:
                if verbose:
                    print(f"    Converged at iter {it}, LL={total_ll:.1f}")
                break

        # M-step shared: dynamics
        shared['A_dyn'], shared['b_dyn'], shared['Q_dyn'] = \
            _hierarchical_m_step_dynamics(
                [er['gamma'] for er in estep_results],
                [er['x_smooth'] for er in estep_results],
                [er['P_smooth'] for er in estep_results],
                [er['Plag_smooth'] for er in estep_results], K, D)

        # M-step shared: transitions
        shared = _hierarchical_m_step_transitions(
            sessions, estep_results, shared, cfg)

        # M-step session-specific: emissions
        n_fac = cfg.n_factors if use_fa else 0
        for n in range(N):
            Y_n = sessions[n][0]
            mask_n = sessions[n][2]
            C_new, d_new, R_new, F_new = slds_m_step_emissions(
                Y_n, mask_n, estep_results[n]['gamma'],
                estep_results[n]['x_smooth'], estep_results[n]['P_smooth'],
                K, D, m, n_factors=n_fac,
                c_shrinkage=cfg.c_shrinkage if use_fa else 0.0)
            sess_params[n]['C_emit'] = C_new
            sess_params[n]['R_emit'] = R_new
            if F_new is not None:
                sess_params[n]['F_emit'] = F_new
            if freeze_d:
                sess_params[n]['d_emit'] = d_inits[n]
            else:
                sess_params[n]['d_emit'] = d_new
            # Null state: always enforce d[0]=0
            if cfg.null_state:
                sess_params[n]['d_emit'][0, :] = 0.0

    if verbose:
        print(f"    Final LL={ll_trace[-1]:.1f}, iters={len(ll_trace)}")

    # BIC
    n_dyn = K * (D * D + D + D * (D + 1) // 2)
    n_trans = K * K * (1 + cfg.D_input) + K - 1
    if cfg.recurrent:
        n_trans += K * K * D
    n_shared = n_dyn + n_trans
    n_per_sess = K * (m * D + m + m)
    if cfg.n_factors > 0:
        n_per_sess += K * m * cfg.n_factors
    n_total = n_shared + N * n_per_sess
    total_T = sum(len(s[0]) for s in sessions)
    bic = -2 * ll_trace[-1] + n_total * np.log(total_T)

    if verbose:
        print(f"  Hierarchical BIC={bic:.1f} (shared={n_shared}, "
              f"per_sess={n_per_sess}, total={n_total})")

    return {
        'shared': shared,
        'sessions': [{
            **sess_params[n],
            'gamma': estep_results[n]['gamma'],
            'x_smooth': estep_results[n]['x_smooth'],
        } for n in range(N)],
        'll_trace': ll_trace,
        'bic': float(bic),
        'n_params': n_total,
        'final_ll': float(ll_trace[-1]),
    }


def iohmm_flexibility_metrics(gamma: np.ndarray, fs: float = 2.0) -> dict:
    """Compute flexibility metrics from IOHMM state posteriors.

    Args:
        gamma: (T, K) posterior state probabilities
        fs: sampling rate in Hz

    Returns:
        dict with per-state and aggregate metrics
    """
    T, K = gamma.shape
    dt = 1.0 / fs
    viterbi_path = np.argmax(gamma, axis=1)

    # State usage
    state_usage = gamma.mean(axis=0).tolist()

    # Transitions (from Viterbi path)
    transitions = np.sum(viterbi_path[1:] != viterbi_path[:-1])
    transition_rate = float(transitions) / (T * dt)

    # Dwell times per state
    dwell_times = {k: [] for k in range(K)}
    current_state = viterbi_path[0]
    current_dwell = 1
    for t in range(1, T):
        if viterbi_path[t] == current_state:
            current_dwell += 1
        else:
            dwell_times[current_state].append(current_dwell * dt)
            current_state = viterbi_path[t]
            current_dwell = 1
    dwell_times[current_state].append(current_dwell * dt)

    mean_dwell = {}
    for k in range(K):
        dwells = dwell_times[k]
        mean_dwell[k] = float(np.mean(dwells)) if dwells else 0.0

    # Shannon entropy of state usage
    usage = np.array(state_usage)
    usage = usage[usage > 0]
    entropy = float(-np.sum(usage * np.log(usage)))

    return {
        'state_usage': state_usage,
        'n_transitions': int(transitions),
        'transition_rate_hz': transition_rate,
        'mean_dwell_s': mean_dwell,
        'shannon_entropy': entropy,
        'K': K,
    }
