"""Sticky HDP-SLDS coupling event detector (V3.3).

2-state Sticky HMM with latent factor emissions for multivariate
coupling event detection.  Exploits cross-channel covariance during
coupling for ~3x effective SNR over per-channel methods.

Generative model (AR(1) conditional emissions):
    z_t | z_{t-1}  ~ Sticky-HMM(pi, kappa)           # K=2 discrete states
    x_t | x_{t-1}, z_t ~ N(A[z_t]*x_{t-1}, Q[z_t])   # d=1 latent AR(1)
    y_t | y_{t-1}, x_t, z_t:                           # C-dim observations
      ε_t = y_t - φ·y_{t-1}
      ε_t ~ StudentT((1-φ)·(μ[z_t] + Λ[z_t]·x_t), σ²[z_t]·(1-φ²), ν)
    where φ = exp(-1/(eval_rate·τ)) is the analytical AR(1) coefficient
    from EWLS exponential weighting.

The AR(1) conditional emission prevents over-counting autocorrelated
samples at native rate.  Innovation ε_t is white noise by construction.
t=0 uses marginal emission (no y_{t-1}).

State 0 = null (Lambda_0=0, A_0=0, Q_0=0.01 all fixed).
State 1 = coupled (Lambda_1 has ARD sparsity, dynamics learned).
mu_0 is fixed to surrogate null mean to break label symmetry.

- MCMC (Gibbs) for real data — full posterior on z, x, and all params.
- Coordinate-ascent VI (simplified 2-state HMM) for surrogates — kappa
  fixed at MCMC-learned value, no latent factor.

Performance: sequential loops (FFBS, Kalman) run on CPU numpy to avoid
CUDA kernel launch overhead on tiny K=2, d=1 tensors.  Vectorised
operations (emissions, parameter updates) run on GPU.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


_LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SLDSParams:
    """All model parameters, stored as tensors on device."""
    A_trans: torch.Tensor     # (2, 2)  transition matrix
    kappa: float              # stickiness
    mu: torch.Tensor          # (2, C)  emission means
    sigma2: torch.Tensor      # (2, C)  emission variances
    Lambda: torch.Tensor      # (2, C)  factor loadings (Lambda[0]=0 fixed)
    tau_ard: torch.Tensor     # (C,)    ARD precision on Lambda[1]
    A_dyn: torch.Tensor       # (2,)    latent AR(1) coeff (A_dyn[0]=0 fixed)
    Q_dyn: torch.Tensor       # (2,)    latent noise var  (Q_dyn[0]=0.01 fixed)


# ---------------------------------------------------------------------------
# Fast CPU forward-backward (numpy)
# ---------------------------------------------------------------------------

def _ffbs_numpy(log_emit, log_A, rng):
    """Forward-filtering backward-sampling on CPU.

    Args:
        log_emit: (T, 2) numpy float64 — log emission probabilities.
        log_A:    (2, 2) numpy float64 — log transition matrix.
        rng:      numpy random Generator.

    Returns:
        z: (T,) int8 numpy array — sampled state sequence.
    """
    T = log_emit.shape[0]
    log_pi = np.array([np.log(0.95), np.log(0.05)])

    # Forward filter (vectorized over 2-state dimension)
    log_alpha = np.empty((T, 2))
    log_alpha[0] = log_pi + log_emit[0]
    log_alpha[0] -= np.logaddexp(log_alpha[0, 0], log_alpha[0, 1])

    for t in range(1, T):
        log_alpha[t] = np.logaddexp(
            log_alpha[t - 1, 0] + log_A[0],
            log_alpha[t - 1, 1] + log_A[1]) + log_emit[t]
        lse = np.logaddexp(log_alpha[t, 0], log_alpha[t, 1])
        log_alpha[t] -= lse

    # Backward sampling
    z = np.empty(T, dtype=np.int8)
    p0 = np.exp(log_alpha[T - 1, 0])
    z[T - 1] = 0 if rng.random() < p0 / (p0 + np.exp(log_alpha[T - 1, 1])) else 1

    for t in range(T - 2, -1, -1):
        lp0 = log_alpha[t, 0] + log_A[0, z[t + 1]]
        lp1 = log_alpha[t, 1] + log_A[1, z[t + 1]]
        lse = np.logaddexp(lp0, lp1)
        z[t] = 0 if rng.random() < np.exp(lp0 - lse) else 1

    return z


def _kalman_rts_numpy(A_t, Q_t, info_prec, info_mean, rng):
    """d=1 Kalman filter + RTS smoother + sample, on CPU.

    All inputs are (T,) numpy arrays (precomputed from GPU).

    Returns:
        x: (T,) numpy float64 — sampled latent trajectory.
    """
    T = len(A_t)

    x_filt = np.empty(T)
    P_filt = np.empty(T)

    # t=0: prior x_0 ~ N(0, 1)
    P_pred = Q_t[0] + 1.0
    P_filt[0] = 1.0 / (1.0 / P_pred + info_prec[0])
    x_filt[0] = P_filt[0] * info_mean[0]

    for t in range(1, T):
        x_p = A_t[t] * x_filt[t - 1]
        P_p = A_t[t] ** 2 * P_filt[t - 1] + Q_t[t]
        P_filt[t] = 1.0 / (1.0 / P_p + info_prec[t])
        x_filt[t] = P_filt[t] * (x_p / P_p + info_mean[t])

    # RTS backward smoother
    x_sm = np.empty(T)
    P_sm = np.empty(T)
    x_sm[T - 1] = x_filt[T - 1]
    P_sm[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_p = A_t[t + 1] ** 2 * P_filt[t] + Q_t[t + 1]
        G = A_t[t + 1] * P_filt[t] / max(P_p, 1e-10)
        x_sm[t] = x_filt[t] + G * (x_sm[t + 1] - A_t[t + 1] * x_filt[t])
        P_sm[t] = P_filt[t] + G ** 2 * (P_sm[t + 1] - P_p)

    # Sample
    return x_sm + np.sqrt(np.maximum(P_sm, 1e-10)) * rng.standard_normal(T)


def _fwd_bwd_numpy(log_emit, log_A, log_pi):
    """Batched forward-backward on CPU for VI surrogates.

    Args:
        log_emit: (S, T, 2) numpy.
        log_A:    (2, 2) numpy.
        log_pi:   (2,) numpy.

    Returns:
        gamma: (S, T, 2) posterior probabilities.
    """
    S, T, _ = log_emit.shape
    log_alpha = np.empty((S, T, 2))

    # Forward (vectorised over S)
    log_alpha[:, 0] = log_pi + log_emit[:, 0]
    lse = np.logaddexp(log_alpha[:, 0, 0], log_alpha[:, 0, 1])
    log_alpha[:, 0] -= lse[:, None]

    for t in range(1, T):
        # alpha[s, j] = logaddexp_i(alpha[s, i] + A[i, j]) + emit[s, j]
        log_alpha[:, t, 0] = np.logaddexp(
            log_alpha[:, t - 1, 0] + log_A[0, 0],
            log_alpha[:, t - 1, 1] + log_A[1, 0]) + log_emit[:, t, 0]
        log_alpha[:, t, 1] = np.logaddexp(
            log_alpha[:, t - 1, 0] + log_A[0, 1],
            log_alpha[:, t - 1, 1] + log_A[1, 1]) + log_emit[:, t, 1]
        lse = np.logaddexp(log_alpha[:, t, 0], log_alpha[:, t, 1])
        log_alpha[:, t] -= lse[:, None]

    # Backward (vectorised over S)
    log_beta = np.zeros((S, T, 2))
    for t in range(T - 2, -1, -1):
        inner = log_emit[:, t + 1] + log_beta[:, t + 1]  # (S, 2)
        log_beta[:, t, 0] = np.logaddexp(
            log_A[0, 0] + inner[:, 0],
            log_A[0, 1] + inner[:, 1])
        log_beta[:, t, 1] = np.logaddexp(
            log_A[1, 0] + inner[:, 0],
            log_A[1, 1] + inner[:, 1])
        lse = np.logaddexp(log_beta[:, t, 0], log_beta[:, t, 1])
        finite = np.isfinite(lse)
        if finite.any():
            log_beta[:, t][finite] -= lse[finite, None]

    # Posterior
    log_gamma = log_alpha + log_beta
    lse = np.logaddexp(log_gamma[:, :, 0], log_gamma[:, :, 1])
    log_gamma -= lse[:, :, None]
    return np.exp(log_gamma)


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------

class GibbsSampler:
    """Full MCMC Gibbs sampler for the SLDS generative model.

    Uses AR(1) conditional emissions to handle autocorrelated dR2
    at native sample rate without over-counting evidence.

    Sequential FFBS and Kalman loops run on CPU (numpy) to avoid CUDA
    kernel launch overhead.  Emissions and parameter updates run on GPU.
    """

    def __init__(self, Y, null_mu, priors, mcmc_cfg, device,
                 eval_rate=1.0, tau_seconds=0.0):
        """
        Args:
            Y: (T, C) observation tensor on device.
            null_mu: (C,) surrogate null mean (numpy).
            priors: dict of prior hyperparameters.
            mcmc_cfg: dict with n_burnin, n_samples, thin.
            device: torch device.
            eval_rate: Hz — sample rate of observations.
            tau_seconds: EWLS decay time — determines AR(1) phi
                analytically as exp(-1/(eval_rate*tau)).
        """
        self.Y = Y
        self.T, self.C = Y.shape
        self.device = device
        self.priors = priors
        self.n_burnin = mcmc_cfg.get('n_burnin', 200)
        self.n_samples = mcmc_cfg.get('n_samples', 300)
        self.thin = mcmc_cfg.get('thin', 3)

        # State variables (GPU tensors for parameter updates)
        self.z = torch.zeros(self.T, dtype=torch.long, device=device)
        self.x = torch.zeros(self.T, dtype=torch.float32, device=device)

        # Posterior tracking
        self.kappa_samples = []
        self.params: Optional[SLDSParams] = None
        self._rng = np.random.default_rng(
            mcmc_cfg.get('seed', None))

        # AR(1) innovation coefficient — analytical from EWLS tau.
        # phi = exp(-1/(rate*tau)):  high rate or long tau → phi≈1 (strong AR).
        if tau_seconds > 0 and eval_rate > 0:
            self._ar_phi = math.exp(-1.0 / (eval_rate * tau_seconds))
        else:
            self._ar_phi = 0.0

    def initialize(self, null_mu, null_sigma, warm_start_posterior=None):
        """Data-driven parameter initialization with warm-start z."""
        C, dev = self.C, self.device
        Y_np = self.Y.cpu().numpy()

        mu_0 = torch.tensor(null_mu, dtype=torch.float32, device=dev)
        upper = np.percentile(Y_np, 75, axis=0)
        mu_1 = torch.tensor(
            np.maximum(upper, null_mu + null_sigma),
            dtype=torch.float32, device=dev)

        sigma2_init = torch.tensor(
            np.maximum(null_sigma, 1e-8) ** 2,
            dtype=torch.float32, device=dev)

        # Scale latent dynamics to match data variance.
        # Q_dyn[1] = data_var ensures x has the same scale as y,
        # so Lambda_1 ≈ O(1) and ARD doesn't over-shrink.
        data_var = max(float(np.var(Y_np)), 1e-8)
        data_std = float(np.sqrt(data_var))

        self.params = SLDSParams(
            A_trans=torch.tensor([[0.99, 0.01], [0.05, 0.95]],
                                  dtype=torch.float32, device=dev),
            kappa=1.0,
            mu=torch.stack([mu_0, mu_1]),
            sigma2=torch.stack([sigma2_init, sigma2_init.clone()]),
            Lambda=torch.zeros(2, C, dtype=torch.float32, device=dev),
            tau_ard=torch.ones(C, dtype=torch.float32, device=dev),
            A_dyn=torch.tensor([0.0, 0.9], dtype=torch.float32, device=dev),
            Q_dyn=torch.tensor([1e-6, data_var], dtype=torch.float32,
                               device=dev),
        )
        self.params.Lambda[1] = torch.randn(C, device=dev) * data_std

        # Warm-start z from upstream z-score posterior (preferred)
        # or per-channel z-score fallback.
        if warm_start_posterior is not None and len(warm_start_posterior) == self.T:
            z_init = (warm_start_posterior > 0.3).astype(np.int64)
        else:
            from scipy.ndimage import uniform_filter1d
            z_per_ch = ((Y_np - null_mu[None, :])
                        / null_sigma[None, :])
            smooth_w = max(1, min(5, self.T // 20))
            z_per_ch = np.array([
                uniform_filter1d(z_per_ch[:, c], size=smooth_w)
                for c in range(C)])
            n_above = (z_per_ch > 1.0).sum(axis=0)
            z_init = (n_above >= 2).astype(np.int64)
            if z_init.sum() < 0.02 * self.T:
                top_pct = np.percentile(n_above, 90)
                if top_pct >= 2:
                    z_init = (n_above >= top_pct).astype(np.int64)
        # Fill small gaps
        if z_init.sum() > 0 and z_init.sum() < 0.8 * self.T:
            from scipy.ndimage import binary_closing
            z_init = binary_closing(
                z_init, structure=np.ones(3)).astype(np.int64)
        self.z = torch.tensor(z_init, dtype=torch.long, device=dev)

    def run(self):
        """Execute the full Gibbs sampler.

        Returns:
            z_samples: (n_posterior, T) int8 numpy array.
        """
        n_total = self.n_burnin + self.n_samples
        z_samples = []
        kappa_cap_burnin = 50.0

        # Anchored + tempered burn-in only when warm-start found signal.
        # Without signal, standard Gibbs from all-null prevents false
        # positives on noise data.
        warm_frac = float((self.z == 1).float().mean())
        has_signal = warm_frac > 0.01
        n_anchored = max(20, self.n_burnin // 5) if has_signal else 0
        n_tempered = max(10, self.n_burnin // 10) if has_signal else 0

        for i in range(n_total):
            if i < n_anchored:
                # Anchored: keep z fixed, update x + emission params.
                self._step_x()
                self._sample_emissions()
                self._sample_loadings()
                self._sample_ard()
            else:
                # Full Gibbs (gently tempered transitions during early FFBS)
                if i < n_anchored + n_tempered:
                    saved_A = self.params.A_trans.clone()
                    hot_A = torch.tensor([[0.85, 0.15], [0.15, 0.85]],
                                          device=self.device)
                    self.params.A_trans = hot_A
                    self._step_z()
                    self.params.A_trans = saved_A
                else:
                    self._step_z()
                self._step_x()
                self._sample_emissions()
                self._sample_loadings()
                self._sample_ard()
                self._sample_transitions()
                self._sample_dynamics()

            # Cap kappa during first half of burn-in
            if i < self.n_burnin // 2:
                self.params.kappa = min(self.params.kappa, kappa_cap_burnin)

            if i >= self.n_burnin and (i - self.n_burnin) % self.thin == 0:
                z_samples.append(self.z.cpu().to(torch.int8).numpy().copy())
                self.kappa_samples.append(float(self.params.kappa))

        return np.array(z_samples)

    # ---- z: FFBS (GPU emissions → CPU forward-backward) ---- #

    def _step_z(self):
        """FFBS for z | x, y, params.

        AR(1) conditional Student-t emission:
          ε_t = y_t - φ·y_{t-1}
          cond_mean = (1-φ)·(μ_k + Λ_k·x_t)
          ε_t ~ StudentT(cond_mean, σ²_k·(1-φ²), ν)
        t=0 uses marginal emission (no y_{t-1}).
        """
        p = self.params
        nu = 5.0  # Student-t degrees of freedom
        phi = self._ar_phi

        # Innovation: y_t - φ·y_{t-1}
        Y_prev = torch.zeros_like(self.Y)
        Y_prev[1:] = self.Y[:-1]
        innovation = self.Y - phi * Y_prev  # (T, C)

        # AR(1) conditional mean: (1-φ)·(μ_k + Λ_k·x_t)   shape (2, T, C)
        cond_mean = ((1 - phi)
                     * (p.mu.unsqueeze(1)
                        + p.Lambda.unsqueeze(1) * self.x.view(1, -1, 1)))

        # Innovation variance: σ²_k·(1-φ²)   shape (2, 1, C)
        sigma2_innov = (p.sigma2 * (1 - phi ** 2)).unsqueeze(1).clamp(min=1e-30)

        res = innovation.unsqueeze(0) - cond_mean  # (2, T, C)
        # Student-t log-pdf summed over channels
        log_e = (-(nu + 1) / 2 * torch.log(1 + res ** 2 / (nu * sigma2_innov))
                 - 0.5 * torch.log(sigma2_innov)).sum(2)  # (2, T)

        # t=0: marginal emission (standard, no AR conditioning)
        if phi > 0:
            res_0 = (self.Y[0]
                     - p.mu
                     - p.Lambda * self.x[0])  # (2, C)
            s2_0 = p.sigma2.clamp(min=1e-30)  # (2, C)
            log_e_0 = (-(nu + 1) / 2
                       * torch.log(1 + res_0 ** 2 / (nu * s2_0))
                       - 0.5 * torch.log(s2_0)).sum(1)  # (2,)
            log_e[:, 0] = log_e_0

        log_emit_np = log_e.T.cpu().numpy()  # (T, 2)

        log_A_np = np.log(np.maximum(
            p.A_trans.cpu().numpy(), 1e-30))

        # Sequential FFBS on CPU
        z_np = _ffbs_numpy(log_emit_np, log_A_np, self._rng)
        self.z = torch.tensor(z_np.astype(np.int64),
                               dtype=torch.long, device=self.device)

    # ---- x: Kalman RTS (GPU info precomp → CPU filter) ---- #

    def _step_x(self):
        """Kalman RTS for x | z, y, params.

        Innovation-based observation info:
          eff_obs = y_t - φ·y_{t-1} - (1-φ)·μ_{z_t}
          Effective loading = (1-φ)·Λ_{z_t}
          Noise variance = σ²_{z_t}·(1-φ²)
        t=0 uses marginal observation info.
        """
        p = self.params
        phi = self._ar_phi

        Lam_t = p.Lambda[self.z]       # (T, C)
        sig2_t = p.sigma2[self.z]      # (T, C)
        mu_t = p.mu[self.z]            # (T, C)

        # Innovation-based observation info
        Y_prev = torch.zeros_like(self.Y)
        Y_prev[1:] = self.Y[:-1]
        eff_obs = self.Y - phi * Y_prev - (1 - phi) * mu_t  # (T, C)

        sig2_innov = (sig2_t * (1 - phi ** 2)).clamp(min=1e-30)  # (T, C)
        inv_sig2_innov = 1.0 / sig2_innov

        Lam_eff = (1 - phi) * Lam_t  # (T, C)
        ip = (Lam_eff ** 2 * inv_sig2_innov).sum(1)  # (T,)
        im = (Lam_eff * eff_obs * inv_sig2_innov).sum(1)  # (T,)

        # t=0: marginal observation info (standard loading, no AR)
        if phi > 0:
            inv_sig2_0 = 1.0 / sig2_t[0].clamp(min=1e-30)  # (C,)
            ip[0] = (Lam_t[0] ** 2 * inv_sig2_0).sum()
            im[0] = (Lam_t[0] * (self.Y[0] - mu_t[0]) * inv_sig2_0).sum()

        A_t = p.A_dyn[self.z].cpu().numpy()                       # (T,)
        Q_t = p.Q_dyn[self.z].cpu().numpy()                       # (T,)

        # Sequential Kalman + RTS + sample on CPU
        x_np = _kalman_rts_numpy(A_t, Q_t,
                                 ip.cpu().numpy(), im.cpu().numpy(),
                                 self._rng)
        self.x = torch.tensor(x_np, dtype=torch.float32,
                               device=self.device)

    # ---- Parameter sampling (GPU, vectorised over C) ---- #

    def _sample_emissions(self):
        """Conjugate NIG for mu_1 and sigma2 from innovations.

        mu_0 stays fixed.  Uses innovation residuals (t >= 1) to
        account for AR(1) correlation structure.
        """
        p, pr = self.params, self.priors
        phi = self._ar_phi
        one_m_phi = 1 - phi
        one_m_phi_sq = 1 - phi ** 2

        # Innovations for t >= 1
        innov = self.Y[1:] - phi * self.Y[:-1]  # (T-1, C)
        z_from1 = self.z[1:]
        x_from1 = self.x[1:]

        for k in range(2):
            mask = (z_from1 == k)
            n_k = mask.sum().float()
            if n_k < 2:
                continue

            # Residual after removing factor: innov - (1-φ)·Λ_k·x
            innov_k = innov[mask]  # (n_k, C)
            x_k = x_from1[mask]   # (n_k,)
            res = innov_k - one_m_phi * p.Lambda[k] * x_k.unsqueeze(1)

            if k == 1:
                # Posterior for mu_1
                # Model: res = (1-φ)·μ_k + noise,  noise_var = σ²·(1-φ²)
                inv_innov_var = 1.0 / (p.sigma2[1] * one_m_phi_sq).clamp(
                    min=1e-30)
                prec_lik = n_k * one_m_phi ** 2 * inv_innov_var
                prec_post = 1.0 + prec_lik
                mean_post = (one_m_phi * res.sum(0) * inv_innov_var
                             / prec_post)
                p.mu[1] = (mean_post
                           + torch.randn(self.C, device=self.device)
                           / prec_post.sqrt())

            # Full residual for sigma2 posterior
            res_mu = res - one_m_phi * p.mu[k]  # (n_k, C)
            ss = (res_mu ** 2).sum(0) / max(one_m_phi_sq, 1e-10)
            alpha_post = float(pr['sigma2_alpha'] + n_k * 0.5)
            beta_post = (pr['sigma2_beta'] + 0.5 * ss).clamp(min=1e-10)
            g = torch.distributions.Gamma(
                torch.full((self.C,), alpha_post, device=self.device),
                beta_post).sample()
            p.sigma2[k] = (1.0 / g).clamp(min=1e-8)

    def _sample_loadings(self):
        """Normal posterior for Lambda_{1,c} with ARD, from innovations.

        Model: innov - (1-φ)·μ_1 = (1-φ)·Λ_1·x + noise
        """
        p = self.params
        phi = self._ar_phi
        one_m_phi = 1 - phi
        one_m_phi_sq = 1 - phi ** 2

        mask = (self.z[1:] == 1)
        if mask.sum() < 2:
            return

        # Innovations (t >= 1) assigned to state 1
        innov = self.Y[1:] - phi * self.Y[:-1]  # (T-1, C)
        x1 = self.x[1:][mask]            # (n_k,)
        o = innov[mask] - one_m_phi * p.mu[1]  # (n_k, C) effective obs

        sum_x2 = (x1 ** 2).sum()
        inv_innov_var = 1.0 / (p.sigma2[1] * one_m_phi_sq).clamp(min=1e-30)

        # Bayesian linear regression: o_t = (1-φ)·Λ·x_t + noise
        prec_post = (p.tau_ard
                     + sum_x2 * one_m_phi ** 2 * inv_innov_var)
        sum_xo = (x1.unsqueeze(1) * o).sum(0)  # (C,)
        mean_post = (one_m_phi * sum_xo * inv_innov_var
                     / prec_post.clamp(min=1e-10))

        p.Lambda[1] = (mean_post
                       + torch.randn(self.C, device=self.device)
                       / prec_post.clamp(min=1e-10).sqrt())
        p.Lambda[0] = 0.0

    def _sample_ard(self):
        """Gamma posterior for ARD precision tau_c."""
        p, pr = self.params, self.priors
        alpha = float(pr['ard_a'] + 0.5)
        beta = (pr['ard_b'] + 0.5 * p.Lambda[1] ** 2).clamp(min=1e-10)
        p.tau_ard = torch.distributions.Gamma(
            torch.full((self.C,), alpha, device=self.device),
            beta).sample()

    def _sample_transitions(self):
        """Sticky Dirichlet for A_trans + kappa."""
        p, pr = self.params, self.priors

        z0, z1 = self.z[:-1], self.z[1:]
        # Vectorized transition count: encode pair as 2*i+j, then bincount
        pair_idx = 2 * z0 + z1  # (T-1,) values in {0,1,2,3}
        counts = torch.bincount(pair_idx, minlength=4).float()
        n_trans = counts.reshape(2, 2)

        for i in range(2):
            conc = n_trans[i].clone() + 1.0
            conc[i] += p.kappa
            g = torch.distributions.Gamma(
                conc.clamp(min=1e-10),
                torch.ones(2, device=self.device)).sample()
            p.A_trans[i] = g / g.sum()

        n_self = float(n_trans[0, 0] + n_trans[1, 1])
        n_total = float(self.T - 1)
        a_post = pr['kappa_a'] + n_self
        b_post = max(pr['kappa_b'] + (n_total - n_self), 1e-10)
        p.kappa = float(self._rng.gamma(a_post, 1.0 / b_post))

    def _sample_dynamics(self):
        """Conjugate for A_dyn[1], Q_dyn[1].  State 0 kept fixed."""
        p, pr = self.params, self.priors

        mask_t = (self.z[1:] == 1)
        n = mask_t.sum()
        if n < 3:
            return

        xp = self.x[:-1][mask_t]
        xc = self.x[1:][mask_t]

        prec_pri = 1.0 / (pr['A_dyn_std'] ** 2)
        prec_lik = (xp ** 2).sum() / p.Q_dyn[1].clamp(min=1e-10)
        prec_post = prec_pri + prec_lik
        mean_post = ((pr['A_dyn_mean'] * prec_pri
                      + (xp * xc).sum() / p.Q_dyn[1].clamp(min=1e-10))
                     / prec_post.clamp(min=1e-10))
        A1 = (mean_post
              + torch.randn(1, device=self.device)
              / prec_post.clamp(min=1e-10).sqrt())
        p.A_dyn = p.A_dyn.clone()
        p.A_dyn[1] = A1.squeeze().clamp(-0.999, 0.999)

        res = xc - p.A_dyn[1] * xp
        alpha_post = float(pr['sigma2_alpha'] + float(n) * 0.5)
        beta_post = (pr['sigma2_beta'] + 0.5 * (res ** 2).sum()).clamp(
            min=1e-10)
        g = torch.distributions.Gamma(
            torch.tensor(alpha_post, device=self.device),
            beta_post).sample()
        p.Q_dyn = p.Q_dyn.clone()
        p.Q_dyn[1] = (1.0 / g).clamp(min=1e-6)
        p.A_dyn[0] = 0.0
        p.Q_dyn[0] = 0.01


# ---------------------------------------------------------------------------
# VI solver  (simplified 2-state Gaussian HMM for surrogates)
# ---------------------------------------------------------------------------

class VISolver:
    """Coordinate-ascent EM for surrogate null distribution.

    Simplified: no latent factor, kappa fixed at MCMC-learned value.
    AR(1) conditional Gaussian emissions match the MCMC model.
    Forward-backward on CPU numpy (vectorised over S surrogates).
    """

    def __init__(self, fixed_kappa, null_mu, null_sigma, vi_cfg, device,
                 tau_seconds=0.0):
        self.fixed_kappa = max(fixed_kappa, 0.01)
        self.null_mu = null_mu.astype(np.float64)
        self.null_sigma = np.maximum(null_sigma, 1e-8).astype(np.float64)
        self.max_iter = vi_cfg.get('max_iter', 50)
        self.tol = vi_cfg.get('tol', 1e-4)
        self.device = device
        self.tau_seconds = tau_seconds

    def fit_surrogates(self, dr2_surr, eval_rate):
        """Return per-surrogate coupling fractions.

        Args:
            dr2_surr: (K, C, T) numpy.
            eval_rate: Hz of the surrogate data.

        Returns:
            fractions: (K,) numpy array.
        """
        K, C, T = dr2_surr.shape

        # AR(1) coefficient for surrogate rate
        if self.tau_seconds > 0 and eval_rate > 0:
            phi = math.exp(-1.0 / (eval_rate * self.tau_seconds))
        else:
            phi = 0.0
        phi_sq = phi ** 2
        one_m_phi = 1.0 - phi
        one_m_phi_sq = 1.0 - phi_sq

        # Build shared transition matrix with fixed kappa
        base_p = 0.02
        kf = self.fixed_kappa / (1.0 + self.fixed_kappa)
        A = np.array([
            [1.0 - base_p + kf * base_p,  base_p - kf * base_p],
            [base_p - kf * base_p,  1.0 - base_p + kf * base_p],
        ])
        A /= A.sum(1, keepdims=True)
        log_A = np.log(np.maximum(A, 1e-30))
        log_pi = np.array([np.log(0.95), np.log(0.05)])

        # Y: (K, T, C)
        Y = dr2_surr.transpose(0, 2, 1).astype(np.float64)
        null_mu = self.null_mu
        null_sig = self.null_sigma

        # Precompute innovations (t >= 1)
        innov = Y[:, 1:] - phi * Y[:, :-1]  # (K, T-1, C)

        # Per-surrogate emission params
        mu = np.zeros((K, 2, C))
        mu[:, 0] = null_mu
        mu[:, 1] = null_mu + null_sig
        sigma2 = np.zeros((K, 2, C))
        sigma2[:, 0] = null_sig ** 2
        sigma2[:, 1] = null_sig ** 2

        prev_frac = None

        for _it in range(self.max_iter):
            # E-step: AR(1) conditional emissions
            log_emit = np.empty((K, T, 2))
            for k in range(2):
                s2_k = np.maximum(sigma2[:, k], 1e-10)   # (K, C)

                # t=0: marginal emission
                diff_0 = Y[:, 0] - mu[:, k]  # (K, C)
                log_emit[:, 0, k] = -0.5 * (
                    np.log(s2_k).sum(1)
                    + (diff_0 ** 2 / s2_k).sum(1))

                # t >= 1: AR(1) conditional emission
                cond_mean = one_m_phi * mu[:, k][:, np.newaxis, :]  # (K, 1, C)
                diff = innov - cond_mean  # (K, T-1, C)
                s2_innov = np.maximum(s2_k * one_m_phi_sq, 1e-10)  # (K, C)
                log_emit[:, 1:, k] = -0.5 * (
                    np.log(s2_innov).sum(1, keepdims=True)          # (K, 1)
                    + (diff ** 2 / s2_innov[:, np.newaxis, :]).sum(2))  # (K, T-1)

            # Forward-backward on CPU
            gamma = _fwd_bwd_numpy(log_emit, log_A, log_pi)  # (K, T, 2)

            # Convergence (approx via mean gamma change)
            frac = gamma[:, :, 1].mean(1)  # (K,)
            if prev_frac is not None and np.abs(frac - prev_frac).max() < self.tol:
                break
            prev_frac = frac.copy()

            # M-step (using innovations for t >= 1)
            for k in range(2):
                g_innov = gamma[:, 1:, k]  # (K, T-1)
                w_innov = np.maximum(
                    g_innov.sum(1, keepdims=True), 1e-10)  # (K, 1)

                if k == 1:
                    # mu_1 from innovations: innov ~ (1-φ)·μ + noise
                    mu[:, 1] = ((g_innov[:, :, np.newaxis] * innov).sum(1)
                                / (w_innov * max(one_m_phi, 1e-10)))
                    mu[:, 1] = np.maximum(
                        mu[:, 1], mu[:, 0] + 0.01 * null_sig)

                # sigma2 from innovation residuals
                diff_innov = (innov
                              - one_m_phi * mu[:, k][:, np.newaxis, :])
                sigma2[:, k] = np.maximum(
                    (g_innov[:, :, np.newaxis] * diff_innov ** 2).sum(1)
                    / (w_innov * max(one_m_phi_sq, 1e-10)),
                    1e-8)

        return gamma[:, :, 1].mean(1)  # (K,)


# ---------------------------------------------------------------------------
# Main detector orchestrator
# ---------------------------------------------------------------------------

class SLDSDetector:
    """Sticky HDP-SLDS coupling event detector.

    Runs at native sample rate using AR(1) conditional emissions to
    handle autocorrelation from EWLS smoothing (no downsampling needed).

    Usage:
        detector = SLDSDetector(slds_config, device='cuda')
        result = detector.fit(dr2_perchannel, dr2_surr_perchannel,
                              eval_rate, tau_seconds=8.0)
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

    def fit(self, dr2_perchannel, dr2_surr_perchannel, eval_rate,
            warm_start_posterior=None, tau_seconds=0.0,
            surr_eval_rate=None):
        """Full SLDS detection pipeline: MCMC + VI surrogates.

        Operates at native eval_rate — AR(1) emissions prevent
        over-counting from EWLS-induced autocorrelation.

        Args:
            dr2_perchannel:      (C, T) per-channel dR2 (numpy).
            dr2_surr_perchannel: (K, C, T) surrogate per-channel dR2 (numpy).
            eval_rate:           Hz of the real data.
            warm_start_posterior: (T,) optional z-score posterior from
                upstream pipeline. Used as warm-start for z if provided.
            tau_seconds:         EWLS decay time for AR(1) phi computation.
            surr_eval_rate:      Hz of surrogate data (if different from
                eval_rate). Defaults to eval_rate.

        Returns:
            dict with coupling_posterior (T,), metadata.
        """
        C, T = dr2_perchannel.shape
        surr_fit = dr2_surr_perchannel
        if surr_eval_rate is None:
            surr_eval_rate = eval_rate

        # --- Null statistics from surrogates (NaN-safe) ---
        C_obs = C
        C_surr = surr_fit.shape[1]
        null_mu_surr = np.nanmean(surr_fit, axis=(0, 2))    # (C_surr,)
        null_sig_surr = np.maximum(
            np.nanstd(surr_fit, axis=(0, 2)), 1e-8)         # (C_surr,)

        # Tile to match observation channels (multi-tau groups)
        n_groups = max(1, C_obs // C_surr) if C_surr > 0 else 1
        null_mu = np.tile(null_mu_surr, n_groups)[:C_obs]
        null_sigma = np.tile(null_sig_surr, n_groups)[:C_obs]

        # Fill NaN with per-channel null mean
        y_fit = dr2_perchannel.copy()
        for c in range(C_obs):
            mask_y = ~np.isfinite(y_fit[c])
            if mask_y.any():
                y_fit[c, mask_y] = null_mu[c]
        for c in range(C_surr):
            mask_s = ~np.isfinite(surr_fit[:, c, :])
            if mask_s.any():
                surr_fit[:, c, :] = np.where(
                    np.isfinite(surr_fit[:, c, :]),
                    surr_fit[:, c, :], null_mu_surr[c])

        # --- Phase 1: MCMC on real data (native rate) ---
        Y = torch.tensor(y_fit.T, dtype=torch.float32,
                          device=self.device)

        # Scale priors to data magnitude
        data_var = max(float(np.var(y_fit)), 1e-10)
        priors = dict(self.config.get('priors', {}))
        priors.setdefault('sigma2_alpha', 2.0)
        priors.setdefault('sigma2_beta', data_var * 2)
        priors.setdefault('ard_a', 1.0)
        priors.setdefault('ard_b', data_var)
        priors.setdefault('kappa_a', 1.0)
        priors.setdefault('kappa_b', 1.0)
        priors.setdefault('A_dyn_mean', 0.9)
        priors.setdefault('A_dyn_std', 0.1)

        # Warm-start posterior — already at native rate (no downsampling)
        ws_post = None
        if warm_start_posterior is not None:
            if len(warm_start_posterior) == T:
                ws_post = warm_start_posterior
            else:
                ws_post = np.interp(
                    np.linspace(0, 1, T),
                    np.linspace(0, 1, len(warm_start_posterior)),
                    warm_start_posterior)

        mcmc_cfg = self.config.get('mcmc', {})
        gibbs = GibbsSampler(Y, null_mu, priors, mcmc_cfg, self.device,
                             eval_rate=eval_rate, tau_seconds=tau_seconds)
        gibbs.initialize(null_mu, null_sigma, warm_start_posterior=ws_post)
        z_samples = gibbs.run()

        coupling_posterior = z_samples.astype(np.float32).mean(axis=0)
        learned_kappa = (float(np.median(gibbs.kappa_samples))
                         if gibbs.kappa_samples else 1.0)

        # Active channels from final loadings
        Lam1 = gibbs.params.Lambda[1].cpu().numpy()
        lam_scale = max(float(np.std(Lam1)), 1e-8)
        active_mask = np.abs(Lam1) > 0.1 * lam_scale
        n_active = int(np.sum(active_mask))

        # --- Phase 2: VI on surrogates ---
        vi_cfg = self.config.get('vi', {})
        vi = VISolver(learned_kappa, null_mu_surr, null_sig_surr, vi_cfg,
                      self.device, tau_seconds=tau_seconds)
        surr_fractions = vi.fit_surrogates(surr_fit, surr_eval_rate)

        # --- Detection criteria ---
        coupling_fraction = float(np.mean(coupling_posterior))
        det_cfg = self.config.get('detection', {})
        min_frac = det_cfg.get('min_coupling_fraction', 0.03)
        min_ch = det_cfg.get('min_active_channels', 2)
        surr_95 = (float(np.percentile(surr_fractions, 95))
                   if len(surr_fractions) > 0 else 0.0)
        detected = (coupling_fraction > max(surr_95, min_frac)
                    and n_active >= min_ch)

        return {
            'coupling_posterior': coupling_posterior,
            'coupling_fraction': coupling_fraction,
            'kappa': learned_kappa,
            'n_active_channels': n_active,
            'active_channels': active_mask,
            'surr_coupling_fractions': surr_fractions,
            'surr_95': surr_95,
            'detected': detected,
            'method': 'slds_mcmc',
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def detect_coupling_slds(dr2_perchannel, dr2_surr_perchannel, eval_rate,
                         slds_cfg, device='cuda', tau_seconds=0.0):
    """Thin wrapper for standalone SLDS detection.

    Returns:
        detected: bool.
        details:  dict with coupling_posterior, metadata.
    """
    detector = SLDSDetector(slds_cfg, device=device)
    result = detector.fit(dr2_perchannel, dr2_surr_perchannel, eval_rate,
                          tau_seconds=tau_seconds)
    return result['detected'], result
