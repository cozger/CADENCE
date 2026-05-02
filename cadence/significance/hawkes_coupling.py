"""Hawkes-process coupling: NMF expression discovery + MMHP temporal localization.

Pipeline:
  1. Joint NMF on raw [0,1] blendshapes → shared expression dictionary
  2. Event detection in NMF component activations
  3. Mutual Hawkes fit for each cross-person pathway (LLR screening)
  4. MMHP (Markov-Modulated Hawkes Process) on significant pathways
     → coupling episodes, per-event attribution, triggering kernels

References:
  - Wu, Ward, Curley, Zheng (2022). Markov-modulated Hawkes processes
    for sporadic and bursty event occurrences. Ann. Appl. Stat. 16(2).
  - Xu, Farajtabar, Zha (2016). Learning Granger Causality for Hawkes
    Processes. ICML.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2
from sklearn.decomposition import NMF
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ── MediaPipe blendshape names (52 coefficients) ─────────────────────

MP_BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


# ── Result dataclasses ────────────────────────────────────────────────

@dataclass
class CouplingEpisode:
    """A detected coupling episode (active state interval)."""
    start_s: float
    end_s: float
    duration_s: float
    n_source_events: int    # P1 events during episode
    n_target_events: int    # P2 events during episode
    n_triggered: int        # P2 events attributed to P1 triggering


@dataclass
class HawkesPathway:
    """Results for one cross-person triggering pathway."""
    source_component: int
    target_component: int
    source_name: str
    target_name: str
    # Hawkes parameters
    mu: float               # target baseline rate (events/s)
    alpha: float            # excitation magnitude
    beta: float             # excitation decay rate (1/s)
    peak_lag_s: float       # 1/beta — mode of exponential kernel
    # Significance
    log_lik: float          # MMHP log-likelihood
    log_lik_null: float     # Poisson-only (no excitation)
    llr: float              # log-likelihood ratio
    p_value: float          # from chi2(2) on 2*LLR
    significant: bool       # after FDR correction
    # MMHP temporal localization
    episodes: List[CouplingEpisode] = field(default_factory=list)
    coupling_fraction: float = 0.0
    n_triggered: int = 0
    n_spontaneous: int = 0
    # State trajectory
    state_times: Optional[np.ndarray] = None   # (N_p2,) event times
    state_labels: Optional[np.ndarray] = None  # (N_p2,) 0=inactive, 1=active


@dataclass
class HawkesCouplingResult:
    """Full NMF + Hawkes coupling analysis."""
    # NMF
    n_components: int
    component_names: List[str]
    H: np.ndarray                    # (k, 52) AU loadings
    explained_variance: float
    # Events
    events_p1: List[np.ndarray]      # k arrays of P1 event times
    events_p2: List[np.ndarray]      # k arrays of P2 event times
    # Pathways (all tested)
    pathways: List[HawkesPathway]
    # Summary
    n_significant: int
    duration_s: float


# ── 1. NMF Expression Discovery ──────────────────────────────────────

def nmf_expression_discovery(p1_raw, p2_raw, n_components=6, seed=42):
    """Joint NMF on raw [0,1] blendshapes → shared expression dictionary.

    Concatenates both participants, fits NMF once, splits activations.
    The shared dictionary H ensures cross-person comparability.

    Args:
        p1_raw, p2_raw: (T, 52) raw blendshapes in [0, 1].
        n_components: number of expression types to discover.
        seed: random state.

    Returns:
        H: (k, 52) shared expression dictionary.
        W_p1, W_p2: (T, k) per-person component activations.
        component_names: auto-generated from top-3 AUs.
        explained: fraction of variance explained.
    """
    T = min(p1_raw.shape[0], p2_raw.shape[0])
    n_aus = min(52, p1_raw.shape[1], p2_raw.shape[1])
    p1 = np.maximum(p1_raw[:T, :n_aus], 0)
    p2 = np.maximum(p2_raw[:T, :n_aus], 0)

    X = np.vstack([p1, p2])

    nmf = NMF(n_components=n_components, init='nndsvda',
              max_iter=500, random_state=seed)
    W = nmf.fit_transform(X)
    H = nmf.components_

    W_p1 = W[:T]
    W_p2 = W[T:]

    names = []
    for i in range(n_components):
        top3 = np.argsort(H[i])[::-1][:3]
        name = '+'.join(MP_BLENDSHAPE_NAMES[j] for j in top3 if j < len(MP_BLENDSHAPE_NAMES))
        names.append(name)

    recon_err = nmf.reconstruction_err_
    total_norm = np.linalg.norm(X, 'fro')
    explained = 1.0 - (recon_err / total_norm)

    return H, W_p1, W_p2, names, explained


def detect_component_events(W, fs=30.0, min_iei_s=2.0, smooth_s=0.3):
    """Detect activation events in each NMF component timecourse.

    Uses adaptive prominence (0.5 × IQR of activation).

    Args:
        W: (T, k) component activations.
        fs: sampling rate.
        min_iei_s: minimum inter-event interval.
        smooth_s: Gaussian smoothing sigma in seconds.

    Returns:
        events: list of k arrays, each containing event times in seconds.
    """
    T, k = W.shape
    events = []

    for comp in range(k):
        w = gaussian_filter1d(W[:, comp], sigma=smooth_s * fs) if smooth_s > 0 else W[:, comp]
        iqr = np.percentile(w, 75) - np.percentile(w, 25)
        prom = max(iqr * 0.5, 0.005)

        pks, _ = find_peaks(w, prominence=prom, distance=int(min_iei_s * fs))
        events.append(pks / fs)

    return events


# ── 2. Mutual Hawkes Process ─────────────────────────────────────────

def _hawkes_R(t, source_times, beta):
    """Hawkes recursive variable: sum of exponential kernels from source events.

    R(t) = Σ_{s_k < t} exp(-β(t - s_k))
    """
    mask = source_times < t
    if not mask.any():
        return 0.0
    return np.sum(np.exp(-beta * (t - source_times[mask])))


def _hawkes_R_vectorized(target_times, source_times, beta):
    """Vectorized R computation for all target event times.

    Returns (N_target,) array where R[n] = Σ_{s_k < t_n} exp(-β(t_n - s_k)).
    """
    N = len(target_times)
    R = np.zeros(N)
    if len(source_times) == 0:
        return R

    for n in range(N):
        t = target_times[n]
        mask = source_times < t
        if mask.any():
            R[n] = np.exp(-beta * (t - source_times[mask])).sum()
    return R


def _hawkes_log_likelihood(params, target_times, source_times, T):
    """Log-likelihood of mutual Hawkes process.

    Target intensity: λ(t) = μ + α × Σ_{s_k < t} exp(-β(t - s_k))

    log L = Σ_n log λ(t_n) - ∫_0^T λ(t) dt

    The integral: ∫λ dt = μT + (α/β) Σ_k (1 - exp(-β(T - s_k)))
    """
    mu, alpha, beta = params

    if mu <= 0 or alpha < 0 or beta <= 0:
        return -1e15

    R = _hawkes_R_vectorized(target_times, source_times, beta)
    intensities = mu + alpha * R

    # Guard against log(0)
    intensities = np.maximum(intensities, 1e-15)
    ll_events = np.sum(np.log(intensities))

    # Integrated intensity
    integral = mu * T
    if len(source_times) > 0:
        integral += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - source_times)))

    return ll_events - integral


def _poisson_log_likelihood(target_times, T):
    """Log-likelihood of homogeneous Poisson (null model)."""
    N = len(target_times)
    if N == 0:
        return 0.0
    rate = N / T
    return N * np.log(max(rate, 1e-15)) - rate * T


def fit_hawkes_pathway(source_times, target_times, T,
                       beta_init=0.4):
    """Fit mutual Hawkes: source → target.

    Args:
        source_times: (N1,) event times of the source.
        target_times: (N2,) event times of the target.
        T: total observation duration.
        beta_init: initial decay rate (0.4 → peak at 2.5s).

    Returns:
        mu, alpha, beta: fitted parameters.
        log_lik: log-likelihood at optimum.
        log_lik_null: Poisson log-likelihood (no excitation).
        llr: log-likelihood ratio.
        p_value: from chi2(2) on 2*LLR.
    """
    N2 = len(target_times)
    if N2 < 3:
        ll_null = _poisson_log_likelihood(target_times, T)
        return 0.0, 0.0, beta_init, ll_null, ll_null, 0.0, 1.0

    mu_init = N2 / T

    def neg_ll(params):
        return -_hawkes_log_likelihood(params, target_times, source_times, T)

    result = minimize(
        neg_ll,
        x0=[mu_init, 0.1, beta_init],
        bounds=[(1e-4, 2.0), (1e-4, 5.0), (0.05, 5.0)],
        method='L-BFGS-B',
    )

    mu, alpha, beta = result.x
    log_lik = -result.fun
    log_lik_null = _poisson_log_likelihood(target_times, T)

    llr = max(0, log_lik - log_lik_null)
    p_value = chi2.sf(2 * llr, df=2)  # 2 extra params (alpha, beta)

    return mu, alpha, beta, log_lik, log_lik_null, llr, p_value


# ── 2a. Sliding-window Hawkes for time-varying coupling ──────────────
#
# A single (μ, α, β) per session yields only a deterministic convolution
# of source events with a fixed kernel — λ(t) varies through events but
# the *coupling strength* α is constant. To get a genuinely time-varying
# coupling timecourse for an rSLDS observation channel, fit α(t) per
# sliding window with β fixed globally (β is hard to fit on small N).

def _fit_alpha_mu_given_beta(target_times, source_times, T, beta):
    """Fit (μ, α) with β fixed. Used inside sliding-window fits."""
    if len(target_times) < 3:
        return float(len(target_times) / max(T, 1e-6)), 0.0, _poisson_log_likelihood(target_times, T)
    mu_init = max(len(target_times) / max(T, 1e-6), 1e-3)

    def neg_ll(params):
        mu, alpha = params
        if mu <= 0 or alpha < 0:
            return 1e15
        R = _hawkes_R_vectorized(target_times, source_times, beta)
        intensities = np.maximum(mu + alpha * R, 1e-15)
        ll_events = np.sum(np.log(intensities))
        integral = mu * T
        if len(source_times) > 0:
            integral += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - source_times)))
        return -(ll_events - integral)

    res = minimize(
        neg_ll, x0=[mu_init, 0.1],
        bounds=[(1e-4, 5.0), (0.0, 5.0)],
        method='L-BFGS-B',
    )
    mu, alpha = res.x
    return float(mu), float(alpha), float(-res.fun)


def fit_hawkes_sliding(
    source_times, target_times, T_total,
    win_s=60.0, hop_s=15.0, beta_global=None,
    min_events_per_window=3,
):
    """Sliding-window time-varying Hawkes coupling α(t).

    β is fit globally first (or supplied), then fixed per window.

    Args:
        source_times, target_times: (Ns,), (Nt,) event times in seconds.
        T_total: full session duration in seconds.
        win_s: window length.
        hop_s: window hop.
        beta_global: if None, fit β on full session via fit_hawkes_pathway.
        min_events_per_window: NaN α(t) if fewer target events in window.

    Returns:
        dict:
            t_centers: (N_win,) window center times.
            mu_t: (N_win,) baseline rate per window.
            alpha_t: (N_win,) coupling strength per window (NaN if too few events).
            beta: scalar global decay.
            n_events_per_window: (N_win,) number of target events in window.
    """
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)

    if beta_global is None:
        if len(target_times) >= 3 and len(source_times) >= 1:
            _, _, beta_global, _, _, _, _ = fit_hawkes_pathway(
                source_times, target_times, T_total)
        else:
            beta_global = 0.4
    beta_global = float(beta_global)

    centers = np.arange(win_s / 2.0, T_total - win_s / 2.0 + 1e-6, hop_s)
    if len(centers) == 0:
        centers = np.array([T_total / 2.0])

    mu_t = np.full(len(centers), np.nan, dtype=np.float64)
    alpha_t = np.full(len(centers), np.nan, dtype=np.float64)
    n_events = np.zeros(len(centers), dtype=np.int64)

    for i, c in enumerate(centers):
        t0 = max(0.0, c - win_s / 2.0)
        t1 = min(T_total, c + win_s / 2.0)
        # Source events: include those within the window AND short prefix
        # so the kernel from earlier events isn't truncated. ~3/β prefix.
        prefix = 3.0 / max(beta_global, 1e-3)
        s_in = source_times[(source_times >= t0 - prefix) & (source_times <= t1)]
        # Target events: within window
        tgt_in = target_times[(target_times >= t0) & (target_times <= t1)]
        n_events[i] = len(tgt_in)
        if len(tgt_in) < min_events_per_window:
            continue

        # Shift to window-local coords for stable optimization
        T_win = t1 - t0 + prefix
        mu, alpha, _ = _fit_alpha_mu_given_beta(
            tgt_in - (t0 - prefix), s_in - (t0 - prefix),
            T_win, beta_global,
        )
        mu_t[i] = mu
        alpha_t[i] = alpha

    return dict(
        t_centers=centers, mu_t=mu_t, alpha_t=alpha_t,
        beta=beta_global, n_events_per_window=n_events,
    )


def hawkes_intensity_timecourse(
    source_times, t_grid, mu_t=None, alpha_t=None, beta=0.4,
    t_centers=None, mu_const=None, alpha_const=None,
):
    """Continuous intensity λ(t) = μ(t) + α(t) Σ_{s_k < t} exp(-β(t - s_k)).

    Either supply per-window (mu_t, alpha_t, t_centers) which are linearly
    interpolated to t_grid, or supply scalar (mu_const, alpha_const).

    Args:
        source_times: (Ns,) event times in seconds.
        t_grid: (T,) time points to evaluate at.
        mu_t, alpha_t, t_centers: per-window fits from fit_hawkes_sliding.
        beta: decay rate (scalar).
        mu_const, alpha_const: scalar baseline / coupling for stationary mode.

    Returns:
        lam: (T,) continuous intensity.
    """
    source_times = np.asarray(source_times, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    if mu_t is not None and alpha_t is not None and t_centers is not None:
        mu_t = np.asarray(mu_t)
        alpha_t = np.asarray(alpha_t)
        t_centers = np.asarray(t_centers)
        # Interpolate, treating NaNs as linear extension from valid neighbors
        valid = np.isfinite(mu_t) & np.isfinite(alpha_t)
        if valid.sum() == 0:
            mu_grid = np.zeros_like(t_grid)
            alpha_grid = np.zeros_like(t_grid)
        else:
            mu_grid = np.interp(t_grid, t_centers[valid], mu_t[valid])
            alpha_grid = np.interp(t_grid, t_centers[valid], alpha_t[valid])
    elif mu_const is not None and alpha_const is not None:
        mu_grid = np.full_like(t_grid, mu_const, dtype=np.float64)
        alpha_grid = np.full_like(t_grid, alpha_const, dtype=np.float64)
    else:
        raise ValueError("Supply either (mu_t, alpha_t, t_centers) or (mu_const, alpha_const)")

    # R(t) = Σ exp(-β(t - s_k)) for s_k < t — vectorize via outer
    if len(source_times) == 0:
        return mu_grid

    # (T, Ns) lag matrix; mask future events. Clip dt before exp to avoid
    # overflow for large negative dt (would not contribute anyway after mask).
    dt = t_grid[:, None] - source_times[None, :]
    valid_kernel = dt > 0
    dt_clipped = np.where(valid_kernel, dt, 0.0)
    R = np.where(valid_kernel, np.exp(-beta * dt_clipped), 0.0).sum(axis=1)
    return mu_grid + alpha_grid * R


def hawkes_sliding_surrogate_z(
    source_times, target_times, T_total,
    real_alpha_t, t_centers, beta_global,
    n_surrogates=200, win_s=60.0, hop_s=15.0,
    min_events_per_window=3, seed=42,
):
    """Surrogate z-score for α(t) via circular shifts of source events.

    Welford accumulator over n_surrogates refits.

    Returns:
        z_t: (N_win,) per-window z-score (NaN where real α is NaN).
    """
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    real_alpha = np.asarray(real_alpha_t, dtype=np.float64)
    n_win = len(t_centers)

    if T_total <= 0 or n_win == 0:
        return np.full(n_win, np.nan)

    rng = np.random.default_rng(seed)
    min_shift = max(5.0, T_total * 0.05)
    max_shift = T_total - min_shift
    if max_shift <= min_shift:
        max_shift = min_shift + 1.0

    mean = np.zeros(n_win)
    m2 = np.zeros(n_win)
    counts = np.zeros(n_win, dtype=np.int64)

    for si in range(n_surrogates):
        shift = float(rng.uniform(min_shift, max_shift))
        s_shift = np.mod(source_times + shift, T_total)
        s_shift.sort()
        surr = fit_hawkes_sliding(
            s_shift, target_times, T_total,
            win_s=win_s, hop_s=hop_s, beta_global=beta_global,
            min_events_per_window=min_events_per_window,
        )
        a = surr['alpha_t']
        valid = np.isfinite(a)
        if valid.any():
            counts[valid] += 1
            n_i = counts[valid]
            d = a[valid] - mean[valid]
            mean[valid] = mean[valid] + d / n_i
            m2[valid] = m2[valid] + d * (a[valid] - mean[valid])

    var = np.where(counts > 1, m2 / np.maximum(counts - 1, 1), 0.0)
    std = np.sqrt(var)
    std = np.where(std > 1e-10, std, 1e-10)
    z = (real_alpha - mean) / std
    z = np.where(np.isfinite(real_alpha), z, np.nan)
    z = np.clip(z, -10, 10)
    return z


# ── 2b. Group-Sparse Multivariate Hawkes (Xu et al. 2016) ────────────
#
# Fits ALL pathways jointly with group-L2 penalty.  Each pathway i→j
# has K basis coefficients.  The group penalty drives entire pathways
# to exact zero — replaces FDR with sparsity-based selection.

def _gaussian_basis(tau, centers, bandwidth):
    """Unnormalized Gaussian RBF basis: exp(-(τ-c)²/(2σ²)).

    Peak = 1.0 so coefficients directly represent intensity contribution.
    Returns (..., K).
    """
    tau = np.asarray(tau, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    return np.exp(-((tau[..., None] - c) ** 2) / (2 * bandwidth ** 2))


def _gaussian_basis_integral(dt, centers, bandwidth):
    """∫_0^dt κ_m(τ) dτ for unnormalized Gaussian = σ√(2π) [Φ((dt-c)/σ) - Φ(-c/σ)]."""
    from scipy.special import erf
    c = np.asarray(centers)
    bw = bandwidth
    s2 = np.sqrt(2)
    phi_hi = 0.5 * (1 + erf((dt - c) / (bw * s2)))
    phi_lo = 0.5 * (1 + erf(-c / (bw * s2)))
    return bw * np.sqrt(2 * np.pi) * (phi_hi - phi_lo)


def _precompute_basis_sums(events_list, centers, bandwidth):
    """B[n, i, m] = Σ_{k: type=i, s_k < t_n} κ_m(t_n - s_k).

    Returns dict[j] → (N_j, D, K) and event counts.
    """
    D = len(events_list)
    K = len(centers)
    B_per_type = {}
    for j in range(D):
        t_j = events_list[j]
        N_j = len(t_j)
        if N_j == 0:
            B_per_type[j] = np.zeros((0, D, K))
            continue
        B = np.zeros((N_j, D, K))
        for i in range(D):
            s_i = events_list[i]
            if len(s_i) == 0:
                continue
            for n in range(N_j):
                mask = s_i < t_j[n]
                if mask.any():
                    lags = t_j[n] - s_i[mask]
                    B[n, i] = _gaussian_basis(lags, centers, bandwidth).sum(axis=0)
        B_per_type[j] = B
    return B_per_type, np.array([len(e) for e in events_list])


def _precompute_integrated_kernels(events_list, T, centers, bandwidth):
    """C[i, m] = Σ_{k: type=i} G_m(T - s_k)."""
    D = len(events_list)
    K = len(centers)
    C = np.zeros((D, K))
    for i in range(D):
        for sk in events_list[i]:
            dt = T - sk
            if dt > 0:
                C[i] += _gaussian_basis_integral(dt, centers, bandwidth)
    return C


def _fit_hawkes_gs_inner(events_list, T, D, K, centers, bandwidth,
                         B_per_type, C_int, event_counts,
                         lambda_group=0.05, lambda_sparse=0.0,
                         max_iter=50, eta=1e-4):
    """EM algorithm for group-sparse Hawkes (Xu et al. 2016, Algorithm 1).

    E-step: compute responsibilities (branching probabilities).
    M-step: closed-form updates for μ and A.
    Shrinkage: sparse-group-lasso on A after each M-step.
    """
    mu = np.array([max(event_counts[j] / T, 1e-4) for j in range(D)])
    A = np.full((D, K, D), 0.01)  # small positive init

    for iteration in range(max_iter):
        # ── E-step: compute responsibilities ─────────────────────

        # p_baseline[j][n] = μ_j / λ_j(t_n)
        # p_trigger[j][n, i, m] = A[j,m,i] * κ_m(τ) / λ_j(t_n)  (summed over source events)
        # We only need aggregated responsibilities for M-step:
        #   sum_p_baseline[j] = Σ_n p_baseline_n  (for μ update)
        #   sum_p_trigger[j, m, i] = Σ_n Σ_{k:type=i} p^m_{nk}  (for A update)

        sum_p_base = np.zeros(D)  # Σ_n p_ii for each target type
        sum_p_trigger = np.zeros((D, K, D))  # sum_p_trigger[j,m,i]
        ll = 0.0

        for j in range(D):
            N_j = int(event_counts[j])
            if N_j == 0:
                continue
            B_j = B_per_type[j]  # (N_j, D, K)

            # λ_j(t_n) = μ_j + Σ_{i,m} A[j,m,i] × B[n,i,m]
            excitation = np.einsum('mi,nim->n', A[j], B_j)  # (N_j,)
            lam = mu[j] + excitation
            lam = np.maximum(lam, 1e-10)
            ll += np.log(lam).sum()

            inv_lam = 1.0 / lam  # (N_j,)

            # Baseline responsibility: Σ_n μ_j / λ_j(t_n)
            sum_p_base[j] = mu[j] * inv_lam.sum()

            # Trigger responsibilities: Σ_n A[j,m,i] * B[n,i,m] / λ_j(t_n)
            for i in range(D):
                for m in range(K):
                    if A[j, m, i] < 1e-15:
                        continue
                    # p^m for source type i = A[j,m,i] * B[n,i,m] / λ(t_n)
                    sum_p_trigger[j, m, i] = A[j, m, i] * (
                        B_j[:, i, m] * inv_lam).sum()

        # Compensator (integrated intensity)
        ll -= mu.sum() * T
        for i in range(D):
            for j in range(D):
                ll -= A[j, :, i] @ C_int[i]

        # ── M-step: update μ and A ──────────────────────────────

        # μ update (Eq 5): μ_u = Σ p_baseline / T
        for j in range(D):
            if event_counts[j] > 0:
                mu[j] = max(sum_p_base[j] / T, 1e-6)

        # A update (Eq 6): quadratic formula per element
        # a^m_{uu'} = (-B + √(B²-4AC)) / (2A)
        # where C_coef = -sum_p_trigger[j,m,i] (negative responsibility sum)
        #       B_coef = C_int[i,m] + lambda_sparse
        #       A_coef = lambda_group / ||a_{ji}||_2

        A_new = np.zeros_like(A)
        for j in range(D):
            for i in range(D):
                a_ji = A[j, :, i]  # (K,) current
                norm_a = np.linalg.norm(a_ji)
                A_coef = lambda_group / max(norm_a, 1e-10)

                for m in range(K):
                    B_coef = C_int[i, m] + lambda_sparse
                    C_coef = -sum_p_trigger[j, m, i]

                    # Quadratic: A_coef * x^2 + B_coef * x + C_coef = 0
                    discriminant = B_coef ** 2 - 4 * A_coef * C_coef
                    if discriminant > 0 and C_coef < 0:
                        A_new[j, m, i] = (-B_coef + np.sqrt(discriminant)) / (2 * A_coef)
                    else:
                        A_new[j, m, i] = 0

        # ── Sparse-group-lasso shrinkage (Eqs 7-8) ──────────────

        for j in range(D):
            for i in range(D):
                v = np.maximum(A_new[j, :, i], 0)
                # Element-wise soft-threshold (L1)
                if lambda_sparse > 0:
                    v = np.sign(v) * np.maximum(np.abs(v) - eta * lambda_sparse, 0)
                # Group test + shrinkage (L2)
                nv = np.linalg.norm(v)
                if nv <= eta * lambda_group:
                    A_new[j, :, i] = 0
                elif nv > 0:
                    A_new[j, :, i] = (1 - eta * lambda_group / nv) * v
                else:
                    A_new[j, :, i] = 0

        A = A_new

    return mu, A, ll


def fit_hawkes_group_sparse(events_list, T, n_basis=5,
                            lag_range=(0.5, 8.0), bandwidth=1.0,
                            lambda_group=None, max_iter=50,
                            bic_select=True):
    """Fit multivariate Hawkes with group-sparse triggering kernels.

    Single joint model across all event types. Group-L2 penalty drives
    inactive pathways to zero — no FDR needed.

    Args:
        events_list: list of D arrays of event times.
        T: observation duration (seconds).
        n_basis: Gaussian RBF basis functions.
        lag_range: (min, max) seconds for basis centers.
        bandwidth: basis bandwidth (seconds).
        lambda_group: penalty strength.  None = BIC selection.
        max_iter: optimization iterations.
        lr: learning rate.
        bic_select: sweep lambda and select by BIC.

    Returns:
        mu: (D,) baseline rates.
        A: (D, K, D) coefficients.  A[j,:,i] = kernel for i→j.
        centers: (K,) basis centers.
        active_pathways: list of (source, target) tuples.
        log_lik: final log-likelihood.
        info: dict with lambda, n_active, etc.
    """
    D = len(events_list)
    K = n_basis
    centers = np.linspace(lag_range[0], lag_range[1], K)

    B_per_type, counts = _precompute_basis_sums(events_list, centers, bandwidth)
    C_int = _precompute_integrated_kernels(events_list, T, centers, bandwidth)
    N_total = int(counts.sum())

    if bic_select and lambda_group is None:
        # Lambda grid: with unnormalized basis (peak=1), coefficients
        # are in the same units as event rate (~0.01-0.5 events/s).
        # Lambda should range from small (all pathways active) to large
        # (all zeroed out).
        lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        best_bic, best = np.inf, None
        for lam in lambdas:
            mu, A, ll = _fit_hawkes_gs_inner(
                events_list, T, D, K, centers, bandwidth,
                B_per_type, C_int, counts,
                lambda_group=lam, max_iter=max_iter)
            n_act = sum(1 for i in range(D) for j in range(D)
                        if np.linalg.norm(A[j, :, i]) > 1e-6)
            bic = -2 * ll + (D + n_act * K) * np.log(max(N_total, 1))
            if bic < best_bic:
                best_bic = bic
                best = (mu.copy(), A.copy(), ll, lam, n_act)
        mu, A, log_lik, best_lam, _ = best
    else:
        lam = lambda_group if lambda_group is not None else 0.05
        mu, A, log_lik = _fit_hawkes_gs_inner(
            events_list, T, D, K, centers, bandwidth,
            B_per_type, C_int, counts,
            lambda_group=lam, max_iter=max_iter)
        best_lam = lam

    active = [(i, j) for i in range(D) for j in range(D)
              if np.linalg.norm(A[j, :, i]) > 1e-6]

    info = {'lambda_group': best_lam, 'n_active': len(active),
            'N_total': N_total, 'centers': centers, 'bandwidth': bandwidth}
    return mu, A, centers, active, log_lik, info


def hawkes_gs_kernel(A, centers, bandwidth, src, tgt, tau_grid=None):
    """Reconstruct triggering kernel φ_{tgt←src}(τ)."""
    if tau_grid is None:
        tau_grid = np.linspace(0, 10, 200)
    a = A[tgt, :, src]
    basis = _gaussian_basis(tau_grid, centers, bandwidth)
    return tau_grid, basis @ a


# ── 3. MMHP: Markov-Modulated Hawkes Process ─────────────────────────

def _ctmc_transition_probs(q1, q2, dt):
    """2×2 CTMC transition matrix over interval dt.

    States: 0 = inactive, 1 = active.
    q1: active → inactive rate.
    q2: inactive → active rate.

    Returns (2, 2) matrix P where P[i,j] = P(state j at t+dt | state i at t).
    """
    total = q1 + q2
    if total < 1e-10 or dt <= 0:
        return np.eye(2)
    e = np.exp(-total * dt)
    p00 = q1 / total + q2 / total * e   # inactive → inactive
    p11 = q2 / total + q1 / total * e   # active → active
    return np.array([[p00, 1 - p00],
                     [1 - p11, p11]])


def _mmhp_integrated_intensity(state, a, b, mu0, mu1, alpha, beta,
                               source_times):
    """Integrated intensity ∫_a^b λ_s(t) dt for a given state.

    State 0: Λ = μ₀ × (b - a)
    State 1: Λ = μ₁ × (b - a) + (α/β) × Σ_{s_k < b} [exp(-β·max(0,a-s_k)) - exp(-β(b-s_k))]
    """
    dt = b - a
    if dt <= 0:
        return 0.0

    if state == 0:
        return mu0 * dt

    # State 1: baseline + excitation integral
    base = mu1 * dt
    if len(source_times) == 0:
        return base

    mask = source_times < b
    if not mask.any():
        return base

    s = source_times[mask]
    # For each source event s_k:
    #   ∫_a^b α·exp(-β(t-s_k)) dt = (α/β) [exp(-β·max(0,a-s_k)) - exp(-β(b-s_k))]
    lower = np.maximum(a - s, 0)
    upper = b - s
    excitation = (alpha / beta) * np.sum(np.exp(-beta * lower) - np.exp(-beta * upper))

    return base + excitation


def _mmhp_forward(target_times, source_times, T, params):
    """Forward algorithm for MMHP.

    Piecewise-constant state approximation: state is constant within
    each inter-event interval. Exact when no CTMC transition occurs
    within an interval — a good approximation for sparse events.

    Args:
        target_times: (N2,) P2 event times.
        source_times: (N1,) P1 event times.
        T: total duration.
        params: (mu0, mu1, alpha, beta, q1, q2).

    Returns:
        log_lik: total log-likelihood.
        forward: (N2, 2) log forward variables.
    """
    mu0, mu1, alpha, beta, q1, q2 = params
    N = len(target_times)

    if N == 0:
        return 0.0, np.zeros((0, 2))

    forward = np.full((N, 2), -np.inf)

    # Stationary distribution of CTMC
    total_q = q1 + q2
    if total_q < 1e-10:
        pi = np.array([0.5, 0.5])
    else:
        pi = np.array([q1 / total_q, q2 / total_q])

    log_pi = np.log(np.maximum(pi, 1e-15))

    # Precompute R values (excitation from source at each target event)
    R = _hawkes_R_vectorized(target_times, source_times, beta)

    # First event
    t0 = target_times[0]
    for s in range(2):
        # Intensity at t0
        if s == 0:
            lam = mu0
        else:
            lam = mu1 + alpha * R[0]
        log_lam = np.log(max(lam, 1e-15))

        # Integrated intensity from 0 to t0
        Lambda = _mmhp_integrated_intensity(s, 0, t0, mu0, mu1, alpha, beta,
                                            source_times)

        forward[0, s] = log_pi[s] + log_lam - Lambda

    # Subsequent events
    for n in range(1, N):
        dt = target_times[n] - target_times[n - 1]
        P = _ctmc_transition_probs(q1, q2, dt)
        log_P = np.log(np.maximum(P, 1e-15))

        for s in range(2):
            # Intensity at t_n
            if s == 0:
                lam = mu0
            else:
                lam = mu1 + alpha * R[n]
            log_lam = np.log(max(lam, 1e-15))

            # Integrated intensity over [t_{n-1}, t_n]
            Lambda = _mmhp_integrated_intensity(
                s, target_times[n - 1], target_times[n],
                mu0, mu1, alpha, beta, source_times)

            # log-sum-exp over previous states
            log_trans = forward[n - 1] + log_P[:, s]
            forward[n, s] = _logsumexp(log_trans) + log_lam - Lambda

    # Final: account for [t_N, T] interval (no event)
    final = np.copy(forward[-1])
    for s in range(2):
        Lambda = _mmhp_integrated_intensity(
            s, target_times[-1], T, mu0, mu1, alpha, beta, source_times)
        final[s] -= Lambda

    log_lik = _logsumexp(final)
    return log_lik, forward


def _mmhp_viterbi(target_times, source_times, T, params):
    """Viterbi decoding for MAP state sequence.

    Returns:
        states: (N2,) array, 0=inactive, 1=active.
        log_prob: log-probability of MAP path.
    """
    mu0, mu1, alpha, beta, q1, q2 = params
    N = len(target_times)

    if N == 0:
        return np.array([], dtype=int), 0.0

    # Same structure as forward but with max instead of logsumexp
    viterbi = np.full((N, 2), -np.inf)
    backptr = np.zeros((N, 2), dtype=int)

    total_q = q1 + q2
    pi = np.array([q1, q2]) / max(total_q, 1e-10) if total_q > 1e-10 else np.array([0.5, 0.5])
    log_pi = np.log(np.maximum(pi, 1e-15))

    R = _hawkes_R_vectorized(target_times, source_times, beta)

    # First event
    t0 = target_times[0]
    for s in range(2):
        lam = mu0 if s == 0 else mu1 + alpha * R[0]
        log_lam = np.log(max(lam, 1e-15))
        Lambda = _mmhp_integrated_intensity(s, 0, t0, mu0, mu1, alpha, beta,
                                            source_times)
        viterbi[0, s] = log_pi[s] + log_lam - Lambda

    # Subsequent events
    for n in range(1, N):
        dt = target_times[n] - target_times[n - 1]
        P = _ctmc_transition_probs(q1, q2, dt)
        log_P = np.log(np.maximum(P, 1e-15))

        for s in range(2):
            lam = mu0 if s == 0 else mu1 + alpha * R[n]
            log_lam = np.log(max(lam, 1e-15))
            Lambda = _mmhp_integrated_intensity(
                s, target_times[n - 1], target_times[n],
                mu0, mu1, alpha, beta, source_times)

            scores = viterbi[n - 1] + log_P[:, s]
            best_prev = np.argmax(scores)
            viterbi[n, s] = scores[best_prev] + log_lam - Lambda
            backptr[n, s] = best_prev

    # Backtrace
    states = np.zeros(N, dtype=int)
    states[-1] = np.argmax(viterbi[-1])
    for n in range(N - 2, -1, -1):
        states[n] = backptr[n + 1, states[n + 1]]

    return states, float(np.max(viterbi[-1]))


def fit_mmhp(source_times, target_times, T,
             hawkes_mu=None, hawkes_alpha=None, hawkes_beta=None):
    """Fit MMHP via MAP estimation (scipy.minimize).

    Uses Hawkes fit as initialization for mu1, alpha, beta.
    Optimizes all 6 parameters: {mu0, mu1, alpha, beta, q1, q2}.

    Args:
        source_times, target_times: event time arrays.
        T: observation duration.
        hawkes_mu/alpha/beta: pre-fitted Hawkes params (optional init).

    Returns:
        params: (mu0, mu1, alpha, beta, q1, q2) fitted.
        log_lik: MMHP log-likelihood.
        states: (N2,) Viterbi state sequence.
        episodes: list of CouplingEpisode.
    """
    N2 = len(target_times)
    if N2 < 5:
        # Not enough events for MMHP
        params = (N2 / T, N2 / T, 0.0, 0.4, 0.1, 0.1)
        return params, -np.inf, np.zeros(N2, dtype=int), []

    base_rate = N2 / T

    # Initialize from Hawkes fit or defaults
    mu0_init = base_rate * 0.5
    mu1_init = hawkes_mu if hawkes_mu else base_rate * 0.5
    alpha_init = hawkes_alpha if hawkes_alpha else 0.1
    beta_init = hawkes_beta if hawkes_beta else 0.4
    q1_init = 0.05   # active episodes ~20s
    q2_init = 0.03   # ~33s between episodes

    x0 = [mu0_init, mu1_init, alpha_init, beta_init, q1_init, q2_init]

    bounds = [
        (1e-4, 2.0),    # mu0
        (1e-4, 2.0),    # mu1
        (1e-4, 5.0),    # alpha
        (0.05, 5.0),    # beta
        (0.005, 1.0),   # q1
        (0.005, 1.0),   # q2
    ]

    def neg_ll(params):
        ll, _ = _mmhp_forward(target_times, source_times, T, params)
        if np.isnan(ll) or np.isinf(ll):
            return 1e15
        return -ll

    result = minimize(neg_ll, x0=x0, bounds=bounds, method='L-BFGS-B',
                      options={'maxiter': 200})

    params = tuple(result.x)
    log_lik = -result.fun

    # Viterbi decode
    states, _ = _mmhp_viterbi(target_times, source_times, T, params)

    # Extract coupling episodes
    episodes = _extract_episodes(target_times, source_times, states, T)

    return params, log_lik, states, episodes


def _extract_episodes(target_times, source_times, states, T):
    """Convert Viterbi state sequence into CouplingEpisode list.

    An episode spans from the first active-state event to the last
    consecutive active-state event, with a buffer of ±1 inter-event
    interval on each side.
    """
    N = len(states)
    if N == 0:
        return []

    episodes = []
    in_episode = False
    ep_start = 0.0
    ep_src = 0
    ep_tgt = 0
    ep_triggered = 0

    for n in range(N):
        if states[n] == 1 and not in_episode:
            # Episode starts
            in_episode = True
            # Start slightly before this event
            if n > 0:
                ep_start = (target_times[n - 1] + target_times[n]) / 2
            else:
                ep_start = max(0, target_times[n] - 1.0)
            ep_tgt = 1
            ep_triggered = 1
            ep_src = int(np.sum((source_times >= ep_start) &
                                (source_times <= target_times[n])))

        elif states[n] == 1 and in_episode:
            # Continue episode
            ep_tgt += 1
            ep_triggered += 1
            ep_src = int(np.sum((source_times >= ep_start) &
                                (source_times <= target_times[n])))

        elif states[n] == 0 and in_episode:
            # Episode ends
            in_episode = False
            if n > 0:
                ep_end = (target_times[n - 1] + target_times[n]) / 2
            else:
                ep_end = target_times[n]

            episodes.append(CouplingEpisode(
                start_s=round(ep_start, 2),
                end_s=round(ep_end, 2),
                duration_s=round(ep_end - ep_start, 2),
                n_source_events=ep_src,
                n_target_events=ep_tgt,
                n_triggered=ep_triggered,
            ))

    # Close final episode if still active
    if in_episode:
        ep_end = min(target_times[-1] + 1.0, T)
        episodes.append(CouplingEpisode(
            start_s=round(ep_start, 2),
            end_s=round(ep_end, 2),
            duration_s=round(ep_end - ep_start, 2),
            n_source_events=ep_src,
            n_target_events=ep_tgt,
            n_triggered=ep_triggered,
        ))

    return episodes


# ── 4. Top-Level Orchestrator ─────────────────────────────────────────

def hawkes_coupling_analysis(p1_raw, p2_raw, fs,
                             n_components=6,
                             min_events=5,
                             n_basis=5,
                             lag_range=(0.5, 8.0),
                             bandwidth=1.0,
                             seed=42):
    """Full NMF → group-sparse Hawkes → MMHP coupling analysis.

    1. Joint NMF discovers shared expression dictionary
    2. Event detection in NMF component activations
    3. Group-sparse multivariate Hawkes fits ALL pathways jointly
       (sparsity replaces FDR — no multiple testing correction needed)
    4. MMHP on each active pathway for temporal localization

    Args:
        p1_raw, p2_raw: (T, C) raw blendshapes in [0, 1] at native rate.
        fs: sampling rate (Hz).
        n_components: NMF components (expression types per person).
        min_events: minimum events per component to include in Hawkes.
        n_basis: Gaussian basis functions per kernel.
        lag_range: (min, max) seconds for basis centers.
        bandwidth: Gaussian basis bandwidth (seconds).
        seed: random state.

    Returns:
        HawkesCouplingResult with discovered pathways and coupling episodes.
    """
    T_samp = min(p1_raw.shape[0], p2_raw.shape[0])
    duration = T_samp / fs
    k = n_components

    # 1. NMF expression discovery
    H, W_p1, W_p2, comp_names, explained = nmf_expression_discovery(
        p1_raw, p2_raw, n_components=k, seed=seed)

    # 2. Event detection
    events_p1 = detect_component_events(W_p1, fs=fs)
    events_p2 = detect_component_events(W_p2, fs=fs)

    # 3. Build joint event list: [P1_comp0, ..., P1_compK, P2_comp0, ..., P2_compK]
    #    Cross-person pathways are i∈[0,k) → j∈[k,2k) (P1 → P2)
    #    and i∈[k,2k) → j∈[0,k) (P2 → P1)
    all_events = []
    type_labels = []
    for i in range(k):
        ev = events_p1[i] if len(events_p1[i]) >= min_events else np.array([])
        all_events.append(ev)
        type_labels.append(f'P1_{comp_names[i][:15]}')
    for j in range(k):
        ev = events_p2[j] if len(events_p2[j]) >= min_events else np.array([])
        all_events.append(ev)
        type_labels.append(f'P2_{comp_names[j][:15]}')

    D = 2 * k

    # 4. Group-sparse Hawkes: fit all pathways jointly
    mu, A, centers, active_raw, gs_ll, gs_info = fit_hawkes_group_sparse(
        all_events, duration,
        n_basis=n_basis, lag_range=lag_range, bandwidth=bandwidth,
        bic_select=True, max_iter=50)

    # 5. Extract cross-person pathways (P1→P2 and P2→P1)
    pathways = []
    for src, tgt in active_raw:
        # Determine if this is cross-person
        src_is_p1 = src < k
        tgt_is_p1 = tgt < k
        if src_is_p1 == tgt_is_p1:
            continue  # within-person self-excitation — skip

        # Map back to component indices
        src_comp = src if src_is_p1 else src - k
        tgt_comp = tgt - k if not tgt_is_p1 else tgt
        src_person = 'P1' if src_is_p1 else 'P2'
        tgt_person = 'P2' if not tgt_is_p1 else 'P1'

        # Reconstruct kernel for peak lag
        tau_grid, kernel = hawkes_gs_kernel(A, centers, bandwidth, src, tgt)
        peak_idx = np.argmax(kernel)
        peak_lag = tau_grid[peak_idx] if kernel.max() > 0 else 0
        alpha_total = kernel.sum() * (tau_grid[1] - tau_grid[0])  # ∫kernel

        pw = HawkesPathway(
            source_component=src_comp,
            target_component=tgt_comp,
            source_name=f'{src_person}:{comp_names[src_comp]}',
            target_name=f'{tgt_person}:{comp_names[tgt_comp]}',
            mu=float(mu[tgt]),
            alpha=float(alpha_total),
            beta=float(1.0 / max(peak_lag, 0.1)),
            peak_lag_s=float(peak_lag),
            log_lik=gs_ll,
            log_lik_null=0,
            llr=0,
            p_value=0,
            significant=True,  # selected by sparsity, not p-value
        )
        pathways.append(pw)

    # 6. MMHP on each active cross-person pathway
    for pw in pathways:
        src_events = events_p1[pw.source_component] if 'P1' in pw.source_name else events_p2[pw.source_component]
        tgt_events = events_p2[pw.target_component] if 'P2' in pw.target_name else events_p1[pw.target_component]

        if len(tgt_events) < 5:
            continue

        params, mmhp_ll, states, episodes = fit_mmhp(
            src_events, tgt_events, duration,
            hawkes_mu=pw.mu, hawkes_alpha=pw.alpha, hawkes_beta=pw.beta)

        pw.log_lik = mmhp_ll
        pw.episodes = episodes
        pw.state_times = tgt_events
        pw.state_labels = states
        pw.n_triggered = int((states == 1).sum())
        pw.n_spontaneous = int((states == 0).sum())
        total_coupled = sum(ep.duration_s for ep in episodes)
        pw.coupling_fraction = total_coupled / duration if duration > 0 else 0

    return HawkesCouplingResult(
        n_components=k,
        component_names=comp_names,
        H=H,
        explained_variance=explained,
        events_p1=events_p1,
        events_p2=events_p2,
        pathways=pathways,
        n_significant=len(pathways),
        duration_s=duration,
    )


# ── Utilities ─────────────────────────────────────────────────────────

def _logsumexp(x):
    """Numerically stable log-sum-exp for a small array."""
    x = np.asarray(x)
    c = x.max()
    if np.isinf(c):
        return c
    return c + np.log(np.sum(np.exp(x - c)))


def _bh_fdr(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns boolean mask of significant entries.
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = alpha * np.arange(1, n + 1) / n

    # Find largest k where p_(k) <= threshold
    passing = sorted_p <= thresholds
    if not passing.any():
        return np.zeros(n, dtype=bool)

    max_k = np.where(passing)[0][-1]
    sig = np.zeros(n, dtype=bool)
    sig[order[:max_k + 1]] = True
    return sig
