"""Stage 3c — lagged sparse CCA on event-gated samples.

Lag sweep ``τ ∈ [-1, +1] s, step 0.1 s`` (21 lags). At each τ:
    SCCA_PMD(latent_dimensions=1, tau=[c,c]).fit([X[:T-τ], Y[τ:]])

Pooling policy:
  episode T_samples >= cca_min_per_episode_samples (200)
      → per-episode CCA
  else if pool of (same-dyad, same-condition, similar-duration) episodes
       reaches cca_min_pooled_samples (600)
      → pooled CCA, all member episodes inherit features + pooled_flag=True
  else
      → NaN features, _3c_valid=False

Outputs (17 features clustered + 1 metadata flag):
  cca_peak_r, cca_peak_lag_s, cca_lag_var,
  cca_p1_region_present[7], cca_p2_region_present[7],
  cca_p1_sparsity, cca_p2_sparsity,
  cca_pooled_flag (metadata only — excluded from clustering)
"""
from __future__ import annotations

import warnings

import numpy as np

from cadence.constants import AU_REGIONS_7
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG

REGION_NAMES = list(AU_REGIONS_7.keys())
N_REGIONS = len(REGION_NAMES)

# Pre-compute AU → region binary mask for region-presence aggregation
_AU_REGION_MASK = np.zeros((52, N_REGIONS), dtype=np.int8)
for r_idx, (_r, lst) in enumerate(AU_REGIONS_7.items()):
    for au in lst:
        _AU_REGION_MASK[au, r_idx] = 1


def _empty_features() -> dict:
    out = {
        'cca_peak_r':       np.nan,
        'cca_peak_lag_s':   np.nan,
        'cca_lag_var':      np.nan,
        'cca_p1_sparsity':  np.nan,
        'cca_p2_sparsity':  np.nan,
        'cca_pooled_flag':  False,
        '_3c_valid':        False,
    }
    for r in REGION_NAMES:
        out[f'cca_p1_region_present__{r}'] = 0
        out[f'cca_p2_region_present__{r}'] = 0
    return out


def _scca_one_lag(X: np.ndarray, Y: np.ndarray, tau_l1: float):
    """Fit SCCA_PMD on (X, Y); return (corr, w_x, w_y) or (NaN, None, None)."""
    if X.shape[0] < 30:
        return np.nan, None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from cca_zoo.linear import SCCA_PMD
            m = SCCA_PMD(latent_dimensions=1, tau=[tau_l1, tau_l1],
                         max_iter=200, tol=1e-5, random_state=42)
            m.fit([X, Y])
            r = float(m.score([X, Y])[0])
            return r, m.weights_[0][:, 0], m.weights_[1][:, 0]
    except Exception:
        return np.nan, None, None


def _features_from_fit(peak_r, peak_lag, lag_var, w1, w2,
                        pooled_flag: bool) -> dict:
    out = _empty_features()
    out['cca_peak_r']      = float(peak_r)
    out['cca_peak_lag_s']  = float(peak_lag)
    out['cca_lag_var']     = float(lag_var)
    out['cca_pooled_flag'] = bool(pooled_flag)
    out['_3c_valid']       = True
    if w1 is not None and w2 is not None:
        # Normalize for sparsity calc
        nz1 = np.abs(w1) > 1e-6
        nz2 = np.abs(w2) > 1e-6
        out['cca_p1_sparsity'] = float(nz1.mean())
        out['cca_p2_sparsity'] = float(nz2.mean())
        # Region presence
        r1 = (nz1[:, None].astype(np.int8) * _AU_REGION_MASK).sum(axis=0) > 0
        r2 = (nz2[:, None].astype(np.int8) * _AU_REGION_MASK).sum(axis=0) > 0
        for i, r in enumerate(REGION_NAMES):
            out[f'cca_p1_region_present__{r}'] = int(r1[i])
            out[f'cca_p2_region_present__{r}'] = int(r2[i])
    return out


def _lag_sweep(p1_seq: np.ndarray, p2_seq: np.ndarray, fs: float,
                config: SynchronyConfig):
    """Run SCCA at every τ; return (r_per_lag, peak_idx, w1_at_peak, w2_at_peak)."""
    lags_s = config.lag_grid_s()
    n_lags = len(lags_s)
    rs = np.full(n_lags, np.nan, dtype=np.float64)
    best = (-1.0, None, None, np.nan)
    for li, tau_s in enumerate(lags_s):
        tau_samp = int(round(tau_s * fs))
        if tau_samp >= 0:
            X = p1_seq[: p1_seq.shape[0] - tau_samp]
            Y = p2_seq[tau_samp:]
        else:
            X = p1_seq[-tau_samp:]
            Y = p2_seq[: p2_seq.shape[0] + tau_samp]
        L = min(X.shape[0], Y.shape[0])
        if L < 30:
            continue
        X = X[:L]
        Y = Y[:L]
        r, w1, w2 = _scca_one_lag(X, Y, config.cca_tau)
        rs[li] = r if np.isfinite(r) else np.nan
        if np.isfinite(r) and r > best[0]:
            best = (r, w1, w2, tau_s)
    if not np.isfinite(rs).any():
        return rs, np.nan, np.nan, None, None, np.nan
    peak_lag_s = best[3]
    peak_r = best[0]
    finite = np.isfinite(rs)
    if finite.sum() >= 2:
        # Concentration of correlation across lags: high if peak is sharp.
        # We define 1 - (mean_finite / max) as "lag selectivity" — higher
        # = more lag-specific.
        lag_var = float(1.0 - rs[finite].mean() / max(peak_r, 1e-6))
    else:
        lag_var = float('nan')
    return rs, peak_lag_s, peak_r, best[1], best[2], lag_var


def compute_cca_features(p1_au_ep: np.ndarray, p2_au_ep: np.ndarray,
                          fs: float = 30.0,
                          config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Per-episode CCA. Caller decides whether to call this directly or pool.

    Returns NaN/_3c_valid=False if the episode is too short for a stable
    per-episode estimate.
    """
    if p1_au_ep.shape[0] < config.cca_min_per_episode_samples:
        return _empty_features()
    L = min(p1_au_ep.shape[0], p2_au_ep.shape[0])
    p1, p2 = p1_au_ep[:L], p2_au_ep[:L]
    rs, peak_lag, peak_r, w1, w2, lag_var = _lag_sweep(p1, p2, fs, config)
    if not np.isfinite(peak_r):
        return _empty_features()
    return _features_from_fit(peak_r, peak_lag, lag_var, w1, w2,
                                pooled_flag=False)


def compute_cca_features_pooled(p1_segments: list[np.ndarray],
                                  p2_segments: list[np.ndarray],
                                  fs: float = 30.0,
                                  config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Pooled CCA — concatenate same-class short episodes for a stable fit.

    Each (p1_seg, p2_seg) is one episode; lengths must match per pair.
    Returns NaN/_3c_valid=False if total pooled length < min_pooled.
    """
    if not p1_segments:
        return _empty_features()
    L_total = sum(p1.shape[0] for p1 in p1_segments)
    if L_total < config.cca_min_pooled_samples:
        return _empty_features()
    # Concatenate
    p1_pool = np.vstack(p1_segments)
    p2_pool = np.vstack(p2_segments)
    rs, peak_lag, peak_r, w1, w2, lag_var = _lag_sweep(p1_pool, p2_pool, fs,
                                                          config)
    if not np.isfinite(peak_r):
        return _empty_features()
    return _features_from_fit(peak_r, peak_lag, lag_var, w1, w2,
                                pooled_flag=True)


def feature_names_clustering() -> list[str]:
    """Stage 3c features used in clustering (cca_pooled_flag is metadata only)."""
    base = ['cca_peak_r', 'cca_peak_lag_s', 'cca_lag_var',
             'cca_p1_sparsity', 'cca_p2_sparsity']
    region_p1 = [f'cca_p1_region_present__{r}' for r in REGION_NAMES]
    region_p2 = [f'cca_p2_region_present__{r}' for r in REGION_NAMES]
    return base + region_p1 + region_p2
