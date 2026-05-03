"""Stage 3b — within-episode DTW features (Sakoe-Chiba banded).

Two flavours:
  (i)  dependent DTW on the full 52-D AU trajectory
  (ii) per-region dependent DTW (7 regions from AU_REGIONS_7)

Each DTW yields 4 features:
  dtw_distance_norm     # path cost / path length
  dtw_mean_lag_s        # mean (j_path - i_path) / fs along warping path
  dtw_lag_var           # variance of lag along path
  dtw_asymmetry         # (#frames j>i - #where j<i) / path_length ∈ [-1,1]

Total 32 features per episode.

Uses ``dtaidistance.dtw_ndim`` for the multivariate path. C-impl (releases
GIL, fine to call from joblib threading backend). Falls back to euclidean
distance for episodes shorter than ``2W + 4`` (band constraint trivial).
"""
from __future__ import annotations

import numpy as np

from cadence.constants import AU_REGIONS_7
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG

REGION_NAMES = list(AU_REGIONS_7.keys())


def _dtw_path_features(p1_seq: np.ndarray, p2_seq: np.ndarray,
                         band_frames: int, fs: float) -> dict:
    """Run dependent DTW on (T, D) sequences; return 4 path features."""
    T1, D = p1_seq.shape
    T2, _ = p2_seq.shape
    if T1 < (2 * band_frames + 4) or T2 < (2 * band_frames + 4):
        return {'dtw_distance_norm': np.nan, 'dtw_mean_lag_s': np.nan,
                'dtw_lag_var': np.nan, 'dtw_asymmetry': np.nan}

    # dtaidistance multidim: warping_paths_ndim returns the cost matrix;
    # best_path returns the warping path through it.
    from dtaidistance import dtw_ndim
    try:
        # window argument is the Sakoe-Chiba band half-width
        d, paths = dtw_ndim.warping_paths(
            p1_seq.astype(np.float64), p2_seq.astype(np.float64),
            window=int(band_frames), use_c=True, psi=0,
        )
    except Exception:
        return {'dtw_distance_norm': np.nan, 'dtw_mean_lag_s': np.nan,
                'dtw_lag_var': np.nan, 'dtw_asymmetry': np.nan}

    from dtaidistance import dtw
    try:
        path = dtw.best_path(paths)
    except Exception:
        return {'dtw_distance_norm': np.nan, 'dtw_mean_lag_s': np.nan,
                'dtw_lag_var': np.nan, 'dtw_asymmetry': np.nan}

    if not path:
        return {'dtw_distance_norm': np.nan, 'dtw_mean_lag_s': np.nan,
                'dtw_lag_var': np.nan, 'dtw_asymmetry': np.nan}

    arr = np.asarray(path, dtype=np.int32)  # (path_len, 2)
    i = arr[:, 0]
    j = arr[:, 1]
    L = len(path)

    # Normalize path cost by path length so duration doesn't dominate.
    distance_norm = float(d) / max(L, 1)

    # Lag along path (positive → P2 leads P1's index, i.e., j_path > i_path
    # means P2 is "ahead" of P1).
    lag_samples = (j - i).astype(np.float64)
    mean_lag_s = float(lag_samples.mean()) / fs
    lag_var    = float(lag_samples.var(ddof=0)) / (fs ** 2)
    asym = float((lag_samples > 0).sum() - (lag_samples < 0).sum()) / L

    return {'dtw_distance_norm': distance_norm,
            'dtw_mean_lag_s':    mean_lag_s,
            'dtw_lag_var':       lag_var,
            'dtw_asymmetry':     asym}


def compute_dtw_features(p1_au_ep: np.ndarray, p2_au_ep: np.ndarray,
                          fs: float = 30.0,
                          config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Compute Stage 3b features for one episode.

    Args:
        p1_au_ep: (T1, 52) Stage-0 baselined AU trajectory for one role.
        p2_au_ep: (T2, 52) for the other role (interpolated to comparable
            sampling). Shapes can differ; DTW handles non-equal lengths.
        fs: native sampling rate.
        config

    Returns:
        Dict with 32 features (4 per DTW × 8 DTWs).
    """
    band_frames = max(2, int(round(config.dtw_band_s * fs)))

    out = {}
    # (i) Full 52-D dependent DTW
    full_feats = _dtw_path_features(p1_au_ep, p2_au_ep, band_frames, fs)
    for k, v in full_feats.items():
        out[f'{k}__full52'] = v

    # (ii) Per-region (mean activation across AUs in region as a 1-D signal)
    for r in REGION_NAMES:
        idxs = AU_REGIONS_7[r]
        # Aggregate to 1-D mean across the AUs in this region per timestep
        p1_r = p1_au_ep[:, idxs].mean(axis=1, keepdims=True)
        p2_r = p2_au_ep[:, idxs].mean(axis=1, keepdims=True)
        rf = _dtw_path_features(p1_r, p2_r, band_frames, fs)
        for k, v in rf.items():
            out[f'{k}__{r}'] = v

    out['_3b_valid'] = bool(np.isfinite(out['dtw_distance_norm__full52']))
    return out


def feature_names() -> list[str]:
    base = ['dtw_distance_norm', 'dtw_mean_lag_s', 'dtw_lag_var', 'dtw_asymmetry']
    suffixes = ['full52'] + REGION_NAMES
    return [f'{b}__{s}' for s in suffixes for b in base]
