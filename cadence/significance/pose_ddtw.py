"""Dependent Dynamic Time Warping (DDTW) for pose-coupling.

Phase 0 alternative to the inherited multi-lag cross-correlation pose channel.
Per the MVP design spec (docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md),
DDTW captures variable-lag whole-body postural mimicry that fixed-lag cross-product
cannot — joints don't move independently, so a *single shared* warping path
(dependent multivariate DTW) is the right alignment object for pose.

Pipeline per session:
  1. Read p{1,2}_pose33 (N, 33, 4) from data/preproc/pose/v1/<sid>.npz
  2. Drop visibility column -> (N, 33, 3) -> flatten to (N, 99) position stream
  3. Project into shared cross-session PCA subspace (10 components)
  4. Resample to 12 Hz (uniform across native rates: 14.4 Hz, 54.1 Hz, 58.3 Hz)
  5. Sliding 4s window @ 500ms stride -> per-stride DDTW alignment cost
  6. Condition-block surrogates: 200 P2-condition-permutations, recompute DDTW
  7. Per-stride z-score against surrogate distribution
  8. AR(1) prewhiten + standardize the surrogate-z timecourse
  9. Output (T,) coupling z at 2 Hz, ready to plug into MVP scaffold pose slot

Implementation notes:
- Uses dtaidistance.dtw_ndim.distance_fast (C backend, ~500x faster than pure Python).
  No Sakoe-Chiba band (window=None) per spec — the 4s window itself bounds warp.
- DDTW score = -1 * normalized cumulative warp cost (higher z = more coupled).
- Validity: surrogate windows with < 50% valid frames are dropped (NaN in output);
  AR(1) prewhitening is validity-mask aware via prewhiten_and_standardize.
- Joblib threading is the caller's responsibility (see _validate_pose_ddtw.py).

Forward-looking GPU port (deferred): if needed, follow gpu_sliding_te_surrogates'
chunking pattern. Per-batch GPU memory at T=48, D=10 is < 100 MB; the full
per-stride (200 surrogates) easily fits in a single batch on 16 GB GPU.

Phase 1 extensions (docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md,
Task 3) — everything above is unchanged for ``feature_mode='pca'``:

- ``FEATURE_MODES``: ``'pca'`` (Phase 0), ``'angles'`` (12 torso-frame
  segment angles from ``pose_angles``, unwrapped radians, each column in
  noise-SD units), ``'angle_speed'`` (angular speed in deg/s, likewise
  noise-normalised). Angle modes need no shared PCA: the features are
  invariant to camera translation / scale / limb proportion by construction.
- ``build_angle_features``: resampled (N, 33, 4) pose -> (N, 12) DDTW input.
- ``sliding_ddtw_path_features``: warping-path timing per stride (mean lag,
  lag variance, asymmetry) for the real pair, so DDTW no longer conflates
  "moved the same way" with "moved at the same time". Sign convention:
  positive lag = P2 frame later than P1 (matches ``synchrony/features/dtw.py``).
- ``compute_session_ddtw(..., compute_path_features=True)`` and
  ``run_session_ddtw(..., feature_mode=...)`` expose both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dtaidistance import dtw as _dtw
from dtaidistance import dtw_ndim as _dtw_ndim
from dtaidistance.dtw_ndim import distance_fast as _ddtw_distance

from cadence.io.resources import limit_blas_threads
from cadence.significance.pose_angles import (
    ANGLE_NAMES, N_ANGLE_FEATURES, angle_features, estimate_fs,
    moving_average_nan, noise_floor, speed_features, unwrap_nan,
)


# ── Constants ───────────────────────────────────────────────────────

POSE_KEYPOINTS = 33                # MediaPipe pose landmarks
POSE_AXES = 3                       # x, y, z (drop visibility)
POSE_FLAT_DIM = POSE_KEYPOINTS * POSE_AXES  # 99

DDTW_TARGET_RATE_HZ = 12.0          # uniform internal rate (lowest native)
DDTW_WINDOW_S = 4.0                 # window size in seconds
DDTW_STRIDE_S = 0.5                 # stride in seconds (= rSLDS rate 2 Hz)
DDTW_PCA_DIM = 10                   # shared PCA subspace dimensionality

DDTW_N_SURROGATES = 200             # surrogate count per stride
DDTW_VALID_FRAC = 0.5               # min valid-frame fraction per window

# Phase 1 feature modes (see module docstring); 'pca' is the Phase 0 path.
FEATURE_MODES = ('pca', 'angles', 'angle_speed')
ANGLE_MODES = ('angles', 'angle_speed')

POSE_COLS = 4                       # x, y, z, visibility
POSE_FLAT_DIM_VIS = POSE_KEYPOINTS * POSE_COLS  # 132 — flattened (N, 33, 4) view
NOISE_FLOOR_WINDOW = 3              # frames; residual vs moving average (dance_sync)
SPEED_SMOOTH_FRAMES = 5             # frames; angular-speed smoothing (pose_angles default)

# Condition order for condition-block surrogate construction (matches V11/v82)
CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1',
                   'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']


# ── Pose loading + alignment ────────────────────────────────────────

def load_pose_streams(preproc_path: str | Path,
                      include_pose33: bool = False) -> dict:
    """Load pose preproc artifact and extract DDTW-ready arrays.

    Returns:
        dict with keys:
          p1_pose, p2_pose: (N_p, 99) float32 position streams (visibility dropped)
          p1_ts, p2_ts:     (N_p,) LSL timestamps (float64)
          p1_valid, p2_valid: (N_p,) bool per-frame validity
          digest_xdf_md5:   provenance hash
          pose_format_in:   one of {mediapipe33, mediapipe33_meta, wholebody_133}
          p1_pose33, p2_pose33: (N_p, 33, 4) float32 raw (x, y, z, vis) —
                            only when ``include_pose33`` (angle feature modes).
    """
    preproc_path = Path(preproc_path)
    npz = np.load(preproc_path, allow_pickle=False)
    sidecar = json.loads(preproc_path.with_suffix('.json').read_text())

    out = {}
    for p in ('p1', 'p2'):
        pose33 = npz[f'{p}_pose33']                  # (N, 33, 4)
        if include_pose33:
            out[f'{p}_pose33'] = pose33.astype(np.float32, copy=False)
        # Drop visibility (column 3) and flatten
        pose_xyz = pose33[..., :3].reshape(pose33.shape[0], -1).astype(np.float32)
        # Visibility-zeroed coordinates remain zeros — those frames are masked
        # by *_pose_features_valid downstream.
        out[f'{p}_pose'] = pose_xyz
        out[f'{p}_ts'] = npz[f'{p}_pose33_ts'].astype(np.float64)
        out[f'{p}_valid'] = npz[f'{p}_pose_features_valid'].astype(bool)

    out['digest_xdf_md5'] = sidecar.get('digest_xdf_md5', '')
    out['pose_format_in'] = sidecar.get('pose_format_in', '')
    return out


def resample_to_uniform_rate(pose: np.ndarray, ts: np.ndarray,
                              target_rate_hz: float = DDTW_TARGET_RATE_HZ
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (N, D) pose stream to a uniform rate in LSL time.

    Linear interpolation per channel. Returns (pose_uniform, ts_uniform) where
    ts_uniform = arange(ts[0], ts[-1], 1/target_rate_hz).
    """
    if pose.shape[0] < 2:
        return pose.astype(np.float32), ts.astype(np.float64)
    dt = 1.0 / target_rate_hz
    ts_uniform = np.arange(ts[0], ts[-1], dt, dtype=np.float64)
    if ts_uniform.size == 0:
        return np.empty((0, pose.shape[1]), dtype=np.float32), ts_uniform
    # Single C-level multi-channel linear interp — releases GIL once per call,
    # so joblib threading actually parallelises across sessions instead of
    # serialising on a 99-iteration Python loop's GIL acquire/release churn.
    # Edge-clamp behaviour matches np.interp (constant extrapolation to ends).
    idx_right = np.searchsorted(ts, ts_uniform, side='left').clip(1, ts.size - 1)
    idx_left = idx_right - 1
    t_left = ts[idx_left]
    t_right = ts[idx_right]
    w = ((ts_uniform - t_left) / np.maximum(t_right - t_left, 1e-12)).clip(0.0, 1.0)
    out = (pose[idx_left] + w[:, None] * (pose[idx_right] - pose[idx_left])).astype(np.float32)
    return out, ts_uniform


def resample_validity_to_uniform(valid: np.ndarray, ts: np.ndarray,
                                  ts_uniform: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resample of a per-frame boolean validity array."""
    if valid.size == 0 or ts_uniform.size == 0:
        return np.zeros(ts_uniform.size, dtype=bool)
    idx = np.searchsorted(ts, ts_uniform, side='left')
    idx = np.clip(idx, 0, valid.size - 1)
    return valid[idx]


# ── Shared cross-session PCA ────────────────────────────────────────

def fit_shared_pca(pose_streams: dict[str, np.ndarray],
                   n_components: int = DDTW_PCA_DIM,
                   max_train_frames_per_session: int = 30000,
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a shared cross-session PCA on pooled pose streams.

    Args:
        pose_streams: {session_id_p: (N, 99) array}. Both p1 and p2 streams
            from each session contribute (so a session contributes 2 entries).
        n_components: target subspace dimensionality.
        max_train_frames_per_session: per-stream cap for memory.

    Returns:
        components: (n_components, 99) PCA basis (rows = components).
        mean: (99,) sample mean for centering.
        diag: dict with 'explained_variance_ratio', 'per_session_recon_error'.
    """
    rng = np.random.default_rng(0)
    blocks = []
    for sid_p, pose in pose_streams.items():
        if pose.shape[0] == 0:
            continue
        if pose.shape[0] > max_train_frames_per_session:
            sel = rng.choice(pose.shape[0], max_train_frames_per_session,
                             replace=False)
            blocks.append(pose[sel])
        else:
            blocks.append(pose)
    if not blocks:
        raise ValueError('fit_shared_pca: no training data')
    X = np.concatenate(blocks, axis=0).astype(np.float64)

    mean = X.mean(axis=0)
    Xc = X - mean

    # Use SVD; rank-cap to n_components
    # Memory: full SVD on (M, 99) is fine — M <= ~600k, 99-D
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]                       # (n_components, 99)
    total_var = (S ** 2).sum() / max(Xc.shape[0] - 1, 1)
    explained = (S[:n_components] ** 2) / max(Xc.shape[0] - 1, 1)
    evr = explained / max(total_var, 1e-12)

    # Per-session reconstruction error (should be cohort-comparable)
    per_session_recon = {}
    for sid_p, pose in pose_streams.items():
        if pose.shape[0] == 0:
            per_session_recon[sid_p] = float('nan')
            continue
        Xs = pose.astype(np.float64) - mean
        proj = Xs @ components.T                         # (N, n_components)
        recon = proj @ components                        # (N, 99)
        err = np.linalg.norm(Xs - recon) / max(np.linalg.norm(Xs), 1e-12)
        per_session_recon[sid_p] = float(err)

    diag = {
        'explained_variance_ratio': evr.tolist(),
        'cumulative_variance_ratio': np.cumsum(evr).tolist(),
        'per_session_recon_error': per_session_recon,
        'n_train_frames': int(X.shape[0]),
    }
    return components.astype(np.float32), mean.astype(np.float32), diag


def project_pca(pose: np.ndarray, components: np.ndarray,
                mean: np.ndarray) -> np.ndarray:
    """Project (N, 99) pose into the shared PCA subspace -> (N, n_components).

    Returns float64 (DDTW expects double).
    """
    if pose.shape[0] == 0:
        return np.empty((0, components.shape[0]), dtype=np.float64)
    return ((pose.astype(np.float64) - mean) @ components.T).astype(np.float64)


# ── Sliding-window DDTW ─────────────────────────────────────────────

def _stride_indices(n_frames: int, window_frames: int,
                    stride_frames: int) -> np.ndarray:
    """Return array of stride start indices [0, stride, 2*stride, ...]."""
    if n_frames < window_frames or stride_frames <= 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_frames - window_frames + 1, stride_frames,
                     dtype=np.int64)


def _ddtw_score(window_a: np.ndarray, window_b: np.ndarray,
                center: bool = True) -> float:
    """DDTW alignment cost normalized by path length.

    Returns -cost (so higher = more coupled); both inputs must be float64 (T, D).

    Per-window centering (default True) subtracts each window's per-PC mean
    before alignment. Without this, individual posture differences (P1 vs P2
    baseline pose, or systematic body-size differences after shared PCA)
    dominate the cost and overwhelm dynamic-coupling signal.
    """
    if center:
        a = window_a - window_a.mean(axis=0, keepdims=True)
        b = window_b - window_b.mean(axis=0, keepdims=True)
        # distance_fast wants C-contiguous double — guarantee it
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
    else:
        a, b = window_a, window_b
    cost = _ddtw_distance(a, b)
    if not np.isfinite(cost):
        return np.nan
    # DTW path length proxy: sqrt(T_a + T_b)
    norm = float(np.sqrt(a.shape[0] + b.shape[0]))
    return -float(cost) / max(norm, 1e-12)


def sliding_ddtw_real(p1_pca: np.ndarray, p2_pca: np.ndarray,
                       window_frames: int, stride_frames: int,
                       p1_valid: np.ndarray, p2_valid: np.ndarray,
                       valid_frac: float = DDTW_VALID_FRAC,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Real (P1 vs P2) DDTW score per stride position.

    Returns:
        scores: (n_strides,) float32 — NaN where window failed validity.
        starts: (n_strides,) int64 — stride start indices in P1 frames.
    """
    # P1 and P2 may have slightly different frame counts after uniform resample;
    # work on min length.
    n = min(p1_pca.shape[0], p2_pca.shape[0])
    if n < window_frames:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    starts = _stride_indices(n, window_frames, stride_frames)
    scores = np.full(starts.size, np.nan, dtype=np.float32)
    p1, p2 = p1_pca[:n], p2_pca[:n]
    v1, v2 = p1_valid[:n], p2_valid[:n]
    min_valid = int(np.ceil(valid_frac * window_frames))
    for i, s in enumerate(starts):
        e = s + window_frames
        if (v1[s:e].sum() < min_valid) or (v2[s:e].sum() < min_valid):
            continue
        scores[i] = _ddtw_score(p1[s:e], p2[s:e])
    return scores, starts


# ── Per-window centering ────────────────────────────────────────────

def _center_window(w: np.ndarray) -> np.ndarray:
    """Subtract per-PC mean within a window. Removes static posture offset
    so DDTW cost reflects within-window dynamic alignment, not individual
    baseline posture differences."""
    return w - w.mean(axis=0, keepdims=True)


# ── Angle feature modes (Phase 1) ───────────────────────────────────

def _fill_nan_linear(x: np.ndarray, edge: str = 'zero') -> np.ndarray:
    """Column-wise linear interpolation across interior NaN gaps. Returns
    float64 (N, D).

    ``edge`` selects the fill for leading / trailing NaN runs (samples before
    the first or after the last finite value of a column):

    - ``'zero'``: -> 0. Correct for signals that genuinely rest at zero
      (angular speed).
    - ``'hold'``: keep ``np.interp``'s natural clamp, i.e. the column stays
      constant at its first / last finite value. Required for torso-frame
      angles, most of which rest near +-pi (limbs anti-parallel to the torso
      axis): a zero fill there is a multi-radian step at the first / last
      appearance of a landmark, which per-window centering cannot remove and
      which the surrogate z then reports as spurious anti-coupling. A
      constant edge run vanishes under per-window centering, so windows with
      no data for that feature contribute nothing to the DTW cost.

    All-NaN columns -> 0 in both modes (also a constant).
    """
    if edge not in ('zero', 'hold'):
        raise ValueError(f"_fill_nan_linear: edge must be 'zero' or 'hold'; got {edge!r}")
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    n = x.shape[0]
    if n == 0:
        return out
    idx = np.arange(n)
    for j in range(x.shape[1]):
        m = np.isfinite(x[:, j])
        if m.all():
            continue
        if not m.any():
            out[:, j] = 0.0
            continue
        fin = idx[m]
        out[:, j] = np.interp(idx, fin, x[m, j])
        if edge == 'zero':
            out[:fin[0], j] = 0.0
            out[fin[-1] + 1:, j] = 0.0
    return out


def _residual_noise_floor(x: np.ndarray, window: int = NOISE_FLOOR_WINDOW,
                          floor_min: float = 1e-4) -> np.ndarray:
    """(D,) robust SD of the residual of ``x`` against its ``window``-frame
    NaN-ignoring moving average: 1.4826 * median |x - MA_w(x)| over finite
    samples, floored at ``floor_min``; all-NaN columns -> ``floor_min``.

    Same definition as ``pose_angles.noise_floor`` but without the unwrap
    step, for signals that are not phases (angular speed in deg/s).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape[1], floor_min, dtype=np.float64)
    if x.shape[0] == 0:
        return out
    resid = np.abs(x - moving_average_nan(x, window))
    finite = np.isfinite(resid)
    has = finite.any(axis=0)
    if has.any():
        med = np.nanmedian(np.where(finite, resid, np.nan)[:, has], axis=0)
        out[has] = np.maximum(1.4826 * med, floor_min)
    return out


def build_angle_features(pose33_uniform: np.ndarray, ts_uniform: np.ndarray,
                         mode: str, noise_normalize: bool = True,
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(N, 33, 4) resampled pose -> (X (N, 12) float64, valid (N,) bool, info).

    ``mode='angles'``: torso-frame segment angles (``pose_angles``), unwrapped
    radians. NaN features are filled by linear interpolation across interior
    gaps so every window is DTW-able; leading / trailing NaN runs are
    edge-held at the column's first / last finite value (torso-frame angles
    rest near +-pi, so a zero fill would be a multi-radian step at the first
    / last appearance of a landmark; a held constant vanishes under
    per-window centering instead). The per-window validity gate in
    ``sliding_ddtw_real`` is what rejects windows with too many invalid
    frames. When ``noise_normalize`` each
    column is divided by its measured noise floor (``pose_angles.noise_floor``:
    robust SD of the residual vs a 3-frame moving average, computed on the
    unfilled unwrapped angles) so DTW cost is in noise-SD units instead of
    raw radians; ``info['noise_floor']`` holds the (12,) radians values
    either way.

    ``mode='angle_speed'``: ``pose_angles.speed_features`` (deg/s, 5-frame
    smoothing at the grid rate estimated from ``ts_uniform``), interior gaps
    interpolated the same way but leading / trailing NaN -> 0 (speed
    genuinely rests at zero, and ``speed_features`` already NaNs the gap
    neighbours), then divided by its own noise floor (robust SD of the
    speed residual vs a 3-frame moving average, deg/s — no unwrap).

    ``valid`` is the angle-frame validity from ``angle_features`` (torso
    frame defined and >= 6 of 12 features finite); callers AND it with the
    preproc ``pose_features_valid`` flag. ``info`` keys: ``feature_mode``,
    ``noise_floor`` (12,), ``noise_floor_units`` ('rad' | 'deg/s'),
    ``noise_normalize``, ``fs``, ``n_features``, ``feature_names``,
    ``n_valid``, ``n_frames``.
    """
    if mode not in ANGLE_MODES:
        raise ValueError(f'build_angle_features: mode must be one of {ANGLE_MODES}; got {mode!r}')
    pose33_uniform = np.asarray(pose33_uniform)
    ts_uniform = np.asarray(ts_uniform, dtype=np.float64)
    fs = estimate_fs(ts_uniform, DDTW_TARGET_RATE_HZ)

    raw, valid = angle_features(pose33_uniform)          # (N, 12) float32, (N,) bool
    unwrapped = unwrap_nan(raw)                          # (N, 12) float64, NaN kept
    if mode == 'angles':
        feats = unwrapped
        floor = noise_floor(unwrapped, window=NOISE_FLOOR_WINDOW)
        units = 'rad'
    else:
        feats = speed_features(unwrapped, fs, smooth_frames=SPEED_SMOOTH_FRAMES)
        floor = _residual_noise_floor(feats, window=NOISE_FLOOR_WINDOW)
        units = 'deg/s'

    X = _fill_nan_linear(feats, edge='hold' if mode == 'angles' else 'zero')
    if noise_normalize:
        X = X / floor[None, :]
    X = np.ascontiguousarray(X, dtype=np.float64)

    info = {
        'feature_mode': mode,
        'noise_floor': floor.astype(np.float64),
        'noise_floor_units': units,
        'noise_normalize': bool(noise_normalize),
        'fs': float(fs),
        'n_features': int(N_ANGLE_FEATURES),
        'feature_names': list(ANGLE_NAMES),
        'n_valid': int(valid.sum()),
        'n_frames': int(valid.size),
    }
    return X, valid.astype(bool), info


def resample_pose33_visibility(pose33: np.ndarray, ts: np.ndarray,
                               ts_grid: np.ndarray) -> np.ndarray:
    """(N, 33, 4) native pose + (N,) ts -> (M, 33) visibility on ``ts_grid``.

    Conservative rule: each grid sample takes the **minimum** visibility of
    the two native frames that bracket it (the same left/right frames that
    linear interpolation of the coordinates blends). A landmark hidden in
    either bracketing frame is therefore hidden on the grid, which masks the
    coordinates that were blended with a visibility-zeroed frame. Grid
    samples outside the native range clamp to the edge frame. This is the
    stricter alternative to thresholding the linearly interpolated
    visibility at 0.5 (which would keep frames blended up to 50% with a
    hidden frame).
    """
    ts = np.asarray(ts, dtype=np.float64)
    vis = np.asarray(pose33)[:, :, 3]
    if vis.shape[0] == 0 or ts_grid.size == 0:
        return np.zeros((ts_grid.size, vis.shape[1]), dtype=np.float32)
    if vis.shape[0] == 1:
        return np.repeat(vis[:1], ts_grid.size, axis=0).astype(np.float32)
    idx_right = np.searchsorted(ts, ts_grid, side='left').clip(1, ts.size - 1)
    idx_left = idx_right - 1
    return np.minimum(vis[idx_left], vis[idx_right]).astype(np.float32)


# ── Warping-path timing features (Phase 1) ──────────────────────────

def _window_path_features(window_a: np.ndarray, window_b: np.ndarray,
                          fs: float) -> tuple[float, float, float]:
    """Warping-path timing for one (T, D) window pair: (lag_s, lag_var, asym).

    Windows are per-channel centred exactly as in ``_ddtw_score`` so the path
    is the one behind the reported cost. Lag along the path is
    ``j - i`` (P2 index minus P1 index): positive = P2 frame later than P1.
    lag_s = mean lag / fs; lag_var = var(lag) / fs^2; asym = (#j>i - #j<i) /
    path length in [-1, 1]. NaN triple if the C backend or best_path fails.
    """
    a = np.ascontiguousarray(_center_window(window_a), dtype=np.float64)
    b = np.ascontiguousarray(_center_window(window_b), dtype=np.float64)
    try:
        _, paths = _dtw_ndim.warping_paths(a, b, use_c=True)
        path = _dtw.best_path(paths)
    except Exception:
        return np.nan, np.nan, np.nan
    if not path:
        return np.nan, np.nan, np.nan
    arr = np.asarray(path, dtype=np.int64)
    lag = (arr[:, 1] - arr[:, 0]).astype(np.float64)
    L = arr.shape[0]
    fs = float(fs)
    return (float(lag.mean()) / fs,
            float(lag.var(ddof=0)) / (fs * fs),
            float((lag > 0).sum() - (lag < 0).sum()) / L)


def sliding_ddtw_path_features(p1: np.ndarray, p2: np.ndarray,
                               window_frames: int, stride_frames: int,
                               p1_valid: np.ndarray, p2_valid: np.ndarray,
                               fs: float,
                               valid_frac: float = DDTW_VALID_FRAC) -> dict:
    """Warping-path timing features per stride for the real pair (no surrogates).

    Same window / stride / validity layout as ``sliding_ddtw_real`` so the
    outputs align element-wise with ``ddtw_z``. Per stride
    ``dtw_ndim.warping_paths(..., use_c=True)`` + ``dtw.best_path``.

    Returns ``{'lag_s': (n_strides,) float32, 'lag_var': (n_strides,) float32,
    'asym': (n_strides,) float32, 'starts': (n_strides,) int64}``; NaN where
    the window fails the validity gate. Sign: positive lag = P2 index ahead
    of P1 index (P2 later).
    """
    n = min(p1.shape[0], p2.shape[0])
    if n < window_frames:
        empty = np.empty(0, dtype=np.float32)
        return {'lag_s': empty, 'lag_var': empty.copy(), 'asym': empty.copy(),
                'starts': np.empty(0, dtype=np.int64)}
    starts = _stride_indices(n, window_frames, stride_frames)
    lag_s = np.full(starts.size, np.nan, dtype=np.float32)
    lag_var = np.full(starts.size, np.nan, dtype=np.float32)
    asym = np.full(starts.size, np.nan, dtype=np.float32)
    a, b = p1[:n], p2[:n]
    v1, v2 = p1_valid[:n], p2_valid[:n]
    min_valid = int(np.ceil(valid_frac * window_frames))
    for i, s in enumerate(starts):
        e = s + window_frames
        if (v1[s:e].sum() < min_valid) or (v2[s:e].sum() < min_valid):
            continue
        lag_s[i], lag_var[i], asym[i] = _window_path_features(a[s:e], b[s:e], fs)
    return {'lag_s': lag_s, 'lag_var': lag_var, 'asym': asym, 'starts': starts}


# ── Surrogate construction ──────────────────────────────────────────

def assign_frame_conditions(ts_uniform: np.ndarray,
                             markers: list[tuple[float, str]]
                             ) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Build a per-frame condition label and condition-block index list.

    Args:
        ts_uniform: (N,) frame timestamps in LSL seconds.
        markers: list of (lsl_t, marker_string) tuples from digest JSON.

    Returns:
        cond_label: (N,) array of condition strings (or '' for unlabeled gaps).
        blocks: list of (cond_name, start_idx, end_idx) for each contiguous
            condition block whose name appears in CONDITION_ORDER.
    """
    cond_label = np.array([''] * ts_uniform.size, dtype=object)
    blocks_ts = {}
    for t_lsl, mtxt in markers:
        for cond in CONDITION_ORDER:
            if mtxt == f'{cond}_start':
                blocks_ts.setdefault(cond, [None, None])
                blocks_ts[cond][0] = float(t_lsl)
            elif mtxt == f'{cond}_stop':
                blocks_ts.setdefault(cond, [None, None])
                blocks_ts[cond][1] = float(t_lsl)
    blocks = []
    for cond, (t0, t1) in blocks_ts.items():
        if t0 is None or t1 is None or t1 <= t0:
            continue
        i0 = int(np.searchsorted(ts_uniform, t0, side='left'))
        i1 = int(np.searchsorted(ts_uniform, t1, side='right'))
        if i1 - i0 < 2:
            continue
        cond_label[i0:i1] = cond
        blocks.append((cond, i0, i1))
    blocks.sort(key=lambda b: b[1])
    return cond_label, blocks


def session_circular_shift(p2_stream: np.ndarray, p2_valid: np.ndarray,
                            rng: np.random.Generator,
                            min_frac: float = 0.1,
                            max_frac: float = 0.9,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Standard CADENCE surrogate: circular shift P2 by a random offset.

    Shift in [min_frac * N, max_frac * N] frames. Preserves all per-channel
    statistics and autocorrelation of P2; destroys cross-participant timing
    alignment. This is the surrogate pattern used by every other CADENCE
    significance module (compute_coupling_excess, pose_velocity_coupling,
    fast_cycles, etc.) — kept for consistency.

    Note on condition-block boundary blending: the spec proposed shifts
    aligned to condition durations to avoid splicing within a condition. In
    practice, at 4s windows and condition durations of 60s+, < 0.5% of
    windows straddle the wrap-around point — empirically negligible, and
    not worth the complication of a non-uniform shift distribution.
    """
    n = p2_stream.shape[0]
    if n < 4:
        return p2_stream.copy(), p2_valid.copy()
    lo = max(1, int(min_frac * n))
    hi = max(lo + 1, int(max_frac * n))
    shift = int(rng.integers(lo, hi))
    return np.roll(p2_stream, shift, axis=0), np.roll(p2_valid, shift, axis=0)


# ── Top-level per-session DDTW ──────────────────────────────────────

def compute_session_ddtw(p1_pca_uniform: np.ndarray, p2_pca_uniform: np.ndarray,
                          p1_valid_uniform: np.ndarray,
                          p2_valid_uniform: np.ndarray,
                          ts_uniform: np.ndarray,
                          markers: list[tuple[float, str]],
                          *,
                          target_rate_hz: float = DDTW_TARGET_RATE_HZ,
                          window_s: float = DDTW_WINDOW_S,
                          stride_s: float = DDTW_STRIDE_S,
                          n_surrogates: int = DDTW_N_SURROGATES,
                          seed: int = 42,
                          compute_path_features: bool = False,
                          fs: float | None = None,
                          ) -> dict:
    """See `_compute_session_ddtw_inner` — wrapper pins BLAS to 1 thread to
    prevent oversubscription when called from a joblib worker pool. Outside a
    pool, `limit_blas_threads(1)` is essentially free (one re-entry)."""
    with limit_blas_threads(1):
        return _compute_session_ddtw_inner(
            p1_pca_uniform, p2_pca_uniform, p1_valid_uniform, p2_valid_uniform,
            ts_uniform, markers,
            target_rate_hz=target_rate_hz, window_s=window_s, stride_s=stride_s,
            n_surrogates=n_surrogates, seed=seed,
            compute_path_features=compute_path_features, fs=fs)


def _compute_session_ddtw_inner(p1_pca_uniform: np.ndarray,
                                  p2_pca_uniform: np.ndarray,
                                  p1_valid_uniform: np.ndarray,
                                  p2_valid_uniform: np.ndarray,
                                  ts_uniform: np.ndarray,
                                  markers: list[tuple[float, str]],
                                  *,
                                  target_rate_hz: float = DDTW_TARGET_RATE_HZ,
                                  window_s: float = DDTW_WINDOW_S,
                                  stride_s: float = DDTW_STRIDE_S,
                                  n_surrogates: int = DDTW_N_SURROGATES,
                                  seed: int = 42,
                                  compute_path_features: bool = False,
                                  fs: float | None = None,
                                  ) -> dict:
    """Compute DDTW coupling z-timecourse for one session.

    Args:
        p1_pca_uniform, p2_pca_uniform: (N_uniform, D) feature streams
            resampled to target_rate_hz — PCA-projected pose (Phase 0) or
            angle features from ``build_angle_features`` (Phase 1).
        p1_valid_uniform, p2_valid_uniform: (N_uniform,) per-frame validity.
        ts_uniform: (N_uniform,) LSL timestamps (uniform spacing).
        markers: per-session digest markers.
        target_rate_hz, window_s, stride_s, n_surrogates: see module constants.
        seed: RNG seed for surrogates.
        compute_path_features: also run ``sliding_ddtw_path_features`` on the
            real pair (Phase 1). Off by default — the Phase 0 output is then
            unchanged.
        fs: frame rate used to convert path lags to seconds; ``None`` ->
            ``target_rate_hz`` (the two coincide for uniform-grid input).

    Returns:
        dict with:
          ddtw_z: (n_strides,) float32 surrogate-z per stride position.
          ddtw_real: (n_strides,) float32 raw DDTW score (real pair).
          surr_mean, surr_std: (n_strides,) float32 surrogate stats.
          stride_ts: (n_strides,) float64 stride center timestamps in LSL.
          n_strides: int.
          ddtw_lag_s, ddtw_lag_var, ddtw_asym: (n_strides,) float32 — only
            when ``compute_path_features``; positive lag = P2 later.
    """
    window_frames = int(round(window_s * target_rate_hz))
    stride_frames = int(round(stride_s * target_rate_hz))
    if window_frames <= 1 or stride_frames <= 0:
        raise ValueError(f'invalid window/stride frames: {window_frames}/{stride_frames}')
    lag_fs = float(target_rate_hz if fs is None else fs)

    real, starts = sliding_ddtw_real(p1_pca_uniform, p2_pca_uniform,
                                      window_frames, stride_frames,
                                      p1_valid_uniform, p2_valid_uniform)
    n_strides = starts.size
    if n_strides == 0:
        out = {
            'ddtw_z': np.empty(0, dtype=np.float32),
            'ddtw_real': real,
            'surr_mean': np.empty(0, dtype=np.float32),
            'surr_std': np.empty(0, dtype=np.float32),
            'stride_ts': np.empty(0, dtype=np.float64),
            'n_strides': 0,
        }
        if compute_path_features:
            for k in ('ddtw_lag_s', 'ddtw_lag_var', 'ddtw_asym'):
                out[k] = np.empty(0, dtype=np.float32)
        return out

    # Stride center timestamps (LSL)
    half = window_frames // 2
    stride_ts = ts_uniform[np.minimum(starts + half, ts_uniform.size - 1)]

    # Build condition blocks for surrogate construction
    _, blocks = assign_frame_conditions(ts_uniform, markers)

    # Welford-style accumulator for per-stride surrogate stats
    surr_mean = np.zeros(n_strides, dtype=np.float64)
    surr_m2 = np.zeros(n_strides, dtype=np.float64)
    n_seen = np.zeros(n_strides, dtype=np.int32)   # per-stride count of finite surr scores

    rng = np.random.default_rng(seed)

    for si in range(n_surrogates):
        # Standard CADENCE pattern: session-wide circular shift in [10%, 90%]
        # of session length. See session_circular_shift docstring for rationale
        # vs the spec's condition-aligned alternative.
        p2_surr, v2_surr = session_circular_shift(p2_pca_uniform,
                                                    p2_valid_uniform, rng)

        surr_scores, _ = sliding_ddtw_real(p1_pca_uniform, p2_surr,
                                            window_frames, stride_frames,
                                            p1_valid_uniform, v2_surr)
        # Welford update on finite entries only
        finite = np.isfinite(surr_scores)
        idx = np.where(finite)[0]
        for j in idx:
            n_seen[j] += 1
            x = float(surr_scores[j])
            delta = x - surr_mean[j]
            surr_mean[j] += delta / n_seen[j]
            delta2 = x - surr_mean[j]
            surr_m2[j] += delta * delta2

    # Finalize per-stride z
    surr_std = np.sqrt(surr_m2 / np.maximum(n_seen - 1, 1))
    surr_std_safe = np.maximum(surr_std, 1e-8)
    z = np.full(n_strides, np.nan, dtype=np.float32)
    valid = (n_seen >= 5) & np.isfinite(real)
    z[valid] = ((real[valid] - surr_mean[valid]) / surr_std_safe[valid]).astype(np.float32)

    out = {
        'ddtw_z': z,
        'ddtw_real': real.astype(np.float32),
        'surr_mean': surr_mean.astype(np.float32),
        'surr_std': surr_std.astype(np.float32),
        'stride_ts': stride_ts.astype(np.float64),
        'n_strides': int(n_strides),
        'n_surrogates_per_stride': n_seen.astype(np.int32),
    }
    if compute_path_features:
        pf = sliding_ddtw_path_features(p1_pca_uniform, p2_pca_uniform,
                                        window_frames, stride_frames,
                                        p1_valid_uniform, p2_valid_uniform,
                                        lag_fs)
        out['ddtw_lag_s'] = pf['lag_s']
        out['ddtw_lag_var'] = pf['lag_var']
        out['ddtw_asym'] = pf['asym']
    return out


# ── Convenience: full per-session pipeline (load -> project -> compute) ───

def run_session_ddtw(session_id: str, components: np.ndarray | None,
                      mean: np.ndarray | None,
                      *,
                      preproc_root: str | Path = 'data/preproc/pose/v1',
                      digest_root: str | Path = 'data/digest/v1',
                      target_rate_hz: float = DDTW_TARGET_RATE_HZ,
                      window_s: float = DDTW_WINDOW_S,
                      stride_s: float = DDTW_STRIDE_S,
                      n_surrogates: int = DDTW_N_SURROGATES,
                      seed: int = 42,
                      feature_mode: str = 'pca',
                      noise_normalize: bool = True,
                      compute_path_features: bool = False,
                      ) -> dict:
    """Top-level per-session DDTW pipeline.

    Loads pose preproc + digest markers, resamples to uniform rate, builds the
    feature stream for ``feature_mode``, runs DDTW + surrogates, returns the
    result dict extended with provenance fields.

    ``feature_mode='pca'`` (default, Phase 0): projects the 99-D positions
    into the provided shared PCA (``components``, ``mean`` required).

    ``feature_mode in ('angles', 'angle_speed')`` (Phase 1): ``components`` /
    ``mean`` may be ``None``. The raw (N, 33, 4) pose is resampled onto the
    same 12 Hz common grid as the PCA path (``resample_to_uniform_rate`` on
    the flattened (N, 132) view, then the per-participant grids are snapped
    onto the common grid exactly as for PCA); the visibility column is then
    replaced by ``resample_pose33_visibility`` (minimum of the two bracketing
    native frames — see its docstring). Angles / speeds are computed on the
    uniform grid (``build_angle_features``), and per-frame validity is
    ``pose_features_valid AND angle valid``.

    Additional keys in every mode: ``feature_mode``, ``n_features``,
    ``noise_floor`` — ``(2, 12)`` float32 (rows P1, P2; radians for
    'angles', deg/s for 'angle_speed') or ``None`` for 'pca' — and
    ``noise_normalize``. With ``compute_path_features`` the dict also has
    ``ddtw_lag_s`` / ``ddtw_lag_var`` / ``ddtw_asym``.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f'run_session_ddtw: feature_mode must be one of {FEATURE_MODES}; '
                         f'got {feature_mode!r}')
    angle_mode = feature_mode in ANGLE_MODES
    if not angle_mode and (components is None or mean is None):
        raise ValueError("run_session_ddtw: feature_mode='pca' requires components and mean")
    streams = load_pose_streams(Path(preproc_root) / f'{session_id}.npz',
                                include_pose33=angle_mode)
    digest = json.loads((Path(digest_root) / f'{session_id}.json').read_text())
    digest_md5 = digest.get('xdf_md5', '')
    # Markers in digest JSON are in ABSOLUTE LSL clock; stream timestamps in
    # NPZ are session-relative seconds (range [0, duration_s]). Convert
    # markers to the stream time base by subtracting t_start_lsl.
    t_start_lsl = float(digest.get('t_start_lsl', 0.0))
    markers = [(float(t) - t_start_lsl, str(label))
               for t, label in digest.get('markers', [])]

    # Resample p1, p2 separately to uniform rate, then snap onto a common grid
    p1_uni, ts1 = resample_to_uniform_rate(streams['p1_pose'], streams['p1_ts'],
                                             target_rate_hz)
    p2_uni, ts2 = resample_to_uniform_rate(streams['p2_pose'], streams['p2_ts'],
                                             target_rate_hz)
    # Common grid spans the intersection of both participants' time ranges
    if ts1.size == 0 or ts2.size == 0:
        raise RuntimeError(f'{session_id}: empty pose stream after resample')
    t0 = max(ts1[0], ts2[0])
    t1 = min(ts1[-1], ts2[-1])
    if t1 <= t0:
        raise RuntimeError(f'{session_id}: no temporal overlap between p1/p2 pose')
    ts_common = np.arange(t0, t1, 1.0 / target_rate_hz, dtype=np.float64)
    # Re-interp each onto common grid (cheap; already uniform)
    def _onto_common(arr_uni, ts_uni):
        out = np.empty((ts_common.size, arr_uni.shape[1]), dtype=np.float32)
        for d in range(arr_uni.shape[1]):
            out[:, d] = np.interp(ts_common, ts_uni, arr_uni[:, d]).astype(np.float32)
        return out
    if not angle_mode:
        p1_common = _onto_common(p1_uni, ts1)
        p2_common = _onto_common(p2_uni, ts2)
    v1_common = resample_validity_to_uniform(streams['p1_valid'], streams['p1_ts'],
                                              ts_common)
    v2_common = resample_validity_to_uniform(streams['p2_valid'], streams['p2_ts'],
                                              ts_common)

    noise_floor_out = None
    if angle_mode:
        # Phase 1: resample the raw (N, 33, 4) pose onto the same common grid
        # (flattened (N, 132) view through the identical two-step path), then
        # restore a conservative visibility column and compute angle features
        # at the fixed grid rate.
        def _pose33_onto_common(pose33, ts):
            flat = pose33.reshape(pose33.shape[0], POSE_FLAT_DIM_VIS)
            flat_uni, ts_uni = resample_to_uniform_rate(flat, ts, target_rate_hz)
            flat_common = _onto_common(flat_uni, ts_uni)
            out33 = flat_common.reshape(ts_common.size, POSE_KEYPOINTS, POSE_COLS)
            out33[:, :, 3] = resample_pose33_visibility(pose33, ts, ts_common)
            return out33
        p1_33 = _pose33_onto_common(streams['p1_pose33'], streams['p1_ts'])
        p2_33 = _pose33_onto_common(streams['p2_pose33'], streams['p2_ts'])
        p1_feat, a1_valid, info1 = build_angle_features(p1_33, ts_common, feature_mode,
                                                        noise_normalize=noise_normalize)
        p2_feat, a2_valid, info2 = build_angle_features(p2_33, ts_common, feature_mode,
                                                        noise_normalize=noise_normalize)
        v1_common = v1_common & a1_valid
        v2_common = v2_common & a2_valid
        noise_floor_out = np.stack([info1['noise_floor'], info2['noise_floor']]
                                   ).astype(np.float32)
        n_features = int(info1['n_features'])
    else:
        # Project to shared PCA subspace
        p1_feat = project_pca(p1_common, components, mean)
        p2_feat = project_pca(p2_common, components, mean)
        n_features = int(components.shape[0])

    out = compute_session_ddtw(p1_feat, p2_feat, v1_common, v2_common, ts_common,
                                markers, target_rate_hz=target_rate_hz,
                                window_s=window_s, stride_s=stride_s,
                                n_surrogates=n_surrogates, seed=seed,
                                compute_path_features=compute_path_features,
                                fs=target_rate_hz)
    out['feature_mode'] = feature_mode
    out['noise_floor'] = noise_floor_out
    out['noise_normalize'] = bool(noise_normalize) if angle_mode else False
    out['n_features'] = n_features
    out['session_id'] = session_id
    out['digest_xdf_md5'] = digest_md5
    out['preproc_pose_md5'] = streams['digest_xdf_md5']  # also xdf_md5
    out['pose_format_in'] = streams['pose_format_in']
    return out
