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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dtaidistance.dtw_ndim import distance_fast as _ddtw_distance

from cadence.io.resources import limit_blas_threads


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

# Condition order for condition-block surrogate construction (matches V11/v82)
CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1',
                   'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']


# ── Pose loading + alignment ────────────────────────────────────────

def load_pose_streams(preproc_path: str | Path) -> dict:
    """Load pose preproc artifact and extract DDTW-ready arrays.

    Returns:
        dict with keys:
          p1_pose, p2_pose: (N_p, 99) float32 position streams (visibility dropped)
          p1_ts, p2_ts:     (N_p,) LSL timestamps (float64)
          p1_valid, p2_valid: (N_p,) bool per-frame validity
          digest_xdf_md5:   provenance hash
          pose_format_in:   one of {mediapipe33, mediapipe33_meta, wholebody_133}
    """
    preproc_path = Path(preproc_path)
    npz = np.load(preproc_path, allow_pickle=False)
    sidecar = json.loads(preproc_path.with_suffix('.json').read_text())

    out = {}
    for p in ('p1', 'p2'):
        pose33 = npz[f'{p}_pose33']                  # (N, 33, 4)
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
                          ) -> dict:
    """See `_compute_session_ddtw_inner` — wrapper pins BLAS to 1 thread to
    prevent oversubscription when called from a joblib worker pool. Outside a
    pool, `limit_blas_threads(1)` is essentially free (one re-entry)."""
    with limit_blas_threads(1):
        return _compute_session_ddtw_inner(
            p1_pca_uniform, p2_pca_uniform, p1_valid_uniform, p2_valid_uniform,
            ts_uniform, markers,
            target_rate_hz=target_rate_hz, window_s=window_s, stride_s=stride_s,
            n_surrogates=n_surrogates, seed=seed)


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
                                  ) -> dict:
    """Compute DDTW coupling z-timecourse for one session.

    Args:
        p1_pca_uniform, p2_pca_uniform: (N_uniform, n_components) PCA-projected
            pose, resampled to target_rate_hz.
        p1_valid_uniform, p2_valid_uniform: (N_uniform,) per-frame validity.
        ts_uniform: (N_uniform,) LSL timestamps (uniform spacing).
        markers: per-session digest markers.
        target_rate_hz, window_s, stride_s, n_surrogates: see module constants.
        seed: RNG seed for surrogates.

    Returns:
        dict with:
          ddtw_z: (n_strides,) float32 surrogate-z per stride position.
          ddtw_real: (n_strides,) float32 raw DDTW score (real pair).
          surr_mean, surr_std: (n_strides,) float32 surrogate stats.
          stride_ts: (n_strides,) float64 stride center timestamps in LSL.
          n_strides: int.
    """
    window_frames = int(round(window_s * target_rate_hz))
    stride_frames = int(round(stride_s * target_rate_hz))
    if window_frames <= 1 or stride_frames <= 0:
        raise ValueError(f'invalid window/stride frames: {window_frames}/{stride_frames}')

    real, starts = sliding_ddtw_real(p1_pca_uniform, p2_pca_uniform,
                                      window_frames, stride_frames,
                                      p1_valid_uniform, p2_valid_uniform)
    n_strides = starts.size
    if n_strides == 0:
        return {
            'ddtw_z': np.empty(0, dtype=np.float32),
            'ddtw_real': real,
            'surr_mean': np.empty(0, dtype=np.float32),
            'surr_std': np.empty(0, dtype=np.float32),
            'stride_ts': np.empty(0, dtype=np.float64),
            'n_strides': 0,
        }

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

    return {
        'ddtw_z': z,
        'ddtw_real': real.astype(np.float32),
        'surr_mean': surr_mean.astype(np.float32),
        'surr_std': surr_std.astype(np.float32),
        'stride_ts': stride_ts.astype(np.float64),
        'n_strides': int(n_strides),
        'n_surrogates_per_stride': n_seen.astype(np.int32),
    }


# ── Convenience: full per-session pipeline (load -> project -> compute) ───

def run_session_ddtw(session_id: str, components: np.ndarray, mean: np.ndarray,
                      *,
                      preproc_root: str | Path = 'data/preproc/pose/v1',
                      digest_root: str | Path = 'data/digest/v1',
                      target_rate_hz: float = DDTW_TARGET_RATE_HZ,
                      window_s: float = DDTW_WINDOW_S,
                      stride_s: float = DDTW_STRIDE_S,
                      n_surrogates: int = DDTW_N_SURROGATES,
                      seed: int = 42,
                      ) -> dict:
    """Top-level per-session DDTW pipeline.

    Loads pose preproc + digest markers, resamples to uniform rate, projects
    into provided shared PCA, runs DDTW + surrogates, returns the result dict
    extended with provenance fields.
    """
    streams = load_pose_streams(Path(preproc_root) / f'{session_id}.npz')
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
    p1_common = _onto_common(p1_uni, ts1)
    p2_common = _onto_common(p2_uni, ts2)
    v1_common = resample_validity_to_uniform(streams['p1_valid'], streams['p1_ts'],
                                              ts_common)
    v2_common = resample_validity_to_uniform(streams['p2_valid'], streams['p2_ts'],
                                              ts_common)

    # Project to shared PCA subspace
    p1_pca = project_pca(p1_common, components, mean)
    p2_pca = project_pca(p2_common, components, mean)

    out = compute_session_ddtw(p1_pca, p2_pca, v1_common, v2_common, ts_common,
                                markers, target_rate_hz=target_rate_hz,
                                window_s=window_s, stride_s=stride_s,
                                n_surrogates=n_surrogates, seed=seed)
    out['session_id'] = session_id
    out['digest_xdf_md5'] = digest_md5
    out['preproc_pose_md5'] = streams['digest_xdf_md5']  # also xdf_md5
    out['pose_format_in'] = streams['pose_format_in']
    return out
