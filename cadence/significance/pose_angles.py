"""Torso-frame segment angles from MediaPipe-33 pose — Phase 1 pose features.

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md (Task 1).

The Phase 0 pose channel (``pose_ddtw``) warps raw 99-D landmark positions
after a shared PCA. Those positions live in *different camera frames* for
the two participants (viewpoint, scale, limb proportion), and mix
image-normalised MediaPipe coordinates with RTMW pixel coordinates across
sessions. This module replaces positions with **12 torso-frame segment
angles** that are invariant to translation, scale, and limb proportion and
need no divisor.

Per frame the torso frame is:

  origin = mid_hip
  u      = unit(mid_shoulder - mid_hip)         (torso axis, "up")
  r      = perp(u) = (-u_y, u_x)                (torso-right axis)

Each body segment ``v = p[to] - p[from]`` is expressed as the signed angle
``arctan2(v . r, v . u)`` in (-pi, pi]: 0 = parallel to the torso axis,
+pi/2 = along ``r``. ``torso_lean`` is the exception — it is the torso axis
measured against image vertical, ``arctan2(u_x, -u_y)`` (image y points
down, so an upright torso has u = (0, -1) and lean 0; rotating the whole
body by phi with the matrix [[cos, -sin], [sin, cos]] changes the lean by
+phi and nothing else).

Only x and y (columns 0, 1) are used: RTMW sessions have z = 0 and MediaPipe
z is the noisiest axis. A feature is NaN whenever either endpoint is hidden
(visibility <= 0.5, matching ``cadence.preprocess.pose.pipeline``) or the
segment is degenerate; a frame is all-NaN and invalid when the torso frame
itself is undefined (hips / shoulders hidden or torso too short).

Also provided: NaN-aware ``np.unwrap``, a NaN-ignoring moving average,
angular speed (deg/s), a per-feature measured **noise floor** (robust SD of
the residual against a 3-frame moving average — dance_sync's trick for
expressing DTW cost in noise-SD units instead of raw radians), and a
weighted whole-body **speed envelope** that feeds the landing / movement
event channel (``pose_event_coincidence``).

Consumers: ``pose_ddtw`` (feature modes ``angles`` / ``angle_speed``) and
``pose_event_coincidence`` (``pose33_to_angle_stream``, ``speed_envelope``).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


# ── Constants ───────────────────────────────────────────────────────

VISIBILITY_THRESHOLD = 0.5    # matches cadence.preprocess.pose.pipeline.VISIBILITY_THRESHOLD
MIN_SEGMENT_LEN = 0.01        # in the pose's own (x, y) units; shorter segments -> NaN feature
MIN_TORSO_LEN = 0.02          # shorter torso -> frame undefined -> all-NaN row, valid=False

# MediaPipe-33 landmark indices (populated for all three supported pose formats)
LM_NOSE = 0
LM_L_EAR, LM_R_EAR = 7, 8
LM_L_SHOULDER, LM_R_SHOULDER = 11, 12
LM_L_ELBOW, LM_R_ELBOW = 13, 14
LM_L_WRIST, LM_R_WRIST = 15, 16
LM_L_HIP, LM_R_HIP = 23, 24
LM_L_KNEE, LM_R_KNEE = 25, 26
LM_L_ANKLE, LM_R_ANKLE = 27, 28

# Virtual points = midpoint of a landmark pair; visible only when BOTH are.
VIRTUAL_POINTS: dict[str, tuple[int, int]] = {
    'mid_hip': (LM_L_HIP, LM_R_HIP),
    'mid_shoulder': (LM_L_SHOULDER, LM_R_SHOULDER),
    'mid_ear': (LM_L_EAR, LM_R_EAR),
}

# (name, tier, from_landmark, to_landmark) — order is fixed; downstream
# modules index columns by position.
ANGLE_FEATURES: tuple[tuple[str, str, str | int, str | int], ...] = (
    ('torso_lean',    'torso',         'mid_hip',      'mid_shoulder'),
    ('neck',          'neck',          'mid_shoulder', 'mid_ear'),
    ('head_twist',    'head_twist',    LM_R_EAR,       LM_L_EAR),   # ear line: longer/steadier than mid_ear->nose
    ('shoulder_line', 'shoulder_line', LM_L_SHOULDER,  LM_R_SHOULDER),
    ('l_thigh',       'thigh',         LM_L_HIP,       LM_L_KNEE),
    ('r_thigh',       'thigh',         LM_R_HIP,       LM_R_KNEE),
    ('l_upper_arm',   'upper_arm',     LM_L_SHOULDER,  LM_L_ELBOW),
    ('r_upper_arm',   'upper_arm',     LM_R_SHOULDER,  LM_R_ELBOW),
    ('l_forearm',     'forearm',       LM_L_ELBOW,     LM_L_WRIST),
    ('r_forearm',     'forearm',       LM_R_ELBOW,     LM_R_WRIST),
    ('l_shin',        'shin',          LM_L_KNEE,      LM_L_ANKLE),
    ('r_shin',        'shin',          LM_R_KNEE,      LM_R_ANKLE),
)
ANGLE_NAMES: list[str] = [f[0] for f in ANGLE_FEATURES]
ANGLE_TIERS: list[str] = [f[1] for f in ANGLE_FEATURES]
N_ANGLE_FEATURES = len(ANGLE_FEATURES)   # 12

# tier -> weight (dance_sync defaults): posture-bearing segments dominate,
# distal segments (forearm, shin) are down-weighted as noisier / more
# task-incidental.
ANGLE_WEIGHTS: dict[str, float] = {
    'torso': 1.0,
    'thigh': 1.0,
    'neck': 1.0,
    'shoulder_line': 0.8,
    'head_twist': 0.6,
    'upper_arm': 0.45,
    'forearm': 0.2,
    'shin': 0.2,
}

MIN_FINITE_FOR_VALID = 6      # valid frame needs >= 6 of 12 finite features
MIN_FINITE_FOR_ENVELOPE = 3   # envelope needs >= 3 finite speed features

_TWO_PI = 2.0 * np.pi
_RAD2DEG = 180.0 / np.pi
_MAD_TO_SD = 1.4826           # Gaussian MAD -> SD consistency factor
_NOISE_FLOOR_MIN = 1e-4       # radians


def feature_weights() -> np.ndarray:
    """(12,) per-feature weights from ``ANGLE_WEIGHTS`` by tier, normalised to sum 1."""
    w = np.array([ANGLE_WEIGHTS[t] for t in ANGLE_TIERS], dtype=np.float64)
    return w / w.sum()


# ── Landmark resolution ─────────────────────────────────────────────

def _resolve_point(key: str | int, xy: np.ndarray,
                   seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ((N,2) coordinates, (N,) visibility mask) for a landmark index or virtual name."""
    if isinstance(key, str):
        a, b = VIRTUAL_POINTS[key]
        pt = 0.5 * (xy[:, a] + xy[:, b])
        ok = seen[:, a] & seen[:, b]
        return pt, ok
    return xy[:, key], seen[:, key]


def _safe_unit(v: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Divide (N,2) vectors by (N,) lengths; zero where length is not positive/finite."""
    good = np.isfinite(length) & (length > 0)
    out = np.zeros_like(v)
    np.divide(v, length[:, None], out=out, where=good[:, None])
    return out


# ── Angle features ──────────────────────────────────────────────────

def angle_features(pose33: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose33 (N,33,4) -> (angles (N,12) float32 radians, valid (N,) bool).

    Torso frame per frame: origin mid-hip, u = unit(mid_shoulder - mid_hip),
    r = perp(u) = (-u_y, u_x). Each segment direction v = p[to] - p[from];
    angle = arctan2(v.r, v.u), in (-pi, pi]. ``torso_lean`` is measured
    against image vertical: arctan2(u_x, -u_y) (signed, 0 = upright).

    NaN for a feature when either endpoint has vis <= VISIBILITY_THRESHOLD
    (virtual midpoints need both constituent landmarks), when any endpoint
    coordinate is non-finite, or when |v| < MIN_SEGMENT_LEN. All-NaN row and
    valid=False when torso length < MIN_TORSO_LEN or hips/shoulders hidden.
    valid[i] is True iff the torso frame is defined AND at least 6 of the 12
    features are finite. Uses only x, y (columns 0, 1).
    """
    pose33 = np.asarray(pose33)
    if pose33.ndim != 3 or pose33.shape[1] != 33 or pose33.shape[2] < 4:
        raise ValueError(f'pose33 must be (N, 33, 4); got {pose33.shape}')
    n = pose33.shape[0]
    xy = pose33[:, :, :2].astype(np.float64, copy=False)
    vis = pose33[:, :, 3].astype(np.float64, copy=False)
    seen = (np.isfinite(vis) & (vis > VISIBILITY_THRESHOLD)
            & np.isfinite(xy).all(axis=2))                       # (N, 33)

    # Torso frame
    mid_hip, ok_hip = _resolve_point('mid_hip', xy, seen)
    mid_sh, ok_sh = _resolve_point('mid_shoulder', xy, seen)
    u_raw = mid_sh - mid_hip
    torso_len = np.hypot(u_raw[:, 0], u_raw[:, 1])
    frame_ok = ok_hip & ok_sh & (torso_len >= MIN_TORSO_LEN)
    u = _safe_unit(u_raw, torso_len)                             # (N, 2)
    r = np.stack([-u[:, 1], u[:, 0]], axis=1)                    # (N, 2)

    angles = np.full((n, N_ANGLE_FEATURES), np.nan, dtype=np.float64)
    for i, (name, _tier, src, dst) in enumerate(ANGLE_FEATURES):
        if name == 'torso_lean':
            ang = np.arctan2(u[:, 0], -u[:, 1])
            feat_ok = frame_ok
        else:
            p_from, ok_from = _resolve_point(src, xy, seen)
            p_to, ok_to = _resolve_point(dst, xy, seen)
            v = p_to - p_from
            v_len = np.hypot(v[:, 0], v[:, 1])
            feat_ok = frame_ok & ok_from & ok_to & (v_len >= MIN_SEGMENT_LEN)
            ang = np.arctan2(v[:, 0] * r[:, 0] + v[:, 1] * r[:, 1],
                             v[:, 0] * u[:, 0] + v[:, 1] * u[:, 1])
        # arctan2 returns [-pi, pi]; fold the -pi edge into (-pi, pi]
        ang = np.where(ang <= -np.pi, ang + _TWO_PI, ang)
        angles[feat_ok, i] = ang[feat_ok]

    n_finite = np.isfinite(angles).sum(axis=1)
    valid = frame_ok & (n_finite >= MIN_FINITE_FOR_VALID)
    return angles.astype(np.float32), valid


# ── NaN-aware unwrap / smoothing ────────────────────────────────────

def unwrap_nan(angles: np.ndarray) -> np.ndarray:
    """Column-wise ``np.unwrap`` that chooses the branch across NaN gaps and restores NaN.

    Finite samples of each column are unwrapped as one compacted sequence
    (equivalent to interpolating the phase across each gap under the
    minimal-jump assumption), so consecutive finite neighbours — even across
    a gap — never differ by more than pi. NaN positions stay NaN. Accepts
    (N,) or (N, D); returns float64 of the same shape.
    """
    a = np.asarray(angles, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    out = np.full(a.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    for j in range(a.shape[1]):
        m = finite[:, j]
        if not m.any():
            continue
        out[m, j] = np.unwrap(a[m, j])
    return out[:, 0] if squeeze else out


def moving_average_nan(x: np.ndarray, window: int) -> np.ndarray:
    """Centered, edge-padded, NaN-ignoring moving average along axis 0.

    ``x`` is (N,) or (N, D). Each output sample is the mean of the finite
    values inside the window (edge samples see the boundary value repeated);
    NaN where the window holds no finite value. ``window <= 1`` returns a
    float64 copy. For even ``window`` the centre sits half a sample early
    (scipy ``uniform_filter1d`` convention).
    """
    x = np.asarray(x, dtype=np.float64)
    window = int(window)
    if window <= 1 or x.shape[0] == 0:
        return x.copy()
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    # uniform_filter1d(mode='nearest') == edge padding; multiply back by the
    # window to turn the mean into a sum so the NaN-ignoring ratio is exact.
    num = uniform_filter1d(filled, size=window, axis=0, mode='nearest') * window
    cnt = uniform_filter1d(finite.astype(np.float64), size=window, axis=0,
                           mode='nearest') * window
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = cnt > 0.5
    np.divide(num, cnt, out=out, where=good)
    return out


# ── Angular speed, noise floor, envelope ────────────────────────────

def speed_features(angles: np.ndarray, fs: float,
                   smooth_frames: int = 5) -> np.ndarray:
    """(N,12) unwrapped-smoothed angular speed magnitude in deg/s.

    unwrap_nan -> moving_average_nan(smooth_frames) -> |np.gradient| * fs,
    converted to degrees. NaN propagates from ``np.gradient`` to the
    immediate neighbours of a NaN gap.
    """
    a = np.asarray(angles, dtype=np.float64)
    if a.shape[0] < 2:
        return np.full(a.shape, np.nan, dtype=np.float64)
    smooth = moving_average_nan(unwrap_nan(a), smooth_frames)
    vel = np.gradient(smooth, axis=0) * float(fs)
    return np.abs(vel) * _RAD2DEG


def noise_floor(angles: np.ndarray, window: int = 3) -> np.ndarray:
    """(12,) per-feature noise SD in radians.

    ``1.4826 * median |unwrapped - moving_average_nan(unwrapped, window)|``
    over finite samples only; floored at 1e-4; all-NaN features -> 1e-4.
    Mirrors dance_sync's measured noise floor; for white noise the residual
    against a ``window``-point mean has SD ``sigma * sqrt((window-1)/window)``
    (~0.82 sigma at window=3), which is retained as-is so the constant is
    comparable across modules that share this definition.
    """
    a = np.asarray(angles, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    d = a.shape[1]
    out = np.full(d, _NOISE_FLOOR_MIN, dtype=np.float64)
    if a.shape[0] == 0:
        return out
    unwrapped = unwrap_nan(a)
    resid = np.abs(unwrapped - moving_average_nan(unwrapped, window))
    finite = np.isfinite(resid)
    has = finite.any(axis=0)
    if has.any():
        med = np.nanmedian(np.where(finite, resid, np.nan)[:, has], axis=0)
        out[has] = np.maximum(_MAD_TO_SD * med, _NOISE_FLOOR_MIN)
    return out


def speed_envelope(angles: np.ndarray, fs: float, smooth_frames: int = 5,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """(N,) weighted NaN-aware mean of ``speed_features`` (deg/s).

    Weights default to ``feature_weights()``; per frame the weights of the
    finite features are renormalised. NaN where fewer than 3 features are
    finite.
    """
    speed = speed_features(angles, fs, smooth_frames=smooth_frames)
    w = feature_weights() if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (speed.shape[1],):
        raise ValueError(f'weights must be ({speed.shape[1]},); got {w.shape}')
    finite = np.isfinite(speed)
    num = np.where(finite, speed, 0.0) @ w
    den = finite.astype(np.float64) @ w
    out = np.full(speed.shape[0], np.nan, dtype=np.float64)
    good = (finite.sum(axis=1) >= MIN_FINITE_FOR_ENVELOPE) & (den > 0)
    np.divide(num, den, out=out, where=good)
    return out


# ── Convenience: pose33 -> angle stream dict ────────────────────────

def estimate_fs(ts: np.ndarray, fs_hint: float = 30.0) -> float:
    """Sampling rate from the median timestamp step; ``fs_hint`` if undeterminable."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        return float(fs_hint)
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float(fs_hint)
    return float(1.0 / np.median(dt))


def pose33_to_angle_stream(pose33: np.ndarray, ts: np.ndarray,
                           fs_hint: float = 30.0) -> dict:
    """Convenience bundle for one participant's pose stream.

    Returns ``{'angles': (N,12) float32 unwrapped radians, 'valid': (N,) bool,
    'speed': (N,12) float32 deg/s, 'envelope': (N,) float32,
    'noise_floor': (12,) float32, 'fs': float}`` with ``fs`` estimated from
    ``ts`` (median step) and ``fs_hint`` as the fallback.

    ``speed`` / ``envelope`` are NaN in the interior of an invalid stretch
    but bleed up to ``smooth_frames // 2`` frames into its edges (the
    NaN-ignoring moving average smooths across short dropouts on purpose);
    consumers that need strict masking should apply ``valid`` themselves.
    """
    fs = estimate_fs(ts, fs_hint)
    raw, valid = angle_features(pose33)
    unwrapped = unwrap_nan(raw)
    return {
        'angles': unwrapped.astype(np.float32),
        'valid': valid,
        'speed': speed_features(unwrapped, fs).astype(np.float32),
        'envelope': speed_envelope(unwrapped, fs).astype(np.float32),
        'noise_floor': noise_floor(unwrapped).astype(np.float32),
        'fs': fs,
    }
