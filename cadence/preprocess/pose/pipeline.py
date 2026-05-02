"""Pose preprocessing pipeline (Step 6).

Per-participant pipeline:

1. Read digest. For each participant, dispatch raw pose to ``(N, 33, 4)`` via
   ``normalize_pose_to_mp33`` (handles mediapipe33 / mediapipe33_meta /
   wholebody_133 transparently).
2. Visibility-aware feature extraction (40 joint-group channels) — vectorized
   reductions that **return ``NaN`` for groups with no visible landmarks** and
   use NaN-safe statistics for downstream z-scoring. This is a deliberate
   behavioral correction over the legacy ``_centroid`` (which computed an
   unconditional mean of all listed landmarks regardless of visibility,
   biasing the head centroid toward origin when face landmarks were missing).
3. Append activity channel -> 41 channels.
4. Save ``data/preproc/pose/v1/<sid>.npz`` + ``.json``.

Output layout (per ``<sid>.npz``)::

    p1_pose33    : (N1, 33, 4)   float32   normalized (x, y, z, visibility)
    p1_pose33_ts : (N1,)         float64   session-relative seconds
    p1_pose_features : (N1, 41)  float32   joint-group + activity, z-scored
    p1_pose_features_valid : (N1,) bool   per-frame validity (>=10 keypoints visible)
    p2_*  (same)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from cadence.preprocess.common import (
    atomic_write_json,
    atomic_write_npz,
    compute_activity_channel,
    default_out_dir,
    staleness_check,
)
from cadence.preprocess.pose.pose_subset import normalize_pose_to_mp33

POSE_MODALITY_VERSION = "v1"
POSE_N_FEATURES = 40              # joint-group features (before activity append)
POSE_N_FEATURES_TOTAL = 41        # +1 activity channel
VISIBILITY_THRESHOLD = 0.5
MIN_KEYPOINTS_VALID = 10
NAN_GROUP = np.float32("nan")


# ---------------------------------------------------------------------------
# Visibility-aware vectorized helpers (deliberate behavioral fix)
# ---------------------------------------------------------------------------

def _centroid(coords: np.ndarray, vis: np.ndarray, lm_list: list[int]) -> np.ndarray:
    """Visibility-weighted mean of selected landmarks. Returns ``(N, 3)``.

    Frames with NO visible landmarks in the group return NaN (downstream
    nanmean/nanstd ignores them) — replaces the legacy unconditional mean
    that biased toward origin when landmarks were missing.
    """
    sub_coords = coords[:, lm_list, :]                   # (N, K, 3)
    sub_vis = vis[:, lm_list] > VISIBILITY_THRESHOLD     # (N, K)
    w = sub_vis.astype(coords.dtype)                     # (N, K)
    num = (sub_coords * w[..., None]).sum(axis=1)        # (N, 3)
    den = w.sum(axis=1, keepdims=True)                   # (N, 1)
    out = np.where(den > 0, num / np.maximum(den, 1.0), NAN_GROUP)
    return out


def _extent(coords: np.ndarray, vis: np.ndarray, lm_list: list[int]) -> np.ndarray:
    """Bounding-box diagonal across visible landmarks. Returns ``(N,)``.

    Frames with fewer than 2 visible landmarks return NaN.
    """
    sub = coords[:, lm_list, :]
    sub_vis = vis[:, lm_list] > VISIBILITY_THRESHOLD
    n_vis = sub_vis.sum(axis=1)
    masked = np.where(sub_vis[..., None], sub, np.nan)
    # Suppress "All-NaN slice" warnings for frames with zero visible landmarks;
    # the n_vis < 2 mask below sets those to NaN_GROUP regardless of the reduction.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        pmax = np.nanmax(masked, axis=1)
        pmin = np.nanmin(masked, axis=1)
    diag = np.linalg.norm(pmax - pmin, axis=1)
    diag[n_vis < 2] = NAN_GROUP
    return diag


def _angle(coords: np.ndarray, vis: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Triplet angle at vertex b. Returns ``(N,)`` radians; NaN if any vertex hidden."""
    valid = ((vis[:, a] > VISIBILITY_THRESHOLD) &
             (vis[:, b] > VISIBILITY_THRESHOLD) &
             (vis[:, c] > VISIBILITY_THRESHOLD))
    ba = coords[:, a, :] - coords[:, b, :]
    bc = coords[:, c, :] - coords[:, b, :]
    cos_a = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    )
    out = np.arccos(np.clip(cos_a, -1, 1))
    out[~valid] = NAN_GROUP
    return out


def _extension(coords: np.ndarray, vis: np.ndarray, a: int, b: int) -> np.ndarray:
    """Distance between two landmarks. Returns ``(N,)``; NaN if either hidden."""
    valid = ((vis[:, a] > VISIBILITY_THRESHOLD) &
             (vis[:, b] > VISIBILITY_THRESHOLD))
    out = np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1)
    out[~valid] = NAN_GROUP
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_pose_features(pose33: np.ndarray, ts: np.ndarray, srate_hint: float = 30.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, frame_valid)``.

    features : (N, 41) float32 — 40 joint-group channels + activity, z-scored
        with NaN-safe statistics. NaN sentinels are filled with 0 after z-score
        so downstream consumers see numeric arrays.
    frame_valid : (N,) bool — at least ``MIN_KEYPOINTS_VALID`` keypoints visible.
    """
    if pose33.ndim != 3 or pose33.shape[1] != 33 or pose33.shape[2] != 4:
        raise ValueError(f"pose33 must be (N, 33, 4); got {pose33.shape}")
    coords = pose33[:, :, :3].astype(np.float32, copy=False)   # (N, 33, 3)
    vis = pose33[:, :, 3].astype(np.float32, copy=False)        # (N, 33)
    n = pose33.shape[0]

    frame_valid = (vis > VISIBILITY_THRESHOLD).sum(axis=1) >= MIN_KEYPOINTS_VALID

    feats = np.full((n, POSE_N_FEATURES), NAN_GROUP, dtype=np.float32)
    col = 0

    # --- Head (8): centroid(3), extent(1), tilt(3), rotation(1) ---
    head_lm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    feats[:, col:col + 3] = _centroid(coords, vis, head_lm); col += 3
    feats[:, col] = _extent(coords, vis, head_lm); col += 1
    ear_valid = (vis[:, 7] > VISIBILITY_THRESHOLD) & (vis[:, 8] > VISIBILITY_THRESHOLD)
    nose_valid = vis[:, 0] > VISIBILITY_THRESHOLD
    tilt_valid = ear_valid & nose_valid
    ear_mid = (coords[:, 7, :] + coords[:, 8, :]) / 2
    tilt = coords[:, 0, :] - ear_mid
    tilt[~tilt_valid] = NAN_GROUP
    feats[:, col:col + 3] = tilt; col += 3
    rot = coords[:, 8, 2] - coords[:, 7, 2]
    rot[~ear_valid] = NAN_GROUP
    feats[:, col] = rot; col += 1

    # --- L.Arm (5): centroid(3), elbow angle(1), extension(1) ---
    feats[:, col:col + 3] = _centroid(coords, vis, [11, 13, 15]); col += 3
    feats[:, col] = _angle(coords, vis, 11, 13, 15); col += 1
    l_arm_ext = _extension(coords, vis, 11, 15)
    feats[:, col] = l_arm_ext; col += 1

    # --- R.Arm (5): centroid(3), elbow angle(1), extension(1) ---
    feats[:, col:col + 3] = _centroid(coords, vis, [12, 14, 16]); col += 3
    feats[:, col] = _angle(coords, vis, 12, 14, 16); col += 1
    r_arm_ext = _extension(coords, vis, 12, 16)
    feats[:, col] = r_arm_ext; col += 1

    # --- Torso (6): centroid(3), lateral lean(1), fwd lean(1), rotation(1) ---
    feats[:, col:col + 3] = _centroid(coords, vis, [11, 12, 23, 24]); col += 3
    shoulder_valid = (vis[:, 11] > VISIBILITY_THRESHOLD) & (vis[:, 12] > VISIBILITY_THRESHOLD)
    hip_valid = (vis[:, 23] > VISIBILITY_THRESHOLD) & (vis[:, 24] > VISIBILITY_THRESHOLD)
    torso_valid = shoulder_valid & hip_valid
    shoulder_mid = (coords[:, 11, :] + coords[:, 12, :]) / 2
    hip_mid = (coords[:, 23, :] + coords[:, 24, :]) / 2
    lean = shoulder_mid - hip_mid
    lean[~torso_valid] = NAN_GROUP
    feats[:, col] = lean[:, 0]; col += 1   # lateral
    feats[:, col] = lean[:, 2]; col += 1   # fwd
    shoulder_line = coords[:, 12, :] - coords[:, 11, :]
    hip_line = coords[:, 24, :] - coords[:, 23, :]
    rotation = np.arctan2(
        shoulder_line[:, 0] * hip_line[:, 2] - shoulder_line[:, 2] * hip_line[:, 0],
        shoulder_line[:, 0] * hip_line[:, 0] + shoulder_line[:, 2] * hip_line[:, 2] + 1e-8,
    )
    rotation[~torso_valid] = NAN_GROUP
    feats[:, col] = rotation; col += 1

    # --- L.Leg (5): centroid(3), knee angle(1), extension(1) ---
    feats[:, col:col + 3] = _centroid(coords, vis, [23, 25, 27, 29]); col += 3
    feats[:, col] = _angle(coords, vis, 23, 25, 27); col += 1
    l_leg_ext = _extension(coords, vis, 23, 27)
    feats[:, col] = l_leg_ext; col += 1

    # --- R.Leg (5): centroid(3), knee angle(1), extension(1) ---
    feats[:, col:col + 3] = _centroid(coords, vis, [24, 26, 28, 30]); col += 3
    feats[:, col] = _angle(coords, vis, 24, 26, 28); col += 1
    r_leg_ext = _extension(coords, vis, 24, 28)
    feats[:, col] = r_leg_ext; col += 1

    # --- Global (6): CoM(3), L/R symmetry(1), velocity(1), openness(1) ---
    com = _centroid(coords, vis, list(range(33)))
    feats[:, col:col + 3] = com; col += 3
    sym = l_arm_ext - r_arm_ext  # NaN propagates if either side hidden
    feats[:, col] = sym; col += 1

    if n > 1:
        dt = np.diff(ts, prepend=ts[0] - 1.0 / max(srate_hint, 1e-3))
        dt = np.clip(dt, 1e-3, 1.0)
        com_diff = np.diff(com, axis=0, prepend=com[:1])
        vel = np.linalg.norm(com_diff, axis=1) / dt
    else:
        vel = np.full(n, NAN_GROUP, dtype=np.float32)
    feats[:, col] = vel; col += 1

    openness = l_arm_ext + r_arm_ext + l_leg_ext + r_leg_ext
    feats[:, col] = openness; col += 1

    assert col == POSE_N_FEATURES, f"expected {POSE_N_FEATURES} cols, got {col}"

    # Z-score each feature with NaN-safe stats over valid frames only
    for ch in range(POSE_N_FEATURES):
        col_vals = feats[frame_valid, ch]
        col_vals = col_vals[np.isfinite(col_vals)]
        if len(col_vals) >= 100:
            mu = float(np.mean(col_vals))
            sigma = float(np.std(col_vals))
            if sigma > 1e-8:
                feats[:, ch] = (feats[:, ch] - mu) / sigma
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    feats = np.clip(feats, -10, 10)

    if n > 1 and ts[-1] > ts[0]:
        effective_hz = n / float(ts[-1] - ts[0])
    else:
        effective_hz = srate_hint
    activity = compute_activity_channel(feats, effective_hz, trailing_seconds=30.0)
    feats_with_act = np.concatenate([feats, activity.astype(np.float32)], axis=1)
    return feats_with_act, frame_valid


# ---------------------------------------------------------------------------
# Per-session entry point
# ---------------------------------------------------------------------------

def preprocess_pose_session(session_id: str,
                            *,
                            digest_dir: str | Path = "data/digest/v1",
                            out_dir: str | Path | None = None,
                            force: bool = False) -> dict:
    """Preprocess one session's pose streams. Returns a small summary dict."""
    from cadence.ingest.digest import load_digest

    digest_dir = Path(digest_dir)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir("pose", POSE_MODALITY_VERSION)
    npz_path = out_dir / f"{session_id}.npz"
    json_path = out_dir / f"{session_id}.json"

    cs = load_digest(session_id, digest_dir=digest_dir)
    if not force and staleness_check(json_path, cs.xdf_md5):
        return {"session_id": session_id, "status": "skip-up-to-date",
                "out_npz": str(npz_path), "out_json": str(json_path)}

    arrays_out: dict[str, np.ndarray] = {}
    per_participant_summary: dict[str, dict] = {}

    for p in ("p1", "p2"):
        raw_key = f"{p}_pose_full"
        ts_key = f"{p}_pose_ts"
        if raw_key not in cs.arrays or ts_key not in cs.arrays:
            continue
        raw = cs.arrays[raw_key]
        ts = cs.arrays[ts_key]
        pose33 = normalize_pose_to_mp33(raw, cs.pose_format)         # (N, 33, 4)
        feats, frame_valid = extract_pose_features(pose33, ts)        # (N, 41) + (N,)

        arrays_out[f"{p}_pose33"] = pose33.astype(np.float32, copy=False)
        arrays_out[f"{p}_pose33_ts"] = ts.astype(np.float64, copy=False)
        arrays_out[f"{p}_pose_features"] = feats.astype(np.float32, copy=False)
        arrays_out[f"{p}_pose_features_valid"] = frame_valid.astype(bool)

        per_participant_summary[p] = {
            "n_frames": int(pose33.shape[0]),
            "valid_frames_pct": float(frame_valid.mean() * 100.0),
            "mean_visibility": float(pose33[:, :, 3].mean()),
        }

    if not arrays_out:
        raise RuntimeError(f"{session_id}: no pose streams in digest")

    sidecar = {
        "session_id": session_id,
        "modality": "pose",
        "modality_version": POSE_MODALITY_VERSION,
        "digest_xdf_md5": cs.xdf_md5,
        "digest_schema_version": cs.schema_version,
        "pose_format_in": cs.pose_format,
        "params": {
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "min_keypoints_valid": MIN_KEYPOINTS_VALID,
            "n_features": POSE_N_FEATURES_TOTAL,
            "feature_layout": (
                "head(8)+l_arm(5)+r_arm(5)+torso(6)+l_leg(5)+r_leg(5)+global(6)+activity(1)"
            ),
            "behavioral_change_vs_legacy": (
                "visibility-aware reductions: groups with no visible landmarks "
                "return NaN (filtered by nanmean/nanstd) instead of biased mean."
            ),
        },
        "participants": per_participant_summary,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    atomic_write_npz(npz_path, **arrays_out)
    atomic_write_json(json_path, sidecar)

    return {"session_id": session_id, "status": "ok",
            "out_npz": str(npz_path), "out_json": str(json_path),
            "summary": per_participant_summary}
