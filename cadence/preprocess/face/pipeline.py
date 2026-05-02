"""Face preprocessing pipeline (Step 6).

Reads the digest's ``{p}_landmarks_raw`` (N, 1489) — first 52 cols are MediaPipe
FaceLandmarker blendshapes (AUs). Produces:

    p{1,2}_au52        : (N, 52) float32   gap-filled + per-AU z-scored
    p{1,2}_au52_ts     : (N,)    float64   session-relative timestamps
    p{1,2}_au_valid    : (N,)    bool      face-detected mask
    p{1,2}_au_activity : (N,)    float32   trailing-mean RMS activity channel
    p{1,2}_au_v2       : (N, 31) float32   PCA(15) + deriv(15) + activity (V7)
    p{1,2}_au_v2_loadings : (15, 52) float32  PCA Vt rows

Pipeline (per participant):

1. Detect face-not-detected frames (all 52 AU channels ~ 0).
2. Linear-interp gaps shorter than 0.5s (sample-rate aware).
3. Z-score each AU using only valid frames.
4. Append RMS activity channel.
5. PCA-reduce 52 AUs to 15 PCs, append smoothed Gaussian derivatives, then
   activity → 31 channels.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cadence.preprocess.common import (
    atomic_write_json,
    atomic_write_npz,
    compute_activity_channel,
    compute_temporal_derivatives,
    default_out_dir,
    fill_short_gaps_linear,
    staleness_check,
    zscore_columns,
)

FACE_MODALITY_VERSION = "v1"
N_AUS = 52
GAP_FILL_SECONDS = 0.5
PCA_N_COMPONENTS = 15
DERIV_SIGMA_S = 0.5


def preprocess_au52(landmarks_raw: np.ndarray, ts: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(au52_z, valid, activity)``.

    au52_z : (N, 52) float32 — gap-filled, per-AU z-scored.
    valid  : (N,) bool — face detected (after gap-fill within max gap window).
    activity : (N, 1) float32 — RMS deviation from trailing 30s mean (ch 53).
    """
    n = landmarks_raw.shape[0]
    au = landmarks_raw[:, :N_AUS].astype(np.float64, copy=True)
    face_present = ~np.all(np.abs(au) < 1e-6, axis=1)

    if n > 1 and ts[-1] > ts[0]:
        srate = n / float(ts[-1] - ts[0])
    else:
        srate = 30.0
    max_gap = max(1, int(GAP_FILL_SECONDS * srate))
    au_filled = fill_short_gaps_linear(au, face_present, max_gap)

    # Reconstruct validity mask after gap-filling: True wherever original
    # was valid OR the gap was successfully filled.
    valid = face_present.copy()
    from cadence.preprocess.common import find_gaps
    starts, lengths = find_gaps(face_present)
    for s, ln in zip(starts, lengths):
        if ln <= max_gap and 0 < s and s + ln < n:
            valid[s:s + ln] = True

    au_z = zscore_columns(au_filled, sample_mask=valid)
    activity = compute_activity_channel(au_z.astype(np.float32), srate,
                                        trailing_seconds=30.0)
    return au_z.astype(np.float32), valid, activity.astype(np.float32)


def extract_au_v2(au52_z: np.ndarray, valid: np.ndarray, ts: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, pca_loadings)``.

    features : (N, 31) — PCA(15) + smoothed-derivatives(15) + activity(1).
    pca_loadings : (15, 52) — Vt rows of the SVD of valid-frame AUs.
    """
    n = au52_z.shape[0]
    valid_mask = valid if valid.sum() > PCA_N_COMPONENTS else np.ones(n, dtype=bool)
    au_valid = au52_z[valid_mask].astype(np.float64)
    mean = au_valid.mean(axis=0)
    centered = au_valid - mean
    # P2.3 — defensive guard: SVD calls into MKL/OpenBLAS and would explode
    # the BLAS thread count if face preprocessing is ever wrapped in joblib
    # parallelism. Currently sequential per-session, but the guard is cheap.
    from cadence.io.resources import limit_blas_threads
    with limit_blas_threads(1):
        _U, _s, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = (au52_z - mean) @ Vt[:PCA_N_COMPONENTS].T
    projected = np.clip(projected, -10, 10).astype(np.float32)

    if n > 1 and ts[-1] > ts[0]:
        srate = n / float(ts[-1] - ts[0])
    else:
        srate = 30.0
    derivatives = compute_temporal_derivatives(projected, srate, sigma_s=DERIV_SIGMA_S)
    derivatives = np.clip(derivatives, -10, 10).astype(np.float32)
    activity = compute_activity_channel(projected, srate, trailing_seconds=30.0).astype(np.float32)
    features = np.concatenate([projected, derivatives, activity], axis=1).astype(np.float32)
    pca_loadings = Vt[:PCA_N_COMPONENTS].astype(np.float32)
    return features, pca_loadings


def preprocess_face_session(session_id: str,
                            *,
                            digest_dir: str | Path = "data/digest/v1",
                            out_dir: str | Path | None = None,
                            force: bool = False) -> dict:
    from cadence.ingest.digest import load_digest

    digest_dir = Path(digest_dir)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir("face", FACE_MODALITY_VERSION)
    npz_path = out_dir / f"{session_id}.npz"
    json_path = out_dir / f"{session_id}.json"

    cs = load_digest(session_id, digest_dir=digest_dir)
    if not force and staleness_check(json_path, cs.xdf_md5):
        return {"session_id": session_id, "status": "skip-up-to-date"}

    arrays_out: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for p in ("p1", "p2"):
        raw_key = f"{p}_landmarks_raw"
        ts_key = f"{p}_landmarks_ts"
        if raw_key not in cs.arrays or ts_key not in cs.arrays:
            continue
        au_z, valid, activity = preprocess_au52(cs.arrays[raw_key], cs.arrays[ts_key])
        au_v2, pca_loadings = extract_au_v2(au_z, valid, cs.arrays[ts_key])

        arrays_out[f"{p}_au52"] = au_z
        arrays_out[f"{p}_au52_ts"] = cs.arrays[ts_key].astype(np.float64, copy=False)
        arrays_out[f"{p}_au_valid"] = valid.astype(bool)
        arrays_out[f"{p}_au_activity"] = activity.squeeze(axis=1).astype(np.float32)
        arrays_out[f"{p}_au_v2"] = au_v2
        arrays_out[f"{p}_au_v2_loadings"] = pca_loadings
        summary[p] = {
            "n_frames": int(au_z.shape[0]),
            "valid_frames_pct": float(valid.mean() * 100.0),
            "pca_n_components": PCA_N_COMPONENTS,
        }

    if not arrays_out:
        raise RuntimeError(f"{session_id}: no landmarks streams in digest")

    sidecar = {
        "session_id": session_id,
        "modality": "face",
        "modality_version": FACE_MODALITY_VERSION,
        "digest_xdf_md5": cs.xdf_md5,
        "digest_schema_version": cs.schema_version,
        "params": {
            "n_aus": N_AUS,
            "gap_fill_seconds": GAP_FILL_SECONDS,
            "pca_n_components": PCA_N_COMPONENTS,
            "deriv_sigma_s": DERIV_SIGMA_S,
            "feature_layout_v2": "pca(15)+derivatives(15)+activity(1)",
        },
        "participants": summary,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    atomic_write_npz(npz_path, **arrays_out)
    atomic_write_json(json_path, sidecar)
    return {"session_id": session_id, "status": "ok",
            "out_npz": str(npz_path), "out_json": str(json_path),
            "summary": summary}
