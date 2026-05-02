"""EEG preprocessing pipeline (Step 6).

Wraps the MATLAB EEGLAB-cleaned EEG (``data/matlab/<sid>_p{1,2}_clean.mat``)
into the standard preproc layout. Per-participant only.

Pipeline (per session):

1. Load digest. Compute the **common-window crop** that the MATLAB export used
   (``t_start_common = max(p1_ts[0], p2_ts[0])``,
    ``t_end_common = min(p1_ts[-1], p2_ts[-1])``) and slice each digest's
   per-participant ``{p}_eeg_ts`` to align with the cleaned MATLAB samples.
2. Validate ``has_fresh_clean_mat`` (xdf_md5 in sidecar matches digest).
   If stale or missing -> raise; the user must re-run MATLAB cleaning.
3. Load clean.mat for p1 and p2 (each ``(M, 14)`` float64 microvolts).
4. Extract 8-ch EEG features at 2 Hz (engagement + aperiodic + theta burst +
   theta/alpha phase + activity) via ``cadence.data.eeg_features``.

Output (``data/preproc/eeg/v1/<sid>.npz``)::

    p{1,2}_eeg_clean        : (M, 14)  float32  EEGLAB-cleaned (microvolts)
    p{1,2}_eeg_clean_ts     : (M,)     float64  digest-derived per-participant ts
    p{1,2}_eeg_clean_valid  : (M,)     bool     all-true placeholder (clean.mat is clean)
    p{1,2}_eeg_features     : (N_out, 8)  float32   8-ch features at 2 Hz
    p{1,2}_eeg_features_ts  : (N_out,) float64
    p{1,2}_eeg_features_valid : (N_out,) bool
    p{1,2}_eeg_ch_labels   : object array of channel name strings (length 14)

Resource-discipline note (P2.4 — 2026-05-01 audit)
--------------------------------------------------
This pipeline currently runs sequentially per session via ``make_modality_cli``
(see ``cadence/preprocess/_cli.py``) and does NOT use joblib at the
session level. The BLAS-heavy stages (MATLAB load + ``extract_eeg_features``
+ ``extract_wavelet_features``) are therefore safe with the default thread
allocation. **If/when this pipeline gets a parallel ``--all`` mode**, every
per-session worker MUST use ``limit_blas_threads(1)`` and the runner MUST
cap ``n_jobs`` via
``pick_n_jobs(per_worker_ram_gb=3.5, requested=args.n_jobs)`` —
the 3.5 GB estimate covers the MATLAB load, the
8-channel feature extraction, and the wavelet feature pass with realistic
60-min sessions. See ``cadence/io/resources.py`` and
``docs/resource_audit_2026_05_01.md`` (finding P2.4) for context.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cadence.preprocess.common import (
    atomic_write_json,
    atomic_write_npz,
    default_out_dir,
    staleness_check,
)
from cadence.preprocess.eeg.matlab_bridge import (
    DEFAULT_MATLAB_DIR,
    MATLAB_PREPROCESS_PARAMS,
    has_fresh_clean_mat,
    load_clean_mat,
)


EEG_MODALITY_VERSION = "v1"


def _common_window_indices(p1_ts: np.ndarray, p2_ts: np.ndarray
                           ) -> tuple[int, int]:
    """Return (p1_start_idx, p2_start_idx) for the common LSL window.

    Mirrors the legacy ``analysis/eeglab_wavelet/session_io.py:load_raw_eeg``
    crop. The MATLAB cleaning was applied to the cropped data, so we need the
    same crop offsets to align timestamps with clean.mat.
    """
    t_start = max(float(p1_ts[0]), float(p2_ts[0]))
    p1_start = int(np.searchsorted(p1_ts, t_start, side="left"))
    p2_start = int(np.searchsorted(p2_ts, t_start, side="left"))
    return p1_start, p2_start


def preprocess_eeg_session(session_id: str,
                           *,
                           digest_dir: str | Path = "data/digest/v1",
                           matlab_dir: str | Path | None = None,
                           out_dir: str | Path | None = None,
                           force: bool = False) -> dict:
    from cadence.ingest.digest import load_digest
    from cadence.data.eeg_features import (
        extract_eeg_features,
        EEG_FEATURE_NAMES,
        EEG_N_FEATURES,
        EEG_FEATURES_SRATE,
    )

    digest_dir = Path(digest_dir)
    matlab_dir = Path(matlab_dir) if matlab_dir is not None else DEFAULT_MATLAB_DIR
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir("eeg", EEG_MODALITY_VERSION)
    npz_path = out_dir / f"{session_id}.npz"
    json_path = out_dir / f"{session_id}.json"

    cs = load_digest(session_id, digest_dir=digest_dir)
    if not force and staleness_check(json_path, cs.xdf_md5):
        return {"session_id": session_id, "status": "skip-up-to-date"}

    if not has_fresh_clean_mat(session_id, cs.xdf_md5, matlab_dir=matlab_dir):
        raise RuntimeError(
            f"{session_id}: no fresh clean.mat. Either the file is missing "
            f"from {matlab_dir} or its xdf_md5 sidecar disagrees with the "
            f"digest. Run MATLAB preprocess_eeg.m or "
            f"scripts/_relocate_clean_mats.py."
        )

    if "p1_eeg_ts" not in cs.arrays or "p2_eeg_ts" not in cs.arrays:
        raise RuntimeError(f"{session_id}: digest missing per-participant EEG ts")

    p1_ts_full = cs.arrays["p1_eeg_ts"]
    p2_ts_full = cs.arrays["p2_eeg_ts"]
    p1_start, p2_start = _common_window_indices(p1_ts_full, p2_ts_full)

    arrays_out: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for participant, (ts_full, start) in [
        ("p1", (p1_ts_full, p1_start)),
        ("p2", (p2_ts_full, p2_start)),
    ]:
        clean = load_clean_mat(session_id, participant, matlab_dir=matlab_dir)
        data = np.asarray(clean["data"], dtype=np.float64)        # (M, 14)
        srate = float(clean["srate"])
        ch_labels = list(clean.get("ch_labels", []))
        if data.ndim != 2 or data.shape[1] != 14:
            raise RuntimeError(
                f"{session_id}/{participant}: unexpected clean.mat shape "
                f"{data.shape}; expected (M, 14)"
            )
        m = data.shape[0]
        # Slice the digest's full per-participant ts to match the cropped MATLAB output.
        ts = ts_full[start:start + m]
        if len(ts) != m:
            raise RuntimeError(
                f"{session_id}/{participant}: clean.mat M={m} but digest "
                f"per-participant ts only has {len(ts)} samples after common-window "
                f"crop (start={start}, full_len={len(ts_full)})"
            )
        # extract_eeg_features expects per-channel validity (N, 14), not (N,).
        # MATLAB-cleaned data is fully valid by construction.
        valid = np.ones(m, dtype=bool)
        eeg_valid_2d = np.ones((m, 14), dtype=bool)

        # 8-ch features at 2 Hz
        features, feat_valid, feat_ts = extract_eeg_features(
            data, eeg_valid_2d, ts.astype(np.float64), srate=int(round(srate)),
            output_hz=EEG_FEATURES_SRATE,
        )

        arrays_out[f"{participant}_eeg_clean"] = data.astype(np.float32, copy=False)
        arrays_out[f"{participant}_eeg_clean_ts"] = ts.astype(np.float64, copy=False)
        arrays_out[f"{participant}_eeg_clean_valid"] = valid
        arrays_out[f"{participant}_eeg_features"] = features.astype(np.float32, copy=False)
        arrays_out[f"{participant}_eeg_features_ts"] = feat_ts.astype(np.float64, copy=False)
        arrays_out[f"{participant}_eeg_features_valid"] = feat_valid.astype(bool)
        arrays_out[f"{participant}_eeg_ch_labels"] = np.array(ch_labels, dtype=object)

        summary[participant] = {
            "n_samples": int(m),
            "srate_hz": srate,
            "common_window_start_idx_in_digest": int(start),
            "n_feature_frames": int(features.shape[0]),
            "feature_valid_pct": float(feat_valid.mean() * 100.0),
            "n_channels": 14,
        }

    sidecar = {
        "session_id": session_id,
        "modality": "eeg",
        "modality_version": EEG_MODALITY_VERSION,
        "digest_xdf_md5": cs.xdf_md5,
        "digest_schema_version": cs.schema_version,
        "matlab_params": MATLAB_PREPROCESS_PARAMS,
        "params": {
            "n_channels": 14,
            "channel_selection_from_raw_19": "cols 3..16 (AF3..AF4)",
            "common_window_crop": True,
            "features_srate_hz": EEG_FEATURES_SRATE,
            "n_features": EEG_N_FEATURES,
            "feature_names": list(EEG_FEATURE_NAMES),
        },
        "participants": summary,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    atomic_write_npz(npz_path, **arrays_out)
    atomic_write_json(json_path, sidecar)
    return {"session_id": session_id, "status": "ok",
            "out_npz": str(npz_path), "out_json": str(json_path),
            "summary": summary}
