"""Smoke tests for ``cadence.preprocess.<modality>`` pipelines.

Each test asserts:
    1. The pipeline produces a non-empty .npz + .json
    2. The sidecar's ``digest_xdf_md5`` matches the digest
    3. Required keys are present and have expected dtypes/shapes

Tests run against ``y_06`` (mediapipe33 pose) which is digested at
``data/digest/v1/y_06.npz``. EEG smoke is skipped if no fresh clean.mat exists.
"""
from __future__ import annotations

import json
from pathlib import Path

# Hoist torch BEFORE numpy on Windows torch 2.10+numpy 2.4 stacks.
import torch  # noqa: F401
import numpy as np
import pytest

DIGEST_DIR = Path("data/digest/v1")
SESSION_ID = "y_06"


def _require_digest():
    if not (DIGEST_DIR / f"{SESSION_ID}.npz").is_file():
        pytest.skip(f"no digest at {DIGEST_DIR / SESSION_ID} — skip smokes")


def _digest_md5() -> str:
    with (DIGEST_DIR / f"{SESSION_ID}.json").open() as fh:
        return json.load(fh)["xdf_md5"]


def _check_sidecar_md5(json_path: Path) -> None:
    with json_path.open() as fh:
        meta = json.load(fh)
    assert meta["digest_xdf_md5"] == _digest_md5(), (
        f"sidecar md5 mismatch: {json_path}"
    )


def test_pose_pipeline_smoke(tmp_path):
    _require_digest()
    from cadence.preprocess.pose.pipeline import preprocess_pose_session
    out = tmp_path / "pose"
    res = preprocess_pose_session(SESSION_ID, out_dir=out, force=True)
    assert res["status"] == "ok"
    npz = np.load(out / f"{SESSION_ID}.npz")
    for k in ("p1_pose33", "p1_pose33_ts", "p1_pose_features",
              "p1_pose_features_valid", "p2_pose33", "p2_pose_features"):
        assert k in npz.files, f"missing {k}"
    assert npz["p1_pose33"].shape[1:] == (33, 4)
    assert npz["p1_pose_features"].shape[1] == 41
    _check_sidecar_md5(out / f"{SESSION_ID}.json")


def test_face_pipeline_smoke(tmp_path):
    _require_digest()
    from cadence.preprocess.face.pipeline import preprocess_face_session
    out = tmp_path / "face"
    res = preprocess_face_session(SESSION_ID, out_dir=out, force=True)
    assert res["status"] == "ok"
    npz = np.load(out / f"{SESSION_ID}.npz")
    for k in ("p1_au52", "p1_au_v2", "p1_au_valid", "p1_au_v2_loadings",
              "p2_au52", "p2_au_v2"):
        assert k in npz.files, f"missing {k}"
    assert npz["p1_au52"].shape[1] == 52
    assert npz["p1_au_v2"].shape[1] == 31
    assert npz["p1_au_v2_loadings"].shape == (15, 52)
    _check_sidecar_md5(out / f"{SESSION_ID}.json")


def test_ecg_pipeline_smoke(tmp_path):
    _require_digest()
    from cadence.preprocess.ecg.pipeline import preprocess_ecg_session
    out = tmp_path / "ecg"
    res = preprocess_ecg_session(SESSION_ID, out_dir=out, force=True)
    assert res["status"] == "ok"
    npz = np.load(out / f"{SESSION_ID}.npz")
    for k in ("p1_ecg_clean", "p1_ecg_clean_ts", "p1_ecg_features",
              "p1_ecg_features_valid", "p2_ecg_clean", "p2_ecg_features"):
        assert k in npz.files, f"missing {k}"
    assert npz["p1_ecg_features"].shape[1] == 7
    _check_sidecar_md5(out / f"{SESSION_ID}.json")


def test_eeg_pipeline_smoke(tmp_path):
    _require_digest()
    # Skip if no fresh clean.mat — Step 5.2 may not have populated this session.
    from cadence.preprocess.eeg.matlab_bridge import has_fresh_clean_mat
    md5 = _digest_md5()
    if not has_fresh_clean_mat(SESSION_ID, md5):
        pytest.skip("no fresh clean.mat — skip EEG smoke")
    from cadence.preprocess.eeg.pipeline import preprocess_eeg_session
    out = tmp_path / "eeg"
    res = preprocess_eeg_session(SESSION_ID, out_dir=out, force=True)
    assert res["status"] == "ok"
    npz = np.load(out / f"{SESSION_ID}.npz", allow_pickle=True)
    for k in ("p1_eeg_clean", "p1_eeg_clean_ts", "p1_eeg_features",
              "p2_eeg_clean", "p2_eeg_features", "p1_eeg_ch_labels"):
        assert k in npz.files, f"missing {k}"
    assert npz["p1_eeg_clean"].shape[1] == 14
    assert npz["p1_eeg_features"].shape[1] == 8
    _check_sidecar_md5(out / f"{SESSION_ID}.json")
