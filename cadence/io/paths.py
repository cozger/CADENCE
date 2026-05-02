"""Canonical paths for the data pipeline."""

from __future__ import annotations

from pathlib import Path


# Resolved relative to the project root (parent of cadence/).
# Caller can override via env var CADENCE_DATA_ROOT.
import os
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("CADENCE_DATA_ROOT", _PROJECT_ROOT / "data"))

DIGEST_DIR = DATA_ROOT / "digest" / "v1"
PREPROC_EEG_DIR = DATA_ROOT / "preproc" / "eeg" / "v1"
PREPROC_FACE_DIR = DATA_ROOT / "preproc" / "face" / "v1"
PREPROC_ECG_DIR = DATA_ROOT / "preproc" / "ecg" / "v1"
PREPROC_POSE_DIR = DATA_ROOT / "preproc" / "pose" / "v1"
MATLAB_DIR = DATA_ROOT / "matlab"

RAW_SESSIONS_DIR = _PROJECT_ROOT / "raw sessions"


def list_digests(digest_dir: Path = DIGEST_DIR) -> list[tuple[str, Path]]:
    """Return [(session_id, npz_path)] for every digest on disk."""
    if not digest_dir.is_dir():
        return []
    return [(p.stem, p) for p in sorted(digest_dir.glob("*.npz"))]
