"""Cache I/O at well-defined boundaries.

Each per-session NPZ records:
- ``face_npz_md5``: SHA-256 of ``data/preproc/face/v1/<sid>.npz``
- ``synchrony_config_hash``: 16-hex prefix of SHA-256 of the config dict

Re-execution skips a stage when both hashes match the cached values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / 'results' / 'synchrony'
FACE_PREPROC_ROOT = REPO_ROOT / 'data' / 'preproc' / 'face' / 'v1'
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'

STAGE_FILENAMES = {
    1: '01_events.npz',
    2: '02_episodes.npz',
    3: '03_features_per_episode.npz',
}
STAGE_SIDECAR_FILENAMES = {
    1: '01_events.json',
    2: '02_episodes.json',
    3: '03_features_per_episode.json',
}


def session_dir(sid: str, ensure: bool = False) -> Path:
    p = RESULTS_ROOT / sid
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p


def cohort_dir(ensure: bool = False) -> Path:
    p = RESULTS_ROOT / 'cohort'
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p


def face_npz_path(sid: str) -> Path:
    return FACE_PREPROC_ROOT / f'{sid}.npz'


def digest_path(sid: str) -> Path:
    return DIGEST_ROOT / f'{sid}.json'


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(chunk), b''):
            h.update(buf)
    return h.hexdigest()


def is_stage_fresh(sid: str, stage: int, config_hash: str) -> bool:
    """Return True iff cached stage NPZ is up-to-date for this config + face NPZ."""
    sidecar = session_dir(sid) / STAGE_SIDECAR_FILENAMES[stage]
    npz = session_dir(sid) / STAGE_FILENAMES[stage]
    if not (sidecar.exists() and npz.exists()):
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get('synchrony_config_hash') != config_hash:
        return False
    face = face_npz_path(sid)
    if not face.exists():
        return False
    return meta.get('face_npz_md5') == file_md5(face)


def write_stage(sid: str, stage: int, arrays: dict, sidecar_extra: dict,
                config_hash: str) -> None:
    """Write stage NPZ + JSON sidecar, stamping freshness metadata."""
    out = session_dir(sid, ensure=True)
    np.savez(out / STAGE_FILENAMES[stage], **arrays)
    meta = {
        'sid': sid,
        'stage': stage,
        'synchrony_config_hash': config_hash,
        'face_npz_md5': file_md5(face_npz_path(sid)),
    }
    meta.update(sidecar_extra)
    (out / STAGE_SIDECAR_FILENAMES[stage]).write_text(json.dumps(meta, indent=2))


def read_stage(sid: str, stage: int) -> tuple[dict, dict]:
    out = session_dir(sid)
    npz = dict(np.load(out / STAGE_FILENAMES[stage], allow_pickle=True))
    meta = json.loads((out / STAGE_SIDECAR_FILENAMES[stage]).read_text())
    return npz, meta


def load_face_npz(sid: str) -> dict:
    return dict(np.load(face_npz_path(sid)))


def load_digest(sid: str) -> dict:
    return json.loads(digest_path(sid).read_text())
