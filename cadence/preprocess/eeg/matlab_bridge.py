"""Bridge between digest layer and MATLAB-cleaned EEG.

Locates / validates / produces ``data/matlab/<sid>_p{1,2}_clean.mat`` files for
the EEG preprocessing submodule. Each clean.mat carries a ``.mat.json``
provenance sidecar so we can detect staleness when the source XDF changes.

Public API:
    relocate_existing_clean_mats(...) -> list[RelocationResult]
        One-off (Step 5.2): copy existing clean.mat from
        ``analysis/eeglab_wavelet/cache/`` into ``data/matlab/`` with sidecars.

    has_fresh_clean_mat(session_id, xdf_md5, *, matlab_dir) -> bool
        Predicate used by ``cadence/preprocess/eeg/pipeline.py``.

    load_clean_mat(session_id, participant, *, matlab_dir) -> dict
        Loader: returns ``{data, srate, ch_labels}`` dict.

The MATLAB cleaning parameters in the sidecars are read directly from
``analysis/eeglab_wavelet/matlab/preprocess_eeg.m`` so they remain in sync
with the script. Existing (pre-Step-5) clean.mat files were produced by this
exact script, so the sidecar values are accurate retroactively.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# These constants encode the MATLAB script's settings. If
# ``analysis/eeglab_wavelet/matlab/preprocess_eeg.m`` changes, bump
# ``EEG_PREPROCESS_VERSION`` and re-run preproc on every session.
EEG_PREPROCESS_VERSION = "v1"
MATLAB_PREPROCESS_PARAMS: dict = {
    "preprocess_version": EEG_PREPROCESS_VERSION,
    "pipeline": "clean_rawdata + pop_interp(spherical) + pop_eegfiltnew(1, 40)",
    "clean_rawdata": {
        "arg_flatline_s": 5,
        "arg_highpass_transition_hz": [0.5, 1.0],
        "arg_channel_correlation": 0.8,
        "arg_line_noise_z": 4,
        "arg_burst_stddev": 20,         # ASR burst threshold (lenient → more retention)
        "arg_window_criterion": "off",  # preserve sample count for marker alignment
    },
    "pop_interp": "spherical",
    "pop_eegfiltnew_hz": [1, 40],
    "data_format": "double, (N_samples, N_channels), microvolts",
    "channels": 14,
}

# Where existing clean.mat files live (legacy) and where the new home is.
LEGACY_CACHE_DIR = Path("analysis/eeglab_wavelet/cache")
DEFAULT_MATLAB_DIR = Path("data/matlab")


# ---------------------------------------------------------------------------
# Sidecar layout
# ---------------------------------------------------------------------------

def _sidecar_path(mat_path: Path) -> Path:
    return mat_path.with_suffix(mat_path.suffix + ".json")


def _build_sidecar(*, session_id: str, participant: str, xdf_md5: str,
                   xdf_basename: str, source: str,
                   eeglab_version: str | None = None,
                   written_atomically: bool = True) -> dict:
    return {
        "session_id": session_id,
        "participant": participant,                    # 'p1' or 'p2'
        "xdf_md5": xdf_md5,                             # staleness key for digest
        "xdf_basename": xdf_basename,
        "matlab_params": MATLAB_PREPROCESS_PARAMS,
        "eeglab_version": eeglab_version or "unknown_legacy",
        "source": source,                               # 'relocated_from_legacy' | 'matlab_run'
        "written_atomically": bool(written_atomically),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` -> ``dst`` via ``dst.tmp`` + os.replace (atomic on same volume).

    Half-written files never have the final name, so a crashed copy is detected
    by an absent .mat.json sidecar (we write the sidecar last).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)  # atomic on Windows + POSIX


def _md5_of_xdf(xdf_path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with xdf_path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 5.2: relocation
# ---------------------------------------------------------------------------

@dataclass
class RelocationResult:
    session_id: str
    participants: list[str]    # ['p1', 'p2']
    cache_stem: str            # legacy stem in eeglab_wavelet/cache (may differ from session_id)
    xdf_basename: str
    xdf_md5: str
    relocated_paths: list[str]  # data/matlab/<sid>_p{1,2}_clean.mat
    sidecar_paths: list[str]


def _resolve_xdf_for_cache_stem(cache_stem: str, raw_dir: Path) -> Path | None:
    """Locate the source XDF for a legacy cache stem (case-insensitive prefix match).

    The legacy cache uses shortened stems for some sessions (``Y_55`` → XDF stem
    ``Y_55_04272026``); for others the cache stem is the full XDF stem.
    """
    candidates = sorted(raw_dir.glob("*.xdf"))
    # Exact match
    for p in candidates:
        if p.stem == cache_stem:
            return p
    cs_lower = cache_stem.lower()
    # Prefix match with delimiter (avoids 'y_06' matching 'y_06b')
    prefix = [p for p in candidates if p.stem.lower().startswith(cs_lower + "_")]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        return min(prefix, key=lambda p: len(p.stem))
    contains = [p for p in candidates if cs_lower in p.stem.lower()]
    if len(contains) == 1:
        return contains[0]
    return None


def relocate_existing_clean_mats(
        cache_dir: Path = LEGACY_CACHE_DIR,
        out_dir: Path = DEFAULT_MATLAB_DIR,
        raw_dir: Path = Path("raw sessions"),
        *,
        session_ids: Iterable[str] | None = None,
        force: bool = False,
        eeglab_version: str | None = "unknown_legacy",
) -> list[RelocationResult]:
    """Copy every ``<stem>_p{1,2}_clean.mat`` from ``cache_dir`` to ``out_dir``,
    write provenance sidecars, return per-session results.

    Idempotent: skips a (sid, p) pair if dst .mat exists, dst sidecar exists,
    AND the sidecar's xdf_md5 matches the current XDF md5. Use ``force=True``
    to overwrite.

    The ``.mat`` files in ``out_dir`` are renamed to use the **canonical
    session_id (XDF stem)**, even if the legacy cache uses a shortened stem
    (e.g. ``Y_55_p1_clean.mat`` becomes ``Y_55_04272026_p1_clean.mat``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_stems = sorted({
        p.name.removesuffix("_p1_clean.mat").removesuffix("_p2_clean.mat")
        for p in cache_dir.glob("*_clean.mat")
    })

    results: list[RelocationResult] = []
    for cs in cache_stems:
        xdf = _resolve_xdf_for_cache_stem(cs, raw_dir)
        if xdf is None:
            print(f"[relocate] SKIP cache_stem={cs!r}: no matching XDF in {raw_dir}",
                  file=sys.stderr)
            continue
        sid = xdf.stem
        if session_ids is not None and sid not in session_ids:
            continue

        xdf_md5 = _md5_of_xdf(xdf)

        relocated_paths: list[str] = []
        sidecar_paths: list[str] = []
        participants: list[str] = []

        for participant in ("p1", "p2"):
            src = cache_dir / f"{cs}_{participant}_clean.mat"
            if not src.is_file():
                print(f"[relocate] WARN {sid}/{participant}: missing legacy "
                      f"clean.mat at {src}", file=sys.stderr)
                continue
            dst = out_dir / f"{sid}_{participant}_clean.mat"
            sidecar = _sidecar_path(dst)

            if not force and dst.exists() and sidecar.exists():
                try:
                    with sidecar.open("r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                    if (existing.get("xdf_md5") == xdf_md5 and
                            existing.get("session_id") == sid):
                        print(f"[relocate] {sid}/{participant}: up-to-date, skipping")
                        relocated_paths.append(str(dst))
                        sidecar_paths.append(str(sidecar))
                        participants.append(participant)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass  # fall through

            _atomic_copy(src, dst)
            sidecar_data = _build_sidecar(
                session_id=sid,
                participant=participant,
                xdf_md5=xdf_md5,
                xdf_basename=xdf.name,
                source="relocated_from_legacy",
                eeglab_version=eeglab_version,
                written_atomically=True,
            )
            with sidecar.open("w", encoding="utf-8") as fh:
                json.dump(sidecar_data, fh, indent=2)
            print(f"[relocate] {sid}/{participant}: {src.name} -> {dst.name}")
            relocated_paths.append(str(dst))
            sidecar_paths.append(str(sidecar))
            participants.append(participant)

        if relocated_paths:
            results.append(RelocationResult(
                session_id=sid,
                participants=participants,
                cache_stem=cs,
                xdf_basename=xdf.name,
                xdf_md5=xdf_md5,
                relocated_paths=relocated_paths,
                sidecar_paths=sidecar_paths,
            ))

    print(f"\n[relocate] done: {len(results)} sessions relocated.")
    return results


# ---------------------------------------------------------------------------
# Runtime predicates / loaders (used by Step 6 EEG submodule)
# ---------------------------------------------------------------------------

def has_fresh_clean_mat(session_id: str, xdf_md5: str, *,
                        matlab_dir: Path = DEFAULT_MATLAB_DIR) -> bool:
    """True iff both p1 and p2 clean.mat + sidecar exist with matching xdf_md5."""
    for p in ("p1", "p2"):
        mat = matlab_dir / f"{session_id}_{p}_clean.mat"
        sidecar = _sidecar_path(mat)
        if not (mat.is_file() and sidecar.is_file()):
            return False
        try:
            with sidecar.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False
        if meta.get("xdf_md5") != xdf_md5 or meta.get("session_id") != session_id:
            return False
    return True


def load_clean_mat(session_id: str, participant: str, *,
                   matlab_dir: Path = DEFAULT_MATLAB_DIR) -> dict:
    """Read clean.mat into a plain dict. Returns ``{data, srate, ch_labels}``."""
    import scipy.io as sio
    mat = matlab_dir / f"{session_id}_{participant}_clean.mat"
    if not mat.is_file():
        raise FileNotFoundError(f"no clean.mat at {mat}")
    raw = sio.loadmat(str(mat), squeeze_me=True, struct_as_record=False)
    return {
        "data": raw["data"],             # (N_samples, N_channels), float64
        "srate": float(raw["srate"]),
        "ch_labels": [str(s).strip() for s in raw.get("ch_labels", [])],
    }
