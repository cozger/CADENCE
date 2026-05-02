"""[DEPRECATED] cadence.data — legacy session-cache surface.

This module was historically the home of the entire data pipeline:
``xdf_loader``, ``preprocessors``, ``alignment``, ``eeg_features``,
``wavelet_features``, ``interbrain_features``. The pipeline has been split into
two cleanly-versioned layers:

    Layer 1 (cadence.ingest)     - raw XDF -> data/digest/v1/<sid>.npz
    Layer 2 (cadence.preprocess) - per-modality cleaning -> data/preproc/<mod>/v1/

This file is the *eager-loading dict view* shim that lets pre-existing scripts
(V11, V10, V8.2 scaffolds; rSLDS hierarchical fits; tests) keep working
unchanged. The plan calls for opportunistic migration of consumers in Step 7;
this shim is the bridge.

Public API preserved:

    load_session_from_cache(session_id_or_path)  -> dict (eager-loaded)
    discover_cached_sessions(cache_dir=None)     -> list[session_id]
    load_and_preprocess_cached(xdf_path, ...)    -> dict
    apply_modality_exclusions(session, sid)      -> dict (mutated)
    EXCLUDED_MODALITIES                          -> dict (frozen view of yaml)

Eager-loading rationale (per the plan):
    * Joblib safety. Returns concrete numpy arrays, not mmap'd NpzFile handles.
      Pickling for loky/threading workers is straightforward.
    * Mutation isolation. Each call constructs a fresh dict with arrays
      write-protected via .setflags(write=False); consumers needing mutation
      must copy explicitly.
    * Determinism. No mid-run filesystem races: artifacts loaded at t=0 are
      what the consumer sees throughout the call.
    * Trivial cost: V11 reads ~30 keys per session, lazy buys nothing.

Coverage. ~30 legacy keys per participant covering:
    eeg, eeg_ts, eeg_valid, eeg_features, eeg_features_ts, eeg_features_valid,
    blendshapes, blendshapes_ts, blendshapes_valid, blendshapes_v2,
    pose, pose_ts, pose_features, pose_features_ts, pose_features_valid,
    ecg, ecg_ts, ecg_valid, ecg_features, ecg_features_ts, ecg_features_valid,
    landmarks_raw, landmarks_ts, role, name.
Plus session-level: ``markers``, ``marker_sources``, ``protocol``,
``duration``, ``t_start_absolute``, ``session_id``, ``xdf_basename``,
``xdf_md5``, ``schema_version``, ``pose_format``.

Wavelet features (``{p}_eeg_wavelet`` and friends) are NOT covered by the
shim; consumers that need them should be migrated explicitly to
``cadence.preprocess.eeg.wavelet`` once that module lands.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

warnings.warn(
    "cadence.data.* is deprecated; use cadence.ingest / cadence.preprocess",
    DeprecationWarning,
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# Re-exports from new home
# ---------------------------------------------------------------------------

from cadence.ingest.quality import (
    apply_modality_exclusions,
    excluded_modalities_legacy_dict as _excluded_dict_loader,
)


def _load_excluded_modalities() -> dict:
    """Frozen-snapshot view of session_quality.yaml in legacy dict shape."""
    try:
        return _excluded_dict_loader()
    except FileNotFoundError:
        return {}


EXCLUDED_MODALITIES = _load_excluded_modalities()


# ---------------------------------------------------------------------------
# Eager-loading dict view
# ---------------------------------------------------------------------------

def _resolve_session_id(arg: str | Path) -> str:
    """Accept either a session_id or a legacy cache path (extract sid from stem)."""
    s = str(arg)
    if "/" in s or "\\" in s or s.endswith(".npz") or s.endswith(".json"):
        # Legacy: cache path like 'session_cache/abc_y_06.npz' -> 'y_06'
        stem = Path(s).stem
        # MCCT prefix: 12 hex chars + '_' + sid. Drop the prefix if present.
        parts = stem.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 12 and all(c in "0123456789abcdef" for c in parts[0]):
            return parts[1]
        return stem
    return s


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        arr.setflags(write=False)
    return arr


def discover_cached_sessions(cache_dir: str | Path | None = None) -> list[tuple[str, str]]:
    """List canonical session IDs available as digests, as (name, path) tuples.

    Returns ``list[(session_id, cache_path_str)]`` for backward compatibility
    with legacy V11/V10/V8.2 scaffold scripts that unpack the tuple. The
    ``cache_path_str`` is the digest .npz path-prefix (without extension), so
    callers can keep their `cache_path_prefix + ".json"` patterns.

    Honors the new home (``data/digest/v1/``); ``cache_dir`` argument retained
    for signature compatibility but ignored.
    """
    digest_dir = Path("data/digest/v1")
    if not digest_dir.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(digest_dir.glob("*.npz")):
        sid = p.stem
        # Pass the .npz path itself as the second element. _resolve_session_id
        # handles legacy-style stems and Path-like arguments equivalently.
        out.append((sid, str(p)))
    return out


def load_session_from_cache(session_id_or_path: str | Path,
                            config: dict | None = None) -> dict:
    """Eager-load a session into the legacy dict layout (~30 keys per participant).

    The ``config`` argument is accepted for backward compatibility but ignored
    — V2 wavelet features are not part of this shim. Use
    ``cadence.preprocess.eeg.wavelet`` directly when the wavelet submodule
    lands.
    """
    from cadence.ingest.digest import load_digest

    session_id = _resolve_session_id(session_id_or_path)

    # ---- Layer 1: digest (always present for canonical sessions) ----
    cs = load_digest(session_id)

    out: dict = {}

    # Top-level metadata
    out["session_id"] = cs.session_id
    out["schema_version"] = cs.schema_version
    out["xdf_basename"] = cs.xdf_basename
    out["xdf_md5"] = cs.xdf_md5
    out["duration"] = cs.duration_s
    out["duration_s"] = cs.duration_s
    out["t_start_absolute"] = cs.t_start_lsl
    out["t_start_lsl"] = cs.t_start_lsl
    out["protocol"] = cs.protocol
    out["pose_format"] = cs.pose_format
    out["modalities"] = list(cs.modalities)
    out["markers"] = list(cs.markers)
    out["marker_sources"] = list(cs.marker_sources)
    out["stream_inventory"] = dict(cs.stream_inventory)

    # Roles
    out["p1_role"] = cs.roles.p1_role
    out["p2_role"] = cs.roles.p2_role
    out["p1_name"] = cs.roles.p1_name
    out["p2_name"] = cs.roles.p2_name
    out["role_source"] = cs.roles.role_source

    # Per-participant raw streams (carried through from digest .npz)
    for k, v in cs.arrays.items():
        out[k] = _make_readonly(np.asarray(v).copy())

    # ---- Layer 2: per-modality preproc (best-effort: missing -> skip) ----
    out.update(_load_preproc_eeg(session_id))
    out.update(_load_preproc_face(session_id))
    out.update(_load_preproc_ecg(session_id))
    out.update(_load_preproc_pose(session_id, cs.pose_format))

    # Apply session_quality.yaml modality exclusions (zeros + False valid)
    out = apply_modality_exclusions(out, session_id)
    return out


def load_and_preprocess_cached(xdf_path: str | Path,
                               cache_dir: str | Path = "session_cache",
                               config: dict | None = None) -> dict:
    """Drop-in for the legacy bulk loader.

    The ``cache_dir`` argument is accepted for signature compatibility but
    ignored — output is read from ``data/digest/v1`` via the digest layer.
    If ``data/digest/v1/<sid>.json`` is missing, this calls
    ``cadence.ingest.digest.digest_xdf`` first.
    """
    from cadence.ingest.digest import digest_xdf

    p = Path(xdf_path)
    sid = p.stem
    digest_json = Path("data/digest/v1") / f"{sid}.json"
    if not digest_json.is_file():
        digest_xdf(p)
    return load_session_from_cache(sid, config=config)


# ---------------------------------------------------------------------------
# Per-modality loaders (silently skip missing artifacts)
# ---------------------------------------------------------------------------

def _try_load_npz(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        d = dict(np.load(path, allow_pickle=True))
    except (OSError, ValueError):
        return None
    return d


def _load_preproc_eeg(session_id: str) -> dict:
    """Map preproc EEG output to legacy keys."""
    npz = _try_load_npz(Path("data/preproc/eeg/v1") / f"{session_id}.npz")
    if npz is None:
        return {}
    out: dict = {}
    for p in ("p1", "p2"):
        if f"{p}_eeg_clean" in npz:
            out[f"{p}_eeg"] = _make_readonly(npz[f"{p}_eeg_clean"])
            out[f"{p}_eeg_ts"] = _make_readonly(npz[f"{p}_eeg_clean_ts"])
            out[f"{p}_eeg_valid"] = _make_readonly(npz[f"{p}_eeg_clean_valid"])
        if f"{p}_eeg_features" in npz:
            out[f"{p}_eeg_features"] = _make_readonly(npz[f"{p}_eeg_features"])
            out[f"{p}_eeg_features_ts"] = _make_readonly(npz[f"{p}_eeg_features_ts"])
            out[f"{p}_eeg_features_valid"] = _make_readonly(npz[f"{p}_eeg_features_valid"])
    return out


def _load_preproc_face(session_id: str) -> dict:
    """Map preproc face output to legacy `blendshapes` keys."""
    npz = _try_load_npz(Path("data/preproc/face/v1") / f"{session_id}.npz")
    if npz is None:
        return {}
    out: dict = {}
    for p in ("p1", "p2"):
        if f"{p}_au52" in npz:
            # Legacy `blendshapes` was 53 ch (52 AU + activity); reconstruct.
            au52 = npz[f"{p}_au52"]
            act = npz.get(f"{p}_au_activity")
            if act is not None:
                bl = np.concatenate([au52, act[:, None].astype(np.float32)], axis=1)
            else:
                bl = au52
            out[f"{p}_blendshapes"] = _make_readonly(bl)
            out[f"{p}_blendshapes_ts"] = _make_readonly(npz[f"{p}_au52_ts"])
            out[f"{p}_blendshapes_valid"] = _make_readonly(npz[f"{p}_au_valid"])
        if f"{p}_au_v2" in npz:
            out[f"{p}_blendshapes_v2"] = _make_readonly(npz[f"{p}_au_v2"])
    return out


def _load_preproc_ecg(session_id: str) -> dict:
    """Map preproc ECG output to legacy keys."""
    npz = _try_load_npz(Path("data/preproc/ecg/v1") / f"{session_id}.npz")
    if npz is None:
        return {}
    out: dict = {}
    for p in ("p1", "p2"):
        if f"{p}_ecg_clean" in npz:
            out[f"{p}_ecg"] = _make_readonly(npz[f"{p}_ecg_clean"])
            out[f"{p}_ecg_ts"] = _make_readonly(npz[f"{p}_ecg_clean_ts"])
            out[f"{p}_ecg_valid"] = _make_readonly(npz[f"{p}_ecg_valid"])
        if f"{p}_ecg_features" in npz:
            out[f"{p}_ecg_features"] = _make_readonly(npz[f"{p}_ecg_features"])
            out[f"{p}_ecg_features_ts"] = _make_readonly(npz[f"{p}_ecg_features_ts"])
            out[f"{p}_ecg_features_valid"] = _make_readonly(npz[f"{p}_ecg_features_valid"])
    return out


def _load_preproc_pose(session_id: str, pose_format: str) -> dict:
    """Map preproc pose output to legacy keys.

    Legacy ``{p}_pose`` was the flattened (N, 99) coordinate-only form. We
    materialize that from the (N, 33, 4) pose33 array for backward compat.
    """
    npz = _try_load_npz(Path("data/preproc/pose/v1") / f"{session_id}.npz")
    if npz is None:
        return {}
    out: dict = {}
    for p in ("p1", "p2"):
        if f"{p}_pose33" in npz:
            pose33 = npz[f"{p}_pose33"]
            pose_99 = pose33[:, :, :3].reshape(pose33.shape[0], 99).astype(np.float32, copy=False)
            out[f"{p}_pose"] = _make_readonly(pose_99)
            out[f"{p}_pose_ts"] = _make_readonly(npz[f"{p}_pose33_ts"])
        if f"{p}_pose_features" in npz:
            out[f"{p}_pose_features"] = _make_readonly(npz[f"{p}_pose_features"])
            out[f"{p}_pose_features_ts"] = _make_readonly(npz[f"{p}_pose33_ts"])
            out[f"{p}_pose_features_valid"] = _make_readonly(npz[f"{p}_pose_features_valid"])
    return out
