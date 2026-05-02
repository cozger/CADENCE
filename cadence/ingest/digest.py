"""Orchestrator: raw XDF -> CanonicalSession -> .npz/.json.

Public API:
    digest_xdf(xdf_path, out_dir, force=False) -> CanonicalSession
    digest_all(raw_dir, out_dir, force=False) -> list[CanonicalSession]
    load_digest(session_id, digest_dir=DIGEST_DIR) -> CanonicalSession
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from cadence.io.paths import DIGEST_DIR, RAW_SESSIONS_DIR
from cadence.ingest.roles import (
    RoleAssignment,
    UnresolvedRolesError,
    resolve_roles,
)
from cadence.ingest.schema import (
    SCHEMA_VERSION,
    SUPPORTED_MODALITIES,
    CanonicalSession,
    QualityFlags,
    UnknownPoseFormatError,
    detect_pose_format,
)
from cadence.ingest.xdf_reader import load_xdf_streams, compute_xdf_md5
from cadence.ingest.quality import (
    load_session_quality,
    list_canonical_sessions,
    load_marker_overrides,
)


def digest_xdf(xdf_path: str | Path,
               out_dir: str | Path = DIGEST_DIR,
               *,
               force: bool = False) -> Optional[CanonicalSession]:
    """Digest one XDF into ``data/digest/v1/<session_id>.npz`` + ``.json``.

    Parameters
    ----------
    xdf_path : str or Path
    out_dir : str or Path
    force : bool
        If False and an up-to-date digest exists (matching xdf_md5), return
        the existing one without re-reading the XDF.

    Returns
    -------
    CanonicalSession on success, None if the session was skipped
    (unresolvable roles, unknown pose format, etc.). All errors are logged
    to stderr and the function returns rather than raising — so a corpus
    run continues even if individual sessions fail.
    """
    xdf_path = Path(xdf_path)
    if not xdf_path.is_file():
        print(f"[digest] ERROR: file not found: {xdf_path}", file=sys.stderr)
        return None

    session_id = xdf_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{session_id}.npz"
    json_path = out_dir / f"{session_id}.json"

    # Staleness check
    xdf_md5 = compute_xdf_md5(str(xdf_path))
    if not force and npz_path.exists() and json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("xdf_md5") == xdf_md5 and existing.get("schema_version") == SCHEMA_VERSION:
                print(f"[digest] {session_id}: up-to-date, skipping (use --force to override)")
                return _load_canonical_session(session_id, out_dir)
        except (json.JSONDecodeError, OSError):
            pass  # fall through to re-digest

    print(f"[digest] {session_id}: reading XDF ...")
    overrides = load_marker_overrides(session_id)
    if overrides:
        print(f"[digest] {session_id}: applying {len(overrides)} marker override(s)")
    try:
        raw = load_xdf_streams(str(xdf_path), marker_overrides=overrides)
    except Exception as e:
        print(f"[digest] ERROR: {session_id}: load_xdf_streams failed: {e!r}", file=sys.stderr)
        traceback.print_exc()
        return None

    # ---- Role resolution ----
    try:
        roles = resolve_roles(raw["streams"])
    except UnresolvedRolesError as e:
        print(f"[digest] SKIP {session_id}: {e}", file=sys.stderr)
        return None

    # ---- Pose format detection ----
    pose_format = None
    if "p1_pose_raw" in raw:
        try:
            pose_format = detect_pose_format(raw["p1_pose_raw"].shape[1])
        except UnknownPoseFormatError as e:
            print(f"[digest] SKIP {session_id}: {e}", file=sys.stderr)
            return None
    if "p2_pose_raw" in raw and pose_format is not None:
        try:
            p2_format = detect_pose_format(raw["p2_pose_raw"].shape[1])
        except UnknownPoseFormatError as e:
            print(f"[digest] SKIP {session_id}: p2_pose: {e}", file=sys.stderr)
            return None
        if p2_format != pose_format:
            print(
                f"[digest] SKIP {session_id}: p1 and p2 have different pose formats "
                f"({pose_format} vs {p2_format})",
                file=sys.stderr,
            )
            return None

    # ---- Quality flags from configs/session_quality.yaml ----
    try:
        sq = load_session_quality(session_id)
        quality_flags = QualityFlags(
            excluded_modalities=sq.excluded_modalities or {"p1": [], "p2": []},
            missing_markers=[],
            notes=list(sq.notes or []),
        )
    except (KeyError, FileNotFoundError):
        # Sessions not yet in YAML: blank quality flags. The audit script
        # populates the YAML in Step 0; explicit out-of-band sessions get
        # default empty flags.
        quality_flags = QualityFlags()

    # ---- Protocol detection from markers ----
    marker_labels = {lab for _t, lab in raw.get("markers", [])}
    protocol = _detect_protocol(marker_labels)
    quality_flags = QualityFlags(
        excluded_modalities=quality_flags.excluded_modalities,
        missing_markers=_missing_required_markers(protocol, marker_labels),
        notes=quality_flags.notes,
    )

    # ---- Modalities present (matches load_xdf_streams keying) ----
    modalities = [
        mod for mod in SUPPORTED_MODALITIES
        if f"p1_{mod}_raw" in raw or f"p2_{mod}_raw" in raw
    ]

    # ---- t_start session-relative ----
    t_start_lsl = float(raw.get("t_start_lsl", 0.0))
    duration_s = float(raw.get("duration_s", 0.0))

    # ---- Build canonical .npz arrays (raw + ts, session-relative ts) ----
    arrays: dict[str, np.ndarray] = {}
    for p in ("p1", "p2"):
        for mod in ("eeg", "ecg", "landmarks", "pose"):
            raw_key = f"{p}_{mod}_raw"
            ts_key = f"{p}_{mod}_ts"
            if raw_key in raw and ts_key in raw:
                # The .npz key for raw arrays uses '_raw' suffix for eeg/ecg/landmarks,
                # but '_full' for pose to signal "untouched original (132 or 532 cols)".
                if mod == "pose":
                    out_data_key = f"{p}_pose_full"
                    out_ts_key = f"{p}_pose_ts"
                else:
                    out_data_key = f"{p}_{mod}_raw"
                    out_ts_key = f"{p}_{mod}_ts"
                arrays[out_data_key] = np.asarray(raw[raw_key], dtype=np.float32)
                arrays[out_ts_key] = (np.asarray(raw[ts_key], dtype=np.float64)
                                      - t_start_lsl)

    # ---- Stream inventory (carry over but adjust pose key naming) ----
    inv = dict(raw.get("stream_inventory", {}))
    # rename p1_pose_raw -> p1_pose_full in inventory keys
    for p in ("p1", "p2"):
        if f"{p}_pose_raw" in inv:
            inv[f"{p}_pose_full"] = inv.pop(f"{p}_pose_raw")

    # ---- Assemble CanonicalSession ----
    canonical = CanonicalSession(
        session_id=session_id,
        schema_version=SCHEMA_VERSION,
        xdf_basename=xdf_path.name,
        xdf_md5=xdf_md5,
        duration_s=duration_s,
        t_start_lsl=t_start_lsl,
        protocol=protocol,
        modalities=modalities,
        roles=roles,
        markers=list(raw.get("markers", [])),
        marker_sources=list(raw.get("marker_sources", [])),
        pose_format=pose_format if pose_format is not None else "mediapipe33",
        quality_flags=quality_flags,
        stream_inventory=inv,
        arrays=arrays,
    )

    _save_canonical_session(canonical, npz_path, json_path)
    print(f"[digest] {session_id}: wrote {npz_path.name} ({len(arrays)} arrays) + {json_path.name}")
    return canonical


def digest_all(raw_dir: str | Path = RAW_SESSIONS_DIR,
               out_dir: str | Path = DIGEST_DIR,
               *,
               force: bool = False,
               only_canonical: bool = True,
               n_jobs: int = 1) -> list[CanonicalSession]:
    """Digest every XDF in ``raw_dir`` (or only canonical-flagged sessions).

    Parameters
    ----------
    n_jobs : int
        Parallel workers. Each worker holds one full XDF in memory (pyxdf
        eagerly loads); 4 workers x ~1.4 GB each ~= 6 GB peak. Use threading
        backend on Windows torch 2.10 stacks (per ``project_win_torch_dll_fix``);
        digestion is heavily I/O-bound (XDF parse + .npz write) so threads
        amortize file-system waits well even under the GIL.

    Returns the successfully-digested CanonicalSession list (skipped sessions
    are logged to stderr and excluded).
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    target_ids: set[str] | None = None
    if only_canonical:
        try:
            target_ids = set(list_canonical_sessions())
            print(f"[digest_all] restricting to {len(target_ids)} canonical sessions per "
                  f"configs/session_quality.yaml")
        except FileNotFoundError:
            print("[digest_all] WARN: configs/session_quality.yaml not found — "
                  "digesting every XDF (Step 0 audit must be run first)")
            target_ids = None

    xdfs = sorted(raw_dir.glob("*.xdf"))
    todo: list[Path] = []
    for xdf in xdfs:
        sid = xdf.stem
        if target_ids is not None and sid not in target_ids:
            print(f"[digest_all] {sid}: not canonical, skipping")
            continue
        todo.append(xdf)

    out: list[CanonicalSession] = []
    skipped: list[str] = []

    if n_jobs == 1 or len(todo) <= 1:
        for xdf in todo:
            result = digest_xdf(xdf, out_dir, force=force)
            (out if result is not None else skipped).append(
                result if result is not None else xdf.stem)
    else:
        from joblib import Parallel, delayed
        print(f"[digest_all] parallelizing {len(todo)} sessions across n_jobs={n_jobs} "
              f"(threading backend)")
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(digest_xdf)(xdf, out_dir, force=force) for xdf in todo
        )
        for xdf, result in zip(todo, results):
            if result is None:
                skipped.append(xdf.stem)
            else:
                out.append(result)

    skipped_ids = [s if isinstance(s, str) else s.session_id for s in skipped]
    print(f"\n[digest_all] {len(out)} succeeded; {len(skipped)} skipped: {skipped_ids}")
    return out


def load_digest(session_id: str,
                digest_dir: str | Path = DIGEST_DIR) -> CanonicalSession:
    """Read a previously-written digest from disk."""
    return _load_canonical_session(session_id, Path(digest_dir))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

PROTOCOL_REQUIRED_MARKERS = {
    "meditation": ["base_EO_start", "base_EC_start", "conv_1_start",
                   "meditate_B_start", "meditate_K_start", "conv_2_start"],
    "PE":         ["base_EO_start", "base_EC_start", "conv_1_start", "conv_2_start"],
}


def _detect_protocol(marker_labels: set[str]) -> str:
    meditation_tags = {"meditate_B_start", "meditate_K_start",
                       "meditate_B_stop", "meditate_K_stop"}
    pe_tags = {"PE_1_start", "PE_2_start", "PE_start", "PE_stop",
               "PE_1_stop", "PE_2_stop"}
    if marker_labels & meditation_tags:
        return "meditation"
    if marker_labels & pe_tags:
        return "PE"
    return "other"


def _missing_required_markers(protocol: str, labels: set[str]) -> list[str]:
    return [m for m in PROTOCOL_REQUIRED_MARKERS.get(protocol, []) if m not in labels]


def _save_canonical_session(cs: CanonicalSession, npz_path: Path, json_path: Path) -> None:
    np.savez(npz_path, **cs.arrays)
    sidecar = {
        "schema_version": cs.schema_version,
        "session_id": cs.session_id,
        "xdf_basename": cs.xdf_basename,
        "xdf_md5": cs.xdf_md5,
        "duration_s": cs.duration_s,
        "t_start_lsl": cs.t_start_lsl,
        "protocol": cs.protocol,
        "modalities": cs.modalities,
        "roles": {
            "p1_role": cs.roles.p1_role,
            "p2_role": cs.roles.p2_role,
            "p1_name": cs.roles.p1_name,
            "p2_name": cs.roles.p2_name,
            "role_source": cs.roles.role_source,
        },
        "markers": [[float(t), str(lab)] for t, lab in cs.markers],
        "marker_sources": list(cs.marker_sources),
        "pose_format": cs.pose_format,
        "quality_flags": {
            "excluded_modalities": dict(cs.quality_flags.excluded_modalities),
            "missing_markers": list(cs.quality_flags.missing_markers),
            "notes": list(cs.quality_flags.notes),
        },
        "stream_inventory": cs.stream_inventory,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


def _load_canonical_session(session_id: str, digest_dir: Path) -> CanonicalSession:
    npz_path = digest_dir / f"{session_id}.npz"
    json_path = digest_dir / f"{session_id}.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"No digest at {npz_path} / {json_path}")
    with json_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = dict(np.load(npz_path))
    roles = RoleAssignment(
        p1_role=meta["roles"]["p1_role"],
        p2_role=meta["roles"]["p2_role"],
        p1_name=meta["roles"]["p1_name"],
        p2_name=meta["roles"]["p2_name"],
        role_source=meta["roles"]["role_source"],
    )
    qf_dict = meta.get("quality_flags", {})
    qflags = QualityFlags(
        excluded_modalities=dict(qf_dict.get("excluded_modalities", {"p1": [], "p2": []})),
        missing_markers=list(qf_dict.get("missing_markers", [])),
        notes=list(qf_dict.get("notes", [])),
    )
    return CanonicalSession(
        session_id=meta["session_id"],
        schema_version=meta["schema_version"],
        xdf_basename=meta["xdf_basename"],
        xdf_md5=meta["xdf_md5"],
        duration_s=float(meta["duration_s"]),
        t_start_lsl=float(meta["t_start_lsl"]),
        protocol=meta["protocol"],
        modalities=list(meta["modalities"]),
        roles=roles,
        markers=[(float(t), str(lab)) for t, lab in meta.get("markers", [])],
        marker_sources=list(meta.get("marker_sources", [])),
        pose_format=meta["pose_format"],
        quality_flags=qflags,
        stream_inventory=dict(meta.get("stream_inventory", {})),
        arrays=arrays,
    )
