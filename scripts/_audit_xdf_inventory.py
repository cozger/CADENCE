#!/usr/bin/env python
"""One-off audit (Step 0): inspect every XDF in raw sessions/ and produce a
draft configs/session_quality.yaml.

For each XDF, reports:
    * EEG stream count (need >= 1; nominal 2 for dyadic recordings)
    * Pose channel count (132 = mediapipe33; 532 = rtmw133; other = unknown)
    * Landmark participant_name fields (resolves therapist/patient roles)
    * Marker stream sources (EventMarkers, BehavioralMarkers, ...)
    * Protocol detection (meditation vs PE vs other)
    * Missing-condition flags (e.g. y_32 missing conv_1)
    * Whether resolve_roles succeeds

Writes:
    configs/session_quality.yaml  (draft, awaiting user sign-off)
    results/audit/xdf_inventory.txt (human-readable report)

Usage:
    python scripts/_audit_xdf_inventory.py [--raw-dir "raw sessions"]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

# Hoist torch BEFORE numpy on Windows torch 2.10+numpy 2.4 stacks
# (per project_win_torch_dll_fix memory).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np

# Ensure project root is importable when run as a script.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from cadence.ingest.roles import (
    THERAPIST_NAME_ALIASES,
    PATIENT_ID_PATTERN,
    UnresolvedRolesError,
    resolve_roles,
)
from cadence.ingest.xdf_reader import (
    EEG_STREAM_NAME,
    normalize_markers,
)


# Minimum required markers for a session to be analytically usable.
# Per CLAUDE.md "Session Protocols":
#   meditation: base_EO, base_EC, conv_1, meditate_B, meditate_K, conv_2
#   PE:         base_EO, base_EC, conv_1, PE_1 (or PE), PE_2 (or PE_stop), conv_2
PROTOCOL_REQUIRED_MARKERS = {
    "meditation": ["base_EO_start", "base_EC_start", "conv_1_start",
                   "meditate_B_start", "meditate_K_start", "conv_2_start"],
    "PE":         ["base_EO_start", "base_EC_start", "conv_1_start",
                   "conv_2_start"],  # PE/PE_1/PE_2 detected separately
}


def _stream_name(s: dict) -> str:
    return s.get("info", {}).get("name", [""])[0]


def _stream_type(s: dict) -> str:
    return s.get("info", {}).get("type", [""])[0]


def _stream_landmark_pname(s: dict) -> str:
    """Extract participant_name from a Landmark stream's desc, if present."""
    info = s.get("info", {})
    desc_list = info.get("desc", [None])
    desc = desc_list[0] if desc_list else None
    if isinstance(desc, dict) and "participant_name" in desc:
        pname_field = desc["participant_name"]
        if isinstance(pname_field, list) and pname_field:
            return str(pname_field[0])
        if isinstance(pname_field, str):
            return pname_field
    return ""


def _detect_protocol(marker_labels: set[str]) -> str:
    """Classify session protocol from marker labels."""
    meditation_tags = {"meditate_B_start", "meditate_K_start", "meditate_B_stop", "meditate_K_stop"}
    pe_tags = {"PE_1_start", "PE_2_start", "PE_start", "PE_stop", "PE_1_stop", "PE_2_stop"}
    if marker_labels & meditation_tags:
        return "meditation"
    if marker_labels & pe_tags:
        return "PE"
    return "other"


def _read_existing_yaml(path: Path) -> dict:
    """Read prior session_quality.yaml so we can preserve user-edited fields
    (canonical flag, marker_overrides, manually-tuned excluded_modalities)
    on re-audit. Returns empty dict if file doesn't exist or is unreadable.
    """
    if not path.is_file():
        return {}
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("sessions", {}) or {}
    except Exception:
        return {}


def _audit_one_xdf(xdf_path: str, overrides_by_session: dict) -> dict:
    """Inspect a single XDF and return a structured audit record."""
    import pyxdf

    record: dict = {
        "xdf_basename": os.path.basename(xdf_path),
        "session_id": Path(xdf_path).stem,
        "errors": [],
    }

    try:
        streams, _hdr = pyxdf.load_xdf(
            xdf_path, dejitter_timestamps=False, synchronize_clocks=False
        )
    except Exception as e:
        record["errors"].append(f"pyxdf.load_xdf failed: {e!r}")
        return record

    # Filter EEG by name: only EmotivDataStream-EEG counts as participant EEG.
    # y_66 has an additional non-Emotiv "EEG"-typed stream (24-ch) we must ignore.
    eeg_streams = [s for s in streams
                   if _stream_type(s) == "EEG" and _stream_name(s) == EEG_STREAM_NAME]
    other_eeg = [s for s in streams
                 if _stream_type(s) == "EEG" and _stream_name(s) != EEG_STREAM_NAME]
    landmark_streams = {_stream_name(s): s for s in streams
                        if _stream_type(s) == "Landmark"}
    marker_streams = [s for s in streams if _stream_type(s) == "Markers"]

    record["eeg_count"] = len(eeg_streams)
    if other_eeg:
        record["eeg_other_streams"] = [_stream_name(s) for s in other_eeg]
    record["landmark_streams"] = sorted(landmark_streams.keys())
    record["marker_sources"] = sorted({_stream_name(s) for s in marker_streams})

    # --- EEG channel counts ---
    eeg_shapes = []
    eeg_sample_counts = []
    for s in eeg_streams:
        try:
            ts = np.asarray(s.get("time_stamps", []), dtype=np.float64)
            data = np.asarray(s.get("time_series", []))
            eeg_shapes.append(tuple(data.shape) if data.ndim == 2 else (len(ts), -1))
            eeg_sample_counts.append(len(ts))
        except Exception as e:
            record["errors"].append(f"EEG stream parse: {e!r}")
    record["eeg_shapes"] = eeg_shapes
    record["eeg_sample_counts"] = eeg_sample_counts

    # --- Pose format detection (delegates to canonical detector) ---
    from cadence.ingest.schema import detect_pose_format, UnknownPoseFormatError
    pose_formats = {}
    for s in streams:
        sname = _stream_name(s)
        if sname not in ("P1_pose", "P2_pose"):
            continue
        try:
            data = np.asarray(s.get("time_series", []))
            n_ch = int(data.shape[1]) if data.ndim == 2 else -1
        except Exception:
            n_ch = -1
        try:
            pose_formats[sname] = detect_pose_format(n_ch)
        except UnknownPoseFormatError:
            pose_formats[sname] = f"unknown({n_ch})"
    record["pose_formats"] = pose_formats

    # --- Role resolution ---
    p1_lm_pname = _stream_landmark_pname(landmark_streams.get("P1_landmarks", {}))
    p2_lm_pname = _stream_landmark_pname(landmark_streams.get("P2_landmarks", {}))
    record["p1_landmark_pname"] = p1_lm_pname
    record["p2_landmark_pname"] = p2_lm_pname

    try:
        ra = resolve_roles(streams)
        record["roles_resolved"] = True
        record["p1_role"] = ra.p1_role
        record["p2_role"] = ra.p2_role
        record["role_source"] = ra.role_source
    except UnresolvedRolesError as e:
        record["roles_resolved"] = False
        record["roles_error"] = str(e)

    # --- Marker collection + legacy normalization + conflict resolution ---
    raw_marker_events: list[tuple[float, str]] = []
    for s in marker_streams:
        try:
            for ts, row in zip(s.get("time_stamps", []), s.get("time_series", [])):
                lab = row[0] if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0 else row
                raw_marker_events.append((float(ts), str(lab)))
        except Exception:
            pass
    raw_marker_events = sorted(set(raw_marker_events), key=lambda x: x[0])
    raw_labels = sorted({lab for _t, lab in raw_marker_events})
    record["marker_labels_raw"] = raw_labels

    # Apply per-session marker_overrides (rename / drop) read from existing YAML,
    # then legacy baseline_start/stop -> base_EO/base_EC pairing,
    # then collapse duplicate _start/_stop events to widest range.
    sid = record["session_id"]
    overrides = overrides_by_session.get(sid, [])
    norm = normalize_markers(raw_marker_events, overrides=overrides)
    if overrides:
        record["marker_overrides_applied"] = len(overrides)
    norm_labels = sorted({lab for _t, lab in norm})
    record["marker_labels"] = norm_labels
    record["protocol"] = _detect_protocol(set(norm_labels))

    # --- Missing markers within protocol (post-normalization) ---
    required = PROTOCOL_REQUIRED_MARKERS.get(record["protocol"], [])
    record["missing_markers"] = [m for m in required if m not in norm_labels]

    return record


def _format_yaml_entry(rec: dict, prior_entry: dict | None = None) -> tuple[str, OrderedDict]:
    """Produce (session_id, yaml_entry_dict) classifying the session.

    Preserves user-edited fields from ``prior_entry`` (canonical flag,
    marker_overrides, manually-tuned excluded_modalities, custom notes).
    """
    sid = rec["session_id"]
    entry: OrderedDict = OrderedDict()
    notes: list[str] = []
    prior_entry = prior_entry or {}

    canonical = True

    # --- Hard exclusions ---
    if rec.get("eeg_count", 0) < 1:
        canonical = False
        notes.append("excluded from canonical: no EEG streams")

    if not rec.get("roles_resolved", False):
        canonical = False
        notes.append(
            f"excluded from canonical: unresolved roles "
            f"(p1={rec.get('p1_landmark_pname','?')!r}, p2={rec.get('p2_landmark_pname','?')!r})"
        )

    if rec.get("protocol") == "other":
        canonical = False
        notes.append("excluded from canonical: no recognized protocol markers")

    # --- Soft warnings (still canonical) ---
    if rec.get("missing_markers"):
        notes.append(f"missing markers: {rec['missing_markers']}")

    if rec.get("errors"):
        notes.append(f"audit errors: {rec['errors']}")

    # User can override the auto-classification (e.g. force canonical=true on
    # a session the audit flagged as 'other'). Prior YAML wins.
    if "canonical" in prior_entry:
        canonical = bool(prior_entry["canonical"])

    # --- Per-modality exclusions: prior YAML wins, else legacy defaults ---
    legacy_excl = {
        "y24_022526": {"p1": ["ecg", "ecg_features"],
                       "p2": ["ecg", "ecg_features", "pose_features", "pose"]},
        "y11_022526": {"p1": ["ecg", "ecg_features"],
                       "p2": ["ecg", "ecg_features"]},
        "Y_45_03302026": {"p1": ["pose", "pose_features"]},
        "y_33_04032026": {"p2": ["pose", "pose_features"]},
        "y_19_3242026": {"p1": ["eeg", "eeg_features"],
                         "p2": ["eeg", "eeg_features"]},
    }
    excluded_modalities = prior_entry.get("excluded_modalities") or legacy_excl.get(sid)

    entry["canonical"] = canonical
    if excluded_modalities:
        entry["excluded_modalities"] = excluded_modalities
    if notes:
        entry["notes"] = notes
    # --- Preserved user fields (always carry through) ---
    if prior_entry.get("marker_overrides"):
        entry["marker_overrides"] = list(prior_entry["marker_overrides"])
    # --- Audit-derived metadata for review ---
    entry["audit"] = OrderedDict([
        ("protocol", rec.get("protocol", "unknown")),
        ("eeg_count", rec.get("eeg_count", 0)),
        ("p1_landmark_pname", rec.get("p1_landmark_pname", "")),
        ("p2_landmark_pname", rec.get("p2_landmark_pname", "")),
        ("p1_role", rec.get("p1_role", "")),
        ("p2_role", rec.get("p2_role", "")),
        ("role_source", rec.get("role_source", "")),
        ("pose_formats", rec.get("pose_formats", {})),
        ("marker_sources", rec.get("marker_sources", [])),
    ])
    if "marker_overrides_applied" in rec:
        entry["audit"]["marker_overrides_applied"] = rec["marker_overrides_applied"]
    return sid, entry


def _dump_yaml(records: list[dict], path: Path,
               prior_sessions: dict | None = None) -> None:
    """Hand-write YAML for nicer diffs / no PyYAML dependency.

    Preserves user-edited fields from ``prior_sessions[sid]`` (canonical flag,
    marker_overrides, manually-tuned excluded_modalities) on re-audit.
    """
    prior_sessions = prior_sessions or {}
    lines: list[str] = []
    lines.append("# CADENCE session quality registry (Step 0 draft)")
    lines.append("# Generated by scripts/_audit_xdf_inventory.py")
    lines.append("# REVIEW BEFORE COMMITTING — flip `canonical:` flags as needed,")
    lines.append("# delete `audit:` blocks once approved.")
    lines.append("")
    lines.append("sessions:")
    for rec in sorted(records, key=lambda r: r["session_id"].lower()):
        prior = prior_sessions.get(rec["session_id"]) or {}
        sid, entry = _format_yaml_entry(rec, prior_entry=prior)
        lines.append(f"  {sid}:")
        lines.append(f"    canonical: {str(entry['canonical']).lower()}")
        if "excluded_modalities" in entry:
            lines.append(f"    excluded_modalities:")
            for p, mods in entry["excluded_modalities"].items():
                lines.append(f"      {p}: {mods}")
        if "notes" in entry:
            lines.append(f"    notes:")
            for n in entry["notes"]:
                # Escape any quotes in note text
                escaped = n.replace('"', '\\"')
                lines.append(f"      - \"{escaped}\"")
        if "marker_overrides" in entry:
            lines.append(f"    marker_overrides:")
            for ov in entry["marker_overrides"]:
                kind = ov.get("kind", "")
                # Lead the block with `- kind:` then dump the rest as siblings.
                lines.append(f"      - kind: {kind}")
                for k, v in ov.items():
                    if k == "kind":
                        continue
                    lines.append(f"        {k}: {v!r}")
        if "audit" in entry:
            lines.append(f"    audit:")
            for k, v in entry["audit"].items():
                if isinstance(v, dict):
                    lines.append(f"      {k}:")
                    for kk, vv in v.items():
                        lines.append(f"        {kk}: {vv!r}")
                elif isinstance(v, list):
                    lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"      {k}: {v!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dump_report(records: list[dict], path: Path) -> None:
    """Human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CADENCE XDF Inventory Audit")
    lines.append("=" * 80)

    canonical = [r for r in records if r.get("roles_resolved") and r.get("eeg_count", 0) >= 1
                 and r.get("protocol") in ("meditation", "PE")]
    lines.append(f"\nTotal XDFs:           {len(records)}")
    lines.append(f"Canonical candidates: {len(canonical)}  "
                 f"(EEG present, roles resolved, protocol detected)")

    by_protocol = {}
    for r in canonical:
        by_protocol.setdefault(r["protocol"], []).append(r["session_id"])
    for proto, sessions in sorted(by_protocol.items()):
        lines.append(f"  {proto}: {len(sessions)}  ({', '.join(sorted(sessions))})")

    lines.append("\n" + "=" * 80)
    lines.append("Per-session details")
    lines.append("=" * 80)

    for rec in sorted(records, key=lambda r: r["session_id"].lower()):
        lines.append(f"\n{rec['session_id']}  ({rec.get('xdf_basename')})")
        if rec.get("errors"):
            lines.append(f"  ERRORS: {rec['errors']}")
        lines.append(f"  protocol={rec.get('protocol','?')}  "
                     f"eeg={rec.get('eeg_count',0)}  "
                     f"landmark_streams={rec.get('landmark_streams',[])}")
        lines.append(f"  p1_landmark_pname={rec.get('p1_landmark_pname','')!r}  "
                     f"p2_landmark_pname={rec.get('p2_landmark_pname','')!r}")
        if rec.get("roles_resolved"):
            lines.append(f"  ROLES OK: p1={rec['p1_role']}  p2={rec['p2_role']}  "
                         f"source={rec['role_source']}")
        else:
            lines.append(f"  ROLES UNRESOLVED: {rec.get('roles_error','?')}")
        lines.append(f"  pose_formats={rec.get('pose_formats',{})}")
        lines.append(f"  marker_sources={rec.get('marker_sources',[])}")
        if rec.get("missing_markers"):
            lines.append(f"  MISSING MARKERS: {rec['missing_markers']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw sessions",
                    help="Directory containing .xdf files")
    ap.add_argument("--out-yaml", default="configs/session_quality.yaml",
                    help="Output YAML path")
    ap.add_argument("--out-report", default="results/audit/xdf_inventory.txt",
                    help="Output text report path")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: raw dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    out_yaml = Path(args.out_yaml)
    out_report = Path(args.out_report)

    # Read prior YAML to preserve user-edited fields and apply marker_overrides
    # to this audit pass.
    prior_sessions = _read_existing_yaml(out_yaml)
    overrides_by_session = {
        sid: list(entry.get("marker_overrides", []) or [])
        for sid, entry in prior_sessions.items()
        if entry
    }

    xdf_paths = sorted(raw_dir.glob("*.xdf"))
    print(f"Auditing {len(xdf_paths)} XDFs in {raw_dir}/ ...\n", flush=True)

    records: list[dict] = []
    for i, xdf in enumerate(xdf_paths, 1):
        sid = xdf.stem
        print(f"  [{i}/{len(xdf_paths)}] {sid} ... ", end="", flush=True)
        try:
            rec = _audit_one_xdf(str(xdf), overrides_by_session)
            records.append(rec)
            ok = "OK" if rec.get("roles_resolved") else "ROLE-?"
            n_ov = rec.get("marker_overrides_applied", 0)
            ov_tag = f"  overrides={n_ov}" if n_ov else ""
            print(f"{ok}  protocol={rec.get('protocol','?')}  "
                  f"eeg={rec.get('eeg_count',0)}{ov_tag}", flush=True)
        except Exception as e:
            print(f"AUDIT ERROR: {e}", flush=True)
            traceback.print_exc()
            records.append({
                "xdf_basename": xdf.name,
                "session_id": sid,
                "errors": [f"audit crashed: {e!r}"],
            })

    _dump_yaml(records, out_yaml, prior_sessions=prior_sessions)
    _dump_report(records, out_report)

    print(f"\nWrote draft YAML:    {out_yaml}")
    print(f"Wrote audit report:  {out_report}")
    print(f"\nReview both files. After approval, the YAML drives Step 5 digestion.")


if __name__ == "__main__":
    main()
