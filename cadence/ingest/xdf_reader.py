"""Raw XDF -> in-memory streams dict.

Pure structural read: extracts EEG, ECG, landmark, pose, and marker streams
without any value transforms. Marker merge logic (originally at
``cadence/data/xdf_loader.py:37-51``) is kept inline rather than factored out
into a separate module — it's tightly coupled to the XDF read pass.

Returns a dict with raw arrays + timestamps + merged markers + per-stream
provenance (so the digest layer can fill in ``stream_inventory``).
"""

from __future__ import annotations

import os

import numpy as np


# Stream identification — matches existing xdf_loader.py conventions.
EEG_TYPE = "EEG"
EEG_STREAM_NAME = "EmotivDataStream-EEG"
"""Only EEG streams with this exact name are dyadic participant EEG.

Some sessions (e.g. y_66) have an additional ``EEG``-typed stream with a
different name (e.g. bare ``"EEG"``, 24 channels). Those are auxiliary /
non-participant and must not be treated as p1/p2 EEG.
"""

MARKER_TYPE = "Markers"
LANDMARK_TYPE = "Landmark"

ECG_NAMES = {"P1_ecg", "P2_ecg"}
LANDMARK_NAMES = {"P1_landmarks", "P2_landmarks"}
POSE_NAMES = {"P1_pose", "P2_pose"}

# Canonical condition names used after marker normalization.
CONDITIONS = (
    "base_EO", "base_EC", "conv_1", "conv_2",
    "meditate_B", "meditate_K",
    "PE", "PE_1", "PE_2",
)


def load_xdf_streams(xdf_path: str, *,
                     p1_eeg_index: int = 0,
                     p2_eeg_index: int = 1,
                     dejitter_timestamps: bool = True,
                     synchronize_clocks: bool = True,
                     marker_overrides: list[dict] | None = None) -> dict:
    """Load every relevant stream from an XDF file.

    Parameters
    ----------
    xdf_path : str
    p1_eeg_index, p2_eeg_index : int
        Which EmotivDataStream-EEG stream is P1/P2 (XDF stream order is not
        consistent — defaults match historical behavior).
    dejitter_timestamps, synchronize_clocks : bool
        Forwarded to ``pyxdf.load_xdf``.

    Returns
    -------
    dict with keys (per participant ``p`` ∈ {p1, p2}, omitting any absent stream):
        ``{p}_eeg_raw, {p}_eeg_ts``       — raw 19-col Emotiv stream + ts
        ``{p}_ecg_raw, {p}_ecg_ts``       — Polar H10 ECG
        ``{p}_landmarks_raw, {p}_landmarks_ts`` — full 1489-col landmark stream
        ``{p}_pose_raw, {p}_pose_ts``     — raw pose stream (132 or 532 cols)
    plus:
        ``markers``         — list[(t, label)] sorted by t (merged across
                              EventMarkers + BehavioralMarkers + ...)
        ``marker_sources``  — sorted list of stream names that contributed markers
        ``streams``         — full pyxdf streams list (for downstream role
                              resolution; do not rely on field stability)
        ``stream_inventory`` — {key: {src_name, shape, srate_hz}} for the JSON sidecar
        ``t_start_lsl``     — earliest timestamp across all streams (LSL absolute time)
        ``duration_s``      — latest minus earliest timestamp
    """
    import pyxdf

    streams, _hdr = pyxdf.load_xdf(
        xdf_path,
        dejitter_timestamps=dejitter_timestamps,
        synchronize_clocks=synchronize_clocks,
    )

    session: dict = {}
    eeg_streams: list[tuple[str, np.ndarray, np.ndarray]] = []
    marker_records: list[tuple[float, str]] = []
    marker_sources: set[str] = set()
    inventory: dict[str, dict] = {}
    all_starts: list[float] = []
    all_ends: list[float] = []

    for s in streams:
        info = s.get("info", {})
        name = info.get("name", [""])[0]
        stype = info.get("type", [""])[0]
        timestamps = np.asarray(s.get("time_stamps", []), dtype=np.float64)

        # ---- Marker streams (merge across all of them) ----
        # Replaces the lines 37-51 logic in cadence/data/xdf_loader.py:
        # accumulates ALL Markers streams (EventMarkers, BehavioralMarkers, ...)
        # so neither protocol markers nor gesture events are lost.
        if stype == MARKER_TYPE:
            ts_data = s.get("time_series", [])
            if len(timestamps) > 0:
                for ts, row in zip(timestamps.tolist(), ts_data):
                    if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0:
                        lab = row[0]
                    else:
                        lab = row
                    marker_records.append((float(ts), str(lab)))
                marker_sources.add(name)
            continue

        # ---- Numeric streams ----
        try:
            data = np.asarray(s.get("time_series", []), dtype=np.float64)
        except (ValueError, TypeError):
            continue

        if data.ndim == 0 or data.shape[0] == 0:
            continue

        srate_hz = _estimate_srate(timestamps)

        # Only EEG/ECG/landmarks/pose streams contribute to t_start_lsl /
        # duration_s. Some auxiliary streams (e.g. P2_gaze, MutualGaze) use a
        # different LSL clock convention (Unix epoch vs session-relative) which
        # would corrupt min/max if mixed in. See Y_55 case where gaze starts at
        # 1.77e9 while video/audio/EEG start at ~47887.
        kept = False
        if stype == EEG_TYPE:
            # Only accept the EmotivDataStream-EEG dyadic participant stream.
            # y_66 has an extra non-Emotiv EEG-typed stream (24-ch) we must skip.
            if name != EEG_STREAM_NAME:
                continue
            eeg_streams.append((name, data, timestamps))
            kept = True
        elif name == "P1_ecg":
            session["p1_ecg_raw"] = data
            session["p1_ecg_ts"] = timestamps
            inventory["p1_ecg_raw"] = _inv(name, data.shape, srate_hz)
            kept = True
        elif name == "P2_ecg":
            session["p2_ecg_raw"] = data
            session["p2_ecg_ts"] = timestamps
            inventory["p2_ecg_raw"] = _inv(name, data.shape, srate_hz)
            kept = True
        elif name == "P1_landmarks":
            session["p1_landmarks_raw"] = data
            session["p1_landmarks_ts"] = timestamps
            inventory["p1_landmarks_raw"] = _inv(name, data.shape, srate_hz)
            kept = True
        elif name == "P2_landmarks":
            session["p2_landmarks_raw"] = data
            session["p2_landmarks_ts"] = timestamps
            inventory["p2_landmarks_raw"] = _inv(name, data.shape, srate_hz)
            kept = True
        elif name == "P1_pose":
            session["p1_pose_raw"] = data
            session["p1_pose_ts"] = timestamps
            inventory["p1_pose_raw"] = _inv(name, data.shape, srate_hz)
            kept = True
        elif name == "P2_pose":
            session["p2_pose_raw"] = data
            session["p2_pose_ts"] = timestamps
            inventory["p2_pose_raw"] = _inv(name, data.shape, srate_hz)
            kept = True

        if kept and len(timestamps) > 0:
            all_starts.append(float(timestamps[0]))
            all_ends.append(float(timestamps[-1]))

    # ---- EEG: assign by index (XDF stream order varies across sessions) ----
    if len(eeg_streams) >= 2:
        p1_name, p1_data, p1_ts = eeg_streams[p1_eeg_index]
        p2_name, p2_data, p2_ts = eeg_streams[p2_eeg_index]
        session["p1_eeg_raw"] = p1_data
        session["p1_eeg_ts"] = p1_ts
        session["p2_eeg_raw"] = p2_data
        session["p2_eeg_ts"] = p2_ts
        inventory["p1_eeg_raw"] = _inv(p1_name, p1_data.shape, _estimate_srate(p1_ts))
        inventory["p2_eeg_raw"] = _inv(p2_name, p2_data.shape, _estimate_srate(p2_ts))
    elif len(eeg_streams) == 1:
        name, data, ts = eeg_streams[0]
        session["p1_eeg_raw"] = data
        session["p1_eeg_ts"] = ts
        inventory["p1_eeg_raw"] = _inv(name, data.shape, _estimate_srate(ts))

    # ---- Marker pipeline: dedupe → overrides → legacy baseline → widest-range ----
    marker_records = sorted(set(marker_records), key=lambda x: x[0])
    marker_records = normalize_markers(marker_records, overrides=marker_overrides)

    session["markers"] = marker_records
    session["marker_sources"] = sorted(marker_sources)
    session["streams"] = streams  # for resolve_roles downstream
    session["stream_inventory"] = inventory
    session["t_start_lsl"] = min(all_starts) if all_starts else 0.0
    session["duration_s"] = (max(all_ends) - min(all_starts)) if all_starts else 0.0

    return session


def _estimate_srate(timestamps: np.ndarray) -> float:
    """Median-based sample-rate estimate (Hz). 0 if fewer than 2 samples."""
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(timestamps)
    median = float(np.median(diffs))
    return 1.0 / median if median > 0 else 0.0


def _inv(src_name: str, shape: tuple, srate_hz: float) -> dict:
    return {"src_name": src_name, "shape": list(shape), "srate_hz": float(srate_hz)}


import math


def normalize_markers(
    raw_events: list[tuple[float, str]],
    *,
    overrides: list[dict] | None = None,
) -> list[tuple[float, str]]:
    """Full marker normalization pipeline.

    Order of operations:

    1. Dedupe + sort by timestamp.
    2. Apply per-session ``marker_overrides`` (rename / drop) if any.
    3. Translate legacy ``baseline_start`` / ``baseline_stop`` pairs into
       ``base_EO`` / ``base_EC`` start/stop.
    4. Collapse duplicate ``_start`` / ``_stop`` events to widest range
       (earliest start, latest stop) per condition.

    The override step is first so corrections can also rescue events that
    would otherwise be discarded by widest-range (e.g. a mis-clicked
    ``conv_2_start`` that should be ``conv_1_start``).
    """
    events = sorted(set(raw_events), key=lambda x: x[0])
    if overrides:
        events = _apply_marker_overrides(events, overrides)
    events = _normalize_legacy_baseline_markers(events)
    events = _resolve_widest_range_conflicts(events)
    return events


def _apply_marker_overrides(
    markers: list[tuple[float, str]],
    overrides: list[dict],
) -> list[tuple[float, str]]:
    """Apply per-session marker corrections from ``configs/session_quality.yaml``.

    Each override is a dict with ``kind`` ∈ {``rename``, ``drop``}:

    * ``{kind: rename, from: <label>, to: <label>, t_min?: float, t_max?: float}``
      Rename matching events to ``to``. Optional time-window match
      (session-relative seconds, computed from earliest marker timestamp).
    * ``{kind: drop, label: <label>, t_min?: float, t_max?: float}``
      Discard matching events.

    Times are session-relative (subtracts earliest marker timestamp) so that
    YAML entries are stable across sessions.
    """
    if not markers:
        return list(markers)
    t0 = min(t for t, _l in markers)

    out: list[tuple[float, str]] = []
    for ts, lab in markers:
        rel = ts - t0
        new_lab = lab
        keep = True
        for ov in overrides:
            kind = ov.get("kind", "")
            t_min = float(ov.get("t_min", -math.inf))
            t_max = float(ov.get("t_max", math.inf))
            if not (t_min <= rel <= t_max):
                continue
            if kind == "rename" and lab == ov.get("from"):
                new_lab = str(ov.get("to", lab))
            elif kind == "drop" and lab == ov.get("label"):
                keep = False
                break
        if keep:
            out.append((ts, new_lab))
    return sorted(out, key=lambda x: x[0])


def _normalize_legacy_baseline_markers(
    markers: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """Translate legacy ``baseline_start`` / ``baseline_stop`` events into
    ``base_EO`` and ``base_EC`` start/stop pairs.

    Older recordings emitted a single ``baseline_*`` block per condition. The
    convention adopted later (per user direction): the **first** start-stop
    pair is eyes-open; the **second** start-stop pair is eyes-closed. Any
    further pairs are dropped with a logged note (no current session has them).

    Original events are preserved alongside the renamed copies so the audit
    trail is not lost — downstream conflict resolution will collapse duplicates.
    """
    out: list[tuple[float, str]] = list(markers)

    # Pair starts and stops in chronological order.
    starts = [(t, l) for (t, l) in markers if l == "baseline_start"]
    stops = [(t, l) for (t, l) in markers if l == "baseline_stop"]
    n_pairs = min(len(starts), len(stops))
    rename_map = ["base_EO", "base_EC"]  # 1st pair -> EO, 2nd -> EC, drop rest.
    for i in range(min(n_pairs, len(rename_map))):
        cond = rename_map[i]
        out.append((starts[i][0], f"{cond}_start"))
        out.append((stops[i][0], f"{cond}_stop"))

    return sorted(out, key=lambda x: x[0])


def _resolve_widest_range_conflicts(
    markers: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """For each condition, collapse multiple start/stop events to the widest range.

    If a condition's ``_start`` appears more than once, keep only the earliest;
    if a ``_stop`` appears more than once, keep only the latest. The interval
    becomes ``[earliest_start, latest_stop]``. All other (non-condition) markers
    are passed through untouched.

    Conditions handled: those listed in ``CONDITIONS`` (base_EO, base_EC,
    conv_1, conv_2, meditate_B, meditate_K, PE, PE_1, PE_2).
    """
    earliest_start: dict[str, float] = {}
    latest_stop: dict[str, float] = {}
    other: list[tuple[float, str]] = []

    for ts, lab in markers:
        matched = False
        for cond in CONDITIONS:
            if lab == f"{cond}_start":
                prev = earliest_start.get(cond)
                if prev is None or ts < prev:
                    earliest_start[cond] = ts
                matched = True
                break
            if lab == f"{cond}_stop":
                prev = latest_stop.get(cond)
                if prev is None or ts > prev:
                    latest_stop[cond] = ts
                matched = True
                break
        if not matched:
            other.append((ts, lab))

    out: list[tuple[float, str]] = list(other)
    for cond, ts in earliest_start.items():
        out.append((ts, f"{cond}_start"))
    for cond, ts in latest_stop.items():
        out.append((ts, f"{cond}_stop"))
    return sorted(out, key=lambda x: x[0])


def compute_xdf_md5(xdf_path: str, chunk: int = 1 << 20) -> str:
    """MD5 of a raw XDF file. Used for digest staleness checks."""
    import hashlib
    h = hashlib.md5()
    with open(xdf_path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
