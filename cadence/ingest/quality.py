"""Reads configs/session_quality.yaml — the single classification scheme for sessions.

Replaces the hardcoded ``EXCLUDED_MODALITIES`` dict at
``cadence/data/alignment.py:409``. The YAML is hand-maintained (drafted by
``scripts/_audit_xdf_inventory.py``, finalized by user sign-off in Step 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_QUALITY_PATH = Path("configs/session_quality.yaml")


@dataclass(frozen=True)
class SessionQuality:
    """Per-session quality metadata."""
    canonical: bool
    excluded_modalities: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    marker_overrides: list[dict] = field(default_factory=list)
    """Per-session marker corrections, applied at digest time.

    Each entry is a dict with ``kind`` ∈ {``rename``, ``drop``}; see
    ``cadence.ingest.xdf_reader._apply_marker_overrides`` for the schema.
    Use sparingly — meant for genuine recording errors (operator mis-clicks,
    spurious labels), not for analytical re-interpretation.
    """


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"session_quality.yaml not found at {path}. "
            f"Run scripts/_audit_xdf_inventory.py to generate a draft."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "sessions" not in data:
        raise ValueError(f"Invalid session_quality.yaml: missing 'sessions' key")
    return data


def load_session_quality(session_id: str,
                         path: Path = DEFAULT_QUALITY_PATH) -> SessionQuality:
    """Return ``SessionQuality`` for one session. Raises if session missing."""
    data = _load_yaml(path)
    sessions = data["sessions"] or {}
    entry = sessions.get(session_id)
    if entry is None:
        raise KeyError(
            f"session_id={session_id!r} not in {path}. "
            f"Add it to configs/session_quality.yaml or rerun the audit."
        )
    return _entry_to_quality(entry)


def _entry_to_quality(entry: dict) -> SessionQuality:
    return SessionQuality(
        canonical=bool(entry.get("canonical", False)),
        excluded_modalities=dict(entry.get("excluded_modalities", {}) or {}),
        notes=list(entry.get("notes", []) or []),
        marker_overrides=list(entry.get("marker_overrides", []) or []),
    )


def load_marker_overrides(session_id: str,
                          path: Path = DEFAULT_QUALITY_PATH) -> list[dict]:
    """Return marker_overrides list for a session (empty if none / session unknown)."""
    try:
        return list(load_session_quality(session_id, path).marker_overrides)
    except (KeyError, FileNotFoundError):
        return []


def list_canonical_sessions(path: Path = DEFAULT_QUALITY_PATH) -> list[str]:
    """Return session IDs with ``canonical: true``."""
    data = _load_yaml(path)
    sessions = data["sessions"] or {}
    return sorted(sid for sid, entry in sessions.items()
                  if entry and entry.get("canonical"))


def list_all_sessions(path: Path = DEFAULT_QUALITY_PATH) -> list[str]:
    """Return all session IDs in the registry (canonical or not)."""
    data = _load_yaml(path)
    sessions = data["sessions"] or {}
    return sorted(sessions.keys())


# Legacy compatibility view for cadence/data/__init__.py shim.
def excluded_modalities_legacy_dict(path: Path = DEFAULT_QUALITY_PATH) -> dict[str, dict[str, list[str]]]:
    """Reproduce the legacy ``EXCLUDED_MODALITIES`` dict shape from the YAML.

    Used by the deprecation shim so legacy code that does
    ``from cadence.data import EXCLUDED_MODALITIES`` continues to work.
    """
    data = _load_yaml(path)
    sessions = data["sessions"] or {}
    out: dict[str, dict[str, list[str]]] = {}
    for sid, entry in sessions.items():
        if not entry:
            continue
        excl = entry.get("excluded_modalities") or {}
        if excl:
            out[sid] = {p: list(mods) for p, mods in excl.items()}
    return out


def apply_modality_exclusions(session: dict, session_id: str,
                              path: Path = DEFAULT_QUALITY_PATH) -> dict:
    """Mask excluded modalities as fully invalid (zeros + False validity).

    Drop-in replacement for ``cadence.data.alignment.apply_modality_exclusions``.
    """
    import numpy as np
    try:
        sq = load_session_quality(session_id, path)
    except KeyError:
        return session  # unknown session — nothing to exclude
    for participant, mods in (sq.excluded_modalities or {}).items():
        for mod in mods:
            data_key = f"{participant}_{mod}"
            valid_key = f"{participant}_{mod}_valid"
            if data_key in session:
                session[data_key] = np.zeros_like(session[data_key])
            if valid_key in session:
                session[valid_key] = np.zeros_like(session[valid_key], dtype=bool)
    return session
