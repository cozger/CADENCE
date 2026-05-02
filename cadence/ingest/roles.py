"""Single source of truth for therapist/patient role resolution from XDF streams.

Replaces every duplicate role-detection path:
    - cadence.data.xdf_loader._detect_roles  (Sway-only, defaults silently)
    - analysis.eeglab_wavelet.session_io._resolve_roles_from_xdf
    - analysis.eeglab_wavelet.session_io.load_au_from_xdf_direct (inline)
    - 6 script-level call sites

Algorithm
---------
For each Landmark-typed stream (P1_landmarks, P2_landmarks), read the
``desc -> participant_name`` field. Classify each name via two rules:

* ``is_therapist(name) := name in THERAPIST_NAME_ALIASES``  (exact match)
* ``is_patient(name)   := PATIENT_ID_PATTERN.match(name) is not None``

Exactly one therapist + exactly one patient → resolved.
Both therapists, both patients, or both unresolvable → raise
``UnresolvedRolesError``.  No silent fallback to defaults — that was the
source of the original bug for RA-tagged sessions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


THERAPIST_NAME_ALIASES: tuple[str, ...] = ("Sway", "RA")
"""Known names of the therapist in the LSL ``participant_name`` field.

* ``Sway`` — older recordings (2024–early 2026).
* ``RA``   — research-assistant alias used in early 2026 onward.

Matched **case-insensitively**: actual data has both ``Sway`` and ``sway``.
"""


PATIENT_ID_PATTERN = re.compile(
    r"^(?:[Yy]_?\d+|SYNC_\d+)(?:[_-].*)?$"
)
"""Patient session ID regex (case-insensitive on Y prefix, suffix-tolerant).

Matches: ``Y10``, ``y_03``, ``Y24``, ``y06``, ``Y_16``, ``Y01_020626``,
``Y26_022728``, ``SYNC_42``, ``SYNC_3``. The optional ``[_-]<suffix>`` covers
the older session-naming convention where the date was appended to the
participant name (``Y01_020626``).

Does not match: ``Sway``, ``sway``, ``RA``, ``unknown``, ``research_assistant``.
"""


class UnresolvedRolesError(RuntimeError):
    """Raised when role resolution cannot identify exactly one therapist and one patient."""

    def __init__(self, p1_name: str, p2_name: str, reason: str = ""):
        msg = (
            f"Unresolved roles: p1_name={p1_name!r}, p2_name={p2_name!r}"
            + (f" — {reason}" if reason else "")
        )
        super().__init__(msg)
        self.p1_name = p1_name
        self.p2_name = p2_name


@dataclass(frozen=True)
class RoleAssignment:
    """Resolved therapist/patient assignment for a session.

    The XDF stream order (p1/p2) is not consistent across sessions — sometimes
    p1 is the therapist, sometimes p2 is. ``RoleAssignment`` stores roles as
    metadata so downstream code can index correctly without reordering arrays.

    No ``role_resolved`` field — use ``role_source != "unresolved"``.
    Unresolved sessions raise ``UnresolvedRolesError`` rather than producing a
    sentinel ``RoleAssignment``.
    """

    p1_role: Literal["therapist", "patient"]
    p2_role: Literal["therapist", "patient"]
    p1_name: str
    p2_name: str
    role_source: Literal["xdf_landmark", "xdf_ra_alias"]


def _read_landmark_participant_names(streams: list[dict]) -> dict[str, str]:
    """Return {stream_name: participant_name} for every Landmark-typed stream."""
    out: dict[str, str] = {}
    for s in streams:
        info = s.get("info", {})
        stype_list = info.get("type", [""])
        stype = stype_list[0] if stype_list else ""
        if stype != "Landmark":
            continue
        name_list = info.get("name", [""])
        sname = name_list[0] if name_list else ""
        desc_list = info.get("desc", [None])
        desc = desc_list[0] if desc_list else None
        if isinstance(desc, dict) and "participant_name" in desc:
            pname_field = desc["participant_name"]
            if isinstance(pname_field, list) and pname_field:
                out[sname] = str(pname_field[0])
            elif isinstance(pname_field, str):
                out[sname] = pname_field
    return out


def _classify(name: str) -> Literal["therapist", "patient", "unknown"]:
    """Classify a participant_name into therapist / patient / unknown.

    Therapist matching is case-insensitive (some older recordings use
    lowercase 'sway'). Patient matching uses ``PATIENT_ID_PATTERN``,
    which is suffix-tolerant for older date-suffixed IDs.
    """
    if name.lower() in (a.lower() for a in THERAPIST_NAME_ALIASES):
        return "therapist"
    if PATIENT_ID_PATTERN.match(name):
        return "patient"
    return "unknown"


def resolve_roles(streams: list[dict]) -> RoleAssignment:
    """Resolve therapist/patient roles from raw pyxdf stream list.

    Parameters
    ----------
    streams : list of dict
        The first return of ``pyxdf.load_xdf(path)``.

    Returns
    -------
    RoleAssignment

    Raises
    ------
    UnresolvedRolesError
        If both Landmark streams classify the same way (both therapist /
        both patient / both unknown) or if either Landmark stream is missing.
    """
    pnames = _read_landmark_participant_names(streams)
    p1_name = pnames.get("P1_landmarks", "unknown")
    p2_name = pnames.get("P2_landmarks", "unknown")

    if "P1_landmarks" not in pnames or "P2_landmarks" not in pnames:
        raise UnresolvedRolesError(
            p1_name, p2_name,
            reason="missing landmark stream(s) or no participant_name in desc",
        )

    p1_class = _classify(p1_name)
    p2_class = _classify(p2_name)

    if p1_class == "therapist" and p2_class == "patient":
        source = "xdf_ra_alias" if p1_name.lower() == "ra" else "xdf_landmark"
        return RoleAssignment(
            p1_role="therapist", p2_role="patient",
            p1_name=p1_name, p2_name=p2_name,
            role_source=source,
        )
    if p1_class == "patient" and p2_class == "therapist":
        source = "xdf_ra_alias" if p2_name.lower() == "ra" else "xdf_landmark"
        return RoleAssignment(
            p1_role="patient", p2_role="therapist",
            p1_name=p1_name, p2_name=p2_name,
            role_source=source,
        )

    # Both therapists / both patients / one unknown — refuse.
    raise UnresolvedRolesError(
        p1_name, p2_name,
        reason=f"p1 classified {p1_class!r}, p2 classified {p2_class!r}",
    )


def roles_dict_from_session(session: dict) -> dict:
    """Backward-compat wrapper for legacy callers using
    ``cadence.data.xdf_loader._detect_roles(session)``.

    Pre-2026-05-01 callers passed the legacy session dict and expected a
    plain dict back (``{p1_role, p2_role, p1_name, p2_name}``). The new
    digest layer already populates these keys, so the new shape is just a
    projection of the session dict; this function exists so legacy scripts
    can drop in the new import path without restructuring their loops.

    Falls back to ``unknown/unknown`` for sessions that don't have the keys
    populated (e.g. an older raw XDF dict from ``cadence.data.xdf_loader``
    that didn't compute roles upstream).
    """
    return {
        "p1_role": str(session.get("p1_role", "unknown")),
        "p2_role": str(session.get("p2_role", "unknown")),
        "p1_name": str(session.get("p1_name", "")),
        "p2_name": str(session.get("p2_name", "")),
        "role_source": str(session.get("role_source", "unresolved")),
    }
