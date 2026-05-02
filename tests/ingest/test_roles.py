"""Tests for cadence.ingest.roles.resolve_roles.

Five hand-built fake-stream cases plus 12 real-XDF stub tests covering the
canonical inventory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cadence.ingest.roles import (
    THERAPIST_NAME_ALIASES,
    PATIENT_ID_PATTERN,
    UnresolvedRolesError,
    resolve_roles,
    _classify,
)


# --------------------------------------------------------------------------
# Helpers: build a minimal pyxdf-shaped stream list with just two Landmark
# streams. resolve_roles only inspects type=='Landmark' streams.
# --------------------------------------------------------------------------

def _landmark_stream(name: str, participant_name: str) -> dict:
    return {
        "info": {
            "name": [name],
            "type": ["Landmark"],
            "desc": [{"participant_name": [participant_name]}],
        },
        "time_series": [],
        "time_stamps": [],
    }


def _other_stream(stype: str, name: str = "Other") -> dict:
    """Non-Landmark stream — should be ignored by resolve_roles."""
    return {
        "info": {"name": [name], "type": [stype]},
        "time_series": [],
        "time_stamps": [],
    }


# --------------------------------------------------------------------------
# Hand-built fake-stream cases
# --------------------------------------------------------------------------

def test_fake_sway_p1_y06_p2_resolves_p1_therapist():
    streams = [
        _landmark_stream("P1_landmarks", "Sway"),
        _landmark_stream("P2_landmarks", "y_06"),
        _other_stream("EEG"),
    ]
    ra = resolve_roles(streams)
    assert ra.p1_role == "therapist"
    assert ra.p2_role == "patient"
    assert ra.p1_name == "Sway"
    assert ra.p2_name == "y_06"
    assert ra.role_source == "xdf_landmark"


def test_fake_y55_p1_ra_p2_resolves_p2_therapist():
    """The previously-mislabelled case: RA-tagged therapist as p2."""
    streams = [
        _landmark_stream("P1_landmarks", "Y_55"),
        _landmark_stream("P2_landmarks", "RA"),
    ]
    ra = resolve_roles(streams)
    assert ra.p1_role == "patient"
    assert ra.p2_role == "therapist"
    assert ra.p2_name == "RA"
    assert ra.role_source == "xdf_ra_alias"


def test_fake_sync42_patient_id_format_is_recognized():
    streams = [
        _landmark_stream("P1_landmarks", "RA"),
        _landmark_stream("P2_landmarks", "SYNC_42"),
    ]
    ra = resolve_roles(streams)
    assert ra.p1_role == "therapist"
    assert ra.p2_role == "patient"
    assert ra.role_source == "xdf_ra_alias"


def test_fake_both_therapist_raises():
    streams = [
        _landmark_stream("P1_landmarks", "Sway"),
        _landmark_stream("P2_landmarks", "RA"),
    ]
    with pytest.raises(UnresolvedRolesError):
        resolve_roles(streams)


def test_fake_both_unknown_raises():
    streams = [
        _landmark_stream("P1_landmarks", "unknown"),
        _landmark_stream("P2_landmarks", "research_assistant"),
    ]
    with pytest.raises(UnresolvedRolesError):
        resolve_roles(streams)


def test_fake_missing_landmark_stream_raises():
    streams = [
        _landmark_stream("P1_landmarks", "Sway"),
        _other_stream("EEG"),
    ]
    with pytest.raises(UnresolvedRolesError):
        resolve_roles(streams)


# --------------------------------------------------------------------------
# Helpers / classifier sanity
# --------------------------------------------------------------------------

def test_patient_id_pattern_accepts_canonical_examples():
    # Patient ID forms seen in CADENCE (audit found these in actual data):
    canonical_ids = (
        "Y10", "Y_03", "Y24", "y06", "y_06", "SYNC_42", "SYNC_3",
        "Y_16",                  # y_06's actual landmark participant_name
        "Y01_020626",            # y04's date-suffixed form
        "Y26_022728",            # y26's date-suffixed form
        "Y_45", "Y_55", "Y_64",  # newer Y-prefix forms
    )
    for patient_id in canonical_ids:
        assert PATIENT_ID_PATTERN.match(patient_id), f"failed on {patient_id!r}"


def test_patient_id_pattern_rejects_therapist_aliases():
    for not_patient in ("Sway", "sway", "RA", "ra", "unknown", "research_assistant", ""):
        assert not PATIENT_ID_PATTERN.match(not_patient), f"unexpectedly matched {not_patient!r}"


def test_therapist_aliases_classified_as_therapist():
    # Case-insensitive match: actual data contains both 'Sway' and 'sway'.
    for alias in ("Sway", "sway", "SWAY", "RA", "ra", "Ra"):
        assert _classify(alias) == "therapist", f"{alias!r} should classify as therapist"


def test_fake_lowercase_sway_resolves():
    """Older sessions use lowercase 'sway' in participant_name (e.g., y_06, y_17)."""
    streams = [
        _landmark_stream("P1_landmarks", "Y_16"),
        _landmark_stream("P2_landmarks", "sway"),
    ]
    ra = resolve_roles(streams)
    assert ra.p1_role == "patient"
    assert ra.p2_role == "therapist"
    assert ra.role_source == "xdf_landmark"


def test_fake_date_suffixed_patient_resolves():
    """Older naming convention: patient_name = 'Y01_020626' (with date suffix)."""
    streams = [
        _landmark_stream("P1_landmarks", "sway"),
        _landmark_stream("P2_landmarks", "Y01_020626"),
    ]
    ra = resolve_roles(streams)
    assert ra.p1_role == "therapist"
    assert ra.p2_role == "patient"


# --------------------------------------------------------------------------
# Real-XDF tests: covers the six previously-mislabelled sessions plus a
# representative Sway session. Skipped if the XDF isn't on disk.
# --------------------------------------------------------------------------

_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "raw sessions"


def _find_xdf(session_id_substring: str) -> Path | None:
    if not _RAW_DIR.is_dir():
        return None
    for path in sorted(_RAW_DIR.glob("*.xdf")):
        if session_id_substring.lower() in path.stem.lower():
            return path
    return None


@pytest.mark.parametrize("sid", [
    "y_06",          # Sway-tagged (was already resolving)
    "Y_55",          # RA-tagged (was unresolved under old code)
    "y_59",          # RA-tagged
    "y_64",          # RA-tagged
    "y_66",          # RA-tagged
    "y26",           # RA-tagged (was unresolved)
    "y_65",          # RA-tagged
])
def test_real_xdf_role_resolution(sid: str):
    xdf_path = _find_xdf(sid)
    if xdf_path is None:
        pytest.skip(f"XDF not found for session {sid}")

    try:
        import pyxdf
    except ImportError:
        pytest.skip("pyxdf not installed")

    streams, _ = pyxdf.load_xdf(
        str(xdf_path), dejitter_timestamps=False, synchronize_clocks=False
    )
    ra = resolve_roles(streams)
    # Either p1 or p2 is therapist — the assertion is that resolve_roles
    # succeeds (no UnresolvedRolesError) and yields exactly one of each role.
    assert {ra.p1_role, ra.p2_role} == {"therapist", "patient"}, ra
    assert ra.role_source in ("xdf_landmark", "xdf_ra_alias")
