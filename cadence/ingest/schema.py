"""Canonical digest schema (v1) — dataclasses + version constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cadence.ingest.roles import RoleAssignment


SCHEMA_VERSION = "v1"


# Modalities the digest layer knows how to ingest. New modalities (respiratory,
# audio, gaze) extend this list AND add corresponding entries in MODALITY_KEYS.
SUPPORTED_MODALITIES: tuple[str, ...] = ("eeg", "ecg", "landmarks", "pose")


# Per-modality canonical .npz key names (per participant `p ∈ {p1, p2}`).
MODALITY_KEYS: dict[str, list[str]] = {
    "eeg":       ["{p}_eeg_raw", "{p}_eeg_ts"],
    "ecg":       ["{p}_ecg_raw", "{p}_ecg_ts"],
    "landmarks": ["{p}_landmarks_raw", "{p}_landmarks_ts"],
    "pose":      ["{p}_pose_full", "{p}_pose_ts"],
}


PoseFormat = Literal["mediapipe33", "mediapipe33_meta", "wholebody_133"]
"""Pose stream layouts encountered in CADENCE recordings.

* ``mediapipe33`` — legacy: ``(N, 132)`` = 33 keypoints × (x, y, z, visibility).
* ``mediapipe33_meta`` — newer mediapipe33: ``(N, 133)`` = mediapipe33 + 1
  trailing metadata flag (binary, near-zero most of the time).
* ``wholebody_133`` — RTMW whole-body (e.g. y_53): ``(N, 400)`` =
  133 keypoints × (x, y, score) + 1 ``provenance`` channel.

The ``rtmw133`` name from earlier plan iterations is dropped — the actual
schema declared in the XDF stream's ``desc`` is ``wholebody_133``.
"""


@dataclass(frozen=True)
class QualityFlags:
    """Per-session quality annotations carried through digest -> consumers."""
    excluded_modalities: dict[str, list[str]] = field(default_factory=lambda: {"p1": [], "p2": []})
    missing_markers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CanonicalSession:
    """In-memory representation of a digested session.

    Loaded from ``data/digest/v1/<session_id>.npz`` + ``.json`` via
    ``cadence.ingest.digest.load_digest``. Persisted via
    ``cadence.ingest.digest.digest_xdf``.

    The .npz holds raw arrays only (no value transforms). The .json holds
    metadata. Schema version is recorded so future bumps are detectable.
    """

    session_id: str
    schema_version: str
    xdf_basename: str
    xdf_md5: str
    duration_s: float
    t_start_lsl: float
    protocol: Literal["meditation", "PE", "other"]
    modalities: list[str]
    roles: RoleAssignment
    markers: list[tuple[float, str]]
    marker_sources: list[str]
    pose_format: PoseFormat
    quality_flags: QualityFlags
    stream_inventory: dict[str, dict]  # {key: {src_name, shape, srate_hz}}
    arrays: dict[str, np.ndarray]      # the .npz payload, keyed by MODALITY_KEYS

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays


class UnknownPoseFormatError(ValueError):
    """Raised when pose stream has an unexpected channel count.

    Expected: 132 (mediapipe33: 33 keypoints × 4 fields) or 532 (rtmw133:
    133 keypoints × 4 fields). Other counts indicate stream corruption,
    truncation, or a new pose model that needs explicit support.
    """


def detect_pose_format(n_channels: int) -> PoseFormat:
    """Map raw pose channel count to a supported pose-format tag.

    Channel-count → format mapping (verified against actual XDF data):

    * 132 → ``mediapipe33``       (33 kp × 4 = x, y, z, visibility)
    * 133 → ``mediapipe33_meta``  (mediapipe33 + 1 metadata flag)
    * 400 → ``wholebody_133``     (133 kp × 3 = x, y, score; + 1 provenance)
    """
    if n_channels == 132:
        return "mediapipe33"
    if n_channels == 133:
        return "mediapipe33_meta"
    if n_channels == 400:
        return "wholebody_133"
    raise UnknownPoseFormatError(
        f"Unknown pose channel count: {n_channels}. Expected one of: "
        f"132 (mediapipe33), 133 (mediapipe33_meta), 400 (wholebody_133)."
    )
