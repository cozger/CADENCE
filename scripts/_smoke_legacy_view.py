"""Smoke-test the legacy dict view shim.

Verifies that ``cadence.data.load_session_from_cache(sid)`` returns a dict
covering all the legacy keys that V11 / V10 / V8.2 scaffolds read.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", DeprecationWarning)
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch  # noqa: F401  # hoist for Win torch 2.10+numpy 2.4

from cadence.data import (
    EXCLUDED_MODALITIES,
    discover_cached_sessions,
    load_session_from_cache,
)

# Top-traffic legacy keys observed in V11/V10/V8.2 scaffold scripts.
REQUIRED_KEYS = [
    # Per-participant raw + features
    "p1_eeg", "p2_eeg", "p1_eeg_ts", "p2_eeg_ts",
    "p1_eeg_valid", "p2_eeg_valid",
    "p1_eeg_features", "p2_eeg_features",
    "p1_eeg_features_ts", "p2_eeg_features_ts",
    "p1_eeg_features_valid", "p2_eeg_features_valid",
    "p1_blendshapes", "p2_blendshapes",
    "p1_blendshapes_ts", "p2_blendshapes_ts",
    "p1_blendshapes_valid", "p2_blendshapes_valid",
    "p1_blendshapes_v2", "p2_blendshapes_v2",
    "p1_pose", "p2_pose",
    "p1_pose_ts", "p2_pose_ts",
    "p1_pose_features", "p2_pose_features",
    "p1_pose_features_ts", "p2_pose_features_ts",
    "p1_pose_features_valid", "p2_pose_features_valid",
    "p1_ecg", "p2_ecg",
    "p1_ecg_ts", "p2_ecg_ts",
    "p1_ecg_valid", "p2_ecg_valid",
    "p1_ecg_features", "p2_ecg_features",
    "p1_ecg_features_ts", "p2_ecg_features_ts",
    "p1_ecg_features_valid", "p2_ecg_features_valid",
    # Roles
    "p1_role", "p2_role", "p1_name", "p2_name", "role_source",
    # Session-level
    "session_id", "duration", "t_start_absolute",
    "markers", "marker_sources", "protocol", "pose_format",
    # Raw streams (passthrough)
    "p1_eeg_raw", "p2_eeg_raw",
    "p1_landmarks_raw", "p2_landmarks_raw",
    "p1_pose_full", "p2_pose_full",
]


def main() -> int:
    sessions = discover_cached_sessions()
    print(f"Discovered {len(sessions)} sessions")
    print(f"EXCLUDED_MODALITIES: {len(EXCLUDED_MODALITIES)} entries")

    test_sids = sys.argv[1:] or ["y_06", "Y_55_04272026"]

    failures = 0
    for sid in test_sids:
        try:
            sess = load_session_from_cache(sid)
        except Exception as e:
            print(f"  [{sid}] LOAD FAILED: {e}")
            failures += 1
            continue
        print(f"\n  [{sid}] loaded {len(sess)} keys")
        missing = [k for k in REQUIRED_KEYS if k not in sess]
        if missing:
            print(f"    MISSING ({len(missing)}): {missing}")
            failures += 1
        else:
            print(f"    All {len(REQUIRED_KEYS)} required keys present")
        # Sample shapes
        for k in ("p1_eeg", "p1_blendshapes", "p1_pose", "p1_pose_features",
                  "p1_ecg_features"):
            v = sess.get(k)
            if v is None:
                continue
            shape = getattr(v, "shape", None)
            print(f"    {k}: shape={shape}")
        print(f"    p1_role={sess['p1_role']!r} ({sess['p1_name']!r}), "
              f"p2_role={sess['p2_role']!r} ({sess['p2_name']!r}), "
              f"src={sess['role_source']!r}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
