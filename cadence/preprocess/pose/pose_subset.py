"""Pose-format dispatch + body-only 33-keypoint subset for backward compat.

Three raw pose formats encountered in CADENCE recordings:

* ``mediapipe33``        — ``(N, 132)`` = 33 keypoints × (x, y, z, visibility)
* ``mediapipe33_meta``   — ``(N, 133)`` = mediapipe33 + 1 trailing metadata flag
* ``wholebody_133``      — ``(N, 400)`` = 133 keypoints × (x, y, score) + 1 provenance

This module normalizes all three into a unified ``(N, 33, 4)`` layout matching
MediaPipe Pose Landmarker (``cols = x, y, z, visibility``) so that downstream
feature extractors (``extract_pose_features``, V8.2 multi-lag, V11 scaffold) run
unchanged regardless of source.

The body-only mapping is **22 of 33 slots populated**. MediaPipe's hand-tip slots
(17–22: pinky/index/thumb × L/R) and dense face slots (1, 3, 4, 6, 9, 10) are
zeroed with ``visibility = 0`` since wholebody_133's hand and face keypoints
are not positionally compatible with MediaPipe's.

References
----------
* MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
* COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody
  - 0–16: COCO body 17 keypoints.
  - 17–22: feet (17=l_big_toe, 18=l_small_toe, 19=l_heel,
            20=r_big_toe, 21=r_small_toe, 22=r_heel).
  - 23–90: 68 face landmarks (zeroed in body-only).
  - 91–111: 21 left-hand keypoints (zeroed).
  - 112–132: 21 right-hand keypoints (zeroed).

The wholebody_133 channel labels are confirmed in the XDF stream's
``info.desc.channels`` field (e.g. ``body_nose_x, body_nose_y, body_nose_score,
body_left_eye_x, ...`` — three channels per keypoint, no separate z).
"""

from __future__ import annotations

import numpy as np


# Body-only RTMW (wholebody_133) keypoint index → MediaPipe Pose 33-slot index.
# 22 of 33 slots populated. -1 means the MediaPipe slot has no body-only analog.
RTMW_TO_MP33: np.ndarray = np.array([
    # Head/face (MP slots 0–10). Only nose / eyes / ears populate.
     0,   # 0  nose                  -> WB 0  body_nose
    -1,   # 1  left_eye_inner        -> ZEROED
     1,   # 2  left_eye              -> WB 1  body_left_eye
    -1,   # 3  left_eye_outer        -> ZEROED
    -1,   # 4  right_eye_inner       -> ZEROED
     2,   # 5  right_eye             -> WB 2  body_right_eye
    -1,   # 6  right_eye_outer       -> ZEROED
     3,   # 7  left_ear              -> WB 3  body_left_ear
     4,   # 8  right_ear             -> WB 4  body_right_ear
    -1,   # 9  mouth_left            -> ZEROED
    -1,   # 10 mouth_right           -> ZEROED
    # Upper body (MP 11–16) <-> WB body 5–10.
     5,   # 11 left_shoulder
     6,   # 12 right_shoulder
     7,   # 13 left_elbow
     8,   # 14 right_elbow
     9,   # 15 left_wrist
    10,   # 16 right_wrist
    # MediaPipe hand markers (MP 17–22). All zeroed — RTMW's 21-pt hand models
    # are not positionally compatible with MediaPipe's pinky/index/thumb.
    -1, -1, -1, -1, -1, -1,
    # Lower body (MP 23–32) <-> WB body 11–16 + feet 17–22.
    11,   # 23 left_hip
    12,   # 24 right_hip
    13,   # 25 left_knee
    14,   # 26 right_knee
    15,   # 27 left_ankle
    16,   # 28 right_ankle
    19,   # 29 left_heel             -> WB 19 foot_left_heel
    22,   # 30 right_heel            -> WB 22 foot_right_heel
    17,   # 31 left_foot_index       -> WB 17 foot_left_big_toe
    20,   # 32 right_foot_index      -> WB 20 foot_right_big_toe
], dtype=np.int32)
assert RTMW_TO_MP33.shape == (33,), "RTMW_TO_MP33 must be length 33"


def normalize_pose_to_mp33(pose_raw: np.ndarray, pose_format: str) -> np.ndarray:
    """Dispatch on ``pose_format`` and return ``(N, 33, 4)`` MediaPipe-shaped pose.

    Parameters
    ----------
    pose_raw : ndarray, (N, C)
        Raw pose stream from ``{p}_pose_full`` in the digest.
    pose_format : {'mediapipe33', 'mediapipe33_meta', 'wholebody_133'}
        Format tag from the digest's ``pose_format`` field.

    Returns
    -------
    pose_33 : ndarray, (N, 33, 4) float32
        Columns (x, y, z, visibility). For wholebody_133, ``z`` is set to 0
        (no z component in 2D RTMW) and ``visibility`` is the keypoint score
        promoted from RTMW's per-keypoint score.
    """
    if pose_format == "mediapipe33":
        return reshape_mediapipe33(pose_raw)
    if pose_format == "mediapipe33_meta":
        # Drop the trailing metadata flag; rest is mediapipe33.
        if pose_raw.ndim != 2 or pose_raw.shape[1] != 133:
            raise ValueError(
                f"mediapipe33_meta expects (N, 133); got {pose_raw.shape}"
            )
        return reshape_mediapipe33(pose_raw[:, :132])
    if pose_format == "wholebody_133":
        return derive_33_subset_from_wholebody133(pose_raw)
    raise ValueError(f"Unknown pose_format: {pose_format!r}")


def reshape_mediapipe33(pose_132: np.ndarray) -> np.ndarray:
    """Reshape mediapipe33 stream ``(N, 132)`` into ``(N, 33, 4)``."""
    if pose_132.ndim != 2 or pose_132.shape[1] != 132:
        raise ValueError(
            f"reshape_mediapipe33 expects (N, 132); got {pose_132.shape}"
        )
    return pose_132.reshape(pose_132.shape[0], 33, 4).astype(np.float32, copy=False)


def derive_33_subset_from_wholebody133(pose_400: np.ndarray) -> np.ndarray:
    """Body-only 33-subset from wholebody_133 ``(N, 400)`` -> ``(N, 33, 4)``.

    The wholebody_133 layout is ``133 keypoints × (x, y, score) + 1 provenance``.
    This function:

    1. Drops the trailing ``provenance`` channel (indices [:399]).
    2. Reshapes to ``(N, 133, 3)``.
    3. Gathers the 22 body-relevant RTMW keypoints into MediaPipe slots via
       ``RTMW_TO_MP33``; the other 11 slots are zeroed.
    4. Inserts ``z = 0`` (RTMW is 2D) and promotes ``score`` to the
       ``visibility`` column so the output ``(N, 33, 4)`` matches MediaPipe's
       layout that ``extract_pose_features`` expects.
    """
    if pose_400.ndim != 2 or pose_400.shape[1] != 400:
        raise ValueError(
            f"derive_33_subset_from_wholebody133 expects (N, 400); got {pose_400.shape}"
        )
    n = pose_400.shape[0]
    rtmw = pose_400[:, :399].reshape(n, 133, 3).astype(np.float32, copy=False)

    out = np.zeros((n, 33, 4), dtype=np.float32)
    valid_mp = np.where(RTMW_TO_MP33 >= 0)[0]
    src_idx = RTMW_TO_MP33[valid_mp].astype(np.int64)

    # Channels: x=0, y=1, score=2 in RTMW; x=0, y=1, z=2, vis=3 in MP.
    out[:, valid_mp, 0] = rtmw[:, src_idx, 0]   # x
    out[:, valid_mp, 1] = rtmw[:, src_idx, 1]   # y
    # z stays 0 (RTMW is 2D).
    out[:, valid_mp, 3] = rtmw[:, src_idx, 2]   # score -> visibility
    # Unmapped slots already 0 (vis=0).
    return out
