"""Unit tests for pose-format dispatch + body-only 33-subset gather."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.preprocess.pose.pose_subset import (
    RTMW_TO_MP33,
    derive_33_subset_from_wholebody133,
    normalize_pose_to_mp33,
    reshape_mediapipe33,
)


# --------------------------------------------------------------------------
# Constants and structural invariants
# --------------------------------------------------------------------------

def test_rtmw_to_mp33_shape_and_validity():
    assert RTMW_TO_MP33.shape == (33,)
    # 22 of 33 slots populated (per body-only design)
    populated = (RTMW_TO_MP33 >= 0).sum()
    assert populated == 22, f"expected 22 populated slots, got {populated}"
    # All populated indices are in [0, 22] (body + feet of wholebody_133)
    valid = RTMW_TO_MP33[RTMW_TO_MP33 >= 0]
    assert valid.max() <= 22, f"body-only must not reach face/hand range, got max={valid.max()}"


def test_zeroed_slots_are_hand_and_partial_face():
    # MP slots 1, 3, 4, 6, 9, 10 are face slots without WB analog;
    # MP slots 17–22 are MediaPipe hand markers without compatible WB layout.
    expected_zero = {1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22}
    actual_zero = {int(i) for i, v in enumerate(RTMW_TO_MP33) if v < 0}
    assert actual_zero == expected_zero


# --------------------------------------------------------------------------
# reshape_mediapipe33: trivial (N, 132) -> (N, 33, 4)
# --------------------------------------------------------------------------

def test_reshape_mediapipe33_basic():
    n = 10
    raw = np.arange(n * 132, dtype=np.float32).reshape(n, 132)
    out = reshape_mediapipe33(raw)
    assert out.shape == (n, 33, 4)
    assert out.dtype == np.float32
    # Per-keypoint stride: kp0 = (0, 1, 2, 3), kp1 = (4, 5, 6, 7), ...
    assert tuple(out[0, 0]) == (0, 1, 2, 3)
    assert tuple(out[0, 1]) == (4, 5, 6, 7)


def test_reshape_mediapipe33_wrong_shape_raises():
    with pytest.raises(ValueError, match="expects \\(N, 132\\)"):
        reshape_mediapipe33(np.zeros((10, 99), dtype=np.float32))


# --------------------------------------------------------------------------
# derive_33_subset_from_wholebody133: (N, 400) -> (N, 33, 4)
# --------------------------------------------------------------------------

def test_wholebody133_to_mp33_shape():
    n = 5
    raw = np.zeros((n, 400), dtype=np.float32)
    out = derive_33_subset_from_wholebody133(raw)
    assert out.shape == (n, 33, 4)
    assert out.dtype == np.float32


def test_wholebody133_gather_is_correct():
    """Synthesize a wholebody_133 frame where each keypoint is encoded as
    (1000 + kp_idx + 0.1, 1000 + kp_idx + 0.2, kp_idx / 133).
    Verify the gather places body keypoints in the right MP slots and
    promotes score -> visibility.
    """
    raw = np.zeros((1, 400), dtype=np.float32)
    flat = raw[0, :399].reshape(133, 3)
    for kp in range(133):
        flat[kp, 0] = 1000.0 + kp + 0.1   # x
        flat[kp, 1] = 1000.0 + kp + 0.2   # y
        flat[kp, 2] = kp / 133.0          # score
    raw[0, :399] = flat.flatten()

    out = derive_33_subset_from_wholebody133(raw)

    # Spot-check a few mappings (RTMW_TO_MP33[mp_slot] = wb_idx)
    # MP slot 0 (nose) -> WB 0; MP slot 11 (l_shoulder) -> WB 5;
    # MP slot 23 (l_hip) -> WB 11; MP slot 29 (l_heel) -> WB 19.
    cases = [
        (0, 0),   # nose
        (11, 5),  # left_shoulder
        (23, 11), # left_hip
        (29, 19), # left_heel
        (32, 20), # right_foot_index
    ]
    for mp_slot, wb_idx in cases:
        assert out[0, mp_slot, 0] == pytest.approx(1000.0 + wb_idx + 0.1), \
            f"mp_slot {mp_slot}: x mismatch"
        assert out[0, mp_slot, 1] == pytest.approx(1000.0 + wb_idx + 0.2), \
            f"mp_slot {mp_slot}: y mismatch"
        assert out[0, mp_slot, 2] == 0.0, \
            f"mp_slot {mp_slot}: z must be 0 (RTMW is 2D)"
        assert out[0, mp_slot, 3] == pytest.approx(wb_idx / 133.0), \
            f"mp_slot {mp_slot}: visibility (=score) mismatch"


def test_wholebody133_zeroed_slots_have_visibility_zero():
    raw = np.random.RandomState(0).rand(2, 400).astype(np.float32)
    out = derive_33_subset_from_wholebody133(raw)
    # Slots 1, 3, 4, 6, 9, 10, 17–22 must have all-zero coords + visibility.
    expected_zero = [1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22]
    for mp_slot in expected_zero:
        assert np.all(out[:, mp_slot, :] == 0.0), \
            f"mp_slot {mp_slot} should be zeroed"


def test_wholebody133_drops_provenance_channel():
    """The trailing provenance channel (index 399) must not leak into output."""
    n = 1
    raw = np.zeros((n, 400), dtype=np.float32)
    raw[0, 399] = 9999.0  # extreme provenance value
    out = derive_33_subset_from_wholebody133(raw)
    assert not np.any(out == 9999.0), "provenance channel leaked into output"


def test_wholebody133_wrong_shape_raises():
    with pytest.raises(ValueError, match="expects \\(N, 400\\)"):
        derive_33_subset_from_wholebody133(np.zeros((10, 132), dtype=np.float32))


# --------------------------------------------------------------------------
# normalize_pose_to_mp33: top-level dispatch
# --------------------------------------------------------------------------

def test_normalize_dispatches_mediapipe33():
    raw = np.arange(5 * 132, dtype=np.float32).reshape(5, 132)
    out = normalize_pose_to_mp33(raw, "mediapipe33")
    assert out.shape == (5, 33, 4)


def test_normalize_dispatches_mediapipe33_meta():
    """133-channel sessions: drop the trailing metadata flag, treat as mediapipe33."""
    n = 4
    raw = np.zeros((n, 133), dtype=np.float32)
    # Fill first 132 channels with a known pattern; last col with 0/1 flags
    raw[:, :132] = np.arange(n * 132, dtype=np.float32).reshape(n, 132)
    raw[:, 132] = [0, 1, 0, 1]
    out = normalize_pose_to_mp33(raw, "mediapipe33_meta")
    assert out.shape == (n, 33, 4)
    # The metadata flag value (0/1) must NOT appear in the gathered output.
    # Since it was just 0/1, check the gathered values match the first-132 pattern.
    expected = raw[:, :132].reshape(n, 33, 4)
    np.testing.assert_array_equal(out, expected)


def test_normalize_dispatches_wholebody_133():
    n = 3
    raw = np.zeros((n, 400), dtype=np.float32)
    out = normalize_pose_to_mp33(raw, "wholebody_133")
    assert out.shape == (n, 33, 4)


def test_normalize_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown pose_format"):
        normalize_pose_to_mp33(np.zeros((2, 132)), "unsupported_pose_xyz")
