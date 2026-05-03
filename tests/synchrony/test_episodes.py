"""Tests for Stage 2 episode segmentation edge cases."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch as _torch  # noqa: F401
import numpy as np

from cadence.synchrony.episodes import _segment_intervals


FS = 30.0


def test_no_above_threshold_returns_empty():
    env = np.zeros(int(10 * FS))
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert out == []


def test_single_long_block():
    env = np.zeros(int(10 * FS))
    env[60:240] = 1.0  # 6 seconds
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert len(out) == 1
    s, e = out[0]
    assert s == 60 and e == 239


def test_too_short_dropped():
    env = np.zeros(int(5 * FS))
    env[10:13] = 1.0  # 3 frames = 100ms (below 200ms min)
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert out == []


def test_merge_gap_collapses():
    env = np.zeros(int(10 * FS))
    # Two bursts separated by 200ms (= 6 frames at 30fps; below 500ms merge_gap)
    env[60:90] = 1.0
    env[96:120] = 1.0
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert len(out) == 1, f'Expected merged single episode, got {len(out)}'


def test_merge_gap_keeps_separate_when_far():
    env = np.zeros(int(10 * FS))
    env[30:60]   = 1.0  # episode 1
    env[150:180] = 1.0  # episode 2 (3 seconds later)
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert len(out) == 2


def test_max_duration_drops_long():
    env = np.ones(int(120 * FS))  # 2 minutes flat above threshold
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert out == []


def test_max_duration_keeps_borderline():
    env = np.zeros(int(70 * FS))
    env[0:int(59.9 * FS)] = 1.0  # just under 60s
    out = _segment_intervals(env, threshold=0.5, fs=FS,
                              merge_gap_s=0.5, min_dur_s=0.2, max_dur_s=60.0)
    assert len(out) == 1
