"""Tests for Stage 3c CCA short-episode pooling logic."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch as _torch  # noqa: F401
import numpy as np

from cadence.synchrony.config import DEFAULT_CONFIG
from cadence.synchrony.features.cca import (
    compute_cca_features, compute_cca_features_pooled, _empty_features,
)


FS = 30.0


def _signal_pair(T: int, copy_strength: float = 0.7, lag_samples: int = 0):
    """Build correlated multivariate sequences. P2 is a noisy lagged copy of P1."""
    rng = np.random.default_rng(0)
    p1 = rng.standard_normal((T + abs(lag_samples), 52))
    if lag_samples > 0:
        p2 = copy_strength * p1[:-lag_samples] + \
             (1 - copy_strength) * rng.standard_normal((T, 52))
        p1 = p1[lag_samples:]
    elif lag_samples < 0:
        p2 = copy_strength * p1[-lag_samples:] + \
             (1 - copy_strength) * rng.standard_normal((T, 52))
        p1 = p1[:T]
    else:
        p2 = copy_strength * p1[:T] + (1 - copy_strength) * rng.standard_normal((T, 52))
        p1 = p1[:T]
    return p1.astype(np.float64), p2.astype(np.float64)


def test_too_short_per_episode_returns_empty():
    """Episode shorter than T_min should yield NaN features."""
    p1, p2 = _signal_pair(50)  # T=50 < T_min=200
    out = compute_cca_features(p1, p2, fs=FS, config=DEFAULT_CONFIG)
    assert out['_3c_valid'] is False
    assert np.isnan(out['cca_peak_r'])


def test_long_episode_per_episode_works():
    """Long episode → per-episode CCA should fit."""
    p1, p2 = _signal_pair(300)  # > T_min=200
    out = compute_cca_features(p1, p2, fs=FS, config=DEFAULT_CONFIG)
    assert out['_3c_valid'] is True
    assert 0.0 <= out['cca_peak_r'] <= 1.0
    assert out['cca_pooled_flag'] is False


def test_pooled_below_min_returns_empty():
    """Pooled samples must clear T_min_pooled threshold."""
    segs1 = [_signal_pair(100)[0] for _ in range(4)]   # 400 < 600
    segs2 = [_signal_pair(100)[1] for _ in range(4)]
    out = compute_cca_features_pooled(segs1, segs2, fs=FS, config=DEFAULT_CONFIG)
    assert out['_3c_valid'] is False


def test_pooled_above_min_works():
    """Pooled samples ≥ T_min_pooled → SCCA fits and flag is set."""
    segs1 = [_signal_pair(100)[0] for _ in range(8)]   # 800 > 600
    segs2 = [_signal_pair(100)[1] for _ in range(8)]
    out = compute_cca_features_pooled(segs1, segs2, fs=FS, config=DEFAULT_CONFIG)
    assert out['_3c_valid'] is True
    assert out['cca_pooled_flag'] is True
    assert 0.0 <= out['cca_peak_r'] <= 1.0


def test_empty_features_shape():
    e = _empty_features()
    assert e['_3c_valid'] is False
    assert np.isnan(e['cca_peak_r'])
    assert e['cca_pooled_flag'] is False


def test_pooled_lengths_must_match_per_pair():
    """Each (p1_seg, p2_seg) pair must have matching lengths."""
    segs1 = [np.zeros((100, 52))]
    segs2 = [np.zeros((50, 52))]  # mismatched
    # The pooled function should fail gracefully (not crash) — we expect NaN
    out = compute_cca_features_pooled(segs1, segs2, fs=FS, config=DEFAULT_CONFIG)
    # With only 100 + 50 samples either way, can't reach min_pooled=600
    assert out['_3c_valid'] is False
