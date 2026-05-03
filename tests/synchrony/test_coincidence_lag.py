"""Tests for Stage 3a NN-lag matching + sign convention.

Sign convention: positive lag = patient FOLLOWS therapist
(i.e., t_patient_event - t_therapist_event > 0 → patient is later).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch as _torch  # noqa: F401
import numpy as np

from cadence.synchrony.features.coincidence import (
    _nn_lag, compute_coincidence_features,
)


def test_nn_lag_empty_inputs():
    idx, lag = _nn_lag(np.array([1.0, 2.0]), np.array([]), max_lag=1.0)
    assert (idx == -1).all()
    assert np.isnan(lag).all()


def test_nn_lag_picks_nearest():
    t_a = np.array([1.0, 2.0, 3.0])
    t_b = np.array([1.05, 2.10, 5.00])
    idx, lag = _nn_lag(t_a, t_b, max_lag=1.0)
    # t_a[0]=1.0 → t_b[0]=1.05 (lag 0.05)
    # t_a[1]=2.0 → t_b[1]=2.10 (lag 0.10)
    # t_a[2]=3.0 → t_b[1]=2.10 is the only candidate within ±1.0 (lag -0.9);
    #              t_b[2]=5.00 is too far.
    np.testing.assert_array_equal(idx, [0, 1, 1])
    np.testing.assert_allclose(lag, [0.05, 0.10, -0.90])


def test_nn_lag_max_lag_filter():
    t_a = np.array([1.0])
    t_b = np.array([3.0])
    idx, lag = _nn_lag(t_a, t_b, max_lag=1.0)
    assert idx[0] == -1
    assert np.isnan(lag[0])


def test_nn_lag_signed_directionality():
    """Positive lag means the b event came AFTER the a event."""
    t_a = np.array([1.0])
    t_b = np.array([1.5])
    _, lag = _nn_lag(t_a, t_b, max_lag=2.0)
    assert lag[0] == 0.5

    _, lag = _nn_lag(t_b, t_a, max_lag=2.0)
    assert lag[0] == -0.5


def test_compute_features_no_partner():
    """Episode with no events from one role → defaults + _3a_valid=False."""
    ev = {
        't_lsl':  np.array([1.0, 1.5, 2.0]),
        'au_idx': np.array([7, 8, 7]),
        'role':   np.array(['therapist'] * 3),
    }
    feats = compute_coincidence_features(ev)
    assert feats['_3a_valid'] is False
    assert feats['n_events_therapist'] == 3
    assert feats['n_events_patient'] == 0
    assert feats['frac_aus_partnered'] == 0.0


def test_compute_features_lead_follow_sign():
    """Therapist firing first, patient firing 0.4s later → positive lag.

    Use 0.4s offset (not 0.5s) so patient events are unambiguously closer
    to the FOLLOWING therapist event than to the previous one — avoids
    the tie-breaking ambiguity that uniform 0.5s spacing creates.
    """
    therapist_t = np.array([1.0, 2.0, 3.0])
    patient_t   = therapist_t + 0.4   # patient consistently 400ms behind
    ev = {
        't_lsl':  np.concatenate([therapist_t, patient_t]),
        'au_idx': np.full(6, 7, dtype=np.int16),
        'role':   np.array(['therapist'] * 3 + ['patient'] * 3),
    }
    feats = compute_coincidence_features(ev)
    assert feats['_3a_valid'] is True
    assert feats['n_events_therapist'] == 3
    assert feats['n_events_patient'] == 3
    assert feats['dominant_lag'] > 0, \
        f'Patient followed therapist by 0.4s, expected lag > 0; got {feats["dominant_lag"]}'
    assert feats['lead_follow'] == 1
