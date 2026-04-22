import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier1_screening.module3_transition_analysis import (
    compute_dwell_times, transition_matrix_from_logits,
    geometric_mean_dwell, transition_event_ks_test
)


def test_compute_dwell_times_basic():
    path = np.array([0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2])
    dwells = compute_dwell_times(path)
    assert dwells[0] == [3, 2]
    assert dwells[1] == [2]
    assert dwells[2] == [4]


def test_compute_dwell_times_single_state():
    path = np.array([1, 1, 1, 1])
    dwells = compute_dwell_times(path)
    assert dwells[1] == [4]
    assert dwells.get(0, []) == []


def test_transition_matrix_rows_sum_to_one(rng):
    W = rng.standard_normal((4, 4)).astype(np.float32)
    T = transition_matrix_from_logits(W)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(4), atol=1e-6)


def test_geometric_mean_dwell_formula():
    for T_kk in [0.5, 0.8, 0.9, 0.95]:
        expected = 1.0 / (1.0 - T_kk)
        assert abs(geometric_mean_dwell(T_kk) - expected) < 1e-9


def test_coincidence_random_transitions_not_significant(rng):
    session_length_s = 1200.0
    event_times = [300.0, 600.0, 900.0]
    transition_times = rng.uniform(0, session_length_s, 60)
    _, p = transition_event_ks_test(transition_times, event_times,
                                    session_length_s, n_boot=500, seed=42)
    assert p > 0.05


def test_coincidence_event_locked_transitions_significant(rng):
    session_length_s = 1200.0
    event_times = [300.0, 600.0, 900.0]
    transition_times = np.array(
        [t + rng.uniform(-3, 3) for t in event_times for _ in range(15)]
    )
    _, p = transition_event_ks_test(transition_times, event_times,
                                    session_length_s, n_boot=500, seed=42)
    assert p < 0.05
