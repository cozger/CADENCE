import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.shared.data_loader import load_session, load_all_sessions, discover_session_names


def test_load_session_shapes(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.Y_raw.shape == (120, 5)
    assert data.Y_pw.shape == (120, 5)
    assert data.U.shape == (120, 3)
    assert data.obs_mask.shape == (120, 5)
    assert data.t_common.shape == (120,)


def test_load_session_keys(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.modality_keys == ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    assert len(data.segments) == 3
    assert data.segments[0][0] == 'base_EO'


def test_load_session_rslds_fields(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.gamma is not None and data.gamma.shape == (120, 4)
    assert data.path_constrained is not None and data.path_constrained.shape == (120,)
    assert data.path_unconstrained is not None


def test_unconstrained_path_is_argmax_gamma(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    expected = np.argmax(data.gamma, axis=1)
    np.testing.assert_array_equal(data.path_unconstrained, expected)


def test_load_session_no_rslds_returns_none(fake_scaffold_only_dir):
    results_dir, session_name = fake_scaffold_only_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.gamma is None
    assert data.path_constrained is None
    assert data.path_unconstrained is None


def test_discover_session_names(fake_session_dir):
    results_dir, _ = fake_session_dir
    names = discover_session_names(str(results_dir))
    assert 'fake_sess' in names


def test_load_all_sessions(fake_session_dir):
    results_dir, _ = fake_session_dir
    sessions = load_all_sessions(str(results_dir))
    assert len(sessions) >= 1
    assert sessions[0].name == 'fake_sess'
