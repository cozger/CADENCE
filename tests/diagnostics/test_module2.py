import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier1_screening.module2_feature_diagnostics import (
    compute_acf, compute_n_eff_geyer, compute_vif, is_slow_drift
)


def test_acf_white_noise_near_zero(rng):
    x = rng.standard_normal(2000)
    acf = compute_acf(x, max_lag=40)
    assert acf[0] == pytest.approx(1.0)
    assert np.abs(acf[10:]).max() < 0.15


def test_acf_ar1_decays_geometrically(rng):
    rho = 0.8
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = rho * x[t-1] + rng.standard_normal()
    acf = compute_acf(x, max_lag=10)
    for lag in range(1, 8):
        assert abs(acf[lag] - rho**lag) < 0.05


def test_n_eff_white_noise_near_n(rng):
    x = rng.standard_normal(1000)
    neff = compute_n_eff_geyer(x)
    assert 700 < neff <= 1000


def test_n_eff_ar1_rho09_much_less_than_n(rng):
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.9 * x[t-1] + rng.standard_normal()
    neff = compute_n_eff_geyer(x)
    assert neff < 200


def test_n_eff_never_negative(rng):
    # Over-whitened signal (negative ACF) must not give negative N_eff
    x = rng.standard_normal(500)
    x = np.diff(x)  # first-differencing creates negative lag-1 ACF
    assert compute_n_eff_geyer(x) > 0


def test_vif_independent_channels(rng):
    X = rng.standard_normal((500, 4))
    vifs = compute_vif(X)
    assert len(vifs) == 4
    assert all(v < 3.0 for v in vifs)


def test_vif_collinear_pair(rng):
    x = rng.standard_normal(500)
    X = np.column_stack([x, x + 0.01 * rng.standard_normal(500), rng.standard_normal(500)])
    vifs = compute_vif(X)
    assert vifs[0] > 10 and vifs[1] > 10
    assert vifs[2] < 3.0


def test_is_slow_drift_ar1_rho097(rng):
    x = np.zeros(1200)
    for t in range(1, 1200):
        x[t] = 0.97 * x[t-1] + 0.01 * rng.standard_normal()
    assert is_slow_drift(x, fs=2.0) is True


def test_is_slow_drift_white_noise(rng):
    assert is_slow_drift(rng.standard_normal(1200), fs=2.0) is False


import os, pandas as pd
from diagnostics.tier1_screening.module2_feature_diagnostics import run_feature_diagnostics
from diagnostics.shared.data_loader import load_session


def test_run_feature_diagnostics_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'module2_out')
    flags = run_feature_diagnostics(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'acf_all_channels.png'))
    assert os.path.exists(os.path.join(out_dir, 'correlation_heatmap.png'))
    assert os.path.exists(os.path.join(out_dir, 'vif_table.csv'))
    assert os.path.exists(os.path.join(out_dir, 'n_eff_per_session.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module2_report.md'))
    assert isinstance(flags, dict)
    assert 'slow_drift' in flags
    assert 'collinear_pairs' in flags
    assert 'high_vif' in flags
