"""Validate Numba _log_transitions_recurrent matches numpy version."""
import torch  # noqa: F401  (Windows torch+numpy DLL ordering)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cadence.significance.rslds_model import (
    _log_transitions_recurrent_ref,    # NEW: numpy reference (renamed)
    _log_transitions_recurrent_numba,  # NEW: numba kernel
)


def test_basic():
    rng = np.random.default_rng(0)
    T, K, D_in, D_lat = 2000, 4, 2, 3
    U = rng.standard_normal((T, D_in))
    x = rng.standard_normal((T, D_lat))
    W = rng.standard_normal((K, K))
    S = rng.standard_normal((K, K, D_in)) * 0.1
    R = rng.standard_normal((K, K, D_lat)) * 0.1
    x0 = rng.standard_normal(D_lat)
    ref = _log_transitions_recurrent_ref(U, x, W, S, R, x0)
    out = _log_transitions_recurrent_numba(U, x, W, S, R, x0)
    err = np.max(np.abs(out - ref))
    print(f'basic: max|delta|={err:.2e}')
    assert err < 1e-12, f'max|delta| = {err:.2e} exceeds 1e-12'


def test_edge_T_small():
    rng = np.random.default_rng(1)
    T, K, D_in, D_lat = 5, 3, 2, 2
    U = rng.standard_normal((T, D_in))
    x = rng.standard_normal((T, D_lat))
    W = rng.standard_normal((K, K))
    S = rng.standard_normal((K, K, D_in)) * 0.1
    R = rng.standard_normal((K, K, D_lat)) * 0.1
    x0 = rng.standard_normal(D_lat)
    ref = _log_transitions_recurrent_ref(U, x, W, S, R, x0)
    out = _log_transitions_recurrent_numba(U, x, W, S, R, x0)
    err = np.max(np.abs(out - ref))
    print(f'small-T: max|delta|={err:.2e}')
    assert err < 1e-12


if __name__ == '__main__':
    test_basic()
    test_edge_T_small()
    print('all PASS')
