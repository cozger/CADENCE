"""Validate vectorized K-loop emissions LL matches per-K-loop output.

Synthetic SLDS-shaped inputs at MVP and V11 sizes, mask + no-mask paths,
factor-analyzed + diagonal paths.
"""
import torch  # noqa: F401
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cadence.significance.rslds_model import (
    _emit_ll_per_k_loop_ref,   # NEW: extracted reference
    _emit_ll_vectorized,        # NEW: batched
)


def make_inputs(T=2000, K=4, D=3, m=6, seed=0, factor=False, mask_frac=0.0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, m))
    x_sm = rng.standard_normal((T, D))
    P_sm = np.tile(0.5 * np.eye(D), (T, 1, 1))
    C = rng.standard_normal((K, m, D)) * 0.3
    d = rng.standard_normal((K, m)) * 0.2
    if factor:
        F = rng.standard_normal((K, m, 2)) * 0.1
    else:
        F = None
    R = np.full((K, m), 0.5)
    if mask_frac > 0:
        mask = rng.random((T, m)) > mask_frac
    else:
        mask = None
    return Y, x_sm, P_sm, C, d, F, R, mask


def test_diagonal_no_mask():
    args = make_inputs(factor=False, mask_frac=0.0)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'diag no-mask: max|delta|={err:.2e}')
    assert err < 1e-10, f'FAIL: {err}'


def test_factor_const_mask():
    Y, x, P, C, d, _, R, _ = make_inputs(factor=True, mask_frac=0.0)
    m = Y.shape[1]
    K = C.shape[0]
    rng = np.random.default_rng(42)
    F = rng.standard_normal((K, m, 2)) * 0.1
    mask = np.ones((Y.shape[0], m), dtype=bool)
    mask[:, 4:] = False  # constant mask (last 2 obs channels missing)
    args = (Y, x, P, C, d, F, R, mask)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'factor const-mask: max|delta|={err:.2e}')
    assert err < 1e-9, f'FAIL: {err}'


def test_factor_var_mask():
    Y, x, P, C, d, _, R, _ = make_inputs(factor=True, mask_frac=0.05)
    m = Y.shape[1]
    K = C.shape[0]
    rng = np.random.default_rng(7)
    F = rng.standard_normal((K, m, 2)) * 0.1
    mask = rng.random((Y.shape[0], m)) > 0.05
    args = (Y, x, P, C, d, F, R, mask)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'factor var-mask: max|delta|={err:.2e}')
    assert err < 1e-9, f'FAIL: {err}'


def test_factor_no_mask():
    """Additional test: factor-analyzed, no mask (else branch)."""
    args = make_inputs(factor=True, mask_frac=0.0, m=6)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'factor no-mask: max|delta|={err:.2e}')
    assert err < 1e-9, f'FAIL: {err}'


def test_factor_all_obs_const_mask():
    """Constant mask where all obs channels are observed (mask_const=True, m_obs_const == m)."""
    args = make_inputs(factor=True, mask_frac=0.0, m=6)
    Y, x, P, C, d, F, R, _ = args
    # All-true mask — should hit the else branch (no subselection)
    mask = np.ones((Y.shape[0], Y.shape[1]), dtype=bool)
    args2 = (Y, x, P, C, d, F, R, mask)
    ref = _emit_ll_per_k_loop_ref(*args2)
    out = _emit_ll_vectorized(*args2)
    err = np.max(np.abs(out - ref))
    print(f'factor all-obs const-mask: max|delta|={err:.2e}')
    assert err < 1e-9, f'FAIL: {err}'


if __name__ == '__main__':
    test_diagonal_no_mask()
    test_factor_no_mask()
    test_factor_all_obs_const_mask()
    test_factor_const_mask()
    test_factor_var_mask()
    print('all PASS')
