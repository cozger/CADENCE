"""Phase 0.3 — numerical equivalence test for Numba Kalman smoother.

Compares _kalman_smoother_weighted_numba (the new soft-mask jitted kernel)
against a reference numpy implementation that mirrors the pre-Phase-0.2b
behavior (variable-shape per-timestep mask extraction).

Tests on three masks:
  1. No mask (all-True)
  2. MVP-y_06 mask (intermittent ~6% missing)
  3. Synthetic worst-case (50% missing channels for whole session)

Pass criterion: max|Δ x_smooth| < 1e-6, max|Δ P_smooth| < 1e-6,
max|Δ Plag_smooth| < 1e-6. Looser than logsumexp's 1e-15 because of the
two structural divergences (soft-mask R-inflation, always-on 1e-10 ridge).
"""
# torch must be imported before numpy on Windows
import torch  # noqa: F401

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the new (numba) version + a copy of the old (numpy) version below.
from cadence.significance.rslds_model import (
    _kalman_smoother_weighted as kalman_numba_wrapper,
    _kalman_update,
)


def _kalman_smoother_weighted_NUMPY_REF(Y, obs_mask, A_t, b_t, Q_t, C_t, d_t,
                                          R_t, x0_mean, P0):
    """Reference numpy implementation — verbatim copy of the pre-Phase-0.2b
    function. Used only by this test script."""
    T, m = Y.shape
    D = A_t.shape[1]

    x_pred = np.zeros((T, D))
    P_pred = np.zeros((T, D, D))
    x_filt = np.zeros((T, D))
    P_filt = np.zeros((T, D, D))

    x_pred[0] = x0_mean
    P_pred[0] = P0.copy()
    mask_0 = obs_mask[0] if obs_mask is not None else None
    x_filt[0], P_filt[0], _ = _kalman_update(x_pred[0], P_pred[0], Y[0],
                                              C_t[0], d_t[0], R_t[0], mask_0)

    for t in range(1, T):
        x_pred[t] = A_t[t] @ x_filt[t - 1] + b_t[t]
        P_pred[t] = A_t[t] @ P_filt[t - 1] @ A_t[t].T + Q_t[t]
        P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T)
        mask_t = obs_mask[t] if obs_mask is not None else None
        x_filt[t], P_filt[t], _ = _kalman_update(x_pred[t], P_pred[t], Y[t],
                                                   C_t[t], d_t[t], R_t[t], mask_t)

    x_smooth = np.zeros((T, D))
    P_smooth = np.zeros((T, D, D))
    Plag_smooth = np.zeros((T - 1, D, D))

    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred_tp1 = P_pred[t + 1]
        eigvals = np.linalg.eigvalsh(P_pred_tp1)
        if eigvals.min() < 1e-10:
            P_pred_tp1 = P_pred_tp1 + (1e-10 - eigvals.min()) * np.eye(D)
        G = P_filt[t] @ A_t[t + 1].T @ np.linalg.inv(P_pred_tp1)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_tp1) @ G.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
        Plag_smooth[t] = P_smooth[t + 1] @ G.T

    return x_smooth, P_smooth, Plag_smooth


def make_synthetic_kalman_inputs(T=500, m=7, D=3, seed=42, mask=None):
    rng = np.random.default_rng(seed)
    A = np.tile(0.95 * np.eye(D), (T, 1, 1))
    b = np.zeros((T, D))
    Q = np.tile(0.1 * np.eye(D), (T, 1, 1))
    C_base = rng.standard_normal((m, D))
    C = np.tile(C_base, (T, 1, 1))
    d = np.zeros((T, m))
    R = np.full((T, m), 0.5)
    x0 = np.zeros(D)
    P0 = np.eye(D)

    # Generate Y
    x_true = np.zeros((T, D))
    x_true[0] = rng.standard_normal(D)
    for t in range(1, T):
        x_true[t] = A[t] @ x_true[t - 1] + np.sqrt(0.1) * rng.standard_normal(D)
    Y = x_true @ C_base.T + np.sqrt(0.5) * rng.standard_normal((T, m))

    if mask is None:
        mask = np.ones((T, m), dtype=bool)
    return Y, mask, A, b, Q, C, d, R, x0, P0


def compare(name, ref_out, jit_out):
    refs = ['x_smooth', 'P_smooth', 'Plag_smooth']
    print(f'\n  {name}')
    all_pass = True
    for n, r, j in zip(refs, ref_out, jit_out):
        d = np.abs(r - j).max()
        rel = d / (np.abs(r).max() + 1e-12)
        ok = d < 1e-6
        flag = 'OK' if ok else 'FAIL'
        print(f'    {n:14s}  max|d|={d:.2e}  rel={rel:.2e}  [{flag}]')
        all_pass = all_pass and ok
    return all_pass


def main():
    OUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'migration' / 'dynamax'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / 'kalman_numba_validation.txt'

    print('=== Phase 0.3 — Numba Kalman numerical equivalence ===')
    log_lines = ['=== Phase 0.3 — Numba Kalman numerical equivalence ===']

    overall_pass = True

    # Test 1: no mask
    Y, mask, A, b, Q, C, d, R, x0, P0 = make_synthetic_kalman_inputs(
        T=500, m=7, D=3, mask=None)
    ref = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R, x0, P0)
    # Numba kernel needs warmup (first call compiles)
    _ = kalman_numba_wrapper(Y[:10], mask[:10], A[:10], b[:10], Q[:10],
                              C[:10], d[:10], R[:10], x0, P0)
    t0 = time.perf_counter()
    jit = kalman_numba_wrapper(Y, mask, A, b, Q, C, d, R, x0, P0)
    t_jit = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R, x0, P0)
    t_ref = time.perf_counter() - t0
    print(f'\n  T=500: numpy_ref {t_ref*1000:.1f}ms, numba {t_jit*1000:.1f}ms '
          f'(speedup {t_ref/t_jit:.1f}x)')
    overall_pass &= compare('Test 1 (no mask)', ref, jit)

    # Test 2: random ~6% missing
    rng = np.random.default_rng(7)
    Y, _, A, b, Q, C, d, R, x0, P0 = make_synthetic_kalman_inputs(T=500, m=7, D=3)
    mask = rng.random((500, 7)) > 0.06
    ref = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R, x0, P0)
    jit = kalman_numba_wrapper(Y, mask, A, b, Q, C, d, R, x0, P0)
    overall_pass &= compare('Test 2 (~6% missing)', ref, jit)

    # Test 3: 2 channels missing for entire session (worst case for soft-mask)
    Y, _, A, b, Q, C, d, R, x0, P0 = make_synthetic_kalman_inputs(T=500, m=7, D=3)
    mask = np.ones((500, 7), dtype=bool)
    mask[:, 5] = False  # whole-session drop ch 5 (resp)
    mask[:, 6] = False  # whole-session drop ch 6 (ecg_hf)
    ref = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R, x0, P0)
    jit = kalman_numba_wrapper(Y, mask, A, b, Q, C, d, R, x0, P0)
    overall_pass &= compare('Test 3 (whole-session drop ch 5,6)', ref, jit)

    # Test 4: real MVP-y_06 mask
    REPO = Path(__file__).resolve().parents[1]
    mvp_path = REPO / 'results' / 'mvp' / 'y_06' / 'mvp_scaffold.npz'
    if mvp_path.exists():
        mvp = np.load(mvp_path)
        T_real = min(500, mvp['obs'].shape[0])
        # construct Kalman inputs with the real mask shape
        m_real = mvp['obs'].shape[1]
        mask_real = mvp['obs_valid'][:T_real].astype(bool)
        Y, _, A, b, Q, C, d, R, x0, P0 = make_synthetic_kalman_inputs(
            T=T_real, m=m_real, D=3, mask=mask_real)
        ref = _kalman_smoother_weighted_NUMPY_REF(Y, mask_real, A, b, Q, C, d, R, x0, P0)
        jit = kalman_numba_wrapper(Y, mask_real, A, b, Q, C, d, R, x0, P0)
        overall_pass &= compare('Test 4 (real MVP y_06 mask)', ref, jit)

    print(f'\n=== Overall: {"PASS" if overall_pass else "FAIL"} ===')
    log_path.write_text('\n'.join(log_lines + [f'overall={"PASS" if overall_pass else "FAIL"}']))
    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
