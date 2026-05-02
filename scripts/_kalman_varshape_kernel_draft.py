"""DRAFT: variable-shape Numba Kalman kernel.

Avoids the BIG_R=1e10 soft-mask inflation that may be causing the local-
optimum drift in Phase 0 hierarchical fit. Uses pre-allocated max-size
buffers + per-t obs_idx + counter for variable shape support.

This is a *staged* draft to validate the equivalence on synthetic data
before swapping into rslds_model.py. Once validated, copy the
@numba.njit function into rslds_model.py to replace
_kalman_smoother_weighted_numba.
"""
# torch must be imported before numpy on Windows
import torch  # noqa: F401

import sys
import time
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@numba.njit(cache=True, fastmath=False)
def _kalman_smoother_weighted_varshape(Y, obs_mask, A_t, b_t, Q_t, C_t, d_t,
                                          R_full, x0_mean, P0):
    """Numba Kalman smoother, variable-shape mask handling.

    Mirrors the original numpy _kalman_update / _kalman_smoother_weighted
    semantics exactly: at each timestep, extract the unmasked sub-block
    (n_obs out of m channels) and do the Kalman update on that sub-block.

    R_full has shape (T, m, m); for diagonal R it's been pre-expanded by
    the wrapper (zero off-diagonals).
    """
    T = Y.shape[0]
    m = Y.shape[1]
    D = A_t.shape[1]
    eye_D = np.eye(D)

    x_pred = np.zeros((T, D))
    P_pred = np.zeros((T, D, D))
    x_filt = np.zeros((T, D))
    P_filt = np.zeros((T, D, D))

    # Pre-allocate max-size buffers (n_obs <= m always)
    obs_idx = np.empty(m, dtype=np.int64)
    y_o = np.empty(m)
    d_o = np.empty(m)
    C_o = np.empty((m, D))
    R_o = np.empty((m, m))
    eye_max = np.eye(m)

    for t in range(T):
        if t == 0:
            x_pred[0] = x0_mean
            P_pred[0] = P0
        else:
            x_pred[t] = A_t[t] @ x_filt[t - 1] + b_t[t]
            P_pred[t] = A_t[t] @ P_filt[t - 1] @ A_t[t].T + Q_t[t]
            P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T)

        # Collect unmasked indices
        n_obs = 0
        for d in range(m):
            if obs_mask[t, d]:
                obs_idx[n_obs] = d
                n_obs += 1

        if n_obs == 0:
            # No update — copy pred to filt
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]
            continue

        # Extract sub-arrays (variable shape)
        for i in range(n_obs):
            oi = obs_idx[i]
            y_o[i] = Y[t, oi]
            d_o[i] = d_t[t, oi]
            for j in range(D):
                C_o[i, j] = C_t[t, oi, j]
            for j in range(n_obs):
                R_o[i, j] = R_full[t, oi, obs_idx[j]]

        # Slice views (Numba handles this fine)
        y_v = y_o[:n_obs]
        d_v = d_o[:n_obs]
        C_v = C_o[:n_obs]
        R_v = R_o[:n_obs, :n_obs]
        I_v = eye_max[:n_obs, :n_obs]

        # Standard Kalman update on the unmasked sub-block
        innov = y_v - C_v @ x_pred[t] - d_v
        S = C_v @ P_pred[t] @ C_v.T + R_v
        K_gain = P_pred[t] @ C_v.T @ np.linalg.solve(S, I_v)

        x_filt[t] = x_pred[t] + K_gain @ innov
        I_KC = eye_D - K_gain @ C_v
        P_filt[t] = I_KC @ P_pred[t] @ I_KC.T + K_gain @ R_v @ K_gain.T
        P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T)

    # ── RTS backward smoother (unchanged) ──
    x_smooth = np.zeros((T, D))
    P_smooth = np.zeros((T, D, D))
    Plag_smooth = np.zeros((T - 1, D, D))

    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred_tp1 = P_pred[t + 1].copy()
        eigvals = np.linalg.eigvalsh(P_pred_tp1)
        ev_min = eigvals[0]
        for i in range(1, D):
            if eigvals[i] < ev_min:
                ev_min = eigvals[i]
        if ev_min < 1e-10:
            shift = 1e-10 - ev_min
            for i in range(D):
                P_pred_tp1[i, i] += shift
        G = P_filt[t] @ A_t[t + 1].T @ np.linalg.solve(P_pred_tp1, eye_D)
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_tp1) @ G.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
        Plag_smooth[t] = P_smooth[t + 1] @ G.T

    return x_smooth, P_smooth, Plag_smooth


def _kalman_smoother_weighted_NUMPY_REF(Y, obs_mask, A_t, b_t, Q_t, C_t, d_t,
                                          R_t, x0_mean, P0):
    """Original numpy Kalman (verbatim from pre-Phase-0.2b)."""
    from cadence.significance.rslds_model import _kalman_update
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


def make_inputs(T=500, m=7, D=3, seed=42, mask_density=0.94):
    rng = np.random.default_rng(seed)
    A = np.tile(0.95 * np.eye(D), (T, 1, 1))
    b = np.zeros((T, D))
    Q = np.tile(0.1 * np.eye(D), (T, 1, 1))
    C_base = rng.standard_normal((m, D))
    C = np.tile(C_base, (T, 1, 1))
    d = np.zeros((T, m))
    R = np.full((T, m), 0.5)
    R_full = np.zeros((T, m, m))
    idx = np.arange(m)
    R_full[:, idx, idx] = R
    x0 = np.zeros(D)
    P0 = np.eye(D)
    x_true = np.zeros((T, D))
    x_true[0] = rng.standard_normal(D)
    for t in range(1, T):
        x_true[t] = A[t] @ x_true[t - 1] + np.sqrt(0.1) * rng.standard_normal(D)
    Y = x_true @ C_base.T + np.sqrt(0.5) * rng.standard_normal((T, m))
    mask = rng.random((T, m)) < mask_density
    return Y, mask, A, b, Q, C, d, R, R_full, x0, P0


def main():
    print('=== Variable-shape Kalman validation ===')
    overall = True

    # Test 1: random sparse mask
    Y, mask, A, b, Q, C, d, R, R_full, x0, P0 = make_inputs(T=500, m=7, D=3,
                                                              mask_density=0.94)
    # Warmup numba
    _ = _kalman_smoother_weighted_varshape(Y[:10], mask[:10], A[:10], b[:10],
                                              Q[:10], C[:10], d[:10],
                                              R_full[:10], x0, P0)
    t0 = time.perf_counter()
    jit_out = _kalman_smoother_weighted_varshape(Y, mask, A, b, Q, C, d,
                                                    R_full, x0, P0)
    t_jit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref_out = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R,
                                                    x0, P0)
    t_ref = time.perf_counter() - t0
    print(f'  T=500 ~6% missing: ref {t_ref*1000:.0f}ms, varshape {t_jit*1000:.0f}ms '
          f'(speedup {t_ref/max(t_jit, 1e-9):.1f}x)')
    diffs = [np.abs(r - j).max() for r, j in zip(ref_out, jit_out)]
    print(f'  max|d|: x_smooth={diffs[0]:.2e}, P_smooth={diffs[1]:.2e}, Plag={diffs[2]:.2e}')
    if max(diffs) > 1e-8:
        print('  FAIL — NOT bit-equivalent to numpy')
        overall = False
    else:
        print('  PASS — equivalent to numpy at <1e-8')

    # Test 2: heavy missing
    Y, mask, A, b, Q, C, d, R, R_full, x0, P0 = make_inputs(T=2000, m=7, D=3,
                                                              mask_density=0.7,
                                                              seed=7)
    jit_out = _kalman_smoother_weighted_varshape(Y, mask, A, b, Q, C, d,
                                                    R_full, x0, P0)
    ref_out = _kalman_smoother_weighted_NUMPY_REF(Y, mask, A, b, Q, C, d, R,
                                                    x0, P0)
    diffs = [np.abs(r - j).max() for r, j in zip(ref_out, jit_out)]
    print(f'  T=2000 ~30% missing: max|d| x_smooth={diffs[0]:.2e}')
    overall &= max(diffs) < 1e-8

    # Test 3: real MVP y_06 mask
    REPO = Path(__file__).resolve().parents[1]
    mvp_path = REPO / 'results' / 'mvp' / 'y_06' / 'mvp_scaffold.npz'
    mvp = np.load(mvp_path)
    T_real = mvp['obs'].shape[0]
    mask_real = mvp['obs_valid'].astype(bool)
    Y, _, A, b, Q, C, d, R, R_full, x0, P0 = make_inputs(T=T_real, m=7, D=3,
                                                           seed=11)
    jit_out = _kalman_smoother_weighted_varshape(Y, mask_real, A, b, Q, C, d,
                                                    R_full, x0, P0)
    ref_out = _kalman_smoother_weighted_NUMPY_REF(Y, mask_real, A, b, Q, C, d,
                                                    R, x0, P0)
    diffs = [np.abs(r - j).max() for r, j in zip(ref_out, jit_out)]
    print(f'  T={T_real} (real MVP y_06): max|d| x_smooth={diffs[0]:.2e}, '
          f'P_smooth={diffs[1]:.2e}, Plag={diffs[2]:.2e}')
    overall &= max(diffs) < 1e-8

    print(f'\n=== Overall: {"PASS" if overall else "FAIL"} ===')
    sys.exit(0 if overall else 1)


if __name__ == '__main__':
    main()
