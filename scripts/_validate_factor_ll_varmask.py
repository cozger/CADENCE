"""Phase 0.2d — numerical equivalence test for vectorized factor-LL.

Reproduces the original per-T loop locally and compares to the new
vectorized soft-mask path on synthetic inputs that mimic the production
shapes (T~6188, K=4, m=7, D=3, n_factors=2).

Pass criterion: max|Δ log_emit| < 1e-10 (the soft-mask identity-
replacement is mathematically equivalent to per-T extraction up to FP
rounding from the inv operation).
"""
# torch must be imported before numpy on Windows
import torch  # noqa: F401

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def factor_ll_loop_REF(Sigma_noise_k, C_k, x_sm, P_sm, Y, d_k, obs_mask):
    """Reference: original per-T loop (lines 1071-1085 pre-Phase-0.2d)."""
    T = Y.shape[0]
    m = Y.shape[1]
    pred_mean = x_sm @ C_k.T + d_k
    resid = Y - pred_mean
    log_emit_k = np.zeros(T)
    for t in range(T):
        oi = np.where(obs_mask[t])[0]
        if len(oi) == 0:
            continue
        Sig_o = Sigma_noise_k[np.ix_(oi, oi)]
        Sig_inv_o = np.linalg.inv(Sig_o)
        logdet_o = np.linalg.slogdet(Sig_o)[1]
        C_o = C_k[oi]
        r = resid[t, oi]
        CPC_t = C_o @ P_sm[t] @ C_o.T
        log_emit_k[t] = -0.5 * (len(oi) * np.log(2 * np.pi)
                                 + logdet_o
                                 + r @ Sig_inv_o @ r
                                 + np.trace(Sig_inv_o @ CPC_t))
    return log_emit_k


def factor_ll_vectorized(Sigma_noise_k, C_k, x_sm, P_sm, Y, d_k, obs_mask):
    """Vectorized soft-mask via identity-replacement. Mirrors the new
    elif branch at slds_e_step:1071-1099 post-Phase-0.2d."""
    T = Y.shape[0]
    m = Y.shape[1]
    pred_mean = x_sm @ C_k.T + d_k
    resid = Y - pred_mean
    pair_mask = obs_mask[:, :, None] & obs_mask[:, None, :]
    pair_mask_f = pair_mask.astype(np.float64)
    Sigma_eff = Sigma_noise_k[None] * pair_mask_f
    diag_idx = np.arange(m)
    diag_add = (~obs_mask).astype(np.float64)
    Sigma_eff[:, diag_idx, diag_idx] += diag_add
    Sigma_inv_eff = np.linalg.inv(Sigma_eff)
    logdet_eff = np.linalg.slogdet(Sigma_eff)[1]
    n_obs_t = obs_mask.sum(axis=1).astype(np.float64)
    resid_eff = resid * obs_mask
    quad = np.einsum('ti,tij,tj->t', resid_eff, Sigma_inv_eff, resid_eff)
    CPC = np.einsum('di,tij,ej->tde', C_k, P_sm, C_k)
    CPC_eff = CPC * pair_mask_f
    trace = np.einsum('tij,tji->t', Sigma_inv_eff, CPC_eff)
    return -0.5 * (n_obs_t * np.log(2 * np.pi)
                    + logdet_eff + quad + trace)


def make_inputs(T=500, m=7, D=3, n_factors=2, seed=42, mask_density=0.94):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.1, (m, n_factors))
    R = np.full(m, 0.5)
    Sigma_noise_k = F @ F.T + np.diag(R) + 0.1 * np.eye(m)  # PD
    C_k = rng.standard_normal((m, D))
    d_k = rng.standard_normal(m) * 0.1
    x_sm = rng.standard_normal((T, D))
    # P_sm: T symmetric PD matrices
    P_sm = np.empty((T, D, D))
    for t in range(T):
        A = rng.standard_normal((D, D))
        P_sm[t] = A @ A.T + 0.1 * np.eye(D)
    Y = x_sm @ C_k.T + d_k + np.sqrt(R) * rng.standard_normal((T, m))
    obs_mask = rng.random((T, m)) < mask_density
    return Sigma_noise_k, C_k, x_sm, P_sm, Y, d_k, obs_mask


def main():
    print('=== Phase 0.2d — vectorized factor-LL equivalence ===')
    overall = True

    # Test 1: production-size T=6188, K=1 (per-state)
    print('\n  Test 1: T=6188, m=7, D=3 (production size)')
    Sigma, C, x, P, Y, d, mask = make_inputs(T=6188, m=7, D=3, mask_density=0.94)
    t0 = time.perf_counter()
    ref = factor_ll_loop_REF(Sigma, C, x, P, Y, d, mask)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    vec = factor_ll_vectorized(Sigma, C, x, P, Y, d, mask)
    t_vec = time.perf_counter() - t0
    diff = np.abs(ref - vec).max()
    print(f'    loop {t_loop*1000:.0f}ms, vec {t_vec*1000:.0f}ms '
          f'(speedup {t_loop/max(t_vec,1e-9):.1f}x)')
    print(f'    max|d| log_emit = {diff:.2e}    [{"OK" if diff < 1e-10 else "FAIL"}]')
    overall &= diff < 1e-10

    # Test 2: lower mask density (more channels missing)
    print('\n  Test 2: T=2000, mask density 0.7 (heavy missing)')
    Sigma, C, x, P, Y, d, mask = make_inputs(T=2000, m=7, D=3,
                                              mask_density=0.7, seed=11)
    ref = factor_ll_loop_REF(Sigma, C, x, P, Y, d, mask)
    vec = factor_ll_vectorized(Sigma, C, x, P, Y, d, mask)
    diff = np.abs(ref - vec).max()
    print(f'    max|d| log_emit = {diff:.2e}    [{"OK" if diff < 1e-10 else "FAIL"}]')
    overall &= diff < 1e-10

    # Test 3: edge case — some all-masked timesteps
    print('\n  Test 3: T=1000 with all-masked timesteps (edge case)')
    Sigma, C, x, P, Y, d, mask = make_inputs(T=1000, m=7, D=3, seed=99)
    mask[100:105] = False  # 5 consecutive all-masked timesteps
    ref = factor_ll_loop_REF(Sigma, C, x, P, Y, d, mask)
    vec = factor_ll_vectorized(Sigma, C, x, P, Y, d, mask)
    diff = np.abs(ref - vec).max()
    # Verify the all-masked timesteps are 0 in both
    print(f'    ref[100:105] = {ref[100:105]}')
    print(f'    vec[100:105] = {vec[100:105]}')
    print(f'    max|d| log_emit = {diff:.2e}    [{"OK" if diff < 1e-10 else "FAIL"}]')
    overall &= diff < 1e-10

    # Test 4: real MVP y_06 mask shape with synthetic factor structure
    REPO = Path(__file__).resolve().parents[1]
    mvp_path = REPO / 'results' / 'mvp' / 'y_06' / 'mvp_scaffold.npz'
    if mvp_path.exists():
        mvp = np.load(mvp_path)
        T_real = mvp['obs'].shape[0]
        m_real = mvp['obs'].shape[1]
        mask_real = mvp['obs_valid'].astype(bool)
        Sigma, C, x, P, Y, d, _ = make_inputs(T=T_real, m=m_real, D=3, seed=7)
        ref = factor_ll_loop_REF(Sigma, C, x, P, Y, d, mask_real)
        vec = factor_ll_vectorized(Sigma, C, x, P, Y, d, mask_real)
        diff = np.abs(ref - vec).max()
        print(f'\n  Test 4: real MVP y_06 mask (T={T_real}, mask coverage '
              f'{mask_real.mean():.3f})')
        print(f'    max|d| log_emit = {diff:.2e}    [{"OK" if diff < 1e-10 else "FAIL"}]')
        overall &= diff < 1e-10

    print(f'\n=== Overall: {"PASS" if overall else "FAIL"} ===')
    sys.exit(0 if overall else 1)


if __name__ == '__main__':
    main()
