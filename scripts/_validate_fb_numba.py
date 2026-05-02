"""Phase 0.2c — numerical equivalence test for Numba forward-backward.

Compares _forward_backward_numba against a reference numpy implementation
that mirrors the pre-Phase-0.2c IOHMM._forward_backward method.

Pass criterion: max|Δ gamma| < 1e-10, max|Δ xi| < 1e-10, |Δ log_lik|/|log_lik| < 1e-12.
The forward-backward kernel is purely additive log-domain reductions; we
expect agreement to near machine epsilon.
"""
# torch must be imported before numpy on Windows
import torch  # noqa: F401

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp as scipy_lse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.significance.rslds_model import _forward_backward_numba


def _fb_numpy_REF(log_emit, log_trans, log_pi):
    """Reference numpy F-B (verbatim copy of pre-0.2c implementation, using
    scipy.special.logsumexp for the most stringent comparison)."""
    T, K = log_emit.shape

    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_pi + log_emit[0]
    log_scale = np.empty(T, dtype=np.float64)
    log_scale[0] = scipy_lse(log_alpha[0])
    log_alpha[0] -= log_scale[0]

    for t in range(1, T):
        msg = log_alpha[t - 1, :, None] + log_trans[t]
        log_alpha[t] = scipy_lse(msg, axis=0) + log_emit[t]
        log_scale[t] = scipy_lse(log_alpha[t])
        log_alpha[t] -= log_scale[t]

    log_lik = float(np.sum(log_scale))

    log_beta = np.zeros((T, K), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        msg = log_trans[t + 1] + log_emit[t + 1] + log_beta[t + 1]
        log_beta[t] = scipy_lse(msg, axis=1)

    log_gamma = log_alpha + log_beta
    log_gamma -= scipy_lse(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    log_xi = (log_alpha[:-1, :, None]
              + log_trans[1:]
              + log_emit[1:, None, :]
              + log_beta[1:, None, :])
    log_xi -= scipy_lse(log_xi.reshape(T - 1, -1), axis=1, keepdims=True).reshape(T - 1, 1, 1)
    xi = np.exp(log_xi)

    return gamma, xi, log_lik


def make_inputs(T=500, K=4, seed=42, with_minus_inf_row=False):
    rng = np.random.default_rng(seed)
    log_emit = -rng.exponential(scale=2.0, size=(T, K))
    # log_trans rows must sum to 1 in linear space
    raw = rng.standard_normal((T, K, K))
    log_trans = raw - scipy_lse(raw, axis=2, keepdims=True)
    log_pi = np.zeros(K)
    log_pi -= scipy_lse(log_pi)

    if with_minus_inf_row:
        # Force a state to be impossible at one timestep
        log_emit[100, 0] = -np.inf
        log_emit[100, 1] = -np.inf

    return log_emit, log_trans, log_pi


def compare(name, ref_out, jit_out, gamma_tol=1e-10, xi_tol=1e-10, ll_tol=1e-12):
    g_ref, xi_ref, ll_ref = ref_out
    g_jit, xi_jit, ll_jit = jit_out
    dg = np.abs(g_ref - g_jit).max()
    dxi = np.abs(xi_ref - xi_jit).max()
    dll = abs(ll_ref - ll_jit) / (abs(ll_ref) + 1e-12)
    ok_g = dg < gamma_tol
    ok_xi = dxi < xi_tol
    ok_ll = dll < ll_tol
    print(f'\n  {name}')
    print(f'    gamma   max|d|={dg:.2e}    [{"OK" if ok_g else "FAIL"}]')
    print(f'    xi      max|d|={dxi:.2e}    [{"OK" if ok_xi else "FAIL"}]')
    print(f'    log_lik rel|d|={dll:.2e}    [{"OK" if ok_ll else "FAIL"}] '
          f'(ref={ll_ref:.4f}, jit={ll_jit:.4f})')
    return ok_g and ok_xi and ok_ll


def main():
    print('=== Phase 0.2c — Numba forward-backward equivalence ===')
    overall = True

    # T=500 K=4 standard
    log_emit, log_trans, log_pi = make_inputs(T=500, K=4)
    # warmup
    _ = _forward_backward_numba(log_emit[:50], log_trans[:50], log_pi)
    t0 = time.perf_counter()
    jit_out = _forward_backward_numba(log_emit, log_trans, log_pi)
    t_jit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref_out = _fb_numpy_REF(log_emit, log_trans, log_pi)
    t_ref = time.perf_counter() - t0
    print(f'\n  T=500 K=4: ref={t_ref*1000:.1f}ms, numba={t_jit*1000:.1f}ms '
          f'(speedup {t_ref/max(t_jit,1e-9):.1f}x)')
    overall &= compare('Test 1 (T=500, K=4)', ref_out, jit_out)

    # Match production T (6188), K=4
    log_emit, log_trans, log_pi = make_inputs(T=6188, K=4, seed=7)
    t0 = time.perf_counter()
    jit_out = _forward_backward_numba(log_emit, log_trans, log_pi)
    t_jit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref_out = _fb_numpy_REF(log_emit, log_trans, log_pi)
    t_ref = time.perf_counter() - t0
    print(f'\n  T=6188 K=4: ref={t_ref*1000:.1f}ms, numba={t_jit*1000:.1f}ms '
          f'(speedup {t_ref/max(t_jit,1e-9):.1f}x)')
    overall &= compare('Test 2 (T=6188, K=4 — production size)', ref_out, jit_out)

    # K=3 (V11 sensitivity)
    log_emit, log_trans, log_pi = make_inputs(T=2000, K=3, seed=11)
    jit_out = _forward_backward_numba(log_emit, log_trans, log_pi)
    ref_out = _fb_numpy_REF(log_emit, log_trans, log_pi)
    overall &= compare('Test 3 (T=2000, K=3)', ref_out, jit_out)

    # Edge case: -inf emission rows
    log_emit, log_trans, log_pi = make_inputs(T=500, K=4, with_minus_inf_row=True)
    jit_out = _forward_backward_numba(log_emit, log_trans, log_pi)
    ref_out = _fb_numpy_REF(log_emit, log_trans, log_pi)
    overall &= compare('Test 4 (with -inf emission rows)', ref_out, jit_out)

    print(f'\n=== Overall: {"PASS" if overall else "FAIL"} ===')
    sys.exit(0 if overall else 1)


if __name__ == '__main__':
    main()
