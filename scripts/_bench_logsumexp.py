"""Microbenchmark: scipy.special.logsumexp vs hand-rolled.

Profile of MVP fit_slds shows scipy.special.logsumexp eating 80% of wall-time
at 2.1M calls. The bulk of those calls are on (K,)=(4,) and (K,K)=(4,4)
arrays inside _forward_backward — exactly the regime where scipy 1.17's
array-API dispatch overhead dominates the actual logsumexp math.

This benchmark confirms the magnitude of the dispatch overhead before we
swap the import in rslds_model.py.
"""
import time

import numpy as np
from scipy.special import logsumexp as scipy_lse


def _logsumexp(x, axis=None, keepdims=False):
    """Stable logsumexp without scipy's array-API dispatch overhead."""
    m = np.max(x, axis=axis, keepdims=True)
    # protect -inf max (e.g. when an entire row is -inf)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    if not keepdims:
        if axis is None:
            out = out.reshape(())
        else:
            out = np.squeeze(out, axis=axis)
    return out


def bench(name, fn, args, kwargs, n):
    # warmup
    for _ in range(10):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f'  {name:25s}  {elapsed*1e6/n:8.3f} us/call  ({elapsed:.3f}s for {n})')


def main():
    print('Microbenchmark: scipy.special.logsumexp vs hand-rolled\n')

    # Scenario 1: scalar (the line 321/328 cases) — vector input, axis=None
    print('Scenario 1: x.shape=(4,), axis=None  (scalar logsumexp; lines 321/328)')
    x = np.random.randn(4)
    bench('scipy_lse',  scipy_lse, (x,), {}, 5000)
    bench('_logsumexp', _logsumexp, (x,), {}, 5000)

    # Scenario 2: (K,) along axis=0 of (K,K) — line 327
    print('\nScenario 2: x.shape=(4,4), axis=0  (line 327 forward msg)')
    x = np.random.randn(4, 4)
    bench('scipy_lse',  scipy_lse, (x,), {'axis': 0}, 5000)
    bench('_logsumexp', _logsumexp, (x,), {'axis': 0}, 5000)

    # Scenario 3: (K,) along axis=1 of (K,K) — line 338
    print('\nScenario 3: x.shape=(4,4), axis=1  (line 338 backward msg)')
    bench('scipy_lse',  scipy_lse, (x,), {'axis': 1}, 5000)
    bench('_logsumexp', _logsumexp, (x,), {'axis': 1}, 5000)

    # Scenario 4: (T,) along axis=1 of (T,K), keepdims=True — line 342
    print('\nScenario 4: x.shape=(6188,4), axis=1, keepdims=True  (line 342 norm)')
    x = np.random.randn(6188, 4)
    bench('scipy_lse',  scipy_lse, (x,), {'axis': 1, 'keepdims': True}, 1000)
    bench('_logsumexp', _logsumexp, (x,), {'axis': 1, 'keepdims': True}, 1000)

    # Scenario 5: (T,K,K), axis=2, keepdims=True — line 295/1056
    print('\nScenario 5: x.shape=(6188,4,4), axis=2, keepdims=True  (line 295/1056)')
    x = np.random.randn(6188, 4, 4)
    bench('scipy_lse',  scipy_lse, (x,), {'axis': 2, 'keepdims': True}, 500)
    bench('_logsumexp', _logsumexp, (x,), {'axis': 2, 'keepdims': True}, 500)

    # Numerical equivalence check (max abs diff)
    print('\nNumerical equivalence (max abs diff):')
    for desc, x, kw in [
        ('(4,) axis=None',      np.random.randn(4),               {}),
        ('(4,4) axis=0',        np.random.randn(4, 4),            {'axis': 0}),
        ('(4,4) axis=1',        np.random.randn(4, 4),            {'axis': 1}),
        ('(6188,4) axis=1 KD',  np.random.randn(6188, 4),         {'axis': 1, 'keepdims': True}),
        ('(6188,4,4) axis=2 KD', np.random.randn(6188, 4, 4),     {'axis': 2, 'keepdims': True}),
    ]:
        a = np.atleast_1d(scipy_lse(x, **kw))
        b = np.atleast_1d(_logsumexp(x, **kw))
        d = np.abs(a - b).max()
        print(f'  {desc:30s}  max|d| = {d:.2e}')

    # Edge case: row of -inf
    print('\nEdge case: -inf entries')
    x = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    print(f'  scipy:      {scipy_lse(x)}')
    print(f'  _logsumexp: {_logsumexp(x)}')

    x = np.array([1.0, -np.inf, 2.0, -np.inf])
    print(f'  partial -inf:')
    print(f'    scipy:      {scipy_lse(x)}')
    print(f'    _logsumexp: {_logsumexp(x)}')


if __name__ == '__main__':
    main()
