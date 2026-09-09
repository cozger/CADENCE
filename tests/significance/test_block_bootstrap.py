"""Tests for cadence.significance.block_bootstrap (Phase 1 pose coupling plan, Task 2)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
try:
    import torch as _torch  # noqa: F401  (Windows torch/numpy import-order guard)
except ImportError:
    pass
import numpy as np

from cadence.significance.block_bootstrap import (
    block_bootstrap_mean, block_bootstrap_paired, block_bootstrap_rank,
    block_len_from_seconds,
)


def _ar1(rng: np.random.Generator, T: int, rho: float, sigma: float = 1.0) -> np.ndarray:
    eps = rng.standard_normal(T) * sigma
    x = np.empty(T)
    x[0] = eps[0] / np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + eps[t]
    return x


# ── (i) coverage on i.i.d. normal ─────────────────────────────────────

def test_iid_ci_covers_true_mean():
    T, mu = 200, 0.7
    covered = 0
    for trial in range(100):
        rng = np.random.default_rng(1000 + trial)
        x = mu + rng.standard_normal(T)
        res = block_bootstrap_mean(x, block_len=1, n_boot=800, seed=trial)
        assert res['mean'].shape == (1,)
        assert res['n_blocks'] == T and res['n_dropped'] == 0
        assert res['boot'].shape == (800, 1)
        if res['ci_lo'][0] <= mu <= res['ci_hi'][0]:
            covered += 1
    assert covered >= 90, covered


# ── (ii) autocorrelation inflates SE ──────────────────────────────────

def test_ar1_block_se_exceeds_iid_se():
    rng = np.random.default_rng(7)
    x = _ar1(rng, 4000, rho=0.9)
    se_iid = block_bootstrap_mean(x, block_len=1, n_boot=1500, seed=1)['se'][0]
    se_blk = block_bootstrap_mean(x, block_len=50, n_boot=1500, seed=1)['se'][0]
    assert se_blk >= 2.0 * se_iid, (se_blk, se_iid)


# ── (iii) ranking ─────────────────────────────────────────────────────

def test_rank_three_columns_with_tie():
    rng = np.random.default_rng(3)
    T = 400
    e0 = rng.standard_normal(T)
    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    scores = np.stack([e0 - e0.mean(), e1 - e1.mean(), 1.0 + e2 - e2.mean()], axis=1)
    rows = block_bootstrap_rank(scores, block_len=5, n_boot=1500, seed=0)
    by_index = {r['index']: r for r in rows}
    assert by_index[2]['rank'] == 1
    assert by_index[2]['tied_with'] == []
    assert by_index[2]['p_ranks_first'] > 0.99
    assert by_index[0]['rank'] == 2 and by_index[1]['rank'] == 2
    assert by_index[0]['tied_with'] == [1] and by_index[1]['tied_with'] == [0]
    assert rows[0]['index'] == 2                       # sorted by rank
    assert abs(by_index[2]['mean'] - 1.0) < 1e-9
    lo, hi = by_index[2]['ci95']
    assert lo < 1.0 < hi
    assert by_index[2]['p_greater'][0] > 0.99
    assert abs(by_index[0]['p_greater'][1] - 0.5) <= 0.35


# ── (iv) NaN handling ─────────────────────────────────────────────────

def test_nans_are_ignored_not_propagated():
    rng = np.random.default_rng(11)
    T = 300
    x = 2.0 + rng.standard_normal((T, 2))
    x[10:40, 0] = np.nan
    x[::7, 1] = np.nan
    res = block_bootstrap_mean(x, block_len=10, n_boot=500, seed=0)
    assert np.all(np.isfinite(res['mean']))
    assert np.all(np.isfinite(res['boot']))
    assert np.all(np.isfinite(res['ci_lo'])) and np.all(np.isfinite(res['ci_hi']))
    np.testing.assert_allclose(res['mean'], np.nanmean(x, axis=0))
    assert np.all(res['ci_lo'] <= res['mean']) and np.all(res['mean'] <= res['ci_hi'])

    # all-NaN column -> NaN stats, other column unaffected
    y = x.copy()
    y[:, 0] = np.nan
    res_y = block_bootstrap_mean(y, block_len=10, n_boot=500, seed=0)
    assert np.isnan(res_y['mean'][0]) and np.isnan(res_y['ci_lo'][0])
    np.testing.assert_allclose(res_y['mean'][1], res['mean'][1])


# ── paired difference ─────────────────────────────────────────────────

def test_paired_difference_and_p_greater():
    rng = np.random.default_rng(5)
    T = 600
    a = 1.0 + _ar1(rng, T, 0.6)
    b = _ar1(rng, T, 0.6)
    a[100:120] = np.nan
    b[300] = np.nan
    res = block_bootstrap_paired(a, b, block_len=20, n_boot=1000, seed=2)
    joint = np.isfinite(a) & np.isfinite(b)
    n_used = (T // 20) * 20
    j = joint[:n_used]
    np.testing.assert_allclose(res['mean_diff'][0], a[:n_used][j].mean() - b[:n_used][j].mean())
    np.testing.assert_allclose(res['mean_diff'], res['mean_a'] - res['mean_b'])
    assert res['p_greater'][0] > 0.95
    assert res['ci_lo'][0] < res['mean_diff'][0] < res['ci_hi'][0]
    assert res['boot'].shape == (1000, 1)

    # symmetric: swapping arguments flips the sign and p_greater -> 1 - p
    swp = block_bootstrap_paired(b, a, block_len=20, n_boot=1000, seed=2)
    np.testing.assert_allclose(swp['mean_diff'], -res['mean_diff'])
    np.testing.assert_allclose(swp['boot'], -res['boot'])
    assert abs(swp['p_greater'][0] + res['p_greater'][0] - 1.0) < 1e-9


# ── determinism, shapes, remainder handling ───────────────────────────

def test_seeded_determinism_and_remainder():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((103, 3))
    r1 = block_bootstrap_mean(x, block_len=10, n_boot=200, seed=4)
    r2 = block_bootstrap_mean(x, block_len=10, n_boot=200, seed=4)
    np.testing.assert_array_equal(r1['boot'], r2['boot'])
    assert r1['n_blocks'] == 10 and r1['n_dropped'] == 3
    assert r1['mean'].shape == (3,) and r1['boot'].shape == (200, 3)
    np.testing.assert_allclose(r1['mean'], x[:100].mean(axis=0))

    # weights: zero weight on a run of samples matches masking them out
    w = np.ones(103)
    w[:20] = 0.0
    rw = block_bootstrap_mean(x, block_len=10, n_boot=200, seed=4, weights=w)
    np.testing.assert_allclose(rw['mean'], x[20:100].mean(axis=0))

    # shorter than one block -> NaN, not an exception
    short = block_bootstrap_mean(x[:5], block_len=10, n_boot=50, seed=0)
    assert short['n_blocks'] == 0 and np.all(np.isnan(short['mean']))


def test_block_len_from_seconds():
    assert block_len_from_seconds(10.0, 2.0) == 20
    assert block_len_from_seconds(0.1, 2.0) == 1
    assert block_len_from_seconds(4.0, 12.0) == 48
