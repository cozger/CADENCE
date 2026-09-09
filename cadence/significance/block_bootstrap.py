"""Moving-block bootstrap for per-session summary statistics.

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md (Task 2).

Per-session summaries of coupling z-timecourses (mean z per condition,
mean difference between two candidate pose channels, ranking of several
candidate channels) are computed from strongly autocorrelated 2 Hz series.
An i.i.d. bootstrap or a naive ``se = sd / sqrt(T)`` badly understates the
uncertainty; the block bootstrap (Künsch 1989) resamples contiguous blocks
so within-block dependence is preserved in every resample.

Design:

  * Non-overlapping blocks of ``block_len`` samples; ``n_blocks = T //
    block_len`` and the trailing remainder is dropped (``n_dropped`` is
    reported so callers can pick a block length that wastes little data).
  * A resample draws ``n_blocks`` blocks with replacement.  Because every
    block has the same length, the mean of the resampled series equals the
    (weight-)ratio of resampled block sums, so a resample is fully
    described by a multinomial *count* vector over blocks.  All
    ``n_boot`` resamples are therefore one ``(n_boot, n_blocks) @
    (n_blocks, K)`` matrix product — no per-resample Python loop and no
    ``(n_boot, T)`` gather.
  * NaN-aware: NaN samples receive weight 0 (they neither contribute to a
    block sum nor to its weight), so blocks with missing data are
    down-weighted rather than propagated.
  * The paired and ranking helpers draw a *single* count matrix and apply
    it to every column, so the comparison shares the block structure and
    the paired difference is estimated on the same resampled time indices.

Sign convention for ranking: higher score = better; rank 1 = the column
with the highest mean.  Ties are decided by the bootstrap pairwise
win-probability, not by the point estimate.
"""
from __future__ import annotations

import warnings

import numpy as np


# ── Constants ───────────────────────────────────────────────────────

DEFAULT_N_BOOT = 4000
DEFAULT_CI = 0.95
DEFAULT_TIE_BAND = 0.35


# ── Helpers ─────────────────────────────────────────────────────────

def block_len_from_seconds(seconds: float, fs: float) -> int:
    """Block length in samples for a block of ``seconds`` at rate ``fs`` Hz (>= 1)."""
    if seconds <= 0 or fs <= 0:
        raise ValueError(f'seconds and fs must be positive, got {seconds}, {fs}')
    return max(1, int(round(seconds * fs)))


def _as_2d(x) -> np.ndarray:
    """(T,) or (T,K) array-like -> (T,K) float64 array (never a view of the input)."""
    arr = np.array(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f'x must be (T,) or (T,K); got shape {arr.shape}')
    return arr


def _weights_2d(weights, shape: tuple[int, int]) -> np.ndarray:
    """None | (T,) | (T,K) weights -> (T,K) float64, validated non-negative."""
    T, K = shape
    if weights is None:
        return np.ones((T, K), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        if w.shape[0] != T:
            raise ValueError(f'weights length {w.shape[0]} != T={T}')
        w = np.repeat(w[:, None], K, axis=1)
    elif w.shape != (T, K):
        raise ValueError(f'weights shape {w.shape} incompatible with x shape {(T, K)}')
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite and non-negative')
    return w.copy()


def _block_sums(x: np.ndarray, w: np.ndarray, block_len: int
                ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per-block sums of w*x and w with NaN samples zero-weighted.

    Returns (wx (n_blocks,K), ww (n_blocks,K), n_blocks, n_dropped).
    """
    if block_len < 1:
        raise ValueError(f'block_len must be >= 1, got {block_len}')
    block_len = int(block_len)
    T, K = x.shape
    n_blocks = T // block_len
    n_dropped = T - n_blocks * block_len
    if n_blocks == 0:
        return (np.zeros((0, K)), np.zeros((0, K)), 0, n_dropped)
    finite = np.isfinite(x)
    w = np.where(finite, w, 0.0)
    xz = np.where(finite, x, 0.0)
    T_used = n_blocks * block_len
    wx = (w[:T_used] * xz[:T_used]).reshape(n_blocks, block_len, K).sum(axis=1)
    ww = w[:T_used].reshape(n_blocks, block_len, K).sum(axis=1)
    return wx, ww, n_blocks, n_dropped


def _resample_counts(rng: np.random.Generator, n_blocks: int, n_boot: int) -> np.ndarray:
    """(n_boot, n_blocks) multinomial counts: how often each block is drawn per resample."""
    p = np.full(n_blocks, 1.0 / n_blocks)
    return rng.multinomial(n_blocks, p, size=n_boot).astype(np.float64)


def _boot_means(counts: np.ndarray, wx: np.ndarray, ww: np.ndarray) -> np.ndarray:
    """(n_boot, K) weighted means of the resampled series (NaN where a resample has zero weight)."""
    num = counts @ wx
    den = counts @ ww
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den <= 0] = np.nan
    return out


def _point_mean(wx: np.ndarray, ww: np.ndarray) -> np.ndarray:
    """(K,) weighted mean over the retained (block-covered) samples; NaN if no weight."""
    num = wx.sum(axis=0)
    den = ww.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den <= 0] = np.nan
    return out


def _ci_bounds(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CI along axis 0, ignoring NaN resamples."""
    if not 0.0 < ci < 1.0:
        raise ValueError(f'ci must be in (0, 1), got {ci}')
    alpha = (1.0 - ci) / 2.0
    if boot.shape[0] == 0 or np.all(np.isnan(boot)):
        K = boot.shape[1]
        return np.full(K, np.nan), np.full(K, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN columns -> NaN
        lo = np.nanpercentile(boot, 100.0 * alpha, axis=0)
        hi = np.nanpercentile(boot, 100.0 * (1.0 - alpha), axis=0)
    return lo, hi


def _boot_se(boot: np.ndarray) -> np.ndarray:
    """(K,) bootstrap SE = SD of the resampled means (ddof=1), NaN-ignoring."""
    K = boot.shape[1]
    if boot.shape[0] < 2:
        return np.full(K, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN columns -> NaN
        return np.nanstd(boot, axis=0, ddof=1)


def _empty_result(K: int, n_boot: int, n_dropped: int) -> dict:
    return {
        'mean': np.full(K, np.nan), 'ci_lo': np.full(K, np.nan),
        'ci_hi': np.full(K, np.nan), 'se': np.full(K, np.nan),
        'boot': np.full((n_boot, K), np.nan), 'n_blocks': 0, 'n_dropped': n_dropped,
    }


# ── Mean + CI ───────────────────────────────────────────────────────

def block_bootstrap_mean(x, block_len: int, n_boot: int = DEFAULT_N_BOOT, seed: int = 0,
                         weights=None, ci: float = DEFAULT_CI) -> dict:
    """Block-bootstrap mean and percentile CI of ``x``.

    Parameters
    ----------
    x : (T,) or (T,K)
        Series; NaN samples get weight 0.  A 1-D input is treated as K=1
        (all outputs keep the trailing K axis).
    block_len : int
        Block length in samples (see ``block_len_from_seconds``).
    n_boot, seed : int
        Number of resamples and RNG seed (``np.random.default_rng``).
    weights : None | (T,) | (T,K)
        Optional non-negative sample weights (multiplied with the finite mask).
    ci : float
        Two-sided coverage of the percentile interval.

    Returns
    -------
    dict with ``mean`` (K,), ``ci_lo`` (K,), ``ci_hi`` (K,), ``se`` (K,),
    ``boot`` (n_boot,K), ``n_blocks`` int, ``n_dropped`` int.  The point
    ``mean`` is computed over the block-covered samples only (the trailing
    ``n_dropped`` samples are excluded from both the estimate and the
    resamples, so they agree).  With ``n_blocks == 0`` every statistic is NaN.
    """
    X = _as_2d(x)
    W = _weights_2d(weights, X.shape)
    wx, ww, n_blocks, n_dropped = _block_sums(X, W, block_len)
    K = X.shape[1]
    if n_blocks == 0:
        return _empty_result(K, n_boot, n_dropped)
    rng = np.random.default_rng(seed)
    counts = _resample_counts(rng, n_blocks, n_boot)
    boot = _boot_means(counts, wx, ww)
    lo, hi = _ci_bounds(boot, ci)
    return {
        'mean': _point_mean(wx, ww), 'ci_lo': lo, 'ci_hi': hi, 'se': _boot_se(boot),
        'boot': boot, 'n_blocks': int(n_blocks), 'n_dropped': int(n_dropped),
    }


# ── Paired comparison ───────────────────────────────────────────────

def block_bootstrap_paired(a, b, block_len: int, n_boot: int = DEFAULT_N_BOOT,
                           seed: int = 0, ci: float = DEFAULT_CI) -> dict:
    """Block-bootstrap paired difference ``a - b`` over a shared block structure.

    ``a`` and ``b`` must have the same shape ((T,) or (T,K)).  A sample is
    used only when *both* ``a`` and ``b`` are finite there (joint mask), so
    ``mean_diff == mean_a - mean_b`` exactly and the reported ``p_greater``
    (fraction of resamples where ``mean(a) > mean(b)``) is the fraction of
    resamples with a positive mean difference.

    Returns dict with ``mean_diff``, ``ci_lo``, ``ci_hi``, ``se``,
    ``p_greater`` (all (K,)), ``mean_a``, ``mean_b`` (K,), ``boot``
    (n_boot,K) resampled differences, ``n_blocks``, ``n_dropped``.
    """
    A = _as_2d(a)
    B = _as_2d(b)
    if A.shape != B.shape:
        raise ValueError(f'a and b must share a shape; got {A.shape} vs {B.shape}')
    joint = np.isfinite(A) & np.isfinite(B)
    W = joint.astype(np.float64)
    wa, ww, n_blocks, n_dropped = _block_sums(A, W, block_len)
    wb, _, _, _ = _block_sums(B, W, block_len)
    K = A.shape[1]
    if n_blocks == 0:
        out = _empty_result(K, n_boot, n_dropped)
        out['mean_diff'] = out.pop('mean')
        out.update(mean_a=np.full(K, np.nan), mean_b=np.full(K, np.nan),
                   p_greater=np.full(K, np.nan))
        return out
    rng = np.random.default_rng(seed)
    counts = _resample_counts(rng, n_blocks, n_boot)
    boot_a = _boot_means(counts, wa, ww)
    boot_b = _boot_means(counts, wb, ww)
    boot = boot_a - boot_b
    lo, hi = _ci_bounds(boot, ci)
    with np.errstate(invalid='ignore'):
        finite = np.isfinite(boot)
        n_fin = finite.sum(axis=0)
        p_greater = np.where(n_fin > 0,
                             (boot > 0).sum(axis=0) / np.maximum(n_fin, 1), np.nan)
    mean_a = _point_mean(wa, ww)
    mean_b = _point_mean(wb, ww)
    return {
        'mean_diff': mean_a - mean_b, 'mean_a': mean_a, 'mean_b': mean_b,
        'ci_lo': lo, 'ci_hi': hi, 'se': _boot_se(boot), 'p_greater': p_greater,
        'boot': boot, 'n_blocks': int(n_blocks), 'n_dropped': int(n_dropped),
    }


# ── Ranking ─────────────────────────────────────────────────────────

def block_bootstrap_rank(scores, block_len: int, n_boot: int = DEFAULT_N_BOOT,
                         seed: int = 0, tie_band: float = DEFAULT_TIE_BAND) -> list[dict]:
    """Competition-style ranking of the columns of ``scores`` (T,K) by block-bootstrap mean.

    A single count matrix is shared by all columns.  For every ordered pair
    ``P(i>j)`` is the fraction of resamples with ``mean_i > mean_j``; columns
    ``i`` and ``j`` tie when ``|P(i>j) - 0.5| <= tie_band``.  Column ``j``
    *beats* ``i`` when ``P(j>i) - 0.5 > tie_band``.  Competition rank of
    ``i`` is ``1 + #{j : j beats i}`` ("1224" style: tied columns share the
    best rank they would receive and the next rank is skipped).

    Each row: ``{'index', 'rank', 'mean', 'ci95': (lo, hi), 'se',
    'p_ranks_first', 'tied_with': list[int], 'p_greater': (K,)}`` where
    ``p_greater[j] = P(i>j)`` and ``p_ranks_first`` is the fraction of
    resamples in which column ``i`` has the largest mean.  Rows are sorted
    by (rank, -mean); ``index`` is the original column index.  A column
    with no finite data gets NaN statistics and the last rank (K).
    """
    S = _as_2d(scores)
    if not 0.0 <= tie_band < 0.5:
        raise ValueError(f'tie_band must be in [0, 0.5), got {tie_band}')
    K = S.shape[1]
    W = np.ones_like(S)
    wx, ww, n_blocks, n_dropped = _block_sums(S, W, block_len)
    if n_blocks == 0:
        return [{'index': i, 'rank': K, 'mean': np.nan, 'ci95': (np.nan, np.nan),
                 'se': np.nan, 'p_ranks_first': np.nan, 'tied_with': [],
                 'p_greater': np.full(K, np.nan)} for i in range(K)]
    rng = np.random.default_rng(seed)
    counts = _resample_counts(rng, n_blocks, n_boot)
    boot = _boot_means(counts, wx, ww)                        # (n_boot, K)
    mean = _point_mean(wx, ww)
    lo, hi = _ci_bounds(boot, 0.95)
    se = _boot_se(boot)
    usable = np.isfinite(mean)

    # Pairwise win probability over resamples where both columns are finite.
    both = np.isfinite(boot)[:, :, None] & np.isfinite(boot)[:, None, :]  # (n_boot,K,K)
    with np.errstate(invalid='ignore'):
        wins = (boot[:, :, None] > boot[:, None, :]) & both
        n_both = both.sum(axis=0)
        p_gt = np.where(n_both > 0, wins.sum(axis=0) / np.maximum(n_both, 1), np.nan)
    np.fill_diagonal(p_gt, np.nan)

    # p_ranks_first: NaN columns never win; resamples with no finite column are ignored.
    boot_filled = np.where(np.isfinite(boot), boot, -np.inf)
    any_finite = np.isfinite(boot).any(axis=1)
    if any_finite.any():
        argmax = boot_filled[any_finite].argmax(axis=1)
        p_first = np.bincount(argmax, minlength=K) / any_finite.sum()
    else:
        p_first = np.full(K, np.nan)

    beats = np.zeros((K, K), dtype=bool)   # beats[j, i]: j significantly better than i
    ties = np.zeros((K, K), dtype=bool)
    with np.errstate(invalid='ignore'):
        dev = p_gt - 0.5
        beats[np.isfinite(dev) & (dev > tie_band)] = True
        ties[np.isfinite(dev) & (np.abs(dev) <= tie_band)] = True
    np.fill_diagonal(ties, False)
    ranks = 1 + beats.sum(axis=0)          # column i beaten by count of j
    ranks = np.where(usable, ranks, K)

    rows = []
    for i in range(K):
        rows.append({
            'index': int(i),
            'rank': int(ranks[i]),
            'mean': float(mean[i]),
            'ci95': (float(lo[i]), float(hi[i])),
            'se': float(se[i]),
            'p_ranks_first': float(p_first[i]) if usable[i] else float('nan'),
            'tied_with': [int(j) for j in np.flatnonzero(ties[i]) if usable[j]] if usable[i] else [],
            'p_greater': p_gt[i].copy(),
        })
    rows.sort(key=lambda r: (r['rank'], -r['mean'] if np.isfinite(r['mean']) else np.inf, r['index']))
    return rows
