"""Standardization classes for the per-episode feature vector.

Four classes (per spec §4 standardization strategy):

  continuous       z-score cohort-wide
  binary           leave at 0/1 (region-presence indicators)
  sign             leave at {-1, 0, +1} (lead_follow)
  fraction         leave at [0, 1] (sparsity, frac-partnered)

The class assignment is recorded in ``cohort_features.json`` so any
re-cluster knows exactly what scaling was applied.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

# Pattern-based classification — keeps the assignment in one place.
PATTERNS_BINARY = (
    re.compile(r'^region_coinc_present__'),
    re.compile(r'^cca_p[12]_region_present__'),
)
PATTERNS_FRACTION = (
    re.compile(r'^cca_p[12]_sparsity$'),
    re.compile(r'^frac_aus_partnered$'),
    re.compile(r'^cca_lag_var$'),
)
PATTERNS_SIGN = (
    re.compile(r'^lead_follow$'),
)
# Anything matching this pattern is metadata, not for clustering.
PATTERNS_METADATA = (
    re.compile(r'^_3[a-d]_valid$'),
    re.compile(r'^cca_pooled_flag$'),
)


def classify_feature(name: str) -> str:
    if any(p.match(name) for p in PATTERNS_METADATA):
        return 'metadata'
    if any(p.match(name) for p in PATTERNS_BINARY):
        return 'binary'
    if any(p.match(name) for p in PATTERNS_FRACTION):
        return 'fraction'
    if any(p.match(name) for p in PATTERNS_SIGN):
        return 'sign'
    return 'continuous'


def classify_all(names: Iterable[str]) -> dict[str, str]:
    return {n: classify_feature(n) for n in names}


def standardize(X: np.ndarray, classes: list[str], means: np.ndarray | None = None,
                 stds: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply standardization in-place; return (X_z, means, stds).

    Continuous columns get z-scored using ``means`` and ``stds`` (computed
    from X if not provided). Other classes are passed through unchanged.
    """
    X_z = X.astype(np.float32, copy=True)
    cont_idx = np.array([i for i, c in enumerate(classes) if c == 'continuous'],
                        dtype=np.int64)
    if len(cont_idx) == 0:
        return X_z, np.zeros(X.shape[1]), np.ones(X.shape[1])
    if means is None or stds is None:
        means_full = np.zeros(X.shape[1], dtype=np.float64)
        stds_full = np.ones(X.shape[1], dtype=np.float64)
        means_full[cont_idx] = X[:, cont_idx].mean(axis=0)
        stds_full[cont_idx] = X[:, cont_idx].std(axis=0, ddof=1)
        # Avoid divide-by-zero for dead features
        stds_full[stds_full < 1e-10] = 1.0
        means = means_full
        stds  = stds_full
    X_z[:, cont_idx] = ((X[:, cont_idx] - means[cont_idx]) /
                          stds[cont_idx]).astype(np.float32)
    return X_z, means, stds


def impute(X: np.ndarray, classes: list[str], medians: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cohort-median impute continuous; 0-impute binary/sign; track imputed mask.

    Returns (X_imputed, medians, imputed_mask).
    """
    X = X.astype(np.float32, copy=True)
    nan_mask = ~np.isfinite(X)
    if medians is None:
        medians = np.nanmedian(np.where(nan_mask, np.nan, X), axis=0)
        # Where the entire column is NaN (degenerate), default to 0
        medians = np.where(np.isfinite(medians), medians, 0.0)
    for i, c in enumerate(classes):
        if not nan_mask[:, i].any():
            continue
        if c == 'continuous':
            X[nan_mask[:, i], i] = medians[i]
        else:
            X[nan_mask[:, i], i] = 0
    return X, medians, nan_mask
