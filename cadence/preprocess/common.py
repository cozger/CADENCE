"""Shared utilities for ``cadence.preprocess.<modality>`` submodules.

Ported from ``cadence/data/preprocessors.py`` (gap-fill, activity channel,
z-score normalization). Submodule pipelines compose these into per-modality
``preprocess_<modality>_session`` functions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Gap detection / linear-interp gap fill
# ---------------------------------------------------------------------------

def find_gaps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gap_starts, gap_lengths) for contiguous False regions in ``mask``."""
    if len(mask) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == -1)[0] + 1
    ends = np.where(changes == 1)[0] + 1
    if not mask[0]:
        starts = np.concatenate([[0], starts])
    if not mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    n = min(len(starts), len(ends))
    return starts[:n], (ends[:n] - starts[:n])


def fill_short_gaps_linear(arr: np.ndarray, valid_mask: np.ndarray,
                           max_gap_samples: int) -> np.ndarray:
    """Fill 1-D or 2-D gaps shorter than ``max_gap_samples`` with linear interp.

    Returns the filled array. ``valid_mask`` is a 1-D per-frame mask;
    interpolation is applied row-wise (broadcasting across columns).
    Modifies a copy.
    """
    out = arr.copy()
    starts, lengths = find_gaps(valid_mask)
    if arr.ndim == 1:
        ncols = 1
        out_2d = out.reshape(-1, 1)
    else:
        ncols = arr.shape[1]
        out_2d = out
    for s, ln in zip(starts, lengths):
        if ln > max_gap_samples or s == 0 or s + ln >= len(arr):
            continue
        for c in range(ncols):
            out_2d[s:s + ln, c] = np.linspace(
                out_2d[s - 1, c], out_2d[s + ln, c], ln
            )
    return out


# ---------------------------------------------------------------------------
# Activity channel (causal trailing-mean deviation)
# ---------------------------------------------------------------------------

def compute_temporal_derivatives(signal: np.ndarray, hz: float,
                                 sigma_s: float = 0.5) -> np.ndarray:
    """Smoothed first derivatives (Gaussian-smoothed forward differences).

    Returns shape ``(N, C)`` matching the input. Used by face PCA + V7
    derivative features. Ported from
    ``cadence/data/preprocessors.py:_compute_temporal_derivatives``.
    """
    from scipy.ndimage import gaussian_filter1d
    dt = 1.0 / max(hz, 1e-3)
    deriv = np.diff(signal, axis=0, prepend=signal[:1]) / dt
    sigma_samples = sigma_s * hz
    if sigma_samples > 0.5:
        for ch in range(deriv.shape[1]):
            deriv[:, ch] = gaussian_filter1d(deriv[:, ch], sigma_samples)
    return deriv.astype(np.float32)


def compute_activity_channel(features: np.ndarray, hz: float,
                             trailing_seconds: float = 30.0) -> np.ndarray:
    """Per-frame RMS deviation from a causal trailing mean.

    Returns shape ``(N, 1)``. Captures activation density independent of
    direction. Ported from ``cadence/data/preprocessors.py:compute_activity_channel``.
    """
    n = len(features)
    win = max(1, int(trailing_seconds * hz))
    activity = np.zeros((n, 1))
    if n == 0:
        return activity
    cum = np.cumsum(features, axis=0)
    for i in range(n):
        lo = max(0, i - win)
        if i == lo:
            activity[i, 0] = 0.0
            continue
        mean = (cum[i] - cum[lo]) / (i - lo)
        deviation = features[i] - mean
        activity[i, 0] = np.sqrt(np.mean(deviation ** 2))
    return activity


# ---------------------------------------------------------------------------
# Per-channel z-score with masked-sample statistics
# ---------------------------------------------------------------------------

def zscore_columns(arr: np.ndarray, sample_mask: np.ndarray | None = None,
                   min_samples: int = 100, eps: float = 1e-8) -> np.ndarray:
    """Z-score each column using only ``sample_mask``-True rows for stats.

    Per-column statistics; columns with fewer than ``min_samples`` valid
    samples or near-zero std are left unchanged.
    """
    if arr.ndim != 2:
        raise ValueError("zscore_columns expects a 2D array")
    out = arr.astype(np.float64, copy=True)
    if sample_mask is None:
        sample_mask = np.ones(out.shape[0], dtype=bool)
    sub = out[sample_mask]
    if sub.shape[0] < min_samples:
        return out
    for c in range(out.shape[1]):
        col = sub[:, c]
        sigma = float(np.std(col))
        if sigma > eps:
            out[:, c] = (out[:, c] - float(np.mean(col))) / sigma
    return out


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save ``arrays`` to ``path`` via ``path.tmp`` + os.replace (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    np.savez(tmp, **arrays)
    # np.savez may append .npz to the tmp path; resolve actual produced file
    if not tmp.exists() and Path(str(tmp) + ".npz").exists():
        produced = Path(str(tmp) + ".npz")
    else:
        produced = tmp
    os.replace(produced, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceMeta:
    session_id: str
    digest_xdf_md5: str
    digest_schema_version: str
    modality: str
    modality_version: str = "v1"
    params: dict = field(default_factory=dict)


def staleness_check(json_path: Path, expected_digest_xdf_md5: str,
                    *, key: str = "digest_xdf_md5") -> bool:
    """Return True iff the existing sidecar's md5 matches the expected one."""
    if not json_path.is_file():
        return False
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False
    return existing.get(key) == expected_digest_xdf_md5


# ---------------------------------------------------------------------------
# Output dir convention
# ---------------------------------------------------------------------------

PREPROC_ROOT = Path("data/preproc")


def default_out_dir(modality: str, version: str = "v1") -> Path:
    return PREPROC_ROOT / modality / version
