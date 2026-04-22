from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax
from scipy.stats import ks_2samp


def compute_dwell_times(path: np.ndarray) -> Dict[int, List[int]]:
    """Return dict mapping state_id -> list of dwell lengths (in samples)."""
    dwells: Dict[int, List[int]] = {}
    if len(path) == 0:
        return dwells
    current, count = int(path[0]), 1
    for s in path[1:]:
        s = int(s)
        if s == current:
            count += 1
        else:
            dwells.setdefault(current, []).append(count)
            current, count = s, 1
    dwells.setdefault(current, []).append(count)
    return dwells


def transition_matrix_from_logits(W_trans: np.ndarray) -> np.ndarray:
    """Convert (K, K) logit matrix to row-stochastic transition matrix."""
    return np.array([softmax(W_trans[k]) for k in range(len(W_trans))])


def geometric_mean_dwell(T_kk: float) -> float:
    """Expected mean dwell in samples for geometric(1 - T_kk)."""
    return 1.0 / (1.0 - T_kk)


def transition_event_ks_test(
    transition_times: np.ndarray,
    event_times: List[float],
    session_length_s: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """KS test: are transitions closer to events than random?

    Returns (ks_statistic, p_value).
    p < 0.05 → transitions cluster near events (real signal).
    Uses alternative='greater': F_real(t) > F_boot(t) when real distances are
    smaller, giving a large statistic and small p-value.
    """
    rng = np.random.default_rng(seed)
    event_arr = np.array(event_times)

    def min_dist(times):
        return np.array([np.min(np.abs(t - event_arr)) for t in times])

    real_dists = min_dist(transition_times)
    n = len(transition_times)
    boot_dists = np.concatenate([
        min_dist(rng.uniform(0, session_length_s, n))
        for _ in range(n_boot)
    ])
    # 'greater': F_real(t) > F_boot(t) when real are smaller → large statistic → small p
    ks_stat, p_val = ks_2samp(real_dists, boot_dists, alternative='greater')
    return float(ks_stat), float(p_val)
