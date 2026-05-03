"""Stage 3a — per-AU event-coincidence features within episode.

For each of 52 AUs, slice both participants' Stage 1 events whose time
falls inside the episode. Run nearest-neighbor matching with a max-lag
window. Aggregate to per-episode features.

Vectorized: ``np.searchsorted`` finds nearest P2-event-time for every
P1-event-time across all 52 AUs simultaneously, no Python loop over AUs.

Features per episode (13):
  n_events_p1, n_events_p2, dominant_lag, lag_variance, lead_follow,
  frac_aus_partnered, region_coinc_present[7]

If either role has 0 events in the episode → defaults of 0/NaN with
``_3a_valid=False``.
"""
from __future__ import annotations

import numpy as np

from cadence.constants import AU_REGIONS_7
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG


N_AUS = 52
REGION_NAMES = list(AU_REGIONS_7.keys())  # ordered
N_REGIONS = len(REGION_NAMES)

# AU index → region index (or -1 if excluded)
_AU_TO_REGION = np.full(N_AUS, -1, dtype=np.int8)
for r_idx, (_r, lst) in enumerate(AU_REGIONS_7.items()):
    for au in lst:
        _AU_TO_REGION[au] = r_idx


def _nn_lag(t_a: np.ndarray, t_b: np.ndarray, max_lag: float):
    """For each event in t_a, find signed lag to nearest in t_b within ±max_lag.

    Returns:
      partner_idx: (len(t_a),) int — index into t_b, or -1 if no partner
      lag:         (len(t_a),) float — signed lag in seconds (+ = b follows a),
                    NaN where no partner.
    """
    n = len(t_a)
    out_idx = np.full(n, -1, dtype=np.int64)
    out_lag = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or len(t_b) == 0:
        return out_idx, out_lag
    pos = np.searchsorted(t_b, t_a)
    # Candidates: t_b[pos-1] and t_b[pos]
    for which in (-1, 0):
        cand = pos + which
        in_range = (cand >= 0) & (cand < len(t_b))
        # Filter
        idx_valid = np.where(in_range)[0]
        if not len(idx_valid):
            continue
        cand_b = cand[idx_valid]
        lag_here = t_b[cand_b] - t_a[idx_valid]
        within = np.abs(lag_here) <= max_lag
        idx_valid = idx_valid[within]
        cand_b = cand_b[within]
        lag_here = lag_here[within]
        # Update where current candidate is closer than what we have
        prev_lag = out_lag[idx_valid]
        better = np.isnan(prev_lag) | (np.abs(lag_here) < np.abs(prev_lag))
        out_idx[idx_valid[better]] = cand_b[better]
        out_lag[idx_valid[better]] = lag_here[better]
    return out_idx, out_lag


def compute_coincidence_features(events_in_ep: dict,
                                   config: SynchronyConfig = DEFAULT_CONFIG) -> dict:
    """Compute Stage 3a feature dict for one episode.

    Args:
        events_in_ep: dict with arrays sliced to this episode:
            't_lsl' (n,), 'au_idx' (n,), 'role' (n,) ['therapist'|'patient']
        config

    Returns:
        Dict with 13 keys + ``_3a_valid`` boolean.
    """
    t = np.asarray(events_in_ep.get('t_lsl', np.zeros(0)))
    au = np.asarray(events_in_ep.get('au_idx', np.zeros(0, dtype=np.int16)))
    role = np.asarray(events_in_ep.get('role', np.array([], dtype=object)))

    is_t = role == 'therapist'
    is_p = role == 'patient'
    n_t = int(is_t.sum())
    n_p = int(is_p.sum())

    out = {
        'n_events_therapist': n_t,
        'n_events_patient':   n_p,
        'dominant_lag':       0.0,
        'lag_variance':       0.0,
        'lead_follow':        0,
        'frac_aus_partnered': 0.0,
    }
    out.update({f'region_coinc_present__{r}': 0 for r in REGION_NAMES})
    out['_3a_valid'] = False

    if n_t == 0 or n_p == 0:
        return out

    # Per-AU NN-lag matching.
    per_au_dom_lag = []
    per_au_n_events_either = []
    region_present = np.zeros(N_REGIONS, dtype=np.int8)

    for au_id in range(N_AUS):
        m_au = (au == au_id)
        t_t = np.sort(t[m_au & is_t])
        t_p = np.sort(t[m_au & is_p])
        if t_t.size == 0 and t_p.size == 0:
            continue
        per_au_n_events_either.append(au_id)
        if t_t.size == 0 or t_p.size == 0:
            continue

        # NN-lag: t -> nearest p, lag positive = patient follows therapist
        _, lag = _nn_lag(t_t, t_p, max_lag=config.coinc_max_lag_s)
        finite = np.isfinite(lag)
        if finite.any():
            per_au_dom_lag.append(np.median(lag[finite]))
            # region presence: any partner within tighter ±coinc_region_lag_s?
            tight = (np.abs(lag[finite]) <= config.coinc_region_lag_s).any()
            r_idx = int(_AU_TO_REGION[au_id])
            if tight and r_idx >= 0:
                region_present[r_idx] = 1

    n_aus_with_either = len(per_au_n_events_either)
    if n_aus_with_either == 0:
        return out
    n_aus_partnered = len(per_au_dom_lag)
    frac_aus_partnered = n_aus_partnered / n_aus_with_either

    if n_aus_partnered == 0:
        out['frac_aus_partnered'] = 0.0
        return out

    per_au_dom_lag = np.asarray(per_au_dom_lag)
    dominant_lag = float(np.median(per_au_dom_lag))
    lag_variance = float(per_au_dom_lag.var(ddof=0))
    lead_follow  = int(np.sign(dominant_lag))

    out.update({
        'dominant_lag':       dominant_lag,
        'lag_variance':       lag_variance,
        'lead_follow':        lead_follow,
        'frac_aus_partnered': float(frac_aus_partnered),
    })
    for i, r in enumerate(REGION_NAMES):
        out[f'region_coinc_present__{r}'] = int(region_present[i])
    out['_3a_valid'] = True
    return out


def feature_names() -> list[str]:
    """Stable ordering for column assembly."""
    return [
        'n_events_therapist', 'n_events_patient',
        'dominant_lag', 'lag_variance', 'lead_follow', 'frac_aus_partnered',
        *[f'region_coinc_present__{r}' for r in REGION_NAMES],
    ]
