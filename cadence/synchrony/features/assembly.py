"""Stage 4 — per-episode feature assembly + cohort pooling.

Per-episode call (sequenced via joblib threading across episodes):
    1. Slice Stage 1 events to episode time window (cheap: pre-sorted by t_lsl).
    2. Slice Stage 0 baselined AU trajectories to episode time window per role.
    3. Run blocks 3a, 3b, 3c (per-episode), 3d (slice from session-level coh).
    4. Compute identity (28), dynamics (10), intensity (3), context (2).
    5. Return one feature dict.

Cohort fan-in (separate function ``pool_cohort_features``):
    Stack per-session caches → impute (cohort medians) → standardize
    (continuous z-score, others passthrough) → write
    ``cohort/cohort_features.npz``.
"""
from __future__ import annotations

# Hoist torch via _gpu before numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from cadence.constants import AU_REGIONS_7
from cadence.significance.face_event_coincidence import (
    _baseline_subtract, _smooth_savgol,
)
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG
from cadence.synchrony.events import _resolve_role_to_p, _interp_invalid
from cadence.synchrony.episodes import (
    _resampled_joint_envelope, compute_cohort_threshold,
)
from cadence.synchrony.features import coincidence as feat_coinc
from cadence.synchrony.features import dtw as feat_dtw
from cadence.synchrony.features import cca as feat_cca
from cadence.synchrony.features import coherence as feat_coh
from cadence.synchrony.features._standardization import (
    classify_all, impute, standardize,
)
from cadence.synchrony.io import (
    cohort_dir, is_stage_fresh, load_digest, load_face_npz, read_stage,
    write_stage,
)


REGION_NAMES = list(AU_REGIONS_7.keys())

# ── Identity / dynamics / intensity / context feature names ───────────

def _identity_feature_names() -> list[str]:
    out = []
    for role in ('therapist', 'patient'):
        for region in REGION_NAMES:
            for stat in ('mean', 'peak'):
                out.append(f'id_{role}_{region}_{stat}')
    return out


def _dynamics_feature_names() -> list[str]:
    out = ['dyn_duration_s', 'dyn_n_events_total']
    for role in ('therapist', 'patient'):
        for stat in ('mean_alpha', 'dominant_alpha', 'peak_time_rel',
                      'envelope_skew'):
            out.append(f'dyn_{role}_{stat}')
    return out


def _intensity_feature_names() -> list[str]:
    return ['int_peak_env', 'int_mean_env', 'int_peak_au_amp']


def _context_feature_names() -> list[str]:
    return ['ctx_time_since_prev_episode_s', 'ctx_recent_episode_rate_60s']


def all_feature_names() -> list[str]:
    """Stable column ordering for the per-episode matrix."""
    return (
        feat_coinc.feature_names()
        + feat_dtw.feature_names()
        + feat_cca.feature_names_clustering()
        + feat_coh.feature_names()
        + _identity_feature_names()
        + _dynamics_feature_names()
        + _intensity_feature_names()
        + _context_feature_names()
    )


# ── Per-episode helpers ───────────────────────────────────────────────

def _identity_features(p1_au_ep: np.ndarray, p2_au_ep: np.ndarray,
                         role_to_p: dict) -> dict:
    """28 features: per-region (mean, peak) AU activation per role."""
    out = {}
    # Note: caller passes (p1, p2) — we map to roles via role_to_p
    # role_to_p is {'therapist': 'p1', 'patient': 'p2'} or swapped
    therapist_au = p1_au_ep if role_to_p['therapist'] == 'p1' else p2_au_ep
    patient_au   = p2_au_ep if role_to_p['therapist'] == 'p1' else p1_au_ep
    role_data = {'therapist': therapist_au, 'patient': patient_au}
    for role, au_data in role_data.items():
        for r in REGION_NAMES:
            idxs = AU_REGIONS_7[r]
            if au_data.shape[0] == 0:
                out[f'id_{role}_{r}_mean'] = 0.0
                out[f'id_{role}_{r}_peak'] = 0.0
                continue
            region_per_t = au_data[:, idxs].mean(axis=1)  # (T,)
            out[f'id_{role}_{r}_mean'] = float(region_per_t.mean())
            out[f'id_{role}_{r}_peak'] = float(region_per_t.max())
    return out


def _dynamics_features(events_in_ep: dict, p1_env: np.ndarray, p2_env: np.ndarray,
                         duration_s: float, role_to_p: dict) -> dict:
    """10 features: duration, total event count, per-role α stats + skew + peak time."""
    from scipy.stats import skew

    out = {'dyn_duration_s': float(duration_s),
            'dyn_n_events_total': int(len(events_in_ep.get('t_lsl', [])))}
    role_arr = np.asarray(events_in_ep.get('role', []))
    alpha_arr = np.asarray(events_in_ep.get('alpha', []), dtype=np.float64)
    amp_arr   = np.asarray(events_in_ep.get('amp', []), dtype=np.float64)
    for role in ('therapist', 'patient'):
        m = (role_arr == role)
        if m.any():
            a = alpha_arr[m]
            out[f'dyn_{role}_mean_alpha'] = float(np.nanmean(a))
            # Dominant: α of the highest-amp event
            dom_idx = int(np.nanargmax(amp_arr[m]))
            out[f'dyn_{role}_dominant_alpha'] = float(a[dom_idx])
        else:
            out[f'dyn_{role}_mean_alpha'] = np.nan
            out[f'dyn_{role}_dominant_alpha'] = np.nan
    # Per-role envelope peak-time-relative + skew (using each role's own envelope)
    role_env = {'therapist': p1_env if role_to_p['therapist'] == 'p1' else p2_env,
                 'patient':   p2_env if role_to_p['therapist'] == 'p1' else p1_env}
    for role, env in role_env.items():
        if env.size > 1:
            out[f'dyn_{role}_peak_time_rel'] = float(np.argmax(env) / max(len(env) - 1, 1))
            out[f'dyn_{role}_envelope_skew'] = float(skew(env, bias=False))
        else:
            out[f'dyn_{role}_peak_time_rel'] = 0.5
            out[f'dyn_{role}_envelope_skew'] = 0.0
    return out


def _intensity_features(joint_env_ep: np.ndarray, ev_amp_in_ep: np.ndarray) -> dict:
    return {
        'int_peak_env':    float(joint_env_ep.max()) if joint_env_ep.size else 0.0,
        'int_mean_env':    float(joint_env_ep.mean()) if joint_env_ep.size else 0.0,
        'int_peak_au_amp': float(ev_amp_in_ep.max()) if ev_amp_in_ep.size else 0.0,
    }


def _context_features(t_start_lsl: float, episode_starts: np.ndarray) -> dict:
    """gap to prev episode + recent rate (60s window)."""
    earlier = episode_starts < t_start_lsl
    if earlier.any():
        gap = float(min(t_start_lsl - episode_starts[earlier].max(), 300.0))
    else:
        gap = 300.0
    recent_n = int(((episode_starts >= (t_start_lsl - 60.0))
                     & (episode_starts < t_start_lsl)).sum())
    return {'ctx_time_since_prev_episode_s': gap,
             'ctx_recent_episode_rate_60s':   float(recent_n)}


# ── Per-session orchestrator ──────────────────────────────────────────

def assemble_session_features(sid: str, config: SynchronyConfig = DEFAULT_CONFIG,
                                force: bool = False) -> dict:
    """Run Stage 4 on one session; cache to ``03_features_per_episode.npz``."""
    config_hash = config.hash()
    if not force and is_stage_fresh(sid, 3, config_hash):
        npz, meta = read_stage(sid, 3)
        return {**npz, '_meta': meta, '_from_cache': True}

    t0 = time.perf_counter()

    # Pull stage 1 + 2 caches
    ev_npz, _ = read_stage(sid, 1)
    ep_npz, ep_meta = read_stage(sid, 2)

    feat_names = all_feature_names()
    n_eps = ep_meta['n_episodes']
    if n_eps == 0:
        # Empty session — nothing to cluster
        empty = {
            'features':    np.zeros((0, len(feat_names)), dtype=np.float32),
            'episode_id':  np.zeros(0, dtype=np.int32),
            't_start_lsl': np.zeros(0, dtype=np.float64),
            'condition':   np.array([], dtype=object),
            'cca_pooled_flag': np.zeros(0, dtype=bool),
            '_3a_valid':   np.zeros(0, dtype=bool),
            '_3b_valid':   np.zeros(0, dtype=bool),
            '_3c_valid':   np.zeros(0, dtype=bool),
            '_3d_valid':   np.zeros(0, dtype=bool),
        }
        sidecar_extra = {'n_episodes_in_features': 0, 'feature_names': feat_names,
                          'wall_seconds': round(time.perf_counter() - t0, 3),
                          'status': ep_meta.get('status', 'no_episodes')}
        write_stage(sid, 3, empty, sidecar_extra, config_hash)
        return {**empty, '_meta': sidecar_extra, '_from_cache': False}

    # Pre-load shared per-session data
    face = load_face_npz(sid)
    digest = load_digest(sid)
    role_to_p = _resolve_role_to_p(face, digest)
    fs = config.fs_native
    lsl_offset = float(digest.get('t_start_lsl', 0.0))

    # Per-role baselined AU trajectories at native rate (Stage 0)
    p_keys = ('p1', 'p2')
    au_baselined = {}
    au_ts_lsl = {}
    au_env = {}
    for p_key in p_keys:
        au_data = np.asarray(face[f'{p_key}_au52'], dtype=np.float64)
        ts = np.asarray(face[f'{p_key}_au52_ts'], dtype=np.float64)
        valid = (np.asarray(face[f'{p_key}_au_valid'], dtype=bool)
                 if f'{p_key}_au_valid' in face
                 else np.ones(au_data.shape[0], dtype=bool))
        au_interp = _interp_invalid(au_data, valid)
        au_smoothed = _smooth_savgol(au_interp, window=config.smooth_window,
                                       poly=config.smooth_poly)
        au_baselined[p_key] = _baseline_subtract(au_smoothed, fs,
                                                   window_s=config.baseline_window_s,
                                                   q=config.baseline_q)
        au_ts_lsl[p_key] = ts + lsl_offset
        au_env[p_key] = np.asarray(face[f'{p_key}_au_activity'],
                                     dtype=np.float64).squeeze()

    # Joint envelope on patient timestamps (matches episode segmenter)
    joint, ts_rel = _resampled_joint_envelope(face, role_to_p)
    ts_lsl_master = ts_rel + lsl_offset

    # Session-level coherence arrays (one CWT pair per session)
    p_pat = role_to_p['patient']
    p_the = role_to_p['therapist']
    coh_freqs = coh_arr = phase_arr = None
    try:
        coh_freqs, coh_arr, phase_arr = feat_coh.compute_session_coherence_arrays(
            au_env[p_the], au_env[p_pat], fs=fs, config=config)
    except Exception as e:
        coh_arr = None  # will fall through to NaN per-episode

    # Episode arrays
    ep_starts_lsl = np.asarray(ep_npz['t_start_lsl'])
    ep_ends_lsl   = np.asarray(ep_npz['t_end_lsl'])
    ep_durs       = np.asarray(ep_npz['duration_s'])
    ep_peak_env   = np.asarray(ep_npz['peak_env'])
    ep_mean_env   = np.asarray(ep_npz['mean_env'])
    ep_cond       = np.asarray(ep_npz['condition'])

    # Member event index pointers
    moff = np.asarray(ep_npz['member_event_offsets'], dtype=np.int64)
    midx = np.asarray(ep_npz['member_event_indices'], dtype=np.int64)
    ev_t_lsl = np.asarray(ev_npz['t_lsl'])
    ev_au    = np.asarray(ev_npz['au_idx'])
    ev_role  = np.asarray(ev_npz['role'])
    ev_alpha = np.asarray(ev_npz['alpha'])
    ev_amp   = np.asarray(ev_npz['amp'])

    # ── CCA pooling pre-pass: assign each (too-short) episode to a pool ──
    # Per-condition greedy duration-bin: collect short episodes in the same
    # condition, run SCCA on each bin's concatenated samples, assign the same
    # 3c features to every member of the bin.
    cca_features_by_episode: dict[int, dict] = {}
    if n_eps:
        from cadence.synchrony.features.cca import (
            compute_cca_features_pooled,
        )
        T_min_per = config.cca_min_per_episode_samples
        T_min_pool = config.cca_min_pooled_samples
        ep_lengths = (np.asarray(ep_durs) * fs).astype(np.int32)
        # Episodes long enough for per-episode → run individually below.
        # Episodes too short → assign to per-condition pools.
        condition_pools: dict[str, list[int]] = {}
        for k in range(n_eps):
            if ep_lengths[k] < T_min_per:
                condition_pools.setdefault(ep_cond[k] or '__no_cond__', []).append(k)
        for cond, eps_in_cond in condition_pools.items():
            # Sort by duration so similar-length episodes get pooled together
            order = sorted(eps_in_cond, key=lambda kk: ep_lengths[kk])
            current_bin = []
            current_total = 0
            bins: list[list[int]] = []
            for kk in order:
                current_bin.append(kk)
                current_total += int(ep_lengths[kk])
                if current_total >= T_min_pool:
                    bins.append(current_bin)
                    current_bin = []
                    current_total = 0
            # Trailing partial bin: include only if it itself has enough samples
            if current_bin and current_total >= T_min_pool:
                bins.append(current_bin)
            for bin_eps in bins:
                # Build per-episode (p1_au_ep, p2_au_ep) slices for this bin
                p1_segs = []
                p2_segs = []
                for kk in bin_eps:
                    t0_ep = float(ep_starts_lsl[kk])
                    t1_ep = float(ep_ends_lsl[kk])
                    i0 = int(np.searchsorted(au_ts_lsl['p1'], t0_ep, side='left'))
                    i1 = int(np.searchsorted(au_ts_lsl['p1'], t1_ep, side='right'))
                    j0 = int(np.searchsorted(au_ts_lsl['p2'], t0_ep, side='left'))
                    j1 = int(np.searchsorted(au_ts_lsl['p2'], t1_ep, side='right'))
                    s1 = au_baselined['p1'][i0:i1]
                    s2 = au_baselined['p2'][j0:j1]
                    L = min(s1.shape[0], s2.shape[0])
                    if L < 4:
                        continue
                    p1_segs.append(s1[:L])
                    p2_segs.append(s2[:L])
                if not p1_segs:
                    continue
                pooled_feats = compute_cca_features_pooled(
                    p1_segs, p2_segs, fs=fs, config=config)
                if pooled_feats.get('_3c_valid', False):
                    for kk in bin_eps:
                        cca_features_by_episode[kk] = pooled_feats

    # ── Per-episode worker ────────────────────────────────────────────
    def _process_one(k):
        t0_ep = float(ep_starts_lsl[k])
        t1_ep = float(ep_ends_lsl[k])
        member = midx[moff[k]:moff[k + 1]]
        ev_in_ep = {
            't_lsl':  ev_t_lsl[member],
            'au_idx': ev_au[member],
            'role':   ev_role[member],
            'alpha':  ev_alpha[member],
            'amp':    ev_amp[member],
        }
        # Slice each role's AU trajectory to episode window
        slices = {}
        for p_key in p_keys:
            ts = au_ts_lsl[p_key]
            i0 = int(np.searchsorted(ts, t0_ep, side='left'))
            i1 = int(np.searchsorted(ts, t1_ep, side='right'))
            slices[p_key] = au_baselined[p_key][i0:i1]

        p1_au_ep = slices['p1']
        p2_au_ep = slices['p2']

        # Per-role envelope slice (for dynamics features)
        env_slices = {}
        for p_key in p_keys:
            ts = au_ts_lsl[p_key]
            i0 = int(np.searchsorted(ts, t0_ep, side='left'))
            i1 = int(np.searchsorted(ts, t1_ep, side='right'))
            env_slices[p_key] = au_env[p_key][i0:i1]

        # Joint envelope slice (master grid)
        i0_master = int(np.searchsorted(ts_lsl_master, t0_ep, side='left'))
        i1_master = int(np.searchsorted(ts_lsl_master, t1_ep, side='right'))
        joint_ep = joint[i0_master:i1_master]

        # ── Feature blocks ──
        f3a = feat_coinc.compute_coincidence_features(ev_in_ep, config=config)
        # 3b/3c require both roles to have non-trivial trajectories
        if min(p1_au_ep.shape[0], p2_au_ep.shape[0]) >= 4:
            f3b = feat_dtw.compute_dtw_features(p1_au_ep, p2_au_ep, fs=fs,
                                                  config=config)
            # Per-episode CCA only if episode is long enough; otherwise use
            # the pooled-bin features assigned in the pre-pass (if any).
            if p1_au_ep.shape[0] >= config.cca_min_per_episode_samples:
                f3c = feat_cca.compute_cca_features(p1_au_ep, p2_au_ep, fs=fs,
                                                      config=config)
            elif k in cca_features_by_episode:
                f3c = cca_features_by_episode[k]
            else:
                f3c = feat_cca._empty_features()
        else:
            f3b = {n: np.nan for n in feat_dtw.feature_names()}
            f3b['_3b_valid'] = False
            f3c = feat_cca._empty_features()

        if coh_arr is not None:
            f3d = feat_coh.compute_episode_features_from_arrays(
                coh_arr, phase_arr, coh_freqs,
                i0_master, i1_master - 1, config=config)
        else:
            f3d = {n: np.nan for n in feat_coh.feature_names()}
            f3d['_3d_valid'] = False

        f_id = _identity_features(p1_au_ep, p2_au_ep, role_to_p)
        f_dyn = _dynamics_features(ev_in_ep, env_slices['p1'], env_slices['p2'],
                                     ep_durs[k], role_to_p)
        f_int = _intensity_features(joint_ep, ev_in_ep['amp'])
        f_ctx = _context_features(t0_ep, ep_starts_lsl)

        all_feats = {**f3a, **f3b, **f3c, **f3d, **f_id, **f_dyn, **f_int, **f_ctx}
        return k, all_feats

    n_jobs = config.n_jobs_per_session if config.n_jobs_per_session > 0 \
              else None  # joblib uses CPU count when None for prefer='threads'
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process_one)(k) for k in range(n_eps))

    # Build matrix
    X = np.full((n_eps, len(feat_names)), np.nan, dtype=np.float32)
    valid_3a = np.zeros(n_eps, dtype=bool)
    valid_3b = np.zeros(n_eps, dtype=bool)
    valid_3c = np.zeros(n_eps, dtype=bool)
    valid_3d = np.zeros(n_eps, dtype=bool)
    pooled_flag = np.zeros(n_eps, dtype=bool)
    name_to_col = {n: i for i, n in enumerate(feat_names)}
    for k, feats in results:
        for n, v in feats.items():
            if n in name_to_col:
                try:
                    X[k, name_to_col[n]] = float(v)
                except (TypeError, ValueError):
                    X[k, name_to_col[n]] = np.nan
        valid_3a[k] = bool(feats.get('_3a_valid', False))
        valid_3b[k] = bool(feats.get('_3b_valid', False))
        valid_3c[k] = bool(feats.get('_3c_valid', False))
        valid_3d[k] = bool(feats.get('_3d_valid', False))
        pooled_flag[k] = bool(feats.get('cca_pooled_flag', False))

    # Compose cache record
    rec = {
        'features':       X,
        'episode_id':     np.asarray(ep_npz['episode_id']),
        't_start_lsl':    ep_starts_lsl.astype(np.float64),
        't_end_lsl':      ep_ends_lsl.astype(np.float64),
        'condition':      ep_cond,
        'duration_s':     ep_durs.astype(np.float32),
        'peak_env':       ep_peak_env.astype(np.float32),
        'mean_env':       ep_mean_env.astype(np.float32),
        'n_events_total': (np.asarray(ep_npz['n_events_therapist']) +
                            np.asarray(ep_npz['n_events_patient'])).astype(np.int16),
        'cca_pooled_flag': pooled_flag,
        '_3a_valid':       valid_3a,
        '_3b_valid':       valid_3b,
        '_3c_valid':       valid_3c,
        '_3d_valid':       valid_3d,
    }

    elapsed = time.perf_counter() - t0
    sidecar_extra = {
        'n_episodes_in_features': n_eps,
        'n_features':             len(feat_names),
        'feature_names':          feat_names,
        'pct_3a_valid':           round(100.0 * valid_3a.mean(), 2),
        'pct_3b_valid':           round(100.0 * valid_3b.mean(), 2),
        'pct_3c_valid':           round(100.0 * valid_3c.mean(), 2),
        'pct_3d_valid':           round(100.0 * valid_3d.mean(), 2),
        'pct_cca_pooled':         round(100.0 * pooled_flag.mean(), 2),
        'wall_seconds':           round(elapsed, 3),
    }
    write_stage(sid, 3, rec, sidecar_extra, config_hash)
    return {**rec, '_meta': sidecar_extra, '_from_cache': False}


# ── Cohort fan-in ─────────────────────────────────────────────────────

def pool_cohort_features(session_ids: list[str],
                           config: SynchronyConfig = DEFAULT_CONFIG,
                           condition_filter: list[str] | None = None) -> dict:
    """Stack per-session caches; impute; standardize; write cohort_features.npz.

    Args:
        session_ids: list of canonical session ids to include.
        config: synchrony config.
        condition_filter: if set, only keep episodes whose ``condition`` is in
            this list. Useful for running a parallel "conversation-only"
            cohort to drop noise from baseline / meditation periods.
    """
    rows = []
    sids = []
    eids = []
    starts = []
    ends = []
    conds = []
    pooled = []
    valid_blocks = {b: [] for b in ('3a', '3b', '3c', '3d')}
    durations = []
    peak_envs = []
    mean_envs = []
    n_events_totals = []

    feat_names = all_feature_names()
    n_kept_sessions = 0
    n_filtered_total = 0
    for sid in session_ids:
        try:
            npz, meta = read_stage(sid, 3)
        except (OSError, FileNotFoundError):
            continue
        if meta.get('n_episodes_in_features', 0) == 0:
            continue
        sess_X = np.asarray(npz['features'])
        sess_cond = np.asarray(npz['condition'])
        if condition_filter is not None:
            keep = np.array([str(c) in condition_filter for c in sess_cond])
            if not keep.any():
                continue
            n_filtered_total += int((~keep).sum())
            sess_X = sess_X[keep]
            row_keep = keep
        else:
            row_keep = np.ones(len(sess_X), dtype=bool)
        rows.append(sess_X)
        n = sess_X.shape[0]
        sids.extend([sid] * n)
        eids.extend(np.asarray(npz['episode_id'])[row_keep].tolist())
        starts.extend(np.asarray(npz['t_start_lsl'])[row_keep].tolist())
        if 't_end_lsl' in npz.files if hasattr(npz, 'files') else 't_end_lsl' in npz:
            ends.extend(np.asarray(npz['t_end_lsl'])[row_keep].tolist())
        else:
            ends.extend((np.asarray(npz['t_start_lsl'])[row_keep] +
                          np.asarray(npz['duration_s'])[row_keep]).tolist())
        conds.extend(sess_cond[row_keep].tolist())
        pooled.extend(np.asarray(npz['cca_pooled_flag'])[row_keep].tolist())
        for b in valid_blocks:
            valid_blocks[b].extend(np.asarray(npz[f'_{b}_valid'])[row_keep].tolist())
        durations.extend(np.asarray(npz['duration_s'])[row_keep].tolist())
        peak_envs.extend(np.asarray(npz['peak_env'])[row_keep].tolist())
        mean_envs.extend(np.asarray(npz['mean_env'])[row_keep].tolist())
        n_events_totals.extend(np.asarray(npz['n_events_total'])[row_keep].tolist())
        n_kept_sessions += 1

    if not rows:
        raise RuntimeError('No sessions had non-empty Stage 4 caches '
                            '(after condition_filter, if any)')

    X_raw = np.vstack(rows)
    classes = [c for n, c in classify_all(feat_names).items()]
    X_imp, medians, imputed_mask = impute(X_raw, classes)
    X_z, means, stds = standardize(X_imp, classes)

    # Per-episode imputation fraction
    imp_frac = imputed_mask.mean(axis=1)
    low_quality = imp_frac > config.low_quality_imputation_threshold

    out = {
        'features':            X_z,
        'features_raw':        X_raw,
        'imputed_mask':        imputed_mask,
        'low_quality':         low_quality,
        'session_id':          np.array(sids, dtype=object),
        'episode_id':          np.array(eids, dtype=np.int32),
        't_start_lsl':         np.array(starts, dtype=np.float64),
        't_end_lsl':           np.array(ends, dtype=np.float64),
        'condition':           np.array(conds, dtype=object),
        'cca_pooled_flag':     np.array(pooled, dtype=bool),
        'duration_s':          np.array(durations, dtype=np.float32),
        'peak_env':            np.array(peak_envs, dtype=np.float32),
        'mean_env':            np.array(mean_envs, dtype=np.float32),
        'n_events_total':      np.array(n_events_totals, dtype=np.int16),
    }
    for b, v in valid_blocks.items():
        out[f'_{b}_valid'] = np.array(v, dtype=bool)

    # Save NPZ + JSON sidecar
    cdir = cohort_dir(ensure=True)
    np.savez(cdir / 'cohort_features.npz', **out, medians=medians,
             means=means, stds=stds,
             feature_classes=np.array(classes, dtype=object))
    sidecar = {
        'n_sessions':    n_kept_sessions,
        'n_episodes':    int(X_raw.shape[0]),
        'n_low_quality': int(low_quality.sum()),
        'n_features':    len(feat_names),
        'feature_names': feat_names,
        'feature_classes': classes,
        'mean_imputation_rate': float(imp_frac.mean()),
        'top_imputed_features': [
            (n, float(imputed_mask[:, i].mean()))
            for i, n in sorted(enumerate(feat_names),
                                key=lambda p: -imputed_mask[:, p[0]].mean())[:10]
        ],
        'condition_filter': condition_filter,
        'n_filtered_out':  n_filtered_total,
        'config_hash':       config.hash(),
    }
    (cdir / 'cohort_features.json').write_text(json.dumps(sidecar, indent=2))

    return {**out, '_meta': sidecar}
