"""Stage 2 — joint activation episode segmentation.

Joint envelope = max(p_therapist_au_activity, p_patient_au_activity) at
30 fps (the precomputed activity envelope from face/v1 preproc — same
signal that drives the MVP ``bl_event_coincidence`` channel).

Threshold ``T*`` is **cohort-anchored**: the 90th percentile of the joint
envelope pooled across all canonical sessions' base_EO segments. This makes
"above-resting-face" mean the same physical level for every dyad — quiet
dyads correctly yield few episodes, expressive dyads yield many. Episode
count is itself a first-class expressivity signal.

For sessions without a base_EO marker (rare), fall back to the cohort
median ``T*``. Sessions where the joint envelope is entirely flat (no
above-threshold frames) still get an entry in the episode cache (with
``n_episodes=0``) — they are legitimate low-expressivity sessions, not
"missing data."
"""
from __future__ import annotations

# Hoist torch via _gpu before numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np

from cadence.ingest.quality import list_canonical_sessions
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG
from cadence.synchrony.events import _resolve_role_to_p
from cadence.synchrony.io import (
    cohort_dir, digest_path, face_npz_path, is_stage_fresh,
    load_digest, load_face_npz, read_stage, session_dir, write_stage,
)


COHORT_THRESHOLD_FILENAME = 'cohort_threshold.json'


# ── Cohort threshold T* ────────────────────────────────────────────────

def _resampled_joint_envelope(face: dict, role_to_p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (joint_env, ts_lsl) at native 30 fps, aligned via interp.

    Uses the patient's timestamps as the master grid (arbitrary; both
    streams are ~30 fps). The other participant's envelope is interpolated
    onto the master grid so the per-frame max is well-defined.

    Returned ts_lsl includes lsl_offset (absolute LSL).
    """
    p_pat = role_to_p['patient']
    p_the = role_to_p['therapist']
    pat_env = np.asarray(face[f'{p_pat}_au_activity'], dtype=np.float64).squeeze()
    pat_ts  = np.asarray(face[f'{p_pat}_au52_ts'], dtype=np.float64)
    the_env = np.asarray(face[f'{p_the}_au_activity'], dtype=np.float64).squeeze()
    the_ts  = np.asarray(face[f'{p_the}_au52_ts'], dtype=np.float64)
    # Interp therapist envelope onto patient timestamps
    the_on_pat = np.interp(pat_ts, the_ts, the_env, left=0.0, right=0.0)
    joint = np.maximum(pat_env, the_on_pat)
    return joint, pat_ts


def _baseline_eo_intervals_lsl(digest: dict) -> list[tuple[float, float]]:
    starts, intervals = {}, []
    for t, lbl in digest.get('markers', []):
        if lbl == 'base_EO_start':
            starts['base_EO'] = t
        elif lbl == 'base_EO_stop' and 'base_EO' in starts:
            intervals.append((starts.pop('base_EO'), t))
    return intervals


def _has_both_face_streams(face: dict, role_to_p: dict) -> bool:
    """Return True iff both participants' AU activity envelopes are present."""
    needed = [f"{role_to_p['therapist']}_au_activity",
              f"{role_to_p['patient']}_au_activity",
              f"{role_to_p['therapist']}_au52_ts",
              f"{role_to_p['patient']}_au52_ts"]
    return all(k in face for k in needed)


def _per_session_joint_eo_pool(sid: str) -> np.ndarray | None:
    """Return joint envelope samples within base_EO intervals, or None.

    Returns None for any reason the session cannot contribute to the pool
    (no face npz, unresolved roles, missing one face stream, no base_EO).
    """
    if not face_npz_path(sid).exists() or not digest_path(sid).exists():
        return None
    face = load_face_npz(sid)
    digest = load_digest(sid)
    try:
        role_to_p = _resolve_role_to_p(face, digest)
    except ValueError:
        return None
    if not _has_both_face_streams(face, role_to_p):
        return None
    eo_lsl = _baseline_eo_intervals_lsl(digest)
    if not eo_lsl:
        return None
    joint, ts_rel = _resampled_joint_envelope(face, role_to_p)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    ts_lsl = ts_rel + lsl_offset
    pool = []
    for t0, t1 in eo_lsl:
        m = (ts_lsl >= t0) & (ts_lsl <= t1)
        if m.any():
            pool.append(joint[m])
    return np.concatenate(pool) if pool else None


def compute_cohort_threshold(config: SynchronyConfig = DEFAULT_CONFIG,
                              force: bool = False, sids: list[str] | None = None
                              ) -> dict:
    """Compute cohort-anchored T* from base_EO joint-envelope pool.

    Caches to ``results/synchrony/cohort/cohort_threshold.json``. Re-run
    when ``force=True`` or the config hash changes.

    Returns ``{'T_star': float, 'config_hash': str, 'n_sessions_pooled':
    int, 'n_samples_pooled': int, 'fallback_T_star': float, ...}``.
    """
    # T* is condition-agnostic (derived from base_EO across the canonical
    # cohort) — always live in the canonical 'cohort' namespace, even if the
    # active cohort name has been switched (e.g. 'cohort_conv_1_conv_2').
    out_path = cohort_dir(ensure=True, name='cohort') / COHORT_THRESHOLD_FILENAME
    config_hash = config.hash()
    if not force and out_path.exists():
        meta = json.loads(out_path.read_text())
        if meta.get('synchrony_config_hash') == config_hash:
            return meta

    if sids is None:
        sids = list_canonical_sessions()

    pools = {}
    skipped = []
    for sid in sids:
        pool = _per_session_joint_eo_pool(sid)
        if pool is None or pool.size < 30:   # require at least 1 s @ 30 fps
            skipped.append(sid)
            continue
        pools[sid] = pool

    if not pools:
        raise RuntimeError('Cohort base_EO pool is empty — cannot compute T*')

    big_pool = np.concatenate(list(pools.values()))
    T_star = float(np.quantile(big_pool, config.cohort_threshold_quantile))

    # Per-session T* for fallback diagnostics
    per_session_T_star = {sid: float(np.quantile(p, config.cohort_threshold_quantile))
                            for sid, p in pools.items()}
    fallback_T_star = float(np.median(list(per_session_T_star.values())))

    meta = {
        'synchrony_config_hash': config_hash,
        'cohort_threshold_quantile': config.cohort_threshold_quantile,
        'T_star': T_star,
        'fallback_T_star': fallback_T_star,
        'n_sessions_pooled': len(pools),
        'n_samples_pooled': int(big_pool.size),
        'sessions_pooled': sorted(pools.keys()),
        'sessions_skipped_no_baseEO': skipped,
        'per_session_T_star': per_session_T_star,
    }
    out_path.write_text(json.dumps(meta, indent=2))
    return meta


# ── Episode segmentation ───────────────────────────────────────────────

def _segment_intervals(envelope: np.ndarray, threshold: float, fs: float,
                        merge_gap_s: float, min_dur_s: float, max_dur_s: float
                        ) -> list[tuple[int, int]]:
    """Above-threshold contiguous intervals → list of (i_start, i_end_inclusive)."""
    above = envelope > threshold
    if not above.any():
        return []
    # Find runs
    diff = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    intervals = list(zip(starts.tolist(), ends.tolist()))

    # Merge gaps shorter than merge_gap_s
    merge_gap_frames = max(1, int(round(merge_gap_s * fs)))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Drop too-short / too-long
    min_frames = max(1, int(round(min_dur_s * fs)))
    max_frames = int(round(max_dur_s * fs))
    kept = [(s, e) for (s, e) in merged
            if (e - s + 1) >= min_frames and (e - s + 1) <= max_frames]
    return kept


def _condition_for_t(t_lsl: float, periods: list[tuple]) -> str:
    for name, t0, t1 in periods:
        if t0 <= t_lsl <= t1:
            return name
    return ''


def segment_session_episodes(sid: str, config: SynchronyConfig = DEFAULT_CONFIG,
                              cohort_threshold: dict | None = None,
                              force: bool = False) -> dict:
    """Run Stage 2 on one session; cache to ``02_episodes.npz``.

    ``cohort_threshold`` may be passed in to avoid recomputation when
    fan-out runs. If None, ``compute_cohort_threshold`` is called.
    """
    config_hash = config.hash()
    if not force and is_stage_fresh(sid, 2, config_hash):
        npz, meta = read_stage(sid, 2)
        return {**npz, '_meta': meta, '_from_cache': True}

    t0 = time.perf_counter()
    if cohort_threshold is None:
        cohort_threshold = compute_cohort_threshold(config)

    face = load_face_npz(sid)
    digest = load_digest(sid)
    role_to_p = _resolve_role_to_p(face, digest)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    fs = config.fs_native

    if not _has_both_face_streams(face, role_to_p):
        # Single-participant face tracking — joint envelope cannot be built.
        # Write an empty episodes cache with status logged so cohort
        # fan-out can skip this session cleanly.
        rec_empty = {
            'episode_id':   np.zeros(0, dtype=np.int32),
            't_start_lsl':  np.zeros(0, dtype=np.float64),
            't_end_lsl':    np.zeros(0, dtype=np.float64),
            'duration_s':   np.zeros(0, dtype=np.float32),
            'peak_env':     np.zeros(0, dtype=np.float32),
            'mean_env':     np.zeros(0, dtype=np.float32),
            'n_events_therapist': np.zeros(0, dtype=np.int16),
            'n_events_patient':   np.zeros(0, dtype=np.int16),
            'condition':    np.array([], dtype=object),
            'member_event_offsets': np.zeros(1, dtype=np.int32),
            'member_event_indices': np.zeros(0, dtype=np.int64),
        }
        sidecar_extra = {
            'n_episodes': 0, 'n_zero_event_episodes': 0,
            'cohort_T_star': float(cohort_threshold['T_star']),
            'fallback_T_star': float(cohort_threshold['fallback_T_star']),
            'used_threshold': float('nan'),
            'session_had_base_EO': bool(_baseline_eo_intervals_lsl(digest)),
            'merge_gap_s': config.episode_merge_gap_s,
            'min_duration_s': config.episode_min_duration_s,
            'max_duration_s': config.episode_max_duration_s,
            'fraction_active': 0.0,
            'session_duration_s': 0.0,
            'episode_rate_per_min': 0.0,
            'wall_seconds': round(time.perf_counter() - t0, 3),
            'status': 'single_face_stream_only',
        }
        write_stage(sid, 2, rec_empty, sidecar_extra, config_hash)
        return {**rec_empty, '_meta': sidecar_extra, '_from_cache': False}

    joint, ts_rel = _resampled_joint_envelope(face, role_to_p)
    ts_lsl = ts_rel + lsl_offset

    # Per-session T* (used in the diagnostic figure but not for segmentation
    # unless the cohort T* is actually inappropriate for this session — which
    # can happen if the dyad is so quiet that no frames clear the cohort
    # threshold; we then log it).
    T_star = cohort_threshold['T_star']
    fallback_T = cohort_threshold['fallback_T_star']
    has_eo = bool(_baseline_eo_intervals_lsl(digest))
    used_threshold = T_star if has_eo else fallback_T

    intervals = _segment_intervals(
        joint, used_threshold, fs,
        merge_gap_s=config.episode_merge_gap_s,
        min_dur_s=config.episode_min_duration_s,
        max_dur_s=config.episode_max_duration_s,
    )

    # Load Stage 1 events for member linking
    ev_npz, _ = read_stage(sid, 1)
    ev_t_lsl = np.asarray(ev_npz['t_lsl'])

    periods = []
    starts = {}
    for t, lbl in digest.get('markers', []):
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                periods.append((n, starts.pop(n), t))
    periods.sort(key=lambda x: x[1])

    n_eps = len(intervals)
    rec = {
        'episode_id':   np.arange(n_eps, dtype=np.int32),
        't_start_lsl':  np.zeros(n_eps, dtype=np.float64),
        't_end_lsl':    np.zeros(n_eps, dtype=np.float64),
        'duration_s':   np.zeros(n_eps, dtype=np.float32),
        'peak_env':     np.zeros(n_eps, dtype=np.float32),
        'mean_env':     np.zeros(n_eps, dtype=np.float32),
        'n_events_therapist': np.zeros(n_eps, dtype=np.int16),
        'n_events_patient':   np.zeros(n_eps, dtype=np.int16),
        'condition':    np.array([''] * n_eps, dtype=object),
        # CSR-style member-event index pointers into 01_events.npz arrays:
        'member_event_offsets': np.zeros(n_eps + 1, dtype=np.int32),
    }
    member_event_indices = []
    ev_role = np.asarray(ev_npz['role'])
    for k, (s, e) in enumerate(intervals):
        t0_ep = ts_lsl[s]
        t1_ep = ts_lsl[e]
        rec['t_start_lsl'][k] = t0_ep
        rec['t_end_lsl'][k]   = t1_ep
        rec['duration_s'][k]  = float(t1_ep - t0_ep)
        env_slice = joint[s:e + 1]
        rec['peak_env'][k]    = float(env_slice.max())
        rec['mean_env'][k]    = float(env_slice.mean())
        # Member events
        member = np.where((ev_t_lsl >= t0_ep) & (ev_t_lsl <= t1_ep))[0]
        member_event_indices.append(member)
        rec['n_events_therapist'][k] = int((ev_role[member] == 'therapist').sum())
        rec['n_events_patient'][k]   = int((ev_role[member] == 'patient').sum())
        rec['condition'][k]   = _condition_for_t((t0_ep + t1_ep) / 2, periods)
        rec['member_event_offsets'][k + 1] = (rec['member_event_offsets'][k]
                                                 + member.size)
    rec['member_event_indices'] = (np.concatenate(member_event_indices)
                                    if member_event_indices
                                    else np.zeros(0, dtype=np.int64))

    # Drop zero-event episodes from the cluster pool but keep them in
    # the descriptive cache.
    n_events_total = (rec['n_events_therapist'] + rec['n_events_patient'])
    n_zero_event = int((n_events_total == 0).sum())

    # Counts used by the expressivity profile:
    total_active_frames = int((joint > used_threshold).sum())
    total_frames = len(joint)
    fraction_active = total_active_frames / max(total_frames, 1)
    duration_session_s = float((ts_lsl[-1] - ts_lsl[0]) if total_frames > 1 else 0.0)
    episode_rate_overall = (n_eps - n_zero_event) / (duration_session_s / 60.0) \
                            if duration_session_s > 0 else 0.0

    elapsed = time.perf_counter() - t0
    sidecar_extra = {
        'n_episodes':               n_eps,
        'n_zero_event_episodes':    n_zero_event,
        'cohort_T_star':            float(T_star),
        'fallback_T_star':          float(fallback_T),
        'used_threshold':           float(used_threshold),
        'session_had_base_EO':      has_eo,
        'merge_gap_s':              config.episode_merge_gap_s,
        'min_duration_s':           config.episode_min_duration_s,
        'max_duration_s':           config.episode_max_duration_s,
        'fraction_active':          fraction_active,
        'session_duration_s':       duration_session_s,
        'episode_rate_per_min':     float(episode_rate_overall),
        'wall_seconds':             round(elapsed, 3),
    }
    write_stage(sid, 2, rec, sidecar_extra, config_hash)
    return {**rec, '_meta': sidecar_extra, '_from_cache': False}
