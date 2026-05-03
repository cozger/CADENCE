"""Stage 1 — multiscale event detection per AU per participant.

Augments ``cadence.significance.face_event_coincidence.detect_au_events``
with:

  * GPU-batched derivative-of-Gaussian pyramid (all 52 AUs × 2 participants
    × n_scales = 624 channels in one ``torch.conv1d`` launch, vs scipy's
    sequential per-channel calls).
  * Hölder regularity α (slope of log|amp| vs log σ along the chain).
  * Optional baseline-derived per-channel noise floor estimated from
    ``base_EO`` segments (when present in the digest markers). Falls back
    to MAD-of-finest-scale-response on the full session when no baseline
    is available.

The MVP channel ``compute_bl_event_coincidence`` is unaffected — it works
on the activity envelope, not per-AU events.
"""
from __future__ import annotations

# Windows torch+numpy DLL ordering: hoist torch via _gpu before any numpy.
from cadence.synchrony import _gpu as _gpu  # noqa: F401

import json
import time
from typing import Optional

import numpy as np

from cadence.significance.face_event_coincidence import (
    chain_link_from_responses,
    _smooth_savgol,
    _baseline_subtract,
)
from cadence.synchrony.config import SynchronyConfig, DEFAULT_CONFIG
from cadence.synchrony.io import (
    is_stage_fresh, write_stage, load_face_npz, load_digest, session_dir,
)


N_AUS = 52
ROLE_NAMES = ('therapist', 'patient')


def _resolve_role_to_p(face_npz: dict, digest: dict) -> dict:
    """Return {'therapist': 'p1', 'patient': 'p2'} or swapped.

    Uses ``digest['roles']`` from the canonical role resolver (Layer 1
    ingest). Format: ``{'p1_role': 'therapist', 'p2_role': 'patient', ...}``
    or vice-versa. Raises if roles are unresolved (we never label
    P1/P2 — see ``feedback_roles.md``).
    """
    roles = digest.get('roles')
    if not roles:
        raise ValueError(f'digest is missing roles dict — '
                         f'session {digest.get("session_id")} not resolved')
    p1_role = roles.get('p1_role', '').lower()
    p2_role = roles.get('p2_role', '').lower()
    if {p1_role, p2_role} != {'therapist', 'patient'}:
        raise ValueError(f'roles must be {{therapist, patient}}; got '
                         f'p1={p1_role!r}, p2={p2_role!r}')
    return {p1_role: 'p1', p2_role: 'p2'}


def _interp_invalid(au_data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linear-interpolate invalid frames per AU column. Returns a copy."""
    out = au_data.astype(np.float64, copy=True)
    if valid_mask.all():
        return out
    inv_idx = np.where(~valid_mask)[0]
    val_idx = np.where(valid_mask)[0]
    if val_idx.size < 3:
        return out
    for c in range(out.shape[1]):
        col_valid = valid_mask & np.isfinite(out[:, c])
        if col_valid.sum() < 3:
            continue
        out[~col_valid, c] = np.interp(np.where(~col_valid)[0],
                                        np.where(col_valid)[0],
                                        out[col_valid, c])
    return out


def _baseline_eo_mask(ts: np.ndarray, lsl_offset: float, markers: list[tuple]) -> np.ndarray:
    """Return per-frame mask for frames inside any base_EO_start/_stop interval.

    ``ts`` is in stream-relative seconds; markers are ``[(t_lsl, label), ...]``
    in absolute LSL seconds.
    """
    mask = np.zeros(len(ts), dtype=bool)
    starts = {}
    intervals = []
    for t, lbl in markers:
        if lbl == 'base_EO_start':
            starts['base_EO'] = t
        elif lbl == 'base_EO_stop' and 'base_EO' in starts:
            intervals.append((starts.pop('base_EO'), t))
    if not intervals:
        return mask
    ts_lsl = ts + lsl_offset
    for t0, t1 in intervals:
        mask |= (ts_lsl >= t0) & (ts_lsl <= t1)
    return mask


def detect_session_events(sid: str, config: SynchronyConfig = DEFAULT_CONFIG,
                            force: bool = False) -> dict:
    """Run Stage 1 on one session; cache to ``01_events.npz``.

    Returns a dict with everything written to the cache. Skips computation
    when an up-to-date cache exists (override with ``force=True``).
    """
    config_hash = config.hash()
    if not force and is_stage_fresh(sid, 1, config_hash):
        from cadence.synchrony.io import read_stage
        npz, meta = read_stage(sid, 1)
        return {**npz, '_meta': meta, '_from_cache': True}

    t0 = time.perf_counter()
    face = load_face_npz(sid)
    digest = load_digest(sid)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    markers = digest.get('markers', [])

    role_to_p = _resolve_role_to_p(face, digest)

    fs = config.fs_native
    scales_seconds = list(config.event_scales_seconds)
    sigmas_frames = [max(1.0, s * fs) for s in scales_seconds]

    # Build (n_role × n_au, T_min) input matrix for batched GPU pyramid.
    # Both participants' AU streams are sampled at ~30 fps but T may differ
    # by a few frames. We process each participant independently; the GPU
    # batch is over (n_au × n_scales) per participant.
    out_au_idx = []
    out_role = []
    out_t_lsl = []
    out_amp = []
    out_alpha = []
    out_chain_len = []
    per_role_meta = {}

    for role in ROLE_NAMES:
        p_key = role_to_p.get(role)
        if p_key not in ('p1', 'p2'):
            per_role_meta[role] = {'status': 'unresolved'}
            continue

        au_key = f'{p_key}_au52'
        ts_key = f'{p_key}_au52_ts'
        valid_key = f'{p_key}_au_valid'

        if au_key not in face or ts_key not in face:
            per_role_meta[role] = {'status': f'missing keys ({au_key}/{ts_key})'}
            continue

        au_data = np.asarray(face[au_key], dtype=np.float64)  # (T, 52)
        ts = np.asarray(face[ts_key], dtype=np.float64)        # (T,)
        valid = (np.asarray(face[valid_key], dtype=bool) if valid_key in face
                 else np.ones(au_data.shape[0], dtype=bool))
        if au_data.ndim != 2 or au_data.shape[1] != N_AUS:
            per_role_meta[role] = {'status': f'unexpected AU shape {au_data.shape}'}
            continue

        # Stage 0 — interp invalid + smooth + baseline-subtract.
        au_interp = _interp_invalid(au_data, valid)
        au_smoothed = _smooth_savgol(au_interp, window=config.smooth_window,
                                       poly=config.smooth_poly)
        au_baselined = _baseline_subtract(au_smoothed, fs,
                                            window_s=config.baseline_window_s,
                                            q=config.baseline_q)

        # Per-channel baseline-noise floor estimation (optional).
        # Computed from base_EO frames only when present; else MAD fallback
        # is applied per-channel inside chain_link_from_responses.
        eo_mask = _baseline_eo_mask(ts, lsl_offset, markers)
        n_eo = int(eo_mask.sum())
        use_baseline_floor = (config.use_baseline_noise_floor and
                                n_eo >= int(2.0 * fs))

        # Stage 1 — batched DoG pyramid (all 52 AUs at once).
        # Input shape: (n_au, T)
        signal_2d = au_baselined.T  # (52, T)
        responses = _gpu.batched_dog_pyramid(signal_2d, scales_seconds, fs,
                                              device='auto')
        # responses: (n_scales, n_au, T)

        per_au_floors = []
        per_au_n_events = []
        for au_idx in range(N_AUS):
            resp_au = responses[:, au_idx, :]  # (n_scales, T)
            if use_baseline_floor:
                # MAD of finest-scale response within base_EO frames.
                finest_eo = np.abs(resp_au[0, eo_mask])
                if finest_eo.size >= 3:
                    mad = 1.4826 * np.median(np.abs(finest_eo - np.median(finest_eo)))
                    floor = max(config.event_noise_thresh_factor * mad, 1e-6)
                else:
                    floor = None  # fallback to MAD-on-full inside helper
            else:
                floor = None
            per_au_floors.append(floor if floor is not None else float('nan'))

            ev_t_rel, ev_amp, ev_alpha, ev_chain_len = chain_link_from_responses(
                resp_au, sigmas_frames, fs,
                noise_thresh_factor=config.event_noise_thresh_factor,
                min_chain_frac=config.event_min_chain_frac,
                noise_floor_override=floor,
                return_extra=True,
            )
            n_au_ev = len(ev_t_rel)
            per_au_n_events.append(n_au_ev)
            if n_au_ev == 0:
                continue
            t_lsl = ts[0] + ev_t_rel + lsl_offset
            out_au_idx.extend([au_idx] * n_au_ev)
            out_role.extend([role] * n_au_ev)
            out_t_lsl.extend(t_lsl.tolist())
            out_amp.extend(ev_amp.tolist())
            out_alpha.extend(ev_alpha.tolist())
            out_chain_len.extend(ev_chain_len.tolist())

        per_role_meta[role] = {
            'status': 'ok',
            'p_stream': p_key,
            'n_frames': int(au_data.shape[0]),
            'n_valid_frames': int(valid.sum()),
            'n_base_EO_frames': n_eo,
            'used_baseline_noise_floor': use_baseline_floor,
            'per_au_n_events': [int(x) for x in per_au_n_events],
            'per_au_noise_floor': per_au_floors,
        }

    # Assemble arrays
    n_events = len(out_au_idx)
    arrays = {
        'au_idx':     np.asarray(out_au_idx, dtype=np.int16),
        'role':       np.asarray(out_role, dtype=object),  # 'therapist' / 'patient'
        't_lsl':      np.asarray(out_t_lsl, dtype=np.float64),
        'amp':        np.asarray(out_amp, dtype=np.float32),
        'alpha':      np.asarray(out_alpha, dtype=np.float32),
        'chain_len':  np.asarray(out_chain_len, dtype=np.int8),
    }
    # Sort by t_lsl ascending for downstream slicing convenience.
    if n_events:
        order = np.argsort(arrays['t_lsl'])
        for k in arrays:
            arrays[k] = arrays[k][order]

    elapsed = time.perf_counter() - t0
    sidecar_extra = {
        'n_events': n_events,
        'role_resolution': role_to_p,
        'per_role': per_role_meta,
        'fs_native': fs,
        'scales_seconds': scales_seconds,
        'used_gpu': _gpu.has_cuda() and config.use_gpu_for_events,
        'wall_seconds': round(elapsed, 3),
    }
    write_stage(sid, 1, arrays, sidecar_extra, config_hash)
    return {**arrays, '_meta': sidecar_extra, '_from_cache': False}
