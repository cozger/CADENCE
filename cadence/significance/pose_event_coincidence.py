"""Pose event coincidence — landing / movement-peak dyadic coupling channel.

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md (Task 4).

The production behavioral (BL) channel is *event coincidence* of
facial-activity peaks (``face_event_coincidence.compute_bl_event_coincidence``).
This module is the pose analogue on the whole-body angular-speed envelope
from ``pose_angles``: do both bodies settle into stillness together (a
**landing** — dance_sync's "pictures"), or start moving together (a
**movement peak**)? Both event kinds reuse the face channel's surrogate
machinery verbatim, so the output is directly comparable to
``bl_event_coincidence`` and interpretable in the same units.

Pipeline per session:

  1. Per participant, ``pose33_to_angle_stream`` on the raw (N, 33, 4)
     pose -> weighted angular-speed envelope (deg/s). Frames that are
     invalid in the preproc artifact (``p{1,2}_pose_features_valid``) or in
     the angle stream (torso frame undefined) are set to NaN.
  2. Events on the envelope:
       ``landing``: local minima (``find_peaks`` on the negated envelope,
                    min gap 0.5 s, prominence = 12 % of the p5-p95 range).
                    NaN gaps are filled with the running max before
                    negation so a tracking dropout is never a "rest".
       ``peak``:    local maxima above the per-session 70th percentile,
                    min separation 1 s (mirror of
                    ``face_event_coincidence.detect_activity_peaks``).
  3. Event times -> 2 Hz binary grid on ``t_common`` (``peaks_to_grid``).
  4. Coincidence z against 200 circular-shift surrogates of P2's grid
     (``_coincidence_z``, +/- ``tau_samples`` bins, default +/-500 ms).
  5. sigma = 15 s Gaussian smoothing + per-session standardisation, exactly
     as the face channel (the raw per-bin z trace is sparse; smoothing turns
     it into a "recent coupling intensity" envelope the rSLDS can use).

Output contract: ``z`` float32 at the ``t_common`` rate, ``info`` dict
mirroring the face channel plus event statistics and a nearest-neighbour
signed lag. Sign convention (matches ``cadence/synchrony/features``):
**positive lag = P2 event later than P1**. Callers map P1/P2 to
therapist/patient via the digest roles; never label outputs by P1/P2.

Time bases: event times are reported in the pose stream's own timestamp
clock (``p{1,2}_pose33_ts``, the same clock as ``pose_ddtw`` stride
timestamps); ``lsl_offset`` is added only when binning onto ``t_common``,
exactly as the face channel treats ``p{1,2}_au52_ts``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from cadence.significance.face_event_coincidence import _coincidence_z, peaks_to_grid
from cadence.significance.pose_angles import pose33_to_angle_stream


# ── Constants ───────────────────────────────────────────────────────

EVENT_KINDS = ('landing', 'peak')

PARTICIPANTS = ('p1', 'p2')
POSE_NPZ_KEYS = ('pose33', 'pose33_ts', 'pose_features_valid')

LAG_MATCH_MAX_S = 1.0       # nearest-neighbour match window for mean_signed_lag_s
LAG_MIN_MATCHES = 5         # fewer matched events -> mean_signed_lag_s is NaN
MIN_ENVELOPE_SAMPLES = 3    # shorter envelopes yield no events

_SURR_STD_FLOOR = 1e-6      # per-session standardisation guard (matches face)


# ── Envelope helpers ────────────────────────────────────────────────

def _fill_nan_running_max(env: np.ndarray) -> np.ndarray:
    """Replace NaN samples with the running max of the finite samples before them.

    Leading NaN (no finite sample yet) are filled with the global finite
    max. The filled value is always >= the last finite sample, so after
    negation a gap can never become a local maximum (i.e. an envelope
    minimum). ``env`` must hold at least one finite value.
    """
    finite = np.isfinite(env)
    filled = np.where(finite, env, -np.inf)
    running = np.maximum.accumulate(filled)
    running = np.where(np.isfinite(running), running, np.nanmax(env))
    return np.where(finite, env, running)


def _neighbours_finite(finite: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """True for indices whose immediate left and right neighbours are finite."""
    if idx.size == 0:
        return np.zeros(0, dtype=bool)
    n = finite.size
    left = np.where(idx > 0, finite[np.maximum(idx - 1, 0)], False)
    right = np.where(idx < n - 1, finite[np.minimum(idx + 1, n - 1)], False)
    return left & right


# ── Event detection ─────────────────────────────────────────────────

def detect_landings(envelope: np.ndarray, fs: float, min_gap_s: float = 0.5,
                    prominence_frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Local minima of the angular-speed envelope = the body coming to rest.

    ``scipy.signal.find_peaks`` on ``-envelope`` with
    ``distance = round(min_gap_s * fs)`` and
    ``prominence = prominence_frac * (nanpercentile(env, 95) - nanpercentile(env, 5))``.
    NaN samples are filled with the running max before negation so a gap is
    never a minimum; minima whose immediate neighbours are NaN (the rim of a
    dropout) are dropped as well, because a rest that coincides with
    tracking loss is not measurable.

    Returns:
        idx:   (n_events,) int frame indices of the minima.
        depth: (n_events,) float prominence of each minimum in envelope
               units (deg/s) — how far the body slowed relative to the
               surrounding movement.
    """
    env = np.asarray(envelope, dtype=np.float64).squeeze()
    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64))
    if env.ndim != 1 or env.size < MIN_ENVELOPE_SAMPLES:
        return empty
    finite = np.isfinite(env)
    if finite.sum() < MIN_ENVELOPE_SAMPLES:
        return empty
    lo, hi = np.nanpercentile(env, [5.0, 95.0])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0.0:
        return empty
    distance = max(1, int(round(float(min_gap_s) * float(fs))))
    prominence = float(prominence_frac) * span
    filled = _fill_nan_running_max(env)
    idx, props = find_peaks(-filled, distance=distance, prominence=prominence)
    keep = _neighbours_finite(finite, idx)
    return idx[keep].astype(int), props['prominences'][keep].astype(np.float64)


def detect_movement_peaks(envelope: np.ndarray, fs: float,
                          quantile_threshold: float = 0.70,
                          min_sep_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``face_event_coincidence.detect_activity_peaks`` on the speed envelope.

    Peaks above the per-session ``quantile_threshold`` quantile of the
    finite envelope with at least ``min_sep_s`` separation. NaN samples are
    filled below the finite minimum so they are never peaks.

    Returns:
        idx: (n_peaks,) int frame indices.
        amp: (n_peaks,) float envelope value at each peak (deg/s).
    """
    env = np.asarray(envelope, dtype=np.float64).squeeze()
    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64))
    if env.ndim != 1 or env.size < MIN_ENVELOPE_SAMPLES:
        return empty
    finite = np.isfinite(env)
    if finite.sum() < MIN_ENVELOPE_SAMPLES:
        return empty
    height = float(np.nanquantile(env, quantile_threshold))
    distance = max(1, int(round(float(min_sep_s) * float(fs))))
    filled = np.where(finite, env, np.nanmin(env) - 1.0)
    idx, _ = find_peaks(filled, height=height, distance=distance)
    return idx.astype(int), env[idx].astype(np.float64)


# ── Per-participant events from a pose NPZ ──────────────────────────

def _participant_envelope(pose_npz_data, participant: str,
                          fs_hint: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(envelope with invalid frames -> NaN, timestamps, fs) for one participant."""
    pose33 = np.asarray(pose_npz_data[f'{participant}_pose33'])
    ts = np.asarray(pose_npz_data[f'{participant}_pose33_ts'], dtype=np.float64)
    valid_npz = np.asarray(pose_npz_data[f'{participant}_pose_features_valid']).astype(bool)
    stream = pose33_to_angle_stream(pose33, ts, fs_hint=fs_hint)
    env = stream['envelope'].astype(np.float64)
    n = env.shape[0]
    valid = stream['valid'].copy()
    if valid_npz.shape[0] == n:
        valid &= valid_npz
    env[~valid] = np.nan
    return env, ts, float(stream['fs'])


def _detect_events(env: np.ndarray, fs: float, event_kind: str, *,
                   min_gap_s: float, prominence_frac: float,
                   quantile_threshold: float, min_sep_s: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    if event_kind == 'landing':
        return detect_landings(env, fs, min_gap_s=min_gap_s,
                               prominence_frac=prominence_frac)
    if event_kind == 'peak':
        return detect_movement_peaks(env, fs, quantile_threshold=quantile_threshold,
                                     min_sep_s=min_sep_s)
    raise ValueError(f"event_kind must be one of {EVENT_KINDS}; got {event_kind!r}")


def events_from_pose_npz(pose_npz_data, participant: str, event_kind: str,
                         **kw) -> np.ndarray:
    """Event times for one participant — used by the validation driver.

    Times are in the pose stream's own timestamp clock
    (``{participant}_pose33_ts[idx]``, the same clock as ``pose_ddtw``
    stride timestamps), not shifted by any ``lsl_offset``.

    Args:
        pose_npz_data: dict-like with ``{p}_pose33``, ``{p}_pose33_ts``,
            ``{p}_pose_features_valid`` (raises ``KeyError`` when absent).
        participant: ``'p1'`` or ``'p2'``.
        event_kind: ``'landing'`` or ``'peak'``.
        **kw: ``fs_hint``, ``min_gap_s``, ``prominence_frac``,
            ``quantile_threshold``, ``min_sep_s`` — same defaults as
            ``compute_pose_event_coincidence``.

    Returns:
        (n_events,) float64 event times.
    """
    if participant not in PARTICIPANTS:
        raise ValueError(f'participant must be one of {PARTICIPANTS}; got {participant!r}')
    env, ts, fs = _participant_envelope(pose_npz_data, participant,
                                        fs_hint=kw.get('fs_hint', 30.0))
    idx, _ = _detect_events(env, fs, event_kind,
                            min_gap_s=kw.get('min_gap_s', 0.5),
                            prominence_frac=kw.get('prominence_frac', 0.12),
                            quantile_threshold=kw.get('quantile_threshold', 0.70),
                            min_sep_s=kw.get('min_sep_s', 1.0))
    return ts[idx].astype(np.float64)


# ── Lag statistics ──────────────────────────────────────────────────

def _nn_signed_lag(t1: np.ndarray, t2: np.ndarray,
                   max_lag_s: float = LAG_MATCH_MAX_S) -> np.ndarray:
    """Nearest-neighbour P2-minus-P1 lag for each P1 event, kept when |lag| <= max_lag_s.

    Positive = the nearest P2 event is later than the P1 event.
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.sort(np.asarray(t2, dtype=np.float64))
    if t1.size == 0 or t2.size == 0:
        return np.zeros(0, dtype=np.float64)
    right = np.searchsorted(t2, t1, side='left')
    left = np.clip(right - 1, 0, t2.size - 1)
    right = np.clip(right, 0, t2.size - 1)
    lag_l = t2[left] - t1
    lag_r = t2[right] - t1
    lag = np.where(np.abs(lag_r) < np.abs(lag_l), lag_r, lag_l)
    return lag[np.abs(lag) <= max_lag_s]


# ── Smoothing + standardisation (identical to the face channel) ─────

def _smooth_and_standardize(z: np.ndarray, t_common: np.ndarray,
                            smooth_sigma_s: float, info: dict) -> np.ndarray:
    """Gaussian-smooth the raw z trace and per-session standardise; fills ``info``."""
    if smooth_sigma_s and smooth_sigma_s > 0 and len(t_common) > 1:
        fs_out = 1.0 / float(np.median(np.diff(t_common)))
        sigma_samples = max(0.5, smooth_sigma_s * fs_out)
        z = gaussian_filter1d(z, sigma=sigma_samples).astype(np.float32)
        info['smooth_sigma_s'] = float(smooth_sigma_s)
        info['mean_z_smoothed'] = float(z.mean())
        info['std_z_smoothed'] = float(z.std())
        s = float(z.std())
        if s > _SURR_STD_FLOOR:
            z = ((z - z.mean()) / s).astype(np.float32)
        info['mean_z'] = float(z.mean())
        info['std_z'] = float(z.std())
    else:
        info['smooth_sigma_s'] = 0.0
        info['mean_z'] = info['mean_z_raw']
        info['std_z'] = info['std_z_raw']
    return z.astype(np.float32)


# ── End-to-end channel ──────────────────────────────────────────────

def compute_pose_event_coincidence(pose_npz_data, t_common, lsl_offset, *,
                                   event_kind: str = 'landing', fs_hint: float = 30.0,
                                   n_surrogates: int = 200, seed: int = 42,
                                   tau_samples: int = 1,
                                   min_gap_s: float = 0.5, prominence_frac: float = 0.12,
                                   quantile_threshold: float = 0.70, min_sep_s: float = 1.0,
                                   smooth_sigma_s: float = 15.0) -> tuple[np.ndarray, dict]:
    """End-to-end: pose NPZ -> per-participant events -> coincidence z trace.

    Args:
        pose_npz_data: dict-like (``np.load`` result) from
            ``data/preproc/pose/v1/<sid>.npz`` with keys ``p{1,2}_pose33``
            (N, 33, 4), ``p{1,2}_pose33_ts`` (N,), ``p{1,2}_pose_features_valid`` (N,).
        t_common: (N_out,) common-time grid (LSL seconds, 2 Hz in production).
        lsl_offset: float added to pose timestamps to align with ``t_common``.
        event_kind: ``'landing'`` (speed minima) or ``'peak'`` (speed maxima).
        fs_hint: fallback native rate when it cannot be estimated from ``ts``.
        n_surrogates, seed, tau_samples: circular-shift surrogate settings
            (``tau_samples=1`` = +/-500 ms at 2 Hz).
        min_gap_s, prominence_frac: ``detect_landings`` settings.
        quantile_threshold, min_sep_s: ``detect_movement_peaks`` settings.
        smooth_sigma_s: Gaussian sigma (s) for the post-z smoothing; 0 disables
            smoothing and standardisation (raw per-bin z is returned).

    Returns:
        z: (N_out,) float32 coincidence z at the ``t_common`` rate.
        info: dict mirroring the face channel (``p{1,2}_total_peaks``,
            ``p{1,2}_grid_density``, ``mean_z_raw``, ``std_z_raw``,
            ``mean_raw_coinc``, ``smooth_sigma_s``, ``mean_z_smoothed``,
            ``std_z_smoothed``, ``mean_z``, ``std_z``, ``status``) plus
            ``event_kind``, ``p{1,2}_n_events``, ``p{1,2}_event_rate_hz``
            (events per second of valid pose), ``p{1,2}_fs``,
            ``p{1,2}_valid_frac``, ``mean_signed_lag_s`` (nearest-neighbour
            P2-minus-P1 lag within +/-1 s over matched events, NaN when
            fewer than 5 matches), ``median_signed_lag_s``, ``n_matched_events``,
            and ``mean_z_raw_event_bins`` (mean raw z over bins holding a P2
            event — the bins where coincidence is actually tested; NaN if none).
            On a missing NPZ key returns zeros and ``{'status': 'missing_data',
            'missing': <key>}`` — the same contract as the face channel.
    """
    if event_kind not in EVENT_KINDS:
        raise ValueError(f"event_kind must be one of {EVENT_KINDS}; got {event_kind!r}")
    t_common = np.asarray(t_common, dtype=np.float64)
    n_out = len(t_common)
    for p in PARTICIPANTS:
        for suffix in POSE_NPZ_KEYS:
            k = f'{p}_{suffix}'
            if k not in pose_npz_data:
                return np.zeros(n_out, dtype=np.float32), {
                    'status': 'missing_data', 'missing': k,
                }

    info: dict = {'event_kind': event_kind}
    event_times: dict[str, np.ndarray] = {}
    for p in PARTICIPANTS:
        env, ts, fs = _participant_envelope(pose_npz_data, p, fs_hint=fs_hint)
        idx, _ = _detect_events(env, fs, event_kind,
                                min_gap_s=min_gap_s, prominence_frac=prominence_frac,
                                quantile_threshold=quantile_threshold, min_sep_s=min_sep_s)
        times = ts[idx].astype(np.float64)
        event_times[p] = times
        n_valid = int(np.isfinite(env).sum())
        valid_dur = n_valid / fs if fs > 0 else 0.0
        if valid_dur <= 0.0 and ts.size > 1:
            valid_dur = float(ts[-1] - ts[0])
        info[f'{p}_n_events'] = int(idx.size)
        info[f'{p}_total_peaks'] = int(idx.size)
        info[f'{p}_event_rate_hz'] = float(idx.size / valid_dur) if valid_dur > 0 else 0.0
        info[f'{p}_fs'] = float(fs)
        info[f'{p}_valid_frac'] = float(n_valid / env.size) if env.size else 0.0

    # Nearest-neighbour signed lag (P2 minus P1), same clock for both streams
    lags = _nn_signed_lag(event_times['p1'], event_times['p2'], LAG_MATCH_MAX_S)
    info['n_matched_events'] = int(lags.size)
    if lags.size >= LAG_MIN_MATCHES:
        info['mean_signed_lag_s'] = float(lags.mean())
        info['median_signed_lag_s'] = float(np.median(lags))
    else:
        info['mean_signed_lag_s'] = float('nan')
        info['median_signed_lag_s'] = float('nan')

    # 2 Hz grids + surrogate z
    p1_grid = peaks_to_grid(event_times['p1'], t_common, lsl_offset)
    p2_grid = peaks_to_grid(event_times['p2'], t_common, lsl_offset)
    info['p1_grid_density'] = float((p1_grid > 0).mean()) if n_out else 0.0
    info['p2_grid_density'] = float((p2_grid > 0).mean()) if n_out else 0.0

    z, raw = _coincidence_z(p1_grid, p2_grid, tau_samples=tau_samples,
                            n_surrogates=n_surrogates, seed=seed)
    z = np.asarray(z, dtype=np.float64)
    info['mean_z_raw'] = float(z.mean()) if n_out else 0.0
    info['std_z_raw'] = float(z.std()) if n_out else 0.0
    info['mean_raw_coinc'] = float(np.mean(raw)) if n_out else 0.0
    p2_bins = p2_grid > 0
    info['mean_z_raw_event_bins'] = float(z[p2_bins].mean()) if p2_bins.any() else float('nan')

    z = _smooth_and_standardize(z, t_common, smooth_sigma_s, info)
    info['status'] = 'ok'
    return z.astype(np.float32), info
