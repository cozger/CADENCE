"""Phase 1 — pose-channel candidate validation driver.

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md (Task 5).

Runs every candidate pose-coupling channel through the Phase 0 validation
harness (``scripts/_validate_pose_ddtw.py``) so each candidate is judged by
exactly the gates that admitted the production DDTW channel:

  Test 1  Semi-synthetic dose response on pseudo-dyads
          (Kendall tau permutation p < 0.05 AND AUC >= 0.65 at kappa = 0.4)
  Test 2  Real-data condition contrast conv vs meditation
          (|dz| >= 0.5 AND paired t or Wilcoxon p < 0.05 — magnitude test,
          imported verbatim from Phase 0)
  Test 3  Pseudo-dyad null (real - pseudo >= 0.5 z). For the event modes the
          mean level is the **raw** circular-shift coincidence z
          (``info['mean_z_raw']``) because the production trace is
          per-session standardised to mean 0 (see ``_t3_level``)
  Test 4  Redundancy vs the Phase 0 DDTW channel and vs the V11 multi-lag
          baseline (descriptive: Pearson r with block-bootstrap 95 % CI,
          10 s blocks at 2 Hz)

Candidate modes (``--modes``):

  pca          Phase 0 DDTW on shared-PCA positions, re-run through this harness
  angles       DDTW on 12 torso-frame segment angles (noise-SD units)
  angle_speed  DDTW on angular speed (deg/s, noise-SD units)
  evt_landing  coincidence of angular-speed-envelope minima (bodies coming to rest)
  evt_peak     coincidence of angular-speed-envelope maxima (movement onsets)

Every mode emits **surrogate z at 2 Hz on stream-relative time** — the same
contract as ``results/mvp/phase0/pose_ddtw_per_session.npz`` — so
``scripts/_run_mvp_scaffold.py`` can slice any of them into the MVP ``pose``
slot: ``results/mvp/phase1_pose/pose_<mode>_per_session.npz`` with keys
``{sid}__z``, ``{sid}__stride_ts``, ``{sid}__z_pw`` (AR(1)-prewhitened +
standardised, validity = finite) and, for DDTW modes, ``{sid}__lag_s``,
``{sid}__lag_var``, ``{sid}__asym`` (warping-path timing, positive lag = P2
later than P1).

Shared objects across modes: each session is loaded once through the Phase 0
loader (99-D positions on the 12 Hz common grid) and, for angle / event
modes, the raw (N, 33, 4) pose is resampled onto the **same** grid
(``pose33_onto_grid``: x, y, z linear through the ``run_session_ddtw``
two-step path, visibility by the conservative bracketing-minimum rule of
``pose_ddtw.resample_pose33_visibility``). Pseudo-dyads and kappa-injection
therefore operate on identical objects for every mode: for angle modes the
injection mixes the **pose33 coordinates** (angles are recomputed from the
mixed skeleton); event modes see the same mixed skeleton. Because the cohort
mixes image-normalised MediaPipe skeletons with RTMW pixel skeletons, P1 is
first re-expressed in P2's body frame (robust mid-hip translation +
torso-length scaling, ``align_pose33_to_frame``) so that kappa blends
*shapes* in proportion rather than being dominated by whichever skeleton has
the larger coordinate magnitude; P2 stays in its native units. The 99-D
``p2_common`` path of Phase 0 is untouched.

Decision: the passing mode (Tests 1-3) with the largest Test 3 margin;
``pose_ddtw`` if only Phase 0 passed (its own report); else ``pose_baseline``.
The Phase 0 report is never touched — this driver writes only
``<out>/phase1_report.md`` plus per-mode NPZ / CSV files.

Usage:
  python scripts/_validate_pose_channels.py --all
  python scripts/_validate_pose_channels.py --sessions y_06 Y_55_04272026 --modes angles evt_landing
  python scripts/_validate_pose_channels.py --all --quick --n-jobs 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import multiprocessing as _mp
import sys
import time
import zlib
from pathlib import Path

try:
    import torch  # noqa: F401  -- must precede numpy on the Windows MCCT stack
except ImportError:  # remote containers without a torch build
    pass
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._validate_pose_ddtw import (
    _auc_from_scores, discover_sessions, inject_coupling_into_segment,
    load_and_resample, make_pseudo_dyad, piecewise_linear_warp,
    test2_condition_contrast,
)
from cadence.io.resources import limit_blas_threads, log_resources, pick_n_jobs
from cadence.significance.block_bootstrap import (
    block_bootstrap_mean, block_bootstrap_paired, block_len_from_seconds,
)
from cadence.significance.pose_ddtw import (
    DDTW_N_SURROGATES, DDTW_PCA_DIM, DDTW_TARGET_RATE_HZ, POSE_COLS,
    POSE_FLAT_DIM_VIS, POSE_KEYPOINTS, assign_frame_conditions,
    build_angle_features, compute_session_ddtw, fit_shared_pca,
    load_pose_streams, project_pca, resample_pose33_visibility,
    resample_to_uniform_rate,
)
from cadence.significance.pose_angles import (
    LM_L_HIP, LM_L_SHOULDER, LM_R_HIP, LM_R_SHOULDER, VISIBILITY_THRESHOLD,
)
from cadence.significance.pose_event_coincidence import compute_pose_event_coincidence

# Process pool context: spawn (see _validate_pose_ddtw for the torch DLL rationale).
_MP_SPAWN = _mp.get_context('spawn')


# ── Constants ───────────────────────────────────────────────────────

PREPROC_POSE_ROOT = REPO_ROOT / 'data' / 'preproc' / 'pose' / 'v1'
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'
V11_SCAFFOLD_ROOT = REPO_ROOT / 'results' / 'v11'
PHASE0_DIR = REPO_ROOT / 'results' / 'mvp' / 'phase0'
DEFAULT_OUT = REPO_ROOT / 'results' / 'mvp' / 'phase1_pose'
PHASE0_REPORT_NAME = 'phase0_report.md'      # never written by this driver
PHASE1_REPORT_NAME = 'phase1_report.md'

DDTW_MODES = ('pca', 'angles', 'angle_speed')
EVENT_MODES = ('evt_landing', 'evt_peak')
ALL_MODES = DDTW_MODES + EVENT_MODES
DEFAULT_MODES = ('angles', 'angle_speed', 'evt_landing', 'evt_peak')
EVENT_KIND = {'evt_landing': 'landing', 'evt_peak': 'peak'}

DECISION_BASELINE = 'pose_baseline'
DECISION_DDTW = 'pose_ddtw'
MODE_TO_DECISION = {
    'pca': DECISION_DDTW,
    'angles': 'pose_angles',
    'angle_speed': 'pose_angle_speed',
    'evt_landing': 'pose_evt_landing',
    'evt_peak': 'pose_evt_peak',
}

FS_OUT_HZ = 2.0                      # output rate of every channel (rSLDS rate)
BLOCK_S = 10.0                       # block-bootstrap block length (s)
N_BOOT = 2000                        # block-bootstrap resamples
KAPPA_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
INJECTION_SEGMENT_S = 60.0
QUICK_N_SURROGATES = 50
EVENT_SMOOTH_SIGMA_S = 15.0          # production smoothing of the event channels

T1_AUC_THRESH = 0.65
T1_P_THRESH = 0.05
T3_DELTA_THRESH = 0.5
# Test 3 mean-level source per mode family (see ``_t3_level``).
T3_LEVEL_SOURCE = {**{m: 'z' for m in DDTW_MODES}, **{m: 'mean_z_raw' for m in EVENT_MODES}}
_TORSO_REF_MIN = 1e-6                # shorter reference torso -> no frame alignment
T4_INDEPENDENT_R = 0.3
T4_REDUNDANT_R = 0.7

# Per-worker RAM (GB) estimates for pick_n_jobs (see _validate_pose_ddtw).
_RAM_PER_LOAD_GB = 0.6               # 99-D + (N, 33, 4) pose on the 12 Hz grid
_RAM_PER_PAIR_GB = 0.7

_PREWHITEN_FALLBACK = 'local_ar1_fallback'
_PREWHITEN_REFERENCE = 'scripts._run_rslds_scaffold_v8.prewhiten_and_standardize'
_PREWHITEN_RHO_THRESH = 0.3
_prewhiten_cache: list = []          # [(callable | None, impl_name)] once resolved


# ── Session loading (shared objects for every mode) ─────────────────

def _interp_onto_grid(arr: np.ndarray, ts: np.ndarray, ts_grid: np.ndarray) -> np.ndarray:
    """(N, D) samples at ``ts`` -> (M, D) float32 on ``ts_grid`` (linear, edge-clamped).

    Same vectorised interpolation as the Phase 0 loader's ``_onto_common``.
    """
    arr = np.asarray(arr)
    ts = np.asarray(ts, dtype=np.float64)
    if arr.shape[0] == 0 or ts_grid.size == 0:
        return np.zeros((ts_grid.size, arr.shape[1]), dtype=np.float32)
    if arr.shape[0] == 1:
        return np.repeat(arr[:1], ts_grid.size, axis=0).astype(np.float32)
    idx_right = np.searchsorted(ts, ts_grid, side='left').clip(1, ts.size - 1)
    idx_left = idx_right - 1
    t_left, t_right = ts[idx_left], ts[idx_right]
    w = ((ts_grid - t_left) / np.maximum(t_right - t_left, 1e-12)).clip(0.0, 1.0)
    return (arr[idx_left] + w[:, None] * (arr[idx_right] - arr[idx_left])).astype(np.float32)


def pose33_onto_grid(pose33: np.ndarray, ts: np.ndarray, ts_grid: np.ndarray,
                     target_rate_hz: float = DDTW_TARGET_RATE_HZ) -> np.ndarray:
    """Native (N, 33, 4) pose -> (M, 33, 4) float32 on ``ts_grid``.

    x, y, z go through the exact two-step path of ``pose_ddtw.run_session_ddtw``
    (``resample_to_uniform_rate`` on the flattened (N, 132) view, then linear
    snap onto the common grid); the visibility column is replaced by the
    conservative bracketing-minimum rule of ``resample_pose33_visibility``.
    """
    pose33 = np.asarray(pose33, dtype=np.float32)
    ts = np.asarray(ts, dtype=np.float64)
    ts_grid = np.asarray(ts_grid, dtype=np.float64)
    flat = pose33.reshape(pose33.shape[0], POSE_FLAT_DIM_VIS)
    flat_uni, ts_uni = resample_to_uniform_rate(flat, ts, target_rate_hz)
    out = _interp_onto_grid(flat_uni, ts_uni, ts_grid).reshape(
        ts_grid.size, POSE_KEYPOINTS, POSE_COLS)
    out[:, :, 3] = resample_pose33_visibility(pose33, ts, ts_grid)
    return out


def add_pose33_to_session(sess: dict, streams: dict) -> dict:
    """Attach ``p{1,2}_pose33`` (M, 33, 4) on ``sess['ts_common']`` from raw streams."""
    ts_grid = sess['ts_common']
    for p in ('p1', 'p2'):
        sess[f'{p}_pose33'] = pose33_onto_grid(streams[f'{p}_pose33'],
                                               streams[f'{p}_ts'], ts_grid)
    return sess


def load_and_resample_pose33(sid: str) -> dict:
    """Phase 0 session dict plus ``p{1,2}_pose33`` on the same 12 Hz common grid."""
    sess = load_and_resample(sid)
    streams = load_pose_streams(PREPROC_POSE_ROOT / f'{sid}.npz', include_pose33=True)
    return add_pose33_to_session(sess, streams)


def _load_session_safe(sid: str, need_pose33: bool) -> dict:
    with limit_blas_threads(1):
        return load_and_resample_pose33(sid) if need_pose33 else load_and_resample(sid)


def make_pseudo_dyad_pose33(sess_a: dict, sess_b: dict) -> dict:
    """Phase 0 pseudo-dyad (P1 from A, P2 from B) that also carries the pose33 arrays.

    Feature caches are deliberately not carried over: pseudo-dyads and
    injected sessions recompute their features from the trimmed objects so
    every mode sees exactly the same inputs.
    """
    pseudo = make_pseudo_dyad(sess_a, sess_b)
    n = pseudo['ts_common'].size
    if 'p1_pose33' in sess_a and 'p2_pose33' in sess_b:
        pseudo['p1_pose33'] = sess_a['p1_pose33'][:n]
        pseudo['p2_pose33'] = sess_b['p2_pose33'][:n]
    pseudo['pose_format_in'] = (f"{sess_a.get('pose_format_in', '')}+"
                                f"{sess_b.get('pose_format_in', '')}")
    return pseudo


# ── kappa-injection on pose33 skeletons ─────────────────────────────

def _torso_frame_reference(seg: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Robust body-frame reference of a (T, 33, 4) segment: (mid_hip (3,), torso_len).

    Frames in which both hips and both shoulders are visible
    (vis > ``VISIBILITY_THRESHOLD``) contribute. ``mid_hip`` is the per-axis
    median of 0.5 * (p[23] + p[24]) over those frames (x, y, z) and
    ``torso_len`` the median |mid_shoulder - mid_hip| in x, y — the same
    quantity ``pose_angles`` uses to define the torso frame, so the scale is
    unit-free across MediaPipe (image-normalised) and RTMW (pixel) skeletons.
    ``None`` when no such frame exists or the torso is degenerate.
    """
    seg = np.asarray(seg, dtype=np.float64)
    hips = seg[:, [LM_L_HIP, LM_R_HIP]]
    shoulders = seg[:, [LM_L_SHOULDER, LM_R_SHOULDER]]
    ok = ((hips[..., 3] > VISIBILITY_THRESHOLD).all(axis=1)
          & (shoulders[..., 3] > VISIBILITY_THRESHOLD).all(axis=1)
          & np.isfinite(hips[..., :3]).all(axis=(1, 2))
          & np.isfinite(shoulders[..., :2]).all(axis=(1, 2)))
    if not ok.any():
        return None
    mid_hip = hips[ok, :, :3].mean(axis=1)                        # (n_ok, 3)
    mid_sh = shoulders[ok, :, :2].mean(axis=1)                    # (n_ok, 2)
    torso = float(np.median(np.hypot(*(mid_sh - mid_hip[:, :2]).T)))
    if not np.isfinite(torso) or torso < _TORSO_REF_MIN:
        return None
    return np.median(mid_hip, axis=0), torso


def align_pose33_to_frame(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """``src`` (T, 33, 4) re-expressed in ``dst``'s body frame (float32 copy).

    xyz_out = (xyz_src - mid_hip_src) * (torso_dst / torso_src) + mid_hip_dst
    with the robust references of ``_torso_frame_reference``; z is scaled by
    the same factor (RTMW z is identically 0, so this is harmless). The
    visibility column is copied unchanged. When either reference is
    undefined the coordinates are returned unchanged (scale 1, zero shift).
    """
    out = np.asarray(src).astype(np.float32, copy=True)
    ref_s, ref_d = _torso_frame_reference(src), _torso_frame_reference(dst)
    if ref_s is None or ref_d is None:
        return out
    (hip_s, torso_s), (hip_d, torso_d) = ref_s, ref_d
    xyz = np.asarray(src, dtype=np.float64)[..., :3]
    out[..., :3] = ((xyz - hip_s) * (torso_d / torso_s) + hip_d).astype(np.float32)
    return out


def inject_pose33_segment(p1_seg: np.ndarray, p2_seg: np.ndarray, kappa: float,
                          rng: np.random.Generator) -> np.ndarray:
    """P2_inj = kappa * warped(P1) + (1 - kappa) * P2 on (T, 33, 4) skeletons.

    P1 is first brought into P2's body frame (``align_pose33_to_frame``:
    robust mid-hip translation + torso-length scaling) so that
    RTMW-pixel and MediaPipe-normalised skeletons contribute in proportion
    to kappa rather than in proportion to their coordinate magnitude — the
    angle / event channels are scale-invariant and would otherwise see a
    step at kappa > 0 instead of a dose. P2 keeps its native units, so
    ``pose_angles`` thresholds (``MIN_SEGMENT_LEN`` / ``MIN_TORSO_LEN``, in
    the pose's own units) are unaffected. Coordinates (x, y, z) are then
    mixed with the Phase 0 ``inject_coupling_into_segment`` on the flattened
    (T, 99) view; the visibility column is the minimum of P1's visibility
    under the **same** warp (the RNG state is restored so both warps share
    their anchors) and P2's visibility — a mixed coordinate is only
    trustworthy when both sources were visible.
    """
    if kappa <= 0:
        return p2_seg.copy()
    T = p1_seg.shape[0]
    p1_al = align_pose33_to_frame(p1_seg, p2_seg)
    state = rng.bit_generator.state
    xyz = inject_coupling_into_segment(
        np.ascontiguousarray(p1_al[..., :3].reshape(T, -1), dtype=np.float32),
        np.ascontiguousarray(p2_seg[..., :3].reshape(T, -1), dtype=np.float32),
        kappa, rng)
    rng.bit_generator.state = state
    vis1_w = piecewise_linear_warp(np.ascontiguousarray(p1_seg[..., 3], dtype=np.float32),
                                   rng=rng)
    out = p2_seg.astype(np.float32, copy=True)
    out[..., :3] = xyz.reshape(T, POSE_KEYPOINTS, 3)
    out[..., 3] = np.minimum(vis1_w, p2_seg[..., 3])
    return out


def inject_session(sess: dict, kappa: float, rng: np.random.Generator,
                   seg_start: int, seg_end: int) -> dict:
    """Copy of ``sess`` whose P2 carries injected coupling on frames [seg_start, seg_end).

    Both the 99-D positions (PCA mode, Phase 0 path — untouched) and, when
    present, the pose33 array (angle / event modes, mixed in P2's body frame
    by ``inject_pose33_segment``) are injected with the same warp. Feature
    caches are dropped so the injected P2 is recomputed.
    """
    out = {k: v for k, v in sess.items() if k != 'features'}
    state = rng.bit_generator.state
    p2 = sess['p2_common'].copy()
    p2[seg_start:seg_end] = inject_coupling_into_segment(
        sess['p1_common'][seg_start:seg_end], sess['p2_common'][seg_start:seg_end],
        kappa, rng)
    out['p2_common'] = p2
    if 'p2_pose33' in sess:
        rng.bit_generator.state = state
        p2_33 = sess['p2_pose33'].copy()
        p2_33[seg_start:seg_end] = inject_pose33_segment(
            sess['p1_pose33'][seg_start:seg_end], sess['p2_pose33'][seg_start:seg_end],
            kappa, rng)
        out['p2_pose33'] = p2_33
    return out


def _pair_seed(sid_a: str, sid_b: str, kappa: float) -> int:
    """Stable (process-independent) seed for one pseudo-dyad pair and kappa."""
    return zlib.crc32(f'{sid_a}|{sid_b}|{int(round(1000 * kappa))}'.encode('utf-8'))


# ── Per-mode feature construction ───────────────────────────────────

def _participant_features(sess: dict, mode: str, p: str, ctx: dict
                          ) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """(X (N, D) float64, valid (N,) bool, info) for one participant in a DDTW mode.

    ``ctx`` carries ``components`` / ``mean`` for 'pca' and
    ``noise_normalize`` for the angle modes. A per-session cache
    (``sess['features'][mode][p]``, filled by ``precompute_features``) is used
    when present.
    """
    cached = sess.get('features', {}).get(mode, {}).get(p)
    if cached is not None:
        return cached
    v = sess[f'v{p[-1]}_common']
    if mode == 'pca':
        X = project_pca(sess[f'{p}_common'], ctx['components'], ctx['mean'])
        return X, v, None
    X, a_valid, info = build_angle_features(sess[f'{p}_pose33'], sess['ts_common'], mode,
                                            noise_normalize=ctx.get('noise_normalize', True))
    return X, v & a_valid, info


def precompute_features(sessions: list[dict], modes: list[str], ctx: dict,
                        n_jobs: int = -1) -> None:
    """Fill ``sess['features'][mode][p]`` for every real session (joblib threads)."""
    ddtw_modes = [m for m in modes if m in DDTW_MODES]
    tasks = [(i, m, p) for i in range(len(sessions)) for m in ddtw_modes for p in ('p1', 'p2')]
    if not tasks:
        return

    def _one(i, m, p):
        with limit_blas_threads(1):
            return i, m, p, _participant_features(sessions[i], m, p, ctx)

    n_jobs_eff = pick_n_jobs(0.2, requested=n_jobs, max_jobs_hard_cap=len(tasks))
    for i, m, p, feat in Parallel(n_jobs=n_jobs_eff, prefer='threads')(
            delayed(_one)(*t) for t in tasks):
        sessions[i].setdefault('features', {}).setdefault(m, {})[p] = feat


def _event_npz_view(sess: dict) -> dict:
    """In-memory view of a session in the ``data/preproc/pose/v1`` NPZ contract."""
    return {
        'p1_pose33': sess['p1_pose33'], 'p1_pose33_ts': sess['ts_common'],
        'p1_pose_features_valid': sess['v1_common'],
        'p2_pose33': sess['p2_pose33'], 'p2_pose33_ts': sess['ts_common'],
        'p2_pose_features_valid': sess['v2_common'],
    }


def event_grid(ts_common: np.ndarray, fs_out: float = FS_OUT_HZ) -> np.ndarray:
    """2 Hz bin centres spanning the common grid (stream-relative seconds)."""
    ts_common = np.asarray(ts_common, dtype=np.float64)
    if ts_common.size < 2:
        return ts_common.copy()
    return np.arange(ts_common[0], ts_common[-1], 1.0 / fs_out, dtype=np.float64)


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            out[k] = v
    return out


# ── Mode-agnostic channel computation ───────────────────────────────

def compute_channel(sess: dict, mode: str, ctx: dict, n_surrogates: int,
                    seed: int = 42, compute_path_features: bool = False) -> dict:
    """One session (real, pseudo or injected) -> surrogate z at 2 Hz for ``mode``.

    Returns ``{'sid', 'mode', 'z' (n,) float32, 'stride_ts' (n,) float64,
    'info'}`` plus ``'raw'`` for DDTW modes and, when
    ``compute_path_features``, ``'lag_s'`` / ``'lag_var'`` / ``'asym'``.
    Event modes are computed on the same 12 Hz pose33 objects (``fs_hint``
    = grid rate, ``lsl_offset`` = 0 because grid and events share a clock)
    with the production sigma = 15 s smoothing + standardisation. Their
    ``info['mean_z_raw']`` (mean of the per-bin coincidence z **before**
    smoothing + standardisation) is what Test 3 reads, see ``_t3_level``.
    """
    if mode not in ALL_MODES:
        raise ValueError(f'mode must be one of {ALL_MODES}; got {mode!r}')
    with limit_blas_threads(1):
        if mode in DDTW_MODES:
            X1, v1, info1 = _participant_features(sess, mode, 'p1', ctx)
            X2, v2, info2 = _participant_features(sess, mode, 'p2', ctx)
            out = compute_session_ddtw(X1, X2, v1, v2, sess['ts_common'], sess['markers_rel'],
                                       n_surrogates=n_surrogates, seed=seed,
                                       compute_path_features=compute_path_features)
            info = {'n_strides': int(out['n_strides']), 'n_features': int(X1.shape[1])}
            if info1 is not None and info2 is not None:
                info['noise_floor'] = np.stack([info1['noise_floor'],
                                                info2['noise_floor']]).tolist()
                info['noise_floor_units'] = info1['noise_floor_units']
            res = {'sid': sess['sid'], 'mode': mode,
                   'z': np.asarray(out['ddtw_z'], dtype=np.float32),
                   'stride_ts': np.asarray(out['stride_ts'], dtype=np.float64),
                   'raw': np.asarray(out['ddtw_real'], dtype=np.float32),
                   'info': info}
            if compute_path_features:
                res['lag_s'] = np.asarray(out['ddtw_lag_s'], dtype=np.float32)
                res['lag_var'] = np.asarray(out['ddtw_lag_var'], dtype=np.float32)
                res['asym'] = np.asarray(out['ddtw_asym'], dtype=np.float32)
        else:
            t_grid = event_grid(sess['ts_common'])
            z, info = compute_pose_event_coincidence(
                _event_npz_view(sess), t_grid, 0.0, event_kind=EVENT_KIND[mode],
                fs_hint=DDTW_TARGET_RATE_HZ, n_surrogates=n_surrogates, seed=seed,
                smooth_sigma_s=ctx.get('smooth_sigma_s', EVENT_SMOOTH_SIGMA_S))
            res = {'sid': sess['sid'], 'mode': mode,
                   'z': np.asarray(z, dtype=np.float32), 'stride_ts': t_grid,
                   'info': _jsonable(info)}
    return res


def _nanmean(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    return float(z[finite].mean()) if finite.any() else float('nan')


def _t3_level(res: dict) -> float:
    """Test 3 mean-level statistic of one ``compute_channel`` result.

    DDTW modes: mean finite ``z`` (surrogate z per stride). Event modes:
    ``info['mean_z_raw']`` — the mean per-bin circular-shift coincidence z
    before the sigma = 15 s smoothing + per-session standardisation, because
    the standardised production trace has mean 0 by construction for real and
    pseudo pairs alike (mirrors how the Task 4 acceptance tests evaluate
    ``mean_z_raw``). Falls back to the trace mean when the key is absent
    (``missing_data`` results). ``T3_LEVEL_SOURCE`` names the quantity.
    """
    if res['mode'] in EVENT_MODES:
        raw = res.get('info', {}).get('mean_z_raw')
        if raw is not None:
            return float(raw)
    return _nanmean(res['z'])


# ── AR(1) prewhitening of the per-session z (validity = finite) ─────

def _resolve_prewhiten():
    """Lazily import the V8 reference implementation (needs matplotlib)."""
    if not _prewhiten_cache:
        try:
            from scripts._run_rslds_scaffold_v8 import prewhiten_and_standardize
            _prewhiten_cache.append((prewhiten_and_standardize, _PREWHITEN_REFERENCE))
        except ImportError:
            _prewhiten_cache.append((None, _PREWHITEN_FALLBACK))
    return _prewhiten_cache[0]


def _prewhiten_fallback(z: np.ndarray, valid: np.ndarray,
                        rho_thresh: float = _PREWHITEN_RHO_THRESH) -> np.ndarray:
    """Single-channel mirror of ``prewhiten_and_standardize`` (validity-aware).

    Lag-1 rho on consecutive valid pairs; up to three AR(1) rounds while
    |rho| > ``rho_thresh`` (filter runs on all samples for continuity, rho is
    re-estimated on valid pairs); then standardise on valid samples and zero
    the invalid ones. Used only when the V8 script cannot be imported (its
    plotting dependencies are absent).
    """
    z = np.asarray(z, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    zeros = np.zeros(z.size, dtype=np.float32)
    zv = z[valid]
    if zv.size < 10 or np.std(zv) < 1e-8:
        return zeros
    pair = valid[:-1] & valid[1:]
    rho = float(np.corrcoef(z[:-1][pair], z[1:][pair])[0, 1]) if pair.sum() >= 10 else 0.0
    z_cur = z.copy()
    n_rounds = 0
    while abs(rho) > rho_thresh and n_rounds < 3:
        z_cur = np.concatenate([[0.0], z_cur[1:] - rho * z_cur[:-1]])
        n_rounds += 1
        pw = valid[1:-1] & valid[2:]
        if pw.sum() >= 10 and np.std(z_cur[1:-1][pw]) > 1e-8:
            rho = float(np.corrcoef(z_cur[1:-1][pw], z_cur[2:][pw])[0, 1])
        else:
            rho = 0.0
    zv = z_cur[valid]
    sd = float(np.std(zv))
    if sd <= 1e-8:
        return zeros
    out = (z_cur - float(np.mean(zv))) / sd
    out[~valid] = 0.0
    return out.astype(np.float32)


def prewhiten_single(z: np.ndarray, key: str = 'pose') -> tuple[np.ndarray, str]:
    """(z_pw float32, implementation name): AR(1)-prewhitened + standardised z.

    NaN samples are zero-filled and passed as invalid (the V11 convention
    for structural zeros), so they neither bias rho nor the standardisation.
    """
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    filled = np.where(finite, z, 0.0)
    fn, impl = _resolve_prewhiten()
    if fn is None:
        return _prewhiten_fallback(filled, finite), impl
    out, _diag = fn(filled[:, None], [key], valid_mask=finite[:, None])
    return np.asarray(out[:, 0], dtype=np.float32), impl


# ── Test 1: semi-synthetic dose response (mode-agnostic) ────────────

def _segment_auc(z: np.ndarray, ts: np.ndarray, t_seg_start: float, t_seg_end: float) -> float:
    in_seg = (ts >= t_seg_start) & (ts < t_seg_end)
    valid = np.isfinite(z)
    in_v, out_v = in_seg & valid, (~in_seg) & valid
    if in_v.sum() < 2 or out_v.sum() < 2:
        return float('nan')
    return _auc_from_scores(z[in_v], z[out_v])


def _proc_test1_pair(args):
    (sess_a, sess_b, mode, ctx, n_surr, injection_segment_s, kappa_levels) = args
    with limit_blas_threads(1):
        pseudo = make_pseudo_dyad_pose33(sess_a, sess_b)
        T = pseudo['ts_common'].size
        seg_len = int(injection_segment_s * DDTW_TARGET_RATE_HZ)
        if T < 2 * seg_len:
            return None
        seg_start = T // 2 - seg_len // 2
        seg_end = seg_start + seg_len
        ts = pseudo['ts_common']
        t_seg_start, t_seg_end = float(ts[seg_start]), float(ts[min(seg_end, T - 1)])
        per_kappa = {}
        for kappa in kappa_levels:
            rng = np.random.default_rng(_pair_seed(sess_a['sid'], sess_b['sid'], kappa))
            inj = inject_session(pseudo, kappa, rng, seg_start, seg_end)
            res = compute_channel(inj, mode, ctx, n_surr, seed=42 + int(round(1000 * kappa)))
            per_kappa[float(kappa)] = _segment_auc(res['z'], res['stride_ts'],
                                                   t_seg_start, t_seg_end)
        return (sess_a['sid'], sess_b['sid'], pseudo.get('pose_format_in', ''), per_kappa)


def _test1_statistics(auc_table: pd.DataFrame, kappa_levels: tuple, seed: int,
                      n_perm: int = 1000) -> dict:
    """Phase 0 Test 1 statistics: mean AUC per kappa, Kendall tau permutation p."""
    if auc_table.empty:
        return {'mean_auc_by_kappa': {}, 'kendall_tau': None, 'kendall_p_perm': None,
                'auc_at_max_kappa': float('nan'), 'pass': False}
    mean_auc = auc_table.groupby('kappa')['auc'].mean().to_dict()
    auc_at_max = float(mean_auc.get(float(max(kappa_levels)), np.nan))
    kappa_arr = auc_table['kappa'].to_numpy(dtype=np.float64)
    auc_arr = auc_table['auc'].to_numpy(dtype=np.float64)
    finite = np.isfinite(auc_arr)
    if finite.sum() >= 6:
        tau, _ = stats.kendalltau(kappa_arr[finite], auc_arr[finite])
        rng = np.random.default_rng(seed + 7)
        ka = kappa_arr[finite]
        null_taus = np.empty(n_perm)
        for i in range(n_perm):
            null_taus[i], _ = stats.kendalltau(ka, rng.permutation(auc_arr[finite]))
        p_perm = float((np.abs(null_taus) >= abs(tau)).mean())
    else:
        tau, p_perm = np.nan, np.nan
    passed = bool(np.isfinite(auc_at_max) and auc_at_max >= T1_AUC_THRESH
                  and np.isfinite(p_perm) and p_perm < T1_P_THRESH)
    return {
        'mean_auc_by_kappa': {float(k): float(v) for k, v in mean_auc.items()},
        'kendall_tau': float(tau) if np.isfinite(tau) else None,
        'kendall_p_perm': float(p_perm) if np.isfinite(p_perm) else None,
        'auc_at_max_kappa': auc_at_max,
        'pass': passed,
    }


def _directed_pairs(n: int, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def _map(fn, args_list, executor):
    """``executor.map`` when a pool is given, otherwise an in-process loop."""
    if executor is not None:
        return list(executor.map(fn, args_list))
    return [fn(a) for a in args_list]


T1_AUC_COLUMNS = ['session_a', 'session_b', 'pose_format_in', 'kappa', 'auc']


def test1_dose_response(sessions: list[dict], mode: str, ctx: dict, n_surr: int,
                        kappa_levels: tuple = KAPPA_LEVELS, max_pairs: int = 30,
                        seed: int = 42, executor=None,
                        injection_segment_s: float = INJECTION_SEGMENT_S) -> dict:
    """Phase 0 Test 1 for ``mode``: AUC(in-segment vs out) per pseudo-dyad and kappa.

    ``auc_table`` carries the pair's ``pose_format_in`` (``"<A fmt>+<B fmt>"``)
    so mixed-format pseudo-dyads are auditable in ``phase1_semisynthetic.csv``.
    """
    if len(sessions) < 2:
        return {'status': 'SKIPPED — need >=2 sessions for pseudo-dyad pairs',
                'pairs': [], 'auc_table': pd.DataFrame(columns=T1_AUC_COLUMNS),
                'mean_auc_by_kappa': {}, 'kendall_tau': None, 'kendall_p_perm': None,
                'auc_at_max_kappa': float('nan'), 'pass': False}
    pairs = _directed_pairs(len(sessions), max_pairs, seed)
    args = [(sessions[i], sessions[j], mode, ctx, n_surr, injection_segment_s, kappa_levels)
            for i, j in pairs]
    results = [r for r in _map(_proc_test1_pair, args, executor) if r is not None]
    rows = [{'session_a': sa, 'session_b': sb, 'pose_format_in': fmt, 'kappa': k, 'auc': a}
            for sa, sb, fmt, perk in results for k, a in perk.items()]
    auc_table = pd.DataFrame(rows, columns=T1_AUC_COLUMNS)
    out = _test1_statistics(auc_table, kappa_levels, seed)
    out['status'] = 'OK' if np.isfinite(out['auc_at_max_kappa']) else 'INSUFFICIENT_DATA'
    out['pairs'] = [(sa, sb) for sa, sb, _fmt, _ in results]
    out['auc_table'] = auc_table
    return out


# ── Test 2: condition contrast (Phase 0 helper, mode-agnostic) ──────

def _as_phase0_results(real_results: list[dict]) -> list[dict]:
    return [{'sid': r['sid'], 'ddtw_z': r['z'], 'stride_ts': r['stride_ts']}
            for r in real_results]


def test2_contrast(real_results: list[dict], sessions: list[dict]) -> dict:
    """Phase 0 ``test2_condition_contrast`` on this mode's z timecourses."""
    return test2_condition_contrast(_as_phase0_results(real_results), sessions)


# ── Test 3: pseudo-dyad null (mode-agnostic) ────────────────────────

def _proc_test3_pair(args):
    sess_a, sess_b, mode, ctx, n_surr = args
    with limit_blas_threads(1):
        pseudo = make_pseudo_dyad_pose33(sess_a, sess_b)
        return _t3_level(compute_channel(pseudo, mode, ctx, n_surr, seed=42))


def test3_pseudo_null(sessions: list[dict], real_results: list[dict], mode: str,
                      ctx: dict, n_surr: int, max_pairs: int = 20, seed: int = 42,
                      executor=None) -> dict:
    """Phase 0 Test 3: mean real-pair level minus mean pseudo-pair level >= 0.5.

    The per-session level is ``_t3_level``: mean surrogate z of the saved
    trace for DDTW modes; for the event modes the mean **raw** per-bin
    coincidence z (``info['mean_z_raw']``), because the production trace
    (``{sid}__z``, sigma = 15 s smoothed + per-session standardised) has mean
    0 for real and pseudo pairs alike. ``level_source`` records which.
    """
    if len(sessions) < 2:
        return {'status': 'SKIPPED', 'real_per_session': {}, 'pseudo_means': [],
                'mean_real': float('nan'), 'mean_pseudo': float('nan'),
                'delta': float('nan'), 'pass': False,
                'level_source': T3_LEVEL_SOURCE.get(mode, 'z')}
    real_means = {r['sid']: _t3_level(r) for r in real_results}
    pairs = _directed_pairs(len(sessions), max_pairs, seed)
    args = [(sessions[i], sessions[j], mode, ctx, n_surr) for i, j in pairs]
    pseudo_means = [m for m in _map(_proc_test3_pair, args, executor) if np.isfinite(m)]
    reals = [v for v in real_means.values() if np.isfinite(v)]
    mean_real = float(np.mean(reals)) if reals else float('nan')
    mean_pseudo = float(np.mean(pseudo_means)) if pseudo_means else float('nan')
    delta = mean_real - mean_pseudo
    return {
        'status': 'OK',
        'real_per_session': real_means,
        'pseudo_means': pseudo_means,
        'mean_real': mean_real,
        'mean_pseudo': mean_pseudo,
        'delta': float(delta),
        'pass': bool(np.isfinite(delta) and delta >= T3_DELTA_THRESH),
        'level_source': T3_LEVEL_SOURCE.get(mode, 'z'),
    }


# ── Test 4: redundancy vs Phase 0 DDTW and V11 baseline ─────────────

def align_on_common_grid(z_a: np.ndarray, ts_a: np.ndarray, z_b: np.ndarray,
                         ts_b: np.ndarray, fs_out: float = FS_OUT_HZ,
                         min_samples: int = 10) -> tuple[np.ndarray, np.ndarray] | None:
    """Two z timecourses -> (x_a, x_b) on a shared ``fs_out`` grid over their overlap.

    Finite samples only are interpolated (Phase 0 Test 4 convention). ``None``
    when the overlap or the finite count is too small, or either is constant.
    """
    z_a = np.asarray(z_a, dtype=np.float64); ts_a = np.asarray(ts_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64); ts_b = np.asarray(ts_b, dtype=np.float64)
    ma, mb = np.isfinite(z_a), np.isfinite(z_b)
    if ma.sum() < min_samples or mb.sum() < min_samples:
        return None
    t0 = max(ts_a[ma][0], ts_b[mb][0])
    t1 = min(ts_a[ma][-1], ts_b[mb][-1])
    if t1 <= t0:
        return None
    grid = np.arange(t0, t1, 1.0 / fs_out)
    if grid.size < min_samples:
        return None
    xa = np.interp(grid, ts_a[ma], z_a[ma])
    xb = np.interp(grid, ts_b[mb], z_b[mb])
    if np.std(xa) < 1e-8 or np.std(xb) < 1e-8:
        return None
    return xa, xb


def pearson_block_ci(x: np.ndarray, y: np.ndarray, block_len: int,
                     n_boot: int = N_BOOT, seed: int = 42) -> dict:
    """Pearson r with a block-bootstrap percentile CI.

    With centred series, r = mean(xy) / sqrt(mean(x^2) mean(y^2)). The three
    moment series are block-resampled **jointly** (one ``block_bootstrap_mean``
    call on a (T, 3) matrix shares the block draws across columns), and the
    ratio is formed per resample, so every bootstrap r lies in [-1, 1]. The
    point estimate is the exact ``np.corrcoef`` r over all samples.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    r = float(np.corrcoef(x, y)[0, 1])
    bb = block_bootstrap_mean(np.stack([xc * yc, xc * xc, yc * yc], axis=1),
                              block_len, n_boot=n_boot, seed=seed)
    boot = bb['boot']
    denom = np.sqrt(np.maximum(boot[:, 1] * boot[:, 2], 1e-24))
    r_boot = np.clip(boot[:, 0] / denom, -1.0, 1.0)
    r_boot = r_boot[np.isfinite(r_boot)]
    if r_boot.size:
        ci_lo, ci_hi = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    else:
        ci_lo = ci_hi = float('nan')
    return {'r': r, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'n': int(x.size), 'n_blocks': int(bb['n_blocks'])}


def load_phase0_reference(sids: list[str], npz_path: Path | None = None
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{sid: (z, stride_ts)} from the Phase 0 per-session NPZ (empty when absent)."""
    path = PHASE0_DIR / 'pose_ddtw_per_session.npz' if npz_path is None else Path(npz_path)
    if not path.exists():
        return {}
    out = {}
    with np.load(path) as npz:
        for sid in sids:
            zk, tk = f'{sid}__ddtw_z', f'{sid}__stride_ts'
            if zk in npz.files and tk in npz.files:
                out[sid] = (npz[zk].astype(np.float64), npz[tk].astype(np.float64))
    return out


def load_v11_baseline(sid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(z_raw_pose, t_rel) of the V11 multi-lag pose baseline in stream-relative seconds."""
    v11_path = V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_ztimecourses.npz'
    digest_path = DIGEST_ROOT / f'{sid}.json'
    if not v11_path.exists() or not digest_path.exists():
        return None
    with np.load(v11_path) as v11:
        if 'z_raw_pose' not in v11.files:
            return None
        z = v11['z_raw_pose'].astype(np.float64)
        t = v11['t_common'].astype(np.float64)
    t_start_lsl = float(json.loads(digest_path.read_text()).get('t_start_lsl', 0.0))
    return z, t - t_start_lsl


def _r_descriptor(mean_r: float) -> str:
    if not np.isfinite(mean_r):
        return 'not available'
    if abs(mean_r) < T4_INDEPENDENT_R:
        return 'largely independent'
    if abs(mean_r) < T4_REDUNDANT_R:
        return 'partial overlap'
    return 'largely redundant'


def _session_level_ci(values: np.ndarray, seed: int = 42, n_boot: int = 1000
                      ) -> tuple[float, tuple[float, float]]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(vals, vals.size, replace=True).mean() for _ in range(n_boot)])
    return float(vals.mean()), (float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5)))


def test4_redundancy(real_results: list[dict], ref_ddtw: dict, block_len: int,
                     v11_loader=load_v11_baseline) -> dict:
    """Per-session Pearson r of this mode vs the Phase 0 DDTW and vs the V11 baseline.

    Each r carries a block-bootstrap 95 % CI; the paired mean-z difference
    (mode minus Phase 0 DDTW on the shared grid) uses
    ``block_bootstrap_paired`` on the same block structure. Cohort-level
    mean r is reported with a session-level bootstrap CI.
    """
    rows = []
    for r in real_results:
        sid = r['sid']
        row = {'sid': sid, 'r_ddtw': np.nan, 'r_ddtw_lo': np.nan, 'r_ddtw_hi': np.nan,
               'dmean_vs_ddtw': np.nan, 'dmean_lo': np.nan, 'dmean_hi': np.nan,
               'r_v11': np.nan, 'r_v11_lo': np.nan, 'r_v11_hi': np.nan, 'n': 0, 'note': ''}
        notes = []
        ref = ref_ddtw.get(sid)
        if ref is None:
            notes.append('no Phase 0 DDTW')
        else:
            al = align_on_common_grid(r['z'], r['stride_ts'], ref[0], ref[1])
            if al is None:
                notes.append('DDTW: no overlap / constant')
            else:
                pc = pearson_block_ci(al[0], al[1], block_len)
                row.update(r_ddtw=pc['r'], r_ddtw_lo=pc['ci_lo'], r_ddtw_hi=pc['ci_hi'],
                           n=pc['n'])
                pd_ = block_bootstrap_paired(al[0], al[1], block_len, n_boot=N_BOOT, seed=42)
                row.update(dmean_vs_ddtw=float(pd_['mean_diff'][0]),
                           dmean_lo=float(pd_['ci_lo'][0]), dmean_hi=float(pd_['ci_hi'][0]))
        v11 = v11_loader(sid)
        if v11 is None:
            notes.append('no V11 scaffold')
        else:
            al = align_on_common_grid(r['z'], r['stride_ts'], v11[0], v11[1])
            if al is None:
                notes.append('V11: no overlap / constant')
            else:
                pc = pearson_block_ci(al[0], al[1], block_len)
                row.update(r_v11=pc['r'], r_v11_lo=pc['ci_lo'], r_v11_hi=pc['ci_hi'])
                row['n'] = max(row['n'], pc['n'])
        row['note'] = '; '.join(notes)
        rows.append(row)
    table = pd.DataFrame(rows)
    mean_r_ddtw, ci_ddtw = _session_level_ci(table['r_ddtw'].to_numpy(dtype=np.float64))
    mean_r_v11, ci_v11 = _session_level_ci(table['r_v11'].to_numpy(dtype=np.float64))
    return {
        'status': 'OK',
        'table': table,
        'mean_r_ddtw': mean_r_ddtw, 'ci_ddtw': ci_ddtw,
        'descriptor_ddtw': _r_descriptor(mean_r_ddtw),
        'mean_r_v11': mean_r_v11, 'ci_v11': ci_v11,
        'descriptor_v11': _r_descriptor(mean_r_v11),
        'block_len': int(block_len),
    }


# ── Per-session condition summaries (block-bootstrap CIs) ───────────

def condition_summary(real_results: list[dict], sessions: list[dict], mode: str,
                      block_len: int, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Mean z per condition per session with block-bootstrap 95 % CIs.

    Columns: mode, sid, condition, mean_z, ci_lo, ci_hi, se, n_samples,
    n_finite, n_blocks. Conditions shorter than one block get the plain
    finite mean and NaN CI (``n_blocks`` = 0).
    """
    rows = []
    for r, s in zip(real_results, sessions):
        _, blocks = assign_frame_conditions(r['stride_ts'], s['markers_rel'])
        for cond, i0, i1 in blocks:
            zc = np.asarray(r['z'][i0:i1], dtype=np.float64)
            bb = block_bootstrap_mean(zc, block_len, n_boot=n_boot, seed=0)
            mean_z = float(bb['mean'][0]) if bb['n_blocks'] > 0 else _nanmean(zc)
            rows.append({'mode': mode, 'sid': r['sid'], 'condition': cond,
                         'mean_z': mean_z,
                         'ci_lo': float(bb['ci_lo'][0]), 'ci_hi': float(bb['ci_hi'][0]),
                         'se': float(bb['se'][0]), 'n_samples': int(zc.size),
                         'n_finite': int(np.isfinite(zc).sum()),
                         'n_blocks': int(bb['n_blocks'])})
    return pd.DataFrame(rows, columns=['mode', 'sid', 'condition', 'mean_z', 'ci_lo',
                                       'ci_hi', 'se', 'n_samples', 'n_finite', 'n_blocks'])


# ── Decision ────────────────────────────────────────────────────────

def parse_decision_line(text: str) -> str | None:
    """Extract the backticked value of a '**Pose channel for MVP scaffold: `x`**' line."""
    for line in text.splitlines():
        if 'Pose channel for MVP scaffold' in line and '`' in line:
            parts = line.split('`')
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def read_phase0_decision(phase0_dir: Path = PHASE0_DIR) -> str | None:
    report = Path(phase0_dir) / PHASE0_REPORT_NAME
    return parse_decision_line(report.read_text(encoding='utf-8')) if report.exists() else None


def mode_passes(tests: dict) -> bool:
    return bool(tests['t1'].get('pass') and tests['t2'].get('pass') and tests['t3'].get('pass'))


def decide(per_mode: dict[str, dict], phase0_decision: str | None) -> tuple[str, str]:
    """(decision, reason): passing mode with the largest Test 3 margin; Phase 0
    DDTW if only Phase 0 passed; else the multi-lag baseline."""
    passing = [(m, float(t['t3'].get('delta', np.nan))) for m, t in per_mode.items()
               if mode_passes(t)]
    passing = [(m, d) for m, d in passing if np.isfinite(d)]
    if passing:
        best, margin = max(passing, key=lambda md: md[1])
        return MODE_TO_DECISION[best], (f'{best} passes Tests 1-3 with the largest '
                                        f'Test 3 margin ({margin:+.3f} z)')
    if phase0_decision == DECISION_DDTW:
        return DECISION_DDTW, 'no Phase 1 candidate passes; Phase 0 DDTW passed'
    return DECISION_BASELINE, 'no Phase 1 candidate passes and Phase 0 did not select DDTW'


# ── Outputs ─────────────────────────────────────────────────────────

def save_mode_outputs(out_dir: Path, mode: str, real_results: list[dict],
                      meta: dict | None = None) -> Path:
    """Write ``pose_<mode>_per_session.npz`` (+ JSON sidecar) in the Phase 0 contract."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save, sidecar_sessions = {}, {}
    for r in real_results:
        sid = r['sid']
        save[f'{sid}__z'] = np.asarray(r['z'], dtype=np.float32)
        save[f'{sid}__stride_ts'] = np.asarray(r['stride_ts'], dtype=np.float64)
        if 'z_pw' in r:
            save[f'{sid}__z_pw'] = np.asarray(r['z_pw'], dtype=np.float32)
        for k in ('lag_s', 'lag_var', 'asym'):
            if k in r:
                save[f'{sid}__{k}'] = np.asarray(r[k], dtype=np.float32)
        sidecar_sessions[sid] = {'n_samples': int(np.size(r['z'])),
                                 'n_finite': int(np.isfinite(r['z']).sum()),
                                 'prewhiten_impl': r.get('prewhiten_impl'),
                                 'info': _jsonable(r.get('info', {}))}
    npz_path = out_dir / f'pose_{mode}_per_session.npz'
    np.savez(npz_path, **save)
    sidecar = {'mode': mode, 'decision_name': MODE_TO_DECISION[mode],
               'contract': '{sid}__z surrogate z at 2 Hz, stream-relative {sid}__stride_ts',
               'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'sessions': sidecar_sessions, **(meta or {})}
    npz_path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2), encoding='utf-8')
    return npz_path


def _fmt(x, spec: str = '+.3f') -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 'N/A'
    return format(xf, spec) if np.isfinite(xf) else 'N/A'


def write_phase1_report(out_dir: Path, sessions: list[dict], per_mode: dict[str, dict],
                        decision: str, reason: str, n_surr: int,
                        phase0_decision: str | None, block_len: int) -> Path:
    """Write ``phase1_report.md`` (per-mode table of Tests 1-4 + decision line)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ['# Phase 1 — Pose-Channel Candidate Validation Report\n',
          f'**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}\n',
          '**Plan:** docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md (Task 5)\n',
          f'**Surrogates per session:** {n_surr}  ',
          f'**Block bootstrap:** {block_len} samples ({block_len / FS_OUT_HZ:.0f} s at '
          f'{FS_OUT_HZ:.0f} Hz)\n',
          '\n## Cohort\n', f'- Sessions evaluated: **{len(sessions)}**\n']
    md += [f'  - `{s["sid"]}` (format: `{s.get("pose_format_in", "")}`)\n' for s in sessions]

    md.append('\n## Summary — Tests 1–4 per mode\n\n')
    md.append('| Mode | T1 AUC@κmax | T1 τ (perm p) | T2 Δz (min p) | T3 real − pseudo '
              '| T4 r vs DDTW [95% CI] | T4 r vs V11 [95% CI] | T1 | T2 | T3 | Pass |\n')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|\n')
    for mode, t in per_mode.items():
        t1, t2, t3, t4 = t['t1'], t['t2'], t['t3'], t['t4']
        md.append(
            f'| `{mode}` | {_fmt(t1.get("auc_at_max_kappa"), ".3f")} '
            f'| {_fmt(t1.get("kendall_tau"))} ({_fmt(t1.get("kendall_p_perm"), ".4f")}) '
            f'| {_fmt(t2.get("mean_delta"))} ({_fmt(t2.get("p_min"), ".4f")}) '
            f'| {_fmt(t3.get("delta"))} '
            f'| {_fmt(t4.get("mean_r_ddtw"))} [{_fmt(t4.get("ci_ddtw", (np.nan, np.nan))[0])}, '
            f'{_fmt(t4.get("ci_ddtw", (np.nan, np.nan))[1])}] '
            f'| {_fmt(t4.get("mean_r_v11"))} [{_fmt(t4.get("ci_v11", (np.nan, np.nan))[0])}, '
            f'{_fmt(t4.get("ci_v11", (np.nan, np.nan))[1])}] '
            f'| {t1.get("pass")} | {t2.get("pass")} | {t3.get("pass")} | {mode_passes(t)} |\n')
    md.append(f'\nGates: T1 Kendall τ perm p < {T1_P_THRESH} AND AUC ≥ {T1_AUC_THRESH} at '
              f'κ = {max(KAPPA_LEVELS)}; T2 |Δz| ≥ 0.5 AND min(paired t, Wilcoxon) p < 0.05 '
              f'(either direction); T3 real − pseudo ≥ {T3_DELTA_THRESH} z; T4 descriptive.\n')
    md.append(f'\nNote on T3 for `{"` / `".join(EVENT_MODES)}`: the real and pseudo levels are '
              'the mean **raw** circular-shift coincidence z (`info[\'mean_z_raw\']`) before '
              f'the σ = {EVENT_SMOOTH_SIGMA_S:.0f} s smoothing + per-session standardisation, '
              'because the saved `{sid}__z` trace is standardised to mean 0 by construction '
              '(real and pseudo alike); DDTW modes use the mean of the saved surrogate-z '
              'trace. T2 and T4 are evaluated on the saved trace for every mode (for the '
              'event modes T2 Δz is therefore in within-session SD units of the smoothed '
              'trace, not surrogate-z units).\n')

    for mode, t in per_mode.items():
        t1, t2, t3, t4 = t['t1'], t['t2'], t['t3'], t['t4']
        md.append(f'\n## Mode `{mode}` → `{MODE_TO_DECISION[mode]}`\n')
        md.append(f'\n### Test 1 — Semi-synthetic dose response ({t1.get("status")})\n')
        for k in sorted(t1.get('mean_auc_by_kappa', {})):
            md.append(f'- κ={k:.2f}: AUC = {t1["mean_auc_by_kappa"][k]:.3f}\n')
        md.append(f'- AUC at max κ: **{_fmt(t1.get("auc_at_max_kappa"), ".3f")}**; Kendall τ = '
                  f'{_fmt(t1.get("kendall_tau"))}, permutation p = '
                  f'{_fmt(t1.get("kendall_p_perm"), ".4f")} — **Pass:** {t1.get("pass")}\n')
        md.append(f'\n### Test 2 — Condition contrast, magnitude ({t2.get("status")})\n')
        md.append(f'- N paired sessions: {len(t2.get("paired", []))}; mean Δz (conv − med) = '
                  f'**{_fmt(t2.get("mean_delta"))}** ({t2.get("direction", "n/a")}); paired t p = '
                  f'{_fmt(t2.get("p_t"), ".4f")}, Wilcoxon p = {_fmt(t2.get("p_wilcoxon"), ".4f")} '
                  f'— **Pass:** {t2.get("pass")}\n')
        md.append(f'\n### Test 3 — Pseudo-dyad null ({t3.get("status")})\n')
        level_src = t3.get('level_source', T3_LEVEL_SOURCE.get(mode, 'z'))
        md.append(f'- Level statistic: `{level_src}`'
                  + (' (raw per-bin coincidence z before smoothing + standardisation)'
                     if level_src == 'mean_z_raw' else ' (mean of the saved surrogate-z trace)')
                  + '\n')
        md.append(f'- Mean real-pair z = **{_fmt(t3.get("mean_real"))}**, mean pseudo-pair z = '
                  f'{_fmt(t3.get("mean_pseudo"))}, Δ = **{_fmt(t3.get("delta"))}** '
                  f'(threshold ≥ +{T3_DELTA_THRESH}) — **Pass:** {t3.get("pass")}\n')
        for sid, v in sorted(t3.get('real_per_session', {}).items()):
            md.append(f'  - `{sid}`: real mean z ({level_src}) = {_fmt(v)}\n')
        md.append('\n### Test 4 — Redundancy (descriptive, block-bootstrap CIs)\n')
        md.append(f'- vs Phase 0 DDTW: mean r = **{_fmt(t4.get("mean_r_ddtw"))}** '
                  f'(session-level 95% CI [{_fmt(t4.get("ci_ddtw", (np.nan, np.nan))[0])}, '
                  f'{_fmt(t4.get("ci_ddtw", (np.nan, np.nan))[1])}]) — '
                  f'{t4.get("descriptor_ddtw")}\n')
        md.append(f'- vs V11 multi-lag baseline: mean r = **{_fmt(t4.get("mean_r_v11"))}** '
                  f'(session-level 95% CI [{_fmt(t4.get("ci_v11", (np.nan, np.nan))[0])}, '
                  f'{_fmt(t4.get("ci_v11", (np.nan, np.nan))[1])}]) — '
                  f'{t4.get("descriptor_v11")}\n')
        table = t4.get('table')
        if table is not None and len(table):
            md.append('- Per session (r [block-bootstrap 95% CI]; Δmean z vs DDTW [CI]):\n')
            for _, row in table.iterrows():
                md.append(f'  - `{row["sid"]}`: r_ddtw = {_fmt(row["r_ddtw"])} '
                          f'[{_fmt(row["r_ddtw_lo"])}, {_fmt(row["r_ddtw_hi"])}], '
                          f'Δmean = {_fmt(row["dmean_vs_ddtw"])} '
                          f'[{_fmt(row["dmean_lo"])}, {_fmt(row["dmean_hi"])}]; '
                          f'r_v11 = {_fmt(row["r_v11"])} '
                          f'[{_fmt(row["r_v11_lo"])}, {_fmt(row["r_v11_hi"])}]'
                          + (f' ({row["note"]})\n' if row['note'] else '\n'))

    md.append('\n## Decision\n')
    md.append(f'- Phase 0 decision (from `{PHASE0_REPORT_NAME}`): '
              f'`{phase0_decision or "not available"}`\n')
    md.append('- Passing Phase 1 modes: ' + (', '.join(
        f'`{m}`' for m, t in per_mode.items() if mode_passes(t)) or 'none') + '\n')
    md.append(f'- Rule: {reason}\n')
    md.append(f'- **Pose channel for MVP scaffold: `{decision}`**\n')

    md.append('\n## Provenance\n')
    md += [f'- `{s["sid"]}` digest_xdf_md5 = `{s.get("digest_xdf_md5", "")}`\n' for s in sessions]
    out_path = out_dir / PHASE1_REPORT_NAME
    out_path.write_text(''.join(md), encoding='utf-8')
    return out_path


# ── Process-pool workers (module-level for pickling) ────────────────

def _proc_compute_real(args):
    sess, mode, ctx, n_surr, compute_path_features = args
    return compute_channel(sess, mode, ctx, n_surr, seed=42,
                           compute_path_features=compute_path_features)


def run_mode_real(sessions: list[dict], mode: str, ctx: dict, n_surr: int,
                  compute_path_features: bool, executor=None) -> list[dict]:
    """Real-pair channel for every session, with ``z_pw`` attached."""
    args = [(s, mode, ctx, n_surr, compute_path_features and mode in DDTW_MODES)
            for s in sessions]
    results = _map(_proc_compute_real, args, executor)
    for r in results:
        r['z_pw'], r['prewhiten_impl'] = prewhiten_single(r['z'], key=mode)
    return results


# ── Main ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modes', nargs='+', default=list(DEFAULT_MODES), choices=ALL_MODES,
                    help=f'Candidate modes to validate (default: {" ".join(DEFAULT_MODES)}; '
                         f'"pca" re-runs the Phase 0 baseline through this harness)')
    ap.add_argument('--sessions', nargs='*', default=None,
                    help='Specific session IDs (default: all canonical with pose preproc)')
    ap.add_argument('--all', action='store_true',
                    help='Use all canonical sessions with pose preproc available')
    ap.add_argument('--quick', action='store_true',
                    help=f'Use {QUICK_N_SURROGATES} surrogates instead of {DDTW_N_SURROGATES} (smoke mode)')
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help='Parallel workers (default -1 = all cores; 1 = in-process, no pool)')
    ap.add_argument('--max-test1-pairs', type=int, default=30,
                    help='Cap on pseudo-dyad pairs for Test 1 (default 30)')
    ap.add_argument('--max-test3-pairs', type=int, default=20,
                    help='Cap on pseudo-dyad pairs for Test 3 (default 20)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'Output directory (default {DEFAULT_OUT.relative_to(REPO_ROOT)})')
    ap.add_argument('--no-path-features', action='store_true',
                    help='Skip warping-path lag / asymmetry features for DDTW modes')
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.all and args.sessions:
        raise SystemExit('--all and --sessions are mutually exclusive')
    modes = list(dict.fromkeys(args.modes))
    out_dir = Path(args.out)
    if out_dir.resolve() == PHASE0_DIR.resolve():
        # Outputs never collide with the Phase 0 report by name, but keep the
        # directories distinct so a Phase 0 rerun cannot be confused with Phase 1.
        raise SystemExit(f'--out must not be the Phase 0 directory ({PHASE0_DIR})')

    sids = discover_sessions(restrict_to=set(args.sessions) if args.sessions else None)
    if not sids:
        raise SystemExit('ERROR: no canonical sessions with pose preproc found.')
    print(f'Discovered {len(sids)} canonical sessions with pose preproc:')
    for s in sids:
        print(f'  - {s}')
    n_surr = QUICK_N_SURROGATES if args.quick else DDTW_N_SURROGATES
    compute_path = not args.no_path_features
    block_len = block_len_from_seconds(BLOCK_S, FS_OUT_HZ)
    print(f'\nModes: {modes}\nN surrogates per session: {n_surr}\nParallel workers: {args.n_jobs}')
    log_resources(prefix='\nResource snapshot at start: ')

    # ── Stage 1: load + resample (joblib threads) ──
    need_pose33 = any(m != 'pca' for m in modes)
    n_jobs_load = pick_n_jobs(_RAM_PER_LOAD_GB, requested=args.n_jobs, max_jobs_hard_cap=len(sids))
    print(f'\n[1/4] Loading + resampling pose streams (pose33={need_pose33}, '
          f'n_jobs={n_jobs_load})...', flush=True)
    t0 = time.time()
    sessions = Parallel(n_jobs=n_jobs_load, prefer='threads')(
        delayed(_load_session_safe)(sid, need_pose33) for sid in sids)
    print(f'  Done in {time.time() - t0:.1f}s')

    # ── Stage 2: shared PCA (pca mode only) + angle feature extraction ──
    ctx: dict = {'components': None, 'mean': None, 'noise_normalize': True,
                 'smooth_sigma_s': EVENT_SMOOTH_SIGMA_S}
    pca_diag = None
    if 'pca' in modes:
        print('\n[2/4] Fitting shared cross-session PCA...', flush=True)
        t0 = time.time()
        streams = {f'{s["sid"]}_{p}': s[f'{p}_common'] for s in sessions for p in ('p1', 'p2')}
        components, mean, pca_diag = fit_shared_pca(streams, n_components=DDTW_PCA_DIM)
        ctx.update(components=components, mean=mean)
        print(f'  Done in {time.time() - t0:.1f}s. '
              f'EVR cumulative: {pca_diag["cumulative_variance_ratio"][-1]:.3f}')
    print('\n[2/4] Extracting per-session features (joblib threads)...', flush=True)
    t0 = time.time()
    precompute_features(sessions, modes, ctx, n_jobs=args.n_jobs)
    print(f'  Done in {time.time() - t0:.1f}s')

    # ── Stage 3: process pool (spawn) shared by every mode ──
    pool = None
    if args.n_jobs != 1:
        pool_size = pick_n_jobs(_RAM_PER_PAIR_GB, requested=args.n_jobs,
                                max_jobs_hard_cap=max(len(sessions), args.max_test1_pairs,
                                                      args.max_test3_pairs))
        print(f'\nStarting shared process pool: {pool_size} workers (mp_context=spawn)...',
              flush=True)
        t_pool = time.time()
        pool = _cf.ProcessPoolExecutor(max_workers=pool_size, mp_context=_MP_SPAWN)
        list(pool.map(int, [0] * pool_size))
        print(f'  Pool warm in {time.time() - t_pool:.1f}s')

    out_dir.mkdir(parents=True, exist_ok=True)
    phase0_decision = read_phase0_decision()
    ref_ddtw = load_phase0_reference(sids)
    per_mode: dict[str, dict] = {}
    auc_tables, contrast_tables, null_rows, redundancy_tables, cond_tables = [], [], [], [], []
    try:
        for mi, mode in enumerate(modes, 1):
            print(f'\n[3/4] Mode {mi}/{len(modes)}: {mode}', flush=True)
            t0 = time.time()
            real = run_mode_real(sessions, mode, ctx, n_surr, compute_path, executor=pool)
            print(f'  real-pair channel: {time.time() - t0:.1f}s')
            npz_path = save_mode_outputs(out_dir, mode, real, meta={
                'n_surrogates': n_surr, 'compute_path_features': compute_path,
                'prewhiten_impl': real[0]['prewhiten_impl'] if real else None,
                'pca_diag': pca_diag if mode == 'pca' else None})
            print(f'  saved -> {npz_path}')
            if mode == 'pca' and not ref_ddtw:
                ref_ddtw = {r['sid']: (r['z'].astype(np.float64), r['stride_ts']) for r in real}
                print('  (no Phase 0 NPZ — this run\'s pca channel is the Test 4 reference)')

            t0 = time.time()
            t1 = test1_dose_response(sessions, mode, ctx, n_surr, max_pairs=args.max_test1_pairs,
                                     executor=pool)
            print(f'  Test 1: {time.time() - t0:.1f}s — pass: {t1["pass"]}')
            t2 = test2_contrast(real, sessions)
            print(f'  Test 2: pass: {t2["pass"]}')
            t0 = time.time()
            t3 = test3_pseudo_null(sessions, real, mode, ctx, n_surr,
                                   max_pairs=args.max_test3_pairs, executor=pool)
            print(f'  Test 3: {time.time() - t0:.1f}s — pass: {t3["pass"]} '
                  f'(Δ = {_fmt(t3["delta"])})')
            t4 = test4_redundancy(real, ref_ddtw, block_len)
            print(f'  Test 4: r vs DDTW = {_fmt(t4["mean_r_ddtw"])}, '
                  f'r vs V11 = {_fmt(t4["mean_r_v11"])}')
            per_mode[mode] = {'t1': t1, 't2': t2, 't3': t3, 't4': t4}

            auc_tables.append(t1['auc_table'].assign(mode=mode))
            contrast_tables.append(t2['table'].assign(mode=mode))
            level_src = t3.get('level_source', T3_LEVEL_SOURCE.get(mode, 'z'))
            null_rows += [{'mode': mode, 'sid': sid, 'real_mean_z': v,
                           'level_source': level_src}
                          for sid, v in t3.get('real_per_session', {}).items()]
            null_rows += [{'mode': mode, 'sid': f'pseudo_{i}', 'real_mean_z': np.nan,
                           'pseudo_mean_z': v, 'level_source': level_src}
                          for i, v in enumerate(t3.get('pseudo_means', []))]
            redundancy_tables.append(t4['table'].assign(mode=mode))
            cond_tables.append(condition_summary(real, sessions, mode, block_len))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    pd.concat(auc_tables, ignore_index=True).to_csv(out_dir / 'phase1_semisynthetic.csv', index=False)
    pd.concat(contrast_tables, ignore_index=True).to_csv(
        out_dir / 'phase1_condition_contrast.csv', index=False)
    pd.DataFrame(null_rows).to_csv(out_dir / 'phase1_pseudo_null.csv', index=False)
    pd.concat(redundancy_tables, ignore_index=True).to_csv(
        out_dir / 'phase1_redundancy.csv', index=False)
    pd.concat(cond_tables, ignore_index=True).to_csv(
        out_dir / 'phase1_condition_summary.csv', index=False)

    # ── Stage 4: decision + report ──
    decision, reason = decide(per_mode, phase0_decision)
    print('\n=== Decision ===')
    for mode, t in per_mode.items():
        print(f'  {mode:12s}: T1 {t["t1"]["pass"]} | T2 {t["t2"]["pass"]} | T3 {t["t3"]["pass"]}')
    print(f'  -> MVP pose channel: {decision} ({reason})')
    report_path = write_phase1_report(out_dir, sessions, per_mode, decision, reason, n_surr,
                                      phase0_decision, block_len)
    print(f'\n[4/4] Report: {report_path}')


if __name__ == '__main__':
    main()
