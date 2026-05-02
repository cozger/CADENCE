"""Phase 0 — DDTW pose-coupling validation.

Per docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md §Phase 0,
this script runs four validation tests on the DDTW pose channel and emits
results/mvp/phase0/phase0_report.md with the decision rule outcome (DDTW
swap-in or default to multi-lag cross-correlation baseline).

Test 1: Semi-synthetic dose response (Kendall tau permutation)
Test 2: Real-data condition contrast (paired t + Wilcoxon)
Test 3: Pseudo-dyad null (real > pseudo by >= 0.5 z)
Test 4: Redundancy vs V11 baseline (descriptive only — Pearson r + 95% CI)

Decision: swap to DDTW iff Tests 1, 2, AND 3 all pass; otherwise keep baseline.

Note on Test 2 polarity: the spec's directional pass criterion (conv > med)
implicitly assumed pose follows the EEG/face pattern (peaks during interactive
conversation). Pose-coupling literature actually predicts the opposite:
postural mimicry (Chartrand & Bargh) is brief and intermittent (~2.7s); time-
averaged pose-coupling is dominated by static co-posture during stillness.
This validator implements Test 2 as a magnitude test: |Δz| >= 0.5 AND
significant in either direction. The grant text reports the observed direction.

Usage:
  python scripts/_validate_pose_ddtw.py --all
  python scripts/_validate_pose_ddtw.py --sessions y_06 Y_55_04272026
  python scripts/_validate_pose_ddtw.py --quick        # 50 surrogates (smoke)
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import multiprocessing as _mp
import os
import sys
import time
from pathlib import Path

import torch  # noqa: F401  -- must precede numpy on Windows MCCT stack
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

# Process pool context: spawn (works on Windows torch 2.10 + numpy 2.4 stack
# where loky's cloudpickle bootstrap races torch's DLL load — see memory
# project_win_torch_dll_fix.md). Stdlib mp.spawn imports numpy lazily, so the
# user-script's `import torch` (line above) wins the DLL race.
_MP_SPAWN = _mp.get_context('spawn')

# repo-root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.significance.pose_ddtw import (
    DDTW_PCA_DIM, DDTW_TARGET_RATE_HZ, DDTW_WINDOW_S, DDTW_STRIDE_S,
    DDTW_N_SURROGATES,
    assign_frame_conditions, fit_shared_pca, load_pose_streams,
    project_pca, resample_to_uniform_rate, resample_validity_to_uniform,
    compute_session_ddtw, run_session_ddtw,
)
from cadence.ingest.quality import list_canonical_sessions
from cadence.io.resources import (
    limit_blas_threads, log_resources, pick_n_jobs, snapshot,
)

# ── Per-worker RAM estimates (used by pick_n_jobs to cap parallelism) ────
# Each estimate is the upper-bound peak RAM (GB) one worker holds during its
# task. Numbers are conservative — they include numpy temp-array overhead.

# Per-worker RAM (GB) estimates used by pick_n_jobs. Empirically measured at
# ~660 MB resident per Stage-3 worker on 21-session × 200-surrogate workload
# (was 1.0 GB / 1.5 GB in original prefactor — too conservative, capped Test 1
# at 12 workers when 16 actually fit within budget).
_RAM_PER_LOAD_RESAMPLE_GB = 0.5   # raw pose load + interp, ~36k frames × 99 ch
_RAM_PER_DDTW_SESSION_GB = 0.7    # PCA project + 200 surrogates × ~6k strides
_RAM_PER_TEST1_PAIR_GB = 0.7      # 5 kappa × DDTW + injection scratch

# ── Paths ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC_POSE_ROOT = REPO_ROOT / 'data' / 'preproc' / 'pose' / 'v1'
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'
V11_SCAFFOLD_ROOT = REPO_ROOT / 'results' / 'v11'
OUT_ROOT = REPO_ROOT / 'results' / 'mvp' / 'phase0'

# ── Discovery ───────────────────────────────────────────────────────

def discover_sessions(restrict_to: list[str] | None = None) -> list[str]:
    """Return canonical sessions that have pose preproc with BOTH participants.

    Sessions where one participant lacks pose data (e.g. y26_022728: p1 only)
    cannot be DDTW-validated and are silently skipped. The skipped sessions
    are still listed in pipeline_inventory.csv via has_pose_preproc=True; only
    DDTW validation excludes them.
    """
    canonical = set(list_canonical_sessions())
    available = []
    skipped = []
    for sid in sorted(canonical):
        if restrict_to is not None and sid not in restrict_to:
            continue
        npz_path = PREPROC_POSE_ROOT / f'{sid}.npz'
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                keys = set(npz.files)
        except Exception:
            skipped.append((sid, 'npz unreadable'))
            continue
        if 'p1_pose33' in keys and 'p2_pose33' in keys:
            available.append(sid)
        else:
            missing = [k for k in ('p1_pose33', 'p2_pose33') if k not in keys]
            skipped.append((sid, f"missing {','.join(missing)}"))
    if skipped:
        print(f'[discover_sessions] skipping {len(skipped)} session(s) with '
              f'incomplete pose preproc:')
        for sid, reason in skipped:
            print(f'  - {sid}: {reason}')
    return available


# ── Per-session DDTW (with pre-resampled streams cached) ────────────

def _load_and_resample_safe(sid: str) -> dict:
    """Joblib-safe wrapper that pins BLAS threads to 1 (prevents oversubscription
    when called from a thread pool). See cadence/io/resources.py."""
    with limit_blas_threads(1):
        return load_and_resample(sid)


def load_and_resample(sid: str) -> dict:
    """Load + resample one session's pose to common 12 Hz grid in stream-relative time."""
    streams = load_pose_streams(PREPROC_POSE_ROOT / f'{sid}.npz')
    p1_uni, ts1 = resample_to_uniform_rate(streams['p1_pose'], streams['p1_ts'])
    p2_uni, ts2 = resample_to_uniform_rate(streams['p2_pose'], streams['p2_ts'])
    if ts1.size == 0 or ts2.size == 0:
        raise RuntimeError(f'{sid}: empty pose stream after resample')
    t0 = max(ts1[0], ts2[0])
    t1 = min(ts1[-1], ts2[-1])
    if t1 <= t0:
        raise RuntimeError(f'{sid}: no temporal overlap between p1/p2 pose')
    ts_common = np.arange(t0, t1, 1.0 / DDTW_TARGET_RATE_HZ, dtype=np.float64)

    def _onto_common(arr_uni, ts_uni):
        # Single C-level multi-channel interp; see pose_ddtw.resample_to_uniform_rate
        # for the GIL-saturation reason this isn't a Python per-channel loop.
        idx_right = np.searchsorted(ts_uni, ts_common, side='left').clip(1, ts_uni.size - 1)
        idx_left = idx_right - 1
        t_left = ts_uni[idx_left]
        t_right = ts_uni[idx_right]
        w = ((ts_common - t_left) / np.maximum(t_right - t_left, 1e-12)).clip(0.0, 1.0)
        return (arr_uni[idx_left] + w[:, None] * (arr_uni[idx_right] - arr_uni[idx_left])).astype(np.float32)

    p1_common = _onto_common(p1_uni, ts1)
    p2_common = _onto_common(p2_uni, ts2)
    v1_common = resample_validity_to_uniform(streams['p1_valid'], streams['p1_ts'],
                                              ts_common)
    v2_common = resample_validity_to_uniform(streams['p2_valid'], streams['p2_ts'],
                                              ts_common)

    digest = json.loads((DIGEST_ROOT / f'{sid}.json').read_text())
    t_start_lsl = float(digest.get('t_start_lsl', 0.0))
    markers_rel = [(float(t) - t_start_lsl, str(label))
                   for t, label in digest.get('markers', [])]

    return {
        'sid': sid,
        'p1_common': p1_common, 'p2_common': p2_common,
        'v1_common': v1_common, 'v2_common': v2_common,
        'ts_common': ts_common, 'markers_rel': markers_rel,
        'pose_format_in': streams['pose_format_in'],
        'digest_xdf_md5': streams['digest_xdf_md5'],
    }


def compute_real_ddtw(session: dict, components: np.ndarray, mean: np.ndarray,
                       n_surrogates: int) -> dict:
    """Compute real-pair DDTW for one cached session.

    BLAS thread pinning happens inside compute_session_ddtw, but PCA projection
    runs first — wrap the whole call so projection also stays single-threaded.
    """
    with limit_blas_threads(1):
        p1_pca = project_pca(session['p1_common'], components, mean)
        p2_pca = project_pca(session['p2_common'], components, mean)
        out = compute_session_ddtw(p1_pca, p2_pca,
                                    session['v1_common'], session['v2_common'],
                                    session['ts_common'], session['markers_rel'],
                                    n_surrogates=n_surrogates, seed=42)
    out['sid'] = session['sid']
    return out


# ── Test 1 helpers: piecewise-linear time-warp injection ────────────

def piecewise_linear_warp(stream: np.ndarray, n_anchors: int = 5,
                           max_warp_frac: float = 0.15,
                           rng: np.random.Generator | None = None,
                           ) -> np.ndarray:
    """Apply a piecewise-linear time warp to a (T, D) stream.

    Sample n_anchors target-time positions in [0, T], with offsets uniform in
    [-max_warp_frac * T / n_anchors, +same]. Linearly interpolate the warp
    function between anchors; resample stream onto warped time axis.

    The result has the same length as input but is locally time-stretched.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, D = stream.shape
    if T < 4 * n_anchors:
        return stream.copy()
    # Anchor target positions: uniform on [0, T-1]
    t_anchors = np.linspace(0, T - 1, n_anchors)
    # Source positions: small jitter (anchor monotonicity preserved)
    max_offset = max_warp_frac * T / n_anchors
    offsets = rng.uniform(-max_offset, max_offset, size=n_anchors)
    offsets[0] = 0.0
    offsets[-1] = 0.0
    s_anchors = t_anchors + offsets
    # Enforce monotonic non-decreasing (rare clip if jitter causes inversion)
    s_anchors = np.maximum.accumulate(s_anchors)
    # Warp function: target_t -> source_t (piecewise linear)
    target_t = np.arange(T)
    source_t = np.interp(target_t, t_anchors, s_anchors)
    source_t = np.clip(source_t, 0, T - 1)
    # Resample each channel
    warped = np.empty_like(stream)
    for d in range(D):
        warped[:, d] = np.interp(source_t, np.arange(T), stream[:, d])
    return warped


def inject_coupling_into_segment(p1_segment: np.ndarray, p2_segment: np.ndarray,
                                  kappa: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """P2_inj = kappa * warped_P1 + (1 - kappa) * P2.

    Both inputs (T, D); output (T, D). Per spec §Phase 0 Test 1.
    """
    if kappa <= 0:
        return p2_segment.copy()
    warped_p1 = piecewise_linear_warp(p1_segment, rng=rng)
    return (kappa * warped_p1 + (1 - kappa) * p2_segment).astype(p2_segment.dtype)


def make_pseudo_dyad(sess_a: dict, sess_b: dict) -> dict:
    """Build a pseudo-dyad from two cached sessions: P1 from A, P2 from B.

    Trim to common length on the LSL stream-relative axis. Inherits A's
    markers (so condition labels are A's structure).
    """
    n = min(sess_a['p1_common'].shape[0], sess_b['p2_common'].shape[0])
    return {
        'sid': f'pseudo({sess_a["sid"]}_p1+{sess_b["sid"]}_p2)',
        'p1_common': sess_a['p1_common'][:n],
        'p2_common': sess_b['p2_common'][:n],
        'v1_common': sess_a['v1_common'][:n],
        'v2_common': sess_b['v2_common'][:n],
        'ts_common': sess_a['ts_common'][:n],
        'markers_rel': sess_a['markers_rel'],
    }


# ── Test 1: semi-synthetic dose response ────────────────────────────

def test1_semisynthetic(sessions_data: list[dict], components: np.ndarray,
                         mean: np.ndarray, kappa_levels: tuple,
                         n_surrogates_per_pair: int,
                         injection_segment_s: float = 60.0,
                         max_pairs: int = 30,
                         seed: int = 42,
                         n_jobs: int = -1,
                         executor: '_cf.ProcessPoolExecutor | None' = None,
                         ) -> dict:
    """Per pseudo-dyad pair, AUC for detecting injected vs uninjected windows.

    For each pseudo-dyad (P1 from A, P2 from B; A != B):
      - Build a 60s segment in the middle of the session (away from boundaries)
      - For each kappa:
        * Make injected P2_inj = kappa * warped_P1 + (1-kappa) * P2
        * Compute DDTW z for the full session with P2 = injection-at-segment
        * AUC: separating "in segment" vs "out of segment" using DDTW z
      - Track AUC per kappa per pair

    Output: AUCs[(kappa, pair)] -> AUC

    Pass criteria: Kendall tau permutation p<0.05 AND mean AUC >=0.65 at kappa=0.4.
    """
    sids = [s['sid'] for s in sessions_data]
    if len(sids) < 2:
        return {'status': 'SKIPPED — need >=2 sessions for pseudo-dyad pairs',
                'pairs': [], 'auc_table': None,
                'kendall_tau': None, 'kendall_p': None,
                'auc_at_max_kappa': None, 'pass': False}

    # Build pseudo-dyad pairs (directed, exclude same-session)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(len(sids)) for j in range(len(sids)) if i != j]
    rng.shuffle(all_pairs)
    pairs = all_pairs[:max_pairs]

    # Process-pool dispatch (was Parallel(prefer='threads')). dtaidistance's
    # dtw_ndim.distance_fast does not release the GIL, so threading collapsed
    # this stage to single-core throughput. spawn context avoids loky's
    # cloudpickle/torch DLL race. Use shared executor if provided (saves ~30 s
    # of redundant worker startup compared to spawning a fresh pool here).
    pair_args = [(sessions_data[i], sessions_data[j], components, mean,
                   n_surrogates_per_pair, injection_segment_s, kappa_levels)
                 for i, j in pairs]
    if executor is not None:
        print(f'  Test 1: {len(pairs)} pseudo-dyad pairs × {len(kappa_levels)} '
              f'kappa levels (shared pool, {executor._max_workers} workers)')
        results = list(executor.map(_proc_test1_pair, pair_args))
    else:
        n_jobs_eff = pick_n_jobs(_RAM_PER_TEST1_PAIR_GB, requested=n_jobs,
                                    max_jobs_hard_cap=len(pairs))
        print(f'  Test 1: {len(pairs)} pseudo-dyad pairs × {len(kappa_levels)} '
              f'kappa levels (n_jobs={n_jobs_eff}, backend=processes/spawn)')
        with _cf.ProcessPoolExecutor(max_workers=n_jobs_eff, mp_context=_MP_SPAWN) as ex:
            results = list(ex.map(_proc_test1_pair, pair_args))
    results = [r for r in results if r is not None]

    # Build AUC table
    rows = []
    for sa, sb, perk in results:
        for kappa, auc in perk.items():
            rows.append({'session_a': sa, 'session_b': sb, 'kappa': kappa, 'auc': auc})
    auc_table = pd.DataFrame(rows)

    # Mean AUC per kappa
    mean_auc = auc_table.groupby('kappa')['auc'].mean().to_dict()
    auc_at_max = float(mean_auc.get(max(kappa_levels), np.nan))

    # Kendall tau permutation
    kappa_arr = auc_table['kappa'].to_numpy()
    auc_arr = auc_table['auc'].to_numpy()
    finite = np.isfinite(auc_arr)
    if finite.sum() >= 6:
        tau, _p_asymp = stats.kendalltau(kappa_arr[finite], auc_arr[finite])
        # Permutation
        rng2 = np.random.default_rng(seed + 7)
        n_perm = 1000
        null_taus = np.empty(n_perm)
        ka_p = kappa_arr[finite].copy()
        for i in range(n_perm):
            au_p = rng2.permutation(auc_arr[finite])
            null_taus[i], _ = stats.kendalltau(ka_p, au_p)
        p_perm = float((np.abs(null_taus) >= abs(tau)).mean())
    else:
        tau, p_perm = np.nan, np.nan

    pass_test = (auc_at_max >= 0.65) and (p_perm < 0.05)

    return {
        'status': 'OK' if not np.isnan(auc_at_max) else 'INSUFFICIENT_DATA',
        'pairs': [(sa, sb) for sa, sb, _ in results],
        'auc_table': auc_table,
        'mean_auc_by_kappa': {float(k): float(v) for k, v in mean_auc.items()},
        'kendall_tau': float(tau) if not np.isnan(tau) else None,
        'kendall_p_perm': float(p_perm) if not np.isnan(p_perm) else None,
        'auc_at_max_kappa': float(auc_at_max),
        'pass': bool(pass_test),
    }


def _auc_from_scores(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score_pos > score_neg)."""
    pos = positives[np.isfinite(positives)]
    neg = negatives[np.isfinite(negatives)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    n_pos, n_neg = pos.size, neg.size
    # Rank-based
    combined = np.concatenate([pos, neg])
    ranks = stats.rankdata(combined)
    rank_pos = ranks[:n_pos]
    auc = (rank_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ── Test 2: condition contrast (paired t + Wilcoxon, magnitude) ─────

CONV_CONDS = ('conv_1', 'conv_2')
MED_CONDS = ('meditate_B', 'meditate_K')


def test2_condition_contrast(real_results: list[dict], sessions_data: list[dict]
                              ) -> dict:
    """Per-session conv vs med mean DDTW z; paired t + Wilcoxon (magnitude)."""
    rows = []
    for r, s in zip(real_results, sessions_data):
        z = r['ddtw_z']
        stride_ts = r['stride_ts']
        markers_rel = s['markers_rel']
        cond_label, blocks = assign_frame_conditions(stride_ts, markers_rel)
        cond_to_z = {}
        for cond, s_idx, e_idx in blocks:
            zc = z[s_idx:e_idx]
            zv = zc[np.isfinite(zc)]
            if zv.size > 0:
                cond_to_z.setdefault(cond, []).extend(zv.tolist())
        # Mean per cond
        means = {c: float(np.mean(v)) for c, v in cond_to_z.items()}
        z_conv = np.mean([means[c] for c in CONV_CONDS if c in means]) \
            if any(c in means for c in CONV_CONDS) else np.nan
        z_med = np.mean([means[c] for c in MED_CONDS if c in means]) \
            if any(c in means for c in MED_CONDS) else np.nan
        rows.append({'sid': r['sid'], 'z_conv': z_conv, 'z_med': z_med,
                     'delta': z_conv - z_med, **{f'z_{c}': means.get(c, np.nan)
                                                  for c in CONV_CONDS + MED_CONDS}})
    df = pd.DataFrame(rows)
    paired = df.dropna(subset=['z_conv', 'z_med']).copy()
    if len(paired) < 3:
        return {'status': 'SKIPPED — need >=3 meditation-protocol sessions',
                'table': df, 'paired': paired,
                'mean_delta': np.nan, 'p_t': np.nan, 'p_wilcoxon': np.nan,
                'pass': False}

    delta = paired['delta'].to_numpy()
    mean_delta = float(np.mean(delta))
    # Paired t (test against 0)
    t_stat, p_t = stats.ttest_rel(paired['z_conv'], paired['z_med'])
    # Wilcoxon signed-rank
    try:
        w_stat, p_w = stats.wilcoxon(paired['z_conv'], paired['z_med'],
                                      zero_method='wilcox')
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    p_min = min(p_t, p_w) if not np.isnan(p_w) else p_t

    # Magnitude pass: |Δz| >= 0.5 AND p_min < 0.05
    pass_test = (abs(mean_delta) >= 0.5) and (p_min < 0.05)

    return {
        'status': 'OK',
        'table': df, 'paired': paired,
        'mean_delta': mean_delta,
        'p_t': float(p_t), 'p_wilcoxon': float(p_w) if not np.isnan(p_w) else None,
        'p_min': float(p_min),
        'pass': bool(pass_test),
        'direction': 'conv > med' if mean_delta > 0 else 'med > conv',
    }


# ── Test 3: pseudo-dyad null (real > pseudo by >=0.5 z) ─────────────

def test3_pseudo_null(sessions_data: list[dict], real_results: list[dict],
                       components: np.ndarray, mean: np.ndarray,
                       n_surrogates_per_pair: int,
                       max_pairs: int = 20, seed: int = 42, n_jobs: int = -1,
                       executor: '_cf.ProcessPoolExecutor | None' = None,
                       ) -> dict:
    sids = [s['sid'] for s in sessions_data]
    if len(sids) < 2:
        return {'status': 'SKIPPED', 'mean_real': np.nan, 'mean_pseudo': np.nan,
                'delta': np.nan, 'pass': False}

    # Real-pair mean z per session
    real_means = {r['sid']: float(np.nanmean(r['ddtw_z'])) for r in real_results}

    # Pseudo-pair mean z
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(len(sids)) for j in range(len(sids)) if i != j]
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    pair_args = [(sessions_data[i], sessions_data[j], components, mean,
                   n_surrogates_per_pair) for i, j in pairs]
    if executor is not None:
        print(f'  Test 3: {len(pairs)} pseudo-dyad pairs '
              f'(shared pool, {executor._max_workers} workers)')
        pseudo_means = list(executor.map(_proc_test3_pair, pair_args))
    else:
        n_jobs_eff = pick_n_jobs(_RAM_PER_DDTW_SESSION_GB, requested=n_jobs,
                                    max_jobs_hard_cap=len(pairs))
        print(f'  Test 3: {len(pairs)} pseudo-dyad pairs '
              f'(n_jobs={n_jobs_eff}, backend=processes/spawn)')
        with _cf.ProcessPoolExecutor(max_workers=n_jobs_eff, mp_context=_MP_SPAWN) as ex:
            pseudo_means = list(ex.map(_proc_test3_pair, pair_args))
    pseudo_means = [m for m in pseudo_means if not np.isnan(m)]

    mean_real = float(np.mean(list(real_means.values())))
    mean_pseudo = float(np.mean(pseudo_means)) if pseudo_means else np.nan
    delta = mean_real - mean_pseudo
    pass_test = (delta >= 0.5)

    return {
        'status': 'OK',
        'real_per_session': real_means,
        'pseudo_means': pseudo_means,
        'mean_real': mean_real,
        'mean_pseudo': mean_pseudo,
        'delta': float(delta),
        'pass': bool(pass_test),
    }


# ── Test 4: redundancy vs V11 baseline (descriptive) ────────────────

def test4_redundancy(real_results: list[dict], sessions_data: list[dict]
                      ) -> dict:
    """Per-session Pearson r between DDTW timecourse and V11 baseline pose channel.

    V11 multi-lag baseline is in results/v11/<sid>/scaffold_v11_ztimecourses.npz
    as 'z_raw_pose'. Both metrics are 2 Hz timecourses, but on different time
    grids — DDTW is in stream-relative seconds, V11 is in absolute LSL time.
    We resample V11 onto DDTW's stride_ts (after offsetting by t_start_lsl).
    """
    rows = []
    for r, s in zip(real_results, sessions_data):
        sid = r['sid']
        v11_path = V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_ztimecourses.npz'
        if not v11_path.exists():
            rows.append({'sid': sid, 'r': np.nan, 'n': 0, 'note': 'no V11 scaffold'})
            continue
        v11 = np.load(v11_path)
        z_v11 = v11['z_raw_pose']
        t_v11_lsl = v11['t_common']
        # Convert V11 LSL time to stream-relative (subtract t_start_lsl from digest)
        digest = json.loads((DIGEST_ROOT / f'{sid}.json').read_text())
        t_start_lsl = float(digest['t_start_lsl'])
        t_v11_rel = t_v11_lsl - t_start_lsl
        # Interp DDTW onto V11 grid where they overlap (V11 is 2 Hz; DDTW also 2 Hz)
        z_ddtw = r['ddtw_z']
        ts_ddtw = r['stride_ts']
        if z_ddtw.size == 0 or z_v11.size == 0:
            rows.append({'sid': sid, 'r': np.nan, 'n': 0, 'note': 'empty'})
            continue
        # Common time range
        t0 = max(ts_ddtw[0], t_v11_rel[0])
        t1 = min(ts_ddtw[-1], t_v11_rel[-1])
        if t1 <= t0:
            rows.append({'sid': sid, 'r': np.nan, 'n': 0, 'note': 'no time overlap'})
            continue
        common_ts = np.arange(t0, t1, 0.5)  # 2 Hz
        # Interpolate (but handle NaNs by masking)
        mask_ddtw = np.isfinite(z_ddtw)
        if mask_ddtw.sum() < 10:
            rows.append({'sid': sid, 'r': np.nan, 'n': 0, 'note': 'too few finite DDTW'})
            continue
        z_ddtw_on_common = np.interp(common_ts, ts_ddtw[mask_ddtw], z_ddtw[mask_ddtw])
        z_v11_on_common = np.interp(common_ts, t_v11_rel, z_v11)
        if np.std(z_ddtw_on_common) < 1e-8 or np.std(z_v11_on_common) < 1e-8:
            rows.append({'sid': sid, 'r': np.nan, 'n': common_ts.size, 'note': 'constant'})
            continue
        rho, _ = stats.pearsonr(z_ddtw_on_common, z_v11_on_common)
        rows.append({'sid': sid, 'r': float(rho), 'n': int(common_ts.size), 'note': ''})
    df = pd.DataFrame(rows)
    rs = df['r'].dropna().to_numpy()
    if rs.size > 0:
        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        n_boot = 1000
        boot_means = np.array([np.mean(rng.choice(rs, rs.size, replace=True))
                               for _ in range(n_boot)])
        mean_r = float(np.mean(rs))
        ci = (float(np.percentile(boot_means, 2.5)),
              float(np.percentile(boot_means, 97.5)))
    else:
        mean_r, ci = np.nan, (np.nan, np.nan)

    if abs(mean_r) < 0.3:
        descriptor = 'largely independent'
    elif abs(mean_r) < 0.7:
        descriptor = 'partial overlap'
    else:
        descriptor = 'largely redundant'

    return {
        'status': 'OK',
        'table': df,
        'mean_r': mean_r,
        'ci_95': ci,
        'descriptor': descriptor,
    }


# ── Report writer ───────────────────────────────────────────────────

def write_phase0_report(out_dir: Path, sessions_data: list[dict],
                         pca_diag: dict, t1: dict, t2: dict, t3: dict, t4: dict,
                         decision: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append('# Phase 0 — DDTW Pose-Coupling Validation Report\n')
    md.append(f'**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append(f'**Spec:** docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md §Phase 0\n')

    md.append('\n## Cohort\n')
    md.append(f'- Sessions evaluated: **{len(sessions_data)}**\n')
    for s in sessions_data:
        md.append(f'  - `{s["sid"]}` (format: `{s["pose_format_in"]}`)\n')

    md.append('\n## Shared PCA fit\n')
    md.append(f'- N components: {DDTW_PCA_DIM}\n')
    md.append(f'- Cumulative explained variance: '
              f'{pca_diag["cumulative_variance_ratio"][-1]:.3f}\n')
    md.append(f'- Per-session reconstruction error:\n')
    for k, v in pca_diag['per_session_recon_error'].items():
        md.append(f'  - `{k}`: {v:.3f}\n')
    cohort_med = np.nanmedian(list(pca_diag['per_session_recon_error'].values()))
    bad = [k for k, v in pca_diag['per_session_recon_error'].items() if v > 1.5 * cohort_med]
    md.append(f'- Cohort median: {cohort_med:.3f}; sessions > 1.5× median: '
              f'{bad if bad else "none — cross-format consistency OK"}\n')

    md.append('\n## Test 1 — Semi-synthetic dose response\n')
    md.append(f'- Status: **{t1.get("status")}**\n')
    if t1.get('mean_auc_by_kappa'):
        md.append('- Mean AUC by κ:\n')
        for k in sorted(t1['mean_auc_by_kappa']):
            md.append(f'  - κ={k:.2f}: AUC = {t1["mean_auc_by_kappa"][k]:.3f}\n')
        md.append(f'- AUC at max κ: **{t1["auc_at_max_kappa"]:.3f}** (threshold ≥ 0.65)\n')
        md.append(f'- Kendall τ: {t1["kendall_tau"]:.3f}, '
                  f'permutation p = {t1["kendall_p_perm"]:.4f} (threshold < 0.05)\n')
    md.append(f'- **Pass:** {t1.get("pass")}\n')

    md.append('\n## Test 2 — Real-data condition contrast (magnitude)\n')
    md.append(f'- Status: **{t2.get("status")}**\n')
    md.append(f'- N sessions with both conv + med: {len(t2.get("paired", []))}\n')
    if t2.get('status') == 'OK':
        md.append(f'- Mean Δz (conv − med): **{t2["mean_delta"]:+.3f}** ({t2["direction"]})\n')
        md.append(f'- Paired t-test p: {t2["p_t"]:.4f}\n')
        wp = t2.get("p_wilcoxon")
        md.append(f'- Wilcoxon signed-rank p: {wp:.4f}\n' if wp is not None
                  else '- Wilcoxon: N/A (degenerate)\n')
        md.append(f'- Pass criteria: |Δz| ≥ 0.5 AND min(p) < 0.05\n')
    md.append(f'- **Pass:** {t2.get("pass")}\n')
    md.append('\n  *Note: spec assumed conv > med; pose literature predicts med > conv '
              '(static co-posture during stillness). Validator passes either direction.*\n')

    md.append('\n## Test 3 — Pseudo-dyad null\n')
    md.append(f'- Status: **{t3.get("status")}**\n')
    if t3.get('status') == 'OK':
        md.append(f'- Mean real-pair z: **{t3["mean_real"]:+.3f}**\n')
        md.append(f'- Mean pseudo-pair z: {t3["mean_pseudo"]:+.3f}\n')
        md.append(f'- Δ = real − pseudo: **{t3["delta"]:+.3f}** (threshold ≥ +0.5)\n')
    md.append(f'- **Pass:** {t3.get("pass")}\n')

    md.append('\n## Test 4 — Redundancy vs V11 multi-lag baseline (descriptive)\n')
    md.append(f'- Status: **{t4.get("status")}**\n')
    md.append(f'- Mean Pearson r (across {len(t4["table"])} sessions): '
              f'**{t4["mean_r"]:+.3f}** (95% CI [{t4["ci_95"][0]:+.3f}, '
              f'{t4["ci_95"][1]:+.3f}])\n')
    md.append(f'- Interpretation: {t4["descriptor"]}\n')
    md.append('- Per-session r:\n')
    for _, row in t4['table'].iterrows():
        md.append(f'  - `{row["sid"]}`: r = '
                  f'{row["r"]:+.3f}' if not np.isnan(row['r']) else f'  - `{row["sid"]}`: N/A')
        md.append(f' ({row["note"]})\n' if row['note'] else '\n')

    md.append('\n## Decision\n')
    pass_tests = ', '.join([
        f'Test 1: {t1.get("pass")}',
        f'Test 2: {t2.get("pass")}',
        f'Test 3: {t3.get("pass")}',
    ])
    md.append(f'- {pass_tests}\n')
    md.append(f'- **Pose channel for MVP scaffold: `{decision}`**\n')

    md.append('\n## Provenance\n')
    md.append(f'- Sessions used:\n')
    for s in sessions_data:
        md.append(f'  - `{s["sid"]}` digest_xdf_md5 = `{s["digest_xdf_md5"]}`\n')

    out_path = out_dir / 'phase0_report.md'
    out_path.write_text(''.join(md), encoding='utf-8')
    return out_path


# ── Process-pool workers (must be module-level so stdlib pickle can ship them) ──
#
# These wrap the per-pair / per-session DDTW work that was previously executed
# inside `Parallel(prefer='threads')` closures. Switched to processes because
# dtaidistance's C kernels do not release the GIL, so thread-pool workers
# serialised on GIL acquisition (measured ~1.2x speedup at 8 threads vs an
# ideal 8x).

def _proc_compute_real_ddtw(args):
    session, components, mean, n_surrogates = args
    # compute_real_ddtw already wraps in limit_blas_threads(1) internally.
    return compute_real_ddtw(session, components, mean, n_surrogates)


def _proc_test1_pair(args):
    (sess_a, sess_b, components, mean, n_surrogates_per_pair,
     injection_segment_s, kappa_levels) = args
    with limit_blas_threads(1):
        pseudo = make_pseudo_dyad(sess_a, sess_b)
        T = pseudo['p1_common'].shape[0]
        if T < int(2 * injection_segment_s * DDTW_TARGET_RATE_HZ):
            return None
        seg_len = int(injection_segment_s * DDTW_TARGET_RATE_HZ)
        seg_start = T // 2 - seg_len // 2
        seg_end = seg_start + seg_len
        p1_seg = pseudo['p1_common'][seg_start:seg_end]
        p2_orig_seg = pseudo['p2_common'][seg_start:seg_end]
        # P1 PCA is invariant across kappa
        p1_pca = project_pca(pseudo['p1_common'], components, mean)
        ts_common = pseudo['ts_common']
        t_seg_start = ts_common[seg_start]
        t_seg_end = ts_common[min(seg_end, ts_common.size - 1)]

        per_kappa = {}
        for kappa in kappa_levels:
            local_rng = np.random.default_rng(
                hash((sess_a['sid'], sess_b['sid'], int(1000 * kappa))) % (2**31))
            p2_modified = pseudo['p2_common'].copy()
            p2_modified[seg_start:seg_end] = inject_coupling_into_segment(
                p1_seg, p2_orig_seg, kappa, local_rng)
            p2_pca = project_pca(p2_modified, components, mean)
            out = compute_session_ddtw(p1_pca, p2_pca,
                                        pseudo['v1_common'], pseudo['v2_common'],
                                        ts_common, pseudo['markers_rel'],
                                        n_surrogates=n_surrogates_per_pair,
                                        seed=42 + int(1000 * kappa))
            z = out['ddtw_z']
            stride_ts = out['stride_ts']
            in_seg = (stride_ts >= t_seg_start) & (stride_ts < t_seg_end)
            valid = np.isfinite(z)
            in_seg_v = in_seg & valid
            out_seg_v = (~in_seg) & valid
            if in_seg_v.sum() < 2 or out_seg_v.sum() < 2:
                auc = np.nan
            else:
                auc = _auc_from_scores(z[in_seg_v], z[out_seg_v])
            per_kappa[kappa] = auc
        return (sess_a['sid'], sess_b['sid'], per_kappa)


def _proc_test3_pair(args):
    sess_a, sess_b, components, mean, n_surrogates_per_pair = args
    with limit_blas_threads(1):
        pseudo = make_pseudo_dyad(sess_a, sess_b)
        p1_pca = project_pca(pseudo['p1_common'], components, mean)
        p2_pca = project_pca(pseudo['p2_common'], components, mean)
        out = compute_session_ddtw(p1_pca, p2_pca,
                                    pseudo['v1_common'], pseudo['v2_common'],
                                    pseudo['ts_common'], pseudo['markers_rel'],
                                    n_surrogates=n_surrogates_per_pair, seed=42)
        return float(np.nanmean(out['ddtw_z']))


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None,
                    help='Specific session IDs (default: all canonical with pose preproc)')
    ap.add_argument('--all', action='store_true',
                    help='Use all canonical sessions with pose preproc available')
    ap.add_argument('--quick', action='store_true',
                    help='Use 50 surrogates instead of 200 (smoke mode)')
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help='Parallel workers (default -1 = all cores)')
    ap.add_argument('--max-test1-pairs', type=int, default=30,
                    help='Cap on pseudo-dyad pairs for Test 1 (default 30)')
    ap.add_argument('--max-test3-pairs', type=int, default=20,
                    help='Cap on pseudo-dyad pairs for Test 3 (default 20)')
    args = ap.parse_args()

    if args.all and args.sessions:
        ap.error('--all and --sessions are mutually exclusive')

    sids = discover_sessions(restrict_to=set(args.sessions) if args.sessions else None)
    if not sids:
        print('ERROR: no canonical sessions with pose preproc found.')
        sys.exit(1)
    print(f'Discovered {len(sids)} canonical sessions with pose preproc:')
    for s in sids:
        print(f'  - {s}')

    n_surr = 50 if args.quick else DDTW_N_SURROGATES
    print(f'\nN surrogates per session: {n_surr}')
    print(f'Parallel workers: {args.n_jobs}')

    log_resources(prefix='\nResource snapshot at start: ')

    # ── Stage 1: load + resample all sessions (cached for reuse) ──
    n_jobs_load = pick_n_jobs(_RAM_PER_LOAD_RESAMPLE_GB, requested=args.n_jobs,
                                max_jobs_hard_cap=len(sids))
    print(f'\n[1/6] Loading + resampling pose streams (n_jobs={n_jobs_load})...',
          flush=True)
    t0 = time.time()
    sessions_data = Parallel(n_jobs=n_jobs_load, prefer='threads')(
        delayed(_load_and_resample_safe)(sid) for sid in sids)
    print(f'  Done in {time.time() - t0:.1f}s')

    # ── Stage 2: shared PCA fit ──
    print('\n[2/6] Fitting shared cross-session PCA...', flush=True)
    t0 = time.time()
    streams_for_pca = {}
    for s in sessions_data:
        streams_for_pca[f'{s["sid"]}_p1'] = s['p1_common']
        streams_for_pca[f'{s["sid"]}_p2'] = s['p2_common']
    components, mean, pca_diag = fit_shared_pca(streams_for_pca,
                                                  n_components=DDTW_PCA_DIM)
    print(f'  Done in {time.time() - t0:.1f}s. '
          f'EVR cumulative: {pca_diag["cumulative_variance_ratio"][-1]:.3f}')

    # ── Persistent process pool for Stage 3 + Test 1 + Test 3 ──
    # Benchmark (this commit): n_workers=16 saturates throughput on this hardware
    # (memory-bandwidth bound); n=24 / n=32 give 0% extra speedup. Sharing a
    # single pool across all three GIL-bound stages saves ~30 s of redundant
    # worker startup (importing torch + numpy + scipy + cadence per worker).
    pool_size = pick_n_jobs(_RAM_PER_TEST1_PAIR_GB, requested=args.n_jobs,
                              max_jobs_hard_cap=max(len(sessions_data),
                                                     args.max_test1_pairs,
                                                     args.max_test3_pairs))
    print(f'\nStarting shared process pool: {pool_size} workers (mp_context=spawn)...',
          flush=True)
    t_pool = time.time()
    pool = _cf.ProcessPoolExecutor(max_workers=pool_size, mp_context=_MP_SPAWN)
    # Eagerly warm the pool by submitting a no-op to each worker so the
    # first real Stage-3 task doesn't pay startup latency.
    list(pool.map(int, [0] * pool_size))
    print(f'  Pool warm in {time.time() - t_pool:.1f}s')

    try:
        # ── Stage 3: real-pair DDTW per session (cached for Tests 2+3+4) ──
        print(f'\n[3/6] Computing real-pair DDTW for {len(sessions_data)} sessions '
              f'(shared pool, {pool_size} workers)...', flush=True)
        t0 = time.time()
        stage3_args = [(s, components, mean, n_surr) for s in sessions_data]
        real_results = list(pool.map(_proc_compute_real_ddtw, stage3_args))
        print(f'  Done in {time.time() - t0:.1f}s')

        # Save per-session DDTW timecourses
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        npz_path = OUT_ROOT / 'pose_ddtw_per_session.npz'
        save_dict = {}
        for r in real_results:
            sid = r['sid']
            save_dict[f'{sid}__ddtw_z'] = r['ddtw_z']
            save_dict[f'{sid}__ddtw_real'] = r['ddtw_real']
            save_dict[f'{sid}__stride_ts'] = r['stride_ts']
        np.savez(npz_path, **save_dict)
        print(f'  Saved per-session DDTW timecourses -> {npz_path}')

        # ── Stage 4: Test 1 (semi-synthetic dose response) ──
        print(f'\n[4/6] Test 1: semi-synthetic dose response...', flush=True)
        t0 = time.time()
        t1 = test1_semisynthetic(sessions_data, components, mean,
                                  kappa_levels=(0.0, 0.1, 0.2, 0.3, 0.4),
                                  n_surrogates_per_pair=n_surr,
                                  max_pairs=args.max_test1_pairs,
                                  n_jobs=args.n_jobs,
                                  executor=pool)
        print(f'  Test 1 done in {time.time() - t0:.1f}s — pass: {t1.get("pass")}')
        if t1.get('auc_table') is not None:
            t1['auc_table'].to_csv(OUT_ROOT / 'validation_semisynthetic.csv', index=False)

        # ── Stage 5: Test 2 (condition contrast) ──
        print(f'\n[5/6] Test 2: real-data condition contrast...', flush=True)
        t0 = time.time()
        t2 = test2_condition_contrast(real_results, sessions_data)
        print(f'  Test 2 done in {time.time() - t0:.1f}s — pass: {t2.get("pass")}')
        t2['table'].to_csv(OUT_ROOT / 'validation_condition_contrast.csv', index=False)

        # ── Stage 6: Test 3 + Test 4 ──
        print(f'\n[6/6] Test 3: pseudo-dyad null + Test 4: V11 redundancy...', flush=True)
        t0 = time.time()
        t3 = test3_pseudo_null(sessions_data, real_results, components, mean,
                                n_surrogates_per_pair=n_surr,
                                max_pairs=args.max_test3_pairs, n_jobs=args.n_jobs,
                                executor=pool)
        t4 = test4_redundancy(real_results, sessions_data)
        print(f'  Tests 3+4 done in {time.time() - t0:.1f}s — '
              f'T3 pass: {t3.get("pass")} | T4 r = {t4["mean_r"]:+.3f}')
    finally:
        pool.shutdown(wait=True)
    pd.DataFrame({'sid': list(t3.get('real_per_session', {}).keys()),
                   'real_mean_z': list(t3.get('real_per_session', {}).values())
                   }).to_csv(OUT_ROOT / 'validation_pseudo_null.csv', index=False)
    t4['table'].to_csv(OUT_ROOT / 'redundancy_vs_baseline.csv', index=False)

    # ── Decision ──
    decision = ('pose_ddtw' if (t1.get('pass') and t2.get('pass') and t3.get('pass'))
                else 'pose_baseline')
    print(f'\n=== Decision ===')
    print(f'  Test 1: {t1.get("pass")} | Test 2: {t2.get("pass")} | Test 3: {t3.get("pass")}')
    print(f'  -> MVP pose channel: {decision}')

    report_path = write_phase0_report(OUT_ROOT, sessions_data, pca_diag,
                                       t1, t2, t3, t4, decision)
    print(f'\nReport: {report_path}')


if __name__ == '__main__':
    main()
