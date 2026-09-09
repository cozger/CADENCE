"""Tests for the Phase 1 pose-channel validation driver and MVP scaffold hooks (Task 5).

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md

Synthetic only (no data files): every per-session function of
``scripts/_validate_pose_channels.py`` runs on in-memory session dicts built
by a small skeleton generator whose joint angles are independent smooth
random walks (pseudo-dyad rule: a null pair is never built from one stream);
the file-backed loaders and the ``_run_mvp_scaffold`` decision / loader
hooks run on temporary fixtures.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
try:
    import torch as _torch  # noqa: F401  (Windows torch/numpy import-order guard)
except ImportError:
    pass
import numpy as np
import pandas as pd
import pytest

from scripts import _run_mvp_scaffold as S
from scripts import _validate_pose_channels as V
from scripts import _validate_pose_ddtw as P0
from cadence.significance.pose_angles import ANGLE_NAMES, angle_features, moving_average_nan
from cadence.significance.pose_ddtw import DDTW_TARGET_RATE_HZ, fit_shared_pca

FS = DDTW_TARGET_RATE_HZ      # 12 Hz common grid
N_SURR = 8                    # keep every DDTW call fast
SECONDS = 150.0


# ── Synthetic sessions ──────────────────────────────────────────────

def smooth_walk(rng: np.random.Generator, n: int, d: int, smooth: int = 12) -> np.ndarray:
    x = np.cumsum(rng.standard_normal((n + 4 * smooth, d)), axis=0)
    x = moving_average_nan(x, smooth)[2 * smooth:2 * smooth + n]
    x = x - x.mean(axis=0)
    return x / x.std(axis=0)


def skeleton_from_drivers(drv: np.ndarray, rng: np.random.Generator,
                          jitter: float = 0.002) -> np.ndarray:
    """(n, 6) joint-angle drivers -> (n, 33, 4) frontal skeleton, all visible."""
    n = drv.shape[0]
    xy = np.zeros((n, 33, 2))
    xy[:, 23] = (-0.15, 0.0); xy[:, 24] = (0.15, 0.0)          # hips
    xy[:, 11] = (-0.20, -1.0); xy[:, 12] = (0.20, -1.0)        # shoulders
    xy[:, 7] = (-0.08, -1.30); xy[:, 8] = (0.08, -1.30)        # ears
    xy[:, 0] = (0.02, -1.24)                                   # nose
    for k in range(1, 7):
        xy[:, k] = (0.03 * (k - 3.5), -1.34)
    xy[:, 9] = (-0.03, -1.20); xy[:, 10] = (0.03, -1.20)

    def seg(base, theta, length):
        return base + length * np.stack([np.sin(theta), np.cos(theta)], axis=1)

    ua_l, fa_l = 0.15 + 0.5 * drv[:, 0], 0.40 + 0.5 * drv[:, 1]
    ua_r, fa_r = -0.15 + 0.5 * drv[:, 2], 0.10 + 0.5 * drv[:, 3]
    th_l, th_r = -0.06 + 0.25 * drv[:, 4], 0.06 + 0.25 * drv[:, 5]
    xy[:, 13] = seg(xy[:, 11], ua_l, 0.35); xy[:, 14] = seg(xy[:, 12], ua_r, 0.35)
    xy[:, 15] = seg(xy[:, 13], fa_l, 0.35); xy[:, 16] = seg(xy[:, 14], fa_r, 0.35)
    for k, base in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        xy[:, k] = xy[:, base] + (0.02 * (k % 3), 0.05)
    xy[:, 25] = seg(xy[:, 23], th_l, 0.5); xy[:, 26] = seg(xy[:, 24], th_r, 0.5)
    xy[:, 27] = xy[:, 25] + (0.0, 0.5); xy[:, 28] = xy[:, 26] + (0.0, 0.5)
    xy[:, 29] = xy[:, 27] + (-0.02, 0.05); xy[:, 31] = xy[:, 27] + (0.05, 0.06)
    xy[:, 30] = xy[:, 28] + (0.02, 0.05); xy[:, 32] = xy[:, 28] + (-0.05, 0.06)
    xy = xy + jitter * rng.standard_normal(xy.shape)
    out = np.zeros((n, 33, 4), dtype=np.float32)
    out[:, :, :2] = xy
    out[:, :, 3] = 1.0
    return out


def delay_frames(x: np.ndarray, d: int) -> np.ndarray:
    return np.vstack([np.repeat(x[:1], d, axis=0), x[:-d]])


def make_session(sid: str, seed: int, coupled: bool, seconds: float = SECONDS,
                 fs: float = FS, delay_s: float = 0.25) -> dict:
    """Session dict in the driver's contract on a uniform ``fs`` grid.

    ``coupled``: P2's joint drivers are P1's delayed by ``delay_s`` plus a
    small independent walk; otherwise P1 / P2 drivers are independent.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs))
    ts = np.arange(n) / fs
    drv1 = smooth_walk(rng, n, 6)
    if coupled:
        drv2 = 0.9 * delay_frames(drv1, int(round(delay_s * fs))) + 0.1 * smooth_walk(rng, n, 6)
    else:
        drv2 = smooth_walk(rng, n, 6)
    p1, p2 = skeleton_from_drivers(drv1, rng), skeleton_from_drivers(drv2, rng)
    markers = [(5.0, 'conv_1_start'), (65.0, 'conv_1_stop'),
               (70.0, 'meditate_B_start'), (140.0, 'meditate_B_stop')]
    return {'sid': sid,
            'p1_common': np.ascontiguousarray(p1[..., :3].reshape(n, -1)),
            'p2_common': np.ascontiguousarray(p2[..., :3].reshape(n, -1)),
            'p1_pose33': p1, 'p2_pose33': p2,
            'v1_common': np.ones(n, dtype=bool), 'v2_common': np.ones(n, dtype=bool),
            'ts_common': ts, 'markers_rel': markers,
            'pose_format_in': 'synthetic', 'digest_xdf_md5': f'md5_{sid}'}


def make_ctx(sessions: list[dict]) -> dict:
    streams = {f'{s["sid"]}_{p}': s[f'{p}_common'] for s in sessions for p in ('p1', 'p2')}
    comps, mean, _ = fit_shared_pca(streams)
    return {'components': comps, 'mean': mean, 'noise_normalize': True, 'smooth_sigma_s': 15.0}


@pytest.fixture(scope='module')
def cohort():
    sessions = [make_session('A', 11, coupled=True), make_session('B', 23, coupled=True)]
    return sessions, make_ctx(sessions)


# ── CLI / syntax gates from the plan ────────────────────────────────

def test_driver_help_and_scaffold_parse():
    proc = subprocess.run([sys.executable, str(REPO / 'scripts' / '_validate_pose_channels.py'),
                           '--help'], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert '--modes' in proc.stdout and 'phase1_pose' in proc.stdout
    ast.parse((REPO / 'scripts' / '_run_mvp_scaffold.py').read_text(encoding='utf-8'))
    assert set(V.DEFAULT_MODES) == {'angles', 'angle_speed', 'evt_landing', 'evt_peak'}
    assert 'pca' in V.ALL_MODES
    assert set(S.POSE_CHANNEL_CHOICES) == {'auto', 'baseline', 'ddtw', 'angles',
                                           'angle_speed', 'evt_landing', 'evt_peak'}


# ── Shared objects: pose33 on the common grid ───────────────────────

def test_pose33_onto_grid_coordinates_and_conservative_visibility():
    rng = np.random.default_rng(0)
    fs_native, seconds = 30.0, 20.0
    n = int(seconds * fs_native)
    ts = np.arange(n) / fs_native
    pose = skeleton_from_drivers(smooth_walk(rng, n, 6), rng)
    pose[200:230, 15, :3] = 0.0            # left wrist hidden for one second
    pose[200:230, 15, 3] = 0.0
    grid = np.arange(ts[0], ts[-1], 1.0 / FS)
    out = V.pose33_onto_grid(pose, ts, grid)
    assert out.shape == (grid.size, 33, 4) and out.dtype == np.float32
    # coordinates: linear interpolation of the native stream
    ref = np.interp(grid, ts, pose[:, 12, 0])
    assert np.allclose(out[:, 12, 0], ref, atol=1e-5)
    # visibility: hidden wherever a bracketing native frame is hidden, else 1
    hidden = (grid >= ts[199]) & (grid <= ts[230])
    assert np.all(out[hidden, 15, 3] == 0.0)
    assert np.all(out[~hidden, 15, 3] == 1.0)
    assert np.all(out[:, 11, 3] == 1.0)


def test_load_and_resample_pose33_from_fixture(tmp_path, monkeypatch):
    sid = 'fx_01'
    rng = np.random.default_rng(5)
    fs_native, seconds = 30.0, 40.0
    n = int(seconds * fs_native)
    arrays = {}
    for p, t_off in (('p1', 0.0), ('p2', 0.07)):
        arrays[f'{p}_pose33'] = skeleton_from_drivers(smooth_walk(rng, n, 6), rng)
        arrays[f'{p}_pose33_ts'] = t_off + np.arange(n) / fs_native
        arrays[f'{p}_pose_features_valid'] = np.ones(n, dtype=bool)
    preproc, digest = tmp_path / 'preproc', tmp_path / 'digest'
    preproc.mkdir(); digest.mkdir()
    np.savez(preproc / f'{sid}.npz', **arrays)
    (preproc / f'{sid}.json').write_text(json.dumps(
        {'digest_xdf_md5': 'abc', 'pose_format_in': 'mediapipe33'}))
    (digest / f'{sid}.json').write_text(json.dumps(
        {'xdf_md5': 'abc', 't_start_lsl': 500.0,
         'markers': [[505.0, 'conv_1_start'], [535.0, 'conv_1_stop']]}))
    monkeypatch.setattr(P0, 'PREPROC_POSE_ROOT', preproc)
    monkeypatch.setattr(P0, 'DIGEST_ROOT', digest)
    monkeypatch.setattr(V, 'PREPROC_POSE_ROOT', preproc)
    sess = V.load_and_resample_pose33(sid)
    m = sess['ts_common'].size
    assert m > 0 and sess['p1_common'].shape == (m, 99)
    for p in ('p1', 'p2'):
        assert sess[f'{p}_pose33'].shape == (m, 33, 4)
        assert np.allclose(sess[f'{p}_pose33'][..., :3].reshape(m, -1), sess[f'{p}_common'],
                           atol=1e-5)
        assert np.all(sess[f'{p}_pose33'][..., 3] == 1.0)
    assert sess['markers_rel'][0] == (5.0, 'conv_1_start')
    assert sess['pose_format_in'] == 'mediapipe33'


# ── Pseudo-dyads and kappa-injection on skeletons ───────────────────

def test_pseudo_dyad_carries_pose33_and_drops_caches(cohort):
    sessions, ctx = cohort
    a = dict(sessions[0]); a['features'] = {'angles': {'p1': ('x', 'y', None)}}
    pseudo = V.make_pseudo_dyad_pose33(a, sessions[1])
    n = pseudo['ts_common'].size
    assert pseudo['p1_pose33'].shape[0] == n and pseudo['p2_pose33'].shape[0] == n
    assert np.array_equal(pseudo['p1_pose33'], sessions[0]['p1_pose33'][:n])
    assert np.array_equal(pseudo['p2_pose33'], sessions[1]['p2_pose33'][:n])
    assert 'features' not in pseudo
    assert pseudo['sid'].startswith('pseudo(')


def test_inject_pose33_segment_identity_mix_and_shared_warp(cohort):
    sessions, _ = cohort
    sess = sessions[0]
    T = 720
    p1, p2 = sess['p1_pose33'][:T], sess['p2_pose33'][:T].copy()
    p2[100:140, 16, 3] = 0.0               # P2 right wrist hidden
    rng = np.random.default_rng(3)
    assert np.array_equal(V.inject_pose33_segment(p1, p2, 0.0, rng), p2)
    rng = np.random.default_rng(3)
    mixed = V.inject_pose33_segment(p1, p2, 0.4, rng)
    assert mixed.shape == p2.shape and mixed.dtype == np.float32
    # coordinates move toward P1 (bounded by the two sources' envelopes)
    lo = np.minimum(p1[..., :2].min(axis=0), p2[..., :2].min(axis=0)) - 0.05
    hi = np.maximum(p1[..., :2].max(axis=0), p2[..., :2].max(axis=0)) + 0.05
    assert np.all(mixed[..., :2] >= lo) and np.all(mixed[..., :2] <= hi)
    assert not np.allclose(mixed[..., :2], p2[..., :2])
    # visibility = min(warped P1 vis, P2 vis): hidden P2 frames stay hidden
    assert np.all(mixed[100:140, 16, 3] == 0.0)
    assert np.all(mixed[:, 11, 3] == 1.0)
    # inject_session: the 99-D path is byte-identical to Phase 0, the skeleton
    # path delegates to inject_pose33_segment with the same RNG state, and for
    # two skeletons already in the same body frame the alignment is a
    # near-identity (jitter-median only), so the two paths still agree closely.
    inj = V.inject_session(sess, 0.4, np.random.default_rng(7), 200, 200 + T)
    expect99 = P0.inject_coupling_into_segment(sess['p1_common'][200:200 + T],
                                               sess['p2_common'][200:200 + T], 0.4,
                                               np.random.default_rng(7))
    assert np.array_equal(inj['p2_common'][200:200 + T], expect99)
    expect33 = V.inject_pose33_segment(sess['p1_pose33'][200:200 + T],
                                       sess['p2_pose33'][200:200 + T], 0.4,
                                       np.random.default_rng(7))
    assert np.array_equal(inj['p2_pose33'][200:200 + T], expect33)
    seg33 = inj['p2_pose33'][200:200 + T, :, :3].reshape(T, -1)
    assert np.allclose(inj['p2_common'][200:200 + T], seg33, atol=2e-3)
    assert np.array_equal(inj['p2_common'][:200], sess['p2_common'][:200])
    assert np.array_equal(inj['p1_pose33'], sess['p1_pose33'])
    assert 'features' not in inj


def _circular_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |wrapped angle difference| over finite entries of two (n, 12) arrays."""
    d = np.angle(np.exp(1j * (a.astype(np.float64) - b.astype(np.float64))))
    return float(np.nanmean(np.abs(d)))


def _scaled_copy(sess: dict, key: str, factor: float) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sess.items()}
    out[key][..., :2] *= factor          # pixel-scale skeleton (RTMW-like units)
    return out


def test_torso_frame_reference_and_alignment(cohort):
    sessions, _ = cohort
    seg = sessions[0]['p1_pose33'][:300]
    ref = V._torso_frame_reference(seg)
    assert ref is not None
    hip, torso = ref
    assert hip.shape == (3,) and torso == pytest.approx(1.0, abs=0.02)
    assert np.allclose(hip[:2], 0.0, atol=0.02)
    scaled = seg.copy(); scaled[..., :2] *= 640.0; scaled[..., :2] += (100.0, 50.0)
    hip_s, torso_s = V._torso_frame_reference(scaled)
    assert torso_s == pytest.approx(640.0 * torso, rel=1e-6)
    al = V.align_pose33_to_frame(scaled, seg)
    assert al.dtype == np.float32 and al.shape == seg.shape
    assert np.allclose(al[..., :2], seg[..., :2], atol=1e-3)
    assert np.array_equal(al[..., 3], scaled[..., 3])
    # degenerate reference (hips hidden everywhere) -> coordinates unchanged
    hidden = seg.copy(); hidden[:, 23:25, 3] = 0.0
    assert V._torso_frame_reference(hidden) is None
    assert np.array_equal(V.align_pose33_to_frame(scaled, hidden), scaled)


@pytest.mark.parametrize('pixel_side', ['p1', 'p2'])
def test_inject_pose33_mixed_units_is_a_dose(pixel_side):
    """kappa blends shapes, not coordinate magnitudes, for mixed-format pseudo-dyads."""
    a = make_session('A', 11, coupled=False)
    b = make_session('B', 23, coupled=False)
    if pixel_side == 'p1':
        a = _scaled_copy(a, 'p1_pose33', 640.0)
    else:
        b = _scaled_copy(b, 'p2_pose33', 640.0)
    pseudo = V.make_pseudo_dyad_pose33(a, b)
    n = pseudo['ts_common'].size
    seg_len = int(60.0 * FS)
    s0 = n // 2 - seg_len // 2; s1 = s0 + seg_len
    ang1, _ = angle_features(pseudo['p1_pose33'][s0:s1])
    ang2, _ = angle_features(pseudo['p2_pose33'][s0:s1])
    col = ANGLE_NAMES.index('l_upper_arm')

    def torso_len(seg):
        return float(np.median(np.hypot(*(seg[:, [11, 12], :2].mean(axis=1)
                                          - seg[:, [23, 24], :2].mean(axis=1)).T)))

    gap_to_p1, gap_to_p2, corr = {}, {}, {}
    for kappa in (0.0, 0.1, 0.2, 0.4, 0.8):
        inj = V.inject_session(pseudo, kappa, np.random.default_rng(0), s0, s1)
        seg = inj['p2_pose33'][s0:s1]
        angi, _ = angle_features(seg)
        gap_to_p1[kappa] = _circular_gap(angi, ang1)
        gap_to_p2[kappa] = _circular_gap(angi, ang2)
        corr[kappa] = float(np.corrcoef(ang1[:, col], angi[:, col])[0, 1])
        # P2 keeps its native units: median torso length within 1 %
        assert torso_len(seg) == pytest.approx(torso_len(pseudo['p2_pose33'][s0:s1]), rel=0.01)
        assert np.array_equal(inj['p2_pose33'][:s0], pseudo['p2_pose33'][:s0])
    # (i) monotone approach to P1 — a dose, not a step at kappa > 0
    assert gap_to_p1[0.0] > gap_to_p1[0.1] > gap_to_p1[0.2] > gap_to_p1[0.4] > gap_to_p1[0.8]
    # (ii) not saturated at the smallest dose
    assert gap_to_p1[0.1] > 0.5 * gap_to_p1[0.0]
    # (iii) the injected P2 actually moves away from the original P2
    assert gap_to_p2[0.4] > 0.01
    assert gap_to_p2[0.1] < gap_to_p2[0.2] < gap_to_p2[0.4] < gap_to_p2[0.8]
    # correlation of a single segment angle rises with kappa in both directions
    assert corr[0.0] < corr[0.4] < corr[0.8]
    assert corr[0.8] > 0.5


def test_pair_seed_is_stable_and_kappa_specific():
    assert V._pair_seed('a', 'b', 0.4) == V._pair_seed('a', 'b', 0.4)
    assert V._pair_seed('a', 'b', 0.4) != V._pair_seed('a', 'b', 0.3)
    assert V._pair_seed('a', 'b', 0.4) != V._pair_seed('b', 'a', 0.4)


# ── Mode-agnostic channel computation ───────────────────────────────

@pytest.mark.parametrize('mode', V.ALL_MODES)
def test_compute_channel_contract_per_mode(cohort, mode):
    sessions, ctx = cohort
    res = V.compute_channel(sessions[0], mode, ctx, N_SURR, seed=1, compute_path_features=True)
    z, ts = res['z'], res['stride_ts']
    assert res['sid'] == 'A' and res['mode'] == mode
    assert z.dtype == np.float32 and z.shape == ts.shape and z.ndim == 1
    assert np.all(np.diff(ts) > 0)
    assert ts[0] >= sessions[0]['ts_common'][0] and ts[-1] <= sessions[0]['ts_common'][-1]
    assert abs(np.median(np.diff(ts)) - 1.0 / V.FS_OUT_HZ) < 1e-6
    assert np.isfinite(z).mean() > 0.9
    if mode in V.DDTW_MODES:
        assert res['info']['n_features'] == (10 if mode == 'pca' else 12)
        for k in ('lag_s', 'lag_var', 'asym'):
            assert res[k].shape == z.shape
        if mode != 'pca':
            assert np.asarray(res['info']['noise_floor']).shape == (2, 12)
        assert 'lag_s' not in V.compute_channel(sessions[0], mode, ctx, N_SURR, seed=1)
    else:
        assert res['info']['event_kind'] == V.EVENT_KIND[mode]
        assert res['info']['p1_n_events'] > 0 and res['info']['status'] == 'ok'
        json.dumps(res['info'])          # sidecar-safe


def test_compute_channel_rejects_unknown_mode(cohort):
    sessions, ctx = cohort
    with pytest.raises(ValueError):
        V.compute_channel(sessions[0], 'velocity', ctx, N_SURR)


def test_precompute_features_cache_matches_uncached(cohort):
    sessions, ctx = cohort
    sess = dict(sessions[0])
    sess.pop('features', None)
    uncached = V.compute_channel(sess, 'angles', ctx, N_SURR, seed=2)
    V.precompute_features([sess], ['angles', 'evt_landing'], ctx, n_jobs=2)
    assert set(sess['features']) == {'angles'} and set(sess['features']['angles']) == {'p1', 'p2'}
    cached = V.compute_channel(sess, 'angles', ctx, N_SURR, seed=2)
    assert np.array_equal(uncached['z'], cached['z'], equal_nan=True)


# ── Tests 1 and 3 on synthetic cohorts (in-process, no pool) ────────

def test_test1_and_test3_serial_angles(cohort):
    sessions, ctx = cohort
    t1 = V.test1_dose_response(sessions, 'angles', ctx, N_SURR, kappa_levels=(0.0, 0.4),
                               max_pairs=2, seed=1)
    assert t1['status'] == 'OK' and len(t1['pairs']) == 2
    assert list(t1['auc_table'].columns) == V.T1_AUC_COLUMNS
    assert list(t1['auc_table'].columns) == ['session_a', 'session_b', 'pose_format_in',
                                             'kappa', 'auc']
    assert set(t1['auc_table']['pose_format_in']) == {'synthetic+synthetic'}
    assert len(t1['auc_table']) == 4
    aucs = t1['mean_auc_by_kappa']
    assert aucs[0.4] > aucs[0.0] and aucs[0.4] > 0.8
    assert t1['auc_at_max_kappa'] == pytest.approx(aucs[0.4])
    assert set(t1) >= {'kendall_tau', 'kendall_p_perm', 'pass'}

    real = V.run_mode_real(sessions, 'angles', ctx, N_SURR, compute_path_features=True)
    assert all('z_pw' in r and 'lag_s' in r for r in real)
    t3 = V.test3_pseudo_null(sessions, real, 'angles', ctx, N_SURR, max_pairs=2, seed=1)
    assert t3['status'] == 'OK' and len(t3['pseudo_means']) == 2
    assert set(t3['real_per_session']) == {'A', 'B'}
    assert t3['level_source'] == 'z'
    assert t3['real_per_session']['A'] == pytest.approx(float(np.nanmean(real[0]['z'])))
    # coupled real pairs beat independent pseudo-dyads by a wide margin
    assert t3['delta'] > V.T3_DELTA_THRESH and t3['pass']
    # positive lag: P2 is P1 delayed by 0.25 s
    lag = np.concatenate([r['lag_s'] for r in real])
    assert np.nanmedian(lag) > 0.1

    t2 = V.test2_contrast(real, sessions)
    assert t2['status'].startswith('SKIPPED') and t2['pass'] is False
    assert len(t2['table']) == 2


def test_test1_skips_with_single_session(cohort):
    sessions, ctx = cohort
    t1 = V.test1_dose_response(sessions[:1], 'angles', ctx, N_SURR)
    assert t1['status'].startswith('SKIPPED') and t1['pass'] is False
    t3 = V.test3_pseudo_null(sessions[:1], [], 'angles', ctx, N_SURR)
    assert t3['status'] == 'SKIPPED' and t3['pass'] is False


# ── Prewhitening ────────────────────────────────────────────────────

def test_prewhiten_single_reduces_rho_and_zeroes_invalid():
    rng = np.random.default_rng(0)
    n = 2000
    z = np.empty(n)
    z[0] = 0.0
    for i in range(1, n):
        z[i] = 0.85 * z[i - 1] + rng.standard_normal()
    z[400:450] = np.nan
    out, impl = V.prewhiten_single(z)
    assert impl in (V._PREWHITEN_FALLBACK, V._PREWHITEN_REFERENCE)
    assert out.dtype == np.float32 and out.shape == z.shape and np.all(np.isfinite(out))
    assert np.all(out[400:450] == 0.0)
    finite = np.isfinite(z)
    assert abs(out[finite].std() - 1.0) < 0.05 and abs(out[finite].mean()) < 0.05
    pair = finite[:-1] & finite[1:]
    rho = np.corrcoef(out[:-1][pair], out[1:][pair])[0, 1]
    assert abs(rho) < V._PREWHITEN_RHO_THRESH
    # degenerate inputs -> zeros, never NaN
    assert np.all(V.prewhiten_single(np.full(50, np.nan))[0] == 0.0)
    assert np.all(V.prewhiten_single(np.ones(50))[0] == 0.0)


# ── Test 4 helpers ──────────────────────────────────────────────────

def test_align_and_pearson_block_ci():
    rng = np.random.default_rng(1)
    n = 1200
    base = smooth_walk(rng, n, 1)[:, 0]
    ts_a = np.arange(n) / 2.0
    ts_b = 3.0 + np.arange(n) / 2.0
    z_a = base + 0.5 * smooth_walk(rng, n, 1)[:, 0]
    z_a[100:120] = np.nan
    z_b = np.interp(ts_b, ts_a, base) + 0.5 * smooth_walk(rng, n, 1)[:, 0]
    al = V.align_on_common_grid(z_a, ts_a, z_b, ts_b)
    assert al is not None
    xa, xb = al
    assert xa.shape == xb.shape and np.all(np.isfinite(xa)) and np.all(np.isfinite(xb))
    block_len = V.block_len_from_seconds(V.BLOCK_S, V.FS_OUT_HZ)
    assert block_len == 20
    pc = V.pearson_block_ci(xa, xb, block_len, n_boot=500)
    assert pc['ci_lo'] <= pc['r'] <= pc['ci_hi'] and pc['r'] > 0.5
    assert -1.0 <= pc['ci_lo'] and pc['ci_hi'] <= 1.0 and pc['n_blocks'] > 0
    identical = V.pearson_block_ci(xa, xa, block_len, n_boot=200)
    assert identical['r'] == pytest.approx(1.0) and identical['ci_hi'] <= 1.0 + 1e-9
    assert V.align_on_common_grid(z_a, ts_a, z_b, ts_b + 10_000.0) is None
    assert V.align_on_common_grid(np.zeros(n), ts_a, z_b, ts_b) is None


def test_test4_redundancy_table_and_descriptors(cohort):
    sessions, ctx = cohort
    real = V.run_mode_real(sessions, 'evt_peak', ctx, N_SURR, compute_path_features=False)
    ref = {'A': (real[0]['z'].astype(np.float64), real[0]['stride_ts'])}
    noise = np.random.default_rng(0).standard_normal(real[1]['z'].size)
    v11 = {'B': (noise, real[1]['stride_ts'])}
    t4 = V.test4_redundancy(real, ref, 20, v11_loader=lambda sid: v11.get(sid))
    tab = t4['table'].set_index('sid')
    assert tab.loc['A', 'r_ddtw'] == pytest.approx(1.0)
    assert np.isnan(tab.loc['B', 'r_ddtw']) and 'no Phase 0 DDTW' in tab.loc['B', 'note']
    assert abs(tab.loc['B', 'r_v11']) < 0.3 and np.isnan(tab.loc['A', 'r_v11'])
    assert tab.loc['A', 'dmean_vs_ddtw'] == pytest.approx(0.0, abs=1e-6)
    assert t4['descriptor_ddtw'] == 'largely redundant'
    assert t4['descriptor_v11'] == 'largely independent'
    assert t4['status'] == 'OK' and t4['block_len'] == 20
    assert V._r_descriptor(float('nan')) == 'not available'
    assert V._r_descriptor(0.5) == 'partial overlap'


# ── Condition summaries ─────────────────────────────────────────────

def test_condition_summary_block_ci(cohort):
    sessions, _ = cohort
    ts = np.arange(0.0, SECONDS, 0.5)
    rng = np.random.default_rng(4)
    z = rng.standard_normal(ts.size).astype(np.float32)
    z[(ts >= 70.0) & (ts < 140.0)] += 2.0       # meditation block is higher
    z[10:14] = np.nan
    results = [{'sid': 'A', 'z': z, 'stride_ts': ts}]
    df = V.condition_summary(results, sessions[:1], 'angles', block_len=20, n_boot=300)
    assert set(df['condition']) == {'conv_1', 'meditate_B'}
    assert set(df['mode']) == {'angles'}
    row = df.set_index('condition')
    assert row.loc['meditate_B', 'mean_z'] > row.loc['conv_1', 'mean_z'] + 1.0
    assert np.all(df['ci_lo'] <= df['mean_z']) and np.all(df['mean_z'] <= df['ci_hi'])
    assert row.loc['conv_1', 'n_finite'] == row.loc['conv_1', 'n_samples'] - 4
    assert np.all(df['n_blocks'] >= 5)


# ── Decision rule, report, NPZ contract, scaffold hooks ─────────────

def _tests(t1, t2, t3, delta):
    return {'t1': {'pass': t1, 'status': 'OK', 'auc_at_max_kappa': 0.7, 'kendall_tau': 0.5,
                   'kendall_p_perm': 0.01, 'mean_auc_by_kappa': {0.0: 0.5, 0.4: 0.7}},
            't2': {'pass': t2, 'status': 'OK', 'mean_delta': 0.6, 'p_min': 0.01, 'p_t': 0.01,
                   'p_wilcoxon': 0.02, 'direction': 'conv > med', 'paired': [1, 2, 3]},
            't3': {'pass': t3, 'status': 'OK', 'delta': delta, 'mean_real': delta,
                   'mean_pseudo': 0.0, 'real_per_session': {'A': delta}},
            't4': {'status': 'OK', 'table': pd.DataFrame(), 'mean_r_ddtw': 0.2,
                   'ci_ddtw': (0.1, 0.3), 'descriptor_ddtw': 'largely independent',
                   'mean_r_v11': float('nan'), 'ci_v11': (float('nan'), float('nan')),
                   'descriptor_v11': 'not available'}}


def test_decide_rule():
    per_mode = {'angles': _tests(True, True, True, 0.8),
                'evt_landing': _tests(True, True, True, 1.4),
                'angle_speed': _tests(True, False, True, 3.0)}
    assert V.decide(per_mode, 'pose_ddtw')[0] == 'pose_evt_landing'
    per_mode = {'angles': _tests(True, True, True, 0.8), 'pca': _tests(True, True, True, 0.9)}
    assert V.decide(per_mode, None)[0] == 'pose_ddtw'
    none_pass = {'angles': _tests(False, True, True, 2.0)}
    assert V.decide(none_pass, 'pose_ddtw')[0] == 'pose_ddtw'
    assert V.decide(none_pass, 'pose_baseline')[0] == 'pose_baseline'
    assert V.decide(none_pass, None)[0] == 'pose_baseline'
    assert V.decide({}, None)[0] == 'pose_baseline'


def test_report_and_scaffold_decision_precedence(tmp_path, monkeypatch, cohort):
    sessions, _ = cohort
    phase1, phase0 = tmp_path / 'phase1_pose', tmp_path / 'phase0'
    per_mode = {'angles': _tests(True, True, True, 0.8)}
    decision, reason = V.decide(per_mode, None)
    path = V.write_phase1_report(phase1, sessions, per_mode, decision, reason, 50, None, 20)
    assert path.name == V.PHASE1_REPORT_NAME and path.exists()
    assert not (phase1 / V.PHASE0_REPORT_NAME).exists()
    text = path.read_text(encoding='utf-8')
    assert '- **Pose channel for MVP scaffold: `pose_angles`**' in text
    assert V.parse_decision_line(text) == 'pose_angles'
    assert S._parse_decision_line(text) == 'pose_angles'

    monkeypatch.setattr(S, 'PHASE1_DIR', phase1)
    monkeypatch.setattr(S, 'PHASE0_DIR', phase0)
    assert S.read_phase_decision() == 'pose_angles'
    phase0.mkdir()
    (phase0 / 'phase0_report.md').write_text(
        '## Decision\n- **Pose channel for MVP scaffold: `pose_ddtw`**\n')
    assert S.read_phase_decision() == 'pose_angles'         # Phase 1 wins
    (phase1 / 'phase1_report.md').write_text('- **Pose channel for MVP scaffold: `pose_bogus`**\n')
    assert S.read_phase_decision() == 'pose_ddtw'           # unknown Phase 1 value skipped
    (phase1 / 'phase1_report.md').unlink()
    assert S.read_phase_decision() == 'pose_ddtw'
    assert S.read_phase1_decision() is None
    (phase0 / 'phase0_report.md').unlink()
    assert S.read_phase_decision() == 'pose_baseline'       # unchanged default
    assert S.read_phase0_decision() is None
    monkeypatch.setattr(V, 'PHASE0_DIR', phase0)
    assert V.read_phase0_decision() is None


def test_save_mode_outputs_and_scaffold_loader(tmp_path, monkeypatch, cohort):
    sessions, ctx = cohort
    real = V.run_mode_real(sessions, 'angle_speed', ctx, N_SURR, compute_path_features=True)
    phase1, phase0, digest = tmp_path / 'phase1_pose', tmp_path / 'phase0', tmp_path / 'digest'
    npz_path = V.save_mode_outputs(phase1, 'angle_speed', real, meta={'n_surrogates': N_SURR})
    assert npz_path == phase1 / 'pose_angle_speed_per_session.npz'
    with np.load(npz_path) as npz:
        keys = set(npz.files)
        for sid in ('A', 'B'):
            assert {f'{sid}__z', f'{sid}__stride_ts', f'{sid}__z_pw', f'{sid}__lag_s',
                    f'{sid}__lag_var', f'{sid}__asym'} <= keys
        assert npz['A__z'].dtype == np.float32 and npz['A__stride_ts'].dtype == np.float64
    sidecar = json.loads(npz_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sidecar['mode'] == 'angle_speed' and sidecar['decision_name'] == 'pose_angle_speed'
    assert set(sidecar['sessions']) == {'A', 'B'} and sidecar['n_surrogates'] == N_SURR

    # scaffold loader: stream-relative NPZ -> absolute-LSL V11 grid
    digest.mkdir()
    t_start = 2000.0
    for sid in ('A', 'B'):
        (digest / f'{sid}.json').write_text(json.dumps({'t_start_lsl': t_start}))
    monkeypatch.setattr(S, 'PHASE1_DIR', phase1)
    monkeypatch.setattr(S, 'PHASE0_DIR', phase0)
    monkeypatch.setattr(S, 'DIGEST_ROOT', digest)
    t_common = t_start + np.arange(0.0, SECONDS, 0.5)
    z, note = S.load_pose_channel_for_session('A', t_common, 'pose_angle_speed')
    assert note == 'OK' and z.dtype == np.float32 and z.shape == t_common.shape
    r = real[0]
    finite = np.isfinite(r['z'])
    expect = np.interp(t_common - t_start, r['stride_ts'][finite], r['z'][finite],
                       left=0.0, right=0.0)
    assert np.allclose(z, expect, atol=1e-5)
    assert S.load_pose_channel_for_session('C', t_common, 'pose_angle_speed')[0] is None
    assert S.load_pose_channel_for_session('A', t_common, 'pose_angles')[0] is None
    assert S.load_pose_channel_for_session('A', t_common, 'pose_baseline')[0] is None
    # Phase 0 DDTW: legacy `__ddtw_z` key through the thin wrapper
    assert S.load_phase0_ddtw_for_session('A', t_common)[0] is None
    phase0.mkdir()
    np.savez(phase0 / 'pose_ddtw_per_session.npz',
             A__ddtw_z=r['z'], A__stride_ts=r['stride_ts'])
    z0, note0 = S.load_phase0_ddtw_for_session('A', t_common)
    assert note0 == 'OK' and np.allclose(z0, expect, atol=1e-5)
    assert S.pose_channel_npz_candidates('pose_ddtw')[0] == phase0 / 'pose_ddtw_per_session.npz'
    assert S.pose_channel_npz_candidates('pose_evt_peak') == [phase1 / 'pose_evt_peak_per_session.npz']
    assert S.pose_channel_npz_candidates('pose_baseline') == []


# ── Event modes: positive control with a shared pause schedule ──────

def _gate(t: np.ndarray, centres: np.ndarray, sigma: float = 0.35) -> np.ndarray:
    g = np.ones_like(t)
    for tc in centres:
        g *= 1.0 - np.exp(-0.5 * ((t - tc) / sigma) ** 2)
    return g


def make_pausing_session(sid: str, seed: int, shared: bool, seconds: float = SECONDS,
                         fs_native: float = 30.0, delay_s: float = 0.3) -> dict:
    """Rotating forearms/shins that pause at scheduled centres; P2 shares P1's
    schedule (delayed by ``delay_s``) or draws its own (pseudo-dyad null)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs_native))
    t = np.arange(n) / fs_native

    def schedule():
        out, tc = [], 2.0 + rng.uniform(3.0, 5.0)
        while tc < seconds - 2.0:
            out.append(tc); tc += rng.uniform(3.0, 5.0)
        return np.asarray(out)

    c1 = schedule()
    c2 = c1 + delay_s if shared else schedule()

    def stream(centres):
        theta = np.cumsum(np.deg2rad(180.0) * _gate(t, centres)) / fs_native
        pose = skeleton_from_drivers(np.zeros((n, 6)), rng, jitter=5e-4)
        for elbow, wrist, sgn in ((13, 15, 1.0), (14, 16, -1.0)):
            pose[:, wrist, 0] = pose[:, elbow, 0] + 0.35 * np.sin(sgn * theta)
            pose[:, wrist, 1] = pose[:, elbow, 1] + 0.35 * np.cos(sgn * theta)
        for knee, ankle, sgn in ((25, 27, 0.6), (26, 28, -0.6)):
            pose[:, ankle, 0] = pose[:, knee, 0] + 0.5 * np.sin(sgn * theta)
            pose[:, ankle, 1] = pose[:, knee, 1] + 0.5 * np.cos(sgn * theta)
        return pose

    grid = np.arange(t[0], t[-1], 1.0 / FS)
    p1 = V.pose33_onto_grid(stream(c1), t, grid)
    p2 = V.pose33_onto_grid(stream(c2), t, grid)
    m = grid.size
    return {'sid': sid,
            'p1_common': np.ascontiguousarray(p1[..., :3].reshape(m, -1)),
            'p2_common': np.ascontiguousarray(p2[..., :3].reshape(m, -1)),
            'p1_pose33': p1, 'p2_pose33': p2,
            'v1_common': np.ones(m, dtype=bool), 'v2_common': np.ones(m, dtype=bool),
            'ts_common': grid, 'markers_rel': [(5.0, 'conv_1_start'), (140.0, 'conv_1_stop')],
            'pose_format_in': 'synthetic', 'digest_xdf_md5': 'evt'}


@pytest.mark.parametrize('mode', V.EVENT_MODES)
def test_event_modes_shared_schedule_beats_independent(mode):
    ctx = {'components': None, 'mean': None, 'noise_normalize': True, 'smooth_sigma_s': 0.0}
    shared = make_pausing_session('S', 1, shared=True)
    indep = make_pausing_session('I', 2, shared=False)
    r_s = V.compute_channel(shared, mode, ctx, n_surrogates=60, seed=3)
    r_i = V.compute_channel(indep, mode, ctx, n_surrogates=60, seed=3)
    assert r_s['info']['p1_n_events'] >= 20 and r_s['info']['p2_n_events'] >= 20
    assert r_s['info']['mean_z_raw_event_bins'] > 1.0          # raw z where P2 events sit
    assert float(np.mean(r_s['z'])) > float(np.mean(r_i['z'])) + 0.1
    assert abs(float(np.mean(r_i['z']))) < 0.5                  # pseudo-dyad-style null
    if mode == 'evt_landing':
        assert r_s['info']['mean_signed_lag_s'] == pytest.approx(0.3, abs=0.15)
    # production smoothing path is standardised per session
    r_sm = V.compute_channel(shared, mode, {**ctx, 'smooth_sigma_s': 15.0}, n_surrogates=60, seed=3)
    assert abs(float(r_sm['z'].std()) - 1.0) < 1e-3 and abs(float(r_sm['z'].mean())) < 1e-3


@pytest.mark.parametrize('mode', V.EVENT_MODES)
def test_event_modes_test3_uses_raw_level_under_production_smoothing(mode):
    """Test 3 for the event modes must not be identically 0 under sigma = 15 s.

    The saved ``{sid}__z`` trace is per-session standardised (mean 0 by
    construction, real and pseudo alike), so the Test 3 level comes from the
    raw per-bin coincidence z (``info['mean_z_raw']``) while the production
    channel contract is unchanged.
    """
    ctx = {'components': None, 'mean': None, 'noise_normalize': True,
           'smooth_sigma_s': V.EVENT_SMOOTH_SIGMA_S}
    sessions = [make_pausing_session(f'S{i}', 10 + i, shared=True) for i in range(3)]
    real = V.run_mode_real(sessions, mode, ctx, 60, compute_path_features=False)
    for r in real:                       # production contract: standardised trace
        assert abs(float(r['z'].mean())) < 1e-3 and abs(float(r['z'].std()) - 1.0) < 1e-3
        assert r['info']['smooth_sigma_s'] == V.EVENT_SMOOTH_SIGMA_S
    t3 = V.test3_pseudo_null(sessions, real, mode, ctx, n_surr=60, max_pairs=2, seed=1)
    assert t3['status'] == 'OK' and t3['level_source'] == 'mean_z_raw'
    assert V.T3_LEVEL_SOURCE[mode] == 'mean_z_raw'
    assert len(t3['pseudo_means']) == 2 and set(t3['real_per_session']) == {'S0', 'S1', 'S2'}
    for r in real:
        assert t3['real_per_session'][r['sid']] == pytest.approx(r['info']['mean_z_raw'])
    assert abs(t3['mean_real']) > 1e-3                      # no longer identically zero
    assert abs(t3['mean_pseudo']) < 0.5                     # pseudo-dyad null
    assert t3['mean_real'] > t3['mean_pseudo'] + 0.1
    assert t3['delta'] == pytest.approx(t3['mean_real'] - t3['mean_pseudo'])
    # a DDTW result is unaffected by the helper
    ddtw_like = {'mode': 'angles', 'z': np.array([1.0, np.nan, 3.0], dtype=np.float32),
                 'info': {'mean_z_raw': 99.0}}
    assert V._t3_level(ddtw_like) == pytest.approx(2.0)
    # missing_data event result (no mean_z_raw key) falls back to the trace mean
    assert V._t3_level({'mode': mode, 'z': np.zeros(4, dtype=np.float32),
                        'info': {'status': 'missing_data'}}) == 0.0
