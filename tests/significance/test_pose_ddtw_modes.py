"""Tests for the Phase 1 feature modes of cadence.significance.pose_ddtw (Task 3).

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md

Synthetic only (no data files): pseudo-dyad null, dose response, warping-path
lag sign, Phase 0 backward compatibility, angle-mode feature construction,
and an end-to-end ``run_session_ddtw`` angle-mode run on a temporary
preproc / digest fixture.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
try:
    import torch as _torch  # noqa: F401  (Windows torch/numpy import-order guard)
except ImportError:
    pass
import numpy as np
import pytest

from cadence.significance import pose_ddtw
from cadence.significance.pose_ddtw import (
    ANGLE_MODES, DDTW_TARGET_RATE_HZ, FEATURE_MODES, _ddtw_score,
    build_angle_features, compute_session_ddtw, resample_pose33_visibility,
    run_session_ddtw, sliding_ddtw_path_features,
)
from cadence.significance.pose_angles import ANGLE_NAMES, moving_average_nan

FS = DDTW_TARGET_RATE_HZ          # 12 Hz internal grid
IDX = {name: i for i, name in enumerate(ANGLE_NAMES)}
PHASE0_KEYS = {'ddtw_z', 'ddtw_real', 'surr_mean', 'surr_std', 'stride_ts',
               'n_strides', 'n_surrogates_per_stride'}


# ── Synthetic streams ───────────────────────────────────────────────

def smooth_walk(rng: np.random.Generator, n: int, d: int = 12,
                smooth: int = 12) -> np.ndarray:
    """(n, d) smooth random walk, per-column standardised — stands in for a
    12-D unwrapped angle stream at 12 Hz."""
    x = np.cumsum(rng.standard_normal((n + 4 * smooth, d)), axis=0)
    x = moving_average_nan(x, smooth)[2 * smooth:2 * smooth + n]
    x = x - x.mean(axis=0)
    return x / x.std(axis=0)


def piecewise_linear_warp(stream: np.ndarray, rng: np.random.Generator,
                          n_anchors: int = 12, max_warp_frac: float = 0.05,
                          ) -> np.ndarray:
    """Local re-implementation of the Phase 0 driver's piecewise-linear time
    warp (scripts/ must not be imported from tests). Offsets are kept below
    ~0.5 s so the warp stays inside what a 4 s DTW window can absorb."""
    T, D = stream.shape
    t_anchors = np.linspace(0, T - 1, n_anchors)
    max_offset = max_warp_frac * T / n_anchors
    offsets = rng.uniform(-max_offset, max_offset, size=n_anchors)
    offsets[0] = offsets[-1] = 0.0
    s_anchors = np.maximum.accumulate(t_anchors + offsets)
    source_t = np.clip(np.interp(np.arange(T), t_anchors, s_anchors), 0, T - 1)
    return np.stack([np.interp(source_t, np.arange(T), stream[:, d])
                     for d in range(D)], axis=1)


def delay_frames(stream: np.ndarray, d: int) -> np.ndarray:
    """P2[t] = P1[t - d] (edge-held): P2 is P1 delayed by d frames."""
    return np.vstack([np.repeat(stream[:1], d, axis=0), stream[:-d]])


def _rotate_about(xy: np.ndarray, phi: np.ndarray, about: np.ndarray) -> np.ndarray:
    """Rotate (n, k, 2) points by per-frame angle phi (n,) about (n, 2)."""
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    rel = xy - about[:, None, :]
    rx = c * rel[:, :, 0] - s * rel[:, :, 1]
    ry = s * rel[:, :, 0] + c * rel[:, :, 1]
    return np.stack([rx, ry], axis=2) + about[:, None, :]


def make_pose_stream(n: int, fs: float, rng: np.random.Generator,
                     amp: float = 0.3, jitter: float = 0.0,
                     hide: tuple[int, int, int] | None = None) -> np.ndarray:
    """(n, 33, 4) frontal skeleton in image coordinates (y down) whose limb
    angles oscillate slowly; optional isotropic position jitter and a hidden
    landmark ``hide=(landmark, i0, i1)`` (vis=0, coordinates zeroed as the
    preproc does)."""
    t = np.arange(n) / fs
    ph = rng.uniform(0, 2 * np.pi, size=8)
    ua_l = 0.15 + amp * np.sin(2 * np.pi * 0.25 * t + ph[0])
    fa_l = 0.40 + amp * np.sin(2 * np.pi * 0.40 * t + ph[1])
    ua_r = -0.15 + amp * np.sin(2 * np.pi * 0.30 * t + ph[2])
    fa_r = 0.10 + amp * np.cos(2 * np.pi * 0.35 * t + ph[3])
    th_l = -0.06 + 0.5 * amp * np.sin(2 * np.pi * 0.20 * t + ph[4])
    th_r = 0.06 + 0.5 * amp * np.sin(2 * np.pi * 0.22 * t + ph[5])
    sh_l = 0.0 + 0.3 * amp * np.sin(2 * np.pi * 0.15 * t + ph[6])
    sh_r = 0.0 + 0.3 * amp * np.cos(2 * np.pi * 0.17 * t + ph[7])
    lean = 0.2 * amp * np.sin(2 * np.pi * 0.10 * t)

    xy = np.zeros((n, 33, 2))
    xy[:, 23] = (-0.15, 0.0); xy[:, 24] = (0.15, 0.0)          # hips
    xy[:, 11] = (-0.20, -1.0); xy[:, 12] = (0.20, -1.0)        # shoulders
    xy[:, 7] = (-0.08, -1.30); xy[:, 8] = (0.08, -1.30)        # ears
    xy[:, 0] = (0.02, -1.24)                                   # nose
    for k in (1, 2, 3):
        xy[:, k] = (-0.03 * k, -1.34)
    for k in (4, 5, 6):
        xy[:, k] = (0.03 * (k - 3), -1.34)
    xy[:, 9] = (-0.03, -1.20); xy[:, 10] = (0.03, -1.20)

    def seg(base, theta, length):
        return base + length * np.stack([np.sin(theta), np.cos(theta)], axis=1)

    xy[:, 13] = seg(xy[:, 11], ua_l, 0.35); xy[:, 14] = seg(xy[:, 12], ua_r, 0.35)
    xy[:, 15] = seg(xy[:, 13], fa_l, 0.35); xy[:, 16] = seg(xy[:, 14], fa_r, 0.35)
    for k, base in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        xy[:, k] = xy[:, base] + (0.02 * (k % 3), 0.05)
    xy[:, 25] = seg(xy[:, 23], th_l, 0.5); xy[:, 26] = seg(xy[:, 24], th_r, 0.5)
    xy[:, 27] = seg(xy[:, 25], sh_l, 0.5); xy[:, 28] = seg(xy[:, 26], sh_r, 0.5)
    xy[:, 29] = xy[:, 27] + (-0.02, 0.05); xy[:, 31] = xy[:, 27] + (0.05, 0.06)
    xy[:, 30] = xy[:, 28] + (0.02, 0.05); xy[:, 32] = xy[:, 28] + (-0.05, 0.06)

    mid_hip = 0.5 * (xy[:, 23] + xy[:, 24])
    xy[:, :23] = _rotate_about(xy[:, :23], lean, mid_hip)
    if jitter > 0:
        xy = xy + jitter * rng.standard_normal(xy.shape)

    out = np.zeros((n, 33, 4), dtype=np.float32)
    out[:, :, :2] = xy
    out[:, :, 3] = 1.0
    if hide is not None:
        lm, i0, i1 = hide
        out[i0:i1, lm, :3] = 0.0
        out[i0:i1, lm, 3] = 0.0
    return out


def _grid(n_seconds: float):
    n = int(round(n_seconds * FS))
    return n, np.arange(n) / FS, np.ones(n, dtype=bool)


# ── Pseudo-dyad null (critical rule) ────────────────────────────────

def test_pseudo_dyad_null_mean_z_near_zero():
    n, ts, valid = _grid(120.0)
    rng = np.random.default_rng(0)
    p1 = smooth_walk(rng, n)
    p2 = smooth_walk(rng, n)          # independent stream = pseudo-dyad
    out = compute_session_ddtw(p1, p2, valid, valid, ts, [], n_surrogates=40, seed=1)
    z = out['ddtw_z']
    assert z.shape == (out['n_strides'],) and out['n_strides'] > 100
    assert np.isfinite(z).mean() > 0.95
    assert -0.5 <= float(np.nanmean(z)) <= 0.5


# ── Dose response ───────────────────────────────────────────────────

def test_dose_response_monotone_in_kappa():
    n, ts, valid = _grid(90.0)
    rng = np.random.default_rng(7)
    p1 = smooth_walk(rng, n)
    indep = smooth_walk(rng, n)
    warped = piecewise_linear_warp(p1, rng)
    means = []
    for kappa in (0.0, 0.4, 0.8):
        p2 = kappa * warped + (1.0 - kappa) * indep
        out = compute_session_ddtw(p1, p2, valid, valid, ts, [], n_surrogates=40, seed=1)
        means.append(float(np.nanmean(out['ddtw_z'])))
    assert means[0] < means[1] < means[2], means
    assert means[2] > 1.0, means


# ── Warping-path lag sign ───────────────────────────────────────────

def test_lag_sign_positive_when_p2_delayed_and_flips_on_swap():
    n, ts, valid = _grid(60.0)
    rng = np.random.default_rng(3)
    p1 = smooth_walk(rng, n)
    p2 = delay_frames(p1, int(round(0.5 * FS)))     # P2 = P1 delayed by 0.5 s
    fwd = compute_session_ddtw(p1, p2, valid, valid, ts, [], n_surrogates=5, seed=1,
                               compute_path_features=True)
    rev = compute_session_ddtw(p2, p1, valid, valid, ts, [], n_surrogates=5, seed=1,
                               compute_path_features=True)
    for out in (fwd, rev):
        for k in ('ddtw_lag_s', 'ddtw_lag_var', 'ddtw_asym'):
            assert out[k].shape == (out['n_strides'],)
            assert out[k].dtype == np.float32
    assert float(np.nanmedian(fwd['ddtw_lag_s'])) > 0.2
    assert float(np.nanmedian(rev['ddtw_lag_s'])) < -0.2
    assert float(np.nanmedian(fwd['ddtw_asym'])) > 0.5
    assert float(np.nanmedian(rev['ddtw_asym'])) < -0.5
    np.testing.assert_allclose(fwd['ddtw_lag_s'], -rev['ddtw_lag_s'], atol=1e-6)
    assert np.all(fwd['ddtw_lag_var'][np.isfinite(fwd['ddtw_lag_var'])] >= 0)


def test_sliding_path_features_validity_gate_and_zero_lag():
    n, ts, valid = _grid(30.0)
    rng = np.random.default_rng(11)
    p1 = smooth_walk(rng, n)
    v2 = valid.copy()
    v2[:120] = False                                  # first 10 s invalid for P2
    pf = sliding_ddtw_path_features(p1, p1.copy(), 48, 6, valid, v2, FS)
    assert set(pf) == {'lag_s', 'lag_var', 'asym', 'starts'}
    assert pf['lag_s'].shape == pf['starts'].shape
    # windows fully inside the invalid stretch are NaN; identical streams -> lag 0
    assert np.isnan(pf['lag_s'][0])
    fin = np.isfinite(pf['lag_s'])
    assert fin.any()
    np.testing.assert_allclose(pf['lag_s'][fin], 0.0, atol=1e-9)
    np.testing.assert_allclose(pf['asym'][fin], 0.0, atol=1e-9)


# ── Backward compatibility of the Phase 0 (pca) path ────────────────

def test_default_path_never_touches_phase1_code(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError('Phase 1 code reached from the default (pca) path')
    monkeypatch.setattr(pose_ddtw, 'build_angle_features', _boom)
    monkeypatch.setattr(pose_ddtw, 'sliding_ddtw_path_features', _boom)
    monkeypatch.setattr(pose_ddtw, '_window_path_features', _boom)
    monkeypatch.setattr(pose_ddtw, 'resample_pose33_visibility', _boom)

    n, ts, valid = _grid(40.0)
    rng = np.random.default_rng(5)
    p1 = rng.standard_normal((n, 10))
    p2 = rng.standard_normal((n, 10))
    markers = [(5.0, 'conv_1_start'), (35.0, 'conv_1_stop')]
    s = _ddtw_score(p1[:48], p2[:48])
    assert np.isfinite(s) and s < 0
    out_a = compute_session_ddtw(p1, p2, valid, valid, ts, markers, n_surrogates=10, seed=2)
    out_b = compute_session_ddtw(p1, p2, valid, valid, ts, markers, n_surrogates=10, seed=2)
    assert set(out_a) == PHASE0_KEYS          # no new keys without opt-in
    for k in PHASE0_KEYS:
        np.testing.assert_array_equal(out_a[k], out_b[k])
    assert out_a['ddtw_z'].dtype == np.float32
    assert FEATURE_MODES == ('pca', 'angles', 'angle_speed')


# ── build_angle_features ────────────────────────────────────────────

def test_build_angle_features_angles_noise_normalised():
    n, ts, _ = _grid(60.0)
    rng = np.random.default_rng(21)
    pose = make_pose_stream(n, FS, rng, amp=0.3, jitter=0.005)
    X, valid, info = build_angle_features(pose, ts, 'angles')
    assert X.shape == (n, 12) and X.dtype == np.float64
    assert valid.shape == (n,) and valid.dtype == bool and valid.all()
    assert np.isfinite(X).all()
    std = X.std(axis=0)
    assert np.all(std >= 0.5) and np.all(std <= 50.0), std
    assert info['noise_floor'].shape == (12,)
    assert np.all(info['noise_floor'] > 1e-4)          # measured, not the floor
    assert info['noise_floor_units'] == 'rad'
    assert info['feature_mode'] == 'angles' and info['n_features'] == 12
    assert info['feature_names'] == ANGLE_NAMES
    assert abs(info['fs'] - FS) < 1e-6
    # un-normalised output is exactly X * floor
    X_raw, _, info_raw = build_angle_features(pose, ts, 'angles', noise_normalize=False)
    np.testing.assert_allclose(X_raw, X * info['noise_floor'][None, :], rtol=1e-10)
    np.testing.assert_allclose(info_raw['noise_floor'], info['noise_floor'])


def test_build_angle_features_angle_speed():
    n, ts, _ = _grid(60.0)
    rng = np.random.default_rng(22)
    pose = make_pose_stream(n, FS, rng, amp=0.3, jitter=0.003)
    X, valid, info = build_angle_features(pose, ts, 'angle_speed')
    assert X.shape == (n, 12) and np.isfinite(X).all() and valid.all()
    assert np.all(X >= 0.0)                              # speed magnitude
    assert info['noise_floor'].shape == (12,)
    assert info['noise_floor_units'] == 'deg/s'
    assert np.all(X.std(axis=0) > 0.3)
    # moving joints carry more speed than the static shoulder line
    assert X[:, IDX['l_forearm']].mean() > 2.0 * X[:, IDX['shoulder_line']].mean()


def test_build_angle_features_nan_fill_and_validity():
    n, ts, _ = _grid(40.0)
    rng = np.random.default_rng(23)
    j = IDX['l_forearm']
    # interior gap: left wrist hidden for frames 100..140
    pose = make_pose_stream(n, FS, rng, jitter=0.002, hide=(15, 100, 140))
    X, valid, _ = build_angle_features(pose, ts, 'angles')
    assert np.isfinite(X).all()
    assert valid[100:140].all()                          # one missing feature keeps the frame valid
    lo, hi = sorted((X[99, j], X[140, j]))
    assert np.all(X[100:140, j] >= lo - 1e-9) and np.all(X[100:140, j] <= hi + 1e-9)
    # leading gap -> 0
    pose = make_pose_stream(n, FS, rng, jitter=0.002, hide=(15, 0, 25))
    X, valid, _ = build_angle_features(pose, ts, 'angles')
    assert np.all(X[:25, j] == 0.0) and X[25, j] != 0.0
    # hidden hips -> frame invalid, still finite (zero) input for the DTW
    pose = make_pose_stream(n, FS, rng, jitter=0.002, hide=(23, 200, 260))
    X, valid, _ = build_angle_features(pose, ts, 'angles')
    assert not valid[200:260].any() and valid[:200].all() and valid[260:].all()
    assert np.isfinite(X).all()


def test_build_angle_features_rejects_bad_mode():
    n, ts, _ = _grid(5.0)
    pose = make_pose_stream(n, FS, np.random.default_rng(0))
    with pytest.raises(ValueError):
        build_angle_features(pose, ts, 'pca')
    with pytest.raises(ValueError):
        build_angle_features(pose, ts, 'nonsense')
    assert ANGLE_MODES == ('angles', 'angle_speed')


def test_resample_pose33_visibility_is_conservative():
    ts = np.arange(5, dtype=np.float64)
    pose = np.ones((5, 33, 4), dtype=np.float32)
    pose[2, 15, 3] = 0.0                                  # left wrist hidden at t=2
    grid = np.array([0.5, 1.5, 2.0, 2.5, 3.5, 9.0])
    vis = resample_pose33_visibility(pose, ts, grid)
    assert vis.shape == (6, 33)
    np.testing.assert_array_equal(vis[:, 15], [1, 0, 0, 0, 1, 1])
    assert vis[:, 16].all()                               # other landmarks untouched


# ── run_session_ddtw end-to-end on a temporary fixture ──────────────

def _write_fixture(tmp_path: Path, sid: str, seed: int = 31,
                   native_fs: float = 30.0, seconds: float = 60.0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    n = int(seconds * native_fs)
    arrays = {}
    for p, t_off in (('p1', 0.0), ('p2', 0.05)):
        pose = make_pose_stream(n, native_fs, rng, amp=0.3, jitter=0.003)
        arrays[f'{p}_pose33'] = pose
        arrays[f'{p}_pose33_ts'] = t_off + np.arange(n) / native_fs
        arrays[f'{p}_pose_features_valid'] = np.ones(n, dtype=bool)
    preproc_root = tmp_path / 'preproc'
    digest_root = tmp_path / 'digest'
    preproc_root.mkdir(); digest_root.mkdir()
    np.savez(preproc_root / f'{sid}.npz', **arrays)
    (preproc_root / f'{sid}.json').write_text(json.dumps(
        {'digest_xdf_md5': 'deadbeef', 'pose_format_in': 'mediapipe33'}))
    t0 = 1000.0
    (digest_root / f'{sid}.json').write_text(json.dumps({
        'xdf_md5': 'deadbeef', 't_start_lsl': t0,
        'markers': [[t0 + 5.0, 'conv_1_start'], [t0 + 55.0, 'conv_1_stop']],
    }))
    return preproc_root, digest_root


@pytest.mark.parametrize('mode', ['angles', 'angle_speed'])
def test_run_session_ddtw_angle_modes(tmp_path, mode):
    sid = 'y_test'
    preproc_root, digest_root = _write_fixture(tmp_path, sid)
    out = run_session_ddtw(sid, None, None, preproc_root=preproc_root,
                           digest_root=digest_root, n_surrogates=6, seed=1,
                           feature_mode=mode, compute_path_features=True)
    n_strides = out['n_strides']
    assert n_strides > 80
    for k in ('ddtw_z', 'ddtw_real', 'ddtw_lag_s', 'ddtw_lag_var', 'ddtw_asym'):
        assert out[k].shape == (n_strides,), k
    assert out['stride_ts'].shape == (n_strides,)
    assert np.isfinite(out['ddtw_z']).mean() > 0.9
    assert np.isfinite(out['ddtw_lag_s']).mean() > 0.9
    assert out['feature_mode'] == mode
    assert out['n_features'] == 12
    assert out['noise_normalize'] is True
    assert out['noise_floor'].shape == (2, 12) and out['noise_floor'].dtype == np.float32
    assert np.all(out['noise_floor'] > 0)
    assert out['session_id'] == sid and out['digest_xdf_md5'] == 'deadbeef'
    assert out['pose_format_in'] == 'mediapipe33'
    # stride centres lie inside the common stream-relative time range
    assert out['stride_ts'][0] >= 0.05 and out['stride_ts'][-1] <= 60.0


def test_run_session_ddtw_pca_requires_components(tmp_path):
    sid = 'y_test'
    preproc_root, digest_root = _write_fixture(tmp_path, sid)
    with pytest.raises(ValueError):
        run_session_ddtw(sid, None, None, preproc_root=preproc_root,
                         digest_root=digest_root, n_surrogates=2)
    with pytest.raises(ValueError):
        run_session_ddtw(sid, None, None, preproc_root=preproc_root,
                         digest_root=digest_root, feature_mode='bogus')
