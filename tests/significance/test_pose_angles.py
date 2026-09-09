"""Tests for cadence.significance.pose_angles (Phase 1 pose coupling, Task 1).

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md
"""
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

from cadence.significance.pose_angles import (
    ANGLE_FEATURES, ANGLE_NAMES, ANGLE_TIERS, ANGLE_WEIGHTS,
    VISIBILITY_THRESHOLD, angle_features, estimate_fs, feature_weights,
    moving_average_nan, noise_floor, pose33_to_angle_stream, speed_envelope,
    speed_features, unwrap_nan,
)

IDX = {name: i for i, name in enumerate(ANGLE_NAMES)}


# ── Synthetic skeleton ──────────────────────────────────────────────

def _rot(xy: np.ndarray, phi: float, about: np.ndarray) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    rel = xy - about
    return np.stack([c * rel[:, 0] - s * rel[:, 1],
                     s * rel[:, 0] + c * rel[:, 1]], axis=1) + about


def make_skeleton(theta_elbow_l: float = 0.4, lean: float = 0.0,
                  scale: float = 1.0, offset=(0.0, 0.0),
                  arm_len: float = 1.0) -> np.ndarray:
    """Upright frontal skeleton in image coordinates (y down), (33, 4).

    ``theta_elbow_l`` sets the left forearm direction, ``lean`` rotates the
    upper body (torso, head, arms) about the mid-hip, ``arm_len`` scales the
    forearms only (limb-proportion change), ``scale``/``offset`` apply to
    all coordinates last.
    """
    xy = np.zeros((33, 2))
    # hips / shoulders / head
    xy[23] = (-0.15, 0.0); xy[24] = (0.15, 0.0)
    xy[11] = (-0.20, -1.0); xy[12] = (0.20, -1.0)
    xy[7] = (-0.08, -1.30); xy[8] = (0.08, -1.30)
    xy[0] = (0.02, -1.24)
    # face fillers (eyes, mouth) near the nose
    for k in (1, 2, 3):
        xy[k] = (-0.03 * k, -1.34)
    for k in (4, 5, 6):
        xy[k] = (0.03 * (k - 3), -1.34)
    xy[9] = (-0.03, -1.20); xy[10] = (0.03, -1.20)
    # arms
    xy[13] = xy[11] + (-0.12, 0.35); xy[14] = xy[12] + (0.12, 0.35)
    fore = 0.35 * arm_len
    xy[15] = xy[13] + fore * np.array([np.sin(theta_elbow_l), np.cos(theta_elbow_l)])
    xy[16] = xy[14] + fore * np.array([0.10, 0.99])
    for k, base in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        xy[k] = xy[base] + (0.02 * (k % 3), 0.05)
    # legs
    xy[25] = xy[23] + (-0.03, 0.5); xy[26] = xy[24] + (0.03, 0.5)
    xy[27] = xy[25] + (0.0, 0.5); xy[28] = xy[26] + (0.0, 0.5)
    xy[29] = xy[27] + (-0.02, 0.05); xy[31] = xy[27] + (0.05, 0.06)
    xy[30] = xy[28] + (0.02, 0.05); xy[32] = xy[28] + (-0.05, 0.06)

    if lean != 0.0:
        mid_hip = 0.5 * (xy[23] + xy[24])
        upper = np.r_[0:23]  # everything above the hips
        xy[upper] = _rot(xy[upper], lean, mid_hip)

    xy = xy * scale + np.asarray(offset, dtype=np.float64)
    out = np.zeros((33, 4))
    out[:, :2] = xy
    out[:, 3] = 1.0
    return out


def _frames(*skeletons) -> np.ndarray:
    return np.stack(skeletons, axis=0)


# ── Constants / metadata ────────────────────────────────────────────

def test_feature_table_layout():
    assert len(ANGLE_FEATURES) == 12
    assert ANGLE_NAMES == ['torso_lean', 'neck', 'head_twist', 'shoulder_line',
                           'l_thigh', 'r_thigh', 'l_upper_arm', 'r_upper_arm',
                           'l_forearm', 'r_forearm', 'l_shin', 'r_shin']
    assert set(ANGLE_TIERS) <= set(ANGLE_WEIGHTS)
    assert VISIBILITY_THRESHOLD == 0.5
    w = feature_weights()
    assert w.shape == (12,)
    assert np.isclose(w.sum(), 1.0)
    assert w[IDX['torso_lean']] > w[IDX['l_forearm']]


# ── angle_features ──────────────────────────────────────────────────

def test_upright_skeleton_all_finite_and_lean_zero():
    ang, valid = angle_features(_frames(make_skeleton()))
    assert ang.shape == (1, 12) and ang.dtype == np.float32
    assert np.isfinite(ang).all()
    assert valid.tolist() == [True]
    assert abs(float(ang[0, IDX['torso_lean']])) < 1e-6
    assert ((ang > -np.pi) & (ang <= np.pi)).all()


def test_translation_invariance():
    base, _ = angle_features(_frames(make_skeleton()))
    moved, _ = angle_features(_frames(make_skeleton(offset=(3.7, -12.25))))
    np.testing.assert_allclose(moved, base, atol=1e-6)


def test_scale_invariance():
    base, _ = angle_features(_frames(make_skeleton()))
    scaled, _ = angle_features(_frames(make_skeleton(scale=2.5)))
    np.testing.assert_allclose(scaled, base, atol=1e-6)


def test_proportion_invariance_forearm_length():
    base, _ = angle_features(_frames(make_skeleton(arm_len=1.0)))
    longer, _ = angle_features(_frames(make_skeleton(arm_len=2.0)))
    np.testing.assert_allclose(longer, base, atol=1e-6)


def test_elbow_angle_is_captured():
    a, _ = angle_features(_frames(make_skeleton(theta_elbow_l=0.2)))
    b, _ = angle_features(_frames(make_skeleton(theta_elbow_l=0.9)))
    # only the left forearm feature changes
    diff = np.abs(a - b)[0]
    assert diff[IDX['l_forearm']] > 0.5
    others = np.delete(diff, IDX['l_forearm'])
    assert np.all(others < 1e-6)


@pytest.mark.parametrize('phi', [0.3, -0.7, 2.0])
def test_whole_body_rotation_changes_only_torso_lean(phi):
    sk = make_skeleton()
    rotated = sk.copy()
    rotated[:, :2] = _rot(sk[:, :2], phi, np.array([0.4, 0.9]))
    base, _ = angle_features(_frames(sk))
    rot, valid = angle_features(_frames(rotated))
    assert valid[0]
    lean_delta = float(rot[0, IDX['torso_lean']] - base[0, IDX['torso_lean']])
    assert abs(lean_delta - phi) < 1e-5
    mask = np.ones(12, bool); mask[IDX['torso_lean']] = False
    np.testing.assert_allclose(rot[0, mask], base[0, mask], atol=1e-5)


def test_lean_parameter_maps_to_torso_lean():
    ang, _ = angle_features(_frames(make_skeleton(lean=0.25)))
    assert abs(float(ang[0, IDX['torso_lean']]) - 0.25) < 1e-5


def test_hidden_wrist_nans_only_forearm():
    sk = make_skeleton()
    sk[15, 3] = 0.0   # left wrist hidden
    ang, valid = angle_features(_frames(sk))
    assert np.isnan(ang[0, IDX['l_forearm']])
    others = np.delete(ang[0], IDX['l_forearm'])
    assert np.isfinite(others).all()
    assert valid[0]


def test_hidden_hips_invalidates_row():
    sk = make_skeleton()
    sk[23, 3] = 0.2
    ang, valid = angle_features(_frames(sk, make_skeleton()))
    assert np.isnan(ang[0]).all()
    assert not valid[0]
    assert np.isfinite(ang[1]).all() and valid[1]


def test_short_torso_and_degenerate_segment():
    sk = make_skeleton(scale=0.01)   # torso length 0.01 < MIN_TORSO_LEN
    ang, valid = angle_features(_frames(sk))
    assert np.isnan(ang).all() and not valid[0]
    sk = make_skeleton()
    sk[15, :2] = sk[13, :2]          # zero-length forearm
    ang, valid = angle_features(_frames(sk))
    assert np.isnan(ang[0, IDX['l_forearm']]) and valid[0]


def test_valid_requires_six_finite_features():
    sk = make_skeleton()
    for lm in (0, 7, 13, 14, 25, 26):   # hides ear 7 -> kills neck + head_twist; arms, thighs, shins
        sk[lm, 3] = 0.0
    ang, valid = angle_features(_frames(sk))
    assert np.isfinite(ang[0]).sum() < 6
    assert not valid[0]


def test_uses_only_xy_and_handles_non_finite():
    sk = make_skeleton()
    sk[:, 2] = np.nan   # z ignored entirely
    ang, valid = angle_features(_frames(sk))
    assert np.isfinite(ang).all() and valid[0]
    sk[15, 0] = np.nan  # non-finite coordinate -> that feature NaN
    ang, valid = angle_features(_frames(sk))
    assert np.isnan(ang[0, IDX['l_forearm']]) and valid[0]


def test_angle_features_rejects_bad_shape():
    with pytest.raises(ValueError):
        angle_features(np.zeros((5, 33, 3)))


# ── unwrap_nan / moving_average_nan ─────────────────────────────────

def test_unwrap_nan_crossing_pi_with_gap():
    t = np.linspace(0, 4 * np.pi, 400)
    wrapped = np.angle(np.exp(1j * t))
    x = np.column_stack([wrapped, -wrapped])
    x[150:172] = np.nan                     # gap that straddles a +-pi crossing
    out = unwrap_nan(x)
    assert out.shape == x.shape
    assert np.isnan(out[150:172]).all()
    for j in range(2):
        fin = out[np.isfinite(out[:, j]), j]
        assert np.all(np.abs(np.diff(fin)) < np.pi)
    # 1-D input works and is close to the true phase modulo a constant
    out1 = unwrap_nan(wrapped)
    np.testing.assert_allclose(np.diff(out1), np.diff(t), atol=1e-9)


def test_moving_average_nan_ignores_nan_and_pads_edges():
    x = np.arange(10, dtype=float)
    x[4] = np.nan
    ma = moving_average_nan(x, 3)
    assert ma.shape == (10,)
    assert np.isclose(ma[0], (0 + 0 + 1) / 3)          # edge padded
    assert np.isclose(ma[4], (3 + 5) / 2)              # NaN skipped
    assert np.isclose(ma[7], 7.0)
    allnan = moving_average_nan(np.full(6, np.nan), 3)
    assert np.isnan(allnan).all()
    np.testing.assert_array_equal(moving_average_nan(x, 1), x)


# ── noise_floor / speed_features / speed_envelope ───────────────────

def test_noise_floor_recovers_jitter_sd():
    rng = np.random.default_rng(0)
    fs, sigma = 30.0, 0.05
    t = np.arange(3000) / fs
    smooth = 0.5 * np.sin(2 * np.pi * 0.5 * t)[:, None] * np.linspace(0.5, 1.5, 12)[None, :]
    x = smooth + rng.normal(0.0, sigma, smooth.shape)
    nf = noise_floor(x)
    assert nf.shape == (12,)
    assert np.all(np.abs(nf - sigma) / sigma < 0.30)
    # NaN-only column -> floor
    x[:, 3] = np.nan
    nf = noise_floor(x)
    assert nf[3] == pytest.approx(1e-4)
    assert np.all(nf >= 1e-4)


def test_speed_features_constant_rotation():
    fs, omega = 30.0, 90.0
    t = np.arange(600) / fs
    theta = np.angle(np.exp(1j * np.deg2rad(omega) * t))
    x = np.tile(theta[:, None], (1, 12))
    sp = speed_features(x, fs)
    assert sp.shape == (600, 12)
    core = sp[20:-20]
    assert np.all(np.abs(core - omega) / omega < 0.05)
    assert np.all(sp >= 0)


def test_speed_envelope_weighting_and_nan_rule():
    fs, omega = 30.0, 60.0
    t = np.arange(300) / fs
    theta = np.deg2rad(omega) * t
    x = np.tile(theta[:, None], (1, 12))
    env = speed_envelope(x, fs)
    assert env.shape == (300,)
    assert np.all(np.abs(env[10:-10] - omega) / omega < 0.05)
    # fewer than 3 finite features -> NaN
    x2 = x.copy()
    x2[100:120, 2:] = np.nan
    env2 = speed_envelope(x2, fs)
    assert np.isnan(env2[105:115]).all()
    assert np.isfinite(env2[:90]).all()
    # custom weights: only feature 0 counts
    w = np.zeros(12); w[0] = 1.0
    x3 = x.copy(); x3[:, 1:] *= 3.0
    env3 = speed_envelope(x3, fs, weights=w)
    assert np.all(np.abs(env3[10:-10] - omega) / omega < 0.05)
    with pytest.raises(ValueError):
        speed_envelope(x, fs, weights=np.ones(5))


# ── pose33_to_angle_stream ──────────────────────────────────────────

def test_estimate_fs():
    assert estimate_fs(np.arange(100) / 25.0) == pytest.approx(25.0)
    assert estimate_fs(np.array([1.0]), fs_hint=12.0) == 12.0
    assert estimate_fs(np.array([1.0, 1.0, 1.0]), fs_hint=12.0) == 12.0


def test_pose33_to_angle_stream_contract():
    fs = 30.0
    n = 240
    t = np.arange(n) / fs
    frames = np.stack([make_skeleton(theta_elbow_l=0.4 + 0.5 * np.sin(2 * np.pi * 0.5 * ti))
                       for ti in t])
    frames[100:110, 23, 3] = 0.0   # hips hidden for a short stretch
    out = pose33_to_angle_stream(frames, t + 1000.0, fs_hint=99.0)
    assert set(out) == {'angles', 'valid', 'speed', 'envelope', 'noise_floor', 'fs'}
    assert out['fs'] == pytest.approx(fs)
    assert out['angles'].shape == (n, 12) and out['angles'].dtype == np.float32
    assert out['speed'].shape == (n, 12) and out['speed'].dtype == np.float32
    assert out['envelope'].shape == (n,) and out['envelope'].dtype == np.float32
    assert out['noise_floor'].shape == (12,) and out['noise_floor'].dtype == np.float32
    assert out['valid'].dtype == bool
    assert not out['valid'][100:110].any() and out['valid'][:100].all()
    assert np.isnan(out['angles'][100:110]).all()
    # NaN-ignoring smoothing bleeds <= 2 frames into the gap edges; interior is NaN
    assert np.isnan(out['envelope'][102:108]).all()
    assert np.isnan(out['speed'][102:108]).all()
    assert np.isfinite(out['envelope'][10:90]).all()
    # the moving elbow is the only feature with appreciable speed
    sp = np.nanmean(out['speed'][10:90], axis=0)
    assert sp[IDX['l_forearm']] > 20.0
    assert np.all(np.delete(sp, IDX['l_forearm']) < 1e-3)
