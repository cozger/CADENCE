"""Tests for cadence.significance.pose_event_coincidence (Phase 1 pose coupling, Task 4).

Plan: docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md

Synthetic 33-landmark skeleton streams: forearms and shins rotate about
their proximal joints at a constant angular rate that is smoothly gated to
zero at scheduled pauses, so the angular-speed envelope has a clean
minimum at every pause centre. Pseudo-dyad rule: the null case uses two
independent pause schedules, never a shared one.
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

from cadence.significance.face_event_coincidence import peaks_to_grid
from cadence.significance.pose_event_coincidence import (
    EVENT_KINDS, compute_pose_event_coincidence, detect_landings,
    detect_movement_peaks, events_from_pose_npz,
)

FS = 30.0
DURATION_S = 150.0
OMEGA_DEG_S = 180.0      # rotation rate of the moving segments while active
PAUSE_SIGMA_S = 0.35     # width of each Gaussian speed well


# ── Synthetic skeleton stream ───────────────────────────────────────

def _base_skeleton() -> np.ndarray:
    """Upright frontal skeleton (33, 4) in image coordinates (y down), all visible."""
    xy = np.zeros((33, 2))
    xy[23] = (-0.15, 0.0); xy[24] = (0.15, 0.0)          # hips
    xy[11] = (-0.20, -1.0); xy[12] = (0.20, -1.0)        # shoulders
    xy[7] = (-0.08, -1.30); xy[8] = (0.08, -1.30)        # ears
    xy[0] = (0.02, -1.24)                                # nose
    for k in range(1, 7):
        xy[k] = (0.03 * (k - 3.5), -1.34)                # eyes
    xy[9] = (-0.03, -1.20); xy[10] = (0.03, -1.20)       # mouth
    xy[13] = xy[11] + (-0.12, 0.35); xy[14] = xy[12] + (0.12, 0.35)   # elbows
    xy[15] = xy[13] + (0.0, 0.35); xy[16] = xy[14] + (0.0, 0.35)      # wrists
    for k, base in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        xy[k] = xy[base] + (0.02, 0.05)                  # hand points
    xy[25] = xy[23] + (-0.03, 0.5); xy[26] = xy[24] + (0.03, 0.5)     # knees
    xy[27] = xy[25] + (0.0, 0.5); xy[28] = xy[26] + (0.0, 0.5)        # ankles
    for k, base in ((29, 27), (31, 27), (30, 28), (32, 28)):
        xy[k] = xy[base] + (0.02, 0.05)                  # feet
    out = np.zeros((33, 4))
    out[:, :2] = xy
    out[:, 3] = 1.0
    return out


def _gate(t: np.ndarray, pause_centres: np.ndarray, sigma: float = PAUSE_SIGMA_S) -> np.ndarray:
    """1 while moving, smoothly 0 at each pause centre (product of Gaussian wells)."""
    g = np.ones_like(t)
    for tc in pause_centres:
        g *= 1.0 - np.exp(-0.5 * ((t - tc) / sigma) ** 2)
    return g


def random_pause_schedule(rng: np.random.Generator, duration_s: float = DURATION_S,
                          gap_range=(3.0, 5.0)) -> np.ndarray:
    """Pause centres separated by uniform gaps, kept away from the edges."""
    centres = []
    t = 2.0 + rng.uniform(*gap_range)
    while t < duration_s - 2.0:
        centres.append(t)
        t += rng.uniform(*gap_range)
    return np.asarray(centres)


def make_pose_stream(pause_centres: np.ndarray, rng: np.random.Generator,
                     fs: float = FS, duration_s: float = DURATION_S,
                     omega_deg_s: float = OMEGA_DEG_S, jitter: float = 5e-4,
                     ts0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pose33 (N,33,4), ts (N,), valid (N,)) with rotating forearms/shins that pause."""
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    g = _gate(t, pause_centres)
    theta = np.cumsum(np.deg2rad(omega_deg_s) * g) / fs        # integrated angle
    base = _base_skeleton()
    pose = np.repeat(base[None], n, axis=0)
    seg = 0.35
    # left / right forearm rotate about the elbow at theta / -theta
    for elbow, wrist, sgn in ((13, 15, 1.0), (14, 16, -1.0)):
        pose[:, wrist, 0] = pose[:, elbow, 0] + seg * np.sin(sgn * theta)
        pose[:, wrist, 1] = pose[:, elbow, 1] + seg * np.cos(sgn * theta)
    # shins rotate more slowly about the knee
    for knee, ankle, sgn in ((25, 27, 0.6), (26, 28, -0.6)):
        pose[:, ankle, 0] = pose[:, knee, 0] + 0.5 * np.sin(sgn * theta)
        pose[:, ankle, 1] = pose[:, knee, 1] + 0.5 * np.cos(sgn * theta)
    pose[:, :, :2] += rng.normal(0.0, jitter, size=(n, 33, 2))
    ts = ts0 + t
    valid = np.ones(n, dtype=bool)
    return pose, ts, valid


def make_npz(p1: tuple, p2: tuple) -> dict:
    d = {}
    for p, (pose, ts, valid) in (('p1', p1), ('p2', p2)):
        d[f'{p}_pose33'] = pose.astype(np.float32)
        d[f'{p}_pose33_ts'] = ts
        d[f'{p}_pose_features_valid'] = valid
    return d


def t_common_for(ts: np.ndarray, fs_out: float = 2.0) -> np.ndarray:
    return np.arange(ts[0], ts[-1], 1.0 / fs_out)


# ── detect_landings / detect_movement_peaks ─────────────────────────

def test_detect_landings_recovers_known_rest_centres():
    fs = FS
    t = np.arange(int(40 * fs)) / fs
    centres = np.array([5.0, 11.0, 18.0, 26.0, 33.0])
    env = 25.0 * _gate(t, centres)
    env[int(21.5 * fs):int(23.0 * fs)] = np.nan          # a tracking dropout, no rest
    idx, depth = detect_landings(env, fs)
    assert idx.size == 5
    true_idx = np.round(centres * fs).astype(int)
    assert np.all(np.abs(np.sort(idx) - true_idx) <= 2)
    assert depth.shape == (5,) and np.all(depth > 0.5 * 25.0)
    # no event inside or on the rim of the NaN gap
    assert not np.any((idx >= int(21.5 * fs) - 1) & (idx <= int(23.0 * fs)))


def test_detect_landings_nan_gap_creates_no_event():
    fs = FS
    t = np.arange(int(20 * fs)) / fs
    env = 20.0 + 2.0 * np.sin(2 * np.pi * 0.1 * t)         # no rest anywhere
    idx_clean, _ = detect_landings(env, fs)
    env_gap = env.copy()
    env_gap[int(8 * fs):int(9 * fs)] = np.nan
    idx_gap, _ = detect_landings(env_gap, fs)
    assert not np.any((idx_gap >= int(8 * fs) - 1) & (idx_gap <= int(9 * fs)))
    assert idx_gap.size <= idx_clean.size


def test_detect_landings_degenerate_inputs():
    idx, depth = detect_landings(np.full(100, np.nan), FS)
    assert idx.size == 0 and depth.size == 0
    idx, depth = detect_landings(np.full(100, 3.0), FS)   # flat: zero range
    assert idx.size == 0
    idx, depth = detect_landings(np.array([1.0, 0.0]), FS)
    assert idx.size == 0


def test_detect_movement_peaks_mirrors_face_detector():
    fs = FS
    t = np.arange(int(30 * fs)) / fs
    env = 10.0 + 8.0 * np.sin(2 * np.pi * 0.25 * t)        # maxima every 4 s
    env[int(12.5 * fs):int(13.5 * fs)] = np.nan
    idx, amp = detect_movement_peaks(env, fs, quantile_threshold=0.70, min_sep_s=1.0)
    assert idx.size >= 5
    assert np.all(np.isfinite(amp))
    assert np.all(amp >= np.nanquantile(env, 0.70))
    assert np.all(np.diff(np.sort(idx)) >= int(round(1.0 * fs)))


# ── End-to-end channel ──────────────────────────────────────────────

@pytest.fixture(scope='module')
def independent_dyad():
    rng = np.random.default_rng(11)
    p1 = make_pose_stream(random_pause_schedule(np.random.default_rng(1)), rng)
    p2 = make_pose_stream(random_pause_schedule(np.random.default_rng(2)), rng)
    return make_npz(p1, p2)


@pytest.fixture(scope='module')
def coupled_dyad():
    rng = np.random.default_rng(12)
    centres = random_pause_schedule(np.random.default_rng(3))
    p1 = make_pose_stream(centres, rng)
    p2 = make_pose_stream(centres + 0.3, rng)               # P2 lands 0.3 s later
    return make_npz(p1, p2), centres


def test_pseudo_dyad_null_raw_z_near_zero(independent_dyad):
    data = independent_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    z, info = compute_pose_event_coincidence(data, t_common, 0.0, event_kind='landing',
                                             n_surrogates=60, seed=0, smooth_sigma_s=0.0)
    assert info['status'] == 'ok'
    assert z.shape == t_common.shape and z.dtype == np.float32
    assert info['p1_n_events'] >= 20 and info['p2_n_events'] >= 20
    assert abs(info['mean_z_raw']) < 0.5
    assert abs(float(z.mean())) < 0.5


def test_shared_schedule_raw_z_positive_and_lag_sign(coupled_dyad):
    data, centres = coupled_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    z, info = compute_pose_event_coincidence(data, t_common, 0.0, event_kind='landing',
                                             n_surrogates=60, seed=0, tau_samples=1,
                                             smooth_sigma_s=0.0)
    assert info['status'] == 'ok'
    # every pause is detected for both participants
    assert abs(info['p1_n_events'] - centres.size) <= 1
    assert abs(info['p2_n_events'] - centres.size) <= 1
    # coincidence bins (where P2 landed) carry strongly positive raw z at tau = +/-500 ms
    p2_times = events_from_pose_npz(data, 'p2', 'landing')
    p2_bins = peaks_to_grid(p2_times, t_common, 0.0) > 0
    assert float(z[p2_bins].mean()) > 1.0
    assert info['mean_z_raw_event_bins'] > 1.0
    assert info['mean_z_raw'] > 0.1
    # positive lag = P2 later than P1
    assert info['n_matched_events'] >= 5
    assert abs(info['mean_signed_lag_s'] - 0.3) <= 0.15


def test_coupled_beats_null_and_swapping_flips_lag(coupled_dyad, independent_dyad):
    data, _ = coupled_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    _, info_c = compute_pose_event_coincidence(data, t_common, 0.0, n_surrogates=60,
                                               seed=0, smooth_sigma_s=0.0)
    t_null = t_common_for(independent_dyad['p1_pose33_ts'])
    _, info_n = compute_pose_event_coincidence(independent_dyad, t_null, 0.0,
                                               n_surrogates=60, seed=0, smooth_sigma_s=0.0)
    assert info_c['mean_z_raw'] > info_n['mean_z_raw'] + 0.1
    swapped = {k.replace('p1_', 'px_').replace('p2_', 'p1_').replace('px_', 'p2_'): v
               for k, v in data.items()}
    _, info_s = compute_pose_event_coincidence(swapped, t_common, 0.0, n_surrogates=60,
                                               seed=0, smooth_sigma_s=0.0)
    assert abs(info_s['mean_signed_lag_s'] + 0.3) <= 0.15


def test_smoothed_output_is_standardized_and_deterministic(coupled_dyad):
    data, _ = coupled_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    z1, info1 = compute_pose_event_coincidence(data, t_common, 0.0, n_surrogates=40, seed=7)
    z2, _ = compute_pose_event_coincidence(data, t_common, 0.0, n_surrogates=40, seed=7)
    assert np.array_equal(z1, z2)
    assert z1.dtype == np.float32 and np.all(np.isfinite(z1))
    assert info1['smooth_sigma_s'] == 15.0
    assert abs(float(z1.mean())) < 1e-4 and abs(float(z1.std()) - 1.0) < 1e-3
    for k in ('p1_total_peaks', 'p2_total_peaks', 'p1_grid_density', 'p2_grid_density',
              'mean_z_raw', 'std_z_raw', 'mean_raw_coinc', 'mean_z_smoothed',
              'std_z_smoothed', 'mean_z', 'std_z', 'event_kind', 'p1_event_rate_hz',
              'p2_event_rate_hz', 'mean_signed_lag_s'):
        assert k in info1
    assert info1['event_kind'] == 'landing'
    assert 0.1 < info1['p1_event_rate_hz'] < 0.5


def test_peak_event_kind_runs(coupled_dyad):
    data, _ = coupled_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    z, info = compute_pose_event_coincidence(data, t_common, 0.0, event_kind='peak',
                                             n_surrogates=40, seed=0, smooth_sigma_s=0.0)
    assert info['status'] == 'ok' and info['event_kind'] == 'peak'
    assert info['p1_n_events'] > 0 and info['p2_n_events'] > 0
    assert z.shape == t_common.shape
    with pytest.raises(ValueError):
        compute_pose_event_coincidence(data, t_common, 0.0, event_kind='bogus')
    assert EVENT_KINDS == ('landing', 'peak')


def test_lsl_offset_aligns_stream_clock_to_t_common(coupled_dyad):
    data, _ = coupled_dyad
    ts = data['p1_pose33_ts']
    t_common = t_common_for(ts)
    z_ref, info_ref = compute_pose_event_coincidence(data, t_common, 0.0, n_surrogates=40,
                                                     seed=0, smooth_sigma_s=0.0)
    z_off, info_off = compute_pose_event_coincidence(data, t_common + 500.0, 500.0,
                                                     n_surrogates=40, seed=0,
                                                     smooth_sigma_s=0.0)
    assert np.array_equal(z_ref, z_off)
    assert info_ref['p1_grid_density'] == info_off['p1_grid_density']


def test_missing_key_returns_zeros_with_missing_data_status(coupled_dyad):
    data, _ = coupled_dyad
    t_common = t_common_for(data['p1_pose33_ts'])
    broken = {k: v for k, v in data.items() if k != 'p2_pose33'}
    z, info = compute_pose_event_coincidence(broken, t_common, 0.0)
    assert info == {'status': 'missing_data', 'missing': 'p2_pose33'}
    assert z.shape == t_common.shape and z.dtype == np.float32
    assert not z.any()


def test_invalid_frames_produce_no_events(coupled_dyad):
    data, centres = coupled_dyad
    masked = dict(data)
    valid = data['p1_pose_features_valid'].copy()
    lo, hi = int(40 * FS), int(80 * FS)
    valid[lo:hi] = False
    masked['p1_pose_features_valid'] = valid
    times = events_from_pose_npz(masked, 'p1', 'landing')
    assert not np.any((times > 40.0) & (times < 80.0))
    n_outside = np.sum((centres < 40.0) | (centres > 80.0))
    assert abs(times.size - n_outside) <= 1


def test_events_from_pose_npz_contract(coupled_dyad):
    data, centres = coupled_dyad
    times = events_from_pose_npz(data, 'p1', 'landing')
    assert times.dtype == np.float64
    assert np.all(np.abs(np.sort(times) - centres) < 0.15)
    with pytest.raises(ValueError):
        events_from_pose_npz(data, 'p3', 'landing')
    with pytest.raises(KeyError):
        events_from_pose_npz({k: v for k, v in data.items() if 'ts' not in k}, 'p1', 'landing')
