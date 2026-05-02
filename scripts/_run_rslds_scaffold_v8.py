"""CADENCE RSLDS Scaffold V8: Fixed 9D coupling z-timecourses.

Fixes from critical review (2026-03-29):
  - EEG: smooth_samples=0 (removes rho=0.95 autocorrelation), per-band z-scores
  - BL: per-segment wavelet coherence z (not full-session diluted surrogates)
  - ECG: Hilbert envelope cross-product of bandpass-filtered IBI (LF/HF)
  - Resp: phase coherence cos(phi1-phi2) at respiratory band
  - Pose: upper-body velocity (first-difference) surrogate cross-product z
  - Conditional prewhitening: only channels with rho > 0.3
  - Variance standardization: all channels to mean=0, std=1

9D observation space:
  0: EEG theta    — per-band surrogate z-scored cross-product (smooth=0)
  1: EEG alpha    — per-band surrogate z-scored cross-product (smooth=0)
  2: EEG beta     — per-band surrogate z-scored cross-product (smooth=0)
  3: BL expression — per-segment wavelet coherence z (0.5-2 Hz)
  4: BL state      — per-segment wavelet coherence z (<0.5 Hz)
  5: ECG LF (SNS)  — Hilbert envelope cross-product, surrogate z (0.04-0.15 Hz)
  6: ECG HF (PNS)  — Hilbert envelope cross-product, surrogate z (0.15-0.4 Hz)
  7: Resp           — phase coherence cos(phi1-phi2) at 0.1-0.5 Hz
  8: Pose           — upper-body velocity cross-product z (11 features)

Usage:
    python scripts/_run_rslds_scaffold_v8.py
    python scripts/_run_rslds_scaffold_v8.py --session y_06 y_17
    python scripts/_run_rslds_scaffold_v8.py --all
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gf1d
import torch
from concurrent.futures import ThreadPoolExecutor

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    AFFECT_AUS, BAND_EXPRESSION, BAND_STATE,
)
from cadence.significance.fast_cycles import eeg_coupling_timecourse
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from scripts._extract_respiratory import extract_respiratory_one, detect_rpeaks

# ── Constants ─────────────────────────────────────────────────────────

FS_BL = 30.0
FS_OUT = 2.0
FS_IBI = 4.0         # IBI interpolation rate (before resampling to FS_OUT)
Z_THRESH = 2.0
PREWHITEN_RHO_THRESH = 0.3

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'

CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1',
                   'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}

# V8 9D modality configuration
MODALITY_KEYS = ['eeg_theta', 'eeg_alpha', 'eeg_beta',
                 'bl_expr', 'bl_state',
                 'ecg_lf', 'ecg_hf',
                 'resp', 'pose']
MODALITY_NAMES = ['EEG theta', 'EEG alpha', 'EEG beta',
                  'BL expression', 'BL state',
                  'ECG LF (SNS)', 'ECG HF (PNS)',
                  'Resp phase', 'Pose velocity']
MODALITY_COLORS = ['#90CAF9', '#A5D6A7', '#FFCC80',
                   '#E91E63', '#9C27B0',
                   '#FF5722', '#795548',
                   '#607D8B', '#4CAF50']

# Upper-body pose feature indices (from 40-channel pose features):
# head centroid xyz (3), head tilt pitch/yaw/roll (3), torso lean x/y/z (3),
# arm extension L/R (2) = 11 features
UPPER_BODY_POSE_IDX = list(range(6)) + [6, 7, 8, 9, 10]  # first 11

ECG_SRATE = 130  # Polar H10


# ── Utility: surrogate cross-product z (GPU) ─────────────────────────

def cross_product_z(p1, p2, n_surrogates=200, seed=42, device='auto'):
    """Cross-product coupling z-score with circular-shift surrogates.

    NO smoothing (smooth_samples=0) — maximizes temporal resolution.
    """
    p1 = np.asarray(p1, dtype=np.float64).copy()
    p2 = np.asarray(p2, dtype=np.float64).copy()
    if p1.ndim == 1:
        p1 = p1[:, None]
        p2 = p2[:, None]
    T, C = p1.shape
    rng = np.random.default_rng(seed)

    # Z-score per channel
    for c in range(C):
        for arr in [p1, p2]:
            mu, sd = arr[:, c].mean(), arr[:, c].std()
            if sd > 1e-8:
                arr[:, c] = (arr[:, c] - mu) / sd
            else:
                arr[:, c] = 0.0

    min_shift = max(1, int(0.1 * T))
    max_shift = max(min_shift + 1, int(0.9 * T))
    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    use_gpu = (device == 'cuda' or
               (device == 'auto' and torch.cuda.is_available()))

    if use_gpu:
        dev = torch.device('cuda')
        p1_t = torch.as_tensor(p1, dtype=torch.float32, device=dev)
        p2_t = torch.as_tensor(p2, dtype=torch.float32, device=dev)
        cp_t = (p1_t * p2_t).mean(dim=1)  # (T,) — no smoothing

        base_idx = torch.arange(T, device=dev)
        shifts_t = torch.as_tensor(shifts, dtype=torch.long, device=dev)
        shifted_idx = (base_idx[None, :] - shifts_t[:, None]) % T
        surr = (p1_t[shifted_idx] * p2_t[None, :, :]).mean(dim=2)  # no smoothing

        null_mean = surr.mean(dim=0)
        null_std = torch.clamp(surr.std(dim=0), min=1e-10)
        z = ((cp_t - null_mean) / null_std).cpu().numpy()
    else:
        cp = (p1 * p2).mean(axis=1)
        base_idx = np.arange(T)
        shifted_idx = (base_idx[None, :] - shifts[:, None]) % T
        surr = (p1[shifted_idx] * p2[None, :, :]).mean(axis=2)
        null_mean = surr.mean(axis=0)
        null_std = np.maximum(surr.std(axis=0), 1e-10)
        z = (cp - null_mean) / null_std

    return z.astype(np.float32)


# ── ECG: Hilbert envelope cross-product ──────────────────────────────

def ecg_hilbert_coupling(p1_ecg_raw, p1_ecg_ts, p2_ecg_raw, p2_ecg_ts,
                         t_common, lsl_offset, srate=ECG_SRATE):
    """Compute ECG LF and HF coupling via Hilbert envelope cross-product.

    1. R-peak detection → IBI timeseries
    2. Cubic spline interpolation to 4 Hz grid
    3. Bandpass (LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz)
    4. Hilbert → envelope → cross-product → surrogate z

    Returns:
        z_lf: (N,) LF coupling z on t_common grid
        z_hf: (N,) HF coupling z on t_common grid
    """
    N = len(t_common)
    z_lf = np.zeros(N, dtype=np.float32)
    z_hf = np.zeros(N, dtype=np.float32)

    # R-peak detection for both participants
    r1 = detect_rpeaks(p1_ecg_raw, p1_ecg_ts, srate)
    r2 = detect_rpeaks(p2_ecg_raw, p2_ecg_ts, srate)
    if r1[0] is None or r2[0] is None:
        return z_lf, z_hf

    _, _, _, ibis1, ibi_times1 = r1
    _, _, _, ibis2, ibi_times2 = r2

    if len(ibis1) < 30 or len(ibis2) < 30:
        return z_lf, z_hf

    # Adjust to LSL time
    ibi_times1_lsl = ibi_times1 + lsl_offset
    ibi_times2_lsl = ibi_times2 + lsl_offset

    # Common IBI grid at 4 Hz
    t_start = max(ibi_times1_lsl[0], ibi_times2_lsl[0], t_common[0])
    t_end = min(ibi_times1_lsl[-1], ibi_times2_lsl[-1], t_common[-1])
    if t_end - t_start < 60:  # need at least 60s
        return z_lf, z_hf

    t_ibi = np.arange(t_start, t_end, 1.0 / FS_IBI)

    # Cubic spline interpolation (avoids broadband artifacts from linear interp)
    f1 = interp1d(ibi_times1_lsl, ibis1, kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(ibi_times2_lsl, ibis2, kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    ibi1_uniform = np.clip(f1(t_ibi), 0.3, 2.0)
    ibi2_uniform = np.clip(f2(t_ibi), 0.3, 2.0)

    T_ibi = len(t_ibi)

    # Process LF and HF bands
    for band_name, lo, hi in [('lf', 0.04, 0.15), ('hf', 0.15, 0.4)]:
        nyq = FS_IBI / 2.0
        if hi >= nyq:
            hi = nyq * 0.95
        sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')

        try:
            bp1 = sosfiltfilt(sos, ibi1_uniform)
            bp2 = sosfiltfilt(sos, ibi2_uniform)
        except ValueError:
            continue

        # Hilbert → analytic envelope
        env1 = np.abs(hilbert(bp1))
        env2 = np.abs(hilbert(bp2))

        # Cross-product surrogate z (no smoothing)
        z_band = cross_product_z(env1, env2, n_surrogates=200, seed=42)

        # Resample to t_common (2 Hz)
        z_on_common = np.interp(t_common, t_ibi, z_band,
                                left=0, right=0).astype(np.float32)

        if band_name == 'lf':
            z_lf = z_on_common
        else:
            z_hf = z_on_common

    return z_lf, z_hf


# ── Respiratory: phase coherence ─────────────────────────────────────

def respiratory_phase_coherence(p1_resp, p2_resp, t_common, lsl_offset):
    """Phase coherence cos(phi1-phi2) from EDR fused signals.

    Bandpass 0.1-0.5 Hz → Hilbert → instantaneous phase → cos(dphi).
    Smooth with 3s Gaussian (single breath cycle).

    Returns:
        z_resp: (N,) phase coherence on t_common grid
    """
    N = len(t_common)
    z_resp = np.zeros(N, dtype=np.float32)

    if p1_resp is None or p2_resp is None:
        return z_resp

    # Get fused EDR waveforms
    fused1 = p1_resp['fused']
    fused2 = p2_resp['fused']
    t1 = p1_resp['t'] + lsl_offset
    t2 = p2_resp['t'] + lsl_offset
    fs_edr = 4.0

    # Common grid at EDR rate
    t_start = max(t1[0], t2[0], t_common[0])
    t_end = min(t1[-1], t2[-1], t_common[-1])
    if t_end - t_start < 30:
        return z_resp

    t_edr = np.arange(t_start, t_end, 1.0 / fs_edr)
    sig1 = np.interp(t_edr, t1, fused1, left=0, right=0)
    sig2 = np.interp(t_edr, t2, fused2, left=0, right=0)

    # Bandpass 0.1-0.5 Hz
    nyq = fs_edr / 2.0
    sos = butter(4, [0.1 / nyq, 0.5 / nyq], btype='band', output='sos')
    try:
        sig1_bp = sosfiltfilt(sos, sig1)
        sig2_bp = sosfiltfilt(sos, sig2)
    except ValueError:
        return z_resp

    # Instantaneous phase via Hilbert
    phase1 = np.angle(hilbert(sig1_bp))
    phase2 = np.angle(hilbert(sig2_bp))

    # Phase coherence: cos(phi1 - phi2)
    coh = np.cos(phase1 - phase2).astype(np.float32)

    # Light smoothing: 3s Gaussian (appropriate for single breath cycle)
    coh_smooth = gf1d(coh, sigma=3.0 * fs_edr).astype(np.float32)

    # Resample to t_common
    z_resp = np.interp(t_common, t_edr, coh_smooth,
                       left=0, right=0).astype(np.float32)

    return z_resp


# ── Pose: upper-body velocity cross-product ──────────────────────────

POSE_MAX_LAG_S = 5.0  # max lag for multi-lag cross-product (±5s)


def pose_velocity_coupling(cached, t_common, lsl_offset):
    """Upper-body multi-lag velocity cross-product z.

    1. Extract upper-body features (11 channels)
    2. First-difference → velocity
    3. Multi-lag cross-product bank: compute cross-product at each lag
       in ±5s (±10 samples at 2 Hz), take max absolute value per timepoint
    4. Surrogate z-score the max-lag cross-product

    The multi-lag bank detects BOTH zero-lag synchrony (postural mirroring)
    AND short-lag reciprocal coupling (head nods with 0.5-3s response delay).
    Without multi-lag, the zero-lag cross-product misses lagged coupling
    because velocity autocorrelation at nonzero lags is negative for
    first-differenced signals.

    Returns:
        z_pose: (N,) coupling z on t_common grid
    """
    N = len(t_common)
    z_pose = np.zeros(N, dtype=np.float32)

    has_pose = ('p1_pose_features' in cached and 'p2_pose_features' in cached
                and 'p1_pose_features_ts' in cached and 'p2_pose_features_ts' in cached)
    if not has_pose:
        return z_pose

    p1_pose = cached['p1_pose_features']
    p2_pose = cached['p2_pose_features']
    t1p = cached['p1_pose_features_ts'] + lsl_offset
    t2p = cached['p2_pose_features_ts'] + lsl_offset

    n_ch = min(p1_pose.shape[1], p2_pose.shape[1], max(UPPER_BODY_POSE_IDX) + 1)
    idx = [i for i in UPPER_BODY_POSE_IDX if i < n_ch]
    if len(idx) < 3:
        return z_pose

    # Interpolate to common 2 Hz grid
    fs_out = 1.0 / max(t_common[1] - t_common[0], 0.01)
    p1_on_grid = np.column_stack([
        np.interp(t_common, t1p, p1_pose[:, c], left=0, right=0) for c in idx])
    p2_on_grid = np.column_stack([
        np.interp(t_common, t2p, p2_pose[:, c], left=0, right=0) for c in idx])

    # First-difference (velocity)
    p1_vel = np.diff(p1_on_grid, axis=0, prepend=p1_on_grid[:1])
    p2_vel = np.diff(p2_on_grid, axis=0, prepend=p2_on_grid[:1])

    # Z-score velocity per channel
    T, C = p1_vel.shape
    for c in range(C):
        for arr in [p1_vel, p2_vel]:
            s = arr[:, c].std()
            if s > 1e-8:
                arr[:, c] = (arr[:, c] - arr[:, c].mean()) / s
            else:
                arr[:, c] = 0.0

    # Multi-lag cross-product bank: lags from -max_lag to +max_lag
    max_lag_samp = int(POSE_MAX_LAG_S * fs_out)
    lags = range(-max_lag_samp, max_lag_samp + 1)

    # Compute cross-product at each lag, take max absolute value per timepoint
    cp_per_lag = np.zeros((len(lags), T), dtype=np.float64)
    for li, lag in enumerate(lags):
        p2_shifted = np.roll(p2_vel, lag, axis=0)
        if lag > 0:
            p2_shifted[:lag] = 0
        elif lag < 0:
            p2_shifted[lag:] = 0
        cp_per_lag[li] = (p1_vel * p2_shifted).mean(axis=1)

    # Best lag per timepoint (max absolute cross-product)
    best_lag_idx = np.argmax(np.abs(cp_per_lag), axis=0)
    cp_best = cp_per_lag[best_lag_idx, np.arange(T)]

    # Surrogate z-score: circular-shift P2 velocity, recompute max-lag cp
    rng = np.random.default_rng(42)
    n_surrogates = 200
    min_shift = max(1, int(0.1 * T))
    max_shift = max(min_shift + 1, int(0.9 * T))
    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    surr_cp = np.zeros((n_surrogates, T), dtype=np.float64)
    for si, shift in enumerate(shifts):
        p2_surr = np.roll(p2_vel, int(shift), axis=0)
        # Compute max-lag cp for this surrogate
        best_surr = np.zeros(T, dtype=np.float64)
        for lag in lags:
            p2s_shifted = np.roll(p2_surr, lag, axis=0)
            if lag > 0:
                p2s_shifted[:lag] = 0
            elif lag < 0:
                p2s_shifted[lag:] = 0
            cp_lag = (p1_vel * p2s_shifted).mean(axis=1)
            # Update best where this lag is better
            better = np.abs(cp_lag) > np.abs(best_surr)
            best_surr[better] = cp_lag[better]
        surr_cp[si] = best_surr

    null_mean = surr_cp.mean(axis=0)
    null_std = np.maximum(surr_cp.std(axis=0), 1e-10)
    z_pose = ((cp_best - null_mean) / null_std).astype(np.float32)

    return z_pose


# ── Conditional prewhitening + standardization ───────────────────────

def prewhiten_and_standardize(z_matrix, keys, rho_thresh=PREWHITEN_RHO_THRESH,
                              valid_mask=None):
    """Conditionally prewhiten channels with lag-1 rho > threshold, then standardize.

    Args:
        z_matrix: (T, D) coupling z-timecourses
        keys: list of D modality names
        rho_thresh: prewhiten only if rho > this
        valid_mask: (T, D) boolean, True = real data. When provided, rho/mean/std
            are computed only on valid timepoints. Invalid timepoints are zeroed
            after prewhitening. This prevents structural zeros (from np.interp
            edge padding, inter-segment gaps) from corrupting AR(1) estimates.

    Returns:
        z_out: (T, D) prewhitened + standardized
        diagnostics: dict with per-channel rho, std before/after
    """
    T, D = z_matrix.shape
    z_out = z_matrix.copy()
    diag = {'rho_before': {}, 'rho_after': {}, 'std_before': {}, 'std_after': {},
            'prewhitened': {}}

    for d in range(D):
        z = z_out[:, d]
        key = keys[d]
        v = valid_mask[:, d] if valid_mask is not None else None

        # Lag-1 autocorrelation (on valid timepoints only)
        z_valid = z[v] if v is not None else z
        if len(z_valid) < 10 or np.std(z_valid) < 1e-8:
            diag['rho_before'][key] = 0.0
            diag['rho_after'][key] = 0.0
            diag['std_before'][key] = 0.0
            diag['std_after'][key] = 0.0
            diag['prewhitened'][key] = False
            continue

        # Compute rho on consecutive valid pairs only
        if v is not None:
            pair_valid = v[:-1] & v[1:]
            if pair_valid.sum() < 10:
                rho = 0.0
            else:
                rho = np.corrcoef(z[:-1][pair_valid], z[1:][pair_valid])[0, 1]
        else:
            rho = np.corrcoef(z[:-1], z[1:])[0, 1]
        diag['rho_before'][key] = float(rho)
        diag['std_before'][key] = float(np.std(z_valid))

        # Conditional prewhitening (iterative — up to 3 rounds until rho < threshold)
        # AR(1) filter runs on ALL timepoints for continuity; rho estimated from valid only
        n_rounds = 0
        z_cur = z.copy()
        while abs(rho) > rho_thresh and n_rounds < 3:
            z_pw = z_cur[1:] - rho * z_cur[:-1]
            z_cur = np.concatenate([[0.0], z_pw])
            n_rounds += 1
            if v is not None:
                # Re-estimate rho on consecutive valid pairs of prewhitened signal
                pw_pairs = v[1:-1] & v[2:]
                if pw_pairs.sum() >= 10:
                    z_valid_pw = z_cur[1:-1][pw_pairs]
                    if np.std(z_valid_pw) > 1e-8:
                        rho = np.corrcoef(z_cur[1:-1][pw_pairs], z_cur[2:][pw_pairs])[0, 1]
                    else:
                        rho = 0.0
                else:
                    rho = 0.0
            else:
                if len(z_cur) > 3 and np.std(z_cur) > 1e-8:
                    rho = np.corrcoef(z_cur[1:-1], z_cur[2:])[0, 1]
                else:
                    rho = 0.0

        if n_rounds > 0:
            z_out[:, d] = z_cur
            diag['prewhitened'][key] = True
        else:
            diag['prewhitened'][key] = False

        rho_after = rho
        diag['rho_after'][key] = float(rho_after)

    # Standardize all channels to mean=0, std=1
    for d in range(D):
        key = keys[d]
        z = z_out[:, d]
        v = valid_mask[:, d] if valid_mask is not None else None
        z_valid = z[v] if v is not None else z

        if len(z_valid) < 10:
            z_out[:, d] = 0.0
            diag['std_after'][key] = 0.0
            continue

        mu = z_valid.mean()
        sd = z_valid.std()
        if sd > 1e-8:
            z_out[:, d] = (z - mu) / sd
        else:
            z_out[:, d] = 0.0
        # Zero invalid timepoints
        if v is not None:
            z_out[~v, d] = 0.0
        diag['std_after'][key] = float(z_out[:, d][v].std() if v is not None else z_out[:, d].std())

    return z_out.astype(np.float32), diag


# ══════════════════════════════════════════════════════════════════════
#  SESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name, config, raw_dir=RAW_DIR):
    """Run V8 scaffold for one session: 9D coupling z-timecourses.

    Returns:
        results: dict with all metrics, or None if session cannot be processed.
    """
    print(f"\n{'='*70}")
    print(f"  RSLDS Scaffold V8 — {session_name}")
    print(f"{'='*70}")

    t_wall = time.time()
    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load session ──────────────────────────────────────────────
    print(f"\n[1/8] Loading {session_name}...", flush=True)

    xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  WARNING: No XDF found for {session_name}, skipping")
        return None
    xdf_path = xdf_files[0]

    session_data = load_xdf_session(xdf_path)
    markers = session_data['markers']
    p1_role = session_data['p1_role']
    p2_role = session_data['p2_role']
    print(f"  Roles: P1={p1_role}, P2={p2_role}")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        print(f"  WARNING: No cache for {session_name}, skipping")
        return None
    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    # Segment boundaries from XDF markers
    segments = []
    for seg_name in CONDITION_ORDER:
        t_start = markers.get(f'{seg_name}_start')
        t_end = markers.get(f'{seg_name}_stop')
        if t_start is not None and t_end is not None:
            segments.append((seg_name, t_start, t_end))

    if not segments:
        print(f"  WARNING: No segments for {session_name}, skipping")
        return None

    session_start_lsl = min(t for _, t, _ in segments) - 30
    session_end_lsl = max(t for _, _, t in segments) + 30
    t_common = np.arange(session_start_lsl, session_end_lsl, 1.0 / FS_OUT)
    N_common = len(t_common)
    print(f"  Grid: {N_common} pts, {(session_end_lsl - session_start_lsl):.0f}s, "
          f"segs: {[s[0] for s in segments]}")

    z_traces = {k: np.zeros(N_common, dtype=np.float32) for k in MODALITY_KEYS}

    # ── 2. EEG: per-band z-scores, smooth_samples=0 ─────────────────
    print(f"\n[2/8] EEG per-band coupling (smooth=0)...", flush=True)
    t0 = time.time()

    _has_eeg = ('p1_eeg' in cached and 'p2_eeg' in cached
                and 'p1_eeg_ts' in cached and 'p2_eeg_ts' in cached)
    if _has_eeg:
        p1_eeg_all = cached['p1_eeg'].astype(np.float64)
        p2_eeg_all = cached['p2_eeg'].astype(np.float64)
        p1_eeg_ts = cached['p1_eeg_ts']
        n_ch = min(14, p1_eeg_all.shape[1], p2_eeg_all.shape[1])
        p1_eeg_all = p1_eeg_all[:, :n_ch]
        p2_eeg_all = p2_eeg_all[:, :n_ch]
        mlen = min(len(p1_eeg_all), len(p2_eeg_all))
        p1_eeg_all = p1_eeg_all[:mlen]; p2_eeg_all = p2_eeg_all[:mlen]
        # Avg-reference
        p1_eeg_all -= p1_eeg_all.mean(axis=1, keepdims=True)
        p2_eeg_all -= p2_eeg_all.mean(axis=1, keepdims=True)
        # Z-score per channel
        for ch in range(n_ch):
            for arr in [p1_eeg_all, p2_eeg_all]:
                sd = arr[:, ch].std()
                if sd > 1e-8:
                    arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd
        fs_eeg = len(p1_eeg_ts) / (p1_eeg_ts[-1] - p1_eeg_ts[0])

        # Key V8 change: smooth_samples=0
        eeg_r = eeg_coupling_timecourse(
            p1_eeg_all, p2_eeg_all, fs_eeg,
            smooth_samples=0, n_surrogates=100, seed=42)

        eeg_times_lsl = eeg_r['times'] + (p1_eeg_ts[0] + lsl_offset)

        for bn in ['theta', 'alpha', 'beta']:
            bd = eeg_r['per_band'].get(bn, {})
            if bd.get('n_valid', 0) > 0:
                z_traces[f'eeg_{bn}'] = np.interp(
                    t_common, eeg_times_lsl, bd['z'],
                    left=0, right=0).astype(np.float32)
                print(f"    {bn}: mean_z={bd['z'].mean():+.2f}, "
                      f"cf={float((bd['z'] > 2.0).mean()):.1%}")

    print(f"  EEG: {time.time() - t0:.1f}s")

    # ── 3. BL: per-segment wavelet coherence ─────────────────────────
    print(f"\n[3/8] BL per-segment wavelet coherence...", flush=True)
    t0 = time.time()

    landmarks = session_data['landmarks']

    for seg_name, t0_seg, t1_seg in segments:
        p1_bl, p2_bl, bl_dur = extract_bl_segment(
            landmarks, t0_seg, t1_seg)
        if p1_bl is None or bl_dur < 5:
            continue

        T_bl = p1_bl.shape[0]
        bl_times = np.linspace(t0_seg, t1_seg, T_bl)

        # CWT on segment affect AUs
        scal_p1 = compute_au_cwt(p1_bl[:, AFFECT_AUS])
        scal_p2 = compute_au_cwt(p2_bl[:, AFFECT_AUS])
        freqs = scal_p1.freqs

        # Segment-local surrogates
        z_coh = surrogate_coherence_z(scal_p1, scal_p2,
                                       200, 0.5,
                                       list(range(len(AFFECT_AUS))), 42, 'auto')

        expr_fmask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
        state_fmask = (freqs >= BAND_STATE[0]) & (freqs < BAND_STATE[1])

        # Expression band: max z across frequencies per timepoint
        if expr_fmask.sum() > 0:
            expr_z_30 = z_coh['z'][expr_fmask].max(axis=0)
            seg_mask = (t_common >= t0_seg) & (t_common <= t1_seg)
            if seg_mask.sum() > 0:
                z_traces['bl_expr'][seg_mask] = np.interp(
                    t_common[seg_mask], bl_times, expr_z_30,
                    left=0, right=0).astype(np.float32)

        # State band: mean z across frequencies per timepoint
        if state_fmask.sum() > 0:
            state_z_30 = z_coh['z'][state_fmask].mean(axis=0)
            seg_mask = (t_common >= t0_seg) & (t_common <= t1_seg)
            if seg_mask.sum() > 0:
                z_traces['bl_state'][seg_mask] = np.interp(
                    t_common[seg_mask], bl_times, state_z_30,
                    left=0, right=0).astype(np.float32)

        print(f"    {seg_name}: {bl_dur:.0f}s, "
              f"expr_z={z_traces['bl_expr'][(t_common >= t0_seg) & (t_common <= t1_seg)].mean():+.2f}")

    print(f"  BL: {time.time() - t0:.1f}s")

    # ── 4. ECG: Hilbert envelope cross-product (LF/HF) ──────────────
    print(f"\n[4/8] ECG Hilbert envelope coupling...", flush=True)
    t0 = time.time()

    has_raw_ecg = ('p1_ecg' in cached and 'p2_ecg' in cached
                   and len(cached.get('p1_ecg', [])) > 1000)
    if has_raw_ecg:
        z_traces['ecg_lf'], z_traces['ecg_hf'] = ecg_hilbert_coupling(
            cached['p1_ecg'], cached['p1_ecg_ts'],
            cached['p2_ecg'], cached['p2_ecg_ts'],
            t_common, lsl_offset, srate=ECG_SRATE)
        print(f"    LF: mean_z={z_traces['ecg_lf'].mean():+.2f}, "
              f"HF: mean_z={z_traces['ecg_hf'].mean():+.2f}")
    else:
        print(f"    No raw ECG, skipping")

    print(f"  ECG: {time.time() - t0:.1f}s")

    # ── 5. Respiratory: phase coherence ──────────────────────────────
    print(f"\n[5/8] Respiratory phase coherence...", flush=True)
    t0 = time.time()

    if has_raw_ecg:
        # Extract EDR from raw ECG
        p1_resp = extract_respiratory_one(cached['p1_ecg'], cached['p1_ecg_ts'], srate=ECG_SRATE)
        p2_resp = extract_respiratory_one(cached['p2_ecg'], cached['p2_ecg_ts'], srate=ECG_SRATE)
        z_traces['resp'] = respiratory_phase_coherence(p1_resp, p2_resp, t_common, lsl_offset)
        print(f"    mean_coh={z_traces['resp'].mean():+.3f}")
    else:
        print(f"    No raw ECG for EDR, skipping")

    print(f"  Resp: {time.time() - t0:.1f}s")

    # ── 6. Pose: upper-body velocity cross-product ───────────────────
    print(f"\n[6/8] Pose upper-body velocity coupling...", flush=True)
    t0 = time.time()

    z_traces['pose'] = pose_velocity_coupling(cached, t_common, lsl_offset)
    print(f"    mean_z={z_traces['pose'].mean():+.2f}, "
          f"cf={float((z_traces['pose'] > 2.0).mean()):.1%}")

    print(f"  Pose: {time.time() - t0:.1f}s")

    # ── 7. Combine + conditional prewhiten + standardize ─────────────
    print(f"\n[7/8] Prewhitening + standardization...", flush=True)

    z_matrix_raw = np.column_stack([z_traces[k] for k in MODALITY_KEYS])
    z_matrix, pw_diag = prewhiten_and_standardize(z_matrix_raw, MODALITY_KEYS)

    # Build per-timepoint observation mask
    # Start with whole-modality mask (False if entire channel is zero)
    obs_mask = np.ones((N_common, len(MODALITY_KEYS)), dtype=bool)
    for d, key in enumerate(MODALITY_KEYS):
        if np.abs(z_matrix_raw[:, d]).max() < 1e-8:
            obs_mask[:, d] = False

    # Per-timepoint masking: face/pose validity from cache
    # BL channels (bl_expr, bl_state) require BOTH participants to have valid face
    # Pose channel requires BOTH participants to have valid pose
    bl_idx = [MODALITY_KEYS.index(k) for k in ['bl_expr', 'bl_state']]
    pose_idx = MODALITY_KEYS.index('pose')

    def _interp_valid(valid_arr, ts_arr, t_out, lsl_off):
        """Interpolate boolean validity to common grid (nearest-neighbor)."""
        v_float = valid_arr.astype(np.float32)
        v_interp = np.interp(t_out, ts_arr + lsl_off, v_float, left=0, right=0)
        return v_interp > 0.5  # threshold back to boolean

    # Face validity
    has_bl_valid = ('p1_blendshapes_valid' in cached and 'p2_blendshapes_valid' in cached
                    and 'p1_blendshapes_ts' in cached and 'p2_blendshapes_ts' in cached)
    if has_bl_valid:
        p1_bl_v = _interp_valid(cached['p1_blendshapes_valid'],
                                cached['p1_blendshapes_ts'], t_common, lsl_offset)
        p2_bl_v = _interp_valid(cached['p2_blendshapes_valid'],
                                cached['p2_blendshapes_ts'], t_common, lsl_offset)
        bl_valid = p1_bl_v & p2_bl_v  # both must have face detected
        for d in bl_idx:
            obs_mask[:, d] &= bl_valid
        n_bl_masked = int((~bl_valid).sum())
        print(f"  BL per-timepoint mask: {n_bl_masked} pts masked "
              f"({n_bl_masked/N_common:.1%} of session)")

    # Pose validity
    has_pose_valid = ('p1_pose_features_valid' in cached and 'p2_pose_features_valid' in cached
                      and 'p1_pose_features_ts' in cached and 'p2_pose_features_ts' in cached)
    if has_pose_valid:
        p1_pose_v = _interp_valid(cached['p1_pose_features_valid'],
                                  cached['p1_pose_features_ts'], t_common, lsl_offset)
        p2_pose_v = _interp_valid(cached['p2_pose_features_valid'],
                                  cached['p2_pose_features_ts'], t_common, lsl_offset)
        pose_valid = p1_pose_v & p2_pose_v
        obs_mask[:, pose_idx] &= pose_valid
        n_pose_masked = int((~pose_valid).sum())
        print(f"  Pose per-timepoint mask: {n_pose_masked} pts masked "
              f"({n_pose_masked/N_common:.1%} of session)")

    print(f"\n  {'Modality':>15s} | rho_raw | rho_pw | prewhitened | std_final")
    print(f"  " + "-" * 65)
    for key in MODALITY_KEYS:
        print(f"  {key:>15s} | {pw_diag['rho_before'].get(key, 0):.3f}   | "
              f"{pw_diag['rho_after'].get(key, 0):.3f}  | "
              f"{'YES' if pw_diag['prewhitened'].get(key, False) else 'no ':3s}         | "
              f"{pw_diag['std_after'].get(key, 0):.3f}")

    # ── 8. Save + visualize ──────────────────────────────────────────
    print(f"\n[8/8] Save + visualize...", flush=True)

    # Save NPZ (V8 format)
    save_dict = {
        't_common': t_common,
        'obs_mask': obs_mask,
    }
    for i, key in enumerate(MODALITY_KEYS):
        save_dict[f'z_{key}'] = z_matrix[:, i]
        save_dict[f'z_raw_{key}'] = z_matrix_raw[:, i]

    np.savez_compressed(os.path.join(out_dir, 'rslds_scaffold_v8_ztimecourses.npz'),
                        **save_dict)

    # Save JSON results
    results = {
        'session': session_name,
        'version': 'v8',
        'n_timepoints': int(N_common),
        'duration_s': float(session_end_lsl - session_start_lsl),
        'fs_out': FS_OUT,
        'modality_keys': MODALITY_KEYS,
        'segments': [(s, float(t0s), float(t1s)) for s, t0s, t1s in segments],
        'prewhitening': {
            'threshold': PREWHITEN_RHO_THRESH,
            'rho_before': pw_diag['rho_before'],
            'rho_after': pw_diag['rho_after'],
            'prewhitened': pw_diag['prewhitened'],
        },
        'per_modality': {},
    }
    for key in MODALITY_KEYS:
        z = z_matrix[:, MODALITY_KEYS.index(key)]
        z_raw = z_matrix_raw[:, MODALITY_KEYS.index(key)]
        results['per_modality'][key] = {
            'mean_z_raw': float(z_raw.mean()),
            'std_z_raw': float(z_raw.std()),
            'rho_raw': pw_diag['rho_before'].get(key, 0),
            'rho_after': pw_diag['rho_after'].get(key, 0),
            'coupling_fraction_raw': float((z_raw > Z_THRESH).mean()),
            'missing': bool(not obs_mask[:, MODALITY_KEYS.index(key)].all()),
        }

    with open(os.path.join(out_dir, 'rslds_scaffold_v8_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization: 9-panel z-timecourses
    fig, axes = plt.subplots(len(MODALITY_KEYS), 1,
                             figsize=(26, 2.5 * len(MODALITY_KEYS)),
                             sharex=True)

    for ax in axes:
        for seg_name, t0s, t1s in segments:
            ax.axvspan(t0s, t1s, alpha=0.25,
                       color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)

    for seg_name, t0s, t1s in segments:
        axes[0].text((t0s + t1s) / 2, 1.08, seg_name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        z = z_matrix[:, idx]
        z_raw = z_matrix_raw[:, idx]

        ax.plot(t_common, z, color=color, linewidth=0.6, alpha=0.8, label='processed')
        ax.plot(t_common, z_raw, color='gray', linewidth=0.3, alpha=0.3, label='raw')
        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
        ax.axhline(2.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

        rho_b = pw_diag['rho_before'].get(key, 0)
        rho_a = pw_diag['rho_after'].get(key, 0)
        pw = 'PW' if pw_diag['prewhitened'].get(key, False) else ''
        ax.text(0.01, 0.92, f'rho={rho_b:.2f}→{rho_a:.2f} {pw}',
                transform=ax.transAxes, fontsize=7, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_ylabel(f'{name}\nz', fontsize=8)

    axes[-1].set_xlabel('LSL time (s)', fontsize=9)

    fig.suptitle(f'{session_name} — V8 Scaffold (9D, prewhitened + standardized)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'rslds_scaffold_v8_timeline.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved timeline to {out_dir}/")

    total = time.time() - t_wall
    print(f"\n  V8 scaffold complete: {total:.0f}s")
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RSLDS V8 Scaffold')
    parser.add_argument('--session', nargs='+', default=['y_06'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    config = load_config()

    if args.all:
        xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
        sessions = []
        for xf in xdf_files:
            name = os.path.splitext(os.path.basename(xf))[0].lower()
            # Normalize session name
            for prefix in ['y_', 'y']:
                if name.startswith(prefix):
                    sessions.append(name)
                    break
        sessions = sorted(set(sessions))
    else:
        sessions = args.session

    print(f"Sessions: {sessions}")

    for sname in sessions:
        run_session(sname, config)


if __name__ == '__main__':
    main()
