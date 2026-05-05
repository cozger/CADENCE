"""CADENCE RSLDS Scaffold Phase 1: Full-session multi-modal coupling z-timecourses.

Computes full-session (not per-segment) coupling z-timecourses for 6+ modalities:
  1. EEG:      volt_amp cross-product z (Stouffer theta/alpha/beta) + per-band
  2. BL expr:  wavelet coherence z (0.5-2 Hz, affect AUs) + smile/speech sub-groups
  3. BL state: wavelet coherence z (<0.5 Hz, affect AUs)
  4. ECG SNS:  multi-channel [IBI_dev, HR_accel, QRS_amp] cross-product z
  5. ECG PNS:  RMSSD cross-product z (parasympathetic proxy)
  6. Pose:     joint-feature cross-product z (all 40 channels)

All aligned to a common 2 Hz grid on LSL timestamps spanning the entire session.

Then computes:
  - Coupling flexibility: transition counts, dwell times, Shannon entropy
  - Distributional stats: skewness, kurtosis of z-timecourses
  - Cross-modal dynamics: mask cross-correlation at ±30s lags
  - Empirical FPR calibration from baseline conditions across sessions
  - Full-session timeline visualization with per-band/AU-group breakdown

Usage:
    python scripts/_run_rslds_scaffold.py                 # y_06 only (default)
    python scripts/_run_rslds_scaffold.py --session y_06 y_17
    python scripts/_run_rslds_scaffold.py --all           # all cached sessions
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.stats import skew as _skew, kurtosis as _kurtosis
import torch
from concurrent.futures import ThreadPoolExecutor

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    AFFECT_AUS, SMILE_AUS, SPEECH_AUS,
    BAND_EXPRESSION, BAND_STATE, BAND_SPEECH,
)
from cadence.significance.fast_cycles import (
    eeg_coupling_timecourse, extract_all_volt_amp, eeg_coupling_from_precomputed,
)
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from scripts._extract_respiratory import extract_respiratory_one, parse_conditions

# ── Constants ─────────────────────────────────────────────────────────

FS_BL = 30.0
FS_OUT = 2.0       # Common output rate (Hz)
Z_THRESH = 2.0     # One-tailed significance threshold
MAX_LAG_S = 30.0   # Cross-modal lag analysis window (seconds)

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
NULL_CONDITIONS = ['base_EO', 'base_EC']  # For empirical FPR calibration
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}

# ECG multi-channel SNS: fastest-changing sympathetic features
ECG_SNS_CHS = [1, 3, 4]   # IBI_dev (autocorr=0.82), HR_accel (0.93), QRS_amp (0.83)
ECG_PNS_CH = 2             # RMSSD (parasympathetic proxy)
ECG_SMOOTH = 2             # 1s sigma at 2 Hz (was 6 = 3s)

# Primary modality configuration (7D observation vector)
MODALITY_KEYS = ['eeg', 'bl_expr', 'bl_state', 'ecg_sns', 'ecg_pns', 'resp', 'pose']
MODALITY_NAMES = ['EEG volt_amp', 'BL expression', 'BL state',
                  'ECG SNS (multi-ch)', 'ECG PNS (RMSSD)', 'Resp rate', 'Pose']
MODALITY_COLORS = ['#2196F3', '#E91E63', '#9C27B0',
                   '#FF5722', '#795548', '#607D8B', '#4CAF50']

# Sub-modality keys for breakdowns (added to NPZ/JSON but not to primary 6D)
EEG_BAND_KEYS = ['eeg_theta', 'eeg_alpha', 'eeg_beta']
EEG_BAND_COLORS = {'eeg_theta': '#90CAF9', 'eeg_alpha': '#A5D6A7', 'eeg_beta': '#FFCC80'}
BL_SUBGROUP_KEYS = ['bl_expr_smile', 'bl_expr_speech']

# All keys that get z-timecourses and flexibility metrics
ALL_TRACE_KEYS = MODALITY_KEYS + EEG_BAND_KEYS + BL_SUBGROUP_KEYS


# ── Utility functions (unchanged from Phase 1) ───────────────────────

def cross_product_z(p1, p2, smooth_samples=3, n_surrogates=200, seed=42, device='auto'):
    """Cross-product coupling z-score with circular-shift surrogates.

    GPU-accelerated: all surrogates computed via batched index gathering,
    smoothed with conv1d Gaussian kernel, z-scored — entirely on GPU.
    Falls back to numpy for device='cpu'.
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

    # Generate shifts on CPU (numpy RNG for reproducibility)
    min_shift = max(1, int(0.1 * T))
    max_shift = max(min_shift + 1, int(0.9 * T))
    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    use_gpu = (device == 'cuda' or
               (device == 'auto' and torch.cuda.is_available()))

    if use_gpu:
        dev = torch.device('cuda')
        p1_t = torch.as_tensor(p1, dtype=torch.float32, device=dev)  # (T, C)
        p2_t = torch.as_tensor(p2, dtype=torch.float32, device=dev)

        # Cross-product
        cp_t = (p1_t * p2_t).mean(dim=1)  # (T,)

        # Gaussian smoothing kernel for conv1d
        if smooth_samples > 0:
            ks = int(6 * smooth_samples) | 1
            t_k = torch.arange(ks, device=dev, dtype=torch.float32) - ks // 2
            kernel = torch.exp(-0.5 * (t_k / smooth_samples) ** 2)
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            pad = ks // 2

            def _smooth(x):
                # x: (N, T) or (T,)
                if x.dim() == 1:
                    return torch.nn.functional.conv1d(
                        x.view(1, 1, -1), kernel, padding=pad).view(-1)
                return torch.nn.functional.conv1d(
                    x.unsqueeze(1), kernel, padding=pad).squeeze(1)

            cp_t = _smooth(cp_t)

        # Batched surrogates: build shifted index tensor
        base_idx = torch.arange(T, device=dev)
        shifts_t = torch.as_tensor(shifts, dtype=torch.long, device=dev)
        shifted_idx = (base_idx[None, :] - shifts_t[:, None]) % T  # (n_surr, T)

        # Gather, multiply, mean across channels: (n_surr, T)
        surr = (p1_t[shifted_idx] * p2_t[None, :, :]).mean(dim=2)

        if smooth_samples > 0:
            surr = _smooth(surr)

        null_mean = surr.mean(dim=0)
        null_std = torch.clamp(surr.std(dim=0), min=1e-10)
        z = ((cp_t - null_mean) / null_std).cpu().numpy()
    else:
        # CPU fallback (numpy vectorized)
        cp = (p1 * p2).mean(axis=1)
        if smooth_samples > 0:
            cp = gf1d(cp, sigma=smooth_samples)

        base_idx = np.arange(T)
        shifted_idx = (base_idx[None, :] - shifts[:, None]) % T
        surr = (p1[shifted_idx] * p2[None, :, :]).mean(axis=2)
        if smooth_samples > 0:
            surr = gf1d(surr, sigma=smooth_samples, axis=1)

        null_mean = surr.mean(axis=0)
        null_std = np.maximum(surr.std(axis=0), 1e-10)
        z = (cp - null_mean) / null_std

    mask = z > Z_THRESH
    return {'z': z.astype(np.float32), 'mask': mask,
            'mean_z': float(z.mean()), 'coupling_fraction': float(mask.mean())}


def flexibility_metrics(z, mask, fs=FS_OUT):
    """Coupling flexibility metrics from z-timecourse and binary mask."""
    T = len(mask)
    dt = 1.0 / fs
    n_transitions = int(np.abs(np.diff(mask.astype(int))).sum())
    transition_rate = n_transitions / (T * dt) if T > 0 else 0.0
    on_dwells, off_dwells = [], []
    if len(mask) > 0:
        current = bool(mask[0])
        run_len = 1
        for i in range(1, len(mask)):
            if mask[i] == current:
                run_len += 1
            else:
                (on_dwells if current else off_dwells).append(run_len * dt)
                current = bool(mask[i])
                run_len = 1
        (on_dwells if current else off_dwells).append(run_len * dt)
    on_dwells = np.array(on_dwells) if on_dwells else np.array([])
    off_dwells = np.array(off_dwells) if off_dwells else np.array([])
    all_dwells = np.concatenate([on_dwells, off_dwells]) if len(on_dwells) + len(off_dwells) > 0 else np.array([])
    if len(all_dwells) > 2:
        n_bins = max(5, int(np.sqrt(len(all_dwells))))
        counts, _ = np.histogram(all_dwells, bins=n_bins)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        shannon_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        shannon_entropy = 0.0
    z_f = z[np.isfinite(z)]
    sk = float(_skew(z_f)) if len(z_f) > 10 else 0.0
    ku = float(_kurtosis(z_f)) if len(z_f) > 10 else 0.0
    return {
        'transition_count': n_transitions, 'transition_rate_hz': round(transition_rate, 4),
        'mean_dwell_on_s': round(float(on_dwells.mean()), 2) if len(on_dwells) > 0 else 0.0,
        'mean_dwell_off_s': round(float(off_dwells.mean()), 2) if len(off_dwells) > 0 else 0.0,
        'n_on_epochs': len(on_dwells), 'n_off_epochs': len(off_dwells),
        'coupling_fraction': round(float(mask.mean()), 4),
        'shannon_entropy': round(shannon_entropy, 4),
        'skewness': round(sk, 4), 'kurtosis': round(ku, 4),
        'mean_z': round(float(z.mean()), 4), 'std_z': round(float(z.std()), 4),
    }


def cross_modal_lag_analysis(masks, keys, names, fs=FS_OUT, max_lag_s=MAX_LAG_S):
    """Cross-correlate binary masks across modality pairs at multiple lags."""
    max_lag = int(max_lag_s * fs)
    lags_seconds = np.arange(-max_lag, max_lag + 1) / fs
    pairs = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1 = masks[keys[i]].astype(np.float64)
            m2 = masks[keys[j]].astype(np.float64)
            m1_z = m1 - m1.mean()
            m2_z = m2 - m2.mean()
            denom = max(np.sqrt(np.sum(m1_z**2) * np.sum(m2_z**2)), 1e-10)
            full_xcorr = np.correlate(m1_z, m2_z, 'full') / denom
            center = len(m1) - 1
            window = full_xcorr[center - max_lag: center + max_lag + 1]
            peak_idx = np.argmax(np.abs(window))
            pair_key = f'{keys[i]}_vs_{keys[j]}'
            pairs[pair_key] = {
                'peak_lag_s': round(float(lags_seconds[peak_idx]), 2),
                'peak_corr': round(float(window[peak_idx]), 4),
                'leader': names[i] if lags_seconds[peak_idx] > 0 else names[j],
                'xcorr_profile': window[::2].tolist(),
            }
    return {'lags_seconds_thinned': lags_seconds[::2].tolist(), 'pairs': pairs}


def _downsample_bl_z(z_30hz, times_30hz, t_common):
    """Downsample BL z-timecourse from 30 Hz to common grid via block averaging."""
    factor = int(FS_BL / FS_OUT)
    n_blocks = len(z_30hz) // factor
    z_2hz = z_30hz[:n_blocks * factor].reshape(n_blocks, factor).mean(axis=1)
    t_2hz = times_30hz[:n_blocks * factor].reshape(n_blocks, factor).mean(axis=1)
    return np.interp(t_common, t_2hz, z_2hz, left=0, right=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  SESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name, config, raw_dir=RAW_DIR):
    """Run full RSLDS scaffold pipeline for one session.

    Returns:
        results: dict with all metrics, or None if session cannot be processed.
        z_traces: dict of {key: (N,) z-timecourse} on common grid.
        z_masks: dict of {key: (N,) boolean mask} on common grid.
    """
    print(f"\n{'='*70}")
    print(f"  RSLDS Scaffold — {session_name}")
    print(f"{'='*70}")

    t_wall = time.time()
    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load session ───────────────────────────────────────────────

    print(f"\n[1/8] Loading {session_name}...", flush=True)

    xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  WARNING: No XDF found for {session_name}, skipping")
        return None, None, None
    xdf_path = xdf_files[0]

    session_data = load_xdf_session(xdf_path)
    markers = session_data['markers']
    p1_role = session_data['p1_role']
    p2_role = session_data['p2_role']
    print(f"  Roles: P1={p1_role}, P2={p2_role}")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name in n]
    if not cache_matches:
        print(f"  WARNING: No cache found for {session_name}, skipping")
        return None, None, None
    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    segments = []
    for seg_name in CONDITION_ORDER:
        t_start = markers.get(f'{seg_name}_start')
        t_end = markers.get(f'{seg_name}_stop')
        if t_start is not None and t_end is not None:
            segments.append((seg_name, t_start, t_end))

    if not segments:
        print(f"  WARNING: No segments found for {session_name}, skipping")
        return None, None, None

    session_start_lsl = min(t for _, t, _ in segments) - 30
    session_end_lsl = max(t for _, _, t in segments) + 30
    t_common = np.arange(session_start_lsl, session_end_lsl, 1.0 / FS_OUT)
    N_common = len(t_common)
    print(f"  Grid: {N_common} pts, {(session_end_lsl - session_start_lsl):.0f}s, "
          f"segs: {[s[0] for s in segments]}")

    z_traces = {}
    z_masks = {}

    # ── 2. EEG full-session coupling (surrogates across entire session) ──

    print(f"\n[2/9] EEG full-session coupling...", flush=True)
    t0 = time.time()

    z_traces['eeg'] = np.zeros(N_common, dtype=np.float32)
    for bk in EEG_BAND_KEYS:
        z_traces[bk] = np.zeros(N_common, dtype=np.float32)

    # Extract full-session EEG from cache, avg-reference + z-score
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
        eeg_r = eeg_coupling_timecourse(
            p1_eeg_all, p2_eeg_all, fs_eeg,
            smooth_samples=3, n_surrogates=100, seed=42)
        # Map to common grid (eeg_r['times'] is relative to session start at t=0)
        eeg_times_lsl = eeg_r['times'] + (p1_eeg_ts[0] + lsl_offset)
        z_traces['eeg'] = np.interp(t_common, eeg_times_lsl,
                                     eeg_r['combined']['z'],
                                     left=0, right=0).astype(np.float32)
        for bn in ['theta', 'alpha', 'beta']:
            bd = eeg_r['per_band'].get(bn, {})
            if bd.get('n_valid', 0) > 0:
                z_traces[f'eeg_{bn}'] = np.interp(
                    t_common, eeg_times_lsl, bd['z'],
                    left=0, right=0).astype(np.float32)
        mz = eeg_r['combined']['mean_z']
        cf = eeg_r['combined']['coupling_fraction']
        print(f"    Full-session: z={mz:+.2f}, cf={cf:.1%}", flush=True)
    else:
        print(f"    No EEG data in cache, skipping", flush=True)

    z_masks['eeg'] = z_traces['eeg'] > Z_THRESH
    for bk in EEG_BAND_KEYS:
        z_masks[bk] = z_traces[bk] > Z_THRESH

    print(f"  EEG total: {time.time() - t0:.1f}s, "
          f"session cf={z_masks['eeg'].mean():.1%}")

    # ── 3. BL wavelet full-session (surrogates across entire session) ───

    print(f"\n[3/9] BL wavelet full-session coherence...", flush=True)
    t0 = time.time()

    landmarks = session_data['landmarks']
    smile_idx = [AFFECT_AUS.index(au) for au in SMILE_AUS]

    # Start respiratory extraction in background (overlaps with BL)
    has_raw_ecg = ('p1_ecg' in cached and 'p2_ecg' in cached
                   and len(cached.get('p1_ecg', [])) > 1000)
    resp_future = None
    if has_raw_ecg:
        _resp_pool = ThreadPoolExecutor(max_workers=1)
        def _extract_resp():
            r1 = extract_respiratory_one(cached['p1_ecg'], cached['p1_ecg_ts'], srate=130)
            r2 = extract_respiratory_one(cached['p2_ecg'], cached['p2_ecg_ts'], srate=130)
            return r1, r2
        resp_future = _resp_pool.submit(_extract_resp)

    # Initialize z-traces
    for bk in ['bl_expr', 'bl_state'] + BL_SUBGROUP_KEYS:
        z_traces[bk] = np.zeros(N_common, dtype=np.float32)

    # Extract full-session BL (entire session span)
    p1_bl, p2_bl, bl_dur = extract_bl_segment(
        landmarks, session_start_lsl, session_end_lsl)
    if p1_bl is not None and bl_dur > 5:
        T_bl = p1_bl.shape[0]
        bl_times = np.linspace(session_start_lsl, session_end_lsl, T_bl)

        # CWT on full-session affect + speech AUs
        scal_p1 = compute_au_cwt(p1_bl[:, AFFECT_AUS])
        scal_p2 = compute_au_cwt(p2_bl[:, AFFECT_AUS])
        freqs = scal_p1.freqs
        expr_fmask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
        state_fmask = (freqs >= BAND_STATE[0]) & (freqs < BAND_STATE[1])

        scal_p1_sp = compute_au_cwt(p1_bl[:, SPEECH_AUS])
        scal_p2_sp = compute_au_cwt(p2_bl[:, SPEECH_AUS])

        # Run surrogates (threaded for GPU overlap) — full session
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(surrogate_coherence_z, scal_p1, scal_p2,
                                200, 0.5, list(range(len(AFFECT_AUS))), 42, 'auto')
            fut_sm = pool.submit(surrogate_coherence_z, scal_p1, scal_p2,
                                 200, 0.5, smile_idx, 42, 'auto')
            fut_sp = pool.submit(surrogate_coherence_z, scal_p1_sp, scal_p2_sp,
                                 200, 0.5, list(range(len(SPEECH_AUS))), 42, 'auto')
            z_aff = fut_a.result()
            z_sm = fut_sm.result()
            z_sp = fut_sp.result()

        # Map full-session z to common grid
        def _bl_z_to_common(z_map, fmask):
            z_30 = z_map[fmask].mean(axis=0) if fmask.sum() > 0 else np.zeros(T_bl)
            return np.interp(t_common, bl_times, z_30,
                             left=0, right=0).astype(np.float32)

        z_traces['bl_expr'] = _bl_z_to_common(z_aff['z'], expr_fmask)
        z_traces['bl_state'] = _bl_z_to_common(z_aff['z'], state_fmask)
        z_traces['bl_expr_smile'] = _bl_z_to_common(z_sm['z'], expr_fmask)
        z_traces['bl_expr_speech'] = _bl_z_to_common(z_sp['z'], expr_fmask)

        print(f"    Full-session BL: expr_mean_z={z_traces['bl_expr'].mean():+.2f}", flush=True)
    else:
        print(f"    No BL data available, skipping", flush=True)

    for bk in ['bl_expr', 'bl_state'] + BL_SUBGROUP_KEYS:
        z_masks[bk] = z_traces[bk] > Z_THRESH

    print(f"  BL total: {time.time() - t0:.1f}s, "
          f"expr_cf={z_masks['bl_expr'].mean():.1%}")

    # ── 4. ECG/Resp/Pose full-session coupling ─────────────────────────
    #
    # All cross-product modalities use full-session surrogates:
    # interpolate to common grid, run cross_product_z once per modality.

    print(f"\n[4/9] ECG + Resp + Pose full-session coupling...", flush=True)
    t0 = time.time()

    has_ecg = ('p1_ecg_features' in cached and 'p2_ecg_features' in cached
               and 'p1_ecg_features_ts' in cached and 'p2_ecg_features_ts' in cached)

    # Collect resp from background thread
    if resp_future is not None:
        p1_resp, p2_resp = resp_future.result()
        _resp_pool.shutdown(wait=False)
    else:
        p1_resp = p2_resp = None

    has_pose = ('p1_pose_features' in cached and 'p2_pose_features' in cached
                and 'p1_pose_features_ts' in cached and 'p2_pose_features_ts' in cached)

    # Initialize all cross-product traces
    for ck in ['ecg_sns', 'ecg_pns', 'resp', 'pose']:
        z_traces[ck] = np.zeros(N_common, dtype=np.float32)

    # ECG full-session
    if has_ecg:
        p1_ecg = cached['p1_ecg_features']
        p2_ecg = cached['p2_ecg_features']
        p1_ts = cached['p1_ecg_features_ts'] + lsl_offset
        p2_ts = cached['p2_ecg_features_ts'] + lsl_offset
        n_ch = min(p1_ecg.shape[1], p2_ecg.shape[1])

        def _interp_full(p_ecg, p_ts, ch):
            return np.interp(t_common, p_ts, p_ecg[:, ch], left=0, right=0)

        sns_chs = [ch for ch in ECG_SNS_CHS if ch < n_ch]
        if sns_chs:
            p1_s = np.column_stack([_interp_full(p1_ecg, p1_ts, ch) for ch in sns_chs])
            p2_s = np.column_stack([_interp_full(p2_ecg, p2_ts, ch) for ch in sns_chs])
            r = cross_product_z(p1_s, p2_s, smooth_samples=ECG_SMOOTH,
                                n_surrogates=200, seed=None)
            z_traces['ecg_sns'] = r['z']
        if ECG_PNS_CH < n_ch:
            p1_p = _interp_full(p1_ecg, p1_ts, ECG_PNS_CH)
            p2_p = _interp_full(p2_ecg, p2_ts, ECG_PNS_CH)
            r = cross_product_z(p1_p, p2_p, smooth_samples=ECG_SMOOTH,
                                n_surrogates=200, seed=None)
            z_traces['ecg_pns'] = r['z']

    # Resp full-session
    if p1_resp is not None and p2_resp is not None:
        r1 = np.interp(t_common, p1_resp['t'] + lsl_offset, p1_resp['rate_bpm'],
                       left=np.nan, right=np.nan)
        r2 = np.interp(t_common, p2_resp['t'] + lsl_offset, p2_resp['rate_bpm'],
                       left=np.nan, right=np.nan)
        v = np.isfinite(r1) & np.isfinite(r2)
        r1[~v] = 0.0; r2[~v] = 0.0
        if v.sum() > 10:
            r = cross_product_z(r1, r2, smooth_samples=ECG_SMOOTH,
                                n_surrogates=200, seed=None)
            z_traces['resp'] = r['z']

    # Pose full-session
    if has_pose:
        p1_pose = cached['p1_pose_features']
        p2_pose = cached['p2_pose_features']
        n_pch = min(p1_pose.shape[1], p2_pose.shape[1], 40)
        t1p = cached['p1_pose_features_ts'] + lsl_offset
        t2p = cached['p2_pose_features_ts'] + lsl_offset
        p1_ps = np.column_stack([np.interp(t_common, t1p, p1_pose[:, c], left=0, right=0)
                                 for c in range(n_pch)])
        p2_ps = np.column_stack([np.interp(t_common, t2p, p2_pose[:, c], left=0, right=0)
                                 for c in range(n_pch)])
        r = cross_product_z(p1_ps, p2_ps, smooth_samples=ECG_SMOOTH,
                            n_surrogates=200, seed=None)
        z_traces['pose'] = r['z']

    for ck in ['ecg_sns', 'ecg_pns', 'resp', 'pose']:
        z_masks[ck] = z_traces[ck] > Z_THRESH

    print(f"  ECG SNS cf={z_masks['ecg_sns'].mean():.1%}, "
          f"PNS cf={z_masks['ecg_pns'].mean():.1%}, "
          f"Resp cf={z_masks['resp'].mean():.1%}, "
          f"Pose cf={z_masks['pose'].mean():.1%} ({time.time() - t0:.1f}s)")

    # ── 6. Flexibility metrics (all traces) ───────────────────────────

    print(f"\n[6/9] Flexibility metrics...", flush=True)

    flex_results = {}
    for key in ALL_TRACE_KEYS:
        if key in z_traces:
            flex_results[key] = flexibility_metrics(z_traces[key], z_masks[key])

    for key, name in zip(MODALITY_KEYS, MODALITY_NAMES):
        fm = flex_results[key]
        print(f"  {name:25s}: cf={fm['coupling_fraction']:.1%}, "
              f"trans={fm['transition_count']}, dwell={fm['mean_dwell_on_s']:.1f}s")

    # Per-condition breakdown
    condition_flex = {}
    for seg_name, t0_lsl, t1_lsl in segments:
        seg_mask = (t_common >= t0_lsl) & (t_common <= t1_lsl)
        if seg_mask.sum() < 10:
            continue
        condition_flex[seg_name] = {}
        for key in ALL_TRACE_KEYS:
            if key in z_traces:
                condition_flex[seg_name][key] = flexibility_metrics(
                    z_traces[key][seg_mask], z_masks[key][seg_mask])

    # ── 7. Cross-modal lag analysis ───────────────────────────────────

    print(f"\n[7/9] Cross-modal lag analysis...", flush=True)
    lag_results = cross_modal_lag_analysis(z_masks, MODALITY_KEYS, MODALITY_NAMES)

    for pair_key, pr in sorted(lag_results['pairs'].items(),
                                key=lambda x: abs(x[1]['peak_corr']), reverse=True)[:5]:
        print(f"  {pair_key:40s} lag={pr['peak_lag_s']:+.1f}s r={pr['peak_corr']:.3f}")

    # ── 7b. Spectral decomposition: z_slow / z_fast + slow PCA ────────

    print(f"\n[7b/9] Spectral decomposition (z_slow/z_fast)...", flush=True)
    from scipy.signal import butter as _butter, sosfiltfilt as _sosfiltfilt
    from sklearn.decomposition import PCA

    SLOW_CUTOFF_HZ = 0.01   # 100s cutoff — separates shared state from coupling
    z_fast = {}
    z_slow = {}

    # Design low-pass filter once (4th order Butterworth, 0.01 Hz at 2 Hz fs)
    _sos_slow = _butter(4, SLOW_CUTOFF_HZ, btype='low', fs=FS_OUT, output='sos')

    for key in MODALITY_KEYS:
        z_orig = z_traces[key]
        if np.abs(z_orig).max() < 1e-8:
            # All zeros (missing modality) — keep both as zeros
            z_slow[key] = np.zeros_like(z_orig)
            z_fast[key] = np.zeros_like(z_orig)
            continue
        z_slow[key] = _sosfiltfilt(_sos_slow, z_orig).astype(np.float32)
        z_fast[key] = (z_orig - z_slow[key]).astype(np.float32)

    # PCA on the 7 z_slow channels → 2D shared-state summary
    z_slow_matrix = np.column_stack([z_slow[k] for k in MODALITY_KEYS])
    # Only include modalities with non-zero signal for PCA
    active_cols = [i for i, k in enumerate(MODALITY_KEYS) if np.abs(z_slow[k]).max() > 1e-8]
    if len(active_cols) >= 2:
        z_slow_active = z_slow_matrix[:, active_cols]
        pca = PCA(n_components=min(2, len(active_cols)))
        z_slow_pcs = pca.fit_transform(z_slow_active).astype(np.float32)
        pca_explained = pca.explained_variance_ratio_
        pca_loadings = pca.components_  # (n_components, n_active_modalities)
        # Build full loading matrix (n_components, 7) with zeros for inactive
        pca_loadings_full = np.zeros((pca.n_components_, len(MODALITY_KEYS)), dtype=np.float32)
        for idx, col in enumerate(active_cols):
            pca_loadings_full[:, col] = pca_loadings[:, idx]
    else:
        z_slow_pcs = np.zeros((N_common, 2), dtype=np.float32)
        pca_explained = np.array([0.0, 0.0])
        pca_loadings_full = np.zeros((2, len(MODALITY_KEYS)), dtype=np.float32)

    # Variance decomposition
    total_var = {k: float(np.var(z_traces[k])) for k in MODALITY_KEYS}
    slow_var = {k: float(np.var(z_slow[k])) for k in MODALITY_KEYS}
    fast_var = {k: float(np.var(z_fast[k])) for k in MODALITY_KEYS}
    slow_frac = {k: slow_var[k] / max(total_var[k], 1e-10) for k in MODALITY_KEYS}

    spectral_decomp = {
        'slow_cutoff_hz': SLOW_CUTOFF_HZ,
        'variance_total': total_var,
        'variance_slow': slow_var,
        'variance_fast': fast_var,
        'slow_fraction': slow_frac,
        'pca_explained_variance_ratio': pca_explained.tolist(),
        'pca_loadings': pca_loadings_full.tolist(),
        'pca_modality_keys': MODALITY_KEYS,
    }

    print(f"  Slow (<{SLOW_CUTOFF_HZ} Hz) variance fractions:")
    for key, name in zip(MODALITY_KEYS, MODALITY_NAMES):
        print(f"    {name:25s}: {slow_frac[key]:5.1%} slow, {1-slow_frac[key]:5.1%} fast")
    print(f"  Slow PCA: {pca_explained[0]:.1%} + {pca_explained[1] if len(pca_explained) > 1 else 0:.1%} = "
          f"{sum(pca_explained):.1%} explained")

    # ── 8. Visualization ──────────────────────────────────────────────

    print(f"\n[8/9] Visualization...", flush=True)

    n_panels = len(MODALITY_KEYS) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(26, 3.2 * n_panels),
                             gridspec_kw={'height_ratios': [1]*len(MODALITY_KEYS) + [0.8]},
                             sharex=True)

    for ax in axes:
        for seg_name, t0s, t1s in segments:
            ax.axvspan(t0s, t1s, alpha=0.3,
                       color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)
            ax.axvline(t0s, color='gray', linewidth=0.3, alpha=0.3)

    for seg_name, t0s, t1s in segments:
        axes[0].text((t0s + t1s) / 2, 1.08, seg_name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        z = z_traces[key]
        mask = z_masks[key]

        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
        ax.axhline(Z_THRESH, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)

        # Per-band EEG overlay (thin, behind combined)
        if key == 'eeg':
            for bk, bc in EEG_BAND_COLORS.items():
                if bk in z_traces:
                    ax.plot(t_common, z_traces[bk], color=bc, linewidth=0.4, alpha=0.5, zorder=1)

        # Main z-trace
        ax.plot(t_common, z, color=color, linewidth=0.7, alpha=0.8, zorder=2)

        # Significance shading
        sig_regions = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(sig_regions == 1)[0]
        ends = np.where(sig_regions == -1)[0]
        for s, e in zip(starts, ends):
            if s < N_common and e <= N_common:
                ax.axvspan(t_common[max(0, s)], t_common[min(e - 1, N_common - 1)],
                           alpha=0.2, color=color, zorder=1)

        # Annotation
        fm = flex_results[key]
        ann = f"cf={fm['coupling_fraction']:.1%}  trans={fm['transition_count']}  H={fm['shannon_entropy']:.2f}"
        if key == 'eeg':
            for bn in ['theta', 'alpha', 'beta']:
                bfm = flex_results.get(f'eeg_{bn}', {})
                ann += f"  {bn[0]}={bfm.get('coupling_fraction', 0):.1%}"
        if key == 'bl_expr':
            sm_cf = flex_results.get('bl_expr_smile', {}).get('coupling_fraction', 0)
            sp_cf = flex_results.get('bl_expr_speech', {}).get('coupling_fraction', 0)
            ann += f"  smile={sm_cf:.1%}  speech={sp_cf:.1%}"
        ax.text(0.01, 0.92, ann, transform=ax.transAxes, fontsize=7, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_ylabel(f'{name}\nz-score', fontsize=9)
        ax.set_ylim(-6, max(8, float(np.nanpercentile(z, 99.5)) + 1))

    # Coupling state strip
    ax_strip = axes[-1]
    n_mod = len(MODALITY_KEYS)
    strip_rgba = np.ones((n_mod, N_common, 4)) * 0.95
    for row, (key, color) in enumerate(zip(MODALITY_KEYS, MODALITY_COLORS)):
        strip_rgba[n_mod - 1 - row, z_masks[key]] = to_rgba(color, alpha=0.85)
    ax_strip.imshow(strip_rgba, aspect='auto', interpolation='nearest',
                    extent=[t_common[0], t_common[-1], 0, n_mod], origin='lower')
    ax_strip.set_yticks(np.arange(n_mod) + 0.5)
    ax_strip.set_yticklabels(list(reversed(MODALITY_NAMES)), fontsize=7)
    ax_strip.set_ylim(0, n_mod)
    ax_strip.set_ylabel('Coupling\nstate', fontsize=9)
    ax_strip.set_xlabel('LSL time (s)')
    ax_strip.set_xlim(session_start_lsl, session_end_lsl)

    fig.suptitle(f'{session_name} RSLDS Scaffold ({p1_role} vs {p2_role})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'rslds_scaffold_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Save results ──────────────────────────────────────────────────

    results = {
        'session': session_name, 'p1_role': p1_role, 'p2_role': p2_role,
        'common_grid': {'fs_hz': FS_OUT, 'n_samples': N_common,
                        'start_lsl': float(t_common[0]), 'end_lsl': float(t_common[-1])},
        'ecg_config': {'sns_channels': ECG_SNS_CHS, 'pns_channel': ECG_PNS_CH,
                       'smooth_samples': ECG_SMOOTH},
        'modalities': {key: {'name': name, 'flexibility': flex_results[key]}
                       for key, name in zip(MODALITY_KEYS, MODALITY_NAMES)},
        'sub_modalities': {key: {'flexibility': flex_results[key]}
                           for key in EEG_BAND_KEYS + BL_SUBGROUP_KEYS if key in flex_results},
        'condition_flexibility': {seg: {k: v for k, v in cf.items()}
                                  for seg, cf in condition_flex.items()},
        'cross_modal_lags': lag_results,
    }

    np.savez_compressed(os.path.join(out_dir, 'rslds_scaffold_ztimecourses.npz'),
        t_common=t_common,
        **{f'z_{k}': z_traces[k] for k in z_traces},
        **{f'mask_{k}': z_masks[k] for k in z_masks},
        # Spectral decomposition: z_fast (coupling) + z_slow PCs (shared state)
        **{f'z_fast_{k}': z_fast[k] for k in MODALITY_KEYS},
        **{f'z_slow_{k}': z_slow[k] for k in MODALITY_KEYS},
        z_slow_pc1=z_slow_pcs[:, 0],
        z_slow_pc2=z_slow_pcs[:, 1] if z_slow_pcs.shape[1] > 1 else np.zeros(N_common, dtype=np.float32),
        pca_loadings=pca_loadings_full)

    results['spectral_decomposition'] = spectral_decomp

    with open(os.path.join(out_dir, 'rslds_scaffold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────

    total = time.time() - t_wall
    print(f"\n  {session_name} complete in {total:.0f}s — {out_dir}/")

    # Per-condition per-band EEG table
    print(f"\n  {'Condition':>12s} | {'combined':>8s} {'theta':>7s} {'alpha':>7s} {'beta':>7s} "
          f"| {'bl_expr':>8s} {'smile':>7s} {'speech':>7s}")
    print("  " + "-" * 80)
    for seg_name, _, _ in segments:
        cf_eeg = condition_flex.get(seg_name, {}).get('eeg', {}).get('coupling_fraction', 0)
        cf_t = condition_flex.get(seg_name, {}).get('eeg_theta', {}).get('coupling_fraction', 0)
        cf_a = condition_flex.get(seg_name, {}).get('eeg_alpha', {}).get('coupling_fraction', 0)
        cf_b = condition_flex.get(seg_name, {}).get('eeg_beta', {}).get('coupling_fraction', 0)
        cf_bl = condition_flex.get(seg_name, {}).get('bl_expr', {}).get('coupling_fraction', 0)
        cf_sm = condition_flex.get(seg_name, {}).get('bl_expr_smile', {}).get('coupling_fraction', 0)
        cf_sp = condition_flex.get(seg_name, {}).get('bl_expr_speech', {}).get('coupling_fraction', 0)
        print(f"  {seg_name:>12s} | {cf_eeg:7.1%} {cf_t:7.1%} {cf_a:7.1%} {cf_b:7.1%} "
              f"| {cf_bl:7.1%} {cf_sm:7.1%} {cf_sp:7.1%}")

    return results, z_traces, z_masks


# ══════════════════════════════════════════════════════════════════════
#  MULTI-SESSION RUNNER + EMPIRICAL FPR
# ══════════════════════════════════════════════════════════════════════

def _compute_pseudodyad_coupling(p1_data, p2_data, N_out, device='auto'):
    """Compute coupling for one pseudo-dyad pair. Returns (cfs, z_ts).

    cfs: {modality_key: coupling_fraction} session-wide.
    z_ts: {modality_key: (z_array, fs)} for per-condition slicing.
    """
    from copy import copy
    cfs = {}
    z_ts = {}

    # EEG: use pre-computed volt_amp if available, else full pipeline
    p1_va = p1_data.get('eeg_precomp')
    p2_va = p2_data.get('eeg_precomp')
    eeg_r = None

    if p1_va is not None and p2_va is not None:
        eeg_r = eeg_coupling_from_precomputed(
            p1_va, p2_va, smooth_samples=3, n_surrogates=100, seed=None,
            device=torch.device(device if device != 'auto'
                                else ('cuda' if torch.cuda.is_available() else 'cpu')))
    elif p1_data.get('eeg') is not None and p2_data.get('eeg') is not None:
        p1_eeg = p1_data['eeg']; p2_eeg = p2_data['eeg']
        mlen = min(len(p1_eeg), len(p2_eeg))
        e1 = p1_eeg[:mlen].copy(); e2 = p2_eeg[:mlen].copy()
        e1 -= e1.mean(axis=1, keepdims=True); e2 -= e2.mean(axis=1, keepdims=True)
        for ch in range(e1.shape[1]):
            for arr in [e1, e2]:
                sd = arr[:, ch].std()
                if sd > 1e-8: arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd
        eeg_r = eeg_coupling_timecourse(e1, e2, p1_data['eeg_fs'], smooth_samples=3,
                                         n_surrogates=100, seed=None)

    if eeg_r is not None:
        cfs['eeg'] = eeg_r['combined']['coupling_fraction']
        z_ts['eeg'] = (eeg_r['combined']['z'], FS_OUT)
        for bn in ['theta', 'alpha', 'beta']:
            bd = eeg_r['per_band'].get(bn, {})
            bz = np.array(bd.get('z', []))
            cfs[f'eeg_{bn}'] = float((bz > Z_THRESH).mean()) if bd.get('n_valid', 0) > 0 else 0.0
            if len(bz) > 0:
                z_ts[f'eeg_{bn}'] = (bz, FS_OUT)

    # BL: trim scalograms to min T, run surrogate coherence
    s1 = p1_data.get('bl_scal')
    s2 = p2_data.get('bl_scal')
    if s1 is not None and s2 is not None:
        T_min = min(s1.coeffs.shape[1], s2.coeffs.shape[1])
        s1t = copy(s1); s2t = copy(s2)
        s1t.coeffs = s1.coeffs[:, :T_min, :]; s2t.coeffs = s2.coeffs[:, :T_min, :]
        z_bl = surrogate_coherence_z(s1t, s2t, n_surrogates=200,
                                      aus=list(range(s1t.coeffs.shape[2])), device=device)
        freqs = s1.freqs
        expr_m = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
        state_m = (freqs >= BAND_STATE[0]) & (freqs < BAND_STATE[1])
        if expr_m.sum() > 0:
            bl_expr_z = z_bl['z'][expr_m].mean(axis=0)
            cfs['bl_expr'] = float((bl_expr_z > Z_THRESH).mean())
            z_ts['bl_expr'] = (bl_expr_z, FS_BL)
        if state_m.sum() > 0:
            bl_state_z = z_bl['z'][state_m].mean(axis=0)
            cfs['bl_state'] = float((bl_state_z > Z_THRESH).mean())
            z_ts['bl_state'] = (bl_state_z, FS_BL)

        # BL smile sub-group (reuses affect scalograms)
        smile_idx = [AFFECT_AUS.index(au) for au in SMILE_AUS]
        z_sm = surrogate_coherence_z(s1t, s2t, n_surrogates=200, aus=smile_idx, device=device)
        if expr_m.sum() > 0:
            sm_z = z_sm['z'][expr_m].mean(axis=0)
            cfs['bl_expr_smile'] = float((sm_z > Z_THRESH).mean())
            z_ts['bl_expr_smile'] = (sm_z, FS_BL)

    # BL speech sub-group
    s1sp = p1_data.get('bl_scal_speech')
    s2sp = p2_data.get('bl_scal_speech')
    if s1sp is not None and s2sp is not None:
        T_min = min(s1sp.coeffs.shape[1], s2sp.coeffs.shape[1])
        s1t = copy(s1sp); s2t = copy(s2sp)
        s1t.coeffs = s1sp.coeffs[:, :T_min, :]; s2t.coeffs = s2sp.coeffs[:, :T_min, :]
        z_sp = surrogate_coherence_z(s1t, s2t, n_surrogates=200,
                                      aus=list(range(s1t.coeffs.shape[2])), device=device)
        freqs = s1sp.freqs
        expr_m = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
        if expr_m.sum() > 0:
            sp_z = z_sp['z'][expr_m].mean(axis=0)
            cfs['bl_expr_speech'] = float((sp_z > Z_THRESH).mean())
            z_ts['bl_expr_speech'] = (sp_z, FS_BL)

    # ECG SNS (multi-channel)
    p1_ecg = p1_data.get('ecg')
    p2_ecg = p2_data.get('ecg')
    if p1_ecg is not None and p2_ecg is not None:
        mlen = min(len(p1_ecg), len(p2_ecg))
        n_ch = min(p1_ecg.shape[1], p2_ecg.shape[1])
        sns_chs = [ch for ch in ECG_SNS_CHS if ch < n_ch]
        if sns_chs:
            r = cross_product_z(p1_ecg[:mlen, sns_chs], p2_ecg[:mlen, sns_chs],
                                smooth_samples=ECG_SMOOTH, n_surrogates=200, seed=None, device=device)
            cfs['ecg_sns'] = r['coupling_fraction']
            z_ts['ecg_sns'] = (r['z'], FS_OUT)
        if ECG_PNS_CH < n_ch:
            r = cross_product_z(p1_ecg[:mlen, ECG_PNS_CH], p2_ecg[:mlen, ECG_PNS_CH],
                                smooth_samples=ECG_SMOOTH, n_surrogates=200, seed=None, device=device)
            cfs['ecg_pns'] = r['coupling_fraction']
            z_ts['ecg_pns'] = (r['z'], FS_OUT)

    # Pose
    p1_pose = p1_data.get('pose')
    p2_pose = p2_data.get('pose')
    if p1_pose is not None and p2_pose is not None:
        mlen = min(len(p1_pose), len(p2_pose))
        n_ch = min(p1_pose.shape[1], p2_pose.shape[1], 40)
        r = cross_product_z(p1_pose[:mlen, :n_ch], p2_pose[:mlen, :n_ch],
                            smooth_samples=ECG_SMOOTH, n_surrogates=200, seed=None, device=device)
        cfs['pose'] = r['coupling_fraction']
        z_ts['pose'] = (r['z'], FS_OUT)

    # Respiratory rate
    p1_rr = p1_data.get('resp_rate')
    p2_rr = p2_data.get('resp_rate')
    if p1_rr is not None and p2_rr is not None:
        mlen = min(len(p1_rr), len(p2_rr))
        r = cross_product_z(p1_rr[:mlen], p2_rr[:mlen],
                            smooth_samples=ECG_SMOOTH, n_surrogates=200, seed=None, device=device)
        cfs['resp'] = r['coupling_fraction']
        z_ts['resp'] = (r['z'], 4.0)

    return cfs, z_ts


def run_pseudodyad_fpr(config, raw_dir=RAW_DIR):
    """Compute empirical FPR from pseudo-dyad pairs (P1 from A, P2 from B)."""

    print(f"\n{'='*70}")
    print("PSEUDO-DYAD FPR CALIBRATION")
    print(f"{'='*70}")

    # Phase A: Pre-load all sessions
    print("\nPhase A: Pre-loading sessions...", flush=True)

    cached_sessions = discover_cached_sessions(config['session_cache'])
    session_names = [name for name, _ in cached_sessions]

    preloaded = {}  # {name: {p1: {...}, p2: {...}}}

    for sname in session_names:
        print(f"  Loading {sname}...", end='', flush=True)
        t0 = time.time()

        # Cache data
        cache_path = [p for n, p in cached_sessions if n == sname][0]
        cached = load_session_from_cache(cache_path, config)

        # XDF for BL
        xdf_files = glob.glob(os.path.join(raw_dir, f'{sname}*.xdf'))
        bl_scal_affect = bl_scal_smile = bl_scal_speech = None
        if xdf_files:
            session_data = load_xdf_session(xdf_files[0])
            landmarks = session_data['landmarks']
            if 'P1' in landmarks and 'P2' in landmarks:
                p1_bl_ts = landmarks['P1'][0]
                p2_bl_ts = landmarks['P2'][0]
                bl_start = max(p1_bl_ts.min(), p2_bl_ts.min())
                bl_end = min(p1_bl_ts.max(), p2_bl_ts.max())
                p1_bl, p2_bl, _ = extract_bl_segment(landmarks, bl_start, bl_end)
                if p1_bl is not None:
                    # CWT on each AU group — store per-participant scalograms
                    bl_scal_affect = (compute_au_cwt(p1_bl[:, AFFECT_AUS]),
                                      compute_au_cwt(p2_bl[:, AFFECT_AUS]))
                    smile_idx_in_affect = [AFFECT_AUS.index(au) for au in SMILE_AUS]
                    bl_scal_smile = (bl_scal_affect[0], bl_scal_affect[1])  # reuse, different aus param
                    bl_scal_speech = (compute_au_cwt(p1_bl[:, SPEECH_AUS]),
                                      compute_au_cwt(p2_bl[:, SPEECH_AUS]))

        # Store P1 and P2 data separately
        p1 = {}; p2 = {}

        # Pre-compute EEG volt_amp (avoids redundant cycle extraction across pairs)
        for person, pdata in [('p1', p1), ('p2', p2)]:
            eeg_key = f'{person}_eeg'
            ts_key = f'{person}_eeg_ts'
            if eeg_key in cached and ts_key in cached:
                eeg_raw = cached[eeg_key][:, :14].astype(np.float64)
                eeg_raw -= eeg_raw.mean(axis=1, keepdims=True)  # avg-ref
                for ch in range(eeg_raw.shape[1]):
                    mu, sd = eeg_raw[:, ch].mean(), eeg_raw[:, ch].std()
                    if sd > 1e-8: eeg_raw[:, ch] = (eeg_raw[:, ch] - mu) / sd
                fs_eeg = len(cached[ts_key]) / max(cached[ts_key][-1] - cached[ts_key][0], 1)
                pdata['eeg_precomp'] = extract_all_volt_amp(eeg_raw, fs_eeg)
                pdata['eeg_fs'] = fs_eeg
            else:
                pdata['eeg_precomp'] = None; pdata['eeg_fs'] = 256.0

        if bl_scal_affect:
            p1['bl_scal'] = bl_scal_affect[0]
            p2['bl_scal'] = bl_scal_affect[1]
            # For smile: same scalograms but will pass different aus param
            # Store the smile-subset indices for surrogate call
            p1['bl_scal_smile'] = bl_scal_affect[0]
            p2['bl_scal_smile'] = bl_scal_affect[1]
            if bl_scal_speech:
                p1['bl_scal_speech'] = bl_scal_speech[0]
                p2['bl_scal_speech'] = bl_scal_speech[1]

        p1['ecg'] = cached.get('p1_ecg_features')
        p2['ecg'] = cached.get('p2_ecg_features')
        p1['pose'] = cached.get('p1_pose_features')
        p2['pose'] = cached.get('p2_pose_features')

        # Respiratory rate from raw ECG
        if 'p1_ecg' in cached and len(cached.get('p1_ecg', [])) > 1000:
            r1 = extract_respiratory_one(cached['p1_ecg'], cached['p1_ecg_ts'], srate=130)
            r2 = extract_respiratory_one(cached['p2_ecg'], cached['p2_ecg_ts'], srate=130)
            p1['resp_rate'] = r1['rate_bpm'] if r1 is not None else None
            p1['resp_t'] = r1['t'] if r1 is not None else None
            p2['resp_rate'] = r2['rate_bpm'] if r2 is not None else None
            p2['resp_t'] = r2['t'] if r2 is not None else None
        else:
            p1['resp_rate'] = p1['resp_t'] = None
            p2['resp_rate'] = p2['resp_t'] = None

        conditions = parse_conditions(cache_path)
        preloaded[sname] = {'p1': p1, 'p2': p2, 'conditions': conditions}
        print(f" {time.time() - t0:.1f}s ({len(conditions)} conds)", flush=True)

    # Phase B: Run all pseudo-dyad pairs + per-condition slicing
    names = list(preloaded.keys())
    n_sessions = len(names)
    pairs = [(a, b) for a in names for b in names if a != b]
    print(f"\nPhase B: {len(pairs)} pseudo-dyad pairs (threaded, shared GPU)...", flush=True)
    t_phase_b = time.time()

    def _run_pair(sa, sb):
        p1_data = preloaded[sa]['p1']
        p2_data = preloaded[sb]['p2']
        cfs, z_ts = _compute_pseudodyad_coupling(p1_data, p2_data, 0, device='auto')
        for cname, t0_s, t1_s in preloaded[sa]['conditions']:
            for mod_key, (z_arr, fs) in z_ts.items():
                i0 = max(0, int(t0_s * fs))
                i1 = min(len(z_arr), int(t1_s * fs))
                if i1 > i0 + 5:
                    cfs[f'{mod_key}@{cname}'] = float((z_arr[i0:i1] > Z_THRESH).mean())
        return {'session_a': sa, 'session_b': sb, **cfs}

    with ThreadPoolExecutor(max_workers=16) as pool:
        pair_results = list(pool.map(lambda args: _run_pair(*args), pairs))
    print(f"  {len(pairs)} pairs in {time.time() - t_phase_b:.1f}s", flush=True)

    # Aggregate FPR per modality
    print(f"\n{'='*70}")
    print("PSEUDO-DYAD FPR RESULTS")
    print(f"{'='*70}")

    fpr_keys = MODALITY_KEYS + EEG_BAND_KEYS + BL_SUBGROUP_KEYS
    print(f"\n  {'Modality':25s} | {'n':>4s} {'mean':>7s} {'std':>7s} {'med':>7s} {'p95':>7s} {'max':>7s}")
    print("  " + "-" * 65)

    fpr_summary = {}
    for key in fpr_keys:
        vals = np.array([pr.get(key, np.nan) for pr in pair_results])
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            fpr_summary[key] = {
                'n_pairs': int(len(vals)),
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'median': round(float(np.median(vals)), 4),
                'p95': round(float(np.percentile(vals, 95)), 4),
                'max': round(float(vals.max()), 4),
            }
            print(f"  {key:25s} | {len(vals):4d} {vals.mean():7.1%} {vals.std():7.1%} "
                  f"{np.median(vals):7.1%} {np.percentile(vals, 95):7.1%} {vals.max():7.1%}")

    # Per-condition FPR — the critical diagnostic for RSLDS
    cond_names = sorted(set(c for s in preloaded.values() for c, _, _ in s['conditions']))
    primary_mods = ['eeg', 'eeg_theta', 'eeg_alpha', 'eeg_beta',
                    'bl_expr', 'ecg_sns', 'resp', 'pose']

    print(f"\n{'='*70}")
    print("PER-CONDITION PSEUDO-DYAD FPR (p95 thresholds)")
    print(f"{'='*70}")
    print(f"\n  {'Condition':>14s} |", end='')
    for mod in primary_mods:
        print(f" {mod:>9s}", end='')
    print()
    print("  " + "-" * (17 + 10 * len(primary_mods)))

    cond_fpr = {}
    for cname in cond_names:
        cond_fpr[cname] = {}
        print(f"  {cname:>14s} |", end='')
        for mod in primary_mods:
            key = f'{mod}@{cname}'
            vals = np.array([pr.get(key, np.nan) for pr in pair_results])
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3:
                p95 = float(np.percentile(vals, 95))
                cond_fpr[cname][mod] = {
                    'n': int(len(vals)), 'mean': round(float(vals.mean()), 4),
                    'p95': round(p95, 4),
                }
                print(f" {p95:8.1%}", end='')
            else:
                print(f" {'n/a':>9s}", end='')
        print()

    fpr_summary['per_condition'] = cond_fpr
    return fpr_summary, pair_results


def main():
    parser = argparse.ArgumentParser(description='RSLDS Scaffold — multi-modal coupling')
    parser.add_argument('--session', nargs='*', default=None,
                        help='Session names (e.g., y_06 y_17). Default: y_06')
    parser.add_argument('--all', action='store_true',
                        help='Process all cached sessions with matching XDFs')
    parser.add_argument('--fpr', action='store_true',
                        help='Run pseudo-dyad FPR calibration (requires --all)')
    args = parser.parse_args()

    config = load_config()

    if args.all:
        cached = discover_cached_sessions(config['session_cache'])
        session_names = [name for name, _ in cached]
    elif args.session:
        session_names = args.session
    else:
        session_names = ['y_06']

    print(f"Sessions to process: {session_names}")

    all_results = {}
    if len(session_names) > 1:
        # Parallel session processing: 3 concurrent processes
        # ProcessPool gives true CPU parallelism (no GIL)
        # GPU memory limits concurrency (~200 MB BL scalograms per session, ~8 GB total)
        from concurrent.futures import ProcessPoolExecutor as _SessionPool
        print(f"Running {len(session_names)} sessions (8 concurrent processes)...")
        with _SessionPool(max_workers=8) as pool:
            futures = {pool.submit(run_session, sname, config): sname
                       for sname in session_names}
            for fut in futures:
                sname = futures[fut]
                try:
                    res, z_traces, z_masks = fut.result()
                    if res is not None:
                        all_results[sname] = res
                except Exception as e:
                    print(f"  ERROR processing {sname}: {e}")
    else:
        for sname in session_names:
            res, z_traces, z_masks = run_session(sname, config)
            if res is not None:
                all_results[sname] = res

    # ── Pseudo-dyad FPR calibration ──────────────────────────────────

    fpr_summary = {}
    if args.fpr and len(all_results) >= 2:
        fpr_summary, pair_results = run_pseudodyad_fpr(config)

    if len(all_results) < 2:
        return

    # Save cross-session summary
    summary = {
        'sessions_processed': list(all_results.keys()),
        'pseudo_dyad_fpr': fpr_summary,
        'per_session': {sname: {key: res['modalities'][key]['flexibility']['coupling_fraction']
                                for key in MODALITY_KEYS}
                        for sname, res in all_results.items()},
    }

    os.makedirs('results/rslds', exist_ok=True)
    with open('results/rslds/cross_session_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cross-session comparison table (with pseudo-dyad p95 threshold row)
    print(f"\n  {'Session':>15s} | ", end='')
    for key in MODALITY_KEYS:
        print(f" {key:>8s}", end='')
    print()
    print("  " + "-" * 75)
    if fpr_summary:
        print(f"  {'pseudo p95':>15s} | ", end='')
        for key in MODALITY_KEYS:
            p95 = fpr_summary.get(key, {}).get('p95', 0)
            print(f" {p95:7.1%}", end='')
        print("  <-- null threshold")
    for sname, res in sorted(all_results.items()):
        print(f"  {sname:>15s} | ", end='')
        for key in MODALITY_KEYS:
            cf = res['modalities'][key]['flexibility']['coupling_fraction']
            # Mark with * if above pseudo-dyad p95
            above = ''
            if fpr_summary and cf > fpr_summary.get(key, {}).get('p95', 1.0):
                above = '*'
            print(f" {cf:6.1%}{above}", end='')
        print()

    print(f"\n  * = exceeds pseudo-dyad 95th percentile")
    print(f"  Saved results/rslds/cross_session_summary.json")


if __name__ == '__main__':
    main()
