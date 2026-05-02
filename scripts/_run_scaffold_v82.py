"""CADENCE V8.2 Scaffold: 12D observation vector with ImCoh + Concordance.

Replaces V8 cross-product EEG features with empirically-validated metrics:
  - EEG ImCoh (3ch): imaginary coherence per band (phase coupling)
  - EEG Concordance (3ch): (z_P1 + z_P2)/2 of band power (shared state)
  - BL expression coupling (1ch): per-segment wavelet coherence z (kept from V8)
  - BL activity concordance (1ch): shared facial activity (replaces BL state)
  - ECG LF/HF coupling (2ch): Hilbert envelope cross-product (kept from V8)
  - Resp phase coherence (1ch): cos(phi1-phi2) (kept from V8)
  - Pose velocity coupling (1ch): upper-body cross-product z (kept from V8)

Validation: 7/12 features significantly separate conversation from baseline.

Usage:
    python scripts/_run_scaffold_v82.py
    python scripts/_run_scaffold_v82.py --session y_06
    python scripts/_run_scaffold_v82.py --all
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert as sci_hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gf1d
import torch
from concurrent.futures import ThreadPoolExecutor

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    AFFECT_AUS, BAND_EXPRESSION,
)
from cadence.significance.fast_cycles import _fft_bandpass, EEG_BANDS
from cadence.preprocess.eeg.coherence import eeg_band_coherence, DEFAULT_EEG_BANDS
from cadence.config import load_config
from cadence.data import discover_cached_sessions, load_session_from_cache
from scripts._extract_respiratory import extract_respiratory_one, detect_rpeaks
from scripts._run_rslds_scaffold_v8 import (
    ecg_hilbert_coupling, respiratory_phase_coherence,
    pose_velocity_coupling, prewhiten_and_standardize,
    cross_product_z, ECG_SRATE,
)

# ── Constants ─────────────────────────────────────────────────────────

FS_BL_DEFAULT = 30.0  # fallback; actual rate detected from timestamps
FS_OUT = 2.0
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

# V8.2 18D modality configuration (15 original + 3 asymmetry)
MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_theta', 'dyn_alpha', 'dyn_beta',
    'asym_theta', 'asym_alpha', 'asym_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf',
    'resp', 'pose',
]
MODALITY_NAMES = [
    'ImCoh theta', 'ImCoh alpha', 'ImCoh beta',
    'Conc theta', 'Conc alpha', 'Conc beta',
    'Dyn theta', 'Dyn alpha', 'Dyn beta',
    'Asym theta', 'Asym alpha', 'Asym beta',
    'BL expression', 'BL activity conc',
    'ECG LF (SNS)', 'ECG HF (PNS)',
    'Resp phase', 'Pose velocity',
]
MODALITY_COLORS = [
    '#1565C0', '#2196F3', '#64B5F6',
    '#E65100', '#FF9800', '#FFB74D',
    '#B71C1C', '#D32F2F', '#E57373',
    '#00695C', '#00897B', '#4DB6AC',
    '#E91E63', '#F48FB1',
    '#FF5722', '#795548',
    '#607D8B', '#4CAF50',
]

DYN_TAU = 3.0  # EMA time constant for dynamics (seconds)


# ── Concordance Dynamics (EWMAD) ──────────────────────────────────────

def compute_concordance_dynamics(conc_raw, tau=DYN_TAU, fs=FS_OUT):
    """Exponentially-weighted mean absolute deviation of concordance.

    Captures the RATE OF CHANGE of shared state — high during
    conversation turn-taking, low during stable rest.

    Computed from RAW concordance (before prewhitening) to preserve
    the variance signal. Log-transformed for Gaussian-like distribution.

    Args:
        conc_raw: (N,) raw concordance timecourse
        tau: EMA time constant in seconds
        fs: sampling rate

    Returns:
        dyn: (N,) log-transformed EWMAD
    """
    alpha = 1.0 - np.exp(-1.0 / (tau * fs))
    delta = np.abs(np.diff(conc_raw, prepend=conc_raw[0]))
    dyn = np.empty_like(delta)
    dyn[0] = delta[0]
    for t in range(1, len(delta)):
        dyn[t] = alpha * delta[t] + (1.0 - alpha) * dyn[t - 1]
    # Log-transform: right-skewed → more symmetric
    dyn = np.log(dyn + 1e-6)
    return dyn.astype(np.float32)


# ── EEG: Imaginary Coherence ─────────────────────────────────────────

def compute_eeg_imcoh(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts):
    """Compute per-band imaginary coherence, resampled to 2 Hz."""
    N = len(t_common)
    result = {f'imcoh_{bn}': np.zeros(N, dtype=np.float32) for bn in ['theta', 'alpha', 'beta']}

    try:
        _, coh_imag, coh_times, _ = eeg_band_coherence(
            p1_eeg, p2_eeg, fs=fs_eeg, use_imcoh=True)
        coh_times_lsl = coh_times + (p1_ts[0] + lsl_offset)
        imag_avg = coh_imag.mean(axis=1)  # (n_bands, n_windows)
        band_names = list(DEFAULT_EEG_BANDS.keys())

        for bi, bn in enumerate(band_names):
            if bi < imag_avg.shape[0]:
                result[f'imcoh_{bn}'] = np.interp(
                    t_common, coh_times_lsl, imag_avg[bi],
                    left=0, right=0).astype(np.float32)
    except Exception as e:
        print(f"    ImCoh error: {e}")

    return result


# ── EEG: Concordance ─────────────────────────────────────────────────

def compute_eeg_concordance(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts):
    """Compute per-band concordance + asymmetry from Hilbert envelope.

    Concordance = (z_P1 + z_P2) / 2  — shared state level
    Asymmetry   = z_P1 - z_P2        — directional: positive = P1 has more power
    """
    N = len(t_common)
    result = {}
    for bn in ['theta', 'alpha', 'beta']:
        result[f'conc_{bn}'] = np.zeros(N, dtype=np.float32)
        result[f'asym_{bn}'] = np.zeros(N, dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_ch = p1_eeg.shape[1]
    T_eeg = len(p1_eeg)
    dur = T_eeg / fs_eeg
    t_grid = np.arange(0, dur, 1.0 / FS_OUT)
    t_eeg = np.arange(T_eeg) / fs_eeg
    t_grid_lsl = t_grid + (p1_ts[0] + lsl_offset)

    both = np.vstack([p1_eeg.T, p2_eeg.T])
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)

    for band_name, (lo, hi) in EEG_BANDS.items():
        filt = _fft_bandpass(both_t, fs_eeg, (lo, hi), device).cpu().numpy()
        p1_env = np.abs(sci_hilbert(filt[:n_ch].T, axis=0)).mean(axis=1)
        p2_env = np.abs(sci_hilbert(filt[n_ch:].T, axis=0)).mean(axis=1)

        p1_2hz = np.interp(t_grid, t_eeg, p1_env)
        p2_2hz = np.interp(t_grid, t_eeg, p2_env)

        for arr in [p1_2hz, p2_2hz]:
            mu, sd = arr.mean(), arr.std()
            if sd > 1e-8:
                arr[:] = (arr - mu) / sd

        conc = ((p1_2hz + p2_2hz) / 2.0).astype(np.float32)
        asym = (p1_2hz - p2_2hz).astype(np.float32)

        result[f'conc_{band_name}'] = np.interp(
            t_common, t_grid_lsl, conc, left=0, right=0).astype(np.float32)
        result[f'asym_{band_name}'] = np.interp(
            t_common, t_grid_lsl, asym, left=0, right=0).astype(np.float32)

    return result


# ── BL Activity Concordance ──────────────────────────────────────────

def compute_bl_activity_concordance(cached, t_common, lsl_offset):
    """BL activity concordance: (z_P1 + z_P2)/2 of facial activity channel."""
    N = len(t_common)
    z = np.zeros(N, dtype=np.float32)

    if ('p1_blendshapes' not in cached or 'p2_blendshapes' not in cached
            or 'p1_blendshapes_ts' not in cached):
        return z

    n_ch1 = cached['p1_blendshapes'].shape[1]
    n_ch2 = cached['p2_blendshapes'].shape[1]
    act_ch = min(n_ch1, n_ch2) - 1
    if act_ch < 52:
        return z

    p1_act = np.interp(t_common, cached['p1_blendshapes_ts'] + lsl_offset,
                        cached['p1_blendshapes'][:, act_ch])
    p2_act = np.interp(t_common, cached['p2_blendshapes_ts'] + lsl_offset,
                        cached['p2_blendshapes'][:, act_ch])

    for arr in [p1_act, p2_act]:
        mu, sd = arr.mean(), arr.std()
        if sd > 1e-8:
            arr[:] = (arr - mu) / sd
        else:
            arr[:] = 0.0

    return ((p1_act + p2_act) / 2.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  SESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name, config, raw_dir=RAW_DIR):
    """Run V8.2 scaffold for one session: 12D observation vector."""
    print(f"\n{'='*70}")
    print(f"  V8.2 Scaffold -- {session_name}")
    print(f"{'='*70}")

    t_wall = time.time()
    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load session ──────────────────────────────────────────────
    print(f"\n[1/9] Loading {session_name}...", flush=True)

    xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  WARNING: No XDF found for {session_name}, skipping")
        return None
    session_data = load_xdf_session(xdf_files[0])
    markers = session_data['markers']
    p1_role = session_data.get('p1_role', 'unknown')
    # Asymmetry sign: always therapist - patient
    # If P1 is patient, flip sign so positive = therapist has more
    asym_sign = -1.0 if p1_role == 'patient' else 1.0
    print(f"  P1={p1_role} -> asym_sign={asym_sign:+.0f} (positive = therapist higher)")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        print(f"  WARNING: No cache for {session_name}, skipping")
        return None
    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    segments = []
    for seg_name in CONDITION_ORDER:
        t_start = markers.get(f'{seg_name}_start')
        t_end = markers.get(f'{seg_name}_stop')
        if t_start is not None and t_end is not None and t_end > t_start:
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

    # ── 2. EEG: ImCoh + Concordance ──────────────────────────────────
    print(f"\n[2/9] EEG ImCoh + Concordance...", flush=True)
    t0 = time.time()

    _has_eeg = ('p1_eeg' in cached and 'p2_eeg' in cached
                and 'p1_eeg_ts' in cached and 'p2_eeg_ts' in cached)
    if _has_eeg:
        p1_eeg = cached['p1_eeg'].astype(np.float64)
        p2_eeg = cached['p2_eeg'].astype(np.float64)
        p1_ts = cached['p1_eeg_ts']
        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]; p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]; p2_eeg = p2_eeg[:mlen]
        p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
        p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
        for ch in range(n_ch):
            for arr in [p1_eeg, p2_eeg]:
                sd = arr[:, ch].std()
                if sd > 1e-8:
                    arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd
        fs_eeg = len(p1_ts) / (p1_ts[-1] - p1_ts[0])

        # ImCoh
        imcoh = compute_eeg_imcoh(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in imcoh.items():
            z_traces[k] = v

        # Concordance + Asymmetry
        conc = compute_eeg_concordance(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in conc.items():
            if k.startswith('asym_'):
                z_traces[k] = v * asym_sign  # flip so positive = therapist higher
            else:
                z_traces[k] = v

        # Dynamics (EWMAD of raw concordance — compute BEFORE prewhitening)
        for bn in ['theta', 'alpha', 'beta']:
            z_traces[f'dyn_{bn}'] = compute_concordance_dynamics(z_traces[f'conc_{bn}'])

        for bn in ['theta', 'alpha', 'beta']:
            print(f"    {bn}: ImCoh={z_traces[f'imcoh_{bn}'].mean():.3f}, "
                  f"Conc={z_traces[f'conc_{bn}'].mean():+.3f}, "
                  f"Dyn={z_traces[f'dyn_{bn}'].mean():+.3f}")
    else:
        print(f"    No EEG data")
    print(f"  EEG: {time.time() - t0:.1f}s")

    # ── 3. BL expression coupling (per-segment wavelet coherence) ────
    print(f"\n[3/9] BL expression coupling...", flush=True)
    t0 = time.time()
    landmarks = session_data['landmarks']

    for seg_name, t0_seg, t1_seg in segments:
        p1_bl, p2_bl, bl_dur = extract_bl_segment(landmarks, t0_seg, t1_seg)
        if p1_bl is None or bl_dur < 5:
            continue
        T_bl = p1_bl.shape[0]
        bl_times = np.linspace(t0_seg, t1_seg, T_bl)
        scal_p1 = compute_au_cwt(p1_bl[:, AFFECT_AUS])
        scal_p2 = compute_au_cwt(p2_bl[:, AFFECT_AUS])
        freqs = scal_p1.freqs
        z_coh = surrogate_coherence_z(scal_p1, scal_p2, 200, 0.5,
                                       list(range(len(AFFECT_AUS))), 42, 'auto')
        expr_fmask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
        if expr_fmask.sum() > 0:
            expr_z_30 = z_coh['z'][expr_fmask].max(axis=0)
            seg_mask = (t_common >= t0_seg) & (t_common <= t1_seg)
            if seg_mask.sum() > 0:
                z_traces['bl_expr'][seg_mask] = np.interp(
                    t_common[seg_mask], bl_times, expr_z_30,
                    left=0, right=0).astype(np.float32)
    print(f"  BL: {time.time() - t0:.1f}s")

    # ── 4. BL activity concordance ───────────────────────────────────
    print(f"\n[4/9] BL activity concordance...", flush=True)
    z_traces['bl_activity_conc'] = compute_bl_activity_concordance(
        cached, t_common, lsl_offset)
    print(f"    mean={z_traces['bl_activity_conc'].mean():+.3f}")

    # ── 5. ECG LF/HF coupling ───────────────────────────────────────
    print(f"\n[5/9] ECG Hilbert coupling...", flush=True)
    t0 = time.time()
    has_raw_ecg = ('p1_ecg' in cached and 'p2_ecg' in cached
                   and len(cached.get('p1_ecg', [])) > 1000)
    if has_raw_ecg:
        z_traces['ecg_lf'], z_traces['ecg_hf'] = ecg_hilbert_coupling(
            cached['p1_ecg'], cached['p1_ecg_ts'],
            cached['p2_ecg'], cached['p2_ecg_ts'],
            t_common, lsl_offset, srate=ECG_SRATE)
    print(f"  ECG: {time.time() - t0:.1f}s")

    # ── 6. Respiratory phase coherence ───────────────────────────────
    print(f"\n[6/9] Respiratory phase coherence...", flush=True)
    t0 = time.time()
    if has_raw_ecg:
        p1_resp = extract_respiratory_one(cached['p1_ecg'], cached['p1_ecg_ts'], srate=ECG_SRATE)
        p2_resp = extract_respiratory_one(cached['p2_ecg'], cached['p2_ecg_ts'], srate=ECG_SRATE)
        z_traces['resp'] = respiratory_phase_coherence(p1_resp, p2_resp, t_common, lsl_offset)
    print(f"  Resp: {time.time() - t0:.1f}s")

    # ── 7. Pose velocity coupling ────────────────────────────────────
    print(f"\n[7/9] Pose velocity coupling...", flush=True)
    z_traces['pose'] = pose_velocity_coupling(cached, t_common, lsl_offset)

    # ── 8. Per-timepoint obs_mask + prewhiten + standardize ──────────
    print(f"\n[8/9] Masking + prewhitening + standardization...", flush=True)

    z_matrix_raw = np.column_stack([z_traces[k] for k in MODALITY_KEYS])
    z_matrix, pw_diag = prewhiten_and_standardize(z_matrix_raw, MODALITY_KEYS)

    # Per-timepoint obs_mask for BL/pose
    obs_mask = np.ones((N_common, len(MODALITY_KEYS)), dtype=bool)
    for d, key in enumerate(MODALITY_KEYS):
        if np.abs(z_matrix_raw[:, d]).max() < 1e-8:
            obs_mask[:, d] = False

    # BL face validity
    bl_idx = [MODALITY_KEYS.index(k) for k in ['bl_expr', 'bl_activity_conc']]
    pose_idx = MODALITY_KEYS.index('pose')

    def _interp_valid(valid_arr, ts_arr, t_out, lsl_off):
        v_float = valid_arr.astype(np.float32)
        v_interp = np.interp(t_out, ts_arr + lsl_off, v_float, left=0, right=0)
        return v_interp > 0.5

    has_bl_valid = ('p1_blendshapes_valid' in cached and 'p2_blendshapes_valid' in cached
                    and 'p1_blendshapes_ts' in cached)
    if has_bl_valid:
        p1_bl_v = _interp_valid(cached['p1_blendshapes_valid'],
                                cached['p1_blendshapes_ts'], t_common, lsl_offset)
        p2_bl_v = _interp_valid(cached['p2_blendshapes_valid'],
                                cached['p2_blendshapes_ts'], t_common, lsl_offset)
        bl_valid = p1_bl_v & p2_bl_v
        for d in bl_idx:
            obs_mask[:, d] &= bl_valid
        print(f"  BL masked: {(~bl_valid).sum()} pts ({(~bl_valid).mean():.1%})")

    has_pose_valid = ('p1_pose_features_valid' in cached and 'p2_pose_features_valid' in cached
                      and 'p1_pose_features_ts' in cached)
    if has_pose_valid:
        p1_pv = _interp_valid(cached['p1_pose_features_valid'],
                              cached['p1_pose_features_ts'], t_common, lsl_offset)
        p2_pv = _interp_valid(cached['p2_pose_features_valid'],
                              cached['p2_pose_features_ts'], t_common, lsl_offset)
        pose_valid = p1_pv & p2_pv
        obs_mask[:, pose_idx] &= pose_valid
        print(f"  Pose masked: {(~pose_valid).sum()} pts ({(~pose_valid).mean():.1%})")

    print(f"\n  {'Modality':>18s} | rho_raw | rho_pw | pw | std")
    print(f"  " + "-" * 55)
    for key in MODALITY_KEYS:
        print(f"  {key:>18s} | {pw_diag['rho_before'].get(key, 0):.3f}   | "
              f"{pw_diag['rho_after'].get(key, 0):.3f}  | "
              f"{'Y' if pw_diag['prewhitened'].get(key, False) else ' ':1s}  | "
              f"{pw_diag['std_after'].get(key, 0):.3f}")

    # ── 9. Save + visualize ──────────────────────────────────────────
    print(f"\n[9/9] Save + visualize...", flush=True)

    save_dict = {'t_common': t_common, 'obs_mask': obs_mask}
    for i, key in enumerate(MODALITY_KEYS):
        save_dict[f'z_{key}'] = z_matrix[:, i]
        save_dict[f'z_raw_{key}'] = z_matrix_raw[:, i]

    np.savez_compressed(os.path.join(out_dir, 'scaffold_v82_ztimecourses.npz'), **save_dict)

    results = {
        'session': session_name, 'version': 'v8.2',
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
    }
    with open(os.path.join(out_dir, 'scaffold_v82_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Timeline plot
    fig, axes = plt.subplots(len(MODALITY_KEYS), 1,
                             figsize=(28, 2.0 * len(MODALITY_KEYS)), sharex=True)
    for ax in axes:
        for seg_name, t0s, t1s in segments:
            ax.axvspan(t0s, t1s, alpha=0.2,
                       color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)
    for seg_name, t0s, t1s in segments:
        axes[0].text((t0s + t1s) / 2, 1.08, seg_name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=7, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        ax.plot(t_common, z_matrix[:, idx], color=color, linewidth=0.5, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
        rho_b = pw_diag['rho_before'].get(key, 0)
        rho_a = pw_diag['rho_after'].get(key, 0)
        pw = 'PW' if pw_diag['prewhitened'].get(key, False) else ''
        ax.text(0.01, 0.92, f'rho={rho_b:.2f}->{rho_a:.2f} {pw}',
                transform=ax.transAxes, fontsize=6, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=5)
    axes[-1].set_xlabel('LSL time (s)', fontsize=8)

    fig.suptitle(f'{session_name} -- V8.2 Scaffold (12D)', fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'scaffold_v82_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    total = time.time() - t_wall
    print(f"\n  V8.2 scaffold complete: {total:.0f}s")
    return results


def run_from_raw(raw_data, t_common, segments=None, label='semisynthetic'):
    """Run V8.2 scaffold on pre-injected raw arrays (for semi-synthetic tests).

    Processes raw EEG/BL/ECG/Pose arrays through the full pipeline:
    ImCoh, concordance, dynamics, asymmetry, BL wavelet, ECG Hilbert,
    respiratory phase coherence, pose velocity, prewhitening, standardization.

    Args:
        raw_data: dict with keys like 'p1_eeg', 'p2_eeg', 'p1_eeg_ts', etc.
            Same format as cached session data + raw landmarks.
        t_common: (N,) common 2 Hz time grid (LSL timestamps).
        segments: list of (name, t_start, t_end) or None.
        label: session label for logging.

    Returns:
        z_matrix: (N, 18) prewhitened/standardized feature matrix.
        z_matrix_raw: (N, 18) raw (pre-prewhitening) features.
        obs_mask: (N, 18) boolean observation mask.
        pw_diag: prewhitening diagnostics dict.
    """
    N_common = len(t_common)
    z_traces = {k: np.zeros(N_common, dtype=np.float32) for k in MODALITY_KEYS}
    lsl_offset = 0.0  # raw_data timestamps are already in LSL time

    # ── EEG ─────────────────────────────────────────────────────────
    if 'p1_eeg' in raw_data and 'p2_eeg' in raw_data:
        p1_eeg = raw_data['p1_eeg'].astype(np.float64)
        p2_eeg = raw_data['p2_eeg'].astype(np.float64)
        p1_ts = raw_data.get('p1_eeg_ts', np.arange(len(p1_eeg)) / 256.0 + t_common[0])
        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]; p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]; p2_eeg = p2_eeg[:mlen]
        # Z-score per channel
        p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
        p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
        for ch in range(n_ch):
            for arr in [p1_eeg, p2_eeg]:
                sd = arr[:, ch].std()
                if sd > 1e-8:
                    arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd
        fs_eeg = len(p1_ts) / max(p1_ts[-1] - p1_ts[0], 1.0)

        # ImCoh
        imcoh = compute_eeg_imcoh(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in imcoh.items():
            z_traces[k] = v

        # Concordance + Asymmetry
        conc = compute_eeg_concordance(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in conc.items():
            z_traces[k] = v  # no role correction in semisynthetic

        # Dynamics
        for bn in ['theta', 'alpha', 'beta']:
            z_traces[f'dyn_{bn}'] = compute_concordance_dynamics(z_traces[f'conc_{bn}'])

    # ── BL ──────────────────────────────────────────────────────────
    if 'p1_blendshapes' in raw_data and 'p2_blendshapes' in raw_data:
        # BL expression coupling (per-segment or full)
        p1_bl = raw_data['p1_blendshapes']
        p2_bl = raw_data['p2_blendshapes']
        bl_ts_p1 = raw_data.get('p1_blendshapes_ts',
                                np.arange(len(p1_bl)) / FS_BL_DEFAULT + t_common[0])
        # Detect actual BL sampling rate from timestamps
        if len(bl_ts_p1) > 100:
            dt = np.median(np.diff(bl_ts_p1[:min(2000, len(bl_ts_p1))]))
            FS_BL = 1.0 / dt if dt > 0 else FS_BL_DEFAULT
        else:
            FS_BL = FS_BL_DEFAULT
        print(f"  BL sampling rate: {FS_BL:.1f} Hz")
        # Ensure timestamps match data length (may differ after injection truncation)
        if len(bl_ts_p1) != len(p1_bl):
            bl_ts_p1 = np.linspace(bl_ts_p1[0], bl_ts_p1[-1], len(p1_bl))

        if segments:
            for seg_name, t0_seg, t1_seg in segments:
                mask_bl = (bl_ts_p1 >= t0_seg) & (bl_ts_p1 < t1_seg)
                if mask_bl.sum() < 150:  # <5s
                    continue
                seg_p1 = p1_bl[mask_bl]
                seg_p2 = p2_bl[mask_bl[:len(p2_bl)] if len(p2_bl) != len(p1_bl) else mask_bl]
                if len(seg_p1) < 150 or len(seg_p2) < 150:
                    continue
                n_au = min(seg_p1.shape[1], seg_p2.shape[1])
                affect_idx = [a for a in AFFECT_AUS if a < n_au]
                if not affect_idx:
                    continue
                scal_p1 = compute_au_cwt(seg_p1[:, affect_idx], fs=FS_BL)
                scal_p2 = compute_au_cwt(seg_p2[:min(len(seg_p1), len(seg_p2)), affect_idx], fs=FS_BL)
                freqs = scal_p1.freqs
                z_coh = surrogate_coherence_z(scal_p1, scal_p2, 200, 0.5,
                                               list(range(len(affect_idx))), 42, 'auto')
                expr_fmask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
                if expr_fmask.sum() > 0:
                    expr_z_30 = z_coh['z'][expr_fmask].max(axis=0)
                    T_z = len(expr_z_30)
                    bl_times = np.linspace(t0_seg, t1_seg, T_z)
                    seg_mask = (t_common >= t0_seg) & (t_common <= t1_seg)
                    if seg_mask.sum() > 0:
                        z_traces['bl_expr'][seg_mask] = np.interp(
                            t_common[seg_mask], bl_times, expr_z_30,
                            left=0, right=0).astype(np.float32)
        else:
            # Process as single segment
            n_au = min(p1_bl.shape[1], p2_bl.shape[1])
            affect_idx = [a for a in AFFECT_AUS if a < n_au]
            mlen_bl = min(len(p1_bl), len(p2_bl))
            if affect_idx and mlen_bl >= 150:
                scal_p1 = compute_au_cwt(p1_bl[:mlen_bl, affect_idx], fs=FS_BL)
                scal_p2 = compute_au_cwt(p2_bl[:mlen_bl, affect_idx], fs=FS_BL)
                freqs = scal_p1.freqs
                z_coh = surrogate_coherence_z(scal_p1, scal_p2, 200, 0.5,
                                               list(range(len(affect_idx))), 42, 'auto')
                expr_fmask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
                if expr_fmask.sum() > 0:
                    expr_z_30 = z_coh['z'][expr_fmask].max(axis=0)
                    T_z = len(expr_z_30)
                    bl_times = np.linspace(t_common[0], t_common[-1], T_z)
                    z_traces['bl_expr'] = np.interp(
                        t_common, bl_times, expr_z_30,
                        left=0, right=0).astype(np.float32)

        # BL activity concordance
        if 'p1_blendshapes' in raw_data:
            z_traces['bl_activity_conc'] = compute_bl_activity_concordance(
                raw_data, t_common, lsl_offset)

    # ── Synchronize all timestamp/data lengths ────────────────────
    for prefix in ['p1_ecg', 'p2_ecg', 'p1_pose_features', 'p2_pose_features']:
        data_key = prefix
        ts_key = f'{prefix}_ts'
        if data_key in raw_data and ts_key in raw_data:
            n_data = len(raw_data[data_key])
            n_ts = len(raw_data[ts_key])
            if n_data != n_ts:
                n_min = min(n_data, n_ts)
                raw_data[data_key] = raw_data[data_key][:n_min]
                raw_data[ts_key] = raw_data[ts_key][:n_min]

    # ── ECG ─────────────────────────────────────────────────────────
    has_ecg = ('p1_ecg' in raw_data and 'p2_ecg' in raw_data
               and len(raw_data.get('p1_ecg', [])) > 1000)
    if has_ecg:
        try:
            # Check for pre-injected ECG IBI (semi-synthetic)
            if '_ecg_ibi2_coupled' in raw_data:
                # Use pre-computed coupled IBI instead of re-detecting R-peaks
                from scripts._run_rslds_scaffold_v8 import FS_IBI
                ibi1 = raw_data['_ecg_ibi1']
                ibi2 = raw_data['_ecg_ibi2_coupled']
                t_ibi = raw_data['_ecg_ibi_t'] + lsl_offset
                T_ibi = min(len(ibi1), len(ibi2), len(t_ibi))
                for band_name, lo, hi in [('lf', 0.04, 0.15), ('hf', 0.15, 0.4)]:
                    nyq = FS_IBI / 2.0
                    if hi >= nyq:
                        hi = nyq * 0.95
                    sos_ecg = butter(4, [lo/nyq, hi/nyq], btype='band', output='sos')
                    bp1 = sosfiltfilt(sos_ecg, ibi1[:T_ibi])
                    bp2 = sosfiltfilt(sos_ecg, ibi2[:T_ibi])
                    env1 = np.abs(sci_hilbert(bp1))
                    env2 = np.abs(sci_hilbert(bp2))
                    z_band = cross_product_z(env1, env2, n_surrogates=200, seed=42)
                    z_on_common = np.interp(t_common, t_ibi[:T_ibi], z_band,
                                            left=0, right=0).astype(np.float32)
                    if band_name == 'lf':
                        z_traces['ecg_lf'] = z_on_common
                    else:
                        z_traces['ecg_hf'] = z_on_common
            else:
                z_traces['ecg_lf'], z_traces['ecg_hf'] = ecg_hilbert_coupling(
                    raw_data['p1_ecg'], raw_data['p1_ecg_ts'],
                    raw_data['p2_ecg'], raw_data['p2_ecg_ts'],
                    t_common, lsl_offset, srate=ECG_SRATE)
        except Exception:
            pass

    # ── Respiratory ─────────────────────────────────────────────────
    if has_ecg:
        try:
            # Check for pre-injected respiratory (semi-synthetic)
            if '_resp_p2_fused_coupled' in raw_data and '_resp_p1' in raw_data:
                p1_resp = raw_data['_resp_p1']
                # Build p2_resp dict with coupled fused signal
                p2_resp = {
                    'fused': raw_data['_resp_p2_fused_coupled'],
                    't': raw_data['_resp_p2_t'] + lsl_offset,
                }
                z_traces['resp'] = respiratory_phase_coherence(
                    p1_resp, p2_resp, t_common, 0.0)  # offset already applied
            else:
                p1_resp = extract_respiratory_one(raw_data['p1_ecg'],
                                                  raw_data['p1_ecg_ts'], srate=ECG_SRATE)
                p2_resp = extract_respiratory_one(raw_data['p2_ecg'],
                                                  raw_data['p2_ecg_ts'], srate=ECG_SRATE)
                z_traces['resp'] = respiratory_phase_coherence(p1_resp, p2_resp,
                                                                t_common, lsl_offset)
        except Exception:
            pass

    # ── Pose ────────────────────────────────────────────────────────
    # Check for pre-injected velocity (semi-synthetic — bypasses position→velocity)
    # For semi-synthetic: if pre-injected velocity is available, build a
    # temporary cached dict with modified pose positions (integrated from
    # coupled velocity) and let the production multi-lag pipeline process it.
    if '_pose_vel_p2_coupled' in raw_data:
        try:
            _p1p = raw_data['p1_pose_features']
            _t1p = raw_data['p1_pose_features_ts'] + lsl_offset
            _t2p = raw_data['p2_pose_features_ts'] + lsl_offset
            _v2_coupled = raw_data['_pose_vel_p2_coupled']  # (T_2hz, n_ch) at 2 Hz

            # Build a synthetic cached dict with position data that produces
            # the coupled velocity when interpolated to 2 Hz and diff'd.
            # Approach: integrate velocity at 2 Hz to get positions at 2 Hz,
            # then treat these as if they were at native rate (the pipeline
            # will interp to 2 Hz which is identity if already at 2 Hz).
            _n_ch = _v2_coupled.shape[1]
            _p2_pos = np.cumsum(_v2_coupled, axis=0).astype(np.float64)
            # Create a synthetic cache with 2 Hz "pose features"
            _synth_cached = dict(raw_data)
            _synth_cached['p2_pose_features'] = _p2_pos
            _synth_cached['p2_pose_features_ts'] = t_common - lsl_offset  # local time

            z_traces['pose'] = pose_velocity_coupling(
                _synth_cached, t_common, lsl_offset)
        except Exception:
            z_traces['pose'] = pose_velocity_coupling(raw_data, t_common, lsl_offset)
    else:
        try:
            z_traces['pose'] = pose_velocity_coupling(raw_data, t_common, lsl_offset)
        except Exception:
            pass

    # ── Prewhiten + standardize ─────────────────────────────────────
    z_matrix_raw = np.column_stack([z_traces[k] for k in MODALITY_KEYS])
    z_matrix, pw_diag = prewhiten_and_standardize(z_matrix_raw, MODALITY_KEYS)

    # Observation mask
    obs_mask = np.ones((N_common, len(MODALITY_KEYS)), dtype=bool)
    for d, key in enumerate(MODALITY_KEYS):
        if np.abs(z_matrix_raw[:, d]).max() < 1e-8:
            obs_mask[:, d] = False

    # Collect extra diagnostic signals (raw cross-products before surrogates)
    extras = {}
    for k in list(z_traces.keys()):
        if k.startswith('_'):
            extras[k] = z_traces[k]

    return z_matrix, z_matrix_raw, obs_mask, pw_diag, extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', nargs='+', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    config = load_config()

    if args.all:
        xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
        sessions = [os.path.splitext(os.path.basename(f))[0].lower() for f in xdf_files]
    elif args.session:
        sessions = args.session
    else:
        sessions = ['y_06']

    print(f"V8.2 Scaffold: {len(sessions)} sessions")
    for sname in sessions:
        run_session(sname, config)


if __name__ == '__main__':
    main()
