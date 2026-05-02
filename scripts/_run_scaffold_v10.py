"""CADENCE V10 Scaffold: 23D observation vector + 5D transition covariates.

Extends V8.2 (18D) with:
  - LZ complexity: concordance + asymmetry for theta/alpha (4ch)
  - Graph metrics: modularity Q + EEG eigenvector centrality (2ch)
  - Transition covariates: coupling flexibility + λ₂ + graph change-point (3ch)

Reuses all V8.2 extraction functions via import (DRY).

Usage:
    python scripts/_run_scaffold_v10.py                    # Single session (y_06)
    python scripts/_run_scaffold_v10.py --session y_17     # Specific session
    python scripts/_run_scaffold_v10.py --all              # All sessions
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gf1d
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

# ── V8.2 imports (DRY — reuse, never copy) ──────────────────────────
from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from scripts._run_scaffold_v82 import (
    compute_eeg_imcoh, compute_eeg_concordance, compute_concordance_dynamics,
    compute_bl_activity_concordance,
    MODALITY_KEYS as V82_KEYS, MODALITY_NAMES as V82_NAMES,
    MODALITY_COLORS as V82_COLORS,
    FS_OUT, DYN_TAU, CONDITION_ORDER, CONDITION_COLORS,
)
from scripts._run_rslds_scaffold_v8 import (
    ecg_hilbert_coupling, respiratory_phase_coherence,
    pose_velocity_coupling, prewhiten_and_standardize,
    ECG_SRATE,
)
from scripts._extract_respiratory import extract_respiratory_one

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.significance.lz_complexity import extract_lz_timecourse, lz_concordance_asymmetry
from cadence.significance.spectral_graph import (
    build_modality_graph, build_modality_graph_windowed,
    coupling_flexibility_index,
    graph_modularity_windowed, eigenvector_centrality_windowed,
    graph_changepoint_score,
)
from cadence.constants import (
    V10_MODALITY_KEYS, V10_MODALITY_NAMES, V10_MODALITY_COLORS,
    V10_COVARIATE_KEYS, FRONTAL_ROI,
)
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    AFFECT_AUS, BAND_EXPRESSION,
)

# ── Constants ────────────────────────────────────────────────────────

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
Z_THRESH = 2.0
PREWHITEN_RHO_THRESH = 0.3

# Graph window parameters
GRAPH_WINDOW_S = 90
GRAPH_STRIDE_S = 15

# z_slow smoothing for transition covariates
ZSLOW_SIGMA_S = 30.0


# ══════════════════════════════════════════════════════════════════════
#  V10-SPECIFIC EXTRACTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def compute_lz_features(cached, t_common, lsl_offset, asym_sign=1.0):
    """Extract LZ concordance + asymmetry for theta and alpha.

    Returns dict with keys: lz_conc_theta, lz_conc_alpha,
                             lz_asym_theta, lz_asym_alpha.
    """
    N = len(t_common)
    result = {k: np.zeros(N, dtype=np.float32)
              for k in ['lz_conc_theta', 'lz_conc_alpha',
                        'lz_asym_theta', 'lz_asym_alpha']}

    if ('p1_eeg' not in cached or 'p2_eeg' not in cached
            or 'p1_eeg_ts' not in cached):
        return result

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    p2_ts = cached.get('p2_eeg_ts', p1_ts)
    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg = p1_eeg[:, :n_ch]
    p2_eeg = p2_eeg[:, :n_ch]

    fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0

    lz_p1 = extract_lz_timecourse(p1_eeg, fs_eeg, roi_channels=FRONTAL_ROI)
    lz_p2 = extract_lz_timecourse(p2_eeg, fs_eeg, roi_channels=FRONTAL_ROI)

    # Convert LZ timecourse timestamps to LSL
    t_lz_p1 = lz_p1['t_out'] + (p1_ts[0] + lsl_offset)
    t_lz_p2 = lz_p2['t_out'] + (p2_ts[0] + lsl_offset)

    for band_name in ['theta', 'alpha']:
        # Interpolate to common time grid
        p1_lz_2hz = np.interp(t_common, t_lz_p1, lz_p1[band_name], left=np.nan, right=np.nan)
        p2_lz_2hz = np.interp(t_common, t_lz_p2, lz_p2[band_name], left=np.nan, right=np.nan)

        # Replace NaN edges with mean
        for arr in [p1_lz_2hz, p2_lz_2hz]:
            valid = np.isfinite(arr)
            if valid.any():
                arr[~valid] = arr[valid].mean()
            else:
                arr[:] = 0.5  # neutral LZ

        conc, asym = lz_concordance_asymmetry(p1_lz_2hz, p2_lz_2hz, asym_sign=asym_sign)
        result[f'lz_conc_{band_name}'] = conc.astype(np.float32)
        result[f'lz_asym_{band_name}'] = asym.astype(np.float32)

    return result


def compute_graph_features(z_18, t_common):
    """Compute graph modularity + EEG centrality from base 18D z-matrix.

    Also returns windowed graph results for transition covariate extraction.

    Args:
        z_18: (N, 18) base V8.2 observation matrix (pre-prewhitened).
        t_common: (N,) time grid in LSL seconds.

    Returns:
        graph_traces: dict with graph_modularity.
        windowed: dict with windowed graph results (for covariates).
        mod_result: dict from graph_modularity_windowed (for change-point).
    """
    N = len(t_common)
    graph_traces = {
        'graph_modularity': np.zeros(N, dtype=np.float32),
    }

    if z_18.shape[0] < int(GRAPH_WINDOW_S * FS_OUT):
        return graph_traces, None, None

    # Windowed graph (flexibility + lambda2 for covariates)
    windowed = build_modality_graph_windowed(
        z_18, window_s=GRAPH_WINDOW_S, stride_s=GRAPH_STRIDE_S, fs=FS_OUT)

    # Modularity (Louvain) — only observation-level graph metric
    # Centrality dropped: near-constant timecourse → spurious AUC after prewhitening.
    # Lambda-2 (algebraic connectivity) already captures graph topology in covariates.
    mod_result = graph_modularity_windowed(
        z_18, window_s=GRAPH_WINDOW_S, stride_s=GRAPH_STRIDE_S, fs=FS_OUT)

    # Interpolate to 2 Hz
    t0 = t_common[0]
    t_mod = mod_result['t_centers'] + t0

    graph_traces['graph_modularity'] = np.interp(
        t_common, t_mod, mod_result['modularity_Q'],
        left=mod_result['modularity_Q'][0],
        right=mod_result['modularity_Q'][-1]).astype(np.float32)

    return graph_traces, windowed, mod_result


def build_transition_covariates(z_18, t_common, windowed, mod_result):
    """Build 5D transition covariate matrix U.

    U[:, 0:2] = z_slow PCs (smoothed slow components)
    U[:, 2]   = coupling flexibility (from windowed graph)
    U[:, 3]   = λ₂ algebraic connectivity
    U[:, 4]   = graph change-point score

    Args:
        z_18: (N, 18) base V8.2 observation matrix.
        t_common: (N,) time grid.
        windowed: dict from build_modality_graph_windowed.
        mod_result: dict from graph_modularity_windowed.

    Returns:
        U: (N, 5) transition covariate matrix.
    """
    N = len(t_common)
    U = np.zeros((N, 5), dtype=np.float32)

    # z_slow PCs: low-pass filter then PCA, standardized
    sigma_samp = ZSLOW_SIGMA_S * FS_OUT
    z_slow = gf1d(z_18, sigma=sigma_samp, axis=0)
    pca = PCA(n_components=2)
    pc_raw = pca.fit_transform(z_slow)
    for c in range(2):
        s = pc_raw[:, c].std()
        if s > 1e-8:
            pc_raw[:, c] /= s
    U[:, 0:2] = pc_raw.astype(np.float32)

    if windowed is None or mod_result is None:
        return U

    t0 = t_common[0]
    t_win = windowed['t_centers'] + t0
    t_mod = mod_result['t_centers'] + t0

    # Coupling flexibility
    U[:, 2] = np.interp(t_common, t_win, windowed['flexibility'],
                         left=windowed['flexibility'][0],
                         right=windowed['flexibility'][-1]).astype(np.float32)

    # Lambda-2 (algebraic connectivity)
    U[:, 3] = np.interp(t_common, t_win, windowed['lambda2'],
                         left=windowed['lambda2'][0],
                         right=windowed['lambda2'][-1]).astype(np.float32)

    # Graph change-point score
    cp = graph_changepoint_score(
        mod_result['modularity_Q'], windowed['lambda2'],
        windowed['n_edges'].astype(np.float64),
        sigma_s=5.0, stride_s=GRAPH_STRIDE_S)
    U[:, 4] = np.interp(t_common, t_mod, cp,
                         left=0, right=0).astype(np.float32)

    return U


# ══════════════════════════════════════════════════════════════════════
#  SESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name, config, raw_dir=RAW_DIR):
    """Run V10 scaffold: 23D observation + 5D covariates."""
    print(f"\n{'='*70}")
    print(f"  V10 Scaffold -- {session_name}")
    print(f"{'='*70}")

    t_wall = time.time()
    out_dir = f'results/v10/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ── [1/12] Load session ──────────────────────────────────────────
    print(f"\n[1/12] Loading {session_name}...", flush=True)

    xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  WARNING: No XDF found for {session_name}, skipping")
        return None
    session_data = load_xdf_session(xdf_files[0])
    markers = session_data['markers']  # dict: marker_name -> timestamp
    p1_role = session_data.get('p1_role', 'unknown')
    asym_sign = -1.0 if p1_role == 'patient' else 1.0
    print(f"  P1={p1_role} -> asym_sign={asym_sign:+.0f}")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        print(f"  WARNING: No cache for {session_name}, skipping")
        return None
    cached = load_session_from_cache(cache_matches[0], config)

    # LSL offset (same as V8.2)
    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    # Segments from markers dict
    segments = []
    for cond in CONDITION_ORDER:
        t_start = markers.get(f'{cond}_start')
        t_end = markers.get(f'{cond}_stop')
        if t_start is not None and t_end is not None and t_end > t_start:
            segments.append((cond, t_start, t_end))

    if not segments:
        print(f"  WARNING: No segments for {session_name}, skipping")
        return None

    # Time grid
    session_start_lsl = min(t for _, t, _ in segments) - 30
    session_end_lsl = max(t for _, _, t in segments) + 30
    t_common = np.arange(session_start_lsl, session_end_lsl, 1.0 / FS_OUT)
    N_common = len(t_common)
    print(f"  N={N_common}, T={session_end_lsl - session_start_lsl:.0f}s, "
          f"segs: {[s[0] for s in segments]}")

    z_traces = {k: np.zeros(N_common, dtype=np.float32) for k in V10_MODALITY_KEYS}

    # ── [2/12] EEG ImCoh + Concordance + Dynamics + Asymmetry ────────
    print(f"\n[2/12] EEG features...", flush=True)
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

        fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000]))
        imcoh = compute_eeg_imcoh(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in imcoh.items():
            z_traces[k] = v

        conc = compute_eeg_concordance(p1_eeg, p2_eeg, fs_eeg, t_common, lsl_offset, p1_ts)
        for k, v in conc.items():
            if k.startswith('asym_'):
                z_traces[k] = v * asym_sign
            else:
                z_traces[k] = v

        z_traces['dyn_mean'] = np.mean([
            compute_concordance_dynamics(z_traces[f'conc_{bn}'])
            for bn in ['theta', 'alpha', 'beta']
        ], axis=0)
    else:
        print(f"    No EEG data")
    print(f"  EEG: {time.time() - t0:.1f}s")

    # ── [3/12] BL expression coupling (same as V8.2) ──────────────────
    print(f"\n[3/12] BL expression...", flush=True)
    t0 = time.time()
    landmarks = session_data['landmarks']
    for seg_name, t0_seg, t1_seg in segments:
        p1_bl, p2_bl, bl_dur = extract_bl_segment(landmarks, t0_seg, t1_seg)
        if p1_bl is None or bl_dur < 5:
            continue
        T_bl = p1_bl.shape[0]
        bl_times = np.linspace(t0_seg, t1_seg, T_bl)
        try:
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
        except Exception as e:
            print(f"    BL error ({seg_name}): {e}")
    print(f"  BL: {time.time() - t0:.1f}s")

    # ── [4/12] BL activity concordance ───────────────────────────────
    print(f"\n[4/12] BL activity concordance...", flush=True)
    z_traces['bl_activity_conc'] = compute_bl_activity_concordance(cached, t_common, lsl_offset)

    # ── [5/12] ECG + Resp ────────────────────────────────────────────
    print(f"\n[5/12] ECG + Respiratory...", flush=True)
    t0 = time.time()
    has_raw_ecg = ('p1_ecg' in cached and 'p2_ecg' in cached
                   and len(cached.get('p1_ecg', [])) > 1000)
    if has_raw_ecg:
        z_traces['ecg_lf'], z_traces['ecg_hf'] = ecg_hilbert_coupling(
            cached['p1_ecg'], cached['p1_ecg_ts'],
            cached['p2_ecg'], cached['p2_ecg_ts'],
            t_common, lsl_offset, srate=ECG_SRATE)
        p1_resp = extract_respiratory_one(cached['p1_ecg'], cached['p1_ecg_ts'], srate=ECG_SRATE)
        p2_resp = extract_respiratory_one(cached['p2_ecg'], cached['p2_ecg_ts'], srate=ECG_SRATE)
        z_traces['resp'] = respiratory_phase_coherence(p1_resp, p2_resp, t_common, lsl_offset)
    print(f"  ECG+Resp: {time.time() - t0:.1f}s")

    # ── [6/12] Pose velocity coupling ────────────────────────────────
    print(f"\n[6/12] Pose velocity...", flush=True)
    z_traces['pose'] = pose_velocity_coupling(cached, t_common, lsl_offset)

    # ── [7/12] LZ complexity (V10 NEW) ───────────────────────────────
    print(f"\n[7/12] LZ complexity...", flush=True)
    t0 = time.time()
    lz_features = compute_lz_features(cached, t_common, lsl_offset, asym_sign=asym_sign)
    for k, v in lz_features.items():
        z_traces[k] = v
    for bn in ['theta', 'alpha']:
        print(f"    LZ conc {bn}: mean={z_traces[f'lz_conc_{bn}'].mean():+.3f}, "
              f"asym={z_traces[f'lz_asym_{bn}'].mean():+.3f}")
    print(f"  LZ: {time.time() - t0:.1f}s")

    # ── [8/12] Graph metrics from base 18D (V10 NEW) ─────────────────
    print(f"\n[8/12] Graph modularity + centrality...", flush=True)
    t0 = time.time()
    # Post-dyn-collapse: "base 18D" is now 16D (V10 first 16 channels).
    # V82_KEYS still references pre-collapse dyn_theta/alpha/beta which aren't in z_traces.
    BASE_16_KEYS = V10_MODALITY_KEYS[:16]
    z_18_raw = np.column_stack([z_traces[k] for k in BASE_16_KEYS])
    graph_traces, windowed, mod_result = compute_graph_features(z_18_raw, t_common)
    for k, v in graph_traces.items():
        z_traces[k] = v
    print(f"    Modularity: mean={z_traces['graph_modularity'].mean():.3f}")
    print(f"  Graph: {time.time() - t0:.1f}s")

    # ── [9/12] Prewhiten + standardize all 24D ───────────────────────
    print(f"\n[9/12] Prewhitening + standardization (23D)...", flush=True)
    z_matrix_raw = np.column_stack([z_traces[k] for k in V10_MODALITY_KEYS])
    z_matrix, pw_diag = prewhiten_and_standardize(z_matrix_raw, V10_MODALITY_KEYS)

    # ── [10/12] Observation mask ─────────────────────────────────────
    print(f"\n[10/12] Observation mask...", flush=True)
    obs_mask = np.ones((N_common, len(V10_MODALITY_KEYS)), dtype=bool)
    for d, key in enumerate(V10_MODALITY_KEYS):
        if np.abs(z_matrix_raw[:, d]).max() < 1e-8:
            obs_mask[:, d] = False

    # BL face validity
    bl_idx = [V10_MODALITY_KEYS.index(k) for k in ['bl_expr', 'bl_activity_conc']]
    pose_idx = V10_MODALITY_KEYS.index('pose')

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

    has_pose_valid = ('p1_pose_features_valid' in cached and 'p2_pose_features_valid' in cached
                      and 'p1_pose_features_ts' in cached)
    if has_pose_valid:
        p1_pv = _interp_valid(cached['p1_pose_features_valid'],
                              cached['p1_pose_features_ts'], t_common, lsl_offset)
        p2_pv = _interp_valid(cached['p2_pose_features_valid'],
                              cached['p2_pose_features_ts'], t_common, lsl_offset)
        pose_valid = p1_pv & p2_pv
        obs_mask[:, pose_idx] &= pose_valid

    print(f"\n  {'Modality':>24s} | rho_raw | rho_pw | pw | std")
    print(f"  " + "-" * 60)
    for key in V10_MODALITY_KEYS:
        print(f"  {key:>24s} | {pw_diag['rho_before'].get(key, 0):.3f}   | "
              f"{pw_diag['rho_after'].get(key, 0):.3f}  | "
              f"{'Y' if pw_diag['prewhitened'].get(key, False) else ' ':1s}  | "
              f"{pw_diag['std_after'].get(key, 0):.3f}")

    # ── [11/12] Transition covariates (5D) ───────────────────────────
    print(f"\n[11/12] Transition covariates (5D)...", flush=True)
    U = build_transition_covariates(z_18_raw, t_common, windowed, mod_result)
    for i, key in enumerate(V10_COVARIATE_KEYS):
        print(f"    {key}: mean={U[:, i].mean():.3f}, std={U[:, i].std():.3f}")

    # ── [12/12] Save + visualize ─────────────────────────────────────
    print(f"\n[12/12] Save + visualize...", flush=True)

    save_dict = {
        't_common': t_common,
        'obs_mask': obs_mask,
        'U_covariates': U,
    }
    for i, key in enumerate(V10_MODALITY_KEYS):
        save_dict[f'z_{key}'] = z_matrix[:, i]
        save_dict[f'z_raw_{key}'] = z_matrix_raw[:, i]
    for i, key in enumerate(V10_COVARIATE_KEYS):
        save_dict[f'u_{key}'] = U[:, i]

    np.savez_compressed(os.path.join(out_dir, 'scaffold_v10_ztimecourses.npz'), **save_dict)

    results = {
        'session': session_name, 'version': 'v10',
        'n_timepoints': int(N_common),
        'duration_s': float(session_end_lsl - session_start_lsl),
        'fs_out': FS_OUT,
        'modality_keys': V10_MODALITY_KEYS,
        'covariate_keys': V10_COVARIATE_KEYS,
        'n_obs_dims': len(V10_MODALITY_KEYS),
        'n_cov_dims': len(V10_COVARIATE_KEYS),
        'segments': [(s, float(t0s), float(t1s)) for s, t0s, t1s in segments],
        'prewhitening': {
            'threshold': PREWHITEN_RHO_THRESH,
            'rho_before': pw_diag['rho_before'],
            'rho_after': pw_diag['rho_after'],
            'prewhitened': pw_diag['prewhitened'],
        },
    }
    with open(os.path.join(out_dir, 'scaffold_v10_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Timeline plot
    n_ch = len(V10_MODALITY_KEYS)
    fig, axes = plt.subplots(n_ch, 1, figsize=(28, 1.8 * n_ch), sharex=True)
    for ax in axes:
        for seg_name, t0s, t1s in segments:
            ax.axvspan(t0s, t1s, alpha=0.2,
                       color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)
    for seg_name, t0s, t1s in segments:
        axes[0].text((t0s + t1s) / 2, 1.08, seg_name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=7, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    for idx, (key, name, color) in enumerate(
            zip(V10_MODALITY_KEYS, V10_MODALITY_NAMES, V10_MODALITY_COLORS)):
        ax = axes[idx]
        ax.plot(t_common, z_matrix[:, idx], color=color, linewidth=0.5, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
        rho_b = pw_diag['rho_before'].get(key, 0)
        rho_a = pw_diag['rho_after'].get(key, 0)
        pw = 'PW' if pw_diag['prewhitened'].get(key, False) else ''
        ax.text(0.01, 0.92, f'rho={rho_b:.2f}->{rho_a:.2f} {pw}',
                transform=ax.transAxes, fontsize=6, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_ylabel(name, fontsize=6, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=5)
    axes[-1].set_xlabel('LSL time (s)', fontsize=8)

    fig.suptitle(f'{session_name} -- V10 Scaffold (23D)', fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0.1, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'scaffold_v10_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    total = time.time() - t_wall
    print(f"\n  V10 scaffold complete: {total:.0f}s")
    return results


# ══════════════════════════════════════════════════════════════════════
#  SEMI-SYNTHETIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_from_raw_v10(raw_data, t_common, segments=None, label='semisynthetic'):
    """Process pre-injected raw arrays through full V10 pipeline.

    Calls V8.2's run_from_raw for base 18D, then adds LZ + graph.

    Args:
        raw_data: dict with p1_eeg, p2_eeg, p1_blendshapes, etc.
        t_common: (N,) 2 Hz time grid.
        segments: list of (name, t_start, t_end) tuples.
        label: identifier string.

    Returns:
        z_matrix: (N, 24) prewhitened/standardized.
        z_matrix_raw: (N, 24) raw.
        obs_mask: (N, 24) boolean.
        pw_diag: prewhitening diagnostics.
        U: (N, 5) transition covariates.
    """
    from scripts._run_scaffold_v82 import run_from_raw as run_from_raw_v82

    # V8.2 base 18D
    z_18, z_18_raw, mask_18, pw_diag_18, extras = run_from_raw_v82(
        raw_data, t_common, segments=segments, label=label)

    N = len(t_common)
    z_traces = {}
    for i, k in enumerate(V82_KEYS):
        z_traces[k] = z_18_raw[:, i]

    # LZ complexity from raw EEG
    lz_traces = {k: np.zeros(N, dtype=np.float32)
                 for k in ['lz_conc_theta', 'lz_conc_alpha',
                            'lz_asym_theta', 'lz_asym_alpha']}
    if 'p1_eeg' in raw_data and 'p2_eeg' in raw_data:
        p1_ts = raw_data.get('p1_eeg_ts', np.linspace(t_common[0], t_common[-1], len(raw_data['p1_eeg'])))
        p2_ts = raw_data.get('p2_eeg_ts', p1_ts)
        fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0

        lz_p1 = extract_lz_timecourse(raw_data['p1_eeg'], fs_eeg, roi_channels=FRONTAL_ROI)
        lz_p2 = extract_lz_timecourse(raw_data['p2_eeg'], fs_eeg, roi_channels=FRONTAL_ROI)

        t_lz_p1 = lz_p1['t_out'] + p1_ts[0]
        t_lz_p2 = lz_p2['t_out'] + p2_ts[0]

        for band_name in ['theta', 'alpha']:
            p1_lz_2hz = np.interp(t_common, t_lz_p1, lz_p1[band_name])
            p2_lz_2hz = np.interp(t_common, t_lz_p2, lz_p2[band_name])
            conc, asym = lz_concordance_asymmetry(p1_lz_2hz, p2_lz_2hz, asym_sign=1.0)
            lz_traces[f'lz_conc_{band_name}'] = conc.astype(np.float32)
            lz_traces[f'lz_asym_{band_name}'] = asym.astype(np.float32)

    # Graph metrics from base 18D raw
    graph_traces, windowed, mod_result = compute_graph_features(z_18_raw, t_common)

    # Assemble 23D raw
    D_v10 = len(V10_MODALITY_KEYS)
    z_v10_raw = np.column_stack([
        z_18_raw,
        lz_traces['lz_conc_theta'], lz_traces['lz_conc_alpha'],
        lz_traces['lz_asym_theta'], lz_traces['lz_asym_alpha'],
        graph_traces['graph_modularity'],
    ])

    # Prewhiten + standardize full 23D
    z_v10, pw_diag = prewhiten_and_standardize(z_v10_raw, V10_MODALITY_KEYS)

    # Obs mask (extend V8.2 mask with always-observed for new channels)
    mask_v10 = np.ones((N, D_v10), dtype=bool)
    mask_v10[:, :18] = mask_18
    for d in range(18, D_v10):
        if np.abs(z_v10_raw[:, d]).max() < 1e-8:
            mask_v10[:, d] = False

    # Transition covariates
    U = build_transition_covariates(z_18_raw, t_common, windowed, mod_result)

    return z_v10, z_v10_raw, mask_v10, pw_diag, U


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CADENCE V10 Scaffold')
    parser.add_argument('--session', type=str, default='y_06')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    config = load_config()

    if args.all:
        from joblib import Parallel, delayed
        cached_sessions = discover_cached_sessions(config['session_cache'])
        session_names = [n for n, _ in cached_sessions]
        # Skip sessions that already have V10 results
        todo = [n for n in session_names
                if not os.path.exists(f'results/v10/{n}/scaffold_v10_ztimecourses.npz')]
        if todo:
            # Resource-disciplined fan-out per audit P1.1. V10 doesn't fire
            # GPU TE surrogates so per-worker RAM is lower than V11 (~2 GB).
            # Use threading backend to share imports + avoid the loky+torch
            # DLL-load issue (defensive: V10 may import torch transitively).
            from cadence.io.resources import (
                limit_blas_threads, log_resources, pick_n_jobs,
            )
            n_jobs = pick_n_jobs(per_worker_ram_gb=2.0, requested=8,
                                  max_jobs_hard_cap=len(todo))
            log_resources(prefix='[v10 --all] pre-fan-out: ')
            print(f"Running V10 scaffold on {len(todo)} remaining sessions "
                  f"(n_jobs={n_jobs}, threading; per-worker RAM ~= 2 GB)")

            def _bounded_run_session(name, cfg):
                with limit_blas_threads(1):
                    return run_session(name, cfg)

            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_bounded_run_session)(name, config) for name in todo
            )
        else:
            print("All sessions already have V10 results.")
            results = []
        valid = [r for r in results if r is not None]
        print(f"\nCompleted: {len(valid)}/{len(todo)} new sessions "
              f"({len(session_names) - len(todo)} already done)")
    else:
        run_session(args.session, config)
