"""CADENCE V11 Scaffold: 28D observation vector + 7D transition covariates.

Extends V10 (23D) with:
  - TE concordance theta/alpha (2ch obs): bidirectional information flow
  - Burst coincidence theta/alpha/beta (3ch obs): inter-brain burst co-occurrence
  - TE asymmetry theta/alpha (2ch cov): directionality as transition modulator

Key design: TE decomposed into concordance (observation) + asymmetry (covariate).
  te_conc = (z_T→P + z_P→T) / 2   "how much bidirectional flow?"  → state property
  te_asym = z_T→P - z_P→T          "who leads?"                    → transition modulator

Reuses all V10 extraction functions via import (DRY).
Single extract_burst_grids() call shared between TE and coincidence.

Usage:
    python scripts/_run_scaffold_v11.py                    # Single session (y_06)
    python scripts/_run_scaffold_v11.py --session y_17     # Specific session
    python scripts/_run_scaffold_v11.py --all              # All sessions
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch must import before numpy on Windows (torch 2.10 + numpy 2.4 DLL-load order bug: shm.dll)
import torch  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gf1d
from sklearn.decomposition import PCA

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

# ── V10 imports (DRY — reuse LZ + graph functions) ──────────────────
from scripts._run_scaffold_v10 import (
    compute_lz_features, compute_graph_features, build_transition_covariates,
    RAW_DIR, Z_THRESH, PREWHITEN_RHO_THRESH,
    GRAPH_WINDOW_S, GRAPH_STRIDE_S, ZSLOW_SIGMA_S,
)

from cadence.config import load_config
from cadence.data import (
    discover_cached_sessions, load_session_from_cache, apply_modality_exclusions
)
from cadence.significance.lz_complexity import extract_lz_timecourse, lz_concordance_asymmetry
from cadence.constants import (
    V11_MODALITY_KEYS, V11_MODALITY_NAMES, V11_MODALITY_COLORS,
    V11_COVARIATE_KEYS, V10_COVARIATE_KEYS, V10_MODALITY_KEYS,
    FRONTAL_ROI,
)
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    AFFECT_AUS, BAND_EXPRESSION,
)

# ── V11 new imports ─────────────────────────────────────────────────
from cadence.significance.fast_cycles import extract_burst_grids, EEG_BANDS
from cadence.significance.burst_coincidence import compute_burst_coincidence
from cadence.significance.directed_burst_coupling import gpu_sliding_te_surrogates


# ── Burst-rate gating for TE reliability ────────────────────────────
# Minimum rolling burst rate (60s window) for TE to be estimable.
# Empirically validated: eliminates burst-rate/TE-episode confound
# (rho: +0.25 -> +0.03) while retaining >=67% of all conditions.
# See scripts/_analyze_burst_rate_thresholds.py for derivation.
MIN_BURST_RATE = 0.05
BURST_RATE_WINDOW = 120  # 60s at 2 Hz, matching TE sliding window


# ══════════════════════════════════════════════════════════════════════
#  V11-SPECIFIC EXTRACTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def compute_burst_features(cached, t_common, lsl_offset, asym_sign=1.0,
                           n_surrogates=200, seed=42):
    """Extract TE concordance/asymmetry + burst coincidence for V11 scaffold.

    Extracts burst grids ONCE (GPU bandpass + cycle detection), then passes
    the shared grids to both TE and coincidence functions.

    TE decomposition:
      te_conc  = (z_P1→P2 + z_P2→P1) / 2   → observation (coupling intensity)
      te_asym  =  z_T→P  - z_P→T            → covariate (directionality)

    Args:
        cached: session cache dict (p1_eeg, p2_eeg, p1_eeg_ts).
        t_common: (N,) scaffold time grid in LSL seconds.
        lsl_offset: LSL time offset for EEG alignment.
        asym_sign: +1 if P1=therapist, -1 if P1=patient.
        n_surrogates: surrogates for both TE and coincidence z-scoring.
        seed: random seed.

    Returns:
        obs: dict with te_conc_theta, te_conc_alpha,
             burst_coinc_theta, burst_coinc_alpha, burst_coinc_beta.
        cov: dict with te_asym_theta, te_asym_alpha.
        rates: dict with p1_burst_rate_theta, p2_burst_rate_theta,
               p1_burst_rate_alpha, p2_burst_rate_alpha,
               burst_gate_theta, burst_gate_alpha.
        All (N,) arrays aligned to t_common.
    """
    import torch

    N = len(t_common)
    obs_keys = ['te_conc_theta', 'te_conc_alpha',
                'burst_coinc_theta', 'burst_coinc_alpha', 'burst_coinc_beta']
    cov_keys = ['te_asym_theta', 'te_asym_alpha']
    obs = {k: np.zeros(N, dtype=np.float32) for k in obs_keys}
    cov = {k: np.zeros(N, dtype=np.float32) for k in cov_keys}

    # Per-participant rolling burst rates + gate
    rates = {
        'p1_burst_rate_theta': np.zeros(N, dtype=np.float32),
        'p2_burst_rate_theta': np.zeros(N, dtype=np.float32),
        'p1_burst_rate_alpha': np.zeros(N, dtype=np.float32),
        'p2_burst_rate_alpha': np.zeros(N, dtype=np.float32),
        'burst_gate_theta': np.zeros(N, dtype=bool),
        'burst_gate_alpha': np.zeros(N, dtype=bool),
    }

    if ('p1_eeg' not in cached or 'p2_eeg' not in cached
            or 'p1_eeg_ts' not in cached):
        return obs, cov, rates

    # Prepare EEG
    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg = p1_eeg[:, :n_ch]
    p2_eeg = p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg = p1_eeg[:mlen]
    p2_eeg = p2_eeg[:mlen]

    fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0

    # EEG-local time grid at 2 Hz
    dur = mlen / fs_eeg
    t_grid_local = np.arange(0, dur, 1.0 / FS_OUT)
    t_grid_lsl = t_grid_local + (p1_ts[0] + lsl_offset)

    # ── Single burst grid extraction (shared) ───────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                 bands=EEG_BANDS, device=device)

    # ── Per-band: TE concordance/asymmetry + burst coincidence ──────
    for band_name, bg in grids.items():
        valid = bg['valid_channels']
        if len(valid) < 3:
            continue

        # Channel filter: both participants >1% burst rate
        p1_ch_rates = bg['p1_burst'].mean(axis=1)
        p2_ch_rates = bg['p2_burst'].mean(axis=1)
        ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)
        if ch_mask.sum() < 3:
            continue

        p1b = bg['p1_burst'][ch_mask]
        p2b = bg['p2_burst'][ch_mask]

        # ── Per-participant rolling burst rate (theta + alpha) ──────
        if band_name in ('theta', 'alpha'):
            C_valid, N_local = p1b.shape
            # Channel-averaged rolling mean via cumsum
            p1_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
            p2_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
            for c in range(C_valid):
                cs1 = np.cumsum(p1b[c].astype(np.float32))
                cs1 = np.insert(cs1, 0, 0.0)
                cs2 = np.cumsum(p2b[c].astype(np.float32))
                cs2 = np.insert(cs2, 0, 0.0)
                for t in range(N_local):
                    t0 = max(0, t - BURST_RATE_WINDOW // 2)
                    t1 = min(N_local, t + BURST_RATE_WINDOW // 2)
                    span = t1 - t0
                    p1_rates_ch[c, t] = (cs1[t1] - cs1[t0]) / span
                    p2_rates_ch[c, t] = (cs2[t1] - cs2[t0]) / span

            p1_rate = np.interp(t_common, t_grid_lsl, p1_rates_ch.mean(axis=0),
                                left=0, right=0).astype(np.float32)
            p2_rate = np.interp(t_common, t_grid_lsl, p2_rates_ch.mean(axis=0),
                                left=0, right=0).astype(np.float32)
            rates[f'p1_burst_rate_{band_name}'] = p1_rate
            rates[f'p2_burst_rate_{band_name}'] = p2_rate
            rates[f'burst_gate_{band_name}'] = (
                (p1_rate >= MIN_BURST_RATE) & (p2_rate >= MIN_BURST_RATE))

        # ── Burst coincidence (all 3 bands) ─────────────────────────
        coinc_z_local, _ = compute_burst_coincidence(
            p1b, p2b, tau_samples=1,
            n_surrogates=n_surrogates, seed=seed)
        coinc_z = np.interp(t_common, t_grid_lsl, coinc_z_local,
                             left=0, right=0).astype(np.float32)
        obs[f'burst_coinc_{band_name}'] = coinc_z

        # ── TE concordance + asymmetry (theta + alpha only) ─────────
        if band_name in ('theta', 'alpha'):
            gpu_te = gpu_sliding_te_surrogates(
                p1b, p2b, window_samples=120, stride_samples=2,
                k=3, n_surrogates=n_surrogates, seed=seed, device=device)

            if len(gpu_te['t_centers']) > 0:
                # t_centers are sample indices into burst grid (2 Hz)
                tc_local = gpu_te['t_centers'].astype(np.float64) / FS_OUT
                tc_lsl = tc_local + (p1_ts[0] + lsl_offset)

                # Concordance: bidirectional flow (observation)
                te_conc_local = (gpu_te['te_p1p2_z'] + gpu_te['te_p2p1_z']) / 2.0
                obs[f'te_conc_{band_name}'] = np.interp(
                    t_common, tc_lsl, te_conc_local,
                    left=0, right=0).astype(np.float32)

                # Asymmetry: who leads (covariate, sign-corrected)
                te_asym_local = gpu_te['te_asym_z']
                cov[f'te_asym_{band_name}'] = np.interp(
                    t_common, tc_lsl, te_asym_local,
                    left=0, right=0).astype(np.float32) * asym_sign

    return obs, cov, rates


# ══════════════════════════════════════════════════════════════════════
#  SESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name, config, raw_dir=RAW_DIR):
    """Run V11 scaffold: 28D observation + 5D covariates."""
    print(f"\n{'='*70}")
    print(f"  V11 Scaffold -- {session_name}")
    print(f"{'='*70}")

    t_wall = time.time()
    out_dir = f'results/v11/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # ── [1/13] Load session ──────────────────────────────────────────
    print(f"\n[1/13] Loading {session_name}...", flush=True)

    xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  WARNING: No XDF found for {session_name}, skipping")
        return None
    session_data = load_xdf_session(xdf_files[0])
    markers = session_data['markers']
    p1_role = session_data.get('p1_role', 'unknown')
    asym_sign = -1.0 if p1_role == 'patient' else 1.0
    print(f"  P1={p1_role} -> asym_sign={asym_sign:+.0f}")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        print(f"  WARNING: No cache for {session_name}, skipping")
        return None
    cached = load_session_from_cache(cache_matches[0], config)
    apply_modality_exclusions(cached, session_name)

    # LSL offset
    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    # Segments from markers
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

    z_traces = {k: np.zeros(N_common, dtype=np.float32) for k in V11_MODALITY_KEYS}

    # ── [2/13] EEG ImCoh + Concordance + Dynamics + Asymmetry ────────
    print(f"\n[2/13] EEG features...", flush=True)
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

        # dyn_theta/alpha/beta are r=0.99 collinear (all EWMAD of concordance at same slow timescale)
        # → collapsed to single dyn_mean channel
        z_traces['dyn_mean'] = np.mean([
            compute_concordance_dynamics(z_traces[f'conc_{bn}'])
            for bn in ['theta', 'alpha', 'beta']
        ], axis=0)
    else:
        print(f"    No EEG data")
    print(f"  EEG: {time.time() - t0:.1f}s")

    # ── [3/13] BL expression coupling ────────────────────────────────
    print(f"\n[3/13] BL expression...", flush=True)
    t0 = time.time()
    landmarks = session_data['landmarks']
    bl_expr_valid = np.zeros(N_common, dtype=bool)  # track valid segments for data_valid
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
                    bl_expr_valid[seg_mask] = True
        except Exception as e:
            print(f"    BL error ({seg_name}): {e}")
    print(f"  BL: {time.time() - t0:.1f}s")

    # ── [4/13] BL activity concordance ───────────────────────────────
    print(f"\n[4/13] BL activity concordance...", flush=True)
    z_traces['bl_activity_conc'] = compute_bl_activity_concordance(cached, t_common, lsl_offset)

    # ── [5/13] ECG + Resp ────────────────────────────────────────────
    print(f"\n[5/13] ECG + Respiratory...", flush=True)
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

    # ── [6/13] Pose velocity coupling ────────────────────────────────
    print(f"\n[6/13] Pose velocity...", flush=True)
    z_traces['pose'] = pose_velocity_coupling(cached, t_common, lsl_offset)

    # ── [7/13] LZ complexity (V10) ───────────────────────────────────
    print(f"\n[7/13] LZ complexity...", flush=True)
    t0 = time.time()
    lz_features = compute_lz_features(cached, t_common, lsl_offset, asym_sign=asym_sign)
    for k, v in lz_features.items():
        z_traces[k] = v
    for bn in ['theta', 'alpha']:
        print(f"    LZ conc {bn}: mean={z_traces[f'lz_conc_{bn}'].mean():+.3f}, "
              f"asym={z_traces[f'lz_asym_{bn}'].mean():+.3f}")
    print(f"  LZ: {time.time() - t0:.1f}s")

    # ── [8/13] Graph metrics from base 16D (V10, post-dyn-collapse) ─
    # Historically "base 18D"; after dyn_theta/alpha/beta collapse to dyn_mean
    # it's 16D. V82_KEYS still references the pre-collapse 18D layout and will
    # KeyError on z_traces['dyn_theta'].
    print(f"\n[8/13] Graph modularity...", flush=True)
    t0 = time.time()
    BASE_16_KEYS = V10_MODALITY_KEYS[:16]
    z_18_raw = np.column_stack([z_traces[k] for k in BASE_16_KEYS])
    graph_traces, windowed, mod_result = compute_graph_features(z_18_raw, t_common)
    for k, v in graph_traces.items():
        z_traces[k] = v
    print(f"    Modularity: mean={z_traces['graph_modularity'].mean():.3f}")
    print(f"  Graph: {time.time() - t0:.1f}s")

    # ── [9/13] Burst features: TE conc/asym + coincidence (V11 NEW) ──
    print(f"\n[9/13] Burst features (TE + coincidence)...", flush=True)
    t0 = time.time()
    burst_obs, burst_cov, burst_rates = compute_burst_features(
        cached, t_common, lsl_offset, asym_sign=asym_sign,
        n_surrogates=200, seed=42)
    for k, v in burst_obs.items():
        z_traces[k] = v
    for bn in ['theta', 'alpha']:
        print(f"    TE conc {bn}: mean={z_traces[f'te_conc_{bn}'].mean():+.3f}, "
              f"std={z_traces[f'te_conc_{bn}'].std():.3f}")
    for bn in ['theta', 'alpha', 'beta']:
        print(f"    Burst coinc {bn}: mean={z_traces[f'burst_coinc_{bn}'].mean():+.3f}, "
              f"std={z_traces[f'burst_coinc_{bn}'].std():.3f}")
    for bn in ['theta', 'alpha']:
        print(f"    TE asym {bn} (cov): mean={burst_cov[f'te_asym_{bn}'].mean():+.3f}, "
              f"std={burst_cov[f'te_asym_{bn}'].std():.3f}")
    for bn in ['theta', 'alpha']:
        gate = burst_rates[f'burst_gate_{bn}']
        p1r = burst_rates[f'p1_burst_rate_{bn}']
        p2r = burst_rates[f'p2_burst_rate_{bn}']
        print(f"    Burst rate {bn}: P1 mean={p1r.mean():.3f}, P2 mean={p2r.mean():.3f}, "
              f"gate={gate.mean():.1%} (min_rate={MIN_BURST_RATE})")
    print(f"  Burst features: {time.time() - t0:.1f}s")

    # ── [10/13] Build data_valid mask + prewhiten ───────────────────
    print(f"\n[10/13] Data validity mask + prewhitening (28D)...", flush=True)

    D_obs = len(V11_MODALITY_KEYS)
    data_valid = np.ones((N_common, D_obs), dtype=bool)

    # EEG time range (12 EEG + 4 LZ + 5 TE/burst channels)
    if _has_eeg:
        eeg_start = p1_ts[0] + lsl_offset
        eeg_end = eeg_start + mlen / fs_eeg
        eeg_valid = (t_common >= eeg_start) & (t_common <= eeg_end)
        eeg_keys = (['imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
                      'conc_theta', 'conc_alpha', 'conc_beta',
                      'dyn_mean',
                      'asym_theta', 'asym_alpha', 'asym_beta',
                      'lz_conc_theta', 'lz_conc_alpha',
                      'lz_asym_theta', 'lz_asym_alpha',
                      'te_conc_theta', 'te_conc_alpha',
                      'burst_coinc_theta', 'burst_coinc_alpha', 'burst_coinc_beta'])
        for k in eeg_keys:
            data_valid[:, V11_MODALITY_KEYS.index(k)] = eeg_valid
    else:
        # No EEG → all EEG-derived channels invalid
        eeg_keys = [k for k in V11_MODALITY_KEYS if k not in
                     ['bl_expr', 'bl_activity_conc', 'ecg_lf', 'ecg_hf',
                      'resp', 'pose', 'graph_modularity']]
        for k in eeg_keys:
            data_valid[:, V11_MODALITY_KEYS.index(k)] = False

    # BL expression: per-segment only (tracked during extraction)
    data_valid[:, V11_MODALITY_KEYS.index('bl_expr')] = bl_expr_valid

    # BL activity concordance: blendshape timestamp range
    if 'p1_blendshapes_ts' in cached and 'p2_blendshapes_ts' in cached:
        bl_start = max(cached['p1_blendshapes_ts'][0],
                       cached['p2_blendshapes_ts'][0]) + lsl_offset
        bl_end = min(cached['p1_blendshapes_ts'][-1],
                     cached['p2_blendshapes_ts'][-1]) + lsl_offset
        bl_act_valid = (t_common >= bl_start) & (t_common <= bl_end)
        data_valid[:, V11_MODALITY_KEYS.index('bl_activity_conc')] = bl_act_valid

    # ECG + Resp: ECG timestamp range
    if has_raw_ecg and 'p1_ecg_ts' in cached and 'p2_ecg_ts' in cached:
        ecg_start = max(cached['p1_ecg_ts'][0],
                        cached['p2_ecg_ts'][0]) + lsl_offset
        ecg_end = min(cached['p1_ecg_ts'][-1],
                      cached['p2_ecg_ts'][-1]) + lsl_offset
        ecg_valid = (t_common >= ecg_start) & (t_common <= ecg_end)
        for k in ['ecg_lf', 'ecg_hf', 'resp']:
            data_valid[:, V11_MODALITY_KEYS.index(k)] = ecg_valid

    # Pose: pose timestamp range
    if ('p1_pose_features_ts' in cached and 'p2_pose_features_ts' in cached
            and len(cached['p1_pose_features_ts']) > 0
            and len(cached['p2_pose_features_ts']) > 0):
        pose_start = max(cached['p1_pose_features_ts'][0],
                         cached['p2_pose_features_ts'][0]) + lsl_offset
        pose_end = min(cached['p1_pose_features_ts'][-1],
                       cached['p2_pose_features_ts'][-1]) + lsl_offset
        pose_time_valid = (t_common >= pose_start) & (t_common <= pose_end)
        data_valid[:, V11_MODALITY_KEYS.index('pose')] = pose_time_valid

    # Graph modularity: trim half-window edges
    graph_idx = V11_MODALITY_KEYS.index('graph_modularity')
    if N_common >= int(GRAPH_WINDOW_S * FS_OUT):
        t_first = t_common[0] + GRAPH_WINDOW_S / 2.0
        t_last = t_common[-1] - GRAPH_WINDOW_S / 2.0
        if t_last > t_first:
            data_valid[:, graph_idx] = (t_common >= t_first) & (t_common <= t_last)

    # Zero out whole-channel zeros
    z_matrix_raw = np.column_stack([z_traces[k] for k in V11_MODALITY_KEYS])
    for d, key in enumerate(V11_MODALITY_KEYS):
        if np.abs(z_matrix_raw[:, d]).max() < 1e-8:
            data_valid[:, d] = False

    n_valid = data_valid.sum(axis=0)
    print(f"  Data valid fractions:")
    for d, key in enumerate(V11_MODALITY_KEYS):
        print(f"    {key:>24s}: {n_valid[d]/N_common:.1%}")

    # Prewhiten with data_valid mask
    z_matrix, pw_diag = prewhiten_and_standardize(
        z_matrix_raw, V11_MODALITY_KEYS, valid_mask=data_valid)

    # ── [11/13] Observation mask ─────────────────────────────────────
    print(f"\n[11/13] Observation mask...", flush=True)
    obs_mask = data_valid.copy()

    # BL face tracker validity
    bl_idx = [V11_MODALITY_KEYS.index(k) for k in ['bl_expr', 'bl_activity_conc']]
    pose_idx = V11_MODALITY_KEYS.index('pose')

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

    # Burst-rate gate: mask TE observation channels at low-rate timepoints
    for te_key in ['te_conc_theta', 'te_conc_alpha']:
        band = te_key.split('_')[-1]
        te_idx = V11_MODALITY_KEYS.index(te_key)
        gate = burst_rates[f'burst_gate_{band}']
        n_gated = (~gate).sum()
        obs_mask[:, te_idx] &= gate
        print(f"    TE gate {band}: {n_gated} timepoints masked "
              f"({n_gated/N_common:.1%} of session)")

    print(f"\n  {'Modality':>24s} | rho_raw | rho_pw | pw | std | valid%")
    print(f"  " + "-" * 70)
    for d, key in enumerate(V11_MODALITY_KEYS):
        print(f"  {key:>24s} | {pw_diag['rho_before'].get(key, 0):.3f}   | "
              f"{pw_diag['rho_after'].get(key, 0):.3f}  | "
              f"{'Y' if pw_diag['prewhitened'].get(key, False) else ' ':1s}  | "
              f"{pw_diag['std_after'].get(key, 0):.3f} | "
              f"{obs_mask[:, d].mean():.1%}")

    # ── [12/13] Transition covariates (7D = 5 V10 + 2 TE asymmetry) ──
    print(f"\n[12/13] Transition covariates (7D)...", flush=True)
    U_v10 = build_transition_covariates(z_18_raw, t_common, windowed, mod_result)
    # Append TE asymmetry covariates (prewhiten with EEG valid mask)
    te_asym_covs = np.column_stack([
        burst_cov['te_asym_theta'],
        burst_cov['te_asym_alpha'],
    ])
    te_asym_valid = np.column_stack([
        data_valid[:, V11_MODALITY_KEYS.index('te_conc_theta')],
        data_valid[:, V11_MODALITY_KEYS.index('te_conc_alpha')],
    ])
    _pw_fn = prewhiten_and_standardize
    te_asym_pw, _ = _pw_fn(te_asym_covs, ['te_asym_theta', 'te_asym_alpha'],
                           valid_mask=te_asym_valid)
    # Zero out AFTER prewhitening — gating before corrupts AR(1)
    te_asym_pw[~burst_rates['burst_gate_theta'], 0] = 0.0
    te_asym_pw[~burst_rates['burst_gate_alpha'], 1] = 0.0
    U = np.column_stack([U_v10, te_asym_pw])
    for i, key in enumerate(V11_COVARIATE_KEYS):
        print(f"    {key}: mean={U[:, i].mean():.3f}, std={U[:, i].std():.3f}")

    # ── [13/13] Save + visualize ─────────────────────────────────────
    print(f"\n[13/13] Save + visualize...", flush=True)

    save_dict = {
        't_common': t_common,
        'obs_mask': obs_mask,
        'U_covariates': U,
    }
    for i, key in enumerate(V11_MODALITY_KEYS):
        save_dict[f'z_{key}'] = z_matrix[:, i]
        save_dict[f'z_raw_{key}'] = z_matrix_raw[:, i]
    for i, key in enumerate(V11_COVARIATE_KEYS):
        save_dict[f'u_{key}'] = U[:, i]

    # Burst rate channels for downstream gating
    for rk in ['p1_burst_rate_theta', 'p2_burst_rate_theta',
               'p1_burst_rate_alpha', 'p2_burst_rate_alpha']:
        save_dict[rk] = burst_rates[rk]
    save_dict['burst_gate_theta'] = burst_rates['burst_gate_theta']
    save_dict['burst_gate_alpha'] = burst_rates['burst_gate_alpha']

    np.savez_compressed(os.path.join(out_dir, 'scaffold_v11_ztimecourses.npz'), **save_dict)

    results = {
        'session': session_name, 'version': 'v11',
        'n_timepoints': int(N_common),
        'duration_s': float(session_end_lsl - session_start_lsl),
        'fs_out': FS_OUT,
        'modality_keys': list(V11_MODALITY_KEYS),
        'covariate_keys': list(V11_COVARIATE_KEYS),
        'n_obs_dims': len(V11_MODALITY_KEYS),
        'n_cov_dims': len(V11_COVARIATE_KEYS),
        'segments': [(s, float(t0s), float(t1s)) for s, t0s, t1s in segments],
        'prewhitening': {
            'threshold': PREWHITEN_RHO_THRESH,
            'rho_before': pw_diag['rho_before'],
            'rho_after': pw_diag['rho_after'],
            'prewhitened': pw_diag['prewhitened'],
        },
        'burst_rate_gating': {
            'min_burst_rate': MIN_BURST_RATE,
            'gate_frac_theta': float(burst_rates['burst_gate_theta'].mean()),
            'gate_frac_alpha': float(burst_rates['burst_gate_alpha'].mean()),
        },
    }
    with open(os.path.join(out_dir, 'scaffold_v11_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Timeline plot
    n_ch = len(V11_MODALITY_KEYS)
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
            zip(V11_MODALITY_KEYS, V11_MODALITY_NAMES, V11_MODALITY_COLORS)):
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

    fig.suptitle(f'{session_name} -- V11 Scaffold (28D)', fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0.1, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'scaffold_v11_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    total = time.time() - t_wall
    print(f"\n  V11 scaffold complete: {total:.0f}s")
    return results


# ══════════════════════════════════════════════════════════════════════
#  SEMI-SYNTHETIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_from_raw_v11(raw_data, t_common, segments=None, label='semisynthetic'):
    """Process pre-injected raw arrays through full V11 pipeline.

    Calls V10's run_from_raw for base 23D, then adds burst features.
    TE decomposed: concordance → observations, asymmetry → covariates.

    Args:
        raw_data: dict with p1_eeg, p2_eeg, p1_blendshapes, etc.
        t_common: (N,) 2 Hz time grid.
        segments: list of (name, t_start, t_end) tuples.
        label: identifier string.

    Returns:
        z_matrix: (N, 28) prewhitened/standardized.
        z_matrix_raw: (N, 28) raw.
        obs_mask: (N, 28) boolean.
        pw_diag: prewhitening diagnostics.
        U: (N, 7) transition covariates.
    """
    import torch
    from scripts._run_scaffold_v10 import run_from_raw_v10

    # V10 base 23D
    z_v10, z_v10_raw, mask_v10, pw_diag_v10, U_v10 = run_from_raw_v10(
        raw_data, t_common, segments=segments, label=label)

    N = len(t_common)

    # Burst features from raw EEG
    obs_traces = {k: np.zeros(N, dtype=np.float32)
                  for k in ['te_conc_theta', 'te_conc_alpha',
                            'burst_coinc_theta', 'burst_coinc_alpha',
                            'burst_coinc_beta']}
    cov_traces = {k: np.zeros(N, dtype=np.float32)
                  for k in ['te_asym_theta', 'te_asym_alpha']}

    burst_rates = {
        'burst_gate_theta': np.zeros(N, dtype=bool),
        'burst_gate_alpha': np.zeros(N, dtype=bool),
    }

    if 'p1_eeg' in raw_data and 'p2_eeg' in raw_data:
        p1_eeg = raw_data['p1_eeg'].astype(np.float64)
        p2_eeg = raw_data['p2_eeg'].astype(np.float64)
        p1_ts = raw_data.get('p1_eeg_ts',
                             np.linspace(t_common[0], t_common[-1], len(p1_eeg)))

        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]; p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]; p2_eeg = p2_eeg[:mlen]

        fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0
        dur = mlen / fs_eeg
        t_grid_local = np.arange(0, dur, 1.0 / FS_OUT)
        t_grid_lsl = t_grid_local + p1_ts[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                     bands=EEG_BANDS, device=device)

        for band_name, bg in grids.items():
            p1_ch_rates = bg['p1_burst'].mean(axis=1)
            p2_ch_rates = bg['p2_burst'].mean(axis=1)
            ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)
            if ch_mask.sum() < 3:
                continue

            p1b = bg['p1_burst'][ch_mask]
            p2b = bg['p2_burst'][ch_mask]

            # Rolling burst rates for gate (theta + alpha)
            if band_name in ('theta', 'alpha'):
                C_valid, N_local = p1b.shape
                p1_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
                p2_rates_ch = np.zeros((C_valid, N_local), dtype=np.float32)
                for c in range(C_valid):
                    cs1 = np.cumsum(p1b[c].astype(np.float32))
                    cs1 = np.insert(cs1, 0, 0.0)
                    cs2 = np.cumsum(p2b[c].astype(np.float32))
                    cs2 = np.insert(cs2, 0, 0.0)
                    for t in range(N_local):
                        t0_ = max(0, t - BURST_RATE_WINDOW // 2)
                        t1_ = min(N_local, t + BURST_RATE_WINDOW // 2)
                        span = t1_ - t0_
                        p1_rates_ch[c, t] = (cs1[t1_] - cs1[t0_]) / span
                        p2_rates_ch[c, t] = (cs2[t1_] - cs2[t0_]) / span
                p1_rate = np.interp(t_common, t_grid_lsl,
                                    p1_rates_ch.mean(axis=0), left=0, right=0)
                p2_rate = np.interp(t_common, t_grid_lsl,
                                    p2_rates_ch.mean(axis=0), left=0, right=0)
                burst_rates[f'burst_gate_{band_name}'] = (
                    (p1_rate >= MIN_BURST_RATE) & (p2_rate >= MIN_BURST_RATE))

            # Burst coincidence
            coinc_z_local, _ = compute_burst_coincidence(
                p1b, p2b, tau_samples=1, n_surrogates=200, seed=42)
            obs_traces[f'burst_coinc_{band_name}'] = np.interp(
                t_common, t_grid_lsl, coinc_z_local, left=0, right=0
            ).astype(np.float32)

            # TE concordance + asymmetry (theta + alpha only)
            if band_name in ('theta', 'alpha'):
                gpu_te = gpu_sliding_te_surrogates(
                    p1b, p2b, window_samples=120, stride_samples=2,
                    k=3, n_surrogates=200, seed=42, device=device)
                if len(gpu_te['t_centers']) > 0:
                    tc_local = gpu_te['t_centers'].astype(np.float64) / FS_OUT
                    tc_lsl = tc_local + p1_ts[0]
                    obs_traces[f'te_conc_{band_name}'] = np.interp(
                        t_common, tc_lsl,
                        (gpu_te['te_p1p2_z'] + gpu_te['te_p2p1_z']) / 2.0,
                        left=0, right=0).astype(np.float32)
                    cov_traces[f'te_asym_{band_name}'] = np.interp(
                        t_common, tc_lsl, gpu_te['te_asym_z'],
                        left=0, right=0).astype(np.float32)

    # Assemble 28D raw observations
    z_v11_raw = np.column_stack([
        z_v10_raw,
        obs_traces['te_conc_theta'], obs_traces['te_conc_alpha'],
        obs_traces['burst_coinc_theta'], obs_traces['burst_coinc_alpha'],
        obs_traces['burst_coinc_beta'],
    ])

    # Build data_valid from V10 mask + EEG time range for new channels
    D_v11 = len(V11_MODALITY_KEYS)
    data_valid = np.ones((N, D_v11), dtype=bool)
    data_valid[:, :23] = mask_v10
    if 'p1_eeg' in raw_data and 'p2_eeg' in raw_data:
        eeg_start = p1_ts[0]
        eeg_end = eeg_start + mlen / fs_eeg
        eeg_valid = (t_common >= eeg_start) & (t_common <= eeg_end)
        for d in range(23, D_v11):
            data_valid[:, d] = eeg_valid
    for d in range(D_v11):
        if np.abs(z_v11_raw[:, d]).max() < 1e-8:
            data_valid[:, d] = False

    # Prewhiten + standardize full 28D with data_valid mask
    z_v11, pw_diag = prewhiten_and_standardize(z_v11_raw, V11_MODALITY_KEYS,
                                                valid_mask=data_valid)

    # Obs mask = data_valid + burst gate for TE channels
    mask_v11 = data_valid.copy()
    for te_key in ['te_conc_theta', 'te_conc_alpha']:
        band = te_key.split('_')[-1]
        te_idx = V11_MODALITY_KEYS.index(te_key)
        mask_v11[:, te_idx] &= burst_rates[f'burst_gate_{band}']

    # 7D covariates (5 V10 + 2 TE asymmetry)
    te_asym_raw = np.column_stack([
        cov_traces['te_asym_theta'],
        cov_traces['te_asym_alpha'],
    ])
    te_asym_valid = np.column_stack([
        data_valid[:, V11_MODALITY_KEYS.index('te_conc_theta')],
        data_valid[:, V11_MODALITY_KEYS.index('te_conc_alpha')],
    ])
    te_asym_pw, _ = prewhiten_and_standardize(
        te_asym_raw, ['te_asym_theta', 'te_asym_alpha'], valid_mask=te_asym_valid)
    # Zero out AFTER prewhitening — same as run_session path
    te_asym_pw[~burst_rates['burst_gate_theta'], 0] = 0.0
    te_asym_pw[~burst_rates['burst_gate_alpha'], 1] = 0.0
    U = np.column_stack([U_v10, te_asym_pw])

    return z_v11, z_v11_raw, mask_v11, pw_diag, U


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CADENCE V11 Scaffold')
    parser.add_argument('--session', type=str, default='y_06')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    config = load_config()

    if args.all:
        from joblib import Parallel, delayed
        cached_sessions = discover_cached_sessions(config['session_cache'])
        excluded = set(config.get('excluded_sessions', []))
        session_names = [n for n, _ in cached_sessions if n not in excluded]
        todo = [n for n in session_names
                if not os.path.exists(f'results/v11/{n}/scaffold_v11_ztimecourses.npz')]
        if todo:
            # Resource-disciplined fan-out per audit P0.3 (this is the call-site
            # that OOMed the user on 2026-05-01). run_session peak RAM ~= 4 GB
            # (raw EEG + cwt + burst grids in flight); threading is mandatory on
            # this stack (torch/numpy DLL bug — loky workers auto-load numpy
            # before user code can hoist torch, triggering shm.dll failure).
            from cadence.io.resources import (
                limit_blas_threads, log_resources, pick_n_jobs,
            )
            n_jobs = pick_n_jobs(per_worker_ram_gb=4.0, requested=4,
                                  max_jobs_hard_cap=len(todo))
            log_resources(prefix='[v11 --all] pre-fan-out: ')
            print(f"Running V11 scaffold on {len(todo)} remaining sessions "
                  f"(n_jobs={n_jobs}, threading; per-worker RAM ~= 4 GB)")

            def _bounded_run_session(name, cfg):
                with limit_blas_threads(1):
                    return run_session(name, cfg)

            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_bounded_run_session)(name, config) for name in todo
            )
        else:
            print("All sessions already have V11 results.")
            results = []
        valid = [r for r in results if r is not None]
        print(f"\nCompleted: {len(valid)}/{len(todo)} new sessions "
              f"({len(session_names) - len(todo)} already done)")
    else:
        run_session(args.session, config)
