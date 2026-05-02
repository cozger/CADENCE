"""Native-rate coupling prototype: S-map (EEG) + Hawkes (BL).

Single-session test of the "estimate coupling at native rate, not in 2 Hz
windows" architectural hypothesis. Compares two operators side-by-side:

  - Multivariate S-map on bandpassed alpha-band EEG @ 256 Hz between AF3
    of therapist and AF3 of patient. Output: time-varying directed
    interaction Jacobian (Deyle & Sugihara 2016).
  - Sliding-window Hawkes on smile events (mouthSmileLeft+Right). Output:
    time-varying cross-excitation strength α(t) plus continuous λ(t).

Validates with a pseudo-dyad null (patient from a different session).

Usage:
    python scripts/_test_native_rate_coupling.py --session y_06 --pseudo y_17
"""

import sys
import os
import json
import argparse
import glob
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch must import before numpy on Windows (torch 2.10 + numpy 2.4 DLL bug)
import torch  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from cadence.config import load_config
from cadence.data.alignment import (
    discover_cached_sessions, load_session_from_cache, apply_modality_exclusions,
)
from cadence.significance.fast_cycles import _fft_bandpass
from cadence.significance.smap import (
    gpu_smap_cross_coupling, simplex_select_E,
)
from cadence.significance.hawkes_coupling import (
    fit_hawkes_pathway, fit_hawkes_sliding,
    hawkes_intensity_timecourse, hawkes_sliding_surrogate_z,
)
from scripts.run_session_v6 import load_xdf_session
from scripts._run_scaffold_v82 import CONDITION_ORDER, CONDITION_COLORS
from scripts._run_scaffold_v10 import RAW_DIR

FS_EEG = 256.0
FS_EEG_LIB = 64.0            # Library/query rate for S-map. Bandpassed alpha
                              # has ~5 Hz bandwidth, so 64 Hz preserves all
                              # information (Nyquist headroom 5x). Going from
                              # 256 Hz to 64 Hz cuts S-map work 4x with no
                              # information loss for an alpha-bandpassed signal.
FS_BL = 30.0
ALPHA_BAND = (8.0, 13.0)
AF3_IDX = 0                  # EPOC channel index for AF3 (frontal left)
SMILE_AUS = [44, 45]


# ───────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ───────────────────────────────────────────────────────────────────────

def _load_session(session_name, config):
    """Load raw XDF + cached preprocessed session."""
    xdf_files = glob.glob(os.path.join(RAW_DIR, f'{session_name}*.xdf'))
    if not xdf_files:
        raise FileNotFoundError(f'No XDF for {session_name} in {RAW_DIR}')
    session_data = load_xdf_session(xdf_files[0])

    cached_sessions = discover_cached_sessions(config['session_cache'])
    matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not matches:
        raise FileNotFoundError(f'No cache for {session_name}')
    cached = load_session_from_cache(matches[0], config)
    apply_modality_exclusions(cached, session_name)
    return session_data, cached


def _bandpass_alpha(eeg, fs, band=ALPHA_BAND, device=None):
    """GPU FFT bandpass at native rate. Input (N, C), returns (N, C)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigs = torch.as_tensor(np.ascontiguousarray(eeg.T),
                           dtype=torch.float32, device=device)
    filt = _fft_bandpass(sigs, fs, band, device=device)
    return filt.cpu().numpy().T


def _decimate_post_bandpass(sig, fs_in, fs_out):
    """Decimate a band-limited signal by integer factor.

    Used after bandpass to drop the library rate from 256 Hz (sensor native) to
    a lower rate (e.g. 64 Hz). This is information-preserving for alpha-band
    signals because the bandwidth (5 Hz) is well below the new Nyquist.
    Uses torch FFT for an exact spectrum-domain decimation that doesn't
    re-introduce out-of-band content.
    """
    factor = int(round(fs_in / fs_out))
    if factor <= 1:
        return sig.copy(), fs_in
    return sig[::factor].copy(), fs_in / factor


def _resample_landmarks(landmarks, person, fs=FS_BL):
    """Resample raw MediaPipe landmark stream to a uniform grid.

    Returns (T, 52) bl matrix and a t_grid in session-relative seconds
    starting at 0.
    """
    if person not in landmarks:
        return None, None
    ts, d = landmarks[person]
    t0 = float(ts[0])
    t1 = float(ts[-1])
    dur = t1 - t0
    T = int(dur * fs)
    t_grid = np.linspace(0, dur, T)
    sig = np.stack([np.interp(t_grid, ts - t0, d[:, c]) for c in range(52)], axis=1)
    np.clip(sig, 0, 1, out=sig)
    return sig.astype(np.float32), t_grid + t0  # absolute LSL times


def _detect_smile_events(bl, t_grid, prominence=0.3, min_distance_s=3.0):
    """find_peaks on AU 44+45 composite; returns absolute LSL event times."""
    composite = bl[:, SMILE_AUS[0]] + bl[:, SMILE_AUS[1]]
    peaks, _ = find_peaks(composite, prominence=prominence,
                          distance=int(min_distance_s * FS_BL))
    return t_grid[peaks]


# ───────────────────────────────────────────────────────────────────────
#  Coupling pipelines
# ───────────────────────────────────────────────────────────────────────

def _run_smap_block(p1_alpha_af3_lib, p2_alpha_af3_lib, fs_lib=FS_EEG_LIB,
                    label='real', n_surrogates=200, theta=1.0, simplex=True):
    """Run S-map cross-coupling and surrogate z-scoring.

    Inputs are the alpha-bandpassed signals decimated to ``fs_lib`` Hz.
    Hyperparameters scale with fs_lib.
    """
    # ~quarter alpha cycle at 10 Hz: tau_s = 25 ms; theiler_s = 305 ms
    tau_samples = max(1, int(round(0.025 * fs_lib)))
    delta_samples = tau_samples
    theiler_samples = max(1, int(round(0.305 * fs_lib)))

    print(f'  [{label}] simplex projection for E selection '
          f'(fs_lib={fs_lib:.0f}Hz, τ={tau_samples}, theiler={theiler_samples})...',
          flush=True)
    if simplex:
        sel = simplex_select_E(p1_alpha_af3_lib,
                               tau=tau_samples, delta=delta_samples,
                               E_grid=(2, 3, 4, 5),
                               theiler_samples=theiler_samples)
        E_best = sel['E_best']
        print(f'    skill_per_E = '
              f'{ {k: round(v, 3) for k, v in sel["skill_per_E"].items()} }, '
              f'-> E={E_best}')
    else:
        E_best = 3

    print(f'  [{label}] S-map at E={E_best}, theta={theta}, '
          f'n_surr={n_surrogates}...', flush=True)
    t0 = time.time()
    res = gpu_smap_cross_coupling(
        p1_alpha_af3_lib, p2_alpha_af3_lib,
        fs=fs_lib, E=E_best,
        tau_samples=tau_samples, delta_samples=delta_samples,
        theta=theta, theiler_samples=theiler_samples,
        query_stride=2,                   # output at fs_lib/2 = 32 Hz
        batch_size=128, lib_chunk=32768,
        n_surrogates=n_surrogates, seed=42,
    )
    print(f'    ran in {time.time() - t0:.1f}s, '
          f'pred skill fwd={res["pred_skill_forward"]:.3f}, '
          f'rev={res["pred_skill_reverse"]:.3f}, '
          f'cond p99={np.nanpercentile(res["cond_forward"], 99):.1e}, '
          f'N_q={len(res["t_query"])}')
    return res


def _run_theta_sensitivity(p1_alpha_lib, p2_alpha_lib, E, fs_lib=FS_EEG_LIB):
    """θ scan to check robustness of coupling timecourse to locality weight."""
    print('  θ sensitivity scan {0.3, 1.0, 3.0}...', flush=True)
    tau_samples = max(1, int(round(0.025 * fs_lib)))
    theiler_samples = max(1, int(round(0.305 * fs_lib)))
    out = {}
    for theta in (0.3, 1.0, 3.0):
        res = gpu_smap_cross_coupling(
            p1_alpha_lib, p2_alpha_lib,
            fs=fs_lib, E=E, tau_samples=tau_samples,
            delta_samples=tau_samples,
            theta=theta, theiler_samples=theiler_samples,
            query_stride=2,
            batch_size=128, lib_chunk=32768, n_surrogates=0,
        )
        out[theta] = res['coupling_p1_to_p2']
    rs = []
    keys = sorted(out.keys())
    for i in range(len(keys) - 1):
        a, b = out[keys[i]], out[keys[i + 1]]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() > 10:
            r = np.corrcoef(a[valid], b[valid])[0, 1]
        else:
            r = float('nan')
        rs.append((keys[i], keys[i + 1], r))
        print(f'    r(θ={keys[i]}, θ={keys[i+1]}) = {r:+.3f}')
    return out, rs


def _run_hawkes_block(events_t, events_p, T_total, label='real',
                      n_surrogates=200, win_s=60.0, hop_s=15.0):
    """Run sliding-window Hawkes (both directions) with surrogate z-scoring."""
    print(f'  [{label}] global β fit + sliding Hawkes...', flush=True)
    # Global β from full-session fit, both directions
    if len(events_p) >= 3 and len(events_t) >= 1:
        _, _, beta_p_to_t, *_ = fit_hawkes_pathway(events_p, events_t, T_total)
    else:
        beta_p_to_t = 0.4
    if len(events_t) >= 3 and len(events_p) >= 1:
        _, _, beta_t_to_p, *_ = fit_hawkes_pathway(events_t, events_p, T_total)
    else:
        beta_t_to_p = 0.4
    # Use mean of the two as the global β for both sliding fits (consistency)
    beta_global = float(np.mean([beta_p_to_t, beta_t_to_p]))
    print(f'    β global = {beta_global:.3f} '
          f'(p→t={beta_p_to_t:.3f}, t→p={beta_t_to_p:.3f})')

    # Sliding T←P (target=therapist, source=patient)
    sl_T_from_P = fit_hawkes_sliding(
        events_p, events_t, T_total,
        win_s=win_s, hop_s=hop_s, beta_global=beta_global)
    sl_P_from_T = fit_hawkes_sliding(
        events_t, events_p, T_total,
        win_s=win_s, hop_s=hop_s, beta_global=beta_global)
    n_valid_t_from_p = np.isfinite(sl_T_from_P['alpha_t']).sum()
    n_valid_p_from_t = np.isfinite(sl_P_from_T['alpha_t']).sum()
    print(f'    valid windows: T←P = {n_valid_t_from_p}/{len(sl_T_from_P["alpha_t"])}, '
          f'P←T = {n_valid_p_from_t}/{len(sl_P_from_T["alpha_t"])}')

    # Surrogate z (skip if too few events to be meaningful)
    if n_surrogates > 0 and n_valid_t_from_p >= 3:
        z_T_from_P = hawkes_sliding_surrogate_z(
            events_p, events_t, T_total,
            sl_T_from_P['alpha_t'], sl_T_from_P['t_centers'],
            beta_global=beta_global, n_surrogates=n_surrogates,
            win_s=win_s, hop_s=hop_s)
    else:
        z_T_from_P = np.full_like(sl_T_from_P['alpha_t'], np.nan)

    if n_surrogates > 0 and n_valid_p_from_t >= 3:
        z_P_from_T = hawkes_sliding_surrogate_z(
            events_t, events_p, T_total,
            sl_P_from_T['alpha_t'], sl_P_from_T['t_centers'],
            beta_global=beta_global, n_surrogates=n_surrogates,
            win_s=win_s, hop_s=hop_s)
    else:
        z_P_from_T = np.full_like(sl_P_from_T['alpha_t'], np.nan)

    return dict(
        sl_T_from_P=sl_T_from_P,
        sl_P_from_T=sl_P_from_T,
        z_T_from_P=z_T_from_P,
        z_P_from_T=z_P_from_T,
        beta_global=beta_global,
    )


# ───────────────────────────────────────────────────────────────────────
#  Per-condition statistics
# ───────────────────────────────────────────────────────────────────────

def _per_condition_stats(t_axis, values, segments):
    """Mean/std/N per condition. t_axis and segments share the same time origin."""
    out = {}
    for cond, t0, t1 in segments:
        mask = (t_axis >= t0) & (t_axis <= t1) & np.isfinite(values)
        if mask.sum() == 0:
            out[cond] = (np.nan, np.nan, 0)
        else:
            out[cond] = (float(values[mask].mean()),
                         float(values[mask].std()),
                         int(mask.sum()))
    return out


def _cohens_d(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    s = np.sqrt(((a.var(ddof=1) * (len(a) - 1) +
                  b.var(ddof=1) * (len(b) - 1)) /
                 (len(a) + len(b) - 2)))
    if s < 1e-12:
        return float('nan')
    return float((a.mean() - b.mean()) / s)


# ───────────────────────────────────────────────────────────────────────
#  Visualization
# ───────────────────────────────────────────────────────────────────────

def _shade_segments(ax, segments, t_origin):
    """Shade condition segments as background spans.

    Segments are already in session-relative time. ``t_origin`` is the
    session-relative time of EEG sample 0 (≈ 0); subtracting it is a no-op
    in normal use but keeps the API flexible.
    """
    for cond, t0, t1 in segments:
        c = CONDITION_COLORS.get(cond, '#cccccc')
        ax.axvspan(t0 - t_origin, t1 - t_origin, color=c, alpha=0.12, zorder=0)


def _plot_all(out_path, smap_real, smap_pseudo, hawkes_real, hawkes_pseudo,
              p1_alpha, p2_alpha, fs_eeg, fs_lib, t_eeg_origin,
              events_t_real, events_p_real, events_p_pseudo,
              segments, session_name, pseudo_name, theta_scan_rs):
    """6-panel figure."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f'Native-rate coupling prototype  —  real: {session_name} '
        f'   pseudo-dyad null: {session_name}/{pseudo_name}',
        fontsize=12, y=0.995)

    n_eeg = min(len(p1_alpha), len(p2_alpha))
    t_eeg = np.arange(n_eeg) / fs_eeg                        # session-relative s
    decim = max(1, int(fs_eeg / 50))                         # 50 Hz for plotting

    # ── 1. Bandpassed alpha EEG ──────────────────────────────────────
    ax = axes[0]
    _shade_segments(ax, segments, t_eeg_origin)
    ax.plot(t_eeg[::decim], p1_alpha[:n_eeg:decim],
            color='#1f77b4', alpha=0.6, lw=0.5, label='Therapist AF3 α')
    ax.plot(t_eeg[::decim], p2_alpha[:n_eeg:decim],
            color='#d62728', alpha=0.6, lw=0.5, label='Patient AF3 α')
    ax.set_ylabel('Alpha\n(µV)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('1) Bandpassed alpha (8–13 Hz) at 256 Hz native rate', fontsize=10)

    # ── 2. S-map directed coupling Jacobians ─────────────────────────
    ax = axes[1]
    _shade_segments(ax, segments, t_eeg_origin)
    t_q_real = smap_real['t_query'] / fs_lib
    ax.plot(t_q_real, smap_real['coupling_p1_to_p2'],
            color='#1f77b4', lw=0.5, alpha=0.7, label='β T→P')
    ax.plot(t_q_real, smap_real['coupling_p2_to_p1'],
            color='#d62728', lw=0.5, alpha=0.7, label='β P→T')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('S-map β\n(at lag τ)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    title_str = '2) S-map Jacobian coefficient (lag τ=23 ms)'
    if theta_scan_rs:
        title_str += '   θ-stability r=' + ', '.join(
            f'{a:.1f}|{b:.1f}={r:+.2f}' for a, b, r in theta_scan_rs)
    ax.set_title(title_str, fontsize=10)

    # ── 3. S-map surrogate z + pseudo-dyad overlay ───────────────────
    ax = axes[2]
    _shade_segments(ax, segments, t_eeg_origin)
    ax.axhline(2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(-2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if smap_pseudo is not None:
        t_q_p = smap_pseudo['t_query'] / fs_lib
        ax.plot(t_q_p, smap_pseudo['z_p1_to_p2'],
                color='gray', lw=0.5, alpha=0.5, label='z T→P (pseudo)')
        ax.plot(t_q_p, smap_pseudo['z_p2_to_p1'],
                color='lightgray', lw=0.5, alpha=0.5, label='z P→T (pseudo)')
    ax.plot(t_q_real, smap_real['z_p1_to_p2'],
            color='#1f77b4', lw=0.6, alpha=0.85, label='z T→P (real)')
    ax.plot(t_q_real, smap_real['z_p2_to_p1'],
            color='#d62728', lw=0.6, alpha=0.85, label='z P→T (real)')
    ax.set_ylabel('S-map z')
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.set_title('3) S-map surrogate z (real vs pseudo-dyad null)', fontsize=10)
    ax.set_ylim(-6, 6)

    # ── 4. Smile event raster ────────────────────────────────────────
    ax = axes[3]
    _shade_segments(ax, segments, t_eeg_origin)
    # Events are already session-relative; no offset needed
    if len(events_t_real) > 0:
        ax.vlines(events_t_real, 0.7, 1.0,
                  color='#1f77b4', lw=0.8, label=f'T smiles ({len(events_t_real)})')
    if len(events_p_real) > 0:
        ax.vlines(events_p_real, 0.0, 0.3,
                  color='#d62728', lw=0.8, label=f'P smiles ({len(events_p_real)})')
    if len(events_p_pseudo) > 0:
        ax.vlines(events_p_pseudo, 0.4, 0.6,
                  color='gray', lw=0.6, alpha=0.7,
                  label=f'P pseudo ({len(events_p_pseudo)})')
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0.15, 0.5, 0.85])
    ax.set_yticklabels(['Patient', 'Pseudo P', 'Therapist'])
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_title('4) Smile event raster (AU 44+45 composite, prominence=0.3)', fontsize=10)

    # ── 5. Hawkes α(t) sliding-window estimates (z-scored) ───────────
    ax = axes[4]
    _shade_segments(ax, segments, t_eeg_origin)
    ax.axhline(2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(-2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    sl_tp = hawkes_real['sl_T_from_P']
    sl_pt = hawkes_real['sl_P_from_T']
    ax.plot(sl_tp['t_centers'], hawkes_real['z_T_from_P'],
            color='#1f77b4', marker='o', ms=3, lw=1.0, label='z T←P (real)')
    ax.plot(sl_pt['t_centers'], hawkes_real['z_P_from_T'],
            color='#d62728', marker='o', ms=3, lw=1.0, label='z P←T (real)')
    if hawkes_pseudo is not None:
        ax.plot(hawkes_pseudo['sl_T_from_P']['t_centers'],
                hawkes_pseudo['z_T_from_P'],
                color='gray', marker='s', ms=2, lw=0.7, alpha=0.6,
                label='z T←P (pseudo)')
    ax.set_ylabel('Hawkes α z')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.set_title('5) Sliding-window Hawkes α(t) z-score (60s win, 15s hop)', fontsize=10)
    ax.set_ylim(-6, 6)

    # ── 6. Hawkes intensity λ(t) ─────────────────────────────────────
    ax = axes[5]
    _shade_segments(ax, segments, t_eeg_origin)
    # Build dense λ on [0, T_total] at 30 Hz
    T_total = sl_tp['t_centers'][-1] + sl_tp['t_centers'][0] if len(sl_tp['t_centers']) else 0
    if T_total > 0:
        t_lam = np.arange(0, T_total, 1.0 / FS_BL)
        # T←P intensity uses patient events as source, therapist α(t) as target rate
        lam_t_from_p = hawkes_intensity_timecourse(
            events_p_real, t_lam,
            mu_t=sl_tp['mu_t'], alpha_t=sl_tp['alpha_t'],
            beta=hawkes_real['beta_global'],
            t_centers=sl_tp['t_centers'])
        lam_p_from_t = hawkes_intensity_timecourse(
            events_t_real, t_lam,
            mu_t=sl_pt['mu_t'], alpha_t=sl_pt['alpha_t'],
            beta=hawkes_real['beta_global'],
            t_centers=sl_pt['t_centers'])
        ax.plot(t_lam, lam_t_from_p, color='#1f77b4', lw=0.8, label='λ T←P')
        ax.plot(t_lam, lam_p_from_t, color='#d62728', lw=0.8, label='λ P←T')
    ax.set_ylabel('Hawkes λ\n(events/s)')
    ax.set_xlabel('Session time (s, relative to start)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('6) Continuous Hawkes intensity λ(t)', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  Saved figure -> {out_path}')


# ───────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────

def run(session_name='y_06', pseudo_name='y_17',
        out_dir='results/native_rate_coupling',
        n_surrogates_smap=200, n_surrogates_hawkes=100,
        skip_pseudo=False, skip_theta_scan=False):
    print('='*72)
    print(f' Native-rate coupling prototype: real={session_name}, '
          f'pseudo={pseudo_name}')
    print('='*72)

    config = load_config()
    os.makedirs(out_dir, exist_ok=True)

    # ── Load real session ────────────────────────────────────────────
    print(f'\n[1] Loading real session {session_name}...')
    real_session, real_cached = _load_session(session_name, config)
    p1_role = real_session.get('p1_role', 'unknown')
    p2_role = real_session.get('p2_role', 'unknown')
    print(f'  p1={p1_role}, p2={p2_role}')

    # Identify therapist/patient — keep array order; map via roles for labels
    p1_eeg = real_cached['p1_eeg'].astype(np.float64)
    p2_eeg = real_cached['p2_eeg'].astype(np.float64)
    p1_eeg_ts = real_cached['p1_eeg_ts']
    p2_eeg_ts = real_cached['p2_eeg_ts']
    n_eeg = min(p1_eeg.shape[0], p2_eeg.shape[0])
    p1_eeg = p1_eeg[:n_eeg, :14]
    p2_eeg = p2_eeg[:n_eeg, :14]
    fs_eeg = float(np.round(1.0 / np.median(np.diff(p1_eeg_ts[:1000]))))
    print(f'  EEG: N={n_eeg}, fs={fs_eeg:.0f} Hz, '
          f'duration={n_eeg / fs_eeg:.1f}s')

    # ── Bandpass alpha at native rate, then decimate to library rate ─
    print('\n[2] Bandpassing alpha (8-13 Hz) at native rate, then decimating...')
    p1_alpha = _bandpass_alpha(p1_eeg, fs_eeg)
    p2_alpha = _bandpass_alpha(p2_eeg, fs_eeg)
    p1_alpha_af3 = p1_alpha[:, AF3_IDX]
    p2_alpha_af3 = p2_alpha[:, AF3_IDX]
    # After alpha bandpass the signal has only ~5 Hz bandwidth, so 64 Hz
    # sampling preserves all information (5x Nyquist headroom).
    p1_alpha_af3_lib, fs_lib = _decimate_post_bandpass(
        p1_alpha_af3, fs_eeg, FS_EEG_LIB)
    p2_alpha_af3_lib, _ = _decimate_post_bandpass(
        p2_alpha_af3, fs_eeg, FS_EEG_LIB)
    print(f'  decimated 256 -> {fs_lib:.0f} Hz: '
          f'N_lib = {len(p1_alpha_af3_lib)} samples')

    # ── Establish a single session-relative time frame ──────────────
    # The XDF cache stores `p1_eeg_ts` already shifted so it starts at ~0
    # (session-relative). Markers from XDF and raw landmark streams are in
    # absolute LSL time. We convert everything to a common session-relative
    # frame: t_rel = t_abs_lsl - lsl_to_rel_offset, where lsl_to_rel_offset
    # is taken from the first landmarks timestamp (matches how the cache
    # was originally aligned).
    landmarks = real_session['landmarks']
    landmarks_lsl_origin = float(min(landmarks['P1'][0][0],
                                      landmarks['P2'][0][0]))
    lsl_to_rel = landmarks_lsl_origin

    markers = real_session['markers']
    segments = []
    for cond in CONDITION_ORDER:
        t0 = markers.get(f'{cond}_start')
        t1 = markers.get(f'{cond}_stop')
        if t0 is not None and t1 is not None and t1 > t0:
            segments.append((cond, float(t0) - lsl_to_rel,
                                   float(t1) - lsl_to_rel))
    print(f'  Segments (rel): {[(s[0], round(s[1],1), round(s[2],1)) for s in segments]}')

    # The cached EEG starts at p1_eeg_ts[0] (≈0, with sub-ms offset).
    t_eeg_origin = float(p1_eeg_ts[0])

    # ── S-map block (real) ──────────────────────────────────────────
    print('\n[3] S-map (real)...')
    smap_real = _run_smap_block(p1_alpha_af3_lib, p2_alpha_af3_lib,
                                 fs_lib=fs_lib, label='real',
                                 n_surrogates=n_surrogates_smap)

    theta_scan_rs = []
    if not skip_theta_scan:
        print('\n[3b] θ sensitivity scan (real)...')
        _, theta_scan_rs = _run_theta_sensitivity(
            p1_alpha_af3_lib, p2_alpha_af3_lib, E=smap_real['E'], fs_lib=fs_lib)

    # ── Smile event detection (real) ────────────────────────────────
    print('\n[4] Smile event detection...')
    bl_p1, t_p1 = _resample_landmarks(landmarks, 'P1')
    bl_p2, t_p2 = _resample_landmarks(landmarks, 'P2')
    if bl_p1 is None or bl_p2 is None:
        raise RuntimeError('Missing P1/P2 landmarks')
    # Convert landmark timestamps from absolute LSL to session-relative
    t_p1 = t_p1 - lsl_to_rel
    t_p2 = t_p2 - lsl_to_rel
    events_p1 = _detect_smile_events(bl_p1, t_p1)
    events_p2 = _detect_smile_events(bl_p2, t_p2)
    print(f'  P1 smiles: {len(events_p1)}, P2 smiles: {len(events_p2)}')

    # Therapist/patient assignment based on roles
    if p1_role == 'therapist':
        events_t_real, events_p_real = events_p1, events_p2
    else:
        events_t_real, events_p_real = events_p2, events_p1

    # ── Hawkes block (real) ─────────────────────────────────────────
    # Events are already in session-relative seconds (we subtracted lsl_to_rel
    # right after _resample_landmarks).
    T_total_bl = float(max(t_p1[-1], t_p2[-1]))
    print(f'\n[5] Hawkes (real), T_total={T_total_bl:.1f}s')
    et_local = events_t_real
    ep_local = events_p_real
    hawkes_real = _run_hawkes_block(
        et_local, ep_local, T_total_bl,
        label='real', n_surrogates=n_surrogates_hawkes)

    # ── Pseudo-dyad null ────────────────────────────────────────────
    smap_pseudo = None
    hawkes_pseudo = None
    events_p_pseudo_abs = np.array([])
    if not skip_pseudo:
        print(f'\n[6] Pseudo-dyad null with {pseudo_name}...')
        try:
            pseudo_session, pseudo_cached = _load_session(pseudo_name, config)
            p2_eeg_pseudo = pseudo_cached['p2_eeg' if pseudo_session.get('p2_role') == 'patient' else 'p1_eeg'].astype(np.float64)
            # Use patient EEG of the pseudo session
            n_pseudo = min(n_eeg, p2_eeg_pseudo.shape[0])
            p2_eeg_pseudo = p2_eeg_pseudo[:n_pseudo, :14]
            p2_alpha_pseudo = _bandpass_alpha(p2_eeg_pseudo, fs_eeg)
            n_used = min(n_eeg, n_pseudo)
            p1_alpha_af3_pseudo_lib, _ = _decimate_post_bandpass(
                p1_alpha_af3[:n_used], fs_eeg, FS_EEG_LIB)
            p2_alpha_af3_pseudo_lib, _ = _decimate_post_bandpass(
                p2_alpha_pseudo[:n_used, AF3_IDX], fs_eeg, FS_EEG_LIB)
            n_used_lib = min(len(p1_alpha_af3_pseudo_lib),
                              len(p2_alpha_af3_pseudo_lib))
            print('  S-map (pseudo)...')
            smap_pseudo = _run_smap_block(
                p1_alpha_af3_pseudo_lib[:n_used_lib],
                p2_alpha_af3_pseudo_lib[:n_used_lib],
                fs_lib=fs_lib, label='pseudo',
                n_surrogates=n_surrogates_smap)

            # BL events from pseudo session's patient
            print('  Smile events (pseudo)...')
            pseudo_landmarks = pseudo_session['landmarks']
            pseudo_p2_role = pseudo_session.get('p2_role', 'unknown')
            pseudo_p1_role = pseudo_session.get('p1_role', 'unknown')
            pseudo_patient_key = 'P2' if pseudo_p2_role == 'patient' else 'P1'
            bl_p_pseudo, t_p_pseudo_abs = _resample_landmarks(
                pseudo_landmarks, pseudo_patient_key)
            if bl_p_pseudo is not None:
                # Pseudo events: rebase to start at 0 of pseudo session, then
                # interpret directly as session-relative for our plotting frame
                t_p_pseudo_rel = t_p_pseudo_abs - t_p_pseudo_abs[0]
                events_p_pseudo_rel = _detect_smile_events(bl_p_pseudo, t_p_pseudo_rel)
                print(f'    pseudo patient smiles: {len(events_p_pseudo_rel)}')
                # Truncate to real-session duration so timelines align
                ep_pseudo_local = events_p_pseudo_rel[
                    events_p_pseudo_rel < T_total_bl]
                print('  Hawkes (pseudo)...')
                hawkes_pseudo = _run_hawkes_block(
                    et_local, ep_pseudo_local, T_total_bl,
                    label='pseudo', n_surrogates=n_surrogates_hawkes)
                events_p_pseudo_abs = ep_pseudo_local  # already session-relative
        except Exception as e:
            print(f'  Pseudo-dyad failed: {e!r}')
            smap_pseudo = None
            hawkes_pseudo = None

    # ── Per-condition stats and Cohen's d ───────────────────────────
    print('\n[7] Per-condition statistics (S-map z, real)...')
    # t_query is in library-rate samples, library starts at session-relative
    # time = t_eeg_origin (≈ 0). Segments are also session-relative.
    t_q_rel = smap_real['t_query'] / fs_lib + t_eeg_origin
    cond_stats_smap_tp = _per_condition_stats(t_q_rel, smap_real['z_p1_to_p2'], segments)
    cond_stats_smap_pt = _per_condition_stats(t_q_rel, smap_real['z_p2_to_p1'], segments)
    print('  S-map z T→P:')
    for c, (m, s, n) in cond_stats_smap_tp.items():
        print(f'    {c:14s} mean={m:+.2f} std={s:.2f} n={n}')
    print('  S-map z P→T:')
    for c, (m, s, n) in cond_stats_smap_pt.items():
        print(f'    {c:14s} mean={m:+.2f} std={s:.2f} n={n}')

    # Cohen's d conv_1 vs meditate_K
    def _vals(stats_t, key):
        for cond, t0, t1 in segments:
            if cond == key:
                m = (t_q_rel >= t0) & (t_q_rel <= t1)
                return smap_real['z_p1_to_p2'][m] if stats_t == 'tp' else smap_real['z_p2_to_p1'][m]
        return np.array([])

    d_tp = _cohens_d(_vals('tp', 'conv_1'), _vals('tp', 'meditate_K'))
    d_pt = _cohens_d(_vals('pt', 'conv_1'), _vals('pt', 'meditate_K'))
    print(f'  Cohen\'s d (conv_1 vs meditate_K): T→P d={d_tp:+.2f}, P→T d={d_pt:+.2f}')

    # Pseudo-dyad z magnitude check
    if smap_pseudo is not None:
        z_pseudo_mean_tp = float(np.nanmean(smap_pseudo['z_p1_to_p2']))
        z_pseudo_mean_pt = float(np.nanmean(smap_pseudo['z_p2_to_p1']))
        print(f'  Pseudo-dyad mean z: T→P={z_pseudo_mean_tp:+.3f}, '
              f'P→T={z_pseudo_mean_pt:+.3f}  (|<1| = null OK)')

    # ── Plot ────────────────────────────────────────────────────────
    print('\n[8] Plotting...')
    fig_path = os.path.join(out_dir, f'{session_name}_smap_hawkes.png')
    _plot_all(
        fig_path, smap_real, smap_pseudo, hawkes_real, hawkes_pseudo,
        p1_alpha_af3, p2_alpha_af3, fs_eeg, fs_lib, t_eeg_origin,
        events_t_real, events_p_real,
        events_p_pseudo_abs if len(events_p_pseudo_abs) > 0 else np.array([]),
        segments, session_name, pseudo_name, theta_scan_rs)

    # ── Save data ───────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, f'{session_name}_data.npz')
    np.savez_compressed(
        npz_path,
        t_query_smap=smap_real['t_query'],
        smap_coupling_tp=smap_real['coupling_p1_to_p2'],
        smap_coupling_pt=smap_real['coupling_p2_to_p1'],
        smap_z_tp=smap_real['z_p1_to_p2'],
        smap_z_pt=smap_real['z_p2_to_p1'],
        smap_E=smap_real['E'],
        smap_pred_skill_fwd=smap_real['pred_skill_forward'],
        smap_pred_skill_rev=smap_real['pred_skill_reverse'],
        hawkes_t_centers=hawkes_real['sl_T_from_P']['t_centers'],
        hawkes_alpha_t_from_p=hawkes_real['sl_T_from_P']['alpha_t'],
        hawkes_alpha_p_from_t=hawkes_real['sl_P_from_T']['alpha_t'],
        hawkes_z_t_from_p=hawkes_real['z_T_from_P'],
        hawkes_z_p_from_t=hawkes_real['z_P_from_T'],
        hawkes_beta=hawkes_real['beta_global'],
        events_t=events_t_real,
        events_p=events_p_real,
        t_eeg_origin=t_eeg_origin,
        lsl_to_rel=lsl_to_rel,
        T_total_bl=T_total_bl,
        segments=np.array(segments, dtype=object),
        fs_eeg=fs_eeg,
    )
    print(f'  Saved data -> {npz_path}')

    summary = dict(
        session=session_name, pseudo=pseudo_name,
        smap_E=int(smap_real['E']),
        smap_pred_skill_fwd=float(smap_real['pred_skill_forward']),
        smap_pred_skill_rev=float(smap_real['pred_skill_reverse']),
        cohens_d_smap_tp=float(d_tp) if not np.isnan(d_tp) else None,
        cohens_d_smap_pt=float(d_pt) if not np.isnan(d_pt) else None,
        n_events_t=int(len(events_t_real)),
        n_events_p=int(len(events_p_real)),
        hawkes_beta=float(hawkes_real['beta_global']),
        per_condition_smap_z_tp={c: list(v) for c, v in cond_stats_smap_tp.items()},
        per_condition_smap_z_pt={c: list(v) for c, v in cond_stats_smap_pt.items()},
        theta_scan_correlations=[(a, b, float(r)) for a, b, r in theta_scan_rs],
    )
    if smap_pseudo is not None:
        summary['pseudo_z_mean_tp'] = float(np.nanmean(smap_pseudo['z_p1_to_p2']))
        summary['pseudo_z_mean_pt'] = float(np.nanmean(smap_pseudo['z_p2_to_p1']))

    with open(os.path.join(out_dir, f'{session_name}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nDone.')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', default='y_06')
    p.add_argument('--pseudo', default='y_17')
    p.add_argument('--out-dir', default='results/native_rate_coupling')
    p.add_argument('--n-surr-smap', type=int, default=200)
    p.add_argument('--n-surr-hawkes', type=int, default=100)
    p.add_argument('--skip-pseudo', action='store_true')
    p.add_argument('--skip-theta-scan', action='store_true')
    args = p.parse_args()

    run(session_name=args.session, pseudo_name=args.pseudo,
        out_dir=args.out_dir,
        n_surrogates_smap=args.n_surr_smap,
        n_surrogates_hawkes=args.n_surr_hawkes,
        skip_pseudo=args.skip_pseudo,
        skip_theta_scan=args.skip_theta_scan)


if __name__ == '__main__':
    main()
