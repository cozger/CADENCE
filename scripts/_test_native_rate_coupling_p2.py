"""Native-rate coupling prototype 2: ROI-averaged + multiband + ECG.

Addresses the three weak hypotheses from P1's S-map result:
  1. Single-channel insufficiency  → ROI-averaged frontal pool
  2. Single-band view              → theta + alpha + beta in parallel
  3. Summary scalar compression    → full Jacobian heatmap

Adds ECG LF and HF bands as a second native-rate test on the cleanest
oscillator in the dataset.

Usage:
    python scripts/_test_native_rate_coupling_p2.py --session y_06 --pseudo y_17
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
from cadence.constants import FRONTAL_ROI
from cadence.significance.fast_cycles import _fft_bandpass
from cadence.significance.smap import (
    gpu_smap_cross_coupling, simplex_select_E,
)
from scripts.run_session_v6 import load_xdf_session
from scripts._run_scaffold_v82 import CONDITION_ORDER, CONDITION_COLORS
from scripts._run_scaffold_v10 import RAW_DIR

FS_EEG = 256.0
FS_ECG = 130.0
FS_HRV_GRID = 4.0   # uniform IBI grid for HRV S-map

EEG_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (15.0, 25.0),
}
EEG_BAND_LIB_RATES = {        # post-bandpass library rate per band
    'theta': 32.0,
    'alpha': 64.0,
    'beta':  96.0,
}
ECG_BANDS = {
    'LF': (0.04, 0.15),
    'HF': (0.15, 0.40),
}


# ───────────────────────────────────────────────────────────────────────
#  Loading
# ───────────────────────────────────────────────────────────────────────

def _load_session(session_name, config):
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


# ───────────────────────────────────────────────────────────────────────
#  Signal preparation: ROI-averaged EEG band + HRV IBI series
# ───────────────────────────────────────────────────────────────────────

def _roi_average_band(eeg, fs, band, roi_idx, device=None):
    """Bandpass each channel at native rate, then ROI-average.

    Args:
        eeg: (N, C) numpy array.
        fs: sampling rate.
        band: (lo, hi) Hz.
        roi_idx: list of channel indices.

    Returns:
        (N,) ROI-averaged bandpassed signal.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigs = torch.as_tensor(np.ascontiguousarray(eeg.T),
                           dtype=torch.float32, device=device)
    filt = _fft_bandpass(sigs, fs, band, device=device)         # (C, N)
    filt = filt.cpu().numpy()
    return filt[roi_idx, :].mean(axis=0)                         # (N,)


def _decimate(sig, fs_in, fs_out):
    factor = int(round(fs_in / fs_out))
    if factor <= 1:
        return sig.copy(), fs_in
    return sig[::factor].copy(), fs_in / factor


def _extract_ibi_series(ecg_filtered, ecg_valid, ecg_ts,
                        fs=FS_ECG, target_fs=FS_HRV_GRID):
    """Extract IBI series → uniform grid at target_fs.

    Mirrors the R-peak detection in cadence.data.preprocessors.extract_ecg_features
    (find_peaks distance≥0.4s, height=0.5*std + median-filter outlier rejection),
    but instead of computing windowed HRV features, returns the raw IBI series
    interpolated onto a uniform target_fs grid for downstream bandpassing.
    """
    if ecg_filtered is None or len(ecg_filtered) < 100:
        return None, None

    valid_samples = ecg_filtered[ecg_valid]
    if len(valid_samples) < 100:
        return None, None
    std_val = valid_samples.std()
    if std_val < 1e-6:
        return None, None

    peaks, _ = find_peaks(
        ecg_filtered,
        distance=int(0.4 * fs),
        height=0.5 * std_val,
    )
    if len(peaks) < 10:
        return None, None
    peak_times = ecg_ts[peaks]
    ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
    ibi_times = peak_times[1:]

    # Outlier rejection (same logic as preprocessors)
    if len(ibis) >= 5:
        from scipy.ndimage import median_filter
        local_median = median_filter(ibis, size=5, mode='reflect')
        outlier = np.abs(ibis - local_median) / np.maximum(local_median, 0.3) > 0.40
        false = set()
        for i in range(len(outlier) - 1):
            if outlier[i] and outlier[i + 1]:
                false.add(i + 1)
        for i in range(len(ibis)):
            if ibis[i] < 0.6 * local_median[i]:
                false.add(i + 1)
        if false:
            keep = [i for i in range(len(peaks)) if i not in false]
            peaks = peaks[keep]
            peak_times = ecg_ts[peaks]
            if len(peaks) < 5:
                return None, None
            ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
            ibi_times = peak_times[1:]

    # Resample to uniform grid
    t_grid = np.arange(ecg_ts[0], ecg_ts[-1], 1.0 / target_fs)
    ibi_grid = np.interp(t_grid, ibi_times, ibis,
                         left=ibis[0], right=ibis[-1])
    return ibi_grid.astype(np.float32), t_grid


def _bandpass_ibi(ibi_grid, fs_grid, band, device=None):
    """FFT bandpass on a 1-D IBI signal."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.as_tensor(ibi_grid[None, :], dtype=torch.float32, device=device)
    out = _fft_bandpass(x, fs_grid, band, device=device)
    return out.squeeze(0).cpu().numpy()


# ───────────────────────────────────────────────────────────────────────
#  S-map per band
# ───────────────────────────────────────────────────────────────────────

def _band_smap(p1_sig_lib, p2_sig_lib, fs_lib, label,
               tau_quarter_cycle_s, theiler_s,
               n_surrogates):
    """One (modality, band) S-map block: simplex E + S-map + condition stats."""
    tau_samples = max(1, int(round(tau_quarter_cycle_s * fs_lib)))
    delta_samples = tau_samples
    theiler_samples = max(1, int(round(theiler_s * fs_lib)))

    print(f'  [{label}] simplex (fs_lib={fs_lib:.1f} Hz, '
          f'τ={tau_samples}, theiler={theiler_samples})...', flush=True)
    sel = simplex_select_E(p1_sig_lib,
                           tau=tau_samples, delta=delta_samples,
                           E_grid=(2, 3, 4, 5),
                           theiler_samples=theiler_samples)
    E_best = sel['E_best']
    print(f'    skill_per_E = '
          f'{ {k: round(v, 3) for k, v in sel["skill_per_E"].items()} } '
          f'-> E={E_best}')

    print(f'  [{label}] S-map (E={E_best}, n_surr={n_surrogates})...',
          flush=True)
    t0 = time.time()
    res = gpu_smap_cross_coupling(
        p1_sig_lib, p2_sig_lib,
        fs=fs_lib, E=E_best,
        tau_samples=tau_samples, delta_samples=delta_samples,
        theta=1.0, theiler_samples=theiler_samples,
        query_stride=2,
        batch_size=512, lib_chunk=65536,
        n_surrogates=n_surrogates, seed=42,
    )
    elapsed = time.time() - t0
    print(f'    {elapsed:.0f}s, '
          f'pred skill fwd={res["pred_skill_forward"]:.3f}, '
          f'rev={res["pred_skill_reverse"]:.3f}, '
          f'cond p99={np.nanpercentile(res["cond_forward"], 99):.1e}, '
          f'N_q={len(res["t_query"])}')
    return res, fs_lib


# ───────────────────────────────────────────────────────────────────────
#  Per-condition stats
# ───────────────────────────────────────────────────────────────────────

def _per_condition_stats(t_axis, values, segments):
    out = {}
    for cond, t0, t1 in segments:
        m = (t_axis >= t0) & (t_axis <= t1) & np.isfinite(values)
        if m.sum() == 0:
            out[cond] = {'mean': None, 'std': None, 'n': 0}
        else:
            out[cond] = {'mean': float(values[m].mean()),
                         'std': float(values[m].std()),
                         'n': int(m.sum())}
    return out


def _cohens_d(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    s = np.sqrt(((a.var(ddof=1) * (len(a) - 1)
                  + b.var(ddof=1) * (len(b) - 1))
                 / (len(a) + len(b) - 2)))
    if s < 1e-12:
        return float('nan')
    return float((a.mean() - b.mean()) / s)


def _vals_in_segment(t_axis, values, segments, key):
    for cond, t0, t1 in segments:
        if cond == key:
            m = (t_axis >= t0) & (t_axis <= t1)
            return values[m]
    return np.array([])


# ───────────────────────────────────────────────────────────────────────
#  Plot
# ───────────────────────────────────────────────────────────────────────

def _shade(ax, segments):
    for cond, t0, t1 in segments:
        c = CONDITION_COLORS.get(cond, '#cccccc')
        ax.axvspan(t0, t1, color=c, alpha=0.12, zorder=0)


def _plot_band_row(axes, sig_p1, sig_p2, fs_lib, smap_real, smap_pseudo,
                   segments, label, t_origin=0.0):
    """4 panels for one (modality, band) row."""
    # Col 1: signal
    ax = axes[0]
    _shade(ax, segments)
    n = min(len(sig_p1), len(sig_p2))
    t_sig = np.arange(n) / fs_lib + t_origin
    decim = max(1, int(fs_lib / 50))
    ax.plot(t_sig[::decim], sig_p1[:n:decim], color='#1f77b4',
            lw=0.5, alpha=0.6, label='T')
    ax.plot(t_sig[::decim], sig_p2[:n:decim], color='#d62728',
            lw=0.5, alpha=0.6, label='P')
    ax.set_ylabel(f'{label}\nsignal')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # Col 2: β summary (coupling at lag τ)
    ax = axes[1]
    _shade(ax, segments)
    t_q = smap_real['t_query'] / fs_lib + t_origin
    ax.plot(t_q, smap_real['coupling_p1_to_p2'], color='#1f77b4',
            lw=0.5, alpha=0.7, label='β T→P')
    ax.plot(t_q, smap_real['coupling_p2_to_p1'], color='#d62728',
            lw=0.5, alpha=0.7, label='β P→T')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('β @ lag τ')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # Col 3: surrogate z + pseudo
    ax = axes[2]
    _shade(ax, segments)
    ax.axhline(2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(-2, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    if smap_pseudo is not None:
        t_p = smap_pseudo['t_query'] / fs_lib + t_origin
        ax.plot(t_p, smap_pseudo['z_p1_to_p2'], color='gray',
                lw=0.5, alpha=0.5)
        ax.plot(t_p, smap_pseudo['z_p2_to_p1'], color='lightgray',
                lw=0.5, alpha=0.5)
    ax.plot(t_q, smap_real['z_p1_to_p2'], color='#1f77b4',
            lw=0.6, alpha=0.85, label='z T→P')
    ax.plot(t_q, smap_real['z_p2_to_p1'], color='#d62728',
            lw=0.6, alpha=0.85, label='z P→T')
    ax.set_ylabel('z')
    ax.set_ylim(-6, 6)
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # Col 4: full Jacobian heatmap (E lags × time, T→P direction)
    ax = axes[3]
    beta_full = smap_real['beta_p1_to_p2']                    # (N_q, E)
    E = beta_full.shape[1]
    extent = [t_q[0], t_q[-1], 0, E]
    vmax = np.nanpercentile(np.abs(beta_full), 98)
    if vmax < 1e-12:
        vmax = 1e-3
    im = ax.imshow(beta_full.T, aspect='auto', origin='lower',
                   extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_ylabel(f'lag-bin\n0…{E-1}')
    ax.set_yticks(np.arange(E) + 0.5)
    ax.set_yticklabels([f'τ·{k}' for k in range(E)])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


def _plot_p2(out_path, real_blocks, pseudo_blocks, segments, session, pseudo,
             summary_table):
    n_rows = len(real_blocks)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.0 * n_rows + 1),
                              sharex=True)
    fig.suptitle(
        f'Native-rate coupling P2 — real {session} | pseudo-dyad {pseudo}\n'
        f'ROI = frontal AF3+F7+F3+F4+F8+AF4 (avg)   '
        f'EEG bands θ/α/β   ECG LF+HF',
        fontsize=11, y=0.99)

    for row_idx, (label, blk) in enumerate(real_blocks.items()):
        psblk = pseudo_blocks.get(label)
        ax_row = axes[row_idx] if n_rows > 1 else axes
        _plot_band_row(ax_row, blk['sig_p1'], blk['sig_p2'], blk['fs_lib'],
                       blk['smap'], psblk['smap'] if psblk else None,
                       segments, label)
    if n_rows > 1:
        axes[-1, 0].set_xlabel('Session time (s)')
        axes[-1, 1].set_xlabel('Session time (s)')
        axes[-1, 2].set_xlabel('Session time (s)')
        axes[-1, 3].set_xlabel('Session time (s)')

    # Inset summary text
    table_lines = ['(modality, band) | E | d T→P | d P→T | pseudo z T→P | pseudo z P→T']
    for label, row in summary_table.items():
        table_lines.append(
            f'{label:18s} | {row["E"]} | '
            f'{row["d_tp"]:+.2f} | {row["d_pt"]:+.2f} | '
            f'{row["pseudo_z_tp"]:+.2f} | {row["pseudo_z_pt"]:+.2f}')
    fig.text(0.01, 0.005, '\n'.join(table_lines),
             fontsize=8, family='monospace',
             verticalalignment='bottom')

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────

def run(session_name='y_06', pseudo_name='y_17',
        out_dir='results/native_rate_coupling_p2',
        n_surrogates=50, skip_pseudo=False, bands_eeg=None, bands_ecg=None):
    print('=' * 72)
    print(f' P2: Native-rate coupling — real={session_name}, pseudo={pseudo_name}')
    print('=' * 72)
    if bands_eeg is None:
        bands_eeg = list(EEG_BANDS.keys())
    if bands_ecg is None:
        bands_ecg = list(ECG_BANDS.keys())
    config = load_config()
    os.makedirs(out_dir, exist_ok=True)

    # ── Load real session ───────────────────────────────────────────
    print(f'\n[1] Loading real session {session_name}...')
    real_session, real_cached = _load_session(session_name, config)
    p1_role = real_session.get('p1_role', 'unknown')
    p2_role = real_session.get('p2_role', 'unknown')
    print(f'  p1={p1_role}, p2={p2_role}')

    p1_eeg = real_cached['p1_eeg'].astype(np.float64)
    p2_eeg = real_cached['p2_eeg'].astype(np.float64)
    p1_eeg_ts = real_cached['p1_eeg_ts']
    n_eeg = min(p1_eeg.shape[0], p2_eeg.shape[0])
    p1_eeg = p1_eeg[:n_eeg, :14]
    p2_eeg = p2_eeg[:n_eeg, :14]
    fs_eeg = float(np.round(1.0 / np.median(np.diff(p1_eeg_ts[:1000]))))

    # Establish session-relative time frame (markers are absolute LSL)
    landmarks = real_session.get('landmarks', {})
    if 'P1' in landmarks and 'P2' in landmarks:
        lsl_to_rel = float(min(landmarks['P1'][0][0], landmarks['P2'][0][0]))
    elif 'P1' in landmarks:
        lsl_to_rel = float(landmarks['P1'][0][0])
    else:
        lsl_to_rel = float(p1_eeg_ts[0])

    markers = real_session['markers']
    segments = []
    for cond in CONDITION_ORDER:
        t0 = markers.get(f'{cond}_start')
        t1 = markers.get(f'{cond}_stop')
        if t0 is not None and t1 is not None and t1 > t0:
            segments.append((cond, float(t0) - lsl_to_rel,
                                   float(t1) - lsl_to_rel))
    print(f'  Segments: {[s[0] for s in segments]}')

    # ── EEG ROI-averaged bandpass per band, real ────────────────────
    print('\n[2] EEG ROI-averaged bandpass (frontal pool) per band...')
    eeg_band_signals_real = {}
    for band_name in bands_eeg:
        band_hz = EEG_BANDS[band_name]
        print(f'  {band_name} {band_hz}...', flush=True)
        p1_band = _roi_average_band(p1_eeg, fs_eeg, band_hz, FRONTAL_ROI)
        p2_band = _roi_average_band(p2_eeg, fs_eeg, band_hz, FRONTAL_ROI)
        # Decimate to band-appropriate library rate
        fs_lib = EEG_BAND_LIB_RATES[band_name]
        p1_lib, fs_lib = _decimate(p1_band, fs_eeg, fs_lib)
        p2_lib, _ = _decimate(p2_band, fs_eeg, fs_lib)
        eeg_band_signals_real[band_name] = (p1_lib, p2_lib, fs_lib)
        print(f'    decimated -> {fs_lib:.0f} Hz, N_lib={len(p1_lib)}')

    # ── ECG IBI extraction, real ────────────────────────────────────
    print('\n[3] ECG IBI extraction (R-peak detection + uniform 4 Hz grid)...')
    if 'p1_ecg' in real_cached and 'p2_ecg' in real_cached:
        p1_ibi, t_ibi_p1 = _extract_ibi_series(
            real_cached['p1_ecg'], real_cached['p1_ecg_valid'],
            real_cached['p1_ecg_ts'])
        p2_ibi, t_ibi_p2 = _extract_ibi_series(
            real_cached['p2_ecg'], real_cached['p2_ecg_valid'],
            real_cached['p2_ecg_ts'])
        if p1_ibi is not None and p2_ibi is not None:
            n_ibi = min(len(p1_ibi), len(p2_ibi))
            p1_ibi = p1_ibi[:n_ibi]
            p2_ibi = p2_ibi[:n_ibi]
            # ECG cache stores ecg_ts in session-relative time (starts ~0).
            # Use that directly as the origin for stats — same convention as EEG.
            ibi_t_offset = float(min(t_ibi_p1[0], t_ibi_p2[0]))
            print(f'  IBI series: N={n_ibi} samples at {FS_HRV_GRID} Hz, '
                  f'origin (session-rel)={ibi_t_offset:+.2f}s')
        else:
            print('  WARNING: IBI extraction failed for one or both participants')
            p1_ibi = p2_ibi = None
            ibi_t_offset = 0.0
    else:
        print('  WARNING: ECG missing in cache')
        p1_ibi = p2_ibi = None
        ibi_t_offset = 0.0

    ecg_band_signals_real = {}
    if p1_ibi is not None:
        for band_name in bands_ecg:
            band_hz = ECG_BANDS[band_name]
            p1_b = _bandpass_ibi(p1_ibi, FS_HRV_GRID, band_hz)
            p2_b = _bandpass_ibi(p2_ibi, FS_HRV_GRID, band_hz)
            ecg_band_signals_real[f'ecg_{band_name}'] = (
                p1_b.astype(np.float32), p2_b.astype(np.float32),
                FS_HRV_GRID)

    # ── Load pseudo and prepare same band signals ───────────────────
    eeg_band_signals_pseudo = {}
    ecg_band_signals_pseudo = {}
    if not skip_pseudo:
        print(f'\n[4] Loading pseudo session {pseudo_name}...')
        try:
            pseudo_session, pseudo_cached = _load_session(pseudo_name, config)
            pseudo_p1_role = pseudo_session.get('p1_role', 'unknown')
            pseudo_patient_eeg_key = ('p1_eeg' if pseudo_p1_role == 'patient'
                                       else 'p2_eeg')
            pseudo_patient_eeg = pseudo_cached[pseudo_patient_eeg_key].astype(np.float64)

            for band_name in bands_eeg:
                band_hz = EEG_BANDS[band_name]
                fs_lib = EEG_BAND_LIB_RATES[band_name]
                # ROI-averaged pseudo patient band
                p2_band_pseudo = _roi_average_band(
                    pseudo_patient_eeg[:, :14], fs_eeg, band_hz, FRONTAL_ROI)
                # Real therapist (P2 in y_06 was therapist)
                if p1_role == 'therapist':
                    p1_band_real = _roi_average_band(p1_eeg, fs_eeg, band_hz, FRONTAL_ROI)
                else:
                    p1_band_real = _roi_average_band(p2_eeg, fs_eeg, band_hz, FRONTAL_ROI)
                # Use the lesser of the two lengths
                n_use = min(len(p1_band_real), len(p2_band_pseudo))
                p1_lib_pseudo, _ = _decimate(p1_band_real[:n_use], fs_eeg, fs_lib)
                p2_lib_pseudo, _ = _decimate(p2_band_pseudo[:n_use], fs_eeg, fs_lib)
                n_use_lib = min(len(p1_lib_pseudo), len(p2_lib_pseudo))
                eeg_band_signals_pseudo[band_name] = (
                    p1_lib_pseudo[:n_use_lib], p2_lib_pseudo[:n_use_lib], fs_lib)

            # Pseudo ECG
            pseudo_patient_ecg_key = ('p1_ecg' if pseudo_p1_role == 'patient'
                                       else 'p2_ecg')
            if (pseudo_patient_ecg_key in pseudo_cached
                    and 'p1_ecg' in real_cached and 'p2_ecg' in real_cached):
                ps_ibi, _ = _extract_ibi_series(
                    pseudo_cached[pseudo_patient_ecg_key],
                    pseudo_cached[pseudo_patient_ecg_key.replace('_ecg', '_ecg_valid')],
                    pseudo_cached[pseudo_patient_ecg_key.replace('_ecg', '_ecg_ts')])
                # therapist IBI from real session
                therapist_ecg_key = 'p2_ecg' if p1_role == 'patient' else 'p1_ecg'
                tibi, _ = _extract_ibi_series(
                    real_cached[therapist_ecg_key],
                    real_cached[therapist_ecg_key.replace('_ecg', '_ecg_valid')],
                    real_cached[therapist_ecg_key.replace('_ecg', '_ecg_ts')])
                if ps_ibi is not None and tibi is not None:
                    n_use = min(len(ps_ibi), len(tibi))
                    for band_name in bands_ecg:
                        band_hz = ECG_BANDS[band_name]
                        p1_b = _bandpass_ibi(tibi[:n_use], FS_HRV_GRID, band_hz)
                        p2_b = _bandpass_ibi(ps_ibi[:n_use], FS_HRV_GRID, band_hz)
                        ecg_band_signals_pseudo[f'ecg_{band_name}'] = (
                            p1_b.astype(np.float32), p2_b.astype(np.float32),
                            FS_HRV_GRID)
        except Exception as e:
            print(f'  Pseudo loading failed: {e!r}')

    # ── Run S-map per (modality, band) for real and pseudo ──────────
    real_blocks = {}
    pseudo_blocks = {}
    summary_table = {}

    def _run_one(label, p1_lib, p2_lib, fs_lib, tau_s, theiler_s,
                 t_origin_for_stats):
        res, _ = _band_smap(
            p1_lib, p2_lib, fs_lib, label,
            tau_quarter_cycle_s=tau_s, theiler_s=theiler_s,
            n_surrogates=n_surrogates,
        )
        # condition stats use session-relative time
        t_q_rel = res['t_query'] / fs_lib + t_origin_for_stats
        cs_tp = _per_condition_stats(t_q_rel, res['z_p1_to_p2'], segments)
        cs_pt = _per_condition_stats(t_q_rel, res['z_p2_to_p1'], segments)
        d_tp = _cohens_d(
            _vals_in_segment(t_q_rel, res['z_p1_to_p2'], segments, 'conv_1'),
            _vals_in_segment(t_q_rel, res['z_p1_to_p2'], segments, 'meditate_K'),
        )
        d_pt = _cohens_d(
            _vals_in_segment(t_q_rel, res['z_p2_to_p1'], segments, 'conv_1'),
            _vals_in_segment(t_q_rel, res['z_p2_to_p1'], segments, 'meditate_K'),
        )
        return res, cs_tp, cs_pt, d_tp, d_pt

    # EEG bands
    print('\n[5] S-map per EEG band (real)...')
    for band_name in bands_eeg:
        if band_name not in eeg_band_signals_real:
            continue
        p1_lib, p2_lib, fs_lib = eeg_band_signals_real[band_name]
        # ~quarter cycle of band center frequency
        band_hz = EEG_BANDS[band_name]
        center = 0.5 * (band_hz[0] + band_hz[1])
        tau_s = 1.0 / (4 * center)              # quarter cycle
        theiler_s = 3.0 / (band_hz[1] - band_hz[0])  # 3x bandwidth-1
        label = f'eeg_{band_name}'
        res, cs_tp, cs_pt, d_tp, d_pt = _run_one(
            label, p1_lib, p2_lib, fs_lib, tau_s, theiler_s,
            t_origin_for_stats=float(p1_eeg_ts[0]))
        real_blocks[label] = dict(sig_p1=p1_lib, sig_p2=p2_lib,
                                  fs_lib=fs_lib, smap=res)
        summary_table[label] = dict(
            E=int(res['E']), d_tp=float(d_tp) if not np.isnan(d_tp) else 0.0,
            d_pt=float(d_pt) if not np.isnan(d_pt) else 0.0,
            pseudo_z_tp=0.0, pseudo_z_pt=0.0,
            cs_tp=cs_tp, cs_pt=cs_pt,
            pred_skill_fwd=float(res['pred_skill_forward']),
            pred_skill_rev=float(res['pred_skill_reverse']),
        )
        print(f'  {label}: d_tp={d_tp:+.2f}, d_pt={d_pt:+.2f}')

    # ECG bands
    print('\n[6] S-map per ECG band (real)...')
    for band_name in bands_ecg:
        label = f'ecg_{band_name}'
        if label not in ecg_band_signals_real:
            continue
        p1_lib, p2_lib, fs_lib = ecg_band_signals_real[label]
        band_hz = ECG_BANDS[band_name]
        center = 0.5 * (band_hz[0] + band_hz[1])
        tau_s = 1.0 / (4 * center)
        theiler_s = 3.0 / (band_hz[1] - band_hz[0])
        res, cs_tp, cs_pt, d_tp, d_pt = _run_one(
            label, p1_lib, p2_lib, fs_lib, tau_s, theiler_s,
            t_origin_for_stats=float(ibi_t_offset))
        real_blocks[label] = dict(sig_p1=p1_lib, sig_p2=p2_lib,
                                  fs_lib=fs_lib, smap=res)
        summary_table[label] = dict(
            E=int(res['E']), d_tp=float(d_tp) if not np.isnan(d_tp) else 0.0,
            d_pt=float(d_pt) if not np.isnan(d_pt) else 0.0,
            pseudo_z_tp=0.0, pseudo_z_pt=0.0,
            cs_tp=cs_tp, cs_pt=cs_pt,
            pred_skill_fwd=float(res['pred_skill_forward']),
            pred_skill_rev=float(res['pred_skill_reverse']),
        )
        print(f'  {label}: d_tp={d_tp:+.2f}, d_pt={d_pt:+.2f}')

    # Pseudo S-map
    if not skip_pseudo:
        print('\n[7] S-map per (modality, band) — pseudo-dyad...')
        for band_name in bands_eeg:
            if band_name not in eeg_band_signals_pseudo:
                continue
            p1_lib, p2_lib, fs_lib = eeg_band_signals_pseudo[band_name]
            band_hz = EEG_BANDS[band_name]
            center = 0.5 * (band_hz[0] + band_hz[1])
            tau_s = 1.0 / (4 * center)
            theiler_s = 3.0 / (band_hz[1] - band_hz[0])
            label = f'eeg_{band_name}'
            res, _, _, _, _ = _run_one(
                f'{label}/pseudo', p1_lib, p2_lib, fs_lib, tau_s, theiler_s,
                t_origin_for_stats=float(p1_eeg_ts[0]))
            pseudo_blocks[label] = dict(sig_p1=p1_lib, sig_p2=p2_lib,
                                         fs_lib=fs_lib, smap=res)
            if label in summary_table:
                summary_table[label]['pseudo_z_tp'] = float(np.nanmean(res['z_p1_to_p2']))
                summary_table[label]['pseudo_z_pt'] = float(np.nanmean(res['z_p2_to_p1']))
        for band_name in bands_ecg:
            label = f'ecg_{band_name}'
            if label not in ecg_band_signals_pseudo:
                continue
            p1_lib, p2_lib, fs_lib = ecg_band_signals_pseudo[label]
            band_hz = ECG_BANDS[band_name]
            center = 0.5 * (band_hz[0] + band_hz[1])
            tau_s = 1.0 / (4 * center)
            theiler_s = 3.0 / (band_hz[1] - band_hz[0])
            res, _, _, _, _ = _run_one(
                f'{label}/pseudo', p1_lib, p2_lib, fs_lib, tau_s, theiler_s,
                t_origin_for_stats=float(ibi_t_offset))
            pseudo_blocks[label] = dict(sig_p1=p1_lib, sig_p2=p2_lib,
                                         fs_lib=fs_lib, smap=res)
            if label in summary_table:
                summary_table[label]['pseudo_z_tp'] = float(np.nanmean(res['z_p1_to_p2']))
                summary_table[label]['pseudo_z_pt'] = float(np.nanmean(res['z_p2_to_p1']))

    # ── Print summary table ─────────────────────────────────────────
    print('\n' + '=' * 72)
    print(' P2 SUMMARY')
    print('=' * 72)
    print(f'{"label":12s} {"E":>2s} {"skill fwd":>9s} {"skill rev":>9s} '
          f'{"d T→P":>7s} {"d P→T":>7s} {"pseudo z T→P":>13s} {"pseudo z P→T":>13s}')
    for label, row in summary_table.items():
        print(f'{label:12s} {row["E"]:>2d} '
              f'{row["pred_skill_fwd"]:>9.3f} {row["pred_skill_rev"]:>9.3f} '
              f'{row["d_tp"]:>+7.2f} {row["d_pt"]:>+7.2f} '
              f'{row["pseudo_z_tp"]:>+13.2f} {row["pseudo_z_pt"]:>+13.2f}')
    print('=' * 72)

    # ── Plot ────────────────────────────────────────────────────────
    fig_path = os.path.join(out_dir, f'{session_name}_p2.png')
    _plot_p2(fig_path, real_blocks, pseudo_blocks, segments,
             session_name, pseudo_name, summary_table)
    print(f'\nFigure saved: {fig_path}')

    # ── Save data + summary JSON ────────────────────────────────────
    npz_path = os.path.join(out_dir, f'{session_name}_p2.npz')
    np.savez_compressed(npz_path,
        **{f'real_{k}_t_query': v['smap']['t_query']
           for k, v in real_blocks.items()},
        **{f'real_{k}_z_tp': v['smap']['z_p1_to_p2']
           for k, v in real_blocks.items()},
        **{f'real_{k}_z_pt': v['smap']['z_p2_to_p1']
           for k, v in real_blocks.items()},
        **{f'real_{k}_beta_tp': v['smap']['beta_p1_to_p2']
           for k, v in real_blocks.items()},
        segments=np.array(segments, dtype=object),
        lsl_to_rel=lsl_to_rel)
    print(f'Data saved: {npz_path}')

    json_path = os.path.join(out_dir, f'{session_name}_p2_summary.json')
    with open(json_path, 'w') as f:
        json.dump({
            'session': session_name, 'pseudo': pseudo_name,
            'n_surrogates': n_surrogates,
            'p1_role': p1_role,
            'p2_role': p2_role,
            'direction_p1_to_p2': f'{p1_role} -> {p2_role}',
            'direction_p2_to_p1': f'{p2_role} -> {p1_role}',
            'summary': summary_table,
            'segments_session_relative': [(c, t0, t1) for c, t0, t1 in segments],
        }, f, indent=2)
    print(f'Summary saved: {json_path}')
    return summary_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', default='y_06')
    p.add_argument('--pseudo', default='y_17')
    p.add_argument('--out-dir', default='results/native_rate_coupling_p2')
    p.add_argument('--n-surr', type=int, default=50)
    p.add_argument('--skip-pseudo', action='store_true')
    p.add_argument('--bands-eeg', default='theta,alpha,beta')
    p.add_argument('--bands-ecg', default='LF,HF')
    args = p.parse_args()
    bands_eeg = [b.strip() for b in args.bands_eeg.split(',') if b.strip()]
    bands_ecg = [b.strip() for b in args.bands_ecg.split(',') if b.strip()]
    run(session_name=args.session, pseudo_name=args.pseudo,
        out_dir=args.out_dir, n_surrogates=args.n_surr,
        skip_pseudo=args.skip_pseudo,
        bands_eeg=bands_eeg, bands_ecg=bands_ecg)


if __name__ == '__main__':
    main()
