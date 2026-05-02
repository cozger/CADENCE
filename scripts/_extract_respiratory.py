"""
ECG-Derived Respiration (EDR) extraction from Polar H10 130 Hz ECG.

Three methods:
  A. FMRR  — frequency modulation of RR intervals (timing only)
  B. AM    — R-wave amplitude modulation (lung volume → electrode distance)
  C. QRS slope — R-S downslope modulation (chest impedance → QRS morphology)

Fused via spectral-concentration weighting (Charlton 2016).

Usage:
  python scripts/_extract_respiratory.py --session y_06
  python scripts/_extract_respiratory.py --all
"""

import sys, os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, welch, hilbert
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# --- CADENCE imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cadence.data.alignment import discover_cached_sessions, _load_session_cache


# ---------------------------------------------------------------------------
# Condition marker parsing
# ---------------------------------------------------------------------------

def parse_conditions(cache_path):
    """Parse condition intervals from session JSON sidecar.

    Returns list of (name, t_start_relative, t_end_relative).
    """
    json_path = cache_path + '.json'
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        meta = json.load(f)

    markers = meta.get('markers', [])
    t_abs_start = meta.get('t_start_absolute', 0)

    # Collect start/stop pairs
    starts = {}
    conditions = []
    for ts_abs, label in markers:
        t_rel = ts_abs - t_abs_start
        if label.endswith('_start'):
            cond_name = label[:-6]  # strip '_start'
            starts[cond_name] = t_rel
        elif label.endswith('_stop'):
            cond_name = label[:-5]  # strip '_stop'
            if cond_name in starts:
                conditions.append((cond_name, starts[cond_name], t_rel))
                del starts[cond_name]  # take the first stop for each start

    return conditions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ECG_SRATE = 130          # Polar H10 native rate
EDR_SRATE = 4.0          # output respiratory waveform rate
RESP_BAND = (0.15, 0.4)  # 9-24 breaths/min (resting range, matches HF-HRV band)
BUTTER_ORDER = 4
SYNC_WINDOW_S = 30.0      # respiratory synchrony window
SYNC_STEP_S = 1.0
SYNC_MAX_LAG_S = 5.0

CACHE_DIR = "C:/Users/optilab/desktop/MCCT/session_cache"
OUT_DIR = "results/respiratory"


# ---------------------------------------------------------------------------
# R-peak detection (replicates extract_ecg_features logic)
# ---------------------------------------------------------------------------

def detect_rpeaks(ecg, ts, srate=ECG_SRATE):
    """Detect R-peaks and compute cleaned IBIs.

    Returns:
        peaks: indices into ecg array
        peak_times: timestamps of R-peaks
        peak_heights: ECG amplitude at each R-peak
        ibis: cleaned inter-beat intervals (len = len(peaks)-1)
        ibi_times: timestamps for IBIs (midpoint convention: at peak[1:])
    """
    std_val = np.std(ecg)
    if std_val < 1e-8:
        return None, None, None, None, None

    peaks, props = find_peaks(
        ecg,
        distance=int(0.4 * srate),
        height=0.5 * std_val,
    )
    if len(peaks) < 3:
        return None, None, None, None, None

    peak_times = ts[peaks]
    peak_heights = props['peak_heights']
    ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
    ibi_times = peak_times[1:]

    # Ectopic beat removal via median filter
    if len(ibis) >= 5:
        local_median = median_filter(ibis, size=5, mode='reflect')
        outlier_mask = np.abs(ibis - local_median) / np.maximum(local_median, 0.3) > 0.40
        false_peaks = set()
        for i in range(len(outlier_mask) - 1):
            if outlier_mask[i] and outlier_mask[i + 1]:
                false_peaks.add(i + 1)
        for i in range(len(ibis)):
            if ibis[i] < 0.6 * local_median[i]:
                false_peaks.add(i + 1)
        if false_peaks:
            keep = [i for i in range(len(peaks)) if i not in false_peaks]
            peaks = peaks[keep]
            peak_times = ts[peaks]
            peak_heights = props['peak_heights'][keep]
            ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
            ibi_times = peak_times[1:]
            if len(peaks) < 3:
                return None, None, None, None, None

    return peaks, peak_times, peak_heights, ibis, ibi_times


# ---------------------------------------------------------------------------
# Bandpass filter helper
# ---------------------------------------------------------------------------

def bandpass(signal, lo, hi, fs, order=BUTTER_ORDER):
    """Zero-phase Butterworth bandpass."""
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.9999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Three EDR methods
# ---------------------------------------------------------------------------

def edr_fmrr(ibis, ibi_times, t_out):
    """Method A: Frequency Modulation of RR intervals.

    Resample IBIs to uniform grid, bandpass to respiratory range.
    Uses only beat TIMING — works even without raw ECG waveform.
    """
    f = interp1d(ibi_times, ibis, kind='cubic', bounds_error=False,
                 fill_value=(ibis[0], ibis[-1]))
    ibi_uniform = f(t_out)
    return bandpass(ibi_uniform, *RESP_BAND, EDR_SRATE)


def edr_amplitude(ecg, peaks, peak_times, peak_heights, t_out):
    """Method B: R-wave Amplitude Modulation.

    Lung expansion changes electrode distance → R-wave amplitude varies
    with breathing cycle. Best on chest-mounted electrodes (Polar H10).
    """
    f = interp1d(peak_times, peak_heights, kind='cubic', bounds_error=False,
                 fill_value=(peak_heights[0], peak_heights[-1]))
    amp_uniform = f(t_out)
    return bandpass(amp_uniform, *RESP_BAND, EDR_SRATE)


def edr_qrs_slope(ecg, ts, peaks, peak_times, t_out, srate=ECG_SRATE):
    """Method C: QRS Downslope (R-S slope).

    Chest impedance changes with lung volume → QRS morphology changes.
    Best single-lead EDR method (Varon 2020).

    Computes slope from R-peak to S-point (minimum within 100ms after R).
    """
    slopes = []
    search_samples = int(0.1 * srate)  # 100ms after R

    for pk in peaks:
        s_end = min(pk + search_samples, len(ecg) - 1)
        if s_end <= pk:
            slopes.append(0.0)
            continue
        s_idx = pk + np.argmin(ecg[pk:s_end + 1])
        dt = (s_idx - pk) / srate
        if dt < 1e-6:
            slopes.append(0.0)
        else:
            slopes.append((ecg[pk] - ecg[s_idx]) / dt)

    slopes = np.array(slopes)
    f = interp1d(peak_times, slopes, kind='cubic', bounds_error=False,
                 fill_value=(slopes[0], slopes[-1]))
    slope_uniform = f(t_out)
    return bandpass(slope_uniform, *RESP_BAND, EDR_SRATE)


# ---------------------------------------------------------------------------
# Spectral concentration quality metric
# ---------------------------------------------------------------------------

def spectral_concentration(signal, fs, band=RESP_BAND):
    """Ratio of power in respiratory band to total power.

    Higher = cleaner respiratory signal = more weight in fusion.
    """
    nperseg = min(len(signal), int(32 * fs))  # 32s window
    if nperseg < int(4 * fs):
        return 0.0
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    total = np.trapezoid(psd, f)
    if total < 1e-12:
        return 0.0
    mask = (f >= band[0]) & (f <= band[1])
    in_band = np.trapezoid(psd[mask], f[mask])
    return in_band / total


# ---------------------------------------------------------------------------
# Multi-method fusion
# ---------------------------------------------------------------------------

def fuse_edr(waveforms, fs=EDR_SRATE):
    """Spectral-concentration-weighted fusion of multiple EDR waveforms.

    Args:
        waveforms: list of (N,) arrays (same length, same fs)
    Returns:
        fused: (N,) weighted average
        weights: per-method weights (sum to 1)
    """
    concentrations = [spectral_concentration(w, fs) for w in waveforms]
    total = sum(concentrations)
    if total < 1e-12:
        weights = [1.0 / len(waveforms)] * len(waveforms)
    else:
        weights = [c / total for c in concentrations]

    fused = np.zeros_like(waveforms[0])
    for w, wt in zip(waveforms, weights):
        fused += wt * (w / max(np.std(w), 1e-8))  # normalize amplitude before weighting
    return fused, weights


# ---------------------------------------------------------------------------
# Respiratory rate from waveform
# ---------------------------------------------------------------------------

def respiratory_rate_from_waveform(waveform, fs=EDR_SRATE, window_s=32.0, step_s=2.0):
    """Extract respiratory rate via spectral peak in sliding Welch windows.

    Much more robust than peak-to-peak estimation on noisy EDR signals.

    Returns:
        rate_bpm: (N,) respiratory rate at each sample (interpolated from windows)
        window_times: center times of estimation windows
        window_rates: rate at each window center
    """
    win_samples = int(window_s * fs)
    step_samples = int(step_s * fs)
    n = len(waveform)

    window_times = []
    window_rates = []

    for start in range(0, n - win_samples, step_samples):
        seg = waveform[start:start + win_samples]
        nperseg = min(len(seg), int(16 * fs))  # 16s sub-window for Welch
        if nperseg < int(4 * fs):
            continue
        f, psd = welch(seg, fs=fs, nperseg=nperseg)
        # Only look in respiratory band
        mask = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
        if not mask.any() or psd[mask].max() < 1e-12:
            continue
        peak_freq = f[mask][np.argmax(psd[mask])]
        window_times.append((start + win_samples / 2) / fs)
        window_rates.append(peak_freq * 60.0)  # Hz → BPM

    if len(window_times) < 2:
        return np.full(n, np.nan), np.array([]), np.array([])

    window_times = np.array(window_times)
    window_rates = np.array(window_rates)

    # Interpolate to full waveform
    t = np.arange(n) / fs
    rate_bpm = np.interp(t, window_times, window_rates,
                         left=window_rates[0], right=window_rates[-1])
    rate_bpm = np.clip(rate_bpm, 4, 40)
    return rate_bpm, window_times, window_rates


# ---------------------------------------------------------------------------
# Respiratory synchrony
# ---------------------------------------------------------------------------

def windowed_crosscorr(sig1, sig2, fs, window_s=SYNC_WINDOW_S,
                       step_s=SYNC_STEP_S, max_lag_s=SYNC_MAX_LAG_S):
    """Windowed cross-correlation between two signals.

    Returns:
        times: center time of each window
        peak_r: peak cross-correlation per window
        peak_lag: lag (seconds) at peak cross-correlation
    """
    win = int(window_s * fs)
    step = int(step_s * fs)
    max_lag = int(max_lag_s * fs)
    n = len(sig1)

    times, peak_rs, peak_lags = [], [], []
    for start in range(0, n - win, step):
        s1 = sig1[start:start + win]
        s2 = sig2[start:start + win]

        # Normalize
        s1 = (s1 - np.mean(s1))
        s2 = (s2 - np.mean(s2))
        std1, std2 = np.std(s1), np.std(s2)
        if std1 < 1e-8 or std2 < 1e-8:
            times.append((start + win / 2) / fs)
            peak_rs.append(0.0)
            peak_lags.append(0.0)
            continue

        # Cross-correlation at lags (proper Pearson r on each sub-slice)
        best_r, best_lag = 0.0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a = s1[lag:]
                b = s2[:len(a)]
            else:
                b = s2[-lag:]
                a = s1[:len(b)]
            if len(a) < 10:
                continue
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
            r = np.dot(a_c, b_c) / denom if denom > 1e-16 else 0.0
            if abs(r) > abs(best_r):
                best_r = r
                best_lag = lag

        times.append((start + win / 2) / fs)
        peak_rs.append(best_r)
        peak_lags.append(best_lag / fs)

    return np.array(times), np.array(peak_rs), np.array(peak_lags)


# ---------------------------------------------------------------------------
# Phase synchronization (PLV) for respiratory synchrony
# ---------------------------------------------------------------------------

N_SURROGATES = 200
MIN_SHIFT_FRAC = 0.1


def phase_sync(sig1, sig2, fs, window_s=15.0, step_s=1.0):
    """Compute windowed Phase Locking Value (PLV) between two respiratory signals.

    PLV measures phase stability: whether the two signals maintain a constant
    phase relationship. For synchronized breathing, PLV → 1. For independent
    breathing at similar rates, phase drifts → PLV → 0.

    Uses Hilbert transform for instantaneous phase extraction.

    Returns:
        times: center time of each window
        plv: PLV per window (0-1)
        mean_phase_diff: mean circular phase difference per window (radians)
    """
    # Instantaneous phase via Hilbert transform
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))

    win = int(window_s * fs)
    step = int(step_s * fs)
    n = len(sig1)

    times, plvs, phase_diffs = [], [], []
    for start in range(0, n - win, step):
        p1 = phase1[start:start + win]
        p2 = phase2[start:start + win]

        # Phase difference
        dphi = p1 - p2
        # PLV = |mean(exp(j * dphi))| — measures phase concentration
        plv = np.abs(np.mean(np.exp(1j * dphi)))
        mean_dphi = np.angle(np.mean(np.exp(1j * dphi)))

        times.append((start + win / 2) / fs)
        plvs.append(plv)
        phase_diffs.append(mean_dphi)

    return np.array(times), np.array(plvs), np.array(phase_diffs)


def _one_plv_surrogate(sig1, sig2, shift, fs, window_s, step_s):
    """Compute PLV for one circular-shifted surrogate."""
    sig2_shifted = np.roll(sig2, shift)
    _, plv, _ = phase_sync(sig1, sig2_shifted, fs, window_s, step_s)
    return plv


def surrogate_zscore_plv(sig1, sig2, fs, n_surrogates=N_SURROGATES,
                         window_s=15.0, step_s=1.0,
                         min_shift_frac=MIN_SHIFT_FRAC, seed=42, n_jobs=-1):
    """Surrogate z-scored Phase Locking Value for respiratory synchrony.

    1. Compute real PLV in sliding windows
    2. For K surrogates: circular-shift sig2, recompute PLV (parallel)
    3. Z-score: z = (PLV_real - mean(PLV_surr)) / std(PLV_surr)

    Returns:
        times: center time of each window
        z_scores: z-score per window
        raw_plv: raw PLV per window (0-1)
        phase_diff: mean phase difference per window (radians)
    """
    times, real_plv, phase_diff = phase_sync(sig1, sig2, fs, window_s, step_s)
    n_windows = len(times)

    if n_windows == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Generate shifts
    rng = np.random.RandomState(seed)
    n = len(sig2)
    min_shift = max(1, int(min_shift_frac * n))
    max_shift = n - min_shift
    shifts = rng.randint(min_shift, max_shift + 1, size=n_surrogates)

    # Parallel surrogate PLV computation
    surr_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_plv_surrogate)(sig1, sig2, int(s), fs, window_s, step_s)
        for s in shifts
    )

    surr_matrix = np.array(surr_results)  # (K, n_windows)
    surr_mean = surr_matrix.mean(axis=0)
    surr_std = np.maximum(surr_matrix.std(axis=0, ddof=1), 1e-8)

    z_scores = (real_plv - surr_mean) / surr_std

    return times, z_scores, real_plv, phase_diff


# ---------------------------------------------------------------------------
# Process one participant
# ---------------------------------------------------------------------------

def extract_respiratory_one(ecg, ts, srate=ECG_SRATE):
    """Extract respiratory signals from one participant's raw ECG.

    Returns dict with all EDR results or None if ECG is bad.
    """
    result = detect_rpeaks(ecg, ts, srate)
    if result[0] is None:
        return None
    peaks, peak_times, peak_heights, ibis, ibi_times = result

    # Uniform output grid at EDR_SRATE
    t_start, t_end = ts[0], ts[-1]
    t_out = np.arange(t_start, t_end, 1.0 / EDR_SRATE)

    # Three EDR methods
    fmrr = edr_fmrr(ibis, ibi_times, t_out)
    am = edr_amplitude(ecg, peaks, peak_times, peak_heights, t_out)
    qrs = edr_qrs_slope(ecg, ts, peaks, peak_times, t_out, srate)

    # Fuse
    fused, weights = fuse_edr([fmrr, am, qrs])

    # Respiratory rate from fused waveform (spectral peak method)
    rate_bpm, win_times, win_rates = respiratory_rate_from_waveform(fused)

    # Per-method respiratory rates (for agreement check)
    rate_fmrr, _, _ = respiratory_rate_from_waveform(fmrr)
    rate_am, _, _ = respiratory_rate_from_waveform(am)
    rate_qrs, _, _ = respiratory_rate_from_waveform(qrs)

    return {
        't': t_out,
        'fmrr': fmrr,
        'am': am,
        'qrs': qrs,
        'fused': fused,
        'weights': weights,
        'rate_bpm': rate_bpm,
        'rate_fmrr': rate_fmrr,
        'rate_am': rate_am,
        'rate_qrs': rate_qrs,
        'rate_window_times': win_times,
        'rate_window_rates': win_rates,
        'rpeak_indices': peaks,
        'rpeak_times': peak_times,
        'ibis': ibis,
    }


# ---------------------------------------------------------------------------
# Process one session
# ---------------------------------------------------------------------------

def process_session(session_name, cache_path, out_dir):
    """Process a single session: extract EDR for both participants, compute synchrony."""
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    session = _load_session_cache(cache_path)

    results = {}
    for p in ['p1', 'p2']:
        ecg_key = f'{p}_ecg'
        ts_key = f'{p}_ecg_ts'
        if ecg_key not in session or ts_key not in session:
            print(f"  {p}: no raw ECG data — skipping")
            results[p] = None
            continue

        ecg = session[ecg_key].astype(np.float64).ravel()
        ts = session[ts_key].astype(np.float64).ravel()
        print(f"  {p}: {len(ecg)} samples @ {ECG_SRATE} Hz = {len(ecg)/ECG_SRATE:.1f}s")

        res = extract_respiratory_one(ecg, ts)
        if res is None:
            print(f"  {p}: R-peak detection failed — skipping")
            results[p] = None
            continue

        mean_rate = np.nanmean(res['rate_bpm'])
        print(f"  {p}: {len(res['rpeak_indices'])} R-peaks, "
              f"mean RR = {mean_rate:.1f} bpm, "
              f"weights = FMRR:{res['weights'][0]:.2f} AM:{res['weights'][1]:.2f} QRS:{res['weights'][2]:.2f}")
        results[p] = res

    # Respiratory synchrony (surrogate z-scored PLV)
    sync_result = None
    if results.get('p1') is not None and results.get('p2') is not None:
        t1, t2 = results['p1']['t'], results['p2']['t']
        t_start = max(t1[0], t2[0])
        t_end = min(t1[-1], t2[-1])

        if t_end - t_start > 40:
            t_common = np.arange(t_start, t_end, 1.0 / EDR_SRATE)
            f1 = np.interp(t_common, t1, results['p1']['fused'])
            f2 = np.interp(t_common, t2, results['p2']['fused'])

            print(f"  Computing PLV + surrogate z-scores ({N_SURROGATES} surrogates, 15s windows)...")
            sync_t, sync_z, sync_plv, sync_phase = surrogate_zscore_plv(
                f1, f2, EDR_SRATE, n_surrogates=N_SURROGATES, window_s=15.0)
            sync_result = {
                'times': sync_t,
                'z_scores': sync_z,
                'plv': sync_plv,
                'phase_diff': sync_phase,
                'mean_z': float(np.mean(sync_z)),
                'mean_plv': float(np.mean(sync_plv)),
                'pct_sig': float(np.mean(sync_z > 1.96) * 100),
            }
            print(f"  Respiratory synchrony: mean z = {sync_result['mean_z']:.2f}, "
                  f"mean PLV = {sync_result['mean_plv']:.3f}, "
                  f"{sync_result['pct_sig']:.0f}% windows sig (z>1.96)")
        else:
            print(f"  Overlap too short for synchrony ({t_end-t_start:.1f}s)")

    # --- Per-condition respiratory synchrony (surrogate z-scored PLV) ---
    conditions = parse_conditions(cache_path)
    condition_sync = {}
    if conditions and results.get('p1') is not None and results.get('p2') is not None:
        t1, t2 = results['p1']['t'], results['p2']['t']

        print(f"\n  Per-condition respiratory synchrony (PLV, surrogate z-scored):")
        print(f"  {'Condition':<20} {'Duration':>8} {'mean z':>8} {'%sig':>6} {'PLV':>8} {'P1 RR':>8} {'P2 RR':>8}")
        print(f"  {'-'*70}")

        for cond_name, t_start_c, t_end_c in sorted(conditions, key=lambda x: x[1]):
            dur = t_end_c - t_start_c
            if dur < 40:
                continue

            mask1 = (t1 >= t_start_c) & (t1 < t_end_c)
            mask2 = (t2 >= t_start_c) & (t2 < t_end_c)
            if mask1.sum() < int(15 * EDR_SRATE) or mask2.sum() < int(15 * EDR_SRATE):
                continue

            seg1 = results['p1']['fused'][mask1]
            seg2 = results['p2']['fused'][mask2]
            min_len = min(len(seg1), len(seg2))
            seg1, seg2 = seg1[:min_len], seg2[:min_len]

            ct, cz, cplv, cphase = surrogate_zscore_plv(
                seg1, seg2, EDR_SRATE, n_surrogates=N_SURROGATES, window_s=15.0)
            if len(cz) == 0:
                continue

            rate1 = results['p1']['rate_bpm'][mask1]
            rate2 = results['p2']['rate_bpm'][mask2]

            cond_info = {
                'duration': float(dur),
                'mean_z': float(np.mean(cz)),
                'pct_sig': float(np.mean(cz > 1.96) * 100),
                'mean_plv': float(np.mean(cplv)),
                'p1_mean_rr': float(np.nanmean(rate1)),
                'p2_mean_rr': float(np.nanmean(rate2)),
            }
            condition_sync[cond_name] = cond_info

            print(f"  {cond_name:<20} {dur:>7.0f}s {cond_info['mean_z']:>8.2f} "
                  f"{cond_info['pct_sig']:>5.0f}% {cond_info['mean_plv']:>8.3f} "
                  f"{cond_info['p1_mean_rr']:>7.1f} {cond_info['p2_mean_rr']:>7.1f}")

    # --- Diagnostic plots ---
    session_dir = os.path.join(out_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    plot_diagnostics(session_name, session, results, sync_result, session_dir,
                     conditions=conditions)

    # --- Summary stats ---
    summary = {'session': session_name}
    for p in ['p1', 'p2']:
        if results.get(p) is not None:
            r = results[p]
            summary[f'{p}_mean_rr'] = float(np.nanmean(r['rate_bpm']))
            summary[f'{p}_std_rr'] = float(np.nanstd(r['rate_bpm']))
            summary[f'{p}_n_rpeaks'] = int(len(r['rpeak_indices']))
            summary[f'{p}_weights'] = [float(w) for w in r['weights']]
            # Method agreement: mean absolute difference between methods
            agreement = np.nanmean([
                np.nanmean(np.abs(r['rate_fmrr'] - r['rate_am'])),
                np.nanmean(np.abs(r['rate_fmrr'] - r['rate_qrs'])),
                np.nanmean(np.abs(r['rate_am'] - r['rate_qrs'])),
            ])
            summary[f'{p}_method_agreement_bpm'] = float(agreement)
        else:
            summary[f'{p}_mean_rr'] = None

    if sync_result is not None:
        summary['sync_mean_z'] = sync_result.get('mean_z', None)
        summary['sync_pct_sig'] = sync_result.get('pct_sig', None)
        summary['sync_mean_plv'] = sync_result.get('mean_plv', None)

    if condition_sync:
        summary['conditions'] = condition_sync

    # Save summary
    with open(os.path.join(session_dir, 'respiratory_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostics(session_name, session, results, sync_result, out_dir,
                     conditions=None):
    """Generate 6-panel diagnostic figure."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
    fig.suptitle(f'EDR Diagnostics: {session_name}', fontsize=14, fontweight='bold')

    colors = {'p1': '#2196F3', 'p2': '#F44336'}

    # Panel 1: Raw ECG excerpt with R-peaks (first 10s of P1)
    ax = axes[0]
    p = 'p1'
    if results.get(p) is not None:
        ecg = session[f'{p}_ecg'].ravel()
        ts = session[f'{p}_ecg_ts'].ravel()
        t0 = ts[0]
        mask = (ts - t0) < 10.0
        ax.plot(ts[mask] - t0, ecg[mask], 'k-', linewidth=0.5, alpha=0.7)
        rpk = results[p]['rpeak_times']
        rpk_mask = (rpk - t0) < 10.0
        rpk_in = rpk[rpk_mask]
        rpk_idx = np.searchsorted(ts, rpk_in)
        rpk_idx = np.clip(rpk_idx, 0, len(ecg) - 1)
        ax.plot(rpk_in - t0, ecg[rpk_idx], 'rv', markersize=6)
    ax.set_title('Panel 1: Raw ECG (P1, first 10s) with R-peaks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ECG (z-scored)')

    # Panel 2: Three EDR waveforms overlaid (60s segment, P1)
    ax = axes[1]
    if results.get('p1') is not None:
        r = results['p1']
        t = r['t'] - r['t'][0]
        mask = t < 60.0
        for name, sig, color, ls in [
            ('FMRR', r['fmrr'], '#4CAF50', '-'),
            ('AM', r['am'], '#FF9800', '--'),
            ('QRS slope', r['qrs'], '#9C27B0', ':'),
        ]:
            normed = sig / max(np.std(sig), 1e-8)
            ax.plot(t[mask], normed[mask], color=color, linestyle=ls,
                    linewidth=1.2, label=f'{name} (w={r["weights"][["FMRR","AM","QRS slope"].index(name)]:.2f})')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Panel 2: Three EDR methods (P1, first 60s, normalized)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')

    # Panel 3: Fused respiratory waveform, both participants
    ax = axes[2]
    for p in ['p1', 'p2']:
        if results.get(p) is not None:
            r = results[p]
            t = r['t'] - r['t'][0]
            ax.plot(t, r['fused'] / max(np.std(r['fused']), 1e-8),
                    color=colors[p], linewidth=0.5, alpha=0.7, label=p.upper())
    ax.legend(loc='upper right')
    ax.set_title('Panel 3: Fused respiratory waveform (full session)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')

    # Panel 4: Respiratory rate timecourse
    ax = axes[3]
    for p in ['p1', 'p2']:
        if results.get(p) is not None:
            r = results[p]
            t = r['t'] - r['t'][0]
            ax.plot(t, r['rate_bpm'], color=colors[p], linewidth=0.8,
                    alpha=0.7, label=p.upper())
    ax.set_ylim(4, 30)
    ax.axhline(12, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(20, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_title('Panel 4: Respiratory rate (breaths/min)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BPM')

    # Panel 5: Surrogate z-scored respiratory synchrony with condition shading
    ax = axes[4]
    cond_colors = {
        'base_EO': '#BDBDBD', 'base_EC': '#9E9E9E',
        'conv_1': '#BBDEFB', 'conv_2': '#90CAF9',
        'meditate_B': '#C8E6C9', 'meditate_K': '#A5D6A7',
    }
    if conditions:
        for cond_name, t_start_c, t_end_c in conditions:
            color = cond_colors.get(cond_name, '#E0E0E0')
            ax.axvspan(t_start_c, t_end_c, alpha=0.2, color=color, label=cond_name)
    if sync_result is not None and 'z_scores' in sync_result:
        z = sync_result['z_scores']
        ax.plot(sync_result['times'], z, 'k-', linewidth=0.8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axhline(1.96, color='red', linewidth=0.8, linestyle='--', alpha=0.7, label='p=0.05')
        ax.fill_between(sync_result['times'], z, 0,
                        where=z > 1.96, alpha=0.3, color='green')
        ax.fill_between(sync_result['times'], z, 0,
                        where=(z > 0) & (z <= 1.96), alpha=0.1, color='green')
        ax.fill_between(sync_result['times'], z, 0,
                        where=z < 0, alpha=0.1, color='red')
        ax.set_title(f'Panel 5: Respiratory synchrony z-score (mean z={sync_result["mean_z"]:.2f}, '
                     f'{sync_result["pct_sig"]:.0f}% sig)')
    elif sync_result is not None:
        ax.plot(sync_result['times'], sync_result['peak_r'], 'k-', linewidth=0.8)
        ax.set_title(f'Panel 5: Respiratory synchrony (raw r)')
    else:
        ax.set_title('Panel 5: Respiratory synchrony (N/A)')
    if conditions:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
        if unique:
            ax.legend(*zip(*unique), loc='upper right', fontsize=7, ncol=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')

    # Panel 6: Method agreement scatter (P1)
    ax = axes[5]
    if results.get('p1') is not None:
        r = results['p1']
        # Subsample for readability
        step = max(1, len(r['rate_fmrr']) // 500)
        ax.scatter(r['rate_fmrr'][::step], r['rate_am'][::step],
                   s=3, alpha=0.3, color='#FF9800', label='AM vs FMRR')
        ax.scatter(r['rate_fmrr'][::step], r['rate_qrs'][::step],
                   s=3, alpha=0.3, color='#9C27B0', label='QRS vs FMRR')
        lims = [4, 30]
        ax.plot(lims, lims, 'k--', linewidth=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Panel 6: Method agreement (P1 respiratory rate)')
    ax.set_xlabel('FMRR rate (bpm)')
    ax.set_ylabel('Other method rate (bpm)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved diagnostics to {out_dir}/diagnostics.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ECG-Derived Respiration extraction')
    parser.add_argument('--session', type=str, help='Single session name (e.g. y_06)')
    parser.add_argument('--all', action='store_true', help='Process all cached sessions')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    sessions = discover_cached_sessions(CACHE_DIR)
    print(f"Found {len(sessions)} cached sessions")

    if args.session:
        matches = [(n, p) for n, p in sessions if n == args.session]
        if not matches:
            print(f"Session '{args.session}' not found. Available: {[n for n,_ in sessions]}")
            return
        sessions = matches
    elif not args.all:
        print("Specify --session <name> or --all")
        return

    all_summaries = []
    for name, path in sessions:
        try:
            summary = process_session(name, path, OUT_DIR)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Global summary table
    if all_summaries:
        print(f"\n{'='*80}")
        print(f"{'Session':<25} {'P1 RR':>7} {'P2 RR':>7} "
              f"{'mean z':>8} {'%sig':>6} {'PLV':>8} {'P1 agree':>9}")
        print(f"{'='*80}")
        for s in all_summaries:
            p1_rr = f"{s['p1_mean_rr']:.1f}" if s.get('p1_mean_rr') else 'N/A'
            p2_rr = f"{s['p2_mean_rr']:.1f}" if s.get('p2_mean_rr') else 'N/A'
            mean_z = f"{s['sync_mean_z']:.2f}" if s.get('sync_mean_z') is not None else 'N/A'
            pct_sig = f"{s['sync_pct_sig']:.0f}%" if s.get('sync_pct_sig') is not None else 'N/A'
            plv = f"{s['sync_mean_plv']:.3f}" if s.get('sync_mean_plv') is not None else 'N/A'
            agree = f"{s['p1_method_agreement_bpm']:.1f}" if s.get('p1_method_agreement_bpm') else 'N/A'
            print(f"{s['session']:<25} {p1_rr:>7} {p2_rr:>7} "
                  f"{mean_z:>8} {pct_sig:>6} {plv:>8} {agree:>9}")

        # Save global summary
        with open(os.path.join(OUT_DIR, 'all_sessions_summary.json'), 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved global summary to {OUT_DIR}/all_sessions_summary.json")


if __name__ == '__main__':
    main()
