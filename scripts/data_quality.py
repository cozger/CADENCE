"""Data quality assessment across sessions (v2).

Computes per-session, per-modality quality metrics grounded in what
empirically separates coupled from null sessions. Compares against
a benchmark session (default: y_32).

Metrics:
- Gap fraction (fraction of recording with timestamp discontinuities)
- Valid sample rate (from preprocessor validity masks)
- Activity variance (std of activity channel — low = static session)
- Dynamics CV (CV of windowed RMS — higher = more dynamic, generally good)
- Autocorrelation at coupling-relevant lags (1s, 5s)
- Cross-participant max correlation (direct coupling signal measure)
- Tracking smoothness (blendshapes only — jerk detects tracking drops)
- Effective duration (usable recording time after gap/validity losses)

Usage:
    python scripts/data_quality.py [--benchmark y_32] [--output results/data_quality]
"""

import argparse
import json
import os
import sys

import numpy as np

from cadence.config import load_config
from cadence.data import discover_cached_sessions, load_session_from_cache
from cadence.preprocess.common import compute_activity_channel


# ---------------------------------------------------------------------------
# V2 pipeline modalities (what CADENCE actually uses)
# ---------------------------------------------------------------------------

MODALITIES = [
    {
        'name': 'EEG wavelet',
        'data_key': '{p}_eeg_wavelet',
        'valid_key': '{p}_eeg_wavelet_valid',
        'ts_key': '{p}_eeg_wavelet_ts',
        'native_hz': 5,
        'activity_ch': None,  # compute on the fly
    },
    {
        'name': 'ECG features',
        'data_key': '{p}_ecg_features_v2',
        'valid_key': '{p}_ecg_features_v2_valid',
        'ts_key': '{p}_ecg_features_v2_ts',
        'native_hz': 2,
        'activity_ch': None,
    },
    {
        'name': 'Blendshapes v2',
        'data_key': '{p}_blendshapes_v2',
        'valid_key': '{p}_blendshapes_v2_valid',
        'ts_key': '{p}_blendshapes_v2_ts',
        'native_hz': 30,
        'activity_ch': 30,  # last channel (index 30 of 31)
    },
    {
        'name': 'Pose features',
        'data_key': '{p}_pose_features',
        'valid_key': '{p}_pose_features_valid',
        'ts_key': '{p}_pose_features_ts',
        'native_hz': 12,
        'activity_ch': 40,  # last channel (index 40 of 41)
    },
]

# For cross-participant correlation
CROSS_MODALITIES = [
    'EEG wavelet', 'ECG features', 'Blendshapes v2', 'Pose features',
]

# Quality thresholds (empirically derived from n=8 sessions)
THRESHOLDS = {
    'pose_gap_critical': 0.10,
    'bl_gap_critical': 0.10,
    'eff_dur_critical': 1200,      # 20 min
    'pose_gap_warning': 0.03,
    'bl_gap_warning': 0.03,
    'eff_dur_warning': 2400,       # 40 min
    'activity_var_low': 0.05,
    'eeg_valid_warning': 0.95,
    'xcorr_high': 0.30,
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def gap_analysis(ts, expected_hz):
    """Analyze timestamp gaps.

    Returns:
        max_gap_s: largest inter-sample interval
        n_gaps: number of gaps > 2.5x expected interval
        gap_frac: fraction of total duration lost to gaps
    """
    if len(ts) < 2:
        return 0.0, 0, 0.0
    dt = np.diff(ts)
    expected_dt = 1.0 / expected_hz
    threshold = expected_dt * 2.5
    gaps = dt[dt > threshold]
    total_gap = float(np.sum(gaps - expected_dt)) if len(gaps) > 0 else 0.0
    total_dur = float(ts[-1] - ts[0])
    return (
        float(np.max(dt)) if len(dt) > 0 else 0.0,
        int(len(gaps)),
        float(total_gap / total_dur) if total_dur > 0 else 0.0,
    )


def dynamics_cv(x, window_sec, hz):
    """Coefficient of variation of windowed RMS.

    Higher = more dynamic variation over time. For coupling detection,
    moderate-to-high CV is generally better (indicates dynamic interaction).
    """
    w = max(1, int(window_sec * hz))
    n = len(x)
    if n < w * 2:
        return np.nan
    n_windows = n // w
    rms_vals = []
    for i in range(n_windows):
        chunk = x[i * w:(i + 1) * w]
        rms = np.sqrt(np.nanmean(chunk ** 2))
        if not np.isnan(rms) and rms > 0:
            rms_vals.append(rms)
    if len(rms_vals) < 2:
        return np.nan
    rms_arr = np.array(rms_vals)
    return float(np.std(rms_arr) / np.mean(rms_arr))


def activity_variance(data, activity_ch_idx, valid, hz):
    """Standard deviation of the activity channel.

    Low activity variance = static session = nothing for CADENCE to detect.
    Uses the pre-computed activity channel if available, otherwise computes it.

    Args:
        data: (N, C) feature array
        activity_ch_idx: index of activity channel, or None to compute
        valid: (N,) boolean validity mask, or None
        hz: sampling rate

    Returns:
        float: std of activity channel over valid samples
    """
    if data.ndim == 1:
        return np.nan

    if activity_ch_idx is not None and activity_ch_idx < data.shape[1]:
        act = data[:, activity_ch_idx]
    else:
        # Compute activity channel from feature data (excluding last col if it's activity)
        act = compute_activity_channel(data, hz, trailing_seconds=30.0).ravel()

    if valid is not None:
        v = np.asarray(valid).ravel()[:len(act)]
        if v.dtype != bool:
            v = v.astype(bool)
        act = act[v] if v.sum() > 10 else act

    return float(np.nanstd(act))


def autocorrelation_at_lags(data_agg, hz, lags_sec=(0.5, 1.0, 2.0, 5.0)):
    """Autocorrelation at specific lags.

    Indicates temporal structure at EWLS-relevant timescales.
    ACF ≈ 0 at these lags means the signal has no predictable dynamics.
    """
    n = len(data_agg)
    x = data_agg - np.nanmean(data_agg)
    var = np.nanvar(x)
    if var < 1e-12 or n < 20:
        return {lag: np.nan for lag in lags_sec}

    result = {}
    for lag_s in lags_sec:
        lag_samples = int(round(lag_s * hz))
        if lag_samples >= n or lag_samples < 1:
            result[lag_s] = np.nan
            continue
        acf = float(np.nanmean(x[lag_samples:] * x[:-lag_samples]) / var)
        result[lag_s] = acf
    return result


def cross_participant_max_corr(p1_data, p2_data, p1_ts, p2_ts,
                                p1_valid, p2_valid, max_lag_s=5.0):
    """Max absolute lagged cross-correlation between P1 and P2.

    The most direct measure of coupling potential: if there is no
    correlation at any lag, CADENCE cannot detect coupling.

    Computed on channel-averaged signals resampled to 1 Hz for speed.
    """
    t_start = max(p1_ts[0], p2_ts[0])
    t_end = min(p1_ts[-1], p2_ts[-1])
    if t_end - t_start < 60:
        return np.nan
    t_common = np.arange(t_start, t_end, 1.0)  # 1 Hz

    # Channel-average
    p1_agg = np.nanmean(p1_data, axis=1) if p1_data.ndim > 1 else p1_data.copy()
    p2_agg = np.nanmean(p2_data, axis=1) if p2_data.ndim > 1 else p2_data.copy()

    # Mask invalid
    if p1_valid is not None:
        v1 = np.asarray(p1_valid)
        if v1.ndim == 2:
            v1 = np.mean(v1, axis=1) > 0.5
        p1_agg = np.where(v1[:len(p1_agg)], p1_agg, np.nan)
    if p2_valid is not None:
        v2 = np.asarray(p2_valid)
        if v2.ndim == 2:
            v2 = np.mean(v2, axis=1) > 0.5
        p2_agg = np.where(v2[:len(p2_agg)], p2_agg, np.nan)

    # Interpolate to common 1Hz grid
    with np.errstate(invalid='ignore'):
        s1 = np.interp(t_common, p1_ts[:len(p1_agg)],
                        np.nan_to_num(p1_agg, nan=0.0))
        s2 = np.interp(t_common, p2_ts[:len(p2_agg)],
                        np.nan_to_num(p2_agg, nan=0.0))

    # Normalize
    std1, std2 = np.std(s1), np.std(s2)
    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0
    s1 = (s1 - np.mean(s1)) / std1
    s2 = (s2 - np.mean(s2)) / std2

    # Max cross-correlation over lags
    max_lag_samples = int(max_lag_s)
    n = len(s1)
    best = 0.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            if n - lag < 10:
                continue
            cc = np.mean(s1[lag:] * s2[:n - lag])
        else:
            if n + lag < 10:
                continue
            cc = np.mean(s1[:n + lag] * s2[-lag:])
        best = max(best, abs(cc))
    return float(best)


def tracking_smoothness(data, valid, hz):
    """Mean absolute jerk (3rd derivative) of blendshape signal.

    High jerk = face tracking instability (drops, reacquisitions).
    Computed on first 10 AU channels. Valid even on z-scored data
    because tracking drops create sharp discontinuities.
    """
    if len(data) < 10 or data.ndim == 1:
        return np.nan
    n_ch = min(10, data.shape[1])
    dt = 1.0 / hz
    d3 = np.diff(data[:, :n_ch], n=3, axis=0) / (dt ** 3)
    if valid is not None and len(valid) > 3:
        v3 = np.asarray(valid).ravel()
        if v3.dtype != bool:
            v3 = v3.astype(bool)
        # Align after 3 diffs
        v3 = v3[3:len(d3) + 3] if len(v3) > len(d3) + 3 else v3[:len(d3)]
        if len(v3) == len(d3) and v3.sum() > 10:
            d3 = d3[v3]
    return float(np.mean(np.abs(d3)))


def effective_duration(duration_s, p_metrics):
    """Actual usable recording time after gap/validity losses.

    Returns duration * min(1 - gap_frac) across key V2 modalities.
    """
    min_available = 1.0
    for mod_name in ['EEG wavelet', 'Blendshapes v2', 'Pose features']:
        mod = p_metrics.get(mod_name)
        if mod is None:
            continue
        gap = mod.get('gap_frac', 0)
        avail = (1.0 - gap) * mod.get('valid_rate', 1.0)
        min_available = min(min_available, avail)
    return duration_s * min_available


# ---------------------------------------------------------------------------
# Per-modality assessment
# ---------------------------------------------------------------------------

def assess_modality(session, mod_def, participant):
    """Compute quality metrics for one modality of one participant."""
    data_key = mod_def['data_key'].format(p=participant)
    valid_key = mod_def['valid_key'].format(p=participant)
    ts_key = mod_def['ts_key'].format(p=participant)

    if data_key not in session:
        return None

    data = session[data_key]
    valid = session.get(valid_key)
    ts = session.get(ts_key)

    if data is None or len(data) == 0:
        return None

    data = np.asarray(data, dtype=np.float64)

    # Valid rate
    if valid is not None:
        valid_arr = np.asarray(valid)
        if valid_arr.ndim == 2:
            valid_rate = float(np.mean(valid_arr))
        else:
            valid_rate = float(np.mean(valid_arr))
    else:
        valid_rate = 1.0

    # Duration
    if ts is not None and len(ts) > 1:
        duration = float(ts[-1] - ts[0])
    else:
        duration = len(data) / mod_def['native_hz']

    n_samples = len(data)
    n_ch = data.shape[1] if data.ndim > 1 else 1
    hz = mod_def['native_hz']

    # Gap analysis
    if ts is not None and len(ts) > 1:
        max_gap, n_gaps, gap_frac = gap_analysis(ts, hz)
    else:
        max_gap, n_gaps, gap_frac = 0.0, 0, 0.0

    # Select valid data for signal analysis
    valid_bool = None
    if valid is not None:
        valid_bool = np.asarray(valid)
        if valid_bool.ndim == 2:
            valid_bool = np.mean(valid_bool, axis=1) > 0.5
        else:
            valid_bool = valid_bool.astype(bool)

    if valid_bool is not None and data.ndim > 1:
        data_v = data[valid_bool] if valid_bool.sum() > 10 else data
    elif valid_bool is not None:
        data_v = data[valid_bool] if valid_bool.sum() > 10 else data
    else:
        data_v = data

    # Channel-averaged signal for aggregate metrics
    data_agg = np.nanmean(data_v, axis=1) if data_v.ndim > 1 else data_v

    # Activity variance
    act_var = activity_variance(data, mod_def.get('activity_ch'), valid_bool, hz)

    # Dynamics CV (30s windows)
    dyn_cv = dynamics_cv(data_agg, window_sec=30, hz=hz)

    # Autocorrelation
    acf = autocorrelation_at_lags(data_agg, hz, lags_sec=(1.0, 5.0))

    # Tracking smoothness (blendshapes only)
    smooth = None
    if 'blendshapes' in mod_def['data_key'].lower() or 'Blendshapes' in mod_def['name']:
        smooth = tracking_smoothness(data, valid_bool, hz)

    return {
        'n_samples': n_samples,
        'duration_s': round(duration, 1),
        'n_channels': n_ch,
        'valid_rate': round(valid_rate, 4),
        'gap_frac': round(gap_frac, 4),
        'n_gaps': n_gaps,
        'max_gap_s': round(max_gap, 2),
        'activity_var': round(act_var, 4) if not np.isnan(act_var) else None,
        'dynamics_cv': round(dyn_cv, 3) if not np.isnan(dyn_cv) else None,
        'acf_1s': round(acf.get(1.0, np.nan), 3) if not np.isnan(acf.get(1.0, np.nan)) else None,
        'acf_5s': round(acf.get(5.0, np.nan), 3) if not np.isnan(acf.get(5.0, np.nan)) else None,
        'tracking_smoothness': round(smooth, 1) if smooth is not None and not np.isnan(smooth) else None,
    }


# ---------------------------------------------------------------------------
# Session-level assessment
# ---------------------------------------------------------------------------

def assess_session(session, name):
    """Full quality assessment for one session."""
    result = {
        'session': name,
        'duration_s': session.get('duration', 0),
        'participants': {},
        'cross_participant': {},
    }

    for p in ['p1', 'p2']:
        p_result = {}
        for mod_def in MODALITIES:
            metrics = assess_modality(session, mod_def, p)
            if metrics is not None:
                p_result[mod_def['name']] = metrics
        result['participants'][p] = p_result

    # Cross-participant correlation for matching modalities
    for mod_def in MODALITIES:
        mod_name = mod_def['name']
        d1_key = mod_def['data_key'].format(p='p1')
        d2_key = mod_def['data_key'].format(p='p2')
        t1_key = mod_def['ts_key'].format(p='p1')
        t2_key = mod_def['ts_key'].format(p='p2')
        v1_key = mod_def['valid_key'].format(p='p1')
        v2_key = mod_def['valid_key'].format(p='p2')

        if d1_key in session and d2_key in session:
            d1 = np.asarray(session[d1_key], dtype=np.float64)
            d2 = np.asarray(session[d2_key], dtype=np.float64)
            t1 = session.get(t1_key)
            t2 = session.get(t2_key)
            if t1 is not None and t2 is not None and len(d1) > 0 and len(d2) > 0:
                xcorr = cross_participant_max_corr(
                    d1, d2, t1, t2,
                    session.get(v1_key), session.get(v2_key))
                result['cross_participant'][mod_name] = round(xcorr, 4) \
                    if not np.isnan(xcorr) else None

    # Interbrain (shared)
    ib_key = 'eeg_interbrain'
    if ib_key in session:
        data = session[ib_key]
        valid = session.get('eeg_interbrain_valid')
        ts = session.get('eeg_interbrain_ts')
        if data is not None and len(data) > 0:
            data = np.asarray(data, dtype=np.float64)
            valid_rate = float(np.mean(valid)) if valid is not None else 1.0
            duration = float(ts[-1] - ts[0]) if ts is not None and len(ts) > 1 else 0
            result['interbrain'] = {
                'n_samples': len(data),
                'n_channels': data.shape[1] if data.ndim > 1 else 1,
                'duration_s': round(duration, 1),
                'valid_rate': round(valid_rate, 4),
            }

    # Effective duration (worst-case across both participants)
    eff_durs = []
    for p in ['p1', 'p2']:
        ed = effective_duration(result['duration_s'], result['participants'].get(p, {}))
        eff_durs.append(ed)
    result['effective_duration_s'] = round(min(eff_durs), 1) if eff_durs else 0

    # Quality flags and grade
    result['flags'] = compute_quality_flags(result)
    result['grade'] = quality_grade(result['flags'])

    return result


def compute_quality_flags(result):
    """Generate quality flags based on empirical thresholds."""
    flags = []
    T = THRESHOLDS

    for p in ['p1', 'p2']:
        mods = result['participants'].get(p, {})

        # Pose gap fraction
        pose = mods.get('Pose features', {})
        gf = pose.get('gap_frac', 0)
        if gf > T['pose_gap_critical']:
            flags.append(('CRITICAL', f'{p} Pose: {gf*100:.1f}% gap fraction (>10%)'))
        elif gf > T['pose_gap_warning']:
            flags.append(('WARNING', f'{p} Pose: {gf*100:.1f}% gap fraction (>3%)'))

        # BL gap fraction
        bl = mods.get('Blendshapes v2', {})
        gf = bl.get('gap_frac', 0)
        if gf > T['bl_gap_critical']:
            flags.append(('CRITICAL', f'{p} BL: {gf*100:.1f}% gap fraction (>10%)'))
        elif gf > T['bl_gap_warning']:
            flags.append(('WARNING', f'{p} BL: {gf*100:.1f}% gap fraction (>3%)'))

        # Activity variance (static session)
        for mod_name in ['Blendshapes v2', 'Pose features']:
            mod = mods.get(mod_name, {})
            av = mod.get('activity_var')
            if av is not None and av < T['activity_var_low']:
                flags.append(('WARNING', f'{p} {mod_name}: low activity variance {av:.3f}'))

        # EEG validity
        eeg = mods.get('EEG wavelet', {})
        vr = eeg.get('valid_rate', 1.0)
        if vr < T['eeg_valid_warning']:
            flags.append(('WARNING', f'{p} EEG: validity {vr*100:.1f}% (<95%)'))

        # Missing key modalities
        for mod_name in ['Blendshapes v2', 'Pose features']:
            if mod_name not in mods:
                flags.append(('CRITICAL', f'{p} {mod_name}: MISSING'))
        if 'ECG features' not in mods:
            flags.append(('INFO', f'{p} ECG features: MISSING'))

    # Effective duration
    ed = result.get('effective_duration_s', 0)
    if ed < T['eff_dur_critical']:
        flags.append(('CRITICAL', f'Effective duration {ed/60:.1f} min (<20 min)'))
    elif ed < T['eff_dur_warning']:
        flags.append(('WARNING', f'Effective duration {ed/60:.1f} min (<40 min)'))

    # Cross-participant correlation
    for mod_name, xcorr in result.get('cross_participant', {}).items():
        if xcorr is not None and xcorr > T['xcorr_high']:
            flags.append(('INFO', f'{mod_name} P1-P2 corr={xcorr:.3f} (high — check for artifact)'))

    return flags


def quality_grade(flags):
    """Assign overall quality grade."""
    levels = [level for level, _ in flags]
    if 'CRITICAL' in levels:
        return 'POOR'
    elif 'WARNING' in levels:
        return 'MARGINAL'
    return 'GOOD'


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt(val, fmt_str, na='N/A'):
    if val is None:
        return na
    return fmt_str.format(val)


def print_comparison(all_results, benchmark_name):
    """Print comparison tables."""
    # Find benchmark
    bench = None
    for r in all_results:
        if benchmark_name in r['session']:
            bench = r
            break

    W = 120
    print(f"\n{'=' * W}")
    print(f"DATA QUALITY REPORT v2  (benchmark: {benchmark_name})")
    print(f"{'=' * W}")

    # ── Table 1: Session Overview ──
    print(f"\n{'Session':<22} {'Duration':>8} {'Eff.Dur':>8} "
          f"{'P1 Pose%':>9} {'P2 Pose%':>9} "
          f"{'P1 BL%':>8} {'P2 BL%':>8} "
          f"{'P1↔P2 BL':>9} {'Grade':>8}")
    print('-' * W)

    for r in sorted(all_results, key=lambda x: x['session']):
        dur = f"{r['duration_s']/60:.0f} min"
        eff = f"{r['effective_duration_s']/60:.0f} min"

        vals = []
        for p in ['p1', 'p2']:
            pose = r['participants'].get(p, {}).get('Pose features', {})
            gf = pose.get('gap_frac', 0)
            vals.append(f"{gf*100:.1f}%")
        for p in ['p1', 'p2']:
            bl = r['participants'].get(p, {}).get('Blendshapes v2', {})
            gf = bl.get('gap_frac', 0)
            vals.append(f"{gf*100:.1f}%")

        xcorr_bl = r.get('cross_participant', {}).get('Blendshapes v2')
        xcorr_s = f"{xcorr_bl:.3f}" if xcorr_bl is not None else 'N/A'

        grade = r.get('grade', '?')
        marker = ' <<<' if benchmark_name in r['session'] else ''

        print(f"{r['session']:<22} {dur:>8} {eff:>8} "
              f"{vals[0]:>9} {vals[1]:>9} "
              f"{vals[2]:>8} {vals[3]:>8} "
              f"{xcorr_s:>9} {grade:>8}{marker}")

    # ── Table 2: Per-Modality Detail ──
    for mod_name in ['Blendshapes v2', 'Pose features', 'EEG wavelet', 'ECG features']:
        print(f"\n{'─' * W}")
        print(f"  {mod_name}")
        print(f"{'─' * W}")

        cols = ['gap%', 'valid%', 'act_var', 'dyn_cv', 'acf_1s', 'acf_5s']
        if 'Blendshapes' in mod_name:
            cols.append('jerk')
        cols.append('xcorr')

        header = f"{'Session':<22} {'P':>3}"
        for c in cols:
            header += f" {c:>9}"
        print(header)
        print('-' * W)

        for r in sorted(all_results, key=lambda x: x['session']):
            for p in ['p1', 'p2']:
                mod = r['participants'].get(p, {}).get(mod_name)
                if mod is None:
                    continue

                marker = ' <<<' if benchmark_name in r['session'] else ''
                row = f"{r['session']:<22} {p:>3}"
                row += f" {mod.get('gap_frac', 0)*100:>8.1f}%"
                row += f" {mod.get('valid_rate', 0)*100:>8.1f}%"
                row += f" {_fmt(mod.get('activity_var'), '{:.3f}'):>9}"
                row += f" {_fmt(mod.get('dynamics_cv'), '{:.3f}'):>9}"
                row += f" {_fmt(mod.get('acf_1s'), '{:.3f}'):>9}"
                row += f" {_fmt(mod.get('acf_5s'), '{:.3f}'):>9}"
                if 'Blendshapes' in mod_name:
                    row += f" {_fmt(mod.get('tracking_smoothness'), '{:.0f}'):>9}"

                # Cross-participant (same for both p1/p2, show once)
                if p == 'p1':
                    xc = r.get('cross_participant', {}).get(mod_name)
                    row += f" {_fmt(xc, '{:.3f}'):>9}"
                else:
                    row += f" {'':>9}"

                row += marker
                print(row)

    # ── Table 3: Quality Flags ──
    print(f"\n{'=' * W}")
    print(f"QUALITY FLAGS")
    print(f"{'=' * W}")

    any_flags = False
    for r in sorted(all_results, key=lambda x: x['session']):
        flags = r.get('flags', [])
        if not flags:
            continue
        any_flags = True
        grade = r.get('grade', '?')
        print(f"\n  {r['session']}  [{grade}]:")
        for level, msg in flags:
            print(f"    [{level:>8}] {msg}")

    if not any_flags:
        print("\n  No quality flags raised.")

    # ── Summary ──
    grades = [r.get('grade', '?') for r in all_results]
    print(f"\n{'─' * W}")
    print(f"  GOOD: {grades.count('GOOD')}  |  "
          f"MARGINAL: {grades.count('MARGINAL')}  |  "
          f"POOR: {grades.count('POOR')}  |  "
          f"Total: {len(all_results)}")
    print(f"{'─' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CADENCE data quality assessment v2')
    parser.add_argument('--benchmark', default='y_32',
                        help='Benchmark session name (default: y_32)')
    parser.add_argument('--output', default='results/data_quality',
                        help='Output directory for JSON report')
    parser.add_argument('--sessions', nargs='*', default=None,
                        help='Specific sessions to assess (default: all)')
    args = parser.parse_args()

    config = load_config()
    sessions = discover_cached_sessions(config['session_cache'])
    print(f"Found {len(sessions)} cached sessions")

    all_results = []

    for name, path in sessions:
        if args.sessions and not any(s in name for s in args.sessions):
            continue

        print(f"\nAssessing: {name}...", end='', flush=True)
        try:
            session = load_session_from_cache(path, config=config)
            result = assess_session(session, name)
            all_results.append(result)
            dur = result['duration_s'] / 60
            grade = result.get('grade', '?')
            print(f" {dur:.0f} min [{grade}]")
        except Exception as e:
            print(f" ERROR: {e}")

    if not all_results:
        print("No sessions assessed!")
        return

    print_comparison(all_results, args.benchmark)

    # Save JSON report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'quality_report.json')

    # Strip non-serializable items
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=clean)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
