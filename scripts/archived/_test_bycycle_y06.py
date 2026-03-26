"""Bycycle cycle-by-cycle analysis on y_06 meditate_K.

Extracts per-cycle features from P1 and P2 theta, then tests:
  1. Burst co-occurrence rate vs chance
  2. Amplitude envelope cross-correlation
  3. Period (instantaneous frequency) cross-correlation
  4. Rise-decay symmetry cross-correlation
  5. Comparison: coupled condition (meditate_K) vs baseline (base_EO)
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import sosfiltfilt, butter
from scipy.ndimage import uniform_filter1d
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0
BAND = (4.0, 8.0)
SEED = 42

# bycycle numpy 2.x fix applied directly to site-packages/bycycle/burst/cycle.py:93
# (.to_numpy() → .to_numpy().copy())
from bycycle import Bycycle

# ── Load y_06 ────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)

p1_role = session.get('p1_role', 'unknown')
p2_role = session.get('p2_role', 'unknown')
intervals = parse_condition_intervals(session)
print(f"Session: {name}, P1={p1_role}, P2={p2_role}")

p1_eeg = session['p1_eeg']
p2_eeg = session['p2_eeg']
p1_ts = session['p1_eeg_ts']
p2_ts = session['p2_eeg_ts']


def extract_segment(start_s, end_s):
    """Extract aligned, avg-ref, z-scored EEG segment."""
    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    p2_m = (p2_ts >= start_s) & (p2_ts < end_s)
    dur = end_s - start_s
    N = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if N < int(10 * FS):
        return None, None, 0

    t_common = np.linspace(0, dur, N)
    p1_t = p1_ts[p1_m] - start_s
    p2_t = p2_ts[p2_m] - start_s

    p1_seg = np.stack([np.interp(t_common, p1_t, p1_eeg[p1_m, c])
                        for c in range(14)], axis=1).astype(np.float64)
    p2_seg = np.stack([np.interp(t_common, p2_t, p2_eeg[p2_m, c])
                        for c in range(14)], axis=1).astype(np.float64)

    # Avg reference
    p1_seg -= p1_seg.mean(axis=1, keepdims=True)
    p2_seg -= p2_seg.mean(axis=1, keepdims=True)

    # Z-normalize per channel
    for ch in range(14):
        for s in [p1_seg, p2_seg]:
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd

    return p1_seg, p2_seg, N


def run_bycycle_channel(sig, fs, band):
    """Run bycycle on one channel, return DataFrame or None."""
    try:
        bc = Bycycle(
            center_extrema='peak',
            burst_method='cycles',
            thresholds={
                'amp_fraction_threshold': 0.,
                'amp_consistency_threshold': 0.4,
                'period_consistency_threshold': 0.5,
                'monotonicity_threshold': 0.7,
                'min_n_cycles': 3,
            }
        )
        bc.fit(sig, fs, f_range=band)
        return bc.df_features
    except Exception as e:
        return None


def cycle_features_to_timeseries(df, N, fs):
    """Convert cycle-level features to regular time series at ~2 Hz."""
    if df is None or len(df) < 3:
        return None

    # Use peak sample as cycle time
    t_cycle = df['sample_peak'].values / fs

    # Features to extract
    features = {}
    for col in ['volt_amp', 'period', 'time_rdsym', 'is_burst']:
        if col in df.columns:
            features[col] = df[col].values.astype(float)

    # Resample to 2 Hz regular grid
    dur = N / fs
    t_out = np.arange(0, dur, 0.5)  # 2 Hz
    resampled = {}
    for col, vals in features.items():
        resampled[col] = np.interp(t_out, t_cycle, vals)

    return t_out, resampled


def cross_correlate_windowed(x1, x2, window_s=20.0, rate=2.0):
    """Sliding-window Pearson correlation between two time series."""
    win = int(window_s * rate)
    N = min(len(x1), len(x2))
    if N < win:
        return np.array([np.corrcoef(x1[:N], x2[:N])[0, 1]])

    n_win = N - win + 1
    corrs = np.zeros(n_win)
    for i in range(n_win):
        a = x1[i:i+win]
        b = x2[i:i+win]
        if a.std() > 1e-8 and b.std() > 1e-8:
            corrs[i] = np.corrcoef(a, b)[0, 1]
    return corrs


def burst_cooccurrence(burst1, burst2, window_s=20.0, rate=2.0,
                        n_surr=200, seed=42):
    """Test burst co-occurrence rate vs chance."""
    rng = np.random.default_rng(seed)
    N = min(len(burst1), len(burst2))
    b1 = burst1[:N] > 0.5
    b2 = burst2[:N] > 0.5

    # Observed co-occurrence
    cooc_real = (b1 & b2).mean()
    rate1 = b1.mean()
    rate2 = b2.mean()
    expected = rate1 * rate2

    # Surrogate: circular shift
    cooc_surr = np.zeros(n_surr)
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N), int(0.9 * N))
        b1_shifted = np.roll(b1, shift)
        cooc_surr[k] = (b1_shifted & b2).mean()

    z = (cooc_real - cooc_surr.mean()) / max(cooc_surr.std(), 1e-10)

    return {
        'cooc_rate': cooc_real,
        'expected_rate': expected,
        'excess': cooc_real - expected,
        'z': z,
        'p1_burst_frac': rate1,
        'p2_burst_frac': rate2,
    }


def analyze_condition(p1_seg, p2_seg, N, label):
    """Full bycycle analysis for one condition."""
    print(f"\n  {'='*60}")
    print(f"  {label} ({N/FS:.0f}s)")
    print(f"  {'='*60}")

    # Run bycycle per channel for both participants
    all_p1_features = {}
    all_p2_features = {}

    t0 = time.perf_counter()
    for ch in range(14):
        df1 = run_bycycle_channel(p1_seg[:, ch], FS, BAND)
        df2 = run_bycycle_channel(p2_seg[:, ch], FS, BAND)
        if df1 is not None and len(df1) > 10:
            all_p1_features[ch] = df1
        if df2 is not None and len(df2) > 10:
            all_p2_features[ch] = df2

    elapsed = time.perf_counter() - t0
    n_ch_ok = len(set(all_p1_features.keys()) & set(all_p2_features.keys()))
    print(f"  Bycycle: {n_ch_ok}/14 channels OK ({elapsed:.1f}s)")

    if n_ch_ok < 3:
        print(f"  Too few channels, skipping")
        return None

    common_ch = sorted(set(all_p1_features.keys()) & set(all_p2_features.keys()))

    # ── Per-channel analysis, then pool across channels ──────────────────
    results = {}

    # 1. Per-channel burst co-occurrence → pooled z
    print(f"\n  Per-channel burst co-occurrence:")
    burst_zs = []
    for ch in common_ch:
        r1 = cycle_features_to_timeseries(all_p1_features[ch], N, FS)
        r2 = cycle_features_to_timeseries(all_p2_features[ch], N, FS)
        if r1 is None or r2 is None:
            continue
        bc_ch = burst_cooccurrence(r1[1]['is_burst'], r2[1]['is_burst'])
        bf1 = all_p1_features[ch]['is_burst'].mean()
        bf2 = all_p2_features[ch]['is_burst'].mean()
        if bf1 > 0.01 and bf2 > 0.01:  # both have bursts
            burst_zs.append(bc_ch['z'])
            print(f"    {EPOC_CHANNEL_NAMES[ch]:>4}: P1={bf1:.0%} P2={bf2:.0%}  "
                  f"cooc={bc_ch['cooc_rate']:.1%} exp={bc_ch['expected_rate']:.1%}  "
                  f"z={bc_ch['z']:+.3f}")
        else:
            print(f"    {EPOC_CHANNEL_NAMES[ch]:>4}: P1={bf1:.0%} P2={bf2:.0%}  (skipped)")

    pooled_burst_z = np.mean(burst_zs) * np.sqrt(len(burst_zs)) if burst_zs else 0
    results['burst_cooc'] = {'z': pooled_burst_z, 'n_ch': len(burst_zs)}
    print(f"    Pooled burst z: {pooled_burst_z:+.3f} ({len(burst_zs)} channels)")

    # 2-4. Per-channel feature cross-correlation → pooled
    feat_labels = {
        'volt_amp': 'Amplitude',
        'period': 'Period (inst. freq)',
        'time_rdsym': 'Rise-decay symmetry',
    }

    print(f"\n  Cross-correlation (20s windows, per-channel → pooled):")
    for feat, label_f in feat_labels.items():
        ch_rs = []
        ch_zs = []
        for ch in common_ch:
            r1 = cycle_features_to_timeseries(all_p1_features[ch], N, FS)
            r2 = cycle_features_to_timeseries(all_p2_features[ch], N, FS)
            if r1 is None or r2 is None or feat not in r1[1] or feat not in r2[1]:
                continue
            x1 = r1[1][feat]
            x2 = r2[1][feat]
            min_len = min(len(x1), len(x2))
            x1, x2 = x1[:min_len], x2[:min_len]
            corrs = cross_correlate_windowed(x1, x2, window_s=20.0, rate=2.0)
            mean_r = corrs.mean()
            # Quick surrogate
            rng = np.random.default_rng(SEED + ch)
            surr_rs = []
            for _ in range(100):
                shift = rng.integers(int(0.1*len(x1)), int(0.9*len(x1)))
                surr_rs.append(cross_correlate_windowed(
                    np.roll(x1, shift), x2, window_s=20.0, rate=2.0).mean())
            surr_rs = np.array(surr_rs)
            z_ch = (mean_r - surr_rs.mean()) / max(surr_rs.std(), 1e-10)
            ch_rs.append(mean_r)
            ch_zs.append(z_ch)

        pooled_z = np.mean(ch_zs) * np.sqrt(len(ch_zs)) if ch_zs else 0
        mean_r_all = np.mean(ch_rs) if ch_rs else 0
        results[feat] = {'mean_r': mean_r_all, 'z': pooled_z, 'n_ch': len(ch_zs)}
        print(f"    {label_f:>25}: r_mean={mean_r_all:+.4f}  "
              f"pooled_z={pooled_z:+.3f} ({len(ch_zs)} ch)")

    return results


# ── Run on target conditions ─────────────────────────────────────────────
target_conditions = ['meditate_K', 'conv_1', 'conv_2', 'base_EO', 'meditate_B']

all_results = {}

for start, end, cond in intervals:
    if cond not in target_conditions:
        continue

    p1_seg, p2_seg, N = extract_segment(start, end)
    if p1_seg is None:
        continue

    r = analyze_condition(p1_seg, p2_seg, N, f"{cond} ({p1_role} vs {p2_role})")
    if r:
        all_results[cond] = r

# ── Summary comparison ───────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  CROSS-CONDITION COMPARISON")
print(f"{'='*70}")

print(f"\n  {'Condition':>15} {'burst_z':>10} {'amp_z':>10} "
      f"{'period_z':>10} {'rdsym_z':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for cond in target_conditions:
    if cond not in all_results:
        continue
    r = all_results[cond]
    bz = r.get('burst_cooc', {}).get('z', 0)
    az = r.get('volt_amp', {}).get('z', 0)
    pz = r.get('period', {}).get('z', 0)
    sz = r.get('time_rdsym', {}).get('z', 0)
    print(f"  {cond:>15} {bz:>+9.3f} {az:>+9.3f} {pz:>+9.3f} {sz:>+9.3f}")

print(f"\n{'='*70}")
