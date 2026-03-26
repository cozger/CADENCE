"""Bycycle cycle-by-cycle analysis on y_06 — optimized with compute_features_2d.

Uses bycycle's native parallelization (n_jobs=-1) for all channels at once.
Per-channel analysis with pooled statistics across channels.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.ndimage import uniform_filter1d
from bycycle.group import compute_features_2d
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0
BAND = (4.0, 8.0)
SEED = 42

BURST_KWARGS = dict(
    burst_method='cycles',
    threshold_kwargs={
        'amp_fraction_threshold': 0.,
        'amp_consistency_threshold': 0.4,
        'period_consistency_threshold': 0.5,
        'monotonicity_threshold': 0.7,
        'min_n_cycles': 3,
    }
)

# ── Load y_06 ────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)

p1_role = session.get('p1_role', 'unknown')
p2_role = session.get('p2_role', 'unknown')
intervals = parse_condition_intervals(session)
print(f"Session: {name}, P1={p1_role}, P2={p2_role}\n")

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
    p1_seg = np.stack([np.interp(t_common, p1_ts[p1_m] - start_s, p1_eeg[p1_m, c])
                        for c in range(14)], axis=1).astype(np.float64)
    p2_seg = np.stack([np.interp(t_common, p2_ts[p2_m] - start_s, p2_eeg[p2_m, c])
                        for c in range(14)], axis=1).astype(np.float64)
    # Avg reference + z-normalize per channel
    for seg in [p1_seg, p2_seg]:
        seg -= seg.mean(axis=1, keepdims=True)
        for ch in range(14):
            mu, sd = seg[:, ch].mean(), max(seg[:, ch].std(), 1e-8)
            seg[:, ch] = (seg[:, ch] - mu) / sd
    return p1_seg, p2_seg, N


def fit_all_channels(eeg_segment):
    """Run bycycle on all 14 channels in parallel. Returns list of DataFrames."""
    sigs = eeg_segment.T  # (14, N)
    dfs = compute_features_2d(sigs, FS, BAND,
                               compute_features_kwargs=BURST_KWARGS,
                               n_jobs=-1)
    return dfs


def cycle_to_regular(df, N, fs, rate=2.0):
    """Resample cycle features to regular time grid."""
    if df is None or len(df) < 5:
        return None
    t_cycle = df['sample_peak'].values / fs
    dur = N / fs
    t_out = np.arange(0, dur, 1.0 / rate)
    out = {}
    for col in ['volt_amp', 'period', 'time_rdsym', 'is_burst']:
        if col in df.columns:
            out[col] = np.interp(t_out, t_cycle, df[col].values.astype(float))
    return t_out, out


def analyze_condition(p1_seg, p2_seg, N, label):
    """Full bycycle analysis for one condition."""
    print(f"\n  {'='*60}")
    print(f"  {label} ({N/FS:.0f}s)")
    print(f"  {'='*60}")

    t0 = time.perf_counter()
    dfs_p1 = fit_all_channels(p1_seg)
    dfs_p2 = fit_all_channels(p2_seg)
    fit_time = time.perf_counter() - t0
    print(f"  Bycycle fit: {fit_time:.1f}s (parallel)")

    results = {}

    # ── Per-channel burst stats ──────────────────────────────────────────
    print(f"\n  Per-channel burst fraction:")
    burst_zs = []
    feat_zs = {'volt_amp': [], 'period': [], 'time_rdsym': []}
    feat_rs = {'volt_amp': [], 'period': [], 'time_rdsym': []}

    rng = np.random.default_rng(SEED)

    for ch in range(14):
        df1, df2 = dfs_p1[ch], dfs_p2[ch]
        if len(df1) < 10 or len(df2) < 10:
            continue

        bf1 = df1['is_burst'].mean()
        bf2 = df2['is_burst'].mean()

        # Resample to regular grid
        r1 = cycle_to_regular(df1, N, FS)
        r2 = cycle_to_regular(df2, N, FS)
        if r1 is None or r2 is None:
            continue

        min_len = min(len(r1[1]['is_burst']), len(r2[1]['is_burst']))

        # Burst co-occurrence
        b1 = r1[1]['is_burst'][:min_len] > 0.5
        b2 = r2[1]['is_burst'][:min_len] > 0.5
        cooc_real = (b1 & b2).mean()
        expected = b1.mean() * b2.mean()

        # Quick surrogate for burst co-occurrence
        cooc_surr = np.array([
            (np.roll(b1, rng.integers(int(0.1*min_len), int(0.9*min_len))) & b2).mean()
            for _ in range(100)])
        z_burst = (cooc_real - cooc_surr.mean()) / max(cooc_surr.std(), 1e-10)

        if bf1 > 0.01 and bf2 > 0.01:
            burst_zs.append(z_burst)

        # Feature cross-correlation
        for feat in ['volt_amp', 'period', 'time_rdsym']:
            x1 = r1[1][feat][:min_len]
            x2 = r2[1][feat][:min_len]
            if len(x1) < 40:
                continue
            r_val = np.corrcoef(x1, x2)[0, 1]
            # Quick surrogate
            surr_rs = np.array([
                np.corrcoef(np.roll(x1, rng.integers(int(0.1*len(x1)), int(0.9*len(x1)))),
                             x2)[0, 1]
                for _ in range(100)])
            z_f = (r_val - surr_rs.mean()) / max(surr_rs.std(), 1e-10)
            feat_zs[feat].append(z_f)
            feat_rs[feat].append(r_val)

        print(f"    {EPOC_CHANNEL_NAMES[ch]:>4}: P1={bf1:4.0%} P2={bf2:4.0%}  "
              f"cooc={cooc_real:.1%}(exp={expected:.1%}) z_b={z_burst:+.2f}")

    # ── Pooled results (Stouffer across channels) ────────────────────────
    print(f"\n  Pooled z-scores (Stouffer across channels):")

    # Burst
    if burst_zs:
        pooled_bz = np.mean(burst_zs) * np.sqrt(len(burst_zs))
    else:
        pooled_bz = 0
    results['burst_cooc'] = {'z': pooled_bz}
    print(f"    {'Burst co-occurrence':>25}: z={pooled_bz:+.3f} ({len(burst_zs)} ch)")

    # Features
    for feat, label_f in [('volt_amp', 'Amplitude'),
                            ('period', 'Period'),
                            ('time_rdsym', 'Rise-decay symmetry')]:
        zs = feat_zs[feat]
        rs = feat_rs[feat]
        if zs:
            pooled = np.mean(zs) * np.sqrt(len(zs))
            mean_r = np.mean(rs)
        else:
            pooled, mean_r = 0, 0
        results[feat] = {'z': pooled, 'mean_r': mean_r}
        print(f"    {label_f:>25}: z={pooled:+.3f}  r={mean_r:+.4f} ({len(zs)} ch)")

    return results


# ── Run ──────────────────────────────────────────────────────────────────
target_conditions = ['meditate_K', 'conv_1', 'conv_2', 'base_EO', 'meditate_B']
all_results = {}

for start, end, cond in intervals:
    if cond not in target_conditions:
        continue
    p1_seg, p2_seg, N = extract_segment(start, end)
    if p1_seg is None:
        continue
    r = analyze_condition(p1_seg, p2_seg, N,
                           f"{cond} ({p1_role} vs {p2_role})")
    if r:
        all_results[cond] = r

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  CROSS-CONDITION COMPARISON (pooled z)")
print(f"{'='*70}")
print(f"  {'Condition':>15} {'burst_z':>10} {'amp_z':>10} "
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

print(f"\n  P1={p1_role}, P2={p2_role}")
print(f"{'='*70}")
