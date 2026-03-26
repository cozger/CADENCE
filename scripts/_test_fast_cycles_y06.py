"""Fast GPU cycle analysis on y_06 — all conditions."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.fast_cycles import analyze_interbrain_cycles
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)
intervals = parse_condition_intervals(session)
p1_role = session.get('p1_role', 'unknown')
p2_role = session.get('p2_role', 'unknown')
print(f"{name}: P1={p1_role}, P2={p2_role}\n")

p1_eeg = session['p1_eeg']
p2_eeg = session['p2_eeg']
p1_ts = session['p1_eeg_ts']
p2_ts = session['p2_eeg_ts']


def extract(start_s, end_s):
    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    p2_m = (p2_ts >= start_s) & (p2_ts < end_s)
    dur = end_s - start_s
    N = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if N < int(10 * FS):
        return None, None
    t = np.linspace(0, dur, N)
    p1 = np.stack([np.interp(t, p1_ts[p1_m]-start_s, p1_eeg[p1_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    p2 = np.stack([np.interp(t, p2_ts[p2_m]-start_s, p2_eeg[p2_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    for s in [p1, p2]:
        s -= s.mean(axis=1, keepdims=True)  # avg ref
        for ch in range(14):
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd
    return p1.astype(np.float32), p2.astype(np.float32)


targets = ['meditate_K', 'conv_1', 'conv_2', 'base_EO', 'meditate_B']
all_r = {}

for start, end, cond in intervals:
    if cond not in targets:
        continue
    p1, p2 = extract(start, end)
    if p1 is None:
        continue

    t0 = time.perf_counter()
    r = analyze_interbrain_cycles(p1, p2, FS, band=(4.0, 8.0),
                                   n_surrogates=200, seed=42)
    elapsed = time.perf_counter() - t0

    all_r[cond] = r
    nv = r['n_valid_channels']
    print(f"{cond:>15} ({end-start:.0f}s, {elapsed:.1f}s) {nv} ch:")
    for feat in ['volt_amp', 'period', 'time_rdsym', 'is_burst']:
        if feat in r:
            fr = r[feat]
            print(f"  {feat:>15}: z={fr['pooled_z']:+.3f}  r={fr['mean_r']:+.4f}")
    bc = r.get('burst_cooc', {})
    print(f"  {'burst_cooc':>15}: z={bc.get('pooled_z', 0):+.3f}  "
          f"({bc.get('n_channels_with_bursts', 0)} ch with bursts)")
    print()

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY (pooled z)")
print(f"{'='*70}")
print(f"  {'cond':>15} {'amp_z':>8} {'period_z':>8} {'rdsym_z':>8} {'burst_z':>8}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for cond in targets:
    if cond not in all_r:
        continue
    r = all_r[cond]
    az = r.get('volt_amp', {}).get('pooled_z', 0)
    pz = r.get('period', {}).get('pooled_z', 0)
    sz = r.get('time_rdsym', {}).get('pooled_z', 0)
    bz = r.get('burst_cooc', {}).get('pooled_z', 0)
    print(f"  {cond:>15} {az:>+7.2f} {pz:>+7.2f} {sz:>+7.2f} {bz:>+7.2f}")
print(f"{'='*70}")
