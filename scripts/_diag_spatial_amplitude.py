"""Spatial diagnostic: is y_06 amplitude coupling genuine or artifact?

Checks:
1. Per-channel amplitude r values — frontal vs occipital
2. Pseudo-dyad null (P1 from y_06, P2 from different session)
3. With vs without avg-ref
"""
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

# Load y_06 and a DIFFERENT session for pseudo-dyad null
y06_name, y06_path = next((n, p) for n, p in entries if 'y_06' in n)
other_name, other_path = next((n, p) for n, p in entries
                               if 'y_06' not in n and 'y_17' not in n)

s_y06 = load_session_from_cache(y06_path, config=cfg)
s_other = load_session_from_cache(other_path, config=cfg)
print(f"Real dyad: {y06_name}")
print(f"Pseudo null: {other_name}")

intervals = parse_condition_intervals(s_y06)
mk = [(s, e) for s, e, c in intervals if c == 'meditate_K'][0]
cv2 = [(s, e) for s, e, c in intervals if c == 'conv_2'][0]


def extract_pair(s1, s2, start_s, end_s, avg_ref=True):
    """Extract aligned segments from two sessions."""
    p1_eeg, p1_ts = s1['p1_eeg'], s1['p1_eeg_ts']
    p2_eeg, p2_ts = s2['p2_eeg'], s2['p2_eeg_ts']

    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    # For pseudo-dyad: use beginning of other session
    dur = end_s - start_s
    p2_start = float(p2_ts[0])
    p2_m = (p2_ts >= p2_start) & (p2_ts < p2_start + dur)

    N = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if N < int(10 * FS):
        return None, None

    t = np.linspace(0, dur, N)
    p1 = np.stack([np.interp(t, p1_ts[p1_m] - start_s, p1_eeg[p1_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    p2 = np.stack([np.interp(t, p2_ts[p2_m] - p2_start, p2_eeg[p2_m, c])
                    for c in range(14)], axis=1).astype(np.float64)

    if avg_ref:
        p1 -= p1.mean(axis=1, keepdims=True)
        p2 -= p2.mean(axis=1, keepdims=True)

    for s in [p1, p2]:
        for ch in range(14):
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd

    return p1.astype(np.float32), p2.astype(np.float32)


def print_spatial(r, label):
    """Print per-channel amplitude r and z, grouped by region."""
    va = r.get('volt_amp', {})
    per_r = va.get('per_channel_r', {})
    per_z = va.get('per_channel_z', {})

    # EPOC layout: frontal=AF3,F7,F3,FC5 / AF4,F8,F4,FC6
    # temporal=T7,T8, parietal=P7,P8, occipital=O1,O2
    regions = {
        'Frontal L': [0, 1, 2, 3],    # AF3, F7, F3, FC5
        'Frontal R': [13, 12, 11, 10], # AF4, F8, F4, FC6
        'Temporal': [4, 9],             # T7, T8
        'Parietal': [5, 8],             # P7, P8
        'Occipital': [6, 7],            # O1, O2
    }

    print(f"\n  {label}")
    print(f"  Pooled z: {va.get('pooled_z', 0):+.3f}, mean r: {va.get('mean_r', 0):+.4f}")
    print(f"  {'Channel':>8} {'r':>8} {'z':>8}")

    for region, chs in regions.items():
        rs = [per_r.get(ch, 0) for ch in chs]
        zs = [per_z.get(ch, 0) for ch in chs]
        mean_r_reg = np.mean(rs)
        mean_z_reg = np.mean(zs)
        print(f"  --- {region} (mean r={mean_r_reg:+.4f}, z={mean_z_reg:+.2f}) ---")
        for ch in chs:
            r_val = per_r.get(ch, 0)
            z_val = per_z.get(ch, 0)
            print(f"  {EPOC_CHANNEL_NAMES[ch]:>8} {r_val:>+7.4f} {z_val:>+7.2f}")


# ── Test 1: Real dyad, meditate_K, with avg-ref ─────────────────────────
print(f"\n{'='*60}")
print(f"  TEST 1: Real dyad — meditate_K (avg-ref)")
print(f"{'='*60}")
p1, p2 = extract_pair(s_y06, s_y06, mk[0], mk[1], avg_ref=True)
# Note: for real dyad, P2 comes from same session (p2_eeg)
# Re-extract properly using p2 from same session
p2_eeg = s_y06['p2_eeg']
p2_ts = s_y06['p2_eeg_ts']
p2_m = (p2_ts >= mk[0]) & (p2_ts < mk[1])
N = min(len(p1), p2_m.sum())
t = np.linspace(0, mk[1]-mk[0], N)
p2_real = np.stack([np.interp(t, p2_ts[p2_m]-mk[0], p2_eeg[p2_m, c])
                     for c in range(14)], axis=1).astype(np.float64)
p2_real -= p2_real.mean(axis=1, keepdims=True)
for ch in range(14):
    mu, sd = p2_real[:, ch].mean(), max(p2_real[:, ch].std(), 1e-8)
    p2_real[:, ch] = (p2_real[:, ch] - mu) / sd
p2_real = p2_real[:N].astype(np.float32)
p1 = p1[:N]

r1 = analyze_interbrain_cycles(p1, p2_real, FS, n_surrogates=200)
print_spatial(r1, "Real dyad, meditate_K, avg-ref")

# ── Test 2: Real dyad, conv_2, with avg-ref ─────────────────────────────
print(f"\n{'='*60}")
print(f"  TEST 2: Real dyad — conv_2 (avg-ref)")
print(f"{'='*60}")
p1c, _ = extract_pair(s_y06, s_y06, cv2[0], cv2[1], avg_ref=True)
p2_m2 = (p2_ts >= cv2[0]) & (p2_ts < cv2[1])
N2 = min(len(p1c), p2_m2.sum())
t2 = np.linspace(0, cv2[1]-cv2[0], N2)
p2c = np.stack([np.interp(t2, p2_ts[p2_m2]-cv2[0], p2_eeg[p2_m2, c])
                 for c in range(14)], axis=1).astype(np.float64)
p2c -= p2c.mean(axis=1, keepdims=True)
for ch in range(14):
    mu, sd = p2c[:, ch].mean(), max(p2c[:, ch].std(), 1e-8)
    p2c[:, ch] = (p2c[:, ch] - mu) / sd
p2c = p2c[:N2].astype(np.float32)
p1c = p1c[:N2]

r2 = analyze_interbrain_cycles(p1c, p2c, FS, n_surrogates=200)
print_spatial(r2, "Real dyad, conv_2, avg-ref")

# ── Test 3: Pseudo-dyad null — meditate_K P1 vs different session P2 ────
print(f"\n{'='*60}")
print(f"  TEST 3: Pseudo-dyad null (P1=y_06, P2={other_name})")
print(f"{'='*60}")
p1_null, p2_null = extract_pair(s_y06, s_other, mk[0], mk[1], avg_ref=True)
if p1_null is not None:
    r3 = analyze_interbrain_cycles(p1_null, p2_null, FS, n_surrogates=200)
    print_spatial(r3, f"Pseudo-dyad: y_06 P1 vs {other_name} P2")

# ── Test 4: Real dyad, meditate_K, WITHOUT avg-ref ──────────────────────
print(f"\n{'='*60}")
print(f"  TEST 4: Real dyad — meditate_K (NO avg-ref)")
print(f"{'='*60}")
# Recompute without avg-ref
p1_eeg_raw = s_y06['p1_eeg']
p1_ts_raw = s_y06['p1_eeg_ts']
p1_m_raw = (p1_ts_raw >= mk[0]) & (p1_ts_raw < mk[1])
N4 = min(p1_m_raw.sum(), p2_m.sum())
t4 = np.linspace(0, mk[1]-mk[0], N4)
p1_noref = np.stack([np.interp(t4, p1_ts_raw[p1_m_raw]-mk[0], p1_eeg_raw[p1_m_raw, c])
                      for c in range(14)], axis=1).astype(np.float64)
p2_noref = np.stack([np.interp(t4, p2_ts[p2_m]-mk[0], p2_eeg[p2_m, c])
                      for c in range(14)], axis=1).astype(np.float64)
# NO avg-ref, just z-normalize
for s in [p1_noref, p2_noref]:
    for ch in range(14):
        mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
        s[:, ch] = (s[:, ch] - mu) / sd

r4 = analyze_interbrain_cycles(p1_noref[:N4].astype(np.float32),
                                p2_noref[:N4].astype(np.float32),
                                FS, n_surrogates=200)
print_spatial(r4, "Real dyad, meditate_K, NO avg-ref")

print(f"\n{'='*60}")
