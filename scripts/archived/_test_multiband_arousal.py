"""Multi-band arousal semi-synthetic + y_06 real data scan.

Part 1: Inject broadband arousal coupling, detect with theta/alpha/beta.
Part 2: Re-scan y_06 with all three bands.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_arousal)
from cadence.significance.fast_cycles import (
    analyze_interbrain_cycles_multiband, EEG_BANDS)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0; SEED = 42; K = 200

# ── Load data ────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])

# Cross-dyad for semi-synthetic
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=1800)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + 1800)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(1800 * FS))
t_target = np.linspace(0, 1800, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(14)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]

for sig in [p1_eeg, p2_eeg]:
    sig -= sig.mean(axis=1, keepdims=True).astype(sig.dtype)
    for ch in range(14):
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)


def print_multiband(r, label):
    """Print multi-band results compactly."""
    print(f"\n  {label}")
    comb = r['combined']
    # Header
    bands = list(r['per_band'].keys())
    hdr = f"  {'feature':>12}"
    for b in bands:
        hdr += f"  {b:>8}"
    hdr += f"  {'combined':>10}"
    print(hdr)
    print(f"  {'-'*12}" + f"  {'-'*8}" * len(bands) + f"  {'-'*10}")

    for feat in ['volt_amp', 'period', 'time_rdsym']:
        row = f"  {feat:>12}"
        for b in bands:
            z = r['per_band'][b].get(feat, {}).get('pooled_z', 0)
            row += f"  {z:>+7.2f}"
        sz = comb.get(feat, {}).get('stouffer_z', 0)
        row += f"  {sz:>+9.2f}"
        print(row)

    # Burst
    row = f"  {'burst_cooc':>12}"
    for b in bands:
        z = r['per_band'][b].get('burst_cooc', {}).get('pooled_z', 0)
        row += f"  {z:>+7.2f}"
    sz = comb.get('burst_cooc', {}).get('stouffer_z', 0)
    row += f"  {sz:>+9.2f}"
    print(row)


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Semi-synthetic — broadband arousal injection
# ══════════════════════════════════════════════════════════════════════════
print(f"{'#'*70}")
print(f"  PART 1: Semi-synthetic — per-band narrowband arousal injection")
print(f"  Each band has its own spatial pattern:")
print(f"    theta (4-8 Hz)  → frontal dominant")
print(f"    alpha (8-13 Hz) → occipital dominant")
print(f"    beta (13-30 Hz) → centroparietal dominant")
print(f"  Inject one band at a time, detect in all three → test selectivity")
print(f"{'#'*70}")

# Band-specific injection configs: (source_band, spatial_mode)
BAND_CONFIGS = {
    'theta': ((4.0, 8.0),   'frontal'),
    'alpha': ((8.0, 13.0),  'occipital'),
    'beta':  ((13.0, 30.0), 'centroparietal'),
}

# Null
t0 = time.perf_counter()
r_null = analyze_interbrain_cycles_multiband(
    p1_eeg, p2_eeg, FS, n_surrogates=K, seed=SEED)
print_multiband(r_null, f"kappa=0.00 NULL ({time.perf_counter()-t0:.1f}s)")

# Per-source-band injection at kappa=0.30
for src_name, (src_band, src_spatial) in BAND_CONFIGS.items():
    t0 = time.perf_counter()
    p2c, kpc, _ = inject_eeg_coupling_arousal(
        p1_eeg, p2_eeg, gate, 0.30,
        lag_s=1.0, source_band=src_band,
        spatial_mode=src_spatial, fs=FS, seed=SEED)
    for ch in range(14):
        mu, sd = p2c[:, ch].mean(), max(p2c[:, ch].std(), 1e-8)
        p2c[:, ch] = (p2c[:, ch] - mu) / sd

    r = analyze_interbrain_cycles_multiband(
        p1_eeg, p2c, FS, n_surrogates=K, seed=SEED)
    elapsed = time.perf_counter() - t0
    print_multiband(r, f"inject {src_name} {src_band} / {src_spatial} ({elapsed:.1f}s)")

# Kappa sweep with theta source (frontal)
print(f"\n  --- Theta/frontal kappa sweep ---")
for kappa in [0.15, 0.30, 0.50]:
    t0 = time.perf_counter()
    p2c, _, _ = inject_eeg_coupling_arousal(
        p1_eeg, p2_eeg, gate, kappa,
        lag_s=1.0, source_band=(4.0, 8.0),
        spatial_mode='frontal', fs=FS, seed=SEED)
    for ch in range(14):
        mu, sd = p2c[:, ch].mean(), max(p2c[:, ch].std(), 1e-8)
        p2c[:, ch] = (p2c[:, ch] - mu) / sd
    r = analyze_interbrain_cycles_multiband(
        p1_eeg, p2c, FS, n_surrogates=K, seed=SEED)
    print_multiband(r, f"kappa={kappa:.2f}, theta/frontal ({time.perf_counter()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Real data — y_06 per condition, multi-band
# ══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*70}")
print(f"  PART 2: Real data y_06 — multi-band per condition")
print(f"{'#'*70}")

y06_name, y06_path = next((n, p) for n, p in entries if 'y_06' in n)
s_y06 = load_session_from_cache(y06_path, config=cfg)
intervals = parse_condition_intervals(s_y06)
p1_role = s_y06.get('p1_role', '?')
p2_role = s_y06.get('p2_role', '?')
print(f"  {y06_name}: P1={p1_role}, P2={p2_role}")

y06_p1 = s_y06['p1_eeg']
y06_p2 = s_y06['p2_eeg']
y06_p1_ts = s_y06['p1_eeg_ts']
y06_p2_ts = s_y06['p2_eeg_ts']


def extract_real(start_s, end_s):
    p1_m = (y06_p1_ts >= start_s) & (y06_p1_ts < end_s)
    p2_m = (y06_p2_ts >= start_s) & (y06_p2_ts < end_s)
    dur = end_s - start_s
    Ns = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if Ns < int(10 * FS):
        return None, None
    t = np.linspace(0, dur, Ns)
    p1 = np.stack([np.interp(t, y06_p1_ts[p1_m]-start_s, y06_p1[p1_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    p2 = np.stack([np.interp(t, y06_p2_ts[p2_m]-start_s, y06_p2[p2_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    for s in [p1, p2]:
        s -= s.mean(axis=1, keepdims=True)
        for ch in range(14):
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd
    return p1.astype(np.float32), p2.astype(np.float32)


targets = ['meditate_K', 'conv_1', 'conv_2', 'base_EO', 'base_EC', 'meditate_B']
real_results = {}

for start, end, cond in intervals:
    if cond not in targets:
        continue
    p1r, p2r = extract_real(start, end)
    if p1r is None:
        continue
    t0 = time.perf_counter()
    r = analyze_interbrain_cycles_multiband(
        p1r, p2r, FS, n_surrogates=K, seed=SEED)
    elapsed = time.perf_counter() - t0
    real_results[cond] = r
    print_multiband(r, f"{cond} ({end-start:.0f}s, {elapsed:.1f}s)")


# ── Summary table ────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  SUMMARY: volt_amp z per band (y_06 real data)")
print(f"{'='*70}")
print(f"  {'condition':>15} {'theta':>8} {'alpha':>8} {'beta':>8} {'combined':>10}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

for cond in targets:
    if cond not in real_results:
        continue
    r = real_results[cond]
    tz = r['per_band'].get('theta', {}).get('volt_amp', {}).get('pooled_z', 0)
    az = r['per_band'].get('alpha', {}).get('volt_amp', {}).get('pooled_z', 0)
    bz = r['per_band'].get('beta', {}).get('volt_amp', {}).get('pooled_z', 0)
    cz = r['combined'].get('volt_amp', {}).get('stouffer_z', 0)
    print(f"  {cond:>15} {tz:>+7.2f} {az:>+7.2f} {bz:>+7.2f} {cz:>+9.2f}")

print(f"{'='*70}")
