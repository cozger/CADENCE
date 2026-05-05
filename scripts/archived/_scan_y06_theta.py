"""Scan y_06 for inter-brain theta coupling per condition.

Runs theta PLV (4-8 Hz) on raw 256 Hz EEG with avg-ref, per condition:
  base_EO, base_EC, conv_1, conv_2, meditate_B, meditate_K

Reports z-scores, coupling fraction, and per-channel breakdown.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.coherence_localization import wpli_temporal_localization
from cadence.constants import EPOC_CHANNEL_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS = 256.0
K = 100
SEED = 42

# ── Load y_06 ────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
print(f"Loading {name} from {path}...")
session = load_session_from_cache(path, config=cfg)

p1_role = session.get('p1_role', 'unknown')
p2_role = session.get('p2_role', 'unknown')
print(f"P1: {p1_role}, P2: {p2_role}")
print(f"Duration: {session.get('duration', 0):.0f}s")

# ── Parse conditions ─────────────────────────────────────────────────────
intervals = parse_condition_intervals(session)
print(f"\nConditions:")
for start, end, cond in intervals:
    print(f"  {cond:>15}: {start:.0f}s - {end:.0f}s ({end-start:.0f}s)")

# ── Get raw EEG + timestamps ─────────────────────────────────────────────
p1_eeg = session['p1_eeg']     # (T, 14) @ 256 Hz
p2_eeg = session['p2_eeg']
p1_ts = session['p1_eeg_ts']   # (T,) session-relative seconds
p2_ts = session['p2_eeg_ts']

print(f"\nP1 EEG: {p1_eeg.shape}, P2 EEG: {p2_eeg.shape}")
print(f"P1 time: {p1_ts[0]:.1f} - {p1_ts[-1]:.1f}s")
print(f"P2 time: {p2_ts[0]:.1f} - {p2_ts[-1]:.1f}s")

# Theta PLV params
freqs_theta = np.logspace(np.log10(4.0), np.log10(8.0), 5)
# Also test broadband for comparison
freqs_broad = np.logspace(np.log10(2.0), np.log10(40.0), 30)


def avg_reference(eeg):
    """Apply average re-reference (removes common CMS/DRL noise)."""
    return eeg - eeg.mean(axis=1, keepdims=True)


def extract_condition(p1_eeg, p2_eeg, p1_ts, p2_ts, start_s, end_s):
    """Extract aligned EEG segment for a condition interval."""
    # P1 mask
    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    p2_m = (p2_ts >= start_s) & (p2_ts < end_s)

    p1_seg = p1_eeg[p1_m].copy()
    p2_seg = p2_eeg[p2_m].copy()

    # Align to same length via interpolation to common time grid
    dur = end_s - start_s
    N = min(len(p1_seg), len(p2_seg), int(dur * FS))
    if N < int(10 * FS):  # skip if < 10s
        return None, None, N

    t_common = np.linspace(0, dur, N)

    p1_t = p1_ts[p1_m] - start_s
    p2_t = p2_ts[p2_m] - start_s

    p1_out = np.stack([np.interp(t_common, p1_t[:len(p1_seg)], p1_seg[:, c])
                        for c in range(14)], axis=1).astype(np.float32)
    p2_out = np.stack([np.interp(t_common, p2_t[:len(p2_seg)], p2_seg[:, c])
                        for c in range(14)], axis=1).astype(np.float32)

    # Average re-reference
    p1_out = avg_reference(p1_out)
    p2_out = avg_reference(p2_out)

    # Z-normalize per channel
    for ch in range(14):
        for sig in [p1_out, p2_out]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd

    return p1_out, p2_out, N


def run_plv(p1, p2, freqs, label, window_s=20.0, smooth_s=15.0):
    """Run PLV and report results."""
    N = len(p1)
    dur = N / FS

    if dur < 30:
        print(f"    {label}: too short ({dur:.0f}s)")
        return None

    t0 = time.perf_counter()
    mask, z_agg, per_ch_z, diag = wpli_temporal_localization(
        p1, p2, FS,
        channels=list(range(14)),
        center_freqs=freqs, n_surrogates=K,
        n_cycles=[3, 7], window_s=window_s, stride_s=0.5,
        smooth_s=smooth_s, target_fa=0.05, min_event_s=5.0,
        metric='plv', seed=SEED, device=device,
        aggregation='pooled')
    elapsed = time.perf_counter() - t0

    cf = mask.mean()
    zm = z_agg.mean()
    zmax = z_agg.max()
    zp95 = np.percentile(z_agg, 95)
    thr = diag['z_threshold']

    print(f"    {label} ({dur:.0f}s, {elapsed:.1f}s):")
    print(f"      z_mean={zm:+.3f}  z_max={zmax:+.3f}  z_p95={zp95:+.3f}  "
          f"thr={thr:.2f}  coupling={cf:.1%}")

    # Per-channel z (mean across time)
    if per_ch_z is not None and per_ch_z.shape[0] == 14:
        ch_z = per_ch_z.mean(axis=1)
        top3 = np.argsort(-ch_z)[:3]
        ch_str = ", ".join(f"{EPOC_CHANNEL_NAMES[i]}={ch_z[i]:+.2f}" for i in top3)
        print(f"      top channels: {ch_str}")

    return {
        'z_mean': zm, 'z_max': zmax, 'z_p95': zp95,
        'coupling_frac': cf, 'threshold': thr, 'duration': dur,
    }


# ── Scan per condition ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  THETA PLV SCAN: {name} ({p1_role} vs {p2_role})")
print(f"{'='*70}")

results = {}

for start, end, cond in intervals:
    print(f"\n  --- {cond} ({end-start:.0f}s) ---")

    p1_seg, p2_seg, N = extract_condition(
        p1_eeg, p2_eeg, p1_ts, p2_ts, start, end)

    if p1_seg is None:
        print(f"    Skipped (too short: {N/FS:.0f}s)")
        continue

    # Theta PLV
    r = run_plv(p1_seg, p2_seg, freqs_theta, "theta (4-8 Hz)")
    if r:
        results[(cond, 'theta')] = r

    # Broadband PLV for comparison
    r2 = run_plv(p1_seg, p2_seg, freqs_broad, "broadband (2-40 Hz)")
    if r2:
        results[(cond, 'broadband')] = r2

# ── Full session (all conditions combined) ───────────────────────────────
print(f"\n  --- FULL SESSION ---")

# Use the longest contiguous overlap
t_start = max(p1_ts[0], p2_ts[0])
t_end = min(p1_ts[-1], p2_ts[-1])
p1_full, p2_full, N_full = extract_condition(
    p1_eeg, p2_eeg, p1_ts, p2_ts, t_start, t_end)
if p1_full is not None:
    r = run_plv(p1_full, p2_full, freqs_theta, "theta full session")
    results[('full', 'theta')] = r

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Condition':>15} {'Band':>10} {'z_mean':>8} {'z_max':>8} "
      f"{'z_p95':>8} {'coupling':>10} {'dur':>6}")
print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

for (cond, band), r in sorted(results.items()):
    print(f"  {cond:>15} {band:>10} {r['z_mean']:>+7.3f} {r['z_max']:>+7.3f} "
          f"{r['z_p95']:>+7.3f} {r['coupling_frac']:>9.1%} {r['duration']:>5.0f}s")

print(f"\n  Roles: P1={p1_role}, P2={p2_role}")
print(f"{'='*70}")
