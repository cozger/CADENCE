"""Show exactly what's happening at the highest coupling timepoints.

For the top-5 z-score peaks, show second-by-second AU values for both
people in a window around the peak.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob

from cadence.significance.bl_coupling import bl_two_stage_coupling
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

FS = 30.0

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

marker_times = {}
for stream in data:
    if stream['info']['type'][0] == 'Markers':
        for t, v in zip(stream['time_stamps'], stream['time_series']):
            marker_times[v[0]] = t

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

t_start = marker_times['conv_1_start']
t_end = marker_times['conv_1_stop']
dur = t_end - t_start
T = int(dur * FS)
t_grid = np.linspace(0, dur, T)

sigs = {}
for p in ['P1', 'P2']:
    ts, d = landmarks[p]
    m = (ts >= t_start) & (ts <= t_end)
    sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)

p1_raw = sigs['P1']
p2_raw = sigs['P2']

result = bl_two_stage_coupling(p1_raw, p2_raw, FS, seed=42)
z = result.z_continuous
out_rate = result.output_rate
z_times = np.arange(len(z)) / out_rate

print(f"conv_1: {dur:.0f}s, coupling={result.mask_continuous.mean():.1%}")
print(f"z range: [{z.min():.2f}, {z.max():.2f}]")
print(f"Estimated lag: {result.estimated_lag_s:.2f}s")

# Find top-5 z-score peaks (at least 10s apart)
peak_indices = []
z_copy = z.copy()
for _ in range(5):
    idx = np.argmax(z_copy)
    peak_indices.append(idx)
    # Zero out ±10s around this peak
    s = max(0, idx - int(10 * out_rate))
    e = min(len(z_copy), idx + int(10 * out_rate))
    z_copy[s:e] = -999

# Key AUs to show (most interpretable)
SHOW_AUS = [
    (44, 'smileL'), (45, 'smileR'),
    (25, 'jawOpen'),
    (7, 'chkSqL'), (8, 'chkSqR'),
    (19, 'eSqL'), (20, 'eSqR'),
    (3, 'brInUp'), (4, 'brOUpL'), (5, 'brOUpR'),
    (1, 'brDnL'), (2, 'brDnR'),
    (30, 'frownL'), (31, 'frownR'),
    (42, 'shrugLo'), (43, 'shrugUp'),
    (9, 'blinkL'), (10, 'blinkR'),
    (33, 'mthL'), (39, 'mthR'),
]

for rank, peak_idx in enumerate(peak_indices):
    peak_t = z_times[peak_idx]
    peak_z = z[peak_idx]

    print(f"\n{'='*80}")
    print(f"  PEAK #{rank+1}: t={peak_t:.1f}s, z={peak_z:.2f}")
    print(f"{'='*80}")

    # Show ±5s window at 1s resolution
    window_s = 5
    print(f"\n  Time-course ±{window_s}s (raw AU values, 1s steps):")

    # Header
    header = f"  {'t':>6s} {'z':>5s} |"
    for au_idx, au_short in SHOW_AUS:
        header += f" {au_short:>6s}"
    print(f"\n  --- P1 (patient) ---")
    print(header)

    for dt in range(-window_s, window_s + 1):
        t = peak_t + dt
        raw_idx = int(t * FS)
        z_idx = int(t * out_rate)
        if raw_idx < 0 or raw_idx >= T or z_idx < 0 or z_idx >= len(z):
            continue
        # Average over ±0.5s for stability
        s = max(0, raw_idx - int(0.5 * FS))
        e = min(T, raw_idx + int(0.5 * FS))
        p1_vals = p1_raw[s:e].mean(axis=0)
        z_val = z[min(z_idx, len(z)-1)]

        marker = '>>>' if dt == 0 else '   '
        row = f"{marker}{t:5.1f}s {z_val:5.2f} |"
        for au_idx, _ in SHOW_AUS:
            v = p1_vals[au_idx]
            row += f" {v:6.3f}"
        print(row)

    print(f"\n  --- P2 (therapist) ---")
    print(header)

    for dt in range(-window_s, window_s + 1):
        t = peak_t + dt
        raw_idx = int(t * FS)
        z_idx = int(t * out_rate)
        if raw_idx < 0 or raw_idx >= T or z_idx < 0 or z_idx >= len(z):
            continue
        s = max(0, raw_idx - int(0.5 * FS))
        e = min(T, raw_idx + int(0.5 * FS))
        p2_vals = p2_raw[s:e].mean(axis=0)
        z_val = z[min(z_idx, len(z)-1)]

        marker = '>>>' if dt == 0 else '   '
        row = f"{marker}{t:5.1f}s {z_val:5.2f} |"
        for au_idx, _ in SHOW_AUS:
            v = p2_vals[au_idx]
            row += f" {v:6.3f}"
        print(row)

    # Summary: what's both people doing at the peak?
    peak_raw = int(peak_t * FS)
    s = max(0, peak_raw - int(0.5 * FS))
    e = min(T, peak_raw + int(0.5 * FS))
    p1_peak = p1_raw[s:e].mean(axis=0)
    p2_peak = p2_raw[s:e].mean(axis=0)

    print(f"\n  At peak (t={peak_t:.1f}s):")
    print(f"    P1 active AUs (>0.15):", end='')
    for i in range(52):
        if p1_peak[i] > 0.15:
            print(f" {MP_BLENDSHAPE_NAMES[i]}={p1_peak[i]:.2f}", end='')
    print()
    print(f"    P2 active AUs (>0.15):", end='')
    for i in range(52):
        if p2_peak[i] > 0.15:
            print(f" {MP_BLENDSHAPE_NAMES[i]}={p2_peak[i]:.2f}", end='')
    print()
