import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import numpy as np, pyxdf, glob
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

target_lsl = 61925.050
window = 5.0  # ±5s

data, _ = pyxdf.load_xdf(glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0], dejitter_timestamps=True)

# Find markers near this time
mt = {v[0]: t for s in data if s['info']['type'][0] == 'Markers' for t, v in zip(s['time_stamps'], s['time_series'])}
print("Markers:")
for name, t in sorted(mt.items(), key=lambda x: x[1]):
    dist = target_lsl - t
    if abs(dist) < 3000:
        marker = " <<<" if abs(dist) < 10 else ""
        print(f"  {name:20s} LSL={t:.3f}  ({dist:+.1f}s from target){marker}")

# Find which segment this falls in
segment = None
for seg in ['conv_1', 'conv_2', 'base_EO', 'base_EC', 'meditate_K', 'meditate_B']:
    s = mt.get(f'{seg}_start')
    e = mt.get(f'{seg}_stop')
    if s and e and s <= target_lsl <= e:
        segment = seg
        seg_offset = target_lsl - s
        print(f"\nTarget LSL={target_lsl:.3f} is in {segment} at offset {seg_offset:.1f}s")
        break

if segment is None:
    print(f"\nTarget LSL={target_lsl:.3f} is NOT inside any segment")
    # Find nearest segment boundaries
    for seg in ['conv_1', 'conv_2', 'base_EO', 'base_EC', 'meditate_K', 'meditate_B']:
        s = mt.get(f'{seg}_start')
        e = mt.get(f'{seg}_stop')
        if s and e:
            print(f"  {seg}: [{s:.1f}, {e:.1f}] dist={min(abs(target_lsl-s), abs(target_lsl-e)):.1f}s")

# Show AU values at target time
landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

for person in ['P1', 'P2']:
    ts, d = landmarks[person]
    # Find samples near target
    m = (ts >= target_lsl - window) & (ts <= target_lsl + window)
    if m.sum() == 0:
        print(f"\n{person}: no data near LSL={target_lsl:.3f}")
        continue

    print(f"\n{person} blendshapes around LSL={target_lsl:.3f}:")
    for dt in np.arange(-window, window + 0.5, 1.0):
        t = target_lsl + dt
        idx = np.argmin(np.abs(ts - t))
        if abs(ts[idx] - t) > 0.5:
            continue
        vals = d[idx, :52]
        active = [(i, MP_BLENDSHAPE_NAMES[i], float(vals[i]))
                  for i in range(52) if vals[i] > 0.10]
        marker = ">>>" if abs(dt) < 0.5 else "   "
        active_str = ', '.join(f'{n}={v:.2f}' for _, n, v in sorted(active, key=lambda x: -x[2])[:6])
        print(f"  {marker} t={dt:+5.1f}s: {active_str}")
