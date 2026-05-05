import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import numpy as np, pyxdf, glob
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

data, _ = pyxdf.load_xdf(glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0], dejitter_timestamps=True)
mt = {v[0]: t for s in data if s['info']['type'][0] == 'Markers' for t, v in zip(s['time_stamps'], s['time_series'])}
landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']), np.array(stream['time_series'], dtype=np.float32))

t_start = mt['conv_2_start']
# The high-z window is at conv_2 offset 269-274s
target_lsl = t_start + 271  # middle of high-z window

print(f"Checking conv_2 offset ~271s (LSL={target_lsl:.1f})")
print(f"This is the HIGH-z window (z=5-6) near the smile event\n")

for person in ['P1', 'P2']:
    ts, d = landmarks[person]
    print(f"{person}:")
    for dt in np.arange(-3, 4, 1.0):
        t = target_lsl + dt
        idx = np.argmin(np.abs(ts - t))
        vals = d[idx, :52]
        active = [(i, MP_BLENDSHAPE_NAMES[i], float(vals[i]))
                  for i in range(52) if vals[i] > 0.10]
        marker = ">>>" if abs(dt) < 0.5 else "   "
        active_str = ', '.join(f'{n}={v:.2f}' for _, n, v in sorted(active, key=lambda x: -x[2])[:6])
        print(f"  {marker} t={271+dt:5.0f}s: {active_str}")
    print()
