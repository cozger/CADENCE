import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import numpy as np, pyxdf, glob
from cadence.significance.bl_coupling import bl_two_stage_coupling

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
t_end = mt['conv_2_stop']
dur = t_end - t_start
T = int(dur * 30)
t_grid = np.linspace(0, dur, T)
sigs = {}
for p in ['P1', 'P2']:
    ts, d = landmarks[p]
    m = (ts >= t_start) & (ts <= t_end)
    sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c]) for c in range(52)], axis=1)

result = bl_two_stage_coupling(sigs['P1'], sigs['P2'], 30.0, seed=42)
z = result.z_continuous
out_rate = result.output_rate

target_offset = 61925.050 - t_start
z_idx = int(target_offset * out_rate)
print(f"conv_2 offset: {target_offset:.1f}s")
print(f"z at target: {z[z_idx]:.2f}")
print(f"mask at target: {result.mask_continuous[z_idx]}")
print(f"\nz in ±10s window:")
for dt in range(-10, 11):
    idx = z_idx + int(dt * out_rate)
    if 0 <= idx < len(z):
        m = '>>>' if dt == 0 else '   '
        coupled = '*' if result.mask_continuous[idx] else ' '
        print(f"  {m} t={target_offset+dt:6.1f}s  z={z[idx]:6.2f} {coupled}")
