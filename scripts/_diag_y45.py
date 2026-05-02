"""Diagnose Y_45 data coverage vs markers."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf

xdf_path = 'C:/Users/optilab/Desktop/CADENCE/raw sessions/Y_45_03302026.xdf'
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

print("XDF Streams:")
for stream in data:
    name = stream['info']['name'][0]
    stype = stream['info']['type'][0]
    ts = stream['time_stamps']
    n = len(ts)
    if n > 0:
        print(f"  {name:>30s} ({stype:>15s}): n={n:>8d}, "
              f"t=[{ts[0]:.1f}, {ts[-1]:.1f}], span={ts[-1]-ts[0]:.1f}s")
    else:
        print(f"  {name:>30s} ({stype:>15s}): EMPTY")

# Find condition markers
print("\nCondition markers:")
for stream in data:
    if stream['info']['type'][0] == 'Markers' or len(stream['time_stamps']) < 100:
        ts = stream['time_stamps']
        samples = stream['time_series']
        for i, (t, s) in enumerate(zip(ts, samples)):
            label = s[0] if isinstance(s, list) else str(s)
            if any(cond in label for cond in ['base_EO', 'base_EC', 'conv_', 'meditate', 'PE']):
                print(f"  t={t:.1f} ({(t-ts[0]):.0f}s from stream start): {label}")
