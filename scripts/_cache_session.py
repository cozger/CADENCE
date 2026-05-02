"""Cache a raw XDF session and inspect its properties."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from cadence.data.alignment import load_and_preprocess_cached

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

xdf_path = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/optilab/Desktop/CADENCE/raw sessions/Y_45_03302026.xdf'
print(f"Caching: {xdf_path}")
session = load_and_preprocess_cached(xdf_path, cache_dir=config['session_cache'])

print(f"\nDuration: {session['duration']:.1f}s ({session['duration']/60:.1f} min)")
print(f"P1 role: {session.get('p1_role', 'unknown')}")
print(f"P2 role: {session.get('p2_role', 'unknown')}")

bl_ts = session.get('p1_blendshapes_ts')
if bl_ts is not None and len(bl_ts) > 100:
    dt = np.median(np.diff(bl_ts[:2000]))
    print(f"BL rate: {1.0/dt:.1f} Hz (n={len(bl_ts)} samples)")

for p in ['p1', 'p2']:
    pv = session.get(f'{p}_pose_features_valid')
    if pv is not None:
        valid_pct = pv.mean() * 100
        print(f"{p} pose valid: {valid_pct:.1f}% ({pv.sum()}/{len(pv)})")
    else:
        print(f"{p} pose: no validity data")

markers = session.get('markers', {})
print(f"\nMarkers ({len(markers)}):")
for k, v in sorted(markers.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0):
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
