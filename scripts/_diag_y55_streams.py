"""Diagnose stream timestamp ranges in Y_55 to find the duration_s leak."""
from __future__ import annotations
import numpy as np
import pyxdf

streams, _ = pyxdf.load_xdf(r"raw sessions/Y_55_04272026.xdf",
                            dejitter_timestamps=False, synchronize_clocks=False)
print(f"{'name':30s} {'type':12s} {'n':>10s} {'t0':>15s} {'tN':>15s} {'span':>10s}")
for s in streams:
    info = s.get("info", {})
    name = info.get("name", [""])[0]
    stype = info.get("type", [""])[0]
    ts = np.asarray(s.get("time_stamps", []), dtype=np.float64)
    if len(ts) == 0:
        continue
    print(f"{name:30s} {stype:12s} {len(ts):>10d} {ts[0]:>15.2f} {ts[-1]:>15.2f} {ts[-1]-ts[0]:>10.2f}")
