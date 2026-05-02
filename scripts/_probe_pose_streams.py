#!/usr/bin/env python
"""Probe pose stream shapes across the corpus.

v2: report ALL streams in target XDFs (not just P*_pose) and inspect channel 132
of the 133-channel pose to understand what the extra channel encodes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pyxdf

_PROJ_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJ_ROOT / "raw sessions"


def probe_one(xdf: Path) -> None:
    print(f"\n=== {xdf.name} ===")
    try:
        streams, _ = pyxdf.load_xdf(
            str(xdf), dejitter_timestamps=False, synchronize_clocks=False,
        )
    except Exception as e:
        print(f"  ERROR: {e!r}")
        return

    print(f"  All streams ({len(streams)}):")
    for s in streams:
        info = s.get("info", {})
        name = info.get("name", [""])[0]
        stype = info.get("type", [""])[0]
        try:
            data = np.asarray(s.get("time_series", []))
            shape = data.shape
        except Exception:
            shape = "?"
        print(f"    {name!r:32s} type={stype!r:12s} shape={shape}")

    print()
    for s in streams:
        info = s.get("info", {})
        name = info.get("name", [""])[0]
        if name not in ("P1_pose", "P2_pose"):
            continue
        try:
            data = np.asarray(s.get("time_series", []))
        except Exception:
            continue
        if data.size == 0 or data.ndim != 2:
            continue

        n_ch = data.shape[1]
        print(f"  {name}: shape={data.shape}")

        if n_ch == 133:
            # Test hypothesis: first 132 are mediapipe33 (33 × 4), last is metadata.
            ch132 = data[:, 132]
            mp_part = data[:, :132]
            print(f"    last channel (idx 132): "
                  f"min={ch132.min():.4f}  max={ch132.max():.4f}  "
                  f"mean={ch132.mean():.4f}  std={ch132.std():.4f}  "
                  f"unique_count={len(np.unique(ch132))}")
            print(f"    last channel first 20: {ch132[:20]}")
            print(f"    monotonic increasing? {bool(np.all(np.diff(ch132[:1000]) >= 0))}")
            mp_reshape_ok = mp_part.size == data.shape[0] * 33 * 4
            print(f"    first 132 reshapable to (N, 33, 4)? {mp_reshape_ok}")
            if mp_reshape_ok:
                kp = mp_part.reshape(data.shape[0], 33, 4)
                # visibility column should be in [0,1] for mediapipe
                vis = kp[:, :, 3]
                print(f"    after reshape (N, 33, 4): visibility col range=[{vis.min():.4f}, {vis.max():.4f}]  "
                      f"mean={vis.mean():.4f}")
        elif n_ch == 132:
            kp = data.reshape(data.shape[0], 33, 4)
            vis = kp[:, :, 3]
            print(f"    standard mediapipe33: visibility col range=[{vis.min():.4f}, {vis.max():.4f}]  "
                  f"mean={vis.mean():.4f}")


def main() -> None:
    targets = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            matches = sorted(RAW_DIR.glob(f"*{arg}*.xdf"))
            targets.extend(matches)
    else:
        # Probe one mediapipe33 session and three "133" sessions
        for sub in ("y_06", "Y_55", "y_64", "y_66"):
            matches = sorted(RAW_DIR.glob(f"*{sub}*.xdf"))
            targets.extend(matches[:1])

    if not targets:
        print("No XDFs to probe", file=sys.stderr)
        sys.exit(1)

    for xdf in targets:
        probe_one(xdf)


if __name__ == "__main__":
    main()
