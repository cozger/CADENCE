"""Step 5.3 acceptance: marker count non-decreasing vs old MCCT cache.

Plan acceptance criterion:
    For Y_45, y_51, y_59, y_64, y_65, y_66: marker count is non-decreasing
    vs old MCCT cache. Strict-greater not required (some sessions may have empty
    EventMarkers); decrease *is* a regression.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

DIGEST = Path("data/digest/v1")
MCCT_CACHE = Path(r"C:\Users\optilab\desktop\MCCT\session_cache")

CHECK = [
    "Y_45_03302026",
    "y_51_04092026",
    "y_59_04172026",
    "y_64_04272026",
    "y_65_04242026",
    "y_66_041626",
]


def _mcct_marker_count(sid: str) -> int | None:
    """Read marker count from the matching MCCT cache .json for a session.

    MCCT uses ``<12-char-hash>_<sid>.json`` naming. Pick the most-recent if
    multiple hashes exist for the same session.
    """
    if not MCCT_CACHE.is_dir():
        return None
    candidates = sorted(MCCT_CACHE.glob(f"*_{sid}.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    cache_json = candidates[0]
    with cache_json.open() as fh:
        meta = json.load(fh)
    if "markers" in meta and isinstance(meta["markers"], list):
        return len(meta["markers"])
    return None


def main() -> int:
    print(f"{'session':25s}  {'new':>8s}  {'mcct':>8s}  status")
    print("-" * 60)
    failed: list[str] = []
    for sid in CHECK:
        path = DIGEST / f"{sid}.json"
        if not path.is_file():
            print(f"{sid:25s}  MISSING DIGEST")
            failed.append(sid)
            continue
        with path.open() as fh:
            m = json.load(fh)
        n_new = len(m.get("markers", []))
        n_mcct = _mcct_marker_count(sid)
        if n_mcct is None:
            print(f"{sid:25s}  {n_new:>8d}  {'?':>8s}  no_mcct_cache")
            continue
        decrease = n_new < n_mcct
        status = "REGRESSION" if decrease else "OK"
        print(f"{sid:25s}  {n_new:>8d}  {n_mcct:>8d}  {status}")
        if decrease:
            failed.append(sid)
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        return 1
    print("\nMarker counts: non-decreasing vs MCCT cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
