"""Step 5.3 acceptance: verify role assignment for previously-mislabelled sessions.

Plan acceptance criterion:
    For Y_55, y_59, y_64, y_66, y26, y_65: roles correctly assigned.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

DIGEST = Path("data/digest/v1")

MISLABELLED = [
    "Y_55_04272026",
    "y_59_04172026",
    "y_64_04272026",
    "y_66_041626",
    "y26_022728",
    "y_65_04242026",
]


def main() -> int:
    print(f"{'session':25s}  {'p1_role':10s}  {'p1_name':14s}  {'p2_role':10s}  {'p2_name':14s}  {'src':16s}")
    print("-" * 100)
    failed = []
    for sid in MISLABELLED:
        path = DIGEST / f"{sid}.json"
        if not path.is_file():
            print(f"{sid:25s}  MISSING DIGEST: {path}")
            failed.append(sid)
            continue
        with path.open() as fh:
            m = json.load(fh)
        r = m["roles"]
        # Acceptance: not unknown, role_source is xdf_landmark or xdf_ra_alias
        ok = (r["p1_name"] != "unknown" and r["p2_name"] != "unknown"
              and r["role_source"] in ("xdf_landmark", "xdf_ra_alias")
              and r["p1_role"] != r["p2_role"]
              and "therapist" in (r["p1_role"], r["p2_role"]))
        tag = "  OK" if ok else "  FAIL"
        print(f"{sid:25s}  {r['p1_role']:10s}  {r['p1_name']:14s}  "
              f"{r['p2_role']:10s}  {r['p2_name']:14s}  {r['role_source']:16s}{tag}")
        if not ok:
            failed.append(sid)
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        return 1
    print("\nAll 6 previously-mislabelled sessions: roles correctly assigned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
