"""Quick inspection of a digest .npz + .json. Used during Step 3 sanity check."""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

DIGEST = Path("data/digest/v1")

def inspect(sid: str) -> None:
    j = DIGEST / f"{sid}.json"
    n = DIGEST / f"{sid}.npz"
    print(f"=== {sid} ===")
    with j.open() as fh:
        m = json.load(fh)
    print(f"  schema={m['schema_version']}  protocol={m['protocol']}  "
          f"duration={m['duration_s']:.1f}s")
    print(f"  pose_format={m['pose_format']}  modalities={m['modalities']}")
    r = m['roles']
    print(f"  roles: p1={r['p1_role']} ({r['p1_name']!r})  "
          f"p2={r['p2_role']} ({r['p2_name']!r})  src={r['role_source']}")
    print(f"  marker_sources={m['marker_sources']}")
    print(f"  n_markers={len(m['markers'])}")
    proto_labels = [lab for _t, lab in m['markers']
                    if any(k in lab for k in ('base_','conv_','meditate','PE_','baseline'))]
    print(f"  protocol_markers={proto_labels}")
    print(f"  missing_markers={m['quality_flags']['missing_markers']}")
    print(f"  excluded_modalities={m['quality_flags']['excluded_modalities']}")
    arrs = np.load(n)
    for k in sorted(arrs.keys()):
        a = arrs[k]
        print(f"  npz {k}: shape={a.shape} dtype={a.dtype}")

if __name__ == "__main__":
    sids = sys.argv[1:] or ["y_06", "Y_55_04272026"]
    for sid in sids:
        inspect(sid)
        print()
