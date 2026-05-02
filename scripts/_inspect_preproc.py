"""Quick inspection of any data/preproc/<modality>/v1/<sid>.{npz,json} pair."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def inspect(modality: str, sid: str, version: str = "v1") -> None:
    base = Path("data/preproc") / modality / version
    j = base / f"{sid}.json"
    n = base / f"{sid}.npz"
    print(f"=== {modality}/{sid} ===")
    if not j.is_file():
        print(f"  MISSING JSON: {j}")
        return
    if not n.is_file():
        print(f"  MISSING NPZ: {n}")
        return
    with j.open() as fh:
        m = json.load(fh)
    print(f"  modality={m.get('modality')}  ver={m.get('modality_version')}  "
          f"digest_md5={m.get('digest_xdf_md5','')[:12]}...")
    if "params" in m:
        for k, v in m["params"].items():
            print(f"  param.{k}: {v}")
    if "participants" in m:
        for p, summ in m["participants"].items():
            print(f"  {p}: {summ}")
    arrs = np.load(n, allow_pickle=True)
    for k in sorted(arrs.keys()):
        a = arrs[k]
        if np.issubdtype(a.dtype, np.floating):
            nan_pct = float(np.isnan(a).mean()) * 100
            print(f"  npz {k}: shape={a.shape} dtype={a.dtype}  NaN%={nan_pct:.2f}")
        else:
            print(f"  npz {k}: shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: _inspect_preproc.py <modality> <sid> [<sid> ...]")
        sys.exit(1)
    modality = args[0]
    for sid in args[1:]:
        inspect(modality, sid)
        print()
