"""Check which sessions are missing pose33 keys."""
import torch  # noqa: F401
import numpy as np
from pathlib import Path

REPO = Path('C:/Users/optilab/desktop/CADENCE')
preproc = REPO / 'data' / 'preproc' / 'pose' / 'v1'

for npz_path in sorted(preproc.glob('*.npz')):
    sid = npz_path.stem
    npz = np.load(npz_path, allow_pickle=False)
    keys = list(npz.keys())
    has_p1 = 'p1_pose33' in keys
    has_p2 = 'p2_pose33' in keys
    flag = 'OK' if (has_p1 and has_p2) else 'MISSING'
    print(f'{sid:25s} {flag:10s} p1_pose33={has_p1} p2_pose33={has_p2} keys={keys}')
