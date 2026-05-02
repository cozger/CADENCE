"""Quick check of V11 scaffold contents for sessions w/o EEG preproc."""
import numpy as np
import json
from pathlib import Path

REPO = Path('C:/Users/optilab/desktop/CADENCE')
sessions = ['y04_020626', 'y11_022526', 'y24_022526', 'y_53_04302026']

for sid in sessions:
    npz = REPO / 'results' / 'v11' / sid / 'scaffold_v11_ztimecourses.npz'
    if not npz.exists():
        print(f'{sid}: NO scaffold')
        continue
    data = np.load(npz)
    print(f'\n{sid}:')
    print(f'  Keys: {list(data.keys())}')
    if 'z_v11' in data:
        z = data['z_v11']
        print(f'  z_v11 shape: {z.shape}, std: {np.std(z, axis=0)[:6].round(3)}...')
    sidecar = REPO / 'results' / 'v11' / sid / 'scaffold_v11_results.json'
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        print(f'  digest_xdf_md5: {meta.get("digest_xdf_md5", "missing")}')
        print(f'  modalities: {meta.get("modalities", "missing")}')
