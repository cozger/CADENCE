"""Stage 0: Inventory + freshness check for MVP autonomous run."""
import json
import csv
from pathlib import Path
from cadence.ingest.quality import list_canonical_sessions
from cadence.io.resources import log_resources

REPO = Path('C:/Users/optilab/desktop/CADENCE')
log_resources(prefix='[Stage 0] start: ')

canonical = list_canonical_sessions()

state = []
for sid in canonical:
    digest_path = REPO / 'data' / 'digest' / 'v1' / f'{sid}.json'
    has_digest = digest_path.exists()
    if has_digest:
        digest_md5 = json.loads(digest_path.read_text()).get('xdf_md5')
        protocol = json.loads(digest_path.read_text()).get('protocol', '')
    else:
        digest_md5, protocol = None, ''

    pre = {m: (REPO / 'data' / 'preproc' / m / 'v1' / f'{sid}.npz').exists()
           for m in ('eeg', 'face', 'ecg', 'pose')}
    has_v11 = (REPO / 'results' / 'v11' / sid / 'scaffold_v11_ztimecourses.npz').exists()
    has_matlab = (
        (REPO / 'data' / 'matlab' / f'{sid}_p1_clean.mat').exists() and
        (REPO / 'data' / 'matlab' / f'{sid}_p2_clean.mat').exists()
    )
    state.append({
        'sid': sid, 'protocol': protocol, 'has_digest': has_digest,
        'digest_md5': digest_md5,
        **{f'preproc_{m}': pre[m] for m in pre},
        'has_v11_scaffold': has_v11, 'has_matlab_clean': has_matlab,
    })

print(f'{"sid":25s} | {"prot":10s} | dig | mat | eeg | fac | ecg | pos | v11 |')
for s in state:
    print(f'{s["sid"]:25s} | {s["protocol"]:10s} | '
          f'{"Y" if s["has_digest"] else "n":3s} | '
          f'{"Y" if s["has_matlab_clean"] else "n":3s} | '
          f'{"Y" if s["preproc_eeg"] else "n":3s} | '
          f'{"Y" if s["preproc_face"] else "n":3s} | '
          f'{"Y" if s["preproc_ecg"] else "n":3s} | '
          f'{"Y" if s["preproc_pose"] else "n":3s} | '
          f'{"Y" if s["has_v11_scaffold"] else "n":3s} |')

# Save inventory
out_path = REPO / 'results' / 'mvp' / 'pipeline_inventory.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(state[0].keys()))
    w.writeheader()
    w.writerows(state)
print(f'\nInventory saved: {out_path}')
print(f'  Canonical: {len(state)}, with_digest: {sum(s["has_digest"] for s in state)}, '
      f'with_matlab: {sum(s["has_matlab_clean"] for s in state)}')
print(f'  preproc: eeg={sum(s["preproc_eeg"] for s in state)}, '
      f'face={sum(s["preproc_face"] for s in state)}, '
      f'ecg={sum(s["preproc_ecg"] for s in state)}, '
      f'pose={sum(s["preproc_pose"] for s in state)}')
print(f'  v11_scaffold: {sum(s["has_v11_scaffold"] for s in state)}')
