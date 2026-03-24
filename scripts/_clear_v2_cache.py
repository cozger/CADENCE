"""Clear V2 features from session cache to force recomputation."""
import sys
import numpy as np

path = sys.argv[1]  # cache path prefix (without .npz)
data = dict(np.load(path + '.npz', allow_pickle=True))
v2_keys = [k for k in data.keys() if 'wavelet' in k or 'blendshapes_v2' in k
           or 'interbrain' in k or 'ecg_features_v2' in k]
print(f'Removing {len(v2_keys)} V2 keys')
for k in v2_keys:
    del data[k]
np.savez_compressed(path + '.npz', **data)
print('Done — V2 features will recompute on next load')
