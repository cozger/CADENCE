"""Quick smoke test of multi-lag bank."""
import sys; sys.path.insert(0, '.')
from cadence.significance.coherence_localization import xcorr_temporal_localization
import numpy as np
p1 = np.random.randn(2560, 5).astype(np.float32)
p2 = np.random.randn(2560, 5).astype(np.float32)
mask, z, lag, d = xcorr_temporal_localization(
    p1, p2, 256.0, max_lag_s=0.05, lag_step_s=0.005,
    smooth_s=0.5, n_surrogates=5, min_event_s=1.0)
print(f'OK: mask={mask.shape}, z={z.shape}, lag={lag.shape}')
print(f'z_mean={z.mean():.3f}, z_max={z.max():.3f}')
