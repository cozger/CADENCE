"""Diagnose the null bias in EEG TL surrogates.

At kappa=0 (pseudo-dyad), the AUC should be 0.50. But we see 0.40.
This means surrogates produce higher cross-products than real data.
Why?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import gaussian_filter1d

from cadence.significance.fast_cycles import (
    _fft_bandpass, extract_cycle_features, EEG_BANDS,
)
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
import torch

# Load pseudo-dyad
config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])

def load_seg(session_name, person, t0, t1):
    cp = [p for n, p in cached_sessions if session_name in n]
    cached = load_session_from_cache(cp[0], config)
    eeg = cached[f'{person}_eeg']
    ts = cached[f'{person}_eeg_ts']
    m = (ts >= t0) & (ts <= t1)
    seg = eeg[m, :14].astype(np.float64)
    seg -= seg.mean(axis=1, keepdims=True)
    for ch in range(14):
        std = seg[:, ch].std()
        if std > 1e-8:
            seg[:, ch] = (seg[:, ch] - seg[:, ch].mean()) / std
    return seg, len(ts[m]) / (ts[m][-1] - ts[m][0])

p1, fs1 = load_seg('y_06', 'p1', 500, 800)
p2, fs2 = load_seg('y_17', 'p2', 500, 800)
min_len = min(len(p1), len(p2))
p1 = p1[:min_len]
p2 = p2[:min_len]
fs = fs1

# Extract theta volt_amp timecourses
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T, C = p1.shape
rate = 2.0
dur = T / fs
t_grid = np.arange(0, dur, 1.0 / rate)
N = len(t_grid)

both = np.vstack([p1.T, p2.T])
both_t = torch.as_tensor(both, dtype=torch.float32, device=device)
both_filt = _fft_bandpass(both_t, fs, (4, 8), device).cpu().numpy()
p1_filt = both_filt[:C].T
p2_filt = both_filt[C:].T

p1_va = np.zeros((C, N), dtype=np.float32)
p2_va = np.zeros((C, N), dtype=np.float32)
valid = []

for ch in range(C):
    cyc1 = extract_cycle_features(p1[:, ch], p1_filt[:, ch], fs, (4, 8))
    cyc2 = extract_cycle_features(p2[:, ch], p2_filt[:, ch], fs, (4, 8))
    if cyc1 is None or cyc2 is None or cyc1['n_cycles'] < 10 or cyc2['n_cycles'] < 10:
        continue
    valid.append(ch)
    t1 = cyc1['peak_sample'] / fs
    t2 = cyc2['peak_sample'] / fs
    p1_va[ch] = np.interp(t_grid, t1, cyc1['volt_amp'])
    p2_va[ch] = np.interp(t_grid, t2, cyc2['volt_amp'])

p1_va = p1_va[valid]
p2_va = p2_va[valid]
C_valid = len(valid)
print(f"Valid channels: {C_valid}, N={N} timepoints at {rate} Hz")

# Z-score per channel
for c in range(C_valid):
    for arr in [p1_va, p2_va]:
        mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
        arr[c] = (arr[c] - mu) / sd

# Real cross-product (no smoothing)
cp_real = (p1_va * p2_va).mean(axis=0)
print(f"\nReal cross-product: mean={cp_real.mean():.4f}, std={cp_real.std():.4f}")

# Surrogates
rng = np.random.default_rng(42)
K = 200
surr_means = np.zeros(K)
surr_stds = np.zeros(K)
surr_all = np.zeros((K, N))

for k in range(K):
    shift = rng.integers(int(0.1 * N), int(0.9 * N))
    p1_shifted = np.roll(p1_va, shift, axis=1)
    cp_surr = (p1_shifted * p2_va).mean(axis=0)
    surr_all[k] = cp_surr
    surr_means[k] = cp_surr.mean()
    surr_stds[k] = cp_surr.std()

print(f"\nSurrogate cross-product:")
print(f"  mean of means: {surr_means.mean():.4f}")
print(f"  std of means:  {surr_means.std():.4f}")
print(f"  mean of stds:  {surr_stds.mean():.4f}")

print(f"\nReal mean vs surrogate mean: {cp_real.mean():.4f} vs {surr_means.mean():.4f}")
print(f"  Difference: {cp_real.mean() - surr_means.mean():.4f}")

# The z-score distribution
surr_mean_per_t = surr_all.mean(axis=0)
surr_std_per_t = np.maximum(surr_all.std(axis=0), 1e-10)
z_real = (cp_real - surr_mean_per_t) / surr_std_per_t

print(f"\nZ-score distribution (no smoothing):")
print(f"  mean={z_real.mean():.3f}, std={z_real.std():.3f}")
print(f"  Should be mean~0, std~1 under null")

# Now with smoothing
for smooth_samp in [3, 5, 10]:
    sigma = smooth_samp / 2.0  # Gaussian sigma in samples
    cp_sm = gaussian_filter1d(cp_real, sigma)
    surr_sm = np.array([gaussian_filter1d(surr_all[k], sigma) for k in range(K)])
    sm_mean = surr_sm.mean(axis=0)
    sm_std = np.maximum(surr_sm.std(axis=0), 1e-10)
    z_sm = (cp_sm - sm_mean) / sm_std
    print(f"\nSmoothed ({smooth_samp} samples, sigma={sigma:.1f}):")
    print(f"  z mean={z_sm.mean():.3f}, std={z_sm.std():.3f}")
    print(f"  cp_real_sm mean={cp_sm.mean():.4f}, surr_sm mean={sm_mean.mean():.4f}")

# Check: is the bias from the cross-product or from the surrogate?
# The circular shift preserves autocorrelation of P1. But does it change
# the cross-correlation with P2?
print(f"\n--- Autocorrelation check ---")
# P1 autocorrelation at lag 1
for c in range(min(3, C_valid)):
    ac_real = np.corrcoef(p1_va[c, :-1], p1_va[c, 1:])[0, 1]
    shifts = [rng.integers(int(0.1*N), int(0.9*N)) for _ in range(50)]
    ac_surrs = [np.corrcoef(np.roll(p1_va[c], s)[:-1], np.roll(p1_va[c], s)[1:])[0, 1]
                for s in shifts]
    print(f"  Ch {valid[c]}: P1 autocorr(1)={ac_real:.3f}, "
          f"surr mean={np.mean(ac_surrs):.3f} (should be same)")

# Check: smoothed cross-product variance
print(f"\n--- Variance of smoothed cross-product ---")
for sigma in [0, 1, 2, 5]:
    if sigma == 0:
        var_real = cp_real.var()
        var_surr = np.mean([surr_all[k].var() for k in range(K)])
    else:
        var_real = gaussian_filter1d(cp_real, sigma).var()
        var_surr = np.mean([gaussian_filter1d(surr_all[k], sigma).var() for k in range(K)])
    print(f"  sigma={sigma}: real var={var_real:.6f}, surr var={var_surr:.6f}, "
          f"ratio={var_real/var_surr:.3f}")
