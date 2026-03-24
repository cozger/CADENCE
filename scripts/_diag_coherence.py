"""Diagnostic: compare old scipy vs new torch coherence detection."""
import numpy as np
import time
from cadence.significance.coherence_localization import (
    coherence_temporal_localization, _welch_msc_batched,
)

# Same test signal as _test_v3_coherence.py Test 2
rng = np.random.RandomState(123)
fs = 30.0; N = int(300 * fs); n_ch = 10
p1 = rng.randn(N, n_ch) * 0.5
p2 = rng.randn(N, n_ch) * 0.5
lag = int(0.3 * fs); kappa = 0.6
for s, e in [(50, 100), (180, 230)]:
    si, ei = int(s * fs), int(e * fs)
    for ch in range(n_ch):
        p2[si+lag:ei+lag, ch] += kappa * p1[si:ei, ch]

ts = np.arange(N) / fs
cfg = {'window_s': 5.0, 'stride_s': 0.5, 'n_surrogates': 50,
       'target_false_alarm': 0.10, 'min_event_s': 2.0, 'seed': 42}

# --- New torch path ---
mask, post, wt, diag = coherence_temporal_localization(
    p1, p2, ts, ts, list(range(n_ch)), fs, cfg=cfg)
print("=== GPU-batched torch ===")
print(f"  threshold={diag['threshold']:.4f}")
print(f"  null_sigma={diag['null_sigma']:.4f}")
print(f"  z_agg_max={diag['z_agg_max']:.4f}")
print(f"  z_agg_p95={diag['z_agg_p95']:.4f}")
print(f"  coupling_frac={diag['coupling_fraction']:.4f}")

# --- Compare: single-channel torch vs scipy ---
import torch
from scipy.signal import csd, welch

x1 = p1[:, 0]
x2 = p2[:, 0]
win_samp = int(5.0 * fs)
nperseg = max(16, win_samp // 2)
starts = np.arange(0, N - win_samp + 1, int(0.5 * fs))

# Scipy coherence for first 5 windows
print("\n=== Per-window comparison (first 5 windows) ===")
print(f"{'window':>6}  {'scipy':>10}  {'torch':>10}  {'diff':>10}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x1_t = torch.as_tensor(x1, dtype=torch.float32, device=device)
x2_t = torch.as_tensor(x2, dtype=torch.float32, device=device)

for i in range(min(5, len(starts))):
    s = starts[i]
    seg1 = x1[s:s+win_samp]
    seg2 = x2[s:s+win_samp]

    # scipy
    freqs_sp, sxy = csd(seg1, seg2, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    _, sxx = welch(seg1, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    _, syy = welch(seg2, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    denom = np.maximum(sxx * syy, 1e-20)
    msc_scipy = np.abs(sxy)**2 / denom
    coh_scipy = float(np.mean(msc_scipy[1:]))

    # torch
    seg1_t = x1_t[s:s+win_samp].unsqueeze(0)
    seg2_t = x2_t[s:s+win_samp].unsqueeze(0)
    coh_torch = float(_welch_msc_batched(seg1_t, seg2_t, fs, nperseg, None, device).cpu().item())

    print(f"{i:>6}  {coh_scipy:>10.6f}  {coh_torch:>10.6f}  {abs(coh_scipy-coh_torch):>10.6f}")
