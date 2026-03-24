"""Deep diagnostic: trace exact difference between scipy and torch Welch."""
import numpy as np
import torch
from scipy.signal import csd, welch

rng = np.random.RandomState(123)
fs = 30.0
win_samp = int(5.0 * fs)  # 150 samples
nperseg = max(16, win_samp // 2)  # 75
noverlap = nperseg // 2  # 37  (overlap, what scipy expects)
sub_step = nperseg - noverlap  # 38  (step, what torch.unfold expects)

x1 = rng.randn(win_samp)
x2 = rng.randn(win_samp)

# --- scipy (noverlap = overlap, NOT step) ---
freqs_sp, sxy_sp = csd(x1, x2, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
_, sxx_sp = welch(x1, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
_, syy_sp = welch(x2, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
msc_sp = np.abs(sxy_sp)**2 / np.maximum(sxx_sp * syy_sp, 1e-20)
print(f"scipy: n_freqs={len(freqs_sp)}, msc_mean(skip DC)={msc_sp[1:].mean():.6f}")

# --- torch (sub_step = step) ---
device = torch.device('cpu')
x1_t = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)  # (1, win_samp)
x2_t = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)

x1_sub = x1_t.unfold(-1, nperseg, sub_step)  # (1, n_sub, nperseg)
x2_sub = x2_t.unfold(-1, nperseg, sub_step)
n_sub = x1_sub.shape[1]
print(f"torch: n_sub={n_sub}, nperseg={nperseg}, sub_step={sub_step}")

# Detrend + Hann
x1_sub = x1_sub - x1_sub.mean(dim=-1, keepdim=True)
x2_sub = x2_sub - x2_sub.mean(dim=-1, keepdim=True)
hann = torch.hann_window(nperseg)
x1_sub = x1_sub * hann
x2_sub = x2_sub * hann

fft1 = torch.fft.rfft(x1_sub, dim=-1)
fft2 = torch.fft.rfft(x2_sub, dim=-1)

Sxy = (fft1.conj() * fft2).mean(dim=-2)
Sxx = fft1.abs().square().mean(dim=-2)
Syy = fft2.abs().square().mean(dim=-2)

msc_torch = (Sxy.abs().square() / torch.clamp(Sxx * Syy, min=1e-20)).squeeze()
print(f"torch: n_freqs={len(msc_torch)}, msc_mean(skip DC)={msc_torch[1:].mean():.6f}")

# Segment count verification
scipy_step = nperseg - noverlap  # 38
scipy_starts = list(range(0, win_samp - nperseg + 1, scipy_step))
torch_starts = list(range(0, win_samp - nperseg + 1, sub_step))
print(f"\nScipy starts (step={scipy_step}): {scipy_starts} (count={len(scipy_starts)})")
print(f"Torch starts (step={sub_step}): {torch_starts} (count={len(torch_starts)})")
assert scipy_starts == torch_starts, "Sub-segment starts should match!"

# Per-frequency comparison
diff = np.abs(msc_sp[1:] - msc_torch[1:].numpy())
print(f"\nPer-freq MSC diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
print(f"Match within 1e-4: {(diff < 1e-4).all()}")
print(f"Match within 1e-3: {(diff < 1e-3).all()}")
print(f"Match within 1e-2: {(diff < 1e-2).all()}")
