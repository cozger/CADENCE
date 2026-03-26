"""Diagnostic: narrowband wavelet coherence — does single-frequency show coupling?"""
import numpy as np
import torch

rng = np.random.RandomState(123)
fs = 30.0; duration = 1800; N = int(duration * fs)
n_ch = 15; n_coupled = 4; kappa = 0.4; lag = int(2.0 * fs)

p1 = rng.randn(N, n_ch).astype(np.float32)
p2 = rng.randn(N, n_ch).astype(np.float32)

from cadence.synthetic import generate_coupling_gate
gate = generate_coupling_gate(N, fs, {'duty_cycle': 0.25, 'event_range_s': (3, 15), 'ramp_s': 1.0}, seed=42)

for ch in range(n_coupled):
    p1_lagged = np.roll(p1[:, ch], lag); p1_lagged[:lag] = 0
    alpha = kappa * gate
    p2[:, ch] = alpha * p1_lagged + np.sqrt(np.maximum(1-alpha**2, 0)) * p2[:, ch]
for ch in range(n_ch):
    p2[:, ch] = (p2[:, ch] - p2[:, ch].mean()) / (p2[:, ch].std() + 1e-8)

ts = np.arange(N) / fs
gate_mask = gate > 0.5
print(f"Session: {duration}s, kappa={kappa}, coupled={n_coupled}/{n_ch}, duty={gate_mask.mean():.1%}")

from cadence.significance.coherence_localization import _morlet_cwt, _gaussian_smooth_1d

device = torch.device('cuda')
x1 = torch.as_tensor(p1.T, dtype=torch.float32, device=device)
x2 = torch.as_tensor(p2.T, dtype=torch.float32, device=device)

# Test individual frequencies
test_freqs = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]

print(f"\n=== Per-frequency, channel 0 (COUPLED) wavelet coherence ===")
print(f"{'Freq':>5}  {'Coupled':>7}  {'Null':>7}  {'Contrast':>8}  {'z':>6}  {'Null_std':>8}")
for freq in test_freqs:
    W1 = _morlet_cwt(x1[0:1], fs, [freq], n_cycles=5, device=device)  # (1, T, 1)
    W2 = _morlet_cwt(x2[0:1], fs, [freq], n_cycles=5, device=device)
    sigma_samp = 5.0 / (2 * np.pi * freq) * fs
    w1 = W1[0, :, 0]; w2 = W2[0, :, 0]
    xy = _gaussian_smooth_1d((w1*w2.conj()).unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    xx = _gaussian_smooth_1d(w1.abs().square().unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    yy = _gaussian_smooth_1d(w2.abs().square().unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    msc = (xy.abs().square() / torch.clamp(xx*yy, min=1e-20)).cpu().numpy()
    T_coh = min(len(msc), len(gate_mask))
    c = msc[:T_coh][gate_mask[:T_coh]].mean()
    n = msc[:T_coh][~gate_mask[:T_coh]].mean()
    s = msc[:T_coh][~gate_mask[:T_coh]].std()
    z = (c-n)/max(s, 1e-8)
    print(f"{freq:>5.1f}  {c:>7.4f}  {n:>7.4f}  {c-n:>8.4f}  {z:>6.2f}  {s:>8.4f}")

print(f"\n=== Same for channel 5 (NULL) ===")
print(f"{'Freq':>5}  {'Coupled':>7}  {'Null':>7}  {'Contrast':>8}  {'z':>6}")
for freq in test_freqs:
    W1 = _morlet_cwt(x1[5:6], fs, [freq], n_cycles=5, device=device)
    W2 = _morlet_cwt(x2[5:6], fs, [freq], n_cycles=5, device=device)
    sigma_samp = 5.0 / (2 * np.pi * freq) * fs
    w1 = W1[0, :, 0]; w2 = W2[0, :, 0]
    xy = _gaussian_smooth_1d((w1*w2.conj()).unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    xx = _gaussian_smooth_1d(w1.abs().square().unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    yy = _gaussian_smooth_1d(w2.abs().square().unsqueeze(0).unsqueeze(0), sigma_samp, dim=-1).squeeze()
    msc = (xy.abs().square() / torch.clamp(xx*yy, min=1e-20)).cpu().numpy()
    T_coh = min(len(msc), len(gate_mask))
    c = msc[:T_coh][gate_mask[:T_coh]].mean()
    n = msc[:T_coh][~gate_mask[:T_coh]].mean()
    s = msc[:T_coh][~gate_mask[:T_coh]].std()
    z = (c-n)/max(s, 1e-8)
    print(f"{freq:>5.1f}  {c:>7.4f}  {n:>7.4f}  {c-n:>8.4f}  {z:>6.2f}")
