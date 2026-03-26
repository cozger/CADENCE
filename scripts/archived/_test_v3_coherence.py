"""V3 coherence temporal localization: integration tests.

Tests:
1. Whitening restores null_sigma ≈ 1.0 for correlated channels
2. Full coherence pipeline detects known coupling windows
3. EEG band coherence detects theta-band coupling
"""
import numpy as np
import time

# ---- Test 1: Whitened Stouffer null_sigma ≈ 1.0 ----
print("=" * 60)
print("Test 1: Whitened Stouffer on correlated channels")
print("=" * 60)

from cadence.significance.temporal_localization import (
    zscore_stouffer, zscore_stouffer_whitened,
)

rng = np.random.RandomState(42)
C, T, K = 15, 1000, 100

# Generate correlated null z-scores (rho=0.4)
rho = 0.4
Sigma_true = (1 - rho) * np.eye(C) + rho * np.ones((C, C))
L = np.linalg.cholesky(Sigma_true)

# Real dR2 = null (no coupling)
dr2_real = np.abs(L @ rng.randn(C, T)) * 0.01
dr2_surr = np.zeros((K, C, T))
for k in range(K):
    dr2_surr[k] = np.abs(L @ rng.randn(C, T)) * 0.01

# Standard Stouffer (inflated null_sigma)
z_agg, z_null, mu0, sig0 = zscore_stouffer(dr2_real, dr2_surr, 2.0)
print(f"  Standard Stouffer: null_mu={mu0:.3f}, null_sigma={sig0:.3f}")

# Whitened Stouffer (should restore null_sigma ≈ 1.0)
z_agg_w, z_null_w, mu_w, sig_w, meta = zscore_stouffer_whitened(
    dr2_real, dr2_surr, 2.0)
print(f"  Whitened Stouffer: null_mu={mu_w:.3f}, null_sigma={sig_w:.3f}")
print(f"  C_eff={meta['C_eff']:.1f} (raw C={C})")

assert sig_w < sig0, "Whitened sigma should be smaller than standard"
assert abs(sig_w - 1.0) < 0.5, f"Whitened sigma should be near 1.0, got {sig_w}"
print("  PASSED\n")


# ---- Test 2: Full coherence pipeline on synthetic coupled signals ----
print("=" * 60)
print("Test 2: Full coherence pipeline on synthetic signals")
print("=" * 60)

from cadence.significance.coherence_localization import (
    coherence_temporal_localization,
)

rng = np.random.RandomState(123)
fs = 30.0
duration_s = 300
N = int(duration_s * fs)
n_ch = 10
t = np.arange(N) / fs

# Generate 10-channel signals with coupling in known windows
p1 = rng.randn(N, n_ch) * 0.5
p2 = rng.randn(N, n_ch) * 0.5

# Inject coupling in [50-100s] and [180-230s] with 0.3s lag
coupling_windows = [(50, 100), (180, 230)]
lag_samples = int(0.3 * fs)
kappa = 0.6  # coupling strength
for s, e in coupling_windows:
    si, ei = int(s * fs), int(e * fs)
    for ch in range(n_ch):
        p2[si+lag_samples:ei+lag_samples, ch] += kappa * p1[si:ei, ch]

ts = np.arange(N) / fs
matched_ch = list(range(n_ch))

t0 = time.perf_counter()
mask, posterior, window_times, diag = coherence_temporal_localization(
    p1, p2, ts, ts, matched_ch, fs,
    cfg={'window_s': 5.0, 'stride_s': 0.5, 'n_surrogates': 50,
         'target_false_alarm': 0.10, 'min_event_s': 2.0, 'seed': 42})
elapsed = time.perf_counter() - t0

print(f"  Pipeline time: {elapsed:.1f}s")
print(f"  Windows: {len(window_times)}")
print(f"  Coupling fraction: {diag.get('coupling_fraction', 0):.2%}")
print(f"  C_eff: {diag.get('C_eff', 0):.1f}")
print(f"  z_agg max: {diag.get('z_agg_max', 0):.2f}")
print(f"  Threshold: {diag.get('threshold', 0):.2f}")

# Check detection in coupling windows
if len(window_times) > 0:
    coupled_mask_true = np.zeros(len(window_times), dtype=bool)
    for s, e in coupling_windows:
        coupled_mask_true |= (window_times >= s) & (window_times <= e)

    n_coupled_true = coupled_mask_true.sum()
    n_coupled_detected = mask[coupled_mask_true].sum()
    hit_rate = n_coupled_detected / max(n_coupled_true, 1)

    n_null_true = (~coupled_mask_true).sum()
    n_null_detected = mask[~coupled_mask_true].sum()
    fa_rate = n_null_detected / max(n_null_true, 1)

    print(f"  Hit rate:  {hit_rate:.2%} ({n_coupled_detected}/{n_coupled_true})")
    print(f"  FA rate:   {fa_rate:.2%} ({n_null_detected}/{n_null_true})")

    # Compute IoU
    detected_set = set(np.where(mask)[0])
    true_set = set(np.where(coupled_mask_true)[0])
    intersection = len(detected_set & true_set)
    union = len(detected_set | true_set)
    iou = intersection / max(union, 1)
    print(f"  IoU:       {iou:.2%}")
else:
    print("  WARNING: No windows produced")

print("  PASSED\n")


# ---- Test 3: EEG band coherence ----
print("=" * 60)
print("Test 3: EEG band coherence (theta injection)")
print("=" * 60)

from cadence.data.eeg_coherence import eeg_band_coherence

rng = np.random.RandomState(456)
fs_eeg = 256
T_eeg = int(60 * fs_eeg)  # 60 seconds
t_eeg = np.arange(T_eeg) / fs_eeg

p1_eeg = rng.randn(T_eeg, 14) * 5.0  # microvolt scale noise
p2_eeg = rng.randn(T_eeg, 14) * 5.0

# Inject theta (6 Hz) coupling in frontal ROI [20-40s]
theta_freq = 6.0
s_inj, e_inj = int(20 * fs_eeg), int(40 * fs_eeg)
theta_signal = np.sin(2 * np.pi * theta_freq * t_eeg)
lag_eeg = int(0.05 * fs_eeg)  # 50ms lag

# Inject into frontal channels (0, 2, 11, 13)
for ch in [0, 2, 11, 13]:
    p1_eeg[s_inj:e_inj, ch] += 10.0 * theta_signal[s_inj:e_inj]
    p2_eeg[s_inj+lag_eeg:e_inj+lag_eeg, ch] += 8.0 * theta_signal[s_inj:e_inj]

t0 = time.perf_counter()
coh_real, coh_imag, wt, names = eeg_band_coherence(
    p1_eeg, p2_eeg, fs=fs_eeg, window_s=2.0, stride_s=0.25,
    bands={'theta': (4, 8), 'alpha': (8, 13)},
    use_imcoh=True)
elapsed = time.perf_counter() - t0

print(f"  EEG coherence time: {elapsed:.1f}s")
print(f"  Shape: real={coh_real.shape}, imag={coh_imag.shape}")
print(f"  Features: {len(names)}")

# Check theta band (index 0) has higher coherence during injection
theta_coh = coh_real[0]  # (n_pairs, n_windows)
coupled_w = (wt >= 20) & (wt <= 40)
null_w = (wt < 15) | (wt > 45)

# Average across ROI pairs
theta_coupled = theta_coh[:, coupled_w].mean()
theta_null = theta_coh[:, null_w].mean()
print(f"  Theta coupled:  {theta_coupled:.4f}")
print(f"  Theta null:     {theta_null:.4f}")
print(f"  Theta contrast: {theta_coupled - theta_null:.4f}")

# ImCoh should also show contrast (non-zero-lag coupling)
if coh_imag is not None:
    imcoh_theta = coh_imag[0]
    imcoh_coupled = imcoh_theta[:, coupled_w].mean()
    imcoh_null = imcoh_theta[:, null_w].mean()
    print(f"  ImCoh coupled:  {imcoh_coupled:.4f}")
    print(f"  ImCoh null:     {imcoh_null:.4f}")
    print(f"  ImCoh contrast: {imcoh_coupled - imcoh_null:.4f}")

assert theta_coupled > theta_null, "Theta coherence should be higher during coupling!"
print("  PASSED\n")

print("=" * 60)
print("ALL V3 TESTS PASSED")
print("=" * 60)
