"""Unit test: cycle-PLV phase reconstruction + integration test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from cadence.significance.fast_cycles import (
    extract_cycle_features, _reconstruct_cycle_phase, _fft_bandpass,
    analyze_interbrain_cycles
)

# ── Test 1: Phase reconstruction from known landmarks ──────────────────
print("Test 1: Phase reconstruction")
troughs = np.array([0, 100, 200, 300])
peaks = np.array([40, 140, 240])
t_grid = np.linspace(0, 3.0, 301)  # 0 to 3.0s at 100 Hz
phase = _reconstruct_cycle_phase(troughs, peaks, 3, 100.0, t_grid)

# At t=0.0 (trough[0]): phase=0
# At t=0.4 (peak[0]): phase=π
# At t=1.0 (trough[1]): phase=2π
# At t=1.4 (peak[1]): phase=3π
# At t=2.0 (trough[2]): phase=4π
assert np.allclose(phase[0], 0.0, atol=0.01), f"t=0: {phase[0]}"
assert np.allclose(phase[40], np.pi, atol=0.05), f"t=0.4: {phase[40]} vs {np.pi}"
assert np.allclose(phase[100], 2*np.pi, atol=0.05), f"t=1.0: {phase[100]} vs {2*np.pi}"
assert np.allclose(phase[140], 3*np.pi, atol=0.05), f"t=1.4: {phase[140]} vs {3*np.pi}"
assert np.allclose(phase[200], 4*np.pi, atol=0.05), f"t=2.0: {phase[200]} vs {4*np.pi}"
# Phase is monotonically increasing
assert np.all(np.diff(phase) >= 0), "Phase not monotonic"
print("  PASSED: landmarks correct, monotonic\n")

# ── Test 2: extract_cycle_features returns trough_sample ────────────────
print("Test 2: extract_cycle_features returns troughs")
fs = 256.0
t = np.arange(0, 5.0, 1.0/fs)
sig = np.sin(2*np.pi*6*t)  # 6 Hz sine
cyc = extract_cycle_features(sig, sig, fs, (4.0, 8.0))
assert cyc is not None
assert 'trough_sample' in cyc, "Missing trough_sample"
assert 'peak_sample' in cyc
n = cyc['n_cycles']
print(f"  {n} cycles detected, {len(cyc['trough_sample'])} troughs, {len(cyc['peak_sample'])} peaks")
assert len(cyc['trough_sample']) == n + 1, f"Expected {n+1} troughs, got {len(cyc['trough_sample'])}"
assert len(cyc['peak_sample']) == n, f"Expected {n} peaks, got {len(cyc['peak_sample'])}"
print("  PASSED\n")

# ── Test 3: Full pipeline with synthetic coupled signals ────────────────
print("Test 3: Full pipeline (coupled vs null)")
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rng = np.random.default_rng(42)
T = int(300 * fs)  # 300s
# P1: 6 Hz oscillation + noise, 14 channels
p1 = np.zeros((T, 14), dtype=np.float32)
for ch in range(14):
    phase_offset = rng.uniform(0, 2*np.pi)
    p1[:, ch] = np.sin(2*np.pi*6*np.arange(T)/fs + phase_offset) + 0.5*rng.standard_normal(T).astype(np.float32)

# P2_coupled: same phase as P1 + noise (strong coupling)
p2_coupled = np.zeros_like(p1)
for ch in range(14):
    p2_coupled[:, ch] = 0.5*p1[:, ch] + 0.5*rng.standard_normal(T).astype(np.float32)

# P2_null: independent
p2_null = np.zeros_like(p1)
for ch in range(14):
    phase_offset = rng.uniform(0, 2*np.pi)
    p2_null[:, ch] = np.sin(2*np.pi*6*np.arange(T)/fs + phase_offset) + 0.5*rng.standard_normal(T).astype(np.float32)

# Z-normalize
for s in [p1, p2_coupled, p2_null]:
    s -= s.mean(axis=1, keepdims=True)
    for ch in range(14):
        sd = max(s[:, ch].std(), 1e-8)
        s[:, ch] /= sd

r_coupled = analyze_interbrain_cycles(p1, p2_coupled, fs, band=(4.0, 8.0),
                                       n_surrogates=100, seed=42, device=device)
r_null = analyze_interbrain_cycles(p1, p2_null, fs, band=(4.0, 8.0),
                                    n_surrogates=100, seed=42, device=device)

print(f"  Coupled: volt_amp z={r_coupled['volt_amp']['pooled_z']:+.2f}, "
      f"cycle_plv z={r_coupled['cycle_plv']['pooled_z']:+.2f}, "
      f"plv={r_coupled['cycle_plv']['mean_plv']:.3f}")
print(f"  Null:    volt_amp z={r_null['volt_amp']['pooled_z']:+.2f}, "
      f"cycle_plv z={r_null['cycle_plv']['pooled_z']:+.2f}, "
      f"plv={r_null['cycle_plv']['mean_plv']:.3f}")

# Coupled should have higher cycle_plv z than null
assert r_coupled['cycle_plv']['pooled_z'] > r_null['cycle_plv']['pooled_z'], \
    "Coupled cycle_plv should be higher than null"
assert r_coupled['cycle_plv']['pooled_z'] > 2.0, \
    f"Coupled cycle_plv z={r_coupled['cycle_plv']['pooled_z']:.2f} should be > 2.0"
print("  PASSED: coupled > null, z > 2.0\n")

print("All tests passed!")
