"""Test Kim filter on synthetic episodic coupling."""
import numpy as np
import time

from cadence.significance.kim_filter import KimSwitchingRegression

rng = np.random.RandomState(42)
fs = 30.0
duration = 300  # 5 min for quick test
N = int(duration * fs)
n_ch = 15
n_coupled = 4
kappa = 0.4
lag_samples = int(2.0 * fs)  # 2s lag

# Generate signals
p1 = rng.randn(N, n_ch).astype(np.float32)
p2 = rng.randn(N, n_ch).astype(np.float32)

# Add AR structure to P2 (makes it more realistic)
for ch in range(n_ch):
    for t in range(3, N):
        p2[t, ch] += 0.3 * p2[t-1, ch] - 0.1 * p2[t-2, ch]

# Generate episodic coupling gate (25% duty)
from cadence.synthetic import generate_coupling_gate
gate = generate_coupling_gate(N, fs, {
    'duty_cycle': 0.25,
    'event_range_s': (3, 15),
    'ramp_s': 1.0,
}, seed=42)

# Inject sparse coupling
for ch in range(n_coupled):
    p1_lagged = np.roll(p1[:, ch], lag_samples)
    p1_lagged[:lag_samples] = 0
    alpha = kappa * gate
    p2[:, ch] += alpha * p1_lagged

# Normalize
for ch in range(n_ch):
    p2[:, ch] = (p2[:, ch] - p2[:, ch].mean()) / (p2[:, ch].std() + 1e-8)
    p1[:, ch] = (p1[:, ch] - p1[:, ch].mean()) / (p1[:, ch].std() + 1e-8)

gate_mask = gate > 0.5
duty = gate_mask.mean()
print(f"Session: {duration}s @ {fs} Hz, N={N}")
print(f"Coupling: kappa={kappa}, lag={lag_samples/fs:.1f}s, "
      f"coupled={n_coupled}/{n_ch}, duty={duty:.1%}")

# Build basis-convolved source (simple: just lagged versions)
# In real pipeline, this comes from raised cosine basis
n_basis = 6
max_lag_s = 5.0
lags = np.linspace(0, max_lag_s, n_basis)
x_basis = np.zeros((N, n_ch, n_basis), dtype=np.float32)
for bi, lag_s in enumerate(lags):
    lag_samp = int(lag_s * fs)
    if lag_samp > 0:
        x_basis[lag_samp:, :, bi] = p1[:-lag_samp, :]
    else:
        x_basis[:, :, bi] = p1

# Transpose to (C, T, n_basis)
x_basis_mc = np.transpose(x_basis, (1, 0, 2))  # (C, T, n_basis)
y_mc = p2.T  # (C, T)

# Sweep Q_coeff to find sensitivity
for Q in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    print(f"\n{'='*60}")
    print(f"Q_coeff = {Q}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    kim = KimSwitchingRegression(
        ar_order=3, n_basis=n_basis,
        p_stay_coupled=0.95,
        p_stay_uncoupled=0.95,
        Q_coeff=Q,
        em_iterations=3)

    _, per_channel, params_list = kim.fit_multichannel(
        y_mc, x_basis_mc)
    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Per-channel stats (just coupled vs null summary)
    coupled_fracs = [(per_channel[c] > 0.5).mean() for c in range(n_coupled)]
    null_fracs = [(per_channel[c] > 0.5).mean() for c in range(n_coupled, n_ch)]
    coupled_ratios = [params_list[c]['sigma2_ratio'] for c in range(n_coupled)]
    null_ratios = [params_list[c]['sigma2_ratio'] for c in range(n_coupled, n_ch)]

    print(f"  Coupled ch mean frac: {np.mean(coupled_fracs):.1%} "
          f"(range {np.min(coupled_fracs):.1%}-{np.max(coupled_fracs):.1%})")
    print(f"  Null ch mean frac:    {np.mean(null_fracs):.1%} "
          f"(range {np.min(null_fracs):.1%}-{np.max(null_fracs):.1%})")
    print(f"  Coupled sig2_ratio:   {np.mean(coupled_ratios):.3f}")
    print(f"  Null sig2_ratio:      {np.mean(null_ratios):.3f}")

    # Hit/FA for best coupled channel
    best_ch = np.argmax(coupled_fracs)
    post_best = per_channel[best_ch] > 0.5
    hits = (gate_mask & post_best).sum() / max(gate_mask.sum(), 1)
    fas = (~gate_mask & post_best).sum() / max((~gate_mask).sum(), 1)
    print(f"  Best coupled ch{best_ch}: hit={hits:.1%} fa={fas:.1%} "
          f"frac={coupled_fracs[best_ch]:.1%}")

# Use best Q for detailed output
Q = 1e-2  # pick one based on sweep

# Rerun with chosen Q for detailed output
print(f"\n{'='*60}")
print(f"Detailed results with Q_coeff={Q}")
print(f"{'='*60}")
kim = KimSwitchingRegression(ar_order=3, n_basis=n_basis,
    p_stay_coupled=0.95, p_stay_uncoupled=0.95, Q_coeff=Q, em_iterations=3)
posterior, per_channel, params_list = kim.fit_multichannel(y_mc, x_basis_mc)

# No pooling — just report per-channel
post_on = per_channel[0] > 0.5  # use channel 0 for IoU
coupling_frac = post_on.mean()
hits = (gate_mask & post_on).sum()
fas = (~gate_mask & post_on).sum()
hit_rate = hits / max(gate_mask.sum(), 1)
fa_rate = fas / max((~gate_mask).sum(), 1)

# IoU
intersection = (gate_mask & post_on).sum()
union = (gate_mask | post_on).sum()
iou = intersection / max(union, 1)

print(f"\n=== Results ===")
print(f"Coupling fraction: {coupling_frac:.1%} (true: {duty:.1%})")
print(f"Hit rate:  {hit_rate:.1%}")
print(f"FA rate:   {fa_rate:.1%}")
print(f"IoU:       {iou:.1%}")

# Per-channel posteriors
print(f"\nPer-channel coupling fractions:")
for ch in range(n_ch):
    cf = (per_channel[ch] > 0.5).mean()
    label = "COUPLED" if ch < n_coupled else "null"
    sig2_ratio = params_list[ch]['sigma2_ratio']
    print(f"  ch{ch:>2}: frac={cf:.1%}  sig2_ratio={sig2_ratio:.3f}  {label}")
