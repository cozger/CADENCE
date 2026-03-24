"""Parameter sweep for Kim filter on synthetic episodic coupling."""
import numpy as np
import time
import itertools
from joblib import Parallel, delayed

from cadence.significance.kim_filter import kim_filter_batched

# --- Generate test data ---
rng = np.random.RandomState(42)
fs = 30.0; duration = 1800; N = int(duration * fs)
n_ch = 15; n_coupled = 4; kappa = 0.6
lag_samples = int(2.0 * fs)

p1 = rng.randn(N, n_ch).astype(np.float32)
p2 = rng.randn(N, n_ch).astype(np.float32)

# AR structure
for ch in range(n_ch):
    for t in range(3, N):
        p2[t, ch] += 0.3 * p2[t-1, ch] - 0.1 * p2[t-2, ch]

# Episodic gate
from cadence.synthetic import generate_coupling_gate
gate = generate_coupling_gate(N, fs, {
    'duty_cycle': 0.05, 'event_range_s': (3, 15), 'ramp_s': 1.0
}, seed=42)

for ch in range(n_coupled):
    p1_lag = np.roll(p1[:, ch], lag_samples); p1_lag[:lag_samples] = 0
    p2[:, ch] += kappa * gate * p1_lag

for ch in range(n_ch):
    p2[:, ch] = (p2[:, ch] - p2[:, ch].mean()) / (p2[:, ch].std() + 1e-8)
    p1[:, ch] = (p1[:, ch] - p1[:, ch].mean()) / (p1[:, ch].std() + 1e-8)

gate_mask = gate > 0.5
duty = gate_mask.mean()

# Build basis
n_basis = 6; max_lag_s = 5.0
lags = np.linspace(0, max_lag_s, n_basis)
x_basis = np.zeros((N, n_ch, n_basis), dtype=np.float32)
for bi, lag_s in enumerate(lags):
    lag_samp = int(lag_s * fs)
    if lag_samp > 0:
        x_basis[lag_samp:, :, bi] = p1[:-lag_samp, :]
    else:
        x_basis[:, :, bi] = p1
x_basis_mc = np.transpose(x_basis, (1, 0, 2))
y_mc = p2.T

print(f"Data: {duration}s @ {fs}Hz, {n_ch}ch ({n_coupled} coupled), "
      f"kappa={kappa}, duty={duty:.1%}")
print(f"Sweeping parameters...\n")


def run_one(Q, p_stay, em_iter):
    """Run Kim filter with one parameter set, return metrics."""
    post, params = kim_filter_batched(
        y_mc, x_basis_mc, ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em_iter)

    # Per-channel metrics
    coupled_fracs = [(post[c] > 0.5).mean() for c in range(n_coupled)]
    null_fracs = [(post[c] > 0.5).mean() for c in range(n_coupled, n_ch)]

    # Best coupled channel hit/FA
    best_c = max(range(n_coupled), key=lambda c: (post[c] > 0.5).mean())
    post_best = post[best_c] > 0.5
    hit = float((gate_mask & post_best).sum() / max(gate_mask.sum(), 1))
    fa = float((~gate_mask & post_best).sum() / max((~gate_mask).sum(), 1))

    # Discrimination: mean coupled frac - mean null frac
    disc = np.mean(coupled_fracs) - np.mean(null_fracs)

    return {
        'Q': Q, 'p_stay': p_stay, 'em': em_iter,
        'coupled_mean': np.mean(coupled_fracs),
        'null_mean': np.mean(null_fracs),
        'disc': disc,
        'best_hit': hit, 'best_fa': fa,
        'best_ch': best_c,
        'sig2_ratio_coupled': float(params['sigma2_ratio'][:n_coupled].mean()),
        'sig2_ratio_null': float(params['sigma2_ratio'][n_coupled:].mean()),
    }


# --- Parameter grid ---
Q_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
p_stay_values = [0.90, 0.95, 0.98, 0.99]
em_values = [3, 5]

configs = list(itertools.product(Q_values, p_stay_values, em_values))
print(f"Total configs: {len(configs)}")

t0 = time.perf_counter()
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(run_one)(Q, p, e) for Q, p, e in configs)
elapsed = time.perf_counter() - t0
print(f"\nSweep done in {elapsed:.1f}s")

# --- Sort by discrimination ---
results.sort(key=lambda r: -r['disc'])

print(f"\n{'Q':>8} {'p_stay':>6} {'em':>3} | "
      f"{'coupled':>7} {'null':>7} {'disc':>6} | "
      f"{'hit':>5} {'fa':>5} | {'sig2r_c':>7} {'sig2r_n':>7}")
print("-" * 85)
for r in results[:20]:
    print(f"{r['Q']:>8.1e} {r['p_stay']:>6.2f} {r['em']:>3d} | "
          f"{r['coupled_mean']:>7.1%} {r['null_mean']:>7.1%} "
          f"{r['disc']:>6.1%} | "
          f"{r['best_hit']:>5.1%} {r['best_fa']:>5.1%} | "
          f"{r['sig2_ratio_coupled']:>7.3f} {r['sig2_ratio_null']:>7.3f}")

# Best by hit rate with FA < 15%
good = [r for r in results if r['best_fa'] < 0.15]
if good:
    good.sort(key=lambda r: -r['best_hit'])
    print(f"\n=== Best by hit rate (FA<15%) ===")
    for r in good[:5]:
        print(f"  Q={r['Q']:.1e} p_stay={r['p_stay']:.2f} em={r['em']}: "
              f"hit={r['best_hit']:.1%} fa={r['best_fa']:.1%} "
              f"coupled={r['coupled_mean']:.1%} null={r['null_mean']:.1%}")
