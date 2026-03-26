"""Quick test: LLR+CUSUM on synthetic data."""
import numpy as np
from cadence.significance.temporal_localization import llr_cusum_localization

rng = np.random.RandomState(42)
C, T, K = 15, 27000, 100  # 15 ch, 1800s @ 15 Hz, 100 surrogates
eval_rate = 15.0
tau = 3.0

# Null dR2: small positive values with noise
dr2_real = rng.randn(C, T) * 0.01 + 0.005  # mean ~0.005
dr2_surr = rng.randn(K, C, T) * 0.01 + 0.005

# Inject coupling into channels 0-3 during 25% duty cycle
gate = np.zeros(T)
# Create ~50 episodes of 3-15s (45-225 samples at 15 Hz)
t = 0
while t < T:
    gap = rng.randint(100, 500)
    t += gap
    if t >= T: break
    dur = rng.randint(45, 225)
    gate[t:min(t+dur, T)] = 1.0
    t += dur
duty = gate.mean()
print(f"Gate duty: {duty:.1%}")

# Coupled channels get boosted dR2 during gate
for ch in range(4):
    dr2_real[ch] += gate * 0.05  # extra dR2 during coupling

mask, posterior, diag = llr_cusum_localization(
    dr2_real, dr2_surr, eval_rate, tau,
    target_fa=0.10, min_event_s=2.0)

print(f"mask.sum()={mask.sum()}, mask.mean()={mask.mean():.3f}")
print(f"posterior.sum()={posterior.sum():.1f}, (posterior>0.5).sum()={(posterior>0.5).sum()}")
print(f"coupling_fraction={diag['coupling_fraction']:.3f}")
print(f"z_multi_max={diag.get('z_multi_max', 0):.2f}, thresh={diag['threshold']:.2f}")
print(f"pos_ch={diag['n_positive_channels']}")
print(f"z_multi_mean={diag.get('z_multi_mean', 0):.2f}, z_multi_p95={diag.get('z_multi_p95', 0):.2f}")

# Check alignment with gate
gate_on = gate > 0.5
post_on = posterior > 0.5
hits = (gate_on & post_on).sum()
fas = (~gate_on & post_on).sum()
print(f"\nhit_rate={hits/max(gate_on.sum(),1):.3f}")
print(f"fa_rate={fas/max((~gate_on).sum(),1):.3f}")
