"""Check AUC variance under pure null (random z vs random gate)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

rng = np.random.default_rng(42)
N = 600  # 300s at 2 Hz

for sigma in [0, 1.5, 2.5, 5.0]:
    aucs = []
    for i in range(500):
        z = rng.standard_normal(N)
        if sigma > 0:
            z = gaussian_filter1d(z, sigma=sigma)
        # Generate a gate similar to our test: 30% duty, 10-30s episodes
        gate = np.zeros(N)
        t = 0
        while t < N:
            gap = rng.integers(20, 60)  # 10-30s gap at 2Hz
            t += gap
            dur = rng.integers(20, 60)  # 10-30s episode
            gate[t:min(t + dur, N)] = 1
            t += dur
        if gate.sum() > 10 and gate.sum() < N - 10:
            aucs.append(roc_auc_score(gate, z))

    aucs = np.array(aucs)
    print(f"sigma={sigma:4.1f}: mean={aucs.mean():.3f} std={aucs.std():.3f} "
          f"range=[{aucs.min():.3f}, {aucs.max():.3f}] "
          f"frac<0.45={np.mean(aucs < 0.45):.0%}")
