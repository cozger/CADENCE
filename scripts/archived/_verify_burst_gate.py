"""Quick verification of burst coupling gate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cadence.synthetic import generate_burst_coupling_gate

gate, info = generate_burst_coupling_gate(460800, 256.0, seed=42)
print(f"Gate shape: {gate.shape}, overall duty: {gate.mean():.3%}")
print(f"Info: {info}")

# Burst structure analysis
transitions = np.sum(np.abs(np.diff(gate > 0.1)))
print(f"Transitions (gate>0.1): {transitions}")

# Typical burst train duration
on = gate > 0.1
edges = np.diff(on.astype(np.int8), prepend=0)
onsets = np.where(edges == 1)[0]
offsets = np.where(edges == -1)[0]
if len(offsets) < len(onsets):
    offsets = np.append(offsets, len(gate))
durs_ms = (offsets[:len(onsets)] - onsets) / 256.0 * 1000
print(f"Burst durations: n={len(durs_ms)}, "
      f"mean={durs_ms.mean():.0f}ms, "
      f"median={np.median(durs_ms):.0f}ms, "
      f"range=[{durs_ms.min():.0f}, {durs_ms.max():.0f}]ms")

# Gap analysis
if len(onsets) > 1:
    gaps_ms = (onsets[1:] - offsets[:len(onsets)-1]) / 256.0 * 1000
    short_gaps = gaps_ms[gaps_ms < 500]  # inter-burst
    long_gaps = gaps_ms[gaps_ms >= 500]  # inter-train
    print(f"Inter-burst gaps (<500ms): n={len(short_gaps)}, "
          f"mean={short_gaps.mean():.0f}ms" if len(short_gaps) else "")
    print(f"Inter-train gaps (≥500ms): n={len(long_gaps)}, "
          f"mean={long_gaps.mean():.0f}ms" if len(long_gaps) else "")
