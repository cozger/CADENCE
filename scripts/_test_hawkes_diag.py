"""Diagnose: does injected mimicry show up in NMF components?"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob
from scipy.signal import find_peaks

from cadence.synthetic import generate_coupling_gate
from cadence.significance.hawkes_coupling import (
    nmf_expression_discovery, detect_component_events,
    fit_hawkes_group_sparse, MP_BLENDSHAPE_NAMES,
)
from scripts._test_hawkes_semisynthetic import load_raw_bl, inject_raw_event_mimicry

FS = 30.0
SMILE_AUS = [44, 45]

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
p1, p2, dur = load_raw_bl(xdf_path, 'conv_1')
T = min(p1.shape[0], p2.shape[0])

gate_profile = {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0}
gate = generate_coupling_gate(T, FS, gate_profile, seed=999)

# Inject at kappa=1.0 for maximum signal
p2_coupled, n_inj, inj_times = inject_raw_event_mimicry(
    p1, p2, gate, kappa=1.0, lag_s=2.5, seed=0)

print(f"Injected {n_inj} mimicry events")
print(f"P2 smile AU44 range: [{p2[:T,44].min():.3f}, {p2[:T,44].max():.3f}] -> "
      f"[{p2_coupled[:T,44].min():.3f}, {p2_coupled[:T,44].max():.3f}]")

# Run NMF on coupled data
H, W1, W2, names, expl = nmf_expression_discovery(p1[:T], p2_coupled[:T], 6, seed=42)

# Which component captures smile?
smile_loading = H[:, SMILE_AUS].sum(axis=1)
smile_comp = np.argmax(smile_loading)
print(f"\nSmile component: {smile_comp} ('{names[smile_comp]}')")
print(f"Smile AU loading: {smile_loading}")

# Detect events in smile component for both people
ev1 = detect_component_events(W1, fs=FS)
ev2 = detect_component_events(W2, fs=FS)

n1 = len(ev1[smile_comp])
n2 = len(ev2[smile_comp])
print(f"P1 smile events: {n1}")
print(f"P2 smile events: {n2}")

# Check co-occurrence of injection times with P2 smile events
if len(inj_times) > 0 and n2 > 0:
    matched = 0
    for t_inj in inj_times:
        if np.any(np.abs(ev2[smile_comp] - t_inj) < 3.0):
            matched += 1
    print(f"Injection times matched by P2 smile events: {matched}/{len(inj_times)}")

# Also check raw AU peaks
raw_smile = p2_coupled[:T, 44] + p2_coupled[:T, 45]
pks, _ = find_peaks(raw_smile, prominence=0.15, distance=int(2.0 * FS))
print(f"\nRaw P2 smile peaks (prom>=0.15): {len(pks)}")
print(f"P2 NMF smile comp events: {n2}")

# Run group-sparse Hawkes on just the smile components (2D: P1_smile, P2_smile)
print(f"\n--- 2D Hawkes on smile only ---")
events_2d = [ev1[smile_comp], ev2[smile_comp]]
for lam in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    mu, A, centers, active, ll, info = fit_hawkes_group_sparse(
        events_2d, dur, n_basis=5, lambda_group=lam, bic_select=False, max_iter=50)
    # A[1,:,0] = P1→P2, A[0,:,1] = P2→P1
    kernel_12 = np.linalg.norm(A[1, :, 0])
    kernel_21 = np.linalg.norm(A[0, :, 1])
    print(f"  lam={lam:5.2f}: P1→P2 kernel={kernel_12:.4f}, "
          f"P2→P1 kernel={kernel_21:.4f}, active={len(active)}")
