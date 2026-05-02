"""Quick BIC sweep to find optimal lambda for group-sparse Hawkes."""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf
import glob

from cadence.significance.hawkes_coupling import (
    nmf_expression_discovery, detect_component_events,
    fit_hawkes_group_sparse,
)

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

marker_times = {}
for stream in data:
    if stream['info']['type'][0] == 'Markers':
        for t, v in zip(stream['time_stamps'], stream['time_series']):
            marker_times[v[0]] = t

t_start = marker_times['conv_1_start']
t_end = marker_times['conv_1_stop']
dur = t_end - t_start

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (
                np.array(stream['time_stamps']),
                np.array(stream['time_series'], dtype=np.float32))

T = int(dur * 30)
t_grid = np.linspace(0, dur, T)

sigs = {}
for p in ['P1', 'P2']:
    ts, d = landmarks[p]
    m = (ts >= t_start) & (ts <= t_end)
    sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)

H, W1, W2, names, expl = nmf_expression_discovery(sigs['P1'], sigs['P2'], 6, seed=42)
ev1 = detect_component_events(W1, fs=30)
ev2 = detect_component_events(W2, fs=30)
all_events = ev1 + ev2

N_total = sum(len(e) for e in all_events)
print(f"N_total={N_total}, dur={dur:.1f}s, D={len(all_events)}")
print(f"{'lam':>8}  {'ll':>12}  {'n_active':>8}  {'n_params':>8}  {'BIC':>12}")

for lam in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    mu, A, centers, active, ll, info = fit_hawkes_group_sparse(
        all_events, dur, n_basis=5, lambda_group=lam,
        bic_select=False, max_iter=50)
    n_act = len(active)
    n_params = len(all_events) + n_act * 5  # D baselines + active kernels
    bic = -2 * ll + n_params * np.log(N_total)

    # Show which cross-person pathways are active
    cross = [(s, t) for s, t in active if (s < 6) != (t < 6)]
    print(f"{lam:8.3f}  {ll:12.1f}  {n_act:8d}  {n_params:8d}  {bic:12.1f}  "
          f"cross={len(cross)}")
