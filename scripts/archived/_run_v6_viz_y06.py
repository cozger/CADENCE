"""Generate V6 visualizations for y_06."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxdf, glob
import numpy as np

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment, DEFAULT_SEGMENTS
from cadence.significance.bl_coupling import facial_event_catalog
from cadence.visualization.v6_plots import (
    plot_shared_smile_timeline, plot_eeg_condition_bars, plot_session_dashboard
)
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

# First run the full session if not already done
results_path = 'results/v6/y_06/v6_results.json'
if not os.path.exists(results_path):
    print("Running V6 analysis first...")
    os.system('conda run -n MCCT python scripts/run_session_v6.py --session y_06')

# Load results
with open(results_path) as f:
    session_results = json.load(f)

print(f"Session: {session_results['session']}")
print(f"Roles: {session_results['p1_role']} / {session_results['p2_role']}")

# Generate plots
out_dir = 'results/v6/y_06'

# 1. EEG condition bars
fig = plot_eeg_condition_bars(session_results,
                              title=f"y_06: EEG Amplitude Coupling",
                              save_path=os.path.join(out_dir, 'eeg_conditions.png'))
if fig:
    print(f"Saved eeg_conditions.png")

# 2. Session dashboard
fig = plot_session_dashboard(session_results,
                              title=f"y_06 Session Dashboard",
                              save_path=os.path.join(out_dir, 'dashboard.png'))
if fig:
    print(f"Saved dashboard.png")

# 3. Per-segment smile timelines (need to regenerate catalogs)
xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
session_data = load_xdf_session(xdf_path)

markers = session_data['markers']
for segment in ['conv_1', 'conv_2']:
    t_start = markers.get(f'{segment}_start')
    t_end = markers.get(f'{segment}_stop')
    if t_start is None:
        continue

    p1, p2, dur = extract_bl_segment(session_data['landmarks'], t_start, t_end)
    if p1 is None:
        continue

    cat = facial_event_catalog(p1, p2, 30.0, lsl_start=t_start, segment_name=segment)

    fig = plot_shared_smile_timeline(
        cat,
        title=f"y_06 {segment}: {cat.n_shared_smiles} shared smiles "
              f"({session_results['p1_role']} = top, {session_results['p2_role']} = bottom)",
        save_path=os.path.join(out_dir, f'smile_timeline_{segment}.png'))
    print(f"Saved smile_timeline_{segment}.png ({cat.n_shared_smiles} shared smiles)")

print(f"\nAll plots saved to {out_dir}")
