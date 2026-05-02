"""Generate full-session timeline visualization for y_06.

Single continuous timeline showing all facial events across all segments,
with condition boundaries marked by vertical lines and shaded regions.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import facial_event_catalog

FS_BL = 30.0

# Condition colors
CONDITION_COLORS = {
    'base_EO': '#E3F2FD',
    'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0',
    'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5',
    'meditate_K': '#E8F5E9',
}

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B',
                   'meditate_K', 'conv_2']

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print(f"Loading {os.path.basename(xdf_path)}...", flush=True)
session_data = load_xdf_session(xdf_path)

markers = session_data['markers']
p1_role = session_data['p1_role']
p2_role = session_data['p2_role']

# Find session time range from markers
all_marker_times = sorted(markers.values())
session_start = all_marker_times[0] - 30  # 30s before first marker
session_end = all_marker_times[-1] + 30

# Collect all segment boundaries
segments = []
for seg_name in CONDITION_ORDER:
    t_start = markers.get(f'{seg_name}_start')
    t_end = markers.get(f'{seg_name}_stop')
    if t_start is not None and t_end is not None:
        segments.append((seg_name, t_start, t_end))

print(f"Session range: {session_start:.0f} - {session_end:.0f} LSL")
print(f"Segments: {len(segments)}")
for name, t0, t1 in segments:
    print(f"  {name}: {t0:.0f} - {t1:.0f} ({t1-t0:.0f}s)")

# Run facial event detection on each segment
all_events_p1 = []
all_events_p2 = []
all_shared_smiles = []
segment_catalogs = {}

for seg_name, t_start, t_end in segments:
    p1, p2, dur = extract_bl_segment(session_data['landmarks'], t_start, t_end)
    if p1 is None:
        continue

    cat = facial_event_catalog(p1, p2, FS_BL, lsl_start=t_start,
                                segment_name=seg_name)
    segment_catalogs[seg_name] = cat

    all_events_p1.extend(cat.events_p1)
    all_events_p2.extend(cat.events_p2)
    all_shared_smiles.extend(cat.shared_smiles)

    print(f"  {seg_name}: {cat.n_events_p1} patient + {cat.n_events_p2} therapist, "
          f"{cat.n_shared_smiles} shared smiles")

# ── Plot full session timeline ────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(20, 10),
                         gridspec_kw={'height_ratios': [1, 1, 0.6]},
                         sharex=True)

ax_patient = axes[0]
ax_therapist = axes[1]
ax_shared = axes[2]

# Draw condition backgrounds on all axes
for ax in axes:
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0, t1, alpha=0.4, color=color, zorder=0)
        # Segment boundary lines
        ax.axvline(t0, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        ax.axvline(t1, color='gray', linewidth=0.5, alpha=0.5, zorder=1)

# Segment labels on top axis
for seg_name, t0, t1 in segments:
    ax_patient.text((t0 + t1) / 2, 1.05, seg_name.replace('_', ' '),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    transform=ax_patient.get_xaxis_transform())

# Patient events (top)
for ev in all_events_p1:
    is_smile = ev.smile_composite > 0.3
    color = '#E91E63' if is_smile else '#BDBDBD'
    size = 2 + ev.smile_composite * 5 if is_smile else 1.5
    alpha = 0.8 if is_smile else 0.3
    ax_patient.plot(ev.lsl_time, ev.smile_composite, 'o',
                    color=color, markersize=size, alpha=alpha, zorder=2)

ax_patient.set_ylabel(f'{p1_role.capitalize()}\nSmile composite')
ax_patient.set_ylim(-0.05, 2.2)
ax_patient.axhline(0.3, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

# Therapist events (middle)
for ev in all_events_p2:
    is_smile = ev.smile_composite > 0.3
    color = '#2196F3' if is_smile else '#BDBDBD'
    size = 2 + ev.smile_composite * 5 if is_smile else 1.5
    alpha = 0.8 if is_smile else 0.3
    ax_therapist.plot(ev.lsl_time, ev.smile_composite, 'o',
                      color=color, markersize=size, alpha=alpha, zorder=2)

ax_therapist.set_ylabel(f'{p2_role.capitalize()}\nSmile composite')
ax_therapist.set_ylim(-0.05, 2.2)
ax_therapist.axhline(0.3, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

# Shared smiles (bottom) — confidence bars
for ss in all_shared_smiles:
    t_mid = (ss.event_a.lsl_time + ss.event_b.lsl_time) / 2
    conf = ss.joint_smile_confidence
    alpha = min(1.0, conf * 3)
    ax_shared.bar(t_mid, conf, width=2.0, color='#F44336', alpha=alpha, zorder=2)

    # Connecting line on patient/therapist axes
    ax_patient.plot(ss.event_a.lsl_time, ss.event_a.smile_composite, 'o',
                    color='#F44336', markersize=4 + conf * 6, alpha=alpha, zorder=3)
    ax_therapist.plot(ss.event_b.lsl_time, ss.event_b.smile_composite, 'o',
                      color='#F44336', markersize=4 + conf * 6, alpha=alpha, zorder=3)

ax_shared.set_ylabel('Shared smile\nconfidence')
ax_shared.set_xlabel('LSL time (s)')
ax_shared.set_ylim(0, 1.0)

# Set x range to cover full session
ax_shared.set_xlim(session_start, session_end)

# Title
fig.suptitle(f'y_06 Full Session: {len(all_shared_smiles)} shared smiles '
             f'({p1_role} = top, {p2_role} = middle)',
             fontsize=13, fontweight='bold')

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63',
           label=f'{p1_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
           label=f'{p2_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
           label='Shared smile', markersize=8),
    Patch(facecolor=CONDITION_COLORS['conv_1'], label='Conversation', alpha=0.6),
    Patch(facecolor=CONDITION_COLORS['meditate_K'], label='Meditation (K)', alpha=0.6),
    Patch(facecolor=CONDITION_COLORS['meditate_B'], label='Meditation (B)', alpha=0.6),
    Patch(facecolor=CONDITION_COLORS['base_EO'], label='Baseline EO', alpha=0.6),
    Patch(facecolor=CONDITION_COLORS['base_EC'], label='Baseline EC', alpha=0.6),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
           bbox_to_anchor=(0.99, 0.98), ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = 'results/v6/y_06/full_session_timeline.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_path}")

# Print per-condition summary
print(f"\nPer-condition summary:")
print(f"{'condition':>15} {'patient_ev':>10} {'therapist_ev':>12} {'shared_sm':>10} {'dur':>6}")
for seg_name, t0, t1 in segments:
    cat = segment_catalogs.get(seg_name)
    if cat is None:
        continue
    print(f"{seg_name:>15} {cat.n_events_p1:>10} {cat.n_events_p2:>12} "
          f"{cat.n_shared_smiles:>10} {t1-t0:>5.0f}s")
