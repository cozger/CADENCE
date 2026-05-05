"""Full-session timeline: EEG coupling z(t) with significance mask + BL smile events.

Runs V6 pipeline on all segments, then plots:
  - EEG: per-band z(t) timecourse with z>2 mask shaded
  - BL: patient/therapist smile events with shared smile connectors
  - Condition boundaries as vertical lines + shaded backgrounds
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scripts.run_session_v6 import (
    load_xdf_session, extract_bl_segment, extract_eeg_segment, DEFAULT_SEGMENTS,
)
from cadence.significance.bl_coupling import facial_event_catalog
from cadence.significance.fast_cycles import eeg_coupling_timecourse
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

FS_BL = 30.0

CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
BAND_COLORS = {'theta': '#2196F3', 'alpha': '#4CAF50', 'beta': '#FF9800'}
CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']

# ── Load session ──────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print("Loading XDF...", flush=True)
session_data = load_xdf_session(xdf_path)
markers = session_data['markers']
p1_role = session_data['p1_role']
p2_role = session_data['p2_role']

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])
cache_path = [p for n, p in cached_sessions if 'y_06' in n][0]
cached = load_session_from_cache(cache_path, config)

p1_bl_ts = cached.get('p1_blendshapes_ts')
lsl_ts = session_data['landmarks']['P1'][0]
lsl_offset = float(lsl_ts[0]) - float(p1_bl_ts[0])
session_data['cached'] = cached
session_data['lsl_offset'] = lsl_offset

# Collect segments
segments = []
for seg_name in CONDITION_ORDER:
    t_start = markers.get(f'{seg_name}_start')
    t_end = markers.get(f'{seg_name}_stop')
    if t_start is not None and t_end is not None:
        segments.append((seg_name, t_start, t_end))

session_start = min(t for _, t, _ in segments) - 30
session_end = max(t for _, _, t in segments) + 30

# ── Run analysis per segment ─────────────────────────────────────────

eeg_tl_data = {}  # seg_name -> tl_result with absolute times
bl_catalogs = {}
all_events_p1 = []
all_events_p2 = []
all_shared_smiles = []

for seg_name, t_start_lsl, t_end_lsl in segments:
    print(f"  {seg_name}...", flush=True)

    # BL
    p1_bl, p2_bl, bl_dur = extract_bl_segment(session_data['landmarks'], t_start_lsl, t_end_lsl)
    if p1_bl is not None:
        cat = facial_event_catalog(p1_bl, p2_bl, FS_BL, lsl_start=t_start_lsl,
                                    segment_name=seg_name)
        bl_catalogs[seg_name] = cat
        all_events_p1.extend(cat.events_p1)
        all_events_p2.extend(cat.events_p2)
        all_shared_smiles.extend(cat.shared_smiles)

    # EEG
    p1_eeg, p2_eeg, eeg_dur, fs_eeg = extract_eeg_segment(
        cached, markers, seg_name, lsl_offset=lsl_offset)
    if p1_eeg is not None:
        t0 = time.time()
        tl = eeg_coupling_timecourse(p1_eeg, p2_eeg, fs_eeg,
                                      smooth_samples=3, n_surrogates=100, seed=42)
        elapsed = time.time() - t0
        # Convert times to absolute LSL
        tl['times_lsl'] = tl['times'] + t_start_lsl
        eeg_tl_data[seg_name] = tl
        cf = tl['combined']['coupling_fraction']
        mz = tl['combined']['mean_z']
        print(f"    EEG TL: mean_z={mz:+.1f}, coupling={cf:.0%} ({elapsed:.1f}s)", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────

print("\nGenerating plot...", flush=True)

fig, axes = plt.subplots(5, 1, figsize=(22, 16),
                         gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.8]},
                         sharex=True)

ax_theta, ax_alpha, ax_beta, ax_patient, ax_shared = axes

eeg_axes = [('theta', ax_theta), ('alpha', ax_alpha), ('beta', ax_beta)]

# Condition backgrounds on all axes
for ax in axes:
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0, t1, alpha=0.3, color=color, zorder=0)
        ax.axvline(t0, color='gray', linewidth=0.5, alpha=0.3)

# Segment labels on top
for seg_name, t0, t1 in segments:
    ax_theta.text((t0 + t1) / 2, 1.08, seg_name.replace('_', ' '),
                  ha='center', va='bottom', fontsize=9, fontweight='bold',
                  transform=ax_theta.get_xaxis_transform())

# ── EEG z(t) per band with significance masking ──────────────────────

for band_name, ax in eeg_axes:
    ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
    ax.axhline(2, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axhline(-2, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)

    for seg_name in [s[0] for s in segments]:
        tl = eeg_tl_data.get(seg_name)
        if tl is None:
            continue

        band_data = tl['per_band'].get(band_name)
        if band_data is None or band_data.get('n_valid', 0) == 0:
            continue

        t_lsl = tl['times_lsl']
        z = band_data['z']
        mask = band_data['mask']

        # Plot z timecourse
        ax.plot(t_lsl, z, color=BAND_COLORS[band_name], linewidth=0.8,
                alpha=0.7, zorder=2)

        # Shade significant regions
        sig_regions = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(sig_regions == 1)[0]
        ends = np.where(sig_regions == -1)[0]
        for s, e in zip(starts, ends):
            if s < len(t_lsl) and e <= len(t_lsl):
                ax.axvspan(t_lsl[max(0, s)], t_lsl[min(e - 1, len(t_lsl) - 1)],
                           alpha=0.25, color=BAND_COLORS[band_name], zorder=1)

    ax.set_ylabel(f'{band_name}\nz-score', fontsize=9)
    ax.set_ylim(-6, 12)

# ── BL: patient smiles + shared smile confidence ─────────────────────

# Patient smiles (colored by phasic onset, not absolute level)
for ev in all_events_p1:
    is_smile = ev.smile_phasic > 0.1 and ev.smile_velocity > 0
    color = '#E91E63' if is_smile else '#BDBDBD'
    size = 2 + max(ev.smile_phasic, 0) * 4 if is_smile else 1
    alpha = 0.7 if is_smile else 0.15
    ax_patient.plot(ev.lsl_time, ev.smile_phasic, 'o',
                    color=color, markersize=size, alpha=alpha, zorder=2)

# Therapist smiles on same axis (below zero line, inverted)
for ev in all_events_p2:
    is_smile = ev.smile_phasic > 0.1 and ev.smile_velocity > 0
    color = '#2196F3' if is_smile else '#BDBDBD'
    size = 2 + max(ev.smile_phasic, 0) * 4 if is_smile else 1
    alpha = 0.7 if is_smile else 0.15
    ax_patient.plot(ev.lsl_time, -ev.smile_phasic, 'o',
                    color=color, markersize=size, alpha=alpha, zorder=2)

# Speech activity overlay (from speech gating)
for seg_name, t0, t1 in segments:
    cat = bl_catalogs.get(seg_name)
    if cat is None or cat.speech_p1 is None:
        continue
    dur = cat.duration_s
    t_arr = np.linspace(t0, t0 + dur, len(cat.speech_p1))
    # P1 (patient) speech as fill band at top of axis
    sp1 = cat.speech_p1
    ax_patient.fill_between(t_arr, 1.8, 1.8 + sp1 * 0.35, alpha=0.5,
                            color='#E91E63', linewidth=0, zorder=1)
    # P2 (therapist) speech as fill band at bottom of axis
    sp2 = cat.speech_p2
    ax_patient.fill_between(t_arr, -1.8, -1.8 - sp2 * 0.35, alpha=0.5,
                            color='#2196F3', linewidth=0, zorder=1)

ax_patient.axhline(0, color='black', linewidth=0.5)
ax_patient.set_ylabel(f'Smile phasic\n{p1_role} (+) / {p2_role} (-)', fontsize=9)
ax_patient.set_ylim(-1.5, 1.5)

# Shared smiles highlight
for ss in all_shared_smiles:
    conf = ss.joint_smile_confidence
    alpha = min(1.0, conf * 5 + 0.3)
    ax_patient.plot(ss.event_a.lsl_time, ss.event_a.smile_phasic, 'o',
                    color='red', markersize=4 + conf * 8, alpha=alpha, zorder=3)
    ax_patient.plot(ss.event_b.lsl_time, -ss.event_b.smile_phasic, 'o',
                    color='red', markersize=4 + conf * 8, alpha=alpha, zorder=3)

# Shared smile confidence bars
for ss in all_shared_smiles:
    t_mid = (ss.event_a.lsl_time + ss.event_b.lsl_time) / 2
    conf = ss.joint_smile_confidence
    alpha = min(1.0, conf * 3)
    ax_shared.bar(t_mid, conf, width=2.0, color='#F44336', alpha=alpha, zorder=2)

# Per-segment counts
for seg_name, t0, t1 in segments:
    cat = bl_catalogs.get(seg_name)
    if cat and cat.n_shared_smiles > 0:
        ax_shared.text((t0 + t1) / 2, 0.95, f'n={cat.n_shared_smiles}',
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color='#F44336', transform=ax_shared.get_xaxis_transform())

ax_shared.set_ylabel('Shared smile\nconfidence', fontsize=9)
ax_shared.set_xlabel('LSL time (s)')
ax_shared.set_ylim(0, 1.0)
ax_shared.set_xlim(session_start, session_end)

# ── Legend + title ────────────────────────────────────────────────────

legend_elements = [
    Line2D([0], [0], color=BAND_COLORS['theta'], label='Theta (4-8 Hz)', linewidth=2),
    Line2D([0], [0], color=BAND_COLORS['alpha'], label='Alpha (8-13 Hz)', linewidth=2),
    Line2D([0], [0], color=BAND_COLORS['beta'], label='Beta (13-30 Hz)', linewidth=2),
    Patch(facecolor=BAND_COLORS['theta'], alpha=0.3, label='EEG coupled (z>2)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63',
           label=f'{p1_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
           label=f'{p2_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           label='Shared smile', markersize=8),
    Patch(facecolor='#E91E63', alpha=0.3, label=f'{p1_role.capitalize()} speech'),
    Patch(facecolor='#2196F3', alpha=0.3, label=f'{p2_role.capitalize()} speech'),
]
for seg_name in ['conv_1', 'meditate_K', 'meditate_B', 'base_EO']:
    legend_elements.append(Patch(facecolor=CONDITION_COLORS.get(seg_name, 'white'),
                                  label=seg_name.replace('_', ' '), alpha=0.5))

fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
           bbox_to_anchor=(0.99, 0.98), ncol=2)

fig.suptitle(f'y_06 Full Session: EEG Coupling + Facial Events ({p1_role} vs {p2_role})',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = 'results/v6/y_06/full_session_eeg_bl_timeline.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_path}")

# Print summary
print(f"\n{'seg':>12} | {'EEG_z_mean':>10} {'coupling%':>10} | {'smiles':>7}")
print("-" * 50)
for seg_name, t0, t1 in segments:
    tl = eeg_tl_data.get(seg_name)
    cat = bl_catalogs.get(seg_name)
    eeg_z = tl['combined']['mean_z'] if tl else 0
    eeg_cf = tl['combined']['coupling_fraction'] if tl else 0
    n_sm = cat.n_shared_smiles if cat else 0
    print(f"{seg_name:>12} | {eeg_z:+10.1f} {eeg_cf:9.0%} | {n_sm:7d}")
