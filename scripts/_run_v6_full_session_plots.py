"""Full-session EEG + BL timeline for y_06.

EEG: per-band (theta/alpha/beta) cycle analysis with all features
     (volt_amp, period, symmetry, burst co-occurrence, cycle-PLV).
BL:  saliency-based facial events with shared smile confidence.

Two separate figures, both spanning the full session with condition markers.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import facial_event_catalog
from cadence.significance.fast_cycles import analyze_interbrain_cycles_multiband, EEG_BANDS
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

FS_BL = 30.0

CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
BAND_COLORS = {'theta': '#2196F3', 'alpha': '#4CAF50', 'beta': '#FF9800'}

# ── Load session ──────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print("Loading XDF...", flush=True)
session_data = load_xdf_session(xdf_path)
markers = session_data['markers']
p1_role = session_data['p1_role']
p2_role = session_data['p2_role']

# Load cache for EEG
config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])
cache_path = [p for n, p in cached_sessions if 'y_06' in n][0]
cached = load_session_from_cache(cache_path, config)

# Compute LSL offset
p1_bl_ts = cached.get('p1_blendshapes_ts')
lsl_ts = session_data['landmarks']['P1'][0]
lsl_offset = float(lsl_ts[0]) - float(p1_bl_ts[0])

# Collect segments
segments = []
for seg_name in CONDITION_ORDER:
    t_start = markers.get(f'{seg_name}_start')
    t_end = markers.get(f'{seg_name}_stop')
    if t_start is not None and t_end is not None:
        segments.append((seg_name, t_start, t_end))

session_start = min(t for _, t, _ in segments) - 30
session_end = max(t for _, _, t in segments) + 30

# ── Run EEG analysis per segment ─────────────────────────────────────

print("\nRunning EEG cycle analysis per segment...", flush=True)
eeg_results = {}

for seg_name, t_start_lsl, t_end_lsl in segments:
    t_start_rel = t_start_lsl - lsl_offset
    t_end_rel = t_end_lsl - lsl_offset

    sigs = {}
    for person in ['p1', 'p2']:
        eeg_all = cached[f'{person}_eeg']
        ts_all = cached[f'{person}_eeg_ts']
        m = (ts_all >= t_start_rel) & (ts_all <= t_end_rel)
        if m.sum() < 1000:
            break
        eeg_seg = eeg_all[m, :14].astype(np.float64)
        eeg_seg -= eeg_seg.mean(axis=1, keepdims=True)
        for ch in range(14):
            std = eeg_seg[:, ch].std()
            if std > 1e-8:
                eeg_seg[:, ch] = (eeg_seg[:, ch] - eeg_seg[:, ch].mean()) / std
        sigs[person] = eeg_seg
        if person == 'p1':
            fs_eeg = len(ts_all[m]) / (ts_all[m][-1] - ts_all[m][0])

    if 'p1' not in sigs or 'p2' not in sigs:
        print(f"  {seg_name}: no EEG", flush=True)
        continue

    min_len = min(len(sigs['p1']), len(sigs['p2']))
    result = analyze_interbrain_cycles_multiband(
        sigs['p1'][:min_len], sigs['p2'][:min_len], fs_eeg,
        n_surrogates=200, seed=42)

    eeg_results[seg_name] = result

    # Print full stats
    combined_z = 0
    band_strs = []
    for band in ['theta', 'alpha', 'beta']:
        br = result.get('per_band', {}).get(band, {})
        va_z = br.get('volt_amp', {}).get('pooled_z', 0)
        per_z = br.get('period', {}).get('pooled_z', 0)
        sym_z = br.get('time_rdsym', {}).get('pooled_z', 0)
        burst_z = br.get('burst_cooc', {}).get('pooled_z', 0)
        plv_z = br.get('cycle_plv', {}).get('pooled_z', 0)
        band_strs.append(f"{band}: va={va_z:+.1f} per={per_z:+.1f} "
                         f"sym={sym_z:+.1f} burst={burst_z:+.1f} plv={plv_z:+.1f}")

    comb = result.get('combined', {}).get('volt_amp', {})
    combined_z = comb.get('stouffer_z', comb.get('pooled_z', 0))
    print(f"  {seg_name}: combined={combined_z:+.1f}", flush=True)
    for bs in band_strs:
        print(f"    {bs}")

# ── Run BL analysis per segment ───────────────────────────────────────

print("\nRunning BL analysis per segment...", flush=True)
bl_catalogs = {}
all_events_p1 = []
all_events_p2 = []
all_shared_smiles = []

for seg_name, t_start, t_end in segments:
    p1, p2, dur = extract_bl_segment(session_data['landmarks'], t_start, t_end)
    if p1 is None:
        continue
    cat = facial_event_catalog(p1, p2, FS_BL, lsl_start=t_start, segment_name=seg_name)
    bl_catalogs[seg_name] = cat
    all_events_p1.extend(cat.events_p1)
    all_events_p2.extend(cat.events_p2)
    all_shared_smiles.extend(cat.shared_smiles)
    print(f"  {seg_name}: {cat.n_shared_smiles} shared smiles", flush=True)

# ── PLOT 1: EEG Full Session ─────────────────────────────────────────

print("\nGenerating EEG plot...", flush=True)

features = ['volt_amp', 'period', 'time_rdsym', 'burst_cooc', 'cycle_plv']
feat_labels = ['Amplitude\n(volt_amp)', 'Period', 'Symmetry\n(rise-decay)',
               'Burst\nco-occurrence', 'Cycle PLV']

fig, axes = plt.subplots(len(features), 1, figsize=(20, 14), sharex=True)

for feat_idx, (feat, feat_label) in enumerate(zip(features, feat_labels)):
    ax = axes[feat_idx]

    # Condition backgrounds
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0, t1, alpha=0.4, color=color, zorder=0)
        ax.axvline(t0, color='gray', linewidth=0.5, alpha=0.3, zorder=1)

    # Segment labels on top axis only
    if feat_idx == 0:
        for seg_name, t0, t1 in segments:
            ax.text((t0 + t1) / 2, 1.05, seg_name.replace('_', ' '),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    transform=ax.get_xaxis_transform())

    # Plot per-band z-scores as bars at segment midpoints
    seg_x = [(t0 + t1) / 2 for _, t0, t1 in segments]
    seg_names = [name for name, _, _ in segments]
    bar_width = min((t1 - t0) * 0.25 for _, t0, t1 in segments)

    for bi, band in enumerate(['theta', 'alpha', 'beta']):
        z_vals = []
        for seg_name in seg_names:
            r = eeg_results.get(seg_name, {})
            br = r.get('per_band', {}).get(band, {})
            z = br.get(feat, {}).get('pooled_z', 0)
            z_vals.append(z)

        offset = (bi - 1) * bar_width
        bars = ax.bar([x + offset for x in seg_x], z_vals, bar_width,
                      color=BAND_COLORS[band], alpha=0.8, label=band if feat_idx == 0 else '',
                      zorder=2)

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axhline(2, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axhline(-2, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_ylabel(feat_label, fontsize=9)

axes[-1].set_xlabel('LSL time (s)')
axes[-1].set_xlim(session_start, session_end)

# Combined z on top
ax_top = axes[0]
for seg_name in seg_names:
    r = eeg_results.get(seg_name, {})
    comb = r.get('combined', {}).get('volt_amp', {})
    cz = comb.get('stouffer_z', comb.get('pooled_z', 0))
    seg_t0 = [t0 for n, t0, _ in segments if n == seg_name][0]
    seg_t1 = [t1 for n, _, t1 in segments if n == seg_name][0]
    ax_top.text((seg_t0 + seg_t1) / 2, ax_top.get_ylim()[1] * 0.9,
                f'z={cz:+.1f}', ha='center', fontsize=7, fontweight='bold',
                color='#F44336' if abs(cz) > 2 else 'gray')

fig.legend(['Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)'],
           loc='upper right', fontsize=9, bbox_to_anchor=(0.99, 0.98))

fig.suptitle(f'y_06 EEG Inter-Brain Cycle Analysis ({p1_role} vs {p2_role})',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

eeg_path = 'results/v6/y_06/full_session_eeg.png'
fig.savefig(eeg_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {eeg_path}")

# ── PLOT 2: BL Full Session ──────────────────────────────────────────

print("Generating BL plot...", flush=True)

fig, axes = plt.subplots(3, 1, figsize=(20, 10),
                         gridspec_kw={'height_ratios': [1, 1, 0.6]},
                         sharex=True)

ax_patient, ax_therapist, ax_shared = axes

for ax in axes:
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0, t1, alpha=0.4, color=color, zorder=0)
        ax.axvline(t0, color='gray', linewidth=0.5, alpha=0.3, zorder=1)

for seg_name, t0, t1 in segments:
    ax_patient.text((t0 + t1) / 2, 1.05, seg_name.replace('_', ' '),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    transform=ax_patient.get_xaxis_transform())

# Patient events
for ev in all_events_p1:
    is_smile = ev.smile_composite > 0.3
    color = '#E91E63' if is_smile else '#BDBDBD'
    size = 2 + ev.smile_composite * 5 if is_smile else 1.5
    alpha = 0.8 if is_smile else 0.2
    ax_patient.plot(ev.lsl_time, ev.smile_composite, 'o',
                    color=color, markersize=size, alpha=alpha, zorder=2)

ax_patient.set_ylabel(f'{p1_role.capitalize()}\nSmile composite')
ax_patient.set_ylim(-0.05, 2.2)

# Therapist events
for ev in all_events_p2:
    is_smile = ev.smile_composite > 0.3
    color = '#2196F3' if is_smile else '#BDBDBD'
    size = 2 + ev.smile_composite * 5 if is_smile else 1.5
    alpha = 0.8 if is_smile else 0.2
    ax_therapist.plot(ev.lsl_time, ev.smile_composite, 'o',
                      color=color, markersize=size, alpha=alpha, zorder=2)

ax_therapist.set_ylabel(f'{p2_role.capitalize()}\nSmile composite')
ax_therapist.set_ylim(-0.05, 2.2)

# Shared smiles
for ss in all_shared_smiles:
    t_mid = (ss.event_a.lsl_time + ss.event_b.lsl_time) / 2
    conf = ss.joint_smile_confidence
    alpha = min(1.0, conf * 3)
    ax_shared.bar(t_mid, conf, width=2.0, color='#F44336', alpha=alpha, zorder=2)
    ax_patient.plot(ss.event_a.lsl_time, ss.event_a.smile_composite, 'o',
                    color='#F44336', markersize=4 + conf * 6, alpha=alpha, zorder=3)
    ax_therapist.plot(ss.event_b.lsl_time, ss.event_b.smile_composite, 'o',
                      color='#F44336', markersize=4 + conf * 6, alpha=alpha, zorder=3)

# Per-segment shared smile count
for seg_name, t0, t1 in segments:
    cat = bl_catalogs.get(seg_name)
    if cat and cat.n_shared_smiles > 0:
        ax_shared.text((t0 + t1) / 2, 0.95, f'n={cat.n_shared_smiles}',
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color='#F44336', transform=ax_shared.get_xaxis_transform())

ax_shared.set_ylabel('Shared smile\nconfidence')
ax_shared.set_xlabel('LSL time (s)')
ax_shared.set_ylim(0, 1.0)
ax_shared.set_xlim(session_start, session_end)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63',
           label=f'{p1_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
           label=f'{p2_role.capitalize()} smile', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
           label='Shared smile', markersize=8),
]
for seg_name in ['conv_1', 'meditate_K', 'meditate_B', 'base_EO', 'base_EC']:
    legend_elements.append(Patch(facecolor=CONDITION_COLORS.get(seg_name, 'white'),
                                  label=seg_name.replace('_', ' '), alpha=0.6))

fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
           bbox_to_anchor=(0.99, 0.98), ncol=2)

fig.suptitle(f'y_06 Facial Expression Events ({p1_role} vs {p2_role})',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

bl_path = 'results/v6/y_06/full_session_bl.png'
fig.savefig(bl_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {bl_path}")

# ── Summary table ─────────────────────────────────────────────────────

print(f"\n{'='*100}")
print(f"FULL SESSION SUMMARY: y_06 ({p1_role} vs {p2_role})")
print(f"{'='*100}")

print(f"\n{'condition':>12} {'dur':>5} | {'th_va':>6} {'al_va':>6} {'be_va':>6} {'comb':>6} | "
      f"{'th_per':>6} {'al_per':>6} {'be_per':>6} | "
      f"{'smiles':>6} {'rate':>6}")
print("-" * 100)

for seg_name, t0, t1 in segments:
    dur = t1 - t0
    r = eeg_results.get(seg_name, {})
    cat = bl_catalogs.get(seg_name)

    eeg_vals = {}
    for band in ['theta', 'alpha', 'beta']:
        br = r.get('per_band', {}).get(band, {})
        eeg_vals[f'{band}_va'] = br.get('volt_amp', {}).get('pooled_z', 0)
        eeg_vals[f'{band}_per'] = br.get('period', {}).get('pooled_z', 0)
    comb = r.get('combined', {}).get('volt_amp', {})
    comb_z = comb.get('stouffer_z', comb.get('pooled_z', 0))

    n_sm = cat.n_shared_smiles if cat else 0
    rate = n_sm / (dur / 60) if dur > 0 else 0

    print(f"{seg_name:>12} {dur:5.0f} | "
          f"{eeg_vals.get('theta_va', 0):+6.1f} {eeg_vals.get('alpha_va', 0):+6.1f} "
          f"{eeg_vals.get('beta_va', 0):+6.1f} {comb_z:+6.1f} | "
          f"{eeg_vals.get('theta_per', 0):+6.1f} {eeg_vals.get('alpha_per', 0):+6.1f} "
          f"{eeg_vals.get('beta_per', 0):+6.1f} | "
          f"{n_sm:6d} {rate:5.1f}/m")
