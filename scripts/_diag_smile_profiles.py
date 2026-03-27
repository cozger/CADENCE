"""Compare average AU profiles: meditation 'smiles' (noise) vs conversation shared smiles (ground truth)."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import facial_event_catalog
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
FS = 30.0

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'smile_diag')
os.makedirs(out_dir, exist_ok=True)

# Collect snapshots from different categories
profiles = {
    'med_smile_p1': [],      # meditation patient "smiles" (noise)
    'med_smile_p2': [],      # meditation therapist "smiles" (noise)
    'med_nonsmile_p1': [],   # meditation patient non-smiles
    'conv_shared_a': [],     # conversation shared smile — person A
    'conv_shared_b': [],     # conversation shared smile — person B
    'conv_nonsmile_p1': [],  # conversation patient non-smiles
}

for seg in ['meditate_K', 'meditate_B']:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue
    cat = facial_event_catalog(p1_bl, p2_bl, FS, lsl_start=t0, segment_name=seg)

    for ev in cat.events_p1:
        if ev.smile_composite > 0.3:
            profiles['med_smile_p1'].append(ev.au_snapshot)
        else:
            profiles['med_nonsmile_p1'].append(ev.au_snapshot)
    for ev in cat.events_p2:
        if ev.smile_composite > 0.3:
            profiles['med_smile_p2'].append(ev.au_snapshot)

for seg in ['conv_1', 'conv_2']:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue
    cat = facial_event_catalog(p1_bl, p2_bl, FS, lsl_start=t0, segment_name=seg)

    for ss in cat.shared_smiles:
        profiles['conv_shared_a'].append(ss.event_a.au_snapshot)
        profiles['conv_shared_b'].append(ss.event_b.au_snapshot)
    for ev in cat.events_p1:
        if ev.smile_composite < 0.15:
            profiles['conv_nonsmile_p1'].append(ev.au_snapshot)

# Average each category
avg = {}
for k, snaps in profiles.items():
    if snaps:
        avg[k] = np.mean(snaps, axis=0)
        print(f"{k}: n={len(snaps)}")
    else:
        print(f"{k}: EMPTY")

# ---- Plot 1: Side-by-side bar comparison (non-eye AUs only) ----
non_eye = [i for i in range(52) if i < 9 or i > 22]
labels = [f"{i:02d} {MP_BLENDSHAPE_NAMES[i][:20]}" for i in non_eye]

categories = [
    ('conv_shared_a', 'Conv shared smile (A)', '#E91E63'),
    ('conv_shared_b', 'Conv shared smile (B)', '#F44336'),
    ('med_smile_p1', 'Meditation "smile" (patient)', '#9E9E9E'),
    ('med_smile_p2', 'Meditation "smile" (therapist)', '#BDBDBD'),
]

fig, ax = plt.subplots(figsize=(14, 10))
y = np.arange(len(non_eye))
bar_h = 0.2

for i, (key, label, color) in enumerate(categories):
    if key in avg:
        vals = avg[key][non_eye]
        ax.barh(y + i * bar_h, vals, height=bar_h, label=label,
                color=color, alpha=0.8)

ax.set_yticks(y + bar_h * 1.5)
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Mean AU activation')
ax.set_title('Average AU profile: Genuine shared smiles vs meditation "smiles"')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'profile_comparison_full.png'), dpi=120)
plt.close(fig)
print(f"\nSaved: profile_comparison_full.png")

# ---- Plot 2: Focused comparison on affect + key AUs ----
focus_aus = [
    (44, 'mouthSmileL'), (45, 'mouthSmileR'),
    (7, 'cheekSquintL'), (8, 'cheekSquintR'),
    (30, 'mouthFrownL'), (31, 'mouthFrownR'),
    (28, 'mouthDimpleL'), (29, 'mouthDimpleR'),
    (50, 'noseSneerL'), (51, 'noseSneerR'),
    (3, 'browInnerUp'), (4, 'browOuterUpL'), (5, 'browOuterUpR'),
    (1, 'browDownL'), (2, 'browDownR'),
    (42, 'mouthShrugLo'), (43, 'mouthShrugUp'),
    (36, 'mouthPressL'), (37, 'mouthPressR'),
    (40, 'mouthRollLo'), (41, 'mouthRollUp'),
]

fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(focus_aus))

cats_focused = [
    ('conv_shared_a', 'Genuine shared smile', '#E91E63'),
    ('med_smile_p1', 'Meditation "smile" (patient)', '#78909C'),
    ('med_smile_p2', 'Meditation "smile" (therapist)', '#B0BEC5'),
    ('conv_nonsmile_p1', 'Conv non-smile event', '#FFB74D'),
]

bar_h = 0.2
for i, (key, label, color) in enumerate(cats_focused):
    if key in avg:
        vals = [avg[key][au] for au, _ in focus_aus]
        ax.barh(y + i * bar_h, vals, height=bar_h, label=label,
                color=color, alpha=0.85)

labels_f = [f"AU{au:02d} {name}" for au, name in focus_aus]
ax.set_yticks(y + bar_h * 1.5)
ax.set_yticklabels(labels_f, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean AU activation')
ax.set_title('AU Profile: Genuine smiles vs noise — which AUs discriminate?')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'profile_comparison_focused.png'), dpi=120)
plt.close(fig)
print(f"Saved: profile_comparison_focused.png")

# ---- Print numerical comparison ----
print(f"\n{'='*80}")
print(f"NUMERICAL COMPARISON: Genuine shared smile vs meditation noise")
print(f"{'='*80}")
print(f"\n{'AU':>4s}  {'Name':20s}  {'Genuine':>8s}  {'Med_P1':>8s}  {'Med_P2':>8s}  {'Ratio_P1':>8s}  {'Discriminative?'}")
print(f"{'--':>4s}  {'--':20s}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'--'}")

genuine = avg.get('conv_shared_a', np.zeros(52))
med_p1 = avg.get('med_smile_p1', np.zeros(52))
med_p2 = avg.get('med_smile_p2', np.zeros(52))

for au, name in focus_aus:
    g = genuine[au]
    m1 = med_p1[au]
    m2 = med_p2[au]
    ratio = g / (m1 + 1e-6)
    disc = ""
    if ratio > 3.0 and g > 0.05:
        disc = "*** STRONG"
    elif ratio > 2.0 and g > 0.03:
        disc = "** moderate"
    elif ratio < 0.5 and m1 > 0.05:
        disc = "-- noise-enriched"
    print(f"AU{au:02d}  {name:20s}  {g:8.3f}  {m1:8.3f}  {m2:8.3f}  {ratio:8.1f}x  {disc}")

print(f"\nAll outputs in: {out_dir}")
