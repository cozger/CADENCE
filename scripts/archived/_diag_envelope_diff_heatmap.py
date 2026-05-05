"""Difference heatmap: conv_2 vs meditation 2-8 Hz AU envelopes.

Shows what's present in conversation that's absent in meditation.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
FS = 30.0

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'speech_diag')
os.makedirs(out_dir, exist_ok=True)


def envelope_all_aus(bl, fs=30.0, lo=2.0, hi=8.0, smooth_s=0.5):
    T, n_ch = bl.shape
    nyq = fs / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype='band', output='sos')
    envs = np.zeros((T, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        x = bl[:, ch].astype(np.float64)
        xf = sosfiltfilt(sos, x)
        env = np.abs(hilbert(xf))
        if smooth_s > 0:
            env = gaussian_filter1d(env, sigma=smooth_s * fs)
        envs[:, ch] = env.astype(np.float32)
    return envs


# Non-eye AUs only
keep = [i for i in range(52) if i < 9 or i > 22]
labels = [f"{i:02d} {MP_BLENDSHAPE_NAMES[i][:20]}" for i in keep]

# Compute mean envelope per AU for each segment/person
seg_means = {}
for seg in ['conv_2', 'meditate_K', 'meditate_B']:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue
    for person, bl, role in [('P1', p1_bl, 'patient'), ('P2', p2_bl, 'therapist')]:
        envs = envelope_all_aus(bl)
        seg_means[(seg, person)] = envs.mean(axis=0)
    print(f"  Computed {seg}")

# Average the two meditation segments for a cleaner reference
med_p1 = (seg_means[('meditate_K', 'P1')] + seg_means[('meditate_B', 'P1')]) / 2
med_p2 = (seg_means[('meditate_K', 'P2')] + seg_means[('meditate_B', 'P2')]) / 2

conv_p1 = seg_means[('conv_2', 'P1')]
conv_p2 = seg_means[('conv_2', 'P2')]

diff_p1 = conv_p1 - med_p1
diff_p2 = conv_p2 - med_p2

# ---- Plot: Difference bar chart (non-eye AUs) ----
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

for ax, diff, conv, med, person, role in [
    (axes[0], diff_p1, conv_p1, med_p1, 'P1', 'patient'),
    (axes[1], diff_p2, conv_p2, med_p2, 'P2', 'therapist')]:

    vals_keep = diff[keep]
    conv_keep = conv[keep]
    med_keep = med[keep]

    y = np.arange(len(keep))
    colors = ['#E91E63' if v > 0 else '#2196F3' for v in vals_keep]

    ax.barh(y, vals_keep, color=colors, alpha=0.7)
    ax.set_title(f'{person} ({role}): conv_2 - meditation', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Envelope difference (conv - med)')
    ax.grid(axis='x', alpha=0.3)

axes[0].set_yticks(range(len(keep)))
axes[0].set_yticklabels(labels, fontsize=7)
axes[0].invert_yaxis()

fig.suptitle('2-8 Hz Envelope: conv_2 minus meditation (red=higher in conv, blue=higher in med)',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'envelope_diff_conv_med.png'), dpi=120)
plt.close(fig)
print(f"Saved: envelope_diff_conv_med.png")

# ---- Plot: Side-by-side absolute values ----
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

for ax, conv_v, med_v, person, role in [
    (axes[0], conv_p1, med_p1, 'P1', 'patient'),
    (axes[1], conv_p2, med_p2, 'P2', 'therapist')]:

    y = np.arange(len(keep))
    ax.barh(y - 0.15, conv_v[keep], height=0.3, color='#E91E63', alpha=0.7, label='conv_2')
    ax.barh(y + 0.15, med_v[keep], height=0.3, color='#78909C', alpha=0.7, label='meditation')
    ax.set_title(f'{person} ({role})', fontsize=11)
    ax.set_xlabel('Mean 2-8 Hz envelope')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

axes[0].set_yticks(range(len(keep)))
axes[0].set_yticklabels(labels, fontsize=7)
axes[0].invert_yaxis()

fig.suptitle('2-8 Hz Envelope: conv_2 vs meditation (absolute)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'envelope_abs_conv_med.png'), dpi=120)
plt.close(fig)
print(f"Saved: envelope_abs_conv_med.png")

# ---- Print numerical top differences ----
print(f"\nTop AUs enriched in conv_2 vs meditation:")
for person, diff in [('P1 (patient)', diff_p1), ('P2 (therapist)', diff_p2)]:
    ranked = [(i, diff[i]) for i in keep]
    ranked.sort(key=lambda x: -x[1])
    print(f"\n  {person}:")
    for i, (au, d) in enumerate(ranked[:10]):
        print(f"    {i+1:2d}. AU{au:02d} {MP_BLENDSHAPE_NAMES[au]:25s}  diff={d:+.5f}")

print(f"\nAll outputs in: {out_dir}")
