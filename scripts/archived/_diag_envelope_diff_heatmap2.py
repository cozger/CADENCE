"""Difference heatmap: patient minus therapist 2-8 Hz envelope over time.

For each segment, compute the full (time x AU) envelope matrix for each person,
subtract therapist from patient, and plot the residual as a heatmap.
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

keep = [i for i in range(52) if i < 9 or i > 22]
labels = [f"{i:02d} {MP_BLENDSHAPE_NAMES[i][:20]}" for i in keep]


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


SEGMENTS = ['conv_2', 'meditate_K', 'meditate_B']

fig, axes = plt.subplots(len(SEGMENTS), 1, figsize=(22, 5 * len(SEGMENTS)),
                         constrained_layout=True)

for row, seg in enumerate(SEGMENTS):
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue

    env_p1 = envelope_all_aus(p1_bl)
    env_p2 = envelope_all_aus(p2_bl)

    # Patient minus therapist
    diff = env_p1 - env_p2

    # Decimate to ~1 Hz for readability
    dec = max(1, int(FS))
    diff_dec = diff[::dec, :]
    t_axis = np.arange(diff_dec.shape[0]) / (FS / dec)

    # Show only non-eye AUs
    diff_show = diff_dec[:, keep].T

    ax = axes[row]
    vmax = np.percentile(np.abs(diff_show), 98)
    im = ax.imshow(diff_show, aspect='auto', interpolation='nearest',
                   extent=[0, t_axis[-1], len(keep) - 0.5, -0.5],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_ylabel(seg, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax, label='patient - therapist', shrink=0.8)
    print(f"  {seg}: vmax={vmax:.4f}")

axes[-1].set_xlabel('Time (s)')
fig.suptitle('2-8 Hz Envelope Difference: Patient minus Therapist (red=patient higher, blue=therapist higher)',
             fontsize=14)
fig.savefig(os.path.join(out_dir, 'envelope_diff_heatmap_p1_minus_p2.png'), dpi=120)
plt.close(fig)
print(f"\nSaved: envelope_diff_heatmap_p1_minus_p2.png")
print(f"Output in: {out_dir}")
