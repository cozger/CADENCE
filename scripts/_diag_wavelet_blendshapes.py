"""Wavelet time-frequency analysis of blendshape signals.

1. Per-AU scalograms: CWT of key AUs for both participants (conv_2, first 60s)
2. Difference scalograms: patient-minus-therapist CWT for speech and affect AU groups
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

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
                       'results', 'v6', 'y_06', 'wavelet_diag')
os.makedirs(out_dir, exist_ok=True)

# Frequency axis: 0.3 to 12 Hz, 40 log-spaced
freqs = np.logspace(np.log10(0.3), np.log10(12.0), 40)
omega = 5.0  # Morlet wavelet parameter


def compute_cwt(signal_1d, fs=FS):
    """CWT magnitude using Morlet wavelet via FFT convolution."""
    x = signal_1d.astype(np.float64)
    x = x - np.mean(x)
    T = len(x)
    X = fft(x)
    f_fft = fftfreq(T, d=1.0 / fs)

    coeffs = np.zeros((len(freqs), T))
    for i, fc in enumerate(freqs):
        # Morlet wavelet in frequency domain: Gaussian centered at fc
        sigma_f = fc / omega
        W = np.exp(-0.5 * ((f_fft - fc) / sigma_f) ** 2)
        coeffs[i] = np.abs(ifft(X * W))
    return coeffs  # (n_freqs, T)


# ---- Load segments ----
segments_bl = {}
for seg in ['conv_2', 'meditate_K']:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue
    segments_bl[seg] = {'p1': p1_bl, 'p2': p2_bl, 'dur': dur}
    print(f"  Loaded {seg}: {dur:.0f}s")

# ====================================================================
# 1. Per-AU scalograms (conv_2, first 60s)
# ====================================================================
print("\nComputing per-AU scalograms (conv_2, first 60s)...")

KEY_AUS = [
    (25, 'jawOpen'),
    (34, 'mouthLowerDownL'),
    (44, 'mouthSmileL'),
    (30, 'mouthFrownL'),
    (4, 'browOuterUpL'),
]

sd = segments_bl['conv_2']
t_crop = int(60 * FS)  # first 60 seconds
t_axis = np.arange(t_crop) / FS

fig, axes = plt.subplots(len(KEY_AUS), 2, figsize=(20, 3 * len(KEY_AUS)),
                         sharex=True, sharey=True)

for row, (au, name) in enumerate(KEY_AUS):
    for col, (person, bl, role) in enumerate([
        ('P1', sd['p1'][:t_crop], 'patient'),
        ('P2', sd['p2'][:t_crop], 'therapist')]):

        ax = axes[row, col]
        cwt_mag = compute_cwt(bl[:, au])

        vmax = np.percentile(cwt_mag, 98)
        ax.pcolormesh(t_axis, freqs, cwt_mag, shading='auto',
                      cmap='hot', vmin=0, vmax=vmax)
        ax.set_yscale('log')
        ax.set_ylim(0.3, 12)

        if row == 0:
            ax.set_title(f'{person} ({role})', fontsize=11)
        if col == 0:
            ax.set_ylabel(f'AU{au:02d} {name}\nFreq (Hz)', fontsize=8)

        # Mark speech band
        ax.axhline(3, color='cyan', linewidth=0.5, alpha=0.5, linestyle='--')
        ax.axhline(6, color='cyan', linewidth=0.5, alpha=0.5, linestyle='--')

axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 1].set_xlabel('Time (s)')
fig.suptitle('conv_2 (first 60s): Per-AU CWT scalograms — cyan lines mark 3-6 Hz speech band',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'per_au_scalograms_conv2.png'), dpi=120)
plt.close(fig)
print("  Saved: per_au_scalograms_conv2.png")

# ====================================================================
# 2. Difference scalograms (full segments)
# ====================================================================
print("\nComputing difference scalograms...")

SPEECH_AUS = [34, 35]
AFFECT_AUS = [44, 45, 30, 31]

for seg in ['conv_2', 'meditate_K']:
    sd = segments_bl[seg]
    T = min(sd['p1'].shape[0], sd['p2'].shape[0])
    t_axis_full = np.arange(T) / FS

    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)

    for row, (aus, group_name) in enumerate([
        (SPEECH_AUS, 'Speech (AU34+35)'),
        (AFFECT_AUS, 'Affect (AU44+45+30+31)')]):

        # Sum CWT across AU group for each person
        cwt_p1 = np.zeros((len(freqs), T))
        cwt_p2 = np.zeros((len(freqs), T))
        for au in aus:
            cwt_p1 += compute_cwt(sd['p1'][:T, au])
            cwt_p2 += compute_cwt(sd['p2'][:T, au])

        diff = cwt_p1 - cwt_p2

        ax = axes[row]
        vmax = np.percentile(np.abs(diff), 97)
        ax.pcolormesh(t_axis_full, freqs, diff, shading='auto',
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_yscale('log')
        ax.set_ylim(0.3, 12)
        ax.set_ylabel(f'{group_name}\nFreq (Hz)', fontsize=9)

        ax.axhline(3, color='black', linewidth=0.5, alpha=0.4, linestyle='--')
        ax.axhline(6, color='black', linewidth=0.5, alpha=0.4, linestyle='--')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Patient minus Therapist CWT (red=patient, blue=therapist)\n'
                 f'Dashed lines: 3-6 Hz speech band', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'diff_scalogram_{seg}.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved: diff_scalogram_{seg}.png")

print(f"\nAll outputs in: {out_dir}")
