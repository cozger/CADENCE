"""Diagnostic: Bandpass [2-8 Hz] + Hilbert envelope for all 52 AUs on y_06.

Produces per-segment heatmaps showing which AUs have oscillatory speech-band
activity, plus line plots of top AUs to verify turn-taking and speech/silence
dynamics.

Ground truths to validate:
  - conv_1, conv_2: back-and-forth turn-taking between therapist & patient
  - meditate_K, meditate_B: only therapist speaks (guiding), patient silent
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d

# --- Load y_06 XDF --------------------------------------------------------
from scripts.run_session_v6 import load_xdf_session, extract_bl_segment

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
if not xdf_files:
    print("No XDF found for y_06"); sys.exit(1)

print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
print(f"Roles: P1={session['p1_role']}, P2={session['p2_role']}")
print(f"Markers: {sorted(markers.keys())}")

# --- AU names (MediaPipe 52 blendshapes) -----------------------------------
MP_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

FS = 30.0

# --- Bandpass + Hilbert envelope -------------------------------------------
def speech_band_envelope(signal_52, fs=FS, lo=2.0, hi=8.0, smooth_s=0.5):
    """Compute [lo, hi] Hz analytic envelope for all 52 AUs.

    Returns (T, 52) envelope array.
    """
    T, n_ch = signal_52.shape
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq - 0.5
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')

    envelopes = np.zeros((T, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        x = signal_52[:, ch].astype(np.float64)
        # Bandpass
        xf = sosfiltfilt(sos, x)
        # Analytic envelope (magnitude of Hilbert transform)
        env = np.abs(hilbert(xf))
        # Smooth
        if smooth_s > 0:
            env = gaussian_filter1d(env, sigma=smooth_s * fs)
        envelopes[:, ch] = env.astype(np.float32)

    return envelopes


# --- Segments to analyze ---------------------------------------------------
SEGMENTS = ['conv_1', 'conv_2', 'meditate_K', 'meditate_B']

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'speech_diag')
os.makedirs(out_dir, exist_ok=True)

# Collect per-segment, per-person mean envelopes for summary heatmap
summary = {}  # segment -> {'P1': (52,), 'P2': (52,)}

for seg in SEGMENTS:
    t_start = markers.get(f'{seg}_start')
    t_end = markers.get(f'{seg}_stop')
    if t_start is None or t_end is None:
        print(f"Skipping {seg}: no markers")
        continue

    dur = t_end - t_start
    print(f"\n{'='*60}")
    print(f"Segment: {seg}  ({dur:.0f}s)")
    print(f"{'='*60}")

    p1_bl, p2_bl, bl_dur = extract_bl_segment(session['landmarks'], t_start, t_end)
    if p1_bl is None:
        print(f"  No BL data"); continue

    p1_env = speech_band_envelope(p1_bl)
    p2_env = speech_band_envelope(p2_bl)

    # Mean envelope per AU
    p1_mean = p1_env.mean(axis=0)
    p2_mean = p2_env.mean(axis=0)
    summary[seg] = {'P1': p1_mean, 'P2': p2_mean}

    # Print top-10 AUs by mean envelope for each person
    for person, env_mean in [('P1', p1_mean), ('P2', p2_mean)]:
        role = session[f'{person.lower()}_role']
        top_idx = np.argsort(env_mean)[::-1][:10]
        print(f"\n  {person} ({role}) — Top 10 AUs by speech-band energy:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. AU{idx:02d} {MP_NAMES[idx]:25s}  env={env_mean[idx]:.4f}")

    # --- Plot 1: Heatmap of all 52 AUs over time (decimated) ---------------
    # Decimate to ~1 Hz for heatmap readability
    dec = max(1, int(FS))
    t_axis = np.arange(0, p1_env.shape[0], dec) / FS

    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    for ax, person, env_full in [(axes[0], 'P1', p1_env), (axes[1], 'P2', p2_env)]:
        role = session[f'{person.lower()}_role']
        env_dec = env_full[::dec, :]
        # Only show non-eye AUs (exclude 9-22)
        keep = [i for i in range(52) if i < 9 or i > 22]
        env_show = env_dec[:, keep].T
        labels = [f"{i:02d} {MP_NAMES[i][:20]}" for i in keep]

        im = ax.imshow(env_show, aspect='auto', interpolation='nearest',
                       extent=[0, t_axis[-1], len(keep)-0.5, -0.5],
                       cmap='hot')
        ax.set_ylabel(f'{person} ({role})')
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, label='Envelope amplitude')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Speech-band [2-8 Hz] envelope — all expression AUs', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_heatmap.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved heatmap: {seg}_heatmap.png")

    # --- Plot 2: Top speech AUs as time series (for turn-taking) -----------
    # Show jawOpen (25), mouthClose (27), mouthFunnel (32), mouthSmileL (44)
    SHOW_AUS = [25, 27, 32, 34, 35, 44, 45]

    fig, axes = plt.subplots(len(SHOW_AUS), 1, figsize=(20, 2.5 * len(SHOW_AUS)),
                             sharex=True)
    t_full = np.arange(p1_env.shape[0]) / FS

    for ax, au_idx in zip(axes, SHOW_AUS):
        role1 = session['p1_role']
        role2 = session['p2_role']
        ax.plot(t_full, p1_env[:, au_idx], alpha=0.8, linewidth=0.8,
                label=f'P1 ({role1})', color='tab:blue')
        ax.plot(t_full, p2_env[:, au_idx], alpha=0.8, linewidth=0.8,
                label=f'P2 ({role2})', color='tab:red')
        ax.set_ylabel(f'AU{au_idx:02d}\n{MP_NAMES[au_idx][:18]}', fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Speech-band envelope — key AUs (P1 blue, P2 red)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_timeseries.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved timeseries: {seg}_timeseries.png")

    # --- Plot 3: Combined speech score (sum of top AUs) --------------------
    # Sum envelope of jawOpen + mouthClose + mouthLowerDownL/R as combined
    SPEECH_CANDIDATE_AUS = [25, 27, 34, 35]
    p1_speech = p1_env[:, SPEECH_CANDIDATE_AUS].sum(axis=1)
    p2_speech = p2_env[:, SPEECH_CANDIDATE_AUS].sum(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    ax.plot(t_full, p1_speech, alpha=0.8, linewidth=1.0,
            label=f'P1 ({session["p1_role"]})', color='tab:blue')
    ax.plot(t_full, -p2_speech, alpha=0.8, linewidth=1.0,
            label=f'P2 ({session["p2_role"]}) [inverted]', color='tab:red')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speech-band energy (AU25+27+34+35)')
    ax.set_title(f'{seg}: Turn-taking visualization (P1 up, P2 down)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_turntaking.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved turn-taking: {seg}_turntaking.png")

# --- Summary heatmap across segments --------------------------------------
if summary:
    fig, axes = plt.subplots(2, len(summary), figsize=(6 * len(summary), 10),
                             sharey=True)
    if len(summary) == 1:
        axes = axes.reshape(2, 1)

    keep = [i for i in range(52) if i < 9 or i > 22]
    labels = [f"{i:02d} {MP_NAMES[i][:18]}" for i in keep]

    for col, seg in enumerate(summary):
        for row, person in enumerate(['P1', 'P2']):
            ax = axes[row, col]
            vals = summary[seg][person][keep]
            role = session[f'{person.lower()}_role']

            colors = ['tab:red' if v > np.percentile(vals, 80) else 'tab:blue' for v in vals]
            ax.barh(range(len(keep)), vals, color=colors, alpha=0.7)
            ax.set_title(f'{seg} — {person} ({role})', fontsize=10)
            if col == 0:
                ax.set_yticks(range(len(keep)))
                ax.set_yticklabels(labels, fontsize=6)
            ax.invert_yaxis()

    fig.suptitle('Mean speech-band [2-8 Hz] envelope per AU, per segment', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'summary_bar.png'), dpi=120)
    plt.close(fig)
    print(f"\nSaved summary: summary_bar.png")

print(f"\nAll outputs in: {out_dir}")
