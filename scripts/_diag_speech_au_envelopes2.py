"""Diagnostic round 2: Refined speech detection from blendshapes.

Key findings from round 1:
  - jawOpen (AU25) is very weak in 2-8 Hz band at 30 Hz sampling
  - mouthLowerDown L/R (34, 35) show clear episodic speech bursts
  - mouthShrugLower (42) is high everywhere (possibly noise, not speech-specific)
  - Eye AUs dominate (blinks/saccades at 3-5 Hz) — must exclude

This round:
  1. Print ALL 52 AU envelope values (not just top 10) for speech band
  2. Try multiple speech composites and compare turn-taking quality
  3. Test broader frequency band (1-10 Hz) since jaw may be slower
  4. Compare speech vs non-speech periods using meditation ground truth
  5. Compute discriminability: therapist/meditation should have high ratio
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

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']

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
EYE_AUS = set(range(9, 23))
SMILE_AUS = {7, 8, 44, 45}  # cheekSquint + mouthSmile

# AUs that are plausible speech indicators (non-eye, non-smile mouth/jaw)
SPEECH_CANDIDATE_AUS = [23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 46, 47, 48, 49]

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'speech_diag')
os.makedirs(out_dir, exist_ok=True)


def bandpass_envelope(signal_1d, fs, lo, hi, smooth_s=0.5):
    """Bandpass + Hilbert envelope for a single channel."""
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq - 0.5
    if lo >= hi:
        lo = 0.5
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    xf = sosfiltfilt(sos, signal_1d.astype(np.float64))
    env = np.abs(hilbert(xf))
    if smooth_s > 0:
        env = gaussian_filter1d(env, sigma=smooth_s * fs)
    return env


def multi_band_envelopes(signal_52, fs=FS, smooth_s=0.5):
    """Compute envelopes in multiple bands for all 52 AUs."""
    bands = {
        'narrow_2_8': (2.0, 8.0),
        'broad_1_10': (1.0, 10.0),
        'low_1_4':    (1.0, 4.0),
        'mid_3_7':    (3.0, 7.0),
    }
    result = {}
    T = signal_52.shape[0]
    for bname, (lo, hi) in bands.items():
        envs = np.zeros((T, 52), dtype=np.float32)
        for ch in range(52):
            envs[:, ch] = bandpass_envelope(signal_52[:, ch], fs, lo, hi, smooth_s)
        result[bname] = envs
    return result


# --- Load segments -----------------------------------------------------
segments_data = {}
for seg in ['conv_1', 'conv_2', 'meditate_K', 'meditate_B']:
    t_start = markers.get(f'{seg}_start')
    t_end = markers.get(f'{seg}_stop')
    if t_start is None:
        continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t_start, t_end)
    if p1_bl is None:
        continue
    segments_data[seg] = {'p1': p1_bl, 'p2': p2_bl, 'dur': dur}
    print(f"Loaded {seg}: {dur:.0f}s, P1 shape={p1_bl.shape}")


# --- Analysis 1: Full AU ranking (non-eye) for speech band ------------
print("\n" + "="*70)
print("ANALYSIS 1: Full non-eye AU ranking by speech-band [2-8 Hz] envelope")
print("="*70)

for seg in ['conv_1', 'meditate_K']:
    sd = segments_data[seg]
    for person, bl in [('P1_patient', sd['p1']), ('P2_therapist', sd['p2'])]:
        envs = np.zeros((bl.shape[0], 52), dtype=np.float32)
        for ch in range(52):
            envs[:, ch] = bandpass_envelope(bl[:, ch], FS, 2.0, 8.0)
        means = envs.mean(axis=0)

        print(f"\n  {seg} / {person} — ALL non-eye AUs ranked:")
        non_eye = [(i, means[i]) for i in range(52) if i not in EYE_AUS]
        non_eye.sort(key=lambda x: -x[1])
        for rank, (idx, val) in enumerate(non_eye):
            marker = ""
            if idx in SMILE_AUS:
                marker = " <-- SMILE"
            elif idx in set(SPEECH_CANDIDATE_AUS):
                marker = " <-- speech-candidate"
            print(f"    {rank+1:2d}. AU{idx:02d} {MP_NAMES[idx]:25s}  env={val:.5f}{marker}")


# --- Analysis 2: Discriminability — conv vs meditation for each AU ----
print("\n" + "="*70)
print("ANALYSIS 2: Discriminability — conv_1(therapist) vs meditate_K(patient)")
print("  Ratio of therapist-in-conv to patient-in-meditation speech energy")
print("  Higher ratio = better speech indicator")
print("="*70)

conv = segments_data['conv_1']
med = segments_data['meditate_K']

# Therapist in conversation (should have speech)
t_conv = np.zeros((conv['p2'].shape[0], 52), dtype=np.float32)
for ch in range(52):
    t_conv[:, ch] = bandpass_envelope(conv['p2'][:, ch], FS, 2.0, 8.0)

# Patient in meditation (should have NO speech)
p_med = np.zeros((med['p1'].shape[0], 52), dtype=np.float32)
for ch in range(52):
    p_med[:, ch] = bandpass_envelope(med['p1'][:, ch], FS, 2.0, 8.0)

t_mean = t_conv.mean(axis=0)
p_mean = p_med.mean(axis=0)
ratio = t_mean / (p_mean + 1e-8)

non_eye = [(i, ratio[i], t_mean[i], p_mean[i]) for i in range(52) if i not in EYE_AUS]
non_eye.sort(key=lambda x: -x[1])

print(f"\n  {'Rank':>4s}  {'AU':>4s}  {'Name':25s}  {'Ratio':>8s}  {'Conv(T)':>10s}  {'Med(P)':>10s}  {'Note'}")
print(f"  {'-'*4}  {'-'*4}  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
for rank, (idx, r, tc, pm) in enumerate(non_eye):
    note = ""
    if idx in SMILE_AUS:
        note = "SMILE"
    elif idx in set(SPEECH_CANDIDATE_AUS):
        note = "speech-cand"
    print(f"  {rank+1:4d}  AU{idx:02d}  {MP_NAMES[idx]:25s}  {r:8.2f}  {tc:10.5f}  {pm:10.5f}  {note}")


# --- Analysis 3: Test multiple speech composites ---------------------
print("\n" + "="*70)
print("ANALYSIS 3: Speech composite candidates — turn-taking quality")
print("="*70)

composites = {
    'jawOpen_only':       [25],
    'lowerDown_LR':       [34, 35],
    'press_LR':           [36, 37],
    'shrug_lower':        [42],
    'jaw+lower+press':    [25, 34, 35, 36, 37],
    'lower+press':        [34, 35, 36, 37],
    'lower+press+shrug':  [34, 35, 36, 37, 42],
    'all_speech_cands':   SPEECH_CANDIDATE_AUS,
    'top_discriminative': [],  # will fill after analysis 2
}

# Pick top-5 discriminative non-eye, non-smile AUs
top_disc = [idx for idx, _, _, _ in non_eye if idx not in SMILE_AUS][:5]
composites['top_discriminative'] = top_disc
print(f"  Top-5 discriminative AUs: {[f'AU{i:02d} {MP_NAMES[i]}' for i in top_disc]}")

# For each composite, compute speech signal for conv_2 and meditate_K
# Show turn-taking plots
fig, axes = plt.subplots(len(composites), 2, figsize=(24, 3 * len(composites)),
                         gridspec_kw={'width_ratios': [1, 1.5]})

for row, (cname, aus) in enumerate(composites.items()):
    if not aus:
        continue

    for col, seg in enumerate(['conv_2', 'meditate_K']):
        sd = segments_data[seg]
        ax = axes[row, col]
        t_full = np.arange(sd['p1'].shape[0]) / FS

        for person, bl, color, label in [
            ('P1', sd['p1'], 'tab:blue', f'P1 (patient)'),
            ('P2', sd['p2'], 'tab:red', f'P2 (therapist)')]:

            env_sum = np.zeros(bl.shape[0])
            for au in aus:
                env_sum += bandpass_envelope(bl[:, au], FS, 2.0, 8.0)

            sign = 1.0 if person == 'P1' else -1.0
            ax.plot(t_full, sign * env_sum, alpha=0.7, linewidth=0.7,
                    label=label, color=color)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(f'{seg}', fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{cname}\n{[f"AU{a}" for a in aus[:4]]}',
                          fontsize=7, rotation=0, ha='right', va='center',
                          labelpad=80)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)

axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 1].set_xlabel('Time (s)')
fig.suptitle('Speech composite comparison: conv_2 (turn-taking) vs meditate_K (therapist-only)',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'composite_comparison.png'), dpi=120)
plt.close(fig)
print(f"  Saved: composite_comparison.png")


# --- Analysis 4: Raw AU timeseries (not enveloped) during speech ------
# Show raw blendshape values to understand what jawOpen actually looks like
print("\n" + "="*70)
print("ANALYSIS 4: Raw AU values during conv_2 (first 30s) — speech texture")
print("="*70)

sd = segments_data['conv_2']
t30 = int(30 * FS)  # first 30 seconds
t_axis = np.arange(t30) / FS

DETAIL_AUS = [25, 27, 34, 35, 36, 37, 42, 44, 45]
fig, axes = plt.subplots(len(DETAIL_AUS), 1, figsize=(20, 2.5 * len(DETAIL_AUS)),
                         sharex=True)

for ax, au in zip(axes, DETAIL_AUS):
    ax.plot(t_axis, sd['p1'][:t30, au], alpha=0.8, linewidth=0.8,
            label='P1 (patient)', color='tab:blue')
    ax.plot(t_axis, sd['p2'][:t30, au], alpha=0.8, linewidth=0.8,
            label='P2 (therapist)', color='tab:red')
    ax.set_ylabel(f'AU{au:02d}\n{MP_NAMES[au][:18]}', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

axes[-1].set_xlabel('Time (s)')
fig.suptitle('conv_2: Raw AU values (first 30s) — looking for speech oscillation texture',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'raw_au_30s.png'), dpi=120)
plt.close(fig)
print(f"  Saved: raw_au_30s.png")


# --- Analysis 5: Correlation between AU composites for P2 in conv -----
# Which AUs co-occur during therapist speech?
print("\n" + "="*70)
print("ANALYSIS 5: AU co-activation during therapist speech (conv_1)")
print("="*70)

conv1 = segments_data['conv_1']
therapist_bl = conv1['p2']  # P2 = therapist

# Compute speech-band envelopes for all non-eye AUs
non_eye_idx = [i for i in range(52) if i not in EYE_AUS]
envs = np.zeros((therapist_bl.shape[0], len(non_eye_idx)))
for j, ch in enumerate(non_eye_idx):
    envs[:, j] = bandpass_envelope(therapist_bl[:, ch], FS, 2.0, 8.0)

# Correlation matrix
corr = np.corrcoef(envs.T)
labels = [f"AU{i:02d} {MP_NAMES[i][:15]}" for i in non_eye_idx]

fig, ax = plt.subplots(1, 1, figsize=(16, 14))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6)
plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Therapist (P2) conv_1: Correlation of speech-band envelopes across AUs')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'au_correlation.png'), dpi=120)
plt.close(fig)
print(f"  Saved: au_correlation.png")

# Print strongest correlated AU pairs for top speech candidates
print(f"\n  Strongest correlations with jawOpen (AU25):")
jaw_idx = non_eye_idx.index(25)
jaw_corrs = [(non_eye_idx[j], corr[jaw_idx, j]) for j in range(len(non_eye_idx))
             if j != jaw_idx and non_eye_idx[j] not in EYE_AUS]
jaw_corrs.sort(key=lambda x: -abs(x[1]))
for idx, c in jaw_corrs[:10]:
    print(f"    AU{idx:02d} {MP_NAMES[idx]:25s}  r={c:+.3f}")

print(f"\n  Strongest correlations with mouthLowerDownLeft (AU34):")
ld_idx = non_eye_idx.index(34)
ld_corrs = [(non_eye_idx[j], corr[ld_idx, j]) for j in range(len(non_eye_idx))
            if j != ld_idx and non_eye_idx[j] not in EYE_AUS]
ld_corrs.sort(key=lambda x: -abs(x[1]))
for idx, c in ld_corrs[:10]:
    print(f"    AU{idx:02d} {MP_NAMES[idx]:25s}  r={c:+.3f}")

print(f"\nAll outputs in: {out_dir}")
