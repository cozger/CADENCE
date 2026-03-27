"""Diagnostic: Blink detection and blink-event overlap analysis.

Questions:
1. What do eyeBlink AUs (9, 10) look like across segments?
2. Which expression AUs co-activate with blinks? (brow, cheek contamination)
3. What fraction of detected facial events coincide with blinks?
4. Ground truth: patient eyes-closed in meditation (few/no blinks),
   therapist looking down reading (blinks present but attenuated)
5. How many events would we remove by subtracting blink-associated events?
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import (
    facial_event_catalog, _detect_facial_events, _EXPR_AUS, _EYE_AUS
)
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
FS = 30.0

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'blink_diag')
os.makedirs(out_dir, exist_ok=True)

# ---- Analysis 1: Blink characteristics per segment/person ----
print("\n" + "="*70)
print("ANALYSIS 1: Blink AU characteristics per segment")
print("="*70)

SEGMENTS = ['conv_1', 'conv_2', 'meditate_K', 'meditate_B', 'base_EO', 'base_EC']
segments_data = {}

for seg in SEGMENTS:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None: continue
    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1_bl is None: continue
    segments_data[seg] = {'p1': p1_bl, 'p2': p2_bl, 'dur': dur, 't0': t0}

    print(f"\n  {seg} ({dur:.0f}s):")
    for person, bl, role in [('P1', p1_bl, 'patient'), ('P2', p2_bl, 'therapist')]:
        blink_l = bl[:, 9]   # eyeBlinkLeft
        blink_r = bl[:, 10]  # eyeBlinkRight
        blink_avg = (blink_l + blink_r) / 2

        # Detect blink peaks
        pks, props = find_peaks(blink_avg, height=0.3, distance=int(0.3 * FS),
                                prominence=0.2)
        blink_rate = len(pks) / dur * 60  # blinks per minute

        print(f"    {person} ({role:>9s}): blink_mean={blink_avg.mean():.3f} "
              f"blink_max={blink_avg.max():.3f} "
              f"n_blinks={len(pks)} rate={blink_rate:.1f}/min "
              f"{'(EYES CLOSED)' if blink_avg.mean() > 0.5 else ''}")


# ---- Analysis 2: Co-activation of expression AUs during blinks ----
print("\n" + "="*70)
print("ANALYSIS 2: Expression AU co-activation during vs outside blinks")
print("  Using conv_1 (forward-looking, easy blinks)")
print("="*70)

seg_data = segments_data.get('conv_1')
if seg_data:
    for person, bl, role in [('P1', seg_data['p1'], 'patient'),
                              ('P2', seg_data['p2'], 'therapist')]:
        blink_avg = (bl[:, 9] + bl[:, 10]) / 2
        pks, _ = find_peaks(blink_avg, height=0.3, distance=int(0.3 * FS),
                            prominence=0.2)

        # Create blink mask: +/- 0.15s around each blink peak
        blink_mask = np.zeros(bl.shape[0], dtype=bool)
        half = int(0.15 * FS)
        for pk in pks:
            s = max(0, pk - half)
            e = min(bl.shape[0], pk + half + 1)
            blink_mask[s:e] = True

        # Compare AU velocities during vs outside blinks
        vel = np.abs(np.diff(bl, axis=0, prepend=bl[:1]))

        vel_during = vel[blink_mask].mean(axis=0) if blink_mask.sum() > 0 else np.zeros(52)
        vel_outside = vel[~blink_mask].mean(axis=0) if (~blink_mask).sum() > 0 else np.zeros(52)
        ratio = vel_during / (vel_outside + 1e-8)

        print(f"\n  {person} ({role}) - {len(pks)} blinks, "
              f"blink time = {blink_mask.mean()*100:.1f}% of segment")
        print(f"  {'AU':>4s}  {'Name':25s}  {'Vel_blink':>10s}  {'Vel_other':>10s}  {'Ratio':>8s}")
        print(f"  {'----':>4s}  {'----':25s}  {'----':>10s}  {'----':>10s}  {'----':>8s}")

        # Sort by ratio, show expression AUs only
        expr_ratios = [(i, ratio[i], vel_during[i], vel_outside[i])
                       for i in _EXPR_AUS]
        expr_ratios.sort(key=lambda x: -x[1])
        for idx, r, vb, vo in expr_ratios[:15]:
            marker = ""
            if idx in {7, 8}: marker = " <-- cheekSquint"
            elif idx in {0, 1, 2, 3, 4, 5}: marker = " <-- brow"
            elif idx in {19, 20}: marker = " <-- eyeSquint (excluded)"
            print(f"  AU{idx:02d}  {MP_BLENDSHAPE_NAMES[idx]:25s}  "
                  f"{vb:10.5f}  {vo:10.5f}  {r:8.2f}x{marker}")


# ---- Analysis 3: Event-blink overlap ----
print("\n" + "="*70)
print("ANALYSIS 3: How many detected events coincide with blinks?")
print("="*70)

for seg in ['conv_1', 'conv_2', 'meditate_K']:
    sd = segments_data.get(seg)
    if sd is None: continue

    print(f"\n  {seg}:")
    for person, bl, role in [('P1', sd['p1'], 'patient'),
                              ('P2', sd['p2'], 'therapist')]:
        # Detect blinks
        blink_avg = (bl[:, 9] + bl[:, 10]) / 2
        pks_blink, _ = find_peaks(blink_avg, height=0.3, distance=int(0.3 * FS),
                                  prominence=0.2)
        blink_times = pks_blink / FS

        # Detect facial events (with speech gating)
        events = _detect_facial_events(bl, FS, lsl_start=sd['t0'])

        # Count events within 0.2s of a blink
        n_near_blink = 0
        for ev in events:
            if len(blink_times) > 0:
                min_dist = np.min(np.abs(blink_times - ev.time))
                if min_dist < 0.2:
                    n_near_blink += 1

        pct = n_near_blink / max(len(events), 1) * 100
        print(f"    {person} ({role:>9s}): {len(events):3d} events, "
              f"{len(pks_blink):3d} blinks, "
              f"{n_near_blink:3d} events near blink ({pct:.0f}%)")


# ---- Analysis 4: Blink timeseries + event overlay for visualization ----
print("\n" + "="*70)
print("ANALYSIS 4: Blink timeseries plots")
print("="*70)

for seg in ['conv_2', 'meditate_K']:
    sd = segments_data.get(seg)
    if sd is None: continue

    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    t = np.arange(sd['p1'].shape[0]) / FS

    for row, (person, bl, role, color) in enumerate([
        ('P1', sd['p1'], 'patient', 'tab:blue'),
        ('P2', sd['p2'], 'therapist', 'tab:red')]):

        ax_blink = axes[row * 2]
        ax_brow = axes[row * 2 + 1]

        # Blink signal
        blink_avg = (bl[:, 9] + bl[:, 10]) / 2
        ax_blink.plot(t, blink_avg, color=color, linewidth=0.7, alpha=0.8)
        ax_blink.set_ylabel(f'{person} ({role})\neyeBlink', fontsize=8)
        ax_blink.set_ylim(-0.1, 1.1)
        ax_blink.axhline(0.3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

        # Brow velocity (should spike during blinks)
        brow_vel = np.abs(np.diff(bl[:, [1, 2, 3, 4, 5]], axis=0,
                                   prepend=bl[:1, [1, 2, 3, 4, 5]]))
        brow_saliency = np.sqrt((brow_vel ** 2).sum(axis=1))
        ax_brow.plot(t, brow_saliency, color=color, linewidth=0.7, alpha=0.8)
        ax_brow.set_ylabel(f'{person}\nbrow velocity', fontsize=8)

        # Mark blinks
        pks, _ = find_peaks(blink_avg, height=0.3, distance=int(0.3 * FS),
                            prominence=0.2)
        ax_blink.plot(pks / FS, blink_avg[pks], 'v', color='black',
                      markersize=4, alpha=0.7)
        for pk in pks:
            ax_brow.axvline(pk / FS, color='gray', linewidth=0.3, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Blinks (top) and brow velocity (bottom) per person',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_blink_brow.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved: {seg}_blink_brow.png")

print(f"\nAll outputs in: {out_dir}")
