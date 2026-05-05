"""Diagnostic: What are the 'smile' events in meditation blocks?

These are almost certainly noise. Examine:
1. AU snapshots at each detected smile event
2. What drives the smile_composite above threshold
3. What the confidence components (amplitude, dominance, bilaterality) look like
4. Temporal context: what's happening around these events
5. Raw AU44/AU45 timeseries in meditation — is there sustained low-level activation?
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import facial_event_catalog, _AFFECT_AUS
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

# ---- Run affect-only detection on meditation + conversation ----
SEGMENTS = ['conv_2', 'meditate_K', 'meditate_B']

for seg in SEGMENTS:
    t0_lsl = markers.get(f'{seg}_start')
    t1_lsl = markers.get(f'{seg}_stop')
    if t0_lsl is None: continue

    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t0_lsl, t1_lsl)
    if p1_bl is None: continue

    cat = facial_event_catalog(p1_bl, p2_bl, FS, lsl_start=t0_lsl,
                                segment_name=seg)

    print(f"\n{'='*70}")
    print(f"SEGMENT: {seg} ({dur:.0f}s)")
    print(f"  P1 events={cat.n_events_p1}  P2 events={cat.n_events_p2}")
    print(f"  Shared={cat.n_shared}  Shared smiles={cat.n_shared_smiles}")
    print(f"{'='*70}")

    # ---- Examine ALL events with smile_composite > 0.15 ----
    print(f"\n  --- ALL events with smile_composite > 0.15 ---")
    for person, events, role in [('P1', cat.events_p1, 'patient'),
                                  ('P2', cat.events_p2, 'therapist')]:
        smile_events = [e for e in events if e.smile_composite > 0.15]
        print(f"\n  {person} ({role}): {len(smile_events)} events with smile>0.15 "
              f"(of {len(events)} total)")

        if not smile_events:
            continue

        print(f"  {'Time':>8s} {'SmileComp':>10s} {'Confidence':>10s} "
              f"{'AU44_smL':>8s} {'AU45_smR':>8s} {'AU7_chkL':>8s} {'AU8_chkR':>8s} "
              f"{'AU30_frL':>8s} {'AU31_frR':>8s} {'AU28_dmL':>8s} {'AU29_dmR':>8s} "
              f"{'Speech':>7s} {'Blink':>6s}  Top AUs")

        for ev in smile_events:
            snap = ev.au_snapshot
            top_str = ", ".join(f"{n[:15]}={v:.2f}" for _, n, v in ev.top_aus[:3])
            print(f"  {ev.time:8.1f} {ev.smile_composite:10.3f} {ev.smile_confidence:10.4f} "
                  f"{snap[44]:8.3f} {snap[45]:8.3f} {snap[7]:8.3f} {snap[8]:8.3f} "
                  f"{snap[30]:8.3f} {snap[31]:8.3f} {snap[28]:8.3f} {snap[29]:8.3f} "
                  f"{ev.speech_prob:7.2f} {ev.blink_prob:6.2f}  {top_str}")

    # ---- Distribution of smile_composite across ALL events ----
    all_comps_p1 = [e.smile_composite for e in cat.events_p1]
    all_comps_p2 = [e.smile_composite for e in cat.events_p2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, comps, person, role in [(axes[0], all_comps_p1, 'P1', 'patient'),
                                     (axes[1], all_comps_p2, 'P2', 'therapist')]:
        ax.hist(comps, bins=50, range=(0, 2), alpha=0.7, color='tab:blue')
        ax.axvline(0.3, color='red', linewidth=2, linestyle='--', label='threshold=0.3')
        ax.set_xlabel('smile_composite (AU44+AU45)')
        ax.set_ylabel('Count')
        ax.set_title(f'{seg} {person} ({role}): n={len(comps)}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_smile_hist.png'), dpi=120)
    plt.close(fig)
    print(f"\n  Saved: {seg}_smile_hist.png")

    # ---- Raw AU44/AU45 timeseries in this segment ----
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    t = np.arange(p1_bl.shape[0]) / FS

    for ax, au, name in [(axes[0], 44, 'mouthSmileLeft'),
                          (axes[1], 45, 'mouthSmileRight')]:
        ax.plot(t, p1_bl[:, au], color='tab:blue', linewidth=0.7, alpha=0.8,
                label='P1 (patient)')
        ax.plot(t, p2_bl[:, au], color='tab:red', linewidth=0.7, alpha=0.8,
                label='P2 (therapist)')
        ax.set_ylabel(f'AU{au}\n{name}', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.axhline(0.15, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.0)

    # Show other affect AUs
    for ax, aus, name in [(axes[2], [30, 31], 'mouthFrown L/R'),
                           (axes[3], [28, 29], 'mouthDimple L/R')]:
        for au in aus:
            ax.plot(t, p1_bl[:, au], color='tab:blue', linewidth=0.5, alpha=0.6)
            ax.plot(t, p2_bl[:, au], color='tab:red', linewidth=0.5, alpha=0.6)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(-0.05, 0.5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Raw affect AU timeseries', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_affect_raw.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved: {seg}_affect_raw.png")

    # ---- Smile composite vs confidence scatter ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for events, person, color in [(cat.events_p1, 'P1 (patient)', 'tab:blue'),
                                   (cat.events_p2, 'P2 (therapist)', 'tab:red')]:
        comps = [e.smile_composite for e in events]
        confs = [e.smile_confidence for e in events]
        ax.scatter(comps, confs, alpha=0.4, s=10, color=color, label=person)
    ax.axvline(0.3, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('smile_composite (AU44 + AU45)')
    ax.set_ylabel('smile_confidence (comp x dominance x bilaterality)')
    ax.set_title(f'{seg}: Smile composite vs confidence')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_comp_vs_conf.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved: {seg}_comp_vs_conf.png")

print(f"\nAll outputs in: {out_dir}")
