"""Test speech gating: compare event counts with and without speech gating."""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_coupling import facial_event_catalog

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
print(f"Roles: P1={session['p1_role']}, P2={session['p2_role']}\n")

FS = 30.0
SEGMENTS = ['conv_1', 'conv_2', 'meditate_K', 'meditate_B']

for seg in SEGMENTS:
    t_start = markers.get(f'{seg}_start')
    t_end = markers.get(f'{seg}_stop')
    if t_start is None:
        continue

    p1_bl, p2_bl, dur = extract_bl_segment(session['landmarks'], t_start, t_end)
    if p1_bl is None:
        continue

    print(f"=== {seg} ({dur:.0f}s) ===")

    # No gating (original behavior)
    t0 = time.time()
    cat_off = facial_event_catalog(p1_bl, p2_bl, FS, lsl_start=t_start,
                                    segment_name=seg,
                                    speech_gating=False, blink_gating=False)
    t_off = time.time() - t0

    # Full gating (speech + blink)
    t0 = time.time()
    cat_on = facial_event_catalog(p1_bl, p2_bl, FS, lsl_start=t_start,
                                   segment_name=seg,
                                   speech_gating=True, blink_gating=True)
    t_on = time.time() - t0

    print(f"  No gating:            P1={cat_off.n_events_p1:3d} P2={cat_off.n_events_p2:3d} "
          f"shared={cat_off.n_shared:3d} smiles={cat_off.n_shared_smiles:2d}  ({t_off:.2f}s)")
    print(f"  Speech+blink gating:  P1={cat_on.n_events_p1:3d} P2={cat_on.n_events_p2:3d} "
          f"shared={cat_on.n_shared:3d} smiles={cat_on.n_shared_smiles:2d}  ({t_on:.2f}s)")

    # Event reduction
    p1_pct = (1 - cat_on.n_events_p1 / max(cat_off.n_events_p1, 1)) * 100
    p2_pct = (1 - cat_on.n_events_p2 / max(cat_off.n_events_p2, 1)) * 100
    print(f"  Reduction: P1={p1_pct:.0f}% P2={p2_pct:.0f}%")

    # Show blink stats
    if cat_on.blink_p1 is not None:
        bp1 = cat_on.blink_p1
        bp2 = cat_on.blink_p2
        print(f"  Blink mask P1: active={(bp1>0.5).mean()*100:.1f}%")
        print(f"  Blink mask P2: active={(bp2>0.5).mean()*100:.1f}%")

    # Show speech probability stats
    if cat_on.speech_p1 is not None:
        sp1 = cat_on.speech_p1
        sp2 = cat_on.speech_p2
        print(f"  Speech prob P1: mean={sp1.mean():.3f} max={sp1.max():.3f} "
              f"active={(sp1>0.5).mean()*100:.1f}%")
        print(f"  Speech prob P2: mean={sp2.mean():.3f} max={sp2.max():.3f} "
              f"active={(sp2>0.5).mean()*100:.1f}%")

    # Check known ground truth smiles (conv_2 only)
    if seg == 'conv_2':
        gt_smiles = [61925.050, 61823.357]
        print(f"\n  Ground truth smile check (LSL times {gt_smiles}):")
        for gt in gt_smiles:
            found_off = any(abs(s.event_a.lsl_time - gt) < 2.0 or
                           abs(s.event_b.lsl_time - gt) < 2.0
                           for s in cat_off.shared_smiles)
            found_on = any(abs(s.event_a.lsl_time - gt) < 2.0 or
                          abs(s.event_b.lsl_time - gt) < 2.0
                          for s in cat_on.shared_smiles)
            print(f"    LSL {gt}: OFF={'FOUND' if found_off else 'MISS'} "
                  f"ON={'FOUND' if found_on else 'MISS'}")

    # Show shared smile details for gated version
    if cat_on.shared_smiles:
        print(f"\n  Shared smiles (gated):")
        for s in cat_on.shared_smiles[:5]:
            print(f"    LSL_a={s.event_a.lsl_time:.1f} LSL_b={s.event_b.lsl_time:.1f} "
                  f"lag={s.lag:.2f}s conf={s.joint_smile_confidence:.4f} "
                  f"speech_a={s.event_a.speech_prob:.2f} speech_b={s.event_b.speech_prob:.2f}")

    print()
