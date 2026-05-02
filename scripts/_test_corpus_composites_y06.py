"""Run bl_two_stage_coupling on y_06 with corpus-NMF-discovered composites.

Compares hand-defined composites vs NMF-discovered composites.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf
import glob
import json

from cadence.significance.bl_coupling import (
    bl_two_stage_coupling, EXPRESSION_COMPOSITES,
)

FS = 30.0

# ── Load corpus NMF composites ────────────────────────────────────────

with open('results/corpus_nmf_k6.json') as f:
    nmf_results = json.load(f)

NMF_COMPOSITES = {}
for comp in nmf_results['components']:
    # Use top 2 AUs as name, suggested AUs as composite
    aus = tuple(comp['suggested_composite_aus'])
    name = comp['name'].split('+')[0]  # short name from first AU
    # Skip pure eye-gaze/blink components (not expressive)
    top_au_names = [a[1] for a in comp['top_aus'][:2]]
    is_eye_only = all('eye' in n.lower() or 'blink' in n.lower() or 'Look' in n
                       for n in top_au_names)
    if is_eye_only:
        continue
    NMF_COMPOSITES[name] = aus

print("Hand-defined composites:")
for name, aus in EXPRESSION_COMPOSITES.items():
    print(f"  {name}: {aus}")

print(f"\nNMF-discovered composites (non-eye):")
for name, aus in NMF_COMPOSITES.items():
    print(f"  {name}: {aus}")

# ── Load y_06 raw blendshapes ─────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

marker_times = {}
for stream in data:
    if stream['info']['type'][0] == 'Markers':
        for t, v in zip(stream['time_stamps'], stream['time_series']):
            marker_times[v[0]] = t

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

# ── Run on both segments ──────────────────────────────────────────────

for segment in ['conv_1', 'conv_2']:
    t_start = marker_times.get(f'{segment}_start')
    t_end = marker_times.get(f'{segment}_stop')
    if t_start is None:
        continue

    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    sigs = {}
    for p in ['P1', 'P2']:
        ts, d = landmarks[p]
        m = (ts >= t_start) & (ts <= t_end)
        sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                            for c in range(52)], axis=1)

    print(f"\n{'='*70}")
    print(f"  {segment} ({dur:.0f}s)")
    print(f"{'='*70}")

    # Run with hand-defined composites
    print(f"\n--- Hand-defined composites ---")
    r_hand = bl_two_stage_coupling(sigs['P1'], sigs['P2'], FS, seed=42)

    print(f"  Stage 1: coupling={r_hand.mask_continuous.mean():.1%}, "
          f"lag={r_hand.estimated_lag_s:.2f}s, "
          f"z_max={r_hand.z_continuous.max():.2f}")

    for name, cat in r_hand.catalogs.items():
        if cat.n_events_a == 0 and cat.n_events_b == 0:
            continue
        print(f"  {name:12s}: P1={cat.n_events_a:3d} P2={cat.n_events_b:3d} "
              f"co-occ={cat.n_cooccurrences:2d} (p={cat.session_p_value:.3f}) "
              f"mimicry={cat.n_mimicry} shared={cat.n_shared_stimulus} "
              f"coinc={cat.n_coincidence}")

    # Run with NMF composites
    print(f"\n--- NMF-discovered composites ---")
    r_nmf = bl_two_stage_coupling(sigs['P1'], sigs['P2'], FS,
                                   composites=NMF_COMPOSITES, seed=42)

    print(f"  Stage 1: coupling={r_nmf.mask_continuous.mean():.1%}, "
          f"lag={r_nmf.estimated_lag_s:.2f}s, "
          f"z_max={r_nmf.z_continuous.max():.2f}")

    for name, cat in r_nmf.catalogs.items():
        if cat.n_events_a == 0 and cat.n_events_b == 0:
            continue
        print(f"  {name:12s}: P1={cat.n_events_a:3d} P2={cat.n_events_b:3d} "
              f"co-occ={cat.n_cooccurrences:2d} (p={cat.session_p_value:.3f}) "
              f"mimicry={cat.n_mimicry} shared={cat.n_shared_stimulus} "
              f"coinc={cat.n_coincidence}")

    # Run with BOTH (hand + NMF combined)
    combined = dict(EXPRESSION_COMPOSITES)
    combined.update(NMF_COMPOSITES)
    print(f"\n--- Combined (hand + NMF) ---")
    r_both = bl_two_stage_coupling(sigs['P1'], sigs['P2'], FS,
                                    composites=combined, seed=42)

    for name, cat in r_both.catalogs.items():
        if cat.n_events_a == 0 and cat.n_events_b == 0:
            continue
        print(f"  {name:12s}: P1={cat.n_events_a:3d} P2={cat.n_events_b:3d} "
              f"co-occ={cat.n_cooccurrences:2d} (p={cat.session_p_value:.3f}) "
              f"mimicry={cat.n_mimicry} shared={cat.n_shared_stimulus} "
              f"coinc={cat.n_coincidence}")
