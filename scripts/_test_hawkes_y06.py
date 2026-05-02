"""Validate Hawkes coupling pipeline on real y_06 data.

Loads raw blendshapes from XDF, runs NMF → Hawkes → MMHP,
reports discovered pathways and coupling episodes.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf
import glob
import time

from cadence.significance.hawkes_coupling import (
    hawkes_coupling_analysis, nmf_expression_discovery,
    detect_component_events, MP_BLENDSHAPE_NAMES,
)

FS = 30.0


def load_raw_bl(xdf_path, segment='conv_1'):
    """Load raw [0,1] blendshapes from XDF."""
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

    marker_times = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                marker_times[v[0]] = t

    t_start = marker_times[f'{segment}_start']
    t_end = marker_times[f'{segment}_stop']

    landmarks = {}
    for stream in data:
        name = stream['info']['name'][0]
        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 52 and person not in landmarks:
                landmarks[person] = (
                    np.array(stream['time_stamps']),
                    np.array(stream['time_series'], dtype=np.float32))

    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    result = {}
    for person in ['P1', 'P2']:
        ts, d = landmarks[person]
        m = (ts >= t_start) & (ts <= t_end)
        sig = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)
        result[person] = sig

    return result['P1'], result['P2'], dur


# ── Main ──────────────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print(f"XDF: {xdf_path}\n")

for segment in ['conv_1', 'conv_2']:
    print(f"{'='*70}")
    print(f"  {segment}")
    print(f"{'='*70}")

    p1, p2, dur = load_raw_bl(xdf_path, segment)
    print(f"Duration: {dur:.1f}s, P1: {p1.shape}, P2: {p2.shape}")

    t0 = time.time()
    result = hawkes_coupling_analysis(p1, p2, FS, n_components=6, seed=42)
    elapsed = time.time() - t0

    print(f"\nNMF: {result.n_components} components, "
          f"{result.explained_variance:.1%} explained")

    # Component summary
    print(f"\nComponents and event counts:")
    for i, name in enumerate(result.component_names):
        n1 = len(result.events_p1[i])
        n2 = len(result.events_p2[i])
        print(f"  {i}: {name}")
        print(f"     P1: {n1} events ({n1/dur:.3f}/s), "
              f"P2: {n2} events ({n2/dur:.3f}/s)")

    # All tested pathways, sorted by p-value
    print(f"\nTested {len(result.pathways)} pathways "
          f"({result.n_significant} significant after FDR):")
    print(f"{'Src→Tgt':<10} {'α':>6} {'β':>6} {'peak_lag':>8} "
          f"{'LLR':>7} {'p-value':>9} {'sig':>4}")
    print("-" * 60)

    for pw in sorted(result.pathways, key=lambda x: x.p_value):
        sig_mark = '***' if pw.significant else ''
        print(f"  {pw.source_component}→{pw.target_component}    "
              f"{pw.alpha:6.3f} {pw.beta:6.2f} {pw.peak_lag_s:7.2f}s "
              f"{pw.llr:7.2f} {pw.p_value:9.4f} {sig_mark}")

    # Significant pathway details
    for pw in result.pathways:
        if not pw.significant:
            continue

        print(f"\n  *** Pathway {pw.source_component}→{pw.target_component}: "
              f"{pw.source_name} → {pw.target_name}")
        print(f"      Hawkes: μ={pw.mu:.4f}, α={pw.alpha:.3f}, "
              f"β={pw.beta:.2f} (peak lag={pw.peak_lag_s:.2f}s)")
        print(f"      Coupling fraction: {pw.coupling_fraction:.1%}")
        print(f"      Triggered: {pw.n_triggered}, "
              f"Spontaneous: {pw.n_spontaneous}")
        if pw.episodes:
            print(f"      Episodes ({len(pw.episodes)}):")
            for ep in pw.episodes:
                print(f"        [{ep.start_s:.1f}s - {ep.end_s:.1f}s] "
                      f"({ep.duration_s:.1f}s) "
                      f"src={ep.n_source_events} tgt={ep.n_target_events} "
                      f"triggered={ep.n_triggered}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print()
