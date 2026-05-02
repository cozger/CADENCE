"""Tune Stage 2 to detect two genuine shared smiles and reject noise.

Ground truth (y_06):
  GENUINE: LSL 61925.050 (conv_2), LSL 61823.357 (conv_2)
  NOISE:   LSL 60055.137 (meditate_B or meditate_K?)
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import numpy as np, pyxdf, glob
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES
from cadence.significance.bl_coupling import bl_two_stage_coupling
from scipy.signal import find_peaks

FS = 30.0
SMILE_AUS = [44, 45]  # mouthSmileLeft + mouthSmileRight

data, _ = pyxdf.load_xdf(
    glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0],
    dejitter_timestamps=True)

mt = {v[0]: t for s in data if s['info']['type'][0] == 'Markers'
      for t, v in zip(s['time_stamps'], s['time_series'])}

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

# ── Show what's at each timestamp ─────────────────────────────────────

targets = [
    (61925.050, 'GENUINE smile 1'),
    (61823.357, 'GENUINE smile 2'),
    (60055.137, 'NOISE'),
]

print("Ground truth timestamps:")
for lsl, label in targets:
    # Find segment
    seg_name = "unknown"
    seg_offset = 0
    for seg in ['conv_1', 'conv_2', 'base_EO', 'base_EC', 'meditate_B', 'meditate_K']:
        s = mt.get(f'{seg}_start')
        e = mt.get(f'{seg}_stop')
        if s and e and s <= lsl <= e:
            seg_name = seg
            seg_offset = lsl - s
            break

    print(f"\n  {label}: LSL={lsl:.3f} -> {seg_name} t={seg_offset:.1f}s")

    for person in ['P1', 'P2']:
        ts, d = landmarks[person]
        idx = np.argmin(np.abs(ts - lsl))
        vals = d[idx, :52]
        smile_val = vals[44] + vals[45]
        active = [(MP_BLENDSHAPE_NAMES[i], float(vals[i]))
                  for i in range(52) if vals[i] > 0.15]
        active_str = ', '.join(f'{n}={v:.2f}' for n, v in
                               sorted(active, key=lambda x: -x[1])[:5])
        print(f"    {person}: smile={smile_val:.2f}  [{active_str}]")

# ── Iterate on parameters ─────────────────────────────────────────────

# Load all segments that contain our targets
segments_to_test = set()
for lsl, label in targets:
    for seg in ['conv_1', 'conv_2', 'base_EO', 'base_EC', 'meditate_B', 'meditate_K']:
        s = mt.get(f'{seg}_start')
        e = mt.get(f'{seg}_stop')
        if s and e and s <= lsl <= e:
            segments_to_test.add(seg)

print(f"\nSegments containing targets: {segments_to_test}")

# Preload segment signals
seg_data = {}
for seg in segments_to_test:
    t_start = mt[f'{seg}_start']
    t_end = mt[f'{seg}_stop']
    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)
    sigs = {}
    for p in ['P1', 'P2']:
        ts, d = landmarks[p]
        m = (ts >= t_start) & (ts <= t_end)
        sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                            for c in range(52)], axis=1)
    seg_data[seg] = (sigs['P1'], sigs['P2'], dur, t_start)


def check_detection(prominence, min_iei, lag_lo, lag_hi, pop_prior):
    """Run Stage 2 and check if ground truth events are detected."""
    results = {}

    for seg in segments_to_test:
        p1, p2, dur, t_start = seg_data[seg]
        res = bl_two_stage_coupling(
            p1, p2, FS,
            composites={'smile': SMILE_AUS},
            max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
            n_surrogates=100, target_fa=0.05,
            event_prominence=prominence,
            min_event_iei_s=min_iei,
            n_surrogates_event=500,
            lag_prior_s=(lag_lo, lag_hi),
            population_prior=pop_prior,
            seed=42)

        cat = res.catalogs['smile']

        # Check each target
        for lsl, label in targets:
            if lsl < t_start or lsl > t_start + dur:
                continue
            seg_t = lsl - t_start

            # Is there a co-occurrence within ±3s of this timestamp?
            detected = False
            det_info = ""
            for co in cat.cooccurrences:
                if abs(co.person_a_time - seg_t) < 3 or abs(co.person_b_time - seg_t) < 3:
                    detected = True
                    det_info = (f"A={co.person_a_time:.1f}s B={co.person_b_time:.1f}s "
                               f"lag={co.lag:.1f}s {co.attribution}")
                    break

            results[label] = {
                'detected': detected,
                'info': det_info,
                'n_events_a': cat.n_events_a,
                'n_events_b': cat.n_events_b,
                'n_cooc': cat.n_cooccurrences,
                'p_value': cat.session_p_value,
            }

    return results


# Parameter sweep
print(f"\n{'='*90}")
print("PARAMETER SWEEP")
print(f"{'='*90}")

configs = [
    # (prominence, min_iei, lag_lo, lag_hi, pop_prior, description)
    (0.30, 3.0, 0.5, 4.0, None, "baseline: prom=0.3, iei=3s, lag=[0.5,4]"),
    (0.20, 3.0, 0.5, 4.0, None, "lower prom=0.2"),
    (0.40, 3.0, 0.5, 4.0, None, "higher prom=0.4"),
    (0.50, 3.0, 0.5, 4.0, None, "high prom=0.5"),
    (0.30, 1.5, 0.5, 4.0, None, "shorter iei=1.5s"),
    (0.30, 2.0, 0.5, 4.0, None, "shorter iei=2.0s"),
    (0.30, 3.0, 0.3, 3.0, None, "narrow lag=[0.3,3]"),
    (0.30, 3.0, 0.5, 5.0, None, "wide lag=[0.5,5]"),
    (0.40, 2.0, 0.5, 3.0, None, "selective: prom=0.4, iei=2, lag=[0.5,3]"),
    (0.50, 2.0, 0.5, 3.0, None, "very selective: prom=0.5, iei=2, lag=[0.5,3]"),
    (0.40, 2.0, 0.3, 2.0, None, "tight: prom=0.4, iei=2, lag=[0.3,2]"),
    (0.50, 3.0, 0.3, 2.0, None, "tight+selective: prom=0.5, iei=3, lag=[0.3,2]"),
]

print(f"\n{'description':>45}  {'G1':>4} {'G2':>4} {'N':>4}  "
      f"{'P1ev':>4} {'P2ev':>4} {'cooc':>4} {'p':>7}")
print("-" * 95)

for prom, iei, lag_lo, lag_hi, pop, desc in configs:
    res = check_detection(prom, iei, lag_lo, lag_hi, pop)

    g1 = res.get('GENUINE smile 1', {})
    g2 = res.get('GENUINE smile 2', {})
    noise = res.get('NOISE', {})

    g1_mark = 'YES' if g1.get('detected') else 'no'
    g2_mark = 'YES' if g2.get('detected') else 'no'
    n_mark = 'NO' if not noise.get('detected') else 'YES!'

    # Get event counts from conv_2 (where genuine smiles are)
    n_a = g1.get('n_events_a', 0)
    n_b = g1.get('n_events_b', 0)
    n_co = g1.get('n_cooc', 0)
    pval = g1.get('p_value', 1.0)

    success = '***' if g1.get('detected') and g2.get('detected') and not noise.get('detected') else ''

    print(f"{desc:>45}  {g1_mark:>4} {g2_mark:>4} {n_mark:>4}  "
          f"{n_a:4d} {n_b:4d} {n_co:4d} {pval:7.3f} {success}")

    if g1.get('detected') and g1.get('info'):
        print(f"{'':>49}G1: {g1['info']}")
    if g2.get('detected') and g2.get('info'):
        print(f"{'':>49}G2: {g2['info']}")
    if noise.get('detected') and noise.get('info'):
        print(f"{'':>49}N:  {noise['info']}")
