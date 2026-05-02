"""Facial saliency event detection: velocity norm on non-eye AUs.

Detects ALL facial events per person, matches co-occurrences,
reports with AU characterization and confidence (amplitude).

Ground truth (y_06):
  GENUINE: LSL 61925.050 (conv_2 t=279s), LSL 61823.357 (conv_2 t=177s)
  NOISE:   LSL 60055.137 (meditate_B t=242s)
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

FS = 30.0

# Non-eye AUs (exclude blink, gaze, squint, wide — indices 9-22)
EYE_AUS = set(range(9, 23))  # eyeBlink*, eyeLook*, eyeSquint*, eyeWide*
EXPR_AUS = [i for i in range(52) if i not in EYE_AUS]
print(f"Expression AUs ({len(EXPR_AUS)}): {[MP_BLENDSHAPE_NAMES[i] for i in EXPR_AUS[:10]]}...")

# Semantic labels for top-AU characterization
def characterize_event(au_snapshot):
    """Label an event from its AU profile."""
    top = sorted([(i, au_snapshot[i]) for i in EXPR_AUS if au_snapshot[i] > 0.1],
                 key=lambda x: -x[1])[:5]
    return top


def load_segment(xdf_path, segment):
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)
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
    t_start = mt[f'{segment}_start']
    t_end = mt[f'{segment}_stop']
    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)
    sigs = {}
    for p in ['P1', 'P2']:
        ts, d = landmarks[p]
        m = (ts >= t_start) & (ts <= t_end)
        sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                            for c in range(52)], axis=1)
    return sigs['P1'], sigs['P2'], dur, t_start, mt


def detect_facial_events(signal, fs=30.0, smooth_s=0.1, prominence=0.03,
                         min_iei_s=1.0):
    """Detect facial events via velocity norm on non-eye AUs.

    Args:
        signal: (T, 52) raw blendshapes [0, 1].
        fs: sampling rate.
        smooth_s: light smoothing on velocity norm.
        prominence: min prominence for peak detection.
        min_iei_s: min inter-event interval.

    Returns:
        event_times: (N,) event times in seconds.
        event_amplitudes: (N,) saliency amplitude at each peak.
        event_snapshots: (N, 52) AU values at each peak.
    """
    T = signal.shape[0]

    # Velocity on non-eye AUs
    vel = np.diff(signal[:, EXPR_AUS], axis=0, prepend=signal[:1, EXPR_AUS])

    # L2 norm across expression AU channels
    saliency = np.sqrt((vel ** 2).sum(axis=1))

    # Light smooth to merge sub-frame jitter
    if smooth_s > 0:
        saliency = gaussian_filter1d(saliency, sigma=smooth_s * fs)

    # Detect peaks
    pks, props = find_peaks(saliency, prominence=prominence,
                            distance=int(min_iei_s * fs))

    event_times = pks / fs
    event_amplitudes = saliency[pks]

    # AU snapshot at each peak (use small window ±0.25s for stability)
    half = int(0.25 * fs)
    snapshots = np.zeros((len(pks), 52))
    for i, pk in enumerate(pks):
        s = max(0, pk - half)
        e = min(T, pk + half + 1)
        snapshots[i] = signal[s:e].mean(axis=0)

    return event_times, event_amplitudes, snapshots


def find_cooccurrences(events_a, snaps_a, amps_a,
                       events_b, snaps_b, amps_b,
                       max_lag_s=3.0):
    """Match co-occurring events between two people.

    For each A event, find nearest B event within ±max_lag_s.
    No greedy consumption — each event can participate in multiple matches.

    Returns list of co-occurrence dicts.
    """
    coocs = []
    used_b = set()

    for i, t_a in enumerate(events_a):
        diffs = events_b - t_a
        in_window = np.where(np.abs(diffs) <= max_lag_s)[0]
        if len(in_window) == 0:
            continue

        # Pick closest unused B
        for j in in_window[np.argsort(np.abs(diffs[in_window]))]:
            if j in used_b:
                continue
            used_b.add(j)
            t_b = events_b[j]
            lag = t_b - t_a

            # Joint confidence = geometric mean of amplitudes
            conf = np.sqrt(amps_a[i] * amps_b[j])

            # Characterize both faces
            top_a = characterize_event(snaps_a[i])
            top_b = characterize_event(snaps_b[j])

            coocs.append({
                't_a': t_a, 't_b': t_b, 'lag': lag,
                'leader': 'A' if lag > 0 else 'B',
                'amp_a': amps_a[i], 'amp_b': amps_b[j],
                'confidence': conf,
                'top_a': top_a, 'top_b': top_b,
                'smile_a': snaps_a[i, 44] + snaps_a[i, 45],
                'smile_b': snaps_b[j, 44] + snaps_b[j, 45],
            })
            break

    return coocs


# ── Run ───────────────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]

# Ground truth
GT = {
    'conv_2': [(279.0, 'GENUINE-1'), (177.3, 'GENUINE-2')],
    'meditate_B': [(242.4, 'NOISE')],
}

for segment in ['conv_2', 'meditate_B']:
    p1, p2, dur, t_start, mt = load_segment(xdf_path, segment)

    print(f"\n{'='*80}")
    print(f"  {segment} ({dur:.0f}s)")
    print(f"{'='*80}")

    # Sweep parameters
    configs = [
        (0.1, 0.02, 1.0, 3.0, "smooth=0.1, prom=0.02, iei=1.0, lag=3"),
        (0.1, 0.03, 1.0, 3.0, "smooth=0.1, prom=0.03, iei=1.0, lag=3"),
        (0.1, 0.04, 1.0, 3.0, "smooth=0.1, prom=0.04, iei=1.0, lag=3"),
        (0.1, 0.05, 1.5, 3.0, "smooth=0.1, prom=0.05, iei=1.5, lag=3"),
        (0.2, 0.03, 1.5, 3.0, "smooth=0.2, prom=0.03, iei=1.5, lag=3"),
        (0.2, 0.04, 2.0, 3.0, "smooth=0.2, prom=0.04, iei=2.0, lag=3"),
    ]

    for smooth, prom, iei, max_lag, desc in configs:
        ev_a, amp_a, snap_a = detect_facial_events(p1, FS, smooth, prom, iei)
        ev_b, amp_b, snap_b = detect_facial_events(p2, FS, smooth, prom, iei)

        coocs = find_cooccurrences(ev_a, snap_a, amp_a,
                                    ev_b, snap_b, amp_b, max_lag)

        # Check ground truth
        gt_results = {}
        for gt_t, gt_label in GT.get(segment, []):
            found = False
            for co in coocs:
                if abs(co['t_a'] - gt_t) < 4 or abs(co['t_b'] - gt_t) < 4:
                    found = True
                    gt_results[gt_label] = co
                    break
            if not found:
                gt_results[gt_label] = None

        # Summary
        g1 = gt_results.get('GENUINE-1')
        g2 = gt_results.get('GENUINE-2')
        noise = gt_results.get('NOISE')

        g1_ok = 'YES' if g1 else 'no'
        g2_ok = 'YES' if g2 else 'no'
        n_ok = 'NO' if noise is None else 'det!'

        # Count smile co-occurrences (both people smile > 0.3)
        smile_coocs = [c for c in coocs if c['smile_a'] > 0.3 and c['smile_b'] > 0.3]

        success = ' ***' if g1 and g2 and noise is None else ''
        print(f"\n  {desc}")
        print(f"    Events: P1={len(ev_a)}, P2={len(ev_b)}, "
              f"co-occ={len(coocs)}, smile_cooc={len(smile_coocs)}")
        print(f"    Ground truth: G1={g1_ok} G2={g2_ok} NOISE={n_ok}{success}")

        if g1:
            print(f"      G1: t_a={g1['t_a']:.1f} t_b={g1['t_b']:.1f} "
                  f"lag={g1['lag']:.1f}s conf={g1['confidence']:.3f} "
                  f"smile_a={g1['smile_a']:.2f} smile_b={g1['smile_b']:.2f}")
        if g2:
            print(f"      G2: t_a={g2['t_a']:.1f} t_b={g2['t_b']:.1f} "
                  f"lag={g2['lag']:.1f}s conf={g2['confidence']:.3f} "
                  f"smile_a={g2['smile_a']:.2f} smile_b={g2['smile_b']:.2f}")
        if noise:
            print(f"      NOISE: t_a={noise['t_a']:.1f} t_b={noise['t_b']:.1f} "
                  f"lag={noise['lag']:.1f}s conf={noise['confidence']:.3f} "
                  f"smile_a={noise['smile_a']:.2f} smile_b={noise['smile_b']:.2f}")

    # Show best config's full smile co-occurrence catalog
    print(f"\n  --- Best config smile co-occurrences ---")
    ev_a, amp_a, snap_a = detect_facial_events(p1, FS, 0.1, 0.03, 1.0)
    ev_b, amp_b, snap_b = detect_facial_events(p2, FS, 0.1, 0.03, 1.0)
    coocs = find_cooccurrences(ev_a, snap_a, amp_a, ev_b, snap_b, amp_b, 3.0)

    smile_coocs = [c for c in coocs if c['smile_a'] > 0.3 and c['smile_b'] > 0.3]
    print(f"  Total events: P1={len(ev_a)}, P2={len(ev_b)}")
    print(f"  Total co-occurrences: {len(coocs)}")
    print(f"  Shared smiles (both > 0.3): {len(smile_coocs)}")

    for co in smile_coocs:
        lsl_a = t_start + co['t_a']
        gt_mark = ""
        for gt_t, gt_label in GT.get(segment, []):
            if abs(co['t_a'] - gt_t) < 4 or abs(co['t_b'] - gt_t) < 4:
                gt_mark = f" <--{gt_label}"
        top_a_str = ', '.join(f"{MP_BLENDSHAPE_NAMES[i]}={v:.2f}" for i, v in co['top_a'][:3])
        top_b_str = ', '.join(f"{MP_BLENDSHAPE_NAMES[i]}={v:.2f}" for i, v in co['top_b'][:3])
        print(f"    t={co['t_a']:6.1f}s lag={co['lag']:+5.1f}s "
              f"conf={co['confidence']:.3f} "
              f"smile=({co['smile_a']:.2f},{co['smile_b']:.2f}) "
              f"LSL={lsl_a:.1f}{gt_mark}")
        print(f"      P1: {top_a_str}")
        print(f"      P2: {top_b_str}")
