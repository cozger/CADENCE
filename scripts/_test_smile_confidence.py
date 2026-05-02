"""Test smile confidence scoring on the shared smile catalog.

Uses high-confidence anchors to validate that the scoring separates
genuine smiles from noise (mouthShrugLower-dominated activations).
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

FS = 30.0
EYE_AUS = set(range(9, 23))
EXPR_AUS = [i for i in range(52) if i not in EYE_AUS]
SMILE_AUS = [44, 45]


def smile_confidence(snapshot):
    """Score how confident we are this is a genuine smile.

    Returns (confidence, smile_composite, dominance, bilaterality).
    """
    smile_l = snapshot[44]
    smile_r = snapshot[45]
    smile_comp = smile_l + smile_r

    # Dominance: fraction of non-eye facial activity that is smile
    total_expr = sum(snapshot[i] for i in EXPR_AUS) + 1e-6
    dominance = smile_comp / total_expr

    # Bilaterality: symmetric smile = genuine
    max_side = max(smile_l, smile_r, 1e-6)
    bilaterality = 1.0 - abs(smile_l - smile_r) / max_side

    confidence = smile_comp * dominance * bilaterality
    return confidence, smile_comp, dominance, bilaterality


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
    return sigs['P1'], sigs['P2'], dur, t_start


def detect_facial_events(signal, fs=30.0, smooth_s=0.1, prominence=0.03,
                         min_iei_s=1.0):
    vel = np.diff(signal[:, EXPR_AUS], axis=0, prepend=signal[:1, EXPR_AUS])
    saliency = np.sqrt((vel ** 2).sum(axis=1))
    if smooth_s > 0:
        saliency = gaussian_filter1d(saliency, sigma=smooth_s * fs)
    pks, _ = find_peaks(saliency, prominence=prominence,
                        distance=int(min_iei_s * fs))
    half = int(0.25 * fs)
    T = signal.shape[0]
    snapshots = np.zeros((len(pks), 52))
    for i, pk in enumerate(pks):
        s = max(0, pk - half)
        e = min(T, pk + half + 1)
        snapshots[i] = signal[s:e].mean(axis=0)
    return pks / fs, saliency[pks], snapshots


def find_cooccurrences(ev_a, snap_a, amp_a, ev_b, snap_b, amp_b, max_lag=3.0):
    coocs = []
    used_b = set()
    for i, t_a in enumerate(ev_a):
        diffs = ev_b - t_a
        in_window = np.where(np.abs(diffs) <= max_lag)[0]
        if len(in_window) == 0:
            continue
        for j in in_window[np.argsort(np.abs(diffs[in_window]))]:
            if j in used_b:
                continue
            used_b.add(j)
            coocs.append({
                't_a': t_a, 't_b': ev_b[j], 'lag': ev_b[j] - t_a,
                'snap_a': snap_a[i], 'snap_b': snap_b[j],
                'amp_a': amp_a[i], 'amp_b': amp_b[j],
            })
            break
    return coocs


# ── Run ───────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]

GT_GENUINE = {61925.050, 61823.357}

all_smile_coocs = []

for segment in ['conv_1', 'conv_2', 'meditate_B', 'meditate_K']:
    try:
        p1, p2, dur, t_start = load_segment(xdf_path, segment)
    except Exception:
        continue

    ev_a, amp_a, snap_a = detect_facial_events(p1, FS, 0.1, 0.03, 1.0)
    ev_b, amp_b, snap_b = detect_facial_events(p2, FS, 0.1, 0.03, 1.0)
    coocs = find_cooccurrences(ev_a, snap_a, amp_a, ev_b, snap_b, amp_b, 3.0)

    for co in coocs:
        sc_a = smile_confidence(co['snap_a'])
        sc_b = smile_confidence(co['snap_b'])

        # Joint smile confidence = geometric mean
        joint_conf = np.sqrt(sc_a[0] * sc_b[0])
        smile_a = sc_a[1]
        smile_b = sc_b[1]

        # Only keep if at least one person has smile > 0.3
        if smile_a < 0.3 and smile_b < 0.3:
            continue

        lsl = t_start + co['t_a']
        is_gt = any(abs(lsl - g) < 5 for g in GT_GENUINE)

        all_smile_coocs.append({
            'segment': segment, 't_a': co['t_a'], 'lsl': lsl,
            'lag': co['lag'],
            'smile_a': smile_a, 'smile_b': smile_b,
            'conf_a': sc_a[0], 'dom_a': sc_a[2], 'bilat_a': sc_a[3],
            'conf_b': sc_b[0], 'dom_b': sc_b[2], 'bilat_b': sc_b[3],
            'joint_conf': joint_conf,
            'is_genuine': is_gt,
            'top_a': sorted([(i, co['snap_a'][i]) for i in EXPR_AUS if co['snap_a'][i] > 0.1],
                            key=lambda x: -x[1])[:3],
            'top_b': sorted([(i, co['snap_b'][i]) for i in EXPR_AUS if co['snap_b'][i] > 0.1],
                            key=lambda x: -x[1])[:3],
        })

# Sort by joint confidence
all_smile_coocs.sort(key=lambda x: -x['joint_conf'])

print(f"Total smile co-occurrences across all segments: {len(all_smile_coocs)}")
print(f"Genuine (ground truth): {sum(1 for c in all_smile_coocs if c['is_genuine'])}")

print(f"\n{'rank':>4} {'seg':>12} {'t':>6} {'lag':>5} "
      f"{'smA':>5} {'smB':>5} {'confA':>6} {'confB':>6} {'JOINT':>6} "
      f"{'domA':>5} {'domB':>5} {'bilA':>5} {'bilB':>5} {'GT':>3}")
print("-" * 110)

for rank, co in enumerate(all_smile_coocs):
    gt = '***' if co['is_genuine'] else ''
    print(f"{rank+1:4d} {co['segment']:>12} {co['t_a']:6.1f} {co['lag']:+5.1f} "
          f"{co['smile_a']:5.2f} {co['smile_b']:5.2f} "
          f"{co['conf_a']:6.3f} {co['conf_b']:6.3f} {co['joint_conf']:6.3f} "
          f"{co['dom_a']:5.2f} {co['dom_b']:5.2f} "
          f"{co['bilat_a']:5.2f} {co['bilat_b']:5.2f} {gt:>3}")

# Find threshold that separates genuine from noise
genuine_confs = [c['joint_conf'] for c in all_smile_coocs if c['is_genuine']]
noise_confs = [c['joint_conf'] for c in all_smile_coocs if not c['is_genuine']]

if genuine_confs and noise_confs:
    print(f"\nGenuine confidence range: [{min(genuine_confs):.4f}, {max(genuine_confs):.4f}]")
    print(f"Noise confidence range:   [{min(noise_confs):.4f}, {max(noise_confs):.4f}]")
    print(f"Genuine min: {min(genuine_confs):.4f}, Noise max: {max(noise_confs):.4f}")

    # How many noise events above each threshold?
    for thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        n_above = sum(1 for c in noise_confs if c >= thresh)
        n_gen = sum(1 for c in genuine_confs if c >= thresh)
        print(f"  threshold={thresh:.2f}: genuine={n_gen}/{len(genuine_confs)}, "
              f"noise_above={n_above}/{len(noise_confs)}")
