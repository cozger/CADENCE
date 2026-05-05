"""NMF expression discovery: learn shared facial expression vocabulary from raw AUs.

Joint NMF across both participants → shared expression dictionary.
Each component = an empirically-discovered expression type.
Then detect events in each person's component activations.

This replaces hand-defined composites (smile, brow, frown, speech) with
data-driven expression types that may capture dyad-specific patterns.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
from sklearn.decomposition import NMF
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pyxdf

# MediaPipe blendshape names (52 coefficients, index 0 = _neutral)
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

FS = 30.0  # resample rate


def load_raw_blendshapes_from_xdf(xdf_path, segment='conv_1'):
    """Load raw [0,1] blendshapes from XDF, segmented by markers."""
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

    # Find markers
    marker_times = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                marker_times[v[0]] = t

    t_start = marker_times.get(f'{segment}_start')
    t_end = marker_times.get(f'{segment}_stop')
    if t_start is None or t_end is None:
        raise ValueError(f"Segment {segment} not found. Available: {list(marker_times.keys())}")

    # Find landmark streams
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

    if 'P1' not in landmarks or 'P2' not in landmarks:
        raise ValueError("Missing landmark streams")

    # Extract + resample to grid
    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    result = {}
    for person in ['P1', 'P2']:
        ts, d = landmarks[person]
        m = (ts >= t_start) & (ts <= t_end)
        ts_local = ts[m] - t_start
        d_local = d[m, :52]  # raw blendshapes [0, 1]

        sig = np.stack([np.interp(t_grid, ts_local, d_local[:, c])
                        for c in range(52)], axis=1)
        result[person] = sig

    return result['P1'], result['P2'], dur


def nmf_expression_discovery(p1_raw, p2_raw, n_components=6, seed=42):
    """Joint NMF to discover shared expression types.

    Args:
        p1_raw, p2_raw: (T, 52) raw blendshapes in [0, 1].
        n_components: number of expression types to discover.

    Returns:
        H: (k, 52) shared expression dictionary (AU loadings per component).
        W_p1: (T, k) P1 component activations.
        W_p2: (T, k) P2 component activations.
        component_names: list of auto-generated names from top AUs.
    """
    T = min(p1_raw.shape[0], p2_raw.shape[0])
    p1 = np.maximum(p1_raw[:T], 0)  # ensure non-negative
    p2 = np.maximum(p2_raw[:T], 0)

    X_joint = np.vstack([p1, p2])  # (2T, 52)

    nmf = NMF(n_components=n_components, init='nndsvda',
              max_iter=500, random_state=seed)
    W = nmf.fit_transform(X_joint)
    H = nmf.components_  # (k, 52)

    W_p1 = W[:T]
    W_p2 = W[T:]

    # Auto-name each component from top 3 AUs
    component_names = []
    for i in range(n_components):
        top3 = np.argsort(H[i])[::-1][:3]
        name = '+'.join(MP_NAMES[j] for j in top3)
        component_names.append(name)

    # Reconstruction quality
    recon_err = nmf.reconstruction_err_
    total_norm = np.linalg.norm(X_joint, 'fro')
    explained = 1 - (recon_err / total_norm)

    return H, W_p1, W_p2, component_names, explained


def detect_component_events(W, fs=30.0, min_iei_s=2.0, prominence_pct=75):
    """Detect activation events in each NMF component's timecourse.

    Uses adaptive prominence: IQR of the component activation.

    Args:
        W: (T, k) component activations.
        fs: sampling rate.
        min_iei_s: minimum inter-event interval.

    Returns:
        events: list of (k,) arrays of event times in seconds.
        prominences: list of (k,) prominence thresholds used.
    """
    T, k = W.shape
    events = []
    proms_used = []

    for comp in range(k):
        w = gaussian_filter1d(W[:, comp], sigma=0.3 * fs)  # light smooth
        iqr = np.percentile(w, 75) - np.percentile(w, 25)
        prom = max(iqr * 0.5, 0.005)  # at least 0.005

        pks, props = find_peaks(w, prominence=prom,
                                distance=int(min_iei_s * fs))
        events.append(pks / fs)
        proms_used.append(prom)

    return events, proms_used


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob
    import os

    # Find y_06 XDF
    search_paths = [
        'C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf',
        'C:/Users/optilab/desktop/MCCT/data/*y_06*.xdf',
        'C:/Users/optilab/desktop/MCCT/*y_06*.xdf',
    ]
    xdf_files = []
    for p in search_paths:
        xdf_files = glob.glob(p)
        if xdf_files:
            break
    if not xdf_files:
        print("No y_06 XDF found")
        sys.exit(1)

    xdf_path = xdf_files[0]
    print(f"Loading: {os.path.basename(xdf_path)}")

    for segment in ['conv_1', 'conv_2']:
        print(f"\n{'='*70}")
        print(f"Segment: {segment}")
        print(f"{'='*70}")

        try:
            p1_raw, p2_raw, dur = load_raw_blendshapes_from_xdf(xdf_path, segment)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        print(f"Duration: {dur:.1f}s, P1: {p1_raw.shape}, P2: {p2_raw.shape}")
        print(f"P1 range: [{p1_raw.min():.3f}, {p1_raw.max():.3f}]")
        print(f"P2 range: [{p2_raw.min():.3f}, {p2_raw.max():.3f}]")

        # Test different k values
        print(f"\nReconstruction quality vs k:")
        for k in [3, 4, 5, 6, 7, 8]:
            _, _, _, _, expl = nmf_expression_discovery(p1_raw, p2_raw, k)
            print(f"  k={k}: {expl:.1%} variance explained")

        # Detailed k=6 analysis
        k = 6
        H, W_p1, W_p2, comp_names, expl = nmf_expression_discovery(
            p1_raw, p2_raw, k)

        events_p1, proms_p1 = detect_component_events(W_p1)
        events_p2, proms_p2 = detect_component_events(W_p2)

        print(f"\nNMF Components (k={k}, {expl:.1%} explained):")
        print(f"-" * 70)

        for i in range(k):
            top5_idx = np.argsort(H[i])[::-1][:5]
            top5_str = ', '.join(f'{MP_NAMES[j]}={H[i,j]:.3f}' for j in top5_idx)

            n1 = len(events_p1[i])
            n2 = len(events_p2[i])
            rate1 = n1 / dur if dur > 0 else 0
            rate2 = n2 / dur if dur > 0 else 0

            print(f"\n  Component {i}: '{comp_names[i]}'")
            print(f"    Top AUs: {top5_str}")
            print(f"    P1: {n1} events ({rate1:.3f}/s), "
                  f"mean act={W_p1[:,i].mean():.4f}, max={W_p1[:,i].max():.3f}")
            print(f"    P2: {n2} events ({rate2:.3f}/s), "
                  f"mean act={W_p2[:,i].mean():.4f}, max={W_p2[:,i].max():.3f}")

            # Quick co-occurrence check (±3s window)
            if n1 > 0 and n2 > 0:
                cooc = 0
                for t1 in events_p1[i]:
                    if np.any(np.abs(events_p2[i] - t1) < 3.0):
                        cooc += 1
                cooc_rate = cooc / max(n1, 1)
                # Expected under independence
                expected = 1 - np.exp(-rate2 * 6.0)  # Poisson in ±3s
                print(f"    Co-occurrences (±3s): {cooc}/{n1} "
                      f"({cooc_rate:.1%} vs {expected:.1%} expected)")
