"""Extract real expression templates from y_06 for semi-synthetic BL injection.

Extracts:
  - smile_template.npy: AU44+45 smile onset (6s @ 30 Hz)
  - duchenne_template.npy: AU44+45+7+8 Duchenne smile (6s @ 30 Hz)
  - frown_template.npy: AU30+31 frown onset (10s @ 30 Hz)

Templates are delta waveforms (event - baseline) aligned to onset.

Usage:
    python scripts/_extract_bl_templates.py
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import find_peaks

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
OUT_DIR = 'results/v6/y_06/coupling_patterns'
FS_BL = 30.0


def extract_event_template(bl_data, target_aus, fs=30.0, window_s=6.0,
                           onset_s=2.0, prominence=0.3, min_distance_s=3.0):
    """Extract averaged delta template from detected events.

    Args:
        bl_data: (T, 52) blendshape array.
        target_aus: list of AU indices to detect/extract.
        fs: sampling rate.
        window_s: total template window in seconds.
        onset_s: onset position within window.
        prominence: peak detection prominence (raw AU units).
        min_distance_s: minimum inter-event interval.

    Returns:
        template: (window_samples, 52) average delta waveform.
        n_events: number of detected events.
    """
    win_samp = int(window_s * fs)
    onset_samp = int(onset_s * fs)
    min_dist = int(min_distance_s * fs)
    T, C = bl_data.shape

    # Composite signal: sum of target AUs
    composite = np.zeros(T, dtype=np.float64)
    for au in target_aus:
        if au < C:
            composite += bl_data[:, au].astype(np.float64)

    # Detect peaks
    peaks, props = find_peaks(composite, prominence=prominence,
                              distance=min_dist)

    if len(peaks) < 3:
        print(f"  Only {len(peaks)} events detected (need >= 3)")
        return None, len(peaks)

    # Extract windows aligned to onset
    deltas = []
    for pk in peaks:
        start = pk - onset_samp
        end = start + win_samp
        if start < 0 or end > T:
            continue

        window = bl_data[start:end].astype(np.float64).copy()
        # Baseline: mean of pre-onset period
        baseline = window[:max(1, onset_samp // 2)].mean(axis=0)
        delta = window - baseline[None, :]
        deltas.append(delta)

    if len(deltas) < 3:
        return None, len(deltas)

    template = np.mean(deltas, axis=0).astype(np.float32)
    return template, len(deltas)


def main():
    print("=" * 60)
    print("  Extracting BL expression templates from y_06")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load y_06 session
    xdf_files = glob.glob(os.path.join(RAW_DIR, 'y_06*.xdf'))
    if not xdf_files:
        xdf_files = glob.glob(os.path.join(RAW_DIR, 'Y_06*.xdf'))
    if not xdf_files:
        print("ERROR: No y_06 XDF file found")
        return

    print(f"\nLoading {xdf_files[0]}...")
    session = load_xdf_session(xdf_files[0])
    markers = session['markers']
    landmarks = session['landmarks']

    # Extract conversation segments (richest in expressions)
    conv_segs = []
    for seg in ['conv_1', 'conv_2']:
        t_start = markers.get(f'{seg}_start')
        t_end = markers.get(f'{seg}_stop')
        if t_start is not None and t_end is not None:
            conv_segs.append((seg, t_start, t_end))
    print(f"  Conversation segments: {[s[0] for s in conv_segs]}")

    # Concatenate BL data from all conversation segments
    all_bl = []
    for seg_name, t0, t1 in conv_segs:
        p1_bl, p2_bl, dur = extract_bl_segment(landmarks, t0, t1)
        if p1_bl is not None:
            # Use both P1 and P2 data for template extraction
            all_bl.append(p1_bl)
            all_bl.append(p2_bl)
            print(f"  {seg_name}: {dur:.0f}s, P1={p1_bl.shape}, P2={p2_bl.shape}")

    if not all_bl:
        print("ERROR: No BL data extracted")
        return

    bl_concat = np.concatenate(all_bl, axis=0)
    print(f"\n  Total BL data: {bl_concat.shape[0] / FS_BL:.0f}s "
          f"({bl_concat.shape[0]} samples, {bl_concat.shape[1]} channels)")

    # --- Smile template (AU44+45) ---
    print("\n[1/3] Smile template (AU44, AU45)...")
    smile_tpl, n_smile = extract_event_template(
        bl_concat, [44, 45], fs=FS_BL, window_s=6.0, onset_s=2.0,
        prominence=0.2, min_distance_s=3.0)
    if smile_tpl is not None:
        path = os.path.join(OUT_DIR, 'smile_template.npy')
        np.save(path, smile_tpl)
        peak_amp = smile_tpl[:, [44, 45]].max()
        print(f"  {n_smile} events, peak AU amplitude={peak_amp:.3f}")
        print(f"  Saved: {path}")
    else:
        print(f"  FAILED: too few events")

    # --- Duchenne smile template (AU44+45+7+8) ---
    print("\n[2/3] Duchenne smile template (AU44, 45, 7, 8)...")
    duchenne_tpl, n_duch = extract_event_template(
        bl_concat, [44, 45, 7, 8], fs=FS_BL, window_s=6.0, onset_s=2.0,
        prominence=0.25, min_distance_s=3.0)
    if duchenne_tpl is not None:
        path = os.path.join(OUT_DIR, 'duchenne_template.npy')
        np.save(path, duchenne_tpl)
        peak_amp = duchenne_tpl[:, [44, 45, 7, 8]].max()
        print(f"  {n_duch} events, peak AU amplitude={peak_amp:.3f}")
        print(f"  Saved: {path}")
    else:
        print(f"  FAILED: too few events")

    # --- Frown template (AU30+31) ---
    print("\n[3/3] Frown template (AU30, AU31)...")
    frown_tpl, n_frown = extract_event_template(
        bl_concat, [30, 31], fs=FS_BL, window_s=10.0, onset_s=3.0,
        prominence=0.15, min_distance_s=5.0)
    if frown_tpl is not None:
        path = os.path.join(OUT_DIR, 'frown_template.npy')
        np.save(path, frown_tpl)
        peak_amp = frown_tpl[:, [30, 31]].max()
        print(f"  {n_frown} events, peak AU amplitude={peak_amp:.3f}")
        print(f"  Saved: {path}")
    else:
        print(f"  FAILED: too few events, using synthetic Gaussian template")
        # Synthetic fallback
        T_tpl = int(10.0 * FS_BL)
        t = np.arange(T_tpl) / FS_BL
        pulse = np.exp(-0.5 * ((t - 3.0) / 1.0) ** 2)
        frown_tpl = np.zeros((T_tpl, 52), dtype=np.float32)
        for au in [30, 31]:
            frown_tpl[:, au] = pulse * 0.25
        np.save(os.path.join(OUT_DIR, 'frown_template.npy'), frown_tpl)
        print(f"  Saved synthetic frown template")

    print(f"\nDone. Templates in: {OUT_DIR}")


if __name__ == '__main__':
    main()
