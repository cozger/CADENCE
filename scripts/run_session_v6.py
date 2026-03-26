"""CADENCE V6 single-session analysis.

Production pipeline using validated methods:
  - EEG: fast_cycles multiband volt_amp (GPU-accelerated)
  - BL:  saliency-based facial event detection + confidence scoring

Usage:
    python scripts/run_session_v6.py --session y_06
    python scripts/run_session_v6.py --session y_06 --segments conv_1 conv_2
    python scripts/run_session_v6.py --session y_06 --eeg-only
    python scripts/run_session_v6.py --session y_06 --bl-only
"""
import argparse
import json
import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf

from cadence.significance.fast_cycles import analyze_interbrain_cycles_multiband, eeg_coupling_timecourse
from cadence.significance.bl_coupling import facial_event_catalog
from cadence.significance.distributional_stats import distributional_stats
from cadence.data.xdf_loader import _detect_roles
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

FS_BL = 30.0
FS_EEG = 256.0

# Default segments to analyze
DEFAULT_SEGMENTS = ['conv_1', 'conv_2', 'meditate_K', 'meditate_B',
                    'base_EO', 'base_EC']


def load_xdf_session(xdf_path):
    """Load XDF and extract landmarks, EEG, markers, and roles."""
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

    roles = _detect_roles(data)
    p1_role = roles.get('p1_role', 'unknown')
    p2_role = roles.get('p2_role', 'unknown')

    markers = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                markers[v[0]] = t

    landmarks = {}
    eeg = {}

    for stream in data:
        name = stream['info']['name'][0]

        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 52 and person not in landmarks:
                landmarks[person] = (
                    np.array(stream['time_stamps']),
                    np.array(stream['time_series'], dtype=np.float32))

        elif 'eeg' in name.lower() or 'emotiv' in name.lower():
            person = 'P1' if 'P1' in name or '1' in name.split('_')[0] else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 14 and person not in eeg:
                eeg[person] = (
                    np.array(stream['time_stamps']),
                    np.array(stream['time_series'], dtype=np.float32))

    return {
        'markers': markers,
        'landmarks': landmarks,
        'eeg': eeg,
        'p1_role': p1_role,
        'p2_role': p2_role,
    }


def extract_bl_segment(landmarks, t_start, t_end, fs=FS_BL):
    """Extract and resample blendshapes for a segment."""
    dur = t_end - t_start
    T = int(dur * fs)
    t_grid = np.linspace(0, dur, T)
    sigs = {}

    for person in ['P1', 'P2']:
        if person not in landmarks:
            return None, None, 0
        ts, d = landmarks[person]
        m = (ts >= t_start) & (ts <= t_end)
        if m.sum() < 100:
            return None, None, 0
        sig = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)
        np.clip(sig, 0, 1, out=sig)
        sigs[person] = sig

    return sigs.get('P1'), sigs.get('P2'), dur


def extract_eeg_segment(cached_session, markers, segment, lsl_offset=0.0):
    """Extract raw EEG for a segment from cached session data.

    The cache has p1_eeg and p2_eeg at native rate (256 Hz for Emotiv)
    with session-relative timestamps (starting at 0). We convert LSL
    marker times to session-relative using lsl_offset, then avg-reference
    and z-score for fast_cycles.
    """
    if cached_session is None:
        return None, None, 0, 0

    t_start_lsl = markers.get(f'{segment}_start')
    t_end_lsl = markers.get(f'{segment}_stop')
    if t_start_lsl is None or t_end_lsl is None:
        return None, None, 0, 0

    # Convert LSL to session-relative time
    t_start = t_start_lsl - lsl_offset
    t_end = t_end_lsl - lsl_offset

    sigs = {}
    fs = 0

    for person in ['p1', 'p2']:
        eeg_key = f'{person}_eeg'
        ts_key = f'{person}_eeg_ts'

        if eeg_key not in cached_session or ts_key not in cached_session:
            return None, None, 0, 0

        eeg_all = cached_session[eeg_key]
        ts_all = cached_session[ts_key]

        m = (ts_all >= t_start) & (ts_all <= t_end)
        if m.sum() < 1000:
            return None, None, 0, 0

        eeg_seg = eeg_all[m].astype(np.float64)
        ts_seg = ts_all[m]

        n_ch = min(14, eeg_seg.shape[1])
        eeg_seg = eeg_seg[:, :n_ch]

        fs = len(ts_seg) / (ts_seg[-1] - ts_seg[0])

        # Average reference
        eeg_seg -= eeg_seg.mean(axis=1, keepdims=True)

        # Z-score per channel
        for ch in range(n_ch):
            std = eeg_seg[:, ch].std()
            if std > 1e-8:
                eeg_seg[:, ch] = (eeg_seg[:, ch] - eeg_seg[:, ch].mean()) / std

        sigs[person.upper()] = eeg_seg

    # Trim to same length
    p1_out = sigs.get('P1')
    p2_out = sigs.get('P2')
    if p1_out is not None and p2_out is not None:
        min_len = min(len(p1_out), len(p2_out))
        p1_out = p1_out[:min_len]
        p2_out = p2_out[:min_len]

    dur = t_end_lsl - t_start_lsl
    return p1_out, p2_out, dur, fs


def analyze_segment(session_data, segment, run_eeg=True, run_bl=True,
                    n_surrogates=200):
    """Run EEG + BL analysis on one segment.

    Returns dict with results, or None if segment not found.
    """
    markers = session_data['markers']
    t_start = markers.get(f'{segment}_start')
    t_end = markers.get(f'{segment}_stop')

    if t_start is None or t_end is None:
        return None

    dur = t_end - t_start
    if dur < 30:
        return None

    result = {
        'segment': segment,
        'duration_s': round(dur, 1),
        'lsl_start': round(t_start, 3),
        'lsl_end': round(t_end, 3),
    }

    # ── BL ────────────────────────────────────────────────────────
    if run_bl:
        p1_bl, p2_bl, bl_dur = extract_bl_segment(
            session_data['landmarks'], t_start, t_end)

        if p1_bl is not None:
            t0 = time.time()
            cat = facial_event_catalog(
                p1_bl, p2_bl, FS_BL,
                lsl_start=t_start, segment_name=segment)
            bl_time = time.time() - t0

            result['bl'] = {
                'n_events_p1': cat.n_events_p1,
                'n_events_p2': cat.n_events_p2,
                'n_shared': cat.n_shared,
                'n_shared_smiles': cat.n_shared_smiles,
                'elapsed_s': round(bl_time, 2),
                'shared_smiles': [
                    {
                        'lsl_a': round(s.event_a.lsl_time, 3),
                        'lsl_b': round(s.event_b.lsl_time, 3),
                        'lag': round(s.lag, 2),
                        'leader': s.leader,
                        'smile_a': round(s.event_a.smile_composite, 3),
                        'smile_b': round(s.event_b.smile_composite, 3),
                        'confidence': round(s.joint_smile_confidence, 4),
                    }
                    for s in cat.shared_smiles
                ],
            }
            if len(cat.shared_events) >= 10:
                lags = np.array([s.lag for s in cat.shared_events])
                result['bl']['lag_dist'] = distributional_stats(lags)
            if len(cat.shared_smiles) >= 10:
                confs = np.array([s.joint_smile_confidence
                                  for s in cat.shared_smiles])
                result['bl']['smile_conf_dist'] = distributional_stats(confs)
            print(f"    BL: {cat.n_events_p1} P1 + {cat.n_events_p2} P2 events, "
                  f"{cat.n_shared_smiles} shared smiles ({bl_time:.1f}s)",
                  flush=True)
        else:
            result['bl'] = None
            print(f"    BL: no data", flush=True)

    # ── EEG ───────────────────────────────────────────────────────
    if run_eeg:
        p1_eeg, p2_eeg, eeg_dur, fs_eeg = extract_eeg_segment(
            session_data.get('cached'), markers, segment,
            lsl_offset=session_data.get('lsl_offset', 0.0))

        if p1_eeg is not None:
            t0 = time.time()
            eeg_result = analyze_interbrain_cycles_multiband(
                p1_eeg, p2_eeg, fs_eeg,
                n_surrogates=n_surrogates, seed=42)
            eeg_time = time.time() - t0

            # Extract key z-scores
            eeg_summary = {'elapsed_s': round(eeg_time, 2)}
            if 'per_band' in eeg_result:
                for band, band_res in eeg_result['per_band'].items():
                    va = band_res.get('volt_amp', {})
                    eeg_summary[f'{band}_volt_amp_z'] = round(
                        va.get('pooled_z', 0), 2)
            if 'combined' in eeg_result:
                va = eeg_result['combined'].get('volt_amp', {})
                eeg_summary['combined_volt_amp_z'] = round(
                    va.get('stouffer_z', va.get('pooled_z', 0)), 2)

            # Time-resolved coupling
            t0_tl = time.time()
            tl_result = eeg_coupling_timecourse(
                p1_eeg, p2_eeg, fs_eeg, n_surrogates=n_surrogates,
                smooth_samples=3, seed=42)
            tl_time = time.time() - t0_tl

            eeg_summary['tl_elapsed_s'] = round(tl_time, 2)
            eeg_summary['coupling_fraction'] = tl_result['combined']['coupling_fraction']
            eeg_summary['tl_mean_z'] = round(tl_result['combined']['mean_z'], 2)

            for band_name, band_data in tl_result['per_band'].items():
                z_arr = band_data.get('z')
                if z_arr is not None and len(z_arr) > 0:
                    eeg_summary[f'{band_name}_tl_dist'] = distributional_stats(z_arr)
            eeg_summary['combined_tl_dist'] = distributional_stats(
                tl_result['combined']['z'])

            # Store timecourses for plotting (not in JSON — too large)
            result['_eeg_tl'] = tl_result

            result['eeg'] = eeg_summary
            combined_z = eeg_summary.get('combined_volt_amp_z', 0)
            cf = eeg_summary.get('coupling_fraction', 0)
            print(f"    EEG: combined z={combined_z:+.2f}, "
                  f"coupling={cf:.0%} ({eeg_time + tl_time:.1f}s)",
                  flush=True)
        else:
            result['eeg'] = None
            print(f"    EEG: no data", flush=True)

    return result


def run_session(args):
    """Run V6 analysis on a single session."""
    import glob

    # Find XDF file
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'raw sessions')
    pattern = os.path.join(raw_dir, f'*{args.session}*.xdf')
    xdf_files = glob.glob(pattern)
    if not xdf_files:
        print(f"No XDF found matching '{args.session}' in {raw_dir}")
        return
    xdf_path = xdf_files[0]

    print(f"Session: {os.path.basename(xdf_path)}", flush=True)
    t0 = time.time()
    session_data = load_xdf_session(xdf_path)
    load_time = time.time() - t0
    print(f"Loaded XDF in {load_time:.1f}s (P1={session_data['p1_role']}, "
          f"P2={session_data['p2_role']})", flush=True)

    # Load cached session for EEG (raw EEG is split per-participant in cache)
    config = load_config()
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_path = None
    for name, path in cached_sessions:
        if args.session in name:
            cache_path = path
            break
    if cache_path:
        cached = load_session_from_cache(cache_path, config)
        session_data['cached'] = cached

        # Compute LSL offset: earliest marker LSL time - corresponding
        # session-relative time. The cached blendshape timestamps start
        # near 0 and the XDF landmarks have LSL times.
        # Use the blendshape timestamp range to estimate session start LSL.
        p1_bl_ts = cached.get('p1_blendshapes_ts')
        if p1_bl_ts is not None and 'P1' in session_data['landmarks']:
            lsl_ts = session_data['landmarks']['P1'][0]
            # Session-relative 0 corresponds to the first LSL timestamp
            session_data['lsl_offset'] = float(lsl_ts[0]) - float(p1_bl_ts[0])
        else:
            session_data['lsl_offset'] = float(min(session_data['markers'].values()))

        print(f"Loaded cache (EEG available, LSL offset={session_data['lsl_offset']:.1f})",
              flush=True)
    else:
        session_data['cached'] = None
        session_data['lsl_offset'] = 0.0
        print(f"No cache found -- EEG will be skipped", flush=True)

    segments = args.segments or DEFAULT_SEGMENTS
    run_eeg = not args.bl_only
    run_bl = not args.eeg_only

    # Output directory
    output_dir = os.path.join(args.output, args.session)
    os.makedirs(output_dir, exist_ok=True)

    session_results = {
        'session': args.session,
        'xdf_file': os.path.basename(xdf_path),
        'p1_role': session_data['p1_role'],
        'p2_role': session_data['p2_role'],
        'segments': {},
    }

    for segment in segments:
        print(f"\n  {segment}:", flush=True)
        result = analyze_segment(session_data, segment,
                                 run_eeg=run_eeg, run_bl=run_bl,
                                 n_surrogates=args.n_surrogates)
        if result is not None:
            session_results['segments'][segment] = result

    # Save
    out_path = os.path.join(output_dir, 'v6_results.json')
    with open(out_path, 'w') as f:
        json.dump(session_results, f, indent=2, default=str)

    total_time = time.time() - t0
    print(f"\nSaved to {out_path} ({total_time:.1f}s total)", flush=True)


def main():
    parser = argparse.ArgumentParser(description='CADENCE V6 session analysis')
    parser.add_argument('--session', required=True, help='Session name (e.g., y_06)')
    parser.add_argument('--segments', nargs='+', default=None,
                        help='Segments to analyze (default: all)')
    parser.add_argument('--output', default='results/v6',
                        help='Output directory')
    parser.add_argument('--eeg-only', action='store_true')
    parser.add_argument('--bl-only', action='store_true')
    parser.add_argument('--n-surrogates', type=int, default=200,
                        help='EEG surrogates (default: 200)')
    args = parser.parse_args()
    run_session(args)


if __name__ == '__main__':
    main()
