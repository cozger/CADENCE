"""CADENCE V6 batch analysis across all sessions.

Runs fast_cycles (EEG) + facial_event_catalog (BL) on all sessions,
aggregates results per condition.

Usage:
    python scripts/run_all_sessions_v6.py
    python scripts/run_all_sessions_v6.py --sessions y_06 y_17
    python scripts/run_all_sessions_v6.py --bl-only
"""
import argparse
import json
import os
import sys
import time
import glob

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from scripts.run_session_v6 import (
    load_xdf_session, analyze_segment, DEFAULT_SEGMENTS,
)
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'raw sessions')


def process_one_session(xdf_path, config, segments, run_eeg, run_bl,
                        n_surrogates, output_dir):
    """Process a single session (called by joblib)."""
    session_name = os.path.splitext(os.path.basename(xdf_path))[0]

    try:
        session_data = load_xdf_session(xdf_path)
    except Exception as e:
        print(f"  {session_name}: LOAD ERROR {e}", flush=True)
        return None

    # Load cache for EEG
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_path = None
    for name, path in cached_sessions:
        if any(part in name for part in session_name.replace('_', ' ').split()):
            cache_path = path
            break

    if cache_path:
        try:
            cached = load_session_from_cache(cache_path, config)
            session_data['cached'] = cached
            p1_bl_ts = cached.get('p1_blendshapes_ts')
            if p1_bl_ts is not None and 'P1' in session_data['landmarks']:
                lsl_ts = session_data['landmarks']['P1'][0]
                session_data['lsl_offset'] = float(lsl_ts[0]) - float(p1_bl_ts[0])
            else:
                session_data['lsl_offset'] = float(min(session_data['markers'].values()))
        except Exception:
            session_data['cached'] = None
            session_data['lsl_offset'] = 0.0
    else:
        session_data['cached'] = None
        session_data['lsl_offset'] = 0.0

    results = {
        'session': session_name,
        'p1_role': session_data['p1_role'],
        'p2_role': session_data['p2_role'],
        'segments': {},
    }

    for segment in segments:
        result = analyze_segment(session_data, segment,
                                 run_eeg=run_eeg, run_bl=run_bl,
                                 n_surrogates=n_surrogates)
        if result is not None:
            results['segments'][segment] = result

    # Save per-session
    sess_dir = os.path.join(output_dir, session_name)
    os.makedirs(sess_dir, exist_ok=True)
    out_path = os.path.join(sess_dir, 'v6_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    n_segs = len(results['segments'])
    print(f"  {session_name}: {n_segs} segments -> {out_path}", flush=True)
    return results


def aggregate_results(all_results):
    """Aggregate across sessions per condition."""
    agg = {}

    for result in all_results:
        if result is None:
            continue
        session = result['session']
        therapist_is = 'P1' if result['p1_role'] == 'therapist' else 'P2'

        for seg_name, seg_data in result['segments'].items():
            if seg_name not in agg:
                agg[seg_name] = {'eeg': [], 'bl': [], 'sessions': []}

            agg[seg_name]['sessions'].append(session)

            if seg_data.get('eeg'):
                eeg = seg_data['eeg']
                entry = {
                    'session': session,
                    'combined_z': eeg.get('combined_volt_amp_z', 0),
                    'theta_z': eeg.get('theta_volt_amp_z', 0),
                    'alpha_z': eeg.get('alpha_volt_amp_z', 0),
                    'beta_z': eeg.get('beta_volt_amp_z', 0),
                    'coupling_fraction': eeg.get('coupling_fraction', 0),
                    'tl_mean_z': eeg.get('tl_mean_z', 0),
                }
                if eeg.get('combined_tl_dist'):
                    entry['combined_tl_dist'] = eeg['combined_tl_dist']
                agg[seg_name]['eeg'].append(entry)

            if seg_data.get('bl'):
                bl = seg_data['bl']
                entry = {
                    'session': session,
                    'n_shared_smiles': bl['n_shared_smiles'],
                    'n_events_p1': bl['n_events_p1'],
                    'n_events_p2': bl['n_events_p2'],
                }
                if bl.get('lag_dist'):
                    entry['lag_dist'] = bl['lag_dist']
                if bl.get('smile_conf_dist'):
                    entry['smile_conf_dist'] = bl['smile_conf_dist']
                agg[seg_name]['bl'].append(entry)

    return agg


def main():
    parser = argparse.ArgumentParser(description='CADENCE V6 batch analysis')
    parser.add_argument('--sessions', nargs='+', default=None,
                        help='Session names (default: all)')
    parser.add_argument('--output', default='results/v6')
    parser.add_argument('--eeg-only', action='store_true')
    parser.add_argument('--bl-only', action='store_true')
    parser.add_argument('--n-surrogates', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel sessions (default 1 — GPU contention)')
    args = parser.parse_args()

    config = load_config()

    # Find XDF files
    xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
    if args.sessions:
        xdf_files = [f for f in xdf_files
                     if any(s in os.path.basename(f) for s in args.sessions)]

    print(f"Found {len(xdf_files)} sessions", flush=True)
    os.makedirs(args.output, exist_ok=True)

    run_eeg = not args.bl_only
    run_bl = not args.eeg_only

    t0 = time.time()

    if args.n_jobs == 1:
        all_results = []
        for xdf_path in xdf_files:
            r = process_one_session(xdf_path, config, DEFAULT_SEGMENTS,
                                     run_eeg, run_bl, args.n_surrogates,
                                     args.output)
            all_results.append(r)
    else:
        all_results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_session)(
                xdf, config, DEFAULT_SEGMENTS, run_eeg, run_bl,
                args.n_surrogates, args.output)
            for xdf in xdf_files)

    total_time = time.time() - t0

    # Aggregate
    agg = aggregate_results([r for r in all_results if r is not None])

    print(f"\n{'='*70}")
    print(f"AGGREGATION ({total_time:.0f}s total)")
    print(f"{'='*70}")

    for seg_name in sorted(agg.keys()):
        seg = agg[seg_name]
        n = len(seg['sessions'])
        print(f"\n  {seg_name} (n={n}):")

        if seg['eeg']:
            z_vals = [e['combined_z'] for e in seg['eeg']]
            print(f"    EEG volt_amp: mean z={np.mean(z_vals):+.2f}, "
                  f"range=[{min(z_vals):+.1f}, {max(z_vals):+.1f}]")
            dfa_vals = [e['combined_tl_dist']['dfa_exponent']
                        for e in seg['eeg']
                        if e.get('combined_tl_dist', {}).get('dfa_exponent') is not None]
            if dfa_vals:
                print(f"    EEG DFA: mean={np.mean(dfa_vals):.2f}, "
                      f"range=[{min(dfa_vals):.2f}, {max(dfa_vals):.2f}]")

        if seg['bl']:
            smiles = [b['n_shared_smiles'] for b in seg['bl']]
            print(f"    BL shared smiles: mean={np.mean(smiles):.1f}, "
                  f"range=[{min(smiles)}, {max(smiles)}]")
            lag_sk = [b['lag_dist']['skewness']
                      for b in seg['bl']
                      if b.get('lag_dist', {}).get('skewness') is not None]
            if lag_sk:
                print(f"    BL lag skewness: mean={np.mean(lag_sk):.2f}, "
                      f"range=[{min(lag_sk):.2f}, {max(lag_sk):.2f}]")

    # Save aggregation
    agg_path = os.path.join(args.output, 'v6_grand_summary.json')
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\nSaved to {agg_path}")


if __name__ == '__main__':
    main()
