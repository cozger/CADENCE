"""Preprocess XDF sessions into session_cache.

Usage:
    python scripts/preprocess_session.py path/to/session.xdf
    python scripts/preprocess_session.py session1.xdf session2.xdf
    python scripts/preprocess_session.py --all  # all XDFs in sessions dir
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.data.alignment import load_and_preprocess_cached


def preprocess_one(xdf_path, cache_dir):
    """Preprocess a single XDF file and print summary."""
    basename = os.path.basename(xdf_path)
    print(f"\n{'='*60}")
    print(f"Preprocessing: {basename}")
    print(f"{'='*60}")

    t0 = time.time()
    session = load_and_preprocess_cached(xdf_path, cache_dir=cache_dir)
    dt = time.time() - t0

    # Print summary
    duration = session.get('duration', 0)
    print(f"\n  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Preprocessing time: {dt:.1f}s")

    # Summarize modalities
    modalities = ['eeg', 'ecg', 'blendshapes', 'pose', 'eeg_features',
                  'ecg_features', 'pose_features']
    for participant in ['p1', 'p2']:
        print(f"\n  {participant.upper()}:")
        for mod in modalities:
            key = f'{participant}_{mod}'
            if key in session and session[key] is not None:
                data = session[key]
                if hasattr(data, 'shape'):
                    print(f"    {mod:20s}: {str(data.shape):>20s}  "
                          f"dtype={data.dtype}")

    return session


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess XDF sessions into session_cache')
    parser.add_argument('xdf_paths', nargs='*', help='XDF file path(s)')
    parser.add_argument('--all', action='store_true',
                        help='Process all XDFs in the sessions directory')
    parser.add_argument('--sessions-dir', default=None,
                        help='Directory containing XDF files (for --all)')
    parser.add_argument('--config', default=None, help='YAML config file')
    args = parser.parse_args()

    # Load config
    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path
    config = load_config(args.config)
    cache_dir = config['session_cache']

    # Collect XDF paths
    xdf_paths = list(args.xdf_paths)

    if args.all:
        sessions_dir = args.sessions_dir
        if sessions_dir is None:
            # Try common locations
            candidates = [
                os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), '..', 'MCCT', 'sessions'),
                os.path.expanduser('~/Desktop/MCCT/sessions'),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    sessions_dir = c
                    break
        if sessions_dir is None or not os.path.isdir(sessions_dir):
            print(f"Sessions directory not found. Use --sessions-dir to specify.")
            return 1
        for f in sorted(os.listdir(sessions_dir)):
            if f.endswith('.xdf'):
                xdf_paths.append(os.path.join(sessions_dir, f))

    if not xdf_paths:
        print("No XDF files specified. Use positional args or --all.")
        parser.print_help()
        return 1

    print(f"Processing {len(xdf_paths)} XDF file(s)")
    print(f"Cache directory: {cache_dir}")

    for xdf_path in xdf_paths:
        if not os.path.exists(xdf_path):
            print(f"\nERROR: File not found: {xdf_path}")
            continue
        preprocess_one(xdf_path, cache_dir)

    print(f"\n{'='*60}")
    print(f"Done. {len(xdf_paths)} session(s) cached to: {cache_dir}")


if __name__ == '__main__':
    main()
