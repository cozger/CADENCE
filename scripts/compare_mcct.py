"""CADENCE vs MCCT side-by-side comparison.

Runs CADENCE on the same sessions as MCCT and compares timecourses.

Usage:
    python scripts/compare_mcct.py --session y_06
    python scripts/compare_mcct.py --session y_06 --mcct-results ../MCCT/results/v7_31ch_all_sessions
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.visualization.comparison import plot_cadence_vs_mcct, compute_correlation
from cadence.constants import MOD_SHORT


def find_mcct_timecourse(mcct_dir, session_name, direction):
    """Find MCCT timecourse JSON for a given session and direction."""
    # Common naming patterns in MCCT results
    patterns = [
        f'{session_name}/{direction}_timecourse.json',
        f'{session_name}/{direction.replace("_to_", "-")}_timecourse.json',
        f'{direction}_timecourse.json',
    ]
    for pat in patterns:
        path = os.path.join(mcct_dir, pat)
        if os.path.exists(path):
            return path

    # Search recursively
    for root, dirs, files in os.walk(mcct_dir):
        for f in files:
            if f.endswith('_timecourse.json') and session_name in root:
                if direction.replace('_to_', '-') in f or direction in f:
                    return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(description='CADENCE vs MCCT comparison')
    parser.add_argument('--session', default='y_06')
    parser.add_argument('--config', default=None)
    parser.add_argument('--mcct-results', default=None,
                        help='MCCT results directory')
    parser.add_argument('--output', default='results/cadence_comparison')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    # Default MCCT results location
    if args.mcct_results is None:
        args.mcct_results = os.path.join(
            os.path.dirname(config['session_cache']), 'results', 'v7_31ch_all_sessions')

    # Find session
    sessions = discover_cached_sessions(config['session_cache'])
    session_match = None
    for name, path in sessions:
        if args.session in name:
            session_match = (name, path)
            break

    if session_match is None:
        print(f"Session '{args.session}' not found")
        return

    name, cache_path = session_match
    print(f"Loading session: {name}")
    session = load_session_from_cache(cache_path)

    os.makedirs(args.output, exist_ok=True)
    estimator = CouplingEstimator(config)

    for direction in ['p1_to_p2', 'p2_to_p1']:
        print(f"\n--- {direction} ---")

        # Run CADENCE
        result = estimator.analyze_session(session, direction)

        # Find MCCT results
        mcct_path = find_mcct_timecourse(args.mcct_results, name, direction)
        if mcct_path is None:
            print(f"  MCCT timecourse not found for {name}/{direction}")
            continue

        print(f"  MCCT: {mcct_path}")

        # Comparison plot
        prefix = f'{name}_{direction.replace("_to_", "-")}'
        plot_cadence_vs_mcct(
            result, mcct_path,
            save_path=os.path.join(args.output, f'{prefix}_comparison.png'))

        # Correlation
        corrs = compute_correlation(result, mcct_path)
        print(f"  Correlations (CADENCE vs MCCT):")
        for mod, r in corrs.items():
            print(f"    {MOD_SHORT.get(mod, mod):>4s}: r={r:.3f}")

    print(f"\nComparison saved: {args.output}")


if __name__ == '__main__':
    main()
