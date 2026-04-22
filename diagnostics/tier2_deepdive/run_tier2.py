"""Tier 2 deep-dive runner — Modules 1, 4, 5, 6.

Usage:
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py --skip-modules 4
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py --screening-report diagnostics/outputs/<ts>/screening_report.md
"""
import argparse, os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.shared.data_loader import load_session, load_all_sessions
from diagnostics.shared.report_utils import make_output_dir
from diagnostics.tier2_deepdive.module1_obs_space import run_obs_space_analysis
from diagnostics.tier2_deepdive.module4_model_comparison import run_model_comparison
from diagnostics.tier2_deepdive.module5_block_pca import run_block_pca
from diagnostics.tier2_deepdive.module6_null_ablation import run_null_ablation


def _read_flags_from_screening_report(report_path: str) -> dict:
    """Parse screening_report.md for Module 2 flags to pass to Module 5."""
    flags = {'slow_drift': [], 'high_vif': [], 'collinear_pairs': [], 'low_info': []}
    if not report_path or not os.path.exists(report_path):
        return flags
    with open(report_path) as f:
        text = f.read()
    for line in text.splitlines():
        for key in flags:
            if line.upper().startswith(key.upper()):
                m = re.search(r'\[([^\]]*)\]', line)
                if m:
                    content = m.group(1).replace("'", '').replace('"', '')
                    flags[key] = [c.strip() for c in content.split(',') if c.strip()]
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default=None)
    parser.add_argument('--results-dir', default='results/v11')
    parser.add_argument('--outputs-dir', default='diagnostics/outputs')
    parser.add_argument('--screening-report', default=None,
                        help='Path to screening_report.md from Tier 1')
    parser.add_argument('--skip-modules', nargs='+', type=int, default=[],
                        help='Module numbers to skip (e.g. --skip-modules 4)')
    parser.add_argument('--n-neighbors', type=int, default=30)
    parser.add_argument('--min-dist', type=float, default=0.1)
    args = parser.parse_args()

    if args.session:
        sessions = [load_session(args.session, results_dir=args.results_dir)]
    else:
        sessions = load_all_sessions(results_dir=args.results_dir)
    print(f'Loaded {len(sessions)} session(s)')

    flags = _read_flags_from_screening_report(args.screening_report)
    if args.screening_report is None:
        print('Warning: no --screening-report provided. Module 5 will not exclude flagged channels.')

    out_root = make_output_dir(args.outputs_dir, 'tier2_deepdive')

    if 1 not in args.skip_modules:
        print('Running Module 1 (observation space)...')
        run_obs_space_analysis(sessions, os.path.join(out_root, 'module1'),
                               n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    if 4 not in args.skip_modules:
        print('Running Module 4 (model comparison — may take 20-40 min)...')
        run_model_comparison(sessions, os.path.join(out_root, 'module4'))

    if 5 not in args.skip_modules:
        print('Running Module 5 (block PCA)...')
        run_block_pca(sessions, flags, os.path.join(out_root, 'module5'))

    if 6 not in args.skip_modules:
        print('Running Module 6 (null-state ablation)...')
        run_null_ablation(sessions, os.path.join(out_root, 'module6'))

    print(f'Tier 2 complete. Outputs: {out_root}')


if __name__ == '__main__':
    main()
