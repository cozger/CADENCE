"""Tier 1 screening runner — Module 2 + Module 3 → screening_report.md.

Usage:
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --session y_06
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --results-dir results/v11
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --outputs-dir diagnostics/outputs
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.shared.data_loader import load_session, load_all_sessions
from diagnostics.shared.report_utils import make_output_dir, write_md_report
from diagnostics.tier1_screening.module2_feature_diagnostics import run_feature_diagnostics
from diagnostics.tier1_screening.module3_transition_analysis import run_transition_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default=None)
    parser.add_argument('--results-dir', default='results/v11')
    parser.add_argument('--outputs-dir', default='diagnostics/outputs')
    args = parser.parse_args()

    if args.session:
        sessions = [load_session(args.session, results_dir=args.results_dir)]
    else:
        sessions = load_all_sessions(results_dir=args.results_dir)
    print(f'Loaded {len(sessions)} session(s)')

    out_root = make_output_dir(args.outputs_dir, 'tier1_screening')
    m2_dir = os.path.join(out_root, 'module2')
    m3_dir = os.path.join(out_root, 'module3')

    print('Running Module 2 (feature diagnostics)...')
    m2_flags = run_feature_diagnostics(sessions, m2_dir)

    print('Running Module 3 (transition analysis)...')
    m3_flags = run_transition_analysis(sessions, m3_dir)

    # Write screening report
    screening_path = os.path.join(out_root, 'screening_report.md')
    with open(screening_path, 'w') as f:
        f.write('# Tier 1 Screening Report\n\n')
        f.write('## Module 2 Flags\n')
        f.write(f'SLOW_DRIFT: {m2_flags["slow_drift"]}\n')
        f.write(f'LOW_INFO: {m2_flags["low_info"]}\n')
        f.write(f'COLLINEAR_PAIRS: {m2_flags["collinear_pairs"]}\n')
        f.write(f'HIGH_VIF: {m2_flags["high_vif"]}\n\n')
        f.write('## Module 3 Flags\n')
        f.write(f'DWELL_RATIO_LOW: {m3_flags["dwell_ratio_low"]}\n')
        f.write(f'HIGH_FLICKER_PCT: {m3_flags["high_flicker_pct"]}\n')
        event_verdict = ('TRANSITIONS_EVENT_LOCKED' if m3_flags['transitions_event_locked']
                         else 'TRANSITIONS_RANDOM' if m3_flags['transitions_random']
                         else 'TRANSITIONS_AMBIGUOUS')
        f.write(f'{event_verdict}\n\n')
        f.write('## Recommendation\n')
        n_flags = (len(m2_flags['slow_drift']) + len(m2_flags['high_vif'])
                   + len(m3_flags['dwell_ratio_low']))
        if n_flags == 0:
            f.write('No critical flags. Proceed to Tier 2.\n')
        else:
            f.write(f'{n_flags} flag(s) raised. Review before running Tier 2 Module 4/5.\n')

    print(f'Tier 1 complete. Outputs: {out_root}')
    print(f'Screening report: {screening_path}')


if __name__ == '__main__':
    main()
