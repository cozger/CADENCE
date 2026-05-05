"""Generate grand average plots from existing JSON results.

Reads grand_summary.json and grand_condition_summary.json from a previous
run_all_sessions.py run and generates the 4 grand average plots.

Usage:
    python scripts/plot_grand_averages.py
    python scripts/plot_grand_averages.py --results results/cadence_all_sessions
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.visualization.grand_average import (
    plot_grand_classification_bars,
    plot_grand_dr2_bars,
    plot_grand_coupling_matrix,
    plot_grand_coupling_by_condition,
)


def main():
    parser = argparse.ArgumentParser(description='Generate grand average plots')
    parser.add_argument('--results', default='results/cadence_all_sessions',
                        help='Directory with grand_summary.json and grand_condition_summary.json')
    args = parser.parse_args()

    grand_path = os.path.join(args.results, 'grand_summary.json')
    grand_cond_path = os.path.join(args.results, 'grand_condition_summary.json')

    if not os.path.exists(grand_path):
        print(f"Error: {grand_path} not found. Run run_all_sessions.py first.")
        return
    if not os.path.exists(grand_cond_path):
        print(f"Error: {grand_cond_path} not found. Run run_all_sessions.py first.")
        return

    with open(grand_path) as f:
        grand_summary = json.load(f)
    with open(grand_cond_path) as f:
        grand_cond = json.load(f)

    print(f"Loaded {len(grand_summary)} direction entries, "
          f"{grand_cond['n_sessions']} sessions")

    # 1. Classification bars
    fig = plot_grand_classification_bars(
        grand_cond, save_path=os.path.join(args.results, 'grand_classification_bars.png'))
    if fig:
        plt.close(fig)
        print("Saved: grand_classification_bars.png")

    # 2. dR2 bars
    fig = plot_grand_dr2_bars(
        grand_cond, save_path=os.path.join(args.results, 'grand_dr2_bars.png'))
    if fig:
        plt.close(fig)
        print("Saved: grand_dr2_bars.png")

    # 3. Coupling matrix
    fig = plot_grand_coupling_matrix(
        grand_summary, save_path=os.path.join(args.results, 'grand_coupling_matrix.png'))
    if fig:
        plt.close(fig)
        print("Saved: grand_coupling_matrix.png")

    # 4. Coupling by condition
    fig = plot_grand_coupling_by_condition(
        grand_cond, save_path=os.path.join(args.results, 'grand_coupling_by_condition.png'))
    if fig:
        plt.close(fig)
        print("Saved: grand_coupling_by_condition.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
