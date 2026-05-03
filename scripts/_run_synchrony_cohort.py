"""Run Stages 5-9 of the synchrony repertoire pipeline (cohort fan-in).

Assumes Stages 1-4 are already cached per-session via
``_run_synchrony_per_session.py``. If they're not, this script will run
them first.

Stage list:
  4 cohort  pool per-session feature matrices
  5+6      UMAP (2D + 10D) + HDBSCAN
  7        per-cluster summarization + paired-face figures
  8        validation suite (bootstrap, LOO, threshold sensitivity)
  9        per-dyad repertoire CSV/fig + expressivity CSV/fig +
           rSLDS cross-tab + REPORT.md

Usage:
    python scripts/_run_synchrony_cohort.py             # everything
    python scripts/_run_synchrony_cohort.py --skip-validation
    python scripts/_run_synchrony_cohort.py --force     # re-run all stages
"""
from __future__ import annotations

# Windows torch+numpy DLL ordering: torch must be imported BEFORE numpy.
import torch as _torch  # noqa: F401

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cadence.ingest.quality import list_canonical_sessions
from cadence.synchrony.config import DEFAULT_CONFIG
from cadence.synchrony.episodes import (
    compute_cohort_threshold, segment_session_episodes,
)
from cadence.synchrony.events import detect_session_events
from cadence.synchrony.features.assembly import (
    assemble_session_features, pool_cohort_features,
)
from cadence.synchrony.clustering import fit_cohort_clusters
from cadence.synchrony.interpretation import summarize_clusters
from cadence.synchrony.viz import plot_cohort_embedding
from cadence.synchrony.validation import run_validation_suite
from cadence.synchrony.io import set_cohort_name
from cadence.synchrony.reporting import (
    per_dyad_repertoire, per_dyad_expressivity,
    repertoire_x_rslds_state, write_report_markdown,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                     help='Re-run all stages from scratch')
    ap.add_argument('--skip-validation', action='store_true',
                     help='Skip Stage 8 validation suite (saves ~2 min)')
    ap.add_argument('--ensure-per-session', action='store_true',
                     help='Run Stages 1-4 first if any session cache is missing')
    ap.add_argument('--conditions', type=str, default=None,
                     help='Comma-separated list of episode conditions to keep '
                     '(e.g. "conv_1,conv_2"). If unset, all episodes are used.')
    ap.add_argument('--cohort-name', type=str, default=None,
                     help='Output directory namespace under results/synchrony/. '
                     'Defaults to "cohort", or "cohort_<conditions_tag>" when '
                     '--conditions is set.')
    args = ap.parse_args()

    condition_filter = None
    if args.conditions:
        condition_filter = [c.strip() for c in args.conditions.split(',') if c.strip()]
    cohort_name = (args.cohort_name or
                    (f'cohort_{"_".join(condition_filter)}' if condition_filter
                     else 'cohort'))
    set_cohort_name(cohort_name)
    print(f'cohort namespace: results/synchrony/{cohort_name}/')
    if condition_filter:
        print(f'episode condition filter: {condition_filter}\n')

    sids = list_canonical_sessions()
    print(f'cohort: {len(sids)} canonical sessions\n')

    # Stage 4 cohort fan-in (after ensuring per-session caches exist if asked)
    if args.ensure_per_session:
        print('=== Ensuring per-session caches (Stages 1-4) ===')
        ct = compute_cohort_threshold(force=args.force)
        for sid in sids:
            try:
                detect_session_events(sid, force=args.force)
                segment_session_episodes(sid, cohort_threshold=ct,
                                            force=args.force)
                assemble_session_features(sid, force=args.force)
            except Exception as e:
                print(f'  {sid}: SKIP ({type(e).__name__}: {e})')

    print('\n=== Stage 4 cohort fan-in ===')
    t0 = time.perf_counter()
    pool = pool_cohort_features(sids, condition_filter=condition_filter)
    m = pool['_meta']
    print(f'  {m["n_sessions"]} sessions, {m["n_episodes"]} episodes, '
           f'{m["n_low_quality"]} low-quality')
    if condition_filter:
        print(f'  filtered out {m["n_filtered_out"]} episodes outside '
               f'{condition_filter}')
    print(f'  mean imputation rate: {m["mean_imputation_rate"]:.2%}')
    print(f'  ({time.perf_counter() - t0:.1f}s)')

    print('\n=== Stages 5+6 UMAP + HDBSCAN ===')
    t1 = time.perf_counter()
    cls = fit_cohort_clusters(force=args.force)
    cm = cls['_meta']
    print(f'  {cm["n_clusters"]} clusters + {cm["n_noise"]} noise '
           f'(min_cluster_size={cm["min_cluster_size"]})')
    print(f'  ({time.perf_counter() - t1:.1f}s)')

    print('\n=== Stage 7 cluster interpretation + figures ===')
    t2 = time.perf_counter()
    plot_cohort_embedding()
    sumry = summarize_clusters()
    print(f'  {sumry["n_clusters"]} clusters summarized + figures rendered '
           f'({time.perf_counter() - t2:.1f}s)')

    if not args.skip_validation:
        print('\n=== Stage 8 validation suite ===')
        t3 = time.perf_counter()
        val = run_validation_suite()
        boot = val['bootstrap_stability']
        loo = val['leave_one_dyad_out']
        thr = val['threshold_sensitivity']
        print(f'  bootstrap mean ARI: {boot["mean_ari"]:.3f} '
               f'(spec≥0.6: {"PASS" if boot["pass_mean_ari_gt_0p6"] else "WARN"})')
        print(f'  LOO mean ARI: {loo["mean_ari"]:.3f} '
               f'(spec≥0.7: {"PASS" if loo["pass_mean_ari_gt_0p7"] else "WARN"})')
        print(f'  threshold ±20% count ratio: '
               f'[{thr["count_ratio_min"]:.2f}, {thr["count_ratio_max"]:.2f}] '
               f'(spec within 0.5–2x: '
               f'{"PASS" if thr["pass_within_2x"] else "WARN"})')
        print(f'  ({time.perf_counter() - t3:.1f}s)')

    print('\n=== Stage 9 reporting ===')
    t4 = time.perf_counter()
    df_rep = per_dyad_repertoire()
    df_exp = per_dyad_expressivity()
    xt = repertoire_x_rslds_state()
    print(f'  per-dyad repertoire: {len(df_rep)} dyads')
    print(f'  per-dyad expressivity: {len(df_exp)} dyads')
    print(f'  rSLDS cross-tab: '
           f'{xt["n_sessions_with_rslds"]} sessions, '
           f'{xt["n_episodes_assigned"]} episodes assigned')
    out = write_report_markdown()
    print(f'  REPORT.md written → {out}')
    print(f'  ({time.perf_counter() - t4:.1f}s)')

    print('\nDone.')


if __name__ == '__main__':
    main()
