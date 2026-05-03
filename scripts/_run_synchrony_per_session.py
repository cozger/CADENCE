"""Run Stages 1-4 of the synchrony repertoire pipeline per session.

Per spec: Stage 1 events + Stage 2 episodes (uses cohort-anchored T*) +
Stage 3a-d features + Stage 4 assembly. Each stage caches to
``results/synchrony/<sid>/`` and is skipped on re-run if the cache hash
matches the current config + face npz.

Usage:
    python scripts/_run_synchrony_per_session.py --session y_06
    python scripts/_run_synchrony_per_session.py --all
    python scripts/_run_synchrony_per_session.py --all --force --n-jobs 4
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
from cadence.synchrony.events import detect_session_events
from cadence.synchrony.episodes import (
    compute_cohort_threshold, segment_session_episodes,
)
from cadence.synchrony.features.assembly import assemble_session_features
from cadence.synchrony.viz import (
    plot_episodes_overlay, plot_events_overlay,
)


def _do_one(sid: str, force: bool, ct: dict, with_figures: bool) -> dict:
    s_t = time.perf_counter()
    info = {'sid': sid, 'status': 'ok', 'note': ''}
    try:
        out_ev = detect_session_events(sid, force=force)
        out_ep = segment_session_episodes(sid, cohort_threshold=ct, force=force)
        out_ft = assemble_session_features(sid, force=force)
        m = out_ft['_meta']
        info['n_episodes'] = m.get('n_episodes_in_features', 0)
        info['pct_3a_valid'] = m.get('pct_3a_valid', 0)
        info['pct_3b_valid'] = m.get('pct_3b_valid', 0)
        info['pct_3c_valid'] = m.get('pct_3c_valid', 0)
        info['pct_3d_valid'] = m.get('pct_3d_valid', 0)
        if with_figures:
            try:
                plot_events_overlay(sid)
                plot_episodes_overlay(sid)
            except Exception as e:
                info['note'] = f'figure error: {type(e).__name__}: {e}'
    except Exception as e:
        info['status'] = 'error'
        info['note'] = f'{type(e).__name__}: {e}'
    info['elapsed_s'] = round(time.perf_counter() - s_t, 1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--force', action='store_true',
                     help='Recompute even if cache is up-to-date')
    ap.add_argument('--no-figures', action='store_true',
                     help='Skip per-session sanity figures')
    args = ap.parse_args()

    if not (args.session or args.all):
        ap.error('Provide --session <sid> or --all')

    sids = (list_canonical_sessions() if args.all else [args.session])

    print(f'=== Computing cohort T* (one pass over base_EO) ===')
    t0 = time.perf_counter()
    ct = compute_cohort_threshold(force=args.force)
    print(f'  T_star = {ct["T_star"]:.5f} '
           f'({ct["n_sessions_pooled"]} sessions, '
           f'{ct["n_samples_pooled"]} samples; '
           f'{time.perf_counter() - t0:.1f}s)')

    print(f'\n=== Running per-session Stages 1-4 on {len(sids)} session(s) ===')
    print(f'{"sid":25s} | {"n_eps":5s} | {"3a%":5s} {"3b%":5s} {"3c%":5s} {"3d%":5s} | '
          f'{"sec":5s} | note')
    print('-' * 100)
    n_ok = n_err = 0
    for sid in sids:
        info = _do_one(sid, force=args.force, ct=ct,
                         with_figures=not args.no_figures)
        status_tag = '✓' if info['status'] == 'ok' else '✗'
        if info['status'] == 'ok':
            n_ok += 1
            print(f'{sid:25s} | {info["n_episodes"]:5d} | '
                   f'{info["pct_3a_valid"]:5.1f} {info["pct_3b_valid"]:5.1f} '
                   f'{info["pct_3c_valid"]:5.1f} {info["pct_3d_valid"]:5.1f} | '
                   f'{info["elapsed_s"]:5.1f} | {status_tag} {info["note"]}')
        else:
            n_err += 1
            print(f'{sid:25s} |       |                            | '
                   f'{info["elapsed_s"]:5.1f} | ✗ {info["note"]}')
    print(f'\nSummary: {n_ok} OK, {n_err} errors')
    sys.exit(0 if n_err == 0 else 1)


if __name__ == '__main__':
    main()
