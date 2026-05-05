"""CADENCE v2 multi-session discovery pipeline.

Runs Stage 1 (group lasso) on all sessions, aggregates with cross-session
consistency filtering, then runs Stage 2 (EWLS) on each session with the
consistent feature set.

Supports multi-GPU parallelism: --n-gpus 2 splits sessions across GPUs
for both Stage 1 and Stage 2.

Usage:
    python scripts/run_discovery.py
    python scripts/run_discovery.py --min-sessions 4 --config configs/default.yaml
    python scripts/run_discovery.py --n-gpus 2 --device cuda
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.coupling.discovery import (
    DiscoveryResult, cross_session_consistency, build_stage2_feature_set,
    discovery_summary,
)
from cadence.visualization.discovery import (
    plot_discovery_report, plot_lambda_path, plot_feature_selection_heatmap,
)
from cadence.visualization.spectral import plot_spectral_coupling_map, extract_spectral_map
from cadence.visualization.heatmaps import plot_coupling_matrix
from cadence.constants import WAVELET_CENTER_FREQS, EEG_ROI_NAMES, MOD_SHORT_V2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _stage1_worker(gpu_id, session_batch, config_path, result_queue):
    """Worker: run Stage 1 discovery on a batch of sessions on one GPU."""
    device = f'cuda:{gpu_id}'
    config = load_config(config_path)
    config['pipeline'] = 'v2'
    config['device'] = device

    print(f"[GPU {gpu_id}] Stage 1 on {len(session_batch)} sessions ({device})")
    discoveries = []

    for session_name, cache_path in session_batch:
        print(f"[GPU {gpu_id}] Session: {session_name}")
        try:
            session = load_session_from_cache(cache_path, config=config)
            estimator = CouplingEstimator(config)

            for direction in ['p1_to_p2', 'p2_to_p1']:
                if direction == 'p1_to_p2':
                    src_p, tgt_p = 'p1', 'p2'
                else:
                    src_p, tgt_p = 'p2', 'p1'

                source_sigs = estimator._extract_signals_v2(session, src_p)
                target_sigs = estimator._extract_signals_v2(session, tgt_p)

                ib_key = 'eeg_interbrain'
                if ib_key in session:
                    ib_valid = session.get(f'{ib_key}_valid',
                                            np.ones(len(session[ib_key]), dtype=bool))
                    if ib_valid.sum() >= 10:
                        source_sigs[ib_key] = (session[ib_key],
                                                session[f'{ib_key}_ts'], ib_valid)

                eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
                t0 = time.time()
                disc = estimator._stage1_discovery(
                    source_sigs, target_sigs, session.get('duration', 0), eval_rate)
                disc.session_name = f"{session_name}_{direction}"
                discoveries.append(disc)
                dt = time.time() - t0

                n_sel = sum(1 for v in disc.n_selected.values() if v > 0)
                print(f"[GPU {gpu_id}]   {direction}: {n_sel} pathways in {dt:.1f}s")

        except Exception:
            print(f"[GPU {gpu_id}] ERROR on {session_name}:")
            traceback.print_exc()

    result_queue.put(discoveries)
    print(f"[GPU {gpu_id}] Stage 1 complete.")


def _stage2_worker(gpu_id, session_batch, config_path, output_dir):
    """Worker: run Stage 2 EWLS on a batch of sessions on one GPU."""
    device = f'cuda:{gpu_id}'
    config = load_config(config_path)
    config['pipeline'] = 'v2'
    config['device'] = device

    print(f"[GPU {gpu_id}] Stage 2 on {len(session_batch)} sessions ({device})")

    for session_name, cache_path in session_batch:
        print(f"[GPU {gpu_id}] Session: {session_name}")
        try:
            session = load_session_from_cache(cache_path, config=config)
            estimator = CouplingEstimator(config)

            for direction in ['p1_to_p2', 'p2_to_p1']:
                t0 = time.time()
                result = estimator.analyze_session(session, direction)
                dt = time.time() - t0

                n_sig = result.n_significant_pathways
                print(f"[GPU {gpu_id}]   {direction}: {n_sig} sig in {dt:.1f}s")

                sess_dir = os.path.join(output_dir, session_name)
                os.makedirs(sess_dir, exist_ok=True)
                prefix = direction.replace('_to_', '-')

                plot_coupling_matrix(result, pipeline='v2',
                                     save_path=os.path.join(sess_dir, f'{prefix}_matrix.png'))

        except Exception:
            print(f"[GPU {gpu_id}] ERROR on {session_name}:")
            traceback.print_exc()

    print(f"[GPU {gpu_id}] Stage 2 complete.")


def run_discovery(args):
    config = load_config(args.config)
    config['pipeline'] = 'v2'
    if args.device:
        config['device'] = args.device

    n_gpus = getattr(args, 'n_gpus', 1) or 1
    if torch.cuda.is_available():
        n_gpus = min(n_gpus, torch.cuda.device_count())
    else:
        n_gpus = 1

    # Override discovery params from CLI
    disc_cfg = config.get('v2_discovery', {})
    min_sessions = args.min_sessions or disc_cfg.get('consistency_min_sessions', 4)

    # Discover sessions
    sessions_list = discover_cached_sessions(config['session_cache'])
    excluded = set(config.get('excluded_sessions', []))
    sessions_list = [(n, p) for n, p in sessions_list if n not in excluded]
    print(f"Found {len(sessions_list)} sessions, using {n_gpus} GPU(s)")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Group Lasso Discovery ──
    print("\n=== Stage 1: Group Lasso Discovery ===")

    if n_gpus > 1:
        # Multi-GPU: split sessions across GPUs
        mp.set_start_method('spawn', force=True)
        gpu_batches = [[] for _ in range(n_gpus)]
        for i, s in enumerate(sessions_list):
            gpu_batches[i % n_gpus].append(s)

        result_queue = mp.Queue()
        processes = []
        for gpu_id, batch in enumerate(gpu_batches):
            if not batch:
                continue
            p = mp.Process(
                target=_stage1_worker,
                args=(gpu_id, batch, args.config, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect discoveries from all workers
        all_discoveries = []
        while not result_queue.empty():
            all_discoveries.extend(result_queue.get())
    else:
        # Single GPU: sequential (original behavior)
        all_discoveries = []
        for session_name, cache_path in sessions_list:
            print(f"\n--- Session: {session_name} ---")
            session = load_session_from_cache(cache_path, config=config)
            print(f"  Duration: {session.get('duration', 0):.0f}s")

            estimator = CouplingEstimator(config)

            for direction in ['p1_to_p2', 'p2_to_p1']:
                print(f"  Direction: {direction}")
                t0 = time.time()

                if direction == 'p1_to_p2':
                    src_p, tgt_p = 'p1', 'p2'
                else:
                    src_p, tgt_p = 'p2', 'p1'

                source_sigs = estimator._extract_signals_v2(session, src_p)
                target_sigs = estimator._extract_signals_v2(session, tgt_p)

                ib_key = 'eeg_interbrain'
                if ib_key in session:
                    ib_valid = session.get(f'{ib_key}_valid',
                                            np.ones(len(session[ib_key]), dtype=bool))
                    if ib_valid.sum() >= 10:
                        source_sigs[ib_key] = (session[ib_key],
                                                session[f'{ib_key}_ts'], ib_valid)

                eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
                disc = estimator._stage1_discovery(
                    source_sigs, target_sigs, session.get('duration', 0), eval_rate)
                disc.session_name = f"{session_name}_{direction}"
                all_discoveries.append(disc)

                dt = time.time() - t0
                n_pathways_with_sel = sum(1 for v in disc.n_selected.values() if v > 0)
                print(f"  Completed in {dt:.1f}s — "
                      f"{n_pathways_with_sel} pathways with selected features")

    # ── Cross-session consistency (CPU) ──
    print(f"\n=== Cross-Session Consistency (min_sessions={min_sessions}) ===")
    consistency = cross_session_consistency(
        all_discoveries, min_sessions=min_sessions)

    summary = discovery_summary(consistency)
    print(summary)

    # Save discovery report
    report_path = os.path.join(output_dir, 'discovery_report.txt')
    with open(report_path, 'w') as f:
        f.write(summary)
    print(f"\nSaved: {report_path}")

    # Visualizations
    fig = plot_discovery_report(consistency,
                                save_path=os.path.join(output_dir, 'discovery_consistency.png'))
    if fig:
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Lambda path plots for first discovery
    if all_discoveries:
        for pathway, cv in all_discoveries[0].cv_scores.items():
            src, tgt = pathway
            src_s = MOD_SHORT_V2.get(src, src)
            tgt_s = MOD_SHORT_V2.get(tgt, tgt)
            # Just plot one example per pathway type
            break

    # Save consistency result as JSON
    cons_data = {
        'n_sessions': consistency.n_sessions,
        'min_sessions': consistency.min_sessions,
        'pathways': {},
    }
    for pw, features in consistency.consistent_features.items():
        src, tgt = pw
        cons_data['pathways'][f'{src}->{tgt}'] = {
            'n_consistent': len(features),
            'feature_indices': features,
            'counts': consistency.feature_counts[pw].tolist(),
        }

    json_path = os.path.join(output_dir, 'consistency_result.json')
    with open(json_path, 'w') as f:
        json.dump(cons_data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {json_path}")

    # ── Stage 2: EWLS on consistent feature set ──
    print("\n=== Stage 2: EWLS on Consistent Features ===")
    stage2_features = build_stage2_feature_set(consistency)

    if not stage2_features:
        print("  No consistent features found — skipping Stage 2")
        return

    if n_gpus > 1:
        # Multi-GPU Stage 2
        gpu_batches = [[] for _ in range(n_gpus)]
        for i, s in enumerate(sessions_list):
            gpu_batches[i % n_gpus].append(s)

        processes = []
        for gpu_id, batch in enumerate(gpu_batches):
            if not batch:
                continue
            p = mp.Process(
                target=_stage2_worker,
                args=(gpu_id, batch, args.config, output_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # Single GPU Stage 2
        for session_name, cache_path in sessions_list:
            print(f"\n--- Session: {session_name} ---")
            session = load_session_from_cache(cache_path, config=config)
            estimator = CouplingEstimator(config)

            for direction in ['p1_to_p2', 'p2_to_p1']:
                print(f"  Direction: {direction}")
                t0 = time.time()

                result = estimator.analyze_session(session, direction)
                dt = time.time() - t0

                n_sig = result.n_significant_pathways
                print(f"  Completed in {dt:.1f}s — {n_sig} significant pathways")

                sess_dir = os.path.join(output_dir, session_name)
                os.makedirs(sess_dir, exist_ok=True)
                prefix = direction.replace('_to_', '-')

                plot_coupling_matrix(result, pipeline='v2',
                                     save_path=os.path.join(sess_dir, f'{prefix}_matrix.png'))

                for pw_key in result.pathway_dr2:
                    src, tgt = pw_key
                    if 'wavelet' in src or 'interbrain' in src:
                        from cadence.coupling.discovery import DiscoveryResult
                        pass

    print(f"\nAll results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='CADENCE v2 multi-session discovery')
    parser.add_argument('--config', default=None, help='YAML config file')
    parser.add_argument('--output', default='results/cadence_v2_discovery',
                        help='Output directory')
    parser.add_argument('--device', default=None, help='Device override')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='Number of GPUs for parallel session processing')
    parser.add_argument('--min-sessions', type=int, default=None,
                        help='Minimum sessions for consistency (default: from config)')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    run_discovery(args)


if __name__ == '__main__':
    main()
