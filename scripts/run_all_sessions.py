"""CADENCE all-sessions analysis with condition-level breakdown.

Runs CADENCE on all sessions, maps results to experimental conditions
(baseline, conversation, meditation, etc.), computes per-condition
coupling statistics, and aggregates grand averages.

Usage:
    python scripts/run_all_sessions.py
    python scripts/run_all_sessions.py --sessions y_06 y_17
    python scripts/run_all_sessions.py --output results/cadence_all_sessions
    python scripts/run_all_sessions.py --parallel-directions  # 2 GPUs: one direction per GPU
"""

import argparse
import json
import os
import sys
import tempfile
import time

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache, ensure_all_cached
from cadence.coupling.estimator import CouplingEstimator
from cadence.conditions import (
    session_condition_summary, aggregate_condition_summaries,
    CONDITION_LABELS,
)
from cadence.visualization.timecourse import plot_coupling_timecourse
from cadence.visualization.heatmaps import plot_coupling_matrix
from cadence.visualization.kernels import plot_coupling_kernels
from cadence.visualization.sparsity import plot_sparsity_summary, plot_block_detail
from cadence.visualization.grand_average import (
    plot_grand_classification_bars,
    plot_grand_dr2_bars,
    plot_grand_coupling_matrix,
    plot_grand_coupling_by_condition,
)
from cadence.coupling.serialization import save_result, load_result
from cadence.significance.detection import detection_summary
from cadence.constants import (
    MOD_SHORT_V2, MODALITY_ORDER_V2, INTERBRAIN_MODALITY,
)


def _parallel_direction_worker(gpu_id, direction, session_data, config_path,
                               device, result_path):
    """Worker process: run one direction on a specific GPU, save result to file."""
    # Set default CUDA device for this process — Triton launches kernels on
    # torch.cuda.current_device(), not the tensor's device. Without this,
    # GPU 1's worker has default device=cuda:0 and Triton can't access
    # cuda:1 pointers ("Pointer argument cannot be accessed from Triton").
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # Per-process Triton cache to prevent JIT compilation race between workers
    triton_cache = os.path.join(tempfile.gettempdir(), f'triton_cache_gpu{gpu_id}')
    os.makedirs(triton_cache, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = triton_cache

    config = load_config(config_path)
    config['device'] = device

    # Warm up Triton JIT at realistic BLOCK sizes (not just tiny T=20)
    try:
        from cadence.regression.ewls import EWLSSolver
        _ws = EWLSSolver(tau_seconds=10.0, lambda_ridge=1e-3,
                         eval_rate=2.0, device=device)
        for _T in [20, 512, 2048]:
            _X = torch.randn(1, _T, 5, device=device)
            _y = torch.randn(1, _T, 1, device=device)
            _ws.solve_batched(_X, _y)
        del _ws, _X, _y
        torch.cuda.empty_cache()
    except Exception:
        pass

    estimator = CouplingEstimator(config)
    print(f"[GPU {gpu_id}] {direction} starting on {device}...", flush=True)
    try:
        result = estimator.analyze_session(session_data, direction)
        save_result(result, result_path)
        print(f"[GPU {gpu_id}] {direction} done.", flush=True)
    except Exception as e:
        is_cuda_error = ('CUDA' in str(e) or 'Triton' in str(e)
                         or 'illegal memory' in str(e))
        if is_cuda_error:
            print(f"[GPU {gpu_id}] {direction} CUDA error: {e}", flush=True)
            print(f"[GPU {gpu_id}] Retrying with Triton disabled...", flush=True)
            try:
                # Reset CUDA state
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()

            # Disable Triton and retry
            from cadence.regression.ewls import disable_triton_scan
            disable_triton_scan()
            estimator2 = CouplingEstimator(config)
            try:
                result = estimator2.analyze_session(session_data, direction)
                save_result(result, result_path)
                print(f"[GPU {gpu_id}] {direction} done (Triton-free retry).", flush=True)
            except Exception as e2:
                import traceback
                print(f"[GPU {gpu_id}] {direction} FAILED on retry: {e2}", flush=True)
                traceback.print_exc()
                sys.exit(1)
        else:
            import traceback
            print(f"[GPU {gpu_id}] {direction} FAILED: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def get_role_directions(session):
    """Map p1/p2 to therapist/patient based on session metadata."""
    p1_role = session.get('p1_role', 'therapist')
    if p1_role == 'therapist':
        return {
            'therapist_to_patient': 'p1_to_p2',
            'patient_to_therapist': 'p2_to_p1',
        }
    else:
        return {
            'therapist_to_patient': 'p2_to_p1',
            'patient_to_therapist': 'p1_to_p2',
        }


def main():
    parser = argparse.ArgumentParser(description='CADENCE all-sessions analysis')
    parser.add_argument('--config', default=None)
    parser.add_argument('--output', default='results/cadence_all_sessions')
    parser.add_argument('--sessions', nargs='+', default=None,
                        help='Specific sessions to run (partial name match)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--parallel-directions', action='store_true',
                        help='Run p1->p2 and p2->p1 on separate GPUs (requires 2+ GPUs)')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device
    mod_short = MOD_SHORT_V2
    modality_order = MODALITY_ORDER_V2

    # Auto-cache any raw XDF files not yet in session_cache
    raw_dir = config.get('raw_sessions')
    if raw_dir:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(raw_dir):
            raw_dir = os.path.join(project_root, raw_dir)
        n_new = ensure_all_cached(raw_dir, config['session_cache'])
        if n_new:
            print(f"Cached {n_new} new session(s) from {raw_dir}", flush=True)

    # Discover sessions
    all_sessions = discover_cached_sessions(config['session_cache'])
    excluded = set(config.get('excluded_sessions', []))

    if args.sessions:
        sessions = [(n, p) for n, p in all_sessions
                    if any(s in n for s in args.sessions)]
    else:
        sessions = [(n, p) for n, p in all_sessions if n not in excluded]

    # Check for parallel directions (dual-GPU)
    parallel_dirs = getattr(args, 'parallel_directions', False)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if parallel_dirs and n_gpus >= 2:
        print(f"Parallel directions: p1->p2 on cuda:0, p2->p1 on cuda:1", flush=True)
        mp.set_start_method('spawn', force=True)
    elif parallel_dirs:
        print(f"Warning: --parallel-directions requires 2+ GPUs, found {n_gpus}. "
              f"Falling back to concurrent CUDA streams.", flush=True)
        parallel_dirs = False

    print(f"Running CADENCE on {len(sessions)} sessions", flush=True)
    os.makedirs(args.output, exist_ok=True)

    estimator = CouplingEstimator(config) if not parallel_dirs else None
    grand_summary = []
    condition_summaries = []  # (name, cond_summary) for grand aggregation

    for sess_idx, (name, cache_path) in enumerate(sessions):
        print(f"\n{'='*60}", flush=True)
        print(f"Session {sess_idx+1}/{len(sessions)}: {name}", flush=True)
        print(f"{'='*60}", flush=True)

        session = load_session_from_cache(cache_path, config=config)
        sess_dir = os.path.join(args.output, name)
        os.makedirs(sess_dir, exist_ok=True)

        # Determine role-based directions
        role_dirs = get_role_directions(session)
        p1_role = session.get('p1_role', 'therapist')
        print(f"  Roles: P1={p1_role}, P2={'patient' if p1_role == 'therapist' else 'therapist'}", flush=True)

        results = {}  # role_label -> CouplingResult

        if parallel_dirs and len(role_dirs) == 2:
            # Dual-GPU: each direction on its own GPU (full 48GB native VRAM)
            t0 = time.time()
            devices = ['cuda:0', 'cuda:1']
            tmp_paths = {}
            procs = []
            for gpu_id, (role_label, p_direction) in enumerate(role_dirs.items()):
                tmp_path = os.path.join(sess_dir, f'_tmp_{p_direction}.npz')
                tmp_paths[role_label] = (p_direction, tmp_path)
                p = mp.Process(
                    target=_parallel_direction_worker,
                    args=(gpu_id, p_direction, session, args.config,
                          devices[gpu_id], tmp_path),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()
            dt = time.time() - t0

            # Check for worker failures
            failed = [p for p in procs if p.exitcode != 0]
            if failed:
                print(f"\n  WARNING: {len(failed)} worker(s) crashed", flush=True)

            print(f"\n  Both directions completed in {dt:.1f}s (parallel GPUs)", flush=True)

            # Load results back from temp files
            for role_label, (p_direction, tmp_path) in tmp_paths.items():
                if not os.path.exists(tmp_path):
                    print(f"  {role_label}: SKIPPED (worker crashed, no result file)", flush=True)
                    continue
                result, _meta = load_result(tmp_path)
                result.direction = p_direction
                short_label = 'T->P' if role_label == 'therapist_to_patient' else 'P->T'
                n_sig = result.n_significant_pathways
                total = len(result.pathway_dr2)
                print(f"  {short_label}: {n_sig}/{total} significant", flush=True)

                for key, is_sig in sorted(result.pathway_significant.items()):
                    dr2_mean = np.nanmean(result.pathway_dr2[key]) if key in result.pathway_dr2 else 0.0
                    src_s = mod_short.get(key[0], key[0])
                    tgt_s = mod_short.get(key[1], key[1])
                    sig = '***' if is_sig else ''
                    print(f"      {src_s}->{tgt_s}: {dr2_mean:+.4f} {sig}", flush=True)

                results[role_label] = result

                # Save permanent result files
                prefix = role_label.replace('_to_', '-')
                try:
                    npz_path = os.path.join(sess_dir, f'{prefix}_full.npz')
                    save_result(result, npz_path, session_name=name,
                                runtime_s=dt / 2)
                    print(f"    Saved NPZ: {npz_path}", flush=True)

                    det = detection_summary(result, config)
                    det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}
                    result_data = {
                        'direction': role_label,
                        'session': name,
                        'n_significant': n_sig,
                        'duration_s': session.get('duration', 0),
                        'analysis_time_s': dt / 2,
                        'pathway_summary': {},
                        'detection': det_json,
                    }
                    for key2 in result.pathway_dr2:
                        pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                        result_data['pathway_summary'][pkey] = {
                            'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                            'significant': bool(result.pathway_significant.get(key2, False)),
                        }
                    json_path = os.path.join(sess_dir, f'{prefix}_results.json')
                    with open(json_path, 'w') as f:
                        json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                except Exception as save_err:
                    print(f"    SAVE ERROR: {save_err}", flush=True)

                # Visualizations (non-critical)
                try:
                    fig = plot_coupling_timecourse(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_timecourse.png'))
                    plt.close(fig)
                    fig = plot_coupling_matrix(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_matrix.png'))
                    plt.close(fig)
                    plot_coupling_kernels(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_kernels.png'))
                    plot_sparsity_summary(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_sparsity.png'))
                    plot_block_detail(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_blocks.png'))
                except Exception as viz_err:
                    print(f"    Visualization error (non-fatal): {viz_err}", flush=True)

                # Store pathway summary for grand aggregation
                sess_summary = {
                    'session': name,
                    'direction': role_label,
                    'p_direction': p_direction,
                    'n_significant': n_sig,
                    'analysis_time_s': dt / 2,
                    'pathways': {},
                }
                for key2 in result.pathway_dr2:
                    pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                    sess_summary['pathways'][pkey] = {
                        'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                        'significant': bool(result.pathway_significant.get(key2, False)),
                    }
                grand_summary.append(sess_summary)

                # Clean up temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        elif 'cuda' in str(config.get('device', 'cpu')) and len(role_dirs) == 2:
            # Single GPU: both directions via concurrent CUDA streams
            t0 = time.time()
            both = estimator.analyze_session_both(session)
            dt = time.time() - t0
            print(f"\n  Both directions completed in {dt:.1f}s", flush=True)

            for role_label, p_direction in role_dirs.items():
                result = both[p_direction]
                short_label = 'T->P' if role_label == 'therapist_to_patient' else 'P->T'
                n_sig = result.n_significant_pathways
                total = len(result.pathway_dr2)
                print(f"  {short_label}: {n_sig}/{total} significant", flush=True)

                for key, is_sig in sorted(result.pathway_significant.items()):
                    dr2_mean = np.nanmean(result.pathway_dr2[key]) if key in result.pathway_dr2 else 0.0
                    src_s = mod_short.get(key[0], key[0])
                    tgt_s = mod_short.get(key[1], key[1])
                    sig = '***' if is_sig else ''
                    print(f"      {src_s}->{tgt_s}: {dr2_mean:+.4f} {sig}", flush=True)

                results[role_label] = result

                # Save results first (before viz)
                prefix = role_label.replace('_to_', '-')
                try:
                    npz_path = os.path.join(sess_dir, f'{prefix}_full.npz')
                    save_result(result, npz_path, session_name=name,
                                runtime_s=dt / 2)
                    print(f"    Saved NPZ: {npz_path}", flush=True)

                    det = detection_summary(result, config)
                    det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}
                    result_data = {
                        'direction': role_label,
                        'session': name,
                        'n_significant': n_sig,
                        'duration_s': session.get('duration', 0),
                        'analysis_time_s': dt / 2,
                        'pathway_summary': {},
                        'detection': det_json,
                    }
                    for key2 in result.pathway_dr2:
                        pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                        result_data['pathway_summary'][pkey] = {
                            'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                            'significant': bool(result.pathway_significant.get(key2, False)),
                        }
                    json_path = os.path.join(sess_dir, f'{prefix}_results.json')
                    with open(json_path, 'w') as f:
                        json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                except Exception as save_err:
                    print(f"    SAVE ERROR: {save_err}", flush=True)

                # Visualizations (non-critical)
                try:
                    fig = plot_coupling_timecourse(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_timecourse.png'))
                    plt.close(fig)
                    fig = plot_coupling_matrix(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_matrix.png'))
                    plt.close(fig)
                    plot_coupling_kernels(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_kernels.png'))
                    plot_sparsity_summary(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_sparsity.png'))
                    plot_block_detail(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_blocks.png'))
                except Exception as viz_err:
                    print(f"    Visualization error (non-fatal): {viz_err}", flush=True)

                # Store pathway summary for grand aggregation
                sess_summary = {
                    'session': name,
                    'direction': role_label,
                    'p_direction': p_direction,
                    'n_significant': n_sig,
                    'analysis_time_s': dt / 2,
                    'pathways': {},
                }
                for key2 in result.pathway_dr2:
                    pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                    sess_summary['pathways'][pkey] = {
                        'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                        'significant': bool(result.pathway_significant.get(key2, False)),
                    }
                grand_summary.append(sess_summary)
        else:
            for role_label, p_direction in role_dirs.items():
                short_label = 'T->P' if role_label == 'therapist_to_patient' else 'P->T'
                print(f"\n  {short_label} ({role_label}):", flush=True)
                t0 = time.time()
                result = estimator.analyze_session(session, p_direction)
                dt = time.time() - t0

                n_sig = result.n_significant_pathways
                total = len(result.pathway_dr2)
                print(f"    {n_sig}/{total} significant pathways ({dt:.1f}s)", flush=True)

                for key, is_sig in sorted(result.pathway_significant.items()):
                    dr2_mean = np.nanmean(result.pathway_dr2[key]) if key in result.pathway_dr2 else 0.0
                    src_s = mod_short.get(key[0], key[0])
                    tgt_s = mod_short.get(key[1], key[1])
                    sig = '***' if is_sig else ''
                    print(f"      {src_s}->{tgt_s}: {dr2_mean:+.4f} {sig}", flush=True)

                results[role_label] = result

                # Save results first (before viz)
                prefix = role_label.replace('_to_', '-')
                try:
                    npz_path = os.path.join(sess_dir, f'{prefix}_full.npz')
                    save_result(result, npz_path, session_name=name,
                                runtime_s=dt)
                    print(f"    Saved NPZ: {npz_path}", flush=True)

                    det = detection_summary(result, config)
                    det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}
                    result_data = {
                        'direction': role_label,
                        'session': name,
                        'n_significant': n_sig,
                        'duration_s': session.get('duration', 0),
                        'analysis_time_s': dt,
                        'pathway_summary': {},
                        'detection': det_json,
                    }
                    for key2 in result.pathway_dr2:
                        pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                        result_data['pathway_summary'][pkey] = {
                            'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                            'significant': bool(result.pathway_significant.get(key2, False)),
                        }
                    json_path = os.path.join(sess_dir, f'{prefix}_results.json')
                    with open(json_path, 'w') as f:
                        json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                except Exception as save_err:
                    print(f"    SAVE ERROR: {save_err}", flush=True)

                # Visualizations (non-critical)
                try:
                    fig = plot_coupling_timecourse(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_timecourse.png'))
                    plt.close(fig)
                    fig = plot_coupling_matrix(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_matrix.png'))
                    plt.close(fig)
                    plot_coupling_kernels(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_kernels.png'))
                    plot_sparsity_summary(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_sparsity.png'))
                    plot_block_detail(
                        result, save_path=os.path.join(sess_dir, f'{prefix}_blocks.png'))
                except Exception as viz_err:
                    print(f"    Visualization error (non-fatal): {viz_err}", flush=True)

                # Store pathway summary for grand aggregation
                sess_summary = {
                    'session': name,
                    'direction': role_label,
                    'p_direction': p_direction,
                    'n_significant': n_sig,
                    'analysis_time_s': dt,
                    'pathways': {},
                }
                for key2 in result.pathway_dr2:
                    pkey = f"{mod_short.get(key2[0], key2[0])}->{mod_short.get(key2[1], key2[1])}"
                    sess_summary['pathways'][pkey] = {
                        'mean_dr2': float(np.nanmean(result.pathway_dr2[key2])),
                        'significant': bool(result.pathway_significant.get(key2, False)),
                    }
                grand_summary.append(sess_summary)

        # Condition-level analysis (needs both directions)
        if 'therapist_to_patient' not in results or 'patient_to_therapist' not in results:
            missing = [k for k in ['therapist_to_patient', 'patient_to_therapist']
                       if k not in results]
            print(f"\n  Condition analysis: SKIPPED (missing {missing})", flush=True)
            continue

        result_tp = results['therapist_to_patient']
        result_pt = results['patient_to_therapist']

        print(f"\n  Condition analysis:", flush=True)
        cond_summary = session_condition_summary(
            result_tp, result_pt, session,
            modality_order=modality_order, mod_short=mod_short)
        condition_summaries.append((name, cond_summary))

        # Print condition breakdown (Overall modality summary)
        overall_stats = cond_summary.get('modality_summary', {}).get('Overall', {})
        for cond, stats in overall_stats.items():
            if cond in ('uncategorized',):
                continue
            label = stats['label']
            print(f"    {label:25s}: "
                  f"sync={stats['sync_pct']:4.1f}%  "
                  f"T-leads={stats['t_leads_pct']:4.1f}%  "
                  f"P-leads={stats['p_leads_pct']:4.1f}%  "
                  f"none={stats['none_pct']:4.1f}%  "
                  f"dR2(T->P)={stats['mean_dr2_tp']:+.3f}  "
                  f"dR2(P->T)={stats['mean_dr2_pt']:+.3f}")

        # Save condition summary JSON
        cond_path = os.path.join(sess_dir, 'condition_summary.json')
        with open(cond_path, 'w') as f:
            json.dump(cond_summary, f, indent=2, cls=NumpyEncoder)

    # === Grand aggregation ===
    print(f"\n{'='*60}", flush=True)
    print("GRAND CONDITION AVERAGES", flush=True)
    print(f"{'='*60}", flush=True)

    grand_cond = aggregate_condition_summaries(condition_summaries)

    # Print grand condition summary
    for cond, cond_data in grand_cond['conditions'].items():
        overall = cond_data.get('modalities', {}).get('Overall', {})
        if not overall:
            continue
        n = overall['n_sessions']
        label = cond_data['label']
        print(f"\n  {label} (n={n} sessions):", flush=True)
        print(f"    {'Metric':15s}  {'Mean':>7s}  {'Std':>6s}", flush=True)
        print(f"    {'-'*35}", flush=True)
        for stat in ['sync_pct', 't_leads_pct', 'p_leads_pct', 'none_pct',
                      'mean_dr2_tp', 'mean_dr2_pt', 'coupling_pct']:
            mean_val = overall.get(f'{stat}_mean', 0)
            std_val = overall.get(f'{stat}_std', 0)
            label_short = stat.replace('_pct', '%').replace('mean_', '')
            if 'dr2' in stat:
                print(f"    {label_short:15s}  {mean_val:+7.4f}  {std_val:6.4f}", flush=True)
            else:
                print(f"    {label_short:15s}  {mean_val:7.1f}  {std_val:6.1f}", flush=True)

    # Per-modality breakdown for key conditions
    print(f"\n{'='*60}", flush=True)
    print("PER-MODALITY CONDITION BREAKDOWN", flush=True)
    print(f"{'='*60}", flush=True)

    for cond, cond_data in grand_cond['conditions'].items():
        if cond in ('uncategorized',):
            continue
        mods = cond_data.get('modalities', {})
        if not mods or 'Overall' not in mods:
            continue

        n = mods['Overall']['n_sessions']
        print(f"\n  {cond_data['label']} (n={n}):", flush=True)
        print(f"    {'Target':>8s}  {'Sync%':>6s}  {'T-lead%':>7s}  {'P-lead%':>7s}  "
              f"{'None%':>6s}  {'dR2(T->P)':>9s}  {'dR2(P->T)':>9s}")
        print(f"    {'-'*60}", flush=True)

        mod_display = [mod_short.get(m, m) for m in modality_order] + ['Overall']
        for mod_key in mod_display:
            if mod_key not in mods:
                continue
            m = mods[mod_key]
            print(f"    {mod_key:>8s}  "
                  f"{m['sync_pct_mean']:5.1f}%  "
                  f"{m['t_leads_pct_mean']:6.1f}%  "
                  f"{m['p_leads_pct_mean']:6.1f}%  "
                  f"{m['none_pct_mean']:5.1f}%  "
                  f"{m['mean_dr2_tp_mean']:+8.4f}  "
                  f"{m['mean_dr2_pt_mean']:+8.4f}")

    # Save grand summaries
    grand_path = os.path.join(args.output, 'grand_summary.json')
    with open(grand_path, 'w') as f:
        json.dump(grand_summary, f, indent=2, cls=NumpyEncoder)

    grand_cond_path = os.path.join(args.output, 'grand_condition_summary.json')
    with open(grand_cond_path, 'w') as f:
        json.dump(grand_cond, f, indent=2, cls=NumpyEncoder)

    # === Grand average visualizations ===
    print(f"\nGenerating grand average visualizations...", flush=True)

    fig = plot_grand_classification_bars(
        grand_cond, save_path=os.path.join(args.output, 'grand_classification_bars.png'))
    if fig:
        plt.close(fig)
        print("  Saved: grand_classification_bars.png", flush=True)

    fig = plot_grand_dr2_bars(
        grand_cond, save_path=os.path.join(args.output, 'grand_dr2_bars.png'))
    if fig:
        plt.close(fig)
        print("  Saved: grand_dr2_bars.png", flush=True)

    fig = plot_grand_coupling_matrix(
        grand_summary, save_path=os.path.join(args.output, 'grand_coupling_matrix.png'))
    if fig:
        plt.close(fig)
        print("  Saved: grand_coupling_matrix.png", flush=True)

    fig = plot_grand_coupling_by_condition(
        grand_cond, save_path=os.path.join(args.output, 'grand_coupling_by_condition.png'))
    if fig:
        plt.close(fig)
        print("  Saved: grand_coupling_by_condition.png", flush=True)

    print(f"\nSaved: {grand_path}", flush=True)
    print(f"Saved: {grand_cond_path}", flush=True)
    print(f"Per-session results in: {args.output}/{{session_name}}/", flush=True)


if __name__ == '__main__':
    main()
