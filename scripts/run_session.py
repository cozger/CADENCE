"""CADENCE single-session analysis.

Usage:
    python scripts/run_session.py --session y_06
    python scripts/run_session.py --session y_17 --config configs/default.yaml
    python scripts/run_session.py --session y_06 --no-surrogates  # fast, F-test only
"""

import argparse
import json
import os
import sys
import time

# Force unbuffered output so progress is visible immediately
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.significance.detection import detection_summary
from cadence.visualization.timecourse import plot_coupling_timecourse
from cadence.visualization.heatmaps import plot_coupling_matrix
from cadence.visualization.kernels import plot_coupling_kernels
from cadence.visualization.sparsity import plot_sparsity_summary, plot_block_detail
from cadence.coupling.serialization import save_result, get_peak_gpu_mb, get_rss_mb
from cadence.constants import MOD_SHORT


def _get_role_directions(session):
    """Map p1/p2 directions to role-based labels."""
    p1_role = session.get('p1_role', 'therapist')
    if p1_role == 'therapist':
        return {
            'p1_to_p2': 'Therapist \u2192 Patient',
            'p2_to_p1': 'Patient \u2192 Therapist',
        }
    else:
        return {
            'p1_to_p2': 'Patient \u2192 Therapist',
            'p2_to_p1': 'Therapist \u2192 Patient',
        }


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


def find_session(session_name, cache_dir):
    """Find a session by partial name match in cache."""
    sessions = discover_cached_sessions(cache_dir)
    for name, path in sessions:
        if session_name in name:
            return name, path
    return None, None


def _direction_worker(gpu_id, direction, session_data, config_path, device,
                      output_dir, session_name, pipeline):
    """Worker: run one direction on a specific GPU."""
    config = load_config(config_path)
    config['device'] = device
    if pipeline:
        config['pipeline'] = pipeline

    # Phase 7: limit per-process VRAM when sharing a single GPU
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        try:
            torch.cuda.set_per_process_memory_fraction(0.48, device=0)
        except Exception:
            pass  # already set or not supported

    session = session_data  # already loaded, passed via fork

    # Warm up Triton JIT compilation with a tiny dummy EWLS solve.
    # Without this, both direction processes race to compile the same
    # Triton kernels via the shared ~/.triton/cache/, causing
    # "illegal memory access" on the loser.
    try:
        from cadence.regression.ewls import EWLSSolver
        _ws = EWLSSolver(tau_seconds=10.0, lambda_ridge=1e-3,
                         eval_rate=2.0, device=device)
        _X = torch.randn(1, 20, 5, device=device)
        _y = torch.randn(1, 20, 1, device=device)
        _ws.solve_batched(_X, _y)
        del _ws, _X, _y
        torch.cuda.empty_cache()
    except Exception:
        pass

    estimator = CouplingEstimator(config)
    role_dirs = _get_role_directions(session)
    role_label = role_dirs.get(direction, direction)
    print(f"[GPU {gpu_id}] Analyzing {role_label}...", flush=True)
    rss_before = get_rss_mb()
    _ = get_peak_gpu_mb()  # reset peak counter
    t0 = time.time()

    result = estimator.analyze_session(session, direction)
    dt = time.time() - t0
    peak_gpu = get_peak_gpu_mb()
    peak_cpu = get_rss_mb() - rss_before
    result.direction = role_label

    n_sig = result.n_significant_pathways
    total = len(result.pathway_dr2)
    print(f"[GPU {gpu_id}] {role_label}: {n_sig}/{total} significant in {dt:.1f}s", flush=True)

    for key, is_sig in sorted(result.pathway_significant.items()):
        dr2_mean = np.nanmean(result.pathway_dr2[key])
        src_short = MOD_SHORT.get(key[0], key[0])
        tgt_short = MOD_SHORT.get(key[1], key[1])
        sig_str = '***' if is_sig else '   '
        print(f"[GPU {gpu_id}]   {src_short:>4s} -> {tgt_short:<4s}: dR2={dr2_mean:+.4f} {sig_str}", flush=True)

    # Generate visualizations
    prefix = direction.replace('_to_', '-')
    plot_coupling_timecourse(
        result, save_path=os.path.join(output_dir, f'{prefix}_timecourse.png'))
    plot_coupling_matrix(
        result, save_path=os.path.join(output_dir, f'{prefix}_matrix.png'))
    plot_coupling_kernels(
        result, save_path=os.path.join(output_dir, f'{prefix}_kernels.png'))
    plot_sparsity_summary(
        result, save_path=os.path.join(output_dir, f'{prefix}_sparsity.png'))
    plot_block_detail(
        result, save_path=os.path.join(output_dir, f'{prefix}_blocks.png'))

    # Detection summary
    det = detection_summary(result, config)
    det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}

    # Save results as JSON
    result_data = {
        'direction': role_label,
        'session': session_name,
        'n_significant': n_sig,
        'duration_s': session.get('duration', 0),
        'analysis_time_s': dt,
        'gpu': gpu_id,
        'pathway_summary': {},
        'detection': det_json,
    }
    for key in result.pathway_dr2:
        src_short = MOD_SHORT.get(key[0], key[0])
        tgt_short = MOD_SHORT.get(key[1], key[1])
        pkey = f'{src_short}->{tgt_short}'
        result_data['pathway_summary'][pkey] = {
            'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
            'significant': result.pathway_significant.get(key, False),
        }

    json_path = os.path.join(output_dir, f'{prefix}_results.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NumpyEncoder)

    # Save full result for offline re-plotting
    npz_path = os.path.join(output_dir, f'{prefix}_full.npz')
    save_result(result, npz_path, session_name=session_name,
                runtime_s=dt, peak_gpu_mb=peak_gpu, peak_cpu_mb=peak_cpu)
    print(f"[GPU {gpu_id}] Saved: {json_path}, {npz_path}", flush=True)


def run_session(args):
    config = load_config(args.config)
    if args.device:
        config['device'] = args.device
    if hasattr(args, 'pipeline') and args.pipeline:
        config['pipeline'] = args.pipeline

    # Find session
    name, cache_path = find_session(args.session, config['session_cache'])
    if name is None:
        print(f"Session '{args.session}' not found in {config['session_cache']}", flush=True)
        sessions = discover_cached_sessions(config['session_cache'])
        print(f"Available: {[s[0] for s in sessions]}", flush=True)
        return

    print(f"Loading session: {name}", flush=True)
    session = load_session_from_cache(cache_path, config=config)
    print(f"  Duration: {session.get('duration', 0):.0f}s", flush=True)

    # Output directory
    output_dir = os.path.join(args.output, name)
    os.makedirs(output_dir, exist_ok=True)

    # Check for parallel directions mode
    parallel_dirs = getattr(args, 'parallel_directions', False)
    n_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if parallel_dirs:
        if n_gpus_available >= 2:
            print(f"Running both directions in parallel on 2 GPUs...", flush=True)
            devices = ['cuda:0', 'cuda:1']
        else:
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Running both directions in parallel on {device}...", flush=True)
            devices = [device, device]

        mp.set_start_method('spawn', force=True)

        processes = []
        for gpu_id, (direction, dev) in enumerate(
                zip(['p1_to_p2', 'p2_to_p1'], devices)):
            p = mp.Process(
                target=_direction_worker,
                args=(gpu_id, direction, session, args.config,
                      dev, output_dir, name,
                      getattr(args, 'pipeline', None)),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"\nAll results saved to: {output_dir}", flush=True)
        return

    # Concurrent mode: both directions on separate CUDA streams (single GPU)
    # Default ON for CUDA devices; --no-concurrent to disable
    concurrent = getattr(args, 'concurrent', True)
    no_concurrent = getattr(args, 'no_concurrent', False)
    if no_concurrent:
        concurrent = False
    device_str = config.get('device', 'cpu')
    if 'cuda' not in str(device_str):
        concurrent = False
    estimator = CouplingEstimator(config)
    directions = ['p1_to_p2', 'p2_to_p1']
    role_dirs = _get_role_directions(session)
    all_results = {}

    if concurrent:
        print(f"\nRunning both directions concurrently (CUDA streams)...", flush=True)
        rss_before = get_rss_mb()
        _ = get_peak_gpu_mb()
        t0 = time.time()

        both_results = estimator.analyze_session_both(session)

        dt = time.time() - t0
        peak_gpu = get_peak_gpu_mb()
        peak_cpu = get_rss_mb() - rss_before
        print(f"  Both directions completed in {dt:.1f}s"
              f"  (GPU: {peak_gpu:.0f} MB, CPU: {peak_cpu:+.0f} MB)")

        for direction in directions:
            result = both_results[direction]
            role_label = role_dirs.get(direction, direction)
            result.direction = role_label
            n_sig = result.n_significant_pathways
            total = len(result.pathway_dr2)
            print(f"\n  {role_label}: {n_sig}/{total} significant", flush=True)
            for key, is_sig in sorted(result.pathway_significant.items()):
                dr2_mean = np.nanmean(result.pathway_dr2[key])
                src_short = MOD_SHORT.get(key[0], key[0])
                tgt_short = MOD_SHORT.get(key[1], key[1])
                sig_str = '***' if is_sig else '   '
                print(f"    {src_short:>4s} -> {tgt_short:<4s}:"
                      f" dR2={dr2_mean:+.4f} {sig_str}", flush=True)
            all_results[direction] = result

            # Generate visualizations
            prefix = direction.replace('_to_', '-')
            plot_coupling_timecourse(
                result, save_path=os.path.join(
                    output_dir, f'{prefix}_timecourse.png'))
            plot_coupling_matrix(
                result, save_path=os.path.join(
                    output_dir, f'{prefix}_matrix.png'))
            plot_coupling_kernels(
                result, save_path=os.path.join(
                    output_dir, f'{prefix}_kernels.png'))
            plot_sparsity_summary(
                result, save_path=os.path.join(
                    output_dir, f'{prefix}_sparsity.png'))
            plot_block_detail(
                result, save_path=os.path.join(
                    output_dir, f'{prefix}_blocks.png'))

            det = detection_summary(result, config)
            det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}

            result_data = {
                'direction': role_label,
                'session': name,
                'n_significant': n_sig,
                'duration_s': session.get('duration', 0),
                'analysis_time_s': dt / 2,  # approximate per-direction
                'pathway_summary': {},
                'detection': det_json,
            }
            for key, is_sig in result.pathway_significant.items():
                pkey = f'{key[0]}->{key[1]}'
                result_data['pathway_summary'][pkey] = {
                    'significant': is_sig,
                    'mean_dr2': float(np.nanmean(result.pathway_dr2.get(
                        key, np.array([0.0])))),
                }
            json_path = os.path.join(output_dir, f'{prefix}_results.json')
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2, cls=NumpyEncoder)

            npz_path = os.path.join(output_dir, f'{prefix}_full.npz')
            save_result(result, npz_path, session_name=name,
                        runtime_s=dt / 2, peak_gpu_mb=peak_gpu,
                        peak_cpu_mb=peak_cpu)

        print(f"\nAll results saved to: {output_dir}", flush=True)
        return

    # Sequential mode (original behavior)
    for direction in directions:
        role_label = role_dirs.get(direction, direction)
        print(f"\nAnalyzing {role_label}...", flush=True)
        rss_before = get_rss_mb()
        _ = get_peak_gpu_mb()  # reset peak counter
        t0 = time.time()

        result = estimator.analyze_session(session, direction)
        dt = time.time() - t0
        peak_gpu = get_peak_gpu_mb()
        peak_cpu = get_rss_mb() - rss_before
        result.direction = role_label
        print(f"  Completed in {dt:.1f}s  (GPU: {peak_gpu:.0f} MB, CPU: {peak_cpu:+.0f} MB)", flush=True)

        # Print summary
        n_sig = result.n_significant_pathways
        total = len(result.pathway_dr2)
        print(f"  Significant pathways: {n_sig}/{total}", flush=True)

        for key, is_sig in sorted(result.pathway_significant.items()):
            dr2_mean = np.nanmean(result.pathway_dr2[key])
            src_short = MOD_SHORT.get(key[0], key[0])
            tgt_short = MOD_SHORT.get(key[1], key[1])
            sig_str = '***' if is_sig else '   '
            print(f"    {src_short:>4s} -> {tgt_short:<4s}: dR2={dr2_mean:+.4f} {sig_str}", flush=True)

        all_results[direction] = result

        # Generate visualizations
        prefix = direction.replace('_to_', '-')
        plot_coupling_timecourse(
            result, save_path=os.path.join(output_dir, f'{prefix}_timecourse.png'))
        plot_coupling_matrix(
            result, save_path=os.path.join(output_dir, f'{prefix}_matrix.png'))
        plot_coupling_kernels(
            result, save_path=os.path.join(output_dir, f'{prefix}_kernels.png'))
        plot_sparsity_summary(
            result, save_path=os.path.join(output_dir, f'{prefix}_sparsity.png'))
        plot_block_detail(
            result, save_path=os.path.join(output_dir, f'{prefix}_blocks.png'))

        # Detection summary
        det = detection_summary(result, config)
        det_json = {f"{k[0]}->{k[1]}": v for k, v in det.items()}

        # Save results as JSON
        result_data = {
            'direction': role_label,
            'session': name,
            'n_significant': n_sig,
            'duration_s': session.get('duration', 0),
            'analysis_time_s': dt,
            'pathway_summary': {},
            'detection': det_json,
        }
        for key in result.pathway_dr2:
            src_short = MOD_SHORT.get(key[0], key[0])
            tgt_short = MOD_SHORT.get(key[1], key[1])
            pkey = f'{src_short}->{tgt_short}'
            result_data['pathway_summary'][pkey] = {
                'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
                'significant': result.pathway_significant.get(key, False),
            }

        json_path = os.path.join(output_dir, f'{prefix}_results.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)

        # Save full result for offline re-plotting
        npz_path = os.path.join(output_dir, f'{prefix}_full.npz')
        save_result(result, npz_path, session_name=name,
                    runtime_s=dt, peak_gpu_mb=peak_gpu, peak_cpu_mb=peak_cpu)
        print(f"  Saved: {json_path}, {npz_path}", flush=True)

    print(f"\nAll results saved to: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='CADENCE single-session analysis')
    parser.add_argument('--session', default='y_06', help='Session name (partial match)')
    parser.add_argument('--config', default=None, help='YAML config file')
    parser.add_argument('--output', default='results/cadence', help='Output directory')
    parser.add_argument('--device', default=None, help='Device override (cpu/cuda)')
    parser.add_argument('--pipeline', default=None, choices=['v1', 'v2'],
                        help='Pipeline version (v1=original, v2=wavelet+group lasso)')
    parser.add_argument('--no-surrogates', action='store_true',
                        help='Skip surrogate testing (F-test only)')
    parser.add_argument('--parallel-directions', action='store_true',
                        help='Run p1->p2 and p2->p1 on separate GPUs (requires 2+ GPUs)')
    parser.add_argument('--concurrent', action='store_true', default=True,
                        help='Run both directions concurrently via CUDA streams (default: ON)')
    parser.add_argument('--no-concurrent', action='store_true',
                        help='Disable concurrent directions (run sequentially)')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    run_session(args)


if __name__ == '__main__':
    main()
