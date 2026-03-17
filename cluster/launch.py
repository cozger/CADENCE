"""CADENCE cluster launcher — dispatches to analysis scripts with GPU assignment.

No DDP needed: CADENCE runs independent regressions per session, so we use
simple torch.multiprocessing with one process per GPU.

Invoked by Slurm sbatch scripts as:
    python cluster/launch.py --script discovery --config configs/cluster.yaml --n-gpus 2
    python cluster/launch.py --script session --session y_06 --n-gpus 1
"""

import sys
import os
import argparse
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator


def parse_args():
    parser = argparse.ArgumentParser(description='CADENCE cluster launcher')
    parser.add_argument('--script', required=True,
                        choices=['discovery', 'session', 'all_sessions',
                                 'synthetic_v2', 'corpus'],
                        help='Which analysis script to run')
    parser.add_argument('--config', default='configs/cluster.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--output', default='results/cluster_discovery',
                        help='Output directory')
    parser.add_argument('--session', default=None,
                        help='Session name (for single-session mode)')
    parser.add_argument('--n-gpus', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration in seconds (for synthetic)')
    parser.add_argument('--min-sessions', type=int, default=None,
                        help='Min sessions for discovery consistency')
    parser.add_argument('--n-coupled', type=int, default=50,
                        help='Number of coupled sessions (corpus mode)')
    parser.add_argument('--n-null', type=int, default=20,
                        help='Number of null sessions (corpus mode)')
    return parser.parse_args()


def _gpu_worker_sessions(gpu_id, session_batch, config_path, output_dir):
    """Worker function: run analysis on a batch of sessions on one GPU."""
    device = f'cuda:{gpu_id}'
    config = load_config(config_path)
    config['device'] = device

    print(f"[GPU {gpu_id}] Processing {len(session_batch)} sessions on {device}")
    print(f"[GPU {gpu_id}] GPU: {torch.cuda.get_device_name(gpu_id)}")

    for session_name, cache_path in session_batch:
        print(f"\n[GPU {gpu_id}] Session: {session_name}")
        try:
            session = load_session_from_cache(cache_path, config=config)
            estimator = CouplingEstimator(config)

            for direction in ['p1_to_p2', 'p2_to_p1']:
                print(f"[GPU {gpu_id}]   {direction}...")
                result = estimator.analyze_session(session, direction)
                n_sig = result.n_significant_pathways
                total = len(result.pathway_dr2)
                print(f"[GPU {gpu_id}]   {n_sig}/{total} significant")

                # Save results
                sess_dir = os.path.join(output_dir, session_name)
                os.makedirs(sess_dir, exist_ok=True)

                import json
                import numpy as np
                prefix = direction.replace('_to_', '-')
                result_data = {
                    'direction': direction,
                    'session': session_name,
                    'n_significant': n_sig,
                    'gpu': gpu_id,
                    'pathway_summary': {},
                }
                for key in result.pathway_dr2:
                    pkey = f'{key[0]}->{key[1]}'
                    result_data['pathway_summary'][pkey] = {
                        'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
                        'significant': result.pathway_significant.get(key, False),
                    }
                json_path = os.path.join(sess_dir, f'{prefix}_results.json')
                with open(json_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                print(f"[GPU {gpu_id}]   Saved: {json_path}")

        except Exception:
            print(f"[GPU {gpu_id}] ERROR on {session_name}:")
            traceback.print_exc()
            continue

    print(f"\n[GPU {gpu_id}] Done.")


def run_discovery(args):
    """Run discovery pipeline with sessions split across GPUs."""
    from scripts.run_discovery import run_discovery as _run_discovery

    config = load_config(args.config)
    n_gpus = min(args.n_gpus, torch.cuda.device_count())

    if n_gpus <= 1:
        # Single GPU: use existing script directly
        print(f"Running discovery on 1 GPU...")
        class DiscoveryArgs:
            pass
        da = DiscoveryArgs()
        da.config = args.config
        da.output = args.output
        da.device = 'cuda:0'
        da.min_sessions = args.min_sessions
        _run_discovery(da)
        return

    # Multi-GPU: split sessions across GPUs for Stage 1 and Stage 2
    print(f"Running discovery on {n_gpus} GPUs...")

    # Discover sessions
    sessions_list = discover_cached_sessions(config['session_cache'])
    excluded = set(config.get('excluded_sessions', []))
    sessions_list = [(n, p) for n, p in sessions_list if n not in excluded]
    print(f"Found {len(sessions_list)} sessions")

    # Split sessions across GPUs (round-robin)
    gpu_batches = [[] for _ in range(n_gpus)]
    for i, session_info in enumerate(sessions_list):
        gpu_batches[i % n_gpus].append(session_info)

    for gpu_id, batch in enumerate(gpu_batches):
        names = [n for n, _ in batch]
        print(f"  GPU {gpu_id}: {names}")

    # Launch workers
    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_id, batch in enumerate(gpu_batches):
        if not batch:
            continue
        p = mp.Process(
            target=_gpu_worker_sessions,
            args=(gpu_id, batch, args.config, args.output),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        print(f"\nWARNING: {len(failed)} GPU worker(s) failed")
    else:
        print(f"\nAll {n_gpus} GPU workers completed successfully.")

    print(f"Results saved to: {args.output}")


def run_session(args):
    """Run single-session analysis."""
    from scripts.run_session import run_session as _run_session

    class SessionArgs:
        pass
    sa = SessionArgs()
    sa.config = args.config
    sa.session = args.session or 'y_06'
    sa.output = args.output
    sa.device = 'cuda:0'
    sa.no_surrogates = False
    _run_session(sa)


def run_all_sessions(args):
    """Run all sessions split across GPUs."""
    config = load_config(args.config)
    n_gpus = min(args.n_gpus, torch.cuda.device_count())

    sessions_list = discover_cached_sessions(config['session_cache'])
    excluded = set(config.get('excluded_sessions', []))
    sessions_list = [(n, p) for n, p in sessions_list if n not in excluded]
    print(f"Found {len(sessions_list)} sessions, using {n_gpus} GPU(s)")

    if n_gpus <= 1:
        _gpu_worker_sessions(0, sessions_list, args.config, args.output)
        return

    # Split across GPUs
    mp.set_start_method('spawn', force=True)
    gpu_batches = [[] for _ in range(n_gpus)]
    for i, session_info in enumerate(sessions_list):
        gpu_batches[i % n_gpus].append(session_info)

    processes = []
    for gpu_id, batch in enumerate(gpu_batches):
        if not batch:
            continue
        p = mp.Process(
            target=_gpu_worker_sessions,
            args=(gpu_id, batch, args.config, args.output),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"All sessions complete. Results in: {args.output}")


def run_synthetic(args):
    """Run synthetic validation."""
    from scripts.run_synthetic import main as _run_synthetic_main
    sys.argv = ['run_synthetic.py',
                '--config', args.config,
                '--device', 'cuda:0',
                '--output', args.output]
    if args.duration:
        sys.argv.extend(['--duration', str(args.duration)])
    _run_synthetic_main()


def _generate_one_session(args):
    """CPU worker: generate one synthetic session. Top-level for pickling."""
    name, kappa_dict, seed, duration, output_dir = args
    import time
    import numpy as np
    from cadence.synthetic import build_synthetic_session_v2

    t0 = time.time()
    session = build_synthetic_session_v2(duration, kappa_dict, seed=seed)
    elapsed = time.time() - t0

    # Save to npz so GPU workers can load without pickling large arrays
    npz_path = os.path.join(output_dir, '.cache', f'{name}.npz')
    arrays = {}
    for key, val in session.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
    # Separate non-array metadata; coupling_gates contain arrays too
    meta = {}
    gate_keys = []
    for k, v in session.items():
        if isinstance(v, np.ndarray):
            continue
        if k == 'coupling_gates' and isinstance(v, dict):
            # Save gate arrays into the npz under prefixed keys
            for gk, gv in v.items():
                if isinstance(gv, np.ndarray):
                    arr_key = f'__gate__{gk}'
                    arrays[arr_key] = gv
                    gate_keys.append(gk)
            meta[k] = list(v.keys())  # just store the gate names
        else:
            meta[k] = v

    np.savez(npz_path, **arrays)
    import json
    meta_path = os.path.join(output_dir, '.cache', f'{name}_meta.json')

    def _jsonify(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        return obj

    with open(meta_path, 'w') as f:
        json.dump(_jsonify(meta), f)

    print(f"  Generated {name} ({elapsed:.1f}s)", flush=True)
    return name, kappa_dict, seed, elapsed


def _load_cached_session(name, output_dir):
    """Load a session from npz + meta json cache."""
    import json
    import numpy as np
    npz_path = os.path.join(output_dir, '.cache', f'{name}.npz')
    meta_path = os.path.join(output_dir, '.cache', f'{name}_meta.json')
    with open(meta_path) as f:
        session = json.load(f)
    data = np.load(npz_path)
    for key in data.files:
        if key.startswith('__gate__'):
            # Reconstruct coupling_gates dict
            gate_name = key[len('__gate__'):]
            if 'coupling_gates' not in session or not isinstance(session['coupling_gates'], dict):
                session['coupling_gates'] = {}
            session['coupling_gates'][gate_name] = data[key]
        else:
            session[key] = data[key]
    # coupling_gates from meta is a list of gate names; replace with dict if needed
    if isinstance(session.get('coupling_gates'), list):
        session['coupling_gates'] = {}
    return session


def _gpu_analyze_corpus(gpu_id, session_names, specs_dict, config_path, output_dir):
    """GPU worker: analyze pre-generated sessions from cache."""
    import json
    import time
    import numpy as np
    device = f'cuda:{gpu_id}'

    config = load_config(config_path)
    config['device'] = device

    from cadence.coupling.estimator import CouplingEstimator

    print(f"[GPU {gpu_id}] Analyzing {len(session_names)} sessions on {device}")
    print(f"[GPU {gpu_id}] GPU: {torch.cuda.get_device_name(gpu_id)}", flush=True)

    estimator = CouplingEstimator(config)
    results = []

    for i, name in enumerate(session_names):
        kappa_dict, seed = specs_dict[name]
        print(f"[GPU {gpu_id}] [{i+1}/{len(session_names)}] Analyzing {name}",
              flush=True)
        t0 = time.time()
        try:
            session = _load_cached_session(name, output_dir)
            result = estimator.analyze_session(session, 'p1_to_p2')

            pathway_summary = {}
            for key in result.pathway_dr2:
                pkey = f'{key[0]}->{key[1]}'
                pathway_summary[pkey] = {
                    'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
                    'significant': result.pathway_significant.get(key, False),
                }

            elapsed = time.time() - t0
            result_data = {
                'name': name,
                'seed': seed,
                'duration': session.get('duration', 0),
                'kappa_dict': kappa_dict,
                'n_significant': result.n_significant_pathways,
                'pathway_summary': pathway_summary,
                'elapsed_s': round(elapsed, 1),
                'gpu': gpu_id,
            }
            results.append(result_data)

            json_path = os.path.join(output_dir, f'{name}.json')
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"[GPU {gpu_id}]   -> {result.n_significant_pathways} sig, "
                  f"{elapsed:.1f}s", flush=True)

            # Free session memory
            del session, result
            torch.cuda.empty_cache()

        except Exception:
            print(f"[GPU {gpu_id}]   -> FAILED")
            traceback.print_exc()

    gpu_results_path = os.path.join(output_dir, f'gpu{gpu_id}_results.json')
    with open(gpu_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[GPU {gpu_id}] Done. {len(results)} sessions analyzed.", flush=True)


def run_corpus(args):
    """Run corpus: Phase 1 parallel CPU generation, Phase 2 parallel GPU analysis."""
    import json
    import time
    from cadence.synthetic import plan_corpus_v2

    config = load_config(args.config)
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    duration = args.duration or 3000

    specs = plan_corpus_v2(n_coupled=args.n_coupled, n_null=args.n_null)
    total = len(specs)
    print(f"Corpus: {total} sessions "
          f"({args.n_coupled} coupled + {args.n_null} null), "
          f"duration={duration}s, {n_gpus} GPU(s)")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, '.cache'), exist_ok=True)

    # Build specs lookup
    specs_dict = {name: (kd, seed) for name, kd, seed in specs}

    # ── Phase 1: Generate all sessions in parallel on CPUs ──
    n_cpus = os.cpu_count() or 1
    n_workers = min(n_cpus, total, 120)  # Leave a few cores for system
    print(f"\nPhase 1: Generating {total} sessions "
          f"with {n_workers} CPU workers...", flush=True)

    gen_args = [(name, kd, seed, duration, args.output)
                for name, kd, seed in specs]

    t0 = time.time()
    ctx = mp.get_context('forkserver')
    with ctx.Pool(n_workers) as pool:
        gen_results = pool.map(_generate_one_session, gen_args)

    gen_elapsed = time.time() - t0
    print(f"Phase 1 complete: {len(gen_results)} sessions in {gen_elapsed:.0f}s\n",
          flush=True)

    # ── Phase 2: Analyze on GPUs ──
    session_names = [name for name, _, _, _ in gen_results]

    print(f"Phase 2: Analyzing on {n_gpus} GPU(s)...", flush=True)

    # Round-robin split across GPUs
    gpu_batches = [[] for _ in range(n_gpus)]
    for i, name in enumerate(session_names):
        gpu_batches[i % n_gpus].append(name)

    if n_gpus <= 1:
        _gpu_analyze_corpus(0, gpu_batches[0], specs_dict,
                            args.config, args.output)
    else:
        spawn_ctx = mp.get_context('spawn')
        processes = []
        for gpu_id, batch in enumerate(gpu_batches):
            if not batch:
                continue
            p = spawn_ctx.Process(
                target=_gpu_analyze_corpus,
                args=(gpu_id, batch, specs_dict, args.config, args.output),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            print(f"\nWARNING: {len(failed)} GPU worker(s) failed")

    # ── Merge results ──
    all_results = []
    for gpu_id in range(n_gpus):
        gpu_path = os.path.join(args.output, f'gpu{gpu_id}_results.json')
        if os.path.exists(gpu_path):
            with open(gpu_path) as f:
                all_results.extend(json.load(f))

    merged_path = os.path.join(args.output, 'corpus_all_results.json')
    with open(merged_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
    from run_corpus import compute_corpus_summary
    summary = compute_corpus_summary(all_results)
    summary_path = os.path.join(args.output, 'corpus_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Clean up cache
    import shutil
    cache_dir = os.path.join(args.output, '.cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Cleaned up session cache.")

    print(f"\nCorpus complete. {len(all_results)} sessions.")
    print(f"Results: {args.output}")
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()

    print(f"CADENCE Cluster Launcher")
    print(f"  Script: {args.script}")
    print(f"  Config: {args.config}")
    print(f"  GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {name} ({mem:.0f} GB)")
    print()

    try:
        if args.script == 'discovery':
            run_discovery(args)
        elif args.script == 'session':
            run_session(args)
        elif args.script == 'all_sessions':
            run_all_sessions(args)
        elif args.script == 'synthetic_v2':
            run_synthetic(args)
        elif args.script == 'corpus':
            run_corpus(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
