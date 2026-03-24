"""Compare block size settings on synthetic data.

Runs the full V2 pipeline twice on 1800s synthetic sessions:
  1. Current block size (120s)
  2. Reduced block size (90s)

Reports per-condition:
  - Detection (TP/FP for each modality)
  - Timing accuracy AUC (dR2 timecourse vs ground-truth coupling gate)
  - Specificity (null test false positive rate)
  - Runtime

Usage:
    python scripts/test_block_size.py
    python scripts/test_block_size.py --device cuda:0
"""

import argparse
import json
import os
import sys
import time
import copy
import torch.multiprocessing as mp

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_v2
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import MOD_SHORT_V2, MODALITY_ORDER_V2


DURATION = 1800  # 30 minutes
BLOCK_SIZES = [120.0, 90.0]

# Test matrix: subset focused on detection + timing
TESTS = [
    ('A_eeg_only',
     {'eeg_wavelet': 0.7, 'ecg_features_v2': 0.0,
      'blendshapes_v2': 0.0, 'pose_features': 0.0},
     None, ['eeg_wavelet']),

    ('B_bl_only',
     {'eeg_wavelet': 0.0, 'ecg_features_v2': 0.0,
      'blendshapes_v2': 0.7, 'pose_features': 0.0},
     0.3, ['blendshapes_v2']),

    ('D_pose_only',
     {'eeg_wavelet': 0.0, 'ecg_features_v2': 0.0,
      'blendshapes_v2': 0.0, 'pose_features': 0.7},
     0.3, ['pose_features']),

    ('E_eeg_bl',
     {'eeg_wavelet': 0.7, 'ecg_features_v2': 0.0,
      'blendshapes_v2': 0.7, 'pose_features': 0.0},
     0.3, ['eeg_wavelet', 'blendshapes_v2']),

    ('F_null',
     {'eeg_wavelet': 0.0, 'ecg_features_v2': 0.0,
      'blendshapes_v2': 0.0, 'pose_features': 0.0},
     None, []),
]


def compute_timing_auc(result, session, expected_coupled):
    """Compute timing accuracy AUC: dR2 timecourse vs ground-truth gate.

    For each coupled modality, resample the coupling gate to the eval
    time grid and compute ROC AUC of the dR2 timecourse predicting
    gate-active periods (gate > 0.1).

    Returns dict {modality: auc} for each coupled modality with detections.
    """
    coupling_gates = session.get('coupling_gates', {})
    aucs = {}

    for mod in expected_coupled:
        if mod not in coupling_gates:
            continue

        # Find the same-modality pathway
        key = (mod, mod)
        if key not in result.pathway_dr2:
            continue

        dr2 = result.pathway_dr2[key]
        gate = coupling_gates[mod]

        # Get eval times and modality times for alignment
        pw_times = result.pathway_times.get(key, result.times)
        n_eval = len(pw_times)
        n_gate = len(gate)

        if n_eval == 0 or n_gate == 0:
            continue

        # Resample gate to eval grid (nearest neighbor)
        # Gate is at the modality's native rate; eval times are in seconds
        t_start = pw_times[0]
        t_end = pw_times[-1]
        gate_ts = np.linspace(t_start, t_end, n_gate)
        gate_resampled = np.interp(pw_times, gate_ts, gate)

        # Binary labels: gate active = 1
        labels = (gate_resampled > 0.1).astype(int)

        # dR2 as prediction score
        scores = np.nan_to_num(dr2[:n_eval], nan=0.0)

        # Need both classes present for AUC
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        try:
            auc = roc_auc_score(labels, scores)
            aucs[mod] = auc
        except ValueError:
            continue

    return aucs


def run_test_suite(config, block_size, output_dir, device='cuda:0'):
    """Run all tests with a given block size. Returns list of result dicts."""
    cfg = copy.deepcopy(config)
    cfg['doubly_sparse']['block_selection']['block_duration_s'] = block_size
    cfg['device'] = device

    tag = f"[GPU {device[-1]}] " if 'cuda' in device else ""
    estimator = CouplingEstimator(cfg)
    results = []

    for test_name, kappa_dict, duty_override, expected_coupled in TESTS:
        print(f"\n  {tag}--- {test_name} (block={block_size}s) ---", flush=True)
        coupled_mods = [m for m, k in kappa_dict.items() if k > 0]
        print(f"  {tag}Coupled: {coupled_mods if coupled_mods else 'NONE (null)'}", flush=True)

        # Generate synthetic session (same seed for both block sizes)
        session = build_synthetic_session_v2(
            DURATION, kappa_dict, seed=42,
            duty_cycle_override=duty_override)

        # Analyze
        t0 = time.time()
        result = estimator.analyze_session(session, 'p1_to_p2')
        runtime = time.time() - t0

        # Detection results
        detected = {}
        mod_order = MODALITY_ORDER_V2
        for src_mod in mod_order:
            key = (src_mod, src_mod)
            detected[src_mod] = result.pathway_significant.get(key, False)

        # Same-modality dR2
        dr2_vals = {}
        for src_mod in mod_order:
            key = (src_mod, src_mod)
            if key in result.pathway_dr2:
                dr2_vals[src_mod] = float(np.nanmean(result.pathway_dr2[key]))

        # Timing AUC
        timing_aucs = compute_timing_auc(result, session, expected_coupled)

        # Pass/fail
        expected_pos = set(expected_coupled)
        expected_neg = set(mod_order) - expected_pos

        if test_name.endswith('_null'):
            false_pos = any(detected.get(m, False) for m in mod_order)
            # Count total false positives across all pathways
            n_fp = sum(1 for k, v in result.pathway_significant.items() if v)
            passed = not false_pos
            status = 'PASS' if passed else f'FAIL (FP: {n_fp})'
        else:
            true_pos = all(detected.get(m, False) for m in expected_pos)
            false_pos = any(detected.get(m, False) for m in expected_neg)
            if not true_pos and not false_pos:
                status = 'FAIL (missed)'
                passed = False
            elif false_pos:
                status = 'FAIL (FP)'
                passed = False
            else:
                status = 'PASS'
                passed = True

        print(f"  {tag}Detection: {status} ({runtime:.1f}s)", flush=True)
        for mod in mod_order:
            short = MOD_SHORT_V2.get(mod, mod)
            det_str = 'YES' if detected.get(mod) else 'no'
            dr2_str = f"dR2={dr2_vals.get(mod, 0):+.4f}" if mod in dr2_vals else ""
            auc_str = f"AUC={timing_aucs.get(mod, 0):.3f}" if mod in timing_aucs else ""
            print(f"  {tag}  {short:>5s}: {det_str:>3s}  {dr2_str:>14s}  {auc_str}", flush=True)

        results.append({
            'test': test_name,
            'block_size': block_size,
            'passed': passed,
            'status': status,
            'runtime_s': runtime,
            'detected': {k: bool(v) for k, v in detected.items()},
            'dr2': dr2_vals,
            'timing_auc': timing_aucs,
        })

    return results


def _worker(config, block_size, output_dir, device, return_dict, key):
    """Worker function for multiprocessing."""
    results = run_test_suite(config, block_size, output_dir, device=device)
    return_dict[key] = results


def main():
    parser = argparse.ArgumentParser(description='Block size comparison')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='results/block_size_test')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)

    # Disable ECG moderation for synthetic tests — synthetic ECG is random
    # Lorenz noise, moderation columns add pure noise.
    config['stage2']['moderation']['enabled'] = False

    os.makedirs(args.output, exist_ok=True)

    # Detect available GPUs
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parallel = n_gpus >= 2 and args.device is None

    print(f"Block Size Comparison Test")
    print(f"Duration: {DURATION}s")
    print(f"Block sizes: {BLOCK_SIZES}")
    print(f"Tests: {[t[0] for t in TESTS]}")
    print(f"GPUs: {n_gpus} ({'parallel' if parallel else 'sequential'})")
    print(f"Output: {args.output}")

    all_results = {}

    if parallel:
        # Run both block sizes in parallel on separate GPUs
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for i, block_size in enumerate(BLOCK_SIZES):
            device = f'cuda:{i}'
            print(f"\n{'='*60}")
            print(f"BLOCK SIZE: {block_size}s -> {device}")
            print(f"  Blocks per session: {DURATION / block_size:.0f}")
            print(f"{'='*60}")

            p = mp.Process(
                target=_worker,
                args=(config, block_size, args.output, device,
                      return_dict, block_size))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for block_size in BLOCK_SIZES:
            all_results[block_size] = return_dict[block_size]
    else:
        # Sequential fallback
        device = args.device or ('cuda:0' if n_gpus > 0 else 'cpu')
        for block_size in BLOCK_SIZES:
            print(f"\n{'='*60}")
            print(f"BLOCK SIZE: {block_size}s -> {device}")
            print(f"  Blocks per session: {DURATION / block_size:.0f}")
            print(f"{'='*60}")

            results = run_test_suite(config, block_size, args.output,
                                     device=device)
            all_results[block_size] = results

    # Comparison summary
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")

    header = f"{'Test':<15s}"
    for bs in BLOCK_SIZES:
        header += f" | {'Pass':>4s} {'Runtime':>8s} {'AUC':>6s}  ({bs:.0f}s)"
    print(header)
    print("-" * 70)

    for test_idx, (test_name, _, _, expected) in enumerate(TESTS):
        row = f"{test_name:<15s}"
        for bs in BLOCK_SIZES:
            r = all_results[bs][test_idx]
            p_str = 'PASS' if r['passed'] else 'FAIL'
            rt_str = f"{r['runtime_s']:.0f}s"
            aucs = list(r['timing_auc'].values())
            auc_str = f"{np.mean(aucs):.3f}" if aucs else "  n/a"
            row += f" | {p_str:>4s} {rt_str:>8s} {auc_str:>6s}"
        print(row)

    # Aggregate metrics
    print(f"\n{'Metric':<25s}", end="")
    for bs in BLOCK_SIZES:
        print(f" | {bs:.0f}s block", end="")
    print()
    print("-" * 55)

    for metric_name, metric_fn in [
        ("Tests passed",
         lambda rs: sum(1 for r in rs if r['passed'])),
        ("Total runtime (s)",
         lambda rs: sum(r['runtime_s'] for r in rs)),
        ("Mean timing AUC",
         lambda rs: np.mean([a for r in rs for a in r['timing_auc'].values()])
                    if any(r['timing_auc'] for r in rs) else float('nan')),
        ("Null FP pathways",
         lambda rs: sum(1 for r in rs if r['test'].endswith('_null')
                        and not r['passed'])),
    ]:
        row = f"{metric_name:<25s}"
        for bs in BLOCK_SIZES:
            val = metric_fn(all_results[bs])
            if isinstance(val, float):
                row += f" | {val:>10.3f}"
            else:
                row += f" | {val:>10d}"
        print(row)

    # Save
    json_path = os.path.join(args.output, 'block_size_comparison.json')
    save_data = {str(bs): results for bs, results in all_results.items()}
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == '__main__':
    main()
