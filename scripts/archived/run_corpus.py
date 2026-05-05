"""Corpus synthetic validation for CADENCE V2 pipeline.

Generates synthetic sessions inline and analyzes with CouplingEstimator in V2
mode. No intermediate files -- generate, analyze, save results per session.

Usage:
    # Quick local test
    python scripts/run_corpus.py --duration 60 --n-coupled 3 --n-null 1 \
        --output results/corpus_test

    # Full corpus (cluster or local)
    python scripts/run_corpus.py --duration 3000 --n-coupled 50 --n-null 20 \
        --output results/corpus_3000s --device cuda
"""

import os
import sys
import json
import time
import argparse
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_v2, plan_corpus_v2
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import MOD_SHORT_V2


def _gen_one(gen_args):
    """Generate a single synthetic session (picklable for multiprocessing)."""
    name, kd, seed, dur = gen_args
    from cadence.synthetic import build_synthetic_session_v2
    t0 = time.time()
    session = build_synthetic_session_v2(dur, kd, seed=seed)
    print(f"  Generated {name} ({time.time()-t0:.1f}s)", flush=True)
    return (name, kd, seed, session)


def analyze_one_session(name, kappa_dict, seed, duration, config, output_dir):
    """Generate + analyze a single synthetic session, save JSON result."""
    t0 = time.time()
    session = build_synthetic_session_v2(duration, kappa_dict, seed=seed)

    estimator = CouplingEstimator(config)
    result = estimator.analyze_session(session, 'p1_to_p2')

    # Collect per-pathway results
    pathway_summary = {}
    for key in result.pathway_dr2:
        pkey = f'{key[0]}->{key[1]}'
        pathway_summary[pkey] = {
            'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
            'significant': result.pathway_significant.get(key, False),
        }
    for key, is_sig in result.pathway_significant.items():
        pkey = f'{key[0]}->{key[1]}'
        if pkey not in pathway_summary:
            pathway_summary[pkey] = {
                'mean_dr2': 0.0,
                'significant': is_sig,
            }

    elapsed = time.time() - t0

    result_data = {
        'name': name,
        'seed': seed,
        'duration': duration,
        'kappa_dict': kappa_dict,
        'n_significant': result.n_significant_pathways,
        'pathway_summary': pathway_summary,
        'elapsed_s': round(elapsed, 1),
    }

    json_path = os.path.join(output_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    return result_data


def compute_corpus_summary(all_results):
    """Compute TP/FP rates per modality from corpus results."""
    v2_mods = ['eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2', 'pose_features']
    mod_labels = {m: MOD_SHORT_V2.get(m, m) for m in v2_mods}

    summary = {'n_sessions': len(all_results), 'per_modality': {}}

    # Per-modality TP/FP tracking
    for mod in v2_mods:
        label = mod_labels[mod]
        tp_count, tp_total = 0, 0
        fp_count, fp_total = 0, 0

        for r in all_results:
            kappa = r['kappa_dict'].get(mod, 0.0)
            # Check if this modality is significant in any same-mod pathway
            mod_sig = False
            for pkey, pinfo in r['pathway_summary'].items():
                src, tgt = pkey.split('->')
                if src == mod and tgt == mod and pinfo['significant']:
                    mod_sig = True
                    break
            # Also check cross-modal pathways where this is source
            if not mod_sig:
                for pkey, pinfo in r['pathway_summary'].items():
                    src, tgt = pkey.split('->')
                    if src == mod and pinfo['significant']:
                        mod_sig = True
                        break

            if kappa > 0:
                tp_total += 1
                if mod_sig:
                    tp_count += 1
            else:
                fp_total += 1
                if mod_sig:
                    fp_count += 1

        summary['per_modality'][label] = {
            'tp_rate': round(tp_count / max(tp_total, 1), 3),
            'tp_count': tp_count,
            'tp_total': tp_total,
            'fp_rate': round(fp_count / max(fp_total, 1), 3),
            'fp_count': fp_count,
            'fp_total': fp_total,
        }

    # Null session false positive rate (any detection)
    null_results = [r for r in all_results
                    if all(v == 0.0 for v in r['kappa_dict'].values())]
    null_fp = sum(1 for r in null_results if r['n_significant'] > 0)
    summary['null_sessions'] = {
        'total': len(null_results),
        'any_fp': null_fp,
        'fp_rate': round(null_fp / max(len(null_results), 1), 3),
    }

    # TP by kappa bin
    kappa_bins = {
        'low_0.3-0.5': (0.3, 0.5),
        'mid_0.5-0.7': (0.5, 0.7),
        'high_0.7-0.9': (0.7, 0.9),
    }
    coupled_results = [r for r in all_results
                       if any(v > 0 for v in r['kappa_dict'].values())]
    summary['tp_by_kappa_bin'] = {}
    for bin_name, (lo, hi) in kappa_bins.items():
        tp, total = 0, 0
        for r in coupled_results:
            # Check EEG kappa (primary coupled modality)
            ek = r['kappa_dict'].get('eeg_wavelet', 0.0)
            if lo <= ek < hi + 0.001:
                total += 1
                if r['n_significant'] > 0:
                    tp += 1
        summary['tp_by_kappa_bin'][bin_name] = {
            'tp_rate': round(tp / max(total, 1), 3),
            'count': total,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='CADENCE V2 corpus synthetic validation')
    parser.add_argument('--duration', type=int, default=300,
                        help='Session duration in seconds (default: 300)')
    parser.add_argument('--n-coupled', type=int, default=50,
                        help='Number of coupled sessions (default: 50)')
    parser.add_argument('--n-null', type=int, default=20,
                        help='Number of null sessions (default: 20)')
    parser.add_argument('--output', default='results/corpus_v2',
                        help='Output directory')
    parser.add_argument('--config', default='configs/default.yaml',
                        help='Config file (default: configs/default.yaml)')
    parser.add_argument('--device', default=None,
                        help='Device override (cuda, cpu)')
    parser.add_argument('--session-idx', nargs=2, type=int, default=None,
                        metavar=('START', 'END'),
                        help='Session index range [START, END) for GPU splitting')
    args = parser.parse_args()

    config = load_config(args.config)
    config['pipeline'] = 'v2'
    if args.device:
        config['device'] = args.device

    os.makedirs(args.output, exist_ok=True)

    specs = plan_corpus_v2(
        n_coupled=args.n_coupled, n_null=args.n_null)

    # Optional sub-range for multi-GPU splitting
    if args.session_idx:
        start, end = args.session_idx
        specs = specs[start:end]
        print(f"Processing sessions [{start}, {end})")

    n_cpus = os.cpu_count() or 1
    n_gen_workers = min(n_cpus, len(specs), 120)

    print(f"CADENCE V2 Corpus Validation")
    print(f"  Duration: {args.duration}s")
    print(f"  Sessions: {len(specs)} "
          f"({args.n_coupled} coupled + {args.n_null} null)")
    print(f"  Device: {config['device']}")
    print(f"  Generation workers: {n_gen_workers}")
    print(f"  Output: {args.output}")
    print()

    # Phase 1: Parallel generation on all CPUs
    import multiprocessing as mproc

    print(f"Phase 1: Generating {len(specs)} sessions "
          f"with {n_gen_workers} workers...", flush=True)
    gen_args = [(name, kd, seed, args.duration) for name, kd, seed in specs]
    t_gen = time.time()

    with mproc.Pool(n_gen_workers) as pool:
        generated = pool.map(_gen_one, gen_args)

    print(f"Phase 1 done: {len(generated)} sessions in "
          f"{time.time()-t_gen:.0f}s\n", flush=True)

    # Phase 2: Sequential GPU analysis
    print(f"Phase 2: Analyzing on {config['device']}...", flush=True)
    all_results = []
    for i, (name, kappa_dict, seed, session) in enumerate(generated):
        coupled_mods = [m for m, k in kappa_dict.items() if k > 0]
        label = ', '.join(coupled_mods) if coupled_mods else 'null'
        print(f"[{i+1}/{len(generated)}] {name} ({label})...", flush=True)

        try:
            t0 = time.time()
            estimator = CouplingEstimator(config)
            result = estimator.analyze_session(session, 'p1_to_p2')

            pathway_summary = {}
            for key in result.pathway_dr2:
                pkey = f'{key[0]}->{key[1]}'
                pathway_summary[pkey] = {
                    'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
                    'significant': result.pathway_significant.get(key, False),
                }
            # Include pathways significant from screening but missing dR2
            # (e.g., Stage 2 EWLS OOM)
            for key, is_sig in result.pathway_significant.items():
                pkey = f'{key[0]}->{key[1]}'
                if pkey not in pathway_summary:
                    pathway_summary[pkey] = {
                        'mean_dr2': 0.0,
                        'significant': is_sig,
                    }

            elapsed = time.time() - t0
            result_data = {
                'name': name, 'seed': seed, 'duration': args.duration,
                'kappa_dict': kappa_dict,
                'n_significant': result.n_significant_pathways,
                'pathway_summary': pathway_summary,
                'elapsed_s': round(elapsed, 1),
            }
            all_results.append(result_data)

            json_path = os.path.join(args.output, f'{name}.json')
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"  -> {result.n_significant_pathways} significant, "
                  f"{elapsed:.1f}s")
        except Exception:
            print(f"  -> FAILED")
            traceback.print_exc()

    # Save merged results
    merged_path = os.path.join(args.output, 'corpus_all_results.json')
    with open(merged_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute and save summary
    summary = compute_corpus_summary(all_results)
    summary_path = os.path.join(args.output, 'corpus_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CORPUS SUMMARY ({len(all_results)} sessions)")
    print(f"{'='*60}")
    for mod_label, stats in summary['per_modality'].items():
        tp = f"TP={stats['tp_rate']*100:.0f}% ({stats['tp_count']}/{stats['tp_total']})"
        fp = f"FP={stats['fp_rate']*100:.0f}% ({stats['fp_count']}/{stats['fp_total']})"
        print(f"  {mod_label:6s}: {tp}  {fp}")
    ns = summary['null_sessions']
    print(f"  Null  : FP={ns['fp_rate']*100:.0f}% ({ns['any_fp']}/{ns['total']})")
    print(f"\nTP by kappa bin:")
    for bin_name, stats in summary['tp_by_kappa_bin'].items():
        print(f"  {bin_name}: {stats['tp_rate']*100:.0f}% (n={stats['count']})")
    print(f"\nResults: {args.output}")


if __name__ == '__main__':
    main()
