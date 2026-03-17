"""Full within- and cross-modality validation for CADENCE V2 pipeline.

Tests each modality individually and in cross-modal pairs to verify:
  - True positives for injected coupling
  - No false positives in uncoupled pathways

Usage:
    python scripts/run_full_modality_test.py --duration 1800 --output results/full_modality_test
"""

import os
import sys
import json
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_v2
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import MOD_SHORT_V2

MODS = ['eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2', 'pose_features']
MOD_LABELS = {m: MOD_SHORT_V2.get(m, m) for m in MODS}

# Test scenarios: name -> kappa_dict
SCENARIOS = {}

# 1. Single-modality tests
for mod in MODS:
    label = MOD_LABELS[mod]
    kd = {m: 0.0 for m in MODS}
    kd[mod] = 0.3
    SCENARIOS[f'single_{label}'] = kd

# 2. Cross-modal pairs (all 6 combinations)
for i, m1 in enumerate(MODS):
    for m2 in MODS[i+1:]:
        l1, l2 = MOD_LABELS[m1], MOD_LABELS[m2]
        kd = {m: 0.0 for m in MODS}
        kd[m1] = 0.3
        kd[m2] = 0.3
        SCENARIOS[f'pair_{l1}+{l2}'] = kd

# 3. Null session
SCENARIOS['null'] = {m: 0.0 for m in MODS}


def analyze_scenario(name, kappa_dict, duration, config, output_dir, seed=42):
    """Run one scenario and return result dict."""
    t0 = time.time()
    print(f"\n{'='*60}")
    coupled = [MOD_LABELS[m] for m in MODS if kappa_dict.get(m, 0) > 0]
    print(f"SCENARIO: {name}  (coupled: {', '.join(coupled) or 'none'})")
    print(f"{'='*60}")

    session = build_synthetic_session_v2(duration, kappa_dict, seed=seed)
    estimator = CouplingEstimator(config)
    result = estimator.analyze_session(session, 'p1_to_p2')

    # Collect pathway results
    pathway_summary = {}
    for key in result.pathway_dr2:
        pkey = f'{key[0]}->{key[1]}'
        pathway_summary[pkey] = {
            'mean_dr2': float(np.nanmean(result.pathway_dr2[key])),
            'significant': bool(result.pathway_significant.get(key, False)),
        }
    for key, is_sig in result.pathway_significant.items():
        pkey = f'{key[0]}->{key[1]}'
        if pkey not in pathway_summary:
            pathway_summary[pkey] = {
                'mean_dr2': 0.0,
                'significant': bool(is_sig),
            }

    elapsed = time.time() - t0

    # Classify results
    tp, fn, fp, tn = [], [], [], []
    for pkey, info in pathway_summary.items():
        src, tgt = pkey.split('->')
        # A pathway is "expected positive" if BOTH src and tgt modalities are coupled
        # (same-modality) or either is coupled (cross-modal is trickier —
        # we only expect same-modality detections for now)
        src_coupled = kappa_dict.get(src, 0) > 0
        tgt_coupled = kappa_dict.get(tgt, 0) > 0
        expected_positive = (src == tgt) and src_coupled

        if expected_positive:
            if info['significant']:
                tp.append(pkey)
            else:
                fn.append(pkey)
        else:
            if info['significant']:
                fp.append(pkey)
            else:
                tn.append(pkey)

    result_data = {
        'name': name,
        'seed': seed,
        'duration': duration,
        'kappa_dict': kappa_dict,
        'n_significant': result.n_significant_pathways,
        'pathway_summary': pathway_summary,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'elapsed_s': round(elapsed, 1),
    }

    # Print result
    print(f"\n  TP ({len(tp)}): {', '.join(tp) or '-'}")
    print(f"  FN ({len(fn)}): {', '.join(fn) or '-'}")
    print(f"  FP ({len(fp)}): {', '.join(fp) or '-'}")
    print(f"  Time: {elapsed:.1f}s")

    # Save individual result
    json_path = os.path.join(output_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    return result_data


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Full within/cross-modality validation')
    parser.add_argument('--duration', type=int, default=1800)
    parser.add_argument('--output', default='results/full_modality_test')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--scenarios', nargs='*', default=None,
                        help='Run only these scenarios (by name)')
    args = parser.parse_args()

    config = load_config(args.config)
    config['pipeline'] = 'v2'
    if args.device:
        config['device'] = args.device

    os.makedirs(args.output, exist_ok=True)

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = {k: v for k, v in SCENARIOS.items() if k in args.scenarios}

    print(f"CADENCE Full Modality Test")
    print(f"  Duration: {args.duration}s")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Device: {config['device']}")

    all_results = []
    for i, (name, kappa_dict) in enumerate(scenarios.items()):
        try:
            r = analyze_scenario(
                name, kappa_dict, args.duration, config, args.output,
                seed=42 + i * 1000)
            all_results.append(r)
        except Exception:
            print(f"  -> FAILED")
            traceback.print_exc()

    # Overall summary
    print(f"\n\n{'='*60}")
    print(f"OVERALL SUMMARY ({len(all_results)} scenarios)")
    print(f"{'='*60}")

    total_tp = sum(len(r['tp']) for r in all_results)
    total_fn = sum(len(r['fn']) for r in all_results)
    total_fp = sum(len(r['fp']) for r in all_results)
    total_tn = sum(len(r['tn']) for r in all_results)

    print(f"\n  True Positives:  {total_tp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  False Positives: {total_fp}")
    print(f"  True Negatives:  {total_tn}")
    if total_tp + total_fn > 0:
        print(f"  Sensitivity:     {total_tp/(total_tp+total_fn)*100:.0f}%")
    if total_tn + total_fp > 0:
        print(f"  Specificity:     {total_tn/(total_tn+total_fp)*100:.0f}%")

    print(f"\nPer-scenario breakdown:")
    print(f"  {'Scenario':<20s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'Time':>6s}")
    print(f"  {'-'*40}")
    for r in all_results:
        print(f"  {r['name']:<20s} {len(r['tp']):>4d} {len(r['fn']):>4d} "
              f"{len(r['fp']):>4d} {r['elapsed_s']:>5.0f}s")

    if total_fp > 0:
        print(f"\n  FALSE POSITIVE DETAILS:")
        for r in all_results:
            for fp in r['fp']:
                print(f"    {r['name']}: {fp}")

    # Save summary
    summary = {
        'total_tp': total_tp, 'total_fn': total_fn,
        'total_fp': total_fp, 'total_tn': total_tn,
        'per_scenario': [{
            'name': r['name'],
            'tp': r['tp'], 'fn': r['fn'], 'fp': r['fp'],
            'elapsed_s': r['elapsed_s'],
        } for r in all_results],
    }
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
