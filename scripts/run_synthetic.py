"""CADENCE synthetic validation.

Tests on synthetic Lorenz sessions with known ground-truth coupling:
  - EEG-coupled: detect EEG pathway, not others
  - BL-coupled: detect BL pathway (with standardized duty cycle)
  - Cross-modal: both detected
  - Null: < 5% false positive rate

Usage:
    python scripts/run_synthetic.py
    python scripts/run_synthetic.py --device cpu
"""

import argparse
import json
import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_v2
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import MOD_SHORT_V2, MODALITY_ORDER_V2, SYNTH_MODALITY_CONFIG_V2


# ---------------------------------------------------------------------------
# V2 test matrix (default): uses V2 modality keys
# ---------------------------------------------------------------------------
SYNTHETIC_TESTS_V2 = [
    ('A_eeg_only',
     {'eeg_wavelet': 0.7, 'blendshapes_v2': 0.0, 'pose_features': 0.0},
     None, ['eeg_wavelet']),

    ('B_bl_only',
     {'eeg_wavelet': 0.0, 'blendshapes_v2': 0.7, 'pose_features': 0.0},
     0.3, ['blendshapes_v2']),

    ('C_pose_only',
     {'eeg_wavelet': 0.0, 'blendshapes_v2': 0.0, 'pose_features': 0.7},
     0.3, ['pose_features']),

    ('D_eeg_bl',
     {'eeg_wavelet': 0.7, 'blendshapes_v2': 0.7, 'pose_features': 0.0},
     0.3, ['eeg_wavelet', 'blendshapes_v2']),

    ('E_null',
     {'eeg_wavelet': 0.0, 'blendshapes_v2': 0.0, 'pose_features': 0.0},
     None, []),
]

def _gen_one_session(args):
    """Worker: generate one synthetic session (top-level for pickling)."""
    test_name, kappa_dict, duty_override, duration = args
    session = build_synthetic_session_v2(
        duration, kappa_dict, seed=42, duty_cycle_override=duty_override)
    return test_name, session


def main():
    parser = argparse.ArgumentParser(description='CADENCE synthetic validation')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--duration', type=int, default=600, help='Session duration (seconds)')
    parser.add_argument('--output', default='results/cadence_synthetic_v2',
                        help='Output directory')
    parser.add_argument('--tests', default=None,
                        help='Comma-separated test prefixes to run (e.g. A,B,C)')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    # Disable ECG moderation for synthetic tests — synthetic ECG is random
    # Lorenz noise with no relationship to coupling, so moderation columns
    # add pure noise that dilutes the signal.
    config['stage2']['moderation']['enabled'] = False

    tests = SYNTHETIC_TESTS_V2
    if args.tests:
        prefixes = [p.strip() for p in args.tests.split(',')]
        tests = [t for t in tests
                 if any(t[0].startswith(p) for p in prefixes)]
    build_fn = build_synthetic_session_v2
    mod_order = MODALITY_ORDER_V2
    mod_short = MOD_SHORT_V2

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Duration: {args.duration}s", flush=True)
    print(f"Output:   {output_dir}", flush=True)
    synth_mods = list(SYNTH_MODALITY_CONFIG_V2.keys())
    print(f"Modalities: {synth_mods}", flush=True)

    estimator = CouplingEstimator(config)

    # Pre-generate all sessions in parallel (CPU-bound Lorenz integration)
    import multiprocessing as mp
    gen_args = [(name, kd, do, args.duration) for name, kd, do, _ in tests]
    n_workers = min(len(tests), os.cpu_count() or 1)
    print(f"Generating {len(tests)} sessions in parallel ({n_workers} workers)...",
          flush=True)
    t0_gen = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        gen_results = pool.map(_gen_one_session, gen_args)
    sessions = {name: sess for name, sess in gen_results}
    print(f"All sessions generated in {time.time()-t0_gen:.1f}s", flush=True)

    results_all = []
    n_passed = 0

    for test_name, kappa_dict, duty_override, expected_coupled in tests:
        print(f"\n{'='*60}", flush=True)
        print(f"Test {test_name}", flush=True)
        coupled_mods = [m for m, k in kappa_dict.items() if k > 0]
        print(f"  Coupled: {coupled_mods if coupled_mods else 'NONE (null)'}", flush=True)
        if duty_override:
            print(f"  Duty cycle override: {duty_override}", flush=True)
        print(f"{'='*60}", flush=True)

        session = sessions[test_name]

        # Analyze (p1->p2 only, since coupling is p1->p2)
        t0 = time.time()
        result = estimator.analyze_session(session, 'p1_to_p2')
        dt = time.time() - t0
        print(f"  Analysis: {dt:.1f}s, {result.n_significant_pathways} significant pathways", flush=True)

        # Report same-modality pathways (primary coupling indicator)
        detected = {}
        print(f"\n  Same-modality pathways:", flush=True)
        for src_mod in mod_order:
            key = (src_mod, src_mod)
            if key in result.pathway_dr2:
                dr2 = result.pathway_dr2[key]
                is_sig = result.pathway_significant.get(key, False)
                detected[src_mod] = is_sig
                mean_dr2 = np.nanmean(dr2)
                sig_str = 'DETECTED' if is_sig else ''
                short = mod_short.get(src_mod, src_mod)
                print(f"    {short:>5s}->{short:<5s}: "
                      f"dR2={mean_dr2:+.6f} {sig_str}")
            else:
                detected[src_mod] = False

        # Also report cross-modal pathways if any are significant
        cross_sig = []
        for key, sig in result.pathway_significant.items():
            if sig and key[0] != key[1]:
                cross_sig.append(key)
        if cross_sig:
            print(f"\n  Cross-modal significant:", flush=True)
            for src_mod, tgt_mod in cross_sig:
                s_short = mod_short.get(src_mod, src_mod)
                t_short = mod_short.get(tgt_mod, tgt_mod)
                dr2 = result.pathway_dr2.get((src_mod, tgt_mod))
                mean_dr2 = np.nanmean(dr2) if dr2 is not None else float('nan')
                print(f"    {s_short:>5s}->{t_short:<5s}: dR2={mean_dr2:+.6f}", flush=True)

        # Evaluate pass/fail
        expected_pos = set(expected_coupled)
        expected_neg = set(mod_order) - expected_pos

        if test_name.endswith('_null'):
            # Null test: no false positives allowed
            false_pos = any(detected.get(m, False) for m in mod_order)
            passed = not false_pos
            status = 'PASS' if passed else 'FAIL (false positive)'
        else:
            # Coupling test: detect expected modalities, no unexpected
            true_pos = all(detected.get(m, False) for m in expected_pos) if expected_pos else True
            false_pos = any(detected.get(m, False) for m in expected_neg)

            if not true_pos and not false_pos:
                status = 'FAIL (missed coupling)'
                passed = False
            elif false_pos:
                status = 'FAIL (false positive)'
                passed = False
            else:
                status = 'PASS'
                passed = True

        print(f"  Result: {status}", flush=True)
        if passed:
            n_passed += 1

        results_all.append({
            'test': test_name,
            'kappa_dict': kappa_dict,
            'duty_override': duty_override,
            'detected': {k: bool(v) for k, v in detected.items()},
            'passed': passed,
            'status': status,
        })

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: {n_passed}/{len(tests)} tests passed", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results_all:
        icon = '[PASS]' if r['passed'] else '[FAIL]'
        print(f"  {icon} {r['test']}: {r['status']}", flush=True)

    # Save results
    json_path = os.path.join(output_dir, 'synthetic_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved: {json_path}", flush=True)


if __name__ == '__main__':
    main()
