"""Pseudo-pair null test for interbrain coupling.

Creates fake sessions by pairing P1 from session A with P2 from session B
(participants who were never in the same room). If interbrain delta coupling
still shows significance, the result is artifactual.

Only tests eeg_interbrain pathways to keep runtime short.

Usage:
    python scripts/run_pseudopair.py
    python scripts/run_pseudopair.py --n-pairs 5 --device cuda
"""

import argparse
import json
import os
import sys
import time
import itertools

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import MOD_SHORT_V2, MODALITY_ORDER_V2, INTERBRAIN_MODALITY


def create_pseudo_session(session_a, session_b, name_a, name_b):
    """Create a pseudo-pair session: P1 from A, P2 from B.

    Recomputes interbrain features from the mismatched EEG signals.
    """
    pseudo = {}

    # Take P1 signals from session A
    for key in session_a:
        if key.startswith('p1_'):
            pseudo[key] = session_a[key]

    # Take P2 signals from session B
    for key in session_b:
        if key.startswith('p2_'):
            pseudo[key] = session_b[key]

    # Use the shorter duration
    dur_a = session_a.get('duration', 0)
    dur_b = session_b.get('duration', 0)
    pseudo['duration'] = min(dur_a, dur_b)

    # Copy metadata from A
    for key in ['p1_role', 'session_name']:
        if key in session_a:
            pseudo[key] = session_a[key]

    # Recompute interbrain features from mismatched EEG
    # The key insight: these two EEGs should have NO genuine phase coupling
    # since the participants were never in the same room.
    try:
        from cadence.data.preprocessors import compute_interbrain_features
        eeg1 = session_a.get('p1_eeg_wavelet')
        eeg2 = session_b.get('p2_eeg_wavelet')
        ts1 = session_a.get('p1_eeg_wavelet_ts')
        ts2 = session_b.get('p2_eeg_wavelet_ts')

        if eeg1 is not None and eeg2 is not None:
            # Trim to common duration
            max_t = pseudo['duration']
            mask1 = ts1 <= max_t
            mask2 = ts2 <= max_t
            eeg1_trim = eeg1[mask1]
            eeg2_trim = eeg2[mask2]
            ts1_trim = ts1[mask1]
            ts2_trim = ts2[mask2]

            # Resample to common grid (5 Hz)
            common_ts = np.arange(0, max_t, 0.2)
            n_ch = eeg1_trim.shape[1]
            eeg1_r = np.zeros((len(common_ts), n_ch), dtype=np.float32)
            eeg2_r = np.zeros((len(common_ts), n_ch), dtype=np.float32)
            for c in range(n_ch):
                eeg1_r[:, c] = np.interp(common_ts, ts1_trim, eeg1_trim[:, c])
                eeg2_r[:, c] = np.interp(common_ts, ts2_trim, eeg2_trim[:, c])

            # Compute phase differences (cos/sin of angle difference)
            # eeg_wavelet features are real/imag pairs: features 0..79 = real, 80..159 = imag
            n_half = n_ch // 2
            real1, imag1 = eeg1_r[:, :n_half], eeg1_r[:, n_half:]
            real2, imag2 = eeg2_r[:, :n_half], eeg2_r[:, n_half:]

            # Phase difference via complex multiplication: z1 * conj(z2)
            cos_diff = real1 * real2 + imag1 * imag2
            sin_diff = imag1 * real2 - real1 * imag2

            # Normalize
            mag = np.sqrt(cos_diff**2 + sin_diff**2) + 1e-10
            cos_diff /= mag
            sin_diff /= mag

            ib_features = np.concatenate([cos_diff, sin_diff], axis=1).astype(np.float32)
            ib_valid = np.ones(len(common_ts), dtype=bool)

            pseudo[INTERBRAIN_MODALITY] = ib_features
            pseudo[f'{INTERBRAIN_MODALITY}_ts'] = common_ts
            pseudo[f'{INTERBRAIN_MODALITY}_valid'] = ib_valid
    except Exception as e:
        print(f"  Warning: interbrain recomputation failed ({e}), using A's features")
        # Fall back to session A's interbrain (still pseudo since P2 is different)
        for key in session_a:
            if INTERBRAIN_MODALITY in key:
                pseudo[key] = session_a[key]

    return pseudo


def main():
    parser = argparse.ArgumentParser(description='Pseudo-pair null test')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--n-pairs', type=int, default=5,
                        help='Number of pseudo-pairs to test')
    parser.add_argument('--output', default='results/pseudopair_null')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    os.makedirs(args.output, exist_ok=True)

    # Load all sessions
    all_sessions = discover_cached_sessions(config['session_cache'])
    excluded = set(config.get('excluded_sessions', []))
    sessions = [(n, p) for n, p in all_sessions if n not in excluded]

    print(f"Available sessions: {[n for n, _ in sessions]}")
    print(f"Loading sessions...")

    loaded = {}
    for name, path in sessions:
        loaded[name] = load_session_from_cache(path, config=config)
        dur = loaded[name].get('duration', 0)
        print(f"  {name}: {dur:.0f}s")

    # Generate pseudo-pairs (all unique A!=B combinations)
    names = list(loaded.keys())
    all_pairs = [(a, b) for a, b in itertools.permutations(names, 2)]
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    pairs = all_pairs[:args.n_pairs]

    print(f"\nTesting {len(pairs)} pseudo-pairs:")
    for a, b in pairs:
        print(f"  P1({a}) x P2({b})")

    estimator = CouplingEstimator(config)

    results = []
    for pair_idx, (name_a, name_b) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Pseudo-pair {pair_idx+1}/{len(pairs)}: P1({name_a}) x P2({name_b})")
        print(f"{'='*60}")

        t0 = time.time()
        pseudo = create_pseudo_session(
            loaded[name_a], loaded[name_b], name_a, name_b)

        if INTERBRAIN_MODALITY not in pseudo:
            print("  No interbrain features, skipping")
            continue

        print(f"  Pseudo session: {pseudo['duration']:.0f}s, "
              f"interbrain shape={pseudo[INTERBRAIN_MODALITY].shape}")

        result = estimator.analyze_session(pseudo, 'p1_to_p2')
        dt = time.time() - t0

        # Report interbrain pathways only
        pair_result = {
            'pair': f'P1({name_a})_x_P2({name_b})',
            'duration': pseudo['duration'],
            'runtime_s': dt,
            'interbrain_pathways': {},
        }

        n_ib_sig = 0
        for key, is_sig in result.pathway_significant.items():
            src_mod, tgt_mod = key
            if src_mod != INTERBRAIN_MODALITY:
                continue
            dr2 = result.pathway_dr2.get(key)
            mean_dr2 = float(np.nanmean(dr2)) if dr2 is not None else 0.0
            sig_str = '*** FALSE POSITIVE' if is_sig else ''
            tgt_s = MOD_SHORT_V2.get(tgt_mod, tgt_mod)
            print(f"  EEGib->{tgt_s}: dR2={mean_dr2:+.4f} {sig_str}")

            pair_result['interbrain_pathways'][f'EEGib->{tgt_s}'] = {
                'significant': is_sig,
                'mean_dr2': mean_dr2,
            }
            if is_sig:
                n_ib_sig += 1

        pair_result['n_interbrain_significant'] = n_ib_sig
        results.append(pair_result)
        print(f"  -> {n_ib_sig} interbrain false positives in {dt:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("PSEUDO-PAIR NULL TEST SUMMARY")
    print(f"{'='*60}")

    total_tests = 0
    total_fp = 0
    for r in results:
        n_tested = len(r['interbrain_pathways'])
        n_fp = r['n_interbrain_significant']
        total_tests += n_tested
        total_fp += n_fp
        status = 'CLEAN' if n_fp == 0 else f'{n_fp} FALSE POSITIVES'
        print(f"  {r['pair']}: {status}")

    fpr = total_fp / total_tests if total_tests > 0 else 0
    print(f"\nFalse positive rate: {total_fp}/{total_tests} = {fpr:.1%}")
    print(f"Expected under null (alpha=0.05): {0.05:.1%}")

    if fpr > 0.10:
        print("\n*** WARNING: FPR > 10% — interbrain delta coupling is likely ARTIFACTUAL ***")
    elif fpr > 0.05:
        print("\n** CAUTION: FPR elevated above nominal 5% **")
    else:
        print("\nFPR within expected range — interbrain coupling may be genuine.")

    # Save
    json_path = os.path.join(args.output, 'pseudopair_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'pairs': results,
            'total_tests': total_tests,
            'total_false_positives': total_fp,
            'false_positive_rate': fpr,
        }, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == '__main__':
    main()
