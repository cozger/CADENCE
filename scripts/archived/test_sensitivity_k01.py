#!/usr/bin/env python
"""Individual sensitivity tests: EEG, BL, Pose at kappa=0.1, duration=3000s."""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.synthetic import (
    build_synthetic_session_v2,
    build_synthetic_wavelet_session,
)
from cadence.data.alignment import _ensure_v2_features
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import WAVELET_CENTER_FREQS

DURATION = 3000
KAPPA = 0.1
SEED = 42

config = load_config('configs/default.yaml')
config['pipeline'] = 'v2'

BEHAVIORAL_MODS = ['eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2', 'pose_features']
null_kd = {m: 0.0 for m in BEHAVIORAL_MODS}


def run_test(name, session, expected_pathways):
    print(f"\n{'='*60}")
    print(f"  {name}  (kappa={KAPPA}, duration={DURATION}s)")
    print(f"{'='*60}")

    # Diagnostic: show session contents
    p1 = session.get('p1', {})
    p2 = session.get('p2', {})
    print(f"  Session keys: {list(session.keys())}")
    print(f"  P1 modalities: {list(p1.keys())}")
    print(f"  P2 modalities: {list(p2.keys())}")
    for p, pname in [(p1, 'P1'), (p2, 'P2')]:
        for mod, data in p.items():
            if isinstance(data, dict) and 'signal' in data:
                sig = data['signal']
                shape = sig.shape if hasattr(sig, 'shape') else 'N/A'
                print(f"    {pname}.{mod}: signal={shape}")

    t0 = time.time()
    estimator = CouplingEstimator(config)
    result = estimator.analyze_session(session, 'p1_to_p2')
    elapsed = time.time() - t0

    # Diagnostic: show all pathway results
    print(f"\n  Pathway results ({len(result.pathway_significant)} pathways):")
    for key, is_sig in result.pathway_significant.items():
        dr2 = float(np.nanmean(result.pathway_dr2.get(key, [0])))
        print(f"    {key[0]}->{key[1]}: sig={is_sig}, dr2={dr2:.6f}")
    if not result.pathway_significant:
        print("    (no pathways found)")

    tp, fn, fp, tn = [], [], [], []
    for key, is_sig in result.pathway_significant.items():
        src, tgt = key
        pkey = f'{src}->{tgt}'
        expected = key in expected_pathways
        if expected:
            (tp if is_sig else fn).append(pkey)
        else:
            (fp if is_sig else tn).append(pkey)

    det = 'DETECTED' if tp else 'MISSED'
    print(f"\n  Result: {det} ({elapsed:.1f}s)")
    print(f"  TP={tp}")
    print(f"  FN={fn}")
    print(f"  FP={fp}")
    return {'tp': tp, 'fn': fn, 'fp': fp, 'elapsed': elapsed}


# --- Test 1: EEG wavelet coupling (6.5 Hz frontal) ---
print("\nBuilding EEG session...")
eeg_session = build_synthetic_wavelet_session(
    DURATION, coupling_freq=6.5, coupling_roi='frontal',
    kappa=KAPPA, seed=SEED)
eeg_session = _ensure_v2_features(eeg_session, config)

eeg_expected = {
    ('eeg_interbrain', 'eeg_wavelet'),
    ('eeg_wavelet', 'eeg_wavelet'),
}
r_eeg = run_test('EEG Wavelet (6.5 Hz frontal)', eeg_session, eeg_expected)
del eeg_session

# --- Test 2: BL coupling ---
print("\nBuilding BL session...")
bl_kd = dict(null_kd)
bl_kd['blendshapes_v2'] = KAPPA
bl_session = build_synthetic_session_v2(DURATION, bl_kd, seed=SEED)

bl_expected = {('blendshapes_v2', 'blendshapes_v2')}
r_bl = run_test('Blendshapes', bl_session, bl_expected)
del bl_session

# --- Test 3: Pose coupling ---
print("\nBuilding Pose session...")
pose_kd = dict(null_kd)
pose_kd['pose_features'] = KAPPA
pose_session = build_synthetic_session_v2(DURATION, pose_kd, seed=SEED)

pose_expected = {('pose_features', 'pose_features')}
r_pose = run_test('Pose', pose_session, pose_expected)
del pose_session

# --- Summary ---
print(f"\n{'='*60}")
print(f"  SUMMARY  (kappa={KAPPA}, duration={DURATION}s)")
print(f"{'='*60}")
for name, r in [('EEG', r_eeg), ('BL', r_bl), ('Pose', r_pose)]:
    det = 'HIT' if r['tp'] else 'MISS'
    fp_str = f", FP={len(r['fp'])}" if r['fp'] else ''
    print(f"  {name:6s}: {det}{fp_str}  ({r['elapsed']:.1f}s)")
