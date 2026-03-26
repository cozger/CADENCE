"""Quick test: does EEG-only synthetic detection work at 1800s?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_v2
from cadence.coupling.estimator import CouplingEstimator

config = load_config('configs/default.yaml')
# Match committed HEAD: disable moderation, use nonlinear only
config['stage2']['moderation']['enabled'] = False

print("Generating 1800s EEG-only session...", flush=True)
session = build_synthetic_session_v2(
    1800,
    {'eeg_wavelet': 0.7, 'ecg_features_v2': 0.0,
     'blendshapes_v2': 0.0, 'pose_features': 0.0},
    seed=42)

print("Analyzing...", flush=True)
estimator = CouplingEstimator(config)
result = estimator.analyze_session(session, 'p1_to_p2')

key = ('eeg_wavelet', 'eeg_wavelet')
sig = result.pathway_significant.get(key, False)
dr2 = np.nanmean(result.pathway_dr2.get(key, np.array([0])))
print(f"\nEEG-EEG: detected={sig}, dR2={dr2:.6f}")
print(f"Total significant: {result.n_significant_pathways}")

# Show all significant pathways
for k, v in result.pathway_significant.items():
    if v:
        d = np.nanmean(result.pathway_dr2.get(k, np.array([0])))
        print(f"  {k[0]} -> {k[1]}: dR2={d:.6f}")
