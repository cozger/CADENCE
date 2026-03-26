"""Diagnostic: measure z-score contrast during coupled vs uncoupled periods."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.synthetic import (
    build_semisynthetic_base, inject_coupling_modality,
    _get_coupled_indices, MODALITY_SPECS_V2,
)

config = load_config()
sessions_info = discover_cached_sessions(config['session_cache'])
sessions = {}
for name, path in sessions_info:
    sessions[name] = load_session_from_cache(path, config=config)
    if len(sessions) >= 2:
        break

names = list(sessions.keys())
base = build_semisynthetic_base(sessions[names[0]], sessions[names[1]], config)
if base is None:
    print("Failed to build base"); sys.exit(1)

# Inject coupling at kappa=0.4 into blendshapes
semi = inject_coupling_modality(base, 'blendshapes_v2', 0.4,
                                 lag_s=2.0, seed=42)
gate = semi['coupling_gates']['blendshapes_v2']
coupled_idx = _get_coupled_indices('blendshapes_v2')
print(f"Coupled channels: {coupled_idx}")
print(f"Gate: {gate.shape}, duty={gate.mean():.2%}, n_events={np.sum(np.diff(gate.astype(int)) == 1)}")

# Run analysis
estimator = CouplingEstimator(config)
result = estimator.analyze_session(semi, 'p1_to_p2')

# Get the BL->BL matched-diagonal results
key = ('blendshapes_v2', 'blendshapes_v2')
if key not in result.pathway_zscore_posterior:
    print("No TL posterior found for BL->BL")
    sys.exit(1)

posterior = result.pathway_zscore_posterior[key]
print(f"\nTL Posterior: mean={posterior.mean():.4f}, max={posterior.max():.4f}, >0.5={np.mean(posterior>0.5):.2%}")

# Get eval_rate and align gate to eval grid
bl_hz = MODALITY_SPECS_V2['blendshapes_v2'][1]  # 30 Hz native
eval_rate = config.get('eval_rate_overrides', {}).get('blendshapes_v2', config['ewls']['eval_rate'])
T_post = len(posterior)
T_gate = len(gate)
# Resample gate to posterior length
gate_eval = np.interp(np.linspace(0, T_gate/bl_hz, T_post),
                       np.arange(T_gate)/bl_hz, gate.astype(float))
gate_bool = gate_eval > 0.5

print(f"\nGate at eval_rate={eval_rate}Hz: {gate_bool.sum()}/{len(gate_bool)} coupled ({gate_bool.mean():.1%})")

# Analyze posterior vs gate
post_coupled = posterior[gate_bool]
post_uncoupled = posterior[~gate_bool]
print(f"Posterior during COUPLED:   mean={post_coupled.mean():.4f}, >0.5={np.mean(post_coupled>0.5):.2%}")
print(f"Posterior during UNCOUPLED: mean={post_uncoupled.mean():.4f}, >0.5={np.mean(post_uncoupled>0.5):.2%}")

# Check if dr2_perchannel is accessible (it's stored internally)
# Try getting the raw dR2 from the result
dr2_key = result.pathway_dr2.get(key)
if dr2_key is not None:
    dr2 = dr2_key
    T_dr2 = len(dr2)
    gate_dr2 = np.interp(np.linspace(0, T_gate/bl_hz, T_dr2),
                          np.arange(T_gate)/bl_hz, gate.astype(float)) > 0.5
    print(f"\ndR2 timecourse: T={T_dr2}")
    print(f"dR2 during COUPLED:   mean={dr2[gate_dr2].mean():.6f}")
    print(f"dR2 during UNCOUPLED: mean={dr2[~gate_dr2].mean():.6f}")
    print(f"dR2 contrast: {dr2[gate_dr2].mean() - dr2[~gate_dr2].mean():.6f}")

# Temporal correlation
corr = np.corrcoef(gate_eval, posterior)[0, 1]
print(f"\nCorr(gate, posterior): {corr:.4f}")

# Also check if we have per-channel info in diagnostics
print("\n--- Per-pathway results ---")
for k, v in result.pathway_significant.items():
    if k[1] == 'blendshapes_v2':
        p = result.pathway_screening_p.get(k, 'N/A')
        dr2 = result.pathway_dr2.get(k)
        dr2_str = f"{dr2.mean():.6f}" if dr2 is not None else "N/A"
        print(f"  {k[0]}->{k[1]}: sig={v}, screen_p={p}, mean_dr2={dr2_str}")
