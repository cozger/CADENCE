"""Quick test: compute BL->BL timing metrics for LLR+HSMM on CPU."""
import sys, os, yaml, time
import numpy as np
sys.path.insert(0, '.')

from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.synthetic import (
    find_valid_window, build_semisynthetic_base,
    inject_coupling_modality,
)
from scripts.run_semisynthetic import compute_timing_metrics, build_cross_dyad_pairs
from cadence.constants import MODALITY_SPECS_V2

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['stage2']['pathway_workers'] = 1  # sequential to avoid CUDA conflicts

cache_dir = cfg.get('session_cache', 'session_cache')
excluded = set(cfg.get('excluded_sessions', []))

print("Loading sessions...", flush=True)
cached = discover_cached_sessions(cache_dir)
cached = [(n, p) for n, p in cached if n not in excluded]
sessions = []
for name, path in cached:
    try:
        s = load_session_from_cache(path, config=cfg)
        sessions.append((name, path, s))
        print(f"  {name} ({s.get('duration',0):.0f}s)", flush=True)
    except Exception as e:
        print(f"  {name}: SKIP ({e})", flush=True)

pairs = build_cross_dyad_pairs(sessions, n_pairs=1, seed=42)
if not pairs:
    print("No valid pairs!")
    sys.exit(1)

name_a, sess_a, win_a, name_b, sess_b, win_b = pairs[0]
print(f"\nPair: {name_a} x {name_b}, window=[{win_a[0]}, {win_a[1]}]", flush=True)

base = build_semisynthetic_base(sess_a, sess_b, win_a[0], win_a[1])
print(f"Base built. Duration={base.get('duration',0):.0f}s", flush=True)

est = CouplingEstimator(cfg)
target_mod = 'blendshapes_v2'
gate_hz = MODALITY_SPECS_V2[target_mod][1]

for kappa in [0.0, 0.2, 0.4]:
    print(f"\n{'='*50}", flush=True)
    print(f"kappa={kappa}", flush=True)

    semi = inject_coupling_modality(base, target_mod, kappa,
                                     lag_s=2.0, seed=42 + int(kappa*100),
                                     duty_cycle=None)  # default=25%
    coupling_gate = semi.get('coupling_gates', {}).get(target_mod)

    t0 = time.time()
    try:
        result = est.analyze_session(semi, 'p1_to_p2')
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        # Reset CUDA after crash
        import torch
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        continue
    dt = time.time() - t0

    key = ('blendshapes_v2', 'blendshapes_v2')
    posterior = result.pathway_coupling_posterior.get(key)
    pw_times = result.pathway_times.get(key, result.times)
    screen_p = getattr(result, 'pathway_screening_p', {}).get(key)
    dr2 = result.pathway_dr2.get(key)
    detected = result.pathway_significant.get(key, False)

    if posterior is not None:
        print(f"  Posterior: mean={np.mean(posterior):.4f} "
              f"max={np.max(posterior):.4f} >0.5={np.mean(posterior > 0.5):.2%}", flush=True)
        if kappa > 0 and coupling_gate is not None:
            tm = compute_timing_metrics(coupling_gate, posterior, gate_hz, pw_times)
            print(f"  hit={tm.get('hit_rate',0):.2%} fa={tm.get('false_alarm',0):.2%} "
                  f"IoU={tm.get('iou',0):.3f} r={tm.get('temporal_corr',0):.3f}", flush=True)
    else:
        print("  No coupling_posterior!", flush=True)

    print(f"  screen_p={screen_p} detected={detected} "
          f"mean_dr2={np.nanmean(dr2) if dr2 is not None else 'N/A':.6f} "
          f"time={dt:.0f}s", flush=True)
