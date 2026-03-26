"""Diagnostic: check z-score contrast between coupled and null timepoints."""
import sys, yaml, numpy as np
sys.path.insert(0, '.')

from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import build_semisynthetic_base, inject_coupling_modality
from scripts.run_semisynthetic import build_cross_dyad_pairs
from cadence.significance.temporal_localization import zscore_stouffer
from cadence.constants import MODALITY_SPECS_V2
from scipy.ndimage import uniform_filter1d

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

cache_dir = cfg.get('session_cache', 'session_cache')
excluded = set(cfg.get('excluded_sessions', []))
cached = discover_cached_sessions(cache_dir)
cached = [(n, p) for n, p in cached if n not in excluded]
sessions = []
for name, path in cached:
    try:
        s = load_session_from_cache(path, config=cfg)
        sessions.append((name, path, s))
    except Exception:
        pass

pairs = build_cross_dyad_pairs(sessions, n_pairs=1, seed=42)
name_a, sess_a, win_a, name_b, sess_b, win_b = pairs[0]
base = build_semisynthetic_base(sess_a, sess_b, win_a[0], win_a[1])

# Load the actual dR2 from a run - we need the EWLS output
# Instead, let's directly check what the z-score pipeline sees
# by loading a session and running just the surrogate comparison

# For now, just check the gate/posterior alignment issue
target_mod = 'blendshapes_v2'
gate_hz = MODALITY_SPECS_V2[target_mod][1]

for kappa in [0.0, 0.2, 0.4]:
    semi = inject_coupling_modality(base, target_mod, kappa,
                                     lag_s=2.0, seed=42 + int(kappa*100),
                                     duty_cycle=0.05)
    gate = semi.get('coupling_gates', {}).get(target_mod)
    if gate is not None:
        duty = np.mean(gate > 0.5)
        n_events = np.sum(np.diff((gate > 0.5).astype(int)) > 0)
        gate_t = np.arange(len(gate)) / gate_hz
        print(f"kappa={kappa}: duty={duty:.2%}, n_events={n_events}, "
              f"gate_len={len(gate)}, gate_hz={gate_hz}")
        # Event durations
        on = gate > 0.5
        changes = np.diff(on.astype(int))
        starts = np.where(changes > 0)[0] + 1
        ends = np.where(changes < 0)[0] + 1
        if on[0]: starts = np.concatenate([[0], starts])
        if on[-1]: ends = np.concatenate([ends, [len(on)]])
        durations = (ends[:len(starts)] - starts[:len(ends)]) / gate_hz
        if len(durations) > 0:
            print(f"  Event durations: {durations[:10]} ...")
            print(f"  Mean={np.mean(durations):.1f}s, min={np.min(durations):.1f}s, max={np.max(durations):.1f}s")
    else:
        print(f"kappa={kappa}: no gate (null)")
