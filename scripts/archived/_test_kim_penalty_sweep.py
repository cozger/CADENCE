"""Sweep complexity penalty on both null and coupled semisynthetic data."""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (find_valid_window, build_semisynthetic_base,
                                inject_coupling_modality, _get_coupled_indices)
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
from cadence.significance.kim_filter import _kim_filter_single_channel
import torch

cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None]

s1, s2 = sess_list[-1][1], sess_list[0][1]
window = find_valid_window(s1, min_duration=1800)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

target_mod = 'blendshapes_v2'
coupled_idx = _get_coupled_indices(target_mod)
fs = float(MODALITY_SPECS_V2[target_mod][1])

# Build null and coupled sessions
session_null = base
session_coupled = inject_coupling_modality(base, target_mod, 0.4,
                                            duty_cycle=0.10, seed=42)
gate = session_coupled['coupling_gates'][target_mod]

# Extract signals at native rate (shared setup)
p1_ts = base[f'p1_{target_mod}_ts']
p2_ts = base[f'p2_{target_mod}_ts']
t_s = max(float(p1_ts[0]), float(p2_ts[0]))
t_e = min(float(p1_ts[-1]), float(p2_ts[-1]))
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)
C = base[f'p1_{target_mod}'].shape[1]

# Source is always P1 from base (same for null and coupled)
src = np.stack([np.interp(times, p1_ts, base[f'p1_{target_mod}'][:, c])
                for c in range(C)]).T

# Two targets: null P2 and coupled P2
tgt_null = np.stack([np.interp(times, p2_ts, session_null[f'p2_{target_mod}'][:, c])
                      for c in range(C)]).T
tgt_coupled = np.stack([np.interp(times, p2_ts, session_coupled[f'p2_{target_mod}'][:, c])
                         for c in range(C)]).T

gate_native = np.interp(times, np.arange(len(gate))/fs, gate.astype(float)) > 0.5

# Basis convolution (shared)
category = get_pathway_category(target_mod, target_mod, cfg)
pw = cfg.get('pathway_temporal', {}).get(category, {'max_lag_seconds': 5.0, 'n_basis': 6})
basis, _ = raised_cosine_basis(n_basis=pw['n_basis'], max_lag_s=pw['max_lag_seconds'],
                                min_lag_s=0.0, sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(basis, ar_order=0, device='cuda')
conv, _ = dm.convolve_source(torch.tensor(src, dtype=torch.float32, device='cuda'),
                              np.ones(T, dtype=bool))
X_src = conv.view(T, C, pw['n_basis']).permute(1, 0, 2).cpu().numpy()

nb = pw['n_basis']
Q, p_stay, em = 1e-6, 0.90, 5

print(f"C={C}, T={T}, gate_duty={gate_native.mean():.1%}, nb={nb}", flush=True)
print(f"Coupled channels: {coupled_idx}", flush=True)
print(f"Params: Q={Q}, p_stay={p_stay}, em={em}", flush=True)

# Sweep penalty values
penalties = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def run_one_penalty(penalty, y_mc, label):
    """Run all channels with one penalty value."""
    results = [_kim_filter_single_channel(
        y_mc[c], X_src[c], ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em, complexity_penalty=penalty)
        for c in range(C)]
    fracs = [(results[c][0] > 0.5).mean() for c in range(C)]
    coupled_mean = np.mean([fracs[c] for c in coupled_idx])
    null_mean = np.mean([fracs[c] for c in range(C) if c not in coupled_idx])
    return label, penalty, coupled_mean, null_mean

total_jobs = len(penalties) * 2 * C
print(f"\nSweeping {len(penalties)} penalties x 2 conditions x {C} ch = {total_jobs} jobs...",
      flush=True)

def run_single(penalty, y, ch, label):
    post, params = _kim_filter_single_channel(
        y, X_src[ch], ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em, complexity_penalty=penalty)
    return (penalty, label, ch, float((post > 0.5).mean()))

t0 = time.perf_counter()
all_results = Parallel(n_jobs=-1, verbose=1)(
    delayed(run_single)(pen, tgt.T[ch] if label == 'coupled' else tgt_null.T[ch], ch, label)
    for pen in penalties
    for label, tgt in [('coupled', tgt_coupled), ('null', tgt_null)]
    for ch in range(C))
print(f"Done in {time.perf_counter()-t0:.1f}s\n", flush=True)

# Regroup
from collections import defaultdict
grouped = defaultdict(lambda: defaultdict(dict))
for pen, label, ch, frac in all_results:
    grouped[pen][label][ch] = frac

print(f"{'penalty':>8} | {'coup_coupled':>12} {'coup_null':>10} {'disc':>6} | "
      f"{'null_all':>8} {'null_disc':>9}", flush=True)
print("-" * 75, flush=True)

for pen in penalties:
    cc = np.mean([grouped[pen]['coupled'][c] for c in coupled_idx])
    cn = np.mean([grouped[pen]['coupled'][c] for c in range(C) if c not in coupled_idx])
    nn = np.mean([grouped[pen]['null'][c] for c in range(C)])
    # Null disc: coupled_idx vs rest on null data (should be ~0)
    nn_c = np.mean([grouped[pen]['null'][c] for c in coupled_idx])
    nn_n = np.mean([grouped[pen]['null'][c] for c in range(C) if c not in coupled_idx])
    print(f"{pen:>8.2f} | {cc:>12.1%} {cn:>10.1%} {cc-cn:>6.1%} | "
          f"{nn:>8.1%} {nn_c-nn_n:>9.1%}", flush=True)
