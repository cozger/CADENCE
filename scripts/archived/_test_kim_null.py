"""Kim filter null test: kappa=0 on semisynthetic data.

Uses the best parameter set from the sweep (Q=1e-6, p_stay=0.90, em=5).
Checks per-channel coupling fractions when there is NO coupling injected.
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, build_semisynthetic_base
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
from cadence.significance.kim_filter import _kim_filter_single_channel
import torch

cfg = load_config('configs/default.yaml')

# --- Build semisynthetic session (NO coupling) ---
print("Loading sessions...", flush=True)
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = []
for name, path in session_entries:
    s = load_session_from_cache(path, cfg)
    if s is not None:
        sess_list.append((name, s))
        print(f"  {name} ({s.get('duration',0):.0f}s)", flush=True)

s1_name, s1 = sess_list[-1]
s2_name, s2 = sess_list[0]
print(f"\nPair: P1={s1_name}, P2={s2_name}", flush=True)

window = find_valid_window(s1, min_duration=1800)
t_start, t_end = window
print(f"  Window: [{t_start:.0f}s, {t_end:.0f}s]", flush=True)

print("Building semisynthetic base (NO coupling)...", flush=True)
base = build_semisynthetic_base(s1, s2, t_start, t_end)

# NO injection — use base directly (kappa=0)
session = base
target_mod = 'blendshapes_v2'

# --- Extract BL signals at native rate ---
p1_bl = session[f'p1_{target_mod}']
p2_bl = session[f'p2_{target_mod}']
p1_ts = session[f'p1_{target_mod}_ts']
p2_ts = session[f'p2_{target_mod}_ts']
fs_native = float(MODALITY_SPECS_V2[target_mod][1])

t_s = max(float(p1_ts[0]), float(p2_ts[0]))
t_e = min(float(p1_ts[-1]), float(p2_ts[-1]))
T_native = int((t_e - t_s) * fs_native)
native_times = np.linspace(t_s, t_e, T_native)

C_all = p1_bl.shape[1]
src_native = np.stack([np.interp(native_times, p1_ts, p1_bl[:, c])
                        for c in range(C_all)]).T
tgt_native = np.stack([np.interp(native_times, p2_ts, p2_bl[:, c])
                        for c in range(C_all)]).T

print(f"Signals: C={C_all}, T={T_native}, fs={fs_native} Hz", flush=True)

# --- Build basis at native rate ---
category = get_pathway_category(target_mod, target_mod, cfg)
pw_temporal = cfg.get('pathway_temporal', {}).get(
    category, {'max_lag_seconds': 5.0, 'n_basis': 6})
n_basis = pw_temporal['n_basis']
max_lag = pw_temporal['max_lag_seconds']

native_basis, _ = raised_cosine_basis(
    n_basis=n_basis, max_lag_s=max_lag, min_lag_s=0.0,
    sample_rate=fs_native, log_spacing=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dm = DesignMatrixBuilder(native_basis, ar_order=0, device=device)
src_t = torch.tensor(src_native, dtype=torch.float32, device=device)
convolved, _ = dm.convolve_source(src_t, np.ones(T_native, dtype=bool))
X_src = convolved.view(T_native, C_all, n_basis).permute(
    1, 0, 2).cpu().numpy()

y_mc = tgt_native.T

# --- Run Kim filter with best params from sweep ---
Q = 1e-6
p_stay = 0.90
em_iter = 5

print(f"\nRunning Kim filter (NULL, kappa=0): Q={Q}, p_stay={p_stay}, em={em_iter}",
      flush=True)

t0 = time.perf_counter()
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(_kim_filter_single_channel)(
        y_mc[c], X_src[c], ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em_iter)
    for c in range(C_all))
elapsed = time.perf_counter() - t0

posteriors = np.array([r[0] for r in results])
params = [r[1] for r in results]

print(f"\nDone in {elapsed:.1f}s", flush=True)
print(f"\n=== NULL (kappa=0) per-channel results ===", flush=True)
print(f"{'ch':>3}  {'frac':>6}  {'sig2r':>7}", flush=True)
for c in range(C_all):
    frac = (posteriors[c] > 0.5).mean()
    sig2r = params[c]['sigma2_ratio']
    print(f"{c:>3}  {frac:>6.1%}  {sig2r:>7.3f}", flush=True)

fracs = [(posteriors[c] > 0.5).mean() for c in range(C_all)]
print(f"\n  Mean coupling fraction (NULL): {np.mean(fracs):.1%}", flush=True)
print(f"  Max coupling fraction (NULL):  {np.max(fracs):.1%}", flush=True)
print(f"  Min coupling fraction (NULL):  {np.min(fracs):.1%}", flush=True)
print(f"  Std coupling fraction (NULL):  {np.std(fracs):.1%}", flush=True)

# Compare: what was the coupled mean at same params?
print(f"\n  (For reference: coupled sweep had coupled=23.7%, null=14.3%)", flush=True)
print(f"  Expected null-only mean: ~14% if the FA floor is intrinsic", flush=True)
