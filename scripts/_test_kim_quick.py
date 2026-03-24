"""Quick Kim filter test: best params on coupled semisynthetic data."""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, build_semisynthetic_base, inject_coupling_modality, _get_coupled_indices
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
session = inject_coupling_modality(base, target_mod, 0.4, duty_cycle=0.10, seed=42)
gate = session['coupling_gates'][target_mod]
fs = float(MODALITY_SPECS_V2[target_mod][1])
coupled_idx = _get_coupled_indices(target_mod)

p1, p2 = session[f'p1_{target_mod}'], session[f'p2_{target_mod}']
p1_ts, p2_ts = session[f'p1_{target_mod}_ts'], session[f'p2_{target_mod}_ts']
t_s, t_e = max(float(p1_ts[0]), float(p2_ts[0])), min(float(p1_ts[-1]), float(p2_ts[-1]))
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)
C = p1.shape[1]

src = np.stack([np.interp(times, p1_ts, p1[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts, p2[:, c]) for c in range(C)]).T
gate_native = np.interp(times, np.arange(len(gate))/fs, gate.astype(float)) > 0.5

category = get_pathway_category(target_mod, target_mod, cfg)
pw = cfg.get('pathway_temporal', {}).get(category, {'max_lag_seconds': 5.0, 'n_basis': 6})
basis, _ = raised_cosine_basis(n_basis=pw['n_basis'], max_lag_s=pw['max_lag_seconds'],
                                min_lag_s=0.0, sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(basis, ar_order=0, device='cuda')
conv, _ = dm.convolve_source(torch.tensor(src, dtype=torch.float32, device='cuda'),
                              np.ones(T, dtype=bool))
X_src = conv.view(T, C, pw['n_basis']).permute(1, 0, 2).cpu().numpy()
y_mc = tgt.T

print(f"kappa=0.4, duty=10%, C={C}, T={T}, gate_duty={gate_native.mean():.1%}", flush=True)
print(f"Coupled channels: {coupled_idx}", flush=True)

# Best params from sweep
Q, p_stay, em = 1e-6, 0.90, 5
print(f"Params: Q={Q}, p_stay={p_stay}, em={em}", flush=True)

t0 = time.perf_counter()
results = Parallel(n_jobs=-1)(
    delayed(_kim_filter_single_channel)(
        y_mc[c], X_src[c], ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em)
    for c in range(C))
print(f"Done in {time.perf_counter()-t0:.1f}s\n", flush=True)

posteriors = np.array([r[0] for r in results])
params = [r[1] for r in results]

print(f"{'ch':>3}  {'frac':>6}  {'sig2r':>7}  {'hit':>5}  {'fa':>5}  {'type'}", flush=True)
for c in range(C):
    post_on = posteriors[c] > 0.5
    frac = post_on.mean()
    sig2r = params[c]['sigma2_ratio']
    hit = float((gate_native & post_on).sum() / max(gate_native.sum(), 1))
    fa = float((~gate_native & post_on).sum() / max((~gate_native).sum(), 1))
    label = "COUPLED" if c in coupled_idx else "null"
    print(f"{c:>3}  {frac:>6.1%}  {sig2r:>7.3f}  {hit:>5.1%}  {fa:>5.1%}  {label}", flush=True)

coupled_fracs = [float((posteriors[c] > 0.5).mean()) for c in coupled_idx]
null_idx = [c for c in range(C) if c not in coupled_idx]
null_fracs = [float((posteriors[c] > 0.5).mean()) for c in null_idx]
print(f"\nCoupled mean: {np.mean(coupled_fracs):.1%}", flush=True)
print(f"Null mean:    {np.mean(null_fracs):.1%}", flush=True)
print(f"Disc:         {np.mean(coupled_fracs)-np.mean(null_fracs):.1%}", flush=True)
