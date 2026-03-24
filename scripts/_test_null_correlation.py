"""Diagnostic: is the 15% FA floor from the Kim filter or the data?

Compute raw correlation between basis-convolved P1 and P2 AR residuals
at kappa=0. If correlations are nonzero, the data has shared structure
independent of any detection method.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, build_semisynthetic_base
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
import torch

cfg = load_config('configs/default.yaml')

# Build null semisynthetic
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = []
for name, path in session_entries:
    s = load_session_from_cache(path, cfg)
    if s is not None:
        sess_list.append((name, s))

s1 = sess_list[-1][1]  # y_32
s2 = sess_list[0][1]   # y01
window = find_valid_window(s1, min_duration=1800)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

target_mod = 'blendshapes_v2'
p1_bl = base[f'p1_{target_mod}']
p2_bl = base[f'p2_{target_mod}']
p1_ts = base[f'p1_{target_mod}_ts']
p2_ts = base[f'p2_{target_mod}_ts']
fs = float(MODALITY_SPECS_V2[target_mod][1])

t_s = max(float(p1_ts[0]), float(p2_ts[0]))
t_e = min(float(p1_ts[-1]), float(p2_ts[-1]))
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)

C = p1_bl.shape[1]
src = np.stack([np.interp(times, p1_ts, p1_bl[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts, p2_bl[:, c]) for c in range(C)]).T

# Basis convolution
category = get_pathway_category(target_mod, target_mod, cfg)
pw = cfg.get('pathway_temporal', {}).get(category, {'max_lag_seconds': 5.0, 'n_basis': 6})
basis, _ = raised_cosine_basis(n_basis=pw['n_basis'], max_lag_s=pw['max_lag_seconds'],
                                min_lag_s=0.0, sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(basis, ar_order=0, device='cuda')
src_t = torch.tensor(src, dtype=torch.float32, device='cuda')
conv, _ = dm.convolve_source(src_t, np.ones(T, dtype=bool))
X_src = conv.view(T, C, pw['n_basis']).permute(1, 0, 2).cpu().numpy()  # (C, T, nb)

print(f"NULL test: C={C}, T={T}, fs={fs}, nb={pw['n_basis']}", flush=True)

# AR(3) residuals for P2
ar_order = 3
print(f"\n=== Session-level correlation: basis-convolved P1 vs P2 AR residual ===", flush=True)
print(f"{'ch':>3}  {'max_r':>7}  {'mean_r':>7}  {'R2_ols':>7}  {'label'}", flush=True)

for c in range(C):
    y = tgt[:, c]
    # AR residual
    Y = y[ar_order:]
    X_ar = np.column_stack([y[ar_order-k-1:T-k-1] for k in range(ar_order)])
    a = np.linalg.solve(X_ar.T @ X_ar + 1e-6 * np.eye(ar_order), X_ar.T @ Y)
    y_res = np.zeros(T)
    y_res[ar_order:] = Y - X_ar @ a

    # Correlations between each basis function and AR residual
    x_basis = X_src[c]  # (T, nb)
    # Standardize
    x_std = x_basis / np.maximum(np.sqrt(np.mean(x_basis**2, axis=0, keepdims=True)), 1e-8)
    y_std = y_res / max(np.std(y_res), 1e-8)

    corrs = np.array([np.corrcoef(x_std[:, b], y_std)[0, 1] for b in range(pw['n_basis'])])
    max_r = float(np.max(np.abs(corrs)))
    mean_r = float(np.mean(np.abs(corrs)))

    # OLS R2: regress y_res on all basis functions
    X_all = x_std[ar_order:]
    Y_res = y_std[ar_order:]
    beta = np.linalg.solve(X_all.T @ X_all + 1e-4 * np.eye(pw['n_basis']), X_all.T @ Y_res)
    pred = X_all @ beta
    ss_res = np.sum((Y_res - pred) ** 2)
    ss_tot = np.sum(Y_res ** 2)
    r2 = max(1.0 - ss_res / max(ss_tot, 1e-8), 0.0)

    label = "PCA" if c < 15 else ("deriv" if c < 30 else "activity")
    print(f"{c:>3}  {max_r:>7.4f}  {mean_r:>7.4f}  {r2:>7.4f}  {label}", flush=True)

print(f"\nIf R2 >> 0 at kappa=0, the data has shared structure.", flush=True)
print(f"If R2 ~ 0, the 15% FA floor is a Kim filter problem.", flush=True)
