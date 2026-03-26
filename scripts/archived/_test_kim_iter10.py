"""Iteration 10: 2-state Gaussian HMM on EWLS dR2 timecourse.

No regression, no basis, no overfitting.
Regime 0: dR2 ~ N(mu_null, sigma_null)
Regime 1: dR2 ~ N(mu_coupled, sigma_coupled)
With mu_coupled > mu_null from the EWLS regression output.

Use per-channel dR2 from tau=8s (where coupled slightly exceeds null),
averaged across oracle coupled channels, then HMM.
"""
import numpy as np
import time, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (find_valid_window, build_semisynthetic_base,
                                inject_coupling_modality, _get_coupled_indices)
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
from cadence.regression.ewls import EWLSSolver
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
coupled_idx = _get_coupled_indices(target_mod)
fs = float(MODALITY_SPECS_V2[target_mod][1])

p1_bl = session[f'p1_{target_mod}']
p2_bl = session[f'p2_{target_mod}']
p1_ts = session[f'p1_{target_mod}_ts']
p2_ts = session[f'p2_{target_mod}_ts']

t_s = max(float(p1_ts[0]), float(p2_ts[0]))
t_e = min(float(p1_ts[-1]), float(p2_ts[-1]))
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)
C = p1_bl.shape[1]
selected = list(range(min(C, 15)))
C_sel = len(selected)

src = np.stack([np.interp(times, p1_ts, p1_bl[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts, p2_bl[:, c]) for c in range(C)]).T

gate_times = np.arange(len(gate)) / fs
gate_native = np.interp(times, gate_times, gate.astype(float)) > 0.5

print(f"C_sel={C_sel}, T={T}, coupled={coupled_idx}, duty={gate_native.mean():.1%}\n", flush=True)

# Compute EWLS per-channel dR2 at multiple tau
device = torch.device('cuda')
nb = 6
native_basis, _ = raised_cosine_basis(n_basis=nb, max_lag_s=5.0, min_lag_s=0.0,
                                       sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(native_basis, ar_order=0, device=device)

src_sel = torch.tensor(src[:, selected], dtype=torch.float32, device=device)
tgt_sel = torch.tensor(tgt[:, selected], dtype=torch.float32, device=device)
conv, _ = dm.convolve_source(src_sel, np.ones(T, dtype=bool))
X_src_batch = conv.view(T, C_sel, nb).permute(1, 0, 2)

ar_order = 3
ar_parts = []
for lag in range(1, ar_order + 1):
    shifted = torch.zeros_like(tgt_sel)
    shifted[lag:] = tgt_sel[:-lag]
    ar_parts.append(shifted)
AR_batch = torch.stack(ar_parts, dim=1).permute(2, 0, 1)

X_full = torch.cat([X_src_batch, AR_batch], dim=2)
y_batch = tgt_sel.T.unsqueeze(2)
valid_batch = torch.ones(C_sel, T, dtype=torch.bool, device=device)

# Compute dR2 for each tau
for tau_s in [3.0, 5.0, 8.0]:
    solver = EWLSSolver(tau_seconds=tau_s, lambda_ridge=1e-3,
                         eval_rate=fs, device=device, min_effective_n=20)
    dr2_all, _, _, _, _ = solver.solve_restricted_batched(
        X_full, AR_batch, y_batch, valid_batch)
    dr2_np = dr2_all.cpu().numpy()  # (C_sel, T)

    # Average dR2 across coupled channels (oracle)
    coupled_ch_idx = [i for i, c in enumerate(selected) if c in coupled_idx]
    dr2_coupled_avg = np.mean([dr2_np[i] for i in coupled_ch_idx], axis=0)

    # Also try: average across ALL selected channels
    dr2_all_avg = dr2_np.mean(axis=0)

    print(f"tau={tau_s}s:", flush=True)
    print(f"  Coupled-ch avg: during_gate={dr2_coupled_avg[gate_native].mean():.5f} "
          f"outside={dr2_coupled_avg[~gate_native].mean():.5f} "
          f"diff={dr2_coupled_avg[gate_native].mean()-dr2_coupled_avg[~gate_native].mean():.5f}", flush=True)
    print(f"  All-ch avg:     during_gate={dr2_all_avg[gate_native].mean():.5f} "
          f"outside={dr2_all_avg[~gate_native].mean():.5f} "
          f"diff={dr2_all_avg[gate_native].mean()-dr2_all_avg[~gate_native].mean():.5f}", flush=True)

    # Run simple 2-state Gaussian HMM on the dR2 timecourse
    def run_gauss_hmm(obs, p01, p11):
        """2-state Gaussian HMM. State 0 = lower quartile stats, State 1 = upper quartile."""
        T_obs = len(obs)
        # Initialize from data quartiles
        q25 = np.percentile(obs, 25); q75 = np.percentile(obs, 75)
        mu0 = float(obs[obs < q25].mean()) if (obs < q25).sum() > 5 else float(obs.mean())
        mu1 = float(obs[obs > q75].mean()) if (obs > q75).sum() > 5 else float(obs.mean()) + 0.001
        mu1 = max(mu1, mu0 + 1e-6)
        sig0 = max(float(obs[obs < q25].std()), 1e-8) if (obs < q25).sum() > 5 else max(float(obs.std()), 1e-8)
        sig1 = max(float(obs[obs > q75].std()), 1e-8) if (obs > q75).sum() > 5 else sig0

        A_tr = np.array([[1-p01, p01], [1-p11, p11]])
        xi = np.zeros((T_obs, 2)); xi[0] = [0.9, 0.1]

        for t in range(1, T_obs):
            xp = A_tr.T @ xi[t-1]; xp = np.maximum(xp, 1e-10)
            ll0 = -0.5*((obs[t]-mu0)**2/sig0**2 + np.log(2*np.pi*sig0**2))
            ll1 = -0.5*((obs[t]-mu1)**2/sig1**2 + np.log(2*np.pi*sig1**2))
            lj = np.array([ll0, ll1]) + np.log(xp)
            lj -= lj.max()
            j = np.exp(lj); xi[t] = j / max(j.sum(), 1e-20)

        # Smoother
        xs = np.zeros((T_obs, 2)); xs[T_obs-1] = xi[T_obs-1]
        for t in range(T_obs-2, -1, -1):
            xp = np.maximum(A_tr.T @ xi[t], 1e-10)
            r = xs[t+1] / xp
            for jj in range(2): xs[t,jj] = xi[t,jj]*(A_tr[jj,0]*r[0]+A_tr[jj,1]*r[1])
            s = xs[t].sum()
            if s > 0: xs[t] /= s
        return np.clip(xs[:, 1], 0.0, 1.0)

    # Sweep smoothing + HMM params on coupled-ch avg dR2
    smooth_values = [1, 30, 90, 150, 300]
    p01_values = [0.005, 0.01, 0.02, 0.05]
    p11_values = [0.95, 0.99]

    def eval_one(sw, p01, p11):
        obs = dr2_coupled_avg.copy()
        if sw > 1: obs = uniform_filter1d(obs, sw, mode='nearest')
        post = run_gauss_hmm(obs, p01, p11)
        T_g = min(len(post), len(gate_native))
        det = post[:T_g] > 0.5
        hit = float((gate_native[:T_g] & det).sum() / max(gate_native[:T_g].sum(), 1))
        fa = float((~gate_native[:T_g] & det).sum() / max((~gate_native[:T_g]).sum(), 1))
        iou = float((gate_native[:T_g] & det).sum() / max((gate_native[:T_g] | det).sum(), 1))
        return sw, p01, p11, hit, fa, iou

    configs = list(itertools.product(smooth_values, p01_values, p11_values))
    results = Parallel(n_jobs=-1)(delayed(eval_one)(s, p, q) for s, p, q in configs)

    good = [r for r in results if r[4] < 0.15]
    if good:
        good.sort(key=lambda r: -r[3])
        print(f"  Best hit (FA<15%) for tau={tau_s}:", flush=True)
        for sw, p01, p11, hit, fa, iou in good[:3]:
            print(f"    smooth={sw} p01={p01} p11={p11}: hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)
    else:
        results.sort(key=lambda r: -r[5])
        print(f"  Best IoU for tau={tau_s}:", flush=True)
        for sw, p01, p11, hit, fa, iou in results[:3]:
            print(f"    smooth={sw} p01={p01} p11={p11}: hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)
