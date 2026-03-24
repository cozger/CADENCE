"""Iteration 8: Direct EWLS dR2 thresholding (no Kim filter).

Compute per-channel dR2 via EWLS at native rate with short tau,
smooth, and threshold. The dR2 IS the coupling signal — no regime
switching needed. Use oracle channels (PCA 0-3) for matched diagonal.

Back to PCA semisynthetic (not raw AUs) since that's what the pipeline uses.
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
fs_native = float(MODALITY_SPECS_V2[target_mod][1])

# Extract PCA signals at native rate
p1_bl = session[f'p1_{target_mod}']
p2_bl = session[f'p2_{target_mod}']
p1_ts = session[f'p1_{target_mod}_ts']
p2_ts = session[f'p2_{target_mod}_ts']

t_s = max(float(p1_ts[0]), float(p2_ts[0]))
t_e = min(float(p1_ts[-1]), float(p2_ts[-1]))
T = int((t_e - t_s) * fs_native)
times = np.linspace(t_s, t_e, T)

C = p1_bl.shape[1]  # 31 (15 PCA + 15 deriv + activity)
src = np.stack([np.interp(times, p1_ts, p1_bl[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts, p2_bl[:, c]) for c in range(C)]).T

gate_times = np.arange(len(gate)) / fs_native
gate_native = np.interp(times, gate_times, gate.astype(float)) > 0.5

n_coupled = len(coupled_idx)
print(f"PCA BL: C={C}, T={T}, fs={fs_native} Hz", flush=True)
print(f"Coupled channels: {coupled_idx} ({n_coupled}/{C})", flush=True)
print(f"Gate duty: {gate_native.mean():.1%}\n", flush=True)

# Build basis at native rate
category = get_pathway_category(target_mod, target_mod, cfg)
pw = cfg.get('pathway_temporal', {}).get(category, {'max_lag_seconds': 5.0, 'n_basis': 6})
nb = pw['n_basis']
native_basis, _ = raised_cosine_basis(n_basis=nb, max_lag_s=pw['max_lag_seconds'],
                                       min_lag_s=0.0, sample_rate=fs_native, log_spacing=True)

device = torch.device('cuda')

# Discovery selects all 15 PCA + 15 deriv = first 30 channels typically
# For this test, use coupled_idx (oracle: PCA 0-3)
selected = list(range(min(C, 15)))  # all 15 PCA channels (what discovery would select)

# Build matched-diagonal EWLS per channel
src_sel = torch.tensor(src[:, selected], dtype=torch.float32, device=device)
tgt_sel = torch.tensor(tgt[:, selected], dtype=torch.float32, device=device)
C_sel = len(selected)

dm = DesignMatrixBuilder(native_basis, ar_order=0, device=device)
convolved, _ = dm.convolve_source(src_sel, np.ones(T, dtype=bool))
X_src_batch = convolved.view(T, C_sel, nb).permute(1, 0, 2)  # (C_sel, T, nb)

# AR terms
ar_order = 3
ar_parts = []
for lag in range(1, ar_order + 1):
    shifted = torch.zeros_like(tgt_sel)
    shifted[lag:] = tgt_sel[:-lag]
    ar_parts.append(shifted)
AR_batch = torch.stack(ar_parts, dim=2).T  # need (C_sel, T, ar_order)
AR_batch = torch.stack(ar_parts, dim=1).permute(2, 0, 1)  # (C_sel, T, ar_order)

# Full design: source basis + AR
X_full = torch.cat([X_src_batch, AR_batch], dim=2)  # (C_sel, T, nb+ar)
y_batch = tgt_sel.T.unsqueeze(2)  # (C_sel, T, 1)
valid_batch = torch.ones(C_sel, T, dtype=torch.bool, device=device)

# Sweep tau values for EWLS
tau_values = [1.0, 2.0, 3.0, 5.0, 8.0]
smooth_values = [1, 30, 90, 150, 300]  # samples at 30 Hz

print("Running EWLS per-channel dR2 sweep...", flush=True)

def run_one_tau(tau_s):
    solver = EWLSSolver(tau_seconds=tau_s, lambda_ridge=1e-3,
                         eval_rate=fs_native, device=device, min_effective_n=20)
    dr2_batch_out, _, _, _, _ = solver.solve_restricted_batched(
        X_full, AR_batch, y_batch, valid_batch)
    return dr2_batch_out.cpu().numpy()  # (C_sel, T)

# Run all taus (sequential on GPU — can't easily parallelize GPU EWLS)
results_tau = {}
for tau_s in tau_values:
    t0 = time.perf_counter()
    dr2 = run_one_tau(tau_s)
    dt = time.perf_counter() - t0
    results_tau[tau_s] = dr2
    # Quick stats
    coupled_dr2 = np.mean([dr2[i][gate_native[:T]].mean() for i, c in enumerate(selected) if c in coupled_idx])
    null_dr2 = np.mean([dr2[i][~gate_native[:T]].mean() for i, c in enumerate(selected) if c in coupled_idx])
    print(f"  tau={tau_s}s: coupled_dr2={coupled_dr2:.4f} null_dr2={null_dr2:.4f} "
          f"diff={coupled_dr2-null_dr2:.4f} ({dt:.1f}s)", flush=True)

# Sweep tau + smoothing + threshold
print(f"\nSweeping tau x smooth x threshold...", flush=True)

def eval_config(tau_s, smooth_w, pct):
    dr2 = results_tau[tau_s]
    # Average dR2 across coupled channels (oracle)
    coupled_ch_idx = [i for i, c in enumerate(selected) if c in coupled_idx]
    dr2_avg = np.mean([dr2[i] for i in coupled_ch_idx], axis=0)  # (T,)
    if smooth_w > 1:
        dr2_s = uniform_filter1d(dr2_avg, smooth_w, mode='nearest')
    else:
        dr2_s = dr2_avg
    thresh = np.percentile(dr2_s, pct)
    det = dr2_s > thresh
    T_g = min(len(det), len(gate_native))
    hit = float((gate_native[:T_g] & det[:T_g]).sum() / max(gate_native[:T_g].sum(), 1))
    fa = float((~gate_native[:T_g] & det[:T_g]).sum() / max((~gate_native[:T_g]).sum(), 1))
    inter = (gate_native[:T_g] & det[:T_g]).sum()
    union = (gate_native[:T_g] | det[:T_g]).sum()
    iou = float(inter / max(union, 1))
    return tau_s, smooth_w, pct, hit, fa, iou

pct_values = [50, 60, 70, 80, 85, 90, 95]
configs = list(itertools.product(tau_values, smooth_values, pct_values))
results = Parallel(n_jobs=-1)(
    delayed(eval_config)(t, s, p) for t, s, p in configs)

results.sort(key=lambda r: -r[5])
print(f"\n{'tau':>4} {'smooth':>6} {'pct':>4} | {'hit':>5} {'fa':>5} {'IoU':>5}", flush=True)
print("-" * 45, flush=True)
for tau_s, sw, pct, hit, fa, iou in results[:20]:
    print(f"{tau_s:>4.1f} {sw:>6} {pct:>4} | {hit:>5.1%} {fa:>5.1%} {iou:>5.1%}", flush=True)

good = [r for r in results if r[4] < 0.15]
if good:
    good.sort(key=lambda r: -r[3])
    print(f"\nBest hit (FA<15%):", flush=True)
    for tau_s, sw, pct, hit, fa, iou in good[:10]:
        print(f"  tau={tau_s} smooth={sw} pct={pct}: hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)

good2 = [r for r in results if r[4] < 0.05]
if good2:
    good2.sort(key=lambda r: -r[3])
    print(f"\nBest hit (FA<5%):", flush=True)
    for tau_s, sw, pct, hit, fa, iou in good2[:10]:
        print(f"  tau={tau_s} smooth={sw} pct={pct}: hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)
