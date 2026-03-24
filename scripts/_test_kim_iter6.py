"""Iteration 6: Warm-start-derived lambda on all channels.

Instead of oracle lambda, estimate lambda from per-channel warm-start R2.
More channels with correct lambda = wider LL gap = better discrimination.
"""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, build_semisynthetic_base, generate_coupling_gate
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
from cadence.significance.kim_filter import _estimate_ar
import torch

cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None]
s1, s2 = sess_list[-1][1], sess_list[0][1]
window = find_valid_window(s1, min_duration=1800)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

p1_raw = s1.get('p1_blendshapes', s1.get('p2_blendshapes'))
p2_raw = s2.get('p1_blendshapes', s2.get('p2_blendshapes'))
p1_ts = s1.get('p1_blendshapes_ts', s1.get('p2_blendshapes_ts'))
p2_ts = s2.get('p1_blendshapes_ts', s2.get('p2_blendshapes_ts'))

fs = 30.0
t_s, t_e = max(float(p1_ts[0]), float(p2_ts[0])), min(float(p1_ts[-1]), float(p2_ts[-1]))
t_e = min(t_e, t_s + 1800)
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)
C = min(p1_raw.shape[1], 52)

src = np.stack([np.interp(times, p1_ts, p1_raw[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts, p2_raw[:, c]) for c in range(C)]).T
for c in range(C):
    src[:, c] = (src[:, c] - src[:, c].mean()) / max(src[:, c].std(), 1e-8)
    tgt[:, c] = (tgt[:, c] - tgt[:, c].mean()) / max(tgt[:, c].std(), 1e-8)

coupled_aus = [22, 23, 27, 28, 29]
kappa, lag_samples = 0.4, int(2.0 * fs)
gate = generate_coupling_gate(T, fs, {'duty_cycle': 0.10, 'event_range_s': (3, 15), 'ramp_s': 1.0}, seed=42)
gate_mask = gate > 0.5

tgt_coupled = tgt.copy()
for ch in coupled_aus:
    p1_lag = np.roll(src[:, ch], lag_samples); p1_lag[:lag_samples] = 0
    alpha = kappa * gate
    tgt_coupled[:, ch] = alpha * p1_lag + np.sqrt(np.maximum(1-alpha**2, 0)) * tgt[:, ch]
for ch in coupled_aus:
    tgt_coupled[:, ch] = (tgt_coupled[:, ch] - tgt_coupled[:, ch].mean()) / max(tgt_coupled[:, ch].std(), 1e-8)

nb = 6
basis, _ = raised_cosine_basis(n_basis=nb, max_lag_s=5.0, min_lag_s=0.0, sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(basis, ar_order=0, device='cuda')
conv, _ = dm.convolve_source(torch.tensor(src, dtype=torch.float32, device='cuda'), np.ones(T, dtype=bool))
X_src = conv.view(T, C, nb).permute(1, 0, 2).cpu().numpy()
for c in range(C):
    rms = np.maximum(np.sqrt(np.mean(X_src[c]**2, axis=0)), 1e-8)
    X_src[c] /= rms

ar_order = 3
print(f"C={C}, T={T}, coupled={coupled_aus}, duty={gate_mask.mean():.1%}\n", flush=True)


def compute_all(y_mc, label):
    """Full pipeline: AR residuals, warm-start, per-channel lambda, fixed-b HMM."""
    y_res = np.zeros((C, T)); sigma2 = np.zeros(C)
    for c in range(C):
        a, s2 = _estimate_ar(y_mc[c], ar_order)
        sigma2[c] = s2
        yr = y_mc[c].copy()
        for k in range(ar_order):
            yr[ar_order:] -= a[k] * y_mc[c, ar_order-k-1:T-k-1]
        yr[:ar_order] = 0.0
        y_res[c] = yr

    # Per-channel warm-start R2 (windowed OLS)
    win_size = int(5.0 * fs); win_step = win_size // 2
    n_wins = max(1, (T - win_size) // win_step)
    r2_per_ch = np.zeros((C, n_wins))
    for wi in range(n_wins):
        s = wi * win_step; e = s + win_size
        for c in range(C):
            Y_w = y_res[c, s:e]; X_w = X_src[c, s:e]
            b_w = np.linalg.solve(X_w.T @ X_w + 1e-4*np.eye(nb), X_w.T @ Y_w)
            ss_res = np.sum((Y_w - X_w @ b_w)**2); ss_tot = np.sum(Y_w**2)
            r2_per_ch[c, wi] = max(1.0 - ss_res/max(ss_tot, 1e-8), 0.0)

    # Per-channel: mean of top-10% windows (captures peak coupling)
    top_pct = 10
    n_top = max(1, n_wins * top_pct // 100)
    lambda_est = np.zeros(C)
    for c in range(C):
        top_r2 = np.sort(r2_per_ch[c])[-n_top:]
        lambda_est[c] = float(np.mean(top_r2))

    # Threshold: only keep channels with lambda above median
    lam_thresh = np.median(lambda_est)
    active_est = lambda_est > lam_thresh
    # Normalize lambda to [0, 1]
    lam_max = max(lambda_est.max(), 1e-8)
    lambda_norm = np.where(active_est, lambda_est / lam_max, 0.0)

    n_active = active_est.sum()
    # Check which coupled AUs are active
    coupled_active = [c for c in coupled_aus if active_est[c]]
    print(f"  [{label}] Active: {n_active}/52, coupled active: {len(coupled_active)}/5",
          flush=True)
    print(f"  [{label}] Top-5 lambda channels: {np.argsort(-lambda_norm)[:5].tolist()}",
          flush=True)

    # Warm-start: use top-z windows for b estimation
    r2_t = np.zeros(T)
    for wi in range(n_wins):
        s = wi * win_step; e = s + win_size
        # Weighted average R2 across active channels
        r2_avg = np.mean([r2_per_ch[c, wi] for c in range(C) if active_est[c]])
        r2_t[s:e] = np.maximum(r2_t[s:e], r2_avg)
    z_ws = (r2_t - r2_t.mean()) / max(r2_t.std(), 1e-8)
    warm = z_ws > 1.5
    print(f"  [{label}] Warm-start frac: {warm.mean():.1%}", flush=True)

    # Estimate shared b from warm-start windows (all active channels)
    if warm.sum() > nb + 5 and n_active > 0:
        Y_s = np.concatenate([y_res[c, warm] for c in range(C) if active_est[c]])
        X_s = np.concatenate([lambda_norm[c] * X_src[c, warm] for c in range(C) if active_est[c]])
        b_fixed = np.linalg.solve(X_s.T @ X_s + 1e-4*np.eye(nb), X_s.T @ Y_s)
    else:
        b_fixed = np.zeros(nb)
    print(f"  [{label}] ||b|| = {np.linalg.norm(b_fixed):.4f}", flush=True)

    # HMM with estimated lambda
    A_tr = np.array([[0.99, 0.01], [0.01, 0.99]])
    xi_filt = np.zeros((T, 2)); xi_filt[0] = [0.5, 0.5]
    b_zero = np.linalg.norm(b_fixed) < 1e-10

    for t in range(1, T):
        xi_pred = A_tr.T @ xi_filt[t-1]; xi_pred = np.maximum(xi_pred, 1e-10)
        ll = np.zeros(2)
        for c in range(C):
            yt = y_res[c, t]; F = sigma2[c]
            ll[0] += -0.5 * (np.log(2*np.pi*F) + yt**2/F)
            if active_est[c] and not b_zero:
                pred = lambda_norm[c] * (X_src[c, t] @ b_fixed)
                ll[1] += -0.5 * (np.log(2*np.pi*F) + (yt-pred)**2/F)
            else:
                ll[1] += -0.5 * (np.log(2*np.pi*F) + yt**2/F)
        lj = ll + np.log(xi_pred); lj -= lj.max()
        j = np.exp(lj); xi_filt[t] = j / max(j.sum(), 1e-20)

    xi_s = np.zeros((T, 2)); xi_s[T-1] = xi_filt[T-1]
    for t in range(T-2, -1, -1):
        xp = np.maximum(A_tr.T @ xi_filt[t], 1e-10)
        r = xi_s[t+1] / xp
        for jj in range(2): xi_s[t,jj] = xi_filt[t,jj] * (A_tr[jj,0]*r[0]+A_tr[jj,1]*r[1])
        s = xi_s[t].sum()
        if s > 0: xi_s[t] /= s

    return np.clip(xi_s[:, 1], 0.0, 1.0), lambda_norm, active_est


# Run
print("=== COUPLED ===", flush=True)
t0 = time.perf_counter()
post_c, lam_c, act_c = compute_all(tgt_coupled.T, "coupled")
print(f"  Time: {time.perf_counter()-t0:.0f}s\n", flush=True)

print("=== NULL ===", flush=True)
t0 = time.perf_counter()
post_n, lam_n, act_n = compute_all(tgt.T, "null")
print(f"  Time: {time.perf_counter()-t0:.0f}s\n", flush=True)

frac_c = float((post_c > 0.5).mean()); frac_n = float((post_n > 0.5).mean())
hit = float((gate_mask & (post_c > 0.5)).sum() / max(gate_mask.sum(), 1))
fa = float((~gate_mask & (post_c > 0.5)).sum() / max((~gate_mask).sum(), 1))
iou = float((gate_mask & (post_c > 0.5)).sum() / max((gate_mask | (post_c > 0.5)).sum(), 1))

print(f"=== RESULTS ===", flush=True)
print(f"Coupled: frac={frac_c:.1%} hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)
print(f"Null:    frac={frac_n:.1%}", flush=True)
print(f"Disc:    {frac_c - frac_n:.1%}", flush=True)
