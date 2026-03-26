"""Iteration 7: Screening-style per-channel Gram statistic as lambda.

Compute the same Gram-whitened cross-covariance that Stage 1.5 uses,
with surrogate calibration. Use the per-channel p-values as lambda.
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
X_src = conv.view(T, C, nb).permute(1, 0, 2).cpu().numpy()  # (C, T, nb)
for c in range(C):
    rms = np.maximum(np.sqrt(np.mean(X_src[c]**2, axis=0)), 1e-8)
    X_src[c] /= rms

ar_order = 3
print(f"C={C}, T={T}, coupled={coupled_aus}, duty={gate_mask.mean():.1%}\n", flush=True)


def gram_screen_per_channel(y_mc, X_src_all, n_surr=100):
    """Compute per-channel Gram-whitened cross-covariance with surrogates.

    Returns per-channel p-values and statistics.
    """
    C, T = y_mc.shape

    # AR whiten target
    y_res = np.zeros((C, T))
    for c in range(C):
        a, _ = _estimate_ar(y_mc[c], ar_order)
        yr = y_mc[c].copy()
        for k in range(ar_order):
            yr[ar_order:] -= a[k] * y_mc[c, ar_order-k-1:T-k-1]
        yr[:ar_order] = 0.0
        y_res[c] = yr

    # Per-channel cross-covariance: cc[c] = X_src[c].T @ y_res[c] / T
    # Shape: (C, nb)
    cc_real = np.zeros((C, nb))
    for c in range(C):
        cc_real[c] = X_src_all[c].T @ y_res[c] / T

    # Gram matrix (average across channels for whitening)
    G = np.zeros((nb, nb))
    for c in range(C):
        G += X_src_all[c].T @ X_src_all[c] / T
    G /= C
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-8)
    G_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Whitened per-channel statistic
    wcc = cc_real @ G_inv_half.T  # (C, nb)
    stat_real = np.sum(wcc**2, axis=1)  # (C,) — chi-squared under null

    # Surrogates: circular shift source
    rng = np.random.RandomState(42)
    min_shift = int(0.1 * T); max_shift = T - min_shift

    def one_surr(seed):
        rng_s = np.random.RandomState(seed)
        shift = rng_s.randint(min_shift, max_shift + 1)
        X_shifted = np.roll(X_src_all, shift, axis=1)
        cc_s = np.zeros((C, nb))
        for c in range(C):
            cc_s[c] = X_shifted[c].T @ y_res[c] / T
        wcc_s = cc_s @ G_inv_half.T
        return np.sum(wcc_s**2, axis=1)  # (C,)

    surr_stats = np.array(Parallel(n_jobs=-1)(
        delayed(one_surr)(42 + k) for k in range(n_surr)))  # (K, C)

    # Per-channel p-values
    p_vals = np.mean(surr_stats >= stat_real[None, :], axis=0)  # (C,)

    return stat_real, p_vals, y_res


# === Run screening on coupled data ===
print("Gram screening on COUPLED data (100 surrogates)...", flush=True)
t0 = time.perf_counter()
stat_c, pvals_c, yres_c = gram_screen_per_channel(tgt_coupled.T, X_src, n_surr=100)
print(f"Done in {time.perf_counter()-t0:.0f}s\n", flush=True)

print("Per-channel Gram statistics (coupled):", flush=True)
print(f"{'ch':>3} {'stat':>8} {'p':>6} {'label'}", flush=True)
for c in range(C):
    label = "COUPLED" if c in coupled_aus else ""
    if pvals_c[c] < 0.20 or c in coupled_aus:
        print(f"{c:>3} {stat_c[c]:>8.2f} {pvals_c[c]:>6.3f} {label}", flush=True)

sig_channels = [c for c in range(C) if pvals_c[c] < 0.10]
print(f"\nSignificant channels (p<0.10): {sig_channels}", flush=True)
print(f"Coupled AUs detected: {[c for c in coupled_aus if c in sig_channels]}", flush=True)

# === Use significant channels as lambda, run fixed-b HMM ===
lam_screen = np.zeros(C)
for c in sig_channels:
    lam_screen[c] = stat_c[c]  # lambda = Gram statistic (stronger coupling = higher weight)
# Normalize
lam_max = max(lam_screen.max(), 1e-8)
lam_screen /= lam_max
active = lam_screen > 0.001

print(f"\nActive channels from screening: {active.sum()}", flush=True)

# Warm-start b
sigma2 = np.zeros(C)
for c in range(C):
    sigma2[c] = max(float(np.var(yres_c[c])), 1e-8)

win_size = int(5.0 * fs); win_step = win_size // 2
n_wins = max(1, (T - win_size) // win_step)
r2_t = np.zeros(T)
for wi in range(n_wins):
    s = wi * win_step; e = s + win_size
    r2_ch = []
    for c in range(C):
        if not active[c]: continue
        Y_w = yres_c[c, s:e]; X_w = lam_screen[c] * X_src[c, s:e]
        b_w = np.linalg.solve(X_w.T @ X_w + 1e-4*np.eye(nb), X_w.T @ Y_w)
        ss_res = np.sum((Y_w - X_w @ b_w)**2); ss_tot = np.sum(Y_w**2)
        r2_ch.append(max(1.0 - ss_res/max(ss_tot, 1e-8), 0.0))
    r2_t[s:e] = np.maximum(r2_t[s:e], np.mean(r2_ch) if r2_ch else 0.0)
z_ws = (r2_t - r2_t.mean()) / max(r2_t.std(), 1e-8)
warm = z_ws > 1.5
print(f"Warm-start frac: {warm.mean():.1%}", flush=True)

if warm.sum() > nb + 5:
    Y_s = np.concatenate([yres_c[c, warm] for c in range(C) if active[c]])
    X_s = np.concatenate([lam_screen[c] * X_src[c, warm] for c in range(C) if active[c]])
    b_fixed = np.linalg.solve(X_s.T @ X_s + 1e-4*np.eye(nb), X_s.T @ Y_s)
else:
    b_fixed = np.zeros(nb)
print(f"||b|| = {np.linalg.norm(b_fixed):.4f}", flush=True)

# Run HMM
A_tr = np.array([[0.99, 0.01], [0.01, 0.99]])
xi_filt = np.zeros((T, 2)); xi_filt[0] = [0.5, 0.5]
b_zero = np.linalg.norm(b_fixed) < 1e-10

for t in range(1, T):
    xi_pred = A_tr.T @ xi_filt[t-1]; xi_pred = np.maximum(xi_pred, 1e-10)
    ll = np.zeros(2)
    for c in range(C):
        yt = yres_c[c, t]; F = sigma2[c]
        ll[0] += -0.5 * (np.log(2*np.pi*F) + yt**2/F)
        if active[c] and not b_zero:
            pred = lam_screen[c] * (X_src[c, t] @ b_fixed)
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
post = np.clip(xi_s[:, 1], 0.0, 1.0)

frac = float((post > 0.5).mean())
hit = float((gate_mask & (post > 0.5)).sum() / max(gate_mask.sum(), 1))
fa = float((~gate_mask & (post > 0.5)).sum() / max((~gate_mask).sum(), 1))
iou = float((gate_mask & (post > 0.5)).sum() / max((gate_mask | (post > 0.5)).sum(), 1))

print(f"\n=== RESULTS (screening-derived lambda) ===", flush=True)
print(f"frac={frac:.1%} hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)
