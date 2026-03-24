"""Fixed-b HMM: warm-start b, sweep p_stay to find hit/FA tradeoff."""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
p1_ts_raw = s1.get('p1_blendshapes_ts', s1.get('p2_blendshapes_ts'))
p2_ts_raw = s2.get('p1_blendshapes_ts', s2.get('p2_blendshapes_ts'))

fs = 30.0
t_s, t_e = max(float(p1_ts_raw[0]), float(p2_ts_raw[0])), min(float(p1_ts_raw[-1]), float(p2_ts_raw[-1]))
t_e = min(t_e, t_s + 1800)
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)
C = min(p1_raw.shape[1], 52)

src = np.stack([np.interp(times, p1_ts_raw, p1_raw[:, c]) for c in range(C)]).T
tgt = np.stack([np.interp(times, p2_ts_raw, p2_raw[:, c]) for c in range(C)]).T
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

lam = np.zeros(C)
for c in coupled_aus: lam[c] = 1.0
active = lam > 0.001
ar_order = 3

print(f"C={C}, T={T}, coupled={coupled_aus}, duty={gate_mask.mean():.1%}\n", flush=True)


def get_warmstart_b(y_mc):
    """Estimate b from warm-start windows."""
    C, T = y_mc.shape
    y_res = np.zeros((C, T))
    sigma2 = np.zeros(C)
    for c in range(C):
        a, s2 = _estimate_ar(y_mc[c], ar_order)
        sigma2[c] = s2
        yr = y_mc[c].copy()
        for k in range(ar_order):
            yr[ar_order:] -= a[k] * y_mc[c, ar_order-k-1:T-k-1]
        yr[:ar_order] = 0.0
        y_res[c] = yr

    win_size = int(5.0 * fs); win_step = win_size // 2
    n_wins = max(1, (T - win_size) // win_step)
    r2_t = np.zeros(T)
    for wi in range(n_wins):
        s = wi * win_step; e = s + win_size
        r2_ch = []
        for c in range(C):
            if not active[c]: continue
            Y_w = y_res[c, s:e]; X_w = lam[c] * X_src[c, s:e]
            b_w = np.linalg.solve(X_w.T @ X_w + 1e-4*np.eye(nb), X_w.T @ Y_w)
            ss_res = np.sum((Y_w - X_w @ b_w)**2); ss_tot = np.sum(Y_w**2)
            r2_ch.append(max(1.0 - ss_res/max(ss_tot, 1e-8), 0.0))
        r2_t[s:e] = np.maximum(r2_t[s:e], np.mean(r2_ch) if r2_ch else 0.0)

    z_ws = (r2_t - r2_t.mean()) / max(r2_t.std(), 1e-8)
    warm = z_ws > 1.5

    if warm.sum() > nb + 5:
        Y_s = np.concatenate([y_res[c, warm] for c in range(C) if active[c]])
        X_s = np.concatenate([lam[c] * X_src[c, warm] for c in range(C) if active[c]])
        b = np.linalg.solve(X_s.T @ X_s + 1e-4*np.eye(nb), X_s.T @ Y_s)
    else:
        b = np.zeros(nb)

    return b, y_res, sigma2


def run_hmm(y_res, sigma2, b_fixed, p_stay=0.99, p01=None, p11=None,
            null_correction=False):
    """Fixed-b HMM forward-backward with optional null-corrected likelihood.

    If null_correction=True, regime 1's F includes the expected prediction
    energy under null: F1 = sigma2 + (h@b)^2. This penalizes regime 1
    for adding prediction noise when coupling isn't active.
    """
    C, T = y_res.shape
    if p01 is not None and p11 is not None:
        A_tr = np.array([[1-p01, p01], [1-p11, p11]])
    else:
        A_tr = np.array([[p_stay, 1-p_stay], [1-p_stay, p_stay]])
    xi_filt = np.zeros((T, 2)); xi_filt[0] = [0.5, 0.5]
    b_zero = np.linalg.norm(b_fixed) < 1e-10

    for t in range(1, T):
        xi_pred = A_tr.T @ xi_filt[t-1]; xi_pred = np.maximum(xi_pred, 1e-10)
        ll = np.zeros(2)
        for c in range(C):
            yt = y_res[c, t]; F0 = sigma2[c]
            ll[0] += -0.5 * (np.log(2*np.pi*F0) + yt**2/F0)
            if active[c] and not b_zero:
                pred = lam[c] * (X_src[c, t] @ b_fixed)
                if null_correction:
                    # F1 includes expected prediction energy under null
                    F1 = sigma2[c] + pred**2
                else:
                    F1 = sigma2[c]
                ll[1] += -0.5 * (np.log(2*np.pi*F1) + (yt-pred)**2/F1)
            else:
                ll[1] += -0.5 * (np.log(2*np.pi*F0) + yt**2/F0)
        lj = ll + np.log(xi_pred); lj -= lj.max()
        j = np.exp(lj); xi_filt[t] = j / max(j.sum(), 1e-20)

    xi_s = np.zeros((T, 2)); xi_s[T-1] = xi_filt[T-1]
    for t in range(T-2, -1, -1):
        xp = np.maximum(A_tr.T @ xi_filt[t], 1e-10)
        r = xi_s[t+1] / xp
        for j in range(2): xi_s[t,j] = xi_filt[t,j] * (A_tr[j,0]*r[0] + A_tr[j,1]*r[1])
        s = xi_s[t].sum()
        if s > 0: xi_s[t] /= s
    return np.clip(xi_s[:, 1], 0.0, 1.0)


# Estimate b once from coupled data
print("Estimating warm-start b from coupled data...", flush=True)
b_c, yres_c, sig2_c = get_warmstart_b(tgt_coupled.T)
print(f"  ||b|| = {np.linalg.norm(b_c):.4f}\n", flush=True)

# Sweep p_stay
# Sweep asymmetric transitions: p01 (enter coupling) vs p11 (stay coupled)
p01_values = [0.001, 0.005, 0.01, 0.02, 0.05]  # probability of entering regime 1
p11_values = [0.90, 0.95, 0.99]  # probability of staying in regime 1

from joblib import Parallel, delayed
import itertools

def compute_ll_diff(y_res, sigma2, b_fixed):
    """Compute per-timepoint log-likelihood difference (regime1 - regime0).

    Positive = coupling prediction helps. Negative = prediction hurts.
    Sum across active channels.
    """
    C, T = y_res.shape
    ll_diff = np.zeros(T)
    for c in range(C):
        if not active[c]: continue
        for t in range(T):
            yt = y_res[c, t]; F = sigma2[c]
            pred = lam[c] * (X_src[c, t] @ b_fixed)
            ll0 = -0.5 * (yt**2 / F)
            ll1 = -0.5 * ((yt - pred)**2 / F)
            ll_diff[t] += ll1 - ll0  # positive if prediction helps
    return ll_diff

def run_one_config(nc, p01, p11, yres, sig2, b, gm):
    """Run one HMM config, return metrics."""
    post = run_hmm(yres, sig2, b, p01=p01, p11=p11, null_correction=nc)
    frac = float((post > 0.5).mean())
    hit = float((gm & (post > 0.5)).sum() / max(gm.sum(), 1))
    fa = float((~gm & (post > 0.5)).sum() / max((~gm).sum(), 1))
    inter = (gm & (post > 0.5)).sum()
    union = (gm | (post > 0.5)).sum()
    iou = float(inter / max(union, 1))
    return nc, p01, p11, frac, hit, fa, iou

configs = list(itertools.product([False, True], p01_values, p11_values))
print(f"Sweeping {len(configs)} configs in parallel...", flush=True)

t0 = time.perf_counter()
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(run_one_config)(nc, p01, p11, yres_c, sig2_c, b_c, gate_mask)
    for nc, p01, p11 in configs)
print(f"Done in {time.perf_counter()-t0:.1f}s\n", flush=True)

print(f"{'corr':>4} {'p01':>6} {'p11':>5} | {'frac':>5} {'hit':>5} {'fa':>5} {'IoU':>5}", flush=True)
print("-" * 55, flush=True)

best_iou = 0; best_params = None
for nc, p01, p11, frac, hit, fa, iou in sorted(results, key=lambda r: -r[6]):
    nc_str = "Y" if nc else "N"
    print(f"  {nc_str}  {p01:>6.3f} {p11:>5.2f} | {frac:>5.1%} {hit:>5.1%} {fa:>5.1%} {iou:>5.1%}", flush=True)
    if best_params is None:
        best_params = (nc, p01, p11, hit, fa, iou)

print(f"\nBest IoU: null_corr={best_params[0]}, p01={best_params[1]}, p11={best_params[2]}, "
      f"hit={best_params[3]:.1%}, fa={best_params[4]:.1%}, IoU={best_params[5]:.1%}", flush=True)

# Also show best by hit with FA<15%
good = [(nc,p01,p11,f,h,fa,iou) for nc,p01,p11,f,h,fa,iou in results if fa < 0.15]
if good:
    good.sort(key=lambda r: -r[4])
    print(f"\nBest hit (FA<15%):", flush=True)
    for nc,p01,p11,f,h,fa,iou in good[:5]:
        nc_s = "Y" if nc else "N"
        print(f"  corr={nc_s} p01={p01} p11={p11}: hit={h:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)

# === Iteration 3: Direct LL difference thresholding (no HMM) ===
print(f"\n{'='*60}", flush=True)
print(f"Direct LL difference (no HMM) with smoothing sweep", flush=True)
print(f"{'='*60}", flush=True)

from scipy.ndimage import uniform_filter1d

ll_diff = compute_ll_diff(yres_c, sig2_c, b_c)
print(f"LL diff: mean={ll_diff.mean():.4f} std={ll_diff.std():.4f} "
      f"max={ll_diff.max():.4f}", flush=True)
print(f"LL diff during coupling: {ll_diff[gate_mask].mean():.4f}", flush=True)
print(f"LL diff during null:     {ll_diff[~gate_mask].mean():.4f}", flush=True)

# Sweep smoothing windows and thresholds
smooth_windows = [1, 15, 30, 60, 150, 450]  # samples at 30 Hz
thresholds_pct = [50, 60, 70, 80, 90, 95]

def run_smooth_thresh(sw, tp, lld, gm):
    if sw > 1:
        lld_s = uniform_filter1d(lld, sw, mode='nearest')
    else:
        lld_s = lld
    thresh = np.percentile(lld_s, tp)
    det = lld_s > thresh
    hit = float((gm & det).sum() / max(gm.sum(), 1))
    fa = float((~gm & det).sum() / max((~gm).sum(), 1))
    inter = (gm & det).sum(); union = (gm | det).sum()
    iou = float(inter / max(union, 1))
    return sw, tp, hit, fa, iou

configs2 = list(itertools.product(smooth_windows, thresholds_pct))
results2 = Parallel(n_jobs=-1)(
    delayed(run_smooth_thresh)(sw, tp, ll_diff, gate_mask)
    for sw, tp in configs2)

print(f"\n{'smooth':>6} {'pct':>4} | {'hit':>5} {'fa':>5} {'IoU':>5}", flush=True)
print("-" * 40, flush=True)
results2.sort(key=lambda r: -r[4])
for sw, tp, hit, fa, iou in results2[:20]:
    print(f"{sw:>6} {tp:>4} | {hit:>5.1%} {fa:>5.1%} {iou:>5.1%}", flush=True)

good2 = [r for r in results2 if r[3] < 0.15]
if good2:
    good2.sort(key=lambda r: -r[2])
    print(f"\nBest hit (FA<15%):", flush=True)
    for sw, tp, hit, fa, iou in good2[:5]:
        print(f"  smooth={sw} pct={tp}: hit={hit:.1%} fa={fa:.1%} IoU={iou:.1%}", flush=True)

# === Iteration 4: Iterative b refinement ===
print(f"\n{'='*60}", flush=True)
print(f"Iterative b refinement (re-estimate b from HMM detections)", flush=True)
print(f"{'='*60}", flush=True)

b_iter = b_c.copy()
p01_use, p11_use = 0.01, 0.99  # best IoU params

for iteration in range(5):
    post = run_hmm(yres_c, sig2_c, b_iter, p01=p01_use, p11=p11_use)
    detected = post > 0.5
    det_frac = detected.mean()

    hit = float((gate_mask & detected).sum() / max(gate_mask.sum(), 1))
    fa = float((~gate_mask & detected).sum() / max((~gate_mask).sum(), 1))
    inter = (gate_mask & detected).sum(); union = (gate_mask | detected).sum()
    iou = float(inter / max(union, 1))

    print(f"  Iter {iteration}: frac={det_frac:.1%} hit={hit:.1%} fa={fa:.1%} "
          f"IoU={iou:.1%} ||b||={np.linalg.norm(b_iter):.4f}", flush=True)

    # Re-estimate b from detected windows
    if detected.sum() > nb + 5:
        Y_s = np.concatenate([yres_c[c, detected] for c in range(C) if active[c]])
        X_s = np.concatenate([lam[c] * X_src[c, detected] for c in range(C) if active[c]])
        b_new = np.linalg.solve(X_s.T @ X_s + 1e-4*np.eye(nb), X_s.T @ Y_s)
        b_iter = b_new
    else:
        print(f"  -> Too few detected windows, stopping", flush=True)
        break
