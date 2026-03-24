"""Real-data PLV temporal localization on y_32_03132026.

Uses pre-computed wavelet features (10 Hz, 160ch) to compute PLV
per (frequency, ROI) in sliding windows with surrogate calibration.
Reports per-band coupling timecourse and condition breakdown.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.constants import WAVELET_CENTER_FREQS, EEG_ROI_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load y_32 session ──────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
session = None
for name, path in entries:
    if 'y_32' in name:
        session = load_session_from_cache(path, config=cfg)
        print(f"Loaded: {name}")
        break
assert session is not None, "y_32 not found"

print(f"  Duration: {session['duration']:.0f}s ({session['duration']/60:.1f} min)")
print(f"  P1: {session['p1_role']}, P2: {session['p2_role']}")

# ── Extract wavelet features (10 Hz, 160 ch) ──────────────────────────
w1 = session['p1_eeg_wavelet']     # (T, 160)
w2 = session['p2_eeg_wavelet']
ts1 = session['p1_eeg_wavelet_ts']
ts2 = session['p2_eeg_wavelet_ts']

# Align to common time grid
t_start = max(ts1[0], ts2[0])
t_end = min(ts1[-1], ts2[-1])
fs = 10.0  # wavelet feature rate
t_common = np.arange(t_start, t_end, 1.0 / fs)
T = len(t_common)

# Interpolate both to common grid
w1_aligned = np.stack([np.interp(t_common, ts1, w1[:, c]) for c in range(160)],
                       axis=1).astype(np.float32)
w2_aligned = np.stack([np.interp(t_common, ts2, w2[:, c]) for c in range(160)],
                       axis=1).astype(np.float32)
print(f"  Aligned wavelet features: ({T}, 160) at {fs} Hz, "
      f"{T/fs:.0f}s duration")

# ── Reconstruct complex CWT from real+imag features ───────────────────
# Structure: first 80 = real (20 freq × 4 ROI), last 80 = imag
n_freq = 20
n_roi = 4
W_p1 = (w1_aligned[:, :80] + 1j * w1_aligned[:, 80:]).reshape(T, n_freq, n_roi)
W_p2 = (w2_aligned[:, :80] + 1j * w2_aligned[:, 80:]).reshape(T, n_freq, n_roi)
# → (T, 20 freq, 4 ROI)

# ── Compute windowed PLV with surrogate calibration ────────────────────
print(f"\n=== Computing PLV ===", flush=True)
WIN_S = 20.0
STRIDE_S = 2.0
N_SURR = 100
SEED = 42

win_samp = int(WIN_S * fs)
stride_samp = int(STRIDE_S * fs)

# Move to GPU: (n_freq*n_roi, T) = (80, T)
w1_t = torch.as_tensor(W_p1.reshape(T, -1).T, device=device)  # (80, T) complex
w2_t = torch.as_tensor(W_p2.reshape(T, -1).T, device=device)

def compute_plv(s1, s2):
    """Windowed PLV for (80, T) complex tensors → (80, n_win)."""
    sxy = s1 * s2.conj()
    sxy_n = sxy / sxy.abs().clamp(min=1e-10)
    # Process per channel to save VRAM
    C = s1.shape[0]
    n_win = max(1, (s1.shape[1] - win_samp) // stride_samp + 1)
    plv = torch.zeros(C, n_win, device=device)
    for c in range(C):
        sr = sxy_n[c].real.unfold(0, win_samp, stride_samp)
        si = sxy_n[c].imag.unfold(0, win_samp, stride_samp)
        plv[c] = torch.sqrt(sr.mean(-1)**2 + si.mean(-1)**2)
    return plv

# Real PLV
t0 = time.perf_counter()
plv_real = compute_plv(w1_t, w2_t)  # (80, n_win)
n_win = plv_real.shape[1]

# Surrogate PLV (circular shift P1)
min_shift = max(1, int(0.1 * T))
max_shift = T - min_shift
gen = torch.Generator(device='cpu')
gen.manual_seed(SEED)
shifts = torch.randint(min_shift, max_shift + 1, (N_SURR,), generator=gen)

plv_surr = torch.zeros(N_SURR, 80, n_win, device=device)
for k in range(N_SURR):
    w1_shifted = torch.roll(w1_t, int(shifts[k].item()), dims=1)
    plv_surr[k] = compute_plv(w1_shifted, w2_t)

elapsed = time.perf_counter() - t0
print(f"  Computed in {elapsed:.1f}s ({n_win} windows)", flush=True)

# ── Z-score (pooled aggregation) ──────────────────────────────────────
plv_real_np = plv_real.cpu().numpy()    # (80, n_win)
plv_surr_np = plv_surr.cpu().numpy()    # (K, 80, n_win)

# Pooled: average across all 80 (freq, ROI) pairs, z-score once
agg_real = plv_real_np.mean(axis=0)           # (n_win,)
agg_surr = plv_surr_np.mean(axis=1)           # (K, n_win)
surr_mean = agg_surr.mean(axis=0)
surr_std = np.maximum(agg_surr.std(axis=0), 1e-10)
z_agg = (agg_real - surr_mean) / surr_std

# Per-band z-scores
band_defs = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}
freqs = WAVELET_CENTER_FREQS
z_bands = {}
for band_name, (flo, fhi) in band_defs.items():
    f_mask = (freqs >= flo) & (freqs <= fhi)
    f_idx = np.where(f_mask)[0]
    # Indices in the 80-channel array: freq_i * 4 + roi (for all 4 ROIs)
    ch_idx = []
    for fi in f_idx:
        for ri in range(n_roi):
            ch_idx.append(fi * n_roi + ri)
    ch_idx = np.array(ch_idx)

    band_real = plv_real_np[ch_idx].mean(axis=0)
    band_surr = plv_surr_np[:, ch_idx].mean(axis=1)
    bm = band_surr.mean(0)
    bs = np.maximum(band_surr.std(0), 1e-10)
    z_bands[band_name] = (band_real - bm) / bs

# Window center times
win_times = t_common[win_samp // 2::stride_samp][:n_win]

# ── Per-ROI z-scores ──────────────────────────────────────────────────
z_rois = {}
for ri, roi_name in enumerate(EEG_ROI_NAMES):
    ch_idx = np.array([fi * n_roi + ri for fi in range(n_freq)])
    roi_real = plv_real_np[ch_idx].mean(axis=0)
    roi_surr = plv_surr_np[:, ch_idx].mean(axis=1)
    rm = roi_surr.mean(0); rs = np.maximum(roi_surr.std(0), 1e-10)
    z_rois[roi_name] = (roi_real - rm) / rs

# ── Condition markers ──────────────────────────────────────────────────
# Load condition summary if available
cond_file = 'results/cluster_all_sessions/y_32_03132026/condition_summary.json'
conditions = []
if os.path.exists(cond_file):
    with open(cond_file) as f:
        cond_data = json.load(f)
    if 'conditions' in cond_data:
        conditions = cond_data['conditions']
        print(f"\n  Conditions: {[c['name'] for c in conditions]}")

# ── Report ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"PLV TEMPORAL ANALYSIS: y_32_03132026")
print(f"{'='*70}")
print(f"  Duration: {T/fs:.0f}s, Windows: {n_win}, "
      f"Window: {WIN_S}s, Stride: {STRIDE_S}s")

print(f"\n  Broadband PLV z-score:")
print(f"    Mean: {z_agg.mean():.3f}, Max: {z_agg.max():.3f}, "
      f"P95: {np.percentile(z_agg, 95):.3f}")
print(f"    Fraction z > 2.0: {(z_agg > 2.0).mean():.1%}")
print(f"    Fraction z > 3.0: {(z_agg > 3.0).mean():.1%}")

print(f"\n  Per-band PLV z-score (mean / max / frac>2):")
for band_name, z_band in z_bands.items():
    print(f"    {band_name:>6}: mean={z_band.mean():.3f} max={z_band.max():.3f} "
          f"frac>2={float((z_band > 2.0).mean()):.1%}")

print(f"\n  Per-ROI PLV z-score (mean / max / frac>2):")
for roi_name, z_roi in z_rois.items():
    print(f"    {roi_name:>12}: mean={z_roi.mean():.3f} max={z_roi.max():.3f} "
          f"frac>2={float((z_roi > 2.0).mean()):.1%}")

# ── Per-condition breakdown ────────────────────────────────────────────
if conditions:
    print(f"\n  Per-condition PLV z-score:")
    print(f"  {'Condition':<15} {'Broadband':>10} {'Theta':>8} {'Alpha':>8} {'Beta':>8}")
    for cond in conditions:
        c_start = cond.get('start_s', 0)
        c_end = cond.get('end_s', T / fs)
        c_mask = (win_times >= c_start) & (win_times < c_end)
        if c_mask.sum() == 0:
            continue
        z_bb = z_agg[c_mask].mean()
        z_th = z_bands['theta'][c_mask].mean()
        z_al = z_bands['alpha'][c_mask].mean()
        z_be = z_bands['beta'][c_mask].mean()
        print(f"  {cond['name']:<15} {z_bb:>10.3f} {z_th:>8.3f} {z_al:>8.3f} {z_be:>8.3f}")

# ── Save plot ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

# Broadband
ax = axes[0]
ax.plot(win_times / 60, z_agg, 'k-', alpha=0.7, lw=0.8)
ax.axhline(2.0, color='r', ls='--', alpha=0.3, label='z=2')
ax.fill_between(win_times / 60, z_agg, where=z_agg > 2.0,
                 color='red', alpha=0.3)
ax.set_ylabel('z (broadband)')
ax.set_title(f'y_32: Inter-brain PLV temporal localization '
             f'(20s windows, K={N_SURR} surrogates)')
ax.legend(loc='upper right')

# Per-band
colors = {'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#e74c3c'}
for i, (band_name, z_band) in enumerate(z_bands.items()):
    ax = axes[i + 1]
    c = colors[band_name]
    ax.plot(win_times / 60, z_band, color=c, alpha=0.7, lw=0.8)
    ax.axhline(2.0, color='r', ls='--', alpha=0.3)
    ax.fill_between(win_times / 60, z_band, where=z_band > 2.0,
                     color=c, alpha=0.3)
    ax.set_ylabel(f'z ({band_name})')

# Condition shading
cond_colors = {'base_EO': '#ecf0f1', 'base_EC': '#d5dbdb',
               'conv_1': '#aed6f1', 'PE': '#f9e79f', 'conv_2': '#a9dfbf'}
for cond in conditions:
    c_s = cond.get('start_s', 0) / 60
    c_e = cond.get('end_s', T / fs) / 60
    cc = cond_colors.get(cond['name'], '#f0f0f0')
    for ax in axes:
        ax.axvspan(c_s, c_e, alpha=0.15, color=cc)
    axes[0].text((c_s + c_e) / 2, axes[0].get_ylim()[1] * 0.9,
                 cond['name'], ha='center', fontsize=8, alpha=0.6)

axes[-1].set_xlabel('Time (minutes)')
plt.tight_layout()
outpath = 'results/y32_plv_temporal.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {outpath}")
