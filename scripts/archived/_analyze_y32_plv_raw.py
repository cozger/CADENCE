"""Real PLV analysis on y_32 from RAW 256 Hz EEG (not decimated features).

Computes Morlet CWT at 256 Hz → windowed PLV with full phase resolution.
Per-frequency chunked to stay within 16 GB VRAM.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
from cadence.constants import WAVELET_CENTER_FREQS, EEG_ROI_NAMES, EEG_ROIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load y_32 raw EEG ─────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
session = None
for name, path in entries:
    if 'y_32' in name:
        session = load_session_from_cache(path, config=cfg)
        print(f"Loaded: {name}")
        break

FS = 256.0
N_CH = 14
p1_eeg = session['p1_eeg']
p2_eeg = session['p2_eeg']
p1_ts = session['p1_eeg_ts']
p2_ts = session['p2_eeg_ts']

# Align to common time window
t_start = max(p1_ts[0], p2_ts[0])
t_end = min(p1_ts[-1], p2_ts[-1])
duration = t_end - t_start

p1_m = (p1_ts >= t_start) & (p1_ts < t_end)
p2_m = (p2_ts >= t_start) & (p2_ts < t_end)
p1_raw = p1_eeg[p1_m].copy()
p2_raw = p2_eeg[p2_m].copy()

# Resample P2 onto P1's time grid
N = len(p1_raw)
t_p1 = p1_ts[p1_m]
t_p2 = p2_ts[p2_m]
if len(p2_raw) != N:
    p2_raw = np.stack([np.interp(t_p1, t_p2, p2_raw[:, c])
                        for c in range(N_CH)], axis=1).astype(np.float32)

t_eeg = t_p1 - t_start
print(f"  Raw EEG: ({N}, {N_CH}) at {FS} Hz, {duration:.0f}s")

# Average re-reference
p1_raw -= p1_raw.mean(axis=1, keepdims=True)
p2_raw -= p2_raw.mean(axis=1, keepdims=True)

# Z-score per channel
for ch in range(N_CH):
    for sig in [p1_raw, p2_raw]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# ── Run PLV per band ──────────────────────────────────────────────────
BANDS = {
    'theta':     (4, 8, 5),      # (lo, hi, n_freqs)
    'alpha':     (8, 13, 5),
    'beta':      (13, 30, 8),
    'broadband': (2, 40, 20),
}

shared = dict(
    channels=list(range(N_CH)),
    n_surrogates=100, n_cycles=[3, 7],
    window_s=20.0, stride_s=2.0,
    target_fa=0.05, min_event_s=5.0,
    metric='plv', seed=42, device=device,
    smooth_s=10,
)

results = {}
for band_name, (flo, fhi, nf) in BANDS.items():
    freqs = np.logspace(np.log10(flo), np.log10(fhi), nf)
    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1_raw, p2_raw, FS, center_freqs=freqs, **shared)
    elapsed = time.perf_counter() - t0

    wt = diag['win_times']
    results[band_name] = {
        'z': z_agg, 'mask': mask, 'win_times': wt,
        'z_mean': z_agg.mean(), 'z_max': z_agg.max(),
        'frac_sig': (z_agg > 2.0).mean(),
        'threshold': diag['z_threshold'],
    }
    print(f"  {band_name:<12}: z_mean={z_agg.mean():.3f} z_max={z_agg.max():.3f} "
          f"frac>2={float((z_agg > 2.0).mean()):.1%} thr={diag['z_threshold']:.2f} "
          f"({elapsed:.0f}s)", flush=True)

# ── Per-ROI analysis (broadband) ──────────────────────────────────────
print(f"\n  Per-ROI PLV (broadband, 256 Hz):")
freqs_bb = np.logspace(np.log10(2), np.log10(40), 20)
for roi_name, roi_ch in EEG_ROIS.items():
    roi_shared = dict(shared)
    roi_shared['channels'] = roi_ch
    _, z_roi, _, _ = wpli_temporal_localization(
        p1_raw, p2_raw, FS, center_freqs=freqs_bb, **roi_shared)
    print(f"    {roi_name:>12}: z_mean={z_roi.mean():.3f} z_max={z_roi.max():.3f} "
          f"frac>2={float((z_roi > 2.0).mean()):.1%}", flush=True)

# ── Report ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"y_32 PLV TEMPORAL ANALYSIS (256 Hz raw EEG, avg-ref)")
print(f"{'='*70}")
print(f"  Session: {duration:.0f}s, P1={session['p1_role']}, P2={session['p2_role']}")
print(f"  Windows: {len(results['broadband']['z'])}, 20s window, 2s stride, "
      f"K=100 surrogates, 10s smoothing")
print(f"\n  Band         z_mean   z_max   frac>2   coupling%")
for bn in ['theta', 'alpha', 'beta', 'broadband']:
    r = results[bn]
    print(f"  {bn:<12} {r['z_mean']:>6.3f}  {r['z_max']:>6.3f}  "
          f"{r['frac_sig']:>6.1%}   {float(r['mask'].mean()):>6.1%}")

# ── Plot ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
colors = {'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#e74c3c',
          'broadband': '#2c3e50'}

for i, (bn, c) in enumerate(colors.items()):
    ax = axes[i]
    r = results[bn]
    t_min = r['win_times'] / 60
    ax.plot(t_min, r['z'], color=c, alpha=0.7, lw=0.8)
    ax.axhline(2.0, color='r', ls='--', alpha=0.3)
    ax.fill_between(t_min, r['z'], where=r['z'] > 2.0, color=c, alpha=0.3)
    ax.set_ylabel(f'z ({bn})')
    ax.set_ylim(-4, max(r['z_max'] + 1, 5))

axes[0].set_title(f'y_32: Inter-brain PLV from 256 Hz raw EEG '
                   f'(avg-ref, 20s windows, K=100 surrogates, 10s smooth)')
axes[-1].set_xlabel('Time (minutes)')
plt.tight_layout()
outpath = 'results/y32_plv_raw256.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {outpath}")
