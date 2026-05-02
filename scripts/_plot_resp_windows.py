"""Test different cross-correlation window sizes on meditate_B vs baselines."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._extract_respiratory import (
    extract_respiratory_one, parse_conditions, discover_cached_sessions,
    _load_session_cache, CACHE_DIR, EDR_SRATE, windowed_crosscorr,
)

OUT = 'results/respiratory/y_06'
os.makedirs(OUT, exist_ok=True)

# Load
sessions = discover_cached_sessions(CACHE_DIR)
cache_path = [p for n, p in sessions if n == 'y_06'][0]
session = _load_session_cache(cache_path)
conditions = parse_conditions(cache_path)

r1 = extract_respiratory_one(session['p1_ecg'].ravel().astype(np.float64),
                              session['p1_ecg_ts'].ravel().astype(np.float64))
r2 = extract_respiratory_one(session['p2_ecg'].ravel().astype(np.float64),
                              session['p2_ecg_ts'].ravel().astype(np.float64))

cond_map = {n: (s, e) for n, s, e in conditions}
cond_order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']

# Test window sizes: 5s, 10s, 15s, 20s, 30s
windows = [5, 10, 15, 20, 30]

# =========================================================================
# Figure 1: Bar chart for each window size
# =========================================================================
fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 5), sharey=True)
fig.suptitle('Respiratory synchrony |r| by cross-correlation window size', fontsize=13)

for wi, win_s in enumerate(windows):
    ax = axes[wi]
    names, vals = [], []
    for cname in cond_order:
        if cname not in cond_map:
            continue
        cs, ce = cond_map[cname]
        t1, t2 = r1['t'], r2['t']
        mask1 = (t1 >= cs) & (t1 < ce)
        mask2 = (t2 >= cs) & (t2 < ce)
        seg1 = r1['fused'][mask1]
        seg2 = r2['fused'][mask2]
        min_len = min(len(seg1), len(seg2))
        if min_len < int(win_s * EDR_SRATE) + 10:
            continue
        seg1, seg2 = seg1[:min_len], seg2[:min_len]
        ct, cr, cl = windowed_crosscorr(seg1, seg2, EDR_SRATE,
                                         window_s=win_s, step_s=1.0, max_lag_s=5.0)
        if len(cr) == 0:
            continue
        names.append(cname)
        vals.append(np.mean(np.abs(cr)))

    colors = ['#BDBDBD', '#9E9E9E', '#BBDEFB', '#C8E6C9', '#A5D6A7', '#90CAF9']
    bars = ax.bar(range(len(names)), vals, color=colors[:len(names)],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_title(f'Window = {win_s}s')
    ax.set_ylim(0, 0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    if wi == 0:
        ax.set_ylabel('Mean |r|')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sync_window_sweep.png'), dpi=150)
plt.close()
print('Saved sync_window_sweep.png')

# =========================================================================
# Figure 2: Synchrony timecourse for meditate_B at different windows
# =========================================================================
fig, axes = plt.subplots(len(windows), 1, figsize=(16, 3 * len(windows)), sharex=True)
fig.suptitle('Respiratory synchrony timecourse during meditate_B (different windows)', fontsize=13)

cs, ce = cond_map['meditate_B']
t1, t2 = r1['t'], r2['t']
mask1 = (t1 >= cs) & (t1 < ce)
mask2 = (t2 >= cs) & (t2 < ce)
seg1 = r1['fused'][mask1]
seg2 = r2['fused'][mask2]
min_len = min(len(seg1), len(seg2))
seg1, seg2 = seg1[:min_len], seg2[:min_len]

for wi, win_s in enumerate(windows):
    ax = axes[wi]
    ct, cr, cl = windowed_crosscorr(seg1, seg2, EDR_SRATE,
                                     window_s=win_s, step_s=1.0, max_lag_s=5.0)
    ax.plot(ct, cr, 'k-', linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.fill_between(ct, cr, 0, where=cr > 0, alpha=0.3, color='green')
    ax.fill_between(ct, cr, 0, where=cr < 0, alpha=0.3, color='red')
    ax.set_ylim(-1, 1)
    ax.set_title(f'Window = {win_s}s  (mean |r| = {np.mean(np.abs(cr)):.3f})')
    ax.set_ylabel('r')

axes[-1].set_xlabel('Time within meditate_B (s)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'meditate_B_window_sweep.png'), dpi=150)
plt.close()
print('Saved meditate_B_window_sweep.png')

# =========================================================================
# Figure 3: Same timecourse sweep but for base_EC (should be low)
# =========================================================================
fig, axes = plt.subplots(len(windows), 1, figsize=(16, 3 * len(windows)), sharex=True)
fig.suptitle('Respiratory synchrony timecourse during base_EC (different windows)', fontsize=13)

cs, ce = cond_map['base_EC']
mask1 = (t1 >= cs) & (t1 < ce)
mask2 = (t2 >= cs) & (t2 < ce)
seg1_b = r1['fused'][mask1]
seg2_b = r2['fused'][mask2]
min_len = min(len(seg1_b), len(seg2_b))
seg1_b, seg2_b = seg1_b[:min_len], seg2_b[:min_len]

for wi, win_s in enumerate(windows):
    ax = axes[wi]
    if min_len < int(win_s * EDR_SRATE) + 10:
        ax.set_title(f'Window = {win_s}s  (too short)')
        continue
    ct, cr, cl = windowed_crosscorr(seg1_b, seg2_b, EDR_SRATE,
                                     window_s=win_s, step_s=1.0, max_lag_s=5.0)
    ax.plot(ct, cr, 'k-', linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.fill_between(ct, cr, 0, where=cr > 0, alpha=0.3, color='green')
    ax.fill_between(ct, cr, 0, where=cr < 0, alpha=0.3, color='red')
    ax.set_ylim(-1, 1)
    ax.set_title(f'Window = {win_s}s  (mean |r| = {np.mean(np.abs(cr)):.3f})')
    ax.set_ylabel('r')

axes[-1].set_xlabel('Time within base_EC (s)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'base_EC_window_sweep.png'), dpi=150)
plt.close()
print('Saved base_EC_window_sweep.png')

print('\nDone.')
