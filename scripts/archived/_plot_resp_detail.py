"""Deep dive: per-condition respiratory waveforms and PSDs for y_06."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._extract_respiratory import (
    extract_respiratory_one, parse_conditions, discover_cached_sessions,
    _load_session_cache, CACHE_DIR, EDR_SRATE, RESP_BAND, windowed_crosscorr,
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
print(f"P1: {len(r1['rpeak_indices'])} R-peaks, P2: {len(r2['rpeak_indices'])} R-peaks")

# All conditions of interest
cond_order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
cond_map = {n: (s, e) for n, s, e in conditions}

# =========================================================================
# Figure 1: 60s waveform excerpts for ALL conditions (fused only, both P)
# =========================================================================
n_conds = len([c for c in cond_order if c in cond_map])
fig, axes = plt.subplots(n_conds, 1, figsize=(16, 3.5 * n_conds))
fig.suptitle('Fused respiratory waveform: 60s excerpt per condition (P1=blue, P2=red)', fontsize=13)

for i, cname in enumerate(cond_order):
    if cname not in cond_map:
        continue
    cs, ce = cond_map[cname]
    ax = axes[i]
    mid = (cs + ce) / 2
    t0, t1_end = mid - 30, mid + 30

    for p_label, res, color in [('P1', r1, '#2196F3'), ('P2', r2, '#F44336')]:
        t = res['t']
        sig = res['fused']
        mask = (t >= t0) & (t < t1_end)
        normed = sig / max(np.std(sig[mask]) if mask.sum() > 10 else 1.0, 1e-8)
        ax.plot(t[mask] - t0, normed[mask], color=color, linewidth=1.0, alpha=0.8, label=p_label)

    ax.set_title(f'{cname} ({ce-cs:.0f}s)')
    ax.set_ylabel('Amp (norm)')
    ax.legend(loc='upper right', fontsize=7)
    if i == n_conds - 1:
        ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'waveforms_all_conditions.png'), dpi=150)
plt.close()
print('Saved waveforms_all_conditions.png')

# =========================================================================
# Figure 2: PSD for all conditions, both participants
# =========================================================================
fig, axes = plt.subplots(n_conds, 2, figsize=(14, 3 * n_conds))
fig.suptitle('Respiratory PSD per condition (P1 left, P2 right)', fontsize=13)

for i, cname in enumerate(cond_order):
    if cname not in cond_map:
        continue
    cs, ce = cond_map[cname]

    for col, (p_label, res) in enumerate([('P1', r1), ('P2', r2)]):
        ax = axes[i, col]
        t = res['t']

        for method_name, key, mc in [('FMRR', 'fmrr', '#4CAF50'), ('AM', 'am', '#FF9800'),
                                      ('QRS slope', 'qrs', '#9C27B0'), ('Fused', 'fused', 'k')]:
            sig = res[key]
            mask = (t >= cs) & (t < ce)
            seg = sig[mask]
            if len(seg) < 32:
                continue
            f, psd = welch(seg, fs=EDR_SRATE, nperseg=min(len(seg), int(32 * EDR_SRATE)))
            fmask = (f >= 0.05) & (f <= 0.6)
            lw = 2.0 if key == 'fused' else 1.0
            alpha = 0.9 if key == 'fused' else 0.5
            ax.plot(f[fmask], psd[fmask] / max(psd[fmask].max(), 1e-12),
                    color=mc, linewidth=lw, label=method_name, alpha=alpha)

        ax.axvline(RESP_BAND[0], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(RESP_BAND[1], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(f'{p_label} - {cname}')
        ax.set_xlabel('Frequency (Hz)')
        if col == 0:
            ax.set_ylabel('Norm PSD')
        if i == 0 and col == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'psd_all_conditions.png'), dpi=150)
plt.close()
print('Saved psd_all_conditions.png')

# =========================================================================
# Figure 3: Per-condition synchrony bar chart
# =========================================================================
sync_results = {}
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
    if min_len < int(30 * EDR_SRATE):
        continue
    seg1, seg2 = seg1[:min_len], seg2[:min_len]
    ct, cr, cl = windowed_crosscorr(seg1, seg2, EDR_SRATE)
    sync_results[cname] = {
        'mean_abs_r': np.mean(np.abs(cr)),
        'mean_r': np.mean(cr),
        'std_r': np.std(cr),
    }

fig, ax = plt.subplots(figsize=(10, 5))
names = [c for c in cond_order if c in sync_results]
vals = [sync_results[c]['mean_abs_r'] for c in names]
stds = [sync_results[c]['std_r'] for c in names]
colors = ['#BDBDBD', '#9E9E9E', '#BBDEFB', '#C8E6C9', '#A5D6A7', '#90CAF9']
bars = ax.bar(names, vals, yerr=stds, color=colors[:len(names)], edgecolor='black',
              linewidth=0.5, capsize=5)
ax.set_ylabel('Mean |r| (respiratory synchrony)')
ax.set_title('Per-condition respiratory synchrony (y_06)')
ax.set_ylim(0, 0.8)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sync_bar_chart.png'), dpi=150)
plt.close()
print('Saved sync_bar_chart.png')
print('\nDone.')
