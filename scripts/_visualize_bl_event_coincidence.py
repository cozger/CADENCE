"""Verify bl_event_coincidence has meaningful information on y_06 before
committing to a full re-fit.

Renders 4 panels:
  1. Per-AU per-participant event raster (top: P1, bottom: P2) with
     experimental-condition shading
  2. New bl_event_coincidence z-score timecourse + condition shading
  3. OLD bl_expr (from MVP scaffold) overlaid
  4. OLD bl_activity_conc overlaid

Plus a per-condition bar chart comparing the three channels' means.

Usage:
    python scripts/_visualize_bl_event_coincidence.py
    python scripts/_visualize_bl_event_coincidence.py --session y_06
"""
from __future__ import annotations

import torch  # noqa: F401  -- precede numpy on Windows
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cadence.significance.face_event_coincidence import (
    compute_bl_event_coincidence, detect_activity_peaks,
)


CONDITION_COLORS = {
    'base_EO':    '#1565C0',
    'base_EC':    '#283593',
    'conv_1':     '#E65100',
    'conv_2':     '#BF360C',
    'meditate_B': '#7B1FA2',
    'meditate_K': '#388E3C',
    'PE_1':       '#C62828',
    'PE_2':       '#B71C1C',
    'PE':         '#C62828',
}

AU_NAMES_AFFECT = {
    7: 'cheekSquintL', 8: 'cheekSquintR',
    28: 'mouthDimpleL', 29: 'mouthDimpleR',
    30: 'mouthFrownL', 31: 'mouthFrownR',
    44: 'mouthSmileL', 45: 'mouthSmileR',
    50: 'noseSneerL', 51: 'noseSneerR',
}


def parse_periods(markers):
    starts, out = {}, []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    return sorted(out, key=lambda x: x[1])


def shade_periods(ax, periods, alpha=0.12, ymin=0, ymax=1):
    for name, t0, t1 in periods:
        c = CONDITION_COLORS.get(name, '#9E9E9E')
        ax.axvspan(t0, t1, color=c, alpha=alpha, ymin=ymin, ymax=ymax,
                   linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='y_06')
    args = ap.parse_args()
    sid = args.session

    sess_dir = REPO / 'results' / 'mvp' / sid
    face_path = REPO / 'data' / 'preproc' / 'face' / 'v1' / f'{sid}.npz'
    digest_path = REPO / 'data' / 'digest' / 'v1' / f'{sid}.json'

    if not face_path.exists():
        print(f'NO face preproc for {sid}'); return
    if not (sess_dir / 'mvp_scaffold.npz').exists():
        print(f'NO MVP scaffold for {sid}'); return

    print(f'=== {sid} ===')
    face = dict(np.load(face_path))
    scaff = np.load(sess_dir / 'mvp_scaffold.npz')
    obs = scaff['obs']
    valid = scaff['obs_valid']
    t_common = scaff['t_common']
    digest = json.load(open(digest_path))
    lsl_offset = digest['t_start_lsl']
    periods = parse_periods(digest['markers'])

    # 7-channel scaffold ordering
    mod_keys = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                'pose', 'resp', 'ecg_hf']
    bl_expr = obs[:, mod_keys.index('bl_expr')]
    bl_act  = obs[:, mod_keys.index('bl_activity_conc')]

    # Compute the new channel
    print('Computing bl_event_coincidence...')
    z_new, info = compute_bl_event_coincidence(face, t_common, lsl_offset)
    print(f'  peaks: P1={info["p1_total_peaks"]} P2={info["p2_total_peaks"]}')
    print(f'  z stats: mean={info["mean_z"]:+.3f} std={info["std_z"]:.3f}')
    print(f'  raw coinc: mean={info["mean_raw_coinc"]:.4f}')

    # Activity envelopes + peaks
    p1_act = np.asarray(face['p1_au_activity']).squeeze()
    p2_act = np.asarray(face['p2_au_activity']).squeeze()
    p1_ts_lsl = face['p1_au52_ts'] + lsl_offset
    p2_ts_lsl = face['p2_au52_ts'] + lsl_offset
    p1_pk_idx, _ = detect_activity_peaks(p1_act, fs_native=30.0,
                                          quantile_threshold=0.70, min_sep_s=1.0)
    p2_pk_idx, _ = detect_activity_peaks(p2_act, fs_native=30.0,
                                          quantile_threshold=0.70, min_sep_s=1.0)
    p1_pk_t = p1_ts_lsl[p1_pk_idx]
    p2_pk_t = p2_ts_lsl[p2_pk_idx]
    p1_thresh = float(np.quantile(p1_act, 0.70))
    p2_thresh = float(np.quantile(p2_act, 0.70))

    # ── PLOT ──
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.6, 1.6, 1.0, 1.0, 1.6],
                          hspace=0.55,
                          left=0.06, right=0.99, top=0.95, bottom=0.06)

    t_min, t_max = t_common[0], t_common[-1]

    # Panel 1: P1 activity envelope + peaks
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(p1_ts_lsl, p1_act, color='#C62828', linewidth=0.6, alpha=0.7)
    ax1.scatter(p1_pk_t, p1_act[p1_pk_idx], s=10, c='#C62828', marker='v', zorder=3)
    ax1.axhline(p1_thresh, color='#C62828', linestyle='--', linewidth=0.6, alpha=0.6,
                label=f'p70 threshold = {p1_thresh:.2f}')
    shade_periods(ax1, periods, alpha=0.15)
    ax1.set_xlim(t_min, t_max)
    ax1.set_ylim(0, np.percentile(p1_act, 99))
    ax1.set_ylabel('p1_au_activity', fontsize=10)
    ax1.set_title(f'{sid}  ·  P1 (therapist) activity envelope + detected peaks  '
                   f'(N={len(p1_pk_idx)})',
                   fontsize=11, fontweight='bold', loc='left', color='#C62828')
    ax1.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    ax1.tick_params(labelsize=8)
    for name, t0, t1 in periods:
        if (t0 + t1) / 2 > t_min and (t0 + t1) / 2 < t_max:
            ax1.text((t0 + t1) / 2, np.percentile(p1_act, 99) * 0.92, name,
                     ha='center', va='top', fontsize=8,
                     color=CONDITION_COLORS.get(name, '#555'),
                     fontweight='bold')

    # Panel 2: P2 activity envelope + peaks
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(p2_ts_lsl, p2_act, color='#1565C0', linewidth=0.6, alpha=0.7)
    ax2.scatter(p2_pk_t, p2_act[p2_pk_idx], s=10, c='#1565C0', marker='v', zorder=3)
    ax2.axhline(p2_thresh, color='#1565C0', linestyle='--', linewidth=0.6, alpha=0.6,
                label=f'p70 threshold = {p2_thresh:.2f}')
    shade_periods(ax2, periods, alpha=0.15)
    ax2.set_xlim(t_min, t_max)
    ax2.set_ylim(0, np.percentile(p2_act, 99))
    ax2.set_ylabel('p2_au_activity', fontsize=10)
    ax2.set_title(f'P2 (patient) activity envelope + detected peaks  (N={len(p2_pk_idx)})',
                   fontsize=11, fontweight='bold', loc='left', color='#1565C0')
    ax2.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    ax2.tick_params(labelsize=8)

    # Panel 3: new bl_event_coincidence
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_common, z_new, color='#388E3C', linewidth=0.8)
    ax3.axhline(0, color='black', linestyle=':', linewidth=0.6)
    ax3.axhline(2, color='red', linestyle='--', linewidth=0.6, alpha=0.5, label='z=2')
    shade_periods(ax3, periods, alpha=0.15)
    ax3.set_xlim(t_min, t_max)
    ax3.set_ylim(min(-1, z_new.min() - 0.5), max(3, z_new.max() * 1.05))
    ax3.set_ylabel('bl_event_coinc\n(NEW)', fontsize=9)
    ax3.set_title(f'NEW: bl_event_coincidence  ·  mean z={z_new.mean():+.3f}, '
                  f'%(z≥2)={(z_new>=2).mean()*100:.1f}%',
                  fontsize=10, fontweight='bold', loc='left', color='#388E3C')
    ax3.tick_params(labelsize=7)

    # Panel 4: OLD bl_expr
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t_common, bl_expr, color='#1976D2', linewidth=0.8, label='bl_expr')
    ax4.plot(t_common, bl_act, color='#E65100', linewidth=0.8, alpha=0.7, label='bl_activity_conc')
    ax4.axhline(0, color='black', linestyle=':', linewidth=0.6)
    shade_periods(ax4, periods, alpha=0.15)
    ax4.set_xlim(t_min, t_max)
    ax4.set_ylabel('OLD channels', fontsize=9)
    ax4.set_title(f'OLD: bl_expr (Morlet wavelet coherence z) + bl_activity_conc ((z_P1+z_P2)/2)',
                  fontsize=10, loc='left', color='#555')
    ax4.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    ax4.set_xlabel('LSL time (s)', fontsize=9)
    ax4.tick_params(labelsize=7)

    # Panel 5: per-condition mean comparison (bar chart)
    ax5 = fig.add_subplot(gs[4])
    cond_means = {}
    cond_present = []
    for name, t0, t1 in periods:
        m = (t_common >= t0) & (t_common <= t1)
        if m.sum() < 5: continue
        cond_present.append(name)
        cond_means[name] = (z_new[m].mean(), bl_expr[m].mean(), bl_act[m].mean(),
                             m.sum())
    x = np.arange(len(cond_present))
    w = 0.27
    ax5.bar(x - w, [cond_means[c][0] for c in cond_present], w, color='#388E3C',
             label='bl_event_coinc (NEW)', edgecolor='white')
    ax5.bar(x,     [cond_means[c][1] for c in cond_present], w, color='#1976D2',
             label='bl_expr (OLD wavelet coh)', edgecolor='white')
    ax5.bar(x + w, [cond_means[c][2] for c in cond_present], w, color='#E65100',
             label='bl_activity_conc (OLD shared act)', edgecolor='white')
    ax5.axhline(0, color='black', linewidth=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{c}\nN={cond_means[c][3]}' for c in cond_present],
                        fontsize=9, rotation=0)
    ax5.set_ylabel('per-condition mean (z)', fontsize=10)
    ax5.set_title('Per-condition channel comparison  (NEW should be elevated WHERE coupling actually happens)',
                  fontsize=10, fontweight='bold', loc='left')
    ax5.legend(loc='upper left', fontsize=9, frameon=True, framealpha=0.9)
    ax5.grid(axis='y', alpha=0.25, linestyle=':')

    out = sess_dir / f'_visualize_bl_event_coincidence_{sid}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
