"""MVP rSLDS example timeline plot.

Loads one session's scaffold (7D obs, 2D cov) + per-session rSLDS results
(path, gamma, state_labels), and the digest's marker list. Produces a
classic CADENCE timecourse: stacked modality traces with Viterbi state
shading, condition spans + labels, latent dynamics, and Viterbi/posterior
strips at the bottom.

Usage:
    python scripts/_plot_mvp_timeline.py --session y_06 --variant med
    python scripts/_plot_mvp_timeline.py --session Y_10_03182026 --variant pe
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

STATE_COLORS = ['#90A4AE', '#1565C0', '#2E7D32', '#FF6F00']  # generic; reordered by label below
STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',  # grey
    'OTHER':  '#9C27B0',  # purple
    'SHARED': '#2E7D32',  # green
    'COUP':   '#FF6F00',  # orange
}

CONDITION_COLORS = {
    'base_EO':    '#90CAF9',  # blue
    'base_EC':    '#9FA8DA',  # indigo
    'conv_1':     '#FFCC80',  # orange
    'conv_2':     '#FFB74D',  # darker orange
    'meditate_B': '#CE93D8',  # purple
    'meditate_K': '#A5D6A7',  # green
    'PE_1':       '#EF9A9A',  # red
    'PE_2':       '#E57373',  # darker red
}

CHANNEL_COLORS = {
    'conc_theta':       '#1976D2',
    'conc_alpha':       '#1565C0',
    'bl_expr':          '#D32F2F',
    'bl_activity_conc': '#C62828',
    'pose':             '#388E3C',
    'resp':             '#F57C00',
    'ecg_hf':           '#7B1FA2',
}


VALID_CONDITIONS = {'base_EO', 'base_EC', 'conv_1', 'conv_2',
                     'meditate_B', 'meditate_K',
                     'PE', 'PE_1', 'PE_2', 'baseline'}


def parse_periods(markers):
    """Convert flat (t, label) markers into (name, t0, t1) period tuples.

    Filters out auto-detected event markers (gesture_*, etc.) that share
    the _start/_stop convention but are not experimental conditions.
    Drops the redundant outer ``baseline`` block when its inner sub-blocks
    (``base_EO``, ``base_EC``) are present.

    LSL times; caller subtracts t_start_lsl to align to t_common.
    """
    starts = {}
    periods = []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            name = lbl[:-len('_start')]
            if name in VALID_CONDITIONS:
                starts[name] = t
        elif lbl.endswith('_stop'):
            name = lbl[:-len('_stop')]
            if name in starts:
                periods.append((name, starts.pop(name), t))
    periods = sorted(periods, key=lambda x: x[1])
    have_subblocks = {n for n, _, _ in periods} & {'base_EO', 'base_EC'}
    if have_subblocks:
        periods = [(n, t0, t1) for n, t0, t1 in periods if n != 'baseline']
    return periods


def state_runs(path, K):
    """For each state k, return list of (start_idx, end_idx) runs."""
    out = {k: [] for k in range(K)}
    if len(path) == 0:
        return out
    cur_start = 0
    cur_state = int(path[0])
    for t in range(1, len(path)):
        if path[t] != cur_state:
            out[cur_state].append((cur_start, t))
            cur_start = t
            cur_state = int(path[t])
    out[cur_state].append((cur_start, len(path)))
    return out


def smooth_with_nans(z, sigma_samples):
    """Gaussian smooth that propagates NaN gaps cleanly (no signal bleed)."""
    if sigma_samples <= 0:
        return z
    valid = np.isfinite(z).astype(float)
    z_filled = np.where(valid > 0, z, 0.0)
    num = gaussian_filter1d(z_filled, sigma_samples, mode='nearest')
    den = gaussian_filter1d(valid, sigma_samples, mode='nearest')
    out = num / np.maximum(den, 1e-9)
    # Re-impose NaN where the smoothing kernel was too sparse
    out[den < 0.1] = np.nan
    return out


def plot_timeline(session_id, variant, out_path, smooth_sigma_s=5.0):
    sess_dir = REPO / 'results' / 'mvp' / session_id
    scaffold = np.load(sess_dir / 'mvp_scaffold.npz')
    with open(sess_dir / 'mvp_scaffold.json') as f:
        meta = json.load(f)
    # 'prod' = production fit (no suffix); other variants get _<variant> suffix
    npz_name = 'mvp_rslds_results.npz' if variant == 'prod' else f'mvp_rslds_results_{variant}.npz'
    rslds = np.load(sess_dir / npz_name, allow_pickle=True)

    # Digest holds the markers (LSL clock; same clock as t_common in scaffold)
    with open(REPO / 'data/digest/v1' / f'{session_id}.json') as f:
        digest = json.load(f)
    periods_lsl = parse_periods(digest['markers'])

    obs = scaffold['obs']                        # (T, 7)
    obs_valid = scaffold['obs_valid']            # (T, 7)
    t_common_lsl = scaffold['t_common']          # (T,) — LSL clock seconds
    # Rebase both to seconds-since-scaffold-start so axis is readable
    t0_anchor = float(t_common_lsl[0])
    t_common = t_common_lsl - t0_anchor
    periods = [(name, t0 - t0_anchor, t1 - t0_anchor) for name, t0, t1 in periods_lsl]
    cov = scaffold['cov']                        # (T, 2)
    path = rslds['path']                         # (T,)
    gamma = rslds['gamma']                       # (T, K)
    x_smooth = rslds['x_smooth']                 # (T, D_latent)
    state_labels = [str(s) for s in rslds['state_labels']]
    K = len(state_labels)
    obs_keys = meta['observation_channels']
    cov_keys = meta['covariate_channels']

    state_colors = [STATE_COLOR_BY_LABEL.get(lbl, STATE_COLORS[k]) for k, lbl in enumerate(state_labels)]

    D_obs = obs.shape[1]
    D_lat = x_smooth.shape[1]
    D_cov = cov.shape[1]

    # Layout: D_obs modality traces + D_cov covariates + latent + posterior + viterbi
    panels = [
        *[('obs', i) for i in range(D_obs)],
        *[('cov', i) for i in range(D_cov)],
        ('latent', None),
        ('gamma', None),
        ('viterbi', None),
    ]
    n_panels = len(panels)
    height_ratios = []
    for kind, _ in panels:
        if kind == 'obs':       height_ratios.append(1.0)
        elif kind == 'cov':     height_ratios.append(0.7)
        elif kind == 'latent':  height_ratios.append(1.4)
        elif kind == 'gamma':   height_ratios.append(0.9)
        elif kind == 'viterbi': height_ratios.append(0.35)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(20, 0.55 * sum(height_ratios) + 4),
        gridspec_kw={'height_ratios': height_ratios},
        sharex=True,
    )

    # Condition shading on every axis (so eye can compare condition vs state path)
    for ax in axes:
        for name, t0, t1 in periods:
            ax.axvspan(t0, t1, alpha=0.40,
                       color=CONDITION_COLORS.get(name, '#FAFAFA'), zorder=0)
        # Solid vertical lines at condition boundaries
        for name, t0, t1 in periods:
            for tb in (t0, t1):
                ax.axvline(tb, color='#263238', linewidth=0.6, alpha=0.55, zorder=1)
        ax.set_xlim(t_common[0], t_common[-1])

    # Condition labels at top
    for name, t0, t1 in periods:
        axes[0].text((t0 + t1) / 2, 1.10, name.replace('_', ' '),
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    # State runs (used only for the Viterbi strip — upper panels show conditions only)
    runs = state_runs(path, K)

    # Smoothing kernel size in samples
    fs = float(meta.get('fs_out', 2.0))
    sigma_samples = max(0, int(round(smooth_sigma_s * fs)))

    # ── Observation channels (raw thin + smoothed bold) ──
    for i, key in enumerate(obs_keys):
        ax = axes[i]
        z = obs[:, i].copy()
        z[~obs_valid[:, i]] = np.nan
        color = CHANNEL_COLORS.get(key, '#37474F')
        # raw trace, very faint
        ax.plot(t_common, z, color=color, linewidth=0.4, alpha=0.25, zorder=3)
        # smoothed envelope, bold
        z_sm = smooth_with_nans(z, sigma_samples)
        ax.plot(t_common, z_sm, color=color, linewidth=1.4, alpha=0.95, zorder=4)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.4, zorder=2)
        ax.set_ylabel(key, fontsize=8, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=7)
        # symmetric ylim around 0 from data percentiles (use raw to avoid clipping spikes)
        finite = z[np.isfinite(z)]
        if finite.size:
            lim = max(np.abs(np.percentile(finite, [1, 99]))) * 1.1 + 0.1
            ax.set_ylim(-lim, lim)

    # ── Covariates ──
    for j, key in enumerate(cov_keys):
        ax = axes[D_obs + j]
        c = cov[:, j]
        ax.plot(t_common, c, color='#455A64', linewidth=0.4, alpha=0.25, zorder=3)
        c_sm = smooth_with_nans(c.astype(float), sigma_samples)
        ax.plot(t_common, c_sm, color='#263238', linewidth=1.3, alpha=0.95, zorder=4)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.4, zorder=2)
        ax.set_ylabel(f'cov:\n{key}', fontsize=7, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=7)

    # ── Latent dynamics (already smoothed by Kalman; no further smoothing) ──
    ax_lat = axes[D_obs + D_cov]
    lat_colors = ['#0288D1', '#388E3C', '#F57C00']
    for d in range(D_lat):
        ax_lat.plot(t_common, x_smooth[:, d], color=lat_colors[d % len(lat_colors)],
                    linewidth=0.9, alpha=0.9, label=f'x{d}', zorder=3)
    ax_lat.axhline(0, color='gray', linewidth=0.3, alpha=0.4)
    ax_lat.set_ylabel('latent\nx_smooth', fontsize=8, rotation=0, ha='right', va='center')
    ax_lat.legend(fontsize=6, loc='upper right', ncol=D_lat)
    ax_lat.tick_params(labelsize=7)

    # ── State posterior γ ──
    ax_g = axes[D_obs + D_cov + 1]
    bottom = np.zeros_like(t_common)
    for k in range(K):
        ax_g.fill_between(t_common, bottom, bottom + gamma[:, k],
                          color=state_colors[k], alpha=0.85, label=state_labels[k],
                          linewidth=0)
        bottom = bottom + gamma[:, k]
    ax_g.set_ylim(0, 1)
    ax_g.set_ylabel('γ posterior', fontsize=8, rotation=0, ha='right', va='center')
    ax_g.legend(fontsize=7, loc='upper right', ncol=K, framealpha=0.85)
    ax_g.tick_params(labelsize=7)

    # ── Viterbi strip ──
    ax_v = axes[-1]
    for k in range(K):
        for s, e in runs[k]:
            ax_v.axvspan(t_common[s], t_common[min(e, len(t_common) - 1)],
                         color=state_colors[k], alpha=0.95, zorder=2)
    ax_v.set_ylim(0, 1)
    ax_v.set_yticks([])
    ax_v.set_ylabel('Viterbi', fontsize=8, rotation=0, ha='right', va='center')
    ax_v.set_xlabel('Time (s, since session start)', fontsize=9)
    ax_v.tick_params(labelsize=7)

    # Title (kept short; second line has key params)
    n_trans = int(np.sum(np.diff(path) != 0))
    usage = np.array([(path == k).mean() for k in range(K)])
    usage_str = '  '.join(f'{lbl}={u*100:.0f}%' for lbl, u in zip(state_labels, usage))
    title_l1 = (f'MVP rSLDS timeline — {session_id} '
                f'(variant={variant}, protocol={digest["protocol"]})')
    title_l2 = (f'K={K}  D_obs={D_obs}  D_lat={D_lat}  D_cov={D_cov}  '
                f'fs={meta["fs_out"]:.1f}Hz  T={len(t_common)}  '
                f'trans={n_trans}  smooth=σ{smooth_sigma_s:.0f}s  |  {usage_str}')
    fig.suptitle(f'{title_l1}\n{title_l2}',
                 fontsize=10, fontfamily='monospace', y=0.995)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.045, hspace=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='y_06')
    ap.add_argument('--variant', default='prod', choices=['prod', 'med', 'pe', 'smoke', 'k3', 'k2', 'no_dwell', 'evtcoinc_k2', 'evtcoinc_k3'])
    ap.add_argument('--smooth-sigma-s', type=float, default=5.0,
                    help='Gaussian sigma in seconds for the obs/cov envelope overlay (default 5.0; 0 disables)')
    ap.add_argument('--out', default=None,
                    help='Output png path (default: results/mvp/<session>/timeline_<variant>.png)')
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO / 'results' / 'mvp' / args.session / f'timeline_{args.variant}.png'
    )
    plot_timeline(args.session, args.variant, out_path,
                  smooth_sigma_s=args.smooth_sigma_s)


if __name__ == '__main__':
    main()
