"""Per-session timecourse plots of bl_event_coincidence with experimental
condition shading.

Renders the new MVP behavioral-coupling channel for the chosen sessions,
with each experimental condition shaded by color and labeled. Useful for
visually verifying that the channel fires where coupling is expected
(conversation, PE_2 teaching) and stays low where not (meditation,
eyes-closed baseline).

Usage:
    python scripts/_plot_bl_event_coincidence_timecourse.py
    python scripts/_plot_bl_event_coincidence_timecourse.py --sessions y_06 Y_10_03182026
    python scripts/_plot_bl_event_coincidence_timecourse.py --grid 6
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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
    'baseline':   '#1565C0',
}
# Only these are real experimental conditions; everything else
# (gesture_*, calibration_*, etc.) is auto-detected event noise that
# must NOT be treated as a period.
VALID_CONDITIONS = set(CONDITION_COLORS.keys())

# Sessions to feature in the default 6-panel grid (mix of meditation + PE protocols).
# Excluded: Y_10, y_37, y_66 (marker overlap — conv_2 wraps PE blocks).
DEFAULT_GRID = ['y_06', 'y_53_04302026', 'y_64_04272026',   # meditation protocol
                 'y01_021726', 'y_65_04242026', 'y_33_04032026']  # PE protocol


def parse_periods(markers):
    """Pair <name>_start with <name>_stop into (name, t0, t1) tuples.

    Filters out auto-detected event markers (gesture_*, etc.) that share
    the _start/_stop convention but are not experimental conditions.
    Also drops the redundant outer ``baseline_*`` block when its inner
    sub-blocks (``base_EO_*``, ``base_EC_*``) are present, to avoid
    overlapping shaded regions.
    """
    starts, out = {}, []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            n = lbl[:-len('_start')]
            if n in VALID_CONDITIONS:
                starts[n] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    out = sorted(out, key=lambda x: x[1])
    # Drop the outer 'baseline' block if its sub-blocks are present —
    # otherwise it overlaps base_EO and base_EC.
    have_subblocks = {n for n, _, _ in out} & {'base_EO', 'base_EC'}
    if have_subblocks:
        out = [(n, t0, t1) for n, t0, t1 in out if n != 'baseline']
    return out


def shade_periods(ax, periods, alpha=0.18, label_at_top=True, ymax=None):
    for name, t0, t1 in periods:
        c = CONDITION_COLORS.get(name, '#9E9E9E')
        ax.axvspan(t0, t1, color=c, alpha=alpha, linewidth=0)
        if label_at_top and ymax is not None:
            ax.text((t0 + t1) / 2, ymax * 0.94, name,
                    ha='center', va='top', fontsize=8,
                    color=c, fontweight='bold')


def load_session(sid):
    sd = REPO / 'results' / 'mvp' / sid
    if not (sd / 'mvp_scaffold.npz').exists():
        return None
    scaff = np.load(sd / 'mvp_scaffold.npz')
    sc_json = json.load(open(sd / 'mvp_scaffold.json'))
    digest = json.load(open(REPO / 'data' / 'digest' / 'v1' / f'{sid}.json'))
    obs_channels = sc_json['observation_channels']
    if 'bl_event_coincidence' not in obs_channels:
        return None
    z = scaff['obs'][:, obs_channels.index('bl_event_coincidence')]
    valid = scaff['obs_valid'][:, obs_channels.index('bl_event_coincidence')]
    t = scaff['t_common']
    periods = parse_periods(digest['markers'])
    return {'sid': sid, 'z': z, 'valid': valid, 't': t, 'periods': periods,
            'protocol': digest.get('protocol', '')}


def render_one_panel(ax, data, smooth_sigma_s=2.5):
    """Render one session: condition shading + raw z + smoothed envelope.

    X-axis is session-relative time (seconds since session start), so all
    sessions are directly comparable visually.
    """
    if data is None:
        ax.text(0.5, 0.5, 'no scaffold', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#888')
        return

    z = data['z']
    t_lsl = data['t']
    periods_lsl = data['periods']

    # Rebase to session-relative time: t=0 at first marker (or at t_common[0]
    # if no markers). Periods and t are both rebased so they stay aligned.
    if periods_lsl:
        t0_session = min(p[1] for p in periods_lsl)
    else:
        t0_session = float(t_lsl[0])
    t = t_lsl - t0_session
    periods = [(n, t0 - t0_session, t1 - t0_session) for n, t0, t1 in periods_lsl]

    # Smooth for envelope (Gaussian, sigma in samples; data is at 2 Hz so 5 samples = 2.5s)
    from scipy.ndimage import gaussian_filter1d
    fs = 1.0 / float(np.median(np.diff(t)))
    sigma_samples = max(0.5, smooth_sigma_s * fs)
    z_smooth = gaussian_filter1d(z, sigma=sigma_samples)

    # Compute per-condition mean
    cond_means = []
    for name, t0, t1 in periods:
        m = (t >= t0) & (t <= t1)
        if m.sum() < 5: continue
        cond_means.append((name, t0, t1, z[m].mean()))

    ymin = min(-1.0, z.min() - 0.3)
    ymax = max(2.5, z.max() * 1.05)

    shade_periods(ax, periods, alpha=0.22, ymax=ymax)
    ax.plot(t, z, color='#388E3C', linewidth=0.5, alpha=0.45,
            label='bl_event_coincidence z (raw)')
    ax.plot(t, z_smooth, color='#1B5E20', linewidth=1.6,
            label=f'smoothed (σ={smooth_sigma_s:.1f}s)')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.6)
    ax.axhline(2, color='red', linestyle='--', linewidth=0.5, alpha=0.45)

    # Per-condition mean as small horizontal bars
    for name, t0, t1, mu in cond_means:
        ax.plot([t0, t1], [mu, mu], color='black', linewidth=2, alpha=0.6)
        ax.text(t1, mu + 0.05, f'{mu:+.2f}', ha='right', va='bottom',
                fontsize=8, color='black')

    ax.set_xlim(0, t[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('z', fontsize=9)
    ax.set_title(f'{data["sid"]}  ·  {data["protocol"]}  ·  '
                 f'overall mean z={z.mean():+.3f},  %(z≥2)={(z>=2).mean()*100:.1f}%',
                 fontsize=10, fontweight='bold', loc='left')
    ax.tick_params(labelsize=7)
    ax.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='+', default=None,
                    help='Session IDs to plot (default: 6-panel preset)')
    ap.add_argument('--smooth-sigma-s', type=float, default=2.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sids = args.sessions if args.sessions else DEFAULT_GRID
    n = len(sids)
    ncols = 1
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2.8 * nrows),
                              sharex=False)
    if n == 1: axes = [axes]
    for ax, sid in zip(axes, sids):
        data = load_session(sid)
        render_one_panel(ax, data, smooth_sigma_s=args.smooth_sigma_s)

    axes[-1].set_xlabel('time (s) since session start', fontsize=10)
    fig.suptitle(f'bl_event_coincidence timecourses — N={n} representative sessions  '
                 f'(meditation + PE protocols), with experimental conditions shaded',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.out) if args.out else (REPO / 'results' / 'mvp' /
                                              f'bl_event_coincidence_timecourses.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
