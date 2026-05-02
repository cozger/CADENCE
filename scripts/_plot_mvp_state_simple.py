"""ONE clean figure to answer reviewer questions:

  1. Why are these distinct coupling states? (not just 1 state with noise)
  2. How does the model characterize a session?

Two panels:

  LEFT  — per-state condition residency, side-by-side. Shows that even though
          NULL / OTHER / SHARED have small instantaneous loadings, they live
          in DIFFERENT experimental conditions: NULL dominates baselines and
          meditation; OTHER + SHARED dominate psychoeducation; COUP peaks in
          conversation and deep meditation. The states are temporally distinct,
          not just observation-magnitude clones.

  RIGHT — per-session state composition stacked bars, ordered by protocol.
          Shows the practical model output: each session is summarised as a
          4-state composition, and PE vs meditation produce visibly different
          compositions.

Designed for non-computational reviewers — no jargon in the labels, no math
in the caption, every quantity is a percentage of time.

Usage:
    python scripts/_plot_mvp_state_simple.py
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'

STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',
    'OTHER':  '#9C27B0',
    'SHARED': '#2E7D32',
    'COUP':   '#FF6F00',
}

# Plain-language descriptors for reviewers
STATE_BLURB = {
    'NULL':   'quiescent / idle',
    'OTHER':  'active, weakly coupled',
    'SHARED': 'shared body expression',
    'COUP':   'EEG concordance',
}

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1',
                    'PE_1', 'PE_2',
                    'meditate_B', 'meditate_K', 'conv_2']
CONDITION_NICE = {
    'base_EO': 'baseline\neyes open',
    'base_EC': 'baseline\neyes closed',
    'conv_1': 'conversation 1',
    'conv_2': 'conversation 2',
    'PE_1': 'psychoed 1',
    'PE_2': 'psychoed 2',
    'meditate_B': 'meditate B',
    'meditate_K': 'meditate K',
}


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


def main():
    with open(HIER / 'mvp_hierarchical_results.json') as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    sids = [s['session'] for s in d['sessions']]
    state_colors = [STATE_COLOR_BY_LABEL.get(l, '#37474F') for l in state_labels]

    # Per-condition pooled state usage (duration-weighted)
    period_acc = {}
    period_dur = {}
    for s in d['sessions']:
        for p in s.get('periods', []):
            if not (isinstance(p, list) and len(p) == 3):
                continue
            name, dur, u = p
            if name not in CONDITION_ORDER:
                continue
            period_acc.setdefault(name, []).append(np.array(u) * dur)
            period_dur[name] = period_dur.get(name, 0.0) + dur

    cond_present = [c for c in CONDITION_ORDER if c in period_acc]
    cond_state = np.zeros((len(cond_present), K))  # cond_state[c, k] = fraction
    for ci, cn in enumerate(cond_present):
        weighted = np.sum(np.stack(period_acc[cn]), axis=0)
        cond_state[ci] = weighted / period_dur[cn]

    # Per-session state composition (using JSON's stored 'usage' field)
    sess_state = np.zeros((len(sids), K))
    sess_protocol = []
    for i, s in enumerate(d['sessions']):
        sess_state[i] = np.array(s['usage'])
        # Determine protocol by looking at periods
        names = [p[0] for p in s['periods'] if isinstance(p, list)]
        if any('PE' in n for n in names):
            sess_protocol.append('PE')
        elif any('meditate' in n for n in names):
            sess_protocol.append('meditation')
        else:
            sess_protocol.append('?')

    # Sort sessions by protocol then by 1st-state usage (for visual coherence)
    proto_order = {'PE': 0, 'meditation': 1, '?': 2}
    sort_idx = sorted(range(len(sids)),
                      key=lambda i: (proto_order[sess_protocol[i]], -sess_state[i].max()))
    sids_sorted = [sids[i] for i in sort_idx]
    sess_state_sorted = sess_state[sort_idx]
    sess_protocol_sorted = [sess_protocol[i] for i in sort_idx]

    # ── PLOT ──
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4],
                          left=0.06, right=0.99, top=0.91, bottom=0.13,
                          wspace=0.25)

    # ── Left: per-condition state composition ──
    ax_l = fig.add_subplot(gs[0])
    bottom = np.zeros(len(cond_present))
    x = np.arange(len(cond_present))
    bar_w = 0.78
    for k in range(K):
        ax_l.bar(x, cond_state[:, k] * 100, bar_w, bottom=bottom * 100,
                 color=state_colors[k], edgecolor='white', linewidth=0.6,
                 label=f'{state_labels[k]} — {STATE_BLURB[state_labels[k]]}')
        # Annotate %
        for ci in range(len(cond_present)):
            seg = cond_state[ci, k] * 100
            if seg >= 6:
                ax_l.text(ci, (bottom[ci] + cond_state[ci, k] / 2) * 100,
                          f'{seg:.0f}%', ha='center', va='center',
                          fontsize=9, color='white' if k != 0 else 'black',
                          fontweight='bold')
        bottom += cond_state[:, k]
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([CONDITION_NICE[c] for c in cond_present],
                          fontsize=10, rotation=15, ha='right')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('% of condition time', fontsize=11)
    ax_l.set_title('A. Each experimental condition has a distinct coupling-state mix\n'
                   '(this is why the four states are distinct: they live in different conditions)',
                   fontsize=11, fontweight='bold', loc='left')
    ax_l.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=9, frameon=False)
    ax_l.grid(axis='y', alpha=0.25, linestyle=':')

    # ── Right: per-session state composition ──
    ax_r = fig.add_subplot(gs[1])
    bottom = np.zeros(len(sids_sorted))
    x = np.arange(len(sids_sorted))
    bar_w = 0.86
    for k in range(K):
        ax_r.bar(x, sess_state_sorted[:, k] * 100, bar_w, bottom=bottom * 100,
                 color=state_colors[k], edgecolor='white', linewidth=0.4,
                 label=f'{state_labels[k]}')
        bottom += sess_state_sorted[:, k]
    # Protocol grouping bar
    pe_count = sum(1 for p in sess_protocol_sorted if p == 'PE')
    med_count = sum(1 for p in sess_protocol_sorted if p == 'meditation')
    if pe_count > 0:
        ax_r.axvspan(-0.5, pe_count - 0.5, ymin=-0.04, ymax=0,
                     color='#C62828', alpha=0.7, clip_on=False, transform=ax_r.transData)
        ax_r.text((pe_count - 1) / 2, -7, f'psychoeducation (N={pe_count})',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  color='#C62828')
    if med_count > 0:
        ax_r.text(pe_count + (med_count - 1) / 2, -7,
                  f'meditation (N={med_count})',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  color='#388E3C')
    ax_r.axvline(pe_count - 0.5, color='black', linewidth=1.5, linestyle='-',
                 alpha=0.7)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(sids_sorted, fontsize=8, rotation=70, ha='right')
    ax_r.set_ylim(-1, 100)
    ax_r.set_ylabel('% of session time', fontsize=11)
    ax_r.set_title('B. Each session is summarised by its 4-state composition\n'
                   '(model output: PE sessions look different from meditation sessions)',
                   fontsize=11, fontweight='bold', loc='left')
    ax_r.grid(axis='y', alpha=0.25, linestyle=':')

    fig.suptitle('Coupling-state model — what the four states are, and what each session looks like '
                 f'(N={len(sids)} canonical dyads)',
                 fontsize=13, fontweight='bold')

    out = HIER / f'state_signatures_simple_{_VARIANT}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
