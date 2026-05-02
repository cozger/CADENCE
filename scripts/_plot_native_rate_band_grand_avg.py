"""Generic per-band aggregator for the P2 native-rate S-map run.

Generates the same set of figures as the theta-specific plotter, but for any
EEG band (theta, alpha, beta). Reads per-session JSONs from:
    theta_grand_avg/         -> theta-only batch
    alpha_beta_grand_avg/    -> alpha+beta batch

Each per-session JSON carries explicit p1_role/p2_role; direction is
resolved per-session (never assumed). Within-session baseline subtraction
falls back base_EC -> baseline -> base_EO. Sessions without any baseline
segment drop out of the baseline-subtracted figure.

Usage:
    python scripts/_plot_native_rate_band_grand_avg.py --band alpha
    python scripts/_plot_native_rate_band_grand_avg.py --band beta
    python scripts/_plot_native_rate_band_grand_avg.py --all
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_DIR = os.path.join(ROOT, 'results', 'native_rate_coupling_p2')

# Per-band: where to look for JSONs (in priority order) and where to write figures.
BAND_LAYOUT = {
    'theta': {
        'json_dirs': [
            os.path.join(P2_DIR, 'theta_grand_avg'),
            P2_DIR,
        ],
        'fig_dir': os.path.join(P2_DIR, 'theta_grand_avg'),
        'json_band_key': 'eeg_theta',
    },
    'alpha': {
        'json_dirs': [
            os.path.join(P2_DIR, 'alpha_beta_grand_avg'),
            P2_DIR,
        ],
        'fig_dir': os.path.join(P2_DIR, 'alpha_beta_grand_avg'),
        'json_band_key': 'eeg_alpha',
    },
    'beta': {
        'json_dirs': [
            os.path.join(P2_DIR, 'alpha_beta_grand_avg'),
            P2_DIR,
        ],
        'fig_dir': os.path.join(P2_DIR, 'alpha_beta_grand_avg'),
        'json_band_key': 'eeg_beta',
    },
}

SESSIONS = {
    'meditation': [
        ('y_06', 'y_06'),
        ('y_17', 'y_17'),
        ('y_19', 'y_19_3242026'),
        ('y_04', 'y04_020626'),
        ('y_11', 'y11_022526'),
        ('y_24', 'y24_022526'),
    ],
    'PE': [
        ('y_01', 'y01_021726'),
        ('y_05', 'y05_02192026'),
        ('y_10', 'Y_10_03182026'),
        ('y_32', 'y_32_03132026'),
        ('y_41', 'Y_41_03192026'),
    ],
}

CONDITIONS = {
    'meditation': ['baseline', 'base_EO', 'base_EC', 'conv_1',
                   'meditate_B', 'meditate_K', 'conv_2'],
    'PE':         ['base_EO', 'base_EC', 'conv_1', 'PE_1', 'PE_2', 'conv_2'],
}

BASELINE_PRIORITY = ['base_EC', 'baseline', 'base_EO']

DIRECTIONS = [
    ('patient_to_therapist', 'patient → therapist'),
    ('therapist_to_patient', 'therapist → patient'),
]

PROTOCOL_COLORS = {
    'meditation': '#1f77b4',
    'PE':         '#d62728',
}


def role_to_cs_field(p1_role, p2_role, target_direction):
    if not {'therapist', 'patient'}.issubset({p1_role, p2_role}):
        return None
    if target_direction == 'patient_to_therapist':
        return 'cs_tp' if (p1_role == 'patient' and p2_role == 'therapist') else 'cs_pt'
    if target_direction == 'therapist_to_patient':
        return 'cs_pt' if (p1_role == 'patient' and p2_role == 'therapist') else 'cs_tp'
    return None


def load_summary(session_arg, json_dirs):
    for d in json_dirs:
        path = os.path.join(d, f'{session_arg}_p2_summary.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def collect_per_session(protocol, band_key, json_dirs, baseline_subtract=False):
    accum = {dk: {cond: [] for cond in CONDITIONS[protocol]} for dk, _ in DIRECTIONS}
    used_sessions = []
    for display, arg in SESSIONS[protocol]:
        data, path = load_summary(arg, json_dirs)
        if data is None:
            continue
        if band_key not in data['summary']:
            continue
        p1_role, p2_role = data.get('p1_role'), data.get('p2_role')
        if p1_role is None or p2_role is None:
            continue
        used_sessions.append((display, arg, path, p1_role, p2_role))
        for dk, _ in DIRECTIONS:
            cs_field = role_to_cs_field(p1_role, p2_role, dk)
            if cs_field is None:
                continue
            cs = data['summary'][band_key][cs_field]
            base_value = None
            if baseline_subtract:
                for ref in BASELINE_PRIORITY:
                    if ref in cs and cs[ref]['n'] > 0:
                        base_value = cs[ref]['mean']
                        break
                if base_value is None:
                    continue
            for cond in CONDITIONS[protocol]:
                if cond in cs and cs[cond]['n'] > 0:
                    val = cs[cond]['mean']
                    if base_value is not None:
                        val = val - base_value
                    accum[dk][cond].append((display, val))
    return accum, used_sessions


def make_grand_avg_figure(band, baseline_subtract):
    layout = BAND_LAYOUT[band]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey='all')
    bsub_str = ('\n(condition − base_EC, within-session subtracted)'
                if baseline_subtract else '\n(raw mean per condition)')
    fig.suptitle(f'{band.title()} S-map (cs = surrogate z) grand averages by condition'
                 + bsub_str
                 + '\nROI-averaged frontal pool (AF3, F7, F3, F4, F8, AF4)',
                 fontsize=12, y=0.99)

    summary_for_save = {}

    for row, protocol in enumerate(['meditation', 'PE']):
        accum, used_sessions = collect_per_session(
            protocol, layout['json_band_key'], layout['json_dirs'],
            baseline_subtract=baseline_subtract,
        )
        n_sess = len(used_sessions)
        protocol_summary = {
            'sessions': [{'display': d, 'arg': a, 'p1_role': p1r, 'p2_role': p2r}
                         for d, a, _, p1r, p2r in used_sessions],
            'n_sessions': n_sess,
            'directions': {},
        }
        for col, (dk, dlabel) in enumerate(DIRECTIONS):
            ax = axes[row, col]
            conds = CONDITIONS[protocol]
            means, sems, ns = [], [], []
            per_session = {c: [] for c in conds}
            for cond in conds:
                pairs = accum[dk][cond]
                vals = [v for _, v in pairs]
                per_session[cond] = [{'session': d, 'value': v} for d, v in pairs]
                if not vals:
                    means.append(np.nan); sems.append(np.nan); ns.append(0)
                else:
                    means.append(float(np.mean(vals)))
                    sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                                if len(vals) > 1 else 0.0)
                    ns.append(len(vals))
            x = np.arange(len(conds))
            ax.errorbar(x, means, yerr=sems, marker='o', ms=8, lw=2, capsize=5,
                        color=PROTOCOL_COLORS[protocol])
            for xi, n in enumerate(ns):
                ax.annotate(f'n={n}',
                            xy=(xi, ax.get_ylim()[1] if ax.get_ylim()[1] else 0),
                            xytext=(0, -2), textcoords='offset points',
                            ha='center', va='top', fontsize=7, color='gray')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha='right')
            ax.set_ylabel('Mean S-map summary z')
            ax.set_title(f'{protocol}  |  {dlabel}  (n={n_sess})', fontsize=10)
            ax.grid(alpha=0.3)

            protocol_summary['directions'][dk] = {
                'label': dlabel,
                'conditions': conds,
                'means_per_condition': means,
                'sems_per_condition': sems,
                'n_sessions_per_condition': ns,
                'per_session_values': per_session,
            }
        summary_for_save[protocol] = protocol_summary

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = '_baseline_subtracted' if baseline_subtract else ''
    out_png = os.path.join(layout['fig_dir'], f'{band}_grand_avg{suffix}.png')
    out_json = os.path.join(layout['fig_dir'], f'{band}_grand_avg{suffix}.json')
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    with open(out_json, 'w') as f:
        json.dump(summary_for_save, f, indent=2)
    print(f'  figure : {out_png}')
    print(f'  summary: {out_json}')
    return summary_for_save


def make_per_session_figure(band, baseline_subtract):
    layout = BAND_LAYOUT[band]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey='all')
    bsub_str = ('\n(condition − base_EC, within-session)' if baseline_subtract
                else '\n(raw mean per condition)')
    fig.suptitle(f'{band.title()} S-map — per-session traces' + bsub_str
                 + '\nROI-averaged frontal pool (AF3, F7, F3, F4, F8, AF4)',
                 fontsize=12, y=0.99)

    cmap_med = plt.cm.Blues(np.linspace(0.45, 0.95, len(SESSIONS['meditation'])))
    cmap_pe = plt.cm.Reds(np.linspace(0.45, 0.95, len(SESSIONS['PE'])))

    for row, protocol in enumerate(['meditation', 'PE']):
        cmap = cmap_med if protocol == 'meditation' else cmap_pe
        for col, (dk, dlabel) in enumerate(DIRECTIONS):
            ax = axes[row, col]
            for si, (display, arg) in enumerate(SESSIONS[protocol]):
                data, _ = load_summary(arg, layout['json_dirs'])
                if data is None or layout['json_band_key'] not in data['summary']:
                    continue
                p1_role, p2_role = data.get('p1_role'), data.get('p2_role')
                if p1_role is None:
                    continue
                cs_field = role_to_cs_field(p1_role, p2_role, dk)
                if cs_field is None:
                    continue
                cs = data['summary'][layout['json_band_key']][cs_field]
                base_value = None
                if baseline_subtract:
                    for ref in BASELINE_PRIORITY:
                        if ref in cs and cs[ref]['n'] > 0:
                            base_value = cs[ref]['mean']
                            break
                    if base_value is None:
                        continue
                xs, ys = [], []
                for ci, cond in enumerate(CONDITIONS[protocol]):
                    if cond in cs and cs[cond]['n'] > 0:
                        v = cs[cond]['mean']
                        if base_value is not None:
                            v = v - base_value
                        xs.append(ci); ys.append(v)
                ax.plot(xs, ys, marker='o', ms=6, lw=1.5, color=cmap[si],
                        label=f'{display} (p1={p1_role[:3]})')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_xticks(np.arange(len(CONDITIONS[protocol])))
            ax.set_xticklabels(CONDITIONS[protocol], rotation=20, ha='right')
            ax.set_ylabel('S-map summary z'
                          + (' − base_EC' if baseline_subtract else ''))
            ax.set_title(f'{protocol}  |  {dlabel}', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = '_baseline_subtracted' if baseline_subtract else ''
    out_png = os.path.join(layout['fig_dir'], f'{band}_per_session{suffix}.png')
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  per-session: {out_png}')


def run_band(band):
    print(f'\n=== {band.upper()} ===')
    make_grand_avg_figure(band, baseline_subtract=False)
    make_grand_avg_figure(band, baseline_subtract=True)
    make_per_session_figure(band, baseline_subtract=False)
    make_per_session_figure(band, baseline_subtract=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--band', choices=list(BAND_LAYOUT.keys()), default=None)
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    if args.all or args.band is None:
        for band in ['theta', 'alpha', 'beta']:
            run_band(band)
    else:
        run_band(args.band)


if __name__ == '__main__':
    main()
