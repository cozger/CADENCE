"""Aggregate per-session theta S-map per-condition means and plot grand averages.

Reads per-session JSONs from results/native_rate_coupling_p2/theta_grand_avg/
(falls back to parent dir for y_06, y_32 if not in subdir). Produces a single
figure with 2x2 subplots:

    Top row    : meditation protocol (y_06, y_17, y_19)
    Bottom row : PE protocol (y_01, y_05, y_10, y_32, y_41)
    Left col   : direction PATIENT -> THERAPIST
    Right col  : direction THERAPIST -> PATIENT

Role mapping per session is explicit, never assumed.  Each per-session JSON
carries 'p1_role' / 'p2_role' (set by the P2 script at run time).  When p1
is the therapist for a particular session, cs_tp (= z_p1_to_p2) actually
encodes therapist -> patient and must be plotted in the right column rather
than the left.  The aggregator does this mapping per session.

X axis labels in protocol-specific order (CADENCE.md):
    Meditation: base_EO -> base_EC -> conv_1 -> meditate_B -> meditate_K -> conv_2
    PE        : base_EO -> base_EC -> conv_1 -> PE_1 -> PE_2 -> conv_2

Sessions missing a condition contribute nothing for that condition (handled in mean).
"""
import json
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_DIR = os.path.join(ROOT, 'results', 'native_rate_coupling_p2')
SUB_DIR = os.path.join(P2_DIR, 'theta_grand_avg')

# Protocol session lists keyed by their --session arg (cache name)
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
    # 'baseline' is the old single-block protocol some sessions used instead
    # of the EO/EC split. Treat it as its own column for visibility, but use
    # it as a within-session reference when EC is missing.
    'meditation': ['baseline', 'base_EO', 'base_EC', 'conv_1',
                   'meditate_B', 'meditate_K', 'conv_2'],
    'PE':         ['base_EO', 'base_EC', 'conv_1', 'PE_1', 'PE_2', 'conv_2'],
}

# Ordered fallback list for within-session baseline subtraction.
BASELINE_PRIORITY = ['base_EC', 'baseline', 'base_EO']

DIRECTIONS = [
    ('patient_to_therapist', 'patient → therapist'),
    ('therapist_to_patient', 'therapist → patient'),
]


def role_to_cs_field(p1_role, p2_role, target_direction):
    """Map a target therapist/patient direction to the cs_tp / cs_pt field
    that holds it for THIS session, given p1/p2 roles. Returns None if the
    session's roles don't include both 'therapist' and 'patient'.
    """
    roles = {p1_role, p2_role}
    if not {'therapist', 'patient'}.issubset(roles):
        return None
    if target_direction == 'patient_to_therapist':
        # patient -> therapist
        if p1_role == 'patient' and p2_role == 'therapist':
            return 'cs_tp'    # cs_tp = z_p1_to_p2 = patient -> therapist
        if p1_role == 'therapist' and p2_role == 'patient':
            return 'cs_pt'    # cs_pt = z_p2_to_p1 = patient -> therapist (flipped session)
    if target_direction == 'therapist_to_patient':
        if p1_role == 'patient' and p2_role == 'therapist':
            return 'cs_pt'
        if p1_role == 'therapist' and p2_role == 'patient':
            return 'cs_tp'
    return None

PROTOCOL_COLORS = {
    'meditation': '#1f77b4',  # blue
    'PE':         '#d62728',  # red
}


def load_summary(session_arg):
    """Try OUT_DIR first, fall back to parent dir. Returns the 'summary' dict
    or None if not found."""
    for base in (SUB_DIR, P2_DIR):
        path = os.path.join(base, f'{session_arg}_p2_summary.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def collect_per_session(protocol, baseline_subtract=False):
    """Return dict {direction_key: {condition: [(session_display, value), ...]}}.

    Each list element pairs a session display name with its per-condition mean.
    Direction is resolved per-session via p1_role/p2_role, never assumed.
    If baseline_subtract is True, every per-session value is reported as
    (condition_mean - base_EC_mean) for that session — removes per-session
    surrogate calibration offset."""
    accum = {dk: {cond: [] for cond in CONDITIONS[protocol]} for dk, _ in DIRECTIONS}
    used_sessions = []

    for display, arg in SESSIONS[protocol]:
        data, path = load_summary(arg)
        if data is None:
            print(f'[skip] {display} ({arg}): no summary JSON found')
            continue
        if 'eeg_theta' not in data['summary']:
            print(f'[skip] {display}: theta band missing')
            continue
        p1_role = data.get('p1_role')
        p2_role = data.get('p2_role')
        if p1_role is None or p2_role is None:
            print(f'[skip] {display}: missing role metadata in JSON')
            continue
        used_sessions.append((display, arg, path, p1_role, p2_role))
        theta = data['summary']['eeg_theta']

        for dk, dlabel in DIRECTIONS:
            cs_field = role_to_cs_field(p1_role, p2_role, dk)
            if cs_field is None:
                print(f'[warn] {display}: unable to map "{dk}" with '
                      f'p1={p1_role} p2={p2_role}; skipping direction')
                continue
            cs = theta[cs_field]
            base_value = None
            base_used = None
            if baseline_subtract:
                for ref in BASELINE_PRIORITY:
                    if ref in cs and cs[ref]['n'] > 0:
                        base_value = cs[ref]['mean']
                        base_used = ref
                        break
                if base_value is None:
                    print(f'[skip-bsub] {display}: no baseline-like segment '
                          f'(tried {BASELINE_PRIORITY}); cannot subtract')
                    continue
            for cond in CONDITIONS[protocol]:
                if cond in cs and cs[cond]['n'] > 0:
                    val = cs[cond]['mean']
                    if base_value is not None:
                        val = val - base_value
                    accum[dk][cond].append((display, val))
    return accum, used_sessions


def make_figure(baseline_subtract, out_png_basename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey='all')
    title_extra = ('\n(condition − base_EC, within-session subtracted)'
                   if baseline_subtract
                   else '\n(raw mean per condition)')
    fig.suptitle('Theta S-map (cs = surrogate z) grand averages by condition'
                 + title_extra
                 + '\nROI-averaged frontal pool (AF3, F7, F3, F4, F8, AF4)',
                 fontsize=12, y=0.99)

    summary_for_save = {}

    for row, protocol in enumerate(['meditation', 'PE']):
        accum, used_sessions = collect_per_session(protocol, baseline_subtract=baseline_subtract)
        n_sess = len(used_sessions)
        print(f'\n=== {protocol} ({n_sess} sessions) ===')
        for d, arg, path, p1r, p2r in used_sessions:
            print(f'   {d}  p1={p1r}  p2={p2r}  <- {os.path.basename(path)}')

        protocol_summary = {
            'sessions': [{'display': d, 'arg': a, 'p1_role': p1r, 'p2_role': p2r}
                         for d, a, _, p1r, p2r in used_sessions],
            'n_sessions': n_sess,
            'directions': {}}

        for col, (dk, dlabel) in enumerate(DIRECTIONS):
            ax = axes[row, col]
            conds = CONDITIONS[protocol]
            means = []
            sems = []
            ns = []
            per_session_for_save = {cond: [] for cond in conds}
            for cond in conds:
                pairs = accum[dk][cond]   # [(display, val), ...]
                vals = [v for _, v in pairs]
                per_session_for_save[cond] = [{'session': d, 'value': v} for d, v in pairs]
                if len(vals) == 0:
                    means.append(np.nan)
                    sems.append(np.nan)
                    ns.append(0)
                else:
                    means.append(float(np.mean(vals)))
                    sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                                if len(vals) > 1 else 0.0)
                    ns.append(len(vals))

            color = PROTOCOL_COLORS[protocol]
            x = np.arange(len(conds))
            ax.errorbar(x, means, yerr=sems, marker='o', ms=8, lw=2,
                        capsize=5, color=color)
            for xi, n in enumerate(ns):
                ax.annotate(f'n={n}', xy=(xi, ax.get_ylim()[1] if ax.get_ylim()[1] else 0),
                            xytext=(0, -2), textcoords='offset points',
                            ha='center', va='top', fontsize=7, color='gray')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_xticks(x)
            ax.set_xticklabels(conds, rotation=20, ha='right')
            ax.set_ylabel('Mean S-map summary z')
            ax.set_title(f'{protocol}  |  {dlabel}  (n={n_sess} sessions)',
                         fontsize=10)
            ax.grid(alpha=0.3)

            protocol_summary['directions'][dk] = {
                'label': dlabel,
                'conditions': conds,
                'means_per_condition': means,
                'sems_per_condition': sems,
                'n_sessions_per_condition': ns,
                'per_session_values': per_session_for_save,
            }

        summary_for_save[protocol] = protocol_summary

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = os.path.join(SUB_DIR, out_png_basename + '.png')
    out_json = os.path.join(SUB_DIR, out_png_basename + '.json')
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    with open(out_json, 'w') as f:
        json.dump(summary_for_save, f, indent=2)

    print(f'Figure saved : {out_png}')
    print(f'Summary saved: {out_json}')
    return summary_for_save


def make_per_session_figure(baseline_subtract):
    """One row per protocol; one column per direction; one line per session."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey='all')
    title_extra = ('\n(condition − base_EC, within-session)'
                   if baseline_subtract
                   else '\n(raw mean per condition)')
    fig.suptitle('Theta S-map — per-session traces' + title_extra
                 + '\nROI-averaged frontal pool (AF3, F7, F3, F4, F8, AF4)',
                 fontsize=12, y=0.99)

    cmap_med = plt.cm.Blues(np.linspace(0.45, 0.95, len(SESSIONS['meditation'])))
    cmap_pe = plt.cm.Reds(np.linspace(0.45, 0.95, len(SESSIONS['PE'])))

    for row, protocol in enumerate(['meditation', 'PE']):
        cmap = cmap_med if protocol == 'meditation' else cmap_pe
        sessions = SESSIONS[protocol]
        for col, (dk, dlabel) in enumerate(DIRECTIONS):
            ax = axes[row, col]
            for si, (display, arg) in enumerate(sessions):
                data, path = load_summary(arg)
                if data is None or 'eeg_theta' not in data['summary']:
                    continue
                p1_role = data.get('p1_role')
                p2_role = data.get('p2_role')
                if p1_role is None or p2_role is None:
                    continue
                cs_field = role_to_cs_field(p1_role, p2_role, dk)
                if cs_field is None:
                    continue
                cs = data['summary']['eeg_theta'][cs_field]
                base_value = None
                if baseline_subtract:
                    for ref in BASELINE_PRIORITY:
                        if ref in cs and cs[ref]['n'] > 0:
                            base_value = cs[ref]['mean']
                            break
                    if base_value is None:
                        continue
                conds = CONDITIONS[protocol]
                xs, ys = [], []
                for ci, cond in enumerate(conds):
                    if cond in cs and cs[cond]['n'] > 0:
                        v = cs[cond]['mean']
                        if base_value is not None:
                            v = v - base_value
                        xs.append(ci)
                        ys.append(v)
                role_tag = ('p1' if p1_role == ('therapist' if 'therapist' in dk
                                               else 'patient') else 'p2')
                # Compact role tag in legend, e.g. y_10 (p1=ther)
                legend_label = f'{display} (p1={p1_role[:3]})'
                ax.plot(xs, ys, marker='o', ms=6, lw=1.5,
                        color=cmap[si], label=legend_label)
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
    out_png = os.path.join(SUB_DIR, f'theta_per_session{suffix}.png')
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-session figure saved: {out_png}')


def main():
    print('=== RAW (per-condition mean) ===')
    raw_summary = make_figure(baseline_subtract=False,
                              out_png_basename='theta_grand_avg')
    print()
    print('=== BASELINE-SUBTRACTED (condition − base_EC, within-session) ===')
    bsub_summary = make_figure(baseline_subtract=True,
                               out_png_basename='theta_grand_avg_baseline_subtracted')

    print()
    print('=== Per-session detail (raw + baseline-subtracted) ===')
    make_per_session_figure(baseline_subtract=False)
    make_per_session_figure(baseline_subtract=True)

    # Console summary tables
    for label, summary_for_save in [('RAW', raw_summary),
                                    ('BASELINE-SUBTRACTED', bsub_summary)]:
        print('\n' + '=' * 100)
        print(f'   {label}')
        print('=' * 100)
        print(f'{"protocol":<11} {"direction":<25} {"condition":<14} '
              f'{"mean":>9} {"sem":>8} {"n":>4}')
        print('-' * 100)
        for protocol, ps in summary_for_save.items():
            for dk, dd in ps['directions'].items():
                for cond, mu, se, n in zip(dd['conditions'], dd['means_per_condition'],
                                           dd['sems_per_condition'], dd['n_sessions_per_condition']):
                    mu_s = f'{mu:+.3f}' if mu == mu else '   nan'
                    se_s = f'{se:.3f}' if se == se else '  nan'
                    print(f'{protocol:<11} {dd["label"]:<25} {cond:<14} '
                          f'{mu_s:>9} {se_s:>8} {n:>4}')
            print()


if __name__ == '__main__':
    main()
