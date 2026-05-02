"""Paired-sample stats on within-session theta S-map condition contrasts.

For each (protocol, condition, direction), gather the within-session
(condition_mean - baseline_mean) values across sessions and report:
  - n sessions contributing
  - paired t-test p-value vs zero
  - Wilcoxon signed-rank p (more robust at small n)
  - 95% bootstrap CI on the grand mean

Direction is resolved per-session via p1_role / p2_role (never assumed).
Baseline reference falls back: base_EC -> baseline -> base_EO.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
P2_DIR = ROOT / 'results' / 'native_rate_coupling_p2'
SUB_DIR = P2_DIR / 'theta_grand_avg'

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

DIRECTIONS = [
    ('patient_to_therapist', 'patient → therapist'),
    ('therapist_to_patient', 'therapist → patient'),
]

BASELINE_PRIORITY = ['base_EC', 'baseline', 'base_EO']


def role_to_cs_field(p1_role, p2_role, target):
    if p1_role == 'patient' and p2_role == 'therapist':
        return 'cs_tp' if target == 'patient_to_therapist' else 'cs_pt'
    if p1_role == 'therapist' and p2_role == 'patient':
        return 'cs_pt' if target == 'patient_to_therapist' else 'cs_tp'
    return None


def load_summary(session_arg):
    for base in (SUB_DIR, P2_DIR):
        path = base / f'{session_arg}_p2_summary.json'
        if path.exists():
            with open(path) as f:
                return json.load(f), path
    return None, None


def collect_contrasts(protocol):
    """{direction: {condition: [(session, contrast_value), ...]}}"""
    out = {dk: {c: [] for c in CONDITIONS[protocol]} for dk, _ in DIRECTIONS}
    for display, arg in SESSIONS[protocol]:
        data, path = load_summary(arg)
        if data is None or 'eeg_theta' not in data['summary']:
            continue
        p1_role, p2_role = data.get('p1_role'), data.get('p2_role')
        if p1_role is None:
            continue
        theta = data['summary']['eeg_theta']
        for dk, _ in DIRECTIONS:
            field = role_to_cs_field(p1_role, p2_role, dk)
            if field is None:
                continue
            cs = theta[field]
            base_value = None
            for ref in BASELINE_PRIORITY:
                if ref in cs and cs[ref]['n'] > 0:
                    base_value = cs[ref]['mean']
                    break
            if base_value is None:
                continue
            for cond in CONDITIONS[protocol]:
                if cond in cs and cs[cond]['n'] > 0:
                    out[dk][cond].append((display, cs[cond]['mean'] - base_value))
    return out


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float('nan'), float('nan')
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def report(protocol):
    contrasts = collect_contrasts(protocol)
    print('=' * 110)
    print(f'  {protocol.upper()} protocol — within-session (condition − baseline) contrasts on θ S-map')
    print('  Baseline ref order: base_EC → baseline → base_EO')
    print('=' * 110)
    print(f'{"direction":<25} {"condition":<14} {"n":>3} '
          f'{"mean":>9} {"std":>8} {"sem":>8}   '
          f'{"t-stat":>8} {"t-p":>8} {"W-p":>8}   {"95% CI":>17}  contributors')
    print('-' * 110)
    for dk, dlabel in DIRECTIONS:
        for cond in CONDITIONS[protocol]:
            pairs = contrasts[dk][cond]
            sessions = [p[0] for p in pairs]
            vals = np.asarray([p[1] for p in pairs])
            if len(vals) < 1:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            sem = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            if len(vals) >= 2:
                t, tp = stats.ttest_1samp(vals, 0.0)
            else:
                t, tp = float('nan'), float('nan')
            if len(vals) >= 1 and not all(v == 0 for v in vals):
                try:
                    if len(vals) >= 2:
                        wp = stats.wilcoxon(vals).pvalue
                    else:
                        wp = float('nan')
                except ValueError:
                    wp = float('nan')
            else:
                wp = float('nan')
            lo, hi = bootstrap_ci(vals)
            ci_str = f'[{lo:+.3f},{hi:+.3f}]' if not math.isnan(lo) else '       -        '
            t_s = f'{t:+.2f}' if not math.isnan(t) else '   -  '
            tp_s = f'{tp:.3f}' if not math.isnan(tp) else '  -  '
            wp_s = f'{wp:.3f}' if not math.isnan(wp) else '  -  '
            print(f'{dlabel:<25} {cond:<14} {len(vals):>3} '
                  f'{mean:>+9.3f} {std:>8.3f} {sem:>8.3f}   '
                  f'{t_s:>8} {tp_s:>8} {wp_s:>8}   {ci_str:>17}  '
                  f'{",".join(sessions)}')
        print()


def main():
    for protocol in ['meditation', 'PE']:
        report(protocol)
    print()
    print('Notes:')
    print(' * t-stat is one-sample t against 0 (paired-difference framing).')
    print(' * W-p is Wilcoxon signed-rank, more robust at small n.')
    print(' * 95% CI is bootstrap (10000 iter) over the per-session contrasts.')
    print(' * "n" is the number of sessions contributing to the contrast for that')
    print('   (condition, direction). Baseline-less sessions (y_11, y_24) drop out.')


if __name__ == '__main__':
    main()
