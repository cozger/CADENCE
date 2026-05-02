"""3-way contrast: y_06 meditate_B vs base_EC vs y_32 avg(PE_1,PE_2) vs base_EC.

Within-session baseline subtraction removes session-specific surrogate calibration
offsets. Compares "active meditation" (y_06) and "active didactic engagement"
(y_32 PE) effects on S-map summary scalars relative to eyes-closed rest.

Both sessions have p1=patient, p2=therapist:
    cs_tp = patient -> therapist
    cs_pt = therapist -> patient
"""

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2_DIR = ROOT / 'results' / 'native_rate_coupling_p2'

with open(P2_DIR / 'y_06_p2_summary.json') as f:
    y06 = json.load(f)['summary']
with open(P2_DIR / 'y_32_p2_summary.json') as f:
    y32 = json.load(f)['summary']


def pooled_std(s1, n1, s2, n2):
    """Pooled std for Cohen's d (Welch-style)."""
    if n1 < 2 or n2 < 2:
        return float('nan')
    return math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


def cohens_d(m1, s1, n1, m2, s2, n2):
    sp = pooled_std(s1, n1, s2, n2)
    if sp == 0 or math.isnan(sp):
        return float('nan')
    return (m1 - m2) / sp


def avg_two_conditions(stat_a, stat_b):
    """Pooled mean/std/n for two conditions treated as one population."""
    n1, n2 = stat_a['n'], stat_b['n']
    m1, m2 = stat_a['mean'], stat_b['mean']
    s1, s2 = stat_a['std'], stat_b['std']
    n = n1 + n2
    m = (n1 * m1 + n2 * m2) / n
    # var via the law of total variance
    var = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2 + n1 * (m1 - m) ** 2 + n2 * (m2 - m) ** 2) / (n - 1)
    return {'mean': m, 'std': math.sqrt(var), 'n': n}


print('=' * 110)
print(' Within-session contrasts on S-map summary scalar (L2 norm of beta over E lags)')
print(' Convention: p1=patient, p2=therapist for BOTH y_06 and y_32')
print('   cs_tp = patient -> therapist')
print('   cs_pt = therapist -> patient')
print('=' * 110)

header = f"{'Modality':<10} {'Band':<5} {'Direction':<25} {'condition':<12} {'mean':>8} {'std':>7} {'n':>7} {'Δ vs base_EC':>13} {'d':>7}"
print(header)
print('-' * len(header))

for mod_band in ['eeg_theta', 'eeg_alpha', 'eeg_beta', 'ecg_LF', 'ecg_HF']:
    for direction_key, direction_label in [('cs_tp', 'patient->therapist'),
                                           ('cs_pt', 'therapist->patient')]:
        # y_06 contrast: meditate_B vs base_EC
        y06_base = y06[mod_band][direction_key]['base_EC']
        y06_med = y06[mod_band][direction_key]['meditate_B']
        d_y06 = cohens_d(y06_med['mean'], y06_med['std'], y06_med['n'],
                         y06_base['mean'], y06_base['std'], y06_base['n'])
        delta_y06 = y06_med['mean'] - y06_base['mean']

        # y_32 contrast: avg(PE_1, PE_2) vs base_EC
        y32_base = y32[mod_band][direction_key]['base_EC']
        y32_pe = avg_two_conditions(y32[mod_band][direction_key]['PE_1'],
                                    y32[mod_band][direction_key]['PE_2'])
        d_y32 = cohens_d(y32_pe['mean'], y32_pe['std'], y32_pe['n'],
                         y32_base['mean'], y32_base['std'], y32_base['n'])
        delta_y32 = y32_pe['mean'] - y32_base['mean']

        mod = mod_band.split('_')[0]
        band = mod_band.split('_')[1]
        # Print y_06 row
        print(f"{mod:<10} {band:<5} {direction_label:<25} "
              f"{'y_06 base_EC':<12} {y06_base['mean']:>8.3f} {y06_base['std']:>7.3f} {y06_base['n']:>7} "
              f"{'-':>13} {'-':>7}")
        print(f"{mod:<10} {band:<5} {direction_label:<25} "
              f"{'y_06 med_B':<12} {y06_med['mean']:>8.3f} {y06_med['std']:>7.3f} {y06_med['n']:>7} "
              f"{delta_y06:>+13.3f} {d_y06:>+7.2f}")
        print(f"{mod:<10} {band:<5} {direction_label:<25} "
              f"{'y_32 base_EC':<12} {y32_base['mean']:>8.3f} {y32_base['std']:>7.3f} {y32_base['n']:>7} "
              f"{'-':>13} {'-':>7}")
        print(f"{mod:<10} {band:<5} {direction_label:<25} "
              f"{'y_32 PE_avg':<12} {y32_pe['mean']:>8.3f} {y32_pe['std']:>7.3f} {y32_pe['n']:>7} "
              f"{delta_y32:>+13.3f} {d_y32:>+7.2f}")
        print()

print('=' * 110)
print(' Side-by-side: meditate_B effect (y_06)  vs  PE engagement effect (y_32)')
print('   Both reported as Δ from each session\'s own base_EC, with Cohen\'s d')
print('=' * 110)

print(f"{'Modality':<10} {'Band':<5} {'Direction':<25} "
      f"{'med_B Δ':>10} {'med_B d':>9}   {'PE_avg Δ':>10} {'PE_avg d':>9}   {'sign agree':>10}")
print('-' * 110)
for mod_band in ['eeg_theta', 'eeg_alpha', 'eeg_beta', 'ecg_LF', 'ecg_HF']:
    for direction_key, direction_label in [('cs_tp', 'patient->therapist'),
                                           ('cs_pt', 'therapist->patient')]:
        y06_base = y06[mod_band][direction_key]['base_EC']
        y06_med = y06[mod_band][direction_key]['meditate_B']
        delta_y06 = y06_med['mean'] - y06_base['mean']
        d_y06 = cohens_d(y06_med['mean'], y06_med['std'], y06_med['n'],
                         y06_base['mean'], y06_base['std'], y06_base['n'])

        y32_base = y32[mod_band][direction_key]['base_EC']
        y32_pe = avg_two_conditions(y32[mod_band][direction_key]['PE_1'],
                                    y32[mod_band][direction_key]['PE_2'])
        delta_y32 = y32_pe['mean'] - y32_base['mean']
        d_y32 = cohens_d(y32_pe['mean'], y32_pe['std'], y32_pe['n'],
                         y32_base['mean'], y32_base['std'], y32_base['n'])

        sign_agree = 'yes' if (delta_y06 * delta_y32) > 0 else ('no' if (delta_y06 * delta_y32) < 0 else '0')
        mod = mod_band.split('_')[0]
        band = mod_band.split('_')[1]
        print(f"{mod:<10} {band:<5} {direction_label:<25} "
              f"{delta_y06:>+10.3f} {d_y06:>+9.2f}   {delta_y32:>+10.3f} {d_y32:>+9.2f}   {sign_agree:>10}")

print()
print('Notes:')
print(' * Δ = condition_mean − base_EC_mean (within-session, removes session calibration offset)')
print(' * |d| > 0.5 conventionally meaningful; |d| > 0.2 small but worth noting at large n')
print(' * sign_agree=yes means meditation and PE shift S-map magnitude in the same direction')
print('   relative to their respective rest baselines — interpret as common engagement signature')
print(' * sign_agree=no means meditation and PE shift in opposite directions — the channel')
print('   discriminates these two active conditions from rest in opposite ways')
