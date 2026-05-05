"""Diagnostic: is alpha amplitude coupling in y_06 genuine?

Checks:
1. Alpha power by condition (confirm eyes-closed has strong alpha)
2. Detrended analysis (remove slow polynomials before correlation)
3. Pseudo-dyad null per band
4. Shorter segment surrogates (more conservative null)
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, hilbert
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.fast_cycles import (
    analyze_interbrain_cycles_multiband, analyze_interbrain_cycles, EEG_BANDS)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])

y06_name, y06_path = next((n, p) for n, p in entries if 'y_06' in n)
s_y06 = load_session_from_cache(y06_path, config=cfg)
intervals = parse_condition_intervals(s_y06)
print(f"{y06_name}: P1={s_y06.get('p1_role')}, P2={s_y06.get('p2_role')}")

# Also load a different session for pseudo-dyad
other_name, other_path = next((n, p) for n, p in entries
                               if 'y_06' not in n and 'y_17' not in n)
s_other = load_session_from_cache(other_path, config=cfg)
print(f"Pseudo-null: {other_name}\n")


def extract_real(session, start_s, end_s, detrend_order=0):
    p1_eeg, p1_ts = session['p1_eeg'], session['p1_eeg_ts']
    p2_eeg, p2_ts = session['p2_eeg'], session['p2_eeg_ts']
    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    p2_m = (p2_ts >= start_s) & (p2_ts < end_s)
    dur = end_s - start_s
    N = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if N < int(10 * FS):
        return None, None
    t = np.linspace(0, dur, N)
    p1 = np.stack([np.interp(t, p1_ts[p1_m]-start_s, p1_eeg[p1_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    p2 = np.stack([np.interp(t, p2_ts[p2_m]-start_s, p2_eeg[p2_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    # Avg ref
    for s in [p1, p2]:
        s -= s.mean(axis=1, keepdims=True)
    # Optional polynomial detrend
    if detrend_order > 0:
        t_norm = np.linspace(-1, 1, N)
        for s in [p1, p2]:
            for ch in range(14):
                coeffs = np.polyfit(t_norm, s[:, ch], detrend_order)
                trend = np.polyval(coeffs, t_norm)
                s[:, ch] -= trend
    # Z-normalize
    for s in [p1, p2]:
        for ch in range(14):
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd
    return p1.astype(np.float32), p2.astype(np.float32)


def extract_pseudodyad(s1, s2, start_s, end_s, detrend_order=0):
    """P1 from s1 at [start_s, end_s], P2 from s2 at beginning."""
    p1_eeg, p1_ts = s1['p1_eeg'], s1['p1_eeg_ts']
    p2_eeg, p2_ts = s2['p2_eeg'], s2['p2_eeg_ts']
    dur = end_s - start_s
    p1_m = (p1_ts >= start_s) & (p1_ts < end_s)
    p2_s = float(p2_ts[0])
    p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + dur)
    N = min(p1_m.sum(), p2_m.sum(), int(dur * FS))
    if N < int(10 * FS):
        return None, None
    t = np.linspace(0, dur, N)
    p1 = np.stack([np.interp(t, p1_ts[p1_m]-start_s, p1_eeg[p1_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    p2 = np.stack([np.interp(t, p2_ts[p2_m]-p2_s, p2_eeg[p2_m, c])
                    for c in range(14)], axis=1).astype(np.float64)
    for s in [p1, p2]:
        s -= s.mean(axis=1, keepdims=True)
    if detrend_order > 0:
        t_norm = np.linspace(-1, 1, N)
        for s in [p1, p2]:
            for ch in range(14):
                coeffs = np.polyfit(t_norm, s[:, ch], detrend_order)
                s[:, ch] -= np.polyval(coeffs, t_norm)
    for s in [p1, p2]:
        for ch in range(14):
            mu, sd = s[:, ch].mean(), max(s[:, ch].std(), 1e-8)
            s[:, ch] = (s[:, ch] - mu) / sd
    return p1.astype(np.float32), p2.astype(np.float32)


# ── CHECK 1: Alpha power per condition ───────────────────────────────────
print(f"{'='*60}")
print(f"  CHECK 1: Alpha band power (8-13 Hz) per condition")
print(f"{'='*60}")

sos_alpha = butter(4, [8.0, 13.0], btype='band', fs=FS, output='sos')

for start, end, cond in intervals:
    p1, p2 = extract_real(s_y06, start, end)
    if p1 is None:
        continue
    # Compute mean alpha power per participant
    p1_alpha_pow = np.mean([np.abs(hilbert(sosfiltfilt(sos_alpha, p1[:, ch])))**2
                             for ch in range(14)])
    p2_alpha_pow = np.mean([np.abs(hilbert(sosfiltfilt(sos_alpha, p2[:, ch])))**2
                             for ch in range(14)])
    print(f"  {cond:>15}: P1 alpha pow={p1_alpha_pow:.3f}  "
          f"P2 alpha pow={p2_alpha_pow:.3f}")


# ── CHECK 2: Standard vs detrended analysis ─────────────────────────────
print(f"\n{'='*60}")
print(f"  CHECK 2: Standard vs 3rd-order polynomial detrend")
print(f"  (volt_amp z per band)")
print(f"{'='*60}")

for cond_name in ['meditate_K', 'conv_2', 'base_EC', 'meditate_B']:
    interval = [(s, e) for s, e, c in intervals if c == cond_name]
    if not interval:
        continue
    start, end = interval[0]

    # Standard
    p1, p2 = extract_real(s_y06, start, end, detrend_order=0)
    if p1 is None:
        continue
    r_std = analyze_interbrain_cycles_multiband(p1, p2, FS, n_surrogates=200)

    # Detrended (order 3 polynomial removed)
    p1d, p2d = extract_real(s_y06, start, end, detrend_order=3)
    r_det = analyze_interbrain_cycles_multiband(p1d, p2d, FS, n_surrogates=200)

    print(f"\n  {cond_name}:")
    for band in ['theta', 'alpha', 'beta']:
        z_std = r_std['per_band'][band].get('volt_amp', {}).get('pooled_z', 0)
        z_det = r_det['per_band'][band].get('volt_amp', {}).get('pooled_z', 0)
        delta = z_det - z_std
        print(f"    {band:>6}: standard z={z_std:+.2f}  "
              f"detrend z={z_det:+.2f}  delta={delta:+.2f}")


# ── CHECK 3: Pseudo-dyad null per band ───────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHECK 3: Pseudo-dyad null (P1=y_06 meditate_K, P2={other_name})")
print(f"{'='*60}")

mk = [(s, e) for s, e, c in intervals if c == 'meditate_K'][0]
p1_null, p2_null = extract_pseudodyad(s_y06, s_other, mk[0], mk[1])
if p1_null is not None:
    r_null = analyze_interbrain_cycles_multiband(p1_null, p2_null, FS, n_surrogates=200)
    for band in ['theta', 'alpha', 'beta']:
        z = r_null['per_band'][band].get('volt_amp', {}).get('pooled_z', 0)
        r_val = r_null['per_band'][band].get('volt_amp', {}).get('mean_r', 0)
        print(f"  {band:>6}: z={z:+.2f}  r={r_val:+.4f}")

    cz = r_null['combined'].get('volt_amp', {}).get('stouffer_z', 0)
    print(f"  combined: z={cz:+.2f}")

print(f"\n{'='*60}")
