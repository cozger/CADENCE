"""Burst-triggered ITC vs continuous theta PLV comparison.

Tests on narrowband mixing + Kuramoto models (fair comparison).
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.signal import sosfiltfilt, butter
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial,
                                inject_eeg_coupling_kuramoto)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
from cadence.significance.burst_itc import burst_itc_temporal_localization
from cadence.constants import EPOC_DISTANCE

FS = 256.0; N_CH = 14; CENTER_CH = 2; DECAY_SIGMA = 0.5; LAG_SAMP = 8
K = 100; SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load EEG ─────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=1800)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + 1800)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(1800 * FS))
t_target = np.linspace(0, 1800, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(N_CH)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]
t_eeg = np.arange(N) / FS

for ch in range(N_CH):
    for sig in [p1_eeg, p2_eeg]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)

# Theta-filtered P1 for narrowband mixing
sos_theta = butter(4, [4.0, 8.0], btype='band', fs=FS, output='sos')
p1_theta = np.stack([sosfiltfilt(sos_theta, p1_eeg[:, ch])
                      for ch in range(N_CH)], axis=1).astype(np.float32)

# Oracle channels
oracle_ch = [i for i in range(N_CH)
             if np.exp(-EPOC_DISTANCE[CENTER_CH][i]**2 / (2*DECAY_SIGMA**2)) > 0.03]

freqs_theta = np.logspace(np.log10(4.0), np.log10(8.0), 5)


def eval_z(z_agg, gate_win):
    best = 0.0
    for thr in np.arange(0.4, 3.0, 0.1):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / 0.5)))
        nc, nn = gate_win.sum(), (~gate_win).sum()
        h = float((m & gate_win).sum() / max(nc, 1))
        f = float((m & ~gate_win).sum() / max(nn, 1))
        if 0.03 <= f <= 0.07:
            best = max(best, h)
    return best


def run_plv(label, p1, p2):
    t0 = time.perf_counter()
    _, z, _, d = wpli_temporal_localization(
        p1, p2, FS, channels=oracle_ch, center_freqs=freqs_theta,
        n_surrogates=K, n_cycles=[3, 7], window_s=20.0, stride_s=0.5,
        smooth_s=15.0, target_fa=0.03, min_event_s=5.0, metric='plv',
        seed=SEED, device=device)
    gw = np.interp(d['win_times'], t_eeg, gate.astype(float)) > 0.5
    zc = z[gw].mean() if gw.any() else 0
    hit = eval_z(z, gw)
    el = time.perf_counter() - t0
    print(f"  {label:>35}: hit={hit:5.1%}  z_c={zc:+.3f}  ({el:.1f}s)")
    return hit, zc


def run_itc(label, p1, p2, burst_pct=90):
    t0 = time.perf_counter()
    _, z, d = burst_itc_temporal_localization(
        p1, p2, FS, channels=oracle_ch, band=(4.0, 8.0),
        burst_percentile=burst_pct, n_surrogates=K,
        window_s=20.0, stride_s=0.5, smooth_s=15.0,
        min_bursts_per_window=3, target_fa=0.03,
        min_event_s=5.0, seed=SEED)
    gw = np.interp(d['win_times'], t_eeg, gate.astype(float)) > 0.5
    zc = z[gw].mean() if gw.any() else 0
    hit = eval_z(z, gw)
    el = time.perf_counter() - t0
    br = d['burst_rate_hz']
    nb = d['mean_bursts_per_window']
    print(f"  {label:>35}: hit={hit:5.1%}  z_c={zc:+.3f}  "
          f"({el:.1f}s) bursts={br:.2f}/s, {nb:.0f}/win")
    return hit, zc


print(f"Device: {device}")
print(f"Oracle channels: {len(oracle_ch)}\n")

for kappa in [0.1, 0.2, 0.4]:
    print(f"\n{'='*70}")
    print(f"  kappa = {kappa}")
    print(f"{'='*70}")

    # ── Narrowband mixing ────────────────────────────────────────────────
    p2_mix, _ = inject_eeg_coupling_spatial(
        p1_theta, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)
    p1_c = p1_eeg.copy()
    for ch in range(N_CH):
        for s in [p1_c, p2_mix]:
            s[:, ch] = (s[:, ch] - s[:, ch].mean()) / max(s[:, ch].std(), 1e-8)

    print(f"\n  Narrowband mixing:")
    run_plv("theta PLV (continuous)", p1_c, p2_mix)
    run_itc("burst ITC (90th pct)", p1_c, p2_mix, 90)
    run_itc("burst ITC (75th pct)", p1_c, p2_mix, 75)

    # ── Kuramoto ─────────────────────────────────────────────────────────
    p2_kur, _ = inject_eeg_coupling_kuramoto(
        p1_eeg, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)
    p1_c2 = p1_eeg.copy()
    for ch in range(N_CH):
        for s in [p1_c2, p2_kur]:
            s[:, ch] = (s[:, ch] - s[:, ch].mean()) / max(s[:, ch].std(), 1e-8)

    print(f"\n  Kuramoto:")
    run_plv("theta PLV (continuous)", p1_c2, p2_kur)
    run_itc("burst ITC (90th pct)", p1_c2, p2_kur, 90)
    run_itc("burst ITC (75th pct)", p1_c2, p2_kur, 75)

# ── Null ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  NULL (kappa=0)")
print(f"{'='*70}")
run_plv("theta PLV", p1_eeg, p2_eeg)
run_itc("burst ITC (90th)", p1_eeg, p2_eeg, 90)

print(f"\n{'='*70}")
