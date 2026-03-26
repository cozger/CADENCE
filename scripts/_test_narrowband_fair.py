"""Fair narrowband comparison: mixing vs Kuramoto with theta-only PLV.

Fixes the broadband bias: bandpass P1 to theta before mixing injection,
and detect with theta-only PLV (5 freqs, 4-8 Hz).
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
from cadence.constants import EPOC_CHANNEL_NAMES, EPOC_DISTANCE

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


def evaluate(z_agg, gate_win):
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


def run(label, p1, p2, freqs, metric='plv'):
    t0 = time.perf_counter()
    oracle = [i for i in range(N_CH)
              if EPOC_DISTANCE[CENTER_CH][i] < 0.8
              and np.exp(-EPOC_DISTANCE[CENTER_CH][i]**2 / (2*DECAY_SIGMA**2)) > 0.03]
    _, z, _, d = wpli_temporal_localization(
        p1, p2, FS, channels=oracle, center_freqs=freqs,
        n_surrogates=K, n_cycles=[3, 7], window_s=20.0, stride_s=0.5,
        smooth_s=15.0, target_fa=0.03, min_event_s=5.0, metric=metric,
        seed=SEED, device=device)
    gw = np.interp(d['win_times'], t_eeg, gate.astype(float)) > 0.5
    zc = z[gw].mean() if gw.any() else 0
    hit = evaluate(z, gw)
    print(f"  {label:>45}: hit={hit:5.1%}  z_c={zc:+.3f}  ({time.perf_counter()-t0:.1f}s)")
    return hit, zc


# ── Narrowband mixing: bandpass P1 to theta before injection ─────────────
sos_theta = butter(4, [4.0, 8.0], btype='band', fs=FS, output='sos')
p1_theta = np.stack([sosfiltfilt(sos_theta, p1_eeg[:, ch])
                      for ch in range(N_CH)], axis=1).astype(np.float32)

# Frequency grids
freqs_broad = np.logspace(np.log10(2.0), np.log10(40.0), 30)
freqs_theta = np.logspace(np.log10(4.0), np.log10(8.0), 5)

print(f"Device: {device}\n")

for kappa in [0.1, 0.2, 0.4]:
    print(f"\n{'='*60}")
    print(f"  kappa = {kappa}")
    print(f"{'='*60}")

    # Broadband mixing (old) + broadband PLV (old baseline)
    p2_mix_bb, _ = inject_eeg_coupling_spatial(
        p1_eeg, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)
    p1_c = p1_eeg.copy()
    for ch in range(N_CH):
        for s in [p1_c, p2_mix_bb]:
            s[:, ch] = (s[:, ch] - s[:, ch].mean()) / max(s[:, ch].std(), 1e-8)
    run("broadband mixing + broadband PLV (old)", p1_c, p2_mix_bb, freqs_broad)

    # Narrowband mixing (theta P1) + broadband PLV
    p2_mix_nb, _ = inject_eeg_coupling_spatial(
        p1_theta, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)
    p1_c2 = p1_eeg.copy()
    for ch in range(N_CH):
        for s in [p1_c2, p2_mix_nb]:
            s[:, ch] = (s[:, ch] - s[:, ch].mean()) / max(s[:, ch].std(), 1e-8)
    run("narrowband mixing + broadband PLV", p1_c2, p2_mix_nb, freqs_broad)
    run("narrowband mixing + theta PLV", p1_c2, p2_mix_nb, freqs_theta)

    # Kuramoto + broadband PLV (from before)
    p2_kur, _ = inject_eeg_coupling_kuramoto(
        p1_eeg, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)
    p1_c3 = p1_eeg.copy()
    for ch in range(N_CH):
        for s in [p1_c3, p2_kur]:
            s[:, ch] = (s[:, ch] - s[:, ch].mean()) / max(s[:, ch].std(), 1e-8)
    run("kuramoto + broadband PLV", p1_c3, p2_kur, freqs_broad)
    run("kuramoto + theta PLV", p1_c3, p2_kur, freqs_theta)

print(f"\n{'='*60}")
