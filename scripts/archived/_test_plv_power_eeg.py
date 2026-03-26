"""Test PLV vs power events vs combined, with per-band breakdown."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0; DURATION = 1800; N_CH = 14; KAPPA = 0.1; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Load and prepare EEG (avg-ref)
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=DURATION)

p1_ts = s1['p1_eeg_ts']; p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_raw = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']; p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + DURATION)
p2_raw = s2['p2_eeg'][p2_m].copy()
N = min(len(p1_raw), len(p2_raw), int(DURATION * FS))
t_target = np.linspace(0, DURATION, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_raw = np.stack([np.interp(t_target, p2_tw, p2_raw[:, c])
                    for c in range(N_CH)], axis=1).astype(np.float32)
p1_raw = p1_raw[:N]; t_eeg = np.arange(N) / FS

# Avg reference + z-score
p1_raw -= p1_raw.mean(axis=1, keepdims=True)
p2_raw -= p2_raw.mean(axis=1, keepdims=True)
for ch in range(N_CH):
    for sig in [p1_raw, p2_raw]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Inject coupling
gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)
gate_mask = gate > 0.5
p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
    p1_raw, p2_raw, gate, KAPPA, center_ch=2, decay_sigma=0.5, lag_samp=8)
p1_c = p1_raw.copy()
for ch in range(N_CH):
    for sig in [p1_c, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd
oracle_ch = [i for i in np.argsort(-kappa_per_ch) if kappa_per_ch[i] > 0.03]
print(f"Oracle: {[EPOC_CHANNEL_NAMES[i] for i in oracle_ch]}\n")

# Band definitions
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'broadband': (2, 40),
}

shared = dict(
    channels=oracle_ch, n_surrogates=100, n_cycles=[3, 7],
    window_s=20.0, stride_s=0.5, target_fa=0.03,
    min_event_s=5.0, seed=SEED, device=device, smooth_s=15,
)

# Run each metric × band combination
print(f"{'Metric':<14} {'Band':<11} {'z_cpl':>6} {'z_nul':>6} {'Hit@5%':>7} {'Time':>5}")
print("-" * 60)

for met in ['plv', 'power_event', 'plv_power']:
    for band_name, (flo, fhi) in BANDS.items():
        n_f = 5 if band_name != 'broadband' else 20
        freqs = np.logspace(np.log10(flo), np.log10(fhi), n_f)
        t0 = time.perf_counter()
        mask, z_agg, pcz, diag = wpli_temporal_localization(
            p1_c, p2_coupled, FS, center_freqs=freqs, metric=met, **shared)
        elapsed = time.perf_counter() - t0

        wt = diag['win_times']
        gw = np.interp(wt, t_eeg, gate.astype(float)) > 0.5
        zc = z_agg[gw].mean(); zn = z_agg[~gw].mean()

        # Find hit at FA~5%
        win_rate = 1.0 / 0.5
        best_hit = 0.0
        for thr in np.arange(0.5, 3.0, 0.1):
            m = _min_event_filter(z_agg > thr, max(1, int(5.0 * win_rate)))
            h = (m & gw).sum() / max(gw.sum(), 1)
            f = (m & ~gw).sum() / max((~gw).sum(), 1)
            if 0.04 <= f <= 0.06:
                best_hit = max(best_hit, float(h))

        print(f"{met:<14} {band_name:<11} {zc:>6.3f} {zn:>6.3f} {best_hit:>6.1%} {elapsed:>4.0f}s",
              flush=True)

# Null test
print(f"\nNull test (combined, broadband):")
freqs_bb = np.logspace(np.log10(2), np.log10(40), 20)
mask_n, z_n, _, _ = wpli_temporal_localization(
    p1_raw, p2_raw, FS, center_freqs=freqs_bb, metric='plv_power', **shared)
print(f"  Coupling: {mask_n.mean():.1%}, z_mean={z_n.mean():.3f}")
