"""Simulate X.on 7-channel EEG: F3, F4, C3, Cz, C4, P3, P4.

Uses the same real EEG sessions but selects only the channels that
overlap between EPOC 14-ch and X.on 7-ch layouts. EPOC has F3, F4
but not C3/Cz/C4/P3/P4. So we simulate the X.on layout by using
EPOC's closest electrodes as proxies:

X.on → EPOC proxy:
  F3  → F3  (idx 2)   — exact match
  F4  → F4  (idx 11)  — exact match
  C3  → FC5 (idx 3)   — closest to C3 on EPOC
  Cz  → (avg F3,F4)   — no central electrode, use frontal midline avg
  C4  → FC6 (idx 10)  — closest to C4 on EPOC
  P3  → P7  (idx 5)   — closest parietal-left on EPOC
  P4  → P8  (idx 8)   — closest parietal-right on EPOC

Spatial decay centered at F3, same as before. Tests kappa=0.2 and 0.4.
"""
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
from cadence.constants import EPOC_CHANNEL_NAMES, EPOC_2D_POS

FS = 256.0; DURATION = 1800; N_CH = 14; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# X.on 7-ch layout: approximate 2D positions (10-20 standard)
XON_CHANNELS = {
    'F3':  (-0.39, 0.69),   # same as EPOC F3
    'F4':  (0.39, 0.69),    # same as EPOC F4
    'C3':  (-0.59, 0.00),   # left central (between FC5 and T7)
    'Cz':  (0.00, 0.00),    # vertex midline
    'C4':  (0.59, 0.00),    # right central
    'P3':  (-0.59, -0.59),  # left parietal
    'P4':  (0.59, -0.59),   # right parietal
}

# Map X.on channels to nearest EPOC electrodes
xon_to_epoc = {}
xon_pos = np.array(list(XON_CHANNELS.values()))
xon_names = list(XON_CHANNELS.keys())
for i, (xn, xp) in enumerate(XON_CHANNELS.items()):
    dists = np.sqrt(((EPOC_2D_POS - np.array(xp)) ** 2).sum(axis=1))
    best = np.argmin(dists)
    xon_to_epoc[xn] = (best, EPOC_CHANNEL_NAMES[best], dists[best])

print("X.on → EPOC mapping:")
for xn, (idx, en, d) in xon_to_epoc.items():
    print(f"  {xn:>3} → {en:>4} (idx {idx}, dist={d:.2f})")

xon_epoc_indices = [v[0] for v in xon_to_epoc.values()]
print(f"\nUsing EPOC indices: {xon_epoc_indices} "
      f"({[EPOC_CHANNEL_NAMES[i] for i in xon_epoc_indices]})")

# Load EEG
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

# Avg reference (over all 14 EPOC channels — the injection happens on 14ch)
p1_raw -= p1_raw.mean(axis=1, keepdims=True)
p2_raw -= p2_raw.mean(axis=1, keepdims=True)
for ch in range(N_CH):
    for sig in [p1_raw, p2_raw]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)
gate_mask = gate > 0.5

freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)
shared = dict(
    center_freqs=freqs, n_surrogates=100, n_cycles=[3, 7],
    window_s=20.0, stride_s=0.5, target_fa=0.03,
    min_event_s=5.0, metric='plv', seed=SEED, device=device, smooth_s=15,
)

for KAPPA in [0.4, 0.2, 0.1]:
    print(f"\n{'='*70}")
    print(f"KAPPA = {KAPPA}")
    print(f"{'='*70}")

    # Inject on all 14 EPOC channels (spatial decay from F3)
    p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
        p1_raw, p2_raw, gate, KAPPA, center_ch=2, decay_sigma=0.5, lag_samp=8)
    p1_c = p1_raw.copy()
    for ch in range(N_CH):
        for sig in [p1_c, p2_coupled]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd

    # X.on oracle: which of the 7 proxy channels have coupling > 0.03?
    xon_oracle = [i for i in xon_epoc_indices if kappa_per_ch[i] > 0.03]
    print(f"  X.on oracle ({len(xon_oracle)} ch): "
          f"{[(EPOC_CHANNEL_NAMES[i], f'{kappa_per_ch[i]:.3f}') for i in xon_oracle]}")

    # Also show EPOC 14ch oracle for comparison
    epoc_oracle = [i for i in range(N_CH) if kappa_per_ch[i] > 0.03]
    print(f"  EPOC oracle ({len(epoc_oracle)} ch): "
          f"{[(EPOC_CHANNEL_NAMES[i], f'{kappa_per_ch[i]:.3f}') for i in epoc_oracle]}")

    # Run both: X.on 7ch and EPOC 14ch
    for label, chs in [("X.on 7ch", xon_oracle), ("EPOC 14ch", epoc_oracle)]:
        if len(chs) == 0:
            print(f"  {label}: no coupled channels")
            continue
        mask, z_agg, pcz, diag = wpli_temporal_localization(
            p1_c, p2_coupled, FS, channels=chs, **shared)
        wt = diag['win_times']
        gw = np.interp(wt, t_eeg, gate.astype(float)) > 0.5
        zc = z_agg[gw].mean(); zn = z_agg[~gw].mean()
        win_rate = 1.0 / 0.5
        min_ev = max(1, int(5.0 * win_rate))

        # Find hit at FA~5%
        best_hit = 0.0
        best_thr = 0.0
        for thr in np.arange(0.5, 5.0, 0.1):
            m = _min_event_filter(z_agg > thr, min_ev)
            h = float((m & gw).sum() / max(gw.sum(), 1))
            f = float((m & ~gw).sum() / max((~gw).sum(), 1))
            if 0.04 <= f <= 0.06 and h > best_hit:
                best_hit = h
                best_thr = thr
        print(f"  {label:<12}: z_cpl={zc:.2f} z_null={zn:.3f} "
              f"hit@FA5%={best_hit:.1%} (thr={best_thr:.1f})")
