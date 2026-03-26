"""Quick test: ROI-averaged coherence with spatial decay injection."""
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
from cadence.constants import EEG_ROIS, EPOC_CHANNEL_NAMES

FS = 256.0; DURATION = 1800; N_CH = 14; KAPPA = 0.1; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=DURATION)

p1_ts = s1['p1_eeg_ts']; p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']; p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + DURATION)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(DURATION * FS))
t_target = np.linspace(0, DURATION, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(N_CH)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]; t_eeg = np.arange(N) / FS

for ch in range(N_CH):
    for sig in [p1_eeg, p2_eeg]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)

p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
    p1_eeg, p2_eeg, gate, KAPPA, center_ch=2, decay_sigma=0.5, lag_samp=8)

p1_c = p1_eeg.copy()
for ch in range(N_CH):
    for sig in [p1_c, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

print(f"Coupling: {[(EPOC_CHANNEL_NAMES[i], f'{kappa_per_ch[i]:.3f}') for i in range(N_CH) if kappa_per_ch[i]>0.01]}")
print(f"EEG_ROIS: {EEG_ROIS}\n")

# Show what ROI averaging does to the coupling signal
for roi, chs in EEG_ROIS.items():
    coupled_in_roi = [i for i in chs if kappa_per_ch[i] > 0.01]
    avg_kappa = np.mean([kappa_per_ch[i] for i in chs])
    print(f"  {roi:>12}: ch={chs}, coupled={coupled_in_roi}, "
          f"avg_kappa={avg_kappa:.4f}, signal_retention={len(coupled_in_roi)/len(chs):.0%}")

freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)
shared = dict(center_freqs=freqs, n_surrogates=100, n_cycles=[3, 7],
              target_fa=0.03, min_event_s=5.0, metric='plv', seed=SEED, device=device)

def roc(label, extra):
    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1_c, p2_coupled, FS, **shared, **extra)
    el = time.perf_counter() - t0
    wt = diag['win_times']
    gw = np.interp(wt, t_eeg, gate.astype(float)) > 0.5
    zc = z_agg[gw].mean(); zn = z_agg[~gw].mean()
    stride = extra.get('stride_s', 0.5)
    print(f"\n{label} ({el:.1f}s)  z_cpl={zc:.3f} z_null={zn:.3f}")
    print(f"  {'Thr':>6} {'Hit':>7} {'FA':>7}")
    for thr in np.arange(0.4, 2.2, 0.1):
        m = _min_event_filter(z_agg > thr, max(1, int(5.0 / stride)))
        h = (m & gw).sum() / max(gw.sum(), 1)
        f = (m & ~gw).sum() / max((~gw).sum(), 1)
        mk = " <--" if 0.04 <= f <= 0.06 else ""
        print(f"  {thr:>6.2f} {h:>6.1%} {f:>6.1%}{mk}")

# Standard 4-ROI (frontal, left_temp, right_temp, posterior)
roc("4-ROI 20s sm15", dict(window_s=20.0, stride_s=0.5, smooth_s=15, roi_map=EEG_ROIS))
roc("4-ROI 20s sm20", dict(window_s=20.0, stride_s=0.5, smooth_s=20, roi_map=EEG_ROIS))
roc("4-ROI 15s sm20", dict(window_s=15.0, stride_s=0.5, smooth_s=20, roi_map=EEG_ROIS))

# Oracle 5ch for comparison
oracle = [i for i in np.argsort(-kappa_per_ch) if kappa_per_ch[i] > 0.03]
roc("Oracle 5ch 20s sm15", dict(channels=oracle, window_s=20.0, stride_s=0.5, smooth_s=15))
