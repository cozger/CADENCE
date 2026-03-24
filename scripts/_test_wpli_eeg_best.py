"""Run only the best config from previous sweep: PLV_bb_10s_14ch_sm10 with pooled aggregation."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import generate_coupling_gate, find_valid_window
from cadence.significance.coherence_localization import wpli_temporal_localization

FS = 256.0; DURATION = 1800; N_CH = 14; KAPPA = 0.1; LAG_S = 0.03; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=DURATION)
t_s, t_e = window

p1_ts = s1['p1_eeg_ts']; p1_m = (p1_ts >= t_s) & (p1_ts < t_e)
p1_eeg = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']; p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + DURATION)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(DURATION * FS))
t_target = np.linspace(0, DURATION, N)
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
gate_mask = gate > 0.5
print(f"EEG: ({N},{N_CH}) @ {FS}Hz, duty={gate_mask.mean():.1%}", flush=True)

# Inject broadband coupling (mixing model)
lag_samp = max(1, int(LAG_S * FS))
p1_lagged = np.roll(p1_eeg, lag_samp, axis=0)
p1_lagged[:lag_samp] = 0
alpha_t = KAPPA * gate
p2_coupled = p2_eeg.copy()
for ch in range(N_CH):
    a = alpha_t
    p2_coupled[:, ch] = a * p1_lagged[:, ch] + np.sqrt(np.maximum(1 - a**2, 0.0)) * p2_eeg[:, ch]
p1_c = p1_eeg.copy()
for ch in range(N_CH):
    for sig in [p1_c, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

freqs = np.logspace(np.log10(2.0), np.log10(40.0), 20)

# Sweep target_fa and smooth_s to find optimal hit/FA tradeoff
print("\n=== ROC analysis: best config with threshold sweep ===", flush=True)

# Run the two best configs and compute full ROC curve
for label, ws, sm in [("20s_sm15", 20.0, 15), ("15s_sm20", 15.0, 20)]:
    cf = np.logspace(np.log10(2.0), np.log10(40.0), 30)
    stride_s = min(ws / 4, 0.5)
    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1_c, p2_coupled, FS,
        channels=list(range(N_CH)), center_freqs=cf,
        n_surrogates=100, window_s=ws, stride_s=stride_s,
        n_cycles=[3, 7], target_fa=0.05, min_event_s=5.0,
        metric='plv', seed=SEED, device=device, smooth_s=sm)
    elapsed = time.perf_counter() - t0

    wt = diag['win_times']
    gw = np.interp(wt, t_eeg, gate.astype(float)) > 0.5

    print(f"\n{label} (computed in {elapsed:.1f}s):")
    print(f"  z_coupled={z_agg[gw].mean():.3f}, z_null={z_agg[~gw].mean():.3f}")
    print(f"  {'Threshold':>10} {'Hit':>8} {'FA':>8} {'IoU':>8}")
    for thr in np.arange(0.6, 2.4, 0.1):
        m = z_agg > thr
        # Apply min event filter
        from cadence.significance.coherence_localization import _min_event_filter
        m = _min_event_filter(m, max(1, int(5.0 / stride_s)))
        h = (m & gw).sum() / max(gw.sum(), 1)
        f = (m & ~gw).sum() / max((~gw).sum(), 1)
        inter = (m & gw).sum(); union = (m | gw).sum()
        iou = inter / max(union, 1)
        marker = " ◄" if 0.04 <= f <= 0.06 else ""
        print(f"  {thr:>10.2f} {h:>7.1%} {f:>7.1%} {iou:>7.1%}{marker}")

print(f"\nTarget: hit>80%, FA<5%")
