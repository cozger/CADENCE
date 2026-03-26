"""Test multi-lag bank cross-correlation on spatial-decay EEG, kappa=0.1.

Multi-lag bank: cross-products at ALL lags → smooth → max over lags.
Surrogate calibration handles the max-over-lags penalty automatically.
No lag estimation needed.
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
    xcorr_temporal_localization, _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0; DURATION = 1800; N_CH = 14; KAPPA = 0.1; SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

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

# Avg reference + z-score
p1_raw -= p1_raw.mean(axis=1, keepdims=True)
p2_raw -= p2_raw.mean(axis=1, keepdims=True)
for ch in range(N_CH):
    for sig in [p1_raw, p2_raw]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Coupling: spatial decay from F3, lag=8 samples (30ms)
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
print(f"Oracle: {[EPOC_CHANNEL_NAMES[i] for i in oracle_ch]}")
print(f"Coupling lag=8 samp (30ms), duty={gate_mask.mean():.1%}\n")

# ── Multi-lag bank sweep ───────────────────────────────────────────────
# Output is at 4 Hz (decimated from 256 Hz)
OUT_RATE = FS / max(1, int(FS / 4))  # 4 Hz

print(f"{'Config':<35} {'Hit':>6} {'FA':>6} {'z_cpl':>6} {'z_nul':>6} "
      f"{'z_thr':>6} {'Time':>5}")
print("-" * 85)

configs = [
    # (name, max_lag_s, lag_step_s, smooth_s, channels)
    ("oracle_lag100ms_sm1",     0.1,  None, 1.0, oracle_ch),
    ("oracle_lag100ms_sm2",     0.1,  None, 2.0, oracle_ch),
    ("oracle_lag100ms_sm3",     0.1,  None, 3.0, oracle_ch),
    ("oracle_lag500ms_sm2",     0.5,  0.005, 2.0, oracle_ch),
    ("oracle_lag500ms_sm3",     0.5,  0.005, 3.0, oracle_ch),
    ("oracle_lag1s_sm3",        1.0,  0.01, 3.0, oracle_ch),
    ("all14_lag100ms_sm2",      0.1,  None, 2.0, list(range(N_CH))),
    ("all14_lag500ms_sm3",      0.5,  0.005, 3.0, list(range(N_CH))),
]

for name, ml, ls, sm, chs in configs:
    t0 = time.perf_counter()
    mask, z_agg, best_lag, diag = xcorr_temporal_localization(
        p1_c, p2_coupled, FS,
        channels=chs, max_lag_s=ml, lag_step_s=ls,
        smooth_s=sm, n_surrogates=100,
        target_fa=0.05, min_event_s=5.0,
        seed=SEED, device=device)
    elapsed = time.perf_counter() - t0

    # Evaluate at output rate (4 Hz)
    gate_out = gate_mask[::int(FS / 4)][:len(mask)]
    nc = gate_out.sum(); nn = (~gate_out).sum()
    hit = float((mask & gate_out).sum() / max(nc, 1))
    fa = float((mask & ~gate_out).sum() / max(nn, 1))
    zc = z_agg[gate_out[:len(z_agg)]].mean()
    zn = z_agg[~gate_out[:len(z_agg)]].mean()

    print(f"{name:<35} {hit:>5.1%} {fa:>5.1%} {zc:>6.3f} {zn:>6.3f} "
          f"{diag['z_threshold']:>6.2f} {elapsed:>4.0f}s", flush=True)

# ── ROC for best config ────────────────────────────────────────────────
print(f"\nROC: oracle, lag=500ms, smooth=3s")
mask, z_agg, best_lag, diag = xcorr_temporal_localization(
    p1_c, p2_coupled, FS,
    channels=oracle_ch, max_lag_s=0.5, lag_step_s=0.005,
    smooth_s=3.0, n_surrogates=100,
    target_fa=0.05, min_event_s=5.0,
    seed=SEED, device=device)

gate_out = gate_mask[::int(FS / 4)][:len(z_agg)]
print(f"  z_cpl={z_agg[gate_out].mean():.3f}, z_null={z_agg[~gate_out].mean():.3f}")
print(f"  {'Thr':>6} {'Hit':>7} {'FA':>7}")
for thr in np.arange(0.5, 3.5, 0.2):
    m = _min_event_filter(z_agg > thr, max(1, int(5.0 * OUT_RATE)))
    h = (m & gate_out).sum() / max(gate_out.sum(), 1)
    f = (m & ~gate_out).sum() / max((~gate_out).sum(), 1)
    mk = " <--" if 0.04 <= f <= 0.06 else ""
    print(f"  {thr:>6.2f} {float(h):>6.1%} {float(f):>6.1%}{mk}")

# ── Null test ──────────────────────────────────────────────────────────
print(f"\nNull test (kappa=0):")
mask_n, z_n, _, dn = xcorr_temporal_localization(
    p1_raw, p2_raw, FS,
    channels=oracle_ch, max_lag_s=0.5, lag_step_s=0.005,
    smooth_s=3.0, n_surrogates=100,
    target_fa=0.05, min_event_s=5.0,
    seed=SEED, device=device)
print(f"  Coupling: {mask_n.mean():.1%}, z_mean={z_n.mean():.3f}")
