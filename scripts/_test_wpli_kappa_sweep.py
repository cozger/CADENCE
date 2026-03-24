"""Sweep wPLI parameters at kappa=0.1 to find detectable configuration.

Tests two injection models:
  1. Additive theta: p1 += kappa*sin, p2 += kappa*sin_shifted (kappa² scaling)
  2. Mixing model: p2 = kappa*p1_lagged + sqrt(1-kappa²)*p2_indep (kappa¹ scaling)

And multiple parameter combos: window size, freq range, n_channels, n_cycles.
"""
import sys, os, time
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (find_valid_window, build_semisynthetic_base,
                                generate_coupling_gate)
from cadence.significance.coherence_localization import wpli_temporal_localization

# ── Configuration ──────────────────────────────────────────────────────
FS_NATIVE = 30.0
FS_TARGET = 256.0
DURATION = 1800
N_CH = 15
KAPPA = 0.1
DUTY_CYCLE = 0.10
SEED = 42
N_SURROGATES = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load sessions & build base ─────────────────────────────────────────
print("Loading sessions...", flush=True)
cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None]
s1 = sess_list[-1][1]
s2 = sess_list[0][1]
window = find_valid_window(s1, min_duration=DURATION)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

target_mod = 'blendshapes_v2'
p1_bl_30 = base[f'p1_{target_mod}'][:, :N_CH].copy()
p2_bl_30 = base[f'p2_{target_mod}'][:, :N_CH].copy()
N_30 = min(len(p1_bl_30), len(p2_bl_30), int(DURATION * FS_NATIVE))
p1_bl_30 = p1_bl_30[:N_30]
p2_bl_30 = p2_bl_30[:N_30]

# Z-score
for ch in range(N_CH):
    for sig in [p1_bl_30, p2_bl_30]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Upsample to 256 Hz
from math import gcd
g = gcd(int(FS_TARGET), int(FS_NATIVE))
up, down = int(FS_TARGET) // g, int(FS_NATIVE) // g
p1_256 = resample_poly(p1_bl_30, up, down, axis=0).astype(np.float32)
p2_256 = resample_poly(p2_bl_30, up, down, axis=0).astype(np.float32)
N_256 = min(len(p1_256), len(p2_256), int(DURATION * FS_TARGET))
p1_256 = p1_256[:N_256]
p2_256 = p2_256[:N_256]
t_256 = np.arange(N_256) / FS_TARGET

# Coupling gate
gate_30 = generate_coupling_gate(N_30, FS_NATIVE, {
    'duty_cycle': DUTY_CYCLE,
    'event_range_s': (5, 20),
    'ramp_s': 1.0,
}, seed=SEED)
gate_256 = np.interp(t_256, np.arange(N_30) / FS_NATIVE, gate_30).astype(np.float32)
gate_mask = gate_256 > 0.5
print(f"Base: {p1_256.shape} @ {FS_TARGET}Hz, duty={gate_mask.mean():.1%}\n",
      flush=True)


def inject_additive_theta(p1, p2, gate, coupled_ch, kappa, fs, seed=42):
    """Additive theta: both P1 and P2 get kappa*sin(6Hz+phase). Scales as kappa²."""
    p1_out, p2_out = p1.copy(), p2.copy()
    rng = np.random.RandomState(seed + 100)
    t = np.arange(len(p1)) / fs
    for ch in coupled_ch:
        phi = rng.uniform(0, 2 * np.pi)
        theta_p1 = kappa * np.sin(2 * np.pi * 6.0 * t + phi)
        theta_p2 = kappa * np.sin(2 * np.pi * 6.0 * t + phi + np.pi / 4)
        p1_out[:, ch] += gate * theta_p1
        p2_out[:, ch] += gate * theta_p2
    # Re-normalize
    for ch in range(p1.shape[1]):
        for sig in [p1_out, p2_out]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd
    return p1_out, p2_out


def inject_mixing_model(p1, p2, gate, coupled_ch, kappa, fs, lag_s=0.1,
                         seed=42):
    """Mixing model: p2 = kappa*p1_lagged + sqrt(1-kappa²)*p2_indep. Scales as kappa¹.

    Uses short lag (0.1s) so phase shift at theta freqs is non-zero for wPLI.
    """
    p1_out, p2_out = p1.copy(), p2.copy()
    lag_samp = max(1, int(lag_s * fs))
    alpha_t = kappa * gate
    p1_lagged = np.roll(p1, lag_samp, axis=0)
    p1_lagged[:lag_samp] = 0

    for ch in coupled_ch:
        a = alpha_t
        noise_scale = np.sqrt(np.maximum(1 - a ** 2, 0.0))
        p2_out[:, ch] = a * p1_lagged[:, ch] + noise_scale * p2[:, ch]

    # Re-normalize
    for ch in range(p1.shape[1]):
        for sig in [p1_out, p2_out]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd
    return p1_out, p2_out


def evaluate(mask, z_agg, gate_at_win):
    """Compute hit, FA, IoU."""
    n_coupled = gate_at_win.sum()
    n_null = (~gate_at_win).sum()
    hits = (mask & gate_at_win).sum()
    fas = (mask & ~gate_at_win).sum()
    hit_rate = hits / max(n_coupled, 1)
    fa_rate = fas / max(n_null, 1)
    intersection = (mask & gate_at_win).sum()
    union = (mask | gate_at_win).sum()
    iou = intersection / max(union, 1)
    return float(hit_rate), float(fa_rate), float(iou)


# ── Sweep configs ──────────────────────────────────────────────────────
# Focus on CSD metric (|<Sxy>|) which captures all coupling phases,
# vs wPLI which loses signal at phases near 0 or pi.
# Focus on the three key tuning dimensions:
# 1. Lag (0.07s avoids blind frequencies at 5/10 Hz for wPLI)
# 2. Temporal smoothing of z_agg (accumulates evidence across windows)
# 3. More channels + wider band
#
# name, injection, win_s, n_coupled, flo, fhi, n_freqs, n_cyc, lag_s, metric, smooth_s
configs = [
    # -- Round 1 baseline: lag=0.07 vs 0.1 --
    ("wPLI_lag01_10s_10ch",    "mixing", 10.0, 10, 1, 14, 15, 5, 0.10, "wpli", 0),
    ("wPLI_lag007_10s_10ch",   "mixing", 10.0, 10, 1, 14, 15, 5, 0.07, "wpli", 0),
    ("PLV_lag007_10s_10ch",    "mixing", 10.0, 10, 1, 14, 15, 5, 0.07, "plv",  0),
    # -- Round 2: temporal smoothing --
    ("wPLI_lag007_5s_15ch_sm10",  "mixing", 5.0, 15, 1, 14, 15, 5, 0.07, "wpli", 10),
    ("wPLI_lag007_10s_10ch_sm10", "mixing", 10.0, 10, 1, 14, 15, 5, 0.07, "wpli", 10),
    ("wPLI_lag007_10s_15ch_sm10", "mixing", 10.0, 15, 1, 14, 15, 5, 0.07, "wpli", 10),
    ("PLV_lag007_10s_10ch_sm10",  "mixing", 10.0, 10, 1, 14, 15, 5, 0.07, "plv",  10),
    ("PLV_lag007_10s_15ch_sm10",  "mixing", 10.0, 15, 1, 14, 15, 5, 0.07, "plv",  10),
    # -- Round 3: aggressive smoothing --
    ("wPLI_lag007_10s_15ch_sm20", "mixing", 10.0, 15, 1, 14, 15, 5, 0.07, "wpli", 20),
    ("PLV_lag007_10s_15ch_sm20",  "mixing", 10.0, 15, 1, 14, 15, 5, 0.07, "plv",  20),
    ("wPLI_lag007_5s_15ch_sm30",  "mixing", 5.0,  15, 1, 14, 15, 5, 0.07, "wpli", 30),
    ("PLV_lag007_5s_15ch_sm30",   "mixing", 5.0,  15, 1, 14, 15, 5, 0.07, "plv",  30),
    # -- Round 4: more freqs --
    ("wPLI_lag007_10s_15ch_20f_sm20", "mixing", 10.0, 15, 1, 14, 20, 5, 0.07, "wpli", 20),
    ("PLV_lag007_10s_15ch_20f_sm20",  "mixing", 10.0, 15, 1, 14, 20, 5, 0.07, "plv",  20),
]

print(f"{'Config':<36} {'Hit':>6} {'FA':>6} {'IoU':>6} {'z_thr':>6} "
      f"{'z_cpl':>6} {'z_nul':>6} {'Time':>6}")
print("-" * 110)

for name, inj_type, win_s, n_coupled, flo, fhi, n_freqs, n_cyc, lag_s, met, sm in configs:
    coupled_ch = list(range(n_coupled))
    stride_s = min(win_s / 4, 0.5)

    p1_test, p2_test = inject_mixing_model(
        p1_256, p2_256, gate_256, coupled_ch, KAPPA, FS_TARGET,
        lag_s=lag_s, seed=SEED)

    freqs = np.logspace(np.log10(flo), np.log10(fhi), n_freqs)

    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1_test, p2_test, FS_TARGET,
        channels=coupled_ch,
        center_freqs=freqs,
        n_surrogates=N_SURROGATES,
        window_s=win_s,
        stride_s=stride_s,
        n_cycles=n_cyc,
        target_fa=0.05,
        min_event_s=2.0,
        metric=met,
        seed=SEED,
        device=device,
        smooth_s=sm)
    elapsed = time.perf_counter() - t0

    win_times = diag['win_times']
    gate_at_win = np.interp(win_times, t_256, gate_256) > 0.5
    hit, fa, iou = evaluate(mask, z_agg, gate_at_win)

    z_coupled = z_agg[gate_at_win].mean() if gate_at_win.any() else 0
    z_null = z_agg[~gate_at_win].mean() if (~gate_at_win).any() else 0

    print(f"{name:<36} {hit:>5.1%} {fa:>5.1%} {iou:>5.1%} "
          f"{diag['z_threshold']:>6.2f} {z_coupled:>6.3f} {z_null:>6.3f} "
          f"{elapsed:>5.1f}s", flush=True)

print(f"\nTarget: hit>80%, FA<1%")
