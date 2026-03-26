"""Test wPLI/PLV temporal localization on EEG at 256 Hz, kappa=0.1.

Uses REAL EEG data at native 256 Hz (14 channels, Emotiv EPOC).
Broadband coupling via mixing model: p2 = kappa*p1_lagged + sqrt(1-kappa²)*p2_indep.
This gives LINEAR kappa scaling and coupling at ALL EEG frequencies (1-45 Hz),
providing much more spectral DOF than the BL test (limited to 0-15 Hz).

Target: hit > 80%, FA < 1% at kappa=0.1
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import generate_coupling_gate, find_valid_window
from cadence.significance.coherence_localization import wpli_temporal_localization

# ── Configuration ──────────────────────────────────────────────────────
FS = 256.0             # EEG native rate
DURATION = 1800        # 30 min
N_CH = 14              # Emotiv EPOC electrodes
KAPPA = 0.1            # Coupling strength
DUTY_CYCLE = 0.10      # 10% episodic
LAG_S = 0.03           # 30ms lag (~8 samples) — physiological for EEG
N_SURROGATES = 100
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load raw EEG sessions ─────────────────────────────────────────────
print("\n=== Loading sessions ===", flush=True)
cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None
             and 'p1_eeg' in s and 'p2_eeg' in s]
print(f"Loaded {len(sess_list)} sessions with raw EEG", flush=True)

s1_name, s1 = sess_list[-1]
s2_name, s2 = sess_list[0]
print(f"P1 from {s1_name}, P2 from {s2_name}", flush=True)

# ── Extract raw EEG at 256 Hz ─────────────────────────────────────────
# Find common valid window
window = find_valid_window(s1, min_duration=DURATION)
t_start, t_end = window

# P1 EEG from session A
p1_ts = s1['p1_eeg_ts']
p1_mask = (p1_ts >= t_start) & (p1_ts < t_end)
p1_eeg = s1['p1_eeg'][p1_mask].copy()  # (N, 14)
p1_ts_win = p1_ts[p1_mask] - t_start

# P2 EEG from session B (cross-dyad → no true coupling at baseline)
p2_ts = s2['p2_eeg_ts']
# Use same time range offset from start of p2's session
p2_start = float(p2_ts[0])
p2_mask = (p2_ts >= p2_start) & (p2_ts < p2_start + DURATION)
p2_eeg = s2['p2_eeg'][p2_mask].copy()
p2_ts_win = p2_ts[p2_mask] - p2_start

# Resample P2 onto P1's time grid if needed
N = min(len(p1_eeg), len(p2_eeg), int(DURATION * FS))
if len(p2_eeg) != N:
    from scipy.interpolate import interp1d
    t_target = np.linspace(0, DURATION, N)
    p2_interp = np.stack([np.interp(t_target, p2_ts_win, p2_eeg[:, ch])
                          for ch in range(N_CH)], axis=1).astype(np.float32)
    p2_eeg = p2_interp
    p1_eeg = p1_eeg[:N]
else:
    p1_eeg = p1_eeg[:N]
    p2_eeg = p2_eeg[:N]

t_eeg = np.arange(N) / FS
print(f"EEG: ({N}, {N_CH}) at {FS} Hz, duration={N/FS:.1f}s", flush=True)

# ── Z-score per channel ───────────────────────────────────────────────
for ch in range(N_CH):
    for sig in [p1_eeg, p2_eeg]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# ── Coupling gate ──────────────────────────────────────────────────────
gate = generate_coupling_gate(N, FS, {
    'duty_cycle': DUTY_CYCLE,
    'event_range_s': (5, 20),
    'ramp_s': 1.0,
}, seed=SEED)
gate_mask = gate > 0.5
print(f"Coupling: kappa={KAPPA}, lag={LAG_S}s ({int(LAG_S*FS)} samples), "
      f"duty={gate_mask.mean():.1%}", flush=True)

# ── Inject broadband coupling (mixing model) ──────────────────────────
print(f"\n=== Injecting broadband coupling (mixing model) ===", flush=True)
lag_samp = max(1, int(LAG_S * FS))
coupled_channels = list(range(N_CH))  # All 14 EEG channels

p1_coupled = p1_eeg.copy()
p2_coupled = p2_eeg.copy()

alpha_t = KAPPA * gate  # (N,) coupling strength timecourse
p1_lagged = np.roll(p1_eeg, lag_samp, axis=0)
p1_lagged[:lag_samp] = 0

for ch in coupled_channels:
    a = alpha_t
    noise_scale = np.sqrt(np.maximum(1 - a ** 2, 0.0))
    p2_coupled[:, ch] = a * p1_lagged[:, ch] + noise_scale * p2_eeg[:, ch]

# Re-normalize
for ch in range(N_CH):
    for sig in [p1_coupled, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Verify coupling injection
from scipy.signal import welch as scipy_welch
ch0 = 0
for label, (s1_test, s2_test) in [("coupled", (p1_coupled, p2_coupled)),
                                    ("null", (p1_eeg, p2_eeg))]:
    # Cross-correlation at lag
    coupled_seg1 = s1_test[gate_mask, ch0]
    coupled_seg2 = s2_test[gate_mask, ch0]
    xcorr = np.corrcoef(coupled_seg1[:len(coupled_seg2)-lag_samp],
                         coupled_seg2[lag_samp:len(coupled_seg1)])[0, 1]
    print(f"  {label} cross-corr at lag={lag_samp} samples: {xcorr:.4f}", flush=True)


def evaluate(mask, z_agg, gate_at_win):
    n_coupled = gate_at_win.sum()
    n_null = (~gate_at_win).sum()
    hit = (mask & gate_at_win).sum() / max(n_coupled, 1)
    fa = (mask & ~gate_at_win).sum() / max(n_null, 1)
    intersection = (mask & gate_at_win).sum()
    union = (mask | gate_at_win).sum()
    iou = intersection / max(union, 1)
    return float(hit), float(fa), float(iou)


# ── Sweep: metric × window × smoothing ────────────────────────────────
# Broadband frequencies: 2-40 Hz (EEG content after 1-45 Hz bandpass)
bb_freqs = np.logspace(np.log10(2.0), np.log10(40.0), 20)
theta_freqs = np.logspace(np.log10(4.0), np.log10(8.0), 5)

# (name, channels, freqs, win_s, n_cycles, metric, smooth_s)
configs = [
    # -- Broadband PLV (captures all phases) --
    ("PLV_bb_2s_14ch",         coupled_channels, bb_freqs, 2.0, 5, "plv", 0),
    ("PLV_bb_5s_14ch",         coupled_channels, bb_freqs, 5.0, 5, "plv", 0),
    ("PLV_bb_5s_14ch_sm10",    coupled_channels, bb_freqs, 5.0, 5, "plv", 10),
    ("PLV_bb_5s_14ch_sm20",    coupled_channels, bb_freqs, 5.0, 5, "plv", 20),
    ("PLV_bb_10s_14ch",        coupled_channels, bb_freqs, 10.0, 5, "plv", 0),
    ("PLV_bb_10s_14ch_sm10",   coupled_channels, bb_freqs, 10.0, 5, "plv", 10),
    # -- Broadband wPLI --
    ("wPLI_bb_5s_14ch",        coupled_channels, bb_freqs, 5.0, 5, "wpli", 0),
    ("wPLI_bb_5s_14ch_sm10",   coupled_channels, bb_freqs, 5.0, 5, "wpli", 10),
    ("wPLI_bb_10s_14ch",       coupled_channels, bb_freqs, 10.0, 5, "wpli", 0),
    ("wPLI_bb_10s_14ch_sm10",  coupled_channels, bb_freqs, 10.0, 5, "wpli", 10),
    # -- Theta only wPLI --
    ("wPLI_th_5s_14ch",        coupled_channels, theta_freqs, 5.0, 5, "wpli", 0),
    ("wPLI_th_5s_14ch_sm10",   coupled_channels, theta_freqs, 5.0, 5, "wpli", 10),
    ("wPLI_th_10s_14ch",       coupled_channels, theta_freqs, 10.0, 5, "wpli", 0),
    ("wPLI_th_10s_14ch_sm10",  coupled_channels, theta_freqs, 10.0, 5, "wpli", 10),
]

print(f"\n{'Config':<30} {'Hit':>6} {'FA':>6} {'IoU':>6} {'z_thr':>6} "
      f"{'z_cpl':>6} {'z_nul':>6} {'Time':>6}")
print("-" * 100)

for name, chs, freqs, win_s, n_cyc, met, sm in configs:
    stride_s = min(win_s / 4, 0.5)

    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1_coupled, p2_coupled, FS,
        channels=chs,
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
    gate_at_win = np.interp(win_times, t_eeg, gate.astype(float)) > 0.5
    hit, fa, iou = evaluate(mask, z_agg, gate_at_win)

    z_c = z_agg[gate_at_win].mean() if gate_at_win.any() else 0
    z_n = z_agg[~gate_at_win].mean() if (~gate_at_win).any() else 0

    print(f"{name:<30} {hit:>5.1%} {fa:>5.1%} {iou:>5.1%} "
          f"{diag['z_threshold']:>6.2f} {z_c:>6.3f} {z_n:>6.3f} "
          f"{elapsed:>5.1f}s", flush=True)

# ── Null test ──────────────────────────────────────────────────────────
print(f"\n=== NULL TEST (kappa=0) ===", flush=True)
best_cfg = ("PLV_bb_5s_14ch_sm10", coupled_channels, bb_freqs, 5.0, 5, "plv", 10)
name, chs, freqs, win_s, n_cyc, met, sm = best_cfg
stride_s = min(win_s / 4, 0.5)
mask_null, z_null, _, diag_null = wpli_temporal_localization(
    p1_eeg, p2_eeg, FS,
    channels=chs, center_freqs=freqs,
    n_surrogates=N_SURROGATES,
    window_s=win_s, stride_s=stride_s,
    n_cycles=n_cyc, target_fa=0.05,
    min_event_s=2.0, metric=met,
    seed=SEED, device=device, smooth_s=sm)
print(f"  Null coupling fraction: {mask_null.mean():.1%}")
print(f"  Null z mean: {z_null.mean():.3f}, max: {z_null.max():.3f}")

print(f"\nTarget: hit>80%, FA<1%")
