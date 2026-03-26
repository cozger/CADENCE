"""Test wPLI temporal localization on semisynthetic BL signals at 256 Hz.

Hypothesis: At 256 Hz with theta-band coupling, per-window spectral DOF
is high enough for wPLI to detect kappa=0.4 episodic coupling.

Setup:
  - Load real BL sessions, build semisynthetic base (cross-dyad)
  - Upsample 30 Hz BL PCA signals to 256 Hz
  - Inject theta-band (6 Hz) oscillatory coupling with pi/4 phase lag
  - 1800s session, 10% duty cycle, kappa=0.4, 5/15 channels coupled
  - Run wPLI temporal localization with K=100 surrogates

Target: hit > 75%, FA < 15%
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
FS_NATIVE = 30.0       # BL native rate
FS_TARGET = 256.0      # Upsampled rate for wPLI
DURATION = 1800        # 30 min session
N_CH = 15              # BL PCA channels
N_COUPLED = 5          # Oracle coupled channels
KAPPA = 0.4            # Coupling strength
DUTY_CYCLE = 0.10      # 10% episodic coupling
THETA_FREQ = 6.0       # Hz — injected coupling frequency
PHASE_LAG = np.pi / 4  # Non-zero for wPLI detection
N_SURROGATES = 100     # Circular-shift surrogates
WINDOW_S = 2.0         # wPLI window (seconds)
STRIDE_S = 0.5         # wPLI stride (seconds)
N_CYCLES = 5           # Morlet cycles
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load real BL sessions ─────────────────────────────────────────────
print("\n=== Loading sessions ===", flush=True)
cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None]
print(f"Loaded {len(sess_list)} sessions", flush=True)

s1_name, s1 = sess_list[-1]
s2_name, s2 = sess_list[0]
print(f"Using P1 from {s1_name}, P2 from {s2_name}", flush=True)

# Build semisynthetic base (cross-dyad, kappa=0)
window = find_valid_window(s1, min_duration=DURATION)
print(f"Valid window: {window[0]:.1f}s - {window[1]:.1f}s "
      f"(duration={window[1]-window[0]:.1f}s)", flush=True)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

# ── Extract BL signals at 30 Hz ───────────────────────────────────────
print("\n=== Preparing signals ===", flush=True)
target_mod = 'blendshapes_v2'
p1_bl_30 = base[f'p1_{target_mod}'][:, :N_CH].copy()  # (T_30, 15)
p2_bl_30 = base[f'p2_{target_mod}'][:, :N_CH].copy()
p1_ts_30 = base[f'p1_{target_mod}_ts']
N_30 = min(len(p1_bl_30), len(p2_bl_30), int(DURATION * FS_NATIVE))
p1_bl_30 = p1_bl_30[:N_30]
p2_bl_30 = p2_bl_30[:N_30]
print(f"BL signals: {p1_bl_30.shape} at {FS_NATIVE} Hz", flush=True)

# ── Z-score per channel ───────────────────────────────────────────────
for ch in range(N_CH):
    for sig in [p1_bl_30, p2_bl_30]:
        mu = sig[:, ch].mean()
        sd = max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# ── Upsample 30 Hz → 256 Hz ──────────────────────────────────────────
# resample_poly(x, up, down) with up=256, down=30 → gcd-reduced
from math import gcd
g = gcd(int(FS_TARGET), int(FS_NATIVE))
up = int(FS_TARGET) // g
down = int(FS_NATIVE) // g
print(f"Resampling: up={up}, down={down} ({FS_NATIVE} → {FS_TARGET} Hz)", flush=True)

t0 = time.perf_counter()
p1_256 = resample_poly(p1_bl_30, up, down, axis=0).astype(np.float32)
p2_256 = resample_poly(p2_bl_30, up, down, axis=0).astype(np.float32)
N_256 = min(len(p1_256), len(p2_256), int(DURATION * FS_TARGET))
p1_256 = p1_256[:N_256]
p2_256 = p2_256[:N_256]
print(f"Upsampled: {p1_256.shape} at {FS_TARGET} Hz "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# ── Generate coupling gate ─────────────────────────────────────────────
gate_30 = generate_coupling_gate(N_30, FS_NATIVE, {
    'duty_cycle': DUTY_CYCLE,
    'event_range_s': (5, 20),
    'ramp_s': 1.0,
}, seed=SEED)

# Upsample gate to 256 Hz
t_30 = np.arange(N_30) / FS_NATIVE
t_256 = np.arange(N_256) / FS_TARGET
gate_256 = np.interp(t_256, t_30, gate_30).astype(np.float32)
gate_mask = gate_256 > 0.5
duty_actual = gate_mask.mean()
print(f"Coupling gate: duty={duty_actual:.1%}, "
      f"coupled channels={list(range(N_COUPLED))}", flush=True)

# ── Inject theta-band oscillatory coupling ─────────────────────────────
print(f"\n=== Injecting theta coupling ===", flush=True)
print(f"  freq={THETA_FREQ} Hz, phase_lag={PHASE_LAG:.3f} rad "
      f"({np.degrees(PHASE_LAG):.1f}°), kappa={KAPPA}", flush=True)

rng = np.random.RandomState(SEED + 100)
coupled_channels = list(range(N_COUPLED))

for ch in coupled_channels:
    # Random starting phase per channel (different for each channel)
    phi = rng.uniform(0, 2 * np.pi)

    # Theta oscillation with known phase lag
    theta_p1 = KAPPA * np.sin(2 * np.pi * THETA_FREQ * t_256 + phi)
    theta_p2 = KAPPA * np.sin(2 * np.pi * THETA_FREQ * t_256 + phi + PHASE_LAG)

    # Inject only during coupling windows (gated)
    p1_256[:, ch] += gate_256 * theta_p1
    p2_256[:, ch] += gate_256 * theta_p2

# Re-normalize after injection
for ch in range(N_CH):
    for sig in [p1_256, p2_256]:
        mu = sig[:, ch].mean()
        sd = max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Verify: check theta-band power increase during coupling
from scipy.signal import welch as scipy_welch
for label, sig in [("P1", p1_256), ("P2", p2_256)]:
    ch0 = 0  # First coupled channel
    coupled_seg = sig[gate_mask, ch0]
    null_seg = sig[~gate_mask, ch0]
    if len(coupled_seg) > 512 and len(null_seg) > 512:
        f_c, psd_c = scipy_welch(coupled_seg, fs=FS_TARGET, nperseg=512)
        f_n, psd_n = scipy_welch(null_seg, fs=FS_TARGET, nperseg=512)
        theta_mask = (f_c >= 4) & (f_c <= 8)
        ratio = psd_c[theta_mask].mean() / max(psd_n[theta_mask].mean(), 1e-20)
        print(f"  {label} ch0 theta power ratio (coupled/null): {ratio:.2f}x",
              flush=True)

# ── Run wPLI temporal localization ─────────────────────────────────────
print(f"\n=== Running wPLI temporal localization ===", flush=True)
print(f"  channels={coupled_channels} (oracle), surrogates={N_SURROGATES}",
      flush=True)
print(f"  window={WINDOW_S}s, stride={STRIDE_S}s, n_cycles={N_CYCLES}",
      flush=True)

# Theta band center frequencies
theta_freqs = np.logspace(np.log10(4.0), np.log10(8.0), 5)
print(f"  center_freqs={[f'{f:.1f}' for f in theta_freqs]} Hz", flush=True)

t0 = time.perf_counter()
mask, z_agg, per_channel_z, diag = wpli_temporal_localization(
    p1_256, p2_256, FS_TARGET,
    channels=coupled_channels,
    center_freqs=theta_freqs,
    n_surrogates=N_SURROGATES,
    window_s=WINDOW_S,
    stride_s=STRIDE_S,
    n_cycles=N_CYCLES,
    target_fa=0.05,
    min_event_s=2.0,
    seed=SEED,
    device=device)
elapsed = time.perf_counter() - t0
print(f"  Completed in {elapsed:.1f}s", flush=True)

# ── Evaluate: hit/FA/IoU ──────────────────────────────────────────────
print(f"\n=== Results ===", flush=True)

# Interpolate gate to window grid
win_times = diag['win_times']
gate_at_windows = np.interp(win_times, t_256, gate_256) > 0.5

n_win = len(mask)
n_coupled_win = gate_at_windows.sum()
n_null_win = (~gate_at_windows).sum()

hits = (mask & gate_at_windows).sum()
fas = (mask & ~gate_at_windows).sum()

hit_rate = hits / max(n_coupled_win, 1)
fa_rate = fas / max(n_null_win, 1)

intersection = (mask & gate_at_windows).sum()
union = (mask | gate_at_windows).sum()
iou = intersection / max(union, 1)

coupling_frac = mask.mean()

print(f"  Windows: {n_win} total, {n_coupled_win} coupled, {n_null_win} null")
print(f"  Detection: {coupling_frac:.1%} flagged as coupled")
print(f"  Hit rate:  {hit_rate:.1%} (target >75%)")
print(f"  FA rate:   {fa_rate:.1%} (target <15%)")
print(f"  IoU:       {iou:.1%}")
print(f"  z_threshold: {diag['z_threshold']:.2f}")
print(f"  z_agg mean: {diag['z_agg_mean']:.3f}, "
      f"max: {diag['z_agg_max']:.3f}, p95: {diag['z_agg_p95']:.3f}")

# Per-channel diagnostics
print(f"\n  Per-channel z-score (mean during coupled / null):")
for ci, ch in enumerate(coupled_channels):
    z_coupled = per_channel_z[ci, gate_at_windows].mean()
    z_null = per_channel_z[ci, ~gate_at_windows].mean()
    print(f"    ch{ch}: coupled={z_coupled:.3f}, null={z_null:.3f}, "
          f"diff={z_coupled - z_null:.3f}")

# z_agg during coupled vs null
z_coupled_agg = z_agg[gate_at_windows].mean()
z_null_agg = z_agg[~gate_at_windows].mean()
print(f"\n  z_agg: coupled={z_coupled_agg:.3f}, null={z_null_agg:.3f}, "
      f"diff={z_coupled_agg - z_null_agg:.3f}")

# ── Also test with all 15 channels (including 10 null) ────────────────
print(f"\n=== Re-running with ALL {N_CH} channels (diluted) ===", flush=True)
t0 = time.perf_counter()
mask_all, z_agg_all, pcz_all, diag_all = wpli_temporal_localization(
    p1_256, p2_256, FS_TARGET,
    channels=list(range(N_CH)),
    center_freqs=theta_freqs,
    n_surrogates=N_SURROGATES,
    window_s=WINDOW_S,
    stride_s=STRIDE_S,
    n_cycles=N_CYCLES,
    target_fa=0.05,
    min_event_s=2.0,
    seed=SEED,
    device=device)
elapsed_all = time.perf_counter() - t0
print(f"  Completed in {elapsed_all:.1f}s", flush=True)

hits_all = (mask_all & gate_at_windows[:len(mask_all)]).sum()
fas_all = (mask_all & ~gate_at_windows[:len(mask_all)]).sum()
n_win_all = len(mask_all)
gate_all = gate_at_windows[:n_win_all]
hit_all = hits_all / max(gate_all.sum(), 1)
fa_all = fas_all / max((~gate_all).sum(), 1)
print(f"  Hit rate:  {hit_all:.1%}")
print(f"  FA rate:   {fa_all:.1%}")

# Summary
print(f"\n{'='*60}")
PASS = hit_rate >= 0.75 and fa_rate <= 0.15
print(f"ORACLE RESULT: hit={hit_rate:.1%} FA={fa_rate:.1%} "
      f"{'PASS' if PASS else 'FAIL'}")
print(f"ALL-CH RESULT: hit={hit_all:.1%} FA={fa_all:.1%}")
print(f"{'='*60}")

# ── Null test (kappa=0): verify FA is controlled ──────────────────────
print(f"\n=== NULL TEST (kappa=0, no coupling) ===", flush=True)
# Use the un-coupled signals (pre-injection, re-load base)
p1_null = resample_poly(base[f'p1_{target_mod}'][:N_30, :N_CH].copy(),
                         up, down, axis=0).astype(np.float32)[:N_256]
p2_null = resample_poly(base[f'p2_{target_mod}'][:N_30, :N_CH].copy(),
                         up, down, axis=0).astype(np.float32)[:N_256]
for ch in range(N_CH):
    for sig in [p1_null, p2_null]:
        mu = sig[:, ch].mean()
        sd = max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

t0 = time.perf_counter()
mask_null, z_null, _, diag_null = wpli_temporal_localization(
    p1_null, p2_null, FS_TARGET,
    channels=coupled_channels,
    center_freqs=theta_freqs,
    n_surrogates=N_SURROGATES,
    window_s=WINDOW_S,
    stride_s=STRIDE_S,
    n_cycles=N_CYCLES,
    target_fa=0.05,
    min_event_s=2.0,
    seed=SEED,
    device=device)
null_frac = mask_null.mean()
print(f"  Null coupling fraction: {null_frac:.1%} "
      f"(expect ~5% or less, {time.perf_counter()-t0:.1f}s)")
print(f"  Null z_agg mean: {z_null.mean():.3f}, max: {z_null.max():.3f}")
print(f"  Null threshold: {diag_null['z_threshold']:.2f}")
