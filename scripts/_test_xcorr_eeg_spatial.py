"""Test cross-correlation temporal localization on spatial-decay EEG, kappa=0.1.

Cross-product + smooth + threshold — the BL agent proved this smashes
through the HMM/spectral bottleneck. Testing on EEG with spatial decay.
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

# ── Load EEG ───────────────────────────────────────────────────────────
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

# ── Coupling ───────────────────────────────────────────────────────────
gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)
gate_mask = gate > 0.5

p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
    p1_eeg, p2_eeg, gate, KAPPA, center_ch=2, decay_sigma=0.5, lag_samp=8)

p1_c = p1_eeg.copy()
for ch in range(N_CH):
    for sig in [p1_c, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

oracle_ch = [i for i in np.argsort(-kappa_per_ch) if kappa_per_ch[i] > 0.03]
print(f"Spatial coupling from F3: {len(oracle_ch)} coupled channels")
for i in oracle_ch:
    print(f"  {EPOC_CHANNEL_NAMES[i]:>4} ({i}): kappa={kappa_per_ch[i]:.3f}")
print(f"Gate duty: {gate_mask.mean():.1%}\n")


def evaluate(mask, gate_native):
    """Evaluate at native rate."""
    T = min(len(mask), len(gate_native))
    m, g = mask[:T], gate_native[:T]
    nc, nn = g.sum(), (~g).sum()
    hit = float((m & g).sum() / max(nc, 1))
    fa = float((m & ~g).sum() / max(nn, 1))
    iou = float((m & g).sum() / max((m | g).sum(), 1))
    return hit, fa, iou


# ── Cross-correlation sweep ────────────────────────────────────────────
print("=" * 80)
print("CROSS-CORRELATION: oracle channels, smooth + direct threshold")
print("=" * 80)

# max_lag_s: pass the KNOWN coupling lag (8 samples = 0.03125s)
# The function averages over ±3 samples around the target lag
KNOWN_LAG_S = 8 / FS  # 30ms

for smooth in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    t0 = time.perf_counter()
    mask, z_agg, cc_ch, diag = xcorr_temporal_localization(
        p1_c, p2_coupled, FS,
        channels=oracle_ch, max_lag_s=KNOWN_LAG_S,
        smooth_s=smooth, n_surrogates=100,
        target_fa=0.05, min_event_s=5.0,
        seed=SEED, device=device)
    elapsed = time.perf_counter() - t0

    hit, fa, iou = evaluate(mask, gate_mask)
    zc = z_agg[gate_mask[:len(z_agg)]].mean()
    zn = z_agg[~gate_mask[:len(z_agg)]].mean()

    print(f"  smooth={smooth}s: hit={hit:.1%} FA={fa:.1%} IoU={iou:.1%} "
          f"z_cpl={zc:.2f} z_null={zn:.3f} thr={diag['z_threshold']:.2f} "
          f"({elapsed:.1f}s)", flush=True)

# ── ROC for best smooth ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ROC SWEEP: xcorr smooth=1.0s, oracle channels")
print(f"{'='*80}")

mask, z_agg, cc_ch, diag = xcorr_temporal_localization(
    p1_c, p2_coupled, FS,
    channels=oracle_ch, max_lag_s=KNOWN_LAG_S,
    smooth_s=1.0, n_surrogates=100,
    target_fa=0.05, min_event_s=5.0,
    seed=SEED, device=device)

zc = z_agg[gate_mask[:len(z_agg)]].mean()
zn = z_agg[~gate_mask[:len(z_agg)]].mean()
print(f"z_coupled={zc:.3f}, z_null={zn:.3f}")
print(f"  {'Thr':>6} {'Hit':>7} {'FA':>7} {'IoU':>7}")

g = gate_mask[:len(z_agg)]
for thr in np.arange(0.5, 4.0, 0.2):
    m = z_agg > thr
    # Decimate for event filter
    dec = max(1, int(FS / 4))
    m_dec = _min_event_filter(m[::dec], max(1, int(5.0 * 4)))
    m_up = np.repeat(m_dec, dec)[:len(z_agg)]
    h = (m_up & g).sum() / max(g.sum(), 1)
    f = (m_up & ~g).sum() / max((~g).sum(), 1)
    iou = (m_up & g).sum() / max((m_up | g).sum(), 1)
    mk = " <--" if 0.04 <= f <= 0.06 else ""
    print(f"  {thr:>6.2f} {h:>6.1%} {f:>6.1%} {iou:>6.1%}{mk}")

# ── Null test ──────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("NULL TEST (kappa=0)")
print(f"{'='*80}")
mask_null, z_null, _, diag_null = xcorr_temporal_localization(
    p1_eeg, p2_eeg, FS,
    channels=oracle_ch, max_lag_s=KNOWN_LAG_S,
    smooth_s=1.0, n_surrogates=100,
    target_fa=0.05, min_event_s=5.0,
    seed=SEED, device=device)
print(f"  Null coupling: {mask_null.mean():.1%}")
print(f"  Null z mean={z_null.mean():.3f}, max={z_null.max():.3f}")
