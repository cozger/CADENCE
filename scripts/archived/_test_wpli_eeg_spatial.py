"""Test wPLI/PLV with spatially-decaying EEG coupling at kappa=0.1.

Injection: Focal source at F3, Gaussian spatial decay (sigma=0.5).
  F3=100%, AF3=85%, FC5=72%, F7=47%, T7=18% of kappa.

Detection variants:
  (a) 14-ch raw, pooled aggregation (baseline)
  (b) 4-ROI averaged, pooled aggregation
  (c) 14-ch + spatial cluster filter (require 2 adjacent ch)
  (d) 4-ROI + spatial cluster filter
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial)
from cadence.significance.coherence_localization import wpli_temporal_localization
from cadence.constants import (EEG_ROIS, EPOC_CHANNEL_NAMES, EPOC_DISTANCE,
                                EPOC_ADJACENCY)

FS = 256.0; DURATION = 1800; N_CH = 14; KAPPA = 0.1; SEED = 42
CENTER_CH = 2  # F3
DECAY_SIGMA = 0.5
LAG_SAMP = 8   # ~30ms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Load EEG ───────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=DURATION)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()

p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
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

# ── Coupling gate ──────────────────────────────────────────────────────
gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)

# ── Inject spatially-decaying coupling ─────────────────────────────────
p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
    p1_eeg, p2_eeg, gate, KAPPA,
    center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)

# Re-normalize
p1_c = p1_eeg.copy()
for ch in range(N_CH):
    for sig in [p1_c, p2_coupled]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

print(f"EEG: ({N},{N_CH}) @ {FS}Hz, duty={gate.mean():.1%}")
print(f"Spatial coupling from {EPOC_CHANNEL_NAMES[CENTER_CH]} (idx {CENTER_CH}):")
for i in np.argsort(-kappa_per_ch):
    if kappa_per_ch[i] > 0.001:
        print(f"  {EPOC_CHANNEL_NAMES[i]:>4} ({i:>2}): kappa={kappa_per_ch[i]:.4f} "
              f"({kappa_per_ch[i]/KAPPA:.0%} of peak)")
coupled_ch = [i for i in range(N_CH) if kappa_per_ch[i] > 0.01]
print(f"Effectively coupled: {len(coupled_ch)} channels\n")


def evaluate(mask, z_agg, gate_at_win):
    nc, nn = gate_at_win.sum(), (~gate_at_win).sum()
    hit = float((mask & gate_at_win).sum() / max(nc, 1))
    fa = float((mask & ~gate_at_win).sum() / max(nn, 1))
    iou = float((mask & gate_at_win).sum() / max((mask | gate_at_win).sum(), 1))
    return hit, fa, iou


# ── Detection variants ─────────────────────────────────────────────────
freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)

# Shared params (best from previous tuning)
shared = dict(
    center_freqs=freqs, n_surrogates=100, n_cycles=[3, 7],
    target_fa=0.03, min_event_s=5.0, metric='plv',
    seed=SEED, device=device,
)

# ROC sweep helper
def run_roc(label, p1, p2, extra_kwargs, gate_arr, t_arr):
    """Run pipeline and print ROC at multiple thresholds."""
    t0 = time.perf_counter()
    mask, z_agg, pcz, diag = wpli_temporal_localization(
        p1, p2, FS, **shared, **extra_kwargs)
    elapsed = time.perf_counter() - t0

    wt = diag['win_times']
    gw = np.interp(wt, t_arr, gate_arr.astype(float)) > 0.5
    zc = z_agg[gw].mean() if gw.any() else 0
    zn = z_agg[~gw].mean() if (~gw).any() else 0

    print(f"\n{label} ({elapsed:.1f}s)  z_coupled={zc:.3f} z_null={zn:.3f}")
    print(f"  {'Thr':>6} {'Hit':>7} {'FA':>7} {'IoU':>7}")
    best_hit_at_5 = 0
    from cadence.significance.coherence_localization import _min_event_filter
    for thr in np.arange(0.6, 2.6, 0.1):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / (extra_kwargs.get('stride_s', 0.5)))))
        h, f, iou = evaluate(m, z_agg, gw)
        marker = ""
        if 0.04 <= f <= 0.06:
            marker = " <-- FA~5%"
            best_hit_at_5 = max(best_hit_at_5, h)
        print(f"  {thr:>6.2f} {h:>6.1%} {f:>6.1%} {iou:>6.1%}{marker}")
    return best_hit_at_5

# Oracle coupled channels: top channels by kappa
oracle_ch = [i for i in np.argsort(-kappa_per_ch) if kappa_per_ch[i] > 0.03]
print(f"Oracle coupled channels: {[EPOC_CHANNEL_NAMES[i] for i in oracle_ch]} "
      f"(kappa: {[f'{kappa_per_ch[i]:.3f}' for i in oracle_ch]})\n")

# Build ROI map for only the coupled hemisphere (left frontal + left temporal)
LEFT_ROIS = {
    'left_frontal': [0, 2],      # AF3, F3
    'left_temp':    [1, 3, 4],   # F7, FC5, T7
}

results = {}

print("=" * 70)
print("VARIANT A: Oracle coupled channels, pooled")
print("=" * 70)
results['A'] = run_roc("oracle_20s_sm15", p1_c, p2_coupled,
    dict(channels=oracle_ch, window_s=20.0, stride_s=0.5, smooth_s=15),
    gate, t_eeg)

print("\n" + "=" * 70)
print("VARIANT B: Left-hemisphere ROIs (focused)")
print("=" * 70)
results['B'] = run_roc("leftROI_20s_sm15", p1_c, p2_coupled,
    dict(window_s=20.0, stride_s=0.5, smooth_s=15, roi_map=LEFT_ROIS),
    gate, t_eeg)

print("\n" + "=" * 70)
print("VARIANT C: Oracle + spatial cluster filter")
print("=" * 70)
# Build adjacency submatrix for oracle channels
oracle_adj = EPOC_ADJACENCY[np.ix_(oracle_ch, oracle_ch)]
results['C'] = run_roc("oracle_spatial_20s_sm15", p1_c, p2_coupled,
    dict(channels=oracle_ch, window_s=20.0, stride_s=0.5, smooth_s=15,
         spatial_adjacency=oracle_adj, min_spatial_cluster=2),
    gate, t_eeg)

print("\n" + "=" * 70)
print("VARIANT D: All 14ch (diluted baseline)")
print("=" * 70)
results['D'] = run_roc("all14_20s_sm15", p1_c, p2_coupled,
    dict(channels=list(range(N_CH)), window_s=20.0, stride_s=0.5, smooth_s=15),
    gate, t_eeg)

# ── Null test ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NULL TEST (kappa=0, oracle channels)")
print("=" * 70)
mask_null, z_null, _, _ = wpli_temporal_localization(
    p1_eeg, p2_eeg, FS,
    channels=oracle_ch,
    window_s=20.0, stride_s=0.5, smooth_s=15, **shared)
print(f"  Null coupling fraction: {mask_null.mean():.1%}")
print(f"  Null z mean: {z_null.mean():.3f}, max: {z_null.max():.3f}")

print(f"\n{'='*70}")
print(f"SUMMARY: Hit rate at FA~5%")
for k, v in results.items():
    print(f"  {k}: {v:.1%}")
print(f"{'='*70}")
