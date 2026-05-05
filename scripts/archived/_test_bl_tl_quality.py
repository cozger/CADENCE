"""Check the temporal resolution and quality of Stage 1 BL coupling mask.

Runs on semi-synthetic (known gate) and real data to assess:
1. What does the z-score timecourse look like?
2. How well does the mask match the true coupling gate? (AUC)
3. What's the temporal resolution in practice?
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob
from sklearn.metrics import roc_auc_score

from cadence.synthetic import generate_coupling_gate
from cadence.significance.bl_coupling import bl_two_stage_coupling

FS = 30.0

# Load y_06 raw BL
xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

marker_times = {}
for stream in data:
    if stream['info']['type'][0] == 'Markers':
        for t, v in zip(stream['time_stamps'], stream['time_series']):
            marker_times[v[0]] = t

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

t_start = marker_times['conv_1_start']
t_end = marker_times['conv_1_stop']
dur = t_end - t_start
T = int(dur * FS)
t_grid = np.linspace(0, dur, T)

sigs = {}
for p in ['P1', 'P2']:
    ts, d = landmarks[p]
    m = (ts >= t_start) & (ts <= t_end)
    sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)

p1_raw = sigs['P1']
p2_raw = sigs['P2']
print(f"Loaded conv_1: {T} samples, {dur:.1f}s")

# ── 1. Real data: what does the z-score timecourse look like? ──

print("\n--- Real data (y_06 conv_1) ---")
result = bl_two_stage_coupling(p1_raw, p2_raw, FS, seed=42)

z = result.z_continuous
mask = result.mask_continuous
times = result.times
out_rate = result.output_rate

print(f"Output rate: {out_rate:.1f} Hz, {len(z)} timepoints")
print(f"z range: [{z.min():.2f}, {z.max():.2f}], mean={z.mean():.2f}")
print(f"z>2 fraction: {(z > 2).mean():.1%}")
print(f"Mask fraction: {mask.mean():.1%}")
print(f"Estimated lag: {result.estimated_lag_s:.2f}s (conf={result.lag_confidence:.2f})")
print(f"Threshold: {result.diagnostics['stage1']['z_threshold']:.2f}")

# z-score distribution
for pct in [50, 75, 90, 95, 99]:
    print(f"  z {pct}th percentile: {np.percentile(z, pct):.2f}")

# ── 2. Semi-synthetic: inject coupling and measure TL AUC ──

print("\n--- Semi-synthetic (kappa sweep) ---")
gate_profile = {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0}
gate = generate_coupling_gate(T, FS, gate_profile, seed=999)

from cadence.significance.coherence_localization import xcorr_temporal_localization
from cadence.significance.kim_filter import _estimate_ar

def inject_and_test(p1, p2, gate, kappa, seed=42):
    """Inject linear coupling at given kappa, run Stage 1, measure TL AUC."""
    T_s = min(p1.shape[0], p2.shape[0])
    n_aus = min(52, p1.shape[1])

    # Z-score internally (matching bl_two_stage_coupling)
    p1_z = p1[:T_s, :n_aus].copy()
    p2_z = p2[:T_s, :n_aus].copy()
    for c in range(n_aus):
        for sig in [p1_z, p2_z]:
            mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
            sig[:, c] = (sig[:, c] - mu) / sd

    # Inject coupling
    lag_samp = int(2.5 * FS)
    p2_coupled = p2_z.copy()
    if kappa > 0:
        p1_lagged = np.roll(p1_z, lag_samp, axis=0)
        p1_lagged[:lag_samp] = 0
        alpha_t = kappa * gate[:T_s, None]
        noise_scale = np.sqrt(np.maximum(1 - alpha_t**2, 0))
        p2_coupled[lag_samp:] = (alpha_t[lag_samp:] * p1_lagged[lag_samp:]
                                  + noise_scale[lag_samp:] * p2_z[lag_samp:])

    # AR residualize P2
    p2_res = p2_coupled.copy()
    for ch in range(n_aus):
        a, _ = _estimate_ar(p2_coupled[:, ch], 3)
        pred = np.zeros(T_s)
        for i in range(len(a)):
            pred[i+1:] += a[i] * p2_coupled[:T_s-i-1, ch]
        p2_res[:, ch] = p2_coupled[:, ch] - pred

    # Run xcorr TL
    mask_out, z_out, lag_out, diag = xcorr_temporal_localization(
        p1_z, p2_res, FS, max_lag_s=5.0, lag_step_s=0.1,
        smooth_s=3.0, n_surrogates=100, target_fa=0.05,
        min_event_s=5.0, seed=seed)

    out_rate = diag['output_rate']

    # Resample gate to output rate for AUC
    gate_times = np.arange(len(z_out)) / out_rate
    gate_resampled = np.interp(gate_times, np.arange(T_s) / FS, gate[:T_s])
    gate_binary = (gate_resampled > 0.5).astype(float)

    if gate_binary.sum() > 0 and gate_binary.sum() < len(gate_binary):
        auc = roc_auc_score(gate_binary, z_out)
    else:
        auc = 0.5

    hit = (mask_out & (gate_binary > 0.5)).sum() / max(gate_binary.sum(), 1)
    fa = (mask_out & (gate_binary < 0.5)).sum() / max((gate_binary < 0.5).sum(), 1)

    return auc, hit, fa, z_out.mean(), z_out.max(), diag

print(f"{'kappa':>6} {'AUC':>6} {'hit':>6} {'FA':>6} {'z_mean':>7} {'z_max':>7}")
for kappa in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    auc, hit, fa, zm, zx, _ = inject_and_test(p1_raw, p2_raw, gate, kappa)
    print(f"{kappa:6.2f} {auc:6.3f} {hit:6.1%} {fa:6.1%} {zm:7.2f} {zx:7.2f}")
