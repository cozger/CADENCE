"""Empirical test: does 60 Hz give same power as 30 Hz at matched kernel samples?

Upsamples real BL data to 60 Hz, injects identical coupling, runs pipeline
at both rates with adaptive smoothing. Compares z-scores.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.interpolate import interp1d
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.significance.coherence_localization import xcorr_temporal_localization

FS_LO = 30.0
FS_HI = 60.0
DURATION = 600
SEED = 42

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
all_sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
all_sess = [(n, s) for n, s in all_sess if s is not None and 'p1_blendshapes' in s]
s1, s2 = all_sess[-1][1], all_sess[0][1]

# Extract 30 Hz BL from two different sessions (pseudo-dyad)
def get_bl(sess, role='therapist'):
    p1r = sess.get('p1_role', 'therapist')
    if (role == 'therapist' and p1r == 'therapist') or \
       (role == 'patient' and p1r == 'patient'):
        return sess['p1_blendshapes'], sess['p1_blendshapes_ts']
    return sess['p2_blendshapes'], sess['p2_blendshapes_ts']

p1_raw, p1_ts = get_bl(s1, 'therapist')
p2_raw, p2_ts = get_bl(s2, 'patient')

p1_s, p2_s = float(p1_ts[0]), float(p2_ts[0])
N30 = int(DURATION * FS_LO)
t30 = np.linspace(0, DURATION, N30)

p1_30 = np.stack([np.interp(t30, p1_ts - p1_s, p1_raw[:, c])
                   for c in range(52)], axis=1).astype(np.float32)[:N30]
p2_30 = np.stack([np.interp(t30, p2_ts - p2_s, p2_raw[:, c])
                   for c in range(52)], axis=1).astype(np.float32)[:N30]

# Z-normalize
for sig in [p1_30, p2_30]:
    for ch in range(52):
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Upsample to 60 Hz (linear interp — preserves autocorrelation structure)
N60 = int(DURATION * FS_HI)
t60 = np.linspace(0, DURATION, N60)
p1_60 = np.stack([np.interp(t60, t30, p1_30[:, c])
                   for c in range(52)], axis=1).astype(np.float32)
p2_60 = np.stack([np.interp(t60, t30, p2_30[:, c])
                   for c in range(52)], axis=1).astype(np.float32)

# Inject coupling: simple mixing model on first 10 AUs
from cadence.synthetic import generate_coupling_gate
gate_30 = generate_coupling_gate(N30, FS_LO, {
    'duty_cycle': 0.15, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)
gate_60 = np.interp(t60, t30, gate_30)

kappa = 0.3
lag_30 = int(2.0 * FS_LO)  # 60 samples
lag_60 = int(2.0 * FS_HI)  # 120 samples
coupled_chs = list(range(10))

p2_30c = p2_30.copy()
p2_60c = p2_60.copy()

for ch in coupled_chs:
    # 30 Hz injection
    alpha30 = kappa * gate_30
    noise30 = np.sqrt(np.maximum(1 - alpha30**2, 0))
    p1_lag30 = np.roll(p1_30[:, ch], lag_30)
    p1_lag30[:lag_30] = 0
    p2_30c[:, ch] = alpha30 * p1_lag30 + noise30 * p2_30[:, ch]

    # 60 Hz injection (same kappa, same physical lag)
    alpha60 = kappa * gate_60
    noise60 = np.sqrt(np.maximum(1 - alpha60**2, 0))
    p1_lag60 = np.roll(p1_60[:, ch], lag_60)
    p1_lag60[:lag_60] = 0
    p2_60c[:, ch] = alpha60 * p1_lag60 + noise60 * p2_60[:, ch]

# Re-normalize
for sig in [p2_30c, p2_60c]:
    for ch in range(52):
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

# Run xcorr_temporal_localization at both rates with adaptive smooth
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = [
    ('30Hz / 3.0s smooth (baseline)', p1_30, p2_30c, FS_LO, 3.0, gate_30, t30),
    ('60Hz / 1.5s smooth (matched N)', p1_60, p2_60c, FS_HI, 1.5, gate_60, t60),
    ('60Hz / 3.0s smooth (max power)', p1_60, p2_60c, FS_HI, 3.0, gate_60, t60),
    ('30Hz / 1.5s smooth (halved N)',  p1_30, p2_30c, FS_LO, 1.5, gate_30, t30),
]

# Also null (no coupling)
configs_null = [
    ('30Hz null', p1_30, p2_30, FS_LO, 3.0, gate_30, t30),
    ('60Hz null', p1_60, p2_60, FS_HI, 1.5, gate_60, t60),
]

print(f"Device: {device}")
print(f"Coupling: kappa={kappa}, 10 AUs, lag=2.0s, duty=15%\n")

print(f"{'Config':>35} {'smooth':>7} {'N_kern':>7} {'z_coupled':>10} "
      f"{'z_null':>8} {'d_prime':>8}")
print(f"{'-'*80}")

for label, p1, p2, fs, sm, gate, t_arr in configs:
    mask, z, lag, diag = xcorr_temporal_localization(
        p1, p2, fs, channels=coupled_chs,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=sm,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0,
        seed=SEED, device=device)

    out_rate = diag['output_rate']
    t_out = np.arange(len(z)) / out_rate
    gw = np.interp(t_out, t_arr, gate.astype(float)) > 0.5
    zc = z[gw].mean() if gw.any() else 0
    zn = z[~gw].mean() if (~gw).any() else 0
    dp = zc - zn
    n_kern = int(sm * fs)

    print(f"{label:>35} {sm:>6.1f}s {n_kern:>7} {zc:>+9.3f} "
          f"{zn:>+7.3f} {dp:>+7.3f}")

print(f"\n{'Config':>35} {'smooth':>7} {'N_kern':>7} {'z_mean':>10}")
print(f"{'-'*65}")
for label, p1, p2, fs, sm, gate, t_arr in configs_null:
    mask, z, lag, diag = xcorr_temporal_localization(
        p1, p2, fs, channels=coupled_chs,
        max_lag_s=5.0, lag_step_s=0.1, smooth_s=sm,
        n_surrogates=100, target_fa=0.05, min_event_s=5.0,
        seed=SEED, device=device)
    print(f"{label:>35} {sm:>6.1f}s {int(sm*fs):>7} {z.mean():>+9.3f}")
