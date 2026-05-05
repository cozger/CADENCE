"""Arousal coupling semi-synthetic validation.

Injects slow amplitude co-modulation at kappa sweep, validates with
fast_cycles (volt_amp should detect, period/symmetry should not).
Includes PLV cross-check and pseudo-dyad null.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_arousal)
from cadence.significance.fast_cycles import analyze_interbrain_cycles
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

FS = 256.0; SEED = 42; K_SURR = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load cross-dyad EEG ─────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=1800)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()
p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + 1800)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(1800 * FS))
t_target = np.linspace(0, 1800, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(14)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]
t_eeg = np.arange(N) / FS

# Avg-ref + z-normalize
for sig in [p1_eeg, p2_eeg]:
    sig -= sig.mean(axis=1, keepdims=True).astype(sig.dtype)
    for ch in range(14):
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=SEED)

freqs_theta = np.logspace(np.log10(4.0), np.log10(8.0), 5)

print(f"Device: {device}")
print(f"EEG: ({N}, 14) @ {FS} Hz, duty={gate.mean():.1%}\n")


def eval_plv(z_agg, gate_win):
    best = 0.0
    for thr in np.arange(0.4, 3.0, 0.1):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / 0.5)))
        nc, nn = gate_win.sum(), (~gate_win).sum()
        h = float((m & gate_win).sum() / max(nc, 1))
        f = float((m & ~gate_win).sum() / max(nn, 1))
        if 0.03 <= f <= 0.07:
            best = max(best, h)
    return best


# ── Kappa sweep ──────────────────────────────────────────────────────────
print(f"{'kappa':>6} | {'volt_amp_z':>10} {'period_z':>10} {'rdsym_z':>10} "
      f"{'burst_z':>10} {'plv_hit':>10} | {'time':>6}")
print(f"{'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*6}")

for kappa in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    t0 = time.perf_counter()

    if kappa == 0:
        p2_coupled = p2_eeg.copy()
    else:
        p2_coupled, kappa_per_ch, modulator = inject_eeg_coupling_arousal(
            p1_eeg, p2_eeg, gate, kappa,
            lag_s=1.0, mod_bandwidth=(0.05, 0.5), source_band=(4.0, 8.0),
            spatial_mode='frontal', fs=FS, seed=SEED)
        # Re-normalize
        for ch in range(14):
            mu, sd = p2_coupled[:, ch].mean(), max(p2_coupled[:, ch].std(), 1e-8)
            p2_coupled[:, ch] = (p2_coupled[:, ch] - mu) / sd

    # fast_cycles analysis
    r = analyze_interbrain_cycles(p1_eeg, p2_coupled, FS,
                                   band=(4.0, 8.0), n_surrogates=K_SURR, seed=SEED)

    va_z = r.get('volt_amp', {}).get('pooled_z', 0)
    pe_z = r.get('period', {}).get('pooled_z', 0)
    rs_z = r.get('time_rdsym', {}).get('pooled_z', 0)
    bu_z = r.get('burst_cooc', {}).get('pooled_z', 0)

    # PLV comparison (theta only)
    _, z_plv, _, d_plv = wpli_temporal_localization(
        p1_eeg, p2_coupled, FS,
        channels=list(range(14)), center_freqs=freqs_theta,
        n_surrogates=100, n_cycles=[3, 7], window_s=20.0, stride_s=0.5,
        smooth_s=15.0, target_fa=0.03, min_event_s=5.0, metric='plv',
        seed=SEED, device=device)
    gw = np.interp(d_plv['win_times'], t_eeg, gate.astype(float)) > 0.5
    plv_hit = eval_plv(z_plv, gw) if kappa > 0 else 0.0

    elapsed = time.perf_counter() - t0

    print(f"{kappa:>6.2f} | {va_z:>+9.2f} {pe_z:>+9.2f} {rs_z:>+9.2f} "
          f"{bu_z:>+9.2f} {plv_hit:>9.1%} | {elapsed:>5.1f}s")

    # Print spatial pattern at kappa=0.30
    if kappa == 0.30:
        va = r.get('volt_amp', {})
        per_r = va.get('per_channel_r', {})
        print(f"        Spatial at kappa=0.30:")
        for ch in range(14):
            rv = per_r.get(ch, 0)
            print(f"          {EPOC_CHANNEL_NAMES[ch]:>4}: r={rv:+.4f}")

print(f"\n{'='*70}")
