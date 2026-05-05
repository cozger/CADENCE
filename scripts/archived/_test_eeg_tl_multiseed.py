"""EEG TL validation with multiple gate seeds for stable AUC estimates.

Uses cross-product with 3-sample smoothing (1.5s resolution) on pseudo-dyad.
10 gate seeds per kappa to average out AUC variance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

from cadence.significance.fast_cycles import (
    analyze_interbrain_cycles_multiband, _fft_bandpass, extract_cycle_features,
)
from cadence.synthetic import generate_coupling_gate, inject_eeg_coupling_arousal
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
import torch

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])

def load_seg(session_name, person, t0, t1):
    cp = [p for n, p in cached_sessions if session_name in n]
    cached = load_session_from_cache(cp[0], config)
    eeg = cached[f'{person}_eeg']
    ts = cached[f'{person}_eeg_ts']
    m = (ts >= t0) & (ts <= t1)
    seg = eeg[m, :14].astype(np.float64)
    fs = len(ts[m]) / (ts[m][-1] - ts[m][0])
    seg -= seg.mean(axis=1, keepdims=True)
    for ch in range(14):
        std = seg[:, ch].std()
        if std > 1e-8:
            seg[:, ch] = (seg[:, ch] - seg[:, ch].mean()) / std
    return seg, fs

p1, fs = load_seg('y_06', 'p1', 500, 800)
p2, _ = load_seg('y_17', 'p2', 500, 800)
min_len = min(len(p1), len(p2))
p1 = p1[:min_len]
p2 = p2[:min_len]
T_samp = min_len
dur = T_samp / fs

print(f"Pseudo-dyad: {T_samp} samples, {dur:.0f}s, fs={fs:.1f} Hz")

# Extract feature timecourses once for the base signals
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C = 14
rate = 2.0
t_grid = np.arange(0, dur, 1.0 / rate)
N = len(t_grid)

def get_volt_amp(p1_eeg, p2_eeg):
    both = np.vstack([p1_eeg.T, p2_eeg.T])
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)
    both_filt = _fft_bandpass(both_t, fs, (4, 8), device).cpu().numpy()
    p1f = both_filt[:C].T
    p2f = both_filt[C:].T
    p1_va = np.zeros((C, N), dtype=np.float32)
    p2_va = np.zeros((C, N), dtype=np.float32)
    valid = []
    for ch in range(C):
        c1 = extract_cycle_features(p1_eeg[:, ch], p1f[:, ch], fs, (4, 8))
        c2 = extract_cycle_features(p2_eeg[:, ch], p2f[:, ch], fs, (4, 8))
        if c1 is None or c2 is None or c1['n_cycles'] < 10 or c2['n_cycles'] < 10:
            continue
        valid.append(ch)
        p1_va[ch] = np.interp(t_grid, c1['peak_sample'] / fs, c1['volt_amp'])
        p2_va[ch] = np.interp(t_grid, c2['peak_sample'] / fs, c2['volt_amp'])
    return p1_va[valid], p2_va[valid]

def crossproduct_z(p1_va, p2_va, sigma_samp=1.5, n_surr=100, seed=42):
    C_v, N_v = p1_va.shape
    rng = np.random.default_rng(seed)
    p1z = p1_va.copy()
    p2z = p2_va.copy()
    for c in range(C_v):
        for arr in [p1z, p2z]:
            mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
            arr[c] = (arr[c] - mu) / sd
    cp = gaussian_filter1d((p1z * p2z).mean(axis=0), sigma=sigma_samp)
    surr = np.zeros((n_surr, N_v))
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N_v), int(0.9 * N_v))
        surr[k] = gaussian_filter1d((np.roll(p1z, shift, axis=1) * p2z).mean(axis=0), sigma=sigma_samp)
    sm = surr.mean(axis=0)
    ss = np.maximum(surr.std(axis=0), 1e-10)
    return (cp - sm) / ss

N_SEEDS = 10

print(f"\n{'kappa':>6} | {'AUC_mean':>8} {'AUC_std':>8} {'AUC_min':>8} {'AUC_max':>8} | {'sess_z':>7}")
print("-" * 65)

for kappa in [0.0, 0.15, 0.30, 0.50, 0.80]:
    aucs = []
    sess_z = 0

    for gate_seed in range(N_SEEDS):
        gate = generate_coupling_gate(T_samp, fs,
            {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0},
            seed=1000 + gate_seed)

        p2_mod, _, _ = inject_eeg_coupling_arousal(p1, p2, gate, kappa, fs=fs, seed=42)
        p1_va, p2_va = get_volt_amp(p1, p2_mod)

        gate_feat = np.interp(t_grid, np.arange(T_samp) / fs, gate[:T_samp])
        gate_bin = (gate_feat > 0.5).astype(float)

        if gate_bin.sum() < 10 or gate_bin.sum() > len(gate_bin) - 10:
            continue

        z = crossproduct_z(p1_va, p2_va, sigma_samp=1.5, n_surr=100,
                           seed=42 + gate_seed)
        aucs.append(roc_auc_score(gate_bin, z))

    # Session z on first gate
    gate0 = generate_coupling_gate(T_samp, fs,
        {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0}, seed=1000)
    p2_mod0, _, _ = inject_eeg_coupling_arousal(p1, p2, gate0, kappa, fs=fs, seed=42)
    try:
        r = analyze_interbrain_cycles_multiband(p1, p2_mod0, fs, n_surrogates=50, seed=42)
        sess_z = r.get('combined', {}).get('volt_amp', {}).get('stouffer_z',
                 r.get('combined', {}).get('volt_amp', {}).get('pooled_z', 0))
    except Exception:
        sess_z = 0

    aucs = np.array(aucs)
    print(f"{kappa:6.2f} | {aucs.mean():8.3f} {aucs.std():8.3f} "
          f"{aucs.min():8.3f} {aucs.max():8.3f} | {sess_z:+7.1f}")
