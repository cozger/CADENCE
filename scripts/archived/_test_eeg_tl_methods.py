"""Compare three EEG temporal localization methods on semi-synthetic data.

Methods:
  1. Windowed fast_cycles (30s windows, 5s stride)
  2. Instantaneous cross-product + smoothing on cycle feature timecourses
  3. Hilbert envelope coherence on cycle feature timecourses

Uses real y_06 EEG with injected arousal coupling via known coupling gate.
Measures temporal localization AUC for each method.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from cadence.significance.fast_cycles import (
    analyze_interbrain_cycles_multiband, _fft_bandpass,
    extract_cycle_features, _reconstruct_cycle_phase, EEG_BANDS,
)
from cadence.synthetic import generate_coupling_gate, inject_eeg_coupling_arousal
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
import torch

# ── Load PSEUDO-DYAD EEG: P1 from y_06, P2 from y_17 ─────────────────
# Critical: real dyad has pre-existing coupling that inflates null AUC.
# Pseudo-dyad guarantees kappa=0 is truly null.

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])

def load_eeg_segment(session_name, person, t_start_rel, t_end_rel):
    """Load EEG segment from cache, avg-ref + z-score."""
    cp = [p for n, p in cached_sessions if session_name in n]
    if not cp:
        return None, 0
    cached = load_session_from_cache(cp[0], config)
    eeg_all = cached[f'{person}_eeg']
    ts_all = cached[f'{person}_eeg_ts']
    m = (ts_all >= t_start_rel) & (ts_all <= t_end_rel)
    if m.sum() < 1000:
        return None, 0
    eeg = eeg_all[m, :14].astype(np.float64)
    ts = ts_all[m]
    fs = len(ts) / (ts[-1] - ts[0])
    eeg -= eeg.mean(axis=1, keepdims=True)
    for ch in range(14):
        std = eeg[:, ch].std()
        if std > 1e-8:
            eeg[:, ch] = (eeg[:, ch] - eeg[:, ch].mean()) / std
    return eeg, fs

# Use middle 300s from each session (avoids edge effects)
p1_eeg, fs1 = load_eeg_segment('y_06', 'p1', 500, 800)
p2_eeg, fs2 = load_eeg_segment('y_17', 'p2', 500, 800)

if p1_eeg is None or p2_eeg is None:
    # Fallback: use p1 from y_06, p2 from y_01
    p2_eeg, fs2 = load_eeg_segment('y01', 'p2', 500, 800)

min_len = min(len(p1_eeg), len(p2_eeg))
p1_eeg = p1_eeg[:min_len]
p2_eeg = p2_eeg[:min_len]
fs_eeg = fs1
T_samp = min_len
dur = T_samp / fs_eeg

print(f"Pseudo-dyad: P1=y_06, P2=y_17, {T_samp} samples, {dur:.0f}s, fs={fs_eeg:.1f} Hz")

# ── Coupling gate ─────────────────────────────────────────────────────

gate = generate_coupling_gate(T_samp, fs_eeg,
                               {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0},
                               seed=999)

# ── Inject arousal coupling ───────────────────────────────────────────

def inject_arousal(p1, p2, gate, kappa, fs, seed=42):
    """Inject amplitude co-modulation: P1's slow theta envelope modulates P2."""
    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract P1 theta envelope (slow modulator)
    p1_t = torch.as_tensor(p1.T, dtype=torch.float32, device=device)
    p1_theta = _fft_bandpass(p1_t, fs, (4, 8), device).cpu().numpy().T

    # Global theta envelope (mean across channels)
    from scipy.signal import hilbert
    env = np.abs(hilbert(p1_theta.mean(axis=1)))
    # Low-pass to get slow modulator
    from scipy.ndimage import gaussian_filter1d
    env_slow = gaussian_filter1d(env, sigma=int(2.0 * fs))
    env_slow = (env_slow - env_slow.mean()) / max(env_slow.std(), 1e-8)

    # Lag
    lag = int(0.5 * fs)
    env_lagged = np.roll(env_slow, lag)
    env_lagged[:lag] = 0

    # Modulate P2
    p2_mod = p2.copy()
    alpha_t = kappa * gate
    for ch in range(p2.shape[1]):
        p2_mod[:, ch] = p2[:, ch] * (1 + alpha_t * env_lagged * 0.3)

    # Re-zscore
    for ch in range(14):
        std = p2_mod[:, ch].std()
        if std > 1e-8:
            p2_mod[:, ch] = (p2_mod[:, ch] - p2_mod[:, ch].mean()) / std

    return p2_mod


# ── Extract cycle feature timecourses ─────────────────────────────────

def extract_feature_timecourses(p1, p2, fs, band=(4, 8)):
    """Extract per-cycle volt_amp resampled to 2Hz for both people."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, C = p1.shape
    rate = 2.0
    dur_s = T / fs
    t_grid = np.arange(0, dur_s, 1.0 / rate)
    n_grid = len(t_grid)

    both = np.vstack([p1.T, p2.T])
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)
    both_filt = _fft_bandpass(both_t, fs, band, device).cpu().numpy()

    p1_filt = both_filt[:C].T
    p2_filt = both_filt[C:].T

    p1_va = np.zeros((C, n_grid), dtype=np.float32)
    p2_va = np.zeros((C, n_grid), dtype=np.float32)
    valid = np.zeros(C, dtype=bool)

    for ch in range(C):
        cyc1 = extract_cycle_features(p1[:, ch], p1_filt[:, ch], fs, band)
        cyc2 = extract_cycle_features(p2[:, ch], p2_filt[:, ch], fs, band)
        if cyc1 is None or cyc2 is None or cyc1['n_cycles'] < 10 or cyc2['n_cycles'] < 10:
            continue
        valid[ch] = True
        t1 = cyc1['peak_sample'] / fs
        t2 = cyc2['peak_sample'] / fs
        p1_va[ch] = np.interp(t_grid, t1, cyc1['volt_amp'])
        p2_va[ch] = np.interp(t_grid, t2, cyc2['volt_amp'])

    return p1_va[valid], p2_va[valid], t_grid, rate


# ── Method 1: Windowed fast_cycles ────────────────────────────────────

def method_windowed(p1, p2, fs, win_s=30, stride_s=5, n_surr=50):
    """Run fast_cycles in sliding windows."""
    T = min(len(p1), len(p2))
    win = int(win_s * fs)
    stride = int(stride_s * fs)
    n_wins = (T - win) // stride + 1

    out_times = np.array([(i * stride + win // 2) / fs for i in range(n_wins)])
    z_vals = np.zeros(n_wins)

    for i in range(n_wins):
        s = i * stride
        e = s + win
        try:
            r = analyze_interbrain_cycles_multiband(
                p1[s:e], p2[s:e], fs, n_surrogates=n_surr, seed=42)
            comb = r.get('combined', {}).get('volt_amp', {})
            z_vals[i] = comb.get('stouffer_z', comb.get('pooled_z', 0))
        except Exception:
            z_vals[i] = 0

    return out_times, z_vals


# ── Method 2: Cross-product + smoothing ───────────────────────────────

def method_crossproduct(p1_va, p2_va, rate, smooth_s=5.0, n_surr=100, seed=42):
    """Instantaneous cross-product of cycle feature timecourses + surrogates."""
    C, N = p1_va.shape
    rng = np.random.default_rng(seed)

    # Z-score each channel
    p1z = p1_va.copy()
    p2z = p2_va.copy()
    for c in range(C):
        for arr in [p1z, p2z]:
            mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
            arr[c] = (arr[c] - mu) / sd

    # Cross-product averaged across channels, smoothed
    cp = (p1z * p2z).mean(axis=0)
    sigma = smooth_s * rate
    cp_smooth = gaussian_filter1d(cp, sigma=sigma)

    # Surrogates
    surr_cp = np.zeros((n_surr, N))
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N), int(0.9 * N))
        p1_shifted = np.roll(p1z, shift, axis=1)
        surr_cp[k] = gaussian_filter1d((p1_shifted * p2z).mean(axis=0), sigma=sigma)

    surr_mean = surr_cp.mean(axis=0)
    surr_std = np.maximum(surr_cp.std(axis=0), 1e-10)
    z = (cp_smooth - surr_mean) / surr_std

    return z


# ── Method 3: Hilbert envelope coherence ──────────────────────────────

def method_envelope_coherence(p1_va, p2_va, rate, smooth_s=5.0, n_surr=100, seed=42):
    """Hilbert envelope coherence on cycle feature timecourses.

    Computes analytic signal of each channel's volt_amp, then windowed
    cross-spectrum magnitude (coherence) with surrogate calibration.
    """
    C, N = p1_va.shape
    rng = np.random.default_rng(seed)
    sigma = smooth_s * rate

    # Z-score
    p1z = p1_va.copy()
    p2z = p2_va.copy()
    for c in range(C):
        for arr in [p1z, p2z]:
            mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
            arr[c] = (arr[c] - mu) / sd

    # Analytic signal per channel
    coh_real = np.zeros(N)
    for c in range(C):
        a1 = hilbert(p1z[c])
        a2 = hilbert(p2z[c])
        sxy = a1 * a2.conj()
        sxy_smooth = gaussian_filter1d(sxy.real, sigma) + 1j * gaussian_filter1d(sxy.imag, sigma)
        pxx = gaussian_filter1d(np.abs(a1) ** 2, sigma)
        pyy = gaussian_filter1d(np.abs(a2) ** 2, sigma)
        coh_real += np.abs(sxy_smooth) / np.sqrt(pxx * pyy + 1e-10)
    coh_real /= C

    # Surrogates
    surr_coh = np.zeros((n_surr, N))
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N), int(0.9 * N))
        p1_shifted = np.roll(p1z, shift, axis=1)
        coh_surr = np.zeros(N)
        for c in range(C):
            a1 = hilbert(p1_shifted[c])
            a2 = hilbert(p2z[c])
            sxy = a1 * a2.conj()
            sxy_s = gaussian_filter1d(sxy.real, sigma) + 1j * gaussian_filter1d(sxy.imag, sigma)
            pxx = gaussian_filter1d(np.abs(a1) ** 2, sigma)
            pyy = gaussian_filter1d(np.abs(a2) ** 2, sigma)
            coh_surr += np.abs(sxy_s) / np.sqrt(pxx * pyy + 1e-10)
        surr_coh[k] = coh_surr / C

    surr_mean = surr_coh.mean(axis=0)
    surr_std = np.maximum(surr_coh.std(axis=0), 1e-10)
    z = (coh_real - surr_mean) / surr_std
    return z


# ── Run comparison ────────────────────────────────────────────────────

# ── Method 4: Channel-pooled cross-product, NO smoothing ─────────────

def method_channel_pooled(p1_va, p2_va, rate, n_surr=100, seed=42):
    """Per-timepoint cross-product pooled across channels, surrogate-calibrated.

    No temporal smoothing. Noise reduction from channel averaging only.
    """
    C, N = p1_va.shape
    rng = np.random.default_rng(seed)

    # Z-score per channel
    p1z = p1_va.copy()
    p2z = p2_va.copy()
    for c in range(C):
        for arr in [p1z, p2z]:
            mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
            arr[c] = (arr[c] - mu) / sd

    # Cross-product averaged across channels (no smoothing)
    cp_real = (p1z * p2z).mean(axis=0)  # (N,)

    # Surrogates: shift P1, compute channel-averaged cross-product
    cp_surr = np.zeros((n_surr, N))
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N), int(0.9 * N))
        p1_shifted = np.roll(p1z, shift, axis=1)
        cp_surr[k] = (p1_shifted * p2z).mean(axis=0)

    surr_mean = cp_surr.mean(axis=0)
    surr_std = np.maximum(cp_surr.std(axis=0), 1e-10)
    z = (cp_real - surr_mean) / surr_std

    return z


# ── Method 5: Two-pass channel-weighted, no smoothing ─────────────────

def method_two_pass(p1_eeg, p2_eeg, p1_va, p2_va, fs_eeg, feat_rate,
                    n_surr=100, seed=42):
    """Two-pass: fast_cycles discovers channels, weighted cross-product tracks time.

    Pass 1: full-segment fast_cycles -> per-channel z-scores
    Pass 2: softmax(z) weights on per-timepoint cross-product, no smoothing
    """
    rng = np.random.default_rng(seed)
    C, N = p1_va.shape

    # Pass 1: get per-channel z-scores from fast_cycles
    try:
        r = analyze_interbrain_cycles_multiband(
            p1_eeg, p2_eeg, fs_eeg, n_surrogates=50, seed=seed)
    except Exception:
        return np.zeros(N)

    # Collect per-channel volt_amp z across bands, take max per channel
    ch_z = np.zeros(C)
    for band in ['theta', 'alpha', 'beta']:
        br = r.get('per_band', {}).get(band, {})
        per_ch = br.get('volt_amp', {}).get('per_channel_z', {})
        for ch_idx_str, z_val in per_ch.items():
            ch_idx = int(ch_idx_str)
            if ch_idx < C:
                ch_z[ch_idx] = max(ch_z[ch_idx], z_val)

    # Softmax weights (temperature=1, clip negative z to 0)
    ch_z_pos = np.maximum(ch_z, 0)
    if ch_z_pos.sum() < 1e-6:
        weights = np.ones(C) / C  # fallback to uniform
    else:
        exp_z = np.exp(ch_z_pos - ch_z_pos.max())
        weights = exp_z / exp_z.sum()

    # Z-score feature timecourses per channel
    p1z = p1_va.copy()
    p2z = p2_va.copy()
    for c in range(C):
        for arr in [p1z, p2z]:
            mu, sd = arr[c].mean(), max(arr[c].std(), 1e-8)
            arr[c] = (arr[c] - mu) / sd

    # Pass 2: weighted cross-product per timepoint (no smoothing)
    cp_real = np.zeros(N)
    for c in range(C):
        cp_real += weights[c] * p1z[c] * p2z[c]

    # Surrogates with same weights
    cp_surr = np.zeros((n_surr, N))
    for k in range(n_surr):
        shift = rng.integers(int(0.1 * N), int(0.9 * N))
        p1_shifted = np.roll(p1z, shift, axis=1)
        for c in range(C):
            cp_surr[k] += weights[c] * p1_shifted[c] * p2z[c]

    surr_mean = cp_surr.mean(axis=0)
    surr_std = np.maximum(cp_surr.std(axis=0), 1e-10)
    z = (cp_real - surr_mean) / surr_std

    return z


# Sweep smoothing widths in samples at 2 Hz feature rate
smooth_samples = [0, 2, 3, 5, 7, 10, 15, 20]
smooth_secs = [s / 2.0 for s in smooth_samples]  # 2 Hz rate

print(f"\n{'kappa':>6} | " + " | ".join(f"{'s='+str(s)+'smp':>8}" for s in smooth_samples) + f" | {'sess_z':>7}")
print("-" * (10 + 11 * len(smooth_samples) + 10))

for kappa in [0.0, 0.15, 0.3, 0.5, 0.8]:
    p2_mod, _, _ = inject_eeg_coupling_arousal(
        p1_eeg, p2_eeg, gate, kappa, fs=fs_eeg, seed=42)

    # Extract feature timecourses once (shared by M2 and M3)
    p1_va, p2_va_orig, t_grid, feat_rate = extract_feature_timecourses(
        p1_eeg, p2_eeg, fs_eeg, band=(4, 8))
    p1_va_mod, p2_va_mod, _, _ = extract_feature_timecourses(
        p1_eeg, p2_mod, fs_eeg, band=(4, 8))

    # Gate resampled to feature rate
    gate_feat = np.interp(t_grid, np.arange(T_samp) / fs_eeg, gate[:T_samp])
    gate_binary = (gate_feat > 0.5).astype(float)

    if gate_binary.sum() == 0 or gate_binary.sum() == len(gate_binary):
        print(f"{kappa:6.1f} | gate degenerate")
        continue

    aucs = []
    for n_samp in smooth_samples:
        s_sec = n_samp / feat_rate if n_samp > 0 else 0
        if n_samp == 0:
            z = method_channel_pooled(p1_va_mod, p2_va_mod, feat_rate, n_surr=100)
        else:
            z = method_crossproduct(p1_va_mod, p2_va_mod, feat_rate,
                                     smooth_s=s_sec, n_surr=100)
        auc = roc_auc_score(gate_binary, z)
        aucs.append(auc)

    # Session-level check
    try:
        r = analyze_interbrain_cycles_multiband(p1_eeg, p2_mod, fs_eeg, n_surrogates=50, seed=42)
        sess_z = r.get('combined', {}).get('volt_amp', {}).get('stouffer_z',
                 r.get('combined', {}).get('volt_amp', {}).get('pooled_z', 0))
    except Exception:
        sess_z = 0

    print(f"{kappa:6.2f} | " + " | ".join(f"{a:8.3f}" for a in aucs) + f" | {sess_z:+7.1f}")
