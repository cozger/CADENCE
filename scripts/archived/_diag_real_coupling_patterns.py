"""Study real speech and smile patterns in the CWT domain.

Characterize what genuine coupling looks like in time-frequency to inform
more realistic semisynthetic injection models.

Questions:
1. What frequency bands carry the smile signal? (just 0.5-2 Hz, or broader?)
2. What's the actual power spectrum of smile AU activity?
3. What does a smile onset look like in the scalogram?
4. What's the cross-wavelet phase structure during real shared smiles?
5. How does speech AU power distribute across frequencies?
6. What's the real duty cycle and temporal structure of facial coupling?
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, wavelet_coherence, AFFECT_AUS, SMILE_AUS, SPEECH_AUS,
    BAND_STATE, BAND_EXPRESSION, BAND_SPEECH
)

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
session = load_xdf_session(glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))[0])
markers = session['markers']
FS = 30.0

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'coupling_patterns')
os.makedirs(out_dir, exist_ok=True)

# Load conv_2 (richest smile segment)
t0 = markers['conv_2_start']
t1 = markers['conv_2_stop']
p1, p2, dur = extract_bl_segment(session['landmarks'], t0, t1)
T = min(p1.shape[0], p2.shape[0])
p1 = p1[:T]; p2 = p2[:T]
print(f"conv_2: {dur:.0f}s, T={T}")

scal_p1 = compute_au_cwt(p1)
scal_p2 = compute_au_cwt(p2)
freqs = scal_p1.freqs
print(f"CWT: {scal_p1.coeffs.shape}")

# ── 1. Power spectrum of smile AUs vs other AUs ────────────────────────
print("\n=== 1. Mean power spectrum per AU group ===")

groups = {
    'smile (44,45)': [44, 45],
    'frown (30,31)': [30, 31],
    'dimple (28,29)': [28, 29],
    'cheekSq (7,8)': [7, 8],
    'speech (34,35)': [34, 35],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (person, scal, role) in [(axes[0], ('P1', scal_p1, 'patient')),
                                   (axes[1], ('P2', scal_p2, 'therapist'))]:
    for gname, aus in groups.items():
        # Mean power across time, summed across AUs in group
        mean_power = sum(scal.power[:, :, au].mean(axis=1) for au in aus)
        ax.plot(freqs, mean_power, label=gname, linewidth=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mean CWT power')
    ax.set_title(f'{person} ({role})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Band boundaries
    for f in [0.5, 2.0]:
        ax.axvline(f, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

fig.suptitle('conv_2: Mean CWT power spectrum per AU group')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'power_spectrum_by_group.png'), dpi=120)
plt.close(fig)
print("  Saved: power_spectrum_by_group.png")

# ── 2. Cross-wavelet coherence per INDIVIDUAL AU ──────────────────────
print("\n=== 2. Per-AU coherence spectrum ===")

from scipy.ndimage import gaussian_filter1d
sigma = 0.5 * FS

fig, ax = plt.subplots(figsize=(10, 6))
for au in [44, 45, 30, 31, 28, 29, 7, 8, 34, 35]:
    w1 = scal_p1.coeffs[:, :, au]
    w2 = scal_p2.coeffs[:, :, au]
    cross = gaussian_filter1d(w1 * np.conj(w2), sigma=sigma, axis=1)
    auto1 = gaussian_filter1d(np.abs(w1)**2, sigma=sigma, axis=1)
    auto2 = gaussian_filter1d(np.abs(w2)**2, sigma=sigma, axis=1)
    coh = np.abs(cross)**2 / (auto1 * auto2 + 1e-10)
    # Mean coherence per frequency
    mean_coh = coh.mean(axis=1)
    from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES
    ax.plot(freqs, mean_coh, label=f'AU{au:02d} {MP_BLENDSHAPE_NAMES[au][:15]}',
            linewidth=1.5)

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Mean coherence')
ax.set_title('conv_2: Per-AU coherence spectrum (patient-therapist)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
for f in [0.5, 2.0]:
    ax.axvline(f, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'per_au_coherence_spectrum.png'), dpi=120)
plt.close(fig)
print("  Saved: per_au_coherence_spectrum.png")

# Print numerical values
print(f"\n  {'AU':>4s}  {'Name':20s}  {'state':>8s}  {'expr':>8s}  {'speech':>8s}  {'total':>8s}")
for au in [44, 45, 30, 31, 28, 29, 7, 8, 34, 35]:
    w1 = scal_p1.coeffs[:, :, au]
    w2 = scal_p2.coeffs[:, :, au]
    cross = gaussian_filter1d(w1 * np.conj(w2), sigma=sigma, axis=1)
    auto1 = gaussian_filter1d(np.abs(w1)**2, sigma=sigma, axis=1)
    auto2 = gaussian_filter1d(np.abs(w2)**2, sigma=sigma, axis=1)
    coh = np.abs(cross)**2 / (auto1 * auto2 + 1e-10)
    mean_coh = coh.mean(axis=1)

    state_mask = (freqs >= BAND_STATE[0]) & (freqs < BAND_STATE[1])
    expr_mask = (freqs >= BAND_EXPRESSION[0]) & (freqs < BAND_EXPRESSION[1])
    speech_mask = (freqs >= BAND_SPEECH[0]) & (freqs < BAND_SPEECH[1])

    print(f"  AU{au:02d}  {MP_BLENDSHAPE_NAMES[au]:20s}  "
          f"{mean_coh[state_mask].mean():8.3f}  "
          f"{mean_coh[expr_mask].mean():8.3f}  "
          f"{mean_coh[speech_mask].mean():8.3f}  "
          f"{mean_coh.mean():8.3f}")

# ── 3. Raw AU44/45 cross-correlation to characterize lag ──────────────
print("\n=== 3. Smile AU cross-correlation (lag structure) ===")

smile_p1 = p1[:, 44] + p1[:, 45]
smile_p2 = p2[:, 44] + p2[:, 45]
# Remove mean
smile_p1 = smile_p1 - smile_p1.mean()
smile_p2 = smile_p2 - smile_p2.mean()

max_lag_samples = int(5 * FS)
lags = np.arange(-max_lag_samples, max_lag_samples + 1)
xcorr = np.correlate(smile_p1, smile_p2, mode='full')
center = len(smile_p1) - 1
xcorr_window = xcorr[center - max_lag_samples:center + max_lag_samples + 1]
xcorr_window /= (np.std(smile_p1) * np.std(smile_p2) * len(smile_p1))

peak_idx = np.argmax(np.abs(xcorr_window))
peak_lag = lags[peak_idx] / FS
print(f"  Peak cross-correlation at lag = {peak_lag:+.2f}s (positive = P1 leads)")
print(f"  Peak r = {xcorr_window[peak_idx]:+.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags / FS, xcorr_window, linewidth=1)
ax.axvline(peak_lag, color='red', linewidth=0.5, linestyle='--')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Lag (s, positive = P1 leads)')
ax.set_ylabel('Cross-correlation')
ax.set_title(f'conv_2: Smile AU (44+45) cross-correlation, peak at {peak_lag:+.2f}s')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'smile_xcorr.png'), dpi=120)
plt.close(fig)
print("  Saved: smile_xcorr.png")

# Do same for speech AUs
speech_p1 = p1[:, 34] + p1[:, 35]
speech_p2 = p2[:, 34] + p2[:, 35]
speech_p1 -= speech_p1.mean()
speech_p2 -= speech_p2.mean()
xcorr_sp = np.correlate(speech_p1, speech_p2, mode='full')
xcorr_sp_window = xcorr_sp[center - max_lag_samples:center + max_lag_samples + 1]
xcorr_sp_window /= (np.std(speech_p1) * np.std(speech_p2) * len(speech_p1))
peak_sp = np.argmax(np.abs(xcorr_sp_window))
print(f"  Speech AU (34+35) peak xcorr at lag = {lags[peak_sp]/FS:+.2f}s, r = {xcorr_sp_window[peak_sp]:+.3f}")

# ── 4. Time-resolved coherence at smile AUs: when is it high? ────────
print("\n=== 4. When is smile-AU coherence high? ===")

# Compute smile-AU-only coherence timecourse in expression band
w1_44 = scal_p1.coeffs[:, :, 44]
w1_45 = scal_p1.coeffs[:, :, 45]
w2_44 = scal_p2.coeffs[:, :, 44]
w2_45 = scal_p2.coeffs[:, :, 45]

cross_smile = (w1_44 * np.conj(w2_44) + w1_45 * np.conj(w2_45))
auto1_smile = (np.abs(w1_44)**2 + np.abs(w1_45)**2)
auto2_smile = (np.abs(w2_44)**2 + np.abs(w2_45)**2)

cross_s = gaussian_filter1d(cross_smile, sigma=sigma, axis=1)
auto1_s = gaussian_filter1d(auto1_smile, sigma=sigma, axis=1)
auto2_s = gaussian_filter1d(auto2_smile, sigma=sigma, axis=1)

coh_smile = np.abs(cross_s)**2 / (auto1_s * auto2_s + 1e-10)

# Expression-band mean coherence timecourse
expr_mask = (freqs >= 0.5) & (freqs < 2.0)
coh_expr_tc = coh_smile[expr_mask].mean(axis=0)

# Raw smile signals for context
raw_smile_p1 = p1[:, 44] + p1[:, 45]
raw_smile_p2 = p2[:, 44] + p2[:, 45]

fig, axes = plt.subplots(3, 1, figsize=(20, 8), sharex=True)
t = np.arange(T) / FS

axes[0].plot(t, raw_smile_p1, color='#E91E63', linewidth=0.7, alpha=0.8, label='Patient')
axes[0].plot(t, raw_smile_p2, color='#2196F3', linewidth=0.7, alpha=0.8, label='Therapist')
axes[0].set_ylabel('Smile AU\n(44+45)')
axes[0].legend(fontsize=8)

axes[1].plot(t, coh_expr_tc, color='purple', linewidth=0.8)
axes[1].set_ylabel('Expression-band\ncoherence')
axes[1].axhline(np.percentile(coh_expr_tc, 75), color='red', linewidth=0.5, linestyle='--', alpha=0.5)

# Expression-band power for each person (smile AUs only)
expr_power_p1 = scal_p1.expression_power[:, 44] + scal_p1.expression_power[:, 45]
expr_power_p2 = scal_p2.expression_power[:, 44] + scal_p2.expression_power[:, 45]
axes[2].fill_between(t, 0, expr_power_p1, color='#E91E63', alpha=0.5, label='Patient')
axes[2].fill_between(t, 0, -expr_power_p2, color='#2196F3', alpha=0.5, label='Therapist')
axes[2].set_ylabel('Smile expr-band\npower')
axes[2].set_xlabel('Time (s)')
axes[2].axhline(0, color='black', linewidth=0.5)

fig.suptitle('conv_2: Smile coupling dynamics')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'smile_coupling_dynamics.png'), dpi=120)
plt.close(fig)
print("  Saved: smile_coupling_dynamics.png")

# Stats on coherence
high_coh = coh_expr_tc > np.percentile(coh_expr_tc, 75)
print(f"  Expression-band smile coherence: mean={coh_expr_tc.mean():.3f}, "
      f"p75={np.percentile(coh_expr_tc, 75):.3f}, "
      f"max={coh_expr_tc.max():.3f}")
print(f"  High-coherence epochs (>p75): {high_coh.mean()*100:.1f}% of time")

# ── 5. Frequency content of actual smile events ──────────────────────
print("\n=== 5. Smile onset frequency signature ===")

# Find smile onsets: rapid increases in smile_p1
smile_vel = np.diff(raw_smile_p1, prepend=raw_smile_p1[0])
smile_vel_smooth = gaussian_filter1d(smile_vel, sigma=0.1*FS)
from scipy.signal import find_peaks
onset_pks, _ = find_peaks(smile_vel_smooth, height=0.01, distance=int(2*FS))
print(f"  {len(onset_pks)} smile onsets detected in patient")

# Average scalogram around smile onsets (event-related spectral perturbation)
window_s = 4.0
half_win = int(window_s * FS)
ersp_stack = []
for pk in onset_pks:
    if pk - half_win >= 0 and pk + half_win < T:
        snippet = scal_p1.power[:, pk-half_win:pk+half_win, 44] + \
                  scal_p1.power[:, pk-half_win:pk+half_win, 45]
        ersp_stack.append(snippet)

if ersp_stack:
    ersp_mean = np.mean(ersp_stack, axis=0)  # (n_freqs, 2*half_win)
    t_ersp = np.linspace(-window_s, window_s, ersp_mean.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.percentile(ersp_mean, 95)
    ax.pcolormesh(t_ersp, freqs, ersp_mean, shading='auto', cmap='hot',
                  vmin=0, vmax=vmax)
    ax.set_yscale('log')
    ax.set_ylim(0.3, 8)
    ax.axvline(0, color='cyan', linewidth=1, linestyle='--')
    ax.set_xlabel('Time relative to smile onset (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Event-related spectral perturbation: {len(ersp_stack)} smile onsets (patient)')
    for f in [0.5, 2.0]:
        ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'smile_onset_ersp.png'), dpi=120)
    plt.close(fig)
    print(f"  Saved: smile_onset_ersp.png ({len(ersp_stack)} events averaged)")

    # Print per-band power at onset vs baseline
    baseline = ersp_mean[:, :half_win//2].mean(axis=1)
    onset = ersp_mean[:, half_win-int(0.5*FS):half_win+int(0.5*FS)].mean(axis=1)
    print(f"\n  Per-frequency onset/baseline ratio:")
    for i, f in enumerate(freqs):
        r = onset[i] / (baseline[i] + 1e-10)
        if r > 1.5:
            print(f"    {f:.2f} Hz: onset={onset[i]:.5f} base={baseline[i]:.5f} ratio={r:.2f}x ***")

print(f"\nAll outputs in: {out_dir}")
