"""Wavelet-native BL coupling analysis (V7).

CWT decomposition of blendshape AU timeseries into frequency bands,
with band-specific event detection, speech/expression separation,
and cross-participant wavelet coherence.

Literature basis: Jeganathan et al. 2022 (eLife) — CWT on AU timeseries
discovers facial states with spectral fingerprints in 0-5 Hz band.
Wavelet coherence on AU timeseries between dyad members is novel.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Try GPU acceleration via torch, fall back to numpy
try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False


# ── Frequency band definitions (empirically grounded) ──────────────────
BAND_STATE = (0.0, 0.5)       # tonic drift, emotional state
BAND_EXPRESSION = (0.5, 2.0)  # expression transitions (onset/offset)
BAND_SPEECH = (2.0, 7.0)      # speech articulation, syllable rate
BAND_NOISE = (7.0, 15.0)      # tracker noise (should be removed by low-pass)

# Affect AUs: expression channels we care about
AFFECT_AUS = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]
SPEECH_AUS = [34, 35]  # mouthLowerDown L/R — best speech discriminators
SMILE_AUS = [44, 45]

# Default CWT parameters (Morlet wavelet)
DEFAULT_FREQS = np.logspace(np.log10(0.3), np.log10(8.0), 30)
MORLET_OMEGA = 5.0


# ── Dataclasses ────────────────────────────────────────────────────────

@dataclass
class AUScalogram:
    """CWT decomposition of blendshape AU timeseries."""
    coeffs: np.ndarray          # (n_freqs, T, n_ch) complex CWT coefficients
    power: np.ndarray           # (n_freqs, T, n_ch) |coeffs|^2
    freqs: np.ndarray           # (n_freqs,) frequency axis
    fs: float                   # sampling rate
    n_ch: int                   # number of AU channels
    # Band-extracted power (summed over frequencies in band)
    state_power: np.ndarray     # (T, n_ch) — <0.5 Hz
    expression_power: np.ndarray  # (T, n_ch) — 0.5-2 Hz
    speech_power: np.ndarray    # (T, n_ch) — 2-7 Hz


# ── CWT computation ───────────────────────────────────────────────────

def compute_au_cwt(signal, fs=30.0, freqs=None, omega=MORLET_OMEGA,
                   device='auto'):
    """Continuous wavelet transform of all AU channels.

    Vectorized FFT-based Morlet convolution. GPU-accelerated when available.

    Args:
        signal: (T, n_ch) blendshape array (already low-pass filtered).
        fs: sampling rate.
        freqs: frequency axis. Default: 30 log-spaced from 0.3 to 8 Hz.
        omega: Morlet wavelet parameter (frequency resolution vs time resolution).
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        AUScalogram with CWT coefficients, power, and band-extracted arrays.
    """
    if freqs is None:
        freqs = DEFAULT_FREQS

    T, n_ch = signal.shape
    n_freqs = len(freqs)

    use_gpu = (device == 'cuda' or (device == 'auto' and _HAS_TORCH))

    if use_gpu:
        coeffs = _cwt_gpu(signal, fs, freqs, omega)
    else:
        coeffs = _cwt_cpu(signal, fs, freqs, omega)

    power = np.abs(coeffs) ** 2

    # Extract band power by summing over frequencies in each band
    state_power = _band_power(power, freqs, *BAND_STATE)
    expression_power = _band_power(power, freqs, *BAND_EXPRESSION)
    speech_power = _band_power(power, freqs, *BAND_SPEECH)

    return AUScalogram(
        coeffs=coeffs, power=power, freqs=freqs, fs=fs, n_ch=n_ch,
        state_power=state_power, expression_power=expression_power,
        speech_power=speech_power,
    )


def _cwt_cpu(signal, fs, freqs, omega):
    """FFT-based Morlet CWT on CPU, vectorized across all channels."""
    T, n_ch = signal.shape
    n_freqs = len(freqs)

    # Remove DC per channel
    x = signal.astype(np.float64) - signal.mean(axis=0, keepdims=True)

    # FFT all channels at once: (T, n_ch) -> (T, n_ch)
    X = np.fft.fft(x, axis=0)
    f_fft = np.fft.fftfreq(T, d=1.0 / fs)  # (T,)

    coeffs = np.empty((n_freqs, T, n_ch), dtype=np.complex128)

    for i, fc in enumerate(freqs):
        # Morlet wavelet in frequency domain: Gaussian centered at fc
        sigma_f = fc / omega
        W = np.exp(-0.5 * ((f_fft - fc) / sigma_f) ** 2)  # (T,)
        # Multiply and inverse FFT — broadcast W across channels
        coeffs[i] = np.fft.ifft(X * W[:, np.newaxis], axis=0)

    return coeffs.astype(np.complex64)


def _cwt_gpu(signal, fs, freqs, omega):
    """FFT-based Morlet CWT on GPU via torch, vectorized across all channels
    AND all frequencies simultaneously."""
    T, n_ch = signal.shape
    n_freqs = len(freqs)

    x = torch.tensor(signal, dtype=torch.float64, device='cuda')
    x = x - x.mean(dim=0, keepdim=True)

    # FFT all channels: (T, n_ch)
    X = torch.fft.fft(x, dim=0)
    f_fft = torch.fft.fftfreq(T, d=1.0 / fs, device='cuda')  # (T,)

    freqs_t = torch.tensor(freqs, dtype=torch.float64, device='cuda')  # (n_freqs,)
    sigma_f = freqs_t / omega  # (n_freqs,)

    # Vectorize across frequencies: (n_freqs, T) Gaussian kernels
    # f_fft: (T,) -> (1, T), freqs: (n_freqs, 1)
    W = torch.exp(-0.5 * ((f_fft[None, :] - freqs_t[:, None]) / sigma_f[:, None]) ** 2)

    # Multiply: (n_freqs, T, 1) * (1, T, n_ch) -> (n_freqs, T, n_ch)
    XW = X[None, :, :] * W[:, :, None]

    # Inverse FFT all at once
    coeffs = torch.fft.ifft(XW, dim=1)

    return coeffs.cpu().numpy().astype(np.complex64)


def _band_power(power, freqs, f_lo, f_hi):
    """Sum power over frequencies within a band.

    Args:
        power: (n_freqs, T, n_ch)
        freqs: (n_freqs,)
        f_lo, f_hi: band edges (Hz)

    Returns:
        (T, n_ch) band-summed power
    """
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if mask.sum() == 0:
        return np.zeros(power.shape[1:], dtype=np.float32)
    return power[mask].sum(axis=0).astype(np.float32)


# ── Speech detection from CWT ─────────────────────────────────────────

def detect_speech(scalogram, noise_floor=0.00002, smooth_s=0.5):
    """Detect speech from CWT speech-band energy on mouthLowerDown L/R.

    Replaces the bandpass+Hilbert speech detector with CWT-native equivalent.
    The scalogram already contains the speech-band power — no redundant
    filtering needed.

    Args:
        scalogram: AUScalogram from compute_au_cwt().
        noise_floor: minimum speech-band energy to consider as speech.
        smooth_s: smoothing on speech probability (seconds).

    Returns:
        speech_prob: (T,) float32, [0, 1].
    """
    # Speech-band energy on AU34 + AU35
    speech_energy = scalogram.speech_power[:, 34] + scalogram.speech_power[:, 35]

    if smooth_s > 0:
        speech_energy = gaussian_filter1d(speech_energy, sigma=smooth_s * scalogram.fs)

    # Threshold using peak-relative scaling with noise floor
    p95 = np.percentile(speech_energy, 95)
    if p95 < noise_floor:
        return np.zeros(len(speech_energy), dtype=np.float32)

    threshold = max(0.15 * p95, noise_floor)
    prob = np.clip((speech_energy - threshold) / (p95 - threshold + 1e-8), 0.0, 1.0)

    return prob.astype(np.float32)


# ── Expression event detection from CWT ────────────────────────────────

def detect_expression_events(scalogram, fs=30.0, prominence=0.03,
                             min_iei_s=1.0, lsl_start=0.0,
                             speech_prob=None):
    """Detect expression events from CWT expression-band energy on affect AUs.

    The expression band (0.5-2 Hz) naturally excludes speech (>2 Hz) and
    tonic drift (<0.5 Hz). No hand-crafted gating needed.

    Args:
        scalogram: AUScalogram from compute_au_cwt().
        fs: sampling rate.
        prominence: min prominence for peak detection.
        min_iei_s: min inter-event interval.
        lsl_start: LSL timestamp of segment start.
        speech_prob: (T,) speech probability for annotation (not used for gating).

    Returns:
        list of dicts with event info.
    """
    # Sum expression-band power across affect AUs
    expr_energy = sum(scalogram.expression_power[:, au] for au in AFFECT_AUS)

    # Smooth lightly
    expr_energy = gaussian_filter1d(expr_energy, sigma=0.1 * fs)

    pks, _ = find_peaks(expr_energy, prominence=prominence,
                        distance=int(min_iei_s * fs))

    # Phasic smile: expression-band power on smile AUs (already tonic-free)
    smile_phasic_ts = (scalogram.expression_power[:, 44]
                       + scalogram.expression_power[:, 45])
    # Smile velocity from expression-band smile signal
    smile_vel_ts = gaussian_filter1d(
        np.diff(smile_phasic_ts, prepend=smile_phasic_ts[0]), sigma=0.1 * fs)

    events = []
    for pk in pks:
        sp = float(speech_prob[pk]) if speech_prob is not None else 0.0
        events.append({
            'time': pk / fs,
            'lsl_time': lsl_start + pk / fs,
            'energy': float(expr_energy[pk]),
            'smile_phasic': float(smile_phasic_ts[pk]),
            'smile_velocity': float(smile_vel_ts[pk]),
            'speech_prob': sp,
        })

    return events


# ── Wavelet coherence ──────────────────────────────────────────────────

def _coherence_from_coeffs_gpu(w1, w2, aus, sigma_samples):
    """GPU-accelerated coherence from pre-computed CWT coefficients.

    Args:
        w1, w2: (n_freqs, T, n_ch) complex CWT coefficients.
        aus: list of AU indices to sum over.
        sigma_samples: smoothing kernel width in samples.

    Returns:
        coherence: (n_freqs, T) float32
        phase: (n_freqs, T) float32
    """
    n_freqs, T, _ = w1.shape

    # Move to GPU, select AUs
    c1 = torch.tensor(w1[:, :, aus], dtype=torch.complex64, device='cuda')  # (F, T, A)
    c2 = torch.tensor(w2[:, :, aus], dtype=torch.complex64, device='cuda')

    # Cross-spectrum and auto-spectra, summed over AUs
    cross = (c1 * c2.conj()).sum(dim=2)     # (F, T)
    auto1 = (c1.abs() ** 2).sum(dim=2)      # (F, T)
    auto2 = (c2.abs() ** 2).sum(dim=2)      # (F, T)

    # Gaussian smoothing in time via 1D convolution
    kernel_size = int(6 * sigma_samples) | 1  # odd
    t = torch.arange(kernel_size, device='cuda', dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)  # (1, 1, K)
    pad = kernel_size // 2

    def smooth(x):
        # x: (F, T) complex or real → treat as (F, 1, T) for conv1d
        if x.is_complex():
            real = torch.nn.functional.conv1d(x.real.unsqueeze(1), kernel, padding=pad).squeeze(1)
            imag = torch.nn.functional.conv1d(x.imag.unsqueeze(1), kernel, padding=pad).squeeze(1)
            return torch.complex(real, imag)
        return torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=pad).squeeze(1)

    cross_s = smooth(cross)
    auto1_s = smooth(auto1)
    auto2_s = smooth(auto2)

    coherence = cross_s.abs() ** 2 / (auto1_s * auto2_s + 1e-10)
    phase = torch.angle(cross_s)

    return coherence.cpu().numpy().astype(np.float32), phase.cpu().numpy().astype(np.float32)


def _coherence_from_coeffs_cpu(w1, w2, aus, sigma_samples):
    """CPU coherence from pre-computed CWT coefficients."""
    cross = np.zeros((w1.shape[0], w1.shape[1]), dtype=np.complex128)
    auto1 = np.zeros_like(cross, dtype=np.float64)
    auto2 = np.zeros_like(cross, dtype=np.float64)

    for au in aus:
        c1 = w1[:, :, au]
        c2 = w2[:, :, au]
        cross += c1 * np.conj(c2)
        auto1 += np.abs(c1) ** 2
        auto2 += np.abs(c2) ** 2

    cross_s = gaussian_filter1d(cross, sigma=sigma_samples, axis=1)
    auto1_s = gaussian_filter1d(auto1, sigma=sigma_samples, axis=1)
    auto2_s = gaussian_filter1d(auto2, sigma=sigma_samples, axis=1)

    coherence = (np.abs(cross_s) ** 2 / (auto1_s * auto2_s + 1e-10))
    phase = np.angle(cross_s)

    return coherence.astype(np.float32), phase.astype(np.float32)


def wavelet_coherence(scal_p1, scal_p2, smooth_s=0.5, au_groups=None,
                      device='auto'):
    """Cross-wavelet coherence between two participants.

    GPU-accelerated when available. Computes time-frequency coherence per AU
    group, giving a multi-scale coupling profile.

    Args:
        scal_p1, scal_p2: AUScalogram from compute_au_cwt().
        smooth_s: temporal smoothing for coherence estimation (seconds).
        au_groups: dict of group_name -> AU index list.
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        dict mapping group names to {'coherence': (F,T), 'phase': (F,T)}
    """
    if au_groups is None:
        au_groups = {'affect': AFFECT_AUS, 'speech': SPEECH_AUS}

    sigma = smooth_s * scal_p1.fs
    use_gpu = (device == 'cuda' or (device == 'auto' and _HAS_TORCH))
    compute_fn = _coherence_from_coeffs_gpu if use_gpu else _coherence_from_coeffs_cpu

    results = {}
    for group_name, aus in au_groups.items():
        coh, phase = compute_fn(scal_p1.coeffs, scal_p2.coeffs, aus, sigma)
        results[group_name] = {'coherence': coh, 'phase': phase}

    return results


# ── Surrogate-based z-scored coherence ─────────────────────────────────

def surrogate_coherence_z(scal_p1, scal_p2, n_surrogates=200,
                          smooth_s=0.5, aus=None, seed=42,
                          device='auto'):
    """Z-score wavelet coherence against circular-shift surrogates.

    Circular-shifts P2's CWT coefficients in time (preserving spectral
    structure), recomputes coherence for each surrogate, and z-scores
    the real coherence against the surrogate null distribution.

    GPU-accelerated: all surrogates computed in a single batched operation.

    Args:
        scal_p1, scal_p2: AUScalogram from compute_au_cwt().
        n_surrogates: number of circular shift surrogates.
        smooth_s: temporal smoothing for coherence.
        aus: AU indices for coherence. Default: AFFECT_AUS.
        seed: random seed.
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        dict with:
            'z': (n_freqs, T) z-scored coherence
            'real_coh': (n_freqs, T) real coherence
            'null_mean': (n_freqs, T) surrogate mean
            'null_std': (n_freqs, T) surrogate std
            'band_z': dict of band_name -> mean z-score in band
    """
    if aus is None:
        aus = AFFECT_AUS

    use_gpu = (device == 'cuda' or (device == 'auto' and _HAS_TORCH))
    sigma = smooth_s * scal_p1.fs
    freqs = scal_p1.freqs
    n_freqs, T, _ = scal_p1.coeffs.shape

    rng = np.random.default_rng(seed)
    min_shift = int(0.1 * T)
    shifts = rng.integers(min_shift, T - min_shift, size=n_surrogates)

    if use_gpu:
        real_coh, null_mean, null_std = _surrogate_z_gpu(
            scal_p1.coeffs, scal_p2.coeffs, aus, sigma, shifts)
    else:
        real_coh, null_mean, null_std = _surrogate_z_cpu(
            scal_p1.coeffs, scal_p2.coeffs, aus, sigma, shifts)

    z = (real_coh - null_mean) / (null_std + 1e-8)

    # Per-band mean z
    bands = {'state': BAND_STATE, 'expression': BAND_EXPRESSION, 'speech': BAND_SPEECH}
    band_z = {}
    for bname, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.sum() > 0:
            band_z[bname] = float(z[mask].mean())
        else:
            band_z[bname] = 0.0

    return {
        'z': z.astype(np.float32),
        'real_coh': real_coh.astype(np.float32),
        'null_mean': null_mean.astype(np.float32),
        'null_std': null_std.astype(np.float32),
        'band_z': band_z,
    }


def _surrogate_z_gpu(w1, w2, aus, sigma_samples, shifts):
    """GPU-batched surrogate coherence z-scoring.

    Computes real coherence + all surrogates in one pass on GPU.
    """
    n_freqs, T, _ = w1.shape
    n_surr = len(shifts)

    # Select AUs and move to GPU
    c1 = torch.tensor(w1[:, :, aus], dtype=torch.complex64, device='cuda')  # (F, T, A)
    c2 = torch.tensor(w2[:, :, aus], dtype=torch.complex64, device='cuda')

    # Pre-compute P1 auto-spectrum (doesn't change across surrogates)
    auto1 = (c1.abs() ** 2).sum(dim=2)  # (F, T)

    # Smoothing kernel
    kernel_size = int(6 * sigma_samples) | 1
    t = torch.arange(kernel_size, device='cuda', dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    pad = kernel_size // 2

    def smooth_real(x):
        return torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=pad).squeeze(1)

    def smooth_complex(x):
        r = torch.nn.functional.conv1d(x.real.unsqueeze(1), kernel, padding=pad).squeeze(1)
        i = torch.nn.functional.conv1d(x.imag.unsqueeze(1), kernel, padding=pad).squeeze(1)
        return torch.complex(r, i)

    auto1_s = smooth_real(auto1)

    # Real coherence
    cross_real = (c1 * c2.conj()).sum(dim=2)
    auto2_real = (c2.abs() ** 2).sum(dim=2)
    cross_real_s = smooth_complex(cross_real)
    auto2_real_s = smooth_real(auto2_real)
    real_coh = cross_real_s.abs() ** 2 / (auto1_s * auto2_real_s + 1e-10)

    # Surrogates: circular-shift c2 in time, recompute coherence
    # Accumulate mean and M2 (Welford's) on GPU to avoid storing all surrogates
    surr_sum = torch.zeros_like(real_coh)
    surr_sum_sq = torch.zeros_like(real_coh)

    for shift in shifts:
        c2_shifted = torch.roll(c2, int(shift), dims=1)
        cross_s = smooth_complex((c1 * c2_shifted.conj()).sum(dim=2))
        auto2_s = smooth_real((c2_shifted.abs() ** 2).sum(dim=2))
        surr_coh = cross_s.abs() ** 2 / (auto1_s * auto2_s + 1e-10)
        surr_sum += surr_coh
        surr_sum_sq += surr_coh ** 2

    null_mean = surr_sum / n_surr
    null_std = torch.sqrt(surr_sum_sq / n_surr - null_mean ** 2 + 1e-10)

    return (real_coh.cpu().numpy(), null_mean.cpu().numpy(), null_std.cpu().numpy())


def _surrogate_z_cpu(w1, w2, aus, sigma_samples, shifts):
    """CPU surrogate coherence z-scoring."""
    n_freqs, T, _ = w1.shape
    n_surr = len(shifts)

    # Real coherence
    real_coh, _ = _coherence_from_coeffs_cpu(w1, w2, aus, sigma_samples)

    # Surrogates
    surr_sum = np.zeros_like(real_coh, dtype=np.float64)
    surr_sum_sq = np.zeros_like(real_coh, dtype=np.float64)

    for shift in shifts:
        w2_shifted = np.roll(w2, int(shift), axis=1)
        surr_coh, _ = _coherence_from_coeffs_cpu(w1, w2_shifted, aus, sigma_samples)
        surr_sum += surr_coh
        surr_sum_sq += surr_coh ** 2

    null_mean = surr_sum / n_surr
    null_std = np.sqrt(surr_sum_sq / n_surr - null_mean ** 2 + 1e-10)

    return real_coh, null_mean.astype(np.float32), null_std.astype(np.float32)


# ── Band-averaged coherence summary ────────────────────────────────────

def coherence_band_summary(coh_result, freqs):
    """Compute mean coherence per frequency band per AU group.

    Args:
        coh_result: dict from wavelet_coherence().
        freqs: frequency axis.

    Returns:
        dict mapping group names to band-level summaries.
    """
    bands = {
        'state': BAND_STATE,
        'expression': BAND_EXPRESSION,
        'speech': BAND_SPEECH,
    }

    summary = {}
    for group_name, group_data in coh_result.items():
        coh = group_data['coherence']  # (n_freqs, T)
        group_summary = {}
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.sum() == 0:
                group_summary[band_name] = {'mean': 0.0, 'max': 0.0}
                continue
            band_coh = coh[mask].mean(axis=0)  # (T,) mean across freqs
            group_summary[band_name] = {
                'mean': float(band_coh.mean()),
                'max': float(band_coh.max()),
                'timecourse': band_coh.astype(np.float32),
            }
        summary[group_name] = group_summary

    return summary
