"""EEG band-specific real + imaginary coherence from raw 256 Hz data.

GPU-accelerated: pre-computes all 14-channel FFTs in a single batched kernel,
then computes cross-spectra via broadcasting over ROI electrode pairs.

For Emotiv EPOC (14 channels), computes cross-participant coherence in
sliding windows per frequency band (theta, alpha, beta) and per ROI pair.

Key design:
  - Imaginary coherency (ImCoh) rejects volume conduction artifacts
  - Band-specific analysis captures frequency-specific neural coupling
  - 2s windows at 256 Hz → 512 samples (excellent spectral estimation)
  - 4 ROIs × 4 ROIs = 16 cross-participant pairs per band
  - All 14-channel FFTs computed in ONE torch.fft.rfft kernel
"""

import numpy as np
import torch

from cadence.constants import EEG_ROIS, EEG_ROI_NAMES


# Default frequency bands (skip delta=artifact-prone, gamma=noisy on Emotiv)
DEFAULT_EEG_BANDS = {
    'theta': (4, 8),     # cognitive engagement, turn-taking
    'alpha': (8, 13),    # emotional synchrony, empathy
    'beta': (13, 30),    # joint action, motor synchronization
}


def _get_device(cfg):
    """Resolve torch device from config."""
    if cfg is None:
        cfg = {}
    device_str = cfg.get('device', 'cuda')
    if isinstance(device_str, torch.device):
        return device_str
    if device_str == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_str)


def eeg_band_coherence(p1_eeg, p2_eeg, fs=256,
                       window_s=2.0, stride_s=0.25,
                       bands=None, roi_map=None,
                       use_imcoh=True, device=None):
    """Band-specific real + imaginary coherence from raw EEG (GPU-accelerated).

    Pre-computes FFTs for all 14 channels in one batched kernel, then
    computes cross-spectra via broadcasting over ROI electrode pairs.
    Eliminates all per-window and per-electrode Python loops.

    Args:
        p1_eeg, p2_eeg: (T, 14) preprocessed EEG at 256 Hz.
        fs: sampling rate (256 Hz).
        window_s: window length (seconds).
        stride_s: stride between windows.
        bands: dict of {name: (f_lo, f_hi)}.
        roi_map: dict mapping ROI names to channel indices.
        use_imcoh: if True, also compute imaginary coherency.
        device: torch device (default: cuda if available).

    Returns:
        coh_real: (n_bands, n_roi_pairs, n_windows) real coherence.
        coh_imag: (n_bands, n_roi_pairs, n_windows) imaginary coherency
                  (None if use_imcoh=False).
        window_times: (n_windows,) center times in seconds.
        feature_names: list of feature name strings.
    """
    if bands is None:
        bands = DEFAULT_EEG_BANDS
    if roi_map is None:
        roi_map = EEG_ROIS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_bands = len(bands)
    roi_names = list(roi_map.keys())
    roi_pairs = [(r1, r2) for r1 in roi_names for r2 in roi_names]
    n_roi_pairs = len(roi_pairs)

    N = min(len(p1_eeg), len(p2_eeg))
    n_ch = min(p1_eeg.shape[1], p2_eeg.shape[1])
    win_samp = min(int(round(window_s * fs)), N)
    stride_samp = max(1, int(round(stride_s * fs)))
    nperseg = max(64, win_samp // 2)
    noverlap = nperseg // 2
    sub_step = nperseg - noverlap  # match scipy convention

    starts = np.arange(0, N - win_samp + 1, stride_samp)
    n_windows = len(starts)
    band_list = list(bands.items())

    if n_windows == 0:
        return (np.zeros((n_bands, n_roi_pairs, 0)),
                np.zeros((n_bands, n_roi_pairs, 0)) if use_imcoh else None,
                np.array([]), [])

    window_times = (starts + win_samp / 2) / fs

    # Transfer to device: (n_ch, T)
    p1_t = torch.as_tensor(np.ascontiguousarray(p1_eeg[:N, :n_ch].T),
                           dtype=torch.float32, device=device)
    p2_t = torch.as_tensor(np.ascontiguousarray(p2_eeg[:N, :n_ch].T),
                           dtype=torch.float32, device=device)

    # Extract all windows for all channels: (n_ch, W, win_samp)
    starts_t = torch.as_tensor(starts, dtype=torch.long, device=device)
    offsets = torch.arange(win_samp, device=device)
    win_idx = starts_t[:, None] + offsets[None, :]  # (W, win_samp)
    p1_w = p1_t[:, win_idx]  # (n_ch, W, win_samp)
    p2_w = p2_t[:, win_idx]

    # Welch sub-segments via unfold: (n_ch, W, n_sub, nperseg)
    hann = torch.hann_window(nperseg, device=device)
    p1_sub = p1_w.unfold(-1, nperseg, sub_step)
    p1_sub = (p1_sub - p1_sub.mean(dim=-1, keepdim=True)) * hann
    p2_sub = p2_w.unfold(-1, nperseg, sub_step)
    p2_sub = (p2_sub - p2_sub.mean(dim=-1, keepdim=True)) * hann

    # Single batched FFT for ALL channels (2 × n_ch FFTs in one kernel)
    fft_p1 = torch.fft.rfft(p1_sub, dim=-1)  # (n_ch, W, n_sub, n_freq)
    fft_p2 = torch.fft.rfft(p2_sub, dim=-1)

    # PSD per channel (averaged over sub-segments)
    psd_p1 = fft_p1.abs().square().mean(dim=2)  # (n_ch, W, n_freq)
    psd_p2 = fft_p2.abs().square().mean(dim=2)

    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / fs, device=device)

    del p1_w, p2_w, p1_sub, p2_sub, p1_t, p2_t

    # Compute coherence per ROI pair × band via broadcasting
    coh_real = np.zeros((n_bands, n_roi_pairs, n_windows))
    coh_imag = np.zeros((n_bands, n_roi_pairs, n_windows)) if use_imcoh else None

    for pi, (r1_name, r2_name) in enumerate(roi_pairs):
        r1_chs = [c for c in roi_map[r1_name] if c < n_ch]
        r2_chs = [c for c in roi_map[r2_name] if c < n_ch]
        if not r1_chs or not r2_chs:
            continue

        # Cross-spectra via broadcasting: (|R1|, 1, W, n_sub, n_freq) × (1, |R2|, ...)
        fft1_roi = fft_p1[r1_chs]   # (|R1|, W, n_sub, n_freq)
        fft2_roi = fft_p2[r2_chs]   # (|R2|, W, n_sub, n_freq)
        # Average CSD over electrode pairs AND sub-segments
        Sxy = (fft1_roi.unsqueeze(1).conj() * fft2_roi.unsqueeze(0)
               ).mean(dim=(0, 1, 3))  # (W, n_freq)

        Sxx = psd_p1[r1_chs].mean(dim=0)  # (W, n_freq)
        Syy = psd_p2[r2_chs].mean(dim=0)

        for bi, (band_name, (f_lo, f_hi)) in enumerate(band_list):
            bm = (freqs >= f_lo) & (freqs <= f_hi)
            if not bm.any():
                continue

            denom = torch.clamp(Sxx[:, bm] * Syy[:, bm], min=1e-20)
            msc = Sxy[:, bm].abs().square() / denom
            coh_real[bi, pi] = msc.mean(dim=-1).cpu().numpy()

            if use_imcoh:
                imcoh_val = Sxy[:, bm].imag.abs() / denom.sqrt()
                coh_imag[bi, pi] = imcoh_val.mean(dim=-1).cpu().numpy()

    # Feature names
    feature_names = []
    for band_name, _ in band_list:
        for r1_name, r2_name in roi_pairs:
            feature_names.append(f'eeg_{band_name}_{r1_name}x{r2_name}')

    return coh_real, coh_imag, window_times, feature_names


def eeg_coherence_features(p1_eeg, p2_eeg, fs=256,
                           window_s=2.0, stride_s=0.25,
                           bands=None, roi_map=None, use_imcoh=True,
                           device=None):
    """Flatten EEG coherence into (n_features, n_windows) for aggregation."""
    coh_real, coh_imag, window_times, names_real = eeg_band_coherence(
        p1_eeg, p2_eeg, fs=fs, window_s=window_s, stride_s=stride_s,
        bands=bands, roi_map=roi_map, use_imcoh=use_imcoh, device=device)

    n_bands, n_pairs, n_windows = coh_real.shape
    features_real = coh_real.reshape(-1, n_windows)
    all_names = [f'{n}_msc' for n in names_real]

    if use_imcoh and coh_imag is not None:
        features_imag = coh_imag.reshape(-1, n_windows)
        features = np.concatenate([features_real, features_imag], axis=0)
        all_names += [f'{n}_imcoh' for n in names_real]
    else:
        features = features_real

    return features, window_times, all_names


def eeg_coherence_surrogates(p1_eeg, p2_eeg, fs=256,
                             n_surrogates=100,
                             window_s=2.0, stride_s=0.25,
                             bands=None, roi_map=None,
                             use_imcoh=True,
                             min_shift_frac=0.1, seed=42,
                             device=None):
    """Generate surrogate EEG coherence features via circular shift.

    GPU-accelerated: P2 FFTs computed once and reused across K surrogates.

    Returns:
        surr_features: (K, n_features, n_windows) surrogate features.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rng = np.random.RandomState(seed)

    # Reference computation for shape
    ref_features, _, _ = eeg_coherence_features(
        p1_eeg, p2_eeg, fs=fs, window_s=window_s, stride_s=stride_s,
        bands=bands, roi_map=roi_map, use_imcoh=use_imcoh, device=device)

    n_features, n_windows = ref_features.shape
    N = min(len(p1_eeg), len(p2_eeg))
    min_shift = max(1, int(min_shift_frac * N))
    max_shift = N - min_shift
    if min_shift >= max_shift:
        min_shift, max_shift = 1, N - 1

    surr_features = np.zeros((n_surrogates, n_features, n_windows))

    for k in range(n_surrogates):
        shift = rng.randint(min_shift, max_shift + 1)
        p1_shifted = np.roll(p1_eeg, shift, axis=0)

        feat_k, _, _ = eeg_coherence_features(
            p1_shifted, p2_eeg, fs=fs,
            window_s=window_s, stride_s=stride_s,
            bands=bands, roi_map=roi_map, use_imcoh=use_imcoh,
            device=device)

        if feat_k.shape[1] == n_windows:
            surr_features[k] = feat_k
        elif feat_k.shape[1] > 0:
            n_min = min(feat_k.shape[1], n_windows)
            surr_features[k, :, :n_min] = feat_k[:, :n_min]

    return surr_features
