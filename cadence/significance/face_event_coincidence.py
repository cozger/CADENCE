"""Facial-activity event coincidence — dyadic behavioral coupling channel.

Replaces ``bl_expr`` (Morlet wavelet coherence — fires on near-stationary
z-score noise during quiet periods, missing the actual behavioral signal)
and ``bl_activity_conc`` (``(z_P1 + z_P2) / 2`` — shared activity LEVEL,
not coupling — confounded by who is speaking).

Pipeline:

  1. Per participant, take the precomputed ``au_activity`` envelope
     (RMS deviation from trailing 30s mean — already in face preproc).
     This signal directly tracks "moments of facial movement" without
     the per-AU-z-score drift artifact that breaks edge detection on
     individual AUs.
  2. Find peaks above the per-session 70th-percentile threshold with
     min-separation 1s. Each peak = one facial-activity event.
     Per-session quantile makes the threshold robust to inter-participant
     baseline-activity differences.
  3. Build per-participant binary peak grid at 2 Hz.
  4. Coincidence = both have a peak within ±tau samples (±500ms default).
  5. Z-score against 200 circular-shift surrogates of P2's peak grid
     (Welford accumulation, mirrors ``burst_coincidence.py``).

Output channel: ``bl_event_coincidence`` — continuous z-score at 2 Hz.

Empirical sanity (y_06):
- Per-participant peak rates: base_EC=23/min, meditation=25/min,
  conv=52–55/min. Patient meditation matches eyes-closed baseline (correct).
- Earlier multiscale-AU-edge approach (inspired by Mallat & Hwang 1992)
  was rejected because per-AU z-scored signals produce spurious edges
  during quiet periods (eyes-closed meditation showed 26 events/min when
  base_EC showed only 0.4/min).

The full multiscale-event-detection + Hölder regularity + UMAP+HDBSCAN
"synchrony repertoire" framework is part of the proposed grant work
(Aim 2). This MVP channel is the activity-level Stage 1 stepping stone.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# AFFECT_AUS — 10 expression-relevant blendshapes (cheekSquintL/R,
# mouthDimpleL/R, mouthFrownL/R, mouthSmileL/R, noseSneerL/R)
AFFECT_AUS = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]


def _smooth_savgol(x, window=5, poly=2):
    """Savitzky–Golay smoothing along axis 0; safe on short arrays."""
    if len(x) < window:
        return x
    return savgol_filter(x, window, poly, axis=0)


def _baseline_subtract(x, fs, window_s=5.0, q=10):
    """Subtract rolling q-th percentile baseline (per-channel)."""
    win = max(3, int(round(window_s * fs)))
    if len(x) < win:
        return x - np.percentile(x, q, axis=0, keepdims=True)

    # Stride trick over a sliding window is overkill — np.percentile in
    # a Python loop with stride is fine for a few hundred channels at 30fps
    # and a few tens of thousands of frames.
    half_l = win // 2
    half_r = win - half_l - 1
    pad = np.pad(x, ((half_l, half_r), (0, 0)), mode='edge')
    # Quantile over a strided view via np.lib.stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(pad, window_shape=win, axis=0)  # (T, C, win)
    base = np.percentile(sw, q, axis=-1)
    return x - base


def detect_au_events(channel_signal, fs, scales_seconds=None,
                      noise_thresh_factor=3.0, min_chain_frac=0.5,
                      noise_floor_override=None, return_extra=False):
    """Multiscale derivative-of-Gaussian event detection on one AU channel.

    Args:
        channel_signal: (T,) preprocessed (smoothed, baseline-subtracted) signal.
        fs: sampling rate (Hz).
        scales_seconds: list of σ values in seconds. Default = dyadic
            [0.05, 0.1, 0.2, 0.4, 0.8, 1.6] s (≈ 1.5 to 48 frames at 30fps).
        noise_thresh_factor: per-channel noise floor = ``factor × MAD``
            of the smallest-scale derivative response.
        min_chain_frac: keep chains spanning ≥ this fraction of scales.
        noise_floor_override: if not None, use this absolute value as the
            noise floor (overriding the MAD-based default). Used by callers
            that estimate the floor from explicit baseline frames.
        return_extra: if True, additionally return Hölder regularity α and
            chain length per event. α is the slope of log|amp| vs log σ
            along the chain (Mallat & Hwang 1992: |W_f(s, t)| ∝ s^α near a
            singularity). Sharp jump → α≈0; ramp → α≈1; over-smooth (>1.5)
            or noise spike (<−0.5) → flagged and dropped.

    Returns:
        If return_extra=False (legacy):
            events: (n_events,) array of event timestamps (s, relative to start)
            amps:   (n_events,) array of peak |response| amplitudes
        If return_extra=True:
            events, amps, alphas, chain_lens
                alphas:     (n_events,) Hölder regularity slope
                chain_lens: (n_events,) int8 — # scales the chain spanned
    """
    T = len(channel_signal)
    if scales_seconds is None:
        scales_seconds = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    sigmas = [max(1.0, s * fs) for s in scales_seconds]
    n_scales = len(sigmas)

    # Multiscale 1st-derivative-of-Gaussian response: shape (n_scales, T)
    resp = np.empty((n_scales, T), dtype=np.float64)
    for j, sigma in enumerate(sigmas):
        # order=1 → first derivative; mode='reflect' avoids edge artifacts
        # Multiply by sigma so amplitude is comparable across scales (Mallat normalization)
        resp[j] = gaussian_filter1d(channel_signal, sigma=sigma, order=1,
                                     mode='reflect') * sigma

    return chain_link_from_responses(resp, sigmas, fs,
                                       noise_thresh_factor=noise_thresh_factor,
                                       min_chain_frac=min_chain_frac,
                                       noise_floor_override=noise_floor_override,
                                       return_extra=return_extra)


def chain_link_from_responses(resp, sigmas, fs, noise_thresh_factor=3.0,
                                min_chain_frac=0.5, noise_floor_override=None,
                                return_extra=False):
    """Run chain-linking + Hölder α extraction on a precomputed response pyramid.

    Factored out of ``detect_au_events`` so a GPU-batched DoG pyramid (see
    ``cadence.synchrony._gpu.batched_dog_pyramid``) can feed in
    pre-convolved responses without paying for per-channel scipy convolution.

    Args:
        resp: (n_scales, T) σ-normalized DoG response array.
        sigmas: per-scale σ in samples (frames). len == n_scales.
        fs: sampling rate (Hz) — used to convert event indices to seconds.
        Other args identical to ``detect_au_events``.

    Returns: same shape as ``detect_au_events``.
    """
    n_scales, T = resp.shape

    if noise_floor_override is not None:
        noise_floor = max(float(noise_floor_override), 1e-6)
    else:
        finest = np.abs(resp[0])
        mad = 1.4826 * np.median(np.abs(finest - np.median(finest)))
        noise_floor = max(noise_thresh_factor * mad, 1e-6)

    max_per_scale = []
    for j in range(n_scales):
        a = np.abs(resp[j])
        if T < 3:
            max_per_scale.append(np.array([], dtype=int))
            continue
        is_max = (a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] > noise_floor)
        idx = np.where(is_max)[0] + 1
        max_per_scale.append(idx)

    empty_extra = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8))
    empty_legacy = (np.zeros(0), np.zeros(0))
    if all(len(m) == 0 for m in max_per_scale):
        return empty_extra if return_extra else empty_legacy
    if max_per_scale[-1].size == 0:
        return empty_extra if return_extra else empty_legacy

    log_sigmas = np.log(np.asarray(sigmas, dtype=np.float64))
    chains = []
    for t_coarse in max_per_scale[-1]:
        chain_t = [None] * n_scales
        chain_t[-1] = int(t_coarse)
        for j in range(n_scales - 2, -1, -1):
            if max_per_scale[j].size == 0:
                break
            tol = max(2.0, 2.0 * sigmas[j + 1])
            distances = np.abs(max_per_scale[j] - chain_t[j + 1])
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= tol:
                chain_t[j] = int(max_per_scale[j][best_idx])
            else:
                break
        spanned = sum(1 for c in chain_t if c is not None)
        if spanned / n_scales < min_chain_frac:
            continue

        finest_idx = next(j for j, c in enumerate(chain_t) if c is not None)
        t_event = chain_t[finest_idx]
        peak_amp = max(np.abs(resp[j, chain_t[j]])
                        for j in range(n_scales) if chain_t[j] is not None)

        chain_amps = np.array([np.abs(resp[j, chain_t[j]])
                                for j in range(n_scales) if chain_t[j] is not None],
                               dtype=np.float64)
        chain_log_sigmas = log_sigmas[[j for j in range(n_scales)
                                         if chain_t[j] is not None]]
        if len(chain_amps) >= 2 and (chain_amps > 0).all():
            slope, _ = np.polyfit(chain_log_sigmas, np.log(chain_amps), 1)
            alpha = float(slope)
        else:
            alpha = np.nan
        if not np.isfinite(alpha):
            continue
        if alpha < -0.5 or alpha > 1.5:
            continue

        chains.append((t_event, peak_amp, alpha, spanned))

    if not chains:
        return empty_extra if return_extra else empty_legacy

    dedup = {}
    for t, a, alpha, span in chains:
        if t not in dedup or a > dedup[t][0]:
            dedup[t] = (a, alpha, span)
    events_idx = np.array(sorted(dedup.keys()), dtype=int)
    amps = np.array([dedup[t][0] for t in events_idx], dtype=np.float64)
    alphas = np.array([dedup[t][1] for t in events_idx], dtype=np.float32)
    chain_lens = np.array([dedup[t][2] for t in events_idx], dtype=np.int8)
    events_s = events_idx / fs
    if return_extra:
        return events_s, amps, alphas, chain_lens
    return events_s, amps


def detect_events_per_au(au_data, valid_mask, ts, au_indices=None,
                          fs_native=30.0, smooth_window=5,
                          baseline_window_s=5.0, noise_thresh_factor=3.0,
                          activity_signal=None,
                          activity_quantile_gate=0.70,
                          activity_min_amp=None):
    """Run multiscale event detection on selected AU channels.

    Args:
        au_data: (T_native, 52) raw blendshape values.
        valid_mask: (T_native,) bool — frames where face was tracked.
        ts: (T_native,) timestamps (seconds) for each row.
        au_indices: list of AU column indices. Defaults to AFFECT_AUS.
        fs_native: native sampling rate (Hz).
        smooth_window: Savitzky-Golay window in frames.
        baseline_window_s: rolling baseline window (seconds).
        noise_thresh_factor: noise floor multiplier (× MAD).
        activity_signal: (T_native,) per-frame activity envelope (e.g.
            ``p{1,2}_au_activity`` from face preproc). If provided, events
            are gated to frames where ``activity > threshold`` — this is
            Stage 2 of the synchrony-repertoire proposal (joint activation
            episode segmentation), and prevents z-score drift during
            quiet periods (e.g. eyes-closed meditation) from registering
            as "expression events".
        activity_quantile_gate: per-session quantile defining the activity
            threshold (default 0.70 = top 30% of frames qualify as
            "active"). Ignored if ``activity_min_amp`` is set.
        activity_min_amp: absolute activity threshold (overrides quantile gate
            if not None).

    Returns:
        events_per_au: dict {au_idx: (event_times_s, event_amps)}.
    """
    if au_indices is None:
        au_indices = AFFECT_AUS

    # Restrict to valid frames; replace invalid with nearest valid via interp
    if (~valid_mask).any():
        # Linear-interpolate invalid stretches per channel
        au_clean = au_data.copy()
        for c in range(au_clean.shape[1]):
            v = valid_mask & np.isfinite(au_clean[:, c])
            if v.sum() < 3:
                continue
            au_clean[~v, c] = np.interp(np.where(~v)[0], np.where(v)[0],
                                         au_clean[v, c])
    else:
        au_clean = au_data

    selected = au_clean[:, au_indices]

    # Stage 0: smooth + baseline-subtract
    smoothed = _smooth_savgol(selected, window=smooth_window, poly=2)
    baselined = _baseline_subtract(smoothed, fs_native,
                                    window_s=baseline_window_s, q=10)

    # Stage 2 gate: only keep events where joint facial activity is high
    if activity_signal is not None:
        act = np.asarray(activity_signal, dtype=np.float64).squeeze()
        if act.shape[0] != au_data.shape[0]:
            raise ValueError(f'activity_signal length {act.shape[0]} != '
                              f'au_data length {au_data.shape[0]}')
        if activity_min_amp is not None:
            act_threshold = float(activity_min_amp)
        else:
            # Per-session quantile (robust to inter-participant baseline diffs)
            act_threshold = float(np.quantile(act[valid_mask], activity_quantile_gate))
        active_frames = act > act_threshold
    else:
        active_frames = None
        act_threshold = None

    # Stage 1: per-AU event detection
    events_per_au = {}
    for col_i, au_idx in enumerate(au_indices):
        et, ea = detect_au_events(baselined[:, col_i], fs_native,
                                   noise_thresh_factor=noise_thresh_factor)
        if len(et) == 0:
            events_per_au[au_idx] = (np.zeros(0), np.zeros(0))
            continue

        # Stage 2 gate: drop events outside active frames
        if active_frames is not None:
            event_idx = np.clip((et * fs_native).astype(int), 0,
                                 len(active_frames) - 1)
            keep = active_frames[event_idx]
            et = et[keep]
            ea = ea[keep]

        # Convert event times (relative to start of selected) to absolute
        if len(et) > 0:
            events_per_au[au_idx] = (ts[0] + et, ea)
        else:
            events_per_au[au_idx] = (np.zeros(0), np.zeros(0))

    return events_per_au


def events_to_grid(events_per_au, t_common, lsl_offset, fs_out=2.0):
    """Convert per-AU event lists → (n_au, T_common) integer event-count grid.

    Args:
        events_per_au: dict {au_idx: (event_times_s, event_amps)}.
        t_common: (T_common,) common-time grid in LSL seconds.
        lsl_offset: float — add to event times to align with t_common's clock.
        fs_out: output rate (Hz). Default 2.0.

    Returns:
        grid: (n_au, T_common) int — number of events at each (au, t) bin.
    """
    n_au = len(events_per_au)
    T = len(t_common)
    grid = np.zeros((n_au, T), dtype=np.int16)
    if T == 0:
        return grid
    bin_w = 1.0 / fs_out
    edges = np.concatenate([t_common - bin_w / 2, [t_common[-1] + bin_w / 2]])
    for col, (au_idx, (et, _)) in enumerate(sorted(events_per_au.items())):
        if len(et) == 0:
            continue
        et_lsl = et + lsl_offset
        idx = np.searchsorted(edges, et_lsl) - 1
        valid = (idx >= 0) & (idx < T)
        for i in idx[valid]:
            grid[col, i] += 1
    return grid


def compute_event_coincidence_z(p1_grid, p2_grid, tau_samples=1,
                                 n_surrogates=200, seed=42, min_event_rate=0.005):
    """Surrogate-z dyadic event coincidence (mirrors burst_coincidence pattern).

    Args:
        p1_grid: (C, N) int event-count grid (AU × 2Hz timepoints).
        p2_grid: (C, N) same.
        tau_samples: coincidence window in 2Hz samples (1 = ±500ms).
        n_surrogates: circular-shift surrogates.
        seed: RNG seed.
        min_event_rate: per-participant minimum event density to bother
            computing — below this z is forced to zero (prevents division
            by near-zero surrogate std at sparse timepoints).

    Returns:
        coincidence_z: (N,) z-scored coincidence timecourse.
        coincidence_raw: (N,) raw mean coincident-AU-fraction.
    """
    C, N = p1_grid.shape
    if C == 0 or N == 0:
        return np.zeros(N), np.zeros(N)

    # Boolean grids: "did this AU have ≥1 event in this 2Hz bin?"
    p1_bool = (p1_grid > 0)
    p2_bool = (p2_grid > 0)

    p1_rate = p1_bool.mean()
    p2_rate = p2_bool.mean()
    if p1_rate < min_event_rate or p2_rate < min_event_rate:
        return np.zeros(N), np.zeros(N)

    # Dilate P1 by ±tau (so coincidence with P2 within ±tau samples counts)
    kernel = np.ones(2 * tau_samples + 1)

    def _dilate(grid_bool):
        out = np.zeros_like(grid_bool)
        for c in range(grid_bool.shape[0]):
            out[c] = np.convolve(grid_bool[c].astype(np.float32), kernel,
                                  mode='same') > 0
        return out

    p1_dil = _dilate(p1_bool)

    # Real coincidence: per-time, fraction of AUs where both fired
    real_coinc = (p1_dil & p2_bool).mean(axis=0).astype(np.float64)

    # Surrogate distribution via Welford
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * N))
    max_shift = N - min_shift
    surr_mean = np.zeros(N, dtype=np.float64)
    surr_m2 = np.zeros(N, dtype=np.float64)
    for si in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        p2_shift = np.roll(p2_bool, shift, axis=1)
        sc = (p1_dil & p2_shift).mean(axis=0).astype(np.float64)
        delta = sc - surr_mean
        surr_mean += delta / (si + 1)
        delta2 = sc - surr_mean
        surr_m2 += delta * delta2
    surr_std = np.sqrt(surr_m2 / max(n_surrogates - 1, 1))
    surr_std = np.maximum(surr_std, 1e-10)
    z = (real_coinc - surr_mean) / surr_std
    z = np.clip(z, -10, 10)
    return z, real_coinc


def detect_activity_peaks(activity_signal, fs_native=30.0,
                            quantile_threshold=0.70, min_sep_s=1.0):
    """Find peaks in the per-frame ``au_activity`` envelope.

    Args:
        activity_signal: (T,) per-frame facial-activity envelope.
        fs_native: native sampling rate (Hz).
        quantile_threshold: per-session quantile defining the peak
            amplitude floor (default 0.70 = top 30% of frames).
        min_sep_s: minimum separation between consecutive peaks (s).

    Returns:
        peak_idx: (n_peaks,) frame indices of peaks.
        peak_amp: (n_peaks,) peak amplitudes.
    """
    from scipy.signal import find_peaks
    a = np.asarray(activity_signal, dtype=np.float64).squeeze()
    if a.size < 3:
        return np.zeros(0, dtype=int), np.zeros(0)
    height = float(np.quantile(a, quantile_threshold))
    distance = max(1, int(round(min_sep_s * fs_native)))
    peaks, props = find_peaks(a, height=height, distance=distance)
    return peaks, a[peaks]


def peaks_to_grid(peak_times_lsl, t_common, lsl_offset=0.0):
    """Convert event timestamps → (T_common,) binary 2 Hz peak grid."""
    T = len(t_common)
    grid = np.zeros(T, dtype=np.int16)
    if len(peak_times_lsl) == 0 or T == 0:
        return grid
    bin_w = float(np.median(np.diff(t_common)))
    edges = np.concatenate([t_common - bin_w / 2, [t_common[-1] + bin_w / 2]])
    pt = peak_times_lsl + lsl_offset
    idx = np.searchsorted(edges, pt) - 1
    valid = (idx >= 0) & (idx < T)
    for i in idx[valid]:
        grid[i] += 1
    return grid


def _coincidence_z(p1_grid, p2_grid, tau_samples=1, n_surrogates=200,
                    seed=42, min_event_rate=0.005):
    """Surrogate-z dyadic coincidence on a single 1D grid pair.

    Adapts the multi-channel event_coincidence_z above for the case where
    each participant contributes a single 1D peak grid.
    """
    N = len(p1_grid)
    if N == 0:
        return np.zeros(N), np.zeros(N)
    p1_bool = (p1_grid > 0).astype(np.float32)
    p2_bool = (p2_grid > 0).astype(np.float32)
    if p1_bool.mean() < min_event_rate or p2_bool.mean() < min_event_rate:
        return np.zeros(N), np.zeros(N)

    kernel = np.ones(2 * tau_samples + 1, dtype=np.float32)
    p1_dil = (np.convolve(p1_bool, kernel, mode='same') > 0).astype(np.float32)

    real_coinc = (p1_dil * p2_bool).astype(np.float64)

    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * N))
    max_shift = N - min_shift
    surr_mean = np.zeros(N, dtype=np.float64)
    surr_m2 = np.zeros(N, dtype=np.float64)
    for si in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        p2_shift = np.roll(p2_bool, shift)
        sc = (p1_dil * p2_shift).astype(np.float64)
        delta = sc - surr_mean
        surr_mean += delta / (si + 1)
        delta2 = sc - surr_mean
        surr_m2 += delta * delta2
    surr_std = np.sqrt(surr_m2 / max(n_surrogates - 1, 1))
    surr_std = np.maximum(surr_std, 1e-10)
    z = (real_coinc - surr_mean) / surr_std
    z = np.clip(z, -10, 10)
    return z, real_coinc


def compute_bl_event_coincidence(face_npz_data, t_common, lsl_offset,
                                  fs_native=30.0, n_surrogates=200, seed=42,
                                  tau_samples=1, activity_quantile_gate=0.70,
                                  min_peak_sep_s=1.0,
                                  smooth_sigma_s=15.0):
    """End-to-end: face activity envelopes → peak coincidence z-trace.

    The raw per-2Hz-bin coincidence z-trace is sparse (mostly zero with
    occasional spikes when peaks coincide), which makes it too low-amplitude
    relative to EEG concordance channels for the rSLDS to form a behavioral
    state around. We Gaussian-smooth (default σ=15s) to produce a continuous
    "recent coupling intensity envelope" — preserves the statistical meaning
    per timepoint (surrogate-corrected) but amplifies regions where coincident
    events cluster, giving the model continuous signal to work with.

    Args:
        face_npz_data: dict-like (np.load result) from
            ``data/preproc/face/v1/<sid>.npz``.
        t_common: (N,) common-time grid in LSL seconds.
        lsl_offset: float — add to face_ts to align with t_common.
        fs_native: native blendshape sampling rate (Hz).
        n_surrogates: circular-shift surrogates for z-scoring.
        seed: RNG seed.
        tau_samples: coincidence window in 2Hz samples (1 = ±500ms).
        activity_quantile_gate: per-session quantile defining the activity
            peak threshold. Default 0.70 = top 30% of frames.
        min_peak_sep_s: minimum separation between consecutive peaks (s).
        smooth_sigma_s: Gaussian smoothing sigma applied to the z-trace
            after surrogate z-scoring. Default 15.0 s. Set to 0 to disable.

    Returns:
        z: (N,) bl_event_coincidence z-score timecourse at t_common rate.
        info: diagnostic dict.
    """
    info = {}
    required = ['p1_au_activity', 'p1_au52_ts',
                'p2_au_activity', 'p2_au52_ts']
    for k in required:
        if k not in face_npz_data:
            return np.zeros(len(t_common), dtype=np.float32), {
                'status': 'missing_data', 'missing': k,
            }

    p1_act = np.asarray(face_npz_data['p1_au_activity']).squeeze()
    p2_act = np.asarray(face_npz_data['p2_au_activity']).squeeze()
    p1_ts  = face_npz_data['p1_au52_ts']
    p2_ts  = face_npz_data['p2_au52_ts']

    # Detect peaks in each participant's activity envelope
    p1_peaks_idx, _ = detect_activity_peaks(p1_act, fs_native=fs_native,
                                              quantile_threshold=activity_quantile_gate,
                                              min_sep_s=min_peak_sep_s)
    p2_peaks_idx, _ = detect_activity_peaks(p2_act, fs_native=fs_native,
                                              quantile_threshold=activity_quantile_gate,
                                              min_sep_s=min_peak_sep_s)
    p1_peak_times = p1_ts[p1_peaks_idx]
    p2_peak_times = p2_ts[p2_peaks_idx]

    info['p1_total_peaks'] = int(len(p1_peaks_idx))
    info['p2_total_peaks'] = int(len(p2_peaks_idx))

    # Build per-participant 2 Hz peak grid (1D)
    p1_grid = peaks_to_grid(p1_peak_times, t_common, lsl_offset)
    p2_grid = peaks_to_grid(p2_peak_times, t_common, lsl_offset)

    info['p1_grid_density'] = float((p1_grid > 0).mean())
    info['p2_grid_density'] = float((p2_grid > 0).mean())

    # Coincidence z
    z, raw = _coincidence_z(p1_grid, p2_grid, tau_samples=tau_samples,
                              n_surrogates=n_surrogates, seed=seed)
    info['mean_z_raw'] = float(z.mean())
    info['std_z_raw'] = float(z.std())
    info['mean_raw_coinc'] = float(raw.mean())

    # Gaussian-smooth the z-trace into a "recent coupling intensity" envelope
    # so the rSLDS sees a continuous signal of comparable amplitude to other
    # observation channels. Smoothing alone *reduces* variance (~ √(window)),
    # so we then per-session standardize to restore unit variance comparable
    # to the EEG concordance channels (which are surrogate-z'd per timepoint
    # and naturally have std ≈ 1).
    if smooth_sigma_s and smooth_sigma_s > 0 and len(t_common) > 1:
        from scipy.ndimage import gaussian_filter1d
        fs_out = 1.0 / float(np.median(np.diff(t_common)))
        sigma_samples = max(0.5, smooth_sigma_s * fs_out)
        z = gaussian_filter1d(z, sigma=sigma_samples).astype(np.float32)
        info['smooth_sigma_s'] = float(smooth_sigma_s)
        info['mean_z_smoothed'] = float(z.mean())
        info['std_z_smoothed'] = float(z.std())
        # Per-session standardize so signal amplitude is comparable to other channels
        s = float(z.std())
        if s > 1e-6:
            z = ((z - z.mean()) / s).astype(np.float32)
        info['mean_z'] = float(z.mean())
        info['std_z'] = float(z.std())
    else:
        info['smooth_sigma_s'] = 0.0
        info['mean_z'] = info['mean_z_raw']
        info['std_z'] = info['std_z_raw']

    info['status'] = 'ok'
    return z.astype(np.float32), info
