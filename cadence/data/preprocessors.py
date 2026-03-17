"""Per-modality preprocessing: filtering, artifact detection, normalization."""

import numpy as np
from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Activity channel: causal trailing-mean deviation (shared across modalities)
# ---------------------------------------------------------------------------

def compute_activity_channel(features, hz, trailing_seconds=30.0):
    """Compute causal activity channel: RMS across channels minus trailing mean.

    Measures "how much is happening right now" compared to the recent past.
    Strictly causal: only uses data from [t - trailing_seconds, t].

    Args:
        features: (N, C) z-scored feature array
        hz: sampling rate
        trailing_seconds: trailing window for local baseline

    Returns:
        activity: (N, 1) activity channel, clipped to [-10, 10]
    """
    # RMS across channels at each timestep
    rms = np.sqrt(np.mean(features ** 2, axis=1))  # (N,)

    # Causal trailing mean via cumsum (vectorized, no loop)
    window_size = max(1, int(trailing_seconds * hz))
    cs = np.concatenate([[0.0], np.cumsum(rms)])
    ends = np.arange(1, len(rms) + 1)
    starts = np.maximum(ends - window_size, 0)
    counts = ends - starts
    trailing_mean = (cs[ends] - cs[starts]) / counts  # (N,)

    # Activity = deviation from recent baseline
    activity = rms - trailing_mean
    return np.clip(activity, -10, 10).astype(np.float32)[:, None]  # (N, 1)


def preprocess_eeg(data, timestamps, srate=256):
    """
    Preprocess EmotivDataStream-EEG.

    Parameters:
        data: (N, 19) raw EEG stream (14 EEG channels in cols 3-16)
        timestamps: (N,) LSL timestamps
        srate: sampling rate (256 Hz for Emotiv)

    Returns:
        eeg: (N, 14) preprocessed EEG
        valid: (N, 14) per-channel validity mask
        timestamps: (N,) timestamps
    """
    # Extract 14 EEG channels (columns 3-16)
    eeg = data[:, 3:17].astype(np.float64)

    # 1. Notch filter 50 Hz and 60 Hz (mains interference)
    for freq in [50.0, 60.0]:
        b, a = iirnotch(freq, Q=30.0, fs=srate)
        for ch in range(14):
            eeg[:, ch] = filtfilt(b, a, eeg[:, ch])

    # 2. Bandpass filter 1-45 Hz
    sos = butter(4, [1.0, 45.0], btype='bandpass', fs=srate, output='sos')
    for ch in range(14):
        eeg[:, ch] = sosfiltfilt(sos, eeg[:, ch])

    # 3. Per-channel validity mask
    valid = np.ones((len(eeg), 14), dtype=bool)

    # Mark high-amplitude artifacts (> 100 uV)
    valid[np.abs(eeg) > 100] = False

    # 4. Z-score normalize per channel (using valid samples only)
    for ch in range(14):
        valid_samples = eeg[valid[:, ch], ch]
        if len(valid_samples) > 100:
            mu, sigma = valid_samples.mean(), valid_samples.std()
            if sigma > 1e-8:
                eeg[:, ch] = (eeg[:, ch] - mu) / sigma

    # Clamp extreme outliers
    eeg = np.clip(eeg, -10, 10)

    return eeg, valid, timestamps


def preprocess_ecg(data, timestamps, target_srate=130):
    """
    Preprocess ECG from Polar H10 (irregular Bluetooth timestamps).

    Parameters:
        data: (N, 1) raw ECG
        timestamps: (N,) irregular LSL timestamps

    Returns:
        ecg: (M,) uniformly sampled ECG
        valid: (M,) boolean validity mask
        t_uniform: (M,) uniform timestamp grid
    """
    ecg_flat = data.flatten().astype(np.float64)

    # Create uniform time grid
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / target_srate)

    if len(t_uniform) < 10:
        return ecg_flat, np.ones(len(ecg_flat), dtype=bool), timestamps

    # Interpolate to uniform grid
    ecg_uniform = np.interp(t_uniform, timestamps, ecg_flat)

    # Detect true dropouts (gaps > 3 sample periods in original timestamps)
    dt = np.diff(timestamps)
    gap_threshold = 3.0 / target_srate
    valid = np.ones(len(t_uniform), dtype=bool)

    gap_indices = np.where(dt > gap_threshold)[0]
    for idx in gap_indices:
        gap_start = timestamps[idx]
        gap_end = timestamps[idx + 1]
        valid[(t_uniform >= gap_start) & (t_uniform <= gap_end)] = False

    # Bandpass filter 0.5-40 Hz
    sos = butter(4, [0.5, 40.0], btype='bandpass', fs=target_srate, output='sos')
    ecg_filtered = sosfiltfilt(sos, ecg_uniform)

    # Z-score normalize
    valid_samples = ecg_filtered[valid]
    if len(valid_samples) > 100:
        mu, sigma = valid_samples.mean(), valid_samples.std()
        if sigma > 1e-8:
            ecg_filtered = (ecg_filtered - mu) / sigma

    return ecg_filtered, valid, t_uniform


def preprocess_blendshapes(data, timestamps, srate=30):
    """
    Preprocess facial blendshapes from landmark stream.

    Parameters:
        data: (N, 1489) raw landmark stream
        timestamps: (N,) LSL timestamps

    Returns:
        blendshapes: (N, 52) preprocessed blendshapes
        valid: (N,) per-frame validity mask
        timestamps: (N,) timestamps
    """
    # Extract 52 blendshapes (first 52 channels)
    blendshapes = data[:, :52].astype(np.float64)

    # Detect face-not-detected frames (all blendshapes ~ 0.0)
    face_present = ~np.all(np.abs(blendshapes) < 1e-6, axis=1)

    # Short gaps (< 0.5s = 15 frames): linear interpolate
    valid = face_present.copy()
    gap_starts, gap_lengths = _find_gaps(face_present)

    for start, length in zip(gap_starts, gap_lengths):
        if length <= 15:  # <= 0.5s at 30 Hz
            end = start + length
            if start > 0 and end < len(blendshapes):
                for ch in range(52):
                    blendshapes[start:end, ch] = np.linspace(
                        blendshapes[start - 1, ch],
                        blendshapes[end, ch],
                        length
                    )
                valid[start:end] = True

    # Z-score normalize per blendshape (using valid frames)
    for ch in range(52):
        valid_samples = blendshapes[valid, ch]
        if len(valid_samples) > 100:
            mu, sigma = valid_samples.mean(), valid_samples.std()
            if sigma > 1e-8:
                blendshapes[:, ch] = (blendshapes[:, ch] - mu) / sigma

    # Clamp extreme outliers
    blendshapes = np.clip(blendshapes, -10, 10)

    # Append activity channel (ch 53): causal RMS deviation from trailing 30s mean
    effective_hz = len(blendshapes) / max(timestamps[-1] - timestamps[0], 1.0)
    activity = compute_activity_channel(blendshapes, effective_hz, trailing_seconds=30.0)
    blendshapes = np.concatenate([blendshapes, activity], axis=1)  # (N, 53)

    return blendshapes, valid, timestamps


def preprocess_pose(data, timestamps, srate=30, visibility_threshold=0.5):
    """
    Preprocess body pose from MediaPipe.

    Parameters:
        data: (N, 132) raw pose (33 landmarks x 4: x, y, z, visibility)
        timestamps: (N,) LSL timestamps

    Returns:
        pose: (N, 99) coordinates (33 x 3), zeroed where invisible
        valid: (N,) per-frame validity mask (True if >= 10 keypoints visible)
        timestamps: (N,) timestamps
    """
    reshaped = data.reshape(-1, 33, 4)
    coords = reshaped[:, :, :3].astype(np.float64)  # (N, 33, 3)
    visibility = reshaped[:, :, 3]  # (N, 33)

    # Zero out low-visibility landmarks
    low_vis = visibility < visibility_threshold
    for lm in range(33):
        coords[low_vis[:, lm], lm, :] = 0.0

    # Frame-level validity: at least 10 keypoints visible
    valid = (visibility >= visibility_threshold).sum(axis=1) >= 10

    # Flatten to (N, 99)
    pose = coords.reshape(-1, 99)

    # Z-score normalize per coordinate (only visible values; leave zeros as zeros)
    for dim in range(99):
        landmark_idx = dim // 3
        visible_frames = valid & (visibility[:, landmark_idx] >= visibility_threshold)
        valid_samples = pose[visible_frames, dim]
        if len(valid_samples) > 100:
            mu, sigma = valid_samples.mean(), valid_samples.std()
            if sigma > 1e-8:
                pose[visible_frames, dim] = (pose[visible_frames, dim] - mu) / sigma

    # Clamp to prevent extreme outliers
    pose = np.clip(pose, -10, 10)

    return pose, valid, timestamps


def _find_gaps(mask):
    """Find contiguous False regions in a boolean mask."""
    if len(mask) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    changes = np.diff(mask.astype(int))
    starts = np.where(changes == -1)[0] + 1  # Start of False region
    ends = np.where(changes == 1)[0] + 1  # End of False region

    # Handle edge cases
    if not mask[0]:
        starts = np.concatenate([[0], starts])
    if not mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    # Ensure equal length
    min_len = min(len(starts), len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]

    lengths = ends - starts
    return starts, lengths


# ---------------------------------------------------------------------------
# V4 feature extractors
# ---------------------------------------------------------------------------

ECG_FEATURES_SRATE = 2  # Output sampling rate for ECG features (Hz)
ECG_N_FEATURES = 6
POSE_N_FEATURES = 40


def extract_ecg_features(ecg_filtered, ecg_valid, ecg_ts, srate=130):
    """
    Extract HRV feature channels from preprocessed ECG at 2 Hz.

    6 channels capturing different cardiac timescales:
        0: Instantaneous HR (beats/min)
        1: IBI deviation from 10s running mean
        2: RMSSD (10s rolling window)
        3: HR acceleration (d(HR)/dt, 5s smoothed)
        4: QRS amplitude (interpolated peak heights)
        5: HR trend (30s linear slope)

    Parameters:
        ecg_filtered: (M,) z-scored, bandpass-filtered ECG
        ecg_valid: (M,) boolean validity mask
        ecg_ts: (M,) uniform timestamps

    Returns:
        features: (N_out, 6) at 2 Hz
        valid_out: (N_out,) boolean
        t_out: (N_out,) timestamps at 2 Hz
    """
    out_dt = 1.0 / ECG_FEATURES_SRATE
    t_out = np.arange(ecg_ts[0], ecg_ts[-1], out_dt)
    n_out = len(t_out)
    features = np.zeros((n_out, ECG_N_FEATURES))
    valid_out = np.zeros(n_out, dtype=bool)

    if n_out == 0 or ecg_valid.sum() < 100:
        return features, valid_out, t_out

    # --- R-peak detection ---
    std_val = np.std(ecg_filtered[ecg_valid])
    if std_val < 1e-8:
        return features, valid_out, t_out

    peaks, props = find_peaks(
        ecg_filtered,
        distance=int(0.4 * srate),   # min 0.4s between beats (max ~150 BPM)
        height=0.5 * std_val,
    )

    if len(peaks) < 3:
        return features, valid_out, t_out

    peak_times = ecg_ts[peaks]
    peak_heights = props['peak_heights']
    ibis = np.clip(np.diff(peak_times), 0.3, 2.0)  # physiological range
    ibi_times = peak_times[1:]

    # --- IBI outlier rejection (ectopic beat removal) ---
    if len(ibis) >= 5:
        from scipy.ndimage import median_filter
        local_median = median_filter(ibis, size=5, mode='reflect')
        outlier_mask = np.abs(ibis - local_median) / np.maximum(local_median, 0.3) > 0.40
        false_peaks = set()
        for i in range(len(outlier_mask) - 1):
            if outlier_mask[i] and outlier_mask[i + 1]:
                false_peaks.add(i + 1)
        for i in range(len(ibis)):
            if ibis[i] < 0.6 * local_median[i]:
                false_peaks.add(i + 1)
        if false_peaks:
            keep = [i for i in range(len(peaks)) if i not in false_peaks]
            peaks = peaks[keep]
            peak_times = ecg_ts[peaks]
            peak_heights = props['peak_heights'][keep]
            ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
            ibi_times = peak_times[1:]
            if len(peaks) < 3:
                return features, valid_out, t_out

    # Interpolate IBIs to 2 Hz output grid
    ibi_interp = np.interp(t_out, ibi_times, ibis, left=ibis[0], right=ibis[-1])
    hr_interp = 60.0 / ibi_interp  # BPM

    # --- Ch0: Instantaneous HR ---
    features[:, 0] = hr_interp

    # --- Ch1: IBI deviation from 10s running mean ---
    w10 = min(20, n_out)  # 10s at 2 Hz
    if w10 > 0:
        kernel10 = np.ones(w10) / w10
        ibi_mean = np.convolve(ibi_interp, kernel10, mode='same')
        features[:, 1] = ibi_interp - ibi_mean

    # --- Ch2: RMSSD (rolling 10s window) ---
    if len(ibis) >= 2:
        succ_diff_sq = np.diff(ibis) ** 2
        sd_times = ibi_times[1:]
        sd_interp = np.interp(t_out, sd_times, succ_diff_sq, left=0, right=0)
        if w10 > 0:
            features[:, 2] = np.sqrt(np.convolve(sd_interp, kernel10, mode='same'))

    # --- Ch3: HR acceleration (gradient of 5s-smoothed HR) ---
    w5 = min(10, n_out)  # 5s at 2 Hz
    if w5 > 1:
        kernel5 = np.ones(w5) / w5
        hr_smooth = np.convolve(hr_interp, kernel5, mode='same')
        features[:, 3] = np.gradient(hr_smooth, out_dt)

    # --- Ch4: QRS amplitude (interpolated peak heights) ---
    amp_interp = np.interp(t_out, peak_times, peak_heights,
                           left=peak_heights[0], right=peak_heights[-1])
    features[:, 4] = amp_interp

    # --- Ch5: HR trend (30s linear slope via convolution) ---
    w30 = min(60, n_out)  # 30s at 2 Hz
    if w30 >= 4:
        x = np.arange(w30) - (w30 - 1) / 2.0
        denom = np.sum(x ** 2)
        if denom > 0:
            slope_kernel = x / denom
            features[:, 5] = np.convolve(hr_interp, slope_kernel[::-1], mode='same')

    # --- Z-score each feature ---
    for ch in range(ECG_N_FEATURES):
        vals = features[:, ch]
        std_ch = np.std(vals)
        if std_ch > 1e-8:
            features[:, ch] = (vals - np.mean(vals)) / std_ch
    features = np.clip(features, -10, 10)

    # --- Validity: must have R-peaks within 5s AND valid original ECG ---
    peak_idx_right = np.clip(np.searchsorted(peak_times, t_out), 0, len(peak_times) - 1)
    peak_idx_left = np.clip(peak_idx_right - 1, 0, len(peak_times) - 1)
    dist_right = np.abs(peak_times[peak_idx_right] - t_out)
    dist_left = np.abs(peak_times[peak_idx_left] - t_out)
    has_nearby_peak = np.minimum(dist_right, dist_left) < 5.0

    ecg_src_idx = np.clip(np.searchsorted(ecg_ts, t_out), 0, len(ecg_valid) - 1)
    valid_out = has_nearby_peak & ecg_valid[ecg_src_idx]

    return features, valid_out, t_out


def extract_pose_features(pose, valid, timestamps, srate=30):
    """
    Extract 40 joint-group feature channels from flat pose coordinates.

    7 body segment groups:
        Head (8):     centroid(3), extent(1), tilt(3), rotation(1)
        L.Arm (5):    centroid(3), elbow angle(1), extension(1)
        R.Arm (5):    centroid(3), elbow angle(1), extension(1)
        Torso (6):    centroid(3), lateral lean(1), fwd lean(1), rotation(1)
        L.Leg (5):    centroid(3), knee angle(1), extension(1)
        R.Leg (5):    centroid(3), knee angle(1), extension(1)
        Global (6):   CoM(3), L/R symmetry(1), velocity(1), openness(1)

    Parameters:
        pose: (N, 99) flat coordinates (33 landmarks x 3)
        valid: (N,) per-frame validity mask
        timestamps: (N,) timestamps

    Returns:
        features: (N, 40) joint-group features
        valid: (N,) validity mask (unchanged)
        timestamps: (N,) timestamps (unchanged)
    """
    N = len(pose)
    coords = pose.reshape(N, 33, 3)

    features = np.zeros((N, POSE_N_FEATURES))
    col = 0

    def _centroid(lm_list):
        return coords[:, lm_list, :].mean(axis=1)  # (N, 3)

    def _extent(lm_list):
        pts = coords[:, lm_list, :]
        return np.linalg.norm(pts.max(axis=1) - pts.min(axis=1), axis=1)  # (N,)

    def _angle(a, b, c):
        ba = coords[:, a, :] - coords[:, b, :]
        bc = coords[:, c, :] - coords[:, b, :]
        cos_a = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
        )
        return np.arccos(np.clip(cos_a, -1, 1))  # (N,) radians

    def _extension(a, b):
        return np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1)  # (N,)

    # --- Head (8): centroid(3), extent(1), tilt(3), rotation(1) ---
    head_lm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    features[:, col:col+3] = _centroid(head_lm); col += 3
    features[:, col] = _extent(head_lm); col += 1
    ear_mid = (coords[:, 7, :] + coords[:, 8, :]) / 2
    tilt = coords[:, 0, :] - ear_mid
    features[:, col:col+3] = tilt; col += 3
    features[:, col] = coords[:, 8, 2] - coords[:, 7, 2]; col += 1

    # --- L.Arm (5): centroid(3), elbow angle(1), extension(1) ---
    features[:, col:col+3] = _centroid([11, 13, 15]); col += 3
    features[:, col] = _angle(11, 13, 15); col += 1
    l_arm_ext = _extension(11, 15)
    features[:, col] = l_arm_ext; col += 1

    # --- R.Arm (5): centroid(3), elbow angle(1), extension(1) ---
    features[:, col:col+3] = _centroid([12, 14, 16]); col += 3
    features[:, col] = _angle(12, 14, 16); col += 1
    r_arm_ext = _extension(12, 16)
    features[:, col] = r_arm_ext; col += 1

    # --- Torso (6): centroid(3), lateral lean(1), fwd lean(1), rotation(1) ---
    features[:, col:col+3] = _centroid([11, 12, 23, 24]); col += 3
    shoulder_mid = (coords[:, 11, :] + coords[:, 12, :]) / 2
    hip_mid = (coords[:, 23, :] + coords[:, 24, :]) / 2
    lean = shoulder_mid - hip_mid
    features[:, col] = lean[:, 0]; col += 1
    features[:, col] = lean[:, 2]; col += 1
    shoulder_line = coords[:, 12, :] - coords[:, 11, :]
    hip_line = coords[:, 24, :] - coords[:, 23, :]
    rotation = np.arctan2(
        shoulder_line[:, 0] * hip_line[:, 2] - shoulder_line[:, 2] * hip_line[:, 0],
        shoulder_line[:, 0] * hip_line[:, 0] + shoulder_line[:, 2] * hip_line[:, 2] + 1e-8,
    )
    features[:, col] = rotation; col += 1

    # --- L.Leg (5): centroid(3), knee angle(1), extension(1) ---
    features[:, col:col+3] = _centroid([23, 25, 27, 29]); col += 3
    features[:, col] = _angle(23, 25, 27); col += 1
    l_leg_ext = _extension(23, 27)
    features[:, col] = l_leg_ext; col += 1

    # --- R.Leg (5): centroid(3), knee angle(1), extension(1) ---
    features[:, col:col+3] = _centroid([24, 26, 28, 30]); col += 3
    features[:, col] = _angle(24, 26, 28); col += 1
    r_leg_ext = _extension(24, 28)
    features[:, col] = r_leg_ext; col += 1

    # --- Global (6): CoM(3), L/R symmetry(1), velocity(1), openness(1) ---
    com = _centroid(list(range(33)))
    features[:, col:col+3] = com; col += 3
    features[:, col] = l_arm_ext - r_arm_ext; col += 1

    if N > 1:
        dt = np.diff(timestamps, prepend=timestamps[0] - 1.0 / srate)
        dt = np.clip(dt, 1e-3, 1.0)
        com_diff = np.diff(com, axis=0, prepend=com[:1])
        features[:, col] = np.linalg.norm(com_diff, axis=1) / dt; col += 1
    else:
        col += 1

    features[:, col] = l_arm_ext + r_arm_ext + l_leg_ext + r_leg_ext; col += 1

    assert col == POSE_N_FEATURES, f"Expected {POSE_N_FEATURES} features, got {col}"

    # Z-score per feature (valid frames only)
    for ch in range(POSE_N_FEATURES):
        vals = features[valid, ch]
        if len(vals) > 100:
            std_ch = np.std(vals)
            if std_ch > 1e-8:
                features[:, ch] = (features[:, ch] - np.mean(vals)) / std_ch
    features = np.clip(features, -10, 10)

    # Append activity channel (ch 41): causal RMS deviation from trailing 30s mean
    effective_hz = len(features) / max(timestamps[-1] - timestamps[0], 1.0)
    activity = compute_activity_channel(features, effective_hz, trailing_seconds=30.0)
    features = np.concatenate([features, activity], axis=1)  # (N, 41)

    return features, valid, timestamps


# ---------------------------------------------------------------------------
# V2: Blendshape PCA + temporal derivatives
# ---------------------------------------------------------------------------

def _compute_temporal_derivatives(signal, hz, sigma_s=0.5):
    """Compute smoothed first derivatives of each channel.

    Uses finite differences smoothed with a Gaussian kernel to suppress
    high-frequency noise from discrete differentiation.

    Args:
        signal: (N, C) feature array
        hz: sampling rate in Hz
        sigma_s: Gaussian smoothing sigma in seconds

    Returns:
        derivatives: (N, C) smoothed d(signal)/dt
    """
    from scipy.ndimage import gaussian_filter1d
    dt = 1.0 / hz
    # Forward finite differences
    deriv = np.diff(signal, axis=0, prepend=signal[:1]) / dt
    # Smooth with Gaussian (sigma in samples)
    sigma_samples = sigma_s * hz
    if sigma_samples > 0.5:
        for ch in range(deriv.shape[1]):
            deriv[:, ch] = gaussian_filter1d(deriv[:, ch], sigma_samples)
    return deriv.astype(np.float32)


def extract_blendshapes_v2(blendshapes, valid, ts, n_components=15,
                            deriv_sigma_s=0.5):
    """PCA-reduce blendshapes and add temporal derivatives.

    Pipeline: 52 AUs -> PCA(15) -> z-score -> derivatives -> activity -> 31 ch

    Args:
        blendshapes: (N, 52+) blendshape array (may include activity channel)
        valid: (N,) per-frame validity mask
        ts: (N,) timestamps
        n_components: number of PCA components (default 15)
        deriv_sigma_s: derivative smoothing sigma in seconds

    Returns:
        features: (N, 31) = 15 PCA + 15 derivatives + 1 activity
        valid: (N,) validity mask (unchanged)
        ts: (N,) timestamps (unchanged)
    """
    N = len(blendshapes)
    # Use only the 52 AU channels (strip activity if present)
    bl_raw = blendshapes[:, :52].astype(np.float64)

    # PCA on valid frames only
    valid_mask = valid if valid.sum() > n_components else np.ones(N, dtype=bool)
    bl_valid = bl_raw[valid_mask]

    mean = bl_valid.mean(axis=0)
    centered = bl_valid - mean

    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project all frames (including invalid, for continuity)
    projected = (bl_raw - mean) @ Vt[:n_components].T  # (N, n_components)

    # Z-score each component
    for c in range(n_components):
        vals = projected[valid_mask, c]
        if len(vals) > 10:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 1e-8:
                projected[:, c] = (projected[:, c] - mu) / sigma
    projected = np.clip(projected, -10, 10).astype(np.float32)

    # Temporal derivatives of PCA components
    effective_hz = len(ts) / max(ts[-1] - ts[0], 1.0) if len(ts) > 1 else 30.0
    derivatives = _compute_temporal_derivatives(
        projected, effective_hz, sigma_s=deriv_sigma_s)

    # Z-score derivatives
    for c in range(n_components):
        vals = derivatives[valid_mask, c]
        if len(vals) > 10:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 1e-8:
                derivatives[:, c] = (derivatives[:, c] - mu) / sigma
    derivatives = np.clip(derivatives, -10, 10).astype(np.float32)

    # Activity channel
    activity = compute_activity_channel(projected, effective_hz,
                                         trailing_seconds=30.0)

    # Stack: [PCA(15) | derivatives(15) | activity(1)] = 31 channels
    features = np.concatenate([projected, derivatives, activity], axis=1)
    return features, valid, ts


# ---------------------------------------------------------------------------
# V2: Cardiac feature expansion (add RMSSD derivative)
# ---------------------------------------------------------------------------

def extract_ecg_features_v2(ecg_filtered, ecg_valid, ecg_ts, srate=130):
    """Extract expanded HRV features: original 6 + RMSSD derivative = 7 channels.

    Args:
        ecg_filtered: (M,) z-scored, bandpass-filtered ECG
        ecg_valid: (M,) boolean validity mask
        ecg_ts: (M,) uniform timestamps

    Returns:
        features: (N_out, 7) at 2 Hz
        valid_out: (N_out,) boolean
        t_out: (N_out,) timestamps at 2 Hz
    """
    # Get base 6 features from v1 extractor
    features_v1, valid_out, t_out = extract_ecg_features(
        ecg_filtered, ecg_valid, ecg_ts, srate=srate)

    n_out = len(t_out)
    features = np.zeros((n_out, 7), dtype=np.float32)
    features[:, :6] = features_v1

    # Ch6: RMSSD derivative (d(RMSSD)/dt, smoothed)
    if n_out > 1:
        rmssd = features_v1[:, 2]  # Ch2 = RMSSD
        dt = 1.0 / ECG_FEATURES_SRATE
        rmssd_deriv = np.diff(rmssd, prepend=rmssd[0]) / dt
        # Smooth with 5s Gaussian
        from scipy.ndimage import gaussian_filter1d
        sigma_samples = 5.0 * ECG_FEATURES_SRATE
        if sigma_samples > 0.5:
            rmssd_deriv = gaussian_filter1d(rmssd_deriv, sigma_samples)
        # Z-score
        std_val = np.std(rmssd_deriv)
        if std_val > 1e-8:
            rmssd_deriv = (rmssd_deriv - np.mean(rmssd_deriv)) / std_val
        features[:, 6] = np.clip(rmssd_deriv, -10, 10)

    return features, valid_out, t_out


# ---------------------------------------------------------------------------
# EEG features: re-exported from cadence.data.eeg_features
# ---------------------------------------------------------------------------

from cadence.data.eeg_features import (  # noqa: E402, F401
    extract_eeg_features, EEG_FEATURE_NAMES, EEG_N_FEATURES, EEG_FEATURES_SRATE,
)
