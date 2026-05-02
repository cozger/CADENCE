"""ECG preprocessing pipeline (Step 6).

Polar H10 transmits ECG at 130 Hz; pyxdf already dejitters BLE timestamps so
no resampling is needed (per ``project_ecg_not_irregular`` memory). We
bandpass-filter, z-score, then extract 7-channel HRV features at 2 Hz
(``extract_ecg_features_v2`` in legacy code).

Output (``data/preproc/ecg/v1/<sid>.npz``)::

    p{1,2}_ecg_clean   : (M,)    float32   bandpassed + z-scored ECG (130 Hz)
    p{1,2}_ecg_clean_ts: (M,)    float64
    p{1,2}_ecg_valid   : (M,)    bool      always-true placeholder
    p{1,2}_ecg_features: (N, 7)  float32   HR + IBI dev + RMSSD + HR-accel
                                          + QRS amp + HR trend + RMSSD-deriv
                                          (z-scored, clipped, sampled at 2 Hz)
    p{1,2}_ecg_features_ts: (N,) float64   2-Hz output grid
    p{1,2}_ecg_features_valid: (N,) bool   has nearby R-peak AND raw ECG valid
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from cadence.preprocess.common import (
    atomic_write_json,
    atomic_write_npz,
    default_out_dir,
    staleness_check,
)


ECG_MODALITY_VERSION = "v1"
ECG_SRATE_HZ = 130
ECG_BANDPASS_HZ = (0.5, 40.0)
ECG_FEATURES_SRATE = 2
ECG_N_FEATURES_V2 = 7

ECG_FEATURE_NAMES = [
    "hr_bpm",                  # 0
    "ibi_dev_5s",              # 1
    "rmssd_5s",                # 2
    "hr_accel_2s",             # 3
    "qrs_amplitude",           # 4
    "hr_trend_10s",            # 5
    "rmssd_derivative",        # 6 — added in v2
]


def preprocess_ecg(data: np.ndarray, ts: np.ndarray, srate: int = ECG_SRATE_HZ
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass-filter + z-score raw ECG. Returns ``(ecg_filtered, valid)``."""
    ecg = data.astype(np.float64).reshape(-1)
    if ecg.size < 10:
        return ecg.astype(np.float32), np.ones(ecg.size, dtype=bool)
    sos = butter(4, list(ECG_BANDPASS_HZ), btype="bandpass", fs=srate, output="sos")
    ecg_filt = sosfiltfilt(sos, ecg)
    mu = float(ecg_filt.mean())
    sigma = float(ecg_filt.std())
    if sigma > 1e-8:
        ecg_filt = (ecg_filt - mu) / sigma
    return ecg_filt.astype(np.float32), np.ones(len(ecg_filt), dtype=bool)


def extract_ecg_features(ecg_filt: np.ndarray, ecg_valid: np.ndarray,
                         ecg_ts: np.ndarray, srate: int = ECG_SRATE_HZ
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, valid_out, t_out)`` — 7 HRV channels at 2 Hz.

    Ports cadence/data/preprocessors.py:extract_ecg_features_v2 (the V8.2
    7-ch HRV used by V8.2/V10/V11). Adds ectopic-beat outlier rejection.
    """
    out_dt = 1.0 / ECG_FEATURES_SRATE
    if len(ecg_ts) < 2:
        return (np.zeros((0, ECG_N_FEATURES_V2), dtype=np.float32),
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float64))
    t_out = np.arange(ecg_ts[0], ecg_ts[-1], out_dt)
    n_out = len(t_out)
    features = np.zeros((n_out, ECG_N_FEATURES_V2), dtype=np.float32)
    valid_out = np.zeros(n_out, dtype=bool)
    if n_out == 0 or ecg_valid.sum() < 100:
        return features, valid_out, t_out

    std_val = float(np.std(ecg_filt[ecg_valid]))
    if std_val < 1e-8:
        return features, valid_out, t_out

    peaks, props = find_peaks(
        ecg_filt,
        distance=int(0.4 * srate),
        height=0.5 * std_val,
    )
    if len(peaks) < 3:
        return features, valid_out, t_out

    peak_times = ecg_ts[peaks]
    peak_heights = props["peak_heights"]
    ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
    ibi_times = peak_times[1:]

    # Ectopic-beat outlier rejection
    if len(ibis) >= 5:
        from scipy.ndimage import median_filter
        local_median = median_filter(ibis, size=5, mode="reflect")
        outlier = np.abs(ibis - local_median) / np.maximum(local_median, 0.3) > 0.40
        false_peaks: set[int] = set()
        for i in range(len(outlier) - 1):
            if outlier[i] and outlier[i + 1]:
                false_peaks.add(i + 1)
        for i in range(len(ibis)):
            if ibis[i] < 0.6 * local_median[i]:
                false_peaks.add(i + 1)
        if false_peaks:
            keep = [i for i in range(len(peaks)) if i not in false_peaks]
            peaks = peaks[keep]
            peak_times = ecg_ts[peaks]
            peak_heights = props["peak_heights"][keep]
            ibis = np.clip(np.diff(peak_times), 0.3, 2.0)
            ibi_times = peak_times[1:]
            if len(peaks) < 3:
                return features, valid_out, t_out

    # Interpolate IBI to 2 Hz output grid
    ibi_interp = np.interp(t_out, ibi_times, ibis, left=ibis[0], right=ibis[-1])
    hr_interp = 60.0 / ibi_interp

    features[:, 0] = hr_interp
    w10 = min(10, n_out)
    if w10 > 0:
        kernel10 = np.ones(w10) / w10
        ibi_mean = np.convolve(ibi_interp, kernel10, mode="same")
        features[:, 1] = ibi_interp - ibi_mean

    if len(ibis) >= 2:
        succ_diff_sq = np.diff(ibis) ** 2
        sd_times = ibi_times[1:]
        sd_interp = np.interp(t_out, sd_times, succ_diff_sq, left=0, right=0)
        if w10 > 0:
            features[:, 2] = np.sqrt(np.convolve(sd_interp, kernel10, mode="same"))

    w5 = min(4, n_out)
    if w5 > 1:
        kernel5 = np.ones(w5) / w5
        hr_smooth = np.convolve(hr_interp, kernel5, mode="same")
        features[:, 3] = np.gradient(hr_smooth, out_dt)

    amp_interp = np.interp(t_out, peak_times, peak_heights,
                           left=peak_heights[0], right=peak_heights[-1])
    features[:, 4] = amp_interp

    w30 = min(20, n_out)
    if w30 >= 4:
        x = np.arange(w30) - (w30 - 1) / 2.0
        denom = np.sum(x ** 2)
        if denom > 0:
            slope_kernel = x / denom
            features[:, 5] = np.convolve(hr_interp, slope_kernel[::-1], mode="same")

    # RMSSD derivative (Gaussian-smoothed) — V2 channel
    if n_out > 1:
        rmssd = features[:, 2].copy()
        rmssd_deriv = np.diff(rmssd, prepend=rmssd[0]) / out_dt
        from scipy.ndimage import gaussian_filter1d
        sigma_samples = 5.0 * ECG_FEATURES_SRATE
        if sigma_samples > 0.5:
            rmssd_deriv = gaussian_filter1d(rmssd_deriv, sigma_samples)
        std_v = float(np.std(rmssd_deriv))
        if std_v > 1e-8:
            rmssd_deriv = (rmssd_deriv - float(np.mean(rmssd_deriv))) / std_v
        features[:, 6] = np.clip(rmssd_deriv, -10, 10)

    # Z-score channels 0..5 (channel 6 already z-scored above)
    for ch in range(6):
        vals = features[:, ch]
        std_ch = float(np.std(vals))
        if std_ch > 1e-8:
            features[:, ch] = (vals - float(np.mean(vals))) / std_ch
    features = np.clip(features, -10, 10).astype(np.float32)

    # Validity: has R-peak within 5s AND raw ECG was valid
    peak_idx_right = np.clip(np.searchsorted(peak_times, t_out), 0, len(peak_times) - 1)
    peak_idx_left = np.clip(peak_idx_right - 1, 0, len(peak_times) - 1)
    dist_right = np.abs(peak_times[peak_idx_right] - t_out)
    dist_left = np.abs(peak_times[peak_idx_left] - t_out)
    has_nearby_peak = np.minimum(dist_right, dist_left) < 5.0
    ecg_src_idx = np.clip(np.searchsorted(ecg_ts, t_out), 0, len(ecg_valid) - 1)
    valid_out = has_nearby_peak & ecg_valid[ecg_src_idx]
    return features, valid_out, t_out


def preprocess_ecg_session(session_id: str,
                           *,
                           digest_dir: str | Path = "data/digest/v1",
                           out_dir: str | Path | None = None,
                           force: bool = False) -> dict:
    from cadence.ingest.digest import load_digest

    digest_dir = Path(digest_dir)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir("ecg", ECG_MODALITY_VERSION)
    npz_path = out_dir / f"{session_id}.npz"
    json_path = out_dir / f"{session_id}.json"

    cs = load_digest(session_id, digest_dir=digest_dir)
    if not force and staleness_check(json_path, cs.xdf_md5):
        return {"session_id": session_id, "status": "skip-up-to-date"}

    arrays_out: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for p in ("p1", "p2"):
        raw_key = f"{p}_ecg_raw"
        ts_key = f"{p}_ecg_ts"
        if raw_key not in cs.arrays or ts_key not in cs.arrays:
            continue
        ecg_clean, ecg_valid = preprocess_ecg(cs.arrays[raw_key], cs.arrays[ts_key])
        feats, feats_valid, feats_ts = extract_ecg_features(
            ecg_clean, ecg_valid, cs.arrays[ts_key]
        )

        arrays_out[f"{p}_ecg_clean"] = ecg_clean
        arrays_out[f"{p}_ecg_clean_ts"] = cs.arrays[ts_key].astype(np.float64, copy=False)
        arrays_out[f"{p}_ecg_valid"] = ecg_valid
        arrays_out[f"{p}_ecg_features"] = feats
        arrays_out[f"{p}_ecg_features_ts"] = feats_ts
        arrays_out[f"{p}_ecg_features_valid"] = feats_valid
        summary[p] = {
            "n_samples": int(ecg_clean.shape[0]),
            "n_feature_frames": int(feats.shape[0]),
            "feature_valid_pct": float(feats_valid.mean() * 100.0),
        }

    if not arrays_out:
        raise RuntimeError(f"{session_id}: no ECG streams in digest")

    sidecar = {
        "session_id": session_id,
        "modality": "ecg",
        "modality_version": ECG_MODALITY_VERSION,
        "digest_xdf_md5": cs.xdf_md5,
        "digest_schema_version": cs.schema_version,
        "params": {
            "srate_hz": ECG_SRATE_HZ,
            "bandpass_hz": list(ECG_BANDPASS_HZ),
            "features_srate_hz": ECG_FEATURES_SRATE,
            "n_features": ECG_N_FEATURES_V2,
            "feature_names": ECG_FEATURE_NAMES,
            "ectopic_outlier_threshold": 0.40,
        },
        "participants": summary,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    atomic_write_npz(npz_path, **arrays_out)
    atomic_write_json(json_path, sidecar)
    return {"session_id": session_id, "status": "ok",
            "out_npz": str(npz_path), "out_json": str(json_path),
            "summary": summary}
