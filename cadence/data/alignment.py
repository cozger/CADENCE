"""Temporal alignment across streams and full session loading pipeline."""

import hashlib
import json
import os

import numpy as np

from cadence.data.xdf_loader import load_session as _load_xdf
from cadence.data.preprocessors import (
    preprocess_eeg, preprocess_ecg, preprocess_blendshapes, preprocess_pose,
    extract_ecg_features, extract_pose_features, extract_eeg_features,
    EEG_N_FEATURES, compute_activity_channel,
)


def align_session(session):
    """
    Find common time range across all streams and trim to alignment.

    Modifies session dict in-place:
    - Trims all data/timestamp arrays to the common time range
    - Timestamps are zero-referenced (start at 0)
    - Adds 'duration' and 't_start_absolute' keys
    """
    all_starts = []
    all_ends = []

    ALL_MODALITIES = ['eeg', 'ecg', 'blendshapes', 'pose',
                      'ecg_features', 'pose_features', 'eeg_features',
                      'eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2']
    for p in ['p1', 'p2']:
        for mod in ALL_MODALITIES:
            ts_key = f'{p}_{mod}_ts'
            if ts_key in session and len(session[ts_key]) > 0:
                all_starts.append(session[ts_key][0])
                all_ends.append(session[ts_key][-1])

    # Also check non-participant streams (e.g., eeg_interbrain)
    for mod in ['eeg_interbrain']:
        ts_key = f'{mod}_ts'
        if ts_key in session and len(session[ts_key]) > 0:
            all_starts.append(session[ts_key][0])
            all_ends.append(session[ts_key][-1])

    if not all_starts:
        raise ValueError("No valid streams found in session")

    t_start = max(all_starts)
    t_end = min(all_ends)

    if t_end <= t_start:
        raise ValueError(f"No overlapping time range: start={t_start}, end={t_end}")

    for p in ['p1', 'p2']:
        for mod in ALL_MODALITIES:
            ts_key = f'{p}_{mod}_ts'
            data_key = f'{p}_{mod}'
            valid_key = f'{p}_{mod}_valid'

            if ts_key not in session:
                continue

            ts = session[ts_key]
            mask = (ts >= t_start) & (ts <= t_end)

            session[ts_key] = ts[mask] - t_start

            if data_key in session:
                session[data_key] = session[data_key][mask]

            if valid_key in session:
                if session[valid_key].ndim == 1:
                    session[valid_key] = session[valid_key][mask]
                else:
                    session[valid_key] = session[valid_key][mask]

    session['duration'] = t_end - t_start
    session['t_start_absolute'] = t_start
    return session


def load_and_preprocess(xdf_path, p1_eeg_index=0, p2_eeg_index=1):
    """
    Full pipeline: load XDF -> preprocess all modalities -> align.

    Returns:
        session: dict with preprocessed, aligned numpy arrays
    """
    session = _load_xdf(xdf_path, p1_eeg_index, p2_eeg_index)

    for p in ['p1', 'p2']:
        eeg_key = f'{p}_eeg_raw'
        if eeg_key in session:
            eeg, eeg_valid, eeg_ts = preprocess_eeg(
                session[eeg_key], session[f'{p}_eeg_ts']
            )
            session[f'{p}_eeg'] = eeg
            session[f'{p}_eeg_valid'] = eeg_valid
            session[f'{p}_eeg_ts'] = eeg_ts

        ecg_key = f'{p}_ecg_raw'
        if ecg_key in session:
            ecg, ecg_valid, ecg_ts = preprocess_ecg(
                session[ecg_key], session[f'{p}_ecg_ts']
            )
            session[f'{p}_ecg'] = ecg
            session[f'{p}_ecg_valid'] = ecg_valid
            session[f'{p}_ecg_ts'] = ecg_ts

        lm_key = f'{p}_landmarks_raw'
        if lm_key in session:
            bl, bl_valid, bl_ts = preprocess_blendshapes(
                session[lm_key], session[f'{p}_landmarks_ts']
            )
            session[f'{p}_blendshapes'] = bl
            session[f'{p}_blendshapes_valid'] = bl_valid
            session[f'{p}_blendshapes_ts'] = bl_ts

        pose_key = f'{p}_pose_raw'
        if pose_key in session:
            pose, pose_valid, pose_ts = preprocess_pose(
                session[pose_key], session[f'{p}_pose_ts']
            )
            session[f'{p}_pose'] = pose
            session[f'{p}_pose_valid'] = pose_valid
            session[f'{p}_pose_ts'] = pose_ts

    for p in ['p1', 'p2']:
        ecg_key = f'{p}_ecg'
        if ecg_key in session:
            ecg_feats, ecg_feats_valid, ecg_feats_ts = extract_ecg_features(
                session[ecg_key], session[f'{p}_ecg_valid'],
                session[f'{p}_ecg_ts'],
            )
            session[f'{p}_ecg_features'] = ecg_feats
            session[f'{p}_ecg_features_valid'] = ecg_feats_valid
            session[f'{p}_ecg_features_ts'] = ecg_feats_ts

        pose_key = f'{p}_pose'
        if pose_key in session:
            pose_feats, pose_feats_valid, pose_feats_ts = extract_pose_features(
                session[pose_key], session[f'{p}_pose_valid'],
                session[f'{p}_pose_ts'],
            )
            session[f'{p}_pose_features'] = pose_feats
            session[f'{p}_pose_features_valid'] = pose_feats_valid
            session[f'{p}_pose_features_ts'] = pose_feats_ts

        eeg_proc_key = f'{p}_eeg'
        if eeg_proc_key in session:
            eeg_feats, eeg_feats_valid, eeg_feats_ts = extract_eeg_features(
                session[eeg_proc_key], session[f'{p}_eeg_valid'],
                session[f'{p}_eeg_ts'],
            )
            session[f'{p}_eeg_features'] = eeg_feats
            session[f'{p}_eeg_features_valid'] = eeg_feats_valid
            session[f'{p}_eeg_features_ts'] = eeg_feats_ts

    raw_keys = [k for k in session if k.endswith('_raw')]
    for k in raw_keys:
        del session[k]
    for p in ['p1', 'p2']:
        lm_ts = f'{p}_landmarks_ts'
        if lm_ts in session:
            del session[lm_ts]

    session = align_session(session)
    return session


_CACHE_VERSION = 'v7'


def _cache_key(xdf_path):
    """Deterministic cache filename from XDF path + modification time + version."""
    basename = os.path.basename(xdf_path)
    mtime = os.path.getmtime(xdf_path)
    raw = f"{basename}:{mtime}:{_CACHE_VERSION}".encode()
    return hashlib.md5(raw).hexdigest()[:12] + '_' + os.path.splitext(basename)[0]


def _save_session_cache(session, cache_path):
    """Save preprocessed session to .npz + sidecar .json for scalars."""
    arrays = {}
    scalars = {}
    for key, val in session.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
        else:
            scalars[key] = val
    np.savez_compressed(cache_path + '.npz', **arrays)
    with open(cache_path + '.json', 'w') as f:
        json.dump(scalars, f)


def _load_session_cache(cache_path):
    """Load preprocessed session from .npz + sidecar .json."""
    data = np.load(cache_path + '.npz', allow_pickle=False)
    session = {k: data[k] for k in data.files}
    with open(cache_path + '.json') as f:
        scalars = json.load(f)
    session.update(scalars)
    return session


def discover_cached_sessions(cache_dir='session_cache'):
    """Discover sessions available in cache (no XDF files needed).

    Returns list of (session_name, cache_path_prefix) sorted by name,
    using the most recent cache file per session name.
    """
    if not os.path.isdir(cache_dir):
        return []

    from collections import defaultdict
    sessions = defaultdict(list)
    for fname in os.listdir(cache_dir):
        if fname.endswith('.npz'):
            prefix = fname[:-4]
            parts = prefix.split('_', 1)
            if len(parts) == 2:
                _hash, name = parts
                cache_path = os.path.join(cache_dir, prefix)
                if os.path.exists(cache_path + '.json'):
                    mtime = os.path.getmtime(cache_path + '.npz')
                    sessions[name].append((mtime, cache_path))

    result = []
    for name in sorted(sessions.keys()):
        best = max(sessions[name], key=lambda x: x[0])
        result.append((name, best[1]))
    return result


def load_session_from_cache(cache_path, config=None):
    """Load a preprocessed session directly from cache path prefix.

    Performs in-place upgrade: if the cache lacks eeg_features keys (v4 cache),
    computes them from cached preprocessed EEG and re-saves.

    If config specifies pipeline='v2', also computes wavelet features,
    inter-brain features, blendshape v2 features, and ECG v2 features.
    """
    session = _load_session_cache(cache_path)

    needs_resave = False
    for p in ['p1', 'p2']:
        eeg_key = f'{p}_eeg'
        feat_key = f'{p}_eeg_features'
        needs_extract = False
        if eeg_key in session:
            if feat_key not in session:
                needs_extract = True
            elif session[feat_key].shape[1] != EEG_N_FEATURES:
                needs_extract = True
        if needs_extract:
            eeg_feats, eeg_feats_valid, eeg_feats_ts = extract_eeg_features(
                session[eeg_key], session[f'{p}_eeg_valid'],
                session[f'{p}_eeg_ts'],
            )
            session[f'{p}_eeg_features'] = eeg_feats
            session[f'{p}_eeg_features_valid'] = eeg_feats_valid
            session[f'{p}_eeg_features_ts'] = eeg_feats_ts
            needs_resave = True

    if needs_resave:
        print(f"  Upgrading cache with {EEG_N_FEATURES}-ch EEG features: {cache_path}")
        _save_session_cache(session, cache_path)

    session = _ensure_activity_channels(session)

    # V2 pipeline: compute wavelet + interbrain + blendshape v2 + ECG v2
    pipeline = 'v1'
    if config is not None:
        pipeline = config.get('pipeline', 'v1')
    if pipeline == 'v2':
        session = _ensure_v2_features(session, config, cache_path)

    return session


def _ensure_v2_features(session, config, cache_path=None):
    """Compute v2 features if missing from session, optionally re-cache.

    Computes: wavelet EEG, inter-brain phase sync, blendshape PCA,
    expanded ECG features.
    """
    needs_resave = False

    # Wavelet EEG features (per participant)
    for p in ['p1', 'p2']:
        wav_key = f'{p}_eeg_wavelet'
        eeg_key = f'{p}_eeg'
        if wav_key not in session and eeg_key in session:
            from cadence.data.wavelet_features import extract_wavelet_features
            print(f"  Computing wavelet features for {p}...")
            feats, valid, ts = extract_wavelet_features(
                session[eeg_key], session[f'{p}_eeg_valid'],
                session[f'{p}_eeg_ts'], config=config,
            )
            session[wav_key] = feats
            session[f'{wav_key}_valid'] = valid
            session[f'{wav_key}_ts'] = ts
            needs_resave = True

    # Inter-brain features (requires both participants' raw EEG)
    ib_key = 'eeg_interbrain'
    if ib_key not in session and 'p1_eeg' in session and 'p2_eeg' in session:
        from cadence.data.interbrain_features import extract_interbrain_features
        print("  Computing inter-brain features...")
        feats, valid, ts = extract_interbrain_features(
            session['p1_eeg'], session['p2_eeg'],
            session['p1_eeg_valid'], session['p2_eeg_valid'],
            session['p1_eeg_ts'], session['p2_eeg_ts'],
            config=config,
        )
        session[ib_key] = feats
        session[f'{ib_key}_valid'] = valid
        session[f'{ib_key}_ts'] = ts
        needs_resave = True

    # Blendshape v2 (PCA + derivatives)
    for p in ['p1', 'p2']:
        bl_v2_key = f'{p}_blendshapes_v2'
        bl_key = f'{p}_blendshapes'
        if bl_v2_key not in session and bl_key in session:
            from cadence.data.preprocessors import extract_blendshapes_v2
            bl_cfg = config.get('blendshapes_v2', {}) if config else {}
            n_comp = bl_cfg.get('n_pca_components', 15)
            sigma = bl_cfg.get('derivative_sigma_s', 0.5)
            print(f"  Computing blendshape v2 for {p}...")
            feats, valid, ts, pca_loadings = extract_blendshapes_v2(
                session[bl_key], session[f'{bl_key}_valid'],
                session[f'{bl_key}_ts'],
                n_components=n_comp, deriv_sigma_s=sigma,
            )
            session[bl_v2_key] = feats
            session[f'{bl_v2_key}_valid'] = valid
            session[f'{bl_v2_key}_ts'] = ts
            session[f'{bl_v2_key}_pca_loadings'] = pca_loadings
            needs_resave = True

    # ECG v2 (add RMSSD derivative)
    for p in ['p1', 'p2']:
        ecg_v2_key = f'{p}_ecg_features_v2'
        ecg_key = f'{p}_ecg'
        if ecg_v2_key not in session and ecg_key in session:
            from cadence.data.preprocessors import extract_ecg_features_v2
            print(f"  Computing ECG v2 features for {p}...")
            feats, valid, ts = extract_ecg_features_v2(
                session[ecg_key], session[f'{ecg_key}_valid'],
                session[f'{ecg_key}_ts'],
            )
            session[ecg_v2_key] = feats
            session[f'{ecg_v2_key}_valid'] = valid
            session[f'{ecg_v2_key}_ts'] = ts
            needs_resave = True

    if needs_resave and cache_path is not None:
        print(f"  Re-caching with v2 features: {cache_path}")
        _save_session_cache(session, cache_path)

    return session


# Modalities that need activity channels, with (expected_n_ch, approx_hz)
_ACTIVITY_MODS = {
    'eeg_features':  (8, 2.0),
    'blendshapes':   (53, 30.0),
    'pose_features': (41, 12.0),
}


def _ensure_activity_channels(session):
    """Add missing activity channels to modalities that need them."""
    for p in ['p1', 'p2']:
        for mod, (expected_n_ch, approx_hz) in _ACTIVITY_MODS.items():
            data_key = f'{p}_{mod}'
            if data_key not in session:
                continue
            arr = session[data_key]
            if arr.ndim == 1:
                continue
            n_ch = arr.shape[1]
            if n_ch == expected_n_ch:
                continue
            if n_ch == expected_n_ch - 1:
                ts_key = f'{p}_{mod}_ts'
                ts = session.get(ts_key)
                if ts is not None and len(ts) > 1:
                    dt = np.median(np.diff(ts[:min(1000, len(ts))]))
                    hz = 1.0 / max(dt, 1e-6)
                else:
                    hz = approx_hz
                activity = compute_activity_channel(arr, hz, trailing_seconds=30.0)
                session[data_key] = np.concatenate([arr, activity], axis=1)
    return session


# Known bad modalities per session (anomalous capture rates etc.)
EXCLUDED_MODALITIES = {
    'y24_022526': {'p2': ['pose_features', 'pose']},
}


def apply_modality_exclusions(session, session_name):
    """Mask excluded modalities as fully invalid (zeros + False validity)."""
    if session_name not in EXCLUDED_MODALITIES:
        return session
    for participant, mods in EXCLUDED_MODALITIES[session_name].items():
        for mod in mods:
            data_key = f'{participant}_{mod}'
            valid_key = f'{participant}_{mod}_valid'
            if data_key in session:
                session[data_key] = np.zeros_like(session[data_key])
            if valid_key in session:
                session[valid_key] = np.zeros_like(session[valid_key], dtype=bool)
    return session


def load_and_preprocess_cached(xdf_path, cache_dir='session_cache',
                                p1_eeg_index=0, p2_eeg_index=1):
    """
    Load preprocessed session from cache if available, otherwise
    preprocess from XDF and cache the result.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(xdf_path)
    cache_path = os.path.join(cache_dir, key)

    if os.path.exists(cache_path + '.npz') and os.path.exists(cache_path + '.json'):
        print(f"Loading cached session: {os.path.basename(xdf_path)}")
        return _load_session_cache(cache_path)

    print(f"Preprocessing session: {os.path.basename(xdf_path)} (will cache)")
    session = load_and_preprocess(xdf_path, p1_eeg_index, p2_eeg_index)
    _save_session_cache(session, cache_path)
    return session
