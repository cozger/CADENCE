"""Coupling burst analysis: continuous intensity, burst detection, per-segment asymmetry.

Core module for the V8.2 burst analysis layer. Provides:
  - Modality group definitions and composite axes
  - Continuous coupling intensity computation
  - Burst event detection (onset, peak, offset, amplitude, duration)
  - Per-segment burst rates, therapist/patient asymmetry
  - Peri-burst averaging for cross-modal timing
  - Pseudo-dyad null and permutation CI for validation

Usage:
    from cadence.significance.burst_analysis import (
        load_session, compute_group_intensity, detect_bursts,
        analyze_session_by_segment, aggregate_sessions,
    )
"""

import os
import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# ── Constants ────────────────────────────────────────────────────────

FS = 2.0  # observation sample rate (Hz)

MOD_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MOD_NAMES = [
    'ImCoh th', 'ImCoh al', 'ImCoh be',
    'Conc th', 'Conc al', 'Conc be',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]
ASYM_KEYS = ['asym_theta', 'asym_alpha', 'asym_beta']

# Modality groups — composites that map to interpretable coupling systems
GROUPS = {
    'EEG phase':  {'idx': [0, 1, 2], 'color': '#1565C0', 'label': 'Phase Coupling (ImCoh)'},
    'EEG power':  {'idx': [3, 4, 5], 'color': '#E65100', 'label': 'Shared Power (Conc)'},
    'Face':       {'idx': [6, 7],    'color': '#E91E63', 'label': 'Facial Coupling (BL)'},
    'Autonomic':  {'idx': [8, 9, 10],'color': '#4CAF50', 'label': 'Autonomic (ECG+Resp)'},
    'Body':       {'idx': [11],      'color': '#795548', 'label': 'Body (Pose)'},
}
GROUP_NAMES = list(GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
STATE_LABELS = ['NULL', 'COUP', 'OTHER', 'SHARED']

# Per-segment condition mapping (no pooling)
SEGMENT_MAP = {
    'base_EO':     ['base_EO'],
    'base_EC':     ['base_EC', 'baseline'],
    'conv_1':      ['conv_1'],
    'conv_2':      ['conv_2'],
    'meditate_B':  ['meditate_B'],
    'meditate_K':  ['meditate_K'],
    'PE_1':        ['PE_1', 'PE'],
    'PE_2':        ['PE_2'],
}
SEGMENT_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
                 'PE_1', 'PE_2', 'conv_2']
SEGMENT_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#5C6BC0',
    'conv_1': '#FFB74D', 'conv_2': '#F57C00',
    'meditate_B': '#CE93D8', 'meditate_K': '#7B1FA2',
    'PE_1': '#F48FB1', 'PE_2': '#C2185B',
}
CONDITION_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#9FA8DA', 'baseline': '#90CAF9',
    'conv_1': '#FFE0B2', 'conv_2': '#FFCC80',
    'PE': '#F8BBD0', 'PE_1': '#F8BBD0', 'PE_2': '#F8BBD0',
    'meditate_B': '#CE93D8', 'meditate_K': '#A5D6A7',
    'gap': '#E0E0E0', 'gap_pre': '#E0E0E0', 'gap_post': '#E0E0E0',
}


# ── Data loading ─────────────────────────────────────────────────────

def get_results_dir():
    """Return absolute path to results/rslds/."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'results', 'rslds')


def load_session(session_id, results_dir=None):
    """Load scaffold observations, Viterbi path, timestamps, segments, asymmetry.

    Returns:
        Y: (T, 12) observation matrix (MOD_KEYS channels)
        viterbi: (T,) integer state assignments
        t: (T,) LSL timestamps
        segments: list of [name, t_start, t_end]
        asym: (T, 3) asymmetry channels (positive = therapist higher)
    """
    if results_dir is None:
        results_dir = get_results_dir()

    sess_dir = os.path.join(results_dir, session_id)
    scaffold_path = os.path.join(sess_dir, 'scaffold_v82_ztimecourses.npz')
    results_path = os.path.join(sess_dir, 'scaffold_v82_results.json')

    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in MOD_KEYS if f'z_{k}' in data])
    t = data['t_common']
    asym = np.column_stack([data[f'z_{k}'] for k in ASYM_KEYS if f'z_{k}' in data])

    # Viterbi path
    v8_path = os.path.join(sess_dir, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(sess_dir, 'rslds_phase2_results.npz')
    if os.path.exists(v8_path):
        viterbi = np.load(v8_path)['full_viterbi']
    elif os.path.exists(p2_path):
        viterbi = np.load(p2_path)['viterbi_path']
    else:
        raise FileNotFoundError(f"No Viterbi path for {session_id}")

    T = min(len(Y), len(viterbi), len(asym))
    Y, viterbi, t, asym = Y[:T], viterbi[:T], t[:T], asym[:T]

    with open(results_path) as f:
        meta = json.load(f)
    segments = meta.get('segments', [])

    return Y, viterbi, t, segments, asym


def load_session_demit(session_id, results_dir=None):
    """Load per-session d_emit (emission means) from v82_rslds_full_results.json."""
    if results_dir is None:
        results_dir = get_results_dir()
    path = os.path.join(results_dir, 'v82_rslds_full_results.json')
    with open(path) as f:
        all_data = json.load(f)
    for entry in all_data:
        if entry['session'] == session_id:
            return np.array(entry['d_emit'])
    raise ValueError(f"Session {session_id} not found in v82_rslds_full_results.json")


# ── Composite axes (domain-informed, not PCA) ───────────────────────

def compute_composite_axes(data, mod_keys=None):
    """Compute interpretable composite scores from observation data.

    Args:
        data: (T, D) or (K, D) array
        mod_keys: list of modality key names (defaults to MOD_KEYS)

    Returns:
        dict of {axis_name: (values, label)}
    """
    if mod_keys is None:
        mod_keys = MOD_KEYS

    def _idx(keys):
        return [mod_keys.index(k) for k in keys if k in mod_keys]

    composites = {}
    imcoh_idx = _idx(['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'])
    conc_idx = _idx(['conc_theta', 'conc_alpha', 'conc_beta'])
    dyn_idx = _idx(['dyn_theta', 'dyn_alpha', 'dyn_beta'])
    body_idx = _idx(['bl_expr', 'bl_activity_conc', 'pose'])
    auto_idx = _idx(['ecg_lf', 'ecg_hf', 'resp'])

    if imcoh_idx:
        composites['phase'] = (data[..., imcoh_idx].mean(axis=-1),
                               'Phase Coupling (mean ImCoh)')
    if conc_idx:
        composites['shared'] = (data[..., conc_idx].mean(axis=-1),
                                'Shared Power (mean Concordance)')
    if dyn_idx:
        composites['dynamics'] = (data[..., dyn_idx].mean(axis=-1),
                                  'EEG Dynamics (mean EWMAD)')
    if body_idx:
        composites['body'] = (data[..., body_idx].mean(axis=-1),
                              'Body Coupling (BL + Pose)')
    if auto_idx:
        composites['autonomic'] = (data[..., auto_idx].mean(axis=-1),
                                   'Autonomic (ECG + Resp)')
    return composites


# ── Coupling intensity ───────────────────────────────────────────────

def compute_group_intensity(Y, smooth_s=3.0):
    """Per-group coupling intensity (smoothed RMS of group channels).

    Returns dict of {group_name: (T,) intensity timecourse}.
    """
    smooth_n = max(1, int(smooth_s * FS))
    intensities = {}
    for gname, ginfo in GROUPS.items():
        rms = np.sqrt(np.mean(Y[:, ginfo['idx']]**2, axis=1))
        intensities[gname] = uniform_filter1d(rms, size=smooth_n, mode='nearest')
    return intensities


def compute_residual_intensity(Y, viterbi, d_emit, smooth_s=3.0):
    """Within-state residual intensity: r_t = y_t - d_emit[z_t].

    Returns per-group residual RMS and raw residuals.
    """
    smooth_n = max(1, int(smooth_s * FS))
    residuals = Y - d_emit[viterbi]
    resid_intensities = {}
    for gname, ginfo in GROUPS.items():
        rms = np.sqrt(np.mean(residuals[:, ginfo['idx']]**2, axis=1))
        resid_intensities[gname] = uniform_filter1d(rms, size=smooth_n, mode='nearest')
    return resid_intensities, residuals


# ── Burst detection ──────────────────────────────────────────────────

def detect_bursts(intensity, threshold_pctl=90, min_duration_s=2.0,
                  merge_gap_s=3.0):
    """Detect coupling bursts as contiguous above-threshold periods.

    Returns:
        bursts: list of dicts with onset, offset, peak_idx, peak_amp, duration_s
        threshold: the percentile threshold used
    """
    threshold = np.percentile(intensity, threshold_pctl)
    above = intensity > threshold
    min_dur = int(min_duration_s * FS)
    merge_gap = int(merge_gap_s * FS)

    changes = np.diff(above.astype(int))
    onsets = np.where(changes == 1)[0] + 1
    offsets = np.where(changes == -1)[0] + 1

    if above[0]:
        onsets = np.concatenate([[0], onsets])
    if above[-1]:
        offsets = np.concatenate([offsets, [len(intensity)]])

    if len(onsets) == 0:
        return [], threshold

    # Merge close bursts
    merged_on, merged_off = [onsets[0]], [offsets[0]]
    for i in range(1, len(onsets)):
        if onsets[i] - merged_off[-1] <= merge_gap:
            merged_off[-1] = offsets[i]
        else:
            merged_on.append(onsets[i])
            merged_off.append(offsets[i])

    bursts = []
    for on, off in zip(merged_on, merged_off):
        if off - on >= min_dur:
            peak = on + np.argmax(intensity[on:off])
            bursts.append({
                'onset': on,
                'offset': off,
                'peak_idx': peak,
                'peak_amp': float(intensity[peak]),
                'duration_s': (off - on) / FS,
            })

    return bursts, threshold


# ── Condition / segment helpers ──────────────────────────────────────

def get_segment_mask(t, segments, segment_name):
    """Boolean mask for timesteps belonging to a named segment."""
    raw_conds = SEGMENT_MAP.get(segment_name, [segment_name])
    mask = np.zeros(len(t), dtype=bool)
    for seg in segments:
        if seg[0] in raw_conds:
            mask |= (t >= seg[1]) & (t <= seg[2])
    return mask


def filter_bursts_by_segment(bursts, t, segments, segment_name):
    """Keep only bursts whose peak falls within the given segment."""
    mask = get_segment_mask(t, segments, segment_name)
    return [b for b in bursts if mask[b['peak_idx']]]


def get_condition_at_time(lsl_time, segments):
    """Map an LSL timestamp to its raw condition name."""
    for seg in segments:
        if seg[1] <= lsl_time <= seg[2]:
            return seg[0]
    return 'gap'


# ── Asymmetry ────────────────────────────────────────────────────────

def compute_burst_asymmetry(asym, bursts, window_s=3.0):
    """Mean asymmetry (therapist - patient) during each burst.

    Returns (n_bursts, 3) array for theta/alpha/beta.
    Positive = therapist has more power during this burst.
    """
    window = int(window_s * FS)
    T = asym.shape[0]
    burst_asym = []
    for burst in bursts:
        peak = burst['peak_idx']
        s = max(0, peak - window)
        e = min(T, peak + window + 1)
        burst_asym.append(asym[s:e].mean(axis=0))
    return np.array(burst_asym) if burst_asym else np.zeros((0, 3))


def compute_participant_power(conc, asym):
    """Decompose concordance + asymmetry into per-participant power.

    Args:
        conc: (T,) concordance timecourse = (z_therapist + z_patient) / 2
        asym: (T,) asymmetry timecourse = z_therapist - z_patient

    Returns:
        z_therapist: (T,) = conc + asym/2
        z_patient: (T,) = conc - asym/2
    """
    return conc + asym / 2.0, conc - asym / 2.0


# ── Peri-burst analysis ──────────────────────────────────────────────

def peri_burst_average(Y, bursts, window_s=10.0):
    """Average all modality group intensities around burst peaks.

    Returns:
        pba: (N_GROUPS, 2*window+1) peri-burst average matrix
        t_axis: (2*window+1,) seconds relative to peak
        n_valid: number of bursts used (excludes edge bursts)
    """
    window = int(window_s * FS)
    T = Y.shape[0]

    raw_int = {}
    for gname, ginfo in GROUPS.items():
        raw_int[gname] = np.sqrt(np.mean(Y[:, ginfo['idx']]**2, axis=1))

    pba = np.zeros((N_GROUPS, 2 * window + 1))
    n_valid = 0
    for burst in bursts:
        peak = burst['peak_idx']
        s, e = peak - window, peak + window + 1
        if s < 0 or e > T:
            continue
        for gi, gname in enumerate(GROUP_NAMES):
            pba[gi] += raw_int[gname][s:e]
        n_valid += 1

    if n_valid > 0:
        pba /= n_valid

    t_axis = np.arange(-window, window + 1) / FS
    return pba, t_axis, n_valid


def peri_burst_peak_latency(Y, bursts, window_s=10.0):
    """Compute peak latency of each modality group's peri-burst average.

    Returns:
        latencies: (N_GROUPS,) peak latencies in seconds (relative to trigger peak)
        n_valid: number of bursts used
    """
    pba, t_axis, n_valid = peri_burst_average(Y, bursts, window_s=window_s)
    if n_valid < 3:
        return np.full(N_GROUPS, np.nan), n_valid
    latencies = np.array([t_axis[np.argmax(pba[gi])] for gi in range(N_GROUPS)])
    return latencies, n_valid


# ── Per-session orchestration ────────────────────────────────────────

def analyze_session_by_segment(session_id, results_dir=None):
    """Full burst analysis for one session, stratified by experimental segment.

    Returns dict with per-segment burst rates, asymmetry, lag matrices.
    """
    Y, viterbi, t, segments, asym = load_session(session_id, results_dir)
    T = Y.shape[0]
    intensities = compute_group_intensity(Y, smooth_s=3.0)

    # Which segments are present in this session?
    available_segments = set()
    for seg in segments:
        for seg_name, raw_conds in SEGMENT_MAP.items():
            if seg[0] in raw_conds:
                available_segments.add(seg_name)

    result = {
        'session': session_id,
        'T': T,
        'duration_min': T / FS / 60.0,
        'segments': {},
    }

    for seg_name in available_segments:
        seg_mask = get_segment_mask(t, segments, seg_name)
        seg_duration_min = seg_mask.sum() / FS / 60.0

        seg_result = {
            'duration_min': seg_duration_min,
            'burst_counts': {},
            'burst_rates': {},
            'burst_asym': {},
            'lag_matrices': {},
        }

        for gi, trigger in enumerate(GROUP_NAMES):
            all_bursts, _ = detect_bursts(intensities[trigger])
            seg_bursts = filter_bursts_by_segment(all_bursts, t, segments, seg_name)

            seg_result['burst_counts'][trigger] = len(seg_bursts)
            seg_result['burst_rates'][trigger] = (
                len(seg_bursts) / seg_duration_min if seg_duration_min > 0 else 0)

            if len(seg_bursts) >= 5:
                latencies, n_valid = peri_burst_peak_latency(Y, seg_bursts)
                seg_result['lag_matrices'][trigger] = latencies.tolist()

                ba = compute_burst_asymmetry(asym, seg_bursts)
                if len(ba) > 0:
                    seg_result['burst_asym'][trigger] = ba.mean(axis=0).tolist()

        result['segments'][seg_name] = seg_result

    return result


# ── Cross-session aggregation ────────────────────────────────────────

def aggregate_sessions(session_results):
    """Aggregate per-session results into cross-session statistics.

    Args:
        session_results: list of dicts from analyze_session_by_segment()

    Returns:
        dict with per-segment mean/SE for burst rates, asymmetry, lag matrices
    """
    agg = {}

    for seg_name in SEGMENT_ORDER:
        seg_rates = {g: [] for g in GROUP_NAMES}
        seg_asym = {g: [] for g in GROUP_NAMES}
        seg_lags = {g: [] for g in GROUP_NAMES}
        n_sessions = 0

        for r in session_results:
            if seg_name not in r['segments']:
                continue
            seg_data = r['segments'][seg_name]
            n_sessions += 1

            for trigger in GROUP_NAMES:
                seg_rates[trigger].append(seg_data['burst_rates'].get(trigger, 0))
                if trigger in seg_data['burst_asym']:
                    seg_asym[trigger].append(seg_data['burst_asym'][trigger])
                if trigger in seg_data['lag_matrices']:
                    seg_lags[trigger].append(seg_data['lag_matrices'][trigger])

        seg_agg = {'n_sessions': n_sessions, 'rates': {}, 'asym': {}, 'lags': {}}

        for trigger in GROUP_NAMES:
            rates = seg_rates[trigger]
            if len(rates) >= 2:
                seg_agg['rates'][trigger] = {
                    'mean': float(np.mean(rates)),
                    'se': float(np.std(rates) / np.sqrt(len(rates))),
                    'n': len(rates),
                }

            asym_vals = seg_asym[trigger]
            if len(asym_vals) >= 2:
                arr = np.array(asym_vals)
                means = arr.mean(axis=0)
                ses = arr.std(axis=0) / np.sqrt(len(arr))
                sig = [bool(abs(means[i]) > 2 * ses[i]) for i in range(3)]
                seg_agg['asym'][trigger] = {
                    'mean': means.tolist(),
                    'se': ses.tolist(),
                    'significant': sig,
                    'n': len(asym_vals),
                    'bands': ['theta', 'alpha', 'beta'],
                }

            lag_vals = seg_lags[trigger]
            if len(lag_vals) >= 2:
                arr = np.array(lag_vals)
                seg_agg['lags'][trigger] = {
                    'mean': np.nanmean(arr, axis=0).tolist(),
                    'se': (np.nanstd(arr, axis=0) / np.sqrt(len(arr))).tolist(),
                    'n': len(lag_vals),
                }

        agg[seg_name] = seg_agg

    return agg


# ── Validation helpers ───────────────────────────────────────────────

def permutation_ci(Y, bursts, n_perm=200, window_s=10.0, seed=42):
    """Circular-shift null distribution for peri-burst peak latencies.

    Returns (n_perm, N_GROUPS) null latencies.
    """
    rng = np.random.default_rng(seed)
    T = Y.shape[0]
    null_lats = np.full((n_perm, N_GROUPS), np.nan)

    for pi in range(n_perm):
        Y_shifted = Y.copy()
        for gname, ginfo in GROUPS.items():
            shift = rng.integers(int(30 * FS), T - int(30 * FS))
            Y_shifted[:, ginfo['idx']] = np.roll(Y[:, ginfo['idx']], shift, axis=0)
        lats, _ = peri_burst_peak_latency(Y_shifted, bursts, window_s=window_s)
        null_lats[pi] = lats

    return null_lats


def pseudo_dyad_lag(sessions_Y, n_pairs=30, seed=42):
    """Pseudo-dyad null: pair EEG from one session with non-EEG from another.

    Args:
        sessions_Y: list of (T_i, D) observation matrices

    Returns:
        (n_pairs, N_GROUPS, N_GROUPS) null lag matrices
    """
    rng = np.random.default_rng(seed)
    n_sess = len(sessions_Y)
    null_lags = []

    for _ in range(n_pairs):
        i, j = rng.choice(n_sess, size=2, replace=False)
        T_min = min(len(sessions_Y[i]), len(sessions_Y[j]))
        Y_pseudo = np.zeros((T_min, sessions_Y[i].shape[1]))
        Y_pseudo[:, :6] = sessions_Y[i][:T_min, :6]   # EEG from session i
        Y_pseudo[:, 6:] = sessions_Y[j][:T_min, 6:]   # non-EEG from session j

        intensities = compute_group_intensity(Y_pseudo, smooth_s=3.0)
        lag_row = np.full((N_GROUPS, N_GROUPS), np.nan)
        for gi, trigger in enumerate(GROUP_NAMES):
            bursts, _ = detect_bursts(intensities[trigger])
            if len(bursts) >= 5:
                lats, _ = peri_burst_peak_latency(Y_pseudo, bursts)
                lag_row[gi] = lats
        null_lags.append(lag_row)

    return np.array(null_lags)
