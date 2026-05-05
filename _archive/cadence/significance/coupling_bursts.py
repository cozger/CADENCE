"""Coupling Excess Analysis for CADENCE V10.

Replaces session-adaptive burst rate (invalid: self-normalizes, can't
distinguish real from pseudo-dyad) with surrogate-calibrated continuous
coupling excess: at each timepoint, how far does the real coupling signal
exceed what circular-shift surrogates produce?

excess_z(t) = (real(t) - surrogate_mean(t)) / surrogate_std(t)

Positive = genuine coupling above chance.
Negative = below chance (e.g., LZ suppression = shared regularity).

Uses circular-shift surrogates on the already-computed 23D scaffold features
(<1s per session for 200 surrogates). No pseudo-dyad recompute needed.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d as gf1d

# ── Channel tier classification ──────────────────────────────────────

COUPLING_GROUPS = {
    'EEG Phase':   {'idx': [0, 1, 2],  'tier': 1, 'label': 'Phase Coupling (ImCoh)'},
    'Facial':      {'idx': [12],        'tier': 1, 'label': 'Facial Expression Coherence'},
    'LZ Shared':   {'idx': [18, 19],    'tier': 1, 'label': 'Shared Neural Complexity'},
    'Respiratory': {'idx': [16],        'tier': 1, 'label': 'Respiratory Phase Coherence'},
    'Postural':    {'idx': [17],        'tier': 1, 'label': 'Postural Coupling'},
    'EEG Power':   {'idx': [3, 4, 5],  'tier': 2, 'label': 'Shared Power (activity-confounded)'},
    'BL Activity': {'idx': [13],        'tier': 2, 'label': 'Facial Activity (activity-confounded)'},
    'Autonomic':   {'idx': [14, 15],    'tier': 2, 'label': 'Autonomic (activity-confounded)'},
}

# Bidirectional groups: coupling may manifest as suppression (z < 0)
BIDIRECTIONAL_GROUPS = {'LZ Shared'}


# ── Core: continuous coupling excess ─────────────────────────────────

def _group_rms(Y, group_idx, smooth_s=3.0, fs=2.0):
    """Smoothed RMS intensity of a channel group."""
    rms = np.sqrt(np.mean(Y[:, group_idx] ** 2, axis=1))
    if smooth_s > 0:
        rms = gf1d(rms, sigma=smooth_s * fs)
    return rms


def compute_coupling_excess(Y, n_surrogates=200, smooth_s=3.0, fs=2.0, seed=42):
    """Compute per-timepoint coupling excess z-score against circular-shift null.

    For each surrogate, each channel is independently circularly shifted by a
    random offset (≥10% of session length). This destroys temporal coupling
    while preserving autocorrelation and amplitude distribution per channel.

    Args:
        Y: (T, D) observation matrix (already prewhitened/z-scored from scaffold).
        n_surrogates: number of circular-shift surrogates.
        smooth_s: RMS smoothing window in seconds.
        fs: sampling rate.
        seed: random seed.

    Returns:
        excess: dict of {group_name: (T,) coupling excess z-score}.
        null_stats: dict of {group_name: {'mean': (T,), 'std': (T,)}}.
    """
    T, D = Y.shape
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * T))
    max_shift = T - min_shift

    # Compute real intensity per group
    real_intensity = {}
    for gname, ginfo in COUPLING_GROUPS.items():
        real_intensity[gname] = _group_rms(Y, ginfo['idx'], smooth_s, fs)

    # Generate surrogates and accumulate null statistics (Welford online)
    null_mean = {g: np.zeros(T) for g in COUPLING_GROUPS}
    null_m2 = {g: np.zeros(T) for g in COUPLING_GROUPS}

    for si in range(n_surrogates):
        # Circularly shift each channel independently
        Y_surr = np.empty_like(Y)
        for ch in range(D):
            shift = rng.integers(min_shift, max_shift)
            Y_surr[:, ch] = np.roll(Y[:, ch], shift)

        # Compute surrogate intensity per group
        for gname, ginfo in COUPLING_GROUPS.items():
            surr_int = _group_rms(Y_surr, ginfo['idx'], smooth_s, fs)
            # Welford update
            delta = surr_int - null_mean[gname]
            null_mean[gname] += delta / (si + 1)
            delta2 = surr_int - null_mean[gname]
            null_m2[gname] += delta * delta2

    # Finalize null statistics
    null_stats = {}
    excess = {}
    for gname in COUPLING_GROUPS:
        null_std = np.sqrt(null_m2[gname] / max(n_surrogates - 1, 1))
        null_std = np.maximum(null_std, 1e-8)  # prevent division by zero
        null_stats[gname] = {
            'mean': null_mean[gname],
            'std': null_std,
        }
        excess[gname] = (real_intensity[gname] - null_mean[gname]) / null_std

    return excess, null_stats


# ── Discrete coupling events ────────────────────────────────────────

def detect_coupling_events(excess_z, threshold=2.0, min_dur_s=2.0,
                           merge_gap_s=3.0, fs=2.0, bidirectional=False):
    """Threshold coupling excess z-score for discrete events.

    z > threshold = coupling excitation event.
    z < -threshold = coupling suppression event (only if bidirectional=True).

    Args:
        excess_z: (T,) coupling excess z-score timecourse.
        threshold: z-score threshold (2.0 = 97.7th percentile).
        min_dur_s: minimum event duration.
        merge_gap_s: merge events within this gap.
        bidirectional: if True, also detect suppression (z < -threshold).
        fs: sampling rate.

    Returns:
        events: list of dicts with onset, offset, duration_s, peak_z, direction.
    """
    min_samp = int(min_dur_s * fs)
    merge_samp = int(merge_gap_s * fs)

    def _detect_one_direction(signal, direction_label):
        above = signal > threshold
        events = []
        in_event = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_event:
                start = i
                in_event = True
            elif not above[i] and in_event:
                events.append((start, i))
                in_event = False
        if in_event:
            events.append((start, len(above)))

        # Merge nearby events
        merged = []
        for s, e in events:
            if merged and s - merged[-1][1] <= merge_samp:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Filter by duration and build output
        result = []
        for s, e in merged:
            dur = (e - s) / fs
            if dur >= min_dur_s:
                peak_idx = s + np.argmax(signal[s:e])
                result.append({
                    'onset': int(s), 'offset': int(e),
                    'duration_s': float(dur),
                    'peak_z': float(signal[peak_idx]),
                    'peak_idx': int(peak_idx),
                    'direction': direction_label,
                })
        return result

    events = _detect_one_direction(excess_z, 'excitation')
    if bidirectional:
        events += _detect_one_direction(-excess_z, 'suppression')
        # For suppression events, flip peak_z sign back
        for e in events:
            if e['direction'] == 'suppression':
                e['peak_z'] = -e['peak_z']

    return sorted(events, key=lambda e: e['onset'])


# ── Event Coincidence Analysis (ECA) ─────────────────────────────────

def event_coincidence_analysis(events_a, events_b, T, tau_s=5.0, fs=2.0):
    """Event Coincidence Analysis between two event sets.

    Tests whether events in A cluster around events in B more than chance.
    Analytic null from Odenweller et al. 2020.

    Args:
        events_a, events_b: lists of event dicts (must have 'peak_idx').
        T: total number of timepoints.
        tau_s: coincidence window in seconds.
        fs: sampling rate.

    Returns:
        dict with trigger_rate, precursor_rate, p_trigger, p_precursor.
    """
    tau = int(tau_s * fs)
    na, nb = len(events_a), len(events_b)

    if na == 0 or nb == 0:
        return {'trigger_rate': 0.0, 'precursor_rate': 0.0,
                'p_trigger': 1.0, 'p_precursor': 1.0, 'n_a': na, 'n_b': nb}

    peaks_a = np.array([e['peak_idx'] for e in events_a])
    peaks_b = np.array([e['peak_idx'] for e in events_b])

    # Trigger coincidence: fraction of A events preceded by B within tau
    n_trigger = 0
    for pa in peaks_a:
        if np.any(np.abs(peaks_b - pa) <= tau):
            n_trigger += 1
    trigger_rate = n_trigger / na

    # Precursor coincidence: fraction of B events followed by A within tau
    n_precursor = 0
    for pb in peaks_b:
        if np.any(np.abs(peaks_a - pb) <= tau):
            n_precursor += 1
    precursor_rate = n_precursor / nb

    # Analytic null: expected coincidence rate under independence
    # P(coincidence) ≈ 1 - (1 - 2*tau/T)^n_other
    p_null_trigger = 1 - (1 - 2 * tau / T) ** nb
    p_null_precursor = 1 - (1 - 2 * tau / T) ** na

    # Binomial p-value (one-sided: more coincidences than expected)
    from scipy.stats import binom
    p_trigger = 1 - binom.cdf(n_trigger - 1, na, p_null_trigger) if p_null_trigger < 1 else 1.0
    p_precursor = 1 - binom.cdf(n_precursor - 1, nb, p_null_precursor) if p_null_precursor < 1 else 1.0

    return {
        'trigger_rate': float(trigger_rate),
        'precursor_rate': float(precursor_rate),
        'p_trigger': float(p_trigger),
        'p_precursor': float(p_precursor),
        'n_a': na, 'n_b': nb,
        'expected_trigger': float(p_null_trigger),
        'expected_precursor': float(p_null_precursor),
    }


def compute_coincidence_matrix(events_by_group, T, tau_s=5.0, fs=2.0):
    """All-pairs ECA between modality groups.

    Returns dict with 'trigger' and 'precursor' matrices + p-values.
    """
    group_names = list(events_by_group.keys())
    n = len(group_names)
    trigger = np.zeros((n, n))
    precursor = np.zeros((n, n))
    p_trigger = np.ones((n, n))
    p_precursor = np.ones((n, n))

    for i, ga in enumerate(group_names):
        for j, gb in enumerate(group_names):
            if i == j:
                continue
            eca = event_coincidence_analysis(
                events_by_group[ga], events_by_group[gb], T, tau_s, fs)
            trigger[i, j] = eca['trigger_rate']
            precursor[i, j] = eca['precursor_rate']
            p_trigger[i, j] = eca['p_trigger']
            p_precursor[i, j] = eca['p_precursor']

    return {
        'group_names': group_names,
        'trigger': trigger,
        'precursor': precursor,
        'p_trigger': p_trigger,
        'p_precursor': p_precursor,
    }


# ── Asymmetry during coupling events ────────────────────────────────

def coupling_event_asymmetry(Y, events, asym_channels=None, window_s=3.0, fs=2.0):
    """Mean asymmetry (therapist - patient) during coupling events.

    Args:
        Y: (T, D) observation matrix.
        events: list of event dicts.
        asym_channels: channel indices for asymmetry. Default: [9,10,11] (theta/alpha/beta).
        window_s: averaging window around peak.
        fs: sampling rate.

    Returns:
        dict with per-band mean asymmetry and per-event values.
    """
    if asym_channels is None:
        asym_channels = [9, 10, 11]
    band_names = ['theta', 'alpha', 'beta']

    if not events:
        return {bn: {'mean': 0.0, 'values': []} for bn in band_names[:len(asym_channels)]}

    win_samp = int(window_s * fs)
    T = Y.shape[0]
    result = {}

    for bi, (ch, bn) in enumerate(zip(asym_channels, band_names)):
        vals = []
        for ev in events:
            pk = ev['peak_idx']
            s = max(0, pk - win_samp)
            e = min(T, pk + win_samp + 1)
            vals.append(float(Y[s:e, ch].mean()))
        result[bn] = {
            'mean': float(np.mean(vals)) if vals else 0.0,
            'se': float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            'n': len(vals),
        }

    return result


# ── Full session analysis ────────────────────────────────────────────

def analyze_session(session_id, results_dir='results/v10', n_surrogates=200,
                    event_threshold=2.0, seed=42):
    """Complete coupling excess analysis for one V10 session.

    1. Load 23D scaffold features
    2. Compute coupling excess (200 surrogates)
    3. Per-condition mean excess
    4. Detect coupling events (z > 2.0 / z < -2.0 for LZ)
    5. ECA coincidence matrix
    6. Asymmetry during coupling events
    """
    from cadence.constants import V10_MODALITY_KEYS

    npz_path = f'{results_dir}/{session_id}/scaffold_v10_ztimecourses.npz'
    json_path = f'{results_dir}/{session_id}/scaffold_v10_results.json'

    import json
    data = np.load(npz_path)
    with open(json_path) as f:
        info = json.load(f)

    Y = np.column_stack([data[f'z_{k}'] for k in V10_MODALITY_KEYS]).astype(np.float64)
    t = data['t_common']
    T = len(t)
    segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

    # 1. Coupling excess
    excess, null_stats = compute_coupling_excess(Y, n_surrogates=n_surrogates,
                                                  seed=seed)

    # 2. Per-condition mean excess
    cond_excess = {}
    for cond_name, t0, t1 in segments:
        mask = (t >= t0) & (t <= t1)
        if mask.sum() < 10:
            continue
        cond_excess[cond_name] = {}
        for gname in COUPLING_GROUPS:
            cond_excess[cond_name][gname] = float(excess[gname][mask].mean())

    # 3. Detect coupling events per group
    events_by_group = {}
    for gname, ginfo in COUPLING_GROUPS.items():
        bidir = gname in BIDIRECTIONAL_GROUPS
        events = detect_coupling_events(
            excess[gname], threshold=event_threshold,
            bidirectional=bidir)
        events_by_group[gname] = events

    # 4. ECA coincidence matrix (Tier 1 groups only)
    tier1_events = {g: events_by_group[g] for g in COUPLING_GROUPS
                    if COUPLING_GROUPS[g]['tier'] == 1}
    coincidence = compute_coincidence_matrix(tier1_events, T)

    # 5. Asymmetry during coupling events (use EEG Phase events as trigger)
    eeg_events = events_by_group.get('EEG Phase', [])
    asymmetry = coupling_event_asymmetry(Y, eeg_events)

    # Per-condition asymmetry
    cond_asymmetry = {}
    for cond_name, t0, t1 in segments:
        cond_events = [e for e in eeg_events
                       if t[e['peak_idx']] >= t0 and t[e['peak_idx']] <= t1]
        if cond_events:
            cond_asymmetry[cond_name] = coupling_event_asymmetry(Y, cond_events)

    # Compile results
    dur_min = (t[-1] - t[0]) / 60.0
    event_summary = {}
    for gname, events in events_by_group.items():
        n_exc = sum(1 for e in events if e['direction'] == 'excitation')
        n_sup = sum(1 for e in events if e['direction'] == 'suppression')
        event_summary[gname] = {
            'n_excitation': n_exc,
            'n_suppression': n_sup,
            'rate_excitation': n_exc / dur_min if dur_min > 0 else 0,
            'rate_suppression': n_sup / dur_min if dur_min > 0 else 0,
            'mean_excess': float(excess[gname].mean()),
        }

    return {
        'session': session_id,
        'n_timepoints': T,
        'duration_min': float(dur_min),
        'per_condition_excess': cond_excess,
        'event_summary': event_summary,
        'coincidence': {
            'group_names': coincidence['group_names'],
            'trigger': coincidence['trigger'].tolist(),
            'p_trigger': coincidence['p_trigger'].tolist(),
        },
        'asymmetry': {bn: {'mean': v['mean'], 'se': v['se'], 'n': v['n']}
                      for bn, v in asymmetry.items()},
        'per_condition_asymmetry': {
            cond: {bn: {'mean': v['mean'], 'se': v['se'], 'n': v['n']}
                   for bn, v in asym.items()}
            for cond, asym in cond_asymmetry.items()
        },
    }
