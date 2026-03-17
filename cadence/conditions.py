"""Condition/segment analysis: map time-varying dR2 to experimental conditions.

Parses XDF markers into condition intervals, classifies each timepoint
as sync/T-leads/P-leads/none, computes per-condition statistics.
"""

import numpy as np
from collections import OrderedDict


# Known condition labels and display names
CONDITION_LABELS = OrderedDict([
    ('base_EO', 'Baseline (EO)'),
    ('base_EC', 'Baseline (EC)'),
    ('baseline', 'Baseline'),
    ('conv_1', 'Conversation 1'),
    ('conv_2', 'Conversation 2'),
    ('PE', 'Psychoeducation'),
    ('PE_1', 'Psychoeducation 1'),
    ('PE_2', 'Psychoeducation 2'),
    ('meditate_B', 'Meditation (Body)'),
    ('meditate_K', 'Meditation (Kindness)'),
])

CONDITION_COLORS = {
    'base_EO': '#E0E0E0',
    'base_EC': '#BDBDBD',
    'baseline': '#D0D0D0',
    'conv_1': '#FFECB3',
    'conv_2': '#FFE082',
    'PE': '#E1BEE7',
    'PE_1': '#E1BEE7',
    'PE_2': '#CE93D8',
    'meditate_B': '#C8E6C9',
    'meditate_K': '#B3E5FC',
}


def parse_condition_intervals(session):
    """Parse session markers into condition intervals.

    Args:
        session: Session dict with 'markers' and 't_start_absolute'.

    Returns:
        List of (start_s, end_s, condition_key) tuples in session-relative seconds.
    """
    markers = session.get('markers', [])
    t0 = session.get('t_start_absolute', 0)
    duration = session.get('duration', 0)

    if not markers:
        return []

    # Convert to session-relative seconds
    events = [(m[0] - t0, m[1]) for m in markers]

    # Pair start/stop markers
    intervals = []
    open_starts = {}

    for t, label in sorted(events):
        if label.endswith('_start'):
            cond = label[:-6]  # strip '_start'
            open_starts[cond] = max(0, t)  # clamp to session start
        elif label.endswith('_stop'):
            cond = label[:-5]  # strip '_stop'
            if cond in open_starts:
                start = open_starts.pop(cond)
                end = min(t, duration)
                if end > start:
                    intervals.append((start, end, cond))
            # else: orphan stop marker — skip

    # Close any unclosed intervals at session end
    for cond, start in open_starts.items():
        if duration > start:
            intervals.append((start, duration, cond))

    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    return intervals


def _align_dr2_pair(result_tp, result_pt, pathway_key):
    """Align dR2 timecourses from both directions onto a shared time grid.

    Returns:
        times, dr2_tp, dr2_pt — all (T,) arrays on the same grid.
    """
    tp_times = result_tp.pathway_times.get(pathway_key, result_tp.times)
    pt_times = result_pt.pathway_times.get(pathway_key, result_pt.times)

    if len(tp_times) <= len(pt_times):
        times = tp_times
        dr2_tp = result_tp.pathway_dr2[pathway_key]
        dr2_pt_raw = result_pt.pathway_dr2[pathway_key]
        dr2_pt = np.interp(times, pt_times, np.nan_to_num(dr2_pt_raw, nan=0.0))
    else:
        times = pt_times
        dr2_pt = result_pt.pathway_dr2[pathway_key]
        dr2_tp_raw = result_tp.pathway_dr2[pathway_key]
        dr2_tp = np.interp(times, tp_times, np.nan_to_num(dr2_tp_raw, nan=0.0))

    dr2_tp = np.nan_to_num(dr2_tp, nan=0.0)
    dr2_pt = np.nan_to_num(dr2_pt, nan=0.0)
    return times, dr2_tp, dr2_pt


def classify_timepoints(dr2_tp, dr2_pt, threshold_tp, threshold_pt):
    """Classify each timepoint as sync/T-leads/P-leads/none.

    Uses per-direction thresholds: a direction is "active" at a timepoint
    only if its dR2 exceeds the threshold for that direction.

    Args:
        dr2_tp: (T,) T->P dR2 values.
        dr2_pt: (T,) P->T dR2 values.
        threshold_tp: Scalar threshold for T->P direction.
        threshold_pt: Scalar threshold for P->T direction.

    Returns:
        labels: (T,) array of 0=none, 1=T-leads, 2=P-leads, 3=sync.
    """
    tp_active = dr2_tp > threshold_tp
    pt_active = dr2_pt > threshold_pt

    labels = np.zeros(len(dr2_tp), dtype=int)
    labels[tp_active & ~pt_active] = 1  # T-leads
    labels[~tp_active & pt_active] = 2  # P-leads
    labels[tp_active & pt_active] = 3   # sync

    return labels


def _compute_adaptive_threshold(dr2, method='median'):
    """Compute an adaptive threshold for one direction's dR2 timecourse.

    Args:
        dr2: (T,) dR2 values.
        method: 'median' (session median), 'q25' (25th percentile),
                or a float for a fixed absolute threshold.

    Returns:
        Scalar threshold.
    """
    if isinstance(method, (int, float)):
        return float(method)
    if method == 'median':
        return float(np.median(dr2))
    if method == 'q25':
        return float(np.percentile(dr2, 25))
    return 0.0


def condition_statistics(times, labels, dr2_tp, dr2_pt, intervals):
    """Compute per-condition coupling statistics.

    Args:
        times: (T,) time array in seconds.
        labels: (T,) 0=none, 1=T-leads, 2=P-leads, 3=sync.
        dr2_tp: (T,) T->P dR2 values.
        dr2_pt: (T,) P->T dR2 values.
        intervals: List of (start_s, end_s, condition_key).

    Returns:
        Dict of condition_key -> stats dict.
    """
    stats = OrderedDict()

    # Build condition masks
    condition_masks = []
    covered = np.zeros(len(times), dtype=bool)

    for start, end, cond in intervals:
        mask = (times >= start) & (times <= end)
        condition_masks.append((cond, mask))
        covered |= mask

    # Add uncategorized and all
    condition_masks.append(('uncategorized', ~covered))
    condition_masks.append(('all', np.ones(len(times), dtype=bool)))

    for cond, mask in condition_masks:
        n = mask.sum()
        if n == 0:
            continue

        cond_labels = labels[mask]
        cond_tp = dr2_tp[mask]
        cond_pt = dr2_pt[mask]

        stats[cond] = {
            'label': CONDITION_LABELS.get(cond, cond.replace('_', ' ').title()),
            'n_timepoints': int(n),
            'duration_s': float(times[mask][-1] - times[mask][0]) if n > 1 else 0.0,
            'sync_pct': float(np.mean(cond_labels == 3) * 100),
            't_leads_pct': float(np.mean(cond_labels == 1) * 100),
            'p_leads_pct': float(np.mean(cond_labels == 2) * 100),
            'none_pct': float(np.mean(cond_labels == 0) * 100),
            'mean_dr2_tp': float(np.mean(cond_tp)),
            'mean_dr2_pt': float(np.mean(cond_pt)),
            'peak_dr2_tp': float(np.max(cond_tp)) if n > 0 else 0.0,
            'peak_dr2_pt': float(np.max(cond_pt)) if n > 0 else 0.0,
            'coupling_pct': float(np.mean((cond_labels == 1) | (cond_labels == 2) | (cond_labels == 3)) * 100),
        }

    return stats


def _average_significant_dr2(result_tp, result_pt, pathway_keys, finest_times):
    """Average dR2 across significant pathways only, interpolated to finest_times.

    For each pathway, a direction is only included if it's significant in that
    direction. Non-significant directions contribute 0 (no coupling signal).

    Returns:
        avg_dr2_tp, avg_dr2_pt: (T,) averaged dR2 arrays.
        n_sig_tp, n_sig_pt: Number of contributing significant pathways per direction.
    """
    avg_dr2_tp = np.zeros(len(finest_times))
    avg_dr2_pt = np.zeros(len(finest_times))
    n_sig_tp = 0
    n_sig_pt = 0

    for key in pathway_keys:
        tp_times = result_tp.pathway_times.get(key, result_tp.times)
        pt_times = result_pt.pathway_times.get(key, result_pt.times)

        # Only include direction if pathway is significant in that direction
        if result_tp.pathway_significant.get(key, False):
            dr2_tp_raw = np.nan_to_num(result_tp.pathway_dr2[key], nan=0.0)
            avg_dr2_tp += np.interp(finest_times, tp_times, dr2_tp_raw)
            n_sig_tp += 1

        if result_pt.pathway_significant.get(key, False):
            dr2_pt_raw = np.nan_to_num(result_pt.pathway_dr2[key], nan=0.0)
            avg_dr2_pt += np.interp(finest_times, pt_times, dr2_pt_raw)
            n_sig_pt += 1

    if n_sig_tp > 0:
        avg_dr2_tp /= n_sig_tp
    if n_sig_pt > 0:
        avg_dr2_pt /= n_sig_pt

    return avg_dr2_tp, avg_dr2_pt, n_sig_tp, n_sig_pt


def session_condition_summary(result_tp, result_pt, session, pathways=None,
                              threshold_method='median', modality_order=None,
                              mod_short=None):
    """Compute full condition summary for one session.

    Uses significance-gated pathways and adaptive thresholds:
    - Only pathways significant in a given direction contribute to that direction
    - Threshold = session median of averaged dR2 (not raw > 0)
    - This means "active coupling" = above the session's baseline coupling level

    Args:
        result_tp: CouplingResult for therapist->patient.
        result_pt: CouplingResult for patient->therapist.
        session: Session dict (for markers).
        pathways: List of (src_mod, tgt_mod) to analyze. If None, uses all
                  cross-modal pathways present in both results.
        threshold_method: 'median' (session median), 'q25' (25th percentile),
                          or a float for fixed threshold.

    Returns:
        Dict with per-pathway, per-condition statistics.
    """
    intervals = parse_condition_intervals(session)

    if pathways is None:
        pathways = [k for k in result_tp.pathway_dr2
                    if k in result_pt.pathway_dr2]

    if mod_short is None:
        from cadence.constants import MOD_SHORT
        mod_short = MOD_SHORT

    summary = {
        'conditions_found': [c for _, _, c in intervals],
        'n_intervals': len(intervals),
        'threshold_method': str(threshold_method),
        'pathways': {},
        'modality_summary': {},
    }

    # Per-pathway condition stats (significance-gated per direction)
    for key in pathways:
        src_mod, tgt_mod = key
        pkey = f"{mod_short.get(src_mod, src_mod)}->{mod_short.get(tgt_mod, tgt_mod)}"

        times, dr2_tp, dr2_pt = _align_dr2_pair(result_tp, result_pt, key)

        # Zero out non-significant directions
        tp_sig = result_tp.pathway_significant.get(key, False)
        pt_sig = result_pt.pathway_significant.get(key, False)
        if not tp_sig:
            dr2_tp = np.zeros_like(dr2_tp)
        if not pt_sig:
            dr2_pt = np.zeros_like(dr2_pt)

        # Adaptive thresholds per direction
        thresh_tp = _compute_adaptive_threshold(dr2_tp, threshold_method) if tp_sig else 0.0
        thresh_pt = _compute_adaptive_threshold(dr2_pt, threshold_method) if pt_sig else 0.0

        labels = classify_timepoints(dr2_tp, dr2_pt, thresh_tp, thresh_pt)
        stats = condition_statistics(times, labels, dr2_tp, dr2_pt, intervals)
        summary['pathways'][pkey] = stats

    # Modality-level summary (average across SIGNIFICANT cross-modal pathways)
    if modality_order is None:
        from cadence.constants import MODALITY_ORDER
        modality_order = MODALITY_ORDER
    finest_times = result_tp.times

    for tgt_mod in modality_order:
        tgt_pathways = [k for k in pathways if k[1] == tgt_mod and k[0] != k[1]]
        if not tgt_pathways:
            continue

        tgt_short = mod_short.get(tgt_mod, tgt_mod)
        avg_dr2_tp, avg_dr2_pt, n_tp, n_pt = _average_significant_dr2(
            result_tp, result_pt, tgt_pathways, finest_times)

        if n_tp == 0 and n_pt == 0:
            continue

        thresh_tp = _compute_adaptive_threshold(avg_dr2_tp, threshold_method) if n_tp > 0 else 0.0
        thresh_pt = _compute_adaptive_threshold(avg_dr2_pt, threshold_method) if n_pt > 0 else 0.0

        labels = classify_timepoints(avg_dr2_tp, avg_dr2_pt, thresh_tp, thresh_pt)
        stats = condition_statistics(
            finest_times, labels, avg_dr2_tp, avg_dr2_pt, intervals)
        summary['modality_summary'][tgt_short] = stats

    # Overall summary (average across all SIGNIFICANT cross-modal pathways)
    cross_modal = [k for k in pathways if k[0] != k[1]]
    if cross_modal:
        avg_dr2_tp, avg_dr2_pt, n_tp, n_pt = _average_significant_dr2(
            result_tp, result_pt, cross_modal, finest_times)

        thresh_tp = _compute_adaptive_threshold(avg_dr2_tp, threshold_method) if n_tp > 0 else 0.0
        thresh_pt = _compute_adaptive_threshold(avg_dr2_pt, threshold_method) if n_pt > 0 else 0.0

        labels = classify_timepoints(avg_dr2_tp, avg_dr2_pt, thresh_tp, thresh_pt)
        stats = condition_statistics(
            finest_times, labels, avg_dr2_tp, avg_dr2_pt, intervals)
        summary['modality_summary']['Overall'] = stats

    return summary


def aggregate_condition_summaries(session_summaries):
    """Aggregate per-session condition summaries into grand averages.

    Args:
        session_summaries: List of (session_name, summary_dict) tuples.

    Returns:
        Grand summary dict with per-condition mean/std across sessions.
    """
    stat_keys = [
        'sync_pct', 't_leads_pct', 'p_leads_pct', 'none_pct',
        'mean_dr2_tp', 'mean_dr2_pt', 'coupling_pct',
    ]

    # Collect all conditions across sessions
    all_conditions = set()
    for _, summ in session_summaries:
        if 'modality_summary' in summ and 'Overall' in summ['modality_summary']:
            all_conditions.update(summ['modality_summary']['Overall'].keys())

    # Known order first, then unknown, then uncategorized/all
    ordered_conds = []
    for c in CONDITION_LABELS:
        if c in all_conditions:
            ordered_conds.append(c)
    for c in sorted(all_conditions - set(CONDITION_LABELS) - {'uncategorized', 'all'}):
        ordered_conds.append(c)
    if 'uncategorized' in all_conditions:
        ordered_conds.append('uncategorized')
    if 'all' in all_conditions:
        ordered_conds.append('all')

    # Collect all modality keys
    all_mod_keys = set()
    for _, summ in session_summaries:
        all_mod_keys.update(summ.get('modality_summary', {}).keys())

    grand = {
        'n_sessions': len(session_summaries),
        'session_names': [n for n, _ in session_summaries],
        'conditions': OrderedDict(),
    }

    for cond in ordered_conds:
        cond_data = {
            'label': CONDITION_LABELS.get(cond, cond.replace('_', ' ').title()),
            'sessions': [],
            'modalities': {},
        }

        for mod_key in sorted(all_mod_keys):
            values = {k: [] for k in stat_keys}
            contributing_sessions = []

            for sess_name, summ in session_summaries:
                mod_stats = summ.get('modality_summary', {}).get(mod_key, {})
                if cond in mod_stats:
                    for k in stat_keys:
                        values[k].append(mod_stats[cond].get(k, 0))
                    contributing_sessions.append(sess_name)

            if not contributing_sessions:
                continue

            mod_result = {
                'n_sessions': len(contributing_sessions),
                'sessions': contributing_sessions,
            }
            for k in stat_keys:
                vals = np.array(values[k])
                mod_result[f'{k}_mean'] = float(np.mean(vals))
                mod_result[f'{k}_std'] = float(np.std(vals))
                mod_result[f'{k}_values'] = [float(v) for v in vals]

            cond_data['modalities'][mod_key] = mod_result
            if mod_key == 'Overall':
                cond_data['sessions'] = contributing_sessions
                cond_data['n_sessions'] = len(contributing_sessions)

        grand['conditions'][cond] = cond_data

    return grand
