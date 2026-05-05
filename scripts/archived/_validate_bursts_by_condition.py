"""Condition-Stratified Burst Validation: lead/lag and asymmetry per experimental segment.

Each protocol segment analyzed individually (no pooling):
  Baselines:     base_EO (eyes open), base_EC (eyes closed)
  Conversations: conv_1 (pre-intervention), conv_2 (post-intervention)
  Meditation:    meditate_B (body scan), meditate_K (kindness)
  PE:            PE_1 (psychoeducation block 1), PE_2 (block 2)

Temporal structure matters:
  - conv_1 vs conv_2: post-intervention conversation may show stronger coupling
  - meditate_B vs meditate_K: different meditation styles, different EEG signatures
  - PE_1 vs PE_2: fatigue/habituation effects
  - base_EO vs base_EC: eyes open has visual coupling, eyes closed does not
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'rslds')

MOD_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
ASYM_KEYS = ['asym_theta', 'asym_alpha', 'asym_beta']

GROUPS = {
    'EEG phase': {'idx': [0, 1, 2], 'color': '#1565C0'},
    'EEG power': {'idx': [3, 4, 5], 'color': '#E65100'},
    'Face':      {'idx': [6, 7],    'color': '#E91E63'},
    'Autonomic': {'idx': [8, 9, 10],'color': '#4CAF50'},
    'Body':      {'idx': [11],      'color': '#795548'},
}
GROUP_NAMES = list(GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)
FS = 2.0

# Each segment is its own condition — no pooling
CONDITION_GROUPS = {
    'base_EO':     ['base_EO'],
    'base_EC':     ['base_EC', 'baseline'],
    'conv_1':      ['conv_1'],
    'conv_2':      ['conv_2'],
    'meditate_B':  ['meditate_B'],
    'meditate_K':  ['meditate_K'],
    'PE_1':        ['PE_1', 'PE'],         # some sessions use single 'PE' marker
    'PE_2':        ['PE_2'],
}
COND_GROUP_COLORS = {
    'base_EO':    '#90CAF9',
    'base_EC':    '#5C6BC0',
    'conv_1':     '#FFB74D',
    'conv_2':     '#F57C00',
    'meditate_B': '#CE93D8',
    'meditate_K': '#7B1FA2',
    'PE_1':       '#F48FB1',
    'PE_2':       '#C2185B',
}
# Temporal order within a session
COND_GROUP_ORDER = ['base_EO', 'base_EC', 'conv_1',
                    'meditate_B', 'meditate_K',
                    'PE_1', 'PE_2',
                    'conv_2']


def load_session(session_id):
    scaffold_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_ztimecourses.npz')
    results_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_results.json')
    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in MOD_KEYS if f'z_{k}' in data])
    t = data['t_common']
    asym = np.column_stack([data[f'z_{k}'] for k in ASYM_KEYS if f'z_{k}' in data])

    v8_path = os.path.join(RESULTS_DIR, session_id, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(RESULTS_DIR, session_id, 'rslds_phase2_results.npz')
    if os.path.exists(v8_path):
        viterbi = np.load(v8_path)['full_viterbi']
    else:
        viterbi = np.load(p2_path)['viterbi_path']

    T = min(len(Y), len(viterbi), len(asym))

    with open(results_path) as f:
        meta = json.load(f)
    segments = meta.get('segments', [])

    return Y[:T], viterbi[:T], t[:T], segments, asym[:T]


def get_condition_group(lsl_time, segments):
    """Map LSL timestamp to condition group."""
    for seg in segments:
        if seg[1] <= lsl_time <= seg[2]:
            raw_cond = seg[0]
            for group_name, cond_list in CONDITION_GROUPS.items():
                if raw_cond in cond_list:
                    return group_name
    return None  # gap — exclude


def get_condition_mask(t, segments, cond_group):
    """Boolean mask for timesteps belonging to a condition group."""
    mask = np.zeros(len(t), dtype=bool)
    for seg in segments:
        raw_cond = seg[0]
        for gname, cond_list in CONDITION_GROUPS.items():
            if gname == cond_group and raw_cond in cond_list:
                mask |= (t >= seg[1]) & (t <= seg[2])
    return mask


def compute_group_intensity(Y, smooth_s=3.0):
    smooth_n = max(1, int(smooth_s * FS))
    intensities = {}
    for gname, ginfo in GROUPS.items():
        rms = np.sqrt(np.mean(Y[:, ginfo['idx']]**2, axis=1))
        intensities[gname] = uniform_filter1d(rms, size=smooth_n, mode='nearest')
    return intensities


def detect_bursts(intensity, threshold_pctl=90, min_duration_s=2.0, merge_gap_s=3.0):
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
        return []

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
            bursts.append({'onset': on, 'offset': off, 'peak_idx': peak,
                          'peak_amp': intensity[peak],
                          'duration_s': (off - on) / FS})
    return bursts


def filter_bursts_by_condition(bursts, t, segments, cond_group):
    """Keep only bursts whose peak falls within the given condition group."""
    mask = get_condition_mask(t, segments, cond_group)
    return [b for b in bursts if mask[b['peak_idx']]]


def peri_burst_peak_latency(Y, bursts, window_s=10.0):
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

    if n_valid < 3:
        return np.full(N_GROUPS, np.nan), n_valid, None

    pba /= n_valid
    t_axis = np.arange(-window, window + 1) / FS
    latencies = np.array([t_axis[np.argmax(pba[gi])] for gi in range(N_GROUPS)])
    return latencies, n_valid, pba


def permutation_ci(Y, bursts, n_perm=200, window_s=10.0, seed=42):
    """Circular-shift null for peri-burst peak latencies."""
    rng = np.random.default_rng(seed)
    T = Y.shape[0]
    null_lats = np.full((n_perm, N_GROUPS), np.nan)

    for pi in range(n_perm):
        Y_shifted = Y.copy()
        for gname, ginfo in GROUPS.items():
            shift = rng.integers(int(30 * FS), T - int(30 * FS))
            Y_shifted[:, ginfo['idx']] = np.roll(Y[:, ginfo['idx']], shift, axis=0)
        lats, _, _ = peri_burst_peak_latency(Y_shifted, bursts, window_s=window_s)
        null_lats[pi] = lats

    return null_lats


def analyze_session_by_condition(session_id):
    """Analyze one session, stratified by condition group."""
    try:
        Y, viterbi, t, segments, asym = load_session(session_id)
    except Exception as e:
        return None

    intensities = compute_group_intensity(Y, smooth_s=3.0)

    # Determine which condition groups this session has
    available_conds = set()
    for seg in segments:
        for gname, cond_list in CONDITION_GROUPS.items():
            if seg[0] in cond_list:
                available_conds.add(gname)

    result = {'session': session_id, 'conditions': {}}

    for cond_group in available_conds:
        cond_result = {
            'lag_matrices': {},  # trigger_group -> latency array
            'burst_counts': {},
            'burst_asym': {},   # trigger_group -> mean asymmetry (3,)
            'burst_rate': {},   # trigger_group -> bursts/min
            'n_bursts_used': {},
        }

        # Duration of this condition in this session
        cond_mask = get_condition_mask(t, segments, cond_group)
        cond_duration_min = cond_mask.sum() / FS / 60.0
        cond_result['duration_min'] = cond_duration_min

        for gi, trigger in enumerate(GROUP_NAMES):
            # All bursts for this modality, then filter to condition
            all_bursts = detect_bursts(intensities[trigger])
            cond_bursts = filter_bursts_by_condition(all_bursts, t, segments, cond_group)

            cond_result['burst_counts'][trigger] = len(cond_bursts)
            cond_result['burst_rate'][trigger] = (len(cond_bursts) / cond_duration_min
                                                   if cond_duration_min > 0 else 0)

            if len(cond_bursts) >= 5:
                latencies, n_valid, _ = peri_burst_peak_latency(Y, cond_bursts, window_s=10.0)
                cond_result['lag_matrices'][trigger] = latencies
                cond_result['n_bursts_used'][trigger] = n_valid

                # Asymmetry during bursts
                burst_asym_vals = []
                for b in cond_bursts:
                    peak = b['peak_idx']
                    window = int(3.0 * FS)
                    s, e = max(0, peak - window), min(len(asym), peak + window + 1)
                    burst_asym_vals.append(asym[s:e].mean(axis=0))
                cond_result['burst_asym'][trigger] = np.mean(burst_asym_vals, axis=0)

        result['conditions'][cond_group] = cond_result

    return result


def main():
    out_dir = os.path.join(RESULTS_DIR, 'quiver_plots')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Condition-Stratified Burst Validation")
    print("=" * 70)

    # ── Analyze all sessions ──
    all_sessions = []
    for d in sorted(glob.glob(os.path.join(RESULTS_DIR, '*', ''))):
        sess = os.path.basename(os.path.normpath(d))
        scaffold = os.path.join(d, 'scaffold_v82_ztimecourses.npz')
        results_json = os.path.join(d, 'scaffold_v82_results.json')
        v8 = os.path.join(d, 'rslds_v8_full_results.npz')
        p2 = os.path.join(d, 'rslds_phase2_results.npz')
        if os.path.exists(scaffold) and os.path.exists(results_json) and \
           (os.path.exists(v8) or os.path.exists(p2)):
            all_sessions.append(sess)

    results = {}
    for sid in all_sessions:
        r = analyze_session_by_condition(sid)
        if r:
            conds = list(r['conditions'].keys())
            n_bursts = sum(sum(c['burst_counts'].values())
                          for c in r['conditions'].values())
            print(f"  {sid}: {conds}, {n_bursts} total bursts")
            results[sid] = r

    n_sess = len(results)
    print(f"\n  {n_sess} sessions analyzed")

    # ── Aggregate by condition group ──
    # For each condition group, collect lead/lag matrices and asymmetry across sessions
    cond_lags = {}       # cond -> trigger -> list of (N_GROUPS,) latency arrays
    cond_asym = {}       # cond -> trigger -> list of (3,) asymmetry arrays
    cond_rates = {}      # cond -> trigger -> list of burst rates
    cond_n_sessions = {} # cond -> count

    for cond_group in COND_GROUP_ORDER:
        cond_lags[cond_group] = {g: [] for g in GROUP_NAMES}
        cond_asym[cond_group] = {g: [] for g in GROUP_NAMES}
        cond_rates[cond_group] = {g: [] for g in GROUP_NAMES}
        cond_n_sessions[cond_group] = 0

        for sid, r in results.items():
            if cond_group not in r['conditions']:
                continue
            cond_data = r['conditions'][cond_group]
            cond_n_sessions[cond_group] += 1

            for trigger in GROUP_NAMES:
                if trigger in cond_data['lag_matrices']:
                    cond_lags[cond_group][trigger].append(cond_data['lag_matrices'][trigger])
                if trigger in cond_data['burst_asym']:
                    cond_asym[cond_group][trigger].append(cond_data['burst_asym'][trigger])
                cond_rates[cond_group][trigger].append(cond_data['burst_rate'].get(trigger, 0))

    # ── Print summary ──
    print(f"\n{'─' * 70}")
    print("Condition-stratified burst rates (mean bursts/min across sessions):")
    for cond in COND_GROUP_ORDER:
        n = cond_n_sessions[cond]
        if n == 0:
            continue
        print(f"\n  {cond} (n={n} sessions):")
        for trigger in GROUP_NAMES:
            rates = cond_rates[cond][trigger]
            if rates:
                print(f"    {trigger:>12s}: {np.mean(rates):5.1f} ± {np.std(rates)/np.sqrt(len(rates)):4.1f} bursts/min")

    print(f"\n{'─' * 70}")
    print("Therapist/patient asymmetry during EEG bursts (+ = therapist higher):")
    band_names = ['θ', 'α', 'β']
    for cond in COND_GROUP_ORDER:
        n = cond_n_sessions[cond]
        if n == 0:
            continue
        print(f"\n  {cond} (n={n}):")
        for trigger in ['EEG phase', 'EEG power']:
            vals = cond_asym[cond][trigger]
            if len(vals) < 2:
                print(f"    {trigger}: insufficient data ({len(vals)} sessions)")
                continue
            arr = np.array(vals)
            means = arr.mean(axis=0)
            ses = arr.std(axis=0) / np.sqrt(len(arr))
            parts = []
            for bi, bn in enumerate(band_names):
                sig = '*' if abs(means[bi]) > 2 * ses[bi] else ''
                parts.append(f"{bn}={means[bi]:+.3f}±{ses[bi]:.3f}{sig}")
            print(f"    {trigger:>12s}: {', '.join(parts)}")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Lead/lag matrices per condition
    # ════════════════════════════════════════════════════════════════════
    active_conds = [c for c in COND_GROUP_ORDER if cond_n_sessions.get(c, 0) >= 2]
    n_conds = len(active_conds)

    fig, axes = plt.subplots(2, n_conds, figsize=(6 * n_conds, 12))
    if n_conds == 1:
        axes = axes.reshape(-1, 1)

    for ci, cond in enumerate(active_conds):
        # Top row: mean lead/lag matrix
        ax = axes[0, ci]
        lag_mat = np.full((N_GROUPS, N_GROUPS), np.nan)
        n_mat = np.zeros((N_GROUPS, N_GROUPS), dtype=int)

        for gi, trigger in enumerate(GROUP_NAMES):
            vals = cond_lags[cond][trigger]
            if len(vals) >= 2:
                arr = np.array(vals)
                lag_mat[gi] = np.nanmean(arr, axis=0)
                n_mat[gi] = np.sum(~np.isnan(arr), axis=0)

        im = ax.imshow(lag_mat, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
        ax.set_xticks(range(N_GROUPS))
        ax.set_xticklabels(GROUP_NAMES, fontsize=7, rotation=30, ha='right')
        ax.set_yticks(range(N_GROUPS))
        ax.set_yticklabels(GROUP_NAMES, fontsize=7)
        n = cond_n_sessions[cond]
        ax.set_title(f'{cond}\n(n={n} sessions)', fontsize=11, fontweight='bold',
                    color=COND_GROUP_COLORS[cond])
        if ci == 0:
            ax.set_ylabel('Trigger → Response\nPeak Latency (s)', fontsize=10)

        for i in range(N_GROUPS):
            for j in range(N_GROUPS):
                v = lag_mat[i, j]
                if not np.isnan(v):
                    c = 'white' if abs(v) > 5 else 'black'
                    ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                           fontsize=7, fontweight='bold', color=c)

        # Bottom row: sign consistency
        ax = axes[1, ci]
        consistency = np.full((N_GROUPS, N_GROUPS), np.nan)
        for gi, trigger in enumerate(GROUP_NAMES):
            vals = cond_lags[cond][trigger]
            if len(vals) >= 3:
                arr = np.array(vals)
                for gj in range(N_GROUPS):
                    col = arr[:, gj]
                    col = col[~np.isnan(col)]
                    if len(col) >= 3:
                        consistency[gi, gj] = np.mean(
                            np.sign(col) == np.sign(np.mean(col)))

        im2 = ax.imshow(consistency, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
        ax.set_xticks(range(N_GROUPS))
        ax.set_xticklabels(GROUP_NAMES, fontsize=7, rotation=30, ha='right')
        ax.set_yticks(range(N_GROUPS))
        ax.set_yticklabels(GROUP_NAMES, fontsize=7)
        if ci == 0:
            ax.set_ylabel('Sign Consistency', fontsize=10)

        for i in range(N_GROUPS):
            for j in range(N_GROUPS):
                v = consistency[i, j]
                if not np.isnan(v):
                    c = 'white' if v > 0.7 else 'black'
                    ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                           fontsize=7, fontweight='bold', color=c)

    plt.suptitle('Cross-Modal Lead/Lag by Condition (top) + Sign Consistency (bottom)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_validation_by_condition_lag.png'),
               dpi=200, bbox_inches='tight')
    print(f"\n  Saved: burst_validation_by_condition_lag.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Asymmetry by segment — all 3 bands × 2 EEG types
    # ════════════════════════════════════════════════════════════════════
    band_names = ['θ', 'α', 'β']
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    for row, trigger in enumerate(['EEG phase', 'EEG power']):
        for col, (bi, bn) in enumerate(zip(range(3), band_names)):
            ax = axes[row, col]

            cond_means, cond_ses, cond_labels, cond_individual = [], [], [], []

            for cond in COND_GROUP_ORDER:
                vals = cond_asym[cond][trigger]
                if len(vals) < 2:
                    continue
                arr = np.array(vals)[:, bi]
                cond_means.append(arr.mean())
                cond_ses.append(arr.std() / np.sqrt(len(arr)))
                cond_labels.append(f'{cond}\nn={len(arr)}')
                cond_individual.append(arr)

            if not cond_labels:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            x = np.arange(len(cond_labels))
            # Color bars by their condition
            raw_conds = [c.split('\n')[0] for c in cond_labels]
            colors = [COND_GROUP_COLORS.get(c, '#999') for c in raw_conds]
            ax.bar(x, cond_means, yerr=[s * 1.96 for s in cond_ses],
                  capsize=4, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)

            for xi, arr in enumerate(cond_individual):
                jitter = np.random.uniform(-0.15, 0.15, len(arr))
                ax.scatter(x[xi] + jitter, arr, color='black', s=15,
                          alpha=0.45, zorder=5)

            ax.axhline(0, color='red', ls='--', lw=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=7, fontweight='bold')
            ax.set_ylabel('Asymmetry (+ = therapist)', fontsize=9)
            ax.set_title(f'{trigger} — {bn} band', fontsize=11, fontweight='bold',
                        color=GROUPS[trigger]['color'])
            ax.grid(True, alpha=0.15, axis='y')

    plt.suptitle('Therapist/Patient EEG Asymmetry During Coupling Bursts — by Segment\n'
                 '(Positive = therapist has higher power; error bars = 95% CI; dots = individual sessions)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_asymmetry_by_segment.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: burst_asymmetry_by_segment.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Burst rate by condition — the reliable metric
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, N_GROUPS, figsize=(4 * N_GROUPS, 5), sharey=True)

    for gi, trigger in enumerate(GROUP_NAMES):
        ax = axes[gi]
        means, ses, labels, individuals = [], [], [], []

        for cond in COND_GROUP_ORDER:
            rates = cond_rates[cond][trigger]
            if len(rates) < 2:
                continue
            arr = np.array(rates)
            means.append(arr.mean())
            ses.append(arr.std() / np.sqrt(len(arr)))
            labels.append(cond)
            individuals.append(arr)

        if not labels:
            continue

        x = np.arange(len(labels))
        colors = [COND_GROUP_COLORS[c] for c in labels]
        ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        for xi, arr in enumerate(individuals):
            jitter = np.random.uniform(-0.12, 0.12, len(arr))
            ax.scatter(x[xi] + jitter, arr, color='black', s=15, alpha=0.4, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
        ax.set_title(trigger, fontsize=11, fontweight='bold',
                    color=GROUPS[trigger]['color'])
        if gi == 0:
            ax.set_ylabel('Bursts / minute', fontsize=10)
        ax.grid(True, alpha=0.15, axis='y')

    plt.suptitle('Coupling Burst Rate by Segment — Cross-Session (95% CI)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_rate_by_segment.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: burst_rate_by_segment.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 4: Peri-burst averages per segment type (side-by-side comparison)
    # ════════════════════════════════════════════════════════════════════
    pba_conds = ['conv_1', 'conv_2', 'meditate_B', 'meditate_K', 'PE_1']
    pba_conds = [c for c in pba_conds if cond_n_sessions.get(c, 0) >= 2]
    print(f"\n── Peri-burst averages per segment: {pba_conds} ──")

    # Collect bursts by (condition, trigger) across sessions
    cond_burst_data = {}  # (cond, trigger) -> list of (burst, Y_sess)
    for cond in pba_conds:
        for trigger in GROUP_NAMES:
            cond_burst_data[(cond, trigger)] = []

    for sid, r in results.items():
        try:
            Y, _, t, segments, _ = load_session(sid)
        except:
            continue
        intensities = compute_group_intensity(Y, smooth_s=3.0)
        for cond in pba_conds:
            if cond not in r['conditions']:
                continue
            for trigger in GROUP_NAMES:
                all_bursts = detect_bursts(intensities[trigger])
                filt = filter_bursts_by_condition(all_bursts, t, segments, cond)
                for b in filt:
                    cond_burst_data[(cond, trigger)].append((b, Y))

    # Plot: rows = trigger groups, columns = conditions
    n_pba_conds = len(pba_conds)
    fig, axes = plt.subplots(N_GROUPS, n_pba_conds,
                             figsize=(5 * n_pba_conds, 2.8 * N_GROUPS),
                             sharex=True, sharey='row')

    window_s = 10.0
    window = int(window_s * FS)

    for ci, cond in enumerate(pba_conds):
        for gi, trigger in enumerate(GROUP_NAMES):
            ax = axes[gi, ci] if N_GROUPS > 1 and n_pba_conds > 1 else axes[gi]
            burst_data = cond_burst_data[(cond, trigger)]

            if len(burst_data) < 3:
                ax.text(0.5, 0.5, f'n={len(burst_data)}', transform=ax.transAxes,
                       ha='center', fontsize=8, alpha=0.5)
                if gi == 0:
                    ax.set_title(cond, fontsize=10, fontweight='bold',
                               color=COND_GROUP_COLORS.get(cond, '#666'))
                if ci == 0:
                    ax.set_ylabel(trigger, fontsize=8, fontweight='bold',
                                 color=GROUPS[trigger]['color'])
                continue

            pba = np.zeros((N_GROUPS, 2 * window + 1))
            n_valid = 0
            for burst, Y_sess in burst_data:
                peak = burst['peak_idx']
                s, e = peak - window, peak + window + 1
                if s < 0 or e > Y_sess.shape[0]:
                    continue
                for gj, gname in enumerate(GROUP_NAMES):
                    idx = GROUPS[gname]['idx']
                    pba[gj] += np.sqrt(np.mean(Y_sess[s:e, idx]**2, axis=1))
                n_valid += 1

            if n_valid < 2:
                ax.text(0.5, 0.5, f'n={n_valid}', transform=ax.transAxes,
                       ha='center', fontsize=8, alpha=0.5)
                continue

            pba /= n_valid
            t_axis = np.arange(-window, window + 1) / FS

            for gj, resp in enumerate(GROUP_NAMES):
                lw = 2.0 if resp == trigger else 0.8
                alpha = 1.0 if resp == trigger else 0.5
                ax.plot(t_axis, pba[gj], color=GROUPS[resp]['color'],
                       lw=lw, alpha=alpha)

            ax.axvline(0, color='black', ls='--', lw=0.8, alpha=0.4)
            ax.grid(True, alpha=0.1)

            if gi == 0:
                ax.set_title(f'{cond}\n(n={n_valid})', fontsize=10, fontweight='bold',
                           color=COND_GROUP_COLORS.get(cond, '#666'))
            if ci == 0:
                ax.set_ylabel(f'{trigger}', fontsize=8, fontweight='bold',
                             color=GROUPS[trigger]['color'])
            if gi == N_GROUPS - 1:
                ax.set_xlabel('s', fontsize=8)

    # Add legend to top-left
    for gj, resp in enumerate(GROUP_NAMES):
        axes[0, 0].plot([], [], color=GROUPS[resp]['color'], lw=2, label=resp)
    axes[0, 0].legend(fontsize=6, loc='upper left', ncol=2)

    plt.suptitle('Peri-Burst Averages by Segment (rows=trigger, cols=condition)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'peri_burst_by_segment.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: peri_burst_by_segment.png")
    plt.close(fig)

    # ── Save results JSON ──
    results_out = {
        'n_sessions': n_sess,
        'condition_groups': COND_GROUP_ORDER,
        'modality_groups': GROUP_NAMES,
    }
    for cond in COND_GROUP_ORDER:
        cond_out = {'n_sessions': cond_n_sessions.get(cond, 0)}
        for trigger in GROUP_NAMES:
            lags = cond_lags[cond][trigger]
            asym_vals = cond_asym[cond][trigger]
            rates = cond_rates[cond][trigger]
            cond_out[trigger] = {
                'n_lag_sessions': len(lags),
                'mean_lag': np.nanmean(lags, axis=0).tolist() if lags else [],
                'mean_asym': np.mean(asym_vals, axis=0).tolist() if asym_vals else [],
                'mean_rate': float(np.mean(rates)) if rates else 0,
            }
        results_out[cond] = cond_out

    with open(os.path.join(out_dir, 'burst_by_condition_results.json'), 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"  Saved: burst_by_condition_results.json")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
