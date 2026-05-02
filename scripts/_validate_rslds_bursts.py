"""Validate rSLDS coupling bursts: cross-session replication, pseudo-dyad null, permutation CIs.

Three validation layers:
  1. Cross-session replication: run burst analysis on all 12 sessions, aggregate lead/lag
  2. Pseudo-dyad null: pair P1/P2 from different sessions, verify null timing
  3. Permutation CIs: circular-shift response modality, 95% CI on peak latency
  4. Directionality: use asymmetry channels to decompose therapist vs patient contribution
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from joblib import Parallel, delayed

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


# ── Core functions (shared with burst analysis) ─────────────────────

def load_session(session_id):
    scaffold_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_ztimecourses.npz')
    results_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_results.json')
    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in MOD_KEYS if f'z_{k}' in data])
    t = data['t_common']

    # Asymmetry channels (positive = therapist higher)
    asym = np.column_stack([data[f'z_{k}'] for k in ASYM_KEYS if f'z_{k}' in data])

    v8_path = os.path.join(RESULTS_DIR, session_id, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(RESULTS_DIR, session_id, 'rslds_phase2_results.npz')
    if os.path.exists(v8_path):
        viterbi = np.load(v8_path)['full_viterbi']
    else:
        viterbi = np.load(p2_path)['viterbi_path']

    T = min(len(Y), len(viterbi), len(asym))
    Y, viterbi, t, asym = Y[:T], viterbi[:T], t[:T], asym[:T]

    with open(results_path) as f:
        meta = json.load(f)
    segments = meta.get('segments', [])

    return Y, viterbi, t, segments, asym


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


def peri_burst_peak_latency(Y, bursts, window_s=15.0):
    """Compute peak latency of each modality group's peri-burst average.

    Returns (N_GROUPS,) array of peak latencies in seconds relative to burst peak.
    """
    window = int(window_s * FS)
    T = Y.shape[0]

    # Raw group intensities (unsmoothed for timing precision)
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
        return np.full(N_GROUPS, np.nan), n_valid

    pba /= n_valid
    t_axis = np.arange(-window, window + 1) / FS

    latencies = np.zeros(N_GROUPS)
    for gi in range(N_GROUPS):
        latencies[gi] = t_axis[np.argmax(pba[gi])]

    return latencies, n_valid


def compute_burst_asymmetry(asym, bursts, window_s=5.0):
    """Mean asymmetry (therapist - patient) during each burst.

    Returns (n_bursts, 3) array: positive = therapist-dominated burst.
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


# ── Session-level analysis ───────────────────────────────────────────

def analyze_session(session_id):
    """Run full burst analysis on one session. Returns result dict."""
    try:
        Y, viterbi, t, segments, asym = load_session(session_id)
    except Exception as e:
        print(f"  {session_id}: SKIP ({e})")
        return None

    T = Y.shape[0]
    intensities = compute_group_intensity(Y, smooth_s=3.0)

    result = {'session': session_id, 'T': T, 'duration_min': T / FS / 60}

    # Detect bursts and compute lead/lag per trigger group
    lag_matrix = np.full((N_GROUPS, N_GROUPS), np.nan)
    burst_counts = {}
    burst_asym_by_group = {}

    for gi, trigger in enumerate(GROUP_NAMES):
        bursts = detect_bursts(intensities[trigger])
        burst_counts[trigger] = len(bursts)

        if len(bursts) >= 5:
            latencies, n_valid = peri_burst_peak_latency(Y, bursts, window_s=15.0)
            lag_matrix[gi] = latencies

        # Asymmetry during EEG-related bursts
        if trigger in ('EEG phase', 'EEG power') and len(bursts) >= 3:
            ba = compute_burst_asymmetry(asym, bursts)
            burst_asym_by_group[trigger] = ba.mean(axis=0) if len(ba) > 0 else np.zeros(3)

    result['lag_matrix'] = lag_matrix
    result['burst_counts'] = burst_counts
    result['burst_asym'] = burst_asym_by_group
    result['intensities'] = intensities
    result['Y'] = Y
    result['asym'] = asym

    return result


# ── Pseudo-dyad null ─────────────────────────────────────────────────

def pseudo_dyad_analysis(sessions_data, n_pairs=30, seed=42):
    """Create pseudo-dyads by pairing observations from different sessions.

    For each pair: take Y[:, :6] (EEG) from session A and Y[:, 6:] (non-EEG)
    from session B. This breaks real coupling while preserving marginal stats.
    """
    rng = np.random.default_rng(seed)
    valid = [(sid, d) for sid, d in sessions_data.items() if d is not None]
    n_sess = len(valid)

    null_lags = []
    for pair_i in range(n_pairs):
        # Pick two different sessions
        i, j = rng.choice(n_sess, size=2, replace=False)
        sid_a, data_a = valid[i]
        sid_b, data_b = valid[j]

        Y_a, Y_b = data_a['Y'], data_b['Y']
        T_min = min(len(Y_a), len(Y_b))

        # Construct pseudo-dyad: EEG from A, non-EEG from B
        Y_pseudo = np.zeros((T_min, Y_a.shape[1]))
        Y_pseudo[:, :6] = Y_a[:T_min, :6]   # EEG channels from session A
        Y_pseudo[:, 6:] = Y_b[:T_min, 6:]   # Non-EEG from session B

        intensities = compute_group_intensity(Y_pseudo, smooth_s=3.0)

        lag_row = np.full((N_GROUPS, N_GROUPS), np.nan)
        for gi, trigger in enumerate(GROUP_NAMES):
            bursts = detect_bursts(intensities[trigger])
            if len(bursts) >= 5:
                latencies, _ = peri_burst_peak_latency(Y_pseudo, bursts, window_s=15.0)
                lag_row[gi] = latencies

        null_lags.append(lag_row)

    return np.array(null_lags)  # (n_pairs, N_GROUPS, N_GROUPS)


# ── Permutation CI ───────────────────────────────────────────────────

def permutation_ci_single(Y, trigger_bursts, n_perm=200, window_s=15.0, seed=42):
    """Circular-shift response modalities to build null distribution for peak latency.

    Returns (n_perm, N_GROUPS) null latencies.
    """
    rng = np.random.default_rng(seed)
    T = Y.shape[0]
    null_latencies = np.full((n_perm, N_GROUPS), np.nan)

    for pi in range(n_perm):
        Y_shifted = Y.copy()
        # Circularly shift each group independently
        for gname, ginfo in GROUPS.items():
            shift = rng.integers(int(30 * FS), T - int(30 * FS))
            Y_shifted[:, ginfo['idx']] = np.roll(Y[:, ginfo['idx']], shift, axis=0)

        lats, _ = peri_burst_peak_latency(Y_shifted, trigger_bursts, window_s=window_s)
        null_latencies[pi] = lats

    return null_latencies


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import glob
    out_dir = os.path.join(RESULTS_DIR, 'quiver_plots')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Burst Validation: Cross-Session + Pseudo-Dyad + Permutation CI")
    print("=" * 70)

    # ── 1. Cross-session analysis ──
    print("\n── Phase 1: Cross-session burst analysis ──")
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

    print(f"  Found {len(all_sessions)} sessions")

    sessions_data = {}
    for sid in all_sessions:
        print(f"  Processing {sid}...", end=' ')
        result = analyze_session(sid)
        if result:
            n_bursts = sum(result['burst_counts'].values())
            print(f"{n_bursts} total bursts, {result['duration_min']:.1f} min")
            sessions_data[sid] = result
        else:
            print("FAILED")

    # Aggregate lag matrices
    all_lags = np.array([d['lag_matrix'] for d in sessions_data.values()])  # (N_sess, G, G)
    n_sess = len(all_lags)

    mean_lag = np.nanmean(all_lags, axis=0)
    std_lag = np.nanstd(all_lags, axis=0)
    # Count how many sessions have valid data per cell
    n_valid = np.sum(~np.isnan(all_lags), axis=0)
    # Standard error
    se_lag = std_lag / np.sqrt(np.maximum(n_valid, 1))

    print(f"\n  Cross-session lead/lag (mean ± SE, n sessions with data):")
    print(f"  {'Trigger':>12s} → {'Response':>12s}  {'Lag':>8s}  {'SE':>6s}  {'n':>3s}  {'Consistent':>10s}")
    for gi, g1 in enumerate(GROUP_NAMES):
        for gj, g2 in enumerate(GROUP_NAMES):
            if gi == gj:
                continue
            m, s, n = mean_lag[gi, gj], se_lag[gi, gj], n_valid[gi, gj]
            if n >= 3:
                # Check sign consistency: fraction of sessions with same sign as mean
                signs = np.sign(all_lags[:, gi, gj])
                signs = signs[~np.isnan(signs)]
                sign_frac = np.mean(signs == np.sign(m)) if len(signs) > 0 else 0
                consistent = f"{sign_frac:.0%}"
                print(f"  {g1:>12s} → {g2:>12s}  {m:+7.1f}s  {s:5.1f}s  {int(n):3d}  {consistent:>10s}")

    # ── 2. Pseudo-dyad null ──
    print(f"\n── Phase 2: Pseudo-dyad null (30 pairs) ──")
    null_lags = pseudo_dyad_analysis(sessions_data, n_pairs=30, seed=42)
    null_mean = np.nanmean(null_lags, axis=0)
    null_std = np.nanstd(null_lags, axis=0)

    # ── 3. Permutation CI on y_06 ──
    print(f"\n── Phase 3: Permutation CI (y_06, 200 permutations) ──")
    y06 = sessions_data.get('y_06')
    perm_results = {}
    if y06:
        Y_y06 = y06['Y']
        intensities_y06 = y06['intensities']

        for gi, trigger in enumerate(GROUP_NAMES):
            bursts = detect_bursts(intensities_y06[trigger])
            if len(bursts) < 5:
                continue
            print(f"  Permuting {trigger} ({len(bursts)} bursts)...")
            null_lats = permutation_ci_single(Y_y06, bursts, n_perm=200, seed=42 + gi)
            real_lats, _ = peri_burst_peak_latency(Y_y06, bursts)

            perm_results[trigger] = {
                'real': real_lats,
                'null_025': np.nanpercentile(null_lats, 2.5, axis=0),
                'null_975': np.nanpercentile(null_lats, 97.5, axis=0),
                'null_mean': np.nanmean(null_lats, axis=0),
                'p_values': np.zeros(N_GROUPS),
            }
            # Two-sided p-value: fraction of null more extreme than real
            for gj in range(N_GROUPS):
                null_col = null_lats[:, gj]
                null_col = null_col[~np.isnan(null_col)]
                if len(null_col) > 0:
                    p = np.mean(np.abs(null_col) >= np.abs(real_lats[gj]))
                    perm_results[trigger]['p_values'][gj] = p

    # ── 4. Directionality: asymmetry during bursts ──
    print(f"\n── Phase 4: Therapist/Patient asymmetry during bursts ──")
    asym_summary = {'EEG phase': [], 'EEG power': []}
    for sid, data in sessions_data.items():
        for gname in ('EEG phase', 'EEG power'):
            if gname in data['burst_asym']:
                asym_summary[gname].append(data['burst_asym'][gname])

    band_names = ['θ', 'α', 'β']
    for gname in ('EEG phase', 'EEG power'):
        vals = asym_summary[gname]
        if len(vals) >= 3:
            arr = np.array(vals)  # (n_sessions, 3)
            mean_a = arr.mean(axis=0)
            se_a = arr.std(axis=0) / np.sqrt(len(arr))
            print(f"\n  {gname} bursts — mean asymmetry (+ = therapist higher):")
            for bi, bn in enumerate(band_names):
                sig = '*' if abs(mean_a[bi]) > 2 * se_a[bi] else ''
                print(f"    {bn}: {mean_a[bi]:+.3f} ± {se_a[bi]:.3f} {sig}")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Cross-session lead/lag with pseudo-dyad null
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A: Real cross-session mean lag
    ax = axes[0]
    im = ax.imshow(mean_lag, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
    ax.set_xticks(range(N_GROUPS))
    ax.set_xticklabels(GROUP_NAMES, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(N_GROUPS))
    ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_title(f'Real Dyads: Mean Lead/Lag\n(n={n_sess} sessions)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Response', fontsize=10)
    ax.set_ylabel('Trigger', fontsize=10)
    plt.colorbar(im, ax=ax, label='seconds', shrink=0.8)
    for i in range(N_GROUPS):
        for j in range(N_GROUPS):
            v = mean_lag[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > 5 else 'black'
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=c)

    # Panel B: Pseudo-dyad null mean lag
    ax = axes[1]
    im = ax.imshow(null_mean, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
    ax.set_xticks(range(N_GROUPS))
    ax.set_xticklabels(GROUP_NAMES, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(N_GROUPS))
    ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_title('Pseudo-Dyad Null: Mean Lead/Lag\n(n=30 random pairs)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Response', fontsize=10)
    plt.colorbar(im, ax=ax, label='seconds', shrink=0.8)
    for i in range(N_GROUPS):
        for j in range(N_GROUPS):
            v = null_mean[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > 5 else 'black'
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=c)

    # Panel C: Difference (real - null) with significance
    ax = axes[2]
    diff = mean_lag - null_mean
    # Significance: real outside null ± 2*null_std
    sig_mask = np.abs(diff) > 2 * null_std

    im = ax.imshow(diff, cmap='RdBu_r', vmin=-8, vmax=8, aspect='auto')
    ax.set_xticks(range(N_GROUPS))
    ax.set_xticklabels(GROUP_NAMES, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(N_GROUPS))
    ax.set_yticklabels(GROUP_NAMES, fontsize=8)
    ax.set_title('Real − Null (★ = |diff| > 2σ)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Response', fontsize=10)
    plt.colorbar(im, ax=ax, label='seconds', shrink=0.8)
    for i in range(N_GROUPS):
        for j in range(N_GROUPS):
            v = diff[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > 4 else 'black'
                star = ' ★' if sig_mask[i, j] else ''
                ax.text(j, i, f'{v:+.1f}{star}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=c)

    plt.suptitle('Cross-Modal Lead/Lag: Real vs Pseudo-Dyad Null',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_validation_lag_matrices.png'),
               dpi=200, bbox_inches='tight')
    print(f"\n  Saved: burst_validation_lag_matrices.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Permutation CI for y_06
    # ════════════════════════════════════════════════════════════════════
    if perm_results:
        fig, axes = plt.subplots(N_GROUPS, 1, figsize=(12, 3 * N_GROUPS), sharex=True)

        for gi, trigger in enumerate(GROUP_NAMES):
            ax = axes[gi]
            if trigger not in perm_results:
                ax.text(0.5, 0.5, f'Insufficient {trigger} bursts',
                       transform=ax.transAxes, ha='center')
                ax.set_ylabel(trigger, fontsize=9, fontweight='bold',
                             color=GROUPS[trigger]['color'])
                continue

            pr = perm_results[trigger]
            x = np.arange(N_GROUPS)

            # Null CI band
            ax.fill_between(x, pr['null_025'], pr['null_975'],
                           alpha=0.3, color='gray', label='Null 95% CI')
            ax.scatter(x, pr['null_mean'], color='gray', s=30, zorder=3,
                      label='Null mean')

            # Real values
            for gj in range(N_GROUPS):
                p = pr['p_values'][gj]
                color = GROUPS[GROUP_NAMES[gj]]['color']
                marker = '*' if p < 0.05 else 'o'
                size = 120 if p < 0.05 else 60
                ax.scatter(gj, pr['real'][gj], color=color, s=size, marker=marker,
                          zorder=5, edgecolors='black', linewidth=1)
                label = f"{pr['real'][gj]:+.1f}s"
                if p < 0.05:
                    label += f"\np={p:.3f}*"
                ax.annotate(label, (gj, pr['real'][gj]),
                           xytext=(8, 5), textcoords='offset points',
                           fontsize=7, fontweight='bold', color=color)

            ax.axhline(0, color='black', ls='--', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(GROUP_NAMES, fontsize=8)
            ax.set_ylabel(f'Triggered on\n{trigger}', fontsize=9, fontweight='bold',
                         color=GROUPS[trigger]['color'])
            ax.set_ylim(-16, 16)
            ax.grid(True, alpha=0.15, axis='y')
            if gi == 0:
                ax.legend(fontsize=7, loc='upper right')

        axes[-1].set_xlabel('Response modality peak latency (seconds)', fontsize=11)
        plt.suptitle('Permutation Test: Real Peak Latency vs Null 95% CI (y_06)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'burst_validation_permutation_ci.png'),
                   dpi=200, bbox_inches='tight')
        print(f"  Saved: burst_validation_permutation_ci.png")
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Cross-session lead/lag distributions + asymmetry
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: distributions of specific lead/lag pairs across sessions
    key_pairs = [
        (2, 0, 'Face → EEG phase'),
        (0, 1, 'EEG phase → EEG power'),
        (4, 2, 'Body → Face'),
    ]
    for pi, (gi, gj, label) in enumerate(key_pairs):
        ax = axes[0, pi]
        real_vals = all_lags[:, gi, gj]
        real_vals = real_vals[~np.isnan(real_vals)]
        null_vals = null_lags[:, gi, gj]
        null_vals = null_vals[~np.isnan(null_vals)]

        if len(null_vals) > 0:
            ax.hist(null_vals, bins=15, alpha=0.4, color='gray', label='Pseudo-dyad null',
                   density=True)
        if len(real_vals) > 0:
            ax.hist(real_vals, bins=min(12, len(real_vals)), alpha=0.6,
                   color=GROUPS[GROUP_NAMES[gi]]['color'], label='Real dyads', density=True)
            # Individual session markers
            for v in real_vals:
                ax.axvline(v, color=GROUPS[GROUP_NAMES[gi]]['color'], lw=0.5, alpha=0.3)

        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('Peak latency (seconds)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    # Bottom row: therapist/patient asymmetry during EEG bursts
    for pi, gname in enumerate(['EEG phase', 'EEG power']):
        ax = axes[1, pi]
        vals = asym_summary[gname]
        if len(vals) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue
        arr = np.array(vals)
        x = np.arange(3)
        means = arr.mean(axis=0)
        ses = arr.std(axis=0) / np.sqrt(len(arr))

        colors = ['#1565C0', '#2196F3', '#64B5F6']
        bars = ax.bar(x, means, yerr=ses * 1.96, capsize=5,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Individual session dots
        for si in range(len(arr)):
            ax.scatter(x + np.random.uniform(-0.1, 0.1, 3), arr[si],
                      color='black', s=15, alpha=0.4, zorder=5)

        ax.axhline(0, color='red', ls='--', lw=1, label='Symmetric')
        ax.set_xticks(x)
        ax.set_xticklabels(['θ', 'α', 'β'], fontsize=11)
        ax.set_ylabel('Asymmetry (+ = therapist higher)', fontsize=10)
        ax.set_title(f'{gname} Burst Asymmetry', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15, axis='y')

    # Bottom right: sign consistency heatmap
    ax = axes[1, 2]
    consistency = np.zeros((N_GROUPS, N_GROUPS))
    for gi in range(N_GROUPS):
        for gj in range(N_GROUPS):
            vals = all_lags[:, gi, gj]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 3:
                consistency[gi, gj] = np.mean(np.sign(vals) == np.sign(np.mean(vals)))
            else:
                consistency[gi, gj] = np.nan

    im = ax.imshow(consistency, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(N_GROUPS))
    ax.set_xticklabels(GROUP_NAMES, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(N_GROUPS))
    ax.set_yticklabels(GROUP_NAMES, fontsize=7)
    ax.set_title('Sign Consistency Across Sessions\n(green = same direction in >70% sessions)',
                fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Fraction same sign', shrink=0.8)
    for i in range(N_GROUPS):
        for j in range(N_GROUPS):
            v = consistency[i, j]
            if not np.isnan(v):
                c = 'white' if v > 0.7 else 'black'
                ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=c)

    plt.suptitle('Burst Validation: Cross-Session Distributions + Directionality',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_validation_distributions.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: burst_validation_distributions.png")
    plt.close(fig)

    # ── Save numeric results ──
    results_out = {
        'n_sessions': n_sess,
        'mean_lag': mean_lag.tolist(),
        'std_lag': std_lag.tolist(),
        'n_valid': n_valid.tolist(),
        'null_mean_lag': null_mean.tolist(),
        'null_std_lag': null_std.tolist(),
        'group_names': GROUP_NAMES,
        'asymmetry_eeg_phase': [v.tolist() for v in asym_summary.get('EEG phase', [])],
        'asymmetry_eeg_power': [v.tolist() for v in asym_summary.get('EEG power', [])],
    }
    if perm_results:
        results_out['permutation_y06'] = {
            trigger: {
                'real_latencies': pr['real'].tolist(),
                'null_025': pr['null_025'].tolist(),
                'null_975': pr['null_975'].tolist(),
                'p_values': pr['p_values'].tolist(),
            } for trigger, pr in perm_results.items()
        }

    out_path = os.path.join(out_dir, 'burst_validation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"  Saved: burst_validation_results.json")

    print(f"\n{'=' * 70}")
    print("Validation complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
