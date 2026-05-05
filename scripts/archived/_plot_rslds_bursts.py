"""rSLDS Coupling Burst Analysis: continuous intensity, burst detection, cross-modal timing.

Goes beyond discrete state assignments to capture:
  1. Continuous coupling intensity per modality group
  2. Discrete burst events (onset, peak, offset, amplitude, dominant modality)
  3. Cross-modal burst timing — peri-burst averages reveal lead/lag structure
  4. Per-condition burst statistics (rate, amplitude, cross-modal composition)
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'rslds')

MOD_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MOD_NAMES = [
    'ImCoh θ', 'ImCoh α', 'ImCoh β',
    'Conc θ', 'Conc α', 'Conc β',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]

# Modality groups — composites that map to interpretable coupling systems
GROUPS = {
    'EEG phase':  {'idx': [0, 1, 2], 'color': '#1565C0', 'label': 'Phase Coupling (ImCoh)'},
    'EEG power':  {'idx': [3, 4, 5], 'color': '#E65100', 'label': 'Shared Power (Conc)'},
    'Face':       {'idx': [6, 7],    'color': '#E91E63', 'label': 'Facial Coupling (BL)'},
    'Autonomic':  {'idx': [8, 9, 10],'color': '#4CAF50', 'label': 'Autonomic (ECG+Resp)'},
    'Body':       {'idx': [11],      'color': '#795548', 'label': 'Body (Pose)'},
}

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
STATE_LABELS = ['NULL', 'COUP', 'OTHER', 'SHARED']
CONDITION_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#9FA8DA', 'baseline': '#90CAF9',
    'conv_1': '#FFE0B2', 'conv_2': '#FFCC80',
    'PE': '#F8BBD0', 'PE_1': '#F8BBD0', 'PE_2': '#F8BBD0',
    'meditate_B': '#CE93D8', 'meditate_K': '#A5D6A7',
    'gap': '#E0E0E0', 'gap_pre': '#E0E0E0', 'gap_post': '#E0E0E0',
}

FS = 2.0  # output sample rate in Hz


# ── Data loading ─────────────────────────────────────────────────────

def load_session(session_id):
    """Load observations, Viterbi, timestamps, and condition segments."""
    scaffold_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_ztimecourses.npz')
    results_path = os.path.join(RESULTS_DIR, session_id, 'scaffold_v82_results.json')
    data = np.load(scaffold_path)
    Y = np.column_stack([data[f'z_{k}'] for k in MOD_KEYS if f'z_{k}' in data])
    t = data['t_common']

    v8_path = os.path.join(RESULTS_DIR, session_id, 'rslds_v8_full_results.npz')
    p2_path = os.path.join(RESULTS_DIR, session_id, 'rslds_phase2_results.npz')
    if os.path.exists(v8_path):
        viterbi = np.load(v8_path)['full_viterbi']
    else:
        viterbi = np.load(p2_path)['viterbi_path']

    T = min(len(Y), len(viterbi))
    Y, viterbi, t = Y[:T], viterbi[:T], t[:T]

    with open(results_path) as f:
        meta = json.load(f)
    segments = meta.get('segments', [])

    # d_emit for this session
    with open(os.path.join(RESULTS_DIR, 'v82_rslds_full_results.json')) as f:
        all_data = json.load(f)
    sess_data = [d for d in all_data if d['session'] == session_id][0]
    d_emit = np.array(sess_data['d_emit'])

    return Y, viterbi, t, segments, d_emit


def get_condition_at_time(lsl_time, segments):
    for seg in segments:
        if seg[1] <= lsl_time <= seg[2]:
            return seg[0]
    return 'gap'


# ── Coupling intensity timecourses ───────────────────────────────────

def compute_group_intensity(Y, smooth_s=3.0):
    """Compute per-group coupling intensity (smoothed RMS of group channels).

    Returns dict of {group_name: (T,) intensity timecourse}.
    Smoothing uses a causal moving average to avoid look-ahead.
    """
    T = Y.shape[0]
    smooth_n = max(1, int(smooth_s * FS))
    intensities = {}

    for gname, ginfo in GROUPS.items():
        idx = ginfo['idx']
        # RMS across channels in the group — captures overall activation
        rms = np.sqrt(np.mean(Y[:, idx]**2, axis=1))
        # Smooth for visual clarity
        smoothed = uniform_filter1d(rms, size=smooth_n, mode='nearest')
        intensities[gname] = smoothed

    return intensities


def compute_residual_intensity(Y, viterbi, d_emit, smooth_s=3.0):
    """Within-state residual intensity: how far observation is from state center.

    r_t = y_t - d_emit[z_t]
    Returns per-group residual RMS — captures amplitude beyond model expectation.
    """
    T = Y.shape[0]
    smooth_n = max(1, int(smooth_s * FS))

    # Residuals from current state expectation
    expected = d_emit[viterbi]  # (T, D)
    residuals = Y - expected

    resid_intensities = {}
    for gname, ginfo in GROUPS.items():
        idx = ginfo['idx']
        rms = np.sqrt(np.mean(residuals[:, idx]**2, axis=1))
        smoothed = uniform_filter1d(rms, size=smooth_n, mode='nearest')
        resid_intensities[gname] = smoothed

    return resid_intensities, residuals


# ── Burst detection ──────────────────────────────────────────────────

def detect_bursts(intensity, threshold_pctl=90, min_duration_s=2.0,
                  merge_gap_s=3.0):
    """Detect coupling bursts as contiguous above-threshold periods.

    Returns list of dicts: {onset, offset, peak_idx, peak_amp, duration_s, mean_amp}
    """
    threshold = np.percentile(intensity, threshold_pctl)
    above = intensity > threshold
    min_dur = int(min_duration_s * FS)
    merge_gap = int(merge_gap_s * FS)

    # Find contiguous above-threshold segments
    changes = np.diff(above.astype(int))
    onsets = np.where(changes == 1)[0] + 1
    offsets = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if above[0]:
        onsets = np.concatenate([[0], onsets])
    if above[-1]:
        offsets = np.concatenate([offsets, [len(intensity)]])

    if len(onsets) == 0:
        return [], threshold

    # Merge close bursts
    merged_onsets = [onsets[0]]
    merged_offsets = [offsets[0]]
    for i in range(1, len(onsets)):
        if onsets[i] - merged_offsets[-1] <= merge_gap:
            merged_offsets[-1] = offsets[i]  # extend previous burst
        else:
            merged_onsets.append(onsets[i])
            merged_offsets.append(offsets[i])

    # Filter by minimum duration and build burst descriptors
    bursts = []
    for on, off in zip(merged_onsets, merged_offsets):
        dur = off - on
        if dur >= min_dur:
            peak_idx = on + np.argmax(intensity[on:off])
            bursts.append({
                'onset': on,
                'offset': off,
                'peak_idx': peak_idx,
                'peak_amp': intensity[peak_idx],
                'duration_s': dur / FS,
                'mean_amp': intensity[on:off].mean(),
            })

    return bursts, threshold


# ── Peri-burst averaging ─────────────────────────────────────────────

def peri_burst_average(Y, bursts, window_s=10.0, group_key=None):
    """Average all modality group intensities around burst peaks.

    Returns (n_groups, 2*window_samples+1) peri-burst average matrix.
    """
    window = int(window_s * FS)
    T = Y.shape[0]
    intensities_raw = {}
    for gname, ginfo in GROUPS.items():
        idx = ginfo['idx']
        intensities_raw[gname] = np.sqrt(np.mean(Y[:, idx]**2, axis=1))

    group_names = list(GROUPS.keys())
    n_groups = len(group_names)
    pba = np.zeros((n_groups, 2 * window + 1))
    n_valid = 0

    for burst in bursts:
        peak = burst['peak_idx']
        start = peak - window
        end = peak + window + 1
        if start < 0 or end > T:
            continue
        for gi, gname in enumerate(group_names):
            pba[gi] += intensities_raw[gname][start:end]
        n_valid += 1

    if n_valid > 0:
        pba /= n_valid

    t_axis = np.arange(-window, window + 1) / FS  # seconds relative to peak
    return pba, t_axis, group_names, n_valid


# ── Main visualization ───────────────────────────────────────────────

def main():
    session_id = 'y_06'
    out_dir = os.path.join(RESULTS_DIR, 'quiver_plots')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"Coupling Burst Analysis — {session_id}")
    print("=" * 70)

    Y, viterbi, t, segments, d_emit = load_session(session_id)
    T, D = Y.shape
    t_rel = (t - t[0]) / 60.0  # minutes

    # ── Compute coupling intensities ──
    intensities = compute_group_intensity(Y, smooth_s=3.0)
    resid_int, residuals = compute_residual_intensity(Y, viterbi, d_emit, smooth_s=3.0)

    # ── Detect bursts per group ──
    all_bursts = {}
    for gname in GROUPS:
        bursts, thresh = detect_bursts(intensities[gname], threshold_pctl=90,
                                       min_duration_s=2.0, merge_gap_s=3.0)
        all_bursts[gname] = {'bursts': bursts, 'threshold': thresh}
        print(f"  {gname:>12s}: {len(bursts):3d} bursts "
              f"(thresh={thresh:.2f}, mean dur={np.mean([b['duration_s'] for b in bursts]):.1f}s)"
              if bursts else f"  {gname:>12s}:   0 bursts")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Full-session coupling intensity timeline
    # ════════════════════════════════════════════════════════════════════
    group_names = list(GROUPS.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(n_groups + 1, 1, figsize=(24, 3.0 * (n_groups + 1)),
                             sharex=True, gridspec_kw={'height_ratios':
                             [1] * n_groups + [0.6]})

    for gi, gname in enumerate(group_names):
        ax = axes[gi]
        ginfo = GROUPS[gname]
        intensity = intensities[gname]
        resid = resid_int[gname]
        bdata = all_bursts[gname]

        # Condition background
        for seg in segments:
            t_s = (seg[1] - t[0]) / 60.0
            t_e = (seg[2] - t[0]) / 60.0
            color = CONDITION_COLORS.get(seg[0], '#F5F5F5')
            ax.axvspan(t_s, t_e, alpha=0.2, color=color, zorder=0)

        # Raw intensity (light) and residual intensity (dark)
        ax.fill_between(t_rel, 0, intensity, alpha=0.25, color=ginfo['color'],
                       label='Total intensity')
        ax.plot(t_rel, intensity, color=ginfo['color'], lw=0.8, alpha=0.6)
        ax.plot(t_rel, resid, color=ginfo['color'], lw=1.2, alpha=0.9,
               label='Residual (above state mean)')

        # Threshold line
        ax.axhline(bdata['threshold'], color='red', ls=':', lw=0.8, alpha=0.4)

        # Mark burst peaks
        for burst in bdata['bursts']:
            peak = burst['peak_idx']
            ax.scatter(t_rel[peak], burst['peak_amp'],
                      color=ginfo['color'], s=30, zorder=5, edgecolors='black',
                      linewidth=0.5)
            # Burst extent
            on = burst['onset']
            off = min(burst['offset'], T - 1)
            ax.axvspan(t_rel[on], t_rel[off], alpha=0.15,
                      color=ginfo['color'], zorder=1)

        ax.set_ylabel(ginfo['label'], fontsize=9, fontweight='bold',
                     color=ginfo['color'])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.1)
        if gi == 0:
            ax.legend(fontsize=7, loc='upper right')
            # Condition labels at top
            for seg in segments:
                t_mid = ((seg[1] + seg[2]) / 2 - t[0]) / 60.0
                ax.text(t_mid, ax.get_ylim()[1] * 0.95, seg[0].replace('_', '\n'),
                       ha='center', va='top', fontsize=6.5, fontweight='bold',
                       alpha=0.6)

    # Bottom panel: Viterbi state timeline
    ax_state = axes[-1]
    for seg in segments:
        t_s = (seg[1] - t[0]) / 60.0
        t_e = (seg[2] - t[0]) / 60.0
        ax_state.axvspan(t_s, t_e, alpha=0.2,
                        color=CONDITION_COLORS.get(seg[0], '#F5F5F5'))

    # Color-coded state blocks
    state_changes = np.where(np.diff(viterbi) != 0)[0]
    boundaries = np.concatenate([[0], state_changes + 1, [T]])
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        state = viterbi[s]
        ax_state.axvspan(t_rel[s], t_rel[min(e, T - 1)],
                        color=STATE_COLORS[state], alpha=0.6)

    for k in range(4):
        ax_state.scatter([], [], c=STATE_COLORS[k], s=40, label=STATE_LABELS[k])
    ax_state.legend(fontsize=7, loc='upper right', ncol=4)
    ax_state.set_ylabel('State', fontsize=9, fontweight='bold')
    ax_state.set_yticks([])
    ax_state.set_xlabel('Time (minutes)', fontsize=11)

    plt.suptitle(f'Coupling Intensity Timeline — {session_id}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'burst_timeline_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"\n  Saved: burst_timeline_{session_id}.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Peri-burst averages — cross-modal timing
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3.0 * n_groups), sharex=True)

    for gi, trigger_group in enumerate(group_names):
        ax = axes[gi]
        bursts = all_bursts[trigger_group]['bursts']
        if len(bursts) < 3:
            ax.text(0.5, 0.5, f'Too few {trigger_group} bursts ({len(bursts)})',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_ylabel(f'Triggered on\n{trigger_group}', fontsize=9,
                         fontweight='bold')
            continue

        pba, t_axis, gn, n_valid = peri_burst_average(Y, bursts, window_s=15.0)

        for gj, resp_group in enumerate(gn):
            ginfo = GROUPS[resp_group]
            lw = 2.5 if resp_group == trigger_group else 1.2
            alpha = 1.0 if resp_group == trigger_group else 0.7
            ax.plot(t_axis, pba[gj], color=ginfo['color'], lw=lw, alpha=alpha,
                   label=resp_group)

        ax.axvline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.set_ylabel(f'Triggered on\n{trigger_group}\n(n={n_valid})',
                     fontsize=9, fontweight='bold',
                     color=GROUPS[trigger_group]['color'])
        ax.grid(True, alpha=0.15)
        if gi == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=n_groups)

    axes[-1].set_xlabel('Time relative to burst peak (seconds)', fontsize=11)
    plt.suptitle(f'Peri-Burst Averages — Cross-Modal Timing ({session_id})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'peri_burst_averages_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: peri_burst_averages_{session_id}.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Per-condition burst statistics
    # ════════════════════════════════════════════════════════════════════
    # Compute per-condition statistics
    cond_names_ordered = [seg[0] for seg in segments]
    # Remove duplicate names (keep order)
    seen = set()
    cond_unique = []
    for c in cond_names_ordered:
        if c not in seen:
            cond_unique.append(c)
            seen.add(c)

    cond_durations = {}
    for seg in segments:
        name = seg[0]
        dur = (seg[2] - seg[1]) / 60.0
        cond_durations[name] = cond_durations.get(name, 0) + dur

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel A: Burst rate per condition per group
    ax = axes[0]
    n_conds = len(cond_unique)
    bar_width = 0.8 / n_groups
    for gi, gname in enumerate(group_names):
        rates = []
        for cond in cond_unique:
            dur = cond_durations.get(cond, 0)
            count = sum(1 for b in all_bursts[gname]['bursts']
                       if get_condition_at_time(t[b['peak_idx']], segments) == cond)
            rates.append(count / dur if dur > 0 else 0)
        x = np.arange(n_conds)
        ax.bar(x + gi * bar_width, rates, bar_width * 0.9,
              color=GROUPS[gname]['color'], alpha=0.75, label=gname)

    ax.set_xticks(np.arange(n_conds) + 0.3)
    ax.set_xticklabels(cond_unique, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Bursts / minute', fontsize=10)
    ax.set_title('Burst Rate by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')

    # Panel B: Mean burst amplitude per condition per group
    ax = axes[1]
    for gi, gname in enumerate(group_names):
        amps = []
        for cond in cond_unique:
            cond_bursts = [b for b in all_bursts[gname]['bursts']
                          if get_condition_at_time(t[b['peak_idx']], segments) == cond]
            amps.append(np.mean([b['peak_amp'] for b in cond_bursts])
                       if cond_bursts else 0)
        x = np.arange(n_conds)
        ax.bar(x + gi * bar_width, amps, bar_width * 0.9,
              color=GROUPS[gname]['color'], alpha=0.75, label=gname)

    ax.set_xticks(np.arange(n_conds) + 0.3)
    ax.set_xticklabels(cond_unique, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Peak Amplitude (z-scored RMS)', fontsize=10)
    ax.set_title('Burst Amplitude by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')

    # Panel C: Cross-modal burst coincidence matrix
    ax = axes[2]
    window_coinc = int(5.0 * FS)  # 5 second coincidence window
    coinc_matrix = np.zeros((n_groups, n_groups))

    for gi, g1 in enumerate(group_names):
        peaks_1 = np.array([b['peak_idx'] for b in all_bursts[g1]['bursts']])
        n1 = len(peaks_1)
        for gj, g2 in enumerate(group_names):
            if gi == gj:
                coinc_matrix[gi, gj] = 1.0
                continue
            peaks_2 = np.array([b['peak_idx'] for b in all_bursts[g2]['bursts']])
            if n1 == 0 or len(peaks_2) == 0:
                continue
            # For each burst in g1, is there a g2 burst within ±5s?
            coincident = 0
            for p1 in peaks_1:
                if np.any(np.abs(peaks_2 - p1) <= window_coinc):
                    coincident += 1
            coinc_matrix[gi, gj] = coincident / n1 if n1 > 0 else 0

    im = ax.imshow(coinc_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names, fontsize=8)
    ax.set_title('Burst Coincidence (±5s window)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='P(col burst | row burst)', shrink=0.8)

    # Annotate cells
    for i in range(n_groups):
        for j in range(n_groups):
            val = coinc_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)

    plt.suptitle(f'Coupling Burst Statistics by Condition — {session_id}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'burst_statistics_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: burst_statistics_{session_id}.png")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 4: Cross-modal lead/lag from peri-burst peak latencies
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 8))

    lag_matrix = np.full((n_groups, n_groups), np.nan)
    window_s = 15.0
    window = int(window_s * FS)

    for gi, trigger_group in enumerate(group_names):
        bursts = all_bursts[trigger_group]['bursts']
        if len(bursts) < 5:
            continue
        pba, t_axis, gn, n_valid = peri_burst_average(Y, bursts, window_s=window_s)
        if n_valid < 3:
            continue

        for gj, resp_group in enumerate(gn):
            # Find peak of response group's peri-burst average
            peak_sample = np.argmax(pba[gj])
            lag_s = t_axis[peak_sample]
            lag_matrix[gi, gj] = lag_s

    im = ax.imshow(lag_matrix, cmap='RdBu_r', vmin=-8, vmax=8, aspect='auto')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names, fontsize=9)
    ax.set_xlabel('Response modality', fontsize=11, fontweight='bold')
    ax.set_ylabel('Trigger modality (burst detected here)', fontsize=11, fontweight='bold')
    ax.set_title(f'Cross-Modal Lead/Lag (seconds) — {session_id}\n'
                 f'Negative = response LEADS trigger, Positive = response FOLLOWS trigger',
                fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Peak latency (seconds)', shrink=0.8)

    for i in range(n_groups):
        for j in range(n_groups):
            val = lag_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 4 else 'black'
                ax.text(j, i, f'{val:+.1f}s', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'burst_lead_lag_{session_id}.png'),
               dpi=200, bbox_inches='tight')
    print(f"  Saved: burst_lead_lag_{session_id}.png")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n{'─' * 60}")
    print("Cross-modal lead/lag summary (trigger → response peak):")
    for gi, g1 in enumerate(group_names):
        for gj, g2 in enumerate(group_names):
            if gi != gj and not np.isnan(lag_matrix[gi, gj]):
                lag = lag_matrix[gi, gj]
                direction = "follows" if lag > 0 else "LEADS"
                print(f"  {g1:>12s} burst → {g2:<12s} peak: {lag:+.1f}s ({direction})")

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
