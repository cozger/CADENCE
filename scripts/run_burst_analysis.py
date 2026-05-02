"""Production burst analysis: per-session + cross-session coupling burst statistics.

Runs on all available sessions with V8.2 scaffold data. Produces:
  1. Per-session JSON results (burst rates, asymmetry, per-segment stats)
  2. Cross-session aggregated JSON (mean/SE across sessions)
  3. Per-session timeline figures (coupling intensity + bursts + Viterbi state)
  4. Cross-session summary figures (burst rates, asymmetry by segment)

Usage:
    python scripts/run_burst_analysis.py                    # All sessions
    python scripts/run_burst_analysis.py --session y_06     # Single session
    python scripts/run_burst_analysis.py --no-plots         # JSON only (fast)
"""

import sys, os, json, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadence.significance.burst_analysis import (
    FS, MOD_KEYS, ASYM_KEYS, GROUPS, GROUP_NAMES, N_GROUPS,
    STATE_COLORS, STATE_LABELS,
    SEGMENT_MAP, SEGMENT_ORDER, SEGMENT_COLORS, CONDITION_COLORS,
    load_session, load_session_demit,
    compute_group_intensity, compute_residual_intensity,
    detect_bursts, filter_bursts_by_segment, get_segment_mask,
    compute_burst_asymmetry, compute_participant_power,
    peri_burst_average, analyze_session_by_segment,
    aggregate_sessions,
)


def find_available_sessions(results_dir):
    """Find all sessions with scaffold + Viterbi data."""
    sessions = []
    for d in sorted(glob.glob(os.path.join(results_dir, '*', ''))):
        sess = os.path.basename(os.path.normpath(d))
        scaffold = os.path.join(d, 'scaffold_v82_ztimecourses.npz')
        results_json = os.path.join(d, 'scaffold_v82_results.json')
        v8 = os.path.join(d, 'rslds_v8_full_results.npz')
        p2 = os.path.join(d, 'rslds_phase2_results.npz')
        if (os.path.exists(scaffold) and os.path.exists(results_json) and
                (os.path.exists(v8) or os.path.exists(p2))):
            sessions.append(sess)
    return sessions


# ── Per-session timeline figure ──────────────────────────────────────

def plot_session_timeline(session_id, results_dir, out_dir):
    """Full-session coupling intensity timeline with bursts and Viterbi state."""
    Y, viterbi, t, segments, asym = load_session(session_id, results_dir)
    T = Y.shape[0]
    t_rel = (t - t[0]) / 60.0

    intensities = compute_group_intensity(Y, smooth_s=3.0)

    try:
        d_emit = load_session_demit(session_id, results_dir)
        resid_int, _ = compute_residual_intensity(Y, viterbi, d_emit, smooth_s=3.0)
    except (ValueError, FileNotFoundError):
        resid_int = None

    n_panels = N_GROUPS + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(24, 2.5 * n_panels),
                             sharex=True, gridspec_kw={'height_ratios':
                             [1] * N_GROUPS + [0.5]})

    for gi, gname in enumerate(GROUP_NAMES):
        ax = axes[gi]
        ginfo = GROUPS[gname]
        intensity = intensities[gname]

        # Condition background
        for seg in segments:
            t_s = (seg[1] - t[0]) / 60.0
            t_e = (seg[2] - t[0]) / 60.0
            color = CONDITION_COLORS.get(seg[0], '#F5F5F5')
            ax.axvspan(t_s, t_e, alpha=0.2, color=color, zorder=0)

        # Intensity fill + line
        ax.fill_between(t_rel, 0, intensity, alpha=0.2, color=ginfo['color'])
        ax.plot(t_rel, intensity, color=ginfo['color'], lw=0.8, alpha=0.6,
               label='Total')

        # Residual intensity if available
        if resid_int is not None:
            ax.plot(t_rel, resid_int[gname], color=ginfo['color'], lw=1.2,
                   alpha=0.9, label='Residual')

        # Detect and mark bursts
        bursts, threshold = detect_bursts(intensity)
        ax.axhline(threshold, color='red', ls=':', lw=0.8, alpha=0.4)
        for burst in bursts:
            on, off = burst['onset'], min(burst['offset'], T - 1)
            ax.axvspan(t_rel[on], t_rel[off], alpha=0.12, color=ginfo['color'])
            ax.scatter(t_rel[burst['peak_idx']], burst['peak_amp'],
                      color=ginfo['color'], s=20, zorder=5, edgecolors='black',
                      linewidth=0.4)

        ax.set_ylabel(ginfo['label'], fontsize=8, fontweight='bold',
                     color=ginfo['color'])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.1)
        if gi == 0:
            ax.legend(fontsize=7, loc='upper right')
            for seg in segments:
                t_mid = ((seg[1] + seg[2]) / 2 - t[0]) / 60.0
                ax.text(t_mid, ax.get_ylim()[1] * 0.92,
                       seg[0].replace('_', '\n'),
                       ha='center', va='top', fontsize=6, fontweight='bold',
                       alpha=0.5)

    # Viterbi state bar
    ax_state = axes[-1]
    for seg in segments:
        t_s = (seg[1] - t[0]) / 60.0
        t_e = (seg[2] - t[0]) / 60.0
        ax_state.axvspan(t_s, t_e, alpha=0.15,
                        color=CONDITION_COLORS.get(seg[0], '#F5F5F5'))

    state_changes = np.where(np.diff(viterbi) != 0)[0]
    boundaries = np.concatenate([[0], state_changes + 1, [T]])
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        ax_state.axvspan(t_rel[s], t_rel[min(e, T - 1)],
                        color=STATE_COLORS[viterbi[s]], alpha=0.6)

    for k in range(4):
        ax_state.scatter([], [], c=STATE_COLORS[k], s=30, label=STATE_LABELS[k])
    ax_state.legend(fontsize=7, loc='upper right', ncol=4)
    ax_state.set_ylabel('State', fontsize=8, fontweight='bold')
    ax_state.set_yticks([])
    ax_state.set_xlabel('Time (minutes)', fontsize=10)

    plt.suptitle(f'Coupling Burst Timeline — {session_id}', fontsize=13,
                fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'burst_timeline_{session_id}.png'),
               dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Cross-session summary figures ────────────────────────────────────

def plot_burst_rates(agg, out_dir):
    """Burst rate by segment, one subplot per modality group."""
    fig, axes = plt.subplots(1, N_GROUPS, figsize=(4 * N_GROUPS, 5), sharey=True)

    for gi, trigger in enumerate(GROUP_NAMES):
        ax = axes[gi]
        means, ses, labels, colors = [], [], [], []

        for seg in SEGMENT_ORDER:
            if seg not in agg or trigger not in agg[seg]['rates']:
                continue
            r = agg[seg]['rates'][trigger]
            if r['n'] < 2:
                continue
            means.append(r['mean'])
            ses.append(r['se'])
            labels.append(f"{seg}\nn={r['n']}")
            colors.append(SEGMENT_COLORS.get(seg, '#999'))

        if not labels:
            continue

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, fontweight='bold')
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
    plt.close(fig)


def plot_asymmetry(agg, out_dir):
    """Therapist/patient asymmetry by segment, 2 triggers x 3 bands."""
    band_names = ['theta', 'alpha', 'beta']
    band_labels = ['th', 'al', 'be']
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    for row, trigger in enumerate(['EEG phase', 'EEG power']):
        for col, (bi, bn) in enumerate(zip(range(3), band_labels)):
            ax = axes[row, col]
            means, ses, labels, colors = [], [], [], []

            for seg in SEGMENT_ORDER:
                if seg not in agg or trigger not in agg[seg]['asym']:
                    continue
                a = agg[seg]['asym'][trigger]
                if a['n'] < 2:
                    continue
                means.append(a['mean'][bi])
                ses.append(a['se'][bi])
                sig = '*' if a['significant'][bi] else ''
                labels.append(f"{seg}{sig}\nn={a['n']}")
                colors.append(SEGMENT_COLORS.get(seg, '#999'))

            if not labels:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            x = np.arange(len(labels))
            ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='red', ls='--', lw=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, fontweight='bold')
            ax.set_ylabel('Asymmetry (+ = therapist)', fontsize=9)
            ax.set_title(f'{trigger} — {band_names[bi]}', fontsize=11,
                        fontweight='bold', color=GROUPS[trigger]['color'])
            ax.grid(True, alpha=0.15, axis='y')

    plt.suptitle('Therapist/Patient EEG Asymmetry During Coupling Bursts — by Segment\n'
                 '(+ = therapist higher; error bars = 95% CI; * = |mean| > 2*SE)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'burst_asymmetry_by_segment.png'),
               dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V8.2 Coupling Burst Analysis')
    parser.add_argument('--session', type=str, default=None,
                       help='Single session to analyze (default: all)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip figure generation (JSON only)')
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', 'rslds')
    out_dir = os.path.join(results_dir, 'burst_analysis')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("V8.2 Coupling Burst Analysis — Production")
    print("=" * 70)

    # ── Find sessions ──
    if args.session:
        sessions = [args.session]
    else:
        sessions = find_available_sessions(results_dir)
    print(f"\nSessions to process: {len(sessions)}")

    # ── Per-session analysis ──
    all_results = []
    for sid in sessions:
        print(f"  {sid}...", end=' ', flush=True)
        try:
            result = analyze_session_by_segment(sid, results_dir)
            n_bursts = sum(sum(s['burst_counts'].values())
                          for s in result['segments'].values())
            segs = sorted(result['segments'].keys())
            print(f"{n_bursts} bursts, segments={segs}")
            all_results.append(result)

            # Save per-session JSON
            sess_out = os.path.join(out_dir, f'{sid}_bursts.json')
            with open(sess_out, 'w') as f:
                json.dump(result, f, indent=2)

            # Per-session timeline figure
            if not args.no_plots:
                plot_session_timeline(sid, results_dir, out_dir)

        except Exception as e:
            print(f"FAILED: {e}")

    if not all_results:
        print("\nNo sessions processed successfully.")
        return

    # ── Cross-session aggregation ──
    print(f"\n── Cross-session aggregation ({len(all_results)} sessions) ──")
    agg = aggregate_sessions(all_results)

    # Print summary
    band_names = ['th', 'al', 'be']
    print(f"\n{'Segment':>12s} | {'n':>3s} | {'EEG ph':>7s} {'EEG pw':>7s} {'Face':>7s} {'Auto':>7s} {'Body':>7s} | Asym (EEG power theta)")
    print("-" * 85)
    for seg in SEGMENT_ORDER:
        if seg not in agg:
            continue
        sa = agg[seg]
        n = sa['n_sessions']
        rates = []
        for g in GROUP_NAMES:
            if g in sa['rates']:
                rates.append(f"{sa['rates'][g]['mean']:5.1f}")
            else:
                rates.append("  -  ")
        rate_str = ' '.join(rates)

        asym_str = ''
        if 'EEG power' in sa['asym']:
            a = sa['asym']['EEG power']
            sig = '*' if a['significant'][0] else ''
            asym_str = f"{a['mean'][0]:+.3f}{sig}"

        print(f"{seg:>12s} | {n:3d} | {rate_str} | {asym_str}")

    # Save aggregated results
    agg_out = os.path.join(out_dir, 'cross_session_burst_results.json')
    with open(agg_out, 'w') as f:
        json.dump({
            'n_sessions': len(all_results),
            'sessions': [r['session'] for r in all_results],
            'segments': agg,
            'group_names': GROUP_NAMES,
            'segment_order': SEGMENT_ORDER,
        }, f, indent=2)
    print(f"\n  Saved: {agg_out}")

    # ── Cross-session figures ──
    if not args.no_plots and len(all_results) >= 2:
        print("\n── Generating cross-session figures ──")
        plot_burst_rates(agg, out_dir)
        print(f"  Saved: burst_rate_by_segment.png")
        plot_asymmetry(agg, out_dir)
        print(f"  Saved: burst_asymmetry_by_segment.png")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.1f}s. Output: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
