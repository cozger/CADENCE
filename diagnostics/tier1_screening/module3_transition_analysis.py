from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax
from scipy.stats import ks_2samp


def compute_dwell_times(path: np.ndarray) -> Dict[int, List[int]]:
    """Return dict mapping state_id -> list of dwell lengths (in samples)."""
    dwells: Dict[int, List[int]] = {}
    if len(path) == 0:
        return dwells
    current, count = int(path[0]), 1
    for s in path[1:]:
        s = int(s)
        if s == current:
            count += 1
        else:
            dwells.setdefault(current, []).append(count)
            current, count = s, 1
    dwells.setdefault(current, []).append(count)
    return dwells


def transition_matrix_from_logits(W_trans: np.ndarray) -> np.ndarray:
    """Convert (K, K) logit matrix to row-stochastic transition matrix."""
    return np.array([softmax(W_trans[k]) for k in range(len(W_trans))])


def geometric_mean_dwell(T_kk: float) -> float:
    """Expected mean dwell in samples for geometric(1 - T_kk)."""
    return 1.0 / (1.0 - T_kk)


def transition_event_ks_test(
    transition_times: np.ndarray,
    event_times: List[float],
    session_length_s: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """KS test: are transitions closer to events than random?

    Returns (ks_statistic, p_value).
    p < 0.05 → transitions cluster near events (real signal).
    Uses alternative='greater': F_real(t) > F_boot(t) when real distances are
    smaller, giving a large statistic and small p-value.
    """
    rng = np.random.default_rng(seed)
    event_arr = np.array(event_times)

    def min_dist(times):
        return np.array([np.min(np.abs(t - event_arr)) for t in times])

    real_dists = min_dist(transition_times)
    n = len(transition_times)
    boot_dists = np.concatenate([
        min_dist(rng.uniform(0, session_length_s, n))
        for _ in range(n_boot)
    ])
    # 'greater': F_real(t) > F_boot(t) when real are smaller → large statistic → small p
    ks_stat, p_val = ks_2samp(real_dists, boot_dists, alternative='greater')
    return float(ks_stat), float(p_val)


import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostics.shared.data_loader import SessionData


_PROTOCOL_PHASES = {
    'meditation': {'base_EO', 'base_EC', 'conv_1', 'conv_2', 'meditate_B', 'meditate_K'},
    'pe':         {'base_EO', 'base_EC', 'conv_1', 'conv_2', 'PE_1', 'PE_2', 'PE_start'},
}

STATE_COLORS = ['#aaaaaa', '#4477AA', '#228833', '#EE6677', '#CCBB44']


def _detect_protocol(session: SessionData) -> str:
    names = {s[0] for s in session.segments}
    if names & {'meditate_B', 'meditate_K'}:
        return 'meditation'
    if names & {'PE_1', 'PE_2', 'PE_start'}:
        return 'pe'
    return 'unknown'


def run_transition_analysis(sessions: list, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    fs = sessions[0].fs
    flags = {'dwell_ratio_low': [], 'high_flicker_pct': [],
              'transitions_event_locked': False, 'transitions_random': False}
    all_coincidence_rows = []

    # ── Dwell-time distributions ─────────────────────────────────────────
    K = 4
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    state_labels = ['S0', 'S1', 'S2', 'S3']

    for sess in sessions:
        if sess.path_unconstrained is None or sess.W_trans is None:
            continue
        T = transition_matrix_from_logits(sess.W_trans)
        sl = sess.state_labels or [f'S{k}' for k in range(K)]
        for k in range(min(K, T.shape[0])):
            dwells = compute_dwell_times(sess.path_unconstrained).get(k, [])
            if not dwells:
                continue
            pred_mean = geometric_mean_dwell(T[k, k])
            emp_mean = float(np.mean(dwells))
            ratio = emp_mean / pred_mean
            if ratio < 0.5:
                flags['dwell_ratio_low'].append(f'{sess.name}:S{k}(ratio={ratio:.2f})')
            pct_short = float(np.mean(np.array(dwells) < (10.0 * fs)))
            if pct_short > 0.5:
                flags['high_flicker_pct'].append(f'{sess.name}:S{k}({pct_short:.0%})')

            axes[k].hist(np.array(dwells) / fs, bins=30, density=True,
                         alpha=0.5, label=sess.name)
            dw_s = np.linspace(0.5, max(dwells) / fs, 200)
            p_geom = 1 - T[k, k] ** np.round(dw_s * fs)
            if not axes[k].lines:
                axes[k].plot(dw_s, np.diff(np.concatenate([[0], p_geom])),
                             color='red', linewidth=1.5, label='geometric null')

        state_labels = sl

    for k in range(K):
        axes[k].set_title(f'State {state_labels[k] if k < len(state_labels) else k}', fontsize=8)
        axes[k].set_xlabel('Dwell (s)')
        if k == 0:
            axes[k].legend(fontsize=5)

    fig.suptitle('Dwell-time distributions vs geometric(1-T_kk) null', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'dwell_time_distributions.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Per-session timelines + coincidence test ─────────────────────────
    for sess in sessions:
        if sess.path_unconstrained is None:
            continue
        t = sess.t_common
        sl = sess.state_labels or [f'S{k}' for k in range(K)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 2.5), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1]})
        for ax, path, label in [(ax1, sess.path_unconstrained, 'Unconstrained'),
                                 (ax2, sess.path_constrained, '10s-min Viterbi')]:
            if path is None:
                continue
            for k in range(K):
                mask = path == k
                if mask.any():
                    ax.fill_between(t, k, k + 1, where=mask,
                                    color=STATE_COLORS[k % len(STATE_COLORS)], alpha=0.8)
            ax.set_ylim(0, K)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=7)
        for seg_name, t0, t1 in sess.segments:
            ax1.axvline(t0, color='black', linewidth=0.8, alpha=0.6)
            ax2.axvline(t0, color='black', linewidth=0.8, alpha=0.6)
            ax1.text(t0, K + 0.1, seg_name[:8], fontsize=5, rotation=45)
        fig.suptitle(f'{sess.name} — state timeline', fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{sess.name}_timeline.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

        trans_idx = np.where(np.diff(sess.path_unconstrained) != 0)[0]
        trans_times = t[trans_idx]
        event_times = [t0 for _, t0, _ in sess.segments]
        session_length = float(t[-1] - t[0])
        if len(trans_times) > 5 and len(event_times) > 0:
            ks, p = transition_event_ks_test(trans_times, event_times, session_length,
                                             n_boot=500, seed=0)
            proto = _detect_protocol(sess)
            all_coincidence_rows.append({
                'session': sess.name, 'protocol': proto,
                'n_transitions': len(trans_times), 'n_events': len(event_times),
                'KS_statistic': ks, 'p_value': p,
                'mean_distance_real_s': float(np.mean([
                    np.min(np.abs(tt - np.array(event_times))) for tt in trans_times])),
            })

    if all_coincidence_rows:
        df = pd.DataFrame(all_coincidence_rows)
        df.to_csv(os.path.join(output_dir, 'coincidence_test_results.csv'), index=False)
        pooled_p = float(df['p_value'].median())
        if pooled_p < 0.05:
            flags['transitions_event_locked'] = True
        if pooled_p > 0.1:
            flags['transitions_random'] = True
    else:
        pd.DataFrame().to_csv(os.path.join(output_dir, 'coincidence_test_results.csv'), index=False)

    with open(os.path.join(output_dir, 'module3_report.md'), 'w') as f:
        f.write('# Module 3: State Transition Analysis\n\n')
        f.write(f'DWELL_RATIO_LOW: {flags["dwell_ratio_low"]}\n')
        f.write(f'HIGH_FLICKER_PCT: {flags["high_flicker_pct"]}\n')
        event_verdict = ('TRANSITIONS_EVENT_LOCKED' if flags['transitions_event_locked']
                         else 'TRANSITIONS_RANDOM' if flags['transitions_random']
                         else 'TRANSITIONS_AMBIGUOUS')
        f.write(f'{event_verdict}\n')

    return flags
