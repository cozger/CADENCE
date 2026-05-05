"""V8.2 Full Analysis: null-state IOHMM K=4 + constrained Viterbi + timeline plots.

Runs on all V8.2 scaffold data. Generates per-session timelines and cross-session
condition breakdown.
"""

import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from cadence.significance.rslds_model import IOHMM, IOHMMConfig

MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MODALITY_NAMES = [
    'ImCoh th', 'ImCoh al', 'ImCoh be',
    'Conc th', 'Conc al', 'Conc be',
    'BL expr', 'BL act conc',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]
MODALITY_COLORS = [
    '#1565C0', '#2196F3', '#64B5F6',
    '#E65100', '#FF9800', '#FFB74D',
    '#E91E63', '#F48FB1',
    '#FF5722', '#795548', '#607D8B', '#4CAF50',
]
STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
FS_OUT = 2.0


def constrained_viterbi(gamma, min_dwell=20):
    path = np.argmax(gamma, axis=1)
    for _ in range(10):
        ch = False
        ss = [0]
        for t in range(1, len(path)):
            if path[t] != path[t-1]:
                ss.append(t)
        ss.append(len(path))
        for i in range(len(ss) - 1):
            s, e = ss[i], ss[i+1]
            if e - s < min_dwell:
                ls = path[ss[i]-1] if i > 0 else path[s]
                lsc = gamma[s:e, ls].mean() if i > 0 else -np.inf
                rs = path[ss[i+1]] if i < len(ss)-2 else path[s]
                rsc = gamma[s:e, rs].mean() if i < len(ss)-2 else -np.inf
                b = ls if lsc >= rsc else rs
                if b != path[s]:
                    path[s:e] = b
                    ch = True
        if not ch:
            break
    return path


def process_session(session_name):
    npz = f'results/rslds/{session_name}/scaffold_v82_ztimecourses.npz'
    json_path = f'results/rslds/{session_name}/scaffold_v82_results.json'
    if not os.path.exists(npz):
        return None

    data = np.load(npz)
    t = data['t_common']
    Y = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
    obs_mask = data['obs_mask'] if 'obs_mask' in data else np.ones_like(Y, dtype=bool)
    T, D = Y.shape
    U = np.zeros((T, 2), dtype=np.float64)

    segments = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            info = json.load(f)
        segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

    # Fit null-state IOHMM
    cfg = IOHMMConfig(K=4, D_obs=D, D_input=2, n_restarts=5, max_em_iter=200,
                      sticky_strength=3.0, null_state=True, null_sigma2_cap=5.0)
    model = IOHMM(cfg)
    params, hist = model.fit(Y, U, obs_mask, seed=42, verbose=False)
    gamma = hist['gamma']
    labels = constrained_viterbi(gamma, min_dwell=20)

    # State labels based on emission profile
    sl = ['NULL', '', '', '']
    # Coupling state: highest ImCoh sum (non-null)
    imcoh_sum = [params.mu[k, 0] + params.mu[k, 1] + params.mu[k, 2] for k in range(4)]
    imcoh_sum[0] = -999
    coup_k = int(np.argmax(imcoh_sum))
    sl[coup_k] = 'COUP'
    # Shared state: highest concordance sum (non-null, non-coup)
    conc_sum = [params.mu[k, 3] + params.mu[k, 4] + params.mu[k, 5] for k in range(4)]
    conc_sum[0] = -999
    conc_sum[coup_k] = -999
    shared_k = int(np.argmax(conc_sum))
    if sl[shared_k] == '':
        sl[shared_k] = 'SHARED'
    for k in range(4):
        if sl[k] == '':
            sl[k] = 'OTHER'

    # Build all periods
    seg_sorted = sorted(segments, key=lambda x: x[1])
    all_periods = []
    if seg_sorted and seg_sorted[0][1] > t[0] + 5:
        all_periods.append(('gap_pre', t[0], seg_sorted[0][1]))
    for i, (name, t0, t1) in enumerate(seg_sorted):
        all_periods.append((name, t0, t1))
        if i < len(seg_sorted) - 1:
            if seg_sorted[i+1][1] - t1 > 3:
                all_periods.append(('gap', t1, seg_sorted[i+1][1]))
    if seg_sorted and seg_sorted[-1][2] < t[-1] - 5:
        all_periods.append(('gap_post', seg_sorted[-1][2], t[-1]))

    period_data = []
    for pname, t0, t1 in all_periods:
        mask = (t >= t0) & (t <= t1)
        if mask.sum() < 3:
            continue
        usage = [float((labels[mask] == k).mean()) for k in range(4)]
        period_data.append((pname, t1 - t0, usage))

    # ── Timeline plot ─────────────────────────────────────────────────
    n_panels = D + 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(28, 1.8 * n_panels),
                             gridspec_kw={'height_ratios': [1]*D + [0.7, 0.3]},
                             sharex=True)

    for ax in axes:
        for sn, t0s, t1s in segments:
            ax.axvspan(t0s, t1s, alpha=0.2,
                       color=CONDITION_COLORS.get(sn, '#F5F5F5'), zorder=0)
        ax.set_xlim(t[0], t[-1])

    for sn, t0s, t1s in segments:
        axes[0].text((t0s + t1s) / 2, 1.08, sn.replace('_', ' '),
                     ha='center', va='bottom', fontsize=6, fontweight='bold',
                     transform=axes[0].get_xaxis_transform())

    for idx in range(D):
        ax = axes[idx]
        # State background
        for k in range(4):
            sm = labels == k
            runs = np.diff(np.concatenate([[0], sm.astype(int), [0]]))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            for s, e in zip(starts, ends):
                if s < T and e <= T:
                    ax.axvspan(t[max(0, s)], t[min(e-1, T-1)],
                               alpha=0.1, color=STATE_COLORS[k], zorder=0)
        ax.plot(t, Y[:, idx], color=MODALITY_COLORS[idx], linewidth=0.4, alpha=0.8, zorder=2)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.3)
        ax.set_ylabel(MODALITY_NAMES[idx], fontsize=6, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=5)

    # State posteriors
    ax_g = axes[D]
    for k in range(4):
        ax_g.fill_between(t, 0, gamma[:, k], alpha=0.5, color=STATE_COLORS[k],
                          label=f'S{k}({sl[k]})')
    ax_g.set_ylim(0, 1)
    ax_g.legend(fontsize=5, loc='upper right', ncol=4)
    ax_g.set_ylabel('Post', fontsize=6, rotation=0, ha='right', va='center')
    ax_g.tick_params(labelsize=5)

    # Viterbi strip
    ax_v = axes[D + 1]
    for k in range(4):
        ax_v.fill_between(t, 0, 1, where=labels == k, color=STATE_COLORS[k], alpha=0.8)
    ax_v.set_ylim(0, 1)
    ax_v.set_yticks([])
    ax_v.set_xlabel('Time (s)', fontsize=7)

    n_trans = int(np.sum(labels[1:] != labels[:-1]))
    usage = [float((labels == k).mean()) for k in range(4)]
    title = (f'{session_name} V8.2 (12D) K=4 null-state | Trans={n_trans} | '
             + ' '.join(f'S{k}({sl[k]})={u:.0%}' for k, u in enumerate(usage)))
    fig.suptitle(title, fontsize=8, fontfamily='monospace')
    plt.tight_layout(rect=[0.06, 0, 1, 0.97])

    out_dir = f'results/rslds/{session_name}'
    fig.savefig(os.path.join(out_dir, 'v82_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'session': session_name,
        'state_labels': sl,
        'usage': usage,
        'n_transitions': n_trans,
        'periods': period_data,
        'emission_mu': params.mu.tolist(),
        'bic': float(hist['bic']),
    }


def main():
    npzs = sorted(glob.glob('results/rslds/*/scaffold_v82_ztimecourses.npz'))
    sessions = [os.path.basename(os.path.dirname(p)) for p in npzs]
    print(f'Fitting {len(sessions)} sessions in parallel...')

    t0 = time.time()
    results = Parallel(n_jobs=-1)(delayed(process_session)(s) for s in sessions)
    results = [r for r in results if r is not None]
    print(f'Done in {time.time()-t0:.0f}s\n')

    # Per-session summary
    for r in results:
        sl = r['state_labels']
        print(f"\n  {r['session']} | BIC={r['bic']:.0f} | Trans={r['n_transitions']} | "
              + ' '.join(f"S{k}({sl[k]})={u:.0%}" for k, u in enumerate(r['usage'])))
        print(f"  {'Period':>20s} | Dur  | "
              f"S0({sl[0][:4]}) S1({sl[1][:4]}) S2({sl[2][:4]}) S3({sl[3][:4]})")
        print(f"  " + "-" * 70)
        for pname, dur, usage in r['periods']:
            u_str = ''.join(f'{u:8.1%}' for u in usage)
            marker = ' ***' if pname == 'gap' and dur > 60 else (
                ' *' if pname.startswith('gap') else '')
            print(f"  {pname:>20s} | {dur:4.0f} | {u_str}{marker}")

    # Condition type aggregation
    print(f"\n{'='*80}")
    print(f"  AGGREGATE: State usage by condition type")
    print(f"{'='*80}")

    cond_data = {}
    for r in results:
        for pname, dur, usage in r['periods']:
            if pname.startswith('base') or pname == 'baseline':
                ct = 'Baseline'
            elif pname.startswith('conv'):
                ct = 'Conversation'
            elif pname.startswith('PE'):
                ct = 'PE'
            elif pname.startswith('meditate'):
                ct = 'Meditation'
            elif pname.startswith('gap'):
                ct = 'Gaps'
            else:
                ct = pname
            if ct not in cond_data:
                cond_data[ct] = {f's{k}': [] for k in range(4)}
            for k in range(4):
                cond_data[ct][f's{k}'].append(usage[k])

    print(f"  {'Condition':>14s} | n  | S0(null)       | S1             | S2             | S3")
    print(f"  " + "-" * 80)
    for ct in ['Conversation', 'PE', 'Meditation', 'Baseline', 'Gaps']:
        if ct not in cond_data:
            continue
        cd = cond_data[ct]
        n = len(cd['s0'])
        parts = []
        for k in range(4):
            vals = cd[f's{k}']
            parts.append(f"{np.mean(vals):.0%}+/-{np.std(vals):.0%}")
        print(f"  {ct:>14s} | {n:2d} | " + " | ".join(f"{p:14s}" for p in parts))

    # Save results
    with open('results/rslds/v82_full_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved results/rslds/v82_full_analysis_results.json")
    print(f"  Timeline plots: results/rslds/*/v82_timeline.png")


if __name__ == '__main__':
    main()
