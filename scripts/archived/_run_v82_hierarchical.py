"""V8.2 Hierarchical rSLDS: shared dynamics across all sessions, 12D features."""

import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cadence.significance.rslds_model import (
    IOHMMConfig, fit_hierarchical_slds, slds_m_step_emissions,
)

MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_theta', 'dyn_alpha', 'dyn_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
MODALITY_NAMES = [
    'ImCoh th', 'ImCoh al', 'ImCoh be',
    'Conc th', 'Conc al', 'Conc be',
    'Dyn th', 'Dyn al', 'Dyn be',
    'BL expr', 'BL act conc',
    'ECG LF', 'ECG HF', 'Resp', 'Pose',
]
STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
MODALITY_COLORS = [
    '#1565C0', '#2196F3', '#64B5F6',
    '#E65100', '#FF9800', '#FFB74D',
    '#B71C1C', '#D32F2F', '#E57373',
    '#E91E63', '#F48FB1',
    '#FF5722', '#795548', '#607D8B', '#4CAF50',
]
FS_OUT = 2.0


def cv(gamma, md=20):
    path = np.argmax(gamma, axis=1)
    for _ in range(10):
        ch = False
        ss = [0]
        for t_i in range(1, len(path)):
            if path[t_i] != path[t_i-1]:
                ss.append(t_i)
        ss.append(len(path))
        for i in range(len(ss) - 1):
            s, e = ss[i], ss[i+1]
            if e - s < md:
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


def main():
    print("=" * 80)
    print("  V8.2 Hierarchical rSLDS (12D, shared dynamics)")
    print("=" * 80)
    t_wall = time.time()

    # Load all sessions
    npzs = sorted(glob.glob('results/rslds/*/scaffold_v82_ztimecourses.npz'))
    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']
        Y = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
        obs_mask = data['obs_mask'] if 'obs_mask' in data else np.ones_like(Y, dtype=bool)
        T, D = Y.shape
        U = np.zeros((T, 2), dtype=np.float64)

        json_path = f'results/rslds/{name}/scaffold_v82_results.json'
        segments = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                info = json.load(f)
            segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

        sessions.append((Y, U, obs_mask, t, name, segments))

    print(f"\n  Loaded {len(sessions)} sessions")
    for Y, U, mask, t, name, segs in sessions:
        print(f"    {name}: {Y.shape[0]} pts, {Y.shape[0]/FS_OUT:.0f}s, "
              f"{len(segs)} conditions")

    # Fit hierarchical rSLDS
    cfg = IOHMMConfig(
        K=4, D_obs=len(MODALITY_KEYS), D_input=2, D_latent=3,
        n_factors=2, recurrent=True,
        n_restarts=2, max_em_iter=80,
        sticky_strength=3.0,
        null_state=True, null_sigma2_cap=5.0,
    )

    session_tuples = [(Y, U, mask) for Y, U, mask, t, name, segs in sessions]
    print(f"\n  Fitting hierarchical rSLDS (K=4, D_latent=3, 12D, null-state)...")
    result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)
    print(f"  BIC: {result['bic']:.0f}")

    # Shared d_emit profiles
    all_d = [result['sessions'][i].get('d_emit') for i in range(len(sessions))]
    mean_d = np.mean(all_d, axis=0)

    # State labels from shared d_emit
    sl = ['NULL', '', '', '']
    imcoh_sum = [mean_d[k, 0] + mean_d[k, 1] + mean_d[k, 2] for k in range(4)]
    imcoh_sum[0] = -999
    coup_k = int(np.argmax(imcoh_sum))
    sl[coup_k] = 'COUP'
    conc_sum = [mean_d[k, 3] + mean_d[k, 4] + mean_d[k, 5] for k in range(4)]
    conc_sum[0] = -999
    conc_sum[coup_k] = -999
    shared_k = int(np.argmax(conc_sum))
    if sl[shared_k] == '':
        sl[shared_k] = 'SHARED'
    for k in range(4):
        if sl[k] == '':
            sl[k] = 'OTHER'

    print(f"\n  Shared d_emit (state labels: {sl}):")
    print(f"  {'State':>8s} | " + ' '.join(f'{k:>10s}' for k in MODALITY_KEYS))
    print(f"  " + "-" * (10 + 11 * len(MODALITY_KEYS)))
    for k in range(4):
        vals = ' '.join(f'{mean_d[k, d]:+10.3f}' for d in range(len(MODALITY_KEYS)))
        print(f"  S{k}({sl[k]:>6s}) | {vals}")

    # Per-session analysis + timeline plots
    all_results = []
    for i, (Y, U, mask, t, name, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path = cv(gamma, md=20)
        T = len(path)

        # Build periods
        seg_sorted = sorted(segments, key=lambda x: x[1])
        all_periods = []
        if seg_sorted and seg_sorted[0][1] > t[0] + 5:
            all_periods.append(('gap_pre', t[0], seg_sorted[0][1]))
        for j, (sn, t0, t1) in enumerate(seg_sorted):
            all_periods.append((sn, t0, t1))
            if j < len(seg_sorted) - 1 and seg_sorted[j+1][1] - t1 > 3:
                all_periods.append(('gap', t1, seg_sorted[j+1][1]))
        if seg_sorted and seg_sorted[-1][2] < t[-1] - 5:
            all_periods.append(('gap_post', seg_sorted[-1][2], t[-1]))

        period_data = []
        for pname, t0, t1 in all_periods:
            pmask = (t >= t0) & (t <= t1)
            if pmask.sum() < 3:
                continue
            usage = [float((path[pmask] == k).mean()) for k in range(4)]
            period_data.append((pname, t1 - t0, usage))

        n_trans = int(np.sum(path[1:] != path[:-1]))
        usage = [float((path == k).mean()) for k in range(4)]

        all_results.append({
            'session': name, 'sl': sl, 'usage': usage,
            'n_trans': n_trans, 'periods': period_data,
        })

        # Timeline plot
        D = Y.shape[1]
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
            for k in range(4):
                sm = path == k
                runs = np.diff(np.concatenate([[0], sm.astype(int), [0]]))
                for s, e in zip(np.where(runs == 1)[0], np.where(runs == -1)[0]):
                    if s < T and e <= T:
                        ax.axvspan(t[max(0, s)], t[min(e-1, T-1)],
                                   alpha=0.1, color=STATE_COLORS[k], zorder=0)
            ax.plot(t, Y[:, idx], color=MODALITY_COLORS[idx],
                    linewidth=0.4, alpha=0.8, zorder=2)
            ax.axhline(0, color='gray', linewidth=0.3, alpha=0.3)
            ax.set_ylabel(MODALITY_NAMES[idx], fontsize=6, rotation=0,
                          ha='right', va='center')
            ax.tick_params(labelsize=5)

        ax_g = axes[D]
        for k in range(4):
            ax_g.fill_between(t, 0, gamma[:, k], alpha=0.5,
                              color=STATE_COLORS[k], label=f'S{k}({sl[k]})')
        ax_g.set_ylim(0, 1)
        ax_g.legend(fontsize=5, loc='upper right', ncol=4)
        ax_g.set_ylabel('Post', fontsize=6, rotation=0, ha='right', va='center')

        ax_v = axes[D + 1]
        for k in range(4):
            ax_v.fill_between(t, 0, 1, where=path == k,
                              color=STATE_COLORS[k], alpha=0.8)
        ax_v.set_ylim(0, 1)
        ax_v.set_yticks([])
        ax_v.set_xlabel('Time (s)', fontsize=7)

        fig.suptitle(
            f'{name} V8.2 Hierarchical rSLDS | BIC={result["bic"]:.0f} | '
            f'Trans={n_trans} | '
            + ' '.join(f'S{k}({sl[k]})={u:.0%}' for k, u in enumerate(usage)),
            fontsize=8, fontfamily='monospace')
        plt.tight_layout(rect=[0.06, 0, 1, 0.97])
        fig.savefig(f'results/rslds/{name}/v82_hierarchical_timeline.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Per-session summary
    print(f"\n{'='*80}")
    print(f"  PER-SESSION SUMMARY")
    print(f"{'='*80}")
    for r in all_results:
        print(f"\n  {r['session']} | Trans={r['n_trans']} | "
              + ' '.join(f"S{k}({sl[k]})={u:.0%}" for k, u in enumerate(r['usage'])))
        print(f"  {'Period':>20s} | Dur  | "
              f"S0({sl[0][:4]}) S1({sl[1][:4]}) S2({sl[2][:4]}) S3({sl[3][:4]})")
        print(f"  " + "-" * 70)
        for pname, dur, usage in r['periods']:
            u_str = ''.join(f'{u:8.1%}' for u in usage)
            marker = ' ***' if pname == 'gap' and dur > 60 else (
                ' *' if pname.startswith('gap') else '')
            print(f"  {pname:>20s} | {dur:4.0f} | {u_str}{marker}")

    # Aggregate by condition type
    print(f"\n{'='*80}")
    print(f"  AGGREGATE: State usage by condition type (aligned via shared dynamics)")
    print(f"{'='*80}")
    cond_data = {}
    for r in all_results:
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

    print(f"  {'Condition':>14s} | n  | S0({sl[0][:4]:>4s})     | "
          f"S1({sl[1][:4]:>4s})     | S2({sl[2][:4]:>4s})     | S3({sl[3][:4]:>4s})")
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
        print(f"  {ct:>14s} | {n:2d} | " + " | ".join(f"{p:12s}" for p in parts))

    # Save
    with open('results/rslds/v82_hierarchical_results.json', 'w') as f:
        json.dump({
            'bic': float(result['bic']),
            'state_labels': sl,
            'd_emit_shared': mean_d.tolist(),
            'sessions': all_results,
        }, f, indent=2, default=str)

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved results/rslds/v82_hierarchical_results.json")
    print(f"  Timelines: results/rslds/*/v82_hierarchical_timeline.png")


if __name__ == '__main__':
    main()
