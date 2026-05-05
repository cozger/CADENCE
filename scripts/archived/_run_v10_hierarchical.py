"""V10 Hierarchical rSLDS: 24D features, 5D covariates, shared dynamics.

Extends V8.2 hierarchical with:
  - 24D observation vector (18 V8.2 + 4 LZ + 2 graph)
  - 5D transition covariates (z_slow PCs + flexibility + λ₂ + change-point)
  - c_shrinkage=0.3 (Phase 3.2 shared loading regularization)
  - recurrent=True (Phase 3.3 latent→discrete feedback)

Usage:
    python scripts/_run_v10_hierarchical.py
"""

import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds
from cadence.constants import V10_MODALITY_KEYS, V10_MODALITY_NAMES, V10_COVARIATE_KEYS

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
FS_OUT = 2.0


def cv(gamma, md=20):
    """Constrained Viterbi: enforce minimum dwell time."""
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
    print("  V10 Hierarchical rSLDS (24D, 5D covariates, shared dynamics)")
    print("=" * 80)
    t_wall = time.time()

    # Load all V10 sessions
    npzs = sorted(glob.glob('results/v10/*/scaffold_v10_ztimecourses.npz'))
    if not npzs:
        print("  No V10 scaffold results found. Run _run_scaffold_v10.py --all first.")
        return

    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']

        # Load 24D observations
        Y = np.column_stack([data[f'z_{k}'] for k in V10_MODALITY_KEYS]).astype(np.float64)
        obs_mask = data['obs_mask'] if 'obs_mask' in data else np.ones_like(Y, dtype=bool)

        # Load 5D transition covariates
        if 'U_covariates' in data:
            U = data['U_covariates'].astype(np.float64)
        else:
            U = np.column_stack([data[f'u_{k}'] for k in V10_COVARIATE_KEYS]).astype(np.float64)

        # Segments
        json_path = f'results/v10/{name}/scaffold_v10_results.json'
        segments = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                info = json.load(f)
            segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

        sessions.append((Y, U, obs_mask, t, name, segments))

    print(f"\n  Loaded {len(sessions)} sessions")
    for Y, U, mask, t, name, segs in sessions:
        print(f"    {name}: {Y.shape[0]} pts ({Y.shape[0]/FS_OUT:.0f}s), "
              f"D_obs={Y.shape[1]}, D_input={U.shape[1]}, "
              f"{len(segs)} conditions")

    # V10 config: 24D obs, 5D covariates, recurrent, c_shrinkage
    D_obs = len(V10_MODALITY_KEYS)  # 23D
    D_input = len(V10_COVARIATE_KEYS)  # 5D
    cfg = IOHMMConfig(
        K=4,
        D_obs=D_obs,
        D_input=D_input,
        D_latent=3,
        n_factors=2,
        recurrent=True,
        c_shrinkage=0.3,
        n_restarts=2,
        max_em_iter=80,
        sticky_strength=3.0,
        null_state=True,
        null_sigma2_cap=5.0,
    )

    session_tuples = [(Y, U, mask) for Y, U, mask, t, name, segs in sessions]
    print(f"\n  Fitting hierarchical rSLDS "
          f"(K={cfg.K}, D_latent={cfg.D_latent}, D_obs={cfg.D_obs}, "
          f"D_input={cfg.D_input}, recurrent={cfg.recurrent}, "
          f"c_shrinkage={cfg.c_shrinkage})...")

    result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)
    print(f"  BIC: {result['bic']:.0f}")
    print(f"  Final LL: {result['final_ll']:.0f}")

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

    print(f"\n  State labels: {sl}")

    # Print shared d_emit for all 24 channels
    print(f"\n  Shared d_emit (24D):")
    for k in range(4):
        top_features = np.argsort(np.abs(mean_d[k]))[::-1][:6]
        top_str = ', '.join(f'{V10_MODALITY_KEYS[d]}={mean_d[k, d]:+.2f}' for d in top_features)
        print(f"    S{k}({sl[k]:>6s}): {top_str}")

    # Check transition covariate effects (S_trans for flexibility, lambda2, change-point)
    if 'S_trans' in result.get('shared', {}):
        S = result['shared']['S_trans']  # (K, K, D_input)
        print(f"\n  Transition covariate effects (S_trans):")
        for ci, cov_name in enumerate(V10_COVARIATE_KEYS):
            S_slice = S[:, :, ci]
            max_effect = np.abs(S_slice).max()
            print(f"    {cov_name}: max|S|={max_effect:.3f}")

    # Save results
    out_dir = 'results/v10/hierarchical'
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        'version': 'v10',
        'n_sessions': len(sessions),
        'config': {
            'K': cfg.K, 'D_obs': cfg.D_obs, 'D_input': cfg.D_input,
            'D_latent': cfg.D_latent, 'recurrent': cfg.recurrent,
            'c_shrinkage': cfg.c_shrinkage, 'n_factors': cfg.n_factors,
        },
        'bic': float(result['bic']),
        'final_ll': float(result['final_ll']),
        'state_labels': sl,
        'modality_keys': V10_MODALITY_KEYS,
        'covariate_keys': V10_COVARIATE_KEYS,
    }

    # Per-session analysis
    all_session_results = []
    for i, (Y, U, mask, t, name, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path = cv(gamma, md=20)

        # Usage per condition
        seg_sorted = sorted(segments, key=lambda x: x[1])
        period_data = []
        for pname, t0, t1 in seg_sorted:
            pmask = (t >= t0) & (t <= t1)
            if pmask.sum() < 3:
                continue
            usage = [float((path[pmask] == k).mean()) for k in range(4)]
            period_data.append((pname, float(t1 - t0), usage))

        n_trans = int(np.sum(path[1:] != path[:-1]))
        usage = [float((path == k).mean()) for k in range(4)]

        all_session_results.append({
            'session': name, 'state_labels': sl,
            'usage': usage, 'n_transitions': n_trans,
            'periods': period_data,
        })

        # Save per-session npz with state assignments
        sess_out = f'results/v10/{name}'
        np.savez_compressed(os.path.join(sess_out, 'v10_rslds_results.npz'),
                            gamma=gamma, path=path, t_common=t)

    summary['sessions'] = all_session_results

    with open(os.path.join(out_dir, 'v10_hierarchical_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Timeline plots for each session
    for i, (Y, U, mask, t, name, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path = cv(gamma, md=20)

        fig, axes = plt.subplots(3, 1, figsize=(24, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 3, 1]})

        # State posterior
        ax = axes[0]
        for k in range(4):
            ax.fill_between(t, 0, gamma[:, k], alpha=0.6,
                             color=STATE_COLORS[k], label=f'S{k}({sl[k]})')
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(z)', fontsize=8)
        ax.legend(loc='upper right', fontsize=6, ncol=4)
        ax.set_title(f'{name} — V10 rSLDS (24D)', fontsize=10, fontweight='bold')

        # Features (top 6 most variable)
        ax = axes[1]
        var_order = np.argsort(np.var(Y, axis=0))[::-1]
        for rank, d in enumerate(var_order[:6]):
            ax.plot(t, Y[:, d] + rank * 3, linewidth=0.5, alpha=0.7,
                     label=V10_MODALITY_KEYS[d])
        ax.legend(loc='upper right', fontsize=6, ncol=3)
        ax.set_ylabel('Features (offset)', fontsize=8)

        # Transition covariates
        ax = axes[2]
        for ci, cov_name in enumerate(V10_COVARIATE_KEYS[2:]):  # skip z_slow PCs
            ax.plot(t, U[:, ci + 2], linewidth=0.8, alpha=0.8, label=cov_name)
        ax.legend(loc='upper right', fontsize=6, ncol=3)
        ax.set_ylabel('Covariates', fontsize=8)
        ax.set_xlabel('LSL time (s)', fontsize=8)

        # Condition shading
        for ax in axes:
            for seg_name, t0s, t1s in segments:
                ax.axvspan(t0s, t1s, alpha=0.15,
                           color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)

        plt.tight_layout()
        fig.savefig(f'results/v10/{name}/v10_rslds_timeline.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    total = time.time() - t_wall
    print(f"\n  V10 hierarchical complete: {total:.0f}s")
    print(f"  Results: {out_dir}/v10_hierarchical_results.json")


if __name__ == '__main__':
    main()
