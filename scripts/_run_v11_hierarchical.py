"""V11 Hierarchical rSLDS: 26D features, 7D covariates, shared dynamics.

Extends V10 hierarchical with:
  - 26D observation vector (21 V10 after dyn collapse + 2 TE concordance + 3 burst coincidence)
  - 7D transition covariates (5 V10 + 2 TE asymmetry as transition modulators)
  - c_shrinkage=0.2 (reduced from 0.3 to let new channels express)

Usage:
    python scripts/_run_v11_hierarchical.py
"""

import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch must import before numpy on Windows (torch 2.10 + numpy 2.4 DLL-load order bug: shm.dll)
import torch  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import parallel_backend
from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds
from cadence.constants import (
    V11_MODALITY_KEYS, V11_MODALITY_NAMES,
    V11_COVARIATE_KEYS,
)

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


def _parse_bool(s):
    return str(s).lower() in ('true', '1', 'yes', 't', 'y')


def _label_states(mean_d, modality_keys, K, null_state):
    """Assign human-readable labels to K rSLDS states based on emission profiles.

    If null_state=True, state 0 is constrained to zero — label it NULL directly.
    If null_state=False, pick the lowest-norm state as the "NULL-analogue" / baseline.

    Then: COUP = highest imcoh_{theta,alpha,beta} sum; SHARED = highest
    conc_{theta,alpha,beta} sum among remaining; OTHER fills any leftovers.
    Works for K=3 (no OTHER) and K=4 (one OTHER).
    """
    labels = [''] * K

    if null_state:
        labels[0] = 'NULL'
        avail = list(range(1, K))
    else:
        norms = np.linalg.norm(mean_d, axis=1)
        null_k = int(np.argmin(norms))
        labels[null_k] = 'NULL'
        avail = [k for k in range(K) if k != null_k]

    imcoh_idx = [modality_keys.index(f'imcoh_{b}') for b in ['theta', 'alpha', 'beta']]
    conc_idx = [modality_keys.index(f'conc_{b}') for b in ['theta', 'alpha', 'beta']]

    imcoh_sums = {k: float(mean_d[k, imcoh_idx].sum()) for k in avail}
    coup_k = max(imcoh_sums, key=imcoh_sums.get)
    labels[coup_k] = 'COUP'
    avail = [k for k in avail if k != coup_k]

    if avail:
        conc_sums = {k: float(mean_d[k, conc_idx].sum()) for k in avail}
        shared_k = max(conc_sums, key=conc_sums.get)
        labels[shared_k] = 'SHARED'
        avail = [k for k in avail if k != shared_k]

    for k in avail:
        labels[k] = 'OTHER'

    return labels


def main():
    parser = argparse.ArgumentParser(
        description='V11 hierarchical rSLDS fit (parameterizable for refit variants).')
    parser.add_argument('--K', type=int, default=4,
                        help='Number of discrete states (default: 4)')
    parser.add_argument('--null-state', type=_parse_bool, default=True,
                        help='Constrain state 0 to zero emissions (default: true)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix on output paths — empty string overwrites the '
                             'production fit, so use a tag like "_k4_nonull" for variants.')
    parser.add_argument('--n-restarts', type=int, default=2)
    parser.add_argument('--max-em-iter', type=int, default=80)
    args = parser.parse_args()

    K = args.K
    null_state = args.null_state
    suffix = args.suffix

    print("=" * 80)
    print(f"  V11 Hierarchical rSLDS  (K={K}, null_state={null_state}, suffix='{suffix}')")
    print("=" * 80)
    t_wall = time.time()

    # Load all V11 sessions
    npzs = sorted(glob.glob('results/v11/*/scaffold_v11_ztimecourses.npz'))
    if not npzs:
        print("  No V11 scaffold results found. Run _run_scaffold_v11.py --all first.")
        return

    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']

        # Load 26D observations
        Y = np.column_stack([data[f'z_{k}'] for k in V11_MODALITY_KEYS]).astype(np.float64)
        obs_mask = data['obs_mask'] if 'obs_mask' in data else np.ones_like(Y, dtype=bool)

        # Load 7D transition covariates
        if 'U_covariates' in data:
            U = data['U_covariates'].astype(np.float64)
        else:
            U = np.column_stack([data[f'u_{k}'] for k in V11_COVARIATE_KEYS]).astype(np.float64)

        # Segments
        json_path = f'results/v11/{name}/scaffold_v11_results.json'
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

    # V11 config: 26D obs, 7D covariates, recurrent, reduced c_shrinkage
    D_obs = len(V11_MODALITY_KEYS)    # 26
    D_input = len(V11_COVARIATE_KEYS)  # 7
    cfg = IOHMMConfig(
        K=K,
        D_obs=D_obs,
        D_input=D_input,
        D_latent=3,
        n_factors=2,
        recurrent=True,
        c_shrinkage=0.2,
        n_restarts=args.n_restarts,
        max_em_iter=args.max_em_iter,
        sticky_strength=3.0,
        null_state=null_state,
        null_sigma2_cap=5.0,
    )

    session_tuples = [(Y, U, mask) for Y, U, mask, t, name, segs in sessions]
    print(f"\n  Fitting hierarchical rSLDS "
          f"(K={cfg.K}, D_latent={cfg.D_latent}, D_obs={cfg.D_obs}, "
          f"D_input={cfg.D_input}, recurrent={cfg.recurrent}, "
          f"c_shrinkage={cfg.c_shrinkage})...")

    # Force threading backend: loky worker spawn triggers a torch 2.10 + numpy 2.4
    # DLL-load bug on Windows (numpy auto-loads in worker before user code can
    # hoist torch). Threads inherit the parent's already-loaded torch.
    with parallel_backend('threading'):
        result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)
    print(f"  BIC: {result['bic']:.0f}")
    print(f"  Final LL: {result['final_ll']:.0f}")

    # Shared d_emit profiles
    all_d = [result['sessions'][i].get('d_emit') for i in range(len(sessions))]
    mean_d = np.mean(all_d, axis=0)

    # State labels from shared d_emit, K-aware and null_state-aware
    sl = _label_states(mean_d, list(V11_MODALITY_KEYS), K, null_state)

    print(f"\n  State labels: {sl}")

    # Print shared d_emit for all 26 channels
    print(f"\n  Shared d_emit (26D):")
    for k in range(K):
        top_features = np.argsort(np.abs(mean_d[k]))[::-1][:8]
        top_str = ', '.join(f'{V11_MODALITY_KEYS[d]}={mean_d[k, d]:+.2f}' for d in top_features)
        print(f"    S{k}({sl[k]:>6s}): {top_str}")

    # Check transition covariate effects
    if 'S_trans' in result.get('shared', {}):
        S = result['shared']['S_trans']  # (K, K, D_input)
        print(f"\n  Transition covariate effects (S_trans):")
        for ci, cov_name in enumerate(V11_COVARIATE_KEYS):
            S_slice = S[:, :, ci]
            max_effect = np.abs(S_slice).max()
            print(f"    {cov_name}: max|S|={max_effect:.3f}")

    # Save results — suffix lets variant fits coexist with the production fit
    out_dir = f'results/v11/hierarchical{suffix}'
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        'version': 'v11',
        'variant': suffix.lstrip('_') or 'production',
        'n_sessions': len(sessions),
        'config': {
            'K': cfg.K, 'D_obs': cfg.D_obs, 'D_input': cfg.D_input,
            'D_latent': cfg.D_latent, 'recurrent': cfg.recurrent,
            'c_shrinkage': cfg.c_shrinkage, 'n_factors': cfg.n_factors,
            'null_state': cfg.null_state,
        },
        'bic': float(result['bic']),
        'final_ll': float(result['final_ll']),
        'state_labels': sl,
        'modality_keys': list(V11_MODALITY_KEYS),
        'covariate_keys': list(V11_COVARIATE_KEYS),
        # Mean emission profile across sessions (K x D_obs) — what each state "looks like"
        'mean_d_emit': mean_d.tolist(),
    }
    # Transition covariate effects (K x K x D_input)
    if 'S_trans' in result.get('shared', {}):
        summary['S_trans'] = result['shared']['S_trans'].tolist()
    # Also save full mean_d_emit + shared params as NPZ for numerical access
    np.savez_compressed(
        os.path.join(out_dir, 'v11_hierarchical_params.npz'),
        mean_d_emit=mean_d,
        state_labels=np.array(sl, dtype=object),
        modality_keys=np.array(list(V11_MODALITY_KEYS), dtype=object),
        covariate_keys=np.array(list(V11_COVARIATE_KEYS), dtype=object),
        W_trans=result.get('shared', {}).get('W_trans', np.zeros(1)),
        S_trans=result.get('shared', {}).get('S_trans', np.zeros(1)),
        A_dyn=result.get('shared', {}).get('A_dyn', np.zeros(1)),
        Q_dyn=result.get('shared', {}).get('Q_dyn', np.zeros(1)),
    )

    # Per-session analysis
    all_session_results = []
    shared_W_trans = result.get('shared', {}).get('W_trans')
    shared_S_trans = result.get('shared', {}).get('S_trans')
    shared_A_dyn = result.get('shared', {}).get('A_dyn')
    shared_Q_dyn = result.get('shared', {}).get('Q_dyn')
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
            usage = [float((path[pmask] == k).mean()) for k in range(K)]
            period_data.append((pname, float(t1 - t0), usage))

        n_trans = int(np.sum(path[1:] != path[:-1]))
        usage = [float((path == k).mean()) for k in range(K)]

        all_session_results.append({
            'session': name, 'state_labels': sl,
            'usage': usage, 'n_transitions': n_trans,
            'periods': period_data,
        })

        # Save per-session npz with state assignments + emission profile
        sess_out = f'results/v11/{name}'
        os.makedirs(sess_out, exist_ok=True)
        sess_result = result['sessions'][i]
        save_kw = dict(gamma=gamma, path=path, t_common=t,
                       state_labels=np.array(sl, dtype=object))
        # Per-session emission parameters (K x D_obs)
        if sess_result.get('d_emit') is not None:
            save_kw['d_emit'] = sess_result['d_emit']
        if sess_result.get('C_emit') is not None:
            save_kw['C_emit'] = sess_result['C_emit']
        if sess_result.get('R_emit') is not None:
            save_kw['R_emit'] = sess_result['R_emit']
        # Latent trajectory (T x D_latent) — needed for latent-space silhouette in diagnostics
        if sess_result.get('x_smooth') is not None:
            save_kw['x_smooth'] = sess_result['x_smooth'].astype(np.float32)
        # Shared transition / dynamics parameters (same across sessions)
        if shared_W_trans is not None:
            save_kw['W_trans'] = shared_W_trans
        if shared_S_trans is not None:
            save_kw['S_trans'] = shared_S_trans
        if shared_A_dyn is not None:
            save_kw['A_dyn'] = shared_A_dyn
        if shared_Q_dyn is not None:
            save_kw['Q_dyn'] = shared_Q_dyn
        np.savez_compressed(os.path.join(sess_out, f'v11_rslds_results{suffix}.npz'),
                            **save_kw)

    summary['sessions'] = all_session_results

    with open(os.path.join(out_dir, 'v11_hierarchical_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Timeline plots for each session
    for i, (Y, U, mask, t, name, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path = cv(gamma, md=20)

        fig, axes = plt.subplots(3, 1, figsize=(24, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 3, 1]})

        # State posterior
        ax = axes[0]
        for k in range(K):
            ax.fill_between(t, 0, gamma[:, k], alpha=0.6,
                             color=STATE_COLORS[k % len(STATE_COLORS)],
                             label=f'S{k}({sl[k]})')
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(z)', fontsize=8)
        ax.legend(loc='upper right', fontsize=6, ncol=4)
        ax.set_title(f'{name} -- V11 rSLDS (26D)', fontsize=10, fontweight='bold')

        # Features (top 8 most variable)
        ax = axes[1]
        var_order = np.argsort(np.var(Y, axis=0))[::-1]
        for rank, d in enumerate(var_order[:8]):
            ax.plot(t, Y[:, d] + rank * 3, linewidth=0.5, alpha=0.7,
                     label=V11_MODALITY_KEYS[d])
        ax.legend(loc='upper right', fontsize=6, ncol=4)
        ax.set_ylabel('Features (offset)', fontsize=8)

        # Transition covariates
        ax = axes[2]
        for ci, cov_name in enumerate(V11_COVARIATE_KEYS[2:]):
            ax.plot(t, U[:, ci + 2], linewidth=0.8, alpha=0.8, label=cov_name)
        ax.legend(loc='upper right', fontsize=6, ncol=5)
        ax.set_ylabel('Covariates', fontsize=8)
        ax.set_xlabel('LSL time (s)', fontsize=8)

        # Condition shading
        for ax in axes:
            for seg_name, t0s, t1s in segments:
                ax.axvspan(t0s, t1s, alpha=0.15,
                           color=CONDITION_COLORS.get(seg_name, '#F5F5F5'), zorder=0)

        plt.tight_layout()
        fig.savefig(f'results/v11/{name}/v11_rslds_timeline{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    total = time.time() - t_wall
    print(f"\n  V11 hierarchical complete: {total:.0f}s")
    print(f"  Results: {out_dir}/v11_hierarchical_results.json")


if __name__ == '__main__':
    main()
