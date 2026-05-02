"""Formalized dynamics summary for the MVP rSLDS production fit.

Quantifies temporal structure with three rigorous measures:

  (1) Empirical Markov transition matrix P(z_{t+1} = j | z_t = i, transition
      occurred), derived directly from the Viterbi paths. Stationary
      distribution + spectral gap + mixing time computed analytically.

  (2) Expected first-passage time matrix E[t : z_t = j | z_0 = i] in seconds,
      from the standard Markov chain calculation
      M_ii = 0;  M_ij = 1/fs + Σ_k P[i,k] M_kj  for i ≠ j.

  (3) Per-state A_dyn[k] eigenvalue spectrum (from the rSLDS itself), giving
      decay timescale τ = -1 / (fs · log|λ|) and oscillation period
      T = 2π / (fs · |arg(λ)|) for each within-state latent mode.

Plus the recurrent W_trans matrix from the rSLDS (which is the model's
internally-learned transition logits before per-timepoint covariate
modulation), and per-state covariate medians (showing what graph topology
each state is associated with).

Usage:
    python scripts/_plot_mvp_state_dynamics.py
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'
_SESS_NPZ = f'mvp_rslds_results{_SUFFIX}.npz'

STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',
    'OTHER':  '#9C27B0',
    'SHARED': '#2E7D32',
    'COUP':   '#FF6F00',
}


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


def collect_paths_and_covs():
    """Collect Viterbi paths (display-aligned) and covariates from canonical sessions."""
    with open(HIER / 'mvp_hierarchical_results.json') as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    canonical = load_canonical_set()
    sids = [s['session'] for s in d['sessions'] if s['session'] in canonical]
    fs = 2.0

    with open(REPO / 'results/mvp' / sids[0] / 'mvp_scaffold.json') as f:
        meta0 = json.load(f)
    cov_keys = meta0['covariate_channels']

    paths, covs = [], []
    for sid in sids:
        sd = REPO / 'results/mvp' / sid
        scaff = np.load(sd / 'mvp_scaffold.npz')
        rs = np.load(sd / _SESS_NPZ, allow_pickle=True)
        sl = list(rs['state_labels'])
        idx_map = {sl.index(lbl): k_disp for k_disp, lbl in enumerate(state_labels)}
        path_disp = np.array([idx_map[int(p)] for p in rs['path']], dtype=np.int8)
        paths.append(path_disp)
        covs.append(scaff['cov'])

    return state_labels, K, fs, cov_keys, sids, paths, covs


def empirical_transition_matrix(paths, K):
    """Compute P[i, j] = P(z_{t+1}=j | z_t=i) over ALL consecutive timepoints
    (NOT just transitions). Returns full row-stochastic K×K matrix."""
    counts = np.zeros((K, K), dtype=int)
    for path in paths:
        for t in range(len(path) - 1):
            counts[int(path[t]), int(path[t+1])] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, counts / np.maximum(row_sums, 1), 0)
    return P, counts


def stationary_distribution(P):
    """Find π s.t. π P = π, π·1 = 1 via the leading left eigenvector."""
    evals, evecs = np.linalg.eig(P.T)
    one_idx = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, one_idx])
    pi = pi / pi.sum()
    return pi


def spectral_gap_and_mixing(P, fs):
    """|λ_2| where λ_2 is the second-largest-magnitude eigenvalue.
    Mixing time ≈ 1/(fs · -log|λ_2|) seconds."""
    evals = np.linalg.eigvals(P)
    mods = np.sort(np.abs(evals))[::-1]
    lam2 = mods[1] if len(mods) > 1 else 0
    if lam2 > 0 and lam2 < 1:
        mix_time = -1 / (fs * np.log(lam2))
    else:
        mix_time = float('inf')
    return lam2, mix_time, evals


def first_passage_times(P, fs, K):
    """E[T : z_T = j | z_0 = i] in seconds. Solved per target column j:
       (I − P̃_j) m = 1/fs · 1, where P̃_j is P with column j zeroed and
       row j zeroed (absorbing at j). m[j] = 0 by construction."""
    M = np.zeros((K, K))
    for j in range(K):
        # Absorbing chain: kill state j by zeroing row j and column j in P
        P_abs = P.copy()
        P_abs[j, :] = 0
        P_abs[:, j] = 0
        # Linear system for non-absorbing states
        non_j = [i for i in range(K) if i != j]
        Q = P_abs[np.ix_(non_j, non_j)]
        try:
            ti = np.linalg.solve(np.eye(len(non_j)) - Q,
                                  np.ones(len(non_j)) / fs)
            for ii, i in enumerate(non_j):
                M[i, j] = ti[ii]
        except np.linalg.LinAlgError:
            for i in non_j:
                M[i, j] = float('nan')
    return M


def main():
    print('=== Loading ===')
    state_labels, K, fs, cov_keys, sids, paths, covs = collect_paths_and_covs()
    print(f'  N={len(sids)} sessions; K={K}; state_labels={state_labels}; covariates={cov_keys}')

    npz = np.load(HIER / 'mvp_hierarchical_params.npz', allow_pickle=True)
    A_dyn = npz['A_dyn']
    W_trans = npz['W_trans']
    S_trans = npz['S_trans']

    # ── Empirical Markov chain ──
    print('\n=== Empirical transition matrix (from Viterbi paths) ===')
    P_mat, counts = empirical_transition_matrix(paths, K)
    print(f'  total step pairs: {counts.sum()}')
    print(f'  transition matrix:')
    for i in range(K):
        row = '  '.join(f'{P_mat[i,j]:.3f}' for j in range(K))
        print(f'    {state_labels[i]:6s} → {row}')

    pi = stationary_distribution(P_mat)
    lam2, mix_time, evals = spectral_gap_and_mixing(P_mat, fs)
    print(f'\n  stationary distribution π = {dict(zip(state_labels, pi.round(3)))}')
    print(f'  spectral gap |λ_2| = {lam2:.4f}')
    print(f'  mixing time ≈ {mix_time:.1f}s')

    M_fp = first_passage_times(P_mat, fs, K)
    print(f'\n  expected first-passage times (s):')
    print(f'    from \\ to    {"  ".join(f"{l:>8s}" for l in state_labels)}')
    for i in range(K):
        cells = '  '.join(f'{M_fp[i,j]:>8.1f}' if M_fp[i,j] != 0 else f'{"-":>8s}'
                          for j in range(K))
        print(f'    {state_labels[i]:8s}    {cells}')

    # ── Per-state A_dyn eigenvalue signatures ──
    print(f'\n=== Per-state A_dyn[k] eigenvalue signatures ===')
    print(f'  state         max|λ|   modes (|λ|, decay τ, osc period)')
    for k in range(K):
        eigs = np.linalg.eigvals(A_dyn[k])
        mods = np.abs(eigs)
        angs = np.angle(eigs)
        modes = []
        for m, a in sorted(zip(mods, angs), key=lambda x: -x[0]):
            tau_s = '∞' if m >= 1 else f'{-1/(fs*np.log(m)):.0f}s'
            osc_s = f', osc {2*np.pi/(abs(a)*fs):.0f}s' if abs(a) > 1e-3 else ''
            modes.append(f'(|λ|={m:.3f}, τ={tau_s}{osc_s})')
        print(f'  {state_labels[k]:6s}      {mods.max():.4f}  {modes}')

    # ── Per-state covariate medians ──
    print(f'\n=== Per-state covariate medians (where each state lives in graph-theoretic space) ===')
    cov = np.concatenate(covs, axis=0)
    path = np.concatenate(paths, axis=0)
    print(f'  state         {"  ".join(f"{c:>22s}" for c in cov_keys)}')
    for k in range(K):
        m = path == k
        medians = [np.nanmedian(cov[m, j]) for j in range(len(cov_keys))]
        cells = '  '.join(f'{med:>22.3f}' for med in medians)
        print(f'  {state_labels[k]:6s}      {cells}')

    # ── PLOT ──
    print('\n=== Rendering ===')
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0],
                          width_ratios=[1.2, 1.2, 1.4, 1.4],
                          hspace=0.42, wspace=0.40,
                          left=0.06, right=0.98, top=0.92, bottom=0.07)

    state_colors = [STATE_COLOR_BY_LABEL.get(l, '#37474F') for l in state_labels]

    # Panel 1: Empirical transition matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P_mat, cmap='viridis', vmin=0, vmax=1, aspect='auto',
                   interpolation='nearest')
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{P_mat[i,j]:.2f}', ha='center', va='center',
                    color='white' if P_mat[i,j] < 0.5 else 'black',
                    fontsize=10, fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(state_labels, fontsize=9, rotation=20)
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel('to state', fontsize=9)
    ax.set_ylabel('from state', fontsize=9)
    ax.set_title('Empirical transition matrix\n'
                 'P(z_{t+1}=j | z_t=i) over consecutive timepoints',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Panel 2: Stationary distribution + mixing time
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(K), pi * 100, color=state_colors, alpha=0.75,
           edgecolor='#263238', linewidth=0.8)
    for i, p in enumerate(pi):
        ax.text(i, p * 100 + 1, f'{p*100:.1f}%', ha='center', fontsize=9,
                fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, fontsize=9, rotation=20)
    ax.set_ylabel('stationary probability (%)', fontsize=9)
    ax.set_ylim(0, max(pi) * 110)
    ax.set_title(f'Stationary distribution π\n'
                 f'spectral gap |λ_2|={lam2:.3f}; mixing time ≈ {mix_time:.0f}s',
                 fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linestyle=':')

    # Panel 3: First-passage time matrix
    ax = fig.add_subplot(gs[0, 2])
    M_disp = np.where(np.eye(K, dtype=bool), np.nan, M_fp)
    im = ax.imshow(M_disp, cmap='magma_r', aspect='auto', interpolation='nearest')
    for i in range(K):
        for j in range(K):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=10, color='#9E9E9E')
                continue
            v = M_fp[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '∞', ha='center', va='center', fontsize=12)
            else:
                ax.text(j, i, f'{v:.0f}s', ha='center', va='center',
                        fontsize=9,
                        color='white' if v > np.nanmedian(M_disp) else 'black',
                        fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(state_labels, fontsize=9, rotation=20)
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel('target state', fontsize=9)
    ax.set_ylabel('starting state', fontsize=9)
    ax.set_title('Expected first-passage time E[T : z_T=j | z_0=i]\n'
                 '(seconds — how long until you reach state j from i)',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='seconds')

    # Panel 4: Per-state A_dyn eigenvalue spectrum
    ax = fig.add_subplot(gs[0, 3])
    for k in range(K):
        eigs = np.linalg.eigvals(A_dyn[k])
        ax.scatter(eigs.real, eigs.imag, s=110, color=state_colors[k],
                   edgecolors='black', linewidths=1.0, alpha=0.85,
                   label=state_labels[k])
    # Unit circle for stability reference
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), 'k--', linewidth=0.8, alpha=0.5)
    ax.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax.set_xlabel('Re(λ)', fontsize=9)
    ax.set_ylabel('Im(λ)', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('A_dyn[k] eigenvalue spectrum\n'
                 '(within-state latent dynamics; unit circle = stability boundary)',
                 fontsize=10, fontweight='bold')

    # Panel 5: Per-state covariate boxplots
    ax = fig.add_subplot(gs[1, :2])
    n_cov = len(cov_keys)
    x_positions = []
    width = 0.18
    for j, ck in enumerate(cov_keys):
        for k in range(K):
            xpos = j + (k - K/2 + 0.5) * width
            vals = cov[path == k, j]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                bp = ax.boxplot([vals], positions=[xpos], widths=width*0.85,
                                vert=True, showfliers=False, patch_artist=True,
                                medianprops=dict(color='black', linewidth=1.2))
                for patch in bp['boxes']:
                    patch.set_facecolor(state_colors[k])
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('#263238')
            x_positions.append(xpos)
    ax.set_xticks(range(n_cov))
    ax.set_xticklabels([c.replace('_', ' ') for c in cov_keys], fontsize=10)
    ax.set_ylabel('z-scored covariate value', fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title('Per-state covariate distributions\n'
                 '(where each state lives in graph-theoretic space — '
                 'flexibility & algebraic connectivity λ₂)',
                 fontsize=10, fontweight='bold')
    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=state_colors[k], edgecolor='black',
                     label=state_labels[k]) for k in range(K)]
    ax.legend(handles=handles, fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.25, linestyle=':')

    # Panel 6: Recurrent W_trans matrix (model's internal transition logits)
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(W_trans, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(W_trans).max(), vmax=np.abs(W_trans).max(),
                   interpolation='nearest')
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{W_trans[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(W_trans[i,j]) > np.abs(W_trans).max()*0.5 else 'black',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(state_labels, fontsize=9, rotation=20)
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel('target', fontsize=9)
    ax.set_ylabel('source', fontsize=9)
    ax.set_title('Recurrent W_trans (rSLDS internal transition logits)\n'
                 'diag = self-stickiness; off-diag = coupling preferences',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Panel 7: Covariate sensitivity S_trans (max|S| per covariate per (i,j))
    ax = fig.add_subplot(gs[1, 3])
    # Reshape S_trans (K, K, n_cov) -> 2 small heatmaps stacked
    n_cov = S_trans.shape[2]
    smax = np.abs(S_trans).max()
    for c_idx, cn in enumerate(cov_keys):
        ax_local = ax if c_idx == 0 else None
        if ax_local is None:
            continue
    # Show as a single heatmap concatenating both covariates side-by-side
    big = np.concatenate([S_trans[:, :, j] for j in range(n_cov)], axis=1)
    im = ax.imshow(big, cmap='RdBu_r', aspect='auto', vmin=-smax, vmax=smax,
                   interpolation='nearest')
    for c_idx, cn in enumerate(cov_keys):
        for i in range(K):
            for j in range(K):
                xpos = c_idx * K + j
                v = S_trans[i, j, c_idx]
                ax.text(xpos, i, f'{v:+.2f}', ha='center', va='center',
                        color='white' if abs(v) > smax*0.5 else 'black',
                        fontsize=8, fontweight='bold')
        # Group label
        ax.text(c_idx * K + (K-1)/2, -0.7, cn.replace('_', ' '),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(K * n_cov))
    ax.set_xticklabels(state_labels * n_cov, fontsize=8, rotation=20)
    ax.set_yticks(range(K))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel('target', fontsize=9)
    ax.set_ylabel('source', fontsize=9)
    # Vertical separator between covariate blocks
    if n_cov > 1:
        for s in range(1, n_cov):
            ax.axvline(s * K - 0.5, color='black', linewidth=1.5)
    ax.set_title('S_trans (covariate sensitivity per source→target transition)\n'
                 'positive = covariate INCREASES P(transition i→j)',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(f'MVP rSLDS formalized state dynamics — N={len(sids)} canonical sessions',
                 fontsize=12, fontweight='bold')

    out = HIER / f'state_dynamics_{_VARIANT}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'Saved {out}')

    # Also write a text summary
    txt_out = HIER / f'state_dynamics_{_VARIANT}.txt'
    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write('MVP rSLDS formalized state dynamics summary\n')
        f.write('=' * 78 + '\n')
        f.write(f'N = {len(sids)} canonical sessions; K = {K}; fs = {fs} Hz\n')
        f.write(f'state labels: {state_labels}\n\n')
        f.write('EMPIRICAL TRANSITION MATRIX P(z_{t+1}=j | z_t=i):\n')
        f.write('  '.join(f'{l:>8s}' for l in [''] + state_labels) + '\n')
        for i in range(K):
            f.write(f'  {state_labels[i]:>6s}  ' +
                    '  '.join(f'{P_mat[i,j]:>8.3f}' for j in range(K)) + '\n')
        f.write(f'\nSTATIONARY DISTRIBUTION π = {dict(zip(state_labels, pi.round(3)))}\n')
        f.write(f'SPECTRAL GAP |λ_2| = {lam2:.4f}\n')
        f.write(f'MIXING TIME ≈ {mix_time:.1f} s\n\n')
        f.write('FIRST-PASSAGE TIMES (s):\n')
        f.write(f'  from \\ to   ' + '  '.join(f'{l:>8s}' for l in state_labels) + '\n')
        for i in range(K):
            f.write(f'  {state_labels[i]:8s}    ' +
                    '  '.join(f'{M_fp[i,j]:>8.1f}' if M_fp[i,j] > 0 else f'{"—":>8s}'
                              for j in range(K)) + '\n')
        f.write('\nPER-STATE A_dyn[k] EIGENVALUES (decay τ in seconds):\n')
        for k in range(K):
            eigs = np.linalg.eigvals(A_dyn[k])
            f.write(f'  {state_labels[k]:6s}: ')
            for e in sorted(eigs, key=lambda x: -abs(x)):
                m, a = abs(e), np.angle(e)
                tau = '∞' if m >= 1 else f'{-1/(fs*np.log(m)):.0f}s'
                osc = f', osc {2*np.pi/(abs(a)*fs):.0f}s' if abs(a) > 1e-3 else ''
                f.write(f'  (|λ|={m:.3f}, τ={tau}{osc})')
            f.write('\n')
        f.write('\nPER-STATE COVARIATE MEDIANS (z-scored):\n')
        f.write('  state    ' + '  '.join(f'{c:>22s}' for c in cov_keys) + '\n')
        for k in range(K):
            m = path == k
            medians = [float(np.nanmedian(cov[m, j])) for j in range(len(cov_keys))]
            f.write(f'  {state_labels[k]:6s}   ' +
                    '  '.join(f'{med:>22.3f}' for med in medians) + '\n')
    print(f'Saved {txt_out}')


if __name__ == '__main__':
    main()
