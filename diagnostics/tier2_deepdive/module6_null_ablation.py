from __future__ import annotations
import os
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0 if np.linalg.norm(a - b) < 1e-12 else 0.0
    return float(np.dot(a, b) / denom)


def hungarian_match_states(
    d_constrained: np.ndarray,   # (K, D)
    d_unconstrained: np.ndarray, # (K, D)
) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian matching: for each constrained state i, find best unconstrained state.

    Returns (perm, sims): perm[i] = unconstrained index matched to constrained i,
    sims[i] = cosine similarity of matched pair.
    """
    K = d_constrained.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = 1.0 - _cosine_sim(d_constrained[i], d_unconstrained[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = col_ind[np.argsort(row_ind)]
    sims = np.array([_cosine_sim(d_constrained[i], d_unconstrained[perm[i]]) for i in range(K)])
    return perm, sims


def run_null_ablation(sessions: list, output_dir: str) -> dict:
    """Compare constrained (null_state=True) vs unconstrained rSLDS emission means.

    Attempts to re-fit with null_state=False. Falls back to constrained-only stats
    if re-fit fails (likely when cadence.significance.rslds_model is not available).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_d_constrained = []
    for sess in sessions:
        if sess.d_emit is not None:
            all_d_constrained.append(sess.d_emit)

    if not all_d_constrained:
        with open(os.path.join(output_dir, 'module6_report.md'), 'w') as f:
            f.write('# Module 6: Null-State Ablation\n\nNo rSLDS fits found.\n')
        pd.DataFrame().to_csv(os.path.join(output_dir, 'hungarian_alignment.csv'), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.savefig(os.path.join(output_dir, 'emission_means_comparison.png'), dpi=120)
        plt.close(fig)
        return {'constrained_null_norm': float('nan')}

    d_con_mean = np.mean(all_d_constrained, axis=0)  # (K, D)
    norms_con = np.linalg.norm(d_con_mean, axis=1)
    null_idx_con = int(np.argmin(norms_con))

    # ── Attempt unconstrained re-fit ─────────────────────────────────────
    d_uncon_mean = None
    perm = np.arange(len(d_con_mean))
    sims = np.ones(len(d_con_mean))
    refit_attempted = False

    try:
        from cadence.significance.rslds_model import IOHMM, IOHMMConfig
        cfg = IOHMMConfig(
            K=4, D_obs=sessions[0].Y_pw.shape[1],
            D_input=sessions[0].U.shape[1],
            D_latent=3, n_factors=2,
            recurrent=True, sticky_strength=3.0,
            c_shrinkage=0.2, null_state=False,
            viterbi_min_dwell=20, max_em_iter=100, n_restarts=2,
        )
        model = IOHMM(cfg)
        refit_attempted = True
        all_d_uncon = []
        for sess in sessions:
            print(f'  Re-fitting {sess.name} (unconstrained)...')
            params, _ = model.fit(sess.Y_pw, sess.U, sess.obs_mask)
            all_d_uncon.append(params.mu)
        d_uncon_mean = np.mean(all_d_uncon, axis=0)
        perm, sims = hungarian_match_states(d_con_mean, d_uncon_mean)
    except Exception as e:
        print(f'  Warning: unconstrained re-fit skipped ({e}). Reporting constrained-only stats.')

    # ── Emission norms plot ───────────────────────────────────────────────
    labels = sessions[0].state_labels or [f'S{k}' for k in range(4)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d_emit, title in [
        (axes[0], d_con_mean, 'Constrained (null_state=True)'),
        (axes[1], d_uncon_mean if d_uncon_mean is not None else d_con_mean,
         'Unconstrained (null_state=False)' if d_uncon_mean is not None else 'N/A (refit skipped)'),
    ]:
        norms = np.linalg.norm(d_emit, axis=1)
        sort_idx = np.argsort(norms)
        ax.barh([str(labels[i]) if i < len(labels) else f'S{i}' for i in sort_idx],
                norms[sort_idx], color='steelblue', alpha=0.8)
        ax.set_xlabel('Emission mean L2 norm')
        ax.set_title(title, fontsize=8)
    fig.suptitle('Emission mean norms: constrained vs unconstrained', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'emission_means_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Hungarian alignment table ─────────────────────────────────────────
    K = len(d_con_mean)
    align_df = pd.DataFrame({
        'constrained_state': [labels[i] if i < len(labels) else f'S{i}' for i in range(K)],
        'matched_unconstrained_state': [labels[perm[i]] if perm[i] < len(labels) else f'S{perm[i]}'
                                        for i in range(K)],
        'cosine_similarity': sims,
    })
    align_df.to_csv(os.path.join(output_dir, 'hungarian_alignment.csv'), index=False)

    # ── Interpretation ────────────────────────────────────────────────────
    min_sim = float(sims.min())
    null_norm_con = float(norms_con[null_idx_con])
    null_norm_uncon = float(np.linalg.norm(d_uncon_mean[perm[null_idx_con]])) \
        if d_uncon_mean is not None else float('nan')

    if min_sim > 0.7:
        interp = ('All states match with cosine similarity > 0.7. '
                  'Null-state constraint encodes a real regime — keep it.')
    elif np.isnan(min_sim) or not refit_attempted:
        interp = 'Re-fit not available — see constrained-only norms above.'
    else:
        interp = ('Low cosine similarity for some states. '
                  'Unconstrained model found different structure — null constraint may be forcing.')

    with open(os.path.join(output_dir, 'module6_report.md'), 'w') as f:
        f.write('# Module 6: Null-State Ablation\n\n')
        f.write(f'Constrained null-state norm: {null_norm_con:.3f}\n')
        if not np.isnan(null_norm_uncon):
            f.write(f'Unconstrained matched-state norm: {null_norm_uncon:.3f}\n')
        f.write(f'Min cosine similarity across matched pairs: {min_sim:.3f}\n\n')
        f.write(f'**Interpretation:** {interp}\n')

    return {
        'constrained_null_norm': null_norm_con,
        'unconstrained_null_norm': null_norm_uncon,
        'min_cosine_similarity': min_sim,
        'refit_attempted': refit_attempted,
    }
