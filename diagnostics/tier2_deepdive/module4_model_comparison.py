from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


def _ll_per_frame_dim(model, X: np.ndarray) -> float:
    """Held-out log-likelihood per frame per dimension."""
    try:
        if hasattr(model, 'score'):
            ll_per_frame = model.score(X)        # already per frame for hmmlearn
            return float(ll_per_frame / X.shape[1])
        else:
            return float(model.score(X) / X.shape[1])
    except Exception:
        return float('nan')


def run_model_comparison(sessions: list, output_dir: str) -> dict:
    """LOO-CV comparison: HMM K=2..5, GMM K=2..5, reference rSLDS K=4."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(sessions)
    rows = []
    K_values = [2, 3, 4, 5]

    for fold, test_sess in enumerate(sessions):
        train_sessions = [s for i, s in enumerate(sessions) if i != fold]
        if not train_sessions:
            train_sessions = [test_sess]   # n=1 edge case for tests

        Y_train = np.vstack([s.Y_pw for s in train_sessions])
        Y_test = test_sess.Y_pw
        T_test, D = Y_test.shape

        row = {'fold': fold, 'test_session': test_sess.name}

        for K in K_values:
            # ── Gaussian HMM ─────────────────────────────────────────────
            hmm = GaussianHMM(n_components=K, covariance_type='diag',
                              n_iter=100, random_state=42)
            try:
                hmm.fit(Y_train)
                row[f'HMM_K{K}'] = _ll_per_frame_dim(hmm, Y_test)
            except Exception:
                row[f'HMM_K{K}'] = float('nan')

            # ── GMM ──────────────────────────────────────────────────────
            gmm = GaussianMixture(n_components=K, covariance_type='diag',
                                  random_state=42, max_iter=200)
            try:
                gmm.fit(Y_train)
                row[f'GMM_K{K}'] = float(gmm.score(Y_test) / D)
            except Exception:
                row[f'GMM_K{K}'] = float('nan')

        # ── rSLDS reference (emission LL conditioned on MAP state path) ──
        if (test_sess.d_emit is not None and test_sess.path_unconstrained is not None):
            sigma2 = np.ones_like(test_sess.d_emit)
            if test_sess.R_emit is not None:
                R = test_sess.R_emit
                if R.ndim == 3:
                    sigma2 = np.array([np.diag(R[k]) for k in range(R.shape[0])])
                else:
                    sigma2 = np.abs(R) + 1e-6

            path = test_sess.path_unconstrained
            d_emit = test_sess.d_emit  # (K, D)
            ll = 0.0
            for t in range(T_test):
                k = int(path[t])
                if k >= len(d_emit):
                    continue
                diff = Y_test[t] - d_emit[k]
                ll += float(np.sum(-0.5 * diff**2 / sigma2[k] - 0.5 * np.log(sigma2[k] + 1e-9)))
            row['rSLDS_K4'] = ll / (T_test * D)
        else:
            row['rSLDS_K4'] = float('nan')

        rows.append(row)
        print(f'  Fold {fold+1}/{n}: {test_sess.name} done')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'loo_cv_results.csv'), index=False)

    # ── Summary bar chart ────────────────────────────────────────────────
    model_cols = [c for c in df.columns if c not in ('fold', 'test_session')]
    means = df[model_cols].mean()
    sems = df[model_cols].sem()

    fig, ax = plt.subplots(figsize=(max(8, len(model_cols) * 0.6), 5))
    x = np.arange(len(model_cols))
    n_hmm = len(K_values)
    n_gmm = len(K_values)
    colors = (['steelblue'] * n_hmm + ['firebrick'] * n_gmm + ['gold'])
    ax.bar(x, means.values, yerr=sems.values, capsize=3,
           color=colors[:len(model_cols)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_cols, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean held-out LL / frame / dim (±SEM)')
    ax.set_title('LOO-CV model comparison (higher = better)')
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison_summary.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Interpretation ───────────────────────────────────────────────────
    rslds_mean = means.get('rSLDS_K4', float('nan'))
    hmm4_mean = means.get('HMM_K4', float('nan'))
    hmm4_sem = sems.get('HMM_K4', 0)
    gmm4_mean = means.get('GMM_K4', float('nan'))

    interp_lines = []
    if not np.isnan(rslds_mean) and not np.isnan(hmm4_mean):
        if rslds_mean - hmm4_mean > hmm4_sem:
            interp_lines.append('rSLDS > HMM K=4 by >1 SEM — recurrent covariates paying off.')
        else:
            interp_lines.append('rSLDS within 1 SEM of HMM K=4 — covariate structure not justified at n.')
    if not np.isnan(gmm4_mean) and not np.isnan(hmm4_mean):
        if abs(gmm4_mean - hmm4_mean) < sems.get('GMM_K4', 0) + hmm4_sem:
            interp_lines.append('GMM K=4 competitive with HMM K=4 — temporal dynamics not contributing.')

    with open(os.path.join(output_dir, 'module4_report.md'), 'w') as f:
        f.write('# Module 4: Alternative-Model Baselines\n\n')
        f.write(means.to_string())
        f.write('\n\n**Interpretation:**\n')
        f.write('\n'.join(interp_lines) or 'Insufficient data for interpretation.')
        f.write('\n')

    return {'mean_ll_per_frame_dim': means.to_dict()}
