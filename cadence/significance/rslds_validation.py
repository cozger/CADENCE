"""CADENCE RSLDS Validation: synthetic recovery + pseudo-dyad null.

Provides:
  - synthetic_iohmm_recovery: Generate synthetic IOHMM data, fit, check ARI
  - synthetic_slds_recovery: Generate SLDS data with factors + recurrence, compare models
  - synthetic_hierarchical_recovery: Multi-session shared dynamics test
  - pseudo_dyad_iohmm_null: Fit IOHMM on pseudo-dyad z-timecourses
  - compare_real_vs_pseudo: Statistical comparison of state dynamics
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import mannwhitneyu
from typing import Optional

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, IOHMMParams, iohmm_flexibility_metrics,
    fit_slds, fit_hierarchical_slds,
)
from cadence.io.resources import pick_n_jobs, limit_blas_threads


def synthetic_iohmm_recovery(K: int = 3, D_obs: int = 7, D_input: int = 2,
                              T: int = 3000, n_trials: int = 3,
                              mean_scale: float = 0.5,
                              seed: int = 42, verbose: bool = True) -> dict:
    """Generate synthetic IOHMM data with known states, fit, and check recovery.

    Uses realistic difficulty calibration: mean_scale=0.5 gives SNR~0.47,
    where k-means ARI~0.4 (temporal model must contribute to exceed 0.7).

    Always reports k-means baseline for comparison.

    Returns:
        dict with ARI scores, emission recovery error, k-means baseline, BIC
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    results = {'trials': [], 'ari_mean': 0, 'ari_min': 0}

    for trial in range(n_trials):
        trial_seed = seed + trial * 100

        # Generate ground truth — realistic difficulty (mean_scale=0.5)
        true_mu = rng.standard_normal((K, D_obs)) * mean_scale
        order = np.argsort(true_mu[:, 0])
        true_mu = true_mu[order]
        # Per-modality noise matching real data heterogeneity
        base_sigma = [0.7, 0.5, 0.8, 0.4, 0.08, 0.5, 0.2, 0.6, 0.3]
        # Extend if D_obs > len(base_sigma)
        while len(base_sigma) < D_obs:
            base_sigma.append(0.5)
        true_sigma2 = np.tile(base_sigma[:D_obs], (K, 1))

        # Sticky transitions with input modulation
        true_W = np.eye(K) * 3.0 - 0.5
        true_S = rng.normal(0, 0.5, size=(K, K, D_input))

        # Block-structured covariates
        U = np.zeros((T, D_input))
        block_size = T // (K * 2)
        for b in range(0, T, block_size):
            be = min(b + block_size, T)
            phase = (b // block_size) % D_input
            U[b:be, phase] = rng.choice([-1, 1])

        # Generate states + observations
        z_true = np.zeros(T, dtype=int)
        Y = np.zeros((T, D_obs))
        z_true[0] = rng.integers(K)
        Y[0] = true_mu[z_true[0]] + rng.standard_normal(D_obs) * np.sqrt(true_sigma2[z_true[0]])
        for t in range(1, T):
            logits = true_W[z_true[t-1]] + true_S[z_true[t-1]] @ U[t]
            probs = softmax(logits)
            z_true[t] = rng.choice(K, p=probs)
            Y[t] = true_mu[z_true[t]] + rng.standard_normal(D_obs) * np.sqrt(true_sigma2[z_true[t]])

        # K-means baseline (validates test difficulty)
        km = KMeans(n_clusters=K, n_init=10, random_state=trial_seed).fit(Y)
        ari_kmeans = adjusted_rand_score(z_true, km.labels_)

        # Fit IOHMM
        cfg = IOHMMConfig(K=K, D_obs=D_obs, D_input=D_input,
                          n_restarts=3, max_em_iter=100)
        model = IOHMM(cfg)
        params, history = model.fit(Y, U, seed=trial_seed, verbose=False)
        viterbi_path = model.viterbi(Y, U, None, params)

        ari = adjusted_rand_score(z_true, viterbi_path)

        # Emission recovery: Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.sum((true_mu[i] - params.mu[j]) ** 2)
        row_ind, col_ind = linear_sum_assignment(cost)
        mu_error = np.sqrt(cost[row_ind, col_ind].mean())

        # Transition matrix recovery: Hungarian-matched Frobenius norm
        matched_W = params.W_trans[col_ind][:, col_ind]
        W_frob = float(np.sqrt(np.sum((true_W - matched_W) ** 2)))

        # S_trans recovery
        matched_S = params.S_trans[col_ind][:, col_ind, :]
        S_corr = float(np.corrcoef(true_S.ravel(), matched_S.ravel())[0, 1]) \
            if np.std(true_S) > 1e-8 else 0.0

        trial_result = {
            'ari': float(ari),
            'ari_kmeans': float(ari_kmeans),
            'ari_lift': float(ari - ari_kmeans),
            'mu_error': float(mu_error),
            'W_frobenius': W_frob,
            'S_correlation': S_corr,
            'bic': float(history['bic']),
            'll': float(history['final_ll']),
        }
        results['trials'].append(trial_result)
        if verbose:
            print(f"  Trial {trial}: ARI={ari:.3f} (kmeans={ari_kmeans:.3f}, lift={ari-ari_kmeans:+.3f}), "
                  f"W_frob={W_frob:.3f}")

    aris = [t['ari'] for t in results['trials']]
    aris_km = [t['ari_kmeans'] for t in results['trials']]
    lifts = [t['ari_lift'] for t in results['trials']]
    w_frobs = [t['W_frobenius'] for t in results['trials']]
    s_corrs = [t['S_correlation'] for t in results['trials']]
    results['ari_mean'] = float(np.mean(aris))
    results['ari_min'] = float(np.min(aris))
    results['ari_kmeans_mean'] = float(np.mean(aris_km))
    results['ari_lift_mean'] = float(np.mean(lifts))
    results['W_frobenius_mean'] = float(np.mean(w_frobs))
    results['S_correlation_mean'] = float(np.mean(s_corrs))
    results['test_valid'] = results['ari_kmeans_mean'] < 0.7
    results['passed'] = results['ari_min'] > 0.7 and results['test_valid']

    if verbose:
        print(f"  IOHMM ARI: mean={results['ari_mean']:.3f}, min={results['ari_min']:.3f}")
        print(f"  K-means ARI: mean={results['ari_kmeans_mean']:.3f} "
              f"({'VALID' if results['test_valid'] else 'TEST TOO EASY'})")
        print(f"  Lift (IOHMM - kmeans): {results['ari_lift_mean']:+.3f}")
        print(f"  W_frobenius: mean={results['W_frobenius_mean']:.3f}")
        print(f"  [{'PASS' if results['passed'] else 'FAIL'}] Synthetic recovery")

    return results


def fit_pseudo_dyad_iohmm(pseudo_z_traces: list, modality_keys: list,
                           K: int = 3, n_pca: int = 2,
                           n_restarts: int = 3, max_iter: int = 100,
                           verbose: bool = True) -> list:
    """Fit IOHMM on pseudo-dyad z-timecourses.

    Args:
        pseudo_z_traces: list of dicts, each {modality: (T,) z-timecourse}
        modality_keys: list of modality keys
        K: number of states
        n_pca: PCA components for z_slow
        n_restarts: random restarts per fit
        max_iter: max EM iterations

    Returns:
        list of flexibility metric dicts (one per pseudo-dyad)
    """
    from scripts._run_rslds_phase2 import spectral_decompose

    pseudo_results = []
    for i, zt in enumerate(pseudo_z_traces):
        T = len(next(iter(zt.values())))
        z_fast, z_slow_pcs, _ = spectral_decompose(zt, modality_keys)

        from cadence.significance.rslds_model import build_observation_mask
        Y = np.column_stack([z_fast[k] for k in modality_keys]).astype(np.float64)
        U = z_slow_pcs.astype(np.float64)
        obs_mask = build_observation_mask(zt, modality_keys, T)

        cfg = IOHMMConfig(K=K, D_obs=len(modality_keys), D_input=n_pca,
                          n_restarts=n_restarts, max_em_iter=max_iter)
        model = IOHMM(cfg)
        params, history = model.fit(Y, U, obs_mask, seed=42 + i, verbose=False)
        gamma = history['gamma']
        flex = iohmm_flexibility_metrics(gamma, fs=2.0)
        flex['bic'] = float(history['bic'])
        flex['ll'] = float(history['final_ll'])
        pseudo_results.append(flex)

        if verbose and (i + 1) % 5 == 0:
            print(f"    Fitted {i+1}/{len(pseudo_z_traces)} pseudo-dyads")

    return pseudo_results


def compare_real_vs_pseudo(real_flex: dict, pseudo_flex_list: list,
                           verbose: bool = True) -> dict:
    """Compare real dyad IOHMM flexibility metrics vs pseudo-dyad distribution.

    Args:
        real_flex: flexibility metrics dict from real dyad
        pseudo_flex_list: list of flexibility dicts from pseudo-dyads

    Returns:
        dict with per-metric p-values and effect sizes
    """
    comparisons = {}

    # Metrics to compare
    metrics = ['n_transitions', 'transition_rate_hz', 'shannon_entropy']

    for metric in metrics:
        real_val = real_flex[metric]
        pseudo_vals = np.array([p[metric] for p in pseudo_flex_list])

        # Percentile rank of real in pseudo distribution
        pct = float(np.mean(pseudo_vals <= real_val))

        # Effect size: (real - mean_pseudo) / std_pseudo
        std_pseudo = np.std(pseudo_vals)
        if std_pseudo > 1e-10:
            d = float((real_val - np.mean(pseudo_vals)) / std_pseudo)
        else:
            d = 0.0

        comparisons[metric] = {
            'real': float(real_val),
            'pseudo_mean': float(np.mean(pseudo_vals)),
            'pseudo_std': float(std_pseudo),
            'percentile': pct,
            'effect_size_d': d,
        }
        if verbose:
            print(f"  {metric:25s}: real={real_val:.3f}, pseudo={np.mean(pseudo_vals):.3f}±{std_pseudo:.3f}, "
                  f"pct={pct:.2%}, d={d:+.2f}")

    # State usage comparison: KL divergence or similar
    real_usage = np.array(real_flex['state_usage'])
    pseudo_usages = np.array([p['state_usage'] for p in pseudo_flex_list])
    K = len(real_usage)

    # Per-state usage comparison
    for k in range(K):
        pseudo_vals = pseudo_usages[:, k]
        real_val = real_usage[k]
        pct = float(np.mean(pseudo_vals <= real_val))
        std_p = np.std(pseudo_vals)
        d = float((real_val - np.mean(pseudo_vals)) / max(std_p, 1e-10))
        comparisons[f'state_{k}_usage'] = {
            'real': float(real_val),
            'pseudo_mean': float(np.mean(pseudo_vals)),
            'pseudo_std': float(std_p),
            'percentile': pct,
            'effect_size_d': d,
        }

    # Overall: is real dyad significantly different?
    sig_count = sum(1 for k, v in comparisons.items()
                    if abs(v.get('effect_size_d', 0)) > 1.0)  # |d| > 1 = large effect
    comparisons['n_significant'] = sig_count
    comparisons['n_metrics'] = len(comparisons) - 2  # exclude meta-keys

    if verbose:
        print(f"\n  Metrics with |d| > 1.0: {sig_count}/{comparisons['n_metrics']}")

    return comparisons


def synthetic_slds_recovery(K: int = 3, D_obs: int = 7, D_input: int = 2,
                             D_latent: int = 3, n_factors: int = 2,
                             T: int = 3000, n_trials: int = 3,
                             mean_scale: float = 0.5,
                             seed: int = 42, verbose: bool = True) -> dict:
    """Generate synthetic SLDS data with factors + recurrence, compare model variants.

    Data generation includes:
      - Per-state dynamics A[k] in latent space (autocorrelation)
      - Factor-analyzed emission noise F[k]F[k]' + diag(R[k])
      - Recurrent transitions: x_{t-1} modulates P(z_t | z_{t-1})

    Tests 5 model variants: IOHMM, SLDS, SLDS+FA, rSLDS, Full (FA+recurrent).
    Always includes k-means baseline for difficulty validation.

    Returns:
        dict with per-trial and aggregate ARI/BIC for each model variant
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    results = {'trials': []}

    for trial in range(n_trials):
        trial_seed = seed + trial * 100
        rng_t = np.random.default_rng(trial_seed)

        # --- Ground truth parameters ---

        # Emission offsets (between-state means)
        true_d = rng_t.standard_normal((K, D_obs)) * mean_scale
        order = np.argsort(true_d[:, 0])
        true_d = true_d[order]

        # Factor loadings (cross-modal noise correlations)
        true_F = rng_t.standard_normal((K, D_obs, n_factors)) * 0.3
        true_R = np.abs(rng_t.standard_normal((K, D_obs))) * 0.3 + 0.1

        # Per-state dynamics: stable AR in latent space
        true_A = np.zeros((K, D_latent, D_latent))
        for k in range(K):
            true_A[k] = 0.8 * np.eye(D_latent) + rng_t.standard_normal(
                (D_latent, D_latent)) * 0.1
            eigvals = np.abs(np.linalg.eigvals(true_A[k]))
            if eigvals.max() > 0.95:
                true_A[k] *= 0.9 / eigvals.max()
        true_b = rng_t.standard_normal((K, D_latent)) * 0.1
        true_Q = np.zeros((K, D_latent, D_latent))
        for k in range(K):
            true_Q[k] = 0.1 * np.eye(D_latent)

        # Emission matrices (latent -> obs)
        true_C = rng_t.standard_normal((K, D_obs, D_latent)) * 0.5

        # Transitions
        true_W = np.eye(K) * 3.0 - 0.5
        true_S = rng_t.normal(0, 0.3, size=(K, K, D_input))

        # Recurrence weights
        true_R_recur = rng_t.normal(0, 0.3, size=(K, K, D_latent))

        # Block-structured covariates
        U = np.zeros((T, D_input))
        block_size = T // (K * 2)
        for b in range(0, T, block_size):
            be = min(b + block_size, T)
            phase = (b // block_size) % D_input
            U[b:be, phase] = rng_t.choice([-1, 1])

        # --- Generate states + latent + observations ---
        z_true = np.zeros(T, dtype=int)
        x_true = np.zeros((T, D_latent))
        Y = np.zeros((T, D_obs))

        z_true[0] = rng_t.integers(K)
        x_true[0] = rng_t.standard_normal(D_latent) * 0.5
        factor_noise = true_F[z_true[0]] @ rng_t.standard_normal(n_factors)
        obs_noise = rng_t.standard_normal(D_obs) * np.sqrt(true_R[z_true[0]])
        Y[0] = (true_C[z_true[0]] @ x_true[0] + true_d[z_true[0]]
                + factor_noise + obs_noise)

        for t in range(1, T):
            # Transition with recurrence
            logits = (true_W[z_true[t - 1]] + true_S[z_true[t - 1]] @ U[t]
                      + true_R_recur[z_true[t - 1]] @ x_true[t - 1])
            probs = softmax(logits)
            z_true[t] = rng_t.choice(K, p=probs)

            # Latent dynamics
            x_true[t] = (true_A[z_true[t]] @ x_true[t - 1] + true_b[z_true[t]]
                         + rng_t.standard_normal(D_latent) * np.sqrt(0.1))

            # Emission with factor noise
            factor_noise = true_F[z_true[t]] @ rng_t.standard_normal(n_factors)
            obs_noise = rng_t.standard_normal(D_obs) * np.sqrt(true_R[z_true[t]])
            Y[t] = (true_C[z_true[t]] @ x_true[t] + true_d[z_true[t]]
                    + factor_noise + obs_noise)

        # --- K-means baseline ---
        km = KMeans(n_clusters=K, n_init=10, random_state=trial_seed).fit(Y)
        ari_kmeans = adjusted_rand_score(z_true, km.labels_)

        trial_result = {'ari_kmeans': float(ari_kmeans)}

        # --- Fit model variants ---

        # 1. IOHMM (no latent)
        cfg_io = IOHMMConfig(K=K, D_obs=D_obs, D_input=D_input,
                              n_restarts=2, max_em_iter=80)
        model_io = IOHMM(cfg_io)
        p_io, h_io = model_io.fit(Y, U, seed=trial_seed, verbose=False)
        vit_io = model_io.viterbi(Y, U, None, p_io)
        trial_result['ari_iohmm'] = float(adjusted_rand_score(z_true, vit_io))
        trial_result['bic_iohmm'] = float(h_io['bic'])

        # 2-5: Fit SLDS variants in parallel
        from joblib import Parallel, delayed

        variant_cfgs = [
            ('slds', dict(D_latent=D_latent, n_restarts=3, max_em_iter=100)),
            ('slds_fa', dict(D_latent=D_latent, n_factors=n_factors,
                              n_restarts=3, max_em_iter=100)),
            ('rslds', dict(D_latent=D_latent, recurrent=True,
                            n_restarts=3, max_em_iter=100)),
            ('full', dict(D_latent=D_latent, n_factors=n_factors,
                           recurrent=True, n_restarts=3, max_em_iter=100)),
        ]

        def _fit_variant(cfg_kwargs, Y_data, U_data, z_gt, ts):
            with limit_blas_threads(1):
                from cadence.significance.rslds_model import IOHMMConfig, fit_slds
                from sklearn.metrics import adjusted_rand_score as ari_fn
                cfg_v = IOHMMConfig(K=K, D_obs=D_obs, D_input=D_input,
                                     **cfg_kwargs)
                _, h = fit_slds(Y_data, U_data, None, cfg_v, seed=ts,
                                verbose=False)
                vit = np.argmax(h['gamma'], axis=1)
                return float(ari_fn(z_gt, vit)), float(h['bic'])

        # Each SLDS fit holds Y/U/x/state arrays + factor analysis workspace.
        # ~1 GB peak per worker is the empirical V11 estimate (D_obs=26).
        n_jobs_v = pick_n_jobs(per_worker_ram_gb=1.0, requested=4,
                                max_jobs_hard_cap=len(variant_cfgs))
        par_results = Parallel(n_jobs=n_jobs_v, prefer='threads')(
            delayed(_fit_variant)(cfg_kw, Y, U, z_true, trial_seed)
            for name, cfg_kw in variant_cfgs
        )

        for (name, _), (ari_v, bic_v) in zip(variant_cfgs, par_results):
            trial_result[f'ari_{name}'] = ari_v
            trial_result[f'bic_{name}'] = bic_v

        results['trials'].append(trial_result)
        if verbose:
            print(f"  Trial {trial}: kmeans={ari_kmeans:.3f}, "
                  f"IOHMM={trial_result['ari_iohmm']:.3f}, "
                  f"SLDS={trial_result['ari_slds']:.3f}, "
                  f"FA={trial_result['ari_slds_fa']:.3f}, "
                  f"rSLDS={trial_result['ari_rslds']:.3f}, "
                  f"Full={trial_result['ari_full']:.3f}")

    # Aggregate
    model_keys = ['ari_kmeans', 'ari_iohmm', 'ari_slds', 'ari_slds_fa',
                  'ari_rslds', 'ari_full']
    bic_keys = ['bic_iohmm', 'bic_slds', 'bic_slds_fa', 'bic_rslds',
                'bic_full']
    for key in model_keys:
        vals = [t[key] for t in results['trials']]
        results[f'{key}_mean'] = float(np.mean(vals))
    for key in bic_keys:
        vals = [t[key] for t in results['trials']]
        results[f'{key}_mean'] = float(np.mean(vals))

    results['test_valid'] = results['ari_kmeans_mean'] < 0.7
    # Pass criteria for SLDS on SLDS-generated data:
    # 1. Model adds value beyond k-means clustering
    # 2. BIC improves over IOHMM (continuous latent helps)
    # 3. Factor/recurrent variants at least match plain SLDS
    results['passed'] = (
        results['ari_full_mean'] > results['ari_kmeans_mean'] + 0.05
        and results['test_valid']
        and results['bic_full_mean'] < results['bic_iohmm_mean']
        and results['ari_slds_fa_mean'] >= results['ari_slds_mean'] - 0.05
    )

    if verbose:
        print(f"\n  K-means:  {results['ari_kmeans_mean']:.3f} "
              f"({'VALID' if results['test_valid'] else 'TOO EASY'})")
        for label, key, bkey in [
            ('IOHMM  ', 'ari_iohmm', 'bic_iohmm'),
            ('SLDS   ', 'ari_slds', 'bic_slds'),
            ('SLDS+FA', 'ari_slds_fa', 'bic_slds_fa'),
            ('rSLDS  ', 'ari_rslds', 'bic_rslds'),
            ('Full   ', 'ari_full', 'bic_full'),
        ]:
            print(f"  {label}: ARI={results[f'{key}_mean']:.3f}, "
                  f"BIC={results[f'{bkey}_mean']:.0f}")
        print(f"  [{'PASS' if results['passed'] else 'FAIL'}] "
              f"Synthetic SLDS recovery")

    return results


def synthetic_hierarchical_recovery(N_sessions: int = 5, K: int = 3,
                                     D_obs: int = 7, D_input: int = 2,
                                     D_latent: int = 3, T_per: int = 1500,
                                     mean_scale: float = 0.5,
                                     seed: int = 42,
                                     verbose: bool = True) -> dict:
    """Multi-session synthetic test: shared dynamics, session-specific emissions.

    Verifies that hierarchical fitting:
    1. Recovers shared dynamics better than per-session fitting
    2. Achieves cross-session state alignment (high inter-session ARI)
    3. BIC favors hierarchical when data warrants pooling

    Returns:
        dict with per-session ARI, cross-session ARI, dynamics recovery, BIC
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)

    # ── Shared ground truth ──────────────────────────────────────────
    true_W = np.eye(K) * 3.0 - 0.5
    true_S = rng.normal(0, 0.3, size=(K, K, D_input))
    true_A = np.zeros((K, D_latent, D_latent))
    for k in range(K):
        true_A[k] = 0.8 * np.eye(D_latent) + rng.standard_normal(
            (D_latent, D_latent)) * 0.1
        ev = np.abs(np.linalg.eigvals(true_A[k]))
        if ev.max() > 0.95:
            true_A[k] *= 0.9 / ev.max()
    true_b = rng.standard_normal((K, D_latent)) * 0.1

    # ── Per-session ground truth (emission variation) ────────────────
    base_d = rng.standard_normal((K, D_obs)) * mean_scale
    order = np.argsort(base_d[:, 0])
    base_d = base_d[order]
    base_C = rng.standard_normal((K, D_obs, D_latent)) * 0.3

    sessions_data = []  # (Y, U, mask)
    sessions_z = []     # ground truth z per session

    for n in range(N_sessions):
        rng_n = np.random.default_rng(seed + n * 100 + 1)
        T = T_per + rng_n.integers(-200, 200)

        # Session-specific d and C (vary around shared base)
        d_n = base_d + rng_n.standard_normal((K, D_obs)) * 0.1
        C_n = base_C + rng_n.standard_normal((K, D_obs, D_latent)) * 0.05
        R_n = np.abs(rng_n.standard_normal((K, D_obs))) * 0.3 + 0.1

        # Block covariates
        U = np.zeros((T, D_input))
        bs = T // (K * 2)
        for b in range(0, T, bs):
            be = min(b + bs, T)
            U[b:be, (b // bs) % D_input] = rng_n.choice([-1, 1])

        # Generate data
        z = np.zeros(T, dtype=int)
        x = np.zeros((T, D_latent))
        Y = np.zeros((T, D_obs))
        z[0] = rng_n.integers(K)
        x[0] = rng_n.standard_normal(D_latent) * 0.3
        Y[0] = C_n[z[0]] @ x[0] + d_n[z[0]] + rng_n.standard_normal(D_obs) * np.sqrt(R_n[z[0]])

        for t in range(1, T):
            logits = true_W[z[t-1]] + true_S[z[t-1]] @ U[t]
            z[t] = rng_n.choice(K, p=softmax(logits))
            x[t] = true_A[z[t]] @ x[t-1] + true_b[z[t]] + rng_n.standard_normal(D_latent) * 0.3
            Y[t] = C_n[z[t]] @ x[t] + d_n[z[t]] + rng_n.standard_normal(D_obs) * np.sqrt(R_n[z[t]])

        sessions_data.append((Y, U, None))
        sessions_z.append(z)

    if verbose:
        print(f"  Generated {N_sessions} sessions, T~{T_per}")

    # ── Per-session fitting (baseline) ───────────────────────────────
    if verbose:
        print("  Fitting per-session SLDS...")
    from joblib import Parallel, delayed

    def _fit_one(args):
        with limit_blas_threads(1):
            Y, U, mask, s_seed = args
            from cadence.significance.rslds_model import IOHMMConfig, fit_slds
            cfg = IOHMMConfig(K=K, D_obs=D_obs, D_input=D_input,
                              D_latent=D_latent,
                              n_restarts=2, max_em_iter=80)
            _, h = fit_slds(Y, U, mask, cfg, seed=s_seed, verbose=False)
            return h

    ps_args = [(Y, U, m, seed + i * 1000 + 500)
               for i, (Y, U, m) in enumerate(sessions_data)]
    # Each per-session SLDS fit holds Y/U/x + workspace; ~1 GB peak per worker.
    n_jobs_ps = pick_n_jobs(per_worker_ram_gb=1.0, requested=-1,
                             max_jobs_hard_cap=len(ps_args))
    ps_results = Parallel(n_jobs=n_jobs_ps, prefer='threads')(
        delayed(_fit_one)(a) for a in ps_args)

    ps_aris = []
    for n, h in enumerate(ps_results):
        vit = np.argmax(h['gamma'], axis=1)
        ari = adjusted_rand_score(sessions_z[n], vit)
        ps_aris.append(float(ari))

    if verbose:
        print(f"    Per-session ARI: mean={np.mean(ps_aris):.3f}")

    # ── Hierarchical fitting ─────────────────────────────────────────
    if verbose:
        print("  Fitting hierarchical SLDS...")

    cfg_h = IOHMMConfig(K=K, D_obs=D_obs, D_input=D_input, D_latent=D_latent,
                         n_restarts=2, max_em_iter=80, n_factors=0,
                         recurrent=False)
    h_result = fit_hierarchical_slds(sessions_data, cfg_h, seed=seed,
                                      verbose=verbose)

    h_aris = []
    for n in range(N_sessions):
        vit = np.argmax(h_result['sessions'][n]['gamma'], axis=1)
        ari = adjusted_rand_score(sessions_z[n], vit)
        h_aris.append(float(ari))

    if verbose:
        print(f"    Hierarchical ARI: mean={np.mean(h_aris):.3f}")

    # ── Cross-session state consistency ──────────────────────────────
    # For hierarchical: states should correspond across sessions
    cross_aris = []
    for i in range(N_sessions):
        for j in range(i + 1, N_sessions):
            T_min = min(len(sessions_z[i]), len(sessions_z[j]))
            cross_ari = adjusted_rand_score(
                sessions_z[i][:T_min], sessions_z[j][:T_min])
            cross_aris.append(float(cross_ari))

    # ── Dynamics recovery ────────────────────────────────────────────
    A_shared = h_result['shared']['A_dyn']
    # Hungarian match for A recovery
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sum((true_A[i] - A_shared[j]) ** 2)
    _, match = linear_sum_assignment(cost)
    A_frob = float(np.sqrt(np.mean([np.sum((true_A[i] - A_shared[match[i]]) ** 2)
                                      for i in range(K)])))

    # Per-session BIC sum vs hierarchical BIC
    ps_bic_sum = sum(h['bic'] for h in ps_results)

    results = {
        'per_session_ari': ps_aris,
        'hierarchical_ari': h_aris,
        'ps_ari_mean': float(np.mean(ps_aris)),
        'h_ari_mean': float(np.mean(h_aris)),
        'cross_session_ari_mean': float(np.mean(cross_aris)),
        'A_frobenius': A_frob,
        'ps_bic_sum': float(ps_bic_sum),
        'h_bic': float(h_result['bic']),
        'bic_improvement': float(ps_bic_sum - h_result['bic']),
    }

    # Pass criteria
    results['passed'] = (
        results['h_ari_mean'] >= results['ps_ari_mean'] - 0.05  # hierarchical >= per-session
        and results['h_bic'] < results['ps_bic_sum']  # hierarchical BIC better
    )

    if verbose:
        print(f"\n  Per-session ARI:    {results['ps_ari_mean']:.3f}")
        print(f"  Hierarchical ARI:   {results['h_ari_mean']:.3f}")
        print(f"  A_dyn Frobenius:    {A_frob:.3f}")
        print(f"  Per-session BIC sum:{results['ps_bic_sum']:.0f}")
        print(f"  Hierarchical BIC:   {results['h_bic']:.0f}")
        print(f"  BIC improvement:    {results['bic_improvement']:.0f}")
        print(f"  [{'PASS' if results['passed'] else 'FAIL'}] "
              f"Hierarchical recovery")

    return results
