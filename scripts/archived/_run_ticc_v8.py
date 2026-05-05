"""CADENCE V8.1: Temporal coupling regime detection via TICC + Sticky IOHMM.

Compares approaches for temporally localized coupling states:
  Track A: TICC (Toeplitz Inverse Covariance-based Clustering)
  Track B: Sticky IOHMM + Constrained Viterbi
  Track C: Block-aggregated IOHMM

All use V8 9D z-timecourses as input.

Usage:
    python scripts/_run_ticc_v8.py                     # y_06 only
    python scripts/_run_ticc_v8.py --session y_06 y_17  # multiple sessions
    python scripts/_run_ticc_v8.py --all                # all V8 sessions
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, det
from sklearn.covariance import graphical_lasso
from joblib import Parallel, delayed

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, iohmm_flexibility_metrics, build_observation_mask,
)
from scripts._run_rslds_phase2 import MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS, FS_OUT

# ── TICC Core Implementation ─────────────────────────────────────────

def _build_stacked_obs(Y, window):
    """Build stacked observation matrix for TICC.

    Each row t contains [Y[t], Y[t-1], ..., Y[t-w+1]] flattened.
    Returns (T-w+1, D*w) matrix.
    """
    T, D = Y.shape
    T_eff = T - window + 1
    X = np.zeros((T_eff, D * window))
    for w in range(window):
        X[:, w*D:(w+1)*D] = Y[w:w+T_eff]
    return X


def _dp_assign(X, Thetas, beta):
    """Dynamic programming cluster assignment with switching penalty.

    cost[t,k] = -log P(X[t] | Theta_k) + beta * 1(cluster_t != cluster_{t-1})

    Log-likelihood under multivariate Gaussian with precision Theta:
        log P(x|Theta) = 0.5 * (log det Theta - x' Theta x - D*log(2pi))
    """
    T, Dw = X.shape
    K = len(Thetas)

    # Precompute emission log-likelihoods
    log_emit = np.zeros((T, K))
    for k in range(K):
        Theta = Thetas[k]
        sign, logdet = np.linalg.slogdet(Theta)
        if sign <= 0:
            logdet = -1e10  # degenerate
        # x' Theta x for each row
        xTx = np.sum((X @ Theta) * X, axis=1)
        log_emit[:, k] = 0.5 * (logdet - xTx - Dw * np.log(2 * np.pi))

    # DP forward pass
    cost = np.full((T, K), -np.inf)
    parent = np.zeros((T, K), dtype=np.int32)
    cost[0] = log_emit[0]

    for t in range(1, T):
        for k in range(K):
            # Stay in same cluster (no penalty)
            stay_score = cost[t-1, k] + log_emit[t, k]
            # Switch from best other cluster (with penalty)
            switch_scores = cost[t-1] + log_emit[t, k] - beta
            switch_scores[k] = stay_score  # no penalty for staying

            best = np.argmax(switch_scores)
            cost[t, k] = switch_scores[best]
            parent[t, k] = best

    # Backtrack
    path = np.empty(T, dtype=np.int32)
    path[T-1] = np.argmax(cost[T-1])
    for t in range(T-2, -1, -1):
        path[t] = parent[t+1, path[t+1]]

    return path, cost


def fit_ticc(Y, K=3, beta=500, window=10, lam=0.01, max_iter=20, seed=42):
    """Fit TICC model to multivariate time series.

    Args:
        Y: (T, D) observations
        K: number of clusters
        beta: switching penalty (higher = longer dwell times)
        window: temporal window size (samples)
        lam: graphical lasso regularization
        max_iter: maximum alternating minimization iterations

    Returns:
        labels: (T,) cluster assignments
        Thetas: list of K precision matrices (Dw, Dw)
        info: dict with convergence info
    """
    T, D = Y.shape
    Dw = D * window
    rng = np.random.default_rng(seed)

    # Build stacked observations
    X = _build_stacked_obs(Y, window)
    T_eff = len(X)

    # Initialize: random assignment
    labels = rng.integers(0, K, size=T_eff)

    # Initialize precision matrices from cluster covariances
    Thetas = []
    for k in range(K):
        mask = labels == k
        if mask.sum() > Dw + 5:
            S = np.cov(X[mask].T) + lam * np.eye(Dw)
            try:
                Theta = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Theta = np.eye(Dw)
        else:
            Theta = np.eye(Dw)
        Thetas.append(Theta)

    prev_labels = labels.copy()
    for it in range(max_iter):
        # Step 1: Update cluster assignments via DP
        labels, cost = _dp_assign(X, Thetas, beta)

        # Step 2: Update precision matrices via graphical lasso
        for k in range(K):
            mask = labels == k
            if mask.sum() > Dw + 5:
                S_k = np.cov(X[mask].T)
                # Ensure positive definite
                S_k += 1e-6 * np.eye(Dw)
                try:
                    _, Theta_k = graphical_lasso(S_k, alpha=lam, max_iter=50)
                    Thetas[k] = Theta_k
                except Exception:
                    pass  # keep previous Theta

        # Check convergence
        changed = (labels != prev_labels).sum()
        if changed == 0:
            break
        prev_labels = labels.copy()

    # Pad labels to full length (first window-1 samples get label of first assigned)
    full_labels = np.empty(T, dtype=np.int32)
    full_labels[:window-1] = labels[0]
    full_labels[window-1:] = labels

    # Compute BIC-like score
    ll = cost[np.arange(T_eff), labels].sum()
    n_params = K * Dw * (Dw + 1) // 2  # precision matrix params
    bic = -2 * ll + n_params * np.log(T_eff)

    return full_labels, Thetas, {
        'n_iter': it + 1,
        'bic': float(bic),
        'll': float(ll),
        'n_params': n_params,
    }


# ── Dwell time statistics ─────────────────────────────────────────────

def dwell_stats(labels, fs=FS_OUT):
    """Compute dwell time statistics from label sequence."""
    T = len(labels)
    K = labels.max() + 1
    transitions = int(np.sum(labels[1:] != labels[:-1]))

    dwells = []
    current = labels[0]
    run = 1
    for t in range(1, T):
        if labels[t] == current:
            run += 1
        else:
            dwells.append((current, run / fs))
            current = labels[t]
            run = 1
    dwells.append((current, run / fs))

    all_durations = [d for _, d in dwells]
    per_state = {}
    for k in range(K):
        state_dwells = [d for s, d in dwells if s == k]
        if state_dwells:
            per_state[k] = {
                'mean': float(np.mean(state_dwells)),
                'median': float(np.median(state_dwells)),
                'min': float(np.min(state_dwells)),
                'max': float(np.max(state_dwells)),
                'count': len(state_dwells),
            }

    usage = np.array([float((labels == k).mean()) for k in range(K)])

    return {
        'n_transitions': transitions,
        'transition_rate_hz': transitions / (T / fs),
        'mean_dwell_s': float(np.mean(all_durations)),
        'median_dwell_s': float(np.median(all_durations)),
        'usage': usage.tolist(),
        'per_state': per_state,
    }


# ── Condition alignment test ──────────────────────────────────────────

def condition_alignment(labels, t_common, session_name, K):
    """Test if cluster/state labels differentiate session conditions."""
    json_path = f'results/rslds/{session_name}/rslds_scaffold_v8_results.json'
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        info = json.load(f)
    segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]
    if len(segments) < 3:
        return None

    condition_usage = {}
    for seg_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        if mask.sum() < 10:
            continue
        usage = np.array([float((labels[mask] == k).mean()) for k in range(K)])
        condition_usage[seg_name] = usage

    if len(condition_usage) < 3:
        return None

    # Permutation test
    usage_matrix = np.array(list(condition_usage.values()))
    uniform = 1.0 / K
    chi2 = np.sum((usage_matrix - uniform) ** 2) / uniform

    rng = np.random.default_rng(42)
    null_chi2s = []
    for _ in range(200):
        perm = rng.permutation(labels)
        perm_usage = []
        for seg_name, t0, t1 in segments:
            mask = (t_common >= t0) & (t_common <= t1)
            if mask.sum() >= 10:
                perm_usage.append([float((perm[mask] == k).mean()) for k in range(K)])
        if perm_usage:
            pm = np.array(perm_usage)
            null_chi2s.append(np.sum((pm - uniform) ** 2) / uniform)

    p_value = float(np.mean(np.array(null_chi2s) >= chi2))
    return {
        'chi2': float(chi2),
        'p_value': p_value,
        'condition_usage': {k: v.tolist() for k, v in condition_usage.items()},
    }


# ── Load V8 data ─────────────────────────────────────────────────────

def load_v8(session_name):
    """Load V8 scaffold data."""
    npz = f'results/rslds/{session_name}/rslds_scaffold_v8_ztimecourses.npz'
    if not os.path.exists(npz):
        return None, None
    data = np.load(npz)
    t = data['t_common']
    z = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
    return z, t


# ══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_session(session_name):
    """Run all temporal localization approaches on one session."""
    print(f"\n{'='*70}")
    print(f"  V8.1 Temporal Localization — {session_name}")
    print(f"{'='*70}")

    Y, t_common = load_v8(session_name)
    if Y is None:
        print(f"  No V8 data for {session_name}")
        return None

    T, D = Y.shape
    U = np.zeros((T, 2), dtype=np.float64)
    obs_mask = np.ones((T, D), dtype=bool)
    print(f"  Loaded: {T} pts, {T/FS_OUT:.0f}s, {D}D")

    results = {}

    # ── Baseline: Standard IOHMM K=4 ─────────────────────────────────
    print(f"\n--- Baseline IOHMM (K=4) ---")
    cfg = IOHMMConfig(K=4, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
    model = IOHMM(cfg)
    params, hist = model.fit(Y, U, obs_mask, seed=42, verbose=False)
    labels_base = model.viterbi(Y, U, obs_mask, params)
    ds = dwell_stats(labels_base)
    cond = condition_alignment(labels_base, t_common, session_name, 4)
    results['baseline'] = {'dwell': ds, 'cond': cond, 'bic': hist['bic']}
    print(f"  Transitions: {ds['n_transitions']}, mean_dwell: {ds['mean_dwell_s']:.1f}s, "
          f"cond_p: {cond['p_value'] if cond else 'N/A'}")

    # ── Track B: Sticky IOHMM (κ sweep) ──────────────────────────────
    for kappa in [1.0, 3.0, 5.0]:
        print(f"\n--- Sticky IOHMM (k={kappa}) ---")
        cfg_s = IOHMMConfig(K=4, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150,
                            sticky_strength=kappa)
        model_s = IOHMM(cfg_s)
        params_s, hist_s = model_s.fit(Y, U, obs_mask, seed=42, verbose=False)
        labels_s = model_s.viterbi(Y, U, obs_mask, params_s)
        ds_s = dwell_stats(labels_s)
        cond_s = condition_alignment(labels_s, t_common, session_name, 4)
        results[f'sticky_k{kappa}'] = {'dwell': ds_s, 'cond': cond_s, 'bic': hist_s['bic']}
        print(f"  Transitions: {ds_s['n_transitions']}, mean_dwell: {ds_s['mean_dwell_s']:.1f}s, "
              f"cond_p: {cond_s['p_value'] if cond_s else 'N/A'}")

    # ── Track B: Constrained Viterbi (min_dwell sweep) ────────────────
    for min_dw in [10, 20, 40]:
        print(f"\n--- Constrained Viterbi (min_dwell={min_dw}, {min_dw/FS_OUT:.0f}s) ---")
        labels_cv = model.viterbi_min_dwell(Y, U, obs_mask, params, min_dwell=min_dw)
        ds_cv = dwell_stats(labels_cv)
        cond_cv = condition_alignment(labels_cv, t_common, session_name, 4)
        results[f'constrained_{min_dw}'] = {'dwell': ds_cv, 'cond': cond_cv}
        print(f"  Transitions: {ds_cv['n_transitions']}, mean_dwell: {ds_cv['mean_dwell_s']:.1f}s, "
              f"cond_p: {cond_cv['p_value'] if cond_cv else 'N/A'}")

    # ── Track B: Block aggregation ────────────────────────────────────
    for block_size in [10, 20]:
        print(f"\n--- Block-aggregated IOHMM (block={block_size}, {block_size/FS_OUT:.0f}s) ---")
        n_blocks = T // block_size
        Y_block = Y[:n_blocks * block_size].reshape(n_blocks, block_size, D).mean(axis=1)
        U_block = np.zeros((n_blocks, 2), dtype=np.float64)
        obs_block = np.ones((n_blocks, D), dtype=bool)

        cfg_b = IOHMMConfig(K=4, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
        model_b = IOHMM(cfg_b)
        params_b, hist_b = model_b.fit(Y_block, U_block, obs_block, seed=42, verbose=False)
        labels_block = model_b.viterbi(Y_block, U_block, obs_block, params_b)
        # Upsample to full resolution
        labels_up = np.repeat(labels_block, block_size)[:T]
        t_block = t_common[:len(labels_up)]

        ds_b = dwell_stats(labels_up)
        cond_b = condition_alignment(labels_up, t_block, session_name, 4)
        results[f'block_{block_size}'] = {'dwell': ds_b, 'cond': cond_b, 'bic': hist_b['bic']}
        print(f"  Transitions: {ds_b['n_transitions']}, mean_dwell: {ds_b['mean_dwell_s']:.1f}s, "
              f"cond_p: {cond_b['p_value'] if cond_b else 'N/A'}")

    # ── Track A: TICC (β sweep) ───────────────────────────────────────
    for beta in [100, 500, 1000]:
        print(f"\n--- TICC (K=4, b={beta}, window=10) ---")
        t0 = time.time()
        labels_t, Thetas, ticc_info = fit_ticc(
            Y, K=4, beta=beta, window=10, lam=0.01, max_iter=30, seed=42)
        elapsed = time.time() - t0
        ds_t = dwell_stats(labels_t)
        cond_t = condition_alignment(labels_t, t_common, session_name, 4)
        results[f'ticc_b{beta}'] = {
            'dwell': ds_t, 'cond': cond_t,
            'bic': ticc_info['bic'], 'ticc_info': ticc_info,
        }
        print(f"  Transitions: {ds_t['n_transitions']}, mean_dwell: {ds_t['mean_dwell_s']:.1f}s, "
              f"cond_p: {cond_t['p_value'] if cond_t else 'N/A'} ({elapsed:.0f}s)")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {session_name}")
    print(f"{'='*70}")
    print(f"  {'Method':>20s} | Trans | Dwell_s | cond_p  | Usage")
    print(f"  " + "-" * 70)
    for name, r in results.items():
        d = r['dwell']
        c = r.get('cond')
        p_str = f"{c['p_value']:.4f}" if c else "  N/A "
        usage = ' '.join(f'{u:.0%}' for u in d['usage'])
        print(f"  {name:>20s} | {d['n_transitions']:5d} | {d['mean_dwell_s']:7.1f} | {p_str} | {usage}")

    # ── Visualization ─────────────────────────────────────────────────
    best_methods = ['baseline', 'sticky_k3.0', 'constrained_20', 'block_20', 'ticc_b500']
    available = [m for m in best_methods if m in results]

    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Save results JSON
    json_results = {}
    for name, r in results.items():
        jr = {k: v for k, v in r.items() if k != 'ticc_info'}
        json_results[name] = jr
    with open(os.path.join(out_dir, 'v81_temporal_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n  Saved {out_dir}/v81_temporal_results.json")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', nargs='+', default=['y_06'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_v8_ztimecourses.npz'))
        sessions = [os.path.basename(os.path.dirname(p)) for p in npzs]
    else:
        sessions = args.session

    for sname in sessions:
        run_session(sname)


if __name__ == '__main__':
    main()
