"""Time-blocked cross-validation and BIC model selection for temporal data.

Creates contiguous folds with gaps between train/test to prevent
temporal leakage from autocorrelation. Also provides Extended BIC
lambda selection for group lasso.
"""

import numpy as np
import torch
from scipy.special import gammaln


def create_time_blocks(T, n_folds=5, gap_samples=150, valid=None):
    """Create contiguous time-block folds with gaps.

    Splits T timepoints into n_folds contiguous blocks.
    Each fold uses one block as test, all others (minus gap) as train.

    Args:
        T: total number of timepoints
        n_folds: number of folds
        gap_samples: number of samples to exclude around test block
                     (prevents leakage from autocorrelation)
        valid: (T,) boolean mask (if provided, masks are ANDed with validity)

    Returns:
        list of (train_mask, test_mask) boolean arrays of length T
    """
    # Compute block boundaries (contiguous, equal-size blocks)
    boundaries = np.linspace(0, T, n_folds + 1, dtype=int)

    folds = []
    for k in range(n_folds):
        test_start = boundaries[k]
        test_end = boundaries[k + 1]

        test_mask = np.zeros(T, dtype=bool)
        test_mask[test_start:test_end] = True

        # Train mask: everything except test block and gap around it
        gap_start = max(0, test_start - gap_samples)
        gap_end = min(T, test_end + gap_samples)

        excluded = np.zeros(T, dtype=bool)
        excluded[gap_start:gap_end] = True

        train_mask = ~excluded

        # AND with validity mask if provided
        if valid is not None:
            valid_np = valid
            if isinstance(valid, torch.Tensor):
                valid_np = valid.cpu().numpy()
            train_mask = train_mask & valid_np
            test_mask = test_mask & valid_np

        folds.append((train_mask, test_mask))

    return folds


def cross_validate_lambda(solver, X, y, valid=None, n_lambdas=20,
                           n_folds=5, gap_seconds=30, eval_rate=5.0):
    """Cross-validate group lasso regularization strength.

    Log-spaced lambda path from lambda_max to lambda_max/100.

    Args:
        solver: GroupLassoSolver instance
        X: (T, p) design matrix
        y: (T, C_tgt) target
        valid: (T,) boolean mask
        n_lambdas: number of lambda values to test
        n_folds: number of CV folds
        gap_seconds: gap in seconds between train/test
        eval_rate: evaluation rate in Hz (for converting gap to samples)

    Returns:
        best_lambda: optimal lambda (lowest CV error)
        cv_scores: (n_lambdas,) mean CV R^2 scores
        lambda_path: (n_lambdas,) lambda values tested
        selected_counts: (n_lambdas,) number of groups selected at each lambda
    """
    T, p = X.shape
    C = y.shape[1] if y.dim() > 1 else 1
    if y.dim() == 1:
        y = y.unsqueeze(1)

    gap_samples = int(gap_seconds * eval_rate)

    # Create time-block folds once
    valid_np = None
    if valid is not None:
        valid_np = valid.cpu().numpy() if isinstance(valid, torch.Tensor) else valid
    folds = create_time_blocks(T, n_folds=n_folds, gap_samples=gap_samples,
                                valid=valid_np)

    # Compute lambda path from lambda_max across all folds
    # Use the global lambda_max (conservative: ensures path covers all folds)
    lambda_max = solver._compute_lambda_max(X, y, valid)
    lambda_path = np.logspace(
        np.log10(lambda_max),
        np.log10(lambda_max * 0.001),
        n_lambdas
    )

    # Accumulate CV scores: (n_lambdas, n_folds)
    fold_scores = np.full((n_lambdas, n_folds), np.nan)

    for fold_idx, (train_mask, test_mask) in enumerate(folds):
        train_mask_t = torch.tensor(train_mask, dtype=torch.bool,
                                     device=solver.device)
        test_mask_t = torch.tensor(test_mask, dtype=torch.bool,
                                    device=solver.device)

        n_test = int(test_mask_t.sum().item())
        if n_test == 0:
            continue

        # Precompute test data for this fold
        X_test = X[test_mask_t]  # (n_test, p)
        y_test = y[test_mask_t]  # (n_test, C)
        y_test_mean = y_test.mean(dim=0, keepdim=True)
        ss_tot = ((y_test - y_test_mean) ** 2).sum().item()

        # Warm-start path along fixed lambda grid (large to small alpha)
        warm = None
        for lam_idx, lam in enumerate(lambda_path):
            beta, _, _ = solver.fit(
                X, y, alpha=float(lam), valid=train_mask_t,
                max_iter=500, tol=1e-5, warm_start=warm
            )
            warm = beta

            # Evaluate R^2 on test fold
            y_hat = X_test @ beta  # (n_test, C)
            ss_res = ((y_test - y_hat) ** 2).sum().item()

            if ss_tot > 1e-10:
                r2 = max(-1.0, min(1.0, 1.0 - ss_res / ss_tot))
            else:
                r2 = 0.0

            fold_scores[lam_idx, fold_idx] = r2

    # Mean CV R^2 across folds (ignoring NaN for folds with no test data)
    cv_scores = np.nanmean(fold_scores, axis=1)  # (n_lambdas,)

    # Count selected groups at each lambda (refit on full data for counts)
    selected_counts = np.zeros(n_lambdas, dtype=int)
    warm = None
    for lam_idx, lam in enumerate(lambda_path):
        beta, selected, _ = solver.fit(
            X, y, alpha=float(lam), valid=valid,
            max_iter=500, tol=1e-5, warm_start=warm
        )
        selected_counts[lam_idx] = len(selected)
        warm = beta

    # Best lambda = highest mean CV R^2
    best_idx = int(np.nanargmax(cv_scores))
    best_lambda = float(lambda_path[best_idx])

    return best_lambda, cv_scores, lambda_path, selected_counts


def gradient_screen(solver, X, y, valid=None, max_features=None):
    """Select source features by partial correlation with target.

    Computes each source group's gradient norm after partialing out
    AR terms, then selects the top-K groups. This is equivalent to
    marginal screening (Fan & Lv 2008) on AR-residualized targets.

    Much more robust than BIC for multivariate targets (large C),
    where in-sample noise overfitting makes BIC unreliable.

    Args:
        solver: GroupLassoSolver instance
        X: (T, p) design matrix [source_basis_cols | AR_cols]
        y: (T, C_tgt) target
        valid: (T,) boolean mask
        max_features: max groups to select (default: min(20, G/2))

    Returns:
        selected: list of selected group indices (sorted)
        gradient_norms: (n_groups,) gradient norms per group
    """
    if y.dim() == 1:
        y = y.unsqueeze(1)

    # Apply validity mask
    if valid is not None:
        if isinstance(valid, torch.Tensor):
            X_v = X[valid]
            y_v = y[valid]
        else:
            valid_t = torch.tensor(valid, dtype=torch.bool, device=X.device)
            X_v = X[valid_t]
            y_v = y[valid_t]
    else:
        X_v = X
        y_v = y

    n = X_v.shape[0]
    G = solver.n_groups
    gs = solver.group_size
    ge = solver._group_end
    C = y_v.shape[1]

    if max_features is None:
        max_features = min(20, max(5, G // 2))

    # Partial out AR terms (columns after group_end)
    n_ar_cols = X_v.shape[1] - ge
    if n_ar_cols > 0:
        X_ar = X_v[:, ge:]
        # Ridge fit AR to get residuals
        lam_ar = 1e-3
        XtX_ar = X_ar.T @ X_ar / n + lam_ar * torch.eye(
            n_ar_cols, device=X.device, dtype=X.dtype)
        Xty_ar = X_ar.T @ y_v / n
        beta_ar = torch.linalg.solve(XtX_ar, Xty_ar)
        y_resid = y_v - X_ar @ beta_ar
    else:
        y_resid = y_v

    # Compute per-group gradient norm on AR-residualized target
    X_src = X_v[:, :ge]
    Xty_resid = X_src.T @ y_resid / n  # (ge, C)
    Xty_3d = Xty_resid.view(G, gs, C)
    gradient_norms = Xty_3d.norm(dim=(1, 2))  # (G,)

    # Select top K
    K = min(max_features, G)
    _, top_indices = gradient_norms.topk(K)
    selected = sorted(top_indices.tolist())

    return selected, gradient_norms


def bic_lambda_selection(solver, X, y, valid=None, n_lambdas=20,
                          lambda_ratio=0.001, ebic_gamma=0.5):
    """Select group lasso lambda via Extended BIC.

    Fits along lambda path from lambda_max to lambda_max * lambda_ratio.
    Selects the lambda that minimizes EBIC.

    Args:
        solver: GroupLassoSolver instance
        X: (T, p) design matrix
        y: (T, C_tgt) target
        valid: (T,) boolean mask
        n_lambdas: number of lambda values to test
        lambda_ratio: lambda_min = lambda_max * ratio
        ebic_gamma: EBIC sparsity control (0=BIC, 0.5-1=sparser)

    Returns:
        best_lambda: optimal lambda (lowest EBIC)
        bic_scores: (n_lambdas,) EBIC scores
        lambda_path: (n_lambdas,) lambda values tested
        selected_counts: (n_lambdas,) number of groups selected at each lambda
    """
    # Fit along warm-started lambda path
    betas, lambda_path, selected_per_lambda = solver.fit_path(
        X, y, valid=valid, n_lambdas=n_lambdas,
        lambda_ratio=lambda_ratio, max_iter=1000, tol=1e-6)

    # Effective sample size
    if valid is not None:
        if isinstance(valid, torch.Tensor):
            n = int(valid.sum().item())
        else:
            n = int(np.sum(valid))
    else:
        n = X.shape[0]

    G = solver.n_groups
    gs = solver.group_size
    C = y.shape[1] if y.dim() > 1 else 1

    bic_scores = np.full(n_lambdas, np.inf)
    selected_counts = np.zeros(n_lambdas, dtype=int)

    for i, (beta, selected) in enumerate(zip(betas, selected_per_lambda)):
        selected_counts[i] = len(selected)
        k = len(selected) * gs  # active params per channel (C cancels in argmin)

        # Compute RSS
        if valid is not None:
            if isinstance(valid, torch.Tensor):
                residual = X[valid] @ beta - y[valid]
            else:
                valid_t = torch.tensor(valid, dtype=torch.bool, device=X.device)
                residual = X[valid_t] @ beta - y[valid_t]
        else:
            residual = X @ beta - y
        rss = (residual ** 2).sum().item()

        # BIC = n * log(RSS/n) + k * log(n)
        bic = n * np.log(rss / n + 1e-20) + k * np.log(n)

        # EBIC combinatorial penalty
        if 0 < len(selected) < G and ebic_gamma > 0:
            log_comb = (gammaln(G + 1) - gammaln(len(selected) + 1)
                        - gammaln(G - len(selected) + 1))
            bic += 2 * ebic_gamma * log_comb

        bic_scores[i] = bic

    best_idx = int(np.argmin(bic_scores))
    best_lambda = float(lambda_path[best_idx])

    return best_lambda, bic_scores, lambda_path, selected_counts
