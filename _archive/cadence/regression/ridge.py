"""Batched static ridge regression via Cholesky decomposition.

Used as a fallback/comparison for EWLS. When tau -> infinity, EWLS
should converge to the static ridge solution.
"""

import torch


def batched_ridge(X, y, lambda_ridge=1e-3, valid=None):
    """Solve static ridge regression: beta = (X'X + lambda*I)^{-1} X'y.

    Args:
        X: (T, p) design matrix tensor.
        y: (T, C) target tensor.
        lambda_ridge: Regularization strength.
        valid: (T,) boolean mask (optional).

    Returns:
        beta: (p, C) regression coefficients.
        y_hat: (T, C) predictions.
        r_squared: scalar R-squared value.
    """
    if valid is not None:
        X = X[valid]
        y = y[valid]

    T, p = X.shape

    # Normal equations with ridge
    XtX = X.T @ X  # (p, p)
    Xty = X.T @ y  # (p, C)
    reg = lambda_ridge * torch.eye(p, device=X.device, dtype=X.dtype)

    # Cholesky solve
    beta = torch.linalg.solve(XtX + reg, Xty)  # (p, C)

    # Predictions on full (unmasked) data
    y_hat = X @ beta  # (T, C)

    # R-squared
    ss_res = ((y - y_hat) ** 2).sum()
    y_mean = y.mean(dim=0, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum()

    r_squared = (1.0 - ss_res / (ss_tot + 1e-10)).clamp(-1.0, 1.0)

    return beta, y_hat, r_squared.item()


def batched_ridge_multi(X_batch, y, lambda_ridge=1e-3, valid=None):
    """Solve K ridge regression problems in parallel on GPU.

    All K systems share the same target y but have different design matrices.
    Uses batched torch.linalg.solve for GPU parallelism — one large kernel
    launch instead of K sequential small ones.

    Args:
        X_batch: (K, T, p) batch of design matrices.
        y: (T, C) shared target tensor.
        lambda_ridge: Regularization strength.
        valid: (T,) boolean mask (optional).

    Returns:
        r2_batch: (K,) R-squared values (one per system).
    """
    if valid is not None:
        X_batch = X_batch[:, valid]
        y = y[valid]

    K, T, p = X_batch.shape
    C = y.shape[1]

    # Batched normal equations: XtX[k] = X[k]^T @ X[k]
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # (K, p, p)

    # Expand y once for all K systems
    y_exp = y.unsqueeze(0).expand(K, -1, -1)  # (K, T, C)
    Xty = torch.bmm(X_batch.transpose(1, 2), y_exp)  # (K, p, C)

    # Regularization (broadcast across batch)
    reg = lambda_ridge * torch.eye(p, device=X_batch.device, dtype=X_batch.dtype)

    # Batched Cholesky solve — one GPU kernel for all K systems
    beta = torch.linalg.solve(XtX + reg, Xty)  # (K, p, C)

    # Predictions
    y_hat = torch.bmm(X_batch, beta)  # (K, T, C)

    # R-squared per system (sum across time and channels)
    ss_res = ((y_exp - y_hat) ** 2).sum(dim=(1, 2))  # (K,)
    y_mean = y.mean(dim=0, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum()

    r2_batch = (1.0 - ss_res / (ss_tot + 1e-10)).clamp(-1.0, 1.0)

    return r2_batch


def batched_ridge_multi_per_ch(X_batch, y, lambda_ridge=1e-3, valid=None):
    """Batched ridge returning per-channel R² for each system.

    Args:
        X_batch: (K, T, p) batch of design matrices.
        y: (T, C) shared target tensor.
        lambda_ridge: Regularization strength.
        valid: (T,) boolean mask (optional).

    Returns:
        r2_per_ch: (K, C) R-squared per system per channel.
    """
    if valid is not None:
        X_batch = X_batch[:, valid]
        y = y[valid]

    K, T, p = X_batch.shape
    C = y.shape[1]

    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # (K, p, p)
    y_exp = y.unsqueeze(0).expand(K, -1, -1)  # (K, T, C)
    Xty = torch.bmm(X_batch.transpose(1, 2), y_exp)  # (K, p, C)
    reg = lambda_ridge * torch.eye(p, device=X_batch.device, dtype=X_batch.dtype)
    beta = torch.linalg.solve(XtX + reg, Xty)  # (K, p, C)
    y_hat = torch.bmm(X_batch, beta)  # (K, T, C)

    ss_res = ((y_exp - y_hat) ** 2).sum(dim=1)  # (K, C)
    y_mean = y.mean(dim=0, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum(dim=0)  # (C,)

    r2_per_ch = (1.0 - ss_res / (ss_tot.unsqueeze(0) + 1e-10)).clamp(-1.0, 1.0)
    return r2_per_ch


def batched_ridge_per_target(X, y, lambda_ridge=1e-3, valid=None):
    """Ridge regression with per-target-channel R-squared.

    Args:
        X: (T, p) design matrix.
        y: (T, C) multi-channel target.
        lambda_ridge: Regularization strength.
        valid: (T,) boolean mask.

    Returns:
        beta: (p, C) coefficients.
        y_hat: (T, C) predictions.
        r2_per_channel: (C,) R-squared per target channel.
    """
    if valid is not None:
        X = X[valid]
        y = y[valid]

    T, p = X.shape
    C = y.shape[1]

    XtX = X.T @ X
    Xty = X.T @ y
    reg = lambda_ridge * torch.eye(p, device=X.device, dtype=X.dtype)

    beta = torch.linalg.solve(XtX + reg, Xty)
    y_hat = X @ beta

    ss_res = ((y - y_hat) ** 2).sum(dim=0)
    y_mean = y.mean(dim=0, keepdim=True)
    ss_tot = ((y - y_mean) ** 2).sum(dim=0)

    r2_per_channel = (1.0 - ss_res / (ss_tot + 1e-10)).clamp(-1.0, 1.0)

    return beta, y_hat, r2_per_channel
