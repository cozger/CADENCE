from __future__ import annotations
import numpy as np
from typing import List


def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized ACF at lags 0..max_lag via FFT."""
    n = len(x)
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return acov / acov[0]


def compute_n_eff_geyer(x: np.ndarray) -> float:
    """Effective sample size via Geyer's (1992) monotone truncation.

    Truncates the ACF sum at the first lag pair where the sum of
    consecutive ACF values is non-positive, preventing negative N_eff
    from over-whitened signals.
    """
    n = len(x)
    acf = compute_acf(x, max_lag=n - 1)
    # Pair consecutive lags (2m-1, 2m) and truncate when pair sum <= 0
    gamma_sum = 0.0
    for m in range(1, n // 2):
        pair = acf[2 * m - 1] + acf[2 * m]
        if pair <= 0:
            break
        gamma_sum += pair
    return float(n / max(1.0, -1.0 + 2.0 * gamma_sum))


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor for each column of X.

    VIF[j] = 1 / (1 - R^2) where R^2 is from regressing column j on all others.
    """
    from sklearn.linear_model import LinearRegression
    n, p = X.shape
    vifs = np.empty(p)
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[j] = 1.0 / max(1e-9, 1.0 - r2)
    return vifs


def is_slow_drift(x: np.ndarray, fs: float,
                  lag_s: float = 10.0, threshold: float = 0.3) -> bool:
    """True if ACF(lag_s) > threshold — channel dominated by slow drift."""
    lag_samples = int(round(lag_s * fs))
    acf = compute_acf(x, max_lag=lag_samples)
    return bool(acf[lag_samples] > threshold)
