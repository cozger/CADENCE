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


import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_feature_diagnostics(sessions, output_dir: str) -> dict:
    """Run all Module 2 diagnostics and write outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    D = len(sessions[0].modality_keys)
    keys = sessions[0].modality_keys

    # ── ACF grid (raw features) ──────────────────────────────────────────
    max_lag_s = 60.0
    fs = sessions[0].fs
    max_lag = int(max_lag_s * fs)
    lags_s = np.arange(max_lag + 1) / fs

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.0 * D), sharex=True)
    slow_drift_flags = []
    for d, key in enumerate(keys):
        ax = axes[d] if D > 1 else axes
        all_acf = []
        for sess in sessions:
            acf = compute_acf(sess.Y_raw[:, d], max_lag)
            all_acf.append(acf)
            ax.plot(lags_s, acf, alpha=0.4, linewidth=0.7, color='steelblue')
        mean_acf = np.mean(all_acf, axis=0)
        ax.plot(lags_s, mean_acf, color='navy', linewidth=1.2)
        ax.axhline(0.3, color='red', linewidth=0.8, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.4)
        flag = bool(mean_acf[max_lag] > 0.3)
        slow_drift_flags.append(flag)
        color = 'firebrick' if flag else 'black'
        ax.set_ylabel(key, fontsize=6, color=color, rotation=0, ha='right')
    axes[-1].set_xlabel('Lag (s)')
    fig.suptitle('ACF per channel (raw features) — red = SLOW_DRIFT flag', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'acf_all_channels.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Power spectra ────────────────────────────────────────────────────
    from scipy.signal import welch
    fig, axes = plt.subplots(D, 1, figsize=(10, 1.8 * D), sharex=True)
    for d, key in enumerate(keys):
        ax = axes[d] if D > 1 else axes
        for sess in sessions:
            f, pxx = welch(sess.Y_raw[:, d], fs=sess.fs, nperseg=min(256, len(sess.Y_raw)))
            ax.semilogy(f, pxx, alpha=0.4, linewidth=0.7, color='steelblue')
        ax.set_ylabel(key, fontsize=6, rotation=0, ha='right')
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('Power spectra (raw features)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_spectra.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Cross-channel correlation heatmap (raw, pooled sessions) ────────
    Y_all = np.vstack([s.Y_raw for s in sessions])
    corr = np.corrcoef(Y_all.T)
    fig, ax = plt.subplots(figsize=(max(6, D * 0.5), max(5, D * 0.45)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(D)); ax.set_xticklabels(keys, rotation=90, fontsize=6)
    ax.set_yticks(range(D)); ax.set_yticklabels(keys, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title('Cross-channel correlation (lag 0)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── VIF table ────────────────────────────────────────────────────────
    vifs = compute_vif(Y_all)
    vif_df = pd.DataFrame({'channel': keys, 'VIF': vifs,
                            'HIGH_VIF': vifs > 10.0})
    vif_df.to_csv(os.path.join(output_dir, 'vif_table.csv'), index=False)

    # ── Collinear pairs ──────────────────────────────────────────────────
    collinear_pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr[i, j]) > 0.8:
                collinear_pairs.append((keys[i], keys[j], float(corr[i, j])))

    # ── N_eff table ──────────────────────────────────────────────────────
    neff_rows = []
    for sess in sessions:
        row = {'session': sess.name}
        for d, key in enumerate(keys):
            row[key] = compute_n_eff_geyer(sess.Y_raw[:, d])
        neff_rows.append(row)
    neff_df = pd.DataFrame(neff_rows)
    neff_df.to_csv(os.path.join(output_dir, 'n_eff_per_session.csv'), index=False)
    low_info = [k for k in keys if neff_df[k].median() < 50]

    # ── Report ───────────────────────────────────────────────────────────
    sd_names = [k for k, f in zip(keys, slow_drift_flags) if f]
    hv_names = list(vif_df[vif_df['HIGH_VIF']]['channel'])
    report_lines = [
        f'SLOW_DRIFT: {sd_names}',
        f'LOW_INFO (median N_eff < 50): {low_info}',
        f'COLLINEAR_PAIRS (|r|>0.8): {collinear_pairs}',
        f'HIGH_VIF (>10): {hv_names}',
    ]
    with open(os.path.join(output_dir, 'module2_report.md'), 'w') as f:
        f.write('# Module 2: Feature Diagnostics\n\n')
        f.write('\n'.join(report_lines) + '\n')

    return {
        'slow_drift': sd_names,
        'low_info': low_info,
        'collinear_pairs': collinear_pairs,
        'high_vif': hv_names,
    }
