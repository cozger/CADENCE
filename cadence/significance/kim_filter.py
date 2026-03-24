"""Markov-Switching Regression via Kim Filter (V3.5).

2-regime switching regression for temporal localization of interpersonal
coupling at native sampling rate.

V3.5b: Multivariate Kim filter — single shared regime switch across all
C matched channels. The shared regime requires correlated evidence from
multiple channels to switch, rejecting per-channel overfitting noise.

Regime 0 (uncoupled): y_c(t) = AR_c(p) + noise_c for all c
Regime 1 (coupled):   y_c(t) = AR_c(p) + z_c(t)' @ b_c + noise_c for all c

The joint observation likelihood is the PRODUCT across channels:
  f(y_1,...,y_C | S_t=j) = Π_c f(y_c | S_t=j)
  log f = Σ_c log f_c

This is the key: random overfitting in channel 3 doesn't help channel 7,
so the sum of log-likelihoods across channels rejects independent noise
while amplifying correlated coupling signal.
"""

import numpy as np
from joblib import Parallel, delayed


def _estimate_ar(y, ar_order, valid=None):
    """Estimate AR coefficients for one channel via OLS."""
    T = len(y)
    p = ar_order
    if T < 2 * p + 10:
        return np.zeros(p), max(float(np.var(y)), 1e-8)

    Y = y[p:]
    X = np.column_stack([y[p - k - 1:T - k - 1] for k in range(p)])
    if valid is not None:
        mask = valid[p:]
        for k in range(p):
            mask = mask & valid[p - k - 1:T - k - 1]
        Y, X = Y[mask], X[mask]
    if len(Y) < p + 5:
        return np.zeros(p), max(float(np.var(y)), 1e-8)

    XtX = X.T @ X + 1e-6 * np.eye(p)
    a = np.linalg.solve(XtX, X.T @ Y)
    return a, max(float(np.var(Y - X @ a)), 1e-8)


def _kim_filter_single_channel(y, x_basis, ar_order=3,
                                p_stay_coupled=0.95,
                                p_stay_uncoupled=0.95,
                                Q_coeff=1e-5,
                                em_iterations=5,
                                valid=None,
                                complexity_penalty=None):
    """Run Kim filter + smoother on one channel (standalone version)."""
    T = len(y)
    nb = x_basis.shape[1]

    z_src = x_basis.copy()
    col_rms = np.sqrt(np.mean(z_src ** 2, axis=0))
    col_rms = np.maximum(col_rms, 1e-8)
    z_src /= col_rms

    a, sigma2_ar = _estimate_ar(y, ar_order, valid)
    y_res = y.copy()
    for k in range(ar_order):
        y_res[ar_order:] -= a[k] * y[ar_order - k - 1:T - k - 1]
    y_res[:ar_order] = 0.0

    sigma2_0 = sigma2_ar
    sigma2_1 = sigma2_ar
    A = np.array([[p_stay_uncoupled, 1 - p_stay_uncoupled],
                  [1 - p_stay_coupled, p_stay_coupled]])
    Q = Q_coeff * np.eye(nb)
    P_init = (sigma2_ar / nb) * np.eye(nb)
    P_max_diag = 10.0 * sigma2_ar / nb
    if complexity_penalty is None:
        complexity_penalty = 0.0

    b_hat = np.zeros(nb)
    P_hat = P_init.copy()
    innov_sq_r1 = np.zeros(T)

    for em_iter in range(em_iterations):
        xi_filt = np.zeros((T, 2))
        xi_filt[0] = [0.5, 0.5]
        b_filt = np.zeros((T, 2, nb))
        P_filt = np.zeros((T, 2, nb, nb))
        for j in range(2):
            b_filt[0, j] = b_hat.copy()
            P_filt[0, j] = P_hat.copy()

        for t in range(1, T):
            if valid is not None and not valid[t]:
                xi_filt[t] = xi_filt[t-1]; b_filt[t] = b_filt[t-1]; P_filt[t] = P_filt[t-1]
                continue
            yt = y_res[t]; zt = z_src[t]
            xi_pred = A.T @ xi_filt[t-1]; xi_pred = np.maximum(xi_pred, 1e-10)
            log_f = np.zeros(2)
            for j in range(2):
                b_pred = b_filt[t-1, j].copy()
                P_pred = P_filt[t-1, j] + Q
                P_diag = np.diag(P_pred)
                if np.any(P_diag > P_max_diag):
                    scale = np.minimum(P_max_diag / np.maximum(P_diag, 1e-10), 1.0)
                    P_pred = P_pred * np.sqrt(np.outer(scale, scale))
                if j == 0:
                    innov = yt; F = sigma2_0
                    b_filt[t, 0] = b_pred; P_filt[t, 0] = P_pred
                else:
                    innov = yt - zt @ b_pred
                    F = zt @ P_pred @ zt + sigma2_1
                    K = P_pred @ zt / F
                    b_filt[t, 1] = b_pred + K * innov
                    P_filt[t, 1] = P_pred - np.outer(K, K) * F
                    innov_sq_r1[t] = innov ** 2
                log_f[j] = -0.5 * (np.log(2*np.pi*max(F, 1e-20)) + innov**2/max(F, 1e-20))
            log_f[1] -= complexity_penalty
            log_joint = log_f + np.log(xi_pred)
            log_joint -= log_joint.max()
            joint = np.exp(log_joint); total = joint.sum()
            xi_filt[t] = joint / max(total, 1e-20)
            for j_to in range(2):
                w = np.array([A[jf, j_to] * xi_filt[t-1, jf] for jf in range(2)])
                ws = w.sum()
                if ws < 1e-10: continue
                w /= ws
                b_m = w[0]*b_filt[t,0] + w[1]*b_filt[t,1]
                P_m = np.zeros((nb, nb))
                for jf in range(2):
                    d = b_filt[t, jf] - b_m
                    P_m += w[jf] * (P_filt[t, jf] + np.outer(d, d))
                b_filt[t, j_to] = b_m; P_filt[t, j_to] = P_m

        xi_smooth = np.zeros((T, 2)); xi_smooth[T-1] = xi_filt[T-1]
        for t in range(T-2, -1, -1):
            xp = A.T @ xi_filt[t]; xp = np.maximum(xp, 1e-10)
            ratio = xi_smooth[t+1] / xp
            for j in range(2):
                xi_smooth[t, j] = xi_filt[t, j] * (A[j,0]*ratio[0] + A[j,1]*ratio[1])
            s = xi_smooth[t].sum()
            if s > 0: xi_smooth[t] /= s
            else: xi_smooth[t] = xi_filt[t]

        posterior = np.clip(xi_smooth[:, 1], 0.0, 1.0)
        w1 = posterior; w0 = 1.0 - w1
        sigma2_0 = max(float((w0 * y_res**2).sum() / max(w0.sum(), 1.0)), 1e-8)
        sigma2_1 = max(float((w1 * innov_sq_r1).sum() / max(w1.sum(), 1.0)), 1e-8)
        if T > 2:
            n00 = float((w0[:-1]*w0[1:]).sum()); n01 = float((w0[:-1]*w1[1:]).sum())
            n10 = float((w1[:-1]*w0[1:]).sum()); n11 = float((w1[:-1]*w1[1:]).sum())
            if n00+n01 > 1: A[0,0] = np.clip(n00/(n00+n01), 0.5, 0.999); A[0,1] = 1-A[0,0]
            if n10+n11 > 1: A[1,1] = np.clip(n11/(n10+n11), 0.5, 0.999); A[1,0] = 1-A[1,1]

    return posterior, {
        'sigma2_0': sigma2_0, 'sigma2_1': sigma2_1,
        'sigma2_ratio': sigma2_1/max(sigma2_0, 1e-8),
        'coupling_fraction': float(np.mean(posterior > 0.5)),
        'transition_matrix': A,
    }


# =========================================================================
# Multivariate Kim filter: shared regime switch across channels
# =========================================================================

def kim_filter_multivariate(y_mc, x_basis_mc, ar_order=3,
                             p_stay_coupled=0.95,
                             p_stay_uncoupled=0.95,
                             Q_coeff=1e-5,
                             em_iterations=5,
                             valid=None,
                             complexity_penalty=None):
    """Multivariate Kim filter with SHARED regime switch across C channels.

    Vectorized across channels at each timestep via batched numpy.
    Single latent state S(t) governs all channels simultaneously.
    Joint observation likelihood = product (sum of log) across channels.

    Args:
        y_mc: (C, T) multichannel target at native rate.
        x_basis_mc: (C, T, nb) per-channel basis-convolved source.
        ar_order: AR order for target.
        p_stay_coupled, p_stay_uncoupled: transition probs.
        Q_coeff: coefficient random walk variance.
        em_iterations: EM iterations.
        valid: (T,) boolean mask.
        complexity_penalty: penalty for regime 1.

    Returns:
        posterior: (T,) shared regime posterior P(coupled | data).
        params: dict with diagnostics.
    """
    C, T = y_mc.shape
    nb = x_basis_mc.shape[2]

    # --- Pre-estimate AR and compute residuals per channel ---
    y_res_mc = np.zeros((C, T))
    sigma2_ar_mc = np.zeros(C)
    z_src_mc = np.zeros((C, T, nb))

    for c in range(C):
        z = x_basis_mc[c].copy()
        col_rms = np.maximum(np.sqrt(np.mean(z**2, axis=0)), 1e-8)
        z /= col_rms
        z_src_mc[c] = z

        a, sigma2_ar = _estimate_ar(y_mc[c], ar_order, valid)
        sigma2_ar_mc[c] = sigma2_ar
        y_res = y_mc[c].copy()
        for k in range(ar_order):
            y_res[ar_order:] -= a[k] * y_mc[c, ar_order-k-1:T-k-1]
        y_res[:ar_order] = 0.0
        y_res_mc[c] = y_res

    # --- Initialize ---
    sigma2_0 = sigma2_ar_mc.copy()  # (C,)
    sigma2_1 = sigma2_ar_mc.copy()  # (C,)
    A = np.array([[p_stay_uncoupled, 1 - p_stay_uncoupled],
                  [1 - p_stay_coupled, p_stay_coupled]])
    Q = Q_coeff * np.eye(nb)  # (nb, nb)
    P_max = 10.0 * sigma2_ar_mc / nb  # (C,)

    if complexity_penalty is None:
        complexity_penalty = 0.0

    # Batched Kalman state: (C, 2, nb) means, (C, 2, nb, nb) covs
    b = np.zeros((C, 2, nb))
    P = np.zeros((C, 2, nb, nb))
    for c in range(C):
        for j in range(2):
            P[c, j] = (sigma2_ar_mc[c] / nb) * np.eye(nb)

    innov_sq_r1 = np.zeros((C, T))

    # --- EM iterations ---
    for em_iter in range(em_iterations):
        xi_filt = np.zeros((T, 2))
        xi_filt[0] = [0.5, 0.5]

        # Storage for smoother + M-step (only need xi_filt history)
        xi_filt_all = np.zeros((T, 2))
        xi_filt_all[0] = xi_filt[0]

        # Reset Kalman states
        b_prev = b.copy()  # (C, 2, nb)
        P_prev = P.copy()  # (C, 2, nb, nb)

        # Store b_filt for regime 1 (needed for M-step innov)
        b_filt_r1 = np.zeros((C, T, nb))

        for t in range(1, T):
            if valid is not None and not valid[t]:
                xi_filt_all[t] = xi_filt_all[t-1]
                continue

            yt = y_res_mc[:, t]    # (C,)
            zt = z_src_mc[:, t, :]  # (C, nb)

            xi_pred = A.T @ xi_filt_all[t-1]
            xi_pred = np.maximum(xi_pred, 1e-10)

            joint_log_f = np.zeros(2)

            for j in range(2):
                # Predict: (C, nb) and (C, nb, nb)
                b_pred = b_prev[:, j, :]          # (C, nb)
                P_pred = P_prev[:, j, :, :] + Q   # (C, nb, nb)

                # Cap P diagonal per channel
                for c in range(C):
                    pd = np.diag(P_pred[c])
                    if np.any(pd > P_max[c]):
                        sc = np.minimum(P_max[c] / np.maximum(pd, 1e-10), 1.0)
                        P_pred[c] *= np.sqrt(np.outer(sc, sc))

                if j == 0:
                    # Regime 0: innovation = yt, F = sigma2_0
                    innov = yt                      # (C,)
                    F = sigma2_0                    # (C,)
                    b_prev[:, 0, :] = b_pred
                    P_prev[:, 0, :, :] = P_pred
                else:
                    # Regime 1: innovation = yt - zt @ b_pred per channel
                    # innov_c = yt_c - sum(zt_c * b_pred_c)
                    innov = yt - np.sum(zt * b_pred, axis=1)  # (C,)
                    # F_c = zt_c @ P_pred_c @ zt_c + sigma2_1_c
                    F = np.einsum('ci,cij,cj->c', zt, P_pred, zt) + sigma2_1  # (C,)
                    # Kalman gain: K_c = P_pred_c @ zt_c / F_c → (C, nb)
                    Pz = np.einsum('cij,cj->ci', P_pred, zt)  # (C, nb)
                    K = Pz / F[:, None]                         # (C, nb)
                    # Update
                    b_prev[:, 1, :] = b_pred + K * innov[:, None]
                    # P_update = P_pred - K @ K' * F
                    P_prev[:, 1, :, :] = P_pred - (
                        K[:, :, None] * K[:, None, :] * F[:, None, None])
                    innov_sq_r1[:, t] = innov ** 2
                    b_filt_r1[:, t, :] = b_prev[:, 1, :]

                # Sum log-likelihood across channels
                F_safe = np.maximum(F, 1e-20)
                joint_log_f[j] += np.sum(
                    -0.5 * (np.log(2 * np.pi * F_safe) + innov**2 / F_safe))

            joint_log_f[1] -= complexity_penalty

            # Hamilton filter
            log_joint = joint_log_f + np.log(xi_pred)
            log_joint -= log_joint.max()
            joint = np.exp(log_joint)
            total = joint.sum()
            xi_filt_all[t] = joint / max(total, 1e-20)

            # Kim moment matching (vectorized across C)
            for j_to in range(2):
                w0 = A[0, j_to] * xi_filt_all[t-1, 0]
                w1 = A[1, j_to] * xi_filt_all[t-1, 1]
                ws = w0 + w1
                if ws < 1e-10:
                    continue
                w0 /= ws; w1 /= ws
                b_m = w0 * b_prev[:, 0, :] + w1 * b_prev[:, 1, :]  # (C, nb)
                # P_m = w0*(P0 + d0@d0') + w1*(P1 + d1@d1')
                d0 = b_prev[:, 0, :] - b_m  # (C, nb)
                d1 = b_prev[:, 1, :] - b_m
                P_m = (w0 * (P_prev[:, 0] + d0[:, :, None] * d0[:, None, :]) +
                       w1 * (P_prev[:, 1] + d1[:, :, None] * d1[:, None, :]))
                b_prev[:, j_to, :] = b_m
                P_prev[:, j_to, :, :] = P_m

        # --- Backward smoother ---
        xi_smooth = np.zeros((T, 2))
        xi_smooth[T-1] = xi_filt_all[T-1]
        for t in range(T-2, -1, -1):
            xp = A.T @ xi_filt_all[t]
            xp = np.maximum(xp, 1e-10)
            ratio = xi_smooth[t+1] / xp
            for j in range(2):
                xi_smooth[t, j] = xi_filt_all[t, j] * (
                    A[j, 0] * ratio[0] + A[j, 1] * ratio[1])
            s = xi_smooth[t].sum()
            if s > 0:
                xi_smooth[t] /= s
            else:
                xi_smooth[t] = xi_filt_all[t]

        posterior = np.clip(xi_smooth[:, 1], 0.0, 1.0)

        # --- EM M-step ---
        w1 = posterior; w0 = 1.0 - w1
        w0s = max(w0.sum(), 1.0); w1s = max(w1.sum(), 1.0)
        sigma2_0 = np.maximum((w0[None, :] * y_res_mc**2).sum(axis=1) / w0s, 1e-8)
        sigma2_1 = np.maximum((w1[None, :] * innov_sq_r1).sum(axis=1) / w1s, 1e-8)

        if T > 2:
            n00 = float((w0[:-1]*w0[1:]).sum())
            n01 = float((w0[:-1]*w1[1:]).sum())
            n10 = float((w1[:-1]*w0[1:]).sum())
            n11 = float((w1[:-1]*w1[1:]).sum())
            if n00+n01 > 1:
                A[0,0] = np.clip(n00/(n00+n01), 0.5, 0.999); A[0,1] = 1-A[0,0]
            if n10+n11 > 1:
                A[1,1] = np.clip(n11/(n10+n11), 0.5, 0.999); A[1,0] = 1-A[1,1]

    # --- Output ---
    coupling_frac = float(np.mean(posterior > 0.5))
    sigma2_ratio_mc = sigma2_1 / np.maximum(sigma2_0, 1e-8)

    params = {
        'coupling_fraction': coupling_frac,
        'coupling_fraction_mc': np.full(C, coupling_frac),
        'sigma2_ratio': sigma2_ratio_mc,
        'transition_matrix': A,
        'per_channel': [{'sigma2_0': sigma2_0[c], 'sigma2_1': sigma2_1[c],
                         'sigma2_ratio': sigma2_ratio_mc[c]} for c in range(C)],
    }
    return posterior, params


# Legacy wrapper
def kim_filter_batched(y_mc, x_basis_mc, ar_order=3,
                       p_stay_coupled=0.95, p_stay_uncoupled=0.95,
                       Q_coeff=1e-5, em_iterations=5, valid=None,
                       n_jobs=-1, complexity_penalty=None,
                       multivariate=True):
    """Run Kim filter. Default: multivariate (shared regime switch)."""
    if multivariate:
        posterior, params = kim_filter_multivariate(
            y_mc, x_basis_mc, ar_order=ar_order,
            p_stay_coupled=p_stay_coupled,
            p_stay_uncoupled=p_stay_uncoupled,
            Q_coeff=Q_coeff, em_iterations=em_iterations,
            valid=valid, complexity_penalty=complexity_penalty)
        # Return as (C, T) for compatibility (shared posterior repeated)
        C = y_mc.shape[0]
        posterior_mc = np.tile(posterior, (C, 1))
        return posterior_mc, params
    else:
        # Independent per-channel (legacy)
        C, T = y_mc.shape
        results = Parallel(n_jobs=n_jobs)(
            delayed(_kim_filter_single_channel)(
                y_mc[c], x_basis_mc[c], ar_order=ar_order,
                p_stay_coupled=p_stay_coupled,
                p_stay_uncoupled=p_stay_uncoupled,
                Q_coeff=Q_coeff, em_iterations=em_iterations,
                valid=valid, complexity_penalty=complexity_penalty)
            for c in range(C))
        posterior_mc = np.array([r[0] for r in results])
        params_list = [r[1] for r in results]
        params = {
            'coupling_fraction_mc': np.array([p['coupling_fraction'] for p in params_list]),
            'sigma2_ratio': np.array([p['sigma2_ratio'] for p in params_list]),
            'per_channel': params_list,
        }
        return posterior_mc, params
