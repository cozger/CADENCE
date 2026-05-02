"""Multivariate S-map for native-rate interpersonal coupling.

Implements Deyle & Sugihara 2016 ("Tracking and forecasting ecosystem
interactions in real time", Nature) for inter-brain bandpassed signals:

  X(t) = [1, p1(t), p1(t-tau), ..., p1(t-(E-1)tau),
              p2(t), p2(t-tau), ..., p2(t-(E-1)tau)]      (2E + 1 dims)

For each query time t, predict p2(t+delta) via locally weighted linear
regression. The 2E coefficients on p1's lags ARE the time-varying directed
coupling p1->p2; symmetrically for p2->p1 by predicting p1(t+delta).

GPU-batched via torch.cdist + torch.linalg.solve, with memory-efficient
Theiler masking (scatter, not full pairwise time-difference matrix) and
einsum-based weighted normal equations.

The library uses every native-rate sample (full information). Queries are
decimated by ``query_stride`` to keep output size and runtime tractable.
"""

import numpy as np
import torch


# ───────────────────────────────────────────────────────────────────────
#  Library construction
# ───────────────────────────────────────────────────────────────────────

def _build_joint_library(p1, p2, E, tau, delta):
    """Build the joint embedding library at native rate.

    Returns:
        X_lib: (N_lib, 2E + 1) state matrix WITH intercept column.
        y2_lib: (N_lib,) target = p2[t+delta] (forward direction).
        y1_lib: (N_lib,) target = p1[t+delta] (reverse direction).
        t_lib: (N_lib,) integer time indices into p1/p2.
    """
    N = p1.shape[0]
    t_min = (E - 1) * tau
    t_max = N - delta
    if t_max <= t_min:
        raise ValueError(
            f"Signal too short for E={E}, tau={tau}, delta={delta} "
            f"(need >= {t_min + delta + 1} samples, got {N})")
    t_lib = torch.arange(t_min, t_max, device=p1.device)

    cols = [torch.ones_like(t_lib, dtype=p1.dtype)]
    for k in range(E):
        cols.append(p1[t_lib - k * tau])
    for k in range(E):
        cols.append(p2[t_lib - k * tau])
    X_lib = torch.stack(cols, dim=1)
    y2_lib = p2[t_lib + delta]
    y1_lib = p1[t_lib + delta]
    return X_lib, y2_lib, y1_lib, t_lib


def _apply_theiler_mask_(W, q_local_idx, theiler):
    """Zero out W (in place) within ±theiler of each query's library index.

    Args:
        W: (B, N_lib) weight tensor on device, modified in place.
        q_local_idx: (B,) int — library-local indices of the queries.
        theiler: ± exclusion (int).
    """
    B, N = W.shape
    if theiler <= 0:
        return W
    offsets = torch.arange(-theiler, theiler + 1, device=W.device)
    j = (q_local_idx.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N - 1)  # (B, 2T+1)
    b = torch.arange(B, device=W.device).unsqueeze(1).expand_as(j)
    W[b.reshape(-1), j.reshape(-1)] = 0.0
    return W


# ───────────────────────────────────────────────────────────────────────
#  Single-direction multivariate S-map
# ───────────────────────────────────────────────────────────────────────

def _smap_both_directions(
    X_lib, y_fwd, y_rev, t_lib, q_indices, theta, theiler_samples,
    batch_size=64, lib_chunk=16384, ridge=1e-6,
):
    """Multivariate S-map for BOTH directions in one pass.

    Predicts y_fwd (=p2[t+δ]) and y_rev (=p1[t+δ]) simultaneously, sharing
    the cdist + weight computation. Halves the GPU work versus calling
    forward and reverse separately.

    Memory-managed: chunks the library dimension when accumulating XTWX,
    XTWy_fwd, XTWy_rev so peak per-chunk live intermediate is
    (B, lib_chunk, P) ~ tens of MB.

    Returns:
        beta_fwd: (N_q, P), beta_rev: (N_q, P),
        cond: (N_q,) shared condition number,
        pred_fwd: (N_q,), pred_rev: (N_q,).
    """
    device = X_lib.device
    dtype = X_lib.dtype
    N_lib, P = X_lib.shape
    N_q = q_indices.shape[0]
    beta_fwd_all = torch.zeros((N_q, P), dtype=dtype, device=device)
    beta_rev_all = torch.zeros((N_q, P), dtype=dtype, device=device)
    cond_all = torch.zeros(N_q, dtype=dtype, device=device)
    pred_fwd_all = torch.zeros(N_q, dtype=dtype, device=device)
    pred_rev_all = torch.zeros(N_q, dtype=dtype, device=device)

    eye_P = torch.eye(P, dtype=dtype, device=device) * ridge

    if theiler_samples > 0:
        tl_offsets = torch.arange(-theiler_samples, theiler_samples + 1,
                                  device=device)
    else:
        tl_offsets = None

    for b0 in range(0, N_q, batch_size):
        b1 = min(b0 + batch_size, N_q)
        q_local = q_indices[b0:b1]
        Xq = X_lib[q_local]                          # (B, P)
        B = Xq.shape[0]

        # ── Pass 1: chunked d_mean computation ──
        d_sum = torch.zeros(B, dtype=dtype, device=device)
        d_count = torch.zeros(B, dtype=dtype, device=device)
        for c0 in range(0, N_lib, lib_chunk):
            c1 = min(c0 + lib_chunk, N_lib)
            X_c = X_lib[c0:c1]
            D_c = torch.cdist(Xq, X_c)
            if tl_offsets is not None:
                j = (q_local.unsqueeze(1) + tl_offsets.unsqueeze(0))
                in_chunk = (j >= c0) & (j < c1)
                if in_chunk.any():
                    j_local = (j - c0).clamp(0, c1 - c0 - 1)
                    bb = torch.arange(B, device=device).unsqueeze(1).expand_as(j_local)
                    D_c[bb[in_chunk], j_local[in_chunk]] = float('nan')
            valid = torch.isfinite(D_c)
            D_finite = torch.where(valid, D_c, torch.zeros_like(D_c))
            d_sum = d_sum + D_finite.sum(dim=1)
            d_count = d_count + valid.sum(dim=1).to(dtype)
            del D_c, D_finite, valid
        d_mean = (d_sum / d_count.clamp(min=1.0)).clamp(min=1e-12)

        # ── Pass 2: chunked XTWX, XTWy_fwd, XTWy_rev ──
        XTWX = torch.zeros((B, P, P), dtype=dtype, device=device)
        XTWy_f = torch.zeros((B, P), dtype=dtype, device=device)
        XTWy_r = torch.zeros((B, P), dtype=dtype, device=device)

        for c0 in range(0, N_lib, lib_chunk):
            c1 = min(c0 + lib_chunk, N_lib)
            X_c = X_lib[c0:c1]
            yf_c = y_fwd[c0:c1]
            yr_c = y_rev[c0:c1]
            D_c = torch.cdist(Xq, X_c)
            W_c = torch.exp(-theta * D_c / d_mean.unsqueeze(1))
            if tl_offsets is not None:
                j = (q_local.unsqueeze(1) + tl_offsets.unsqueeze(0))
                in_chunk = (j >= c0) & (j < c1)
                if in_chunk.any():
                    j_local = (j - c0).clamp(0, c1 - c0 - 1)
                    bb = torch.arange(B, device=device).unsqueeze(1).expand_as(j_local)
                    W_c[bb[in_chunk], j_local[in_chunk]] = 0.0
            XW = X_c.unsqueeze(0) * W_c.unsqueeze(2)
            XTWX = XTWX + torch.bmm(
                XW.transpose(1, 2),
                X_c.unsqueeze(0).expand(B, -1, -1),
            )
            XTWy_f = XTWy_f + (XW * yf_c.unsqueeze(0).unsqueeze(2)).sum(dim=1)
            XTWy_r = XTWy_r + (XW * yr_c.unsqueeze(0).unsqueeze(2)).sum(dim=1)
            del D_c, W_c, XW

        XTWX_reg = XTWX + eye_P.unsqueeze(0)
        cond_b = torch.linalg.cond(XTWX_reg)
        try:
            beta_f = torch.linalg.solve(XTWX_reg, XTWy_f.unsqueeze(2)).squeeze(2)
            beta_r = torch.linalg.solve(XTWX_reg, XTWy_r.unsqueeze(2)).squeeze(2)
        except torch._C._LinAlgError:
            beta_f = torch.linalg.lstsq(XTWX_reg, XTWy_f.unsqueeze(2)).solution.squeeze(2)
            beta_r = torch.linalg.lstsq(XTWX_reg, XTWy_r.unsqueeze(2)).solution.squeeze(2)

        pred_f = (Xq * beta_f).sum(dim=1)
        pred_r = (Xq * beta_r).sum(dim=1)
        beta_fwd_all[b0:b1] = beta_f
        beta_rev_all[b0:b1] = beta_r
        cond_all[b0:b1] = cond_b
        pred_fwd_all[b0:b1] = pred_f
        pred_rev_all[b0:b1] = pred_r

        del XTWX, XTWy_f, XTWy_r, XTWX_reg
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return beta_fwd_all, beta_rev_all, cond_all, pred_fwd_all, pred_rev_all


# ───────────────────────────────────────────────────────────────────────
#  Public: simplex projection for E selection
# ───────────────────────────────────────────────────────────────────────

def simplex_select_E(
    sig, tau, delta, E_grid=(2, 3, 4, 5), theiler_samples=78,
    query_stride=64, batch_size=64, lib_chunk=32768,
    device=None, dtype=torch.float32,
):
    """Univariate simplex projection: pick E maximizing 1-step forecast skill.

    Memory-managed via library chunking. The cdist result for each batch is
    accumulated as a top-K running set so we never hold the full (B, N_lib)
    distance matrix.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sig_t = torch.as_tensor(np.asarray(sig), dtype=dtype, device=device)
    N = sig_t.shape[0]
    skill = {}

    for E in E_grid:
        K = E + 1
        t_min = (E - 1) * tau
        t_max = N - delta
        if t_max - t_min < 4 * K:
            skill[E] = float('nan')
            continue
        t_idx = torch.arange(t_min, t_max, device=device)
        cols = [sig_t[t_idx - k * tau] for k in range(E)]
        X = torch.stack(cols, dim=1)
        y = sig_t[t_idx + delta]
        N_lib = X.shape[0]

        q_idx = torch.arange(0, N_lib, query_stride, device=device)
        N_q = q_idx.shape[0]
        preds = torch.zeros(N_q, dtype=dtype, device=device)
        truths = y[q_idx]

        if theiler_samples > 0:
            tl_offsets = torch.arange(-theiler_samples,
                                       theiler_samples + 1, device=device)
        else:
            tl_offsets = None

        for b0 in range(0, N_q, batch_size):
            b1 = min(b0 + batch_size, N_q)
            ql = q_idx[b0:b1]
            Xq = X[ql]
            B = Xq.shape[0]
            # Running top-K across library chunks
            best_d = torch.full((B, K), float('inf'), dtype=dtype, device=device)
            best_y = torch.zeros((B, K), dtype=dtype, device=device)

            for c0 in range(0, N_lib, lib_chunk):
                c1 = min(c0 + lib_chunk, N_lib)
                X_c = X[c0:c1]
                y_c = y[c0:c1]
                D_c = torch.cdist(Xq, X_c)             # (B, Cc)
                if tl_offsets is not None:
                    j = (ql.unsqueeze(1) + tl_offsets.unsqueeze(0))
                    in_chunk = (j >= c0) & (j < c1)
                    if in_chunk.any():
                        j_local = (j - c0).clamp(0, c1 - c0 - 1)
                        bb = torch.arange(B, device=device).unsqueeze(1).expand_as(j_local)
                        sel = in_chunk
                        D_c[bb[sel], j_local[sel]] = float('inf')
                # Combine with running top-K and re-rank
                tk_d, tk_i = torch.topk(D_c, K, dim=1, largest=False)
                tk_y = y_c[tk_i]
                cand_d = torch.cat([best_d, tk_d], dim=1)             # (B, 2K)
                cand_y = torch.cat([best_y, tk_y], dim=1)
                ord_d, ord_i = torch.topk(cand_d, K, dim=1, largest=False)
                best_d = ord_d
                best_y = torch.gather(cand_y, 1, ord_i)
                del D_c, tk_d, tk_i, tk_y, cand_d, cand_y, ord_d, ord_i

            d_min = best_d[:, :1].clamp(min=1e-12)
            w = torch.exp(-best_d / d_min)
            w_norm = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
            preds[b0:b1] = (w_norm * best_y).sum(dim=1)
            del best_d, best_y, w, w_norm
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        valid = torch.isfinite(preds) & torch.isfinite(truths)
        if valid.sum() < 10:
            skill[E] = float('nan')
            continue
        p = preds[valid] - preds[valid].mean()
        t = truths[valid] - truths[valid].mean()
        denom = torch.sqrt((p * p).sum() * (t * t).sum()).clamp(min=1e-12)
        skill[E] = float((p * t).sum() / denom)
        del t_idx, X, y, preds, truths
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    valid_skill = {E: s for E, s in skill.items()
                   if s is not None and not (isinstance(s, float) and np.isnan(s))}
    if not valid_skill:
        E_best = 3
    else:
        E_best = max(valid_skill.keys(), key=lambda k: valid_skill[k])
    return {'E_best': E_best, 'skill_per_E': skill}


# ───────────────────────────────────────────────────────────────────────
#  Public: GPU multivariate S-map cross-coupling
# ───────────────────────────────────────────────────────────────────────

def gpu_smap_cross_coupling(
    p1_sig, p2_sig,
    fs=256.0, E=3, tau_samples=6, delta_samples=6,
    theta=1.0, theiler_samples=78,
    query_stride=32, batch_size=64, lib_chunk=16384,
    n_surrogates=200, seed=42,
    device=None, dtype=torch.float32,
):
    """Multivariate S-map directed coupling between two bandpassed signals.

    Args:
        p1_sig, p2_sig: (N,) numpy arrays at native sampling rate.
        fs: sampling rate.
        E: embedding dim per channel (joint state has 2E+1 dims with intercept).
        tau_samples: lag in samples (default 6 = 23 ms at 256 Hz, ~quarter
                     alpha cycle at 10 Hz).
        delta_samples: prediction horizon.
        theta: S-map locality weighting.
        theiler_samples: ± exclusion (default 78 ≈ 305 ms for alpha bandpass).
        query_stride: queries every K-th library sample (default 8 → 32 Hz at 256 Hz fs).
                      Library uses every native-rate sample regardless.
        batch_size: queries per GPU batch.
        n_surrogates: 200 circular-shift surrogates for z-scoring (0 to skip).
        seed: random seed for surrogates.

    Returns:
        dict with t_query, beta_p1_to_p2, beta_p2_to_p1, coupling_*, z_*,
        cond_*, pred_skill_*, plus echoed hyperparams.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p1 = torch.as_tensor(np.asarray(p1_sig), dtype=dtype, device=device)
    p2 = torch.as_tensor(np.asarray(p2_sig), dtype=dtype, device=device)
    N = min(p1.shape[0], p2.shape[0])
    p1, p2 = p1[:N], p2[:N]

    X_lib, y2_lib, y1_lib, t_lib = _build_joint_library(p1, p2, E, tau_samples, delta_samples)
    N_lib = X_lib.shape[0]

    # Decimated query indices into the library
    q_idx = torch.arange(0, N_lib, query_stride, device=device)

    beta_fwd, beta_rev, cond_shared, pred_fwd, pred_rev = _smap_both_directions(
        X_lib, y2_lib, y1_lib, t_lib, q_idx, theta, theiler_samples,
        batch_size=batch_size, lib_chunk=lib_chunk,
    )
    cond_fwd = cond_shared
    cond_rev = cond_shared

    beta_p1_to_p2 = beta_fwd[:, 1:1 + E]
    beta_p2_to_p1 = beta_rev[:, 1 + E:1 + 2 * E]
    summary_idx = 1 if E >= 2 else 0
    coupling_fwd = beta_p1_to_p2[:, summary_idx]
    coupling_rev = beta_p2_to_p1[:, summary_idx]

    # Predict skill on diagonal (truth at the query times)
    truth_fwd = y2_lib[q_idx]
    truth_rev = y1_lib[q_idx]

    def _r(p, t):
        v = torch.isfinite(p) & torch.isfinite(t)
        if v.sum() < 10:
            return float('nan')
        p_, t_ = p[v] - p[v].mean(), t[v] - t[v].mean()
        denom = torch.sqrt((p_ * p_).sum() * (t_ * t_).sum()).clamp(min=1e-12)
        return float((p_ * t_).sum() / denom)

    out = dict(
        t_query=t_lib[q_idx].cpu().numpy(),
        beta_p1_to_p2=beta_p1_to_p2.cpu().numpy(),
        beta_p2_to_p1=beta_p2_to_p1.cpu().numpy(),
        coupling_p1_to_p2=coupling_fwd.cpu().numpy(),
        coupling_p2_to_p1=coupling_rev.cpu().numpy(),
        cond_forward=cond_fwd.cpu().numpy(),
        cond_reverse=cond_rev.cpu().numpy(),
        pred_skill_forward=_r(pred_fwd, truth_fwd),
        pred_skill_reverse=_r(pred_rev, truth_rev),
        E=E, tau_samples=tau_samples, delta_samples=delta_samples,
        theta=theta, theiler_samples=theiler_samples,
        query_stride=query_stride,
    )

    if n_surrogates > 0:
        out['z_p1_to_p2'], out['z_p2_to_p1'] = _surrogate_z(
            p1, p2, E, tau_samples, delta_samples, theta, theiler_samples,
            q_idx=q_idx, real_fwd=coupling_fwd, real_rev=coupling_rev,
            n_surrogates=n_surrogates, seed=seed,
            batch_size=batch_size, lib_chunk=lib_chunk,
            device=device, dtype=dtype,
        )
    else:
        out['z_p1_to_p2'] = np.zeros_like(out['coupling_p1_to_p2'])
        out['z_p2_to_p1'] = np.zeros_like(out['coupling_p2_to_p1'])

    return out


def _surrogate_z(
    p1, p2, E, tau, delta, theta, theiler_samples,
    q_idx, real_fwd, real_rev,
    n_surrogates, seed, batch_size, lib_chunk, device, dtype,
):
    """Welford z-scores from circular-shift surrogates of p2."""
    N = p1.shape[0]
    rng = np.random.default_rng(seed)
    min_shift = max(1280, N // 50)        # ≥5s at 256 Hz, or 2% of N
    if min_shift >= N:
        min_shift = max(1, N // 4)
    max_shift = N - min_shift
    if max_shift <= min_shift:
        max_shift = min_shift + 1

    N_q = q_idx.shape[0]
    summary_idx = 1 if E >= 2 else 0
    p1_lag_col = 1 + summary_idx                # column in beta for p1 at lag tau
    p2_lag_col = 1 + E + summary_idx            # column in beta for p2 at lag tau

    mean_fwd = torch.zeros(N_q, dtype=dtype, device=device)
    m2_fwd = torch.zeros(N_q, dtype=dtype, device=device)
    mean_rev = torch.zeros(N_q, dtype=dtype, device=device)
    m2_rev = torch.zeros(N_q, dtype=dtype, device=device)

    real_fwd_t = real_fwd.to(device)
    real_rev_t = real_rev.to(device)

    for si in range(n_surrogates):
        shift = int(rng.integers(min_shift, max_shift))
        p2_s = torch.roll(p2, shift, dims=0)

        X_s, y2_s, y1_s, t_s = _build_joint_library(p1, p2_s, E, tau, delta)
        beta_fwd_s, beta_rev_s, _, _, _ = _smap_both_directions(
            X_s, y2_s, y1_s, t_s, q_idx, theta, theiler_samples,
            batch_size=batch_size, lib_chunk=lib_chunk,
        )
        c_fwd = beta_fwd_s[:, p1_lag_col]
        c_rev = beta_rev_s[:, p2_lag_col]

        n_i = si + 1
        d_fwd = c_fwd - mean_fwd
        mean_fwd = mean_fwd + d_fwd / n_i
        m2_fwd = m2_fwd + d_fwd * (c_fwd - mean_fwd)

        d_rev = c_rev - mean_rev
        mean_rev = mean_rev + d_rev / n_i
        m2_rev = m2_rev + d_rev * (c_rev - mean_rev)

        del X_s, y2_s, y1_s, t_s, beta_fwd_s, beta_rev_s, c_fwd, c_rev, p2_s
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    n_eff = max(n_surrogates - 1, 1)
    std_fwd = torch.sqrt(m2_fwd / n_eff).clamp(min=1e-10)
    std_rev = torch.sqrt(m2_rev / n_eff).clamp(min=1e-10)
    z_fwd = ((real_fwd_t - mean_fwd) / std_fwd).clamp(min=-10, max=10)
    z_rev = ((real_rev_t - mean_rev) / std_rev).clamp(min=-10, max=10)
    return z_fwd.cpu().numpy(), z_rev.cpu().numpy()
