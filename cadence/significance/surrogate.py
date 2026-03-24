"""Circular-shift surrogate significance testing for CADENCE coupling.

Key optimization: shift commutes with convolution.
  shift(phi_j * x) = phi_j * shift(x)
So we pre-compute basis-convolved source signals once, then just
circular-shift the convolved outputs for each surrogate.
"""

import numpy as np
import torch

from cadence.surrogates import (
    circular_shift_surrogate_batched, fourier_surrogate_gpu_batched,
)
from cadence.regression.ewls import EWLSSolver
from cadence.basis.design_matrix import DesignMatrixBuilder


def surrogate_significance(dm_builder, solver, source_signal, target_signal,
                           source_valid, target_valid, eval_times,
                           source_times, target_times, eval_rate,
                           n_surrogates=200, min_shift_frac=0.1, seed=42):
    """Compute surrogate p-values for one pathway.

    Optimization: circular shift commutes with linear convolution.
    We pre-compute the basis-convolved source, then for each surrogate
    we circular-shift the convolved output (cheap) rather than
    reconvolving shifted raw data (expensive).

    Args:
        dm_builder: DesignMatrixBuilder with basis functions.
        solver: EWLSSolver for time-varying regression.
        source_signal: (T_src, C_src) source features.
        target_signal: (T_tgt, C_tgt) target features.
        source_valid: (T_src,) boolean mask.
        target_valid: (T_tgt,) boolean mask.
        eval_times: (T_eval,) evaluation timestamps.
        source_times: (T_src,) source timestamps.
        target_times: (T_tgt,) target timestamps.
        eval_rate: Evaluation rate Hz.
        n_surrogates: Number of circular-shift surrogates.
        min_shift_frac: Minimum shift as fraction of T.
        seed: Random seed.

    Returns:
        p_values: (T_eval,) per-timepoint p-values.
        dr2_real: (T_eval,) real delta-R2.
        dr2_surrogates: (n_surrogates, T_eval) surrogate delta-R2 values.
    """
    device = dm_builder.device

    # Step 1: Real model
    X_full, y, valid = dm_builder.build(
        source_signal, target_signal,
        source_valid=source_valid, target_valid=target_valid,
        eval_times=eval_times, source_times=source_times,
        target_times=target_times, eval_rate=eval_rate,
    )
    X_restricted = dm_builder._build_ar_terms(
        target_signal, target_times, eval_times, eval_rate)

    dr2_real, _, _, _, _ = solver.solve_restricted(X_full, X_restricted, y, valid)
    dr2_real_np = dr2_real.cpu().numpy()

    # Step 2: Pre-compute basis-convolved source (the expensive part, done once)
    convolved, _ = dm_builder.convolve_source(source_signal, source_valid)
    # Resample to eval grid
    convolved_eval = dm_builder._resample_to_eval(
        convolved, source_times, eval_times)
    # Shape: (T_eval, C_src * n_basis)

    T_eval = len(eval_times)

    # Step 3: Generate surrogates by circular-shifting the convolved output
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Reshape for circular_shift: (1, C_conv, T_eval)
    conv_for_shift = convolved_eval.T.unsqueeze(0)  # (1, C_conv, T_eval)

    dr2_surrogates = np.zeros((n_surrogates, T_eval))

    # Process in batches to limit memory
    batch_size = min(50, n_surrogates)
    for batch_start in range(0, n_surrogates, batch_size):
        batch_end = min(batch_start + batch_size, n_surrogates)
        n_batch = batch_end - batch_start

        # Generate shifted convolved outputs
        shifted = circular_shift_surrogate_batched(
            conv_for_shift, n_batch, min_shift_frac=min_shift_frac,
            generator=gen)  # (n_batch, C_conv, T_eval)

        for k in range(n_batch):
            # Build surrogate full design matrix: shifted_conv + AR
            X_surr_basis = shifted[k].T  # (T_eval, C_conv)
            X_surr_full = torch.cat([X_surr_basis, X_restricted], dim=1)

            dr2_surr, _, _, _, _ = solver.solve_restricted(
                X_surr_full, X_restricted, y, valid)
            dr2_surrogates[batch_start + k] = dr2_surr.cpu().numpy()

    # Step 4: Compute p-values (fraction of surrogates >= real)
    dr2_real_expanded = dr2_real_np[None, :]  # (1, T_eval)
    p_values = np.mean(dr2_surrogates >= dr2_real_expanded, axis=0)

    # Minimum p-value = 1 / (n_surrogates + 1) for finite samples
    p_values = np.maximum(p_values, 1.0 / (n_surrogates + 1))

    return p_values, dr2_real_np, dr2_surrogates


def surrogate_pvalues_from_design(solver, X_augmented, X_restricted, y, valid,
                                   n_source_cols, dr2_real,
                                   n_surrogates=20, min_shift_frac=0.1,
                                   seed=42, smooth_samples=0,
                                   surrogate_method='circular_shift',
                                   n_basis=0, selected=None,
                                   feat_dr2_real=None,
                                   gpd_pool_half=50,
                                   gpd_threshold_quantile=0.9):
    """Per-timepoint p-values by circular-shifting source columns of X_augmented.

    Shifts only the first n_source_cols columns (basis-convolved source).
    Augmented terms (nonlinear, moderation) are rebuilt from the shifted
    source so that the surrogate model matches the real model structure.

    Both real and surrogate dR2 are smoothed identically before computing
    p-values, which reduces null variance and prevents sensitivity loss.

    Optionally computes per-feature p-values when n_basis, selected, and
    feat_dr2_real are provided. Uses beta energy decomposition from the
    surrogate EWLS solve (beta_full already computed, zero extra EWLS cost).

    Args:
        solver: EWLSSolver instance.
        X_augmented: (T_eval, p_full) full design matrix (source + aug + AR).
        X_restricted: (T_eval, p_ar) AR-only design matrix.
        y: (T_eval, C_tgt) target signal.
        valid: (T_eval,) boolean mask.
        n_source_cols: Number of basis-convolved source columns in X_augmented
            (before nonlinear/moderation augmentations).
        dr2_real: (T_eval,) real dR2 timecourse from Stage 2.
        n_surrogates: Number of circular-shift surrogates (default 20).
        min_shift_frac: Minimum shift as fraction of T.
        seed: Random seed.
        smooth_samples: If > 0, apply uniform smoothing before p-value computation.
        surrogate_method: 'circular_shift' or 'fourier_phase'.
        n_basis: Number of basis functions per feature (for per-feature decomposition).
        selected: List of selected feature indices (for per-feature decomposition).
        feat_dr2_real: Dict {feat_idx: (T,) dr2} from real solve.

    Returns:
        p_values: (T_eval,) per-timepoint p-values.
        dr2_surrogates: (n_surrogates, T_eval) surrogate dR2 timecourses.
        feat_pvalues: Dict {feat_idx: (T,) pvalues} or None if not computing.
    """
    device = X_augmented.device
    T_eval = X_augmented.shape[0]
    p_total = X_augmented.shape[1]
    p_ar = X_restricted.shape[1]

    # Per-feature p-value computation enabled?
    compute_feat = (n_basis > 0 and selected is not None
                    and feat_dr2_real is not None and len(selected) > 1)

    # Identify which columns are what
    # Layout: [source_basis(n_source_cols) | augmented | AR(p_ar)]
    # The augmented block sits between source and AR
    n_aug = p_total - n_source_cols - p_ar
    X_source = X_augmented[:, :n_source_cols]

    # Detect augmentation structure from X_augmented:
    # - Nonlinear: X_source ** 2 (same shape as source)
    # - Moderation: X_source * moderator (same shape per moderator)
    # We'll reconstruct these from shifted source columns.

    # Determine how many augmentation "blocks" of size n_source_cols exist
    if n_source_cols > 0:
        n_aug_blocks = n_aug // n_source_cols
    else:
        n_aug_blocks = 0

    # Extract moderator/nonlinear structure from original X_augmented
    # by computing the element-wise ratio of each aug block to source
    aug_multipliers = []
    for b in range(n_aug_blocks):
        start = n_source_cols + b * n_source_cols
        end = start + n_source_cols
        aug_block = X_augmented[:, start:end]

        # For nonlinear: multiplier = X_source (since aug = source * source)
        # For moderation: multiplier = moderator (since aug = source * mod)
        # Safe division: where source is zero, multiplier is zero
        safe_src = X_source.clone()
        safe_src[safe_src.abs() < 1e-10] = 1.0
        multiplier = aug_block / safe_src
        # Take column median to get the per-timepoint multiplier
        # (all source cols share the same moderator signal)
        multiplier = multiplier.median(dim=1, keepdim=True).values
        aug_multipliers.append(multiplier)

    # Reshape source for surrogate generation: (1, n_source_cols, T_eval)
    X_src_3d = X_source.T.unsqueeze(0)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dr2_surrogates = np.zeros((n_surrogates, T_eval))

    # Per-feature surrogate dR2 accumulator
    if compute_feat:
        feat_dr2_surr = {fi: np.zeros((n_surrogates, T_eval))
                         for fi in selected}

    batch_size = min(10, n_surrogates)
    for batch_start in range(0, n_surrogates, batch_size):
        batch_end = min(batch_start + batch_size, n_surrogates)
        n_batch = batch_end - batch_start

        if surrogate_method == 'fourier_phase':
            shifted = fourier_surrogate_gpu_batched(
                X_src_3d, n_batch, base_seed=seed + batch_start)
        else:
            shifted = circular_shift_surrogate_batched(
                X_src_3d, n_batch, min_shift_frac=min_shift_frac,
                generator=gen)  # (n_batch, n_source_cols, T_eval)

        for k in range(n_batch):
            X_src_shifted = shifted[k].T  # (T_eval, n_source_cols)

            # Rebuild augmented design matrix with shifted source
            parts = [X_src_shifted]
            for b, mult in enumerate(aug_multipliers):
                parts.append(X_src_shifted * mult)
            parts.append(X_restricted)
            X_surr_full = torch.cat(parts, dim=1)

            dr2_surr, _, _, beta_surr, _ = solver.solve_restricted(
                X_surr_full, X_restricted, y, valid)
            surr_idx = batch_start + k
            dr2_surr_np = dr2_surr.cpu().numpy()
            dr2_surrogates[surr_idx] = dr2_surr_np

            # Per-feature energy decomposition from surrogate beta
            if compute_feat and beta_surr is not None:
                beta_src = beta_surr[:, :n_source_cols, :]  # (T, n_src_cols, C)
                total_e = (beta_src ** 2).sum(dim=(1, 2)).cpu().numpy()
                for i, fi in enumerate(selected):
                    cs = i * n_basis
                    ce = cs + n_basis
                    fe = (beta_src[:, cs:ce, :] ** 2).sum(dim=(1, 2)).cpu().numpy()
                    feat_dr2_surr[fi][surr_idx] = dr2_surr_np * (fe / (total_e + 1e-20))

        del shifted
        torch.cuda.empty_cache()

    # Apply smoothing to both real and surrogate dR2 before p-value computation
    if smooth_samples > 1:
        from scipy.ndimage import uniform_filter1d
        dr2_real_s = uniform_filter1d(dr2_real, smooth_samples, mode='nearest')
        dr2_surr_s = np.array([
            uniform_filter1d(dr2_surrogates[i], smooth_samples, mode='nearest')
            for i in range(n_surrogates)
        ])
    else:
        dr2_real_s = dr2_real
        dr2_surr_s = dr2_surrogates

    # Per-timepoint p-values via GPD tail fitting on pooled surrogates
    from cadence.significance.gpd_pvalues import gpd_tail_pvalues
    p_values = gpd_tail_pvalues(dr2_real_s, dr2_surr_s,
                                pool_half=gpd_pool_half,
                                threshold_quantile=gpd_threshold_quantile)

    # Per-feature p-values
    feat_pvalues = None
    if compute_feat:
        if smooth_samples <= 1:
            from scipy.ndimage import uniform_filter1d
        feat_pvalues = {}
        for fi in selected:
            real_f = feat_dr2_real[fi]
            surr_f = feat_dr2_surr[fi]
            if smooth_samples > 1:
                real_f = uniform_filter1d(real_f, smooth_samples, mode='nearest')
                surr_f = np.array([
                    uniform_filter1d(surr_f[j], smooth_samples, mode='nearest')
                    for j in range(n_surrogates)
                ])
            fp = gpd_tail_pvalues(real_f, surr_f,
                                  pool_half=gpd_pool_half,
                                  threshold_quantile=gpd_threshold_quantile)
            feat_pvalues[fi] = fp

    return p_values, dr2_surrogates, feat_pvalues


def surrogate_pvalues_perchannel(solver, X_full_batch, X_restr_batch,
                                  y_batch, valid_batch, n_src_cols,
                                  dr2_real_perchannel, dr2_real_agg,
                                  n_surrogates=20, min_shift_frac=0.1,
                                  seed=42, smooth_samples=0,
                                  gpd_pool_half=50,
                                  gpd_threshold_quantile=0.9):
    """Per-timepoint p-values for matched-diagonal (per-channel) EWLS.

    All C_min channels share the same circular shift per surrogate so
    cross-channel coupling structure is preserved in the null.

    Args:
        solver: EWLSSolver instance.
        X_full_batch: (C_min, T, p_ch) batched full design matrices.
        X_restr_batch: (C_min, T, p_ar) batched AR-only matrices.
        y_batch: (C_min, T, 1) batched single-channel targets.
        valid_batch: (C_min, T) boolean mask, or None.
        n_src_cols: Number of source basis columns per channel (before
            nonlinear/moderation augmentation).
        dr2_real_perchannel: (C_min, T) real per-channel dR2.
        dr2_real_agg: (T,) aggregated real dR2 (mean across channels).
        n_surrogates: Number of circular-shift surrogates.
        min_shift_frac: Minimum shift as fraction of T.
        seed: Random seed.
        smooth_samples: Smoothing window (applied identically to real + null).
        gpd_pool_half: GPD temporal pooling half-width.
        gpd_threshold_quantile: GPD threshold quantile.

    Returns:
        p_values: (T,) per-timepoint p-values on aggregated dR2.
        dr2_null_flat: (K*T,) pooled null dR2 for null stats (mu_0, sigma_0).
        dr2_surr_perchannel: (n_surrogates, C_min, T) per-channel surrogate dR2.
    """
    device = X_full_batch.device
    C_min, T, p_ch = X_full_batch.shape
    p_ar = X_restr_batch.shape[2]

    # Detect augmentation blocks in X_full
    # Layout per channel: [src_basis(n_src_cols) | aug_blocks | AR(p_ar)]
    n_aug = p_ch - n_src_cols - p_ar
    n_aug_blocks = n_aug // n_src_cols if n_src_cols > 0 else 0

    # Extract per-channel augmentation multipliers from original X
    # (nonlinear = source, moderation = moderator signal)
    aug_multipliers = []
    if n_aug_blocks > 0:
        X_src = X_full_batch[:, :, :n_src_cols]  # (C_min, T, n_src)
        for b in range(n_aug_blocks):
            start = n_src_cols + b * n_src_cols
            end = start + n_src_cols
            aug_block = X_full_batch[:, :, start:end]  # (C_min, T, n_src)
            safe_src = X_src.clone()
            safe_src[safe_src.abs() < 1e-10] = 1.0
            mult = aug_block / safe_src
            # Per-timepoint multiplier (median across source cols)
            mult = mult.median(dim=2, keepdim=True).values  # (C_min, T, 1)
            aug_multipliers.append(mult)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    N = T
    min_shift = max(1, int(min_shift_frac * N))
    max_shift = N - min_shift
    if min_shift >= max_shift:
        min_shift, max_shift = 1, N - 1

    dr2_surr_agg = np.zeros((n_surrogates, T))
    dr2_surr_perchannel = np.zeros((n_surrogates, C_min, T))

    # Mega-batch: stack M surrogates into one EWLS solve for better GPU
    # utilization (C_min=15 → M*C_min=60 batch elements fills more warps).
    MEGA_BATCH = min(4, n_surrogates)

    # Pre-generate all shifts at once
    all_shifts = torch.randint(min_shift, max_shift + 1, (n_surrogates,),
                               generator=gen, device=device)

    X_src_orig = X_full_batch[:, :, :n_src_cols]  # (C_min, T, n_src)

    for kb in range(0, n_surrogates, MEGA_BATCH):
        M = min(MEGA_BATCH, n_surrogates - kb)

        # Build M shifted design matrices and stack into mega-batch
        X_surr_list = []
        for m in range(M):
            shift = int(all_shifts[kb + m].item())
            X_src_shifted = torch.roll(X_src_orig, shifts=shift, dims=1)
            parts = [X_src_shifted]
            for mult in aug_multipliers:
                parts.append(X_src_shifted * mult)
            parts.append(X_restr_batch)
            X_surr_list.append(torch.cat(parts, dim=2))

        # (M*C_min, T, p_ch)
        X_surr_mega = torch.cat(X_surr_list, dim=0)
        X_restr_mega = X_restr_batch.repeat(M, 1, 1)
        y_mega = y_batch.repeat(M, 1, 1)
        valid_mega = valid_batch.repeat(M, 1) if valid_batch is not None else None

        dr2_mega, _, _, _, _ = solver.solve_restricted_batched(
            X_surr_mega, X_restr_mega, y_mega, valid_mega)
        # dr2_mega: (M*C_min, T)
        dr2_mega_np = dr2_mega.cpu().numpy()

        for m in range(M):
            dr2_k_np = dr2_mega_np[m*C_min:(m+1)*C_min]  # (C_min, T)
            dr2_surr_perchannel[kb + m] = dr2_k_np
            dr2_surr_agg[kb + m] = dr2_k_np.mean(axis=0)

        del X_surr_mega, X_restr_mega, y_mega, valid_mega, dr2_mega
        torch.cuda.empty_cache()

    # Smooth real and surrogate identically before p-value computation
    if smooth_samples > 1:
        from scipy.ndimage import uniform_filter1d
        dr2_real_s = uniform_filter1d(dr2_real_agg, smooth_samples,
                                      mode='nearest')
        dr2_surr_s = np.array([
            uniform_filter1d(dr2_surr_agg[i], smooth_samples, mode='nearest')
            for i in range(n_surrogates)
        ])
    else:
        dr2_real_s = dr2_real_agg
        dr2_surr_s = dr2_surr_agg

    # GPD tail p-values on aggregated timecourses
    from cadence.significance.gpd_pvalues import gpd_tail_pvalues
    p_values = gpd_tail_pvalues(dr2_real_s, dr2_surr_s,
                                pool_half=gpd_pool_half,
                                threshold_quantile=gpd_threshold_quantile)

    # Flat null for HMM null stats
    dr2_null_flat = dr2_surr_agg.ravel()

    return p_values, dr2_null_flat, dr2_surr_perchannel


def zscore_posterior_perchannel(dr2_real_perchannel, dr2_surr_perchannel,
                                 smooth_samples=10, sigmoid_center=2.0,
                                 eval_rate=2.0,
                                 multi_scale_s=(3.0, 8.0, 15.0),
                                 sigmoid_scale=1.0,
                                 hmm_enabled=True,
                                 hmm_max_iter=20):
    """Coupling posterior via per-channel z-score + Stouffer + HMM pipeline.

    5-stage pipeline:
      A) Per-channel pointwise z-scores against time-varying null
      B) Stouffer aggregation across channels (SNR boost √C)
      C) Multi-scale smoothing (max-envelope over event durations)
      D) Sigmoid sharpening (z-score → [0,1] posterior)
      E) 2-state HMM segmentation (null state anchored at N(0,1))

    Args:
        dr2_real_perchannel: (C_min, T) real per-channel dR2.
        dr2_surr_perchannel: (K, C_min, T) surrogate per-channel dR2.
        smooth_samples: Smoothing window for dr2_smooth return value.
        sigmoid_center: z-score threshold for 0.5 posterior.
        eval_rate: Samples per second.
        multi_scale_s: Smoothing scales in seconds for max-envelope.
        sigmoid_scale: Sigmoid sharpness (lower = sharper).
        hmm_enabled: Whether to run 2-state Baum-Welch HMM.
        hmm_max_iter: Max EM iterations for HMM.

    Returns:
        posterior: (T,) coupling posterior in [0, 1].
        dr2_smooth: (T,) smoothed aggregated dR2.
        null_stats: dict with null statistics and z-score diagnostics.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.special import expit

    C_min, T = dr2_real_perchannel.shape
    K = dr2_surr_perchannel.shape[0]

    # --- Stage A: Per-channel pointwise z-scores (vectorized) ---
    # Smooth null stats over 15s to reduce noise (K is finite)
    null_smooth = max(1, int(15.0 * eval_rate))
    real_smooth = max(1, int(3.0 * eval_rate))

    # Compute null mean/std across surrogates → (C_min, T)
    surr_mean = np.nanmean(dr2_surr_perchannel, axis=0)  # (C_min, T)
    surr_std = np.nanstd(dr2_surr_perchannel, axis=0)    # (C_min, T)

    # uniform_filter1d supports axis= for 2D arrays — one C call for all channels
    mu_ct = uniform_filter1d(surr_mean, null_smooth, axis=1, mode='nearest')
    sig_ct = np.maximum(
        uniform_filter1d(surr_std, null_smooth, axis=1, mode='nearest'), 1e-8)

    real_clean = np.nan_to_num(dr2_real_perchannel, nan=0.0)  # (C_min, T)
    real_smooth_arr = uniform_filter1d(real_clean, real_smooth, axis=1, mode='nearest')

    z_perchannel = (real_smooth_arr - mu_ct) / sig_ct

    # --- Stage B: Stouffer aggregation across channels ---
    z_agg = np.nansum(z_perchannel, axis=0) / np.sqrt(C_min)

    # --- Stage C: Multi-scale smoothing (max-envelope) ---
    z_scales = []
    for s in multi_scale_s:
        w = max(1, int(s * eval_rate))
        z_scales.append(uniform_filter1d(z_agg, w, mode='nearest'))
    z_multi = np.maximum.reduce(z_scales)

    # --- Stage D: Sigmoid sharpening ---
    posterior_raw = expit((z_multi - sigmoid_center) / sigmoid_scale)

    # --- Stage E: 2-state HMM segmentation (Baum-Welch) ---
    if hmm_enabled and T > 20:
        from cadence.significance.detection import _forward_backward

        # State 0: N(0, 1) — anchored by z-score definition (NEVER updated)
        mu_0_hmm, sigma_0_hmm = 0.0, 1.0

        # State 1: initialize from upper quartile of z
        z_upper = z_multi[z_multi > np.percentile(z_multi, 75)]
        if len(z_upper) < 5:
            z_upper = z_multi[z_multi > 0]
        mu_1 = max(float(np.mean(z_upper)), 1.5) if len(z_upper) >= 5 else 2.0
        sigma_1 = max(float(np.std(z_upper)), 0.5) if len(z_upper) >= 5 else 1.0

        # Transitions: agnostic P(stay) = 0.95 for both states
        alpha = 0.05   # P(0->1)
        beta = 0.05    # P(1->0)
        log_pi = np.log(np.array([0.5, 0.5]))

        em_tol = 1e-6
        prev_ll = -np.inf

        for iteration in range(hmm_max_iter):
            log_A = np.log(np.array([[1 - alpha, alpha], [beta, 1 - beta]]))

            # Emission log-probabilities (vectorized)
            log_em = np.column_stack([
                -0.5 * np.log(2 * np.pi) - np.log(sigma_0_hmm)
                    - 0.5 * ((z_multi - mu_0_hmm) / sigma_0_hmm) ** 2,
                -0.5 * np.log(2 * np.pi) - np.log(sigma_1)
                    - 0.5 * ((z_multi - mu_1) / sigma_1) ** 2,
            ])

            # E-step
            gamma, ll = _forward_backward(log_em, log_A, log_pi)

            if abs(ll - prev_ll) < em_tol * max(1.0, abs(ll)):
                break
            prev_ll = ll

            # M-step
            g0, g1 = gamma[:, 0], gamma[:, 1]
            w0, w1 = np.sum(g0), np.sum(g1)

            # State 0: FIXED at N(0, 1) — anchored, never updated
            # State 1: update emission from data
            if w1 > 5:
                mu_1 = max(float(np.sum(g1 * z_multi) / w1), 0.5)
                sigma_1 = max(np.sqrt(float(np.sum(g1 * (z_multi - mu_1) ** 2) / w1)), 0.3)
            mu_1 = max(mu_1, 0.5)  # enforce separation from null

            # Update transitions from expected transition counts (vectorized)
            xi = (gamma[:-1, :, None]
                  * np.exp(log_A[None, :, :] + log_em[1:, None, :])
                  * gamma[1:, None, :])
            n_ij = xi.sum(axis=0)  # (2, 2)
            if n_ij[0, 0] + n_ij[0, 1] > 1e-10:
                alpha = np.clip(n_ij[0, 1] / (n_ij[0, 0] + n_ij[0, 1]), 1e-6, 0.3)
            if n_ij[1, 0] + n_ij[1, 1] > 1e-10:
                beta = np.clip(n_ij[1, 0] / (n_ij[1, 0] + n_ij[1, 1]), 1e-6, 0.3)

        posterior = gamma[:, 1]
    else:
        posterior = posterior_raw

    # --- Return values (backward compatible) ---
    dr2_agg = np.nanmean(dr2_real_perchannel, axis=0)
    dr2_smooth = uniform_filter1d(
        np.nan_to_num(dr2_agg, nan=0.0), smooth_samples, mode='nearest')
    surr_agg = np.nanmean(dr2_surr_perchannel, axis=1)

    null_stats = {
        'mu_0': float(np.nanmean(surr_agg)),
        'sigma_0': max(float(np.nanstd(surr_agg)), 1e-8),
        'z_agg_mean': float(np.mean(z_multi)),
        'z_agg_max': float(np.max(z_multi)),
        'hmm_enabled': hmm_enabled,
    }

    return np.clip(posterior, 0.0, 1.0), dr2_smooth, null_stats
