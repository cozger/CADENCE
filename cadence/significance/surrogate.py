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
                                   surrogate_method='circular_shift'):
    """Per-timepoint p-values by circular-shifting source columns of X_augmented.

    Shifts only the first n_source_cols columns (basis-convolved source).
    Augmented terms (nonlinear, moderation) are rebuilt from the shifted
    source so that the surrogate model matches the real model structure.

    Both real and surrogate dR2 are smoothed identically before computing
    p-values, which reduces null variance and prevents sensitivity loss.

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
        surrogate_method: 'circular_shift' or 'fourier_phase'. Fourier phase
            randomization destroys phase structure while preserving power spectrum,
            providing a stronger null for autocorrelated features (e.g., delta-band
            interbrain PLV).

    Returns:
        p_values: (T_eval,) per-timepoint p-values.
        dr2_surrogates: (n_surrogates, T_eval) surrogate dR2 timecourses.
    """
    device = X_augmented.device
    T_eval = X_augmented.shape[0]
    p_total = X_augmented.shape[1]
    p_ar = X_restricted.shape[1]

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

            dr2_surr, _, _, _, _ = solver.solve_restricted(
                X_surr_full, X_restricted, y, valid)
            dr2_surrogates[batch_start + k] = dr2_surr.cpu().numpy()

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

    # Per-timepoint p-values: fraction of surrogates >= real
    p_values = np.mean(dr2_surr_s >= dr2_real_s[None, :], axis=0)
    p_values = np.maximum(p_values, 1.0 / (n_surrogates + 1))

    return p_values, dr2_surrogates
