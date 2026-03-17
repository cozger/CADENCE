"""Circular-shift surrogate significance testing for CADENCE coupling.

Key optimization: shift commutes with convolution.
  shift(phi_j * x) = phi_j * shift(x)
So we pre-compute basis-convolved source signals once, then just
circular-shift the convolved outputs for each surrogate.
"""

import numpy as np
import torch

from cadence.surrogates import circular_shift_surrogate_batched
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
