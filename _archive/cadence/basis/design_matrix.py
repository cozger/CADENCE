"""Design matrix builder: convolve source signals with basis functions via GPU conv1d.

For each source modality, the design matrix columns are:
    [phi_0 * x_src_ch0, phi_1 * x_src_ch0, ..., phi_0 * x_src_ch1, ...]

For cross-rate pathways, convolved outputs are resampled to the target's rate.
AR terms from the target signal are appended at the end.
"""

import numpy as np
import torch
import torch.nn.functional as F


class DesignMatrixBuilder:
    """Build design matrices for distributed lag regression.

    Pre-computes basis-convolved source signals on GPU. The design matrix
    for a given (source_mod, target_mod) pathway is the horizontal stack
    of all basis-convolved channels plus autoregressive target terms.
    """

    def __init__(self, basis, ar_order=3, device='cuda'):
        """
        Args:
            basis: (n_lag_samples, n_basis) numpy array of basis functions.
            ar_order: Number of AR lags to include (0 = no AR terms).
            device: Torch device for computation.
        """
        self.n_lag_samples, self.n_basis = basis.shape
        self.ar_order = ar_order
        self.device = device

        # Store basis as conv1d weight: (n_basis, 1, n_lag_samples)
        # Flip for convolution (conv1d does cross-correlation)
        basis_flipped = basis[::-1, :].copy()  # flip lag axis
        self.basis_weight = torch.tensor(
            basis_flipped.T[:, None, :],  # (n_basis, 1, n_lag)
            dtype=torch.float32, device=device
        )

    def convolve_source(self, source_signal, source_valid=None):
        """Convolve each channel of source signal with all basis functions.

        Args:
            source_signal: (T_src, C_src) numpy array or tensor.
            source_valid: (T_src,) boolean mask (optional).

        Returns:
            convolved: (T_src, C_src * n_basis) tensor on device.
            valid_out: (T_src,) boolean tensor.
        """
        if isinstance(source_signal, np.ndarray):
            source_signal = torch.tensor(source_signal, dtype=torch.float32,
                                         device=self.device)
        T, C = source_signal.shape

        # Reshape for grouped conv1d: (1, C, T)
        x = source_signal.T.unsqueeze(0)  # (1, C, T)

        # Expand basis weight for grouped convolution: (C * n_basis, 1, n_lag)
        # Each input channel gets its own set of n_basis filters
        w = self.basis_weight.repeat(C, 1, 1)  # (C * n_basis, 1, n_lag)

        # Grouped conv1d: groups=C, each group has n_basis filters
        # Input (1, C, T) -> reshape to (1, C, T) with groups=C
        # Need to reshape x to (1, C, T) and use groups=C
        # conv1d with groups=C: each of C input channels convolved with n_basis filters
        # Output: (1, C * n_basis, T_out)
        pad = self.n_lag_samples - 1  # causal padding (only past lags)
        x_padded = F.pad(x, (pad, 0))  # left-pad for causal
        convolved = F.conv1d(x_padded, w, groups=C)  # (1, C * n_basis, T)

        # Transpose to (T, C * n_basis)
        convolved = convolved.squeeze(0).T  # (T, C * n_basis)

        # Validity: if source has NaN/invalid regions, propagate
        if source_valid is not None:
            if isinstance(source_valid, np.ndarray):
                source_valid = torch.tensor(source_valid, dtype=torch.bool,
                                            device=self.device)
            valid_out = source_valid
        else:
            valid_out = torch.ones(T, dtype=torch.bool, device=self.device)

        return convolved, valid_out

    def build(self, source_signal, target_signal, source_valid=None,
              target_valid=None, eval_times=None, source_times=None,
              target_times=None, eval_rate=2.0):
        """Build complete design matrix for one pathway.

        Handles cross-rate pathways by resampling convolved outputs to eval rate.

        Args:
            source_signal: (T_src, C_src) source modality features.
            target_signal: (T_tgt, C_tgt) target modality features (for AR terms).
            source_valid: (T_src,) boolean validity mask.
            target_valid: (T_tgt,) boolean validity mask.
            eval_times: (T_eval,) output evaluation timestamps.
            source_times: (T_src,) source timestamps (for cross-rate resampling).
            target_times: (T_tgt,) target timestamps (for AR term alignment).
            eval_rate: Evaluation rate in Hz.

        Returns:
            X: (T_eval, p) design matrix tensor on device.
               p = C_src * n_basis + ar_order * C_tgt
            y: (T_eval, C_tgt) target signal resampled to eval times.
            valid: (T_eval,) boolean validity mask.
        """
        # Step 1: Convolve source with basis functions
        convolved, src_valid = self.convolve_source(source_signal, source_valid)
        T_src = convolved.shape[0]

        # Step 2: Determine eval grid
        if eval_times is None:
            # Default: use source rate
            T_eval = T_src
            X_basis = convolved
        else:
            T_eval = len(eval_times)
            # Resample convolved output to eval times
            if source_times is not None:
                X_basis = self._resample_to_eval(
                    convolved, source_times, eval_times)
            else:
                X_basis = convolved[:T_eval]

        # Step 3: Build AR terms from target signal
        if self.ar_order > 0 and target_signal is not None:
            ar_terms = self._build_ar_terms(
                target_signal, target_times, eval_times, eval_rate)
            X = torch.cat([X_basis, ar_terms], dim=1)
        else:
            X = X_basis

        # Step 4: Resample target to eval grid
        if target_signal is not None:
            if isinstance(target_signal, np.ndarray):
                target_signal = torch.tensor(target_signal, dtype=torch.float32,
                                             device=self.device)
            if target_times is not None and eval_times is not None:
                y = self._resample_to_eval(
                    target_signal, target_times, eval_times)
            else:
                y = target_signal[:T_eval]
        else:
            y = torch.zeros(T_eval, 1, device=self.device)

        # Step 5: Combine validity masks (GPU-native searchsorted)
        valid = torch.ones(T_eval, dtype=torch.bool, device=self.device)
        if source_valid is not None and eval_times is not None and source_times is not None:
            valid = valid & self._lookup_valid_gpu(
                source_valid, source_times, eval_times)
        if target_valid is not None and eval_times is not None and target_times is not None:
            valid = valid & self._lookup_valid_gpu(
                target_valid, target_times, eval_times)

        return X, y, valid

    def _lookup_valid_gpu(self, valid_mask, src_times, eval_times):
        """Nearest-neighbor validity lookup on GPU."""
        if isinstance(valid_mask, np.ndarray):
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        elif not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.tensor(np.asarray(valid_mask), dtype=torch.bool, device=self.device)
        else:
            valid_mask = valid_mask.to(self.device)

        if isinstance(src_times, np.ndarray):
            st = torch.tensor(src_times, dtype=torch.float32, device=self.device)
        else:
            st = src_times.to(self.device).float()

        if isinstance(eval_times, np.ndarray):
            et = torch.tensor(eval_times, dtype=torch.float32, device=self.device)
        else:
            et = eval_times.to(self.device).float()

        idx = torch.searchsorted(st, et).clamp(0, len(valid_mask) - 1)
        return valid_mask[idx]

    def _resample_to_eval(self, signal, signal_times, eval_times):
        """Resample signal from its native timestamps to eval timestamps.

        GPU-native linear interpolation via searchsorted + lerp.
        Avoids GPU->CPU->GPU transfers that bottleneck the pipeline.
        """
        # Ensure everything is on GPU as tensors
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        elif signal.device != torch.device(self.device):
            signal = signal.to(self.device)

        if isinstance(signal_times, np.ndarray):
            sig_t = torch.tensor(signal_times, dtype=torch.float32, device=self.device)
        else:
            sig_t = signal_times.to(self.device).float()

        if isinstance(eval_times, np.ndarray):
            eval_t = torch.tensor(eval_times, dtype=torch.float32, device=self.device)
        else:
            eval_t = eval_times.to(self.device).float()

        # searchsorted: find insertion indices for eval_times in signal_times
        idx_right = torch.searchsorted(sig_t, eval_t).clamp(1, len(sig_t) - 1)
        idx_left = idx_right - 1

        # Lerp weights
        t_left = sig_t[idx_left]
        t_right = sig_t[idx_right]
        dt = (t_right - t_left).clamp(min=1e-10)
        w = ((eval_t - t_left) / dt).clamp(0.0, 1.0)  # (T_eval,)

        # Vectorized interpolation across all channels
        val_left = signal[idx_left]    # (T_eval, C)
        val_right = signal[idx_right]  # (T_eval, C)
        result = val_left + w.unsqueeze(1) * (val_right - val_left)

        return result

    def _build_ar_terms(self, target_signal, target_times, eval_times, eval_rate):
        """Build autoregressive terms: lagged target signal values.

        AR(p) means we include y(t-1), y(t-2), ..., y(t-p) as predictors.
        """
        if isinstance(target_signal, np.ndarray):
            target_signal = torch.tensor(target_signal, dtype=torch.float32,
                                         device=self.device)

        # Resample target to eval grid
        if target_times is not None and eval_times is not None:
            y_eval = self._resample_to_eval(target_signal, target_times, eval_times)
        else:
            y_eval = target_signal

        T_eval, C_tgt = y_eval.shape
        ar_cols = []
        for lag in range(1, self.ar_order + 1):
            shifted = torch.zeros_like(y_eval)
            shifted[lag:] = y_eval[:-lag]
            ar_cols.append(shifted)

        return torch.cat(ar_cols, dim=1)  # (T_eval, ar_order * C_tgt)
