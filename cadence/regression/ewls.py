"""Forward-backward Exponentially Weighted Least Squares (EWLS) solver.

Time-varying regression via two-pass exponential weighting:
  Forward:  S_fwd(t) = gamma * S_fwd(t-1) + x(t) x(t)'
  Backward: S_bwd(t) = gamma * S_bwd(t+1) + x(t+1) x(t+1)'
  Combined: S_xx(t) = S_fwd(t) + S_bwd(t)
  Solve:    beta(t) = solve(S_xx(t) + lambda*I, S_xy(t))

Optimizations:
  - Batched: process B pathways simultaneously for better GPU occupancy
  - Triton strided forward scan: O(log T) depth, no transpose/copy overhead.
  - Triton fused backward scan: computes outer products on-the-fly and
    combines with forward in-place. Replaces O(T) Python loop with single
    kernel launch per tensor.
  - Falls back to sequential Python loops if Triton unavailable.
"""

import warnings

import numpy as np
import torch

# --- Scan implementations (priority order) ---

# 1. Triton parallel scan (fastest: O(log T) depth per block)
_HAS_TRITON_SCAN = False
_HAS_TRITON_BWD = False
try:
    from cadence.regression.triton_scan import (
        triton_exp_scan_fwd_, triton_bwd_scan_combine, HAS_TRITON
    )
    _HAS_TRITON_SCAN = HAS_TRITON
    _HAS_TRITON_BWD = HAS_TRITON
except ImportError:
    pass


def disable_triton_scan():
    """Disable Triton kernels globally (e.g., after CUDA context corruption)."""
    global _HAS_TRITON_SCAN, _HAS_TRITON_BWD
    _HAS_TRITON_SCAN = False
    _HAS_TRITON_BWD = False


def _multi_exp_scan_fwd_(tensors, gamma, force_sequential=False):
    """In-place forward exponential scan across multiple tensors.

    Each tensor has shape (B, T, ...). Dispatches to Triton parallel scan
    if available, otherwise falls back to sequential Python loop.
    """
    T = tensors[0].shape[1]
    if (_HAS_TRITON_SCAN and tensors[0].is_cuda and not force_sequential
            and T >= 64):  # Triton kernels unsafe for very small T
        triton_exp_scan_fwd_(tensors, gamma)
    else:
        _sequential_exp_scan_fwd_(tensors, gamma)


def _sequential_exp_scan_fwd_(tensors, gamma):
    """Fallback: sequential Python loop scan. O(T) per tensor."""
    T = tensors[0].shape[1]
    for t in range(1, T):
        for A in tensors:
            A[:, t].add_(A[:, t - 1], alpha=gamma)


def _check_vram(label, estimated_bytes, device):
    """Phase 5: VRAM pre-flight check. Warn if estimated peak > 85% free."""
    try:
        free, total = torch.cuda.mem_get_info(device)
        if estimated_bytes > 0.85 * free:
            warnings.warn(
                f"EWLS {label}: estimated {estimated_bytes / 1e9:.1f} GB "
                f"peak vs {free / 1e9:.1f} GB free VRAM "
                f"({total / 1e9:.1f} GB total). May OOM.",
                ResourceWarning, stacklevel=3)
    except Exception:
        pass


class EWLSSolver:
    """GPU-accelerated forward-backward EWLS for time-varying regression."""

    def __init__(self, tau_seconds=30.0, lambda_ridge=1e-3, eval_rate=2.0,
                 device='cuda', min_effective_n=20, force_sequential=False):
        self.tau = tau_seconds
        self.lam = lambda_ridge
        self.eval_rate = eval_rate
        self.device = device
        self.min_effective_n = min_effective_n
        self.force_sequential = force_sequential

    def solve(self, X, y, valid=None):
        """Single-pathway solve. Auto-selects checkpointed path for large p."""
        T, p = X.shape
        xx_bytes = T * p * p * 4
        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
        except Exception:
            free_vram = 8 * 1024**3  # conservative fallback

        if xx_bytes > 0.4 * free_vram:
            # Large p: use checkpointed solve to avoid OOM
            return self.solve_batched_checkpointed(
                X.unsqueeze(0), y.unsqueeze(0),
                valid.unsqueeze(0) if valid is not None else None,
                squeeze=True)

        X_b = X.unsqueeze(0)
        y_b = y.unsqueeze(0)
        valid_b = valid.unsqueeze(0) if valid is not None else None
        beta_b, y_hat_b, r2_b, n_eff_b = self.solve_batched(X_b, y_b, valid_b)
        return beta_b[0], y_hat_b[0], r2_b[0], n_eff_b[0]

    def solve_batched(self, X, y, valid=None):
        """Solve B time-varying regressions simultaneously.

        When Triton is available, both forward and backward scans use parallel
        prefix scan kernels — no Python loops over T. The backward kernel
        computes outer products on-the-fly and adds to the forward result
        in-place (zero extra allocation beyond the forward tensors).

        Falls back to streaming Python loop when Triton is unavailable.

        Args:
            X: (B, T, p) batched design matrices.
            y: (B, T, C) batched target signals.
            valid: (B, T) boolean mask, or None.

        Returns:
            beta: (B, T, p, C) time-varying regression coefficients.
            y_hat: (B, T, C) predicted target signals.
            r_squared: (B, T) coefficient of determination.
            effective_n: (B, T) effective sample counts.
        """
        B, T, p = X.shape
        C = y.shape[2]

        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=self.device)

        dt = 1.0 / self.eval_rate
        gamma = np.exp(-dt / self.tau) if self.tau > 0 else 0.0

        valid_f = valid.float()  # (B, T)

        # Phase 5: VRAM pre-flight
        est_bytes = B * T * (p * p + p * C) * 4  # streaming formula
        _check_vram('solve_batched', est_bytes, self.device)

        # === Pre-compute outer products (vectorized, one GPU call each) ===
        xx = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))   # (B, T, p, p)
        xy = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))   # (B, T, p, C)

        # Zero invalid timepoints
        inv_mask = (~valid).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        xx.masked_fill_(inv_mask, 0.0)
        xy.masked_fill_(inv_mask, 0.0)

        # === Forward + backward scans ===
        use_triton = (
            _HAS_TRITON_BWD and X.is_cuda and T > 1
            and not self.force_sequential
            and T >= 64  # Triton kernels unsafe for very small T (BLOCK > T)
        )

        if use_triton:
            # --- Multi-stream Triton path ---
            # Launch forward + backward scans for xx, xy, n_eff on
            # 3 concurrent CUDA streams. Within each stream, operations
            # execute in order (forward then backward). Across streams,
            # they overlap — tripling GPU utilization during scan phases.
            n_fwd = torch.zeros(B, T, 1, 1, device=self.device, dtype=torch.float32)
            n_fwd[:, :, 0, 0] = valid_f

            # Prepare bwd_n data (shifted valid, reversed for forward scan)
            bwd_n = torch.zeros(B, T, 1, device=self.device, dtype=torch.float32)
            bwd_n[:, :-1, 0] = valid_f[:, 1:]
            bwd_n = bwd_n.flip(1).contiguous()

            # Record event so non-default streams know inputs are ready
            ready = torch.cuda.Event()
            ready.record()

            s_xx = torch.cuda.Stream(device=self.device)
            s_xy = torch.cuda.Stream(device=self.device)
            s_n = torch.cuda.Stream(device=self.device)

            # Stream 1: forward scan xx → backward scan xx
            with torch.cuda.stream(s_xx):
                s_xx.wait_event(ready)
                triton_exp_scan_fwd_([xx], gamma)
                triton_bwd_scan_combine(X, X, valid_f, xx, gamma)

            # Stream 2: forward scan xy → backward scan xy
            with torch.cuda.stream(s_xy):
                s_xy.wait_event(ready)
                triton_exp_scan_fwd_([xy], gamma)
                triton_bwd_scan_combine(X, y, valid_f, xy, gamma)

            # Stream 3: forward scan n_fwd + backward n_eff
            with torch.cuda.stream(s_n):
                s_n.wait_event(ready)
                triton_exp_scan_fwd_([n_fwd], gamma)
                triton_exp_scan_fwd_([bwd_n], gamma)

            # Wait for all streams
            s_xx.synchronize()
            s_xy.synchronize()
            s_n.synchronize()

            n_fwd_2d = n_fwd.squeeze(-1).squeeze(-1)
            bwd_n = bwd_n.flip(1).contiguous().squeeze(-1)
            del n_fwd
        else:
            # --- Sequential fallback ---
            n_fwd = torch.zeros(B, T, 1, 1, device=self.device, dtype=torch.float32)
            n_fwd[:, :, 0, 0] = valid_f
            _multi_exp_scan_fwd_([xx, xy, n_fwd], gamma,
                                 force_sequential=self.force_sequential)
            n_fwd_2d = n_fwd.squeeze(-1).squeeze(-1)
            del n_fwd
            bwd_n = self._streaming_backward(X, y, valid_f, xx, xy, gamma)

        # Combined effective sample count
        n_eff = n_fwd_2d + bwd_n
        del n_fwd_2d, bwd_n

        # xx and xy now contain S_xx and S_xy (forward + backward combined)
        S_xx = xx
        S_xy = xy

        torch.cuda.empty_cache()

        # === Solve ===
        reg = self.lam * torch.eye(p, device=self.device, dtype=torch.float32)
        S_xx += reg  # broadcasts (p, p) -> (B, T, p, p)

        # Phase 12: launch EWMA on separate stream to overlap with solve
        stream_ewma = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream_ewma):
            y_mean = self._ewma_mean_batched(y, valid_f, gamma,
                                             force_sequential=self.force_sequential)

        beta = torch.linalg.solve(S_xx, S_xy)  # (B, T, p, C)
        del S_xx, S_xy

        # Predicted values
        y_hat = torch.matmul(X.unsqueeze(2), beta).squeeze(2)  # (B, T, C)

        # R-squared (wait for EWMA to finish)
        stream_ewma.synchronize()

        residuals = y - y_hat
        ss_res = (residuals ** 2).sum(dim=2)  # (B, T)
        ss_tot = ((y - y_mean) ** 2).sum(dim=2)  # (B, T)

        with torch.no_grad():
            r_squared = torch.where(
                ss_tot > 1e-10,
                1.0 - ss_res / ss_tot,
                torch.zeros_like(ss_res)
            )
            r_squared = r_squared.clamp(-1.0, 1.0)

        low_n = n_eff < self.min_effective_n
        r_squared[low_n] = float('nan')
        beta[low_n] = 0.0

        return beta, y_hat, r_squared, n_eff

    def solve_restricted(self, X_full, X_restricted, y, valid=None):
        """Solve both full and restricted models, return delta-R2.

        When checkpointed solve is used (large p), beta_full will be None.
        """
        beta_full, _, r2_full, n_eff = self.solve(X_full, y, valid)
        _, _, r2_restr, _ = self.solve(X_restricted, y, valid)
        dr2 = r2_full - r2_restr
        return dr2, r2_full, r2_restr, beta_full, n_eff

    def solve_restricted_batched(self, X_full_batch, X_restr_batch, y_batch,
                                  valid_batch=None):
        """Batched solve_restricted: process B pathways simultaneously."""
        beta_full, _, r2_full, n_eff = self.solve_batched(
            X_full_batch, y_batch, valid_batch)
        _, _, r2_restr, _ = self.solve_batched(
            X_restr_batch, y_batch, valid_batch)
        dr2 = r2_full - r2_restr
        return dr2, r2_full, r2_restr, beta_full, n_eff

    def solve_batched_checkpointed(self, X, y, valid=None, squeeze=False):
        """Gradient-checkpointed EWLS solve for large p.

        Uses O(sqrt(T) * p²) memory instead of O(T * p²) by saving
        checkpoints every K steps and replaying forward within segments.
        Produces bit-identical results to solve_batched.

        Args:
            X: (B, T, p) batched design matrices.
            y: (B, T, C) batched target signals.
            valid: (B, T) boolean mask, or None.
            squeeze: if True, squeeze batch dim from outputs (for single-pathway).

        Returns:
            beta: None (not stored to save memory).
            y_hat: (B, T, C) predicted target signals.
            r_squared: (B, T) coefficient of determination.
            effective_n: (B, T) effective sample counts.
        """
        B, T, p = X.shape
        C = y.shape[2]

        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=self.device)

        dt = 1.0 / self.eval_rate
        gamma = np.exp(-dt / self.tau) if self.tau > 0 else 0.0

        valid_f = valid.float()  # (B, T)
        K = max(1, int(T ** 0.5))  # checkpoint interval
        n_ckpts = (T + K - 1) // K  # number of checkpoints

        # --- Phase 1: Forward pass with checkpoints ---
        # Running accumulators
        acc_xx = torch.zeros(B, p, p, device=self.device, dtype=X.dtype)
        acc_xy = torch.zeros(B, p, C, device=self.device, dtype=X.dtype)
        acc_n = torch.zeros(B, device=self.device, dtype=torch.float32)

        # Checkpoint storage: save at the START of each segment (before segment's first step)
        ckpt_xx = torch.zeros(n_ckpts, B, p, p, device=self.device, dtype=X.dtype)
        ckpt_xy = torch.zeros(n_ckpts, B, p, C, device=self.device, dtype=X.dtype)
        ckpt_n = torch.zeros(n_ckpts, B, device=self.device, dtype=torch.float32)

        tmp_xx = torch.empty(B, p, p, device=self.device, dtype=X.dtype)
        tmp_xy = torch.empty(B, p, C, device=self.device, dtype=X.dtype)

        for t in range(T):
            seg_idx = t // K
            if t % K == 0:
                # Save checkpoint at start of segment
                ckpt_xx[seg_idx].copy_(acc_xx)
                ckpt_xy[seg_idx].copy_(acc_xy)
                ckpt_n[seg_idx].copy_(acc_n)

            # Accumulate: acc = gamma * acc + outer(x[t], x[t]) * v[t]
            x_t = X[:, t]  # (B, p)
            v_t = valid_f[:, t]  # (B,)

            acc_xx.mul_(gamma)
            acc_xy.mul_(gamma)
            acc_n.mul_(gamma)

            torch.bmm(x_t.unsqueeze(2), x_t.unsqueeze(1), out=tmp_xx)
            tmp_xx.mul_(v_t.view(B, 1, 1))
            acc_xx.add_(tmp_xx)

            y_t = y[:, t]  # (B, C)
            torch.bmm(x_t.unsqueeze(2), y_t.unsqueeze(1), out=tmp_xy)
            tmp_xy.mul_(v_t.view(B, 1, 1))
            acc_xy.add_(tmp_xy)

            acc_n.add_(v_t)

        del acc_xx, acc_xy, acc_n, tmp_xx, tmp_xy

        # --- EWMA mean (small memory: B,T,C) ---
        y_mean = self._ewma_mean_batched(y, valid_f, gamma,
                                         force_sequential=self.force_sequential)

        # --- Phase 2: Backward pass with per-segment solve ---
        # Output tensors
        y_hat = torch.zeros(B, T, C, device=self.device, dtype=X.dtype)
        r_squared = torch.zeros(B, T, device=self.device, dtype=torch.float32)
        n_eff = torch.zeros(B, T, device=self.device, dtype=torch.float32)

        # Regularization
        reg = self.lam * torch.eye(p, device=self.device, dtype=torch.float32)

        # Backward running accumulators (carry between segments)
        acc_bwd_xx = torch.zeros(B, p, p, device=self.device, dtype=X.dtype)
        acc_bwd_xy = torch.zeros(B, p, C, device=self.device, dtype=X.dtype)
        acc_bwd_n = torch.zeros(B, device=self.device, dtype=torch.float32)

        tmp_bwd_xx = torch.empty(B, p, p, device=self.device, dtype=X.dtype)
        tmp_bwd_xy = torch.empty(B, p, C, device=self.device, dtype=X.dtype)

        # Process segments right-to-left
        for seg_idx in range(n_ckpts - 1, -1, -1):
            seg_start = seg_idx * K
            seg_end = min((seg_idx + 1) * K, T)
            seg_len = seg_end - seg_start

            # Pre-allocate segment buffers
            seg_bwd_xx = torch.zeros(seg_len, B, p, p, device=self.device, dtype=X.dtype)
            seg_bwd_xy = torch.zeros(seg_len, B, p, C, device=self.device, dtype=X.dtype)
            seg_bwd_n = torch.zeros(seg_len, B, device=self.device, dtype=torch.float32)

            # --- Backward scan through segment (right-to-left) ---
            # Backward carry propagates naturally across segments
            for i in range(seg_len - 1, -1, -1):
                t = seg_start + i
                # Backward uses shifted indexing: outer(x[t+1], x[t+1]) * v[t+1]
                if t + 1 < T:
                    x_t1 = X[:, t + 1]
                    y_t1 = y[:, t + 1]
                    v_t1 = valid_f[:, t + 1]

                    acc_bwd_xx.mul_(gamma)
                    acc_bwd_xy.mul_(gamma)
                    acc_bwd_n.mul_(gamma)

                    torch.bmm(x_t1.unsqueeze(2), x_t1.unsqueeze(1), out=tmp_bwd_xx)
                    tmp_bwd_xx.mul_(v_t1.view(B, 1, 1))
                    acc_bwd_xx.add_(tmp_bwd_xx)

                    torch.bmm(x_t1.unsqueeze(2), y_t1.unsqueeze(1), out=tmp_bwd_xy)
                    tmp_bwd_xy.mul_(v_t1.view(B, 1, 1))
                    acc_bwd_xy.add_(tmp_bwd_xy)

                    acc_bwd_n.add_(v_t1)

                seg_bwd_xx[i].copy_(acc_bwd_xx)
                seg_bwd_xy[i].copy_(acc_bwd_xy)
                seg_bwd_n[i].copy_(acc_bwd_n)

            # --- Forward replay from checkpoint ---
            seg_fwd_xx = torch.zeros(seg_len, B, p, p, device=self.device, dtype=X.dtype)
            seg_fwd_xy = torch.zeros(seg_len, B, p, C, device=self.device, dtype=X.dtype)
            seg_fwd_n = torch.zeros(seg_len, B, device=self.device, dtype=torch.float32)

            replay_xx = ckpt_xx[seg_idx].clone()
            replay_xy = ckpt_xy[seg_idx].clone()
            replay_n = ckpt_n[seg_idx].clone()

            tmp_fwd_xx = torch.empty(B, p, p, device=self.device, dtype=X.dtype)
            tmp_fwd_xy = torch.empty(B, p, C, device=self.device, dtype=X.dtype)

            for i in range(seg_len):
                t = seg_start + i
                x_t = X[:, t]
                y_t = y[:, t]
                v_t = valid_f[:, t]

                replay_xx.mul_(gamma)
                replay_xy.mul_(gamma)
                replay_n.mul_(gamma)

                torch.bmm(x_t.unsqueeze(2), x_t.unsqueeze(1), out=tmp_fwd_xx)
                tmp_fwd_xx.mul_(v_t.view(B, 1, 1))
                replay_xx.add_(tmp_fwd_xx)

                torch.bmm(x_t.unsqueeze(2), y_t.unsqueeze(1), out=tmp_fwd_xy)
                tmp_fwd_xy.mul_(v_t.view(B, 1, 1))
                replay_xy.add_(tmp_fwd_xy)

                replay_n.add_(v_t)

                seg_fwd_xx[i].copy_(replay_xx)
                seg_fwd_xy[i].copy_(replay_xy)
                seg_fwd_n[i].copy_(replay_n)

            del replay_xx, replay_xy, replay_n, tmp_fwd_xx, tmp_fwd_xy

            # --- Combine and solve ---
            S_xx = seg_fwd_xx + seg_bwd_xx + reg  # (seg_len, B, p, p)
            S_xy = seg_fwd_xy + seg_bwd_xy  # (seg_len, B, p, C)
            seg_n_eff = seg_fwd_n + seg_bwd_n  # (seg_len, B)

            del seg_fwd_xx, seg_fwd_xy, seg_fwd_n
            del seg_bwd_xx, seg_bwd_xy, seg_bwd_n

            # Solve and compute y_hat per timestep to avoid MAGMA batched
            # warnings for large p, and to avoid storing full beta_seg
            X_seg = X[:, seg_start:seg_end]  # (B, seg_len, p)
            y_hat_seg = torch.empty(seg_len, B, C, device=self.device,
                                    dtype=X.dtype)
            for i in range(seg_len):
                # S_xx[i]: (B, p, p), S_xy[i]: (B, p, C)
                beta_i = torch.linalg.solve(S_xx[i], S_xy[i])  # (B, p, C)
                # y_hat = X @ beta: (B, 1, p) @ (B, p, C) -> (B, C)
                y_hat_seg[i] = torch.matmul(
                    X_seg[:, i].unsqueeze(1), beta_i).squeeze(1)
            del S_xx, S_xy

            # R² from residuals vs EWMA mean
            y_seg = y[:, seg_start:seg_end].permute(1, 0, 2)  # (seg_len, B, C)
            y_mean_seg = y_mean[:, seg_start:seg_end].permute(1, 0, 2)

            residuals = y_seg - y_hat_seg
            ss_res = (residuals ** 2).sum(dim=2)  # (seg_len, B)
            ss_tot = ((y_seg - y_mean_seg) ** 2).sum(dim=2)  # (seg_len, B)

            with torch.no_grad():
                r2_seg = torch.where(
                    ss_tot > 1e-10,
                    1.0 - ss_res / ss_tot,
                    torch.zeros_like(ss_res)
                ).clamp(-1.0, 1.0)

            # Write to output tensors
            y_hat[:, seg_start:seg_end] = y_hat_seg.permute(1, 0, 2)
            r_squared[:, seg_start:seg_end] = r2_seg.permute(1, 0)
            n_eff[:, seg_start:seg_end] = seg_n_eff.permute(1, 0)

            del y_hat_seg, r2_seg, seg_n_eff, residuals, ss_res, ss_tot

        del ckpt_xx, ckpt_xy, ckpt_n
        del acc_bwd_xx, acc_bwd_xy, acc_bwd_n, tmp_bwd_xx, tmp_bwd_xy

        # Mask low-n timepoints
        low_n = n_eff < self.min_effective_n
        r_squared[low_n] = float('nan')

        if squeeze:
            return None, y_hat[0], r_squared[0], n_eff[0]
        return None, y_hat, r_squared, n_eff

    def _streaming_backward(self, X, y, valid_f, xx, xy, gamma):
        """Fallback streaming backward pass when Triton is unavailable.

        Uses a Python loop over T with running (B,p,p) accumulators.
        Combines with forward result in-place. Returns bwd_n (B, T).
        """
        B, T, p = X.shape
        C = y.shape[2]

        acc_xx = torch.zeros(B, p, p, device=self.device, dtype=X.dtype)
        acc_xy = torch.zeros(B, p, C, device=self.device, dtype=X.dtype)
        acc_n = torch.zeros(B, device=self.device, dtype=torch.float32)
        bwd_n = torch.zeros(B, T, device=self.device, dtype=torch.float32)

        tmp_xx = torch.empty(B, p, p, device=self.device, dtype=X.dtype)
        tmp_xy = torch.empty(B, p, C, device=self.device, dtype=X.dtype)

        for t in range(T - 2, -1, -1):
            x_t1 = X[:, t + 1]
            y_t1 = y[:, t + 1]
            v_t1 = valid_f[:, t + 1]

            acc_xx.mul_(gamma)
            acc_xy.mul_(gamma)
            acc_n.mul_(gamma)

            torch.bmm(x_t1.unsqueeze(2), x_t1.unsqueeze(1), out=tmp_xx)
            tmp_xx.mul_(v_t1.view(B, 1, 1))
            acc_xx.add_(tmp_xx)

            torch.bmm(x_t1.unsqueeze(2), y_t1.unsqueeze(1), out=tmp_xy)
            tmp_xy.mul_(v_t1.view(B, 1, 1))
            acc_xy.add_(tmp_xy)

            acc_n.add_(v_t1)

            xx[:, t].add_(acc_xx)
            xy[:, t].add_(acc_xy)
            bwd_n[:, t] = acc_n

        del acc_xx, acc_xy, acc_n, tmp_xx, tmp_xy
        return bwd_n

    def _ewma_mean_batched(self, y, valid_f, gamma, force_sequential=False):
        """Batched EWMA using forward + reversed backward multi-tensor scan."""
        B, T, C = y.shape

        y_valid = y * valid_f.unsqueeze(2)

        # Forward: scan sum (B,T,C) and n (B,T,1) together
        fwd_sum = y_valid.clone()
        fwd_n = valid_f.unsqueeze(-1).clone()  # (B, T, 1)
        _multi_exp_scan_fwd_([fwd_sum, fwd_n], gamma,
                             force_sequential=force_sequential)

        # Backward (shifted + reversed, multi-tensor scan)
        bwd_sum = torch.zeros_like(y)
        bwd_sum[:, :-1] = y_valid[:, 1:]
        bwd_n = torch.zeros(B, T, 1, device=self.device, dtype=torch.float32)
        bwd_n[:, :-1, 0] = valid_f[:, 1:]

        bwd_sum = bwd_sum.flip(1).contiguous()
        bwd_n = bwd_n.flip(1).contiguous()
        _multi_exp_scan_fwd_([bwd_sum, bwd_n], gamma,
                             force_sequential=force_sequential)

        bwd_sum = bwd_sum.flip(1)
        bwd_n = bwd_n.flip(1)

        total_n = fwd_n + bwd_n  # (B, T, 1)
        total_sum = fwd_sum + bwd_sum  # (B, T, C)
        y_mean = torch.where(
            total_n > 0,
            total_sum / (total_n + 1e-20),
            torch.zeros_like(y)
        )
        return y_mean

    def _ewma_mean(self, y, valid_f, valid_cpu, gamma):
        """Legacy single-pathway EWMA."""
        result = self._ewma_mean_batched(
            y.unsqueeze(0), valid_f.unsqueeze(0), gamma)
        return result[0]
