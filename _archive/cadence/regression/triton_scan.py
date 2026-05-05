"""Triton parallel prefix scan for exponential weighted accumulation.

Implements S[t] = gamma * S[t-1] + x[t] using Blelloch-style parallel scan.
O(log T) parallel depth per block instead of O(T) sequential Python steps.

Architecture (following accelerated-scan pattern):
  - Within-block: tl.associative_scan (Blelloch parallel prefix, ~11 steps for BLOCK=2048)
  - Between-blocks: sequential carry propagation via scalar h0
  - Across (B, D): fully parallel via Triton grid

Kernels:
  1. _strided_fwd_scan_kernel: Forward scan on (B,T,*trailing) layout using strides.
     No transpose/contiguous copy needed — saves ~20 GB for large tensors.
  2. _bwd_scan_combine_kernel: Fused backward scan that computes outer products
     on-the-fly from X and adds results to the forward tensor in-place.
     Replaces the O(T) Python loop with a single kernel launch.

Requires: triton-windows >= 3.6 (pip install triton-windows)
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _combine_fn(b_left, a_left, b_right, a_right):
        """Associative operator for first-order linear recurrence.

        Represents S = a * S_prev + b.
        Composition: (b1,a1) . (b2,a2) = (a2*b1 + b2, a1*a2)
        """
        return a_right * b_left + b_right, a_left * a_right

    # ------------------------------------------------------------------
    # Legacy kernels (kept for reference / fallback)
    # ------------------------------------------------------------------

    @triton.jit
    def _exp_scan_kernel(
        input_ptr, output_ptr,
        gamma,
        seqlen,
        stride_b,   # stride for batch dim
        stride_d,   # stride for feature dim
        BLOCK: tl.constexpr,
    ):
        """Forward exponential scan kernel: out[t] = gamma * out[t-1] + in[t].

        Grid: (B, D) — one program per batch element per feature dimension.
        Each program processes T elements in chunks of BLOCK.
        """
        bid = tl.program_id(0)   # batch index
        did = tl.program_id(1)   # feature index

        base_offset = bid * stride_b + did * stride_d
        nblocks = tl.cdiv(seqlen, BLOCK)
        idx = tl.arange(0, BLOCK)

        h0 = tl.zeros((), dtype=tl.float32)  # inter-block carry

        for block_id in tl.range(0, nblocks):
            t = block_id * BLOCK + idx
            offset = base_offset + t

            x = tl.load(input_ptr + offset, mask=t < seqlen, other=0.0)
            # Forget factor: gamma for all elements (constant)
            f = tl.full((BLOCK,), gamma, dtype=tl.float32)

            # Parallel prefix scan within block
            h, f_scan = tl.associative_scan((x, f), axis=0, combine_fn=_combine_fn)

            # Incorporate carry from previous blocks:
            # h_corrected[i] = h[i] + h0 * f_cumulative[i]
            h = h + h0 * f_scan

            tl.store(output_ptr + offset, h, mask=t < seqlen)

            # Update carry: last element of corrected scan
            # Use reduction to extract the last valid element
            last_idx = tl.minimum(BLOCK - 1, seqlen - 1 - block_id * BLOCK)
            h0 = tl.sum(tl.where(idx == last_idx, h, 0.0), axis=0)

    @triton.jit
    def _exp_scan_inplace_kernel(
        data_ptr,
        gamma,
        seqlen,
        stride_b,
        stride_d,
        BLOCK: tl.constexpr,
    ):
        """In-place forward exponential scan: data[t] += gamma * data[t-1].

        Same as _exp_scan_kernel but reads and writes to the same buffer.
        """
        bid = tl.program_id(0)
        did = tl.program_id(1)

        base_offset = bid * stride_b + did * stride_d
        nblocks = tl.cdiv(seqlen, BLOCK)
        idx = tl.arange(0, BLOCK)

        h0 = tl.zeros((), dtype=tl.float32)

        for block_id in tl.range(0, nblocks):
            t = block_id * BLOCK + idx
            offset = base_offset + t

            x = tl.load(data_ptr + offset, mask=t < seqlen, other=0.0)
            f = tl.full((BLOCK,), gamma, dtype=tl.float32)

            h, f_scan = tl.associative_scan((x, f), axis=0, combine_fn=_combine_fn)
            h = h + h0 * f_scan

            tl.store(data_ptr + offset, h, mask=t < seqlen)

            last_idx = tl.minimum(BLOCK - 1, seqlen - 1 - block_id * BLOCK)
            h0 = tl.sum(tl.where(idx == last_idx, h, 0.0), axis=0)

    # ------------------------------------------------------------------
    # Change 1: Strided forward scan — no transpose/copy overhead
    # ------------------------------------------------------------------

    @triton.jit
    def _strided_fwd_scan_kernel(
        data_ptr,
        gamma,
        T,
        D,
        stride_batch,
        stride_time,
        stride_feat,
        BLOCK: tl.constexpr,
    ):
        """In-place forward exponential scan on strided (B, T, D) layout.

        Grid: (B * D,) — one program per (batch, feature) pair.
        Each program scans T time steps at stride `stride_time` apart,
        avoiding the transpose + contiguous copy of the legacy kernel.
        """
        pid = tl.program_id(0)
        b = pid // D
        d = pid % D

        base = b * stride_batch + d * stride_feat
        nblocks = tl.cdiv(T, BLOCK)
        idx = tl.arange(0, BLOCK)

        h0 = tl.zeros((), dtype=tl.float32)

        for block_id in tl.range(0, nblocks):
            t = block_id * BLOCK + idx
            offset = base + t * stride_time

            x = tl.load(data_ptr + offset, mask=t < T, other=0.0)
            f = tl.full((BLOCK,), gamma, dtype=tl.float32)

            h, f_scan = tl.associative_scan((x, f), axis=0, combine_fn=_combine_fn)
            h = h + h0 * f_scan

            tl.store(data_ptr + offset, h, mask=t < T)

            last_idx = tl.minimum(BLOCK - 1, T - 1 - block_id * BLOCK)
            h0 = tl.sum(tl.where(idx == last_idx, h, 0.0), axis=0)

    # ------------------------------------------------------------------
    # Change 2: Fused backward scan + in-place combine
    # ------------------------------------------------------------------

    @triton.jit
    def _bwd_scan_combine_kernel(
        A_ptr,          # (B, T, D1) — first factor (X for both xx and xy)
        B_ptr,          # (B, T, D2) — second factor (X for xx, y for xy)
        valid_ptr,      # (B, T)     — validity mask as float
        fwd_ptr,        # (B, T, D1, D2) — forward result, combined in-place
        gamma,
        T,
        D1, D2,
        stride_A_b, stride_A_t, stride_A_d,
        stride_B_b, stride_B_t, stride_B_d,
        stride_v_b, stride_v_t,
        stride_fwd_b, stride_fwd_t, stride_fwd_d1, stride_fwd_d2,
        BLOCK: tl.constexpr,
    ):
        """Fused backward exponential scan with in-place combine.

        For each (b, i, j) element, computes the reverse scan:
            bwd[t] = gamma * bwd[t+1] + A[t+1, i] * B[t+1, j] * valid[t+1]
        then adds bwd[t] to fwd[t, i, j] in-place.

        Grid: (B * D1 * D2,) — one program per independent scan.

        Implements reverse scan by processing blocks end-to-start with
        reversed positions within each block, using the same associative
        scan primitive as the forward kernel.
        """
        pid = tl.program_id(0)
        total_d = D1 * D2
        b_idx = pid // total_d
        rem = pid % total_d
        ii = rem // D2
        jj = rem % D2

        nblocks = tl.cdiv(T, BLOCK)
        idx = tl.arange(0, BLOCK)

        h0 = tl.zeros((), dtype=tl.float32)  # inter-block carry

        for fwd_iter in tl.range(0, nblocks):
            blk = nblocks - 1 - fwd_iter  # process blocks end → start

            # Reversed positions within the block
            t_rev = blk * BLOCK + (BLOCK - 1 - idx)
            # Backward reads data from t+1
            t_src = t_rev + 1

            # Valid: t_rev in bounds AND t_src in bounds
            mask_valid = (t_rev < T) & (t_src < T)

            # Compute outer product element on-the-fly
            a_offset = b_idx * stride_A_b + t_src * stride_A_t + ii * stride_A_d
            b_offset = b_idx * stride_B_b + t_src * stride_B_t + jj * stride_B_d
            v_offset = b_idx * stride_v_b + t_src * stride_v_t

            a_val = tl.load(A_ptr + a_offset, mask=mask_valid, other=0.0)
            b_val = tl.load(B_ptr + b_offset, mask=mask_valid, other=0.0)
            v_val = tl.load(valid_ptr + v_offset, mask=mask_valid, other=0.0)

            inp = a_val * b_val * v_val

            f = tl.full((BLOCK,), gamma, dtype=tl.float32)
            h, f_scan = tl.associative_scan((inp, f), axis=0, combine_fn=_combine_fn)
            h = h + h0 * f_scan

            # Add backward value to forward tensor in-place
            # Skip t=T-1 (boundary: bwd[T-1]=0) and out-of-bounds positions
            mask_store = t_rev < T - 1
            fwd_offset = (b_idx * stride_fwd_b + t_rev * stride_fwd_t +
                          ii * stride_fwd_d1 + jj * stride_fwd_d2)
            fwd_val = tl.load(fwd_ptr + fwd_offset, mask=mask_store, other=0.0)
            tl.store(fwd_ptr + fwd_offset, fwd_val + h, mask=mask_store)

            # Carry: value at last scan position (idx=BLOCK-1 = earliest time in block)
            h0 = tl.sum(tl.where(idx == BLOCK - 1, h, 0.0), axis=0)


# ======================================================================
# Python wrappers
# ======================================================================

def triton_exp_scan_fwd_(tensors, gamma):
    """Strided in-place forward scan — no transpose/copy overhead.

    Drop-in replacement for the legacy version. Each tensor has shape
    (B, T, ...). Flattens trailing dims to D, then launches one Triton
    program per (b, d) pair that scans T time steps at stride D.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    for A in tensors:
        B, T = A.shape[0], A.shape[1]
        trailing = A.shape[2:]
        D = 1
        for s in trailing:
            D *= s

        # Reshape to (B, T, D) — view on contiguous tensor, no copy
        A_flat = A.reshape(B, T, D)
        stride_batch = A_flat.stride(0)
        stride_time = A_flat.stride(1)
        stride_feat = A_flat.stride(2)

        BLOCK = min(2048, triton.next_power_of_2(T))

        grid = (B * D,)
        _strided_fwd_scan_kernel[grid](
            A_flat, float(gamma), T, D,
            stride_batch, stride_time, stride_feat,
            BLOCK=BLOCK,
        )


def triton_bwd_scan_combine(A, B_tensor, valid_f, fwd, gamma):
    """Fused backward scan + in-place combine via Triton kernel.

    Computes for each (i, j):
        bwd[t, i, j] = gamma * bwd[t+1, i, j] + A[t+1, i] * B[t+1, j] * valid[t+1]
    and adds bwd to fwd in-place.

    Args:
        A: (B, T, D1) first factor (typically X).
        B_tensor: (B, T, D2) second factor (X for xx, y for xy).
        valid_f: (B, T) validity mask as float.
        fwd: (B, T, D1, D2) forward scan result — modified in-place.
        gamma: exponential decay factor.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    Batch, T, D1 = A.shape
    D2 = B_tensor.shape[2]

    BLOCK = min(2048, triton.next_power_of_2(T))

    grid = (Batch * D1 * D2,)
    _bwd_scan_combine_kernel[grid](
        A, B_tensor, valid_f, fwd,
        float(gamma),
        T, D1, D2,
        A.stride(0), A.stride(1), A.stride(2),
        B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
        valid_f.stride(0), valid_f.stride(1),
        fwd.stride(0), fwd.stride(1), fwd.stride(2), fwd.stride(3),
        BLOCK=BLOCK,
    )
