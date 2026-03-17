"""Correctness + performance tests for Triton kernels at V2-scale dimensions.

V2 pipeline typical dimensions:
  - EEG->BL:    B=1, T=3000, p=293, C=31
  - Pose->Pose: B=1, T=3000, p=283, C=41
  - BL->BL:     B=1, T=3000, p=165, C=31

Run: python scripts/test_triton_kernels.py
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')

from cadence.regression.ewls import (
    _sequential_exp_scan_fwd_,
    _multi_exp_scan_fwd_,
    _HAS_TRITON_SCAN,
    _HAS_TRITON_BWD,
    EWLSSolver,
)
from cadence.regression.triton_scan import HAS_TRITON


def test_strided_forward_scan():
    """Strided forward scan vs sequential reference — V2 scale shapes."""
    print("=== Test 1: Strided Forward Scan (V2 scale) ===")

    if not HAS_TRITON:
        print("  SKIP: Triton not available")
        return True

    torch.manual_seed(42)
    device = 'cuda'
    gamma = 0.99  # V2 uses eval_rate=5, tau=30 → gamma≈0.993

    shapes = [
        (1, 3000, 30, 30),       # V2 medium pathway xx
        (1, 3000, 30, 8),        # V2 medium pathway xy
        (1, 1500, 50, 50),       # V2 larger p, shorter T
        (1, 3000, 1, 1),         # n_fwd (tiny D)
    ]

    all_pass = True
    for shape in shapes:
        ref = torch.randn(*shape, device=device, dtype=torch.float32)
        test = ref.clone()

        _sequential_exp_scan_fwd_([ref], gamma)
        _multi_exp_scan_fwd_([test], gamma)

        max_err = (ref - test).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-10)

        status = "PASS" if rel_err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  shape={shape}: max_err={max_err:.2e}, rel_err={rel_err:.2e} [{status}]")

    return all_pass


def test_backward_scan_combine():
    """Backward scan + combine correctness at V2 dimensions."""
    print("\n=== Test 2: Backward Scan + Combine (V2 scale) ===")

    if not _HAS_TRITON_BWD:
        print("  SKIP: Triton backward scan not available")
        return True

    from cadence.regression.triton_scan import triton_bwd_scan_combine

    torch.manual_seed(123)
    device = 'cuda'

    # V2-representative configs: (B, T, p, C)
    configs = [
        (1, 500,  30,  8),    # V2 medium — fits in VRAM for ref
        (1, 300,  50, 10),    # V2 large p, short T
        (1, 1000, 20,  6),    # V2 small p, longer T
    ]

    all_pass = True
    for B, T, p, C in configs:
        gamma = 0.993

        X = torch.randn(B, T, p, device=device, dtype=torch.float32)
        y = torch.randn(B, T, C, device=device, dtype=torch.float32)
        valid = torch.ones(B, T, dtype=torch.bool, device=device)
        valid[:, :5] = False
        valid[:, T//2:T//2+10] = False  # gap in middle
        valid_f = valid.float()

        # -- Reference: sequential --
        xx_ref = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
        xy_ref = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))
        inv_mask = (~valid).unsqueeze(-1).unsqueeze(-1)
        xx_ref.masked_fill_(inv_mask, 0.0)
        xy_ref.masked_fill_(inv_mask, 0.0)
        _sequential_exp_scan_fwd_([xx_ref, xy_ref], gamma)

        acc_xx = torch.zeros(B, p, p, device=device, dtype=torch.float32)
        acc_xy = torch.zeros(B, p, C, device=device, dtype=torch.float32)
        for t in range(T - 2, -1, -1):
            x_t1 = X[:, t + 1]
            y_t1 = y[:, t + 1]
            v_t1 = valid_f[:, t + 1]
            acc_xx.mul_(gamma)
            acc_xy.mul_(gamma)
            acc_xx.add_(torch.bmm(x_t1.unsqueeze(2), x_t1.unsqueeze(1)) *
                        v_t1.view(B, 1, 1))
            acc_xy.add_(torch.bmm(x_t1.unsqueeze(2), y_t1.unsqueeze(1)) *
                        v_t1.view(B, 1, 1))
            xx_ref[:, t].add_(acc_xx)
            xy_ref[:, t].add_(acc_xy)

        # -- Test: Triton --
        xx_test = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
        xy_test = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))
        xx_test.masked_fill_(inv_mask, 0.0)
        xy_test.masked_fill_(inv_mask, 0.0)
        _multi_exp_scan_fwd_([xx_test, xy_test], gamma)

        triton_bwd_scan_combine(X, X, valid_f, xx_test, gamma)
        triton_bwd_scan_combine(X, y, valid_f, xy_test, gamma)

        rel_xx = (xx_ref - xx_test).abs().max().item() / (xx_ref.abs().max().item() + 1e-10)
        rel_xy = (xy_ref - xy_test).abs().max().item() / (xy_ref.abs().max().item() + 1e-10)

        status = "PASS" if rel_xx < 1e-3 and rel_xy < 1e-3 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  B={B} T={T} p={p} C={C}: "
              f"xx rel={rel_xx:.2e}, xy rel={rel_xy:.2e} [{status}]")

        del xx_ref, xy_ref, xx_test, xy_test, X, y
        torch.cuda.empty_cache()

    return all_pass


def test_full_solve_v2():
    """End-to-end EWLS at V2 dimensions: Triton vs sequential reference."""
    print("\n=== Test 3: Full EWLS Solve (V2 scale) ===")

    torch.manual_seed(7)
    device = 'cuda'

    # V2 BL->BL pathway scale: p=165, C=31, T=1500 (300s @ 5Hz)
    B, T, p, C = 1, 1500, 40, 8

    X = torch.randn(B, T, p, device=device, dtype=torch.float32)
    y = torch.randn(B, T, C, device=device, dtype=torch.float32)
    valid = torch.ones(B, T, dtype=torch.bool, device=device)
    valid[:, :5] = False

    solver = EWLSSolver(tau_seconds=30.0, lambda_ridge=1e-3, eval_rate=5.0,
                        device=device)

    # Triton path
    beta_tri, yhat_tri, r2_tri, neff_tri = solver.solve_batched(X, y, valid)

    # Sequential reference
    beta_seq, yhat_seq, r2_seq, neff_seq = _solve_sequential(solver, X, y, valid)

    beta_err = (beta_tri - beta_seq).abs().max().item()
    beta_scale = beta_seq.abs().max().item() + 1e-10
    r2_mask = ~torch.isnan(r2_tri) & ~torch.isnan(r2_seq)
    r2_err = (r2_tri[r2_mask] - r2_seq[r2_mask]).abs().max().item() if r2_mask.any() else 0.0
    neff_err = (neff_tri - neff_seq).abs().max().item()
    neff_scale = neff_seq.abs().max().item() + 1e-10

    print(f"  beta: max_err={beta_err:.2e}, rel={beta_err/beta_scale:.2e}")
    print(f"  r2:   max_err={r2_err:.2e}")
    print(f"  neff: max_err={neff_err:.2e}, rel={neff_err/neff_scale:.2e}")

    ok = (beta_err / beta_scale < 1e-2 and r2_err < 5e-3 and neff_err / neff_scale < 1e-3)
    print(f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def _solve_sequential(solver, X, y, valid):
    """EWLS with forced sequential (no Triton) backward pass."""
    B, T, p = X.shape
    C = y.shape[2]

    dt = 1.0 / solver.eval_rate
    gamma = np.exp(-dt / solver.tau) if solver.tau > 0 else 0.0
    valid_f = valid.float()

    xx = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
    xy = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))

    inv_mask = (~valid).unsqueeze(-1).unsqueeze(-1)
    xx.masked_fill_(inv_mask, 0.0)
    xy.masked_fill_(inv_mask, 0.0)

    n_fwd = torch.zeros(B, T, 1, 1, device=X.device, dtype=torch.float32)
    n_fwd[:, :, 0, 0] = valid_f
    _sequential_exp_scan_fwd_([xx, xy, n_fwd], gamma)
    n_fwd_2d = n_fwd.squeeze(-1).squeeze(-1)

    bwd_n = solver._streaming_backward(X, y, valid_f, xx, xy, gamma)
    n_eff = n_fwd_2d + bwd_n

    reg = solver.lam * torch.eye(p, device=X.device, dtype=torch.float32)
    xx += reg
    beta = torch.linalg.solve(xx, xy)
    y_hat = torch.matmul(X.unsqueeze(2), beta).squeeze(2)

    y_mean = solver._ewma_mean_batched(y, valid_f, gamma)
    ss_res = ((y - y_hat) ** 2).sum(dim=2)
    ss_tot = ((y - y_mean) ** 2).sum(dim=2)

    with torch.no_grad():
        r2 = torch.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, torch.zeros_like(ss_res))
        r2 = r2.clamp(-1.0, 1.0)

    low_n = n_eff < solver.min_effective_n
    r2[low_n] = float('nan')
    beta[low_n] = 0.0
    return beta, y_hat, r2, n_eff


def test_timing_v2():
    """Benchmark at V2 scale: Triton multi-stream vs sequential backward."""
    print("\n=== Test 4: Timing (V2 scale) ===")

    if not _HAS_TRITON_BWD:
        print("  SKIP: Triton backward scan not available")
        return True

    from cadence.regression.triton_scan import triton_exp_scan_fwd_, triton_bwd_scan_combine

    device = 'cuda'

    # Sweep V2-representative sizes
    configs = [
        ("V1 small",       1,  1200,  20,   4),
        ("V2 ECG->BL",     1,  3000,  60,  31),
        ("V2 BL->BL",      1,  3000, 100,  31),
        ("V2 EEG->Pose",   1,  3000, 150,  41),
    ]

    for name, B, T, p, C in configs:
        gamma = 0.993
        torch.manual_seed(0)
        X = torch.randn(B, T, p, device=device, dtype=torch.float32)
        y = torch.randn(B, T, C, device=device, dtype=torch.float32)
        valid_f = torch.ones(B, T, device=device, dtype=torch.float32)

        # -- Warmup Triton (JIT compile) --
        xx_w = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
        xy_w = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))
        triton_exp_scan_fwd_([xx_w], gamma)
        triton_bwd_scan_combine(X, X, valid_f, xx_w, gamma)
        triton_bwd_scan_combine(X, y, valid_f, xy_w, gamma)
        torch.cuda.synchronize()
        del xx_w, xy_w
        torch.cuda.empty_cache()

        # -- Triton multi-stream (matches solve_batched path) --
        xx_t = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
        xy_t = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))

        bwd_n_t = torch.zeros(B, T, 1, device=device, dtype=torch.float32)
        bwd_n_t[:, :-1, 0] = valid_f[:, 1:]
        bwd_n_t = bwd_n_t.flip(1).contiguous()

        n_fwd_t = valid_f.unsqueeze(-1).unsqueeze(-1).clone()

        ready = torch.cuda.Event()
        s_xx = torch.cuda.Stream(device=device)
        s_xy = torch.cuda.Stream(device=device)
        s_n = torch.cuda.Stream(device=device)

        torch.cuda.synchronize()
        ready.record()
        t0 = time.perf_counter()

        with torch.cuda.stream(s_xx):
            s_xx.wait_event(ready)
            triton_exp_scan_fwd_([xx_t], gamma)
            triton_bwd_scan_combine(X, X, valid_f, xx_t, gamma)
        with torch.cuda.stream(s_xy):
            s_xy.wait_event(ready)
            triton_exp_scan_fwd_([xy_t], gamma)
            triton_bwd_scan_combine(X, y, valid_f, xy_t, gamma)
        with torch.cuda.stream(s_n):
            s_n.wait_event(ready)
            triton_exp_scan_fwd_([n_fwd_t], gamma)
            triton_exp_scan_fwd_([bwd_n_t], gamma)

        s_xx.synchronize()
        s_xy.synchronize()
        s_n.synchronize()
        t_triton = time.perf_counter() - t0

        del xx_t, xy_t, bwd_n_t, n_fwd_t
        torch.cuda.empty_cache()

        # -- Sequential backward --
        xx_s = torch.matmul(X.unsqueeze(3), X.unsqueeze(2))
        xy_s = torch.matmul(X.unsqueeze(3), y.unsqueeze(2))
        _sequential_exp_scan_fwd_([xx_s, xy_s], gamma)

        acc_xx = torch.zeros(B, p, p, device=device, dtype=torch.float32)
        acc_xy = torch.zeros(B, p, C, device=device, dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(T - 2, -1, -1):
            acc_xx.mul_(gamma)
            acc_xy.mul_(gamma)
            acc_xx.add_(torch.bmm(X[:, t+1].unsqueeze(2), X[:, t+1].unsqueeze(1)) *
                        valid_f[:, t+1].view(B, 1, 1))
            acc_xy.add_(torch.bmm(X[:, t+1].unsqueeze(2), y[:, t+1].unsqueeze(1)) *
                        valid_f[:, t+1].view(B, 1, 1))
            xx_s[:, t].add_(acc_xx)
            xy_s[:, t].add_(acc_xy)
        torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0

        speedup = t_seq / (t_triton + 1e-10)
        print(f"  {name:16s} (p={p:3d}, C={C:2d}): "
              f"Triton={t_triton*1000:7.1f}ms  Seq={t_seq*1000:7.1f}ms  "
              f"Speedup={speedup:5.0f}x")

        del xx_s, xy_s, acc_xx, acc_xy, X, y
        torch.cuda.empty_cache()

    return True


def test_full_solve_timing_v2():
    """End-to-end solve_batched timing at V2 dimensions."""
    print("\n=== Test 5: Full solve_batched Timing (V2 scale) ===")

    device = 'cuda'

    configs = [
        ("V2 small",   1, 1500,  40,  8),
        ("V2 medium",  1, 3000,  80, 16),
        ("V2 large",   1, 3000, 150, 31),
    ]

    solver = EWLSSolver(tau_seconds=30.0, lambda_ridge=1e-3, eval_rate=5.0,
                        device=device)

    for name, B, T, p, C in configs:
        torch.manual_seed(42)
        X = torch.randn(B, T, p, device=device, dtype=torch.float32)
        y = torch.randn(B, T, C, device=device, dtype=torch.float32)
        valid = torch.ones(B, T, dtype=torch.bool, device=device)

        # Warmup
        try:
            solver.solve_batched(X.clone(), y.clone(), valid.clone())
        except Exception:
            pass
        torch.cuda.empty_cache()

        # Timed run
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        beta, yhat, r2, neff = solver.solve_batched(X, y, valid)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        xx_mb = B * T * p * p * 4 / 1e6

        print(f"  {name:12s} (p={p:3d}, C={C:2d}, T={T}): "
              f"{elapsed*1000:7.1f}ms  peak={peak_mb:6.0f}MB  "
              f"xx_tensor={xx_mb:.0f}MB")

        del X, y, valid, beta, yhat, r2, neff
        torch.cuda.empty_cache()

    return True


def test_memory_v2():
    """Memory check at V2 scale."""
    print("\n=== Test 6: Memory Check (V2 scale) ===")

    torch.manual_seed(42)
    device = 'cuda'
    B, T, p, C = 1, 3000, 80, 16

    solver = EWLSSolver(tau_seconds=30.0, lambda_ridge=1e-3, eval_rate=5.0,
                        device=device)

    X = torch.randn(B, T, p, device=device, dtype=torch.float32)
    y = torch.randn(B, T, C, device=device, dtype=torch.float32)
    valid = torch.ones(B, T, dtype=torch.bool, device=device)

    torch.cuda.reset_peak_memory_stats()
    _ = solver.solve_batched(X, y, valid)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # xx alone: B*T*p*p*4 bytes
    xx_mb = B * T * p * p * 4 / 1e6
    ratio = peak_mb / (xx_mb + 1e-10)

    print(f"  Peak: {peak_mb:.0f} MB, xx alone: {xx_mb:.0f} MB, Peak/xx: {ratio:.1f}x")
    # Should be ~3-4x xx (xx + xy + beta + solve intermediates), not 5-6x
    ok = ratio < 6.0
    print(f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == '__main__':
    print(f"Triton available: {HAS_TRITON}")
    print(f"Triton forward scan: {_HAS_TRITON_SCAN}")
    print(f"Triton backward scan: {_HAS_TRITON_BWD}")
    print()

    results = []
    results.append(("Strided Forward Scan", test_strided_forward_scan()))
    results.append(("Backward Scan+Combine", test_backward_scan_combine()))
    results.append(("Full Solve (V2)", test_full_solve_v2()))
    results.append(("Timing (V2)", test_timing_v2()))
    results.append(("Full Solve Timing (V2)", test_full_solve_timing_v2()))
    results.append(("Memory (V2)", test_memory_v2()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name}: {status}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
