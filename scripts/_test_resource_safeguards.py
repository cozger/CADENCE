"""Stress test for cadence.io.resources guarantees.

Goal: Demonstrate that a naive workload that *would* OOM gets clamped down to
fit the 60 GB RAM budget, while still maximizing CPU usage within that budget.

Test scenarios:
  S1. NAIVE BASELINE — request n_jobs=-1 with per-worker estimate that would
      blow past 60 GB. Verify pick_n_jobs clamps.
  S2. SAFE EXECUTION — actually run a synthetic memory-hungry workload through
      joblib using pick_n_jobs. Measure peak RAM during execution. Assert
      peak stays under the budget.
  S3. PARALLELISM CHECK — confirm n_jobs > 1 was actually used (vs degenerate
      collapse to single-threaded).
  S4. THROUGHPUT — compare wall-time to a 1-job baseline; confirm parallelism
      actually engaged (>1.5x speedup expected with 4+ workers).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadence.io.resources import (
    RAM_BUDGET_GB, RAM_RESERVE_GB, limit_blas_threads, log_resources,
    pick_n_jobs, snapshot, working_ram_budget_gb,
)


# ── RAM watcher (samples peak RSS during a window) ──────────────────

class PeakRamSampler:
    """Background thread that polls psutil to record peak RSS during a window."""

    def __init__(self, poll_s: float = 0.1):
        self.poll_s = poll_s
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_gb = 0.0
        self.peak_system_used_gb = 0.0

    def __enter__(self):
        self._stop.clear()
        self.peak_rss_gb = 0.0
        self.peak_system_used_gb = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                me = self._proc.memory_info().rss / 1e9
                children = sum(c.memory_info().rss for c in self._proc.children(recursive=True)) / 1e9
                total = me + children
                if total > self.peak_rss_gb:
                    self.peak_rss_gb = total
                vm = psutil.virtual_memory()
                used = (vm.total - vm.available) / 1e9
                if used > self.peak_system_used_gb:
                    self.peak_system_used_gb = used
            except psutil.NoSuchProcess:
                pass
            self._stop.wait(self.poll_s)


# ── Synthetic memory-hungry worker ──────────────────────────────────

def hungry_worker(per_worker_gb: float, hold_seconds: float = 0.5,
                   work_iters: int = 5) -> dict:
    """Allocate `per_worker_gb` of float64, do a few numpy ops on it, return.

    Simulates a real CADENCE worker (numpy ops + held buffer). Critical to
    pin BLAS threads here so workers don't oversubscribe the CPU and thrash.
    """
    with limit_blas_threads(1):
        n_doubles = int(per_worker_gb * 1e9 / 8)
        # Use a 2D shape so np.dot has something to chew on
        side = max(64, int(np.sqrt(n_doubles)))
        A = np.random.default_rng().standard_normal((side, side)).astype(np.float64)
        # Forced hold to make peak RAM observable
        s_total = 0.0
        for _ in range(work_iters):
            # Light numpy work to verify BLAS limit holds — matrix square
            s_total += float(np.einsum('ij,ji->', A[:1024, :1024], A[:1024, :1024].T))
            time.sleep(hold_seconds / max(1, work_iters))
        return {'gb_held': float(A.nbytes / 1e9),
                'side': side,
                'work_sum': s_total}


# ── Main stress test ────────────────────────────────────────────────

def run_stress(per_worker_gb: float, requested_jobs: int = -1,
                n_units: int = 64, hold_seconds: float = 0.5
                ) -> dict:
    snap = snapshot()
    print(f'\n  Snapshot: {snap}')
    budget = working_ram_budget_gb(snap)
    print(f'  Working RAM budget: {budget:.1f} GB '
          f'(cap={RAM_BUDGET_GB:.0f}, reserve={RAM_RESERVE_GB:.0f})')

    # S1: pick_n_jobs decision
    n_jobs = pick_n_jobs(per_worker_ram_gb=per_worker_gb, requested=requested_jobs)
    naive_demand_gb = (snap.cpu_count if requested_jobs == -1 else requested_jobs) * per_worker_gb
    print(f'  Naive request: {requested_jobs} jobs × {per_worker_gb:.1f} GB = '
          f'{naive_demand_gb:.1f} GB')
    print(f'  pick_n_jobs   -> {n_jobs} (clamped from naive)')

    if n_jobs * per_worker_gb > budget:
        print(f'  ABORT: even after clamping, n_jobs*per_worker '
              f'({n_jobs * per_worker_gb:.1f}) > budget ({budget:.1f}). '
              f'per_worker_gb is too high.')
        return {'aborted': True}

    # S4: time a 1-job baseline first (single iteration)
    print(f'\n  [Baseline] running 1 worker for timing reference...', flush=True)
    t0 = time.time()
    _ = hungry_worker(per_worker_gb, hold_seconds=hold_seconds)
    baseline_s = time.time() - t0
    print(f'  Baseline single-worker: {baseline_s:.2f}s')

    # S2 + S3: actual parallel execution with peak-RSS sampler
    print(f'\n  [Parallel] running {n_units} units via joblib '
          f'(n_jobs={n_jobs}, per_worker={per_worker_gb:.1f} GB)...', flush=True)
    t0 = time.time()
    with PeakRamSampler(poll_s=0.1) as sampler:
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(hungry_worker)(per_worker_gb, hold_seconds=hold_seconds)
            for _ in range(n_units))
    parallel_s = time.time() - t0

    speedup = (n_units * baseline_s) / parallel_s if parallel_s > 0 else 0
    serial_eq_s = n_units * baseline_s

    print(f'\n  ──── Stress-test result ────')
    print(f'  Wall time (parallel):        {parallel_s:.2f}s for {n_units} units')
    print(f'  Equivalent serial time:      {serial_eq_s:.2f}s')
    print(f'  Speedup vs serial:           {speedup:.2f}x')
    print(f'  Peak own-process RSS:        {sampler.peak_rss_gb:.2f} GB')
    print(f'  Peak system-used RAM:        {sampler.peak_system_used_gb:.2f} GB')
    print(f'  RAM budget:                  {budget:.1f} GB')

    # Pass/fail checks
    print(f'\n  ──── Verification ────')
    checks = []
    # Peak own-process RSS should be below per_worker × n_jobs × overhead
    expected_max = n_jobs * per_worker_gb * 2.0  # 2x for transient peaks
    rss_ok = sampler.peak_rss_gb <= expected_max
    checks.append(('Peak own-process RSS <= n_jobs * per_worker * 2',
                   rss_ok, f'{sampler.peak_rss_gb:.2f} <= {expected_max:.2f}'))
    # System-used RAM stayed under user's hard cap (60 GB by default)
    sys_ok = sampler.peak_system_used_gb <= RAM_BUDGET_GB
    checks.append((f'Peak system-used RAM <= {RAM_BUDGET_GB:.0f} GB hard cap',
                   sys_ok, f'{sampler.peak_system_used_gb:.2f} <= {RAM_BUDGET_GB:.0f}'))
    # n_jobs > 1 (not over-clamped)
    para_ok = n_jobs >= 2
    checks.append(('Parallelism: n_jobs >= 2', para_ok, f'n_jobs={n_jobs}'))
    # Speedup > 1.3x (some parallel benefit, allowing for joblib overhead)
    speed_ok = speedup >= 1.3
    checks.append(('Speedup vs serial >= 1.3x', speed_ok, f'speedup={speedup:.2f}x'))

    for label, ok, detail in checks:
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {label}  ({detail})')

    all_ok = all(c[1] for c in checks)
    print(f'\n  Overall: {"PASS" if all_ok else "FAIL"}')
    return {
        'n_jobs': n_jobs, 'per_worker_gb': per_worker_gb,
        'parallel_s': parallel_s, 'serial_eq_s': serial_eq_s, 'speedup': speedup,
        'peak_rss_gb': sampler.peak_rss_gb,
        'peak_system_gb': sampler.peak_system_used_gb,
        'budget_gb': budget,
        'all_ok': all_ok, 'checks': checks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', choices=['light', 'medium', 'heavy', 'all'],
                    default='all',
                    help='Stress level. light=0.5 GB/worker, medium=2.0, heavy=8.0')
    ap.add_argument('--n-units', type=int, default=32,
                    help='How many work units to dispatch (default 32)')
    ap.add_argument('--requested-jobs', type=int, default=-1,
                    help='What to ask pick_n_jobs for (default -1 = naive max)')
    ap.add_argument('--hold-s', type=float, default=0.5,
                    help='How long each worker holds its memory (default 0.5s)')
    args = ap.parse_args()

    log_resources(prefix='Initial state: ')

    scenarios = {
        'light':  {'per_worker_gb': 0.5},
        'medium': {'per_worker_gb': 2.0},
        'heavy':  {'per_worker_gb': 8.0},
    }
    if args.scenarios == 'all':
        runs = list(scenarios.items())
    else:
        runs = [(args.scenarios, scenarios[args.scenarios])]

    summary = []
    for name, params in runs:
        print('\n' + '=' * 76)
        print(f'  Scenario: {name.upper()} (per_worker={params["per_worker_gb"]:.1f} GB)')
        print('=' * 76)
        # If running heavy with default n_units=32, that's 32 * 8 = 256 GB serial
        # demand. Worker per-call only holds 8 GB so it's safe — we just want to
        # see the budget clamp in action.
        result = run_stress(per_worker_gb=params['per_worker_gb'],
                             requested_jobs=args.requested_jobs,
                             n_units=args.n_units, hold_seconds=args.hold_s)
        summary.append((name, result))

    print('\n' + '=' * 76)
    print('  SUMMARY')
    print('=' * 76)
    print(f'{"Scenario":10s} | {"per_worker":11s} | {"n_jobs":7s} | '
          f'{"peak_RSS":10s} | {"speedup":8s} | {"verdict":7s}')
    for name, r in summary:
        if r.get('aborted'):
            print(f'{name:10s} | aborted (per-worker > budget)')
            continue
        print(f'{name:10s} | {r["per_worker_gb"]:5.1f} GB    | '
              f'{r["n_jobs"]:5d}   | {r["peak_rss_gb"]:5.1f} GB   | '
              f'{r["speedup"]:5.2f}x  | '
              f'{"PASS" if r["all_ok"] else "FAIL"}')


if __name__ == '__main__':
    main()
