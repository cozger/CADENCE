"""Adaptive resource budgeting for CADENCE compute.

After the Phase 0 OOM incident (2026-05-01), every parallel CADENCE compute
must stay within hard caps:

    RAM_BUDGET_GB  = 60.0   (user-specified upper limit)
    VRAM_BUDGET_GB = 16.0   (matches GPU hardware)

The OOM cause was joblib `prefer='threads'` with `n_jobs=-1` on CPU work that
calls into numpy/scipy: each python thread spawns ~16 BLAS threads by default,
so a 16-thread joblib pool became 256 contending threads, each carrying its
own BLAS workspace. The cure is twofold:

    1. Inside every worker that does numpy/scipy work, call
       `with limit_blas_threads(1):` (a thin wrapper over
       threadpoolctl.threadpool_limits) — pins each worker to 1 BLAS thread.

    2. Set n_jobs adaptively from a per-worker memory estimate:
            n_jobs = pick_n_jobs(per_worker_ram_gb=<estimate>)
       which clamps to min(requested_jobs, cpu_count, RAM_BUDGET / per_worker).

For GPU operations, follow the chunking pattern from
`extract_burst_grids` / `gpu_sliding_te_surrogates`: hold no more than
VRAM_BUDGET_GB on device at peak; transfer chunks rather than full tensors.

This module is intentionally small and dependency-light (psutil, optional
threadpoolctl, optional nvidia-smi parse) so it can be imported by any
CADENCE script without pulling heavy deps into hot paths.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import psutil

# ── User-configurable hard caps (override via env if ever needed) ────

RAM_BUDGET_GB: float = float(os.environ.get('CADENCE_RAM_BUDGET_GB', '60.0'))
VRAM_BUDGET_GB: float = float(os.environ.get('CADENCE_VRAM_BUDGET_GB', '16.0'))

# Safety reserve subtracted from the budget when computing parallelism — keeps
# headroom for the OS, IDE, browsers, and any other background processes the
# user is actively using. Defaults give a useful working budget on a 64 GB box.
RAM_RESERVE_GB: float = float(os.environ.get('CADENCE_RAM_RESERVE_GB', '8.0'))
VRAM_RESERVE_GB: float = float(os.environ.get('CADENCE_VRAM_RESERVE_GB', '1.5'))


# ── Memory introspection ────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    ram_total_gb: float
    ram_available_gb: float
    vram_total_gb: float
    vram_free_gb: float
    cpu_count: int

    def __repr__(self):
        return (f'MemorySnapshot(RAM={self.ram_available_gb:.1f}/'
                f'{self.ram_total_gb:.1f}GB free, '
                f'VRAM={self.vram_free_gb:.1f}/{self.vram_total_gb:.1f}GB free, '
                f'CPUs={self.cpu_count})')


def _query_nvidia_smi() -> tuple[float, float]:
    """Return (vram_total_gb, vram_free_gb). Returns (0, 0) if no GPU."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0 or not out.stdout.strip():
            return 0.0, 0.0
        # Sum across GPUs (typically just one for CADENCE)
        total_mb = 0.0
        free_mb = 0.0
        for line in out.stdout.strip().splitlines():
            parts = line.split(',')
            if len(parts) >= 2:
                total_mb += float(parts[0])
                free_mb += float(parts[1])
        return total_mb / 1024.0, free_mb / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 0.0, 0.0


def snapshot() -> MemorySnapshot:
    """Take a snapshot of currently-available compute resources."""
    vm = psutil.virtual_memory()
    vram_total, vram_free = _query_nvidia_smi()
    return MemorySnapshot(
        ram_total_gb=vm.total / 1e9,
        ram_available_gb=vm.available / 1e9,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        cpu_count=os.cpu_count() or 1,
    )


# ── Adaptive parallelism ────────────────────────────────────────────

def working_ram_budget_gb(snap: Optional[MemorySnapshot] = None) -> float:
    """Headroom for OUR new allocations such that system-used stays within caps.

    The 60 GB user cap is interpreted as a *system-wide* upper bound on total
    used RAM (not "we may allocate up to 60 GB regardless of other processes").
    So the per-call headroom is the user cap MINUS what other processes are
    already consuming, also subject to the hardware-reserve floor.

    Returns:
        min(
            RAM_BUDGET_GB - already_used_gb,
            ram_total - RAM_RESERVE_GB - already_used_gb,
        )

    where ``already_used_gb = ram_total - ram_available``. The 2026-05-01
    stress test demonstrated that the previous formula
    (``min(RAM_BUDGET_GB, ram_total - RAM_RESERVE_GB, ram_available)``)
    breached the 60 GB cap whenever the IDE/browser/OS were already consuming
    ~15-20 GB at run start, because ``RAM_BUDGET_GB`` was treated as our
    quota rather than as a system-wide ceiling.
    """
    snap = snap or snapshot()
    already_used_gb = max(0.0, snap.ram_total_gb - snap.ram_available_gb)
    headroom_user_cap = max(0.0, RAM_BUDGET_GB - already_used_gb)
    headroom_hw = max(0.0, snap.ram_total_gb - RAM_RESERVE_GB - already_used_gb)
    return float(min(headroom_user_cap, headroom_hw))


def working_vram_budget_gb(snap: Optional[MemorySnapshot] = None) -> float:
    """Effective VRAM budget for a single GPU operation (peak)."""
    snap = snap or snapshot()
    by_user_cap = VRAM_BUDGET_GB
    by_hw = max(0.0, snap.vram_total_gb - VRAM_RESERVE_GB)
    by_now = max(0.0, snap.vram_free_gb)
    return float(min(by_user_cap, by_hw, by_now))


def pick_n_jobs(per_worker_ram_gb: float, requested: int = -1,
                snap: Optional[MemorySnapshot] = None,
                blas_overhead_factor: float = 2.0,
                budget_safety_factor: float = 0.9,
                max_jobs_hard_cap: Optional[int] = None,
                ) -> int:
    """Compute a safe n_jobs given a per-worker RAM estimate.

    Args:
        per_worker_ram_gb: peak RAM (GB) one worker is expected to use
            (estimate from data size; do NOT include BLAS/numpy overhead — that
            is applied via blas_overhead_factor).
        requested: user's requested n_jobs (-1 = "as many as safe").
        snap: optional pre-taken memory snapshot.
        blas_overhead_factor: even with limit_blas_threads(1), each worker
            carries BLAS workspace, Python object overhead, and transient
            numpy/scipy temp arrays. The 2026-05-01 stress test measured
            actual overhead ~1.83x for a worker holding a single big numpy
            array; 2.0x is the conservative default.
        budget_safety_factor: only commit this fraction of the working budget
            to allocations (default 0.9 = leave 10% slack for misc growth).
        max_jobs_hard_cap: optional ceiling regardless of memory math.

    Returns:
        n_jobs >= 1 that fits in budget_safety_factor * working_ram_budget_gb.
    """
    snap = snap or snapshot()
    budget = working_ram_budget_gb(snap) * float(budget_safety_factor)
    eff_per_worker = max(per_worker_ram_gb * blas_overhead_factor, 0.05)
    by_ram = max(1, int(budget / eff_per_worker))
    by_cpu = snap.cpu_count
    if requested == -1 or requested == 0:
        wanted = by_cpu
    else:
        wanted = int(requested)
    n = min(wanted, by_cpu, by_ram)
    if max_jobs_hard_cap is not None:
        n = min(n, int(max_jobs_hard_cap))
    return max(1, n)


# ── BLAS thread guard ───────────────────────────────────────────────

_THREADPOOLCTL_AVAILABLE: Optional[bool] = None


def _have_threadpoolctl() -> bool:
    global _THREADPOOLCTL_AVAILABLE
    if _THREADPOOLCTL_AVAILABLE is None:
        try:
            import threadpoolctl  # noqa: F401
            _THREADPOOLCTL_AVAILABLE = True
        except ImportError:
            _THREADPOOLCTL_AVAILABLE = False
    return _THREADPOOLCTL_AVAILABLE


@contextmanager
def limit_blas_threads(n: int = 1):
    """Pin BLAS / OpenMP thread count for the duration of the context.

    Required inside every joblib worker that does numpy/scipy work, otherwise
    each worker oversubscribes the CPU and bloats memory. No-op (with a warning
    on first use) if threadpoolctl is not installed.
    """
    if _have_threadpoolctl():
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=int(n), user_api='blas'):
            yield
    else:
        warnings.warn(
            'threadpoolctl unavailable — install it (`pip install threadpoolctl`) '
            'to prevent BLAS thread bloat in joblib workers.',
            RuntimeWarning, stacklevel=2)
        yield


# ── Optional helper: GPU chunk-size estimator ───────────────────────

def gpu_chunk_size(per_unit_vram_gb: float, n_units: int,
                    snap: Optional[MemorySnapshot] = None,
                    safety_factor: float = 0.7,
                    ) -> int:
    """Pick chunk size for a GPU loop so peak VRAM stays under budget.

    Args:
        per_unit_vram_gb: VRAM cost of processing one unit (e.g. one surrogate,
            one frequency band, one window).
        n_units: total units in the full computation.
        snap: optional snapshot.
        safety_factor: leave headroom for autograd buffers, kernel scratch,
            cuFFT plans. Default 0.7 (use 70% of free VRAM at most).

    Returns:
        chunk_size in [1, n_units] such that
        chunk_size * per_unit_vram_gb <= safety_factor * working_vram_budget_gb.
    """
    snap = snap or snapshot()
    budget = working_vram_budget_gb(snap) * safety_factor
    if per_unit_vram_gb <= 0:
        return n_units
    chunk = max(1, int(budget / per_unit_vram_gb))
    return min(chunk, n_units)


# ── Convenience: log resource state ────────────────────────────────

def log_resources(prefix: str = '') -> MemorySnapshot:
    """Print a resource snapshot to stdout for traceability in long runs."""
    snap = snapshot()
    print(f'{prefix}{snap}', flush=True)
    return snap
