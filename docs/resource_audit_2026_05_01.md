# Resource-discipline audit (2026-05-01)

Scope: `cadence/ingest/`, `cadence/preprocess/`, `cadence/significance/` GPU
users (`fast_cycles.py`, `bl_wavelet.py`, `bl_wavelet_regional.py`,
`directed_burst_coupling.py`, `burst_coincidence.py`, `coupling_bursts.py`),
and the three `--all`-capable runners
(`scripts/_run_scaffold_v11.py`, `scripts/_run_scaffold_v10.py`,
`scripts/_run_v11_hierarchical.py`).

Reference module: `cadence/io/resources.py` (`pick_n_jobs`,
`limit_blas_threads`, `gpu_chunk_size`, `snapshot`, `log_resources`).

## Summary

- Total findings: 14 (P0: 4 / P1: 6 / P2: 4)
- Two of the four P0 findings — `fast_cycles.extract_burst_grids` and
  `_run_scaffold_v11.py` `--all` — are direct repeats of the OOM mechanism
  that triggered this audit (joblib threads × unlimited BLAS, ringed by
  GPU work that re-imports torch into every worker).
- The Phase-1 P0 patches alone close the OOM-replay path; the P1 work then
  brings every adaptive-`n_jobs` site onto `pick_n_jobs(...)`.

## P0 — Active OOM risk

### P0.1 — `cadence/significance/fast_cycles.py:517`  (`Parallel(n_jobs=-1, prefer='threads')`)

**Pattern.** Inside `analyze_interbrain_cycles_multiband`, every band × channel ×
participant cycle-extraction job is fanned out as
`Parallel(n_jobs=-1, prefer='threads')` over CPU-bound numpy work
(`extract_cycle_features`, scipy `find_peaks`, repeated `np.percentile` /
`np.diff`). Each thread inherits the parent's BLAS thread pool with no cap,
which is the textbook reproduction of the OOM that triggered this audit.

**Concrete fix.**

Before:

```python
results_list = Parallel(n_jobs=-1, prefer='threads')(jobs)
```

After:

```python
from cadence.io.resources import pick_n_jobs, limit_blas_threads

# extract_cycle_features peaks at ~50–80 MB per worker on a 30-min EEG channel.
n_jobs = pick_n_jobs(per_worker_ram_gb=0.1, requested=-1)

def _bounded_extract(*args, **kwargs):
    with limit_blas_threads(1):
        return _extract_one(*args, **kwargs)

results_list = Parallel(n_jobs=n_jobs, prefer='threads')(
    delayed(_bounded_extract)(*j.args, **j.keywords) for j in jobs)
```

(The wrapper makes the BLAS guard mandatory; do not rely on individual
`extract_cycle_features` callsites to remember it.)

### P0.2 — `cadence/significance/fast_cycles.py:803, 947`  (`ThreadPoolExecutor(max_workers=8)`)

**Pattern.** Both `extract_all_volt_amp` and the much-hotter
`extract_burst_grids` (the V11 scaffold's single biggest CPU stage) use
`ThreadPoolExecutor(max_workers=8)` over numpy/scipy work with no BLAS
guard. `extract_burst_grids` is called inside `compute_burst_features`,
which is itself launched 4-wide via the V11 `--all` pool — so the realised
parallelism is `4 × 8 × ~16 BLAS = 512` threads. This is the OOM
configuration in compounded form.

**Concrete fix.** Replace both pools with `joblib.Parallel`
(repo convention per `feedback_joblib_loky`) under
`limit_blas_threads(1)`:

Before (line 980):

```python
from concurrent.futures import ThreadPoolExecutor as _TPE
extracted = {'p1': {}, 'p2': {}}
with _TPE(max_workers=8) as pool:
    futures = []
    for band_name, ch, eeg, filt, band, person in tasks:
        fut = pool.submit(_extract_one, band_name, ch, eeg, filt, band)
        futures.append((person, fut))
    for person, fut in futures:
        result = fut.result()
        ...
```

After:

```python
from joblib import Parallel, delayed
from cadence.io.resources import pick_n_jobs, limit_blas_threads

def _bounded_extract_one(band_name, ch, eeg, filt, band, person):
    with limit_blas_threads(1):
        return person, _extract_one(band_name, ch, eeg, filt, band)

n_jobs = pick_n_jobs(per_worker_ram_gb=0.15, requested=8)
out = Parallel(n_jobs=n_jobs, prefer='threads')(
    delayed(_bounded_extract_one)(band_name, ch, eeg, filt, band, person)
    for band_name, ch, eeg, filt, band, person in tasks)
extracted = {'p1': {}, 'p2': {}}
for person, result in out:
    if result is not None:
        bn, ch, va, burst = result
        extracted[person].setdefault(bn, {})[ch] = (va, burst)
```

Apply the same pattern to `extract_all_volt_amp:803`.

### P0.3 — `scripts/_run_scaffold_v11.py:888`  (`n_jobs=4`, no BLAS guard, GPU-heavy work)

**Pattern.**

```python
results = Parallel(n_jobs=4, backend='threading')(
    delayed(run_session)(name, config) for name in todo
)
```

`run_session` runs the full V11 pipeline per session. With the inner
ThreadPoolExecutor in `extract_burst_grids` still alive (P0.2), the
realised footprint per outer worker is ~3–4 GB (cached EEG) +
ThreadPoolExecutor BLAS spawn + GPU resident state. Four concurrent
`run_session` workers competing for the same 16 GB GPU regularly exceed
both the user's 60 GB RAM cap and 16 GB VRAM cap. The `--all` runner
already hit OOM in this configuration.

**Concrete fix.** Adaptive jobs + per-worker BLAS guard. Even with
threading, the explicit `pick_n_jobs` keeps the caller honest about the
per-worker peak and lets the resource module shrink the pool when the
user has IDE/browser open.

Before:

```python
print(f"Running V11 scaffold on {len(todo)} remaining sessions "
      f"(n_jobs=4, threading)...")
results = Parallel(n_jobs=4, backend='threading')(
    delayed(run_session)(name, config) for name in todo
)
```

After:

```python
from cadence.io.resources import pick_n_jobs, limit_blas_threads, log_resources

# run_session peak RAM ≈ 4 GB (raw EEG + cwt + burst grids in-flight).
# Threading backend is mandatory on this stack (torch/numpy DLL bug).
n_jobs = pick_n_jobs(per_worker_ram_gb=4.0, requested=4)
log_resources(prefix='[v11 --all] pre-fan-out: ')
print(f"Running V11 scaffold on {len(todo)} sessions "
      f"(n_jobs={n_jobs}, threading; per-worker RAM ~= 4 GB)")

def _bounded_run_session(name, config):
    with limit_blas_threads(1):
        return run_session(name, config)

results = Parallel(n_jobs=n_jobs, backend='threading')(
    delayed(_bounded_run_session)(name, config) for name in todo
)
```

The companion fix for `_run_scaffold_v10.py:638` is in P1.1 — V10 doesn't
issue GPU surrogates per session, so its parallelism is less acute, but
the same `n_jobs=8` hardcode should adopt the same pattern.

### P0.4 — `cadence/significance/rslds_model.py:1714, 1801`  (`Parallel(n_jobs=-1, prefer='processes')` then `prefer='threads'`)

**Pattern.** Hierarchical-rSLDS init uses `n_jobs=-1, prefer='processes'`
(line 1714), which copies the entire `sessions` list (each session is
many MB of Y, U, mask) into every forked worker. On a 16-CPU host with
N=12 sessions and ~50 MB/session, that's ~10 GB of redundant copies
even before the worker does any work. The E-step loop (line 1801) is
already thread-pinned with the manual `threadpool_limits` guard
(line 1795), but the parent process forks the BLAS pool unbounded
between calls.

**Concrete fix.** Cap `n_jobs` by per-worker RAM in both calls; switch
the init path to threading so workers share the session arrays in-place;
adopt `limit_blas_threads(1)` in `_init_one`.

Before (1714):

```python
init_results = Parallel(n_jobs=-1, prefer='processes')(
    delayed(_init_one)(a) for a in init_args)
```

After:

```python
from cadence.io.resources import pick_n_jobs, limit_blas_threads

# Each SLDS init carries one session's Y/U/mask + an SLDS workspace.
# Empirically ~1 GB peak per session at D_obs=26.
n_jobs = pick_n_jobs(per_worker_ram_gb=1.0, requested=-1)

def _bounded_init_one(args):
    with limit_blas_threads(1):
        return _init_one(args)

init_results = Parallel(n_jobs=n_jobs, prefer='threads')(
    delayed(_bounded_init_one)(a) for a in init_args)
```

The duplicated guard at 1795 should then be replaced by
`limit_blas_threads(1)` for consistency, but it does not need to be a
new pattern — the manual `threadpoolctl.threadpool_limits` is the same
thing the new helper wraps.

## P1 — Should adopt resource module

### P1.1 — `scripts/_run_scaffold_v10.py:638`  (`n_jobs=8`)

`Parallel(n_jobs=8)` (default backend = loky / processes on Windows when
threading is not requested). Each forked process re-loads the full V10
pipeline imports (torch, sklearn, etc.) plus its own session cache.
At 8-wide that's ~8 × ~3 GB and the same DLL-load order issue that
forced V11 to threading. Should mirror the V11 pattern (P0.3) plus the
`hoist torch` shim — V10 doesn't currently `import torch` at the top of
the module, so loky workers will reproduce the shm.dll bug under a
torch 2.10 reinstall (out of scope for this audit but worth flagging).

Concrete fix is structurally identical to P0.3 except `per_worker_ram_gb=2.0`
(no GPU TE surrogates).

### P1.2 — `cadence/significance/rslds_validation.py:407, 559`  (`Parallel(n_jobs=4|-1, prefer='processes')`)

Two synthetic-recovery test loops fan out SLDS fits across processes.
At `n_jobs=-1` on a 16-core box, six concurrent SLDS fits with
`max_em_iter=80` each carry their own BLAS pool. These are
test-only paths (not on the production hot path), but they fail the
same way. Same fix as P0.4: thread + `limit_blas_threads(1)` +
`pick_n_jobs(per_worker_ram_gb=1.0)`.

### P1.3 — `cadence/surrogates.py:235`  (`Parallel(n_jobs=-1)`)

IAAFT surrogate generation parallelizes per-surrogate over CPU FFTs.
Default backend (`loky`) forks the full input array into every worker
(`(N, C)` data, K copies). For a 30-min session at 30 Hz with 200
surrogates and 16 cores, peak is ~16 × 5–10 MB; not catastrophic, but a
trivial one-line fix:

Before:

```python
results = Parallel(n_jobs=-1)(
    delayed(iaaft_surrogate)(...))
```

After:

```python
from cadence.io.resources import pick_n_jobs, limit_blas_threads
n_jobs = pick_n_jobs(per_worker_ram_gb=0.05, requested=-1)
def _bounded(*a, **kw):
    with limit_blas_threads(1):
        return iaaft_surrogate(*a, **kw)
results = Parallel(n_jobs=n_jobs)(delayed(_bounded)(...))
```

### P1.4 — `cadence/significance/kim_filter.py:401`  (`Parallel(n_jobs=n_jobs)` from caller default `n_jobs=-1`)

Per-channel Kim-filter loop parallelizes EM (forward-backward + numerical
inversions). Default `n_jobs=-1` from the wrapper signature on
line 384. No BLAS guard — and Kalman/EM is heavy in `np.linalg.solve`
calls, which is exactly the kind of work BLAS oversubscription destroys.
Same fix as P0.1, with `per_worker_ram_gb=0.2` (basis matrices per
channel).

### P1.5 — `cadence/coupling/estimator.py:2010, 3887`  (`ThreadPoolExecutor`)

Two pathway-discovery dispatches use `ThreadPoolExecutor(max_workers=...)`
in lieu of joblib. These are deliberate — each thread grabs its own
`torch.cuda.Stream` to overlap GPU kernels (line 2016) — so the
threading model is correct and the workers are GPU-bound, not BLAS-bound.
Two minor adoptions are still warranted:

1. `max_workers` is taken from a free function (`n_workers`) without an
   explicit cap; should be clamped via
   `gpu_chunk_size(per_unit_vram_gb=<peak_per_pathway>, n_units=n_pathways)`
   so the chosen worker count cannot exceed VRAM budget.
2. There is no `torch.cuda.empty_cache()` between successive
   `_run_pathway` invocations after the loop completes (one is at line
   2052, but the recursive structure in stage 2 — line 3928 — does not
   match this pattern).

This is below P0 only because GPU stream contention bounds the realised
parallelism in practice; under heavier coupling configs the user would
hit a CUDA OOM long before RAM is exhausted.

### P1.6 — `cadence/ingest/digest.py:265`  (`backend='threading'`, no BLAS guard)

`digest_all` already uses threading and explicitly documents the
per-worker memory in `cli.py:54` ("4 ~= 6 GB peak"). It does NOT use
the resource module to **enforce** that — a user passing `--n-jobs 16`
on a 60 GB box would still get 96 GB peak. The fix is two lines:

Before (cli.py:74, 79):

```python
results = digest_all(raw_dir, out_dir, force=args.force, only_canonical=True,
                     n_jobs=args.n_jobs)
```

After:

```python
from cadence.io.resources import pick_n_jobs, log_resources
log_resources(prefix='[ingest --all] ')
# pyxdf.load_xdf holds one full XDF in memory; ~1.5 GB/worker is realistic.
safe_jobs = pick_n_jobs(per_worker_ram_gb=1.5, requested=args.n_jobs)
if safe_jobs != args.n_jobs:
    print(f"[ingest] resource cap: {args.n_jobs} -> {safe_jobs} workers")
results = digest_all(raw_dir, out_dir, force=args.force, only_canonical=True,
                     n_jobs=safe_jobs)
```

(The threading backend already protects against BLAS spawn proliferation
at the digest level — XDF parsing is not BLAS-bound — so a per-worker
`limit_blas_threads(1)` is not required here.)

## P2 — Minor

### P2.1 — `cadence/significance/fast_cycles.py:484`  has `torch.cuda.empty_cache()` once but not after each band

The multi-band CWT pipeline only releases VRAM after all bands finish
the bandpass step (line 484). Surrogates for theta/alpha/beta are then
generated sequentially on GPU at lines 601–658, accumulating allocations
that compete with `extract_burst_grids`'s next call. Add an
`empty_cache()` at the bottom of each band's loop body. Cost: zero on
modern PyTorch (lazy free); benefit: bounded VRAM peak.

### P2.2 — `cadence/significance/bl_wavelet.py` — full-session CWT on GPU, no chunking

`_cwt_gpu` (line 127) builds a single `(n_freqs=30, T, n_ch)` complex64
tensor. For a 30-min session at 30 Hz with 52 AUs that's
30 × 54000 × 52 × 8 B ≈ 670 MB — fine in isolation, but a single GPU
worker doing wavelet coherence + concurrent EEG burst extraction +
surrogate accumulation can push past 8 GB before the OS reports it. Add
the same chunked pattern that `wavelet_features._cwt_chunked` uses
(`_CHUNK_SECONDS = 60`, `_OVERLAP_SECONDS = 5` — already proven for
EEG). Use `gpu_chunk_size(per_unit_vram_gb=<size_per_chunk_gb>, n_units=...)`
to keep the chunk size adaptive.

### P2.3 — `cadence/preprocess/face/pipeline.py:94`  full-session SVD

`np.linalg.svd(centered, full_matrices=False)` is run on the full
`(N_valid_frames, 52)` matrix. For a 60-min session at 30 Hz this is
~108k × 52 — small enough that an explicit cap is not needed, but the
SVD calls inside `joblib.Parallel(prefer='threads')` would compete for
BLAS threads if face preprocessing ever gets parallelized over sessions.
Defensive adoption: when face preprocessing is wrapped in a future
`--all` runner, use `limit_blas_threads(1)` inside the worker.

### P2.4 — `cadence/preprocess/eeg/pipeline.py` — no `--all` parallelism, but unbounded per-process

The current `make_modality_cli` runs preprocess sessions in a sequential
`for sid in session_ids` loop (line 55 of `_cli.py`) with no
parallelism. That's safe under the new resource discipline. If/when the
user wants `--n-jobs N` across sessions, the pattern from P1.6 should
be adopted — and the per-worker memory will need an empirical estimate
(EEG MATLAB load + extract_eeg_features + extract_wavelet_features is
~3-4 GB realistically).

## Recommended sequence of changes

1. **`cadence/significance/fast_cycles.py`** — fix P0.1 + P0.2 first.
   These two functions are called from every V10/V11 path and from the
   semi-synthetic battery; closing the ThreadPoolExecutor + BLAS hole
   in `extract_burst_grids` alone reduces realised thread count from
   `~4 × 8 × 16 = 512` to `~4 × 4 × 1 = 16` on the worst V11 `--all`
   path. **Do this before re-running V11 `--all`**.
2. **`scripts/_run_scaffold_v11.py:888`** — fix P0.3. Adopt
   `pick_n_jobs(per_worker_ram_gb=4.0, ...)` + `limit_blas_threads(1)`.
   This is the call-site that reproduced the user's OOM and is the
   first one a user will re-trigger.
3. **`scripts/_run_scaffold_v10.py:638`** — fix P1.1 (same pattern).
   Even though V10 hasn't OOMed yet, the same fix is trivial and
   prevents regressions when running the V10 audit battery.
4. **`cadence/significance/rslds_model.py:1714`** — fix P0.4 init
   forking. The hierarchical fit is a frequent user touch point and
   process forking with `prefer='processes'` is the worst case for
   memory under pinned BLAS.
5. **`cadence/ingest/digest.py` (via `cli.py`)** — fix P1.6. One-line
   `pick_n_jobs` wrap.
6. **`cadence/surrogates.py`** + **`cadence/significance/kim_filter.py`**
   + **`cadence/significance/rslds_validation.py`** — sweep the
   remaining `Parallel(n_jobs=-1)` sites in one PR (P1.2-1.4); these
   are off the V11 hot path but easy to adopt.
7. **GPU chunking polish** — fix P2.1 (per-band `empty_cache`) and
   P2.2 (chunked CWT in `bl_wavelet.py`). These are below the OOM
   threshold but matter for a future longer-session batch (V12 gaze
   sessions, larger AU panels).
8. **Defensive adoption** for `cadence/preprocess/` — leave as
   sequential, document in each pipeline's docstring that any future
   `--all` parallelism MUST go through `pick_n_jobs` +
   `limit_blas_threads`.

After steps 1–4 the audited code base is OOM-safe under the documented
60 GB / 16 GB caps; steps 5–8 are cleanups that bring everything onto
the same adaptive-resources path.
