# Resource-discipline P1 + P2 patches (2026-05-01)

Companion to `docs/resource_audit_2026_05_01.md`. The P0 patches were
applied by the main session; this document records the P1 and P2
follow-ups completed in the subagent pass.

## Summary

| Finding | File | Pattern applied | Notes |
| --- | --- | --- | --- |
| P1.2 | `cadence/significance/rslds_validation.py` (lines ~407, ~559) | `pick_n_jobs(per_worker_ram_gb=1.0) + limit_blas_threads(1)` + threading backend | Two SLDS-recovery test loops; switched from `prefer='processes'` to `prefer='threads'`. |
| P1.3 | `cadence/surrogates.py` (~235) | `pick_n_jobs(per_worker_ram_gb=0.05) + limit_blas_threads(1)` | IAAFT surrogate generation; default backend retained for FFT throughput. |
| P1.4 | `cadence/significance/kim_filter.py` (~401, default at ~384) | `pick_n_jobs(per_worker_ram_gb=0.2) + limit_blas_threads(1)` + threading | Independent per-channel Kim-filter EM loop; module-level import added. |
| P1.5 | `cadence/coupling/estimator.py` (~2010, ~3887) | `gpu_chunk_size(per_unit_vram_gb=2.0/3.0) cap on max_workers` + retained `torch.cuda.empty_cache()` | ThreadPoolExecutor kept (CUDA-stream aware); only `max_workers` is now VRAM-aware. |
| P1.6 | `cadence/ingest/cli.py` (was line 73-79) | `pick_n_jobs(per_worker_ram_gb=1.5)` + `log_resources` wrap of `digest_all` | Combined `--all` and `--every-xdf` branches into one resource-capped block. |
| P2.1 | `cadence/significance/fast_cycles.py` (4 sites: bandpass loops in `extract_all_volt_amp` ~823, `extract_burst_grids` ~986, plus per-band surrogate loops in `analyze_interbrain_cycles_multiband` ~576 and `eeg_coupling_from_precomputed` ~891 and `eeg_coupling_timecourse` ~1090) | `if device.type == 'cuda': torch.cuda.empty_cache()` at end of each band loop body + `del intermediate; empty_cache` after bandpass build | Fully additive; bounds peak VRAM across theta/alpha/beta accumulation. |
| P2.2 | `cadence/significance/bl_wavelet.py` (`_cwt_gpu`, ~127) | Frequency-axis chunking via `gpu_chunk_size(per_unit_vram_gb=2*T*n_ch*16/1e9)` + per-chunk `torch.cuda.empty_cache()` | Replaces the single `(n_freqs, T, n_ch)` complex128 tensor with adaptive freq-chunk iteration; output shape unchanged. |
| P2.3 | `cadence/preprocess/face/pipeline.py` (~94) | `with limit_blas_threads(1): np.linalg.svd(...)` | Defensive guard; sequential today, but matters under any future `--all` parallelism. |
| P2.4 | `cadence/preprocess/eeg/pipeline.py` (module docstring) | Documentation note — no code change | Records the empirical 3.5 GB/worker estimate and the required `pick_n_jobs` + `limit_blas_threads(1)` pattern for future parallelism. |

## Verification

```text
$ python -c "from cadence.significance import rslds_validation, kim_filter, \\
                       fast_cycles, bl_wavelet
                from cadence.coupling import estimator
                from cadence.ingest import cli
                from cadence import surrogates
                from cadence.preprocess.face import pipeline as face_pipeline
                from cadence.preprocess.eeg import pipeline as eeg_pipeline
                print('All P1+P2 patched modules import cleanly')"
All P1+P2 patched modules import cleanly
```

Structural grep for the resource-module API across the patched set:

```text
cadence/coupling/estimator.py:                6
cadence/preprocess/face/pipeline.py:          2
cadence/surrogates.py:                        3
cadence/significance/fast_cycles.py:          7
cadence/significance/bl_wavelet.py:           3
cadence/significance/kim_filter.py:           3
cadence/significance/rslds_validation.py:     5
cadence/ingest/cli.py:                        2
                                              ===
                                              31
```

## Notes / assumptions

* P1.5 (estimator.py): the per-pathway VRAM peak was estimated at 2.0 GB
  (Stage 1) and 3.0 GB (Stage 2). The Stage 2 site already had a
  `mem_get_info` heuristic; the new `gpu_chunk_size` cap is applied
  *first*, then the existing heuristic acts as a secondary clamp.
* P1.6 (ingest cli): the threading backend in `digest_all` already
  prevents BLAS spawn proliferation, so a per-worker
  `limit_blas_threads(1)` was deemed redundant at the digest level
  (XDF parsing is not BLAS-bound). The cap enforces only the RAM
  ceiling (`pick_n_jobs(per_worker_ram_gb=1.5)`).
* P2.2 (bl_wavelet chunking): kept under 50 lines of changes. Output
  semantics are identical (same complex64 array of shape
  `(n_freqs, T, n_ch)`), only the construction is now chunked along the
  frequency axis. The chunk size is adaptive via `gpu_chunk_size`.
