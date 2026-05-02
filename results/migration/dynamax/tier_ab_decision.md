# Tier A+B Decision Gate — 2026-05-02

**Profile:** 3 sessions (y_06, y_17, y24_022526), 10 EM iters (early-stop at iter 3), K=4, D_obs=7, D_input=2, D_latent=3, n_factors=2, recurrent=True

**Wall-clock:** 178.16s

**Gate measurement (sequential pooled M-step):**
- `_hierarchical_m_step_transitions`: cumtime=0.396s, ncalls=3
- `_hierarchical_m_step_dynamics`: cumtime=0.016s, ncalls=3
- **Total sequential M-step: 0.412s = 0.23% of wall-clock**

Note: ncalls=3 reflects early convergence (EM stopped at iteration 3 via em_tol). The
per-call cumtime (0.132s/call for transitions, 0.005s/call for dynamics) is the
load-bearing quantity; at max_em_iter=10 these would sum to ≤1.4s / ≤0.8% of
wall-clock at the same convergence rate.

**Time breakdown (approximate, from cumtime ÷ thread-count):**
- Phase 1 init (parallel, 3 threads): dominated by time.sleep=177.3s / threading idle
- Phase 4 E-step (parallel, 3 threads): also folded into the threading wait pool
- Phase 4 M-step sequential (measured above): 0.412s = 0.23% (PRECISE)
- Other / orchestration: ~0.7s (numpy ops, emissions M-step, joblib dispatch)

NOTE: With the threading backend, cProfile captures all wall-time in `time.sleep`
(joblib's polling loop, 177.3s) — the main thread waits for worker threads while they
run the heavy Kalman smoother and per-session SLDS init. The sequential M-step
functions execute on the main thread and are captured exactly. All parallel work
runs in worker threads and is only partially captured by cProfile under threading.

**Top-5 cumulative-time functions:**
```
   ncalls  tottime  percall  cumtime  percall  function
        1    0.001    0.001  178.136  178.136  rslds_model:1892(fit_hierarchical_slds)
        5    0.000    0.000  177.443   35.489  joblib/parallel.py:1969(__call__)
       25    0.000    0.000  177.443    7.098  joblib/parallel.py:1670(_get_outputs)
        5    0.071    0.014  177.430   35.486  joblib/parallel.py:1776(_retrieve)
    10950  177.336    0.016  177.336    0.016  {built-in method time.sleep}
```

**Top-5 tottime functions:**
```
   ncalls  tottime  percall  cumtime  percall  function
    10950  177.336    0.016  177.336    0.016  {built-in method time.sleep}
      220    0.182    0.001    0.182    0.001  {method 'acquire' of '_thread.lock' objects}
     1551    0.157    0.000    0.157    0.000  {built-in method numpy._core._multiarray_umath.c_einsum}
     1870    0.097    0.000    0.097    0.000  {method 'reduce' of 'numpy.ufunc' objects}
       66    0.085    0.001    0.390    0.006  rslds_model:1835(_objective)
```

**Gate-target functions (verbatim from profile):**
```
   ncalls  tottime  percall  cumtime  percall  function
        3    0.013    0.004    0.016    0.005  rslds_model:1759(_hierarchical_m_step_dynamics)
        3    0.000    0.000    0.396    0.132  rslds_model:1817(_hierarchical_m_step_transitions)
```

**Decision rule:**
- mstep_seq_pct ≥ 30% → Tier A + Tier B
- 15% ≤ mstep_seq_pct < 30% → Tier A only (Tier B deferred)
- mstep_seq_pct < 15% → Tier A only

**Decision: Tier A only**

**Rationale:** The sequential hierarchical M-step (`_hierarchical_m_step_transitions` +
`_hierarchical_m_step_dynamics`) consumes a measured **0.23% of wall-clock** (0.412s /
178.16s), far below both the 15% deferred threshold and the 30% full-proceed threshold.
The true bottleneck is the parallel E-step (Kalman smoother + per-session SLDS init),
which runs in worker threads and consumes nearly all wall-time. Tier B's numba LBFGS
GIL-release is not cost-justified at current cohort size: addressing 0.23% of wall-time
would yield negligible end-to-end speedup.

**Note for ~2027-2028 horizon (per project_jax_post_phase0_decision):**
At N=450 the sequential M-step grows linearly with N (single softmax-regression
over pooled suff-stats from all sessions), while parallel E-step is bounded by
core count. This pushes M-step% upward by ~24× — a re-profile at that scale
will likely cross the 30% threshold and re-enable Tier B.

## Session Substitution

`y_24` (spec) → `y24_022526` (actual directory): the date-suffixed scaffold exists at
`results/mvp/y24_022526/mvp_scaffold.npz`. The script handles this automatically.
All 3 sessions loaded successfully (y_06: 6188 timesteps, y_17: 5361 timesteps,
y24_022526: 2968 timesteps).

## Profile Interpretation Notes

- cProfile with `prefer='threads'` captures only main-thread execution in parallel
  sections; worker-thread time is not attributed to individual functions. The sequential
  M-step runs on the main thread and is captured exactly.
- `time.sleep` cumtime=177.3s is joblib's polling loop waiting for worker threads.
  It does not reflect actual idle time on the wall clock.
- The previous run (267.09s) used a different threading configuration that showed
  cumtime inflation across all threads; the current 178.16s run uses the threading
  backend cleanly with main-thread-only cProfile capture.
- Previous document stated "~7% / ~20s" for the M-step — this was an extrapolation
  from the per-session `_m_step_transitions` timing (0.586s/call × 3 sessions × ~10
  iters). The measured values are 0.132s/call × 3 calls = 0.396s total, confirming
  the hierarchical M-step is far cheaper than extrapolated (the pooling over sessions
  is done in the objective function's numpy ops, not in repeated single-session calls).
