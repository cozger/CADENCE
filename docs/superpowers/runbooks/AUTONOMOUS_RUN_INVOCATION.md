# Autonomous run — invocation instructions

To kick off the full MVP pipeline autonomously, paste this into a fresh
Claude Code session at the project root (`C:\Users\optilab\desktop\CADENCE\`):

---

## Short invocation (recommended)

```
You are running the full CADENCE MVP rSLDS pipeline autonomously.

Read the runbook at:
docs/superpowers/runbooks/2026-05-01-mvp-full-pipeline-autonomous-run.md

Execute every stage in order (0 -> 8). Use TaskCreate to track each stage.
At the end, write results/mvp/AUTONOMOUS_RUN_REPORT.md and finish with the
status line specified in the runbook.

Operating constraints (non-negotiable):
- ENVIRONMENT: run every Python command under the MCCT conda env (CUDA torch
  2.10.0+cu128 + Python 3.11). Two acceptable forms:
    1. PowerShell session activated with `conda activate MCCT` once at the
       start, then `$env:KMP_DUPLICATE_LIB_OK = "TRUE"`.
    2. Per-call `KMP_DUPLICATE_LIB_OK=TRUE conda run -n MCCT python ...`
       (works with the Bash tool; conda run does NOT accept multi-line
       `python -c` snippets — write a temp script if you need one).
  Do NOT use the base env — it has CPU-only torch and the wrong Python.
- 60 GB RAM cap, 16 GB VRAM cap. Use cadence/io/resources.py everywhere.
- Run preprocessing modalities sequentially (NOT parallel across modalities).
- Skip sessions with missing MATLAB EEG cleaning; do NOT block on them.
- If you hit OOM (peak system RAM > 60 GB), stop and report — do not retry.
- No interactive prompts; every command must be self-contained.

Total expected wall-clock: 5-8 hours. Run in auto mode.
```

---

## Long invocation (if the runbook needs supplementing)

If the runbook reference doesn't load reliably for the agent, paste the full
contents of the runbook directly into the prompt (it's ~250 lines).

---

## Pre-flight checks before invoking

All checks must be run **inside the MCCT env** with the OMP override set.
Run from a Bash-tool prompt (POSIX env-var prefix); for PowerShell, set
`$env:KMP_DUPLICATE_LIB_OK = "TRUE"` first, then drop the `KMP_...=TRUE`
prefix from each command.

Verify these are already in place — if not, the run will fail early:

1. `KMP_DUPLICATE_LIB_OK=TRUE conda run -n MCCT pip show dtaidistance`
   returns version 2.4.0. (If missing:
   `conda run -n MCCT pip install -r cadence/requirements_mvp.txt`.)
2. `KMP_DUPLICATE_LIB_OK=TRUE conda run -n MCCT pip show threadpoolctl`
   returns 3.x.
3. `KMP_DUPLICATE_LIB_OK=TRUE conda run -n MCCT python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"`
   prints `True 12.8 NVIDIA GeForce RTX 5080` (or your local CUDA-capable
   device). Failure here means the wrong env is active — do NOT proceed
   with the CPU-only torch in `base`; the runbook's GPU stages were sized
   assuming CUDA.
4. `KMP_DUPLICATE_LIB_OK=TRUE conda run -n MCCT python -c "from cadence.io.resources import pick_n_jobs, limit_blas_threads, snapshot; print(snapshot())"`
   returns a sensible MemorySnapshot. If you get `Windows fatal exception:
   code 0xc06d007f` instead, the torch DLL hoist (see
   `project_win_torch_dll_fix.md`) is broken — investigate before running.
5. At least 50 GB of RAM and 13 GB of VRAM are free at run start
   (close other heavy applications first).

---

## Mid-run monitoring

The run produces these progress markers you can `tail` from another terminal:

- `results/mvp/pipeline_inventory.csv` (after Stage 0)
- `data/preproc/{eeg,face,ecg,pose}/v1/*.json` (per session, as Stage 2 progresses)
- `results/v11/<sid>/scaffold_v11_results.json` (per session, as Stage 3 progresses)
- `results/mvp/phase0/phase0_report.md` (after Stage 4)
- `results/mvp/cohort_protocol_assignment.csv` (after Stage 5)
- `results/mvp/hierarchical*/mvp_hierarchical_results.json` (per fit, Stage 6)
- `results/mvp/diagnostics/verification_report.md` (after Stage 7)
- `results/mvp/figures/figure[12]_*.png` (after Stage 8)
- `results/mvp/AUTONOMOUS_RUN_REPORT.md` (final)

---

## Resuming a partial run

If the run is interrupted mid-stage, simply re-invoke with the same prompt. All
stages are designed to be resumable:

- Ingestion / preprocessing CLIs are idempotent.
- V11 scaffold regen skips sessions with up-to-date `digest_xdf_md5`.
- MVP scaffold slice is cheap; just re-runs.
- Hierarchical fits write per-session NPZs incrementally; if interrupted,
  re-run the specific fit variant whose suffix's output dir is missing.

The agent should detect existing outputs and skip ahead.
