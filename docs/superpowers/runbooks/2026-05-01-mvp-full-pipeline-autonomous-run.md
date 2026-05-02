# Autonomous Full-Pipeline Run — MVP rSLDS Grant Figures

**Audience:** A Claude Code agent (subagent or fresh session) tasked with
running the full MVP pipeline end-to-end without further human input.

**Goal:** Take the project from current state (digests + partial preproc +
some V11 scaffolds) all the way through Figure 1 + Figure 2, with verification
results, in a single autonomous run. Should tolerate partial-data sessions,
skip cleanly when blocked, and finish with a complete status report.

**Prerequisites already in place:**
- **Conda environment: `MCCT`** (Python 3.11, torch 2.10.0+cu128, CUDA 12.8,
  RTX-class GPU). Activate before any command:
  `conda activate MCCT` (PowerShell) and set
  `$env:KMP_DUPLICATE_LIB_OK = "TRUE"` once per shell.
  *Do not use `base` — it has CPU-only torch.*
- `cadence/io/resources.py` — adaptive parallelism with `pick_n_jobs()`,
  `limit_blas_threads(1)`, hard caps at **60 GB RAM / 16 GB VRAM**.
- All P0 sites in `fast_cycles.py`, `_run_scaffold_v11.py`,
  `_run_scaffold_v10.py`, and `rslds_model.py` are resource-disciplined.
- `cadence/significance/pose_ddtw.py` + `scripts/_validate_pose_ddtw.py` exist
  (Phase 0 DDTW pipeline).
- `scripts/_run_mvp_scaffold.py` + `scripts/_run_mvp_hierarchical.py` exist.
- `dtaidistance==2.4.0` is installed in MCCT (see
  `cadence/requirements_mvp.txt`; install with
  `conda run -n MCCT pip install -r cadence/requirements_mvp.txt`).
- `KMP_DUPLICATE_LIB_OK=TRUE` is set in the active shell — without it the
  MCCT env crashes with `OMP: Error #15` on duplicate `libiomp5md.dll`.
- Canonical session registry: `configs/session_quality.yaml`
  (`canonical: true` flag, ~22 sessions).

---

## Operating principles

1. **Resource discipline is non-negotiable.** Every script you launch must use
   the patched code. If you encounter an unpatched `Parallel(n_jobs=-1)` site,
   stop and patch it before running. The 60 GB / 16 GB caps are hard.
2. **Sessions that cannot be processed are skipped, not blockers.** Four
   sessions (`y04_020626`, `y11_022526`, `y24_022526`, `y_53_04302026`) need
   fresh MATLAB EEG cleaning the user must do externally. Note them in the
   final report and proceed without them.
3. **Idempotent steps.** All preproc CLIs honor staleness — re-running is
   safe. V11 scaffold regen IS expensive; only re-run when staleness check
   demands.
4. **Log resources at every stage entry** (`log_resources(prefix=...)` in any
   inline Python). Halt and warn the user if RAM available drops below 10 GB.
5. **Use TaskCreate** to track each stage. Mark completed as you go. Final
   report references task IDs.
6. **NO interactive prompts.** All commands must run with explicit args.

---

## Stage 0: Inventory + freshness check (~2 min)

Run in Python:

```python
import json
from pathlib import Path
from cadence.ingest.quality import list_canonical_sessions
from cadence.io.resources import log_resources

REPO = Path('C:/Users/optilab/desktop/CADENCE')
log_resources(prefix='[Stage 0] start: ')

canonical = list_canonical_sessions()

state = []
for sid in canonical:
    digest_path = REPO / 'data' / 'digest' / 'v1' / f'{sid}.json'
    has_digest = digest_path.exists()
    if has_digest:
        digest_md5 = json.loads(digest_path.read_text()).get('xdf_md5')
        protocol = json.loads(digest_path.read_text()).get('protocol', '')
    else:
        digest_md5, protocol = None, ''

    pre = {m: (REPO / 'data' / 'preproc' / m / 'v1' / f'{sid}.npz').exists()
           for m in ('eeg', 'face', 'ecg', 'pose')}
    has_v11 = (REPO / 'results' / 'v11' / sid / 'scaffold_v11_ztimecourses.npz').exists()
    has_matlab = (
        (REPO / 'data' / 'matlab' / f'{sid}_p1_clean.mat').exists() and
        (REPO / 'data' / 'matlab' / f'{sid}_p2_clean.mat').exists()
    )
    state.append({
        'sid': sid, 'protocol': protocol, 'has_digest': has_digest,
        'digest_md5': digest_md5,
        **{f'preproc_{m}': pre[m] for m in pre},
        'has_v11_scaffold': has_v11, 'has_matlab_clean': has_matlab,
    })

# Print table
print(f'{"sid":25s} | {"prot":10s} | dig | mat | eeg | fac | ecg | pos | v11 |')
for s in state:
    print(f'{s["sid"]:25s} | {s["protocol"]:10s} | '
          f'{"Y" if s["has_digest"] else "n":3s} | '
          f'{"Y" if s["has_matlab_clean"] else "n":3s} | '
          f'{"Y" if s["preproc_eeg"] else "n":3s} | '
          f'{"Y" if s["preproc_face"] else "n":3s} | '
          f'{"Y" if s["preproc_ecg"] else "n":3s} | '
          f'{"Y" if s["preproc_pose"] else "n":3s} | '
          f'{"Y" if s["has_v11_scaffold"] else "n":3s} |')

# Save inventory for later stages
import csv
with open('results/mvp/pipeline_inventory.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(state[0].keys()))
    w.writeheader()
    w.writerows(state)
print(f'\nInventory saved: results/mvp/pipeline_inventory.csv')
print(f'  Canonical: {len(state)}, with_digest: {sum(s["has_digest"] for s in state)}, '
      f'with_matlab: {sum(s["has_matlab_clean"] for s in state)}')
```

**Expected:** all 22 canonical sessions have digests; ~18 have MATLAB clean.mats.

---

## Stage 1: Backfill ingestion if needed (~5–10 min if any missing)

If Stage 0 showed any session WITHOUT `has_digest=True`, run:

```bash
python -m cadence.ingest --all
```

The CLI is idempotent (skips fresh sessions). Threading backend already in
place (`cadence/ingest/cli.py` is on the audit P1.6 list — apply the
`pick_n_jobs(per_worker_ram_gb=1.5)` patch first if not yet done).

Re-run Stage 0 inventory verification at the end.

---

## Stage 2: Preprocessing — all 4 modalities (~30–60 min total)

Run **sequentially** (NOT parallel across modalities — each CLI is itself
parallelizable; running them in series keeps RAM bounded):

```bash
python -m cadence.preprocess.pose --all
python -m cadence.preprocess.face --all
python -m cadence.preprocess.ecg  --all
python -m cadence.preprocess.eeg  --all
```

The EEG step will FAIL for sessions without fresh MATLAB clean.mat (the four
sessions named above). That's expected — capture the error message, log the
list of failed sessions, and proceed. Other 18 sessions should preproc cleanly.

After completion, re-run Stage 0 inventory and verify:
- All 22 sessions have face/pose/ecg preproc.
- 18 sessions have eeg preproc; 4 are flagged for the user.

---

## Stage 3: V11 scaffold regeneration (~30–90 min)

Per `docs/data_pipeline_v1.md`: V11 scaffolds built before 2026-05-01 lack the
`digest_xdf_md5` provenance field, so the MVP staleness fallback uses the
preproc-sidecars-content-hash check. To get fresh provenance and ensure all 22
canonical sessions are covered:

```bash
python scripts/_run_scaffold_v11.py --all
```

The patched runner uses `pick_n_jobs(per_worker_ram_gb=4.0)` so it auto-caps
its 4-wide threading pool to whatever fits the budget. Expect ~15–30 min per
session × ~4 wide = 1–2 hours total.

Sessions missing EEG preproc will fail this stage; log and skip.

After completion: ~18 sessions should have fresh V11 scaffolds (matching
digest_xdf_md5 in their sidecar JSON).

---

## Stage 4: Phase 0 — DDTW pose validation (~10–20 min)

Now that pose preproc exists for the full canonical cohort:

```bash
python scripts/_validate_pose_ddtw.py --all
```

Output: `results/mvp/phase0/phase0_report.md` with a decision line.

The decision is binary: `pose_ddtw` (if Tests 1, 2, 3 all pass) or
`pose_baseline`. NOTE: per smoke-test findings (2026-05-01), pose-coupling
peaks during STILLNESS not interaction — Test 2 in the validator was already
implemented as a magnitude test (|Δz| ≥ 0.5, significant in either direction).
This is NOT a bug; pose channel decision should still be DDTW if the magnitude
+ semi-synthetic + pseudo-null tests all pass.

---

## Stage 5: MVP scaffold slice (~5 min for 18 sessions)

Read Phase 0 decision automatically:

```bash
python scripts/_run_mvp_scaffold.py --all --pose-channel auto
```

Output:
- `results/mvp/<sid>/mvp_scaffold.{npz,json}` per session
- `results/mvp/cohort_protocol_assignment.csv` — cohort-level table

Check the cohort table: every session in the production fit should have
`included_in_production_fit=True`.

---

## Stage 6: MVP hierarchical fits (~3–5 hours total)

Run sequentially (each fit holds the full session set in memory; running them
in parallel would exceed the 60 GB cap):

```bash
# Production K=4 fit
python scripts/_run_mvp_hierarchical.py --K 4 --suffix ""

# K=3 sensitivity fit (per spec V2)
python scripts/_run_mvp_hierarchical.py --K 3 --suffix _k3

# Protocol-stratified fits (per spec V3)
python scripts/_run_mvp_hierarchical.py --K 4 --protocol meditation --suffix _med
python scripts/_run_mvp_hierarchical.py --K 4 --protocol pe --suffix _pe

# Unconstrained-Viterbi sensitivity (per spec V1)
python scripts/_run_mvp_hierarchical.py --K 4 --viterbi-min-dwell 0 --suffix _no_dwell
```

NOT in scope yet (`--share-demit` is flagged in the script as not-yet-wired):
- shared-d_emit sensitivity (spec V4)

After all five fits, output is in:
- `results/mvp/hierarchical/` (production)
- `results/mvp/hierarchical_k3/` (K=3 sensitivity)
- `results/mvp/hierarchical_med/`, `results/mvp/hierarchical_pe/` (protocol)
- `results/mvp/hierarchical_no_dwell/` (Viterbi sensitivity)

---

## Stage 7: Verification — V1, V2, V3, V5 (~5 min)

```bash
python scripts/_run_mvp_verification.py
```

Output to `results/mvp/diagnostics/`:
- `dwell_report.md` — V1 (per-state dwell ratio + short-dwell fraction)
- `k_comparison_report.md` — V2 (BIC + emission-mean separation; held-out
  LL CV is documented but not yet implemented)
- `protocol_stratified_report.md` — V3 (Spearman ρ between pooled and
  protocol-stratified rankings)
- `flexibility_circularity_report.md` — V5 (partial r between flexibility
  covariate and emission residual top PCs)
- `verification_report.md` — top-level summary with K_winner verdict

V4 (shared-d_emit sensitivity) is skipped automatically — `--share-demit`
flag is not yet wired in `fit_hierarchical_slds`.

---

## Stage 8: Figures (~2 min)

```bash
python scripts/_make_mvp_figures.py
```

Auto-detects K_winner from `verification_report.md` (override with `--K 3` or
`--K 4`). Per-session ranking by `mean((COUP+SHARED) gamma)` over
`conv_1+conv_2`.

Output:
- `results/mvp/figures/figure1_dyad_variability.{png,pdf}` — 3 panels:
  highest-coupling dyad, lowest-coupling dyad, cohort condition aggregate
- `results/mvp/figures/figure2_protocol_comparison.{png,pdf}` — 3 panels:
  top meditation dyad, top PE dyad, side-by-side protocol aggregates

---

## Final report

Write `results/mvp/AUTONOMOUS_RUN_REPORT.md` with:

1. Stage-by-stage timing (wall-clock per stage)
2. Sessions used in production fit (exact n)
3. Sessions excluded (with reason — most likely "needs MATLAB EEG cleaning")
4. K_winner from V2 (3 or 4)
5. Per-condition state usage table (cohort + per-protocol)
6. Phase 0 decision (DDTW vs baseline)
7. Verification verdicts (PASS/FAIL per V1/V2/V3/V5)
8. Final figure file paths

End with the line:
```
RUN COMPLETE: <X> sessions used, K=<3|4>, pose_channel=<ddtw|baseline>
```

---

## Failure handling

- **OOM** (peak RSS > 60 GB): STOP. The resource discipline failed somewhere.
  Capture which stage, which script, full traceback, and notify the user.
  Do NOT continue.
- **MATLAB EEG missing for >4 sessions**: STOP. Something has changed since
  2026-05-01. List the missing sessions and ask the user.
- **V11 scaffold fails for >50% of canonical sessions**: STOP. Likely a
  preproc problem upstream. Capture failures and ask user.
- **Phase 0 fails to produce any report**: Default to `pose_baseline` and
  proceed; note in final report.
- **Hierarchical fit diverges or produces NaN BIC**: Log the failure, continue
  with whichever variants did succeed.

---

## Total expected wall-clock

| Stage | Time |
|---|---|
| 0. Inventory | 1 min |
| 1. Ingestion backfill | 0–10 min |
| 2. Preprocessing × 4 modalities | 30–60 min |
| 3. V11 scaffold regen × ~18 sessions | 60–120 min |
| 4. Phase 0 DDTW validation | 10–20 min |
| 5. MVP scaffold slice | 5 min |
| 6. Hierarchical fits × 5 variants | 180–300 min |
| 7. Verification | 5 min |
| 8. Figures | 5 min |
| **Total** | **~5–8 hours** |

Run autonomously; the user does not need to intervene unless one of the
failure-handling triggers fires.
