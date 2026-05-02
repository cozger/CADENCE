# cadence.ingest — Layer 1: Structural ingestion

Reads raw XDF files and writes canonical session artifacts to `data/digest/v1/`.
Pure structural I/O — no signal cleaning or value transforms.

## Public API

```python
from cadence.ingest import resolve_roles, RoleAssignment, CanonicalSession
from cadence.ingest.digest import digest_xdf, load_digest, digest_all
from cadence.ingest.quality import list_canonical_sessions, load_session_quality
```

## CLI

```bash
# Digest one session
python -m cadence.ingest --session y_06

# Digest every canonical session per configs/session_quality.yaml
python -m cadence.ingest --all

# Parallel digestion (threading backend; each worker holds one XDF in memory)
python -m cadence.ingest --all --n-jobs 4

# Force re-digestion (e.g., after raw XDF re-recorded under same name)
python -m cadence.ingest --session y_06 --force
```

## What lives here

* `xdf_reader.py` — raw XDF -> in-memory streams dict. Inlines the
  EventMarkers + BehavioralMarkers merge fix (was at
  `cadence/data/xdf_loader.py:37-51`) and the four-pass marker
  normalization pipeline (`normalize_markers`: dedupe -> overrides ->
  legacy baseline -> widest-range conflict resolution).
* `roles.py` — single source of truth for therapist/patient resolution.
  Replaces `_detect_roles`, `_resolve_roles_from_xdf`, and the script-level
  fallback call sites. Therapist alias `{Sway, RA}` (case-insensitive);
  patient pattern `^[Yy]_?\d+|SYNC_\d+` with optional date-suffix tolerance.
* `quality.py` — reads `configs/session_quality.yaml` (the single classification
  scheme; replaces hardcoded `EXCLUDED_MODALITIES`). Includes
  `marker_overrides` mechanism for per-session recording-error corrections
  (rename / drop, with t_min/t_max time windows).
* `schema.py` — `CanonicalSession` + `QualityFlags` dataclasses,
  `SCHEMA_VERSION = "v1"`, `detect_pose_format` (132 / 133 / 400 ->
  mediapipe33 / mediapipe33_meta / wholebody_133).
* `digest.py` — orchestrator: `digest_xdf(xdf_path, ...) -> CanonicalSession`,
  `digest_all(..., n_jobs=N)` for parallel digestion (threading backend).
* `cli.py` — `python -m cadence.ingest`.

The 133->33 body-only gather (`pose_subset.py`) lives in
`cadence/preprocess/pose/`, not here — it's a value transform, not
structural I/O.

See `docs/data_pipeline_v1.md` for the canonical schema and migration record.
