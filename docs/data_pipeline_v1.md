# CADENCE Data Pipeline v1

**Status:** Production. Migrated 2026-05-01. Replaces the MCCT-style cache and
the EEGLAB-wavelet-pipeline cache as a single source of truth for raw-data
ingestion and per-modality preprocessing.

## Why two layers, not one

The pipeline is split into two cleanly-versioned layers:

```
raw sessions/<session>.xdf
        │
        ▼  Layer 1: cadence/ingest/      (structural I/O — no value transforms)
data/digest/v1/<session_id>.npz    +    .json
        │
        ▼  Layer 2: cadence/preprocess/<modality>/
data/preproc/<modality>/v1/<session_id>.npz + .json
        │
        ▼  Analytics (V11/V10/V8.2 scaffolds, rSLDS, wavelet)
results/v11/, results/v10/, ...
```

EEG MATLAB preprocessing takes ~30 min/session × ~18 ≈ 9 hrs. ECG/face/pose
preprocessing takes seconds. Versioning them together would force re-running
MATLAB whenever (e.g.) ECG bandpass parameters change. Independent versioning
(`data/preproc/eeg/v1/`, `data/preproc/face/v1/`) is the right call given that
asymmetry.

`xdf_md5` keys flow downstream: each preproc sidecar carries the source
digest's `xdf_md5`, and a digest re-run invalidates all four preproc
artifacts.

## Layer 1 — Digestion (`cadence.ingest`)

Pure structural I/O: stream identification, role resolution, marker merge,
pose-format detection, quality flags. **No signal cleaning, no value
transforms.**

### What's in a digest

`data/digest/v1/<session_id>.npz` arrays (per participant `p ∈ {p1, p2}`):

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `{p}_eeg_raw` | `(N_eeg, 19)` | float32 | unprocessed Emotiv EPOC frame |
| `{p}_eeg_ts` | `(N_eeg,)` | float64 | session-relative seconds |
| `{p}_ecg_raw` | `(N_ecg, 1)` | float32 | Polar H10 raw |
| `{p}_ecg_ts` | `(N_ecg,)` | float64 | dejittered |
| `{p}_landmarks_raw` | `(N_lm, 1489)` | float32 | first 52 cols are AU blendshapes |
| `{p}_landmarks_ts` | `(N_lm,)` | float64 | |
| `{p}_pose_full` | `(N_pose, C)` | float32 | C=132/133/400 per format |
| `{p}_pose_ts` | `(N_pose,)` | float64 | |

`data/digest/v1/<session_id>.json` sidecar:

```json
{
  "schema_version": "v1",
  "session_id": "Y_55_04272026",
  "xdf_basename": "Y_55_04272026.xdf",
  "xdf_md5": "<md5 of raw xdf>",        // staleness key
  "duration_s": 3422.8,
  "t_start_lsl": 47887.42,
  "protocol": "meditation",              // {meditation, PE, other}
  "modalities": ["eeg", "ecg", "landmarks", "pose"],
  "roles": {
    "p1_role": "patient",                // {therapist, patient}
    "p2_role": "therapist",
    "p1_name": "Y_55",
    "p2_name": "Sway",
    "role_source": "xdf_landmark"        // {xdf_landmark, xdf_ra_alias, ...}
  },
  "markers": [[0.0, "base_EO_start"], "..."],
  "marker_sources": ["EventMarkers", "BehavioralMarkers"],
  "pose_format": "mediapipe33_meta",     // {mediapipe33, mediapipe33_meta, wholebody_133}
  "quality_flags": {
    "excluded_modalities": {"p1": [], "p2": []},
    "missing_markers": [],
    "notes": []
  },
  "stream_inventory": {
    "p1_eeg_raw": {"shape": [872662, 19], "srate_hz": 256.14, "src_name": "EmotivDataStream-EEG"}
  }
}
```

### Pose format dispatch

Three raw pose formats encountered in CADENCE recordings:

| Format | n_channels | Layout | Sessions |
|---|---|---|---|
| `mediapipe33` | 132 | 33 keypoints × (x, y, z, visibility) | y_06, older Sway-tag sessions |
| `mediapipe33_meta` | 133 | mediapipe33 + 1 trailing metadata flag | Y_55, y_59, y_64, y_65, y_66 |
| `wholebody_133` | 400 | 133 keypoints × (x, y, score) + 1 provenance | y_53 (RTMW) |

Detection is by channel count (`cadence.ingest.schema.detect_pose_format`).
Unknown counts raise; we fail-loud rather than silently mis-shape.

### Role resolution

**Stream order (P1/P2) in the XDF is NOT consistent across sessions.** The
canonical resolver lives in `cadence.ingest.roles`:

- Therapist alias: `{"Sway", "RA"}`, case-insensitive.
- Patient pattern: `r"^(?:[Yy]_?\d+|SYNC_\d+)(?:[_-].*)?$"` (suffix-tolerant
  to handle date-suffixed names like `Y01_020626`).
- Source: read from BOTH `P1_landmarks` and `P2_landmarks` `desc->participant_name`.
- Exactly-one-therapist + exactly-one-patient → resolved. Otherwise raise
  `UnresolvedRolesError` and the digest CLI logs + skips.

Old behavior (in legacy `cadence.data.xdf_loader._detect_roles`) silently
defaulted six RA-tag sessions (Y_55, y_59, y_64, y_66, y26, y_65) to
`unknown/unknown`. The new resolver fixes this.

### Marker pipeline

`cadence.ingest.xdf_reader.normalize_markers` applies four passes in order:

1. **Dedupe** — sort by time and drop exact `(t, label)` duplicates from the
   merged Markers stream union.
2. **Per-session overrides** — apply `marker_overrides` from
   `configs/session_quality.yaml` (rename or drop). Use only for genuine
   recording errors (operator mis-clicks, spurious labels), not analytical
   re-interpretation.
3. **Legacy baseline pairing** — older sessions emitted `baseline_start/stop`
   pairs without `base_EO_*`/`base_EC_*`. Treat the first pair as eyes-open
   and the second as eyes-closed.
4. **Widest-range conflict resolution** — for each condition with multiple
   `_start` or multiple `_stop` events, keep the earliest `_start` and latest
   `_stop`.

### Session quality registry

`configs/session_quality.yaml` is the single classification scheme:

```yaml
sessions:
  <session_id>:
    canonical: true                    # included in canonical analyses
    excluded_modalities:               # per-participant modality blacklist
      p1: [ecg, ecg_features]
      p2: [pose, pose_features]
    notes: ["free-text rationale"]
    marker_overrides:                  # per-session corrections (rare)
      - kind: rename
        from: conv_2_start
        to:   conv_1_start
        t_min: 560.0
        t_max: 600.0
        reason: "mis-clicked button at 577s"
      - kind: drop
        label: PE_2_start
        t_min: 3800.0
```

The audit script (`scripts/_audit_xdf_inventory.py`) inspects every XDF and
generates this YAML. User reviews and signs off; subsequent runs preserve
user-edited fields (`canonical`, `excluded_modalities`, `marker_overrides`)
while refreshing audit metadata.

### Public API

```python
from cadence.ingest import resolve_roles, RoleAssignment, CanonicalSession
from cadence.ingest.digest import digest_xdf, load_digest, digest_all
from cadence.ingest.quality import (
    list_canonical_sessions,
    load_session_quality,
    apply_modality_exclusions,
)
```

CLI: `python -m cadence.ingest --session y_06` or `--all` (with optional
`--n-jobs N` for parallel digestion via `joblib(threading)`).

## Layer 2 — Preprocessing (`cadence.preprocess.<modality>`)

Each submodule reads only the digest, writes one atomic
`.npz`+`.json` pair. Per-participant only — cross-participant features go
in `cadence.coupling`.

### Submodule contract

```python
preprocess_<modality>_session(session_id, *,
                              digest_dir="data/digest/v1",
                              out_dir=None,    # default: data/preproc/<mod>/v1
                              force=False) -> dict
```

- **Input:** session_id (string). Reads `data/digest/v1/<sid>.npz` only.
- **Output:** `data/preproc/<mod>/v1/<sid>.npz` + `.json`.
- **Idempotent:** if output's `digest_xdf_md5` matches the digest, skip.
- **Per-participant only.** Output keys begin with `p1_` or `p2_`.

### Pose (`cadence.preprocess.pose`)

Format dispatch via `normalize_pose_to_mp33` → `(N, 33, 4)` MediaPipe-shaped
output regardless of source format. Then visibility-aware feature
extraction:

- 40 joint-group features + 1 activity channel = `(N, 41)`.
- **Deliberate behavioral fix:** `_centroid` and `_angle` return NaN for
  groups with no visible landmarks; downstream z-score uses NaN-safe stats
  (`np.isfinite` mask before mean/std). Legacy `_centroid` computed an
  unconditional mean across all listed landmarks regardless of visibility,
  biasing the head centroid toward origin when face landmarks were missing.
  Frames with face occlusion will produce different head features post-migration
  — this is intended.
- Frame validity: `(visibility > 0.5).sum() >= 10` keypoints.
- For wholebody_133, only 22 of 33 MP slots are populated (mouth/eye_inner/eye_outer/hand
  markers are zeroed); RTMW score is promoted to MediaPipe visibility.

### Face (`cadence.preprocess.face`)

- 52 MediaPipe blendshapes (AUs) extracted from `{p}_landmarks_raw[:, :52]`.
- Linear interp gaps shorter than 0.5s.
- Per-AU z-score using only valid frames.
- Activity channel (RMS deviation from trailing 30s mean).
- Optional V7 PCA(15) + Gaussian-smoothed derivatives(15) + activity = `(N, 31)`.

### ECG (`cadence.preprocess.ecg`)

- Polar H10 at 130 Hz, dejittered by pyxdf — no resampling.
- 0.5–40 Hz Butterworth bandpass + z-score.
- R-peak detection (`scipy.signal.find_peaks`) with ectopic-beat outlier
  rejection (median filter, drop bursts where successive |Δ| > 40% of local
  median).
- 7-channel HRV at 2 Hz: `hr_bpm`, `ibi_dev_5s`, `rmssd_5s`, `hr_accel_2s`,
  `qrs_amplitude`, `hr_trend_10s`, `rmssd_derivative`. All z-scored, clipped
  to [-10, 10].

### EEG (`cadence.preprocess.eeg`)

EEG cleaning lives in MATLAB EEGLAB and is bridged here:

1. The bridge looks for `data/matlab/<sid>_p{1,2}_clean.mat` with a fresh
   `.mat.json` provenance sidecar. The sidecar's `xdf_md5` must match the
   digest. Mismatch → re-run MATLAB (or hand-relocate via
   `scripts/_relocate_clean_mats.py`).
2. The MATLAB script (`analysis/eeglab_wavelet/matlab/preprocess_eeg.m`) runs
   `clean_rawdata` (ASR burst stddev=20, FIR 1–40 Hz) + `pop_interp` +
   `pop_eegfiltnew`. Sample count is preserved (`arg_window='off'`) so
   downstream marker alignment remains valid.
3. clean.mat data is `(M, 14)` float64 microvolts (Emotiv channels 3..16
   from the raw 19); both participants are cropped to a common LSL window.
4. Per-participant timestamps are derived from the digest by re-deriving the
   common-window start index via searchsorted.
5. 8 features at 2 Hz via `cadence.data.eeg_features`: engagement index,
   frontal aperiodic exponent, theta burst fraction, frontal theta/alpha
   phase (cos/sin), activity.

#### Atomic writes + provenance sidecars

`preprocess_eeg.m` writes to `<id>_p{1,2}_clean.mat.tmp` then `os.replace`.
The sidecar is written last. Half-written files never have a sidecar; the
bridge skips them and re-runs.

Per-modality preprocess sidecars look like:

```json
{
  "session_id": "Y_55_04272026",
  "modality": "eeg",
  "modality_version": "v1",
  "digest_xdf_md5": "<staleness key>",
  "digest_schema_version": "v1",
  "matlab_params": { /* exact MATLAB script params */ },
  "params": { /* per-modality parameters */ },
  "participants": { "p1": {...}, "p2": {...} },
  "written_at": "2026-05-01T14:30:00-0700"
}
```

### CLI

```bash
python -m cadence.preprocess.pose  y_06              # one session
python -m cadence.preprocess.face  --all             # every canonical session
python -m cadence.preprocess.ecg   y_06 --force      # force re-run
python -m cadence.preprocess.eeg   --all
```

## Backward compatibility — the legacy dict view

Pre-existing scripts (V11/V10/V8.2 scaffolds, rSLDS hierarchical fits, tests)
read a flat dict from `cadence.data.alignment.load_session_from_cache`. The
shim in `cadence/data/__init__.py` keeps them working unchanged:

```python
from cadence.data import load_session_from_cache, discover_cached_sessions

sess = load_session_from_cache("y_06")   # ~30 keys per participant
sess["p1_eeg"]              # (M, 14) — from preproc.eeg
sess["p1_blendshapes"]      # (N, 53) — from preproc.face (52 AU + activity)
sess["p1_pose"]             # (N, 99) — from preproc.pose (33 kp × 3 coords)
sess["p1_pose_features"]    # (N, 41) — from preproc.pose
sess["p1_ecg_features"]     # (N, 7)  — from preproc.ecg
sess["p1_role"], sess["p1_name"], sess["role_source"]
```

**Eager-loading rationale:**

- **Joblib safety.** Returns concrete numpy arrays, not mmap'd `NpzFile`
  handles. Pickling for loky/threading workers is straightforward.
- **Mutation isolation.** Each call constructs a fresh dict with arrays
  write-protected via `setflags(write=False)`; consumer mutations don't
  bleed across calls.
- **Determinism.** Artifacts loaded at t=0 are what the consumer sees
  throughout — no mid-run filesystem races.

The shim accepts legacy MCCT-style cache paths (`session_cache/<hash>_<sid>.npz`)
or session IDs interchangeably. The MCCT prefix-hash is parsed and dropped.

## How to add a modality

1. Create `cadence/preprocess/<mod>/__init__.py`, `pipeline.py`, `cli.py`.
2. Implement `preprocess_<mod>_session(session_id, *, digest_dir, out_dir, force=False)`.
3. Add the modality name to `cadence.ingest.schema.SUPPORTED_MODALITIES`.
4. Update `cadence.ingest.xdf_reader.load_xdf_streams` to recognize the
   raw stream name.
5. Add a smoke test in `tests/preprocess/test_pipeline_smokes.py`.
6. If the new modality should appear in the legacy dict view, extend
   `cadence/data/__init__.py`'s `_load_preproc_<mod>` helper.

## Migration record (2026-05-01)

- Backup: `D:\backup\2026-05-01\` — full snapshot of MCCT/session_cache,
  analysis/eeglab_wavelet/cache, analysis/eeglab_wavelet/results
  (50 GB / 11705 files; `manifest.md5` byte-clean).
- 22/22 canonical sessions digested.
- 36 clean.mat files relocated from `analysis/eeglab_wavelet/cache/` to
  `data/matlab/` with provenance sidecars.
- 4 sessions still need fresh MATLAB EEG cleaning: y04_020626, y11_022526,
  y24_022526, y_53_04302026.
- 6 previously mislabelled sessions (Y_55, y_59, y_64, y_66, y26, y_65) now
  correctly resolve as therapist/patient (was `unknown/unknown`).
- Y_45 / y_51 marker counts +12 each from the BehavioralMarkers merge fix
  baked into `xdf_reader.py`.

## See also

- `cadence/ingest/README.md` — Layer 1 module index.
- `cadence/preprocess/README.md` — Layer 2 module index + modality recipe.
- `analysis/eeglab_wavelet/matlab/preprocess_eeg.m` — MATLAB preprocessing
  source (do not modify; bump `EEG_PREPROCESS_VERSION` in
  `cadence/preprocess/eeg/matlab_bridge.py` if you do).
- `configs/session_quality.yaml` — canonical session registry.
- Plan of record: `C:/Users/optilab/.claude/plans/review-all-available-raw-deep-kurzweil.md`.
