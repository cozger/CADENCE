# Pose Coupling Phase 1 — Angle Features, Landing Coincidence, Path Timing, Noise Floor, Block Bootstrap

**Date**: 2026-09-09
**Owner**: cozger
**Status**: Plan — implementing via multi-agent workflow
**Builds on**: `cadence/significance/pose_ddtw.py` (MVP pose channel, Phase 0),
`cadence/significance/face_event_coincidence.py` (production BL channel substrate),
`scripts/_validate_pose_ddtw.py` (Phase 0 validation gates)
**Motivation**: comparison against `jeremyipark/vision-demos/dance_sync`
(torso-frame segment angles, posture/timing split, measured noise floor,
"pictures" = landing coincidence, block-bootstrap ranking). See §1.

> **For agentic workers:** each Task below is self-contained and names every
> file it may touch. Do not edit files outside your Task's file list. Steps use
> checkbox (`- [ ]`) syntax. Run the named tests before returning.

---

## 1. Why

The production MVP pose channel (`pose_ddtw`, Phase 0) runs dependent DTW on
raw 99-D MediaPipe-33 positions after shared cross-session PCA and per-window
mean centering. Three weaknesses, all made concrete by the dance_sync
comparison:

1. **Raw positions are not in a shared frame.** P1 and P2 are captured by
   *different cameras*. Per-window centering removes the mean offset but not
   viewpoint, scale, or limb-proportion differences between therapist and
   patient. `pose_format_in` also mixes image-normalized MediaPipe coords with
   RTMW pixel coords across sessions. Torso-frame **segment angles** are
   invariant to translation, scale, and limb proportion, and need no divisor.
2. **DDTW cost conflates shape and rate.** A single warp cost cannot say
   whether the pair moved *in the same way* or *at the same time*, and the
   warping path (which carries lead/lag) is thrown away. We already extract
   path lag/asymmetry for face AUs in `cadence/synchrony/features/dtw.py`.
3. **No pose event channel.** The production BL channel is *event
   coincidence* of facial-activity peaks. The pose analogue — do both bodies
   settle into stillness (a "landing") or start moving together — reuses the
   same surrogate machinery and is directly interpretable.

Two smaller borrowings: a **measured per-feature noise floor** (residual vs a
3-frame moving average) so DTW cost is expressed in noise-SD units rather than
raw radians; and a **block bootstrap** for per-session summary CIs, which no
cadence module currently provides.

What we deliberately do **not** borrow: clip-relative 0–100 anchors (breaks
cross-session comparability required by the hierarchical rSLDS), the clamp at
zero (discards anti-coupling, which is informative for us), and same-instant
(±67 ms) comparison (our coupling lives at 0.5–3 s lags).

---

## 2. Architecture

```
data/preproc/pose/v1/<sid>.npz  ({p}_pose33 (N,33,4), {p}_pose_features_valid)
   |
   |-- cadence/significance/pose_angles.py            [NEW, Task 1]
   |     pose33 -> 12 torso-frame segment angles (N,12) + valid (N,)
   |     unwrap, angular speed (deg/s), per-feature noise floor
   |
   |-- cadence/significance/pose_ddtw.py               [EXTEND, Task 3]
   |     feature_mode: 'pca' (Phase 0, unchanged) | 'angles' | 'angle_speed'
   |     noise-floor normalisation (angles modes)
   |     warping-path features per stride: mean lag, lag var, asymmetry
   |
   |-- cadence/significance/pose_event_coincidence.py  [NEW, Task 4]
   |     speed envelope -> landings (speed minima) / movement peaks
   |     -> 2 Hz grid -> circular-shift coincidence z -> sigma=15s smooth + standardize
   |     (imports peaks_to_grid, _coincidence_z from face_event_coincidence)
   |
   |-- cadence/significance/block_bootstrap.py         [NEW, Task 2]
   |     block-bootstrap mean / CI / paired comparison for per-session summaries
   |
   |-- scripts/_validate_pose_channels.py              [NEW, Task 5]
   |     Phase 1 driver: runs every candidate mode through Phase 0 Tests 1-3
   |     writes results/mvp/phase1_pose/pose_<mode>_per_session.npz + phase1_report.md
   |
   '-- scripts/_run_mvp_scaffold.py                    [EXTEND, Task 5]
         --pose-channel gains: angles | angle_speed | evt_landing | evt_peak
```

All new channels emit **surrogate z at 2 Hz on stream-relative time**, the
same contract as `pose_ddtw_per_session.npz` (`{sid}__z`, `{sid}__stride_ts`),
so `_run_mvp_scaffold.py` can slice any of them into the MVP `pose` slot.

Coordinate convention used throughout: MediaPipe-33 `(x, y, z, vis)`;
**only x and y are used for angles** (RTMW sessions have z=0; MediaPipe z is
the noisiest axis). Landmark indices (populated for all three pose formats):
nose 0, ears 7/8, shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24,
knees 25/26, ankles 27/28. Visibility threshold 0.5 (matches
`cadence.preprocess.pose.pipeline.VISIBILITY_THRESHOLD`).

Sign convention for lag (matches `cadence/synchrony/features/coincidence.py`
and `features/dtw.py`): **positive lag = P2 event/frame later than P1**.
Callers map P1/P2 to therapist/patient via the digest roles; never label
outputs by P1/P2 in reports.

---

## 3. File Map

| File | Action | Task | Responsibility |
|------|--------|------|----------------|
| `cadence/significance/pose_angles.py` | Create | 1 | Torso-frame segment angles, unwrap, speed, noise floor |
| `tests/significance/__init__.py` | Create | 1 | package marker |
| `tests/significance/test_pose_angles.py` | Create | 1 | invariance + NaN tests |
| `cadence/significance/block_bootstrap.py` | Create | 2 | block bootstrap utilities |
| `tests/significance/test_block_bootstrap.py` | Create | 2 | coverage + autocorrelation tests |
| `cadence/significance/pose_ddtw.py` | Modify | 3 | feature modes, noise normalisation, path features |
| `tests/significance/test_pose_ddtw_modes.py` | Create | 3 | pseudo-dyad null + dose response + lag sign |
| `cadence/significance/pose_event_coincidence.py` | Create | 4 | landing / peak coincidence channel |
| `tests/significance/test_pose_event_coincidence.py` | Create | 4 | null ≈ 0, injected coincidence > 0, lag sign |
| `scripts/_validate_pose_channels.py` | Create | 5 | Phase 1 driver over all modes |
| `scripts/_run_mvp_scaffold.py` | Modify | 5 | new `--pose-channel` modes + generic loader |
| `CLAUDE.md` | Modify | 6 | project structure + MVP note |
| `docs/superpowers/specs/2026-09-09-pose-coupling-phase1-design.md` | Create | 6 | design spec (post-hoc, mirrors this plan) |

Read-only references: `cadence/preprocess/pose/pipeline.py` (feature layout,
visibility threshold), `cadence/synchrony/features/dtw.py` (path feature
definitions), `scripts/_run_rslds_scaffold_v8.py::prewhiten_and_standardize`,
`scripts/_validate_pose_ddtw.py` (Phase 0 test helpers — import, do not copy).

Test preamble convention (existing tests hoist torch before numpy for Windows;
torch may be absent in CI/remote containers):

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
try:
    import torch as _torch  # noqa: F401  (Windows torch/numpy import-order guard)
except ImportError:
    pass
import numpy as np
```

Run tests with `python -m pytest tests/significance -q` from the repo root.

---

### Task 1: `pose_angles.py` — torso-frame segment angles

**Files:** create `cadence/significance/pose_angles.py`,
`tests/significance/__init__.py`, `tests/significance/test_pose_angles.py`.

Public API (exact names — Tasks 3 and 4 code against these):

```python
VISIBILITY_THRESHOLD = 0.5
MIN_SEGMENT_LEN = 0.01        # in the pose's own (x, y) units; frames whose torso is shorter -> all NaN
MIN_TORSO_LEN = 0.02

# (name, tier, from_landmark, to_landmark); 'mid_hip', 'mid_shoulder', 'mid_ear' are virtual points
ANGLE_FEATURES: tuple[tuple[str, str, str | int, str | int], ...]  # 12 entries, order fixed:
#  torso_lean(torso)  neck(neck)  head_twist(head_twist)  shoulder_line(shoulder_line)
#  l_thigh(thigh) r_thigh(thigh) l_upper_arm(upper_arm) r_upper_arm(upper_arm)
#  l_forearm(forearm) r_forearm(forearm) l_shin(shin) r_shin(shin)
ANGLE_NAMES: list[str]          # 12 names in that order
ANGLE_TIERS: list[str]
ANGLE_WEIGHTS: dict[str, float] # tier -> weight, dance_sync defaults
                                # torso 1.0 thigh 1.0 neck 1.0 shoulder_line 0.8 head_twist 0.6
                                # upper_arm 0.45 forearm 0.2 shin 0.2
def feature_weights() -> np.ndarray   # (12,) normalised to sum 1

def angle_features(pose33: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose33 (N,33,4) -> (angles (N,12) float32 radians, valid (N,) bool).

    Torso frame per frame: origin mid-hip, u = unit(mid_shoulder - mid_hip), r = perp(u).
    Each segment direction v = p[to]-p[from]; angle = arctan2(v.r, v.u), in (-pi, pi].
    torso_lean is measured against image vertical: arctan2(u_x, -u_y) (signed, 0 = upright).
    NaN for a feature when either endpoint has vis <= VISIBILITY_THRESHOLD, or |v| < MIN_SEGMENT_LEN.
    All-NaN row and valid=False when torso length < MIN_TORSO_LEN or hips/shoulders hidden.
    valid[i] is True iff torso frame defined AND at least 6 of 12 features finite.
    Uses only x, y (columns 0, 1)."""

def unwrap_nan(angles: np.ndarray) -> np.ndarray:
    """Column-wise np.unwrap that interpolates across NaN gaps for branch choice and restores NaN."""

def moving_average_nan(x: np.ndarray, window: int) -> np.ndarray:
    """Centered, edge-padded, NaN-ignoring moving average along axis 0 (x is (N,) or (N,D))."""

def speed_features(angles: np.ndarray, fs: float, smooth_frames: int = 5) -> np.ndarray:
    """(N,12) unwrapped-smoothed angular speed magnitude in deg/s (np.gradient * fs, abs)."""

def noise_floor(angles: np.ndarray, window: int = 3) -> np.ndarray:
    """(12,) per-feature noise SD in radians: 1.4826 * median |unwrapped - moving_average(unwrapped, window)|,
    finite samples only; floored at 1e-4; NaN features -> 1e-4."""

def speed_envelope(angles: np.ndarray, fs: float, smooth_frames: int = 5,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """(N,) weighted (feature_weights by default) NaN-aware mean of speed_features; NaN where <3 features finite."""

def pose33_to_angle_stream(pose33: np.ndarray, ts: np.ndarray, fs_hint: float = 30.0) -> dict:
    """Convenience: {'angles': (N,12) float32 unwrapped radians, 'valid': (N,) bool,
    'speed': (N,12) float32 deg/s, 'envelope': (N,) float32, 'noise_floor': (12,) float32,
    'fs': float (estimated from ts, fallback fs_hint)}."""
```

- [ ] Implement with vectorised numpy (no per-frame Python loop over N; a loop over the 12 features is fine).
- [ ] Tests (`test_pose_angles.py`), using a hand-built synthetic skeleton generator
      `make_skeleton(theta_elbow_l=..., lean=..., scale=1.0, offset=(0,0), arm_len=1.0)` returning (33,4):
  - all 12 features finite for a fully visible upright skeleton; torso_lean ≈ 0.
  - **translation invariance**: adding an (x,y) offset leaves angles unchanged (allclose 1e-6).
  - **scale invariance**: multiplying coordinates by 2.5 leaves angles unchanged.
  - **proportion invariance**: doubling forearm length leaves forearm angle unchanged.
  - **rotation of the whole body by φ** changes torso_lean by φ and leaves every other feature unchanged.
  - **hidden landmark** (vis=0 on left wrist) → l_forearm NaN, others finite, valid still True.
  - **hidden hips** → whole row NaN, valid False.
  - `unwrap_nan` on a signal crossing ±π with a NaN gap has no jump > π between finite neighbours.
  - `noise_floor` on a smooth sinusoid + N(0, σ) jitter recovers σ within 30%.
  - `speed_features` of a constant rotation at ω deg/s returns ≈ ω (within 5%) away from the edges.

### Task 2: `block_bootstrap.py`

**Files:** create `cadence/significance/block_bootstrap.py`,
`tests/significance/test_block_bootstrap.py`.

```python
def block_bootstrap_mean(x, block_len: int, n_boot: int = 4000, seed: int = 0,
                         weights=None, ci: float = 0.95) -> dict:
    """x (T,) or (T,K), NaN-aware (NaN samples get weight 0). Resample contiguous
    blocks of block_len with replacement (n_blocks = T // block_len, trailing
    remainder dropped — report 'n_dropped'). Returns {'mean': (K,), 'ci_lo': (K,),
    'ci_hi': (K,), 'se': (K,), 'boot': (n_boot, K), 'n_blocks': int, 'n_dropped': int}.
    Vectorised: mean of resampled blocks == mean of block means (equal-length blocks)."""

def block_bootstrap_paired(a, b, block_len, n_boot=4000, seed=0, ci=0.95) -> dict:
    """Paired difference a-b over a shared block structure. Returns mean diff, CI, and
    'p_greater' = fraction of resamples where mean(a) > mean(b)."""

def block_bootstrap_rank(scores, block_len, n_boot=4000, seed=0, tie_band=0.35) -> list[dict]:
    """scores (T,K) -> competition-style ranks; two columns tie when |P(i>j) - 0.5| <= tie_band.
    Each row: {'index', 'rank', 'mean', 'ci95', 'p_ranks_first', 'tied_with'}."""

def block_len_from_seconds(seconds: float, fs: float) -> int
```

- [ ] Tests: (i) i.i.d. normal, block_len=1: CI covers the true mean in ≥ 90/100 seeded
      trials; (ii) AR(1) with ρ=0.9: `se` with block_len=50 is at least 2× the `se` with
      block_len=1; (iii) `block_bootstrap_rank` on three columns with means 0, 0, 1
      returns rank 1 for column 2 and ties columns 0 and 1; (iv) NaNs are ignored, not propagated.

### Task 3: extend `pose_ddtw.py` — feature modes, noise normalisation, path features

**Files:** modify `cadence/significance/pose_ddtw.py`; create
`tests/significance/test_pose_ddtw_modes.py`. Depends on Task 1 API.

Keep every existing public function and default behaviour **byte-for-byte
equivalent for `feature_mode='pca'`** (Phase 0 results must stay
reproducible). Additions:

```python
FEATURE_MODES = ('pca', 'angles', 'angle_speed')

def build_angle_features(pose33_uniform: np.ndarray, ts_uniform: np.ndarray,
                         mode: str, noise_normalize: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """(N,33,4) resampled pose -> (X (N,D) float64, valid (N,) bool, info).
    'angles': unwrapped radians, NaN filled by linear interpolation inside valid runs
              (leading/trailing NaN -> 0), each column divided by its noise floor when
              noise_normalize (info['noise_floor'] holds the (12,) radians values).
    'angle_speed': speed_features (deg/s), smoothed 5 frames, divided by its own
              noise floor (median-absolute-residual of the speed vs 3-frame MA)."""

def sliding_ddtw_path_features(p1: np.ndarray, p2: np.ndarray, window_frames, stride_frames,
                               p1_valid, p2_valid, fs: float, valid_frac=DDTW_VALID_FRAC) -> dict:
    """Real pair only (no surrogates). Per stride via dtw_ndim.warping_paths(..., use_c=True)
    + dtw.best_path: {'lag_s': (n_strides,), 'lag_var': (n_strides,), 'asym': (n_strides,)}.
    Sign: positive = P2 index ahead of P1 index (P2 later). NaN where window invalid."""
```

- `compute_session_ddtw(...)` gains kwargs `compute_path_features: bool = False`
  and `fs: float = DDTW_TARGET_RATE_HZ`; when true, the returned dict also has
  `ddtw_lag_s`, `ddtw_lag_var`, `ddtw_asym`.
- `run_session_ddtw(...)` gains `feature_mode: str = 'pca'`,
  `noise_normalize: bool = True`, `compute_path_features: bool = False`. In
  angle modes `components`/`mean` may be `None`; the (N,33,4) array is
  resampled per landmark coordinate (reuse `resample_to_uniform_rate` on the
  flattened (N,132) view, then reshape), and validity is
  `pose_features_valid AND angle valid`. Info returned in the dict:
  `feature_mode`, `noise_floor` (or None), `n_features`.
- Resampling note: resample the **raw pose33** to 12 Hz first and compute
  angles on the uniform grid (so the unwrap/speed operate at a fixed fs). The
  visibility column is resampled with nearest-neighbour (reuse
  `resample_validity_to_uniform` per landmark, or threshold the linearly
  interpolated vis at 0.5 — document which).

- [ ] Tests (`test_pose_ddtw_modes.py`), synthetic, no data files, fast (< 30 s):
  - **Pseudo-dyad null (critical rule)**: two *independent* random-walk angle
    streams (12-D, smooth, 12 Hz, 120 s) as P1 and P2; run `compute_session_ddtw`
    with `n_surrogates=40`; mean finite z ∈ [-0.5, 0.5].
  - **Dose response**: P2 = κ·warped(P1) + (1-κ)·independent, κ ∈ {0, 0.4, 0.8}
    (reuse `piecewise_linear_warp` logic locally — do not import from scripts/);
    mean z strictly increases with κ and mean z at κ=0.8 > 1.0.
  - **Lag sign**: P2 = P1 delayed by 0.5 s → `ddtw_lag_s` median > 0.2 s;
    swapping arguments flips the sign.
  - **Backward compatibility**: `_ddtw_score` and `compute_session_ddtw` with
    default args on random PCA input produce identical output to a frozen
    call made before the edit (compute once at test time by calling the
    unchanged private path — i.e. assert `feature_mode='pca'` path never
    touches the new code; simplest: monkeypatch `build_angle_features` to
    raise and confirm the default call succeeds).
  - `build_angle_features('angles')` output columns have std within
    [0.5, 50] after noise normalisation on a jittered smooth signal, and
    `info['noise_floor']` has shape (12,).

### Task 4: `pose_event_coincidence.py` — landing / peak coincidence channel

**Files:** create `cadence/significance/pose_event_coincidence.py`,
`tests/significance/test_pose_event_coincidence.py`. Depends on Task 1 API
(import `pose33_to_angle_stream`, `speed_envelope`).

```python
EVENT_KINDS = ('landing', 'peak')

def detect_landings(envelope: np.ndarray, fs: float, min_gap_s: float = 0.5,
                    prominence_frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Local minima of the angular-speed envelope = the body coming to rest.
    scipy.signal.find_peaks on -envelope with distance=round(min_gap_s*fs) and
    prominence = prominence_frac * (nanpercentile(env,95) - nanpercentile(env,5)).
    NaN samples are filled with the running max before negation so a gap is never a minimum.
    Returns (idx, depth)."""

def detect_movement_peaks(envelope, fs, quantile_threshold=0.70, min_sep_s=1.0):
    """Mirror of face_event_coincidence.detect_activity_peaks on the speed envelope."""

def compute_pose_event_coincidence(pose_npz_data, t_common, lsl_offset, *,
                                   event_kind='landing', fs_hint=30.0,
                                   n_surrogates=200, seed=42, tau_samples=1,
                                   min_gap_s=0.5, prominence_frac=0.12,
                                   quantile_threshold=0.70, min_sep_s=1.0,
                                   smooth_sigma_s=15.0) -> tuple[np.ndarray, dict]:
    """End-to-end from data/preproc/pose/v1/<sid>.npz contents
    (keys p{1,2}_pose33, p{1,2}_pose33_ts, p{1,2}_pose_features_valid).
    Envelope per participant = pose33_to_angle_stream(...)['envelope'] with invalid
    frames set to NaN. Events -> peaks_to_grid -> _coincidence_z (both imported from
    face_event_coincidence) -> gaussian smooth + per-session standardize exactly as
    compute_bl_event_coincidence. Returns (z float32 (N,), info) with info keys
    mirroring the face version plus 'event_kind', 'p1_n_events', 'p2_n_events',
    'p1_event_rate_hz', 'p2_event_rate_hz', and 'mean_signed_lag_s' (nearest-neighbour
    P2-minus-P1 lag within ±1 s over matched events; NaN if < 5 matches)."""

def events_from_pose_npz(pose_npz_data, participant: str, event_kind: str, **kw) -> np.ndarray:
    """Event times (stream-relative s) for one participant — used by the validation driver."""
```

- [ ] Tests (synthetic 33-landmark skeleton streams built with a small local
      generator: oscillating elbow/knee angles that pause every few seconds):
  - Independent P1/P2 (different random pause schedules, **pseudo-dyad rule**):
    `n_surrogates=60`, |mean z| < 0.5 before smoothing.
  - Shared pause schedule with P2 delayed 0.3 s: mean raw z > 1.0 at
    τ=±500 ms; `mean_signed_lag_s` ≈ +0.3 (± 0.15).
  - `detect_landings` on an envelope with 5 known rest periods returns 5 events
    within ±2 frames of the true rest centres; a NaN gap creates no event.
  - Missing NPZ key → zeros + `{'status': 'missing_data'}` (contract identical to face).

### Task 5: Phase 1 validation driver + MVP scaffold sourcing

**Files:** create `scripts/_validate_pose_channels.py`; modify
`scripts/_run_mvp_scaffold.py`. Depends on Tasks 1, 3, 4.

`_validate_pose_channels.py`:
- CLI: `--modes` (default `angles angle_speed evt_landing evt_peak`; `pca` allowed
  to re-run the Phase 0 baseline through the same harness), `--sessions`, `--all`,
  `--quick`, `--n-jobs`, `--max-test1-pairs`, `--max-test3-pairs`, `--out results/mvp/phase1_pose`.
- Import (do not copy) from `scripts/_validate_pose_ddtw.py`: `discover_sessions`,
  `load_and_resample`, `make_pseudo_dyad`, `inject_coupling_into_segment`,
  `piecewise_linear_warp`, `_auc_from_scores`, and the Test 2/3 statistics
  helpers where they are mode-agnostic. `load_and_resample` returns PCA-ready
  99-D streams; for angle/event modes add `load_and_resample_pose33(sid)` that
  keeps the (N,33,4) array (resample x,y,z linearly and vis by nearest) on the
  same 12 Hz common grid, so pseudo-dyads and κ-injection operate on the same
  objects for every mode. κ-injection for angle modes mixes the **pose33
  coordinates** (so angles are recomputed from mixed skeletons); for event
  modes the injected P2 is the same mixed skeleton.
- Per mode: run real sessions → `{sid}__z`, `{sid}__stride_ts`, plus
  `{sid}__z_pw` (after `prewhiten_and_standardize` from
  `scripts/_run_rslds_scaffold_v8.py`, single channel, validity = finite) and,
  for DDTW modes with `compute_path_features=True`, `{sid}__lag_s`, `{sid}__asym`.
  Save `results/mvp/phase1_pose/pose_<mode>_per_session.npz`.
- Tests 1–3 exactly as Phase 0 (same thresholds: Kendall τ perm p<0.05 &
  AUC≥0.65 at κ=0.4; Δz≥0.5 conv vs meditation with paired t or Wilcoxon p<0.05;
  real−pseudo ≥0.5 z). Test 4 (descriptive): Pearson r of each mode vs the
  Phase 0 DDTW and vs the V11 baseline, with **block-bootstrap 95% CI**
  (`block_bootstrap_paired` / `block_bootstrap_mean`, block = 10 s at 2 Hz).
- Per-session condition summaries (mean z per condition) with block-bootstrap
  CIs, written to `phase1_condition_summary.csv`.
- `phase1_report.md`: per-mode table of Tests 1–4; decision line
  `- **Pose channel for MVP scaffold: `<mode>`**` where `<mode>` ∈
  `{pose_baseline, pose_ddtw, pose_angles, pose_angle_speed, pose_evt_landing, pose_evt_peak}`
  chosen as: the passing mode with the largest Test 3 margin; `pose_ddtw` if
  only Phase 0 passes; else `pose_baseline`. Never silently overwrite
  `results/mvp/phase0/phase0_report.md`.
- Must run without a GPU; joblib threading for angle feature extraction,
  `ProcessPoolExecutor(spawn)` for DDTW as in Phase 0.

`_run_mvp_scaffold.py`:
- `--pose-channel` choices become `auto, baseline, ddtw, angles, angle_speed, evt_landing, evt_peak`.
- `read_phase_decision()` reads `results/mvp/phase1_pose/phase1_report.md`
  first, then falls back to `phase0_report.md`, then `'pose_baseline'`.
- Generalise `load_phase0_ddtw_for_session` into
  `load_pose_channel_for_session(sid, t_common, mode)` mapping modes to NPZ
  paths (`phase0/pose_ddtw_per_session.npz` for `pose_ddtw`, `phase1_pose/pose_<m>_per_session.npz` otherwise, key `__z`);
  keep the existing function name as a thin wrapper. Record `pose_source` in
  the sidecar as today.
- [ ] No behaviour change for `--pose-channel auto` when no phase1 report exists.
- [ ] `python -c "import ast,sys; ast.parse(open('scripts/_run_mvp_scaffold.py').read())"` and
      `python scripts/_validate_pose_channels.py --help` both succeed.

### Task 6: docs + CLAUDE.md

**Files:** modify `CLAUDE.md`; create
`docs/superpowers/specs/2026-09-09-pose-coupling-phase1-design.md`.

- [ ] CLAUDE.md project structure: add `pose_angles.py`, `pose_event_coincidence.py`,
      `block_bootstrap.py` under `significance/`; add `_validate_pose_channels.py` under
      scripts; one paragraph in the MVP section noting Phase 1 pose candidates and that
      the production channel is unchanged until `phase1_report.md` exists.
- [ ] Spec doc: header block in house style, problem statement, the borrowed-vs-rejected
      table from §1, API summary, validation gates, and a "Follow-ups" section listing
      (a) YQP-Explorer `BodyPoseDTWAnalyzer` null calibration, (b) YouQuantiPy real-time
      pose synchrony (see `docs/superpowers/specs/2026-09-09-yqp-realtime-pose-sync-design.md`).

---

## 4. Validation gates (unchanged from Phase 0, applied per mode)

| Test | Gate |
|------|------|
| 1 Semi-synthetic dose response (pseudo-dyad base) | Kendall τ perm p < 0.05 AND AUC ≥ 0.65 at κ=0.4 |
| 2 Condition contrast conv vs meditation | Δz ≥ 0.5 AND (paired t p<0.05 OR Wilcoxon p<0.05) |
| 3 Pseudo-dyad null | real − pseudo ≥ 0.5 z |
| 4 Redundancy vs Phase 0 DDTW / V11 baseline | descriptive, block-bootstrap CI |

Data-dependent runs (`_validate_pose_channels.py --all`) happen on the lab
machine; this plan's workflow delivers code + synthetic tests only.

## 5. Out of scope

- Changing the production K=3 fit or `results/mvp/hierarchical_evtcoinc_smooth/`.
- Multi-person tracking / person re-identification (not a CADENCE problem).
- 3D torso frames (revisit when stereo gaze hardware lands, V12).
