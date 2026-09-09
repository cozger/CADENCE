# Pose Coupling Phase 1 — Design Spec

**Date**: 2026-09-09
**Owner**: cozger
**Status**: Code-complete (post-hoc spec, mirrors the plan) — data-dependent validation (`_validate_pose_channels.py --all`) pending on the lab machine
**Cohort**: 19 canonical MVP-eligible sessions
**Builds on**: `cadence/significance/pose_ddtw.py` (MVP pose channel, Phase 0), `cadence/significance/face_event_coincidence.py` (production BL channel substrate), `scripts/_validate_pose_ddtw.py` (Phase 0 validation gates)
**Plan**: `docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md`
**Motivation**: comparison against `jeremyipark/vision-demos/dance_sync` (torso-frame segment angles, posture/timing split, measured noise floor, "pictures" = landing coincidence, block-bootstrap ranking)
**Production impact**: none until `results/mvp/phase1_pose/phase1_report.md` exists — the K=3 fit at `results/mvp/hierarchical_evtcoinc_smooth/` and the `--pose-channel auto` decision path are unchanged

---

## 1. Problem statement

The production MVP pose channel (`pose_ddtw`, Phase 0) runs dependent DTW on
raw 99-D MediaPipe-33 positions after a shared cross-session PCA and
per-window mean centering. Three weaknesses, all made concrete by the
dance_sync comparison:

1. **Raw positions are not in a shared frame.** P1 and P2 are captured by
   *different cameras*. Per-window centering removes the mean offset but not
   viewpoint, scale, or limb-proportion differences between therapist and
   patient, and `pose_format_in` mixes image-normalised MediaPipe coordinates
   with RTMW pixel coordinates across sessions. **Torso-frame segment
   angles** are invariant to translation, scale, and limb proportion and need
   no divisor.
2. **DDTW cost conflates shape and rate.** A single warp cost cannot say
   whether the pair moved *in the same way* or *at the same time*, and the
   warping path (which carries lead/lag) is thrown away. Path lag/asymmetry
   is already extracted for face AUs in `cadence/synchrony/features/dtw.py`.
3. **No pose event channel.** The production BL channel is *event
   coincidence* of facial-activity peaks. The pose analogue — do both bodies
   settle into stillness (a **landing**) or start moving together (a
   **movement peak**) — reuses the same surrogate machinery and is directly
   interpretable in the same units.

Two smaller borrowings: a **measured per-feature noise floor** (robust SD of
the residual against a 3-frame moving average) so DTW cost is expressed in
noise-SD units rather than raw radians; and a **block bootstrap** for
per-session summary CIs, which no cadence module previously provided.

Phase 1 delivers four candidate pose channels (`angles`, `angle_speed`,
`evt_landing`, `evt_peak`) that all emit the Phase 0 output contract, plus a
driver that runs every candidate through the Phase 0 validation gates and
writes a decision the MVP scaffold builder can consume.

## 2. Borrowed vs rejected from dance_sync

| dance_sync ingredient | Decision | Where | Rationale |
|---|---|---|---|
| Torso-frame segment angles (origin mid-hip, u = mid_shoulder − mid_hip, r = perp(u)) | **Borrowed** | `pose_angles.angle_features` | Invariant to camera translation, scale, limb proportion; removes the need for shared PCA and per-window centering |
| Tiered feature weights (posture segments 1.0 → shins/forearms 0.2) | **Borrowed** | `pose_angles.ANGLE_WEIGHTS` | Distal segments are noisier and more task-incidental; used for the speed envelope only (DTW cost uses all 12 columns equally in noise-SD units) |
| Posture vs timing split | **Borrowed** | `pose_ddtw.sliding_ddtw_path_features` | DDTW cost = "same shape"; warping-path mean lag / lag variance / asymmetry = "same time" |
| Measured noise floor (residual vs 3-frame MA) | **Borrowed** | `pose_angles.noise_floor`, `pose_ddtw.build_angle_features` | DTW cost in noise-SD units; per-feature, per-participant, per-session, so hardware / format differences cancel |
| "Pictures" = both dancers reach a still pose together | **Borrowed** as landings | `pose_event_coincidence.detect_landings` | Speed-envelope minima → binary grid → circular-shift coincidence z, identical to `bl_event_coincidence` |
| Block bootstrap for per-clip summaries and ranking | **Borrowed** | `block_bootstrap` | 2 Hz z traces are strongly autocorrelated; i.i.d. SEs understate uncertainty |
| Clip-relative 0–100 anchors | **Rejected** | — | Breaks cross-session comparability required by the hierarchical rSLDS; surrogate z is already session-calibrated |
| Clamp at zero | **Rejected** | — | Discards anti-coupling, which is informative for CADENCE |
| Same-instant (±67 ms) comparison | **Rejected** | — | Our coupling lives at 0.5–3 s lags; DDTW windows of 4 s and ±500 ms coincidence tolerance instead |

## 3. Architecture

```
data/preproc/pose/v1/<sid>.npz  (p{1,2}_pose33 (N,33,4), p{1,2}_pose33_ts, p{1,2}_pose_features_valid)
   |
   |-- cadence/significance/pose_angles.py               [Task 1]
   |     pose33 -> 12 torso-frame segment angles (N,12) + valid (N,)
   |     NaN-aware unwrap, angular speed (deg/s), per-feature noise floor,
   |     weighted whole-body speed envelope
   |
   |-- cadence/significance/pose_ddtw.py                  [Task 3, extended]
   |     feature_mode: 'pca' (Phase 0, byte-identical) | 'angles' | 'angle_speed'
   |     noise-floor normalisation (angle modes)
   |     warping-path timing features per stride: mean lag, lag var, asymmetry
   |
   |-- cadence/significance/pose_event_coincidence.py     [Task 4]
   |     speed envelope -> landings (speed minima) / movement peaks
   |     -> 2 Hz grid -> circular-shift coincidence z -> sigma=15 s smooth + standardise
   |     (imports peaks_to_grid, _coincidence_z from face_event_coincidence)
   |
   |-- cadence/significance/block_bootstrap.py            [Task 2]
   |     moving-block bootstrap: mean / CI, paired difference, competition ranking
   |
   |-- scripts/_validate_pose_channels.py                 [Task 5]
   |     Phase 1 driver: every candidate mode through Phase 0 Tests 1-4
   |     writes results/mvp/phase1_pose/pose_<mode>_per_session.npz + phase1_report.md
   |
   '-- scripts/_run_mvp_scaffold.py                       [Task 5, extended]
         --pose-channel: auto | baseline | ddtw | angles | angle_speed | evt_landing | evt_peak
```

Conventions used throughout:

- **Coordinates**: MediaPipe-33 `(x, y, z, vis)`; only x and y are used for
  angles (RTMW sessions have z = 0; MediaPipe z is the noisiest axis).
  Landmark indices are populated for all three pose formats: nose 0, ears
  7/8, shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26,
  ankles 27/28. Visibility threshold 0.5 (matches
  `cadence.preprocess.pose.pipeline.VISIBILITY_THRESHOLD`).
- **Lag sign** (matches `cadence/synchrony/features/coincidence.py` and
  `features/dtw.py`): **positive lag = P2 event/frame later than P1**.
  Callers map P1/P2 to therapist/patient via the digest roles; never label
  outputs by P1/P2 in reports.
- **Output contract**: every candidate channel emits surrogate z at 2 Hz on
  stream-relative time, the same contract as `pose_ddtw_per_session.npz`
  (`{sid}__z`, `{sid}__stride_ts`), so `_run_mvp_scaffold.py` can slice any
  of them into the MVP `pose` slot.
- **Time bases**: pose event times and DDTW stride timestamps live in the
  pose stream's own clock (`p{1,2}_pose33_ts`); `lsl_offset` is added only
  when binning onto `t_common`, exactly as the face channel treats
  `p{1,2}_au52_ts`.

## 4. API summary (as implemented)

Signatures below are the ones in the modules. Where the implementation
deviates from or extends the plan text, the deviation is called out inline
so downstream code targets the real contract.

### 4.1 `cadence/significance/pose_angles.py` — torso-frame segment angles

```python
VISIBILITY_THRESHOLD = 0.5
MIN_SEGMENT_LEN = 0.01        # pose's own (x, y) units; shorter segment -> NaN feature
MIN_TORSO_LEN = 0.02          # shorter torso -> all-NaN row, valid=False
ANGLE_FEATURES: tuple[tuple[name, tier, from, to], ...]   # 12 entries, fixed order
ANGLE_NAMES, ANGLE_TIERS: list[str]                        # length 12
ANGLE_WEIGHTS: dict[str, float]   # torso 1.0 thigh 1.0 neck 1.0 shoulder_line 0.8
                                  # head_twist 0.6 upper_arm 0.45 forearm 0.2 shin 0.2
N_ANGLE_FEATURES = 12; MIN_FINITE_FOR_VALID = 6; MIN_FINITE_FOR_ENVELOPE = 3
LM_NOSE, LM_L_EAR, ..., LM_R_ANKLE      # landmark index constants
VIRTUAL_POINTS = {'mid_hip': (23, 24), 'mid_shoulder': (11, 12), 'mid_ear': (7, 8)}

def feature_weights() -> np.ndarray                                   # (12,), sums to 1
def angle_features(pose33) -> tuple[np.ndarray, np.ndarray]           # (N,12) float32 rad, (N,) bool
def unwrap_nan(angles) -> np.ndarray                                  # float64, NaN restored
def moving_average_nan(x, window) -> np.ndarray                       # float64, centred, edge-padded
def speed_features(angles, fs, smooth_frames=5) -> np.ndarray         # (N,12) float64 deg/s
def noise_floor(angles, window=3) -> np.ndarray                       # (12,) float64 rad, >= 1e-4
def speed_envelope(angles, fs, smooth_frames=5, weights=None) -> np.ndarray   # (N,) float64 deg/s
def estimate_fs(ts, fs_hint=30.0) -> float                            # median-step rate
def pose33_to_angle_stream(pose33, ts, fs_hint=30.0) -> dict
    # {'angles': (N,12) float32 unwrapped rad, 'valid': (N,) bool, 'speed': (N,12) float32,
    #  'envelope': (N,) float32, 'noise_floor': (12,) float32, 'fs': float}
```

Feature order (fixed; downstream modules index by position):
`torso_lean, neck, head_twist, shoulder_line, l_thigh, r_thigh, l_upper_arm,
r_upper_arm, l_forearm, r_forearm, l_shin, r_shin`.

Semantics:

- Torso frame per frame: origin mid-hip, `u = unit(mid_shoulder − mid_hip)`,
  `r = perp(u) = (−u_y, u_x)`. Segment `v = p[to] − p[from]`, angle
  `arctan2(v·r, v·u)` in (−π, π]. `torso_lean = arctan2(u_x, −u_y)` against
  image vertical (image y points down; 0 = upright). Rotating the whole body
  by φ with `[[cos, −sin], [sin, cos]]` changes `torso_lean` by +φ and
  nothing else.
- Segment endpoints for the non-obvious features: `neck` = mid_shoulder →
  mid_ear; `head_twist` = R ear (8) → L ear (7) (the ear line: longer and steadier than mid_ear → nose); `shoulder_line` = L shoulder (11)
  → R shoulder (12). Virtual midpoints require **both** constituent
  landmarks visible.
- A feature is NaN when either endpoint has `vis <= 0.5`, a non-finite
  coordinate, or `|v| < MIN_SEGMENT_LEN`. The row is all-NaN and
  `valid=False` when hips/shoulders are hidden or the torso is shorter than
  `MIN_TORSO_LEN`; `valid[i]` is True iff the torso frame is defined AND at
  least 6 of 12 features are finite. Input not shaped `(N, 33, >=4)` raises
  `ValueError`.
- `unwrap_nan` unwraps the **compacted finite sequence** of each column
  (equivalent to minimal-jump interpolation across each gap), then restores
  NaN — this is what guarantees no jump > π between finite neighbours, even
  across a gap. It is not a literal linear interpolation of wrapped values.
- `moving_average_nan` is `scipy.ndimage.uniform_filter1d(mode='nearest')`
  on the finite-masked sum and count; for even `window` the centre sits half
  a sample early. `window <= 1` returns a float64 copy.
- `noise_floor` is the plan formula verbatim
  (`1.4826 · median |x − MA_w(x)|`, finite samples, floored at 1e-4). For
  white noise this reads ≈ 0.82 σ at `window=3` (the residual against a
  3-point mean has SD `σ·sqrt(2/3)`); the constant is retained as-is because
  it cancels in the noise normalisation and keeps the definition comparable
  across modules.
- `speed_features` = `|np.gradient(MA(unwrap(angles)))| · fs` in deg/s;
  `speed_envelope` renormalises the weights over the finite features per
  frame and is NaN where fewer than 3 features are finite. Because the
  moving average is NaN-ignoring by design, speed and envelope bleed up to
  `smooth_frames // 2` (= 2) frames into the edges of an invalid stretch;
  the gap interior is NaN. Consumers apply the `valid` mask themselves
  (`pose_ddtw` and `pose_event_coincidence` both do).
- dtypes: `speed_features`, `speed_envelope`, `noise_floor`, `unwrap_nan`,
  `moving_average_nan` return float64; `angle_features` returns float32
  angles; only `pose33_to_angle_stream` casts its bundle to float32.

### 4.2 `cadence/significance/block_bootstrap.py` — moving-block bootstrap

```python
DEFAULT_N_BOOT = 4000; DEFAULT_CI = 0.95; DEFAULT_TIE_BAND = 0.35

def block_len_from_seconds(seconds, fs) -> int                       # max(1, round(seconds*fs))
def block_bootstrap_mean(x, block_len, n_boot=4000, seed=0, weights=None, ci=0.95) -> dict
    # {'mean','ci_lo','ci_hi','se': (K,), 'boot': (n_boot,K), 'n_blocks', 'n_dropped'}
def block_bootstrap_paired(a, b, block_len, n_boot=4000, seed=0, ci=0.95) -> dict
    # {'mean_diff','mean_a','mean_b','ci_lo','ci_hi','se','p_greater': (K,),
    #  'boot': (n_boot,K) resampled differences, 'n_blocks', 'n_dropped'}
def block_bootstrap_rank(scores, block_len, n_boot=4000, seed=0, tie_band=0.35) -> list[dict]
    # rows: {'index','rank','mean','ci95': (lo,hi),'se','p_ranks_first','tied_with': list[int],
    #        'p_greater': (K,)}   sorted by (rank, -mean)
```

Semantics:

- Non-overlapping blocks of `block_len` samples, `n_blocks = T // block_len`,
  trailing remainder dropped and reported as `n_dropped`. The point `mean` is
  computed over the **block-covered samples only** so the estimate and the
  resamples agree; pick `block_len` so `n_dropped` is small.
- Shapes are always `(K,)` / `(n_boot, K)`: a 1-D `(T,)` input yields K=1,
  so `res['mean']` is shape `(1,)`, not a scalar — index `[0]`.
- NaN samples get weight 0 (down-weighted, never propagated). `T < block_len`
  (`n_blocks == 0`) returns NaN-filled results with `n_blocks=0` rather than
  raising — driver-friendly for short condition segments. `block_len < 1`,
  negative or non-finite weights, `ci` outside (0, 1), `tie_band` outside
  [0, 0.5) and shape mismatches raise `ValueError`.
- Resamples are represented as multinomial block-count vectors and combined
  with one `(n_boot, n_blocks) @ (n_blocks, K)` matmul — memory is
  O(n_boot · n_blocks), not O(n_boot · T). The paired and rank helpers draw
  a **single** count matrix shared across all columns.
- `block_bootstrap_paired` uses the **joint** finite mask (a sample counts
  only where both `a` and `b` are finite), so `mean_diff == mean_a − mean_b`
  exactly and `p_greater` is the fraction of resamples with a positive mean
  difference. Its point-estimate key is `mean_diff` (not `mean`).
- `block_bootstrap_rank`: higher score = better, rank 1 = highest mean.
  Column `j` beats `i` when `P(j>i) − 0.5 > tie_band`; competition rank =
  `1 + #{j beats i}` ("1224" style). `ci95` is a `(lo, hi)` tuple at fixed
  95 %; `p_greater[j] = P(i>j)` with NaN on the diagonal. Use `row['index']`
  to map back to the column; an all-NaN column gets NaN stats and rank K.

### 4.3 `cadence/significance/pose_ddtw.py` — feature modes, noise normalisation, path features

Everything Phase 0 is byte-for-byte unchanged for `feature_mode='pca'`
(verified by a monkeypatch test that `build_angle_features` is never
touched on the default path).

```python
FEATURE_MODES = ('pca', 'angles', 'angle_speed'); ANGLE_MODES = ('angles', 'angle_speed')
POSE_COLS = 4; POSE_FLAT_DIM_VIS = 132; NOISE_FLOOR_WINDOW = 3; SPEED_SMOOTH_FRAMES = 5

def build_angle_features(pose33_uniform, ts_uniform, mode, noise_normalize=True)
        -> tuple[np.ndarray, np.ndarray, dict]         # X (N,12) float64, valid (N,) bool, info
def resample_pose33_visibility(pose33, ts, ts_grid) -> np.ndarray    # (M,33) float32
def sliding_ddtw_path_features(p1, p2, window_frames, stride_frames, p1_valid, p2_valid,
                               fs, valid_frac=DDTW_VALID_FRAC) -> dict
    # {'lag_s','lag_var','asym': (n_strides,) float32, 'starts': (n_strides,) int64}
def compute_session_ddtw(..., compute_path_features=False, fs: float | None = None) -> dict
def run_session_ddtw(session_id, components: np.ndarray | None, mean: np.ndarray | None, *,
                     ..., feature_mode='pca', noise_normalize=True,
                     compute_path_features=False) -> dict
def load_pose_streams(preproc_path, include_pose33=False) -> dict
```

Semantics:

- `build_angle_features('angles')`: `angle_features` → `unwrap_nan` →
  noise floor measured on the **unfilled** unwrapped angles → NaN fill
  (interior gaps linear-interpolated between finite neighbours;
  leading/trailing NaN and all-NaN columns → 0) → divide each column by its
  floor when `noise_normalize`. A boundary step at the first finite value can
  be seen by early/late windows; the per-window `valid_frac` gate in
  `sliding_ddtw_real` is the only guard. `'angle_speed'`:
  `speed_features` (5-frame smoothing at the grid rate estimated from
  `ts_uniform`) divided by its own residual floor computed **without**
  unwrap (`_residual_noise_floor`), in deg/s.
- `info` carries `feature_mode`, `noise_floor` (12,) float64 (always the
  measured floor, even when `noise_normalize=False`), `noise_floor_units`
  (`'rad'` for angles, `'deg/s'` for angle_speed — i.e. in speed mode the
  floor is the **speed** floor, not radians), `noise_normalize`, `fs`
  (`estimate_fs(ts_uniform)`, fallback 12 Hz), `n_features`,
  `feature_names`, `n_valid`, `n_frames`. Returned `valid` is the angle
  validity only; `run_session_ddtw` ANDs it with `pose_features_valid`.
- **Visibility resampling** (deviation from the plan's two named options):
  `resample_pose33_visibility` takes, per grid sample, the **minimum**
  visibility of the two native frames that bracket it (`searchsorted`
  side='left', edge-clamped) — the same frames the linear coordinate
  interpolation blends. Any grid frame whose coordinates were blended with a
  visibility-zeroed frame is therefore masked; this is stricter than
  thresholding the interpolated visibility at 0.5. Coordinates go through the
  exact PCA two-step path (`resample_to_uniform_rate` on the flattened
  `(N, 132)` view → snap onto the common grid → reshape) so `stride_ts` and
  the common grid are identical across modes.
- `sliding_ddtw_path_features`: same window/stride/validity layout as
  `sliding_ddtw_real` so outputs align element-wise with `ddtw_z`. Windows
  are per-channel centred (as in `_ddtw_score`) before
  `dtw_ndim.warping_paths(use_c=True)` + `dtw.best_path`, so the path is the
  one behind the reported cost. Returns the NaN triple on a backend failure.
  The extra `'starts'` key holds stride start indices. Positive `lag_s` =
  P2 later than P1, confirmed empirically and matching
  `cadence/synchrony/features/dtw.py`.
- `compute_session_ddtw(..., fs=None)`: `None` resolves to
  `target_rate_hz`, so lag seconds stay correct if a caller changes the
  target rate without passing `fs` (plan wrote `fs: float =
  DDTW_TARGET_RATE_HZ`; explicit `fs` behaves identically). With
  `compute_path_features=True` the dict gains `ddtw_lag_s`, `ddtw_lag_var`,
  `ddtw_asym`.
- `run_session_ddtw`: raises `ValueError` for an unknown `feature_mode` and
  for `'pca'` with `components`/`mean` None. Output gains in **every** mode
  (additive; Phase 0 keys unchanged): `feature_mode`, `n_features` (12 for
  angle modes, `components.shape[0]` for pca), `noise_floor` — `(2, 12)`
  float32 stacked [P1, P2] (radians for angles, deg/s for angle_speed) or
  `None` for pca — and `noise_normalize` (`False` in pca).
- `load_pose_streams(include_pose33=True)` adds `'p1_pose33'` /
  `'p2_pose33'` `(N, 33, 4)` float32 to the dict; default output unchanged.

### 4.4 `cadence/significance/pose_event_coincidence.py` — landing / peak coincidence

```python
EVENT_KINDS = ('landing', 'peak'); PARTICIPANTS = ('p1', 'p2')
POSE_NPZ_KEYS = ('pose33', 'pose33_ts', 'pose_features_valid')
LAG_MATCH_MAX_S = 1.0; LAG_MIN_MATCHES = 5; MIN_ENVELOPE_SAMPLES = 3

def detect_landings(envelope, fs, min_gap_s=0.5, prominence_frac=0.12) -> (idx, depth)
def detect_movement_peaks(envelope, fs, quantile_threshold=0.70, min_sep_s=1.0) -> (idx, amp)
def events_from_pose_npz(pose_npz_data, participant, event_kind, **kw) -> np.ndarray   # (n,) float64 s
def compute_pose_event_coincidence(pose_npz_data, t_common, lsl_offset, *,
                                   event_kind='landing', fs_hint=30.0,
                                   n_surrogates=200, seed=42, tau_samples=1,
                                   min_gap_s=0.5, prominence_frac=0.12,
                                   quantile_threshold=0.70, min_sep_s=1.0,
                                   smooth_sigma_s=15.0) -> tuple[np.ndarray, dict]
```

Semantics:

- Envelope per participant = `pose33_to_angle_stream(...)['envelope']` with
  invalid frames → NaN, where invalid means `NOT (pose_features_valid AND
  angle-stream valid)` — both must hold for a frame's envelope sample to be
  used.
- `detect_landings`: `find_peaks` on `−envelope` with
  `distance = round(min_gap_s·fs)` and
  `prominence = prominence_frac · (p95 − p5)`; NaN samples are filled with
  the running max before negation so a gap is never a minimum, and minima
  whose immediate left/right neighbour is NaN (the rim of a dropout) are
  dropped as well. `depth` is the `find_peaks` prominence in deg/s.
- `detect_movement_peaks` mirrors `face_event_coincidence.detect_activity_peaks`
  (height = `np.nanquantile(env, 0.70)`, NaN filled with `nanmin − 1`);
  returns `(idx, amp)`.
- `events_from_pose_npz` returns times in the pose stream's **own** clock
  (`p{1,2}_pose33_ts[idx]`, not shifted by `ts[0]`), raises `ValueError` for
  a participant outside `('p1', 'p2')` and `KeyError` for a missing NPZ key
  (only `compute_*` returns the zeros/`missing_data` contract).
- `compute_pose_event_coincidence` raises `ValueError` for an unknown
  `event_kind` (checked before the missing-key contract); a missing NPZ key
  returns `zeros(len(t_common))` + `{'status': 'missing_data', 'missing':
  <key>}`, identical to the face channel. `smooth_sigma_s=0` returns the raw
  per-bin z without standardisation, same as the face channel.
- `info` keys: face-mirror `p{1,2}_total_peaks`, `p{1,2}_grid_density`,
  `mean_z_raw`, `std_z_raw`, `mean_raw_coinc`, `smooth_sigma_s`,
  `mean_z_smoothed`, `std_z_smoothed`, `mean_z`, `std_z`, `status`; plus
  `event_kind`, `p{1,2}_n_events`, `p{1,2}_event_rate_hz` (events per second
  of **valid** pose, falling back to the ts span if no valid frames),
  `p{1,2}_fs`, `p{1,2}_valid_frac`, `mean_signed_lag_s` (nearest-neighbour
  P2 − P1 lag within ±1 s over matched events; NaN if < 5 matches),
  `median_signed_lag_s`, `n_matched_events`, and `mean_z_raw_event_bins`
  (mean raw z over bins holding a P2 event — the bins where coincidence is
  actually tested).
- Note on the plan's test threshold "mean raw z > 1.0": that is unattainable
  as a whole-trace mean under `_coincidence_z` (bins outside P1's dilated
  events have z = 0 and non-matched dilated bins are negative, so the
  whole-trace mean is bounded by ≈ `sqrt(q(1−q)) <= 0.5`). The test asserts
  mean raw z > 1.0 over P2-event bins (measured 2.67), whole-trace mean > 0.1
  (measured 0.24), and coupled > null.

### 4.5 `scripts/_validate_pose_channels.py` and `scripts/_run_mvp_scaffold.py` (Task 5)

`_validate_pose_channels.py` — Phase 1 driver, delivered by Task 5 of the
same workflow (`docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md`
§Task 5 is the contract):

- CLI: `--modes` (default `angles angle_speed evt_landing evt_peak`; `pca`
  re-runs the Phase 0 baseline through the same harness), `--sessions`,
  `--all`, `--quick`, `--n-jobs`, `--max-test1-pairs`, `--max-test3-pairs`,
  `--out results/mvp/phase1_pose`.
- Imports (does not copy) `discover_sessions`, `load_and_resample`,
  `make_pseudo_dyad`, `inject_coupling_into_segment`,
  `piecewise_linear_warp`, `_auc_from_scores` and the mode-agnostic Test 2/3
  helpers from `scripts/_validate_pose_ddtw.py`; adds
  `load_and_resample_pose33(sid)` that keeps the `(N, 33, 4)` array on the
  same 12 Hz common grid so pseudo-dyads and κ-injection operate on the same
  objects for every mode. κ-injection for angle modes mixes the **pose33
  coordinates** (angles are recomputed from mixed skeletons); event modes use
  the same mixed skeleton.
- Per mode writes `results/mvp/phase1_pose/pose_<mode>_per_session.npz` with
  `{sid}__z`, `{sid}__stride_ts`, `{sid}__z_pw` (after
  `prewhiten_and_standardize` from `scripts/_run_rslds_scaffold_v8.py`,
  validity = finite) and, for DDTW modes with `compute_path_features=True`,
  `{sid}__lag_s`, `{sid}__asym`. Per-session condition summaries with
  block-bootstrap CIs go to `phase1_condition_summary.csv`
  (`block_bootstrap_mean` / `block_bootstrap_paired`, block = 10 s at 2 Hz).
- `phase1_report.md`: per-mode table of Tests 1–4 and the decision line
  ``- **Pose channel for MVP scaffold: `<mode>`**`` with `<mode>` ∈
  `{pose_baseline, pose_ddtw, pose_angles, pose_angle_speed,
  pose_evt_landing, pose_evt_peak}`: the passing mode with the largest Test 3
  margin; `pose_ddtw` if only Phase 0 passes; else `pose_baseline`. Never
  overwrites `results/mvp/phase0/phase0_report.md`. Runs without a GPU
  (joblib threading for angle features, `ProcessPoolExecutor(spawn)` for
  DDTW as in Phase 0).

`_run_mvp_scaffold.py`:

- `--pose-channel` choices become `auto, baseline, ddtw, angles,
  angle_speed, evt_landing, evt_peak`.
- `read_phase_decision()` reads `results/mvp/phase1_pose/phase1_report.md`
  first, then falls back to `phase0_report.md`, then `'pose_baseline'`, so
  `--pose-channel auto` is behaviour-identical to today while no Phase 1
  report exists.
- `load_pose_channel_for_session(sid, t_common, mode)` maps modes to NPZ
  paths (`phase0/pose_ddtw_per_session.npz` for `pose_ddtw`,
  `phase1_pose/pose_<m>_per_session.npz` otherwise, key `__z`);
  `load_phase0_ddtw_for_session` stays as a thin wrapper. `pose_source` is
  recorded in the sidecar as today.

## 5. Validation gates (unchanged from Phase 0, applied per mode)

| Test | Gate |
|------|------|
| 1 Semi-synthetic dose response (pseudo-dyad base) | Kendall τ perm p < 0.05 AND AUC ≥ 0.65 at κ = 0.4 |
| 2 Condition contrast conv vs meditation | \|Δz\| ≥ 0.5 AND (paired t p < 0.05 OR Wilcoxon p < 0.05) — magnitude test, direction reported (see `_validate_pose_ddtw.py` note on pose polarity) |
| 3 Pseudo-dyad null | real − pseudo ≥ 0.5 z |
| 4 Redundancy vs Phase 0 DDTW / V11 baseline | descriptive: Pearson r with block-bootstrap 95 % CI |

The **critical testing rule** applies: semi-synthetic tests use pseudo-dyads
(P1 from session A + P2 from session B) as the base signal so κ = 0 is a
true null (AUC ≈ 0.50).

Synthetic test coverage shipped with the code (`tests/significance/`, 56
tests, ≈ 5 s, all seeded, no data files):

| File | What it pins down |
|---|---|
| `test_pose_angles.py` | 12 finite features on an upright skeleton; translation / scale / proportion invariance; whole-body rotation moves only `torso_lean`; hidden wrist → one NaN, hidden hips → invalid row; `unwrap_nan` across a NaN gap; `noise_floor` recovers σ within 30 %; constant rotation speed within 5 % |
| `test_block_bootstrap.py` | i.i.d. CI coverage ≥ 90/100 seeded trials; AR(1) ρ = 0.9 block SE ≥ 2× i.i.d. SE; rank ties (0, 0, 1); NaN ignored, not propagated |
| `test_pose_ddtw_modes.py` | pseudo-dyad null mean z ∈ [−0.5, 0.5]; dose response κ ∈ {0, 0.4, 0.8} monotone with z(0.8) > 1; lag sign (0.5 s delay → median `ddtw_lag_s` > 0.2 s, swap flips); `feature_mode='pca'` never touches the new code; angle columns std ∈ [0.5, 50] after noise normalisation |
| `test_pose_event_coincidence.py` | independent pause schedules \|mean z\| < 0.5; shared schedule with P2 delayed 0.3 s → event-bin z > 1 and `mean_signed_lag_s` ≈ +0.3 ± 0.15; 5 rest periods → 5 landings within ±2 frames, NaN gap creates none; missing key → zeros + `missing_data` |

## 6. Production status and decision path

- The production K=3 fit (`results/mvp/hierarchical_evtcoinc_smooth/`) and
  the MVP `pose` slot are **unchanged**. `--pose-channel auto` keeps reading
  `phase0_report.md` (or defaulting to `pose_baseline`) until
  `results/mvp/phase1_pose/phase1_report.md` exists.
- Data-dependent runs (`python scripts/_validate_pose_channels.py --all`)
  happen on the lab machine; the workflow that produced this spec delivers
  code + synthetic tests only. Adopting a Phase 1 mode into the production
  scaffold is a separate, explicit decision after reviewing
  `phase1_report.md`.

## 7. Out of scope

- Changing the production K=3 fit or `results/mvp/hierarchical_evtcoinc_smooth/`.
- Multi-person tracking / person re-identification (not a CADENCE problem).
- 3D torso frames (revisit when stereo gaze hardware lands, V12).

## 8. Follow-ups

- **(a) YQP-Explorer `BodyPoseDTWAnalyzer` null calibration.**
  `YQP-Explorer/core/analysis/bodypose_dtw.py` reports raw windowed DTW
  distance on z-scored / jointly-normalised landmark positions with no
  surrogate or null model, so its values are not comparable across sessions
  or against chance. Port the Phase 1 substrate: torso-frame angles
  (`pose_angles`) as the DTW input, the measured noise floor for cost units,
  and a circular-shift / condition-block surrogate z as in
  `pose_ddtw.compute_session_ddtw`. The dance_sync clip-relative 0–100
  anchor is acceptable there (single-session display tool), unlike in
  CADENCE.
- **(b) YouQuantiPy real-time pose synchrony.** A streaming variant of the
  angle + landing-coincidence channel for the live recording platform
  (companion spec:
  `docs/superpowers/specs/2026-09-09-yqp-realtime-pose-sync-design.md`).
  Reuse `angle_features` / `speed_envelope` per frame; the surrogate z must
  be replaced by a running calibration (e.g. a rolling circular-shift
  reference or a fixed per-session warm-up null) since whole-session
  surrogates are unavailable online.
- Stereo (3D) torso frames and gaze-conditioned pose coupling once the V12
  hardware lands (`docs/v12_gaze_design.md`).
- EDA/SC remains the #1 hardware addition for therapy synchrony
  (`/literature`); pose Phase 1 does not change that priority.
