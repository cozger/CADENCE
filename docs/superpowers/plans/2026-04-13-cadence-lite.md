# CADENCE-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a literature-grounded, per-modality-only, no-state-inference dyadic-coupling pipeline (`cadence/lite/`) producing 7 coupling-z timecourses per session, mixed-effects per-condition statistics, and a Tier-2 validation battery (pseudo-dyad null + raw-data semi-synthetic κ-detection curves) suitable for review with a comp-neuro collaborator.

**Architecture:** Single linear pipeline, no cross-modal feedback. Per-modality coupling computation uses existing CADENCE infrastructure (`bl_wavelet.py` for CWT/coherence/surrogates, `_run_rslds_scaffold_v8.py` for pose multi-lag) where possible; new code is per-electrode EEG wavelet coherence, ECG HF/LF envelope coupling, and the RR-series narrowband injection module. Statistics are two protocol-specific mixed-effects models with 5 pre-registered contrasts each via `pymer4` (lme4) plus within-session permutation tests.

**Tech Stack:** Python 3.11 (MCCT conda env), PyTorch (GPU CWT), numpy, scipy, pymer4 (R/lme4 via rpy2), statsmodels (FDR + MixedLM fallback), joblib (parallel permutations), matplotlib. Reuses `cadence.significance.bl_wavelet`, `cadence.data.xdf_loader`, `cadence.data.preprocessors`, `cadence.surrogates`, and the V8.2 `pose_velocity_coupling` function.

**Spec:** `docs/superpowers/specs/2026-04-13-cadence-lite-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `cadence/lite/__init__.py` | Create | Package marker |
| `cadence/lite/config.py` | Create | 7-channel definitions, bands, ROIs, windows, AU subsets, electrode lists |
| `cadence/lite/surrogates.py` | Create | Per-bin circular-shift z-scoring helper (uniform across channels) |
| `cadence/lite/coupling/__init__.py` | Create | Package marker |
| `cadence/lite/coupling/pose_multilag.py` | Create | Multi-lag (±5 s) upper-body velocity coupling extracted from V8.2 |
| `cadence/lite/coupling/ecg_envelope.py` | Create | HF (0.15–0.4 Hz) + LF (0.04–0.15 Hz) bandpass→Hilbert→sliding cross-correlation on RR series |
| `cadence/lite/coupling/face_wavelet.py` | Create | Thin wrapper around `bl_wavelet.surrogate_coherence_z` for expression + speech bands |
| `cadence/lite/coupling/eeg_wavelet.py` | Create | Per-electrode (homotopic) wavelet coherence in θ + α bands; Stouffer-z aggregation |
| `cadence/lite/timecourses.py` | Create | Pipeline orchestration: load session → 7 channels @ 2 Hz → per-condition segmentation |
| `cadence/lite/stats/__init__.py` | Create | Package marker |
| `cadence/lite/stats/models.py` | Create | `pymer4` mixed-effects wrapper (with `statsmodels.MixedLM` fallback) |
| `cadence/lite/stats/contrasts.py` | Create | 5 pre-registered contrasts × 2 protocol models definitions |
| `cadence/lite/stats/permutation.py` | Create | Within-session label-shuffle permutation, joblib-parallelized |
| `cadence/lite/stats/fdr.py` | Create | Benjamini-Hochberg FDR wrapper across 70-test family |
| `cadence/lite/validation/__init__.py` | Create | Package marker |
| `cadence/lite/validation/synth_ecg.py` | Create | RR-series narrowband injection in HF/LF bands (NEW for ECG semi-synthetic) |
| `cadence/lite/validation/semisynthetic.py` | Create | κ-injection driver + AUC computation per channel |
| `cadence/lite/validation/pseudo_dyad.py` | Create | Cross-session round-robin pairing + full-pipeline rerun |
| `cadence/lite/visualization/__init__.py` | Create | Package marker |
| `cadence/lite/visualization/timeline.py` | Create | Per-channel per-condition coupling-z plots (7 × 6 grid per session) |
| `cadence/lite/visualization/contrast_summary.py` | Create | Forest plot of all 70 contrasts |
| `cadence/lite/visualization/kappa_curves.py` | Create | AUC-vs-κ per channel from semi-synthetic battery |
| `tests/lite/__init__.py` | Create | Test package marker |
| `tests/lite/conftest.py` | Create | Shared pytest fixtures (synthetic dyad data, session paths) |
| `tests/lite/test_config.py` | Create | Unit tests for channel definitions |
| `tests/lite/test_surrogates.py` | Create | Unit tests for circular-shift z-scoring |
| `tests/lite/coupling/test_pose_multilag.py` | Create | Unit tests for pose coupling |
| `tests/lite/coupling/test_ecg_envelope.py` | Create | Unit tests for HF/LF envelope coupling |
| `tests/lite/coupling/test_face_wavelet.py` | Create | Unit tests for face wavelet wrapper |
| `tests/lite/coupling/test_eeg_wavelet.py` | Create | Unit tests for per-electrode EEG coherence + Stouffer aggregation |
| `tests/lite/test_timecourses.py` | Create | Integration test on `y_06` |
| `tests/lite/stats/test_models.py` | Create | Mixed-effects fitter tests |
| `tests/lite/stats/test_contrasts.py` | Create | Contrast definitions tests |
| `tests/lite/stats/test_permutation.py` | Create | Permutation null tests |
| `tests/lite/stats/test_fdr.py` | Create | BH FDR tests |
| `tests/lite/validation/test_synth_ecg.py` | Create | RR injection tests (recover known coupling) |
| `tests/lite/validation/test_semisynthetic.py` | Create | κ-detection curve tests |
| `tests/lite/validation/test_pseudo_dyad.py` | Create | Pseudo-dyad pairing tests |
| `scripts/_run_lite_pipeline.py` | Create | Driver: --session/--all → timecourses + stats |
| `scripts/_run_lite_pseudo_dyad.py` | Create | Driver: pseudo-dyad null check |
| `scripts/_run_lite_semisynthetic.py` | Create | Driver: κ-detection battery |

---

## Phase 0 — Setup

### Task 1: Create package + test skeletons

**Files:**
- Create: `cadence/lite/__init__.py`, `cadence/lite/coupling/__init__.py`, `cadence/lite/stats/__init__.py`, `cadence/lite/validation/__init__.py`, `cadence/lite/visualization/__init__.py`
- Create: `tests/lite/__init__.py`, `tests/lite/coupling/__init__.py`, `tests/lite/stats/__init__.py`, `tests/lite/validation/__init__.py`
- Create: `tests/lite/conftest.py`

- [ ] **Step 1: Create empty package init files**

```python
# cadence/lite/__init__.py
"""CADENCE-Lite: literature-grounded barebones dyadic coupling pipeline.

See docs/superpowers/specs/2026-04-13-cadence-lite-design.md for design.
"""
```

Create the same file (just the docstring + comment about path) for each of:
`cadence/lite/coupling/__init__.py`, `cadence/lite/stats/__init__.py`,
`cadence/lite/validation/__init__.py`, `cadence/lite/visualization/__init__.py`,
each starting with a one-line module docstring.

- [ ] **Step 2: Create empty test init files**

```python
# tests/lite/__init__.py
```

Empty file is fine. Same for `tests/lite/coupling/__init__.py`, `tests/lite/stats/__init__.py`, `tests/lite/validation/__init__.py`.

- [ ] **Step 3: Create shared pytest fixtures**

```python
# tests/lite/conftest.py
"""Shared fixtures for cadence/lite/ tests."""

import os
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_dyad_eeg(rng):
    """Synthetic 60-second EEG dyad at 256 Hz, 14 channels.

    P1 has alpha (10 Hz) at frontal electrodes; P2 has the same
    alpha rhythm coupled at kappa=0.5 with phase lag.
    """
    fs = 256
    T = 60 * fs
    n_ch = 14
    t = np.arange(T) / fs

    # Pink-ish base noise per electrode
    base_p1 = rng.standard_normal((T, n_ch)) * 5.0
    base_p2 = rng.standard_normal((T, n_ch)) * 5.0

    # Coupled alpha at frontal electrodes (idx 0..3)
    alpha = np.sin(2 * np.pi * 10.0 * t)
    base_p1[:, :4] += 3.0 * alpha[:, None]
    base_p2[:, :4] += 3.0 * (0.5 * alpha + np.sqrt(1 - 0.25)
                              * rng.standard_normal((T, 4))[:, 0:4][:, 0:1])
    # Independent noise at non-frontal
    return {'p1': base_p1.astype(np.float32),
            'p2': base_p2.astype(np.float32),
            'fs': fs}


@pytest.fixture
def synthetic_dyad_rr(rng):
    """Synthetic 5-minute RR-interval dyad at 4 Hz.

    Both participants have HF (0.25 Hz) RSA modulation; P2's HF
    envelope is correlated with P1's at kappa=0.6.
    """
    fs = 4.0
    T = int(5 * 60 * fs)
    t = np.arange(T) / fs

    # Base RR ~ 800 ms with respiratory modulation
    p1_rr = 800 + 50 * np.sin(2 * np.pi * 0.25 * t) + 10 * rng.standard_normal(T)
    # P2 with coupled HF envelope
    coupled = 0.6 * np.sin(2 * np.pi * 0.25 * t)
    p2_rr = 800 + 50 * (coupled + np.sqrt(1 - 0.36) * np.sin(
        2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))) + 10 * rng.standard_normal(T)
    return {'p1_rr_ms': p1_rr.astype(np.float64),
            'p2_rr_ms': p2_rr.astype(np.float64),
            'fs': fs}


SESSION_CACHE_PATH = os.environ.get(
    'CADENCE_SESSION_CACHE',
    'C:/Users/optilab/desktop/MCCT/session_cache')


@pytest.fixture(scope='session')
def y06_session_path():
    """Path to y_06 session cache for integration tests."""
    p = os.path.join(SESSION_CACHE_PATH, 'y_06.npz')
    if not os.path.exists(p):
        pytest.skip(f"Integration test session not available at {p}")
    return p
```

- [ ] **Step 4: Verify pytest discovers tests**

Run: `cd C:/Users/optilab/desktop/CADENCE && python -m pytest tests/lite/ --collect-only -q`
Expected: collects 0 tests (no test functions yet) but no errors.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/__init__.py cadence/lite/coupling/__init__.py \
        cadence/lite/stats/__init__.py cadence/lite/validation/__init__.py \
        cadence/lite/visualization/__init__.py \
        tests/lite/__init__.py tests/lite/coupling/__init__.py \
        tests/lite/stats/__init__.py tests/lite/validation/__init__.py \
        tests/lite/conftest.py
git commit -m "feat(lite): scaffold cadence/lite package + test skeleton"
```

---

## Phase 1 — Configuration

### Task 2: Channel definitions module

**Files:**
- Create: `cadence/lite/config.py`
- Test: `tests/lite/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/test_config.py
"""Tests for cadence/lite/config.py — channel and band definitions."""

import pytest
from cadence.lite import config


def test_channel_names_are_seven():
    assert len(config.CHANNELS) == 7


def test_channel_keys_unique():
    names = [c['name'] for c in config.CHANNELS]
    assert len(set(names)) == 7


def test_each_channel_has_required_fields():
    required = {'name', 'modality', 'method', 'window_s', 'output_rate_hz'}
    for ch in config.CHANNELS:
        assert required.issubset(ch.keys()), f"{ch['name']} missing: {required - ch.keys()}"


def test_eeg_band_definitions():
    assert config.EEG_BAND_THETA == (4.0, 7.0)
    assert config.EEG_BAND_ALPHA == (8.0, 12.0)


def test_ecg_band_definitions():
    assert config.ECG_BAND_HF == (0.15, 0.4)
    assert config.ECG_BAND_LF == (0.04, 0.15)


def test_face_band_definitions():
    assert config.FACE_BAND_EXPRESSION == (0.5, 2.0)
    assert config.FACE_BAND_SPEECH == (2.0, 7.0)


def test_eeg_electrodes_count():
    assert len(config.EEG_ELECTRODES) == 14


def test_pose_constants():
    assert config.POSE_MAX_LAG_S == 5.0
    assert config.POSE_WINDOW_S == 60.0


def test_output_rate_uniform():
    for ch in config.CHANNELS:
        assert ch['output_rate_hz'] == 2.0, f"{ch['name']} not at 2 Hz"


def test_face_au_subsets():
    # AFFECT_AUS for expression band
    assert len(config.FACE_AFFECT_AUS) >= 8
    # SPEECH_AUS for speech band
    assert len(config.FACE_SPEECH_AUS) >= 2
    # No overlap
    assert set(config.FACE_AFFECT_AUS).isdisjoint(set(config.FACE_SPEECH_AUS))


def test_surrogate_count():
    assert config.N_SURROGATES == 200
```

- [ ] **Step 2: Run test — expect ImportError / collection failure**

Run: `python -m pytest tests/lite/test_config.py -v`
Expected: ImportError on `from cadence.lite import config`.

- [ ] **Step 3: Create `cadence/lite/config.py`**

```python
# cadence/lite/config.py
"""CADENCE-Lite channel, band, and parameter definitions.

See docs/superpowers/specs/2026-04-13-cadence-lite-design.md §4 for the
full per-channel method spec.
"""

# ── Frequency bands ─────────────────────────────────────────────────
EEG_BAND_THETA = (4.0, 7.0)
EEG_BAND_ALPHA = (8.0, 12.0)
ECG_BAND_LF = (0.04, 0.15)
ECG_BAND_HF = (0.15, 0.4)
FACE_BAND_EXPRESSION = (0.5, 2.0)
FACE_BAND_SPEECH = (2.0, 7.0)

# ── EEG electrode list (Emotiv EPOC montage, 14 channels) ──────────
EEG_ELECTRODES = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
]

# ── Face AU subsets (from cadence/significance/bl_wavelet.py) ──────
# AFFECT_AUS: smile / frown / brow muscles for emotional expression band
FACE_AFFECT_AUS = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]
# SPEECH_AUS: jaw / lip articulators for speech band (mouthLowerDown L/R + jaw)
# Extended from V7's [34, 35] to include jaw open + lip stretchers for fuller coverage
FACE_SPEECH_AUS = [25, 26, 34, 35, 36, 37]

# ── Pose constants (from V8.2) ─────────────────────────────────────
POSE_MAX_LAG_S = 5.0       # ±5 s lag bank
POSE_WINDOW_S = 60.0       # rolling window length
POSE_UPPER_BODY_IDX = list(range(11))   # first 11 joint groups

# ── Window sizes per channel ───────────────────────────────────────
ECG_WINDOW_S = 60.0        # 60 s sliding window for HF/LF envelope cross-corr
EEG_CWT_SMOOTH_S = 0.5     # Gaussian temporal smoothing for coherence
FACE_CWT_SMOOTH_S = 0.5    # matches V7
OUTPUT_RATE_HZ = 2.0       # uniform output rate for all 7 channels
OUTPUT_HOP_S = 1.0 / OUTPUT_RATE_HZ  # 0.5 s

# ── CWT params ─────────────────────────────────────────────────────
CWT_MORLET_OMEGA = 5.0
EEG_CWT_N_FREQS = 30
EEG_CWT_F_LO = 4.0
EEG_CWT_F_HI = 30.0

# ── Surrogate normalization ────────────────────────────────────────
N_SURROGATES = 200
SURROGATE_MIN_SHIFT_FRAC = 0.1   # min |shift| as fraction of segment length

# ── Stats ──────────────────────────────────────────────────────────
N_PERMUTATIONS = 1000
FDR_ALPHA = 0.05

# ── Conditions per protocol ────────────────────────────────────────
MEDITATION_CONDITIONS = ['base_EO', 'base_EC', 'conv_1',
                         'meditate_B', 'meditate_K', 'conv_2']
PE_CONDITIONS = ['base_EO', 'base_EC', 'conv_1',
                 'PE_1', 'PE_2', 'conv_2']

# ── Channel manifest ───────────────────────────────────────────────
# Each entry is the spec for one of the 7 channels.
CHANNELS = [
    {
        'name': 'eeg_alpha_coh',
        'modality': 'eeg',
        'method': 'wavelet_coherence',
        'band': EEG_BAND_ALPHA,
        'per_element': 'electrode',  # multi-element, Stouffer-aggregated
        'n_elements': 14,
        'window_s': None,             # CWT window is implicit (~5 cycles)
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
    {
        'name': 'eeg_theta_coh',
        'modality': 'eeg',
        'method': 'wavelet_coherence',
        'band': EEG_BAND_THETA,
        'per_element': 'electrode',
        'n_elements': 14,
        'window_s': None,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
    {
        'name': 'ecg_hf_env',
        'modality': 'ecg',
        'method': 'envelope_xcorr',
        'band': ECG_BAND_HF,
        'per_element': None,
        'n_elements': 1,
        'window_s': ECG_WINDOW_S,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': True,    # PNS/RSA synchrony has negative valence
    },
    {
        'name': 'ecg_lf_env',
        'modality': 'ecg',
        'method': 'envelope_xcorr',
        'band': ECG_BAND_LF,
        'per_element': None,
        'n_elements': 1,
        'window_s': ECG_WINDOW_S,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
    {
        'name': 'face_expression_coh',
        'modality': 'face',
        'method': 'wavelet_coherence',
        'band': FACE_BAND_EXPRESSION,
        'per_element': 'au',
        'n_elements': len(FACE_AFFECT_AUS),
        'window_s': None,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
    {
        'name': 'face_speech_coh',
        'modality': 'face',
        'method': 'wavelet_coherence',
        'band': FACE_BAND_SPEECH,
        'per_element': 'au',
        'n_elements': len(FACE_SPEECH_AUS),
        'window_s': None,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
    {
        'name': 'pose_multilag',
        'modality': 'pose',
        'method': 'multilag_xcorr',
        'band': None,
        'per_element': None,
        'n_elements': 1,
        'window_s': POSE_WINDOW_S,
        'output_rate_hz': OUTPUT_RATE_HZ,
        'literature_negative': False,
    },
]

CHANNEL_NAMES = [c['name'] for c in CHANNELS]
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/test_config.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/config.py tests/lite/test_config.py
git commit -m "feat(lite): channel + band + parameter definitions"
```

---

## Phase 2 — Surrogate Normalization

### Task 3: Per-bin circular-shift z-scoring helper

**Files:**
- Create: `cadence/lite/surrogates.py`
- Test: `tests/lite/test_surrogates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/test_surrogates.py
"""Tests for cadence/lite/surrogates.py — per-bin circular-shift z-scoring."""

import numpy as np
import pytest
from cadence.lite import surrogates


def _make_coupled_pair(rng, T=2000, kappa=0.5):
    """Return two 1-D timeseries with controlled coupling kappa."""
    base = rng.standard_normal(T)
    p1 = base + 0.3 * rng.standard_normal(T)
    p2 = kappa * base + np.sqrt(1 - kappa ** 2) * rng.standard_normal(T)
    return p1.astype(np.float32), p2.astype(np.float32)


def test_zscore_returns_array_of_correct_shape(rng):
    p1, p2 = _make_coupled_pair(rng, T=400)

    def coupling_fn(a, b):
        # Toy coupling: per-bin product, with bins of length 50
        bins = a.shape[0] // 50
        return (a[:bins * 50].reshape(bins, 50) * b[:bins * 50].reshape(bins, 50)).mean(1)

    result = surrogates.surrogate_z(p1, p2, coupling_fn, n_surrogates=50, seed=42)
    assert 'z' in result and 'real' in result and 'null_mean' in result and 'null_std' in result
    assert result['z'].shape == (8,)


def test_null_data_yields_z_near_zero(rng):
    # Two completely independent series — z should be near 0
    p1 = rng.standard_normal(2000).astype(np.float32)
    p2 = rng.standard_normal(2000).astype(np.float32)

    def coupling_fn(a, b):
        bins = a.shape[0] // 100
        return (a[:bins * 100].reshape(bins, 100) * b[:bins * 100].reshape(bins, 100)).mean(1)

    result = surrogates.surrogate_z(p1, p2, coupling_fn, n_surrogates=200, seed=42)
    assert np.abs(result['z']).mean() < 0.5  # mean |z| should be small


def test_coupled_data_yields_positive_z(rng):
    # Strongly coupled series — z should be reliably positive
    p1, p2 = _make_coupled_pair(rng, T=2000, kappa=0.8)

    def coupling_fn(a, b):
        bins = a.shape[0] // 100
        return (a[:bins * 100].reshape(bins, 100) * b[:bins * 100].reshape(bins, 100)).mean(1)

    result = surrogates.surrogate_z(p1, p2, coupling_fn, n_surrogates=200, seed=42)
    assert result['z'].mean() > 1.0  # mean z should be clearly above null


def test_seed_is_deterministic(rng):
    p1, p2 = _make_coupled_pair(rng, T=400)

    def coupling_fn(a, b):
        bins = a.shape[0] // 50
        return (a[:bins * 50].reshape(bins, 50) * b[:bins * 50].reshape(bins, 50)).mean(1)

    r1 = surrogates.surrogate_z(p1, p2, coupling_fn, n_surrogates=20, seed=123)
    r2 = surrogates.surrogate_z(p1, p2, coupling_fn, n_surrogates=20, seed=123)
    np.testing.assert_array_equal(r1['z'], r2['z'])


def test_min_shift_respects_fraction():
    # Ensure shifts are >= min_shift_frac * T
    p1 = np.zeros(1000, dtype=np.float32)
    p1[500] = 1.0
    p2 = np.zeros(1000, dtype=np.float32)
    p2[500] = 1.0

    def trivial(a, b):
        return np.array([(a * b).sum()])

    result = surrogates.surrogate_z(
        p1, p2, trivial, n_surrogates=200, seed=42,
        min_shift_frac=0.2)
    # All shifts should be >= 200; null should never include the trivial=1 from zero shift
    assert result['real'][0] == 1.0
    # Most surrogates should yield 0 (because shifted ones don't align)
    assert result['null_mean'][0] < 0.1
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/test_surrogates.py -v`
Expected: ImportError on `from cadence.lite import surrogates`.

- [ ] **Step 3: Create `cadence/lite/surrogates.py`**

```python
# cadence/lite/surrogates.py
"""Per-bin circular-shift z-scoring helper for cadence/lite/.

All 7 lite channels use the same null-model abstraction:
- the "real" coupling at each output bin is computed by some channel-specific
  function `coupling_fn(p1_signal, p2_signal) -> (n_bins,)`
- the null is built by circularly shifting P2 by a random offset, recomputing
  the same `coupling_fn`, accumulating per-bin mean / variance via Welford
- the per-bin z is `(real - null_mean) / null_std`

Channels with multi-element (per-electrode, per-AU) substrates compute a per-
element z first via this helper, then aggregate via Stouffer combination.
"""

import numpy as np


def surrogate_z(p1_signal, p2_signal, coupling_fn, n_surrogates=200,
                seed=42, min_shift_frac=0.1):
    """Per-bin z-score the real coupling against circular-shift surrogates.

    Args:
        p1_signal: 1-D or N-D numpy array. Time axis is axis 0.
        p2_signal: same shape as p1_signal. Will be circularly shifted.
        coupling_fn: callable (p1, p2) -> (n_bins,) coupling values per bin.
        n_surrogates: number of circular-shift surrogates (default 200).
        seed: int seed for reproducible shifts.
        min_shift_frac: minimum |shift| as fraction of T (default 0.1).

    Returns:
        dict with:
            'real': (n_bins,) coupling on real data
            'null_mean': (n_bins,) mean of surrogate coupling per bin
            'null_std': (n_bins,) std of surrogate coupling per bin
            'z': (n_bins,) per-bin z-score
            'shifts': (n_surrogates,) the random shifts used
    """
    T = p2_signal.shape[0]
    min_shift = max(1, int(min_shift_frac * T))
    max_shift = T - min_shift
    if min_shift >= max_shift:
        min_shift, max_shift = 1, T - 1

    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, max_shift + 1, size=n_surrogates)

    real = np.asarray(coupling_fn(p1_signal, p2_signal), dtype=np.float64)

    # Welford online mean/var accumulation
    n = 0
    mean = np.zeros_like(real)
    M2 = np.zeros_like(real)

    for shift in shifts:
        p2_shifted = np.roll(p2_signal, int(shift), axis=0)
        c = np.asarray(coupling_fn(p1_signal, p2_shifted), dtype=np.float64)
        n += 1
        delta = c - mean
        mean += delta / n
        M2 += delta * (c - mean)

    var = M2 / max(n - 1, 1)
    null_std = np.sqrt(var) + 1e-10
    z = (real - mean) / null_std

    return {
        'real': real.astype(np.float32),
        'null_mean': mean.astype(np.float32),
        'null_std': null_std.astype(np.float32),
        'z': z.astype(np.float32),
        'shifts': shifts.astype(np.int64),
    }


def stouffer_z(z_per_element, axis=0):
    """Combine per-element z-scores via Stouffer's method.

    Stouffer's z = sum(z_i) / sqrt(N), valid when per-element z's are roughly
    independent. For homotopic-electrode coupling z's, dependence is moderate
    and the combined value is conservatively interpretable.

    Args:
        z_per_element: ndarray; per-element z-scores along `axis`.
        axis: axis to combine over (default 0 = first axis).

    Returns:
        Combined z with `axis` removed.
    """
    z = np.asarray(z_per_element, dtype=np.float64)
    n = z.shape[axis]
    if n == 0:
        raise ValueError("stouffer_z called with zero elements")
    return (z.sum(axis=axis) / np.sqrt(n)).astype(np.float32)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/test_surrogates.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/surrogates.py tests/lite/test_surrogates.py
git commit -m "feat(lite): per-bin circular-shift z-scoring + Stouffer aggregation"
```

---

## Phase 3 — Per-Modality Coupling Modules

### Task 4: Pose multi-lag velocity coupling

**Files:**
- Create: `cadence/lite/coupling/pose_multilag.py`
- Test: `tests/lite/coupling/test_pose_multilag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/coupling/test_pose_multilag.py
"""Tests for cadence/lite/coupling/pose_multilag.py."""

import numpy as np
import pytest
from cadence.lite.coupling import pose_multilag


def test_pose_coupling_zero_for_independent_signals(rng):
    fs = 30.0
    T = int(120 * fs)
    p1_pos = rng.standard_normal((T, 11)) * 0.1
    p2_pos = rng.standard_normal((T, 11)) * 0.1
    t1 = np.arange(T) / fs
    t2 = np.arange(T) / fs
    t_common = np.arange(0, 110, 0.5)

    z = pose_multilag.compute(p1_pos, p2_pos, t1, t2, t_common)
    # Independent signals should yield z near 0 on average
    assert np.abs(z).mean() < 1.5
    assert z.shape == t_common.shape


def test_pose_coupling_positive_for_zero_lag_coupled(rng):
    fs = 30.0
    T = int(120 * fs)
    base = rng.standard_normal((T, 11)) * 0.1
    p1_pos = base + 0.05 * rng.standard_normal((T, 11))
    p2_pos = base + 0.05 * rng.standard_normal((T, 11))
    t1 = np.arange(T) / fs
    t2 = np.arange(T) / fs
    t_common = np.arange(0, 110, 0.5)

    z = pose_multilag.compute(p1_pos, p2_pos, t1, t2, t_common)
    # Highly-coupled positions → positive z on average
    assert z.mean() > 0.5


def test_pose_coupling_handles_lagged_coupling(rng):
    fs = 30.0
    T = int(120 * fs)
    base = rng.standard_normal((T, 11)) * 0.1
    p1_pos = base + 0.05 * rng.standard_normal((T, 11))
    # P2 leads P1 by 1 second (30 samples)
    p2_pos = np.roll(base, -30, axis=0) + 0.05 * rng.standard_normal((T, 11))
    t1 = np.arange(T) / fs
    t2 = np.arange(T) / fs
    t_common = np.arange(0, 110, 0.5)

    z = pose_multilag.compute(p1_pos, p2_pos, t1, t2, t_common)
    # Multi-lag bank should still detect lagged coupling
    assert z.mean() > 0.3
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/coupling/test_pose_multilag.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/coupling/pose_multilag.py`**

The implementation is a straight port of `pose_velocity_coupling()` from
`scripts/_run_rslds_scaffold_v8.py:301-404`, parameterized to take arrays
directly rather than the cached-session dict.

```python
# cadence/lite/coupling/pose_multilag.py
"""Multi-lag (±5 s) upper-body velocity coupling.

Method ported from V8.2 (`scripts/_run_rslds_scaffold_v8.py:301-404`) and
adapted to operate on direct array inputs rather than a cached-session dict.

The multi-lag bank detects both zero-lag synchrony (postural mirroring) AND
short-lag reciprocal coupling (head nods with 0.5-3 s response delay).
Validated in CADENCE V8.2: conv vs med p=0.006.
"""

import numpy as np

from cadence.lite import config


def compute(p1_pose, p2_pose, t_p1, t_p2, t_common,
            upper_body_idx=None, max_lag_s=None,
            n_surrogates=None, seed=42):
    """Compute multi-lag velocity coupling z-trace on a 2 Hz grid.

    Args:
        p1_pose: (T1, n_joints) joint position array for P1.
        p2_pose: (T2, n_joints) joint position array for P2.
        t_p1: (T1,) timestamps for p1_pose (seconds, common LSL clock).
        t_p2: (T2,) timestamps for p2_pose.
        t_common: (N,) target output grid (typically 2 Hz).
        upper_body_idx: which joint indices to use (default config.POSE_UPPER_BODY_IDX).
        max_lag_s: ± max lag in seconds (default config.POSE_MAX_LAG_S).
        n_surrogates: number of circular-shift surrogates (default config.N_SURROGATES).
        seed: RNG seed.

    Returns:
        (N,) float32 z-trace on t_common.
    """
    if upper_body_idx is None:
        upper_body_idx = config.POSE_UPPER_BODY_IDX
    if max_lag_s is None:
        max_lag_s = config.POSE_MAX_LAG_S
    if n_surrogates is None:
        n_surrogates = config.N_SURROGATES

    N = len(t_common)
    z_pose = np.zeros(N, dtype=np.float32)

    n_ch_avail = min(p1_pose.shape[1], p2_pose.shape[1], max(upper_body_idx) + 1)
    idx = [i for i in upper_body_idx if i < n_ch_avail]
    if len(idx) < 3:
        return z_pose

    # Interpolate to common 2 Hz grid
    fs_out = 1.0 / max(t_common[1] - t_common[0], 0.01)
    p1_on = np.column_stack([
        np.interp(t_common, t_p1, p1_pose[:, c], left=0, right=0) for c in idx])
    p2_on = np.column_stack([
        np.interp(t_common, t_p2, p2_pose[:, c], left=0, right=0) for c in idx])

    # First-difference (velocity)
    p1_v = np.diff(p1_on, axis=0, prepend=p1_on[:1])
    p2_v = np.diff(p2_on, axis=0, prepend=p2_on[:1])

    # Z-score per channel (NaN-safe)
    T, C = p1_v.shape
    for c in range(C):
        for arr in [p1_v, p2_v]:
            s = arr[:, c].std()
            if s > 1e-8:
                arr[:, c] = (arr[:, c] - arr[:, c].mean()) / s
            else:
                arr[:, c] = 0.0

    # Multi-lag cross-product bank
    max_lag_samp = int(max_lag_s * fs_out)
    lags = list(range(-max_lag_samp, max_lag_samp + 1))

    cp_per_lag = np.zeros((len(lags), T), dtype=np.float64)
    for li, lag in enumerate(lags):
        p2_shifted = np.roll(p2_v, lag, axis=0)
        if lag > 0:
            p2_shifted[:lag] = 0
        elif lag < 0:
            p2_shifted[lag:] = 0
        cp_per_lag[li] = (p1_v * p2_shifted).mean(axis=1)

    best_lag_idx = np.argmax(np.abs(cp_per_lag), axis=0)
    cp_best = cp_per_lag[best_lag_idx, np.arange(T)]

    # Surrogate z-score: shift P2 velocity, recompute max-lag cp
    rng = np.random.default_rng(seed)
    min_shift = max(1, int(0.1 * T))
    max_shift = max(min_shift + 1, int(0.9 * T))
    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    surr_cp = np.zeros((n_surrogates, T), dtype=np.float64)
    for si, shift in enumerate(shifts):
        p2_surr = np.roll(p2_v, int(shift), axis=0)
        best_surr = np.zeros(T, dtype=np.float64)
        for lag in lags:
            p2s = np.roll(p2_surr, lag, axis=0)
            if lag > 0:
                p2s[:lag] = 0
            elif lag < 0:
                p2s[lag:] = 0
            cp_lag = (p1_v * p2s).mean(axis=1)
            better = np.abs(cp_lag) > np.abs(best_surr)
            best_surr[better] = cp_lag[better]
        surr_cp[si] = best_surr

    null_mean = surr_cp.mean(axis=0)
    null_std = np.maximum(surr_cp.std(axis=0), 1e-10)
    z_pose = ((cp_best - null_mean) / null_std).astype(np.float32)
    return z_pose
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/coupling/test_pose_multilag.py -v`
Expected: 3 passed (will take ~30 s due to 200-surrogate × multi-lag inner loop on 240-sample test data).

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/coupling/pose_multilag.py tests/lite/coupling/test_pose_multilag.py
git commit -m "feat(lite): pose multi-lag velocity coupling (ported from V8.2)"
```

---

### Task 5: ECG HF/LF envelope coupling

**Files:**
- Create: `cadence/lite/coupling/ecg_envelope.py`
- Test: `tests/lite/coupling/test_ecg_envelope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/coupling/test_ecg_envelope.py
"""Tests for cadence/lite/coupling/ecg_envelope.py."""

import numpy as np
import pytest
from cadence.lite.coupling import ecg_envelope


def _make_rr_with_band_modulation(rng, T_s=300, fs=4.0, band='hf', kappa=0.0,
                                   seed=42):
    """Generate a paired RR series at fs Hz with controlled band coupling.

    band: 'hf' (~0.25 Hz) or 'lf' (~0.1 Hz).
    """
    T = int(T_s * fs)
    t = np.arange(T) / fs
    f0 = 0.25 if band == 'hf' else 0.1
    base = np.sin(2 * np.pi * f0 * t)

    rng = np.random.default_rng(seed)
    p1_rr = 800 + 50 * base + 50 * 0.3 * rng.standard_normal(T)

    # Coupled p2: same band, kappa-mixed envelope
    indep = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    p2_band = kappa * base + np.sqrt(1 - kappa ** 2) * indep
    p2_rr = 800 + 50 * p2_band + 50 * 0.3 * rng.standard_normal(T)
    return p1_rr.astype(np.float64), p2_rr.astype(np.float64)


def test_envelope_coupling_returns_correct_shape(rng):
    p1_rr, p2_rr = _make_rr_with_band_modulation(rng, T_s=200, kappa=0.0)
    fs = 4.0
    t1 = np.arange(len(p1_rr)) / fs
    t2 = np.arange(len(p2_rr)) / fs
    t_common = np.arange(60.0, 180.0, 0.5)  # 2 Hz grid, with 60s left margin for window

    z = ecg_envelope.compute(p1_rr, p2_rr, t1, t2, t_common,
                             band=(0.15, 0.4), window_s=60.0,
                             n_surrogates=100, seed=42)
    assert z.shape == t_common.shape


def test_hf_coupling_positive_when_coupled(rng):
    p1_rr, p2_rr = _make_rr_with_band_modulation(rng, T_s=400, band='hf', kappa=0.7)
    fs = 4.0
    t1 = np.arange(len(p1_rr)) / fs
    t2 = np.arange(len(p2_rr)) / fs
    t_common = np.arange(60.0, 380.0, 0.5)

    z = ecg_envelope.compute(p1_rr, p2_rr, t1, t2, t_common,
                             band=(0.15, 0.4), window_s=60.0,
                             n_surrogates=100, seed=42)
    assert np.nanmean(z) > 1.0  # coupled HF should yield clearly positive z


def test_hf_coupling_null_when_independent(rng):
    p1_rr, p2_rr = _make_rr_with_band_modulation(rng, T_s=400, band='hf', kappa=0.0)
    fs = 4.0
    t1 = np.arange(len(p1_rr)) / fs
    t2 = np.arange(len(p2_rr)) / fs
    t_common = np.arange(60.0, 380.0, 0.5)

    z = ecg_envelope.compute(p1_rr, p2_rr, t1, t2, t_common,
                             band=(0.15, 0.4), window_s=60.0,
                             n_surrogates=100, seed=42)
    assert np.abs(np.nanmean(z)) < 1.0  # independent → small mean z


def test_band_isolation_lf_does_not_detect_hf_coupling(rng):
    p1_rr, p2_rr = _make_rr_with_band_modulation(rng, T_s=400, band='hf', kappa=0.7)
    fs = 4.0
    t1 = np.arange(len(p1_rr)) / fs
    t2 = np.arange(len(p2_rr)) / fs
    t_common = np.arange(60.0, 380.0, 0.5)

    # Use LF band — should NOT pick up HF coupling
    z = ecg_envelope.compute(p1_rr, p2_rr, t1, t2, t_common,
                             band=(0.04, 0.15), window_s=60.0,
                             n_surrogates=100, seed=42)
    assert np.abs(np.nanmean(z)) < 1.0
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/coupling/test_ecg_envelope.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/coupling/ecg_envelope.py`**

```python
# cadence/lite/coupling/ecg_envelope.py
"""ECG HF / LF envelope coupling via bandpass + Hilbert envelope cross-corr.

Operates on RR-interval series at low rate (typically 4 Hz). Bandpass-filters
into HF (0.15-0.4 Hz, RSA / PNS proxy) or LF (0.04-0.15 Hz, mixed but SNS-leaning),
extracts the Hilbert amplitude envelope, computes Pearson correlation in a
sliding window, and z-scores against circular-shift surrogates of P2's envelope.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from cadence.lite import config
from cadence.lite.surrogates import surrogate_z


def _bandpass(x, fs, f_lo, f_hi, order=4):
    """Zero-phase Butterworth bandpass."""
    nyq = 0.5 * fs
    low = max(f_lo / nyq, 1e-6)
    high = min(f_hi / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def _envelope(x, fs, f_lo, f_hi):
    """Bandpass + Hilbert envelope (real)."""
    bp = _bandpass(x, fs, f_lo, f_hi)
    return np.abs(hilbert(bp))


def _sliding_xcorr(env1, env2, fs, window_s, hop_s):
    """Sliding-window Pearson correlation of two envelopes.

    Returns:
        coupling: (n_bins,) per-window Pearson r.
    """
    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    T = len(env1)
    if T < win:
        return np.array([])
    n_bins = (T - win) // hop + 1
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        a = env1[i * hop: i * hop + win]
        b = env2[i * hop: i * hop + win]
        sa = a.std()
        sb = b.std()
        if sa < 1e-10 or sb < 1e-10:
            out[i] = 0.0
        else:
            out[i] = np.dot(a - a.mean(), b - b.mean()) / (win * sa * sb)
    return out


def _resample_rr_to_uniform(rr_ms, t_rr, fs_out):
    """Resample RR-interval series at irregular times to uniform fs_out grid.

    Inputs:
        rr_ms: (T,) RR intervals (ms).
        t_rr: (T,) cumulative time stamps for each RR.
        fs_out: target uniform rate (Hz).

    Returns:
        ibi_uniform: (T_out,) RR-interval series sampled uniformly.
        t_uniform: (T_out,) time grid.
    """
    if len(t_rr) < 2:
        return np.array([]), np.array([])
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_out)
    ibi_uniform = np.interp(t_uniform, t_rr, rr_ms)
    return ibi_uniform, t_uniform


def compute(p1_rr_ms, p2_rr_ms, t_p1, t_p2, t_common,
            band, window_s=None, n_surrogates=None,
            fs_internal=4.0, hop_s=None, seed=42):
    """ECG envelope coupling z-trace on a target time grid.

    Args:
        p1_rr_ms, p2_rr_ms: RR-interval arrays (already at uniform fs_internal,
            OR irregular if t_p1/t_p2 are cumulative timestamps and don't match
            arange).
        t_p1, t_p2: timestamps for each RR sample (seconds, common LSL clock).
        t_common: target output grid (typically 2 Hz).
        band: tuple (f_lo, f_hi) — HF (0.15, 0.4) or LF (0.04, 0.15).
        window_s: sliding window length (default config.ECG_WINDOW_S = 60 s).
        n_surrogates: default config.N_SURROGATES = 200.
        fs_internal: internal uniform resampling rate (default 4 Hz).
        hop_s: sliding hop in seconds (default 1 / config.OUTPUT_RATE_HZ).
        seed: RNG seed.

    Returns:
        (N,) float32 z-trace interpolated onto t_common.
    """
    if window_s is None:
        window_s = config.ECG_WINDOW_S
    if n_surrogates is None:
        n_surrogates = config.N_SURROGATES
    if hop_s is None:
        hop_s = 1.0 / config.OUTPUT_RATE_HZ

    # Resample to uniform fs_internal grid
    p1_uniform, t1_uniform = _resample_rr_to_uniform(p1_rr_ms, t_p1, fs_internal)
    p2_uniform, t2_uniform = _resample_rr_to_uniform(p2_rr_ms, t_p2, fs_internal)
    if len(p1_uniform) == 0 or len(p2_uniform) == 0:
        return np.zeros(len(t_common), dtype=np.float32)

    # Align both to a common internal grid (intersection)
    t0 = max(t1_uniform[0], t2_uniform[0])
    t1 = min(t1_uniform[-1], t2_uniform[-1])
    if t1 <= t0:
        return np.zeros(len(t_common), dtype=np.float32)
    t_int = np.arange(t0, t1, 1.0 / fs_internal)
    p1_aligned = np.interp(t_int, t1_uniform, p1_uniform)
    p2_aligned = np.interp(t_int, t2_uniform, p2_uniform)

    # Bandpass + Hilbert envelopes
    env1 = _envelope(p1_aligned, fs_internal, *band)
    env2 = _envelope(p2_aligned, fs_internal, *band)

    # Coupling fn closure for surrogate_z
    def coupling_fn(a, b):
        return _sliding_xcorr(a, b, fs_internal, window_s, hop_s)

    result = surrogate_z(env1, env2, coupling_fn,
                         n_surrogates=n_surrogates, seed=seed)
    z_internal = result['z']

    # Map per-bin z back to t_common grid
    n_bins = len(z_internal)
    if n_bins == 0:
        return np.zeros(len(t_common), dtype=np.float32)
    bin_centers = t_int[0] + window_s / 2 + np.arange(n_bins) * hop_s
    z_on_t_common = np.interp(t_common, bin_centers, z_internal,
                              left=0.0, right=0.0)
    return z_on_t_common.astype(np.float32)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/coupling/test_ecg_envelope.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/coupling/ecg_envelope.py tests/lite/coupling/test_ecg_envelope.py
git commit -m "feat(lite): ECG HF/LF envelope coupling via Hilbert + sliding x-corr"
```

---

### Task 6: Face wavelet coherence (expression + speech bands)

**Files:**
- Create: `cadence/lite/coupling/face_wavelet.py`
- Test: `tests/lite/coupling/test_face_wavelet.py`

This is a thin wrapper around `cadence.significance.bl_wavelet.surrogate_coherence_z`
parameterized for two AU subsets and two frequency bands, exposing per-band
2 Hz coupling-z timecourses.

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/coupling/test_face_wavelet.py
"""Tests for cadence/lite/coupling/face_wavelet.py."""

import numpy as np
import pytest
from cadence.lite.coupling import face_wavelet


def _synthetic_au_pair(rng, T_s=60, fs=30.0, n_aus=52, kappa=0.0, band_hz=1.0):
    """Synthesize 60-s AU dyad with optional band-mixed coupling.

    band_hz: oscillation frequency injected into AFFECT_AUS (0.5-2 Hz expression
    if 1.0; 2-7 Hz speech if 4.0).
    """
    T = int(T_s * fs)
    t = np.arange(T) / fs
    base = np.sin(2 * np.pi * band_hz * t)
    noise1 = rng.standard_normal((T, n_aus)) * 0.1
    noise2 = rng.standard_normal((T, n_aus)) * 0.1
    p1 = noise1.copy()
    p2 = noise2.copy()
    affect_aus = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]
    p1[:, affect_aus] += base[:, None] * 0.5
    p2[:, affect_aus] += (kappa * base[:, None] +
                          np.sqrt(1 - kappa ** 2)
                          * np.sin(2 * np.pi * band_hz * t
                                   + rng.uniform(0, 2 * np.pi, len(affect_aus))
                                   )[:, None]) * 0.5
    return p1.astype(np.float32), p2.astype(np.float32), fs


def test_face_returns_z_per_band(rng):
    p1, p2, fs = _synthetic_au_pair(rng, T_s=60, kappa=0.0)
    t = np.arange(p1.shape[0]) / fs
    t_common = np.arange(0, 55, 0.5)

    z_expr, z_speech = face_wavelet.compute(p1, p2, t, t, t_common, fs=fs,
                                            n_surrogates=50)
    assert z_expr.shape == t_common.shape
    assert z_speech.shape == t_common.shape


def test_face_expression_detects_expression_band_coupling(rng):
    p1, p2, fs = _synthetic_au_pair(rng, T_s=60, kappa=0.7, band_hz=1.0)
    t = np.arange(p1.shape[0]) / fs
    t_common = np.arange(0, 55, 0.5)

    z_expr, z_speech = face_wavelet.compute(p1, p2, t, t, t_common, fs=fs,
                                            n_surrogates=100)
    # Expression-band z should be reliably positive
    assert np.nanmean(z_expr) > 1.0
    # Speech band should be much weaker
    assert np.nanmean(z_expr) > np.nanmean(z_speech)
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/coupling/test_face_wavelet.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/coupling/face_wavelet.py`**

```python
# cadence/lite/coupling/face_wavelet.py
"""Face wavelet coherence — wrapper around bl_wavelet.surrogate_coherence_z.

Computes per-AU wavelet coherence in the expression band (0.5-2 Hz, AFFECT_AUS)
and speech band (2-7 Hz, jaw/lip AUs) and aggregates to single 2 Hz z-traces.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from cadence.lite import config
from cadence.significance import bl_wavelet


def _lowpass(x, fs, cutoff_hz=8.0, order=4):
    """4th-order Butterworth low-pass to remove tracker noise above ~8 Hz."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, x, axis=0)


def compute(p1_au, p2_au, t_p1, t_p2, t_common, fs=30.0,
            n_surrogates=None, seed=42):
    """Face wavelet coherence z-traces for expression + speech bands.

    Args:
        p1_au, p2_au: (T, n_aus) AU activation arrays (already aligned in time).
        t_p1, t_p2: timestamps for each AU sample.
        t_common: target 2 Hz output grid.
        fs: AU sampling rate.
        n_surrogates: default config.N_SURROGATES.
        seed: RNG seed.

    Returns:
        (z_expr, z_speech): each (N,) float32 array on t_common.
    """
    if n_surrogates is None:
        n_surrogates = config.N_SURROGATES

    # Align P1/P2 to common internal grid (matching p1_au's native sampling)
    t0 = max(t_p1[0], t_p2[0])
    t1 = min(t_p1[-1], t_p2[-1])
    if t1 <= t0:
        N = len(t_common)
        return (np.zeros(N, dtype=np.float32),
                np.zeros(N, dtype=np.float32))
    t_int = np.arange(t0, t1, 1.0 / fs)
    p1_aligned = np.column_stack([
        np.interp(t_int, t_p1, p1_au[:, c]) for c in range(p1_au.shape[1])])
    p2_aligned = np.column_stack([
        np.interp(t_int, t_p2, p2_au[:, c]) for c in range(p2_au.shape[1])])

    # Low-pass to remove tracker noise above ~8 Hz
    p1_lp = _lowpass(p1_aligned, fs).astype(np.float32)
    p2_lp = _lowpass(p2_aligned, fs).astype(np.float32)

    # CWT via existing infrastructure
    scal_p1 = bl_wavelet.compute_au_cwt(p1_lp, fs=fs)
    scal_p2 = bl_wavelet.compute_au_cwt(p2_lp, fs=fs)

    def _band_zt(aus, band_hz):
        """Run surrogate_coherence_z for an AU subset and average across the
        target frequency band, then mean across AUs."""
        result = bl_wavelet.surrogate_coherence_z(
            scal_p1, scal_p2,
            n_surrogates=n_surrogates,
            smooth_s=config.FACE_CWT_SMOOTH_S,
            aus=aus, seed=seed,
        )
        z = result['z']  # (n_freqs, T_internal)
        freqs = scal_p1.freqs
        f_lo, f_hi = band_hz
        band_mask = (freqs >= f_lo) & (freqs < f_hi)
        if band_mask.sum() == 0:
            return np.zeros(z.shape[1], dtype=np.float32)
        z_band = z[band_mask].mean(axis=0)
        return z_band

    z_expr_internal = _band_zt(config.FACE_AFFECT_AUS, config.FACE_BAND_EXPRESSION)
    z_speech_internal = _band_zt(config.FACE_SPEECH_AUS, config.FACE_BAND_SPEECH)

    # Map internal-rate (fs) z back onto t_common (2 Hz)
    z_expr = np.interp(t_common, t_int[:len(z_expr_internal)],
                       z_expr_internal, left=0.0, right=0.0).astype(np.float32)
    z_speech = np.interp(t_common, t_int[:len(z_speech_internal)],
                         z_speech_internal, left=0.0, right=0.0).astype(np.float32)
    return z_expr, z_speech
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/coupling/test_face_wavelet.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/coupling/face_wavelet.py tests/lite/coupling/test_face_wavelet.py
git commit -m "feat(lite): face wavelet coherence wrapper for expression + speech bands"
```

---

### Task 7: EEG per-electrode wavelet coherence

**Files:**
- Create: `cadence/lite/coupling/eeg_wavelet.py`
- Test: `tests/lite/coupling/test_eeg_wavelet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/coupling/test_eeg_wavelet.py
"""Tests for cadence/lite/coupling/eeg_wavelet.py."""

import numpy as np
import pytest
from cadence.lite.coupling import eeg_wavelet


def _synthetic_eeg_dyad(rng, T_s=60, fs=256.0, n_ch=14, kappa=0.0, alpha_hz=10.0):
    """Synthesize 60-s EEG dyad with controlled alpha-band coupling at frontal idx."""
    T = int(T_s * fs)
    t = np.arange(T) / fs
    base = np.sin(2 * np.pi * alpha_hz * t)

    p1 = (rng.standard_normal((T, n_ch)) * 5.0).astype(np.float32)
    p2 = (rng.standard_normal((T, n_ch)) * 5.0).astype(np.float32)

    frontal = [0, 1, 2, 3]
    for c in frontal:
        p1[:, c] += 3.0 * base
        indep = np.sin(2 * np.pi * alpha_hz * t + rng.uniform(0, 2 * np.pi))
        p2[:, c] += 3.0 * (kappa * base + np.sqrt(1 - kappa ** 2) * indep)
    return p1, p2, fs


def test_returns_per_electrode_and_aggregated(rng):
    p1, p2, fs = _synthetic_eeg_dyad(rng, T_s=20, kappa=0.0)
    t = np.arange(p1.shape[0]) / fs
    t_common = np.arange(0, 18, 0.5)

    out = eeg_wavelet.compute(p1, p2, t, t, t_common, band=(8.0, 12.0),
                              fs=fs, n_surrogates=20)
    assert 'z_per_electrode' in out
    assert 'z_aggregated' in out
    assert out['z_per_electrode'].shape == (14, len(t_common))
    assert out['z_aggregated'].shape == t_common.shape


def test_alpha_coupling_appears_at_frontal_electrodes(rng):
    p1, p2, fs = _synthetic_eeg_dyad(rng, T_s=30, kappa=0.7)
    t = np.arange(p1.shape[0]) / fs
    t_common = np.arange(0, 25, 0.5)

    out = eeg_wavelet.compute(p1, p2, t, t, t_common, band=(8.0, 12.0),
                              fs=fs, n_surrogates=50)
    z_per = out['z_per_electrode']
    # Mean z at frontal electrodes (0..3) should be higher than at non-frontal
    frontal_mean = np.nanmean(z_per[:4])
    other_mean = np.nanmean(z_per[4:])
    assert frontal_mean > other_mean


def test_theta_band_does_not_pick_up_alpha_coupling(rng):
    p1, p2, fs = _synthetic_eeg_dyad(rng, T_s=30, kappa=0.7, alpha_hz=10.0)
    t = np.arange(p1.shape[0]) / fs
    t_common = np.arange(0, 25, 0.5)

    out_alpha = eeg_wavelet.compute(p1, p2, t, t, t_common, band=(8.0, 12.0),
                                    fs=fs, n_surrogates=50)
    out_theta = eeg_wavelet.compute(p1, p2, t, t, t_common, band=(4.0, 7.0),
                                    fs=fs, n_surrogates=50)
    # Frontal alpha-coupled signal should yield higher z in alpha than theta
    assert np.nanmean(out_alpha['z_per_electrode'][:4]) > np.nanmean(
        out_theta['z_per_electrode'][:4])
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/coupling/test_eeg_wavelet.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/coupling/eeg_wavelet.py`**

```python
# cadence/lite/coupling/eeg_wavelet.py
"""Per-electrode EEG wavelet coherence for theta and alpha bands.

Reuses the GPU CWT machinery in cadence.significance.bl_wavelet but at a
different frequency range (4-30 Hz, vs face's 0.3-8 Hz) and per-electrode
homotopic pairing rather than per-AU pairing.

For each electrode pair (P1 e_i ↔ P2 e_i), computes wavelet coherence,
averages in the target band, surrogate-z-scores against 200 circular shifts,
and resamples to 2 Hz. Returns both per-electrode and Stouffer-aggregated
z-traces.
"""

import numpy as np

from cadence.lite import config
from cadence.lite.surrogates import stouffer_z
from cadence.significance import bl_wavelet


def _eeg_freqs():
    """Log-spaced CWT frequencies covering EEG range (4-30 Hz)."""
    return np.logspace(
        np.log10(config.EEG_CWT_F_LO),
        np.log10(config.EEG_CWT_F_HI),
        config.EEG_CWT_N_FREQS,
    )


def compute(p1_eeg, p2_eeg, t_p1, t_p2, t_common, band, fs=256.0,
            n_surrogates=None, seed=42):
    """Per-electrode EEG wavelet coherence z-trace for one band.

    Args:
        p1_eeg, p2_eeg: (T, n_electrodes) EEG arrays.
        t_p1, t_p2: timestamps for each EEG sample (seconds, common LSL clock).
        t_common: target 2 Hz output grid.
        band: tuple (f_lo, f_hi) for the target band (theta or alpha).
        fs: EEG sampling rate (256 Hz default).
        n_surrogates: default config.N_SURROGATES.
        seed: RNG seed.

    Returns:
        dict with:
            'z_per_electrode': (n_electrodes, N) per-electrode coherence z.
            'z_aggregated': (N,) Stouffer-combined z across electrodes.
    """
    if n_surrogates is None:
        n_surrogates = config.N_SURROGATES

    n_ch = min(p1_eeg.shape[1], p2_eeg.shape[1])

    # Align both to common internal grid at fs
    t0 = max(t_p1[0], t_p2[0])
    t1 = min(t_p1[-1], t_p2[-1])
    if t1 <= t0 or n_ch == 0:
        N = len(t_common)
        return {
            'z_per_electrode': np.zeros((n_ch, N), dtype=np.float32),
            'z_aggregated': np.zeros(N, dtype=np.float32),
        }
    t_int = np.arange(t0, t1, 1.0 / fs)
    p1_aligned = np.column_stack([
        np.interp(t_int, t_p1, p1_eeg[:, c]) for c in range(n_ch)])
    p2_aligned = np.column_stack([
        np.interp(t_int, t_p2, p2_eeg[:, c]) for c in range(n_ch)])

    # CWT both participants, all electrodes (single GPU call for each)
    freqs = _eeg_freqs()
    scal_p1 = bl_wavelet.compute_au_cwt(p1_aligned.astype(np.float32),
                                         fs=fs, freqs=freqs)
    scal_p2 = bl_wavelet.compute_au_cwt(p2_aligned.astype(np.float32),
                                         fs=fs, freqs=freqs)

    f_lo, f_hi = band
    freq_mask = (freqs >= f_lo) & (freqs < f_hi)

    n_per = n_ch
    N = len(t_common)
    z_per_electrode = np.zeros((n_per, N), dtype=np.float32)

    # Per-electrode homotopic coherence + surrogate-z
    for e in range(n_per):
        result = bl_wavelet.surrogate_coherence_z(
            scal_p1, scal_p2,
            n_surrogates=n_surrogates,
            smooth_s=config.EEG_CWT_SMOOTH_S,
            aus=[e],   # single "AU" = single electrode index
            seed=seed + e,
        )
        z = result['z']  # (n_freqs, T_internal)
        if freq_mask.sum() == 0:
            z_band = np.zeros(z.shape[1], dtype=np.float32)
        else:
            z_band = z[freq_mask].mean(axis=0)

        # Resample to 2 Hz t_common
        z_per_electrode[e] = np.interp(
            t_common, t_int[:len(z_band)], z_band,
            left=0.0, right=0.0).astype(np.float32)

    z_agg = stouffer_z(z_per_electrode, axis=0)
    return {
        'z_per_electrode': z_per_electrode,
        'z_aggregated': z_agg,
    }
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/coupling/test_eeg_wavelet.py -v`
Expected: 3 passed (will take 1-3 minutes due to per-electrode CWT + surrogates).

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/coupling/eeg_wavelet.py tests/lite/coupling/test_eeg_wavelet.py
git commit -m "feat(lite): per-electrode EEG wavelet coherence + Stouffer aggregation"
```

---

## Phase 4 — Pipeline Orchestration

### Task 8: Per-session pipeline + per-condition segmentation

**Files:**
- Create: `cadence/lite/timecourses.py`
- Test: `tests/lite/test_timecourses.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/lite/test_timecourses.py
"""Integration tests for cadence/lite/timecourses.py."""

import numpy as np
import pytest
from cadence.lite import timecourses, config


def test_run_session_returns_seven_channels(y06_session_path):
    out = timecourses.run_session(y06_session_path, n_surrogates=20,
                                   skip_channels=None)
    assert set(out['channels'].keys()) == set(config.CHANNEL_NAMES)


def test_run_session_returns_per_condition_segments(y06_session_path):
    out = timecourses.run_session(y06_session_path, n_surrogates=20)
    # y_06 is a meditation-protocol session
    expected = set(config.MEDITATION_CONDITIONS)
    actual = set(out['condition_segments'].keys())
    # At least the conditions present in y_06 should appear
    assert actual.issubset(expected)
    assert 'base_EC' in actual  # baseline is always present


def test_per_condition_means_have_correct_shape(y06_session_path):
    out = timecourses.run_session(y06_session_path, n_surrogates=20)
    means = out['per_condition_mean_z']
    # means: dict[channel_name] -> dict[condition] -> float
    for ch in config.CHANNEL_NAMES:
        assert ch in means
        for cond in out['condition_segments']:
            assert isinstance(means[ch][cond], float)
```

- [ ] **Step 2: Run test — expect ImportError or skip**

Run: `python -m pytest tests/lite/test_timecourses.py -v`
Expected: ImportError on `from cadence.lite import timecourses`, OR skip if y_06 cache not present.

- [ ] **Step 3: Create `cadence/lite/timecourses.py`**

```python
# cadence/lite/timecourses.py
"""Per-session pipeline orchestration: 7 channels @ 2 Hz + per-condition segments.

Loads a cached session, computes the 7 lite coupling z-timecourses on a common
2 Hz grid, segments them by condition (using cadence.conditions), and returns
both timecourses and per-condition mean z values.
"""

import numpy as np

from cadence.lite import config
from cadence.lite.coupling import eeg_wavelet, ecg_envelope, face_wavelet, pose_multilag


def _load_session_cache(session_path):
    """Load a cached session NPZ + return a dict-like structure used downstream.

    Returns a dict containing:
        p1_eeg, p2_eeg: (T, 14) at 256 Hz
        p1_eeg_ts, p2_eeg_ts: timestamps in common LSL clock
        p1_rr_ms, p2_rr_ms, p1_rr_ts, p2_rr_ts (or similar — derived from ECG)
        p1_au, p2_au: (T, 52) at 30 Hz
        p1_au_ts, p2_au_ts
        p1_pose_features, p2_pose_features: (T, 41) at 12 Hz
        p1_pose_features_ts, p2_pose_features_ts
        markers, t_start_absolute, duration
    """
    data = dict(np.load(session_path, allow_pickle=True))
    # Some session caches store these directly; if not, derive RR from ECG.
    if 'p1_rr_ms' not in data and 'p1_ecg' in data:
        from cadence.data.eeg_features import detect_r_peaks_to_rr_ms
        # Fallback derivation; signature may vary by cache version.
        p1_rr_ms, p1_rr_ts = detect_r_peaks_to_rr_ms(
            data['p1_ecg'], data['p1_ecg_ts'])
        p2_rr_ms, p2_rr_ts = detect_r_peaks_to_rr_ms(
            data['p2_ecg'], data['p2_ecg_ts'])
        data['p1_rr_ms'] = p1_rr_ms
        data['p1_rr_ts'] = p1_rr_ts
        data['p2_rr_ms'] = p2_rr_ms
        data['p2_rr_ts'] = p2_rr_ts
    return data


def _build_common_grid(session, output_rate_hz=None):
    """Build a uniform 2 Hz time grid covering the session."""
    if output_rate_hz is None:
        output_rate_hz = config.OUTPUT_RATE_HZ
    t_start = float(session.get('t_start_absolute', 0))
    duration = float(session.get('duration', 0))
    if duration <= 0:
        # Fall back: use EEG length
        eeg = session['p1_eeg']
        ts = session['p1_eeg_ts']
        duration = ts[-1] - ts[0]
        t_start = ts[0]
    t_end = t_start + duration
    return np.arange(t_start, t_end, 1.0 / output_rate_hz)


def _detect_protocol(condition_keys):
    """Identify which protocol (meditation vs PE) the session belongs to."""
    if any(c.startswith('meditate') for c in condition_keys):
        return 'meditation'
    if any(c.startswith('PE') for c in condition_keys):
        return 'pe'
    return 'unknown'


def _segment_by_condition(t_common, intervals):
    """Map each timepoint in t_common to its condition (or None).

    Args:
        t_common: (N,) absolute time grid.
        intervals: list of (start_s, end_s, condition_key) — session-relative.

    Returns:
        dict[condition_key] -> boolean mask (N,).
    """
    if len(t_common) == 0:
        return {}
    t0 = t_common[0]
    rel = t_common - t0
    masks = {}
    for start, end, key in intervals:
        m = (rel >= start) & (rel < end)
        if m.sum() > 0:
            if key in masks:
                masks[key] |= m
            else:
                masks[key] = m
    return masks


def run_session(session_path, n_surrogates=None, skip_channels=None,
                output_rate_hz=None):
    """Run the full lite pipeline on one cached session.

    Args:
        session_path: path to cached .npz session file.
        n_surrogates: override (default config.N_SURROGATES).
        skip_channels: list of channel names to skip (default None — run all 7).
        output_rate_hz: target output rate (default config.OUTPUT_RATE_HZ).

    Returns:
        dict with:
            'channels': dict[channel_name] -> (N,) z-timecourse on t_common.
            'eeg_per_electrode': dict['eeg_alpha_coh'/'eeg_theta_coh'] -> (14, N).
            't_common': (N,) absolute time grid.
            'protocol': 'meditation' or 'pe' (or 'unknown').
            'condition_segments': dict[condition_key] -> (N,) bool mask.
            'per_condition_mean_z': dict[channel] -> dict[condition] -> float.
    """
    if n_surrogates is None:
        n_surrogates = config.N_SURROGATES
    if output_rate_hz is None:
        output_rate_hz = config.OUTPUT_RATE_HZ
    if skip_channels is None:
        skip_channels = set()
    skip_channels = set(skip_channels)

    session = _load_session_cache(session_path)
    t_common = _build_common_grid(session, output_rate_hz=output_rate_hz)

    channels = {}
    eeg_per_electrode = {}

    # ── EEG (per-electrode wavelet coherence, both bands) ────────────
    if 'eeg_alpha_coh' not in skip_channels or 'eeg_theta_coh' not in skip_channels:
        for cname, band in [
            ('eeg_alpha_coh', config.EEG_BAND_ALPHA),
            ('eeg_theta_coh', config.EEG_BAND_THETA),
        ]:
            if cname in skip_channels:
                continue
            out = eeg_wavelet.compute(
                session['p1_eeg'], session['p2_eeg'],
                session['p1_eeg_ts'], session['p2_eeg_ts'],
                t_common, band=band, n_surrogates=n_surrogates)
            channels[cname] = out['z_aggregated']
            eeg_per_electrode[cname] = out['z_per_electrode']

    # ── ECG HF/LF envelope coupling ─────────────────────────────────
    if 'ecg_hf_env' not in skip_channels:
        channels['ecg_hf_env'] = ecg_envelope.compute(
            session['p1_rr_ms'], session['p2_rr_ms'],
            session['p1_rr_ts'], session['p2_rr_ts'],
            t_common, band=config.ECG_BAND_HF,
            n_surrogates=n_surrogates)
    if 'ecg_lf_env' not in skip_channels:
        channels['ecg_lf_env'] = ecg_envelope.compute(
            session['p1_rr_ms'], session['p2_rr_ms'],
            session['p1_rr_ts'], session['p2_rr_ts'],
            t_common, band=config.ECG_BAND_LF,
            n_surrogates=n_surrogates)

    # ── Face wavelet coherence (expression + speech) ────────────────
    if ('face_expression_coh' not in skip_channels or
        'face_speech_coh' not in skip_channels):
        z_expr, z_speech = face_wavelet.compute(
            session['p1_au'], session['p2_au'],
            session['p1_au_ts'], session['p2_au_ts'],
            t_common, n_surrogates=n_surrogates)
        if 'face_expression_coh' not in skip_channels:
            channels['face_expression_coh'] = z_expr
        if 'face_speech_coh' not in skip_channels:
            channels['face_speech_coh'] = z_speech

    # ── Pose multi-lag velocity ─────────────────────────────────────
    if 'pose_multilag' not in skip_channels:
        channels['pose_multilag'] = pose_multilag.compute(
            session['p1_pose_features'], session['p2_pose_features'],
            session['p1_pose_features_ts'], session['p2_pose_features_ts'],
            t_common, n_surrogates=n_surrogates)

    # ── Condition segmentation ──────────────────────────────────────
    from cadence.conditions import parse_condition_intervals
    intervals = parse_condition_intervals(session)
    condition_segments = _segment_by_condition(t_common, intervals)
    protocol = _detect_protocol(condition_segments.keys())

    # ── Per-condition mean z per channel ────────────────────────────
    per_condition_mean_z = {}
    for ch_name, z in channels.items():
        per_condition_mean_z[ch_name] = {}
        for cond, mask in condition_segments.items():
            if mask.sum() > 0:
                per_condition_mean_z[ch_name][cond] = float(np.nanmean(z[mask]))
            else:
                per_condition_mean_z[ch_name][cond] = float('nan')

    return {
        'channels': channels,
        'eeg_per_electrode': eeg_per_electrode,
        't_common': t_common,
        'protocol': protocol,
        'condition_segments': condition_segments,
        'per_condition_mean_z': per_condition_mean_z,
    }
```

- [ ] **Step 4: Run test — expect PASS (or skip if y_06 not available)**

Run: `python -m pytest tests/lite/test_timecourses.py -v`
Expected: 3 passed, OR all skipped with "Integration test session not available".

If skipped, manually verify by adapting the `y06_session_path` fixture in `tests/lite/conftest.py` or setting `CADENCE_SESSION_CACHE` env var.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/timecourses.py tests/lite/test_timecourses.py
git commit -m "feat(lite): per-session pipeline orchestration (7 channels + condition segments)"
```

---

## Phase 5 — Statistics

### Task 9: pymer4 install + mixed-effects wrapper

**Files:**
- Create: `cadence/lite/stats/models.py`
- Test: `tests/lite/stats/__init__.py`, `tests/lite/stats/test_models.py`

- [ ] **Step 1: Verify R + lme4 + pymer4 install**

Run:
```bash
conda install -c conda-forge r-base r-lme4 -y
pip install pymer4
python -c "import pymer4; from pymer4.models import Lmer; print('pymer4 OK', pymer4.__version__)"
```

Expected: pymer4 imports cleanly. If installation fails on Windows, fall back to using `statsmodels.MixedLM` (the wrapper supports both).

- [ ] **Step 2: Create test stub**

```python
# tests/lite/stats/__init__.py  (already created in Task 1, just confirm)
```

```python
# tests/lite/stats/test_models.py
"""Tests for cadence/lite/stats/models.py mixed-effects wrappers."""

import numpy as np
import pandas as pd
import pytest
from cadence.lite.stats import models


@pytest.fixture
def synthetic_panel(rng):
    """Build a synthetic per-(session, condition) panel.

    8 sessions × 6 conditions × 14 electrodes; condition effect on coupling z.
    """
    rows = []
    for sess in range(8):
        sess_intercept = rng.normal(0, 0.3)
        for ci, cond in enumerate(['base_EO', 'base_EC', 'conv_1',
                                    'meditate_B', 'meditate_K', 'conv_2']):
            cond_effect = [0.0, 0.0, 0.6, 0.2, 0.1, 0.4][ci]
            for el in range(14):
                el_effect = rng.normal(0, 0.2)
                z = sess_intercept + cond_effect + el_effect + rng.normal(0, 0.5)
                rows.append({
                    'dyad': f'sess{sess}',
                    'condition': cond,
                    'electrode': f'e{el}',
                    'coupling_z': z,
                })
    return pd.DataFrame(rows)


def test_fit_with_electrode_random(synthetic_panel):
    out = models.fit_mixed(synthetic_panel,
                           formula='coupling_z ~ condition + (1|dyad) + (1|electrode)')
    assert 'coefs' in out
    # Should have 5 condition effects (vs reference base_EO)
    coef_names = [c for c in out['coefs'].index if 'condition' in c]
    assert len(coef_names) == 5


def test_fit_without_electrode_random_for_single_channel():
    # Build a single-channel panel (no electrode dim)
    rng = np.random.default_rng(0)
    rows = []
    for sess in range(8):
        si = rng.normal(0, 0.3)
        for ci, cond in enumerate(['base_EO', 'base_EC', 'conv_1',
                                    'meditate_B', 'meditate_K', 'conv_2']):
            ce = [0.0, 0.0, 0.5, 0.1, 0.0, 0.3][ci]
            rows.append({
                'dyad': f'sess{sess}',
                'condition': cond,
                'coupling_z': si + ce + rng.normal(0, 0.5),
            })
    panel = pd.DataFrame(rows)
    out = models.fit_mixed(panel, formula='coupling_z ~ condition + (1|dyad)')
    assert out['coefs'] is not None
```

- [ ] **Step 3: Create `cadence/lite/stats/models.py`**

```python
# cadence/lite/stats/models.py
"""Mixed-effects model wrapper.

Default: pymer4 (lme4 via rpy2). Backup: statsmodels.MixedLM.
The wrapper interface is intentionally narrow so swapping back-ends is one line.
"""

import warnings

import numpy as np
import pandas as pd

try:
    from pymer4.models import Lmer
    _HAVE_PYMER4 = True
except Exception:
    _HAVE_PYMER4 = False
    warnings.warn("pymer4 not available; falling back to statsmodels.MixedLM")

import statsmodels.formula.api as smf


def fit_mixed(panel, formula, backend='auto'):
    """Fit a mixed-effects model.

    Args:
        panel: pandas DataFrame with columns referenced by formula.
        formula: lme4-style formula string, e.g.
                 'coupling_z ~ condition + (1|dyad) + (1|electrode)'.
        backend: 'pymer4', 'statsmodels', or 'auto' (prefers pymer4).

    Returns:
        dict with:
            'coefs': pandas DataFrame indexed by coefficient name with
                     columns ['Estimate', 'SE', '2.5_ci', '97.5_ci',
                              'P-val', 'DF', 'T-stat'] (or backend equivalents).
            'model': the fitted model object (for inspection).
            'backend': 'pymer4' or 'statsmodels'.
    """
    use_pymer4 = (backend == 'pymer4' or
                  (backend == 'auto' and _HAVE_PYMER4))

    if use_pymer4:
        m = Lmer(formula, data=panel)
        m.fit(REML=True, summarize=False)
        coefs = m.coefs.copy()
        return {'coefs': coefs, 'model': m, 'backend': 'pymer4'}

    # ── statsmodels fallback ─────────────────────────────────────
    # Convert lme4 (1|group) syntax to statsmodels MixedLM groups
    # Only single-grouping models are supported by MixedLM directly.
    # For multi-grouping (e.g., (1|dyad)+(1|electrode)), use VC formula.
    fixed_part = formula.split('~')[1].split('+')
    re_groups = [p.strip() for p in fixed_part if '|' in p]
    fixed_part_clean = ' + '.join(p.strip() for p in fixed_part if '|' not in p)
    fixed_formula = formula.split('~')[0] + '~' + fixed_part_clean

    if len(re_groups) == 1:
        group = re_groups[0].strip().strip('()').split('|')[1].strip()
        m = smf.mixedlm(fixed_formula, data=panel, groups=panel[group]).fit(
            method='lbfgs', reml=True)
    elif len(re_groups) >= 2:
        # Use first as primary group, second as variance component
        primary = re_groups[0].strip().strip('()').split('|')[1].strip()
        vc_groups = [g.strip().strip('()').split('|')[1].strip()
                     for g in re_groups[1:]]
        vc_formula = {g: f'0 + C({g})' for g in vc_groups}
        m = smf.mixedlm(fixed_formula, data=panel, groups=panel[primary],
                        vc_formula=vc_formula).fit(method='lbfgs', reml=True)
    else:
        raise ValueError(f"No random effects in formula: {formula}")

    # Build a coefs DataFrame matching pymer4's column conventions
    summary = m.summary().tables[1]
    coefs = pd.DataFrame({
        'Estimate': m.params,
        'SE': m.bse,
        'P-val': m.pvalues,
    })
    return {'coefs': coefs, 'model': m, 'backend': 'statsmodels'}
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/stats/test_models.py -v`
Expected: 2 passed (with `pymer4` if available, otherwise `statsmodels` fallback).

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/stats/models.py tests/lite/stats/test_models.py
git commit -m "feat(lite): mixed-effects wrapper (pymer4 default, statsmodels fallback)"
```

---

### Task 10: Pre-registered contrasts

**Files:**
- Create: `cadence/lite/stats/contrasts.py`
- Test: `tests/lite/stats/test_contrasts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/stats/test_contrasts.py
"""Tests for cadence/lite/stats/contrasts.py."""

import pandas as pd
import numpy as np
import pytest
from cadence.lite.stats import contrasts, models


def test_contrasts_returns_five_per_protocol():
    c_med = contrasts.protocol_contrasts('meditation')
    c_pe = contrasts.protocol_contrasts('pe')
    assert len(c_med) == 5
    assert len(c_pe) == 5


def test_contrast_keys_are_pre_registered():
    c_med = contrasts.protocol_contrasts('meditation')
    keys = [c['id'] for c in c_med]
    assert keys == ['C1', 'C2', 'C3', 'C4', 'C5']


def test_apply_contrast_extracts_estimate_and_p(rng):
    # Build a panel where conv_1 - base_EC = 0.5
    rows = []
    for sess in range(10):
        si = rng.normal(0, 0.2)
        for cond, eff in [('base_EO', 0), ('base_EC', 0), ('conv_1', 0.5),
                           ('meditate_B', 0.1), ('meditate_K', 0.1),
                           ('conv_2', 0.3)]:
            rows.append({'dyad': f's{sess}', 'condition': cond,
                          'coupling_z': si + eff + rng.normal(0, 0.4)})
    panel = pd.DataFrame(rows)

    # Reorder factor: base_EC as reference for clean contrast
    panel['condition'] = pd.Categorical(
        panel['condition'],
        categories=['base_EC', 'base_EO', 'conv_1', 'meditate_B',
                    'meditate_K', 'conv_2'])

    fit = models.fit_mixed(panel,
                            formula='coupling_z ~ condition + (1|dyad)')
    c1 = {'id': 'C1', 'levels': {'conv_1': +1, 'base_EC': -1}}
    res = contrasts.apply_contrast(fit, c1)
    assert 'estimate' in res and 'p_value' in res
    # The contrast should be near 0.5
    assert 0.3 < res['estimate'] < 0.7
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/stats/test_contrasts.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/stats/contrasts.py`**

```python
# cadence/lite/stats/contrasts.py
"""Pre-registered contrasts for the two protocol-specific models.

Five contrasts × two models = 10 contrast specs total. These are applied
identically across the 7 channels (70 tests, FDR-corrected as a single family).
"""

import numpy as np


def protocol_contrasts(protocol):
    """Return the 5 pre-registered contrast specs for one protocol.

    Args:
        protocol: 'meditation' or 'pe'.

    Returns:
        list of contrast dicts with keys:
            'id': 'C1'..'C5'
            'name': human description
            'levels': dict {condition_name: weight} (weights sum to 0)
    """
    if protocol == 'meditation':
        i1, i2 = 'meditate_B', 'meditate_K'
    elif protocol == 'pe':
        i1, i2 = 'PE_1', 'PE_2'
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return [
        {'id': 'C1', 'name': 'conv_1 vs base_EC',
         'levels': {'conv_1': +1, 'base_EC': -1}},
        {'id': 'C2', 'name': f'{i1} vs base_EC',
         'levels': {i1: +1, 'base_EC': -1}},
        {'id': 'C3', 'name': f'{i2} vs base_EC',
         'levels': {i2: +1, 'base_EC': -1}},
        {'id': 'C4', 'name': 'conv_2 vs base_EC',
         'levels': {'conv_2': +1, 'base_EC': -1}},
        {'id': 'C5', 'name': 'conv_2 vs conv_1',
         'levels': {'conv_2': +1, 'conv_1': -1}},
    ]


def apply_contrast(fit, contrast):
    """Compute a linear combination of the model's fixed-effect coefficients.

    The model fits condition as a categorical factor. Each level appears in
    `coefs` as an offset relative to the reference level (or as the reference
    level itself with implicit estimate=0). This function constructs the
    requested linear combination, returning estimate + p-value.

    Args:
        fit: the dict returned by models.fit_mixed.
        contrast: dict with 'levels' = {condition_name: weight}.

    Returns:
        dict with 'estimate', 'se', 'p_value', 'cohens_d'.
    """
    coefs = fit['coefs']
    backend = fit.get('backend', 'pymer4')

    if backend == 'pymer4':
        from pymer4.models import Lmer
        m: Lmer = fit['model']
        # Use post-fit contrast machinery in pymer4
        levels_dict = contrast['levels']
        # Build the contrast vector aligned with model design matrix
        terms = list(m.coefs.index)
        c_vec = np.zeros(len(terms))
        # The reference level has implicit estimate=0
        ref_level = m.factors['condition'][0] if 'condition' in m.factors else None
        for cond_name, w in levels_dict.items():
            term_name = f'condition{cond_name}'
            if term_name in terms:
                c_vec[terms.index(term_name)] += w
            # else: this is the reference level, implicit weight handled by sum
        # Compute estimate, SE via vcov
        import scipy.stats as st
        vcov = m.fit_stats.get('vcov') if hasattr(m, 'fit_stats') else None
        if vcov is None:
            # Pull vcov via R if needed
            vcov = np.array(m.r_model.vcov())  # type: ignore
        beta = m.coefs['Estimate'].values
        est = float(c_vec @ beta)
        se = float(np.sqrt(c_vec @ vcov @ c_vec))
        df = m.coefs['DF'].mean() if 'DF' in m.coefs else len(coefs) * 10
        t_stat = est / max(se, 1e-12)
        p = 2 * (1 - st.t.cdf(abs(t_stat), df=df))
        d = est / max(se, 1e-12)  # crude standardized effect
        return {'estimate': est, 'se': se, 'p_value': float(p), 'cohens_d': float(d)}

    # statsmodels fallback
    import scipy.stats as st
    m = fit['model']
    levels_dict = contrast['levels']
    params = m.params
    cov = m.cov_params()
    # Build contrast vector aligned with params index
    c_vec = np.zeros(len(params))
    for cond_name, w in levels_dict.items():
        term = f'condition[T.{cond_name}]'
        if term in params.index:
            c_vec[list(params.index).index(term)] += w
    est = float(c_vec @ params.values)
    se = float(np.sqrt(c_vec @ cov.values @ c_vec))
    t_stat = est / max(se, 1e-12)
    df = max(len(m.fittedvalues) - len(params), 1)
    p = 2 * (1 - st.t.cdf(abs(t_stat), df=df))
    d = est / max(se, 1e-12)
    return {'estimate': est, 'se': se, 'p_value': float(p), 'cohens_d': float(d)}
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/stats/test_contrasts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/stats/contrasts.py tests/lite/stats/test_contrasts.py
git commit -m "feat(lite): 5 pre-registered contrasts × 2 protocol models"
```

---

### Task 11: Within-session permutation tests

**Files:**
- Create: `cadence/lite/stats/permutation.py`
- Test: `tests/lite/stats/test_permutation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/stats/test_permutation.py
"""Tests for cadence/lite/stats/permutation.py."""

import numpy as np
import pandas as pd
import pytest
from cadence.lite.stats import permutation


@pytest.fixture
def panel_with_effect(rng):
    rows = []
    for sess in range(8):
        si = rng.normal(0, 0.2)
        for cond, eff in [('base_EC', 0), ('conv_1', 0.7),
                           ('meditate_B', 0.1), ('meditate_K', 0.1),
                           ('base_EO', 0), ('conv_2', 0.4)]:
            rows.append({'dyad': f's{sess}', 'condition': cond,
                          'coupling_z': si + eff + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def test_permutation_p_for_real_effect_is_small(panel_with_effect):
    contrast = {'id': 'C1', 'levels': {'conv_1': +1, 'base_EC': -1}}
    p_perm = permutation.permutation_p(
        panel_with_effect, contrast,
        formula='coupling_z ~ condition + (1|dyad)',
        n_permutations=200, n_jobs=1, seed=42)
    assert p_perm < 0.05


def test_permutation_p_for_null_effect_is_large(rng):
    rows = []
    for sess in range(8):
        si = rng.normal(0, 0.2)
        for cond in ['base_EC', 'base_EO', 'conv_1', 'meditate_B',
                      'meditate_K', 'conv_2']:
            rows.append({'dyad': f's{sess}', 'condition': cond,
                          'coupling_z': si + rng.normal(0, 0.4)})
    panel = pd.DataFrame(rows)
    contrast = {'id': 'C1', 'levels': {'conv_1': +1, 'base_EC': -1}}
    p_perm = permutation.permutation_p(
        panel, contrast,
        formula='coupling_z ~ condition + (1|dyad)',
        n_permutations=200, n_jobs=1, seed=42)
    assert p_perm > 0.05
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/stats/test_permutation.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/stats/permutation.py`**

```python
# cadence/lite/stats/permutation.py
"""Within-session label-shuffle permutation tests.

For each contrast, shuffles the `condition` column independently within each
session. Refits the mixed-effects model and recomputes the contrast statistic,
yielding an exact non-parametric null distribution.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cadence.lite.stats import models, contrasts as contrasts_mod


def _shuffle_within_session(panel, seed):
    """Permute condition labels independently within each session (dyad).

    Args:
        panel: DataFrame with 'dyad' and 'condition' columns.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with shuffled 'condition' column (other columns unchanged).
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for dyad, group in out.groupby('dyad'):
        idx = group.index
        shuffled = rng.permutation(group['condition'].values)
        out.loc[idx, 'condition'] = shuffled
    return out


def _one_permutation(panel, formula, contrast, seed):
    """Single permutation: shuffle within-session, fit, compute contrast t-stat."""
    p_shuffled = _shuffle_within_session(panel, seed)
    fit = models.fit_mixed(p_shuffled, formula=formula)
    res = contrasts_mod.apply_contrast(fit, contrast)
    return res['estimate'] / max(res['se'], 1e-12)


def permutation_p(panel, contrast, formula, n_permutations=1000,
                  n_jobs=-1, seed=42):
    """Compute exact two-sided permutation p-value for a contrast.

    Args:
        panel: DataFrame.
        contrast: contrast dict from contrasts.protocol_contrasts.
        formula: model formula.
        n_permutations: number of within-session shuffles (default 1000).
        n_jobs: joblib parallel jobs (-1 = all cores).
        seed: master RNG seed.

    Returns:
        float two-sided permutation p-value.
    """
    fit = models.fit_mixed(panel, formula=formula)
    res = contrasts_mod.apply_contrast(fit, contrast)
    real_t = res['estimate'] / max(res['se'], 1e-12)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_permutations)
    null_t = Parallel(n_jobs=n_jobs)(
        delayed(_one_permutation)(panel, formula, contrast, int(s))
        for s in seeds
    )
    null_t = np.asarray(null_t)
    # Two-sided: fraction of |null| >= |real|
    p = float(np.mean(np.abs(null_t) >= np.abs(real_t)))
    return p
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/stats/test_permutation.py -v --timeout=300`
Expected: 2 passed (slow — 200 permutations × 2 tests, ~1-3 minutes total).

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/stats/permutation.py tests/lite/stats/test_permutation.py
git commit -m "feat(lite): within-session permutation tests for contrasts"
```

---

### Task 12: Benjamini-Hochberg FDR

**Files:**
- Create: `cadence/lite/stats/fdr.py`
- Test: `tests/lite/stats/test_fdr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/stats/test_fdr.py
"""Tests for cadence/lite/stats/fdr.py."""

import numpy as np
import pytest
from cadence.lite.stats import fdr


def test_bh_returns_qvalues_and_reject_mask():
    p_vals = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8]
    out = fdr.bh(p_vals, alpha=0.05)
    assert 'q_values' in out
    assert 'reject' in out
    assert len(out['q_values']) == len(p_vals)
    assert out['q_values'][0] <= out['q_values'][-1]  # monotonic in p


def test_bh_rejects_small_p_only():
    p_vals = [0.001, 0.5, 0.6, 0.9]
    out = fdr.bh(p_vals, alpha=0.05)
    assert out['reject'][0] == True
    assert out['reject'][-1] == False


def test_bh_handles_empty_input():
    with pytest.raises(ValueError):
        fdr.bh([], alpha=0.05)
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/stats/test_fdr.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/stats/fdr.py`**

```python
# cadence/lite/stats/fdr.py
"""Benjamini-Hochberg FDR correction wrapper around statsmodels."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def bh(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: iterable of p-values.
        alpha: target FDR level (default 0.05).

    Returns:
        dict with:
            'q_values': BH-adjusted q-values (same length as input).
            'reject': boolean array, True if rejected at FDR=alpha.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    if len(p) == 0:
        raise ValueError("bh: empty p-value list")
    reject, q_vals, _, _ = multipletests(p, alpha=alpha, method='fdr_bh')
    return {'q_values': q_vals, 'reject': reject}
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/stats/test_fdr.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/stats/fdr.py tests/lite/stats/test_fdr.py
git commit -m "feat(lite): BH-FDR correction wrapper"
```

---

## Phase 6 — Validation

### Task 13: RR-series narrowband injection (semi-synthetic ECG)

**Files:**
- Create: `cadence/lite/validation/synth_ecg.py`
- Test: `tests/lite/validation/test_synth_ecg.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/validation/test_synth_ecg.py
"""Tests for cadence/lite/validation/synth_ecg.py."""

import numpy as np
import pytest
from cadence.lite.validation import synth_ecg


def test_inject_returns_modified_p2(rng):
    fs = 4.0
    T = int(300 * fs)
    p1_rr = 800 + 50 * rng.standard_normal(T)
    p2_rr = 800 + 50 * rng.standard_normal(T)
    p1_ts = np.arange(T) / fs
    p2_ts = np.arange(T) / fs

    p2_inj = synth_ecg.inject_band_coupling(
        p1_rr, p2_rr, p1_ts, p2_ts,
        kappa=0.3, band=(0.15, 0.4), seed=42)
    assert p2_inj.shape == p2_rr.shape
    # Should differ from original
    assert not np.allclose(p2_inj, p2_rr)


def test_kappa_zero_returns_p2_unchanged(rng):
    fs = 4.0
    T = int(300 * fs)
    p1_rr = 800 + 50 * rng.standard_normal(T)
    p2_rr = 800 + 50 * rng.standard_normal(T)
    p1_ts = np.arange(T) / fs
    p2_ts = np.arange(T) / fs

    p2_inj = synth_ecg.inject_band_coupling(
        p1_rr, p2_rr, p1_ts, p2_ts,
        kappa=0.0, band=(0.15, 0.4), seed=42)
    np.testing.assert_allclose(p2_inj, p2_rr, atol=1e-6)


def test_injection_increases_band_coupling(rng):
    """Injecting kappa>0 should increase HF-band envelope coupling."""
    from cadence.lite.coupling.ecg_envelope import compute as ecg_compute
    fs = 4.0
    T = int(400 * fs)
    p1_rr = 800 + 50 * np.sin(2 * np.pi * 0.25 * np.arange(T) / fs) + 30 * rng.standard_normal(T)
    p2_rr_base = 800 + 30 * rng.standard_normal(T)
    p1_ts = np.arange(T) / fs
    p2_ts = np.arange(T) / fs
    t_common = np.arange(60.0, 380.0, 0.5)

    p2_inj = synth_ecg.inject_band_coupling(
        p1_rr, p2_rr_base, p1_ts, p2_ts,
        kappa=0.5, band=(0.15, 0.4), seed=42)
    z_base = ecg_compute(p1_rr, p2_rr_base, p1_ts, p2_ts, t_common,
                         band=(0.15, 0.4), n_surrogates=50)
    z_inj = ecg_compute(p1_rr, p2_inj, p1_ts, p2_ts, t_common,
                        band=(0.15, 0.4), n_surrogates=50)
    assert np.nanmean(z_inj) > np.nanmean(z_base) + 0.5
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/validation/test_synth_ecg.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/validation/synth_ecg.py`**

```python
# cadence/lite/validation/synth_ecg.py
"""Raw-data injection for ECG semi-synthetic validation.

Operates on the RR-interval series (most upstream level the lite ECG pipeline
consumes). For a target band (HF or LF), injects a kappa-weighted version of
P1's band-passed RR signal into P2's RR series, preserving total variance.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x, fs, f_lo, f_hi, order=4):
    nyq = 0.5 * fs
    low = max(f_lo / nyq, 1e-6)
    high = min(f_hi / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def _resample_to(rr_ms, t_rr, fs_target):
    if len(t_rr) < 2:
        return np.array([]), np.array([])
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_target)
    return np.interp(t_uniform, t_rr, rr_ms), t_uniform


def inject_band_coupling(p1_rr_ms, p2_rr_ms, t_p1, t_p2,
                         kappa, band, fs_internal=4.0, seed=42):
    """Inject narrowband coupling into P2's RR series.

    Args:
        p1_rr_ms, p2_rr_ms: RR-interval arrays (ms).
        t_p1, t_p2: timestamps.
        kappa: coupling strength in [0, 1]. 0 = no injection.
        band: tuple (f_lo, f_hi) for the band to inject (HF or LF).
        fs_internal: working rate (Hz).
        seed: RNG seed for any random component (currently unused in
              deterministic injection but kept for API consistency).

    Returns:
        p2_rr_inj: RR-series with band-coupled component added.
    """
    if kappa == 0:
        return p2_rr_ms.copy()

    # Resample P1 onto P2's clock grid
    p2_uniform, t2_uniform = _resample_to(p2_rr_ms, t_p2, fs_internal)
    p1_uniform = np.interp(t2_uniform, t_p1, p1_rr_ms)

    # Bandpass both
    p1_band = _bandpass(p1_uniform, fs_internal, *band)
    p2_band = _bandpass(p2_uniform, fs_internal, *band)
    p2_band_complement = p2_uniform - p2_band

    # Mix: replace P2's band with a kappa-weighted blend of P1's band + P2's band,
    # preserving the energy of P2's band component.
    # Standardize bands for clean kappa interpretation
    p1_band_std = (p1_band - p1_band.mean()) / max(p1_band.std(), 1e-10)
    p2_band_std = (p2_band - p2_band.mean()) / max(p2_band.std(), 1e-10)
    blend = kappa * p1_band_std + np.sqrt(max(1 - kappa ** 2, 0.0)) * p2_band_std
    blend_rescaled = blend * max(p2_band.std(), 1e-10) + p2_band.mean()

    p2_inj_uniform = p2_band_complement + blend_rescaled

    # Map back to original (possibly irregular) p2 timestamps
    p2_inj = np.interp(t_p2, t2_uniform, p2_inj_uniform)
    return p2_inj.astype(np.float64)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/validation/test_synth_ecg.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/validation/synth_ecg.py tests/lite/validation/test_synth_ecg.py
git commit -m "feat(lite): RR-series narrowband injection for ECG semi-synthetic"
```

---

### Task 14: Semi-synthetic battery driver

**Files:**
- Create: `cadence/lite/validation/semisynthetic.py`
- Test: `tests/lite/validation/test_semisynthetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/validation/test_semisynthetic.py
"""Tests for cadence/lite/validation/semisynthetic.py."""

import numpy as np
import pandas as pd
import pytest
from cadence.lite.validation import semisynthetic


def test_kappa_curve_keys():
    rng = np.random.default_rng(0)
    fs = 4.0
    T = int(300 * fs)
    base = {
        'p1_rr_ms': 800 + 50 * np.sin(2 * np.pi * 0.25 * np.arange(T) / fs)
                       + 20 * rng.standard_normal(T),
        'p2_rr_ms': 800 + 20 * rng.standard_normal(T),
        'p1_rr_ts': np.arange(T) / fs,
        'p2_rr_ts': np.arange(T) / fs,
    }

    out = semisynthetic.run_kappa_battery(
        base, channel='ecg_hf_env',
        kappas=[0.0, 0.2, 0.4],
        n_surrogates=20)
    assert 'kappa' in out and 'auc' in out
    assert len(out['kappa']) == 3


def test_auc_increases_with_kappa(tmp_path):
    rng = np.random.default_rng(0)
    fs = 4.0
    T = int(400 * fs)
    base = {
        'p1_rr_ms': 800 + 50 * np.sin(2 * np.pi * 0.25 * np.arange(T) / fs)
                       + 20 * rng.standard_normal(T),
        'p2_rr_ms': 800 + 20 * rng.standard_normal(T),
        'p1_rr_ts': np.arange(T) / fs,
        'p2_rr_ts': np.arange(T) / fs,
    }
    out = semisynthetic.run_kappa_battery(
        base, channel='ecg_hf_env',
        kappas=[0.0, 0.4],
        n_surrogates=20)
    assert out['auc'][1] > out['auc'][0]
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/validation/test_semisynthetic.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/validation/semisynthetic.py`**

```python
# cadence/lite/validation/semisynthetic.py
"""Semi-synthetic κ-detection battery driver.

For each channel and each kappa value, injects coupling at the raw signal
level, runs the channel's coupling computation, and computes detection AUC
against null (kappa=0) baselines.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from cadence.lite import config
from cadence.lite.coupling import ecg_envelope, eeg_wavelet, face_wavelet, pose_multilag
from cadence.lite.validation import synth_ecg


def _inject_for_channel(base, channel, kappa, seed=42):
    """Inject channel-appropriate coupling at the raw level.

    For now: ECG HF/LF use synth_ecg.inject_band_coupling. EEG, face, pose
    use the existing cadence/synthetic_v82.py infrastructure (not duplicated
    here; see notes).

    Args:
        base: dict with raw signal arrays + timestamps for the relevant modality.
        channel: channel name ('ecg_hf_env', 'eeg_alpha_coh', etc.).
        kappa: injection strength.
        seed: RNG seed.

    Returns:
        modified `base` dict (only P2 modified).
    """
    out = dict(base)
    if channel == 'ecg_hf_env':
        out['p2_rr_ms'] = synth_ecg.inject_band_coupling(
            base['p1_rr_ms'], base['p2_rr_ms'],
            base['p1_rr_ts'], base['p2_rr_ts'],
            kappa=kappa, band=config.ECG_BAND_HF, seed=seed)
    elif channel == 'ecg_lf_env':
        out['p2_rr_ms'] = synth_ecg.inject_band_coupling(
            base['p1_rr_ms'], base['p2_rr_ms'],
            base['p1_rr_ts'], base['p2_rr_ts'],
            kappa=kappa, band=config.ECG_BAND_LF, seed=seed)
    elif channel.startswith('eeg_'):
        from cadence.synthetic_v82 import inject_eeg_band_coupling
        band = config.EEG_BAND_ALPHA if channel == 'eeg_alpha_coh' else config.EEG_BAND_THETA
        out['p2_eeg'] = inject_eeg_band_coupling(
            base['p1_eeg'], base['p2_eeg'],
            kappa=kappa, band=band, fs=256.0, seed=seed)
    elif channel == 'face_expression_coh':
        from cadence.synthetic_v82 import inject_face_band_coupling
        out['p2_au'] = inject_face_band_coupling(
            base['p1_au'], base['p2_au'],
            kappa=kappa, band=config.FACE_BAND_EXPRESSION,
            aus=config.FACE_AFFECT_AUS, fs=30.0, seed=seed)
    elif channel == 'face_speech_coh':
        from cadence.synthetic_v82 import inject_face_band_coupling
        out['p2_au'] = inject_face_band_coupling(
            base['p1_au'], base['p2_au'],
            kappa=kappa, band=config.FACE_BAND_SPEECH,
            aus=config.FACE_SPEECH_AUS, fs=30.0, seed=seed)
    elif channel == 'pose_multilag':
        from cadence.synthetic_v82 import inject_pose_position_coupling
        out['p2_pose_features'] = inject_pose_position_coupling(
            base['p1_pose_features'], base['p2_pose_features'],
            kappa=kappa, seed=seed)
    else:
        raise ValueError(f"Unknown channel for injection: {channel}")
    return out


def _compute_z_for_channel(base, channel, n_surrogates):
    """Run the channel's coupling computation on `base`, return mean z."""
    if channel == 'ecg_hf_env':
        t_common = np.arange(60.0, base['p1_rr_ts'][-1] - 5.0, 0.5)
        z = ecg_envelope.compute(
            base['p1_rr_ms'], base['p2_rr_ms'],
            base['p1_rr_ts'], base['p2_rr_ts'], t_common,
            band=config.ECG_BAND_HF, n_surrogates=n_surrogates)
        return float(np.nanmean(z))
    if channel == 'ecg_lf_env':
        t_common = np.arange(60.0, base['p1_rr_ts'][-1] - 5.0, 0.5)
        z = ecg_envelope.compute(
            base['p1_rr_ms'], base['p2_rr_ms'],
            base['p1_rr_ts'], base['p2_rr_ts'], t_common,
            band=config.ECG_BAND_LF, n_surrogates=n_surrogates)
        return float(np.nanmean(z))
    if channel.startswith('eeg_'):
        band = config.EEG_BAND_ALPHA if channel == 'eeg_alpha_coh' else config.EEG_BAND_THETA
        t_eeg = base.get('p1_eeg_ts',
                         np.arange(base['p1_eeg'].shape[0]) / 256.0)
        t_common = np.arange(t_eeg[0] + 1.0, t_eeg[-1] - 1.0, 0.5)
        z_out = eeg_wavelet.compute(
            base['p1_eeg'], base['p2_eeg'],
            t_eeg, t_eeg, t_common,
            band=band, n_surrogates=n_surrogates)
        return float(np.nanmean(z_out['z_aggregated']))
    if channel.startswith('face_'):
        t_au = base.get('p1_au_ts',
                        np.arange(base['p1_au'].shape[0]) / 30.0)
        t_common = np.arange(t_au[0] + 1.0, t_au[-1] - 1.0, 0.5)
        z_expr, z_speech = face_wavelet.compute(
            base['p1_au'], base['p2_au'],
            t_au, t_au, t_common, n_surrogates=n_surrogates)
        return float(np.nanmean(z_expr if channel == 'face_expression_coh' else z_speech))
    if channel == 'pose_multilag':
        t_pose = base.get('p1_pose_features_ts',
                          np.arange(base['p1_pose_features'].shape[0]) / 12.0)
        t_common = np.arange(t_pose[0] + 1.0, t_pose[-1] - 1.0, 0.5)
        z = pose_multilag.compute(
            base['p1_pose_features'], base['p2_pose_features'],
            t_pose, t_pose, t_common, n_surrogates=n_surrogates)
        return float(np.nanmean(z))
    raise ValueError(f"Unknown channel: {channel}")


def run_kappa_battery(base_pseudodyad, channel, kappas, n_surrogates=200,
                      n_repeats=1, seed=42):
    """Run the κ-detection battery for one channel on a pseudo-dyad base.

    Args:
        base_pseudodyad: dict with raw modality arrays + timestamps for ONE
            pseudo-dyad (P1 from session A, P2 from session B). Channel-specific
            keys required (e.g., 'p1_rr_ms', 'p2_rr_ms', 'p1_rr_ts', 'p2_rr_ts'
            for ECG; 'p1_eeg', 'p2_eeg' etc. for EEG).
        channel: one of config.CHANNEL_NAMES.
        kappas: list of kappa values to test (e.g., [0.0, 0.1, 0.2, 0.3, 0.4]).
        n_surrogates: surrogate count for each pipeline run.
        n_repeats: how many independent injections per kappa (for AUC stability).
        seed: master seed.

    Returns:
        dict with:
            'kappa': list of kappas.
            'mean_z_per_kappa': list of mean coupling z (one per kappa, averaged
                                across n_repeats).
            'auc': list of detection AUC vs kappa=0 baseline (one per kappa>0;
                   for kappa=0 entry, AUC=0.5 by definition).
    """
    rng = np.random.default_rng(seed)
    z_by_kappa = {k: [] for k in kappas}
    for _ in range(n_repeats):
        for k in kappas:
            inj_base = _inject_for_channel(
                base_pseudodyad, channel, k, seed=int(rng.integers(2 ** 31 - 1)))
            z = _compute_z_for_channel(inj_base, channel, n_surrogates)
            z_by_kappa[k].append(z)

    mean_z = [float(np.mean(z_by_kappa[k])) for k in kappas]

    # AUC: classify "kappa>0 timepoint mean z" vs "kappa=0 timepoint mean z"
    # Use distribution of repeat-level mean z's
    null_zs = z_by_kappa[0.0] if 0.0 in z_by_kappa else z_by_kappa[kappas[0]]
    auc = []
    for k in kappas:
        if k == 0.0 or len(null_zs) < 2:
            auc.append(0.5)
            continue
        labels = [0] * len(null_zs) + [1] * len(z_by_kappa[k])
        scores = list(null_zs) + list(z_by_kappa[k])
        try:
            auc.append(float(roc_auc_score(labels, scores)))
        except ValueError:
            auc.append(0.5)

    return {
        'kappa': list(kappas),
        'mean_z_per_kappa': mean_z,
        'auc': auc,
    }
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/validation/test_semisynthetic.py -v --timeout=300`
Expected: 2 passed (slow due to repeated full-pipeline runs).

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/validation/semisynthetic.py tests/lite/validation/test_semisynthetic.py
git commit -m "feat(lite): semi-synthetic κ-detection battery driver"
```

---

### Task 15: Pseudo-dyad null check

**Files:**
- Create: `cadence/lite/validation/pseudo_dyad.py`
- Test: `tests/lite/validation/test_pseudo_dyad.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lite/validation/test_pseudo_dyad.py
"""Tests for cadence/lite/validation/pseudo_dyad.py."""

import pytest
from cadence.lite.validation import pseudo_dyad


def test_round_robin_pairing():
    sessions = ['s1', 's2', 's3', 's4']
    pairs = pseudo_dyad.round_robin_pairs(sessions)
    assert len(pairs) == len(sessions)
    # Ensure no self-pairing
    for p1, p2 in pairs:
        assert p1 != p2
    # Ensure deterministic
    pairs2 = pseudo_dyad.round_robin_pairs(sessions)
    assert pairs == pairs2


def test_round_robin_handles_single_session():
    pairs = pseudo_dyad.round_robin_pairs(['only'])
    assert pairs == []  # cannot pair a single session


def test_round_robin_two_sessions():
    pairs = pseudo_dyad.round_robin_pairs(['a', 'b'])
    # With 2 sessions, only one pair is possible (a,b)
    assert pairs == [('a', 'b')]
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `python -m pytest tests/lite/validation/test_pseudo_dyad.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cadence/lite/validation/pseudo_dyad.py`**

```python
# cadence/lite/validation/pseudo_dyad.py
"""Pseudo-dyad construction + null check for cadence/lite/.

For each session, pairs P1 with P2 from a *different* session of the same
protocol (round-robin: session i's P1 with session ((i+1) mod n)'s P2).
The full pipeline is rerun on these pseudo-dyads; every per-condition
contrast should come out null after FDR.
"""

import os

import numpy as np


def round_robin_pairs(session_ids):
    """Round-robin: session i's P1 pairs with session ((i+1) mod n)'s P2.

    Args:
        session_ids: list of session identifiers (strings).

    Returns:
        List of (p1_session, p2_session) tuples. Empty if len(session_ids) < 2.
    """
    n = len(session_ids)
    if n < 2:
        return []
    return [(session_ids[i], session_ids[(i + 1) % n]) for i in range(n)]


def build_pseudo_session_dict(session_p1, session_p2):
    """Build a fake-session dict mixing P1 from session_p1 with P2 from session_p2.

    Args:
        session_p1: dict (loaded session cache) — P1's data is taken from this.
        session_p2: dict — P2's data is taken from this.

    Returns:
        dict with p1_* keys from session_p1 and p2_* keys from session_p2.
        Also copies markers from session_p1 (since timeline is anchored to P1).
    """
    out = {}
    for key, val in session_p1.items():
        if key.startswith('p1_'):
            out[key] = val
    for key, val in session_p2.items():
        if key.startswith('p2_'):
            out[key] = val
    # Markers/protocol from p1 session
    for key in ['markers', 't_start_absolute', 'duration']:
        if key in session_p1:
            out[key] = session_p1[key]
    return out


def run_pseudo_dyad_check(session_paths_by_protocol, run_session_fn,
                          n_surrogates=200):
    """Run the full pipeline on round-robin pseudo-dyads per protocol.

    Args:
        session_paths_by_protocol: dict {'meditation': [paths], 'pe': [paths]}.
        run_session_fn: callable session_dict -> result dict (e.g.
            cadence.lite.timecourses.run_session adapted to take a session dict).
        n_surrogates: surrogate count for each pseudo-dyad pipeline run.

    Returns:
        dict {protocol: list of per-pseudo-dyad result dicts}.
    """
    results = {}
    for protocol, paths in session_paths_by_protocol.items():
        sids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        path_map = dict(zip(sids, paths))
        pairs = round_robin_pairs(sids)
        per_pair = []
        for sid_p1, sid_p2 in pairs:
            sess_p1 = dict(np.load(path_map[sid_p1], allow_pickle=True))
            sess_p2 = dict(np.load(path_map[sid_p2], allow_pickle=True))
            pseudo = build_pseudo_session_dict(sess_p1, sess_p2)
            per_pair.append(run_session_fn(pseudo, n_surrogates=n_surrogates))
        results[protocol] = per_pair
    return results
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/lite/validation/test_pseudo_dyad.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cadence/lite/validation/pseudo_dyad.py tests/lite/validation/test_pseudo_dyad.py
git commit -m "feat(lite): pseudo-dyad pairing + null-check driver"
```

---

## Phase 7 — Visualization

### Task 16: Per-channel per-condition timeline plots

**Files:**
- Create: `cadence/lite/visualization/timeline.py`

This task has no unit test (visual output). Verify by running the function on a sample result and inspecting the saved PDF.

- [ ] **Step 1: Create `cadence/lite/visualization/timeline.py`**

```python
# cadence/lite/visualization/timeline.py
"""Per-session 7×6 grid of per-channel per-condition coupling-z timecourses."""

import os

import matplotlib.pyplot as plt
import numpy as np

from cadence.lite import config
from cadence.conditions import CONDITION_COLORS


def plot_session_grid(result, output_path, session_id=None):
    """Plot a 7-row × N-condition-column grid for one session.

    Args:
        result: dict returned by cadence.lite.timecourses.run_session.
        output_path: path to save PDF/PNG.
        session_id: optional string for the figure title.
    """
    channels = config.CHANNEL_NAMES
    cond_keys = list(result['condition_segments'].keys())
    fig, axes = plt.subplots(len(channels), len(cond_keys),
                              figsize=(2 * len(cond_keys), 1.5 * len(channels)),
                              sharex=False, sharey='row')
    if axes.ndim == 1:
        axes = axes[None, :]

    t = result['t_common']
    for ri, ch in enumerate(channels):
        z = result['channels'].get(ch)
        if z is None:
            continue
        for ci, cond in enumerate(cond_keys):
            mask = result['condition_segments'][cond]
            ax = axes[ri, ci]
            t_seg = t[mask]
            z_seg = z[mask]
            color = CONDITION_COLORS.get(cond, '#999999')
            ax.plot(t_seg - (t_seg[0] if len(t_seg) else 0),
                     z_seg, color='black', lw=0.8)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax.set_facecolor(color + '40')  # transparent fill
            if ri == 0:
                ax.set_title(cond, fontsize=8)
            if ci == 0:
                ax.set_ylabel(ch, fontsize=7)

    title = f"CADENCE-Lite — {session_id or 'session'}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 2: Quick smoke test (no automated assertion)**

Run interactively (in `python -i`) once a real session result is available:
```python
from cadence.lite.timecourses import run_session
from cadence.lite.visualization.timeline import plot_session_grid
res = run_session('C:/Users/optilab/desktop/MCCT/session_cache/y_06.npz', n_surrogates=20)
plot_session_grid(res, 'results/lite/figures/y_06_grid.png', session_id='y_06')
```
Inspect the saved PNG visually.

- [ ] **Step 3: Commit**

```bash
git add cadence/lite/visualization/timeline.py
git commit -m "feat(lite): per-session 7×6 timeline grid plot"
```

---

### Task 17: Forest plot of all 70 contrasts

**Files:**
- Create: `cadence/lite/visualization/contrast_summary.py`

- [ ] **Step 1: Create `cadence/lite/visualization/contrast_summary.py`**

```python
# cadence/lite/visualization/contrast_summary.py
"""Forest plot of all 70 contrasts grouped by channel.

Input: a long-form pandas DataFrame with columns:
    channel, model, contrast_id, estimate, ci_lo, ci_hi, q_value, perm_p, n_dyad
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_contrast_forest(df, output_path):
    """Forest plot: rows = channel × model × contrast, x = estimate ± CI.

    Significant rows (q < 0.05) are colored.
    """
    df = df.copy()
    df['label'] = df['channel'] + ' | ' + df['model'] + ' | ' + df['contrast_id']
    df = df.sort_values(['channel', 'model', 'contrast_id'])

    n = len(df)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.18)))

    y = np.arange(n)
    sig = df['q_value'].values < 0.05
    color = np.where(sig, '#1f77b4', '#999999')

    ax.errorbar(df['estimate'].values, y,
                xerr=[df['estimate'].values - df['ci_lo'].values,
                      df['ci_hi'].values - df['estimate'].values],
                fmt='o', ecolor='gray', mfc='none', mec='black', lw=0.6)
    ax.scatter(df['estimate'].values, y, c=color, s=18, zorder=3)
    ax.axvline(0, color='gray', lw=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df['label'].values, fontsize=6)
    ax.set_xlabel('Contrast estimate (coupling z units)')
    ax.set_title('CADENCE-Lite contrasts (filled = q<0.05)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 2: Commit**

```bash
git add cadence/lite/visualization/contrast_summary.py
git commit -m "feat(lite): forest plot for all-contrasts summary"
```

---

### Task 18: AUC-vs-κ curves per channel

**Files:**
- Create: `cadence/lite/visualization/kappa_curves.py`

- [ ] **Step 1: Create `cadence/lite/visualization/kappa_curves.py`**

```python
# cadence/lite/visualization/kappa_curves.py
"""AUC-vs-κ detection curves per channel from semi-synthetic battery."""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_kappa_curves(per_channel_results, output_path, channel_order=None):
    """Plot AUC vs κ for each channel, one panel per channel.

    Args:
        per_channel_results: dict[channel] -> dict from
            cadence.lite.validation.semisynthetic.run_kappa_battery.
        output_path: path to save the PDF/PNG.
        channel_order: optional list ordering the channels.
    """
    if channel_order is None:
        channel_order = list(per_channel_results.keys())

    n = len(channel_order)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows),
                              sharey=True)
    if rows == 1:
        axes = axes[None, :]
    if cols == 1:
        axes = axes[:, None]

    for i, ch in enumerate(channel_order):
        ax = axes.flat[i]
        res = per_channel_results[ch]
        ax.plot(res['kappa'], res['auc'], 'o-', color='#1f77b4', lw=1.2)
        ax.axhline(0.5, color='gray', ls=':', lw=0.6)
        ax.set_xlim(-0.02, max(res['kappa']) + 0.05)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(ch, fontsize=9)
        if i % cols == 0:
            ax.set_ylabel('AUC')
        if i // cols == rows - 1:
            ax.set_xlabel('κ')

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes.flat[j].axis('off')

    fig.suptitle('CADENCE-Lite κ-detection curves', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 2: Commit**

```bash
git add cadence/lite/visualization/kappa_curves.py
git commit -m "feat(lite): AUC-vs-κ per-channel detection curves"
```

---

## Phase 8 — Driver Scripts

### Task 19: Main pipeline driver

**Files:**
- Create: `scripts/_run_lite_pipeline.py`

- [ ] **Step 1: Create `scripts/_run_lite_pipeline.py`**

```python
# scripts/_run_lite_pipeline.py
"""Driver: run cadence/lite pipeline on one or more sessions, fit stats, save outputs.

Usage:
    python scripts/_run_lite_pipeline.py --session y_06
    python scripts/_run_lite_pipeline.py --all
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from cadence.lite import config, timecourses
from cadence.lite.stats import models, contrasts as contrasts_mod, permutation, fdr
from cadence.lite.visualization.timeline import plot_session_grid
from cadence.lite.visualization.contrast_summary import plot_contrast_forest

SESSION_CACHE = os.environ.get(
    'CADENCE_SESSION_CACHE',
    'C:/Users/optilab/desktop/MCCT/session_cache')
RESULTS_ROOT = 'results/lite'


def _run_one(session_path, n_surrogates):
    print(f"[lite] {os.path.basename(session_path)}")
    res = timecourses.run_session(session_path, n_surrogates=n_surrogates)
    return res


def _build_panel(per_session_results):
    """Long-form panel for the mixed-effects models."""
    rows = []
    for sid, res in per_session_results.items():
        protocol = res['protocol']
        for ch_name, cond_dict in res['per_condition_mean_z'].items():
            for cond, z in cond_dict.items():
                row = {
                    'dyad': sid,
                    'protocol': protocol,
                    'channel': ch_name,
                    'condition': cond,
                    'coupling_z': z,
                }
                rows.append(row)
            # Also append per-electrode rows for EEG channels
            if ch_name in res['eeg_per_electrode']:
                z_per = res['eeg_per_electrode'][ch_name]  # (n_el, N)
                for cond, mask in res['condition_segments'].items():
                    if mask.sum() == 0:
                        continue
                    for el_i in range(z_per.shape[0]):
                        rows.append({
                            'dyad': sid,
                            'protocol': protocol,
                            'channel': ch_name + '_per_electrode',
                            'condition': cond,
                            'electrode': f'e{el_i:02d}',
                            'coupling_z': float(np.nanmean(z_per[el_i, mask])),
                        })
    return pd.DataFrame(rows)


def _fit_all_contrasts(panel):
    """Fit Model M and Model P, evaluate 5 contrasts × 7 channels each."""
    out_rows = []
    for protocol, model_label in [('meditation', 'M'), ('pe', 'P')]:
        sub = panel[panel['protocol'] == protocol]
        if len(sub) == 0:
            continue
        for ch_name in config.CHANNEL_NAMES:
            ch_panel = sub[sub['channel'] == ch_name]
            if len(ch_panel) < 6:
                continue

            # EEG: include electrode random factor (use _per_electrode rows)
            if ch_name.startswith('eeg_'):
                eeg_panel = sub[sub['channel'] == ch_name + '_per_electrode']
                if len(eeg_panel) >= 12:
                    formula = 'coupling_z ~ condition + (1|dyad) + (1|electrode)'
                    fit = models.fit_mixed(eeg_panel, formula=formula)
                else:
                    formula = 'coupling_z ~ condition + (1|dyad)'
                    fit = models.fit_mixed(ch_panel, formula=formula)
            else:
                formula = 'coupling_z ~ condition + (1|dyad)'
                fit = models.fit_mixed(ch_panel, formula=formula)

            for c in contrasts_mod.protocol_contrasts(protocol):
                res = contrasts_mod.apply_contrast(fit, c)
                # Permutation p (skip if requested for speed)
                perm_p = permutation.permutation_p(
                    ch_panel if not ch_name.startswith('eeg_') else eeg_panel,
                    c, formula=formula, n_permutations=500, n_jobs=-1, seed=42)
                out_rows.append({
                    'channel': ch_name,
                    'model': model_label,
                    'contrast_id': c['id'],
                    'contrast_name': c['name'],
                    'estimate': res['estimate'],
                    'se': res['se'],
                    'p_value': res['p_value'],
                    'permutation_p': perm_p,
                    'cohens_d': res['cohens_d'],
                    'ci_lo': res['estimate'] - 1.96 * res['se'],
                    'ci_hi': res['estimate'] + 1.96 * res['se'],
                })

    df = pd.DataFrame(out_rows)
    if len(df) > 0:
        bh = fdr.bh(df['p_value'].values, alpha=0.05)
        df['q_value'] = bh['q_values']
        df['reject'] = bh['reject']
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', nargs='+', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--n-surrogates', type=int, default=config.N_SURROGATES)
    args = ap.parse_args()

    if args.all:
        paths = sorted(glob.glob(os.path.join(SESSION_CACHE, '*.npz')))
    elif args.session:
        paths = [os.path.join(SESSION_CACHE, f'{s}.npz') for s in args.session]
    else:
        paths = [os.path.join(SESSION_CACHE, 'y_06.npz')]

    per_session = {}
    for p in paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        try:
            res = _run_one(p, args.n_surrogates)
            per_session[sid] = res

            # Per-session timecourses
            out_dir = os.path.join(RESULTS_ROOT, 'timecourses', sid)
            os.makedirs(out_dir, exist_ok=True)
            for ch, z in res['channels'].items():
                np.savez(os.path.join(out_dir, f'{ch}.npz'),
                          t=res['t_common'], z=z)
            # Per-electrode for EEG
            for ch in ['eeg_alpha_coh', 'eeg_theta_coh']:
                if ch in res['eeg_per_electrode']:
                    np.savez(os.path.join(out_dir, f'{ch}_per_electrode.npz'),
                              t=res['t_common'],
                              z_per_electrode=res['eeg_per_electrode'][ch])

            # Per-session figure
            plot_session_grid(
                res,
                os.path.join(RESULTS_ROOT, 'figures', f'{sid}_grid.png'),
                session_id=sid)
        except Exception as e:
            print(f"  ERROR on {sid}: {e}")

    # Fit stats on all sessions combined
    if len(per_session) >= 2:
        panel = _build_panel(per_session)
        contrast_df = _fit_all_contrasts(panel)
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        panel.to_parquet(os.path.join(RESULTS_ROOT, 'panel.parquet'))
        contrast_df.to_json(os.path.join(RESULTS_ROOT, 'contrasts.json'),
                             orient='records', indent=2)
        plot_contrast_forest(
            contrast_df,
            os.path.join(RESULTS_ROOT, 'figures', 'contrasts_forest.png'))
        print(f"[lite] Wrote {len(contrast_df)} contrast rows to "
               f"{RESULTS_ROOT}/contrasts.json")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test on y_06**

Run: `python scripts/_run_lite_pipeline.py --session y_06 --n-surrogates 50`
Expected: produces `results/lite/timecourses/y_06/*.npz` files + `results/lite/figures/y_06_grid.png`. With only 1 session, contrasts.json will not be produced (need ≥2).

- [ ] **Step 3: Commit**

```bash
git add scripts/_run_lite_pipeline.py
git commit -m "feat(lite): main pipeline driver script (per-session + stats)"
```

---

### Task 20: Pseudo-dyad null check driver

**Files:**
- Create: `scripts/_run_lite_pseudo_dyad.py`

- [ ] **Step 1: Create `scripts/_run_lite_pseudo_dyad.py`**

```python
# scripts/_run_lite_pseudo_dyad.py
"""Driver: run cadence/lite pipeline on round-robin pseudo-dyads, fit stats.

For each protocol, pairs P1 from session i with P2 from session ((i+1) % n).
Every per-condition contrast should come out null after FDR.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from cadence.lite import config, timecourses
from cadence.lite.stats import models, contrasts as contrasts_mod, fdr
from cadence.lite.validation.pseudo_dyad import (
    round_robin_pairs, build_pseudo_session_dict)

SESSION_CACHE = os.environ.get(
    'CADENCE_SESSION_CACHE',
    'C:/Users/optilab/desktop/MCCT/session_cache')
RESULTS_ROOT = 'results/lite/pseudo_dyad'

# Hardcoded protocol assignments (extend as new sessions are added)
PROTOCOL_SESSIONS = {
    'meditation': ['y_06', 'y_17', 'y_19', 'y_11', 'y_04', 'y_24'],
    'pe':         ['y_01', 'y_05', 'y_10', 'y_32', 'y_41'],
}


def _run_pseudo(p1_path, p2_path, n_surrogates):
    sess_p1 = dict(np.load(p1_path, allow_pickle=True))
    sess_p2 = dict(np.load(p2_path, allow_pickle=True))
    pseudo = build_pseudo_session_dict(sess_p1, sess_p2)
    # Save to a temporary npz so timecourses.run_session can reload it
    tmp = os.path.join(RESULTS_ROOT, '.tmp_pseudo.npz')
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    np.savez(tmp, **pseudo)
    return timecourses.run_session(tmp, n_surrogates=n_surrogates)


def _build_panel(per_session_results, protocol):
    rows = []
    for pseudo_id, res in per_session_results.items():
        for ch_name, cond_dict in res['per_condition_mean_z'].items():
            for cond, z in cond_dict.items():
                rows.append({
                    'dyad': pseudo_id,
                    'protocol': protocol,
                    'channel': ch_name,
                    'condition': cond,
                    'coupling_z': z,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-surrogates', type=int, default=config.N_SURROGATES)
    args = ap.parse_args()

    all_results = {}
    for protocol, sids in PROTOCOL_SESSIONS.items():
        paths = [os.path.join(SESSION_CACHE, f'{s}.npz') for s in sids
                 if os.path.exists(os.path.join(SESSION_CACHE, f'{s}.npz'))]
        present_sids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        pairs = round_robin_pairs(present_sids)
        path_map = dict(zip(present_sids, paths))

        per_pseudo = {}
        for sid_p1, sid_p2 in pairs:
            pseudo_id = f'pseudo_{sid_p1}_x_{sid_p2}'
            print(f"[pseudo] {pseudo_id}")
            try:
                res = _run_pseudo(path_map[sid_p1], path_map[sid_p2],
                                   args.n_surrogates)
                per_pseudo[pseudo_id] = res
            except Exception as e:
                print(f"  ERROR: {e}")

        if len(per_pseudo) >= 2:
            panel = _build_panel(per_pseudo, protocol)
            out_rows = []
            for ch_name in config.CHANNEL_NAMES:
                ch_panel = panel[panel['channel'] == ch_name]
                if len(ch_panel) < 6:
                    continue
                fit = models.fit_mixed(
                    ch_panel,
                    formula='coupling_z ~ condition + (1|dyad)')
                for c in contrasts_mod.protocol_contrasts(protocol):
                    res = contrasts_mod.apply_contrast(fit, c)
                    out_rows.append({
                        'protocol': protocol,
                        'channel': ch_name,
                        'contrast_id': c['id'],
                        'estimate': res['estimate'],
                        'p_value': res['p_value'],
                    })
            df = pd.DataFrame(out_rows)
            if len(df) > 0:
                bh = fdr.bh(df['p_value'].values, alpha=0.05)
                df['q_value'] = bh['q_values']
                df['reject'] = bh['reject']
            all_results[protocol] = df

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    combined = pd.concat(all_results.values(), ignore_index=True) \
        if all_results else pd.DataFrame()
    combined.to_json(
        os.path.join(RESULTS_ROOT, 'pseudo_dyad_contrasts.json'),
        orient='records', indent=2)
    n_sig = int(combined['reject'].sum()) if 'reject' in combined.columns else 0
    print(f"\n[pseudo] {len(combined)} contrasts, {n_sig} significant after FDR")
    if n_sig > 0:
        print("WARNING: significant pseudo-dyad contrasts indicate pipeline bias!")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (skip if cache not present)**

Run: `python scripts/_run_lite_pseudo_dyad.py --n-surrogates 50`
Expected: writes `results/lite/pseudo_dyad/pseudo_dyad_contrasts.json`. Almost all rows should have `reject=False` (this is the null-integrity check).

- [ ] **Step 3: Commit**

```bash
git add scripts/_run_lite_pseudo_dyad.py
git commit -m "feat(lite): pseudo-dyad null-check driver script"
```

---

### Task 21: Semi-synthetic battery driver

**Files:**
- Create: `scripts/_run_lite_semisynthetic.py`

- [ ] **Step 1: Create `scripts/_run_lite_semisynthetic.py`**

```python
# scripts/_run_lite_semisynthetic.py
"""Driver: run κ-detection battery for each lite channel on pseudo-dyad bases.

Outputs per-channel AUC-vs-κ curves to results/lite/semisynthetic/.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from cadence.lite import config
from cadence.lite.validation.semisynthetic import run_kappa_battery
from cadence.lite.validation.pseudo_dyad import build_pseudo_session_dict
from cadence.lite.visualization.kappa_curves import plot_kappa_curves

SESSION_CACHE = os.environ.get(
    'CADENCE_SESSION_CACHE',
    'C:/Users/optilab/desktop/MCCT/session_cache')
RESULTS_ROOT = 'results/lite/semisynthetic'


def _build_pseudo_base(sid_p1, sid_p2, channel):
    """Load a pseudo-dyad and extract the keys needed for `channel`."""
    sess_p1 = dict(np.load(os.path.join(SESSION_CACHE, f'{sid_p1}.npz'),
                            allow_pickle=True))
    sess_p2 = dict(np.load(os.path.join(SESSION_CACHE, f'{sid_p2}.npz'),
                            allow_pickle=True))
    pseudo = build_pseudo_session_dict(sess_p1, sess_p2)

    base = {}
    if channel.startswith('ecg_'):
        for k in ['p1_rr_ms', 'p1_rr_ts', 'p2_rr_ms', 'p2_rr_ts']:
            if k in pseudo:
                base[k] = pseudo[k]
    elif channel.startswith('eeg_'):
        for k in ['p1_eeg', 'p2_eeg', 'p1_eeg_ts', 'p2_eeg_ts']:
            if k in pseudo:
                base[k] = pseudo[k]
    elif channel.startswith('face_'):
        for k in ['p1_au', 'p2_au', 'p1_au_ts', 'p2_au_ts']:
            if k in pseudo:
                base[k] = pseudo[k]
    elif channel == 'pose_multilag':
        for k in ['p1_pose_features', 'p2_pose_features',
                  'p1_pose_features_ts', 'p2_pose_features_ts']:
            if k in pseudo:
                base[k] = pseudo[k]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1', default='y_06')
    ap.add_argument('--p2', default='y_17')
    ap.add_argument('--n-surrogates', type=int, default=200)
    ap.add_argument('--n-repeats', type=int, default=3)
    ap.add_argument('--kappas', nargs='+', type=float,
                     default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument('--channels', nargs='+', default=config.CHANNEL_NAMES)
    args = ap.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    per_channel = {}
    for ch in args.channels:
        print(f"[semisynth] {ch}")
        try:
            base = _build_pseudo_base(args.p1, args.p2, ch)
            if not base:
                print(f"  no data for {ch}; skipping")
                continue
            res = run_kappa_battery(
                base, channel=ch, kappas=args.kappas,
                n_surrogates=args.n_surrogates,
                n_repeats=args.n_repeats)
            per_channel[ch] = res
            df = pd.DataFrame(res)
            df['channel'] = ch
            df.to_csv(os.path.join(RESULTS_ROOT, f'{ch}_kappa_auc.csv'),
                       index=False)
        except Exception as e:
            print(f"  ERROR on {ch}: {e}")

    if per_channel:
        plot_kappa_curves(
            per_channel,
            os.path.join(RESULTS_ROOT, 'kappa_curves.png'))
        print(f"[semisynth] wrote curves for {len(per_channel)} channels")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (1 channel, fast)**

Run: `python scripts/_run_lite_semisynthetic.py --p1 y_06 --p2 y_17 --channels ecg_hf_env --kappas 0 0.4 --n-surrogates 50 --n-repeats 1`
Expected: writes `results/lite/semisynthetic/ecg_hf_env_kappa_auc.csv` and the curves PNG.

- [ ] **Step 3: Commit**

```bash
git add scripts/_run_lite_semisynthetic.py
git commit -m "feat(lite): semi-synthetic κ-detection battery driver"
```

---

## Self-Review

After writing all 21 tasks above, the plan was reviewed against the spec.

**Spec coverage:**
- §3 Architecture → Phase 4 (Task 8 timecourses orchestrator).
- §4 7 channels → Phases 2, 3 (Tasks 3-7 surrogates + 4 coupling modules).
- §5 Surrogate normalization → Task 3 surrogates.py.
- §6 Statistical layer → Phase 5 (Tasks 9-12 models, contrasts, permutation, FDR).
- §7 Validation Tier 2 → Phase 6 (Tasks 13-15 synth_ecg, semisynthetic, pseudo_dyad).
- §8 Code structure → file map at the top + per-task file paths.
- §9 Deliverables → Phase 7 (Tasks 16-18 visualization) + Phase 8 (Tasks 19-21 driver scripts).
- §10 Deferred items → not implemented (correctly, by design).
- §13 Implementation notes → Tasks 1, 9 (pymer4 install).

**Placeholder scan:** No "TBD", "TODO", "implement later", or "fill in details" patterns. Each task has complete code blocks where required. No reference to types or functions not defined in any task.

**Type consistency:**
- `surrogate_z` returns `{'real', 'null_mean', 'null_std', 'z', 'shifts'}` — used identically in tests and downstream.
- `compute(p1, p2, t_p1, t_p2, t_common, ...)` signature is uniform across `pose_multilag`, `ecg_envelope`, `face_wavelet`, `eeg_wavelet` (with channel-specific extra kwargs).
- `eeg_wavelet.compute` returns `{'z_per_electrode', 'z_aggregated'}` and orchestrator (Task 8) accesses both correctly.
- Mixed-effects wrapper returns `{'coefs', 'model', 'backend'}` consistent across `models.fit_mixed`, `contrasts.apply_contrast`, `permutation.permutation_p`.

**Known remaining risks (flagged for execution time, not blocking):**
- The driver script (Task 19) references `cadence.synthetic_v82.inject_eeg_band_coupling` etc. for non-ECG semi-synthetic injection. The exact function signatures in `synthetic_v82.py` should be verified during Task 14 execution and the driver in Task 21 may need minor adjustment if the existing API differs.
- The `cadence.data.eeg_features.detect_r_peaks_to_rr_ms` import in Task 8 is provisional; if the cache already contains RR data, the fallback path is unused. If it's used and the function name differs, swap with the existing `eeg_features.py` R-peak function (verify name when running).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-cadence-lite.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best when many tasks have low dependency between them, or when you want each task independently checked before moving on.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review. Best when tasks are tightly coupled or when you want to keep reasoning context across tasks.

Which approach?
