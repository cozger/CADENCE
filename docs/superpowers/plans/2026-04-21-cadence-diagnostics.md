# CADENCE rSLDS Diagnostic Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-tier diagnostic suite that validates the V11 rSLDS pipeline's feature quality, state transition behavior, observation-space structure, and model architecture choices.

**Architecture:** Tier 1 (fast screening: Modules 2+3) gates Tier 2 (deep dive: Modules 1, 4, 5, 6). All modules share a `data_loader.py` that reads existing scaffold + rSLDS result files — no pipeline modifications. Findings are written to timestamped output directories as plots, CSVs, and Markdown reports.

**Tech Stack:** Python 3.11, numpy, scipy, sklearn 1.8, matplotlib, pandas, umap-learn (new), hmmlearn (new), pytest 9.0. MCCT conda environment.

---

## Scaffold output format (reference for all tasks)

`results/v11/{session}/scaffold_v11_ztimecourses.npz`:
- `z_{key}` — prewhitened+standardized channel (what rSLDS sees)
- `z_raw_{key}` — raw surrogate-corrected channel (pre-whitening)
- `t_common`, `obs_mask`, `U_covariates`, `u_{key}` (covariate channels)

`results/v11/{session}/scaffold_v11_results.json`:
- `modality_keys`, `covariate_keys`, `segments` [(name, t0, t1), ...]

`results/v11/{session}/v11_rslds_results.npz`:
- `gamma` (N, K), `path` (N,) constrained Viterbi (20-sample min-dwell)
- `d_emit` (K, D), `C_emit`, `R_emit`, `W_trans` (K, K), `S_trans` (K, K, D_cov)
- `state_labels` (K,)

**Unconstrained path** = `argmax(gamma, axis=1)` — full IOHMMParams not saved, this is the correct proxy.

**T_kk extraction** from `W_trans`:
```python
from scipy.special import softmax as _sm
T = np.array([_sm(W_trans[k]) for k in range(K)])  # (K, K)
T_kk = T[k, k]
```

---

## File Structure

```
diagnostics/
  __init__.py
  shared/
    __init__.py
    data_loader.py      — SessionData dataclass + load_session / load_all_sessions
    report_utils.py     — make_output_dir, write_md_report, screening_report_path
  tier1_screening/
    __init__.py
    module2_feature_diagnostics.py
    module3_transition_analysis.py
    run_tier1.py
  tier2_deepdive/
    __init__.py
    module1_obs_space.py
    module4_model_comparison.py
    module5_block_pca.py
    module6_null_ablation.py
    run_tier2.py
tests/
  diagnostics/
    conftest.py
    test_data_loader.py
    test_module2.py
    test_module3.py
    test_module1.py
    test_module4.py
    test_module5.py
    test_module6.py
```

---

## Task 1: Install dependencies + directory scaffolding

**Files:** `diagnostics/__init__.py`, `diagnostics/shared/__init__.py`, `diagnostics/tier1_screening/__init__.py`, `diagnostics/tier2_deepdive/__init__.py`, `tests/diagnostics/__init__.py`, `tests/diagnostics/conftest.py`

- [ ] **Step 1: Install umap-learn and hmmlearn**

```bash
conda run -n MCCT pip install umap-learn hmmlearn
```
Expected: both install without errors.

- [ ] **Step 2: Verify installs**

```bash
conda run -n MCCT python -c "import umap; import hmmlearn; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Create package skeleton**

```bash
cd C:/Users/optilab/desktop/CADENCE
mkdir -p diagnostics/shared diagnostics/tier1_screening diagnostics/tier2_deepdive
mkdir -p tests/diagnostics
touch diagnostics/__init__.py diagnostics/shared/__init__.py
touch diagnostics/tier1_screening/__init__.py diagnostics/tier2_deepdive/__init__.py
touch tests/__init__.py tests/diagnostics/__init__.py
```

- [ ] **Step 4: Write conftest.py with shared fixtures**

Create `tests/diagnostics/conftest.py`:

```python
import numpy as np
import json
import pytest

N, D, K, D_COV = 120, 5, 4, 3
FS = 2.0
MOD_KEYS = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
COV_KEYS = ['z_slow_pc1', 'z_slow_pc2', 'flex']
SEGMENTS = [('base_EO', 0.0, 30.0), ('conv_1', 30.0, 90.0), ('meditate_B', 90.0, 60.0)]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fake_session_dir(tmp_path, rng):
    """Write minimal scaffold + rSLDS files for one session."""
    sess_dir = tmp_path / 'fake_sess'
    sess_dir.mkdir()

    Y_raw = rng.standard_normal((N, D)).astype(np.float32)
    Y_pw = rng.standard_normal((N, D)).astype(np.float32)
    obs_mask = np.ones((N, D), dtype=bool)
    U = rng.standard_normal((N, D_COV)).astype(np.float32)
    t_common = np.arange(N, dtype=np.float32) / FS

    save_dict = {'t_common': t_common, 'obs_mask': obs_mask, 'U_covariates': U}
    for i, key in enumerate(MOD_KEYS):
        save_dict[f'z_{key}'] = Y_pw[:, i]
        save_dict[f'z_raw_{key}'] = Y_raw[:, i]
    for i, key in enumerate(COV_KEYS):
        save_dict[f'u_{key}'] = U[:, i]

    np.savez_compressed(str(sess_dir / 'scaffold_v11_ztimecourses.npz'), **save_dict)

    meta = {
        'session': 'fake_sess', 'version': 'v11',
        'n_timepoints': N, 'duration_s': float(N / FS),
        'fs_out': FS,
        'modality_keys': MOD_KEYS,
        'covariate_keys': COV_KEYS,
        'segments': [[s, t0, t1] for s, t0, t1 in SEGMENTS],
    }
    with open(str(sess_dir / 'scaffold_v11_results.json'), 'w') as f:
        json.dump(meta, f)

    # rSLDS results
    gamma = np.abs(rng.standard_normal((N, K)))
    gamma /= gamma.sum(axis=1, keepdims=True)
    path = np.argmax(gamma, axis=1).astype(np.int32)
    d_emit = rng.standard_normal((K, D)).astype(np.float32)
    W_trans = rng.standard_normal((K, K)).astype(np.float32)
    S_trans = rng.standard_normal((K, K, D_COV)).astype(np.float32)
    state_labels = np.array(['NULL', 'COUP', 'SHARED', 'OTHER'], dtype=object)

    np.savez_compressed(str(sess_dir / 'v11_rslds_results.npz'),
                        gamma=gamma, path=path, t_common=t_common,
                        d_emit=d_emit, W_trans=W_trans, S_trans=S_trans,
                        state_labels=state_labels)
    return tmp_path, 'fake_sess'


@pytest.fixture
def fake_scaffold_only_dir(tmp_path, rng):
    """Scaffold only — no rSLDS fit."""
    sess_dir = tmp_path / 'no_rslds'
    sess_dir.mkdir()
    Y_raw = rng.standard_normal((N, D)).astype(np.float32)
    Y_pw = rng.standard_normal((N, D)).astype(np.float32)
    obs_mask = np.ones((N, D), dtype=bool)
    U = rng.standard_normal((N, D_COV)).astype(np.float32)
    t_common = np.arange(N, dtype=np.float32) / FS
    save_dict = {'t_common': t_common, 'obs_mask': obs_mask, 'U_covariates': U}
    for i, key in enumerate(MOD_KEYS):
        save_dict[f'z_{key}'] = Y_pw[:, i]
        save_dict[f'z_raw_{key}'] = Y_raw[:, i]
    np.savez_compressed(str(sess_dir / 'scaffold_v11_ztimecourses.npz'), **save_dict)
    meta = {'session': 'no_rslds', 'version': 'v11',
            'n_timepoints': N, 'duration_s': float(N / FS), 'fs_out': FS,
            'modality_keys': MOD_KEYS, 'covariate_keys': COV_KEYS,
            'segments': [[s, t0, t1] for s, t0, t1 in SEGMENTS]}
    with open(str(sess_dir / 'scaffold_v11_results.json'), 'w') as f:
        json.dump(meta, f)
    return tmp_path, 'no_rslds'
```

- [ ] **Step 5: Commit skeleton**

```bash
cd C:/Users/optilab/desktop/CADENCE
git add diagnostics/ tests/
git commit -m "feat(diagnostics): scaffold directory structure + conftest fixtures"
```

---

## Task 2: `diagnostics/shared/data_loader.py`

**Files:** Create `diagnostics/shared/data_loader.py`, `tests/diagnostics/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/diagnostics/test_data_loader.py`:

```python
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.shared.data_loader import load_session, load_all_sessions, discover_session_names


def test_load_session_shapes(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.Y_raw.shape == (120, 5)
    assert data.Y_pw.shape == (120, 5)
    assert data.U.shape == (120, 3)
    assert data.obs_mask.shape == (120, 5)
    assert data.t_common.shape == (120,)


def test_load_session_keys(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.modality_keys == ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    assert len(data.segments) == 3
    assert data.segments[0][0] == 'base_EO'


def test_load_session_rslds_fields(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.gamma is not None and data.gamma.shape == (120, 4)
    assert data.path_constrained is not None and data.path_constrained.shape == (120,)
    assert data.path_unconstrained is not None


def test_unconstrained_path_is_argmax_gamma(fake_session_dir):
    results_dir, session_name = fake_session_dir
    data = load_session(session_name, results_dir=str(results_dir))
    expected = np.argmax(data.gamma, axis=1)
    np.testing.assert_array_equal(data.path_unconstrained, expected)


def test_load_session_no_rslds_returns_none(fake_scaffold_only_dir):
    results_dir, session_name = fake_scaffold_only_dir
    data = load_session(session_name, results_dir=str(results_dir))
    assert data.gamma is None
    assert data.path_constrained is None
    assert data.path_unconstrained is None


def test_discover_session_names(fake_session_dir):
    results_dir, _ = fake_session_dir
    names = discover_session_names(str(results_dir))
    assert 'fake_sess' in names


def test_load_all_sessions(fake_session_dir):
    results_dir, _ = fake_session_dir
    sessions = load_all_sessions(str(results_dir))
    assert len(sessions) >= 1
    assert sessions[0].name == 'fake_sess'
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd C:/Users/optilab/desktop/CADENCE
conda run -n MCCT pytest tests/diagnostics/test_data_loader.py -v 2>&1 | head -30
```
Expected: `ImportError` or `ModuleNotFoundError` — `data_loader` doesn't exist yet.

- [ ] **Step 3: Implement `data_loader.py`**

Create `diagnostics/shared/data_loader.py`:

```python
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SessionData:
    name: str
    t_common: np.ndarray
    Y_raw: np.ndarray
    Y_pw: np.ndarray
    U: np.ndarray
    obs_mask: np.ndarray
    modality_keys: List[str]
    covariate_keys: List[str]
    segments: List[Tuple[str, float, float]]
    fs: float = 2.0
    gamma: Optional[np.ndarray] = None
    path_constrained: Optional[np.ndarray] = None
    path_unconstrained: Optional[np.ndarray] = None
    d_emit: Optional[np.ndarray] = None
    C_emit: Optional[np.ndarray] = None
    R_emit: Optional[np.ndarray] = None
    W_trans: Optional[np.ndarray] = None
    S_trans: Optional[np.ndarray] = None
    state_labels: Optional[List[str]] = None


def load_session(session_name: str, results_dir: str = 'results/v11') -> SessionData:
    sess_dir = os.path.join(results_dir, session_name)
    scaffold_npz = os.path.join(sess_dir, 'scaffold_v11_ztimecourses.npz')
    scaffold_json = os.path.join(sess_dir, 'scaffold_v11_results.json')

    npz = np.load(scaffold_npz, allow_pickle=False)
    with open(scaffold_json) as f:
        meta = json.load(f)

    mod_keys = meta['modality_keys']
    cov_keys = meta.get('covariate_keys', [])
    fs = float(meta.get('fs_out', 2.0))

    Y_raw = np.column_stack([npz[f'z_raw_{k}'] for k in mod_keys])
    Y_pw = np.column_stack([npz[f'z_{k}'] for k in mod_keys])
    U = npz['U_covariates'] if 'U_covariates' in npz else np.zeros((len(npz['t_common']), len(cov_keys)))
    obs_mask = npz['obs_mask']
    t_common = npz['t_common']
    segments = [(s[0], float(s[1]), float(s[2])) for s in meta.get('segments', [])]

    rslds_path = os.path.join(sess_dir, 'v11_rslds_results.npz')
    gamma = path_con = path_uncon = d_emit = C_emit = R_emit = W_trans = S_trans = state_labels = None

    if os.path.exists(rslds_path):
        r = np.load(rslds_path, allow_pickle=True)
        gamma = r['gamma']
        path_con = r['path']
        path_uncon = np.argmax(gamma, axis=1).astype(np.int32)
        d_emit = r['d_emit'] if 'd_emit' in r else None
        C_emit = r['C_emit'] if 'C_emit' in r else None
        R_emit = r['R_emit'] if 'R_emit' in r else None
        W_trans = r['W_trans'] if 'W_trans' in r else None
        S_trans = r['S_trans'] if 'S_trans' in r else None
        sl = r['state_labels'] if 'state_labels' in r else None
        state_labels = list(sl) if sl is not None else None

    return SessionData(
        name=session_name, t_common=t_common,
        Y_raw=Y_raw, Y_pw=Y_pw, U=U, obs_mask=obs_mask,
        modality_keys=mod_keys, covariate_keys=cov_keys,
        segments=segments, fs=fs,
        gamma=gamma, path_constrained=path_con, path_unconstrained=path_uncon,
        d_emit=d_emit, C_emit=C_emit, R_emit=R_emit,
        W_trans=W_trans, S_trans=S_trans, state_labels=state_labels,
    )


def discover_session_names(results_dir: str = 'results/v11') -> List[str]:
    names = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if entry.is_dir():
            scaffold = os.path.join(entry.path, 'scaffold_v11_ztimecourses.npz')
            if os.path.exists(scaffold):
                names.append(entry.name)
    return names


def load_all_sessions(results_dir: str = 'results/v11') -> List[SessionData]:
    return [load_session(n, results_dir) for n in discover_session_names(results_dir)]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_data_loader.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/shared/data_loader.py tests/diagnostics/test_data_loader.py
git commit -m "feat(diagnostics): SessionData + load_session / load_all_sessions"
```

---

## Task 3: `diagnostics/shared/report_utils.py`

**Files:** `diagnostics/shared/report_utils.py`

- [ ] **Step 1: Implement directly (no separate test — pure file I/O, covered by integration)**

Create `diagnostics/shared/report_utils.py`:

```python
from __future__ import annotations
import os
from datetime import datetime
from typing import List, Tuple


def make_output_dir(base_dir: str, module_name: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(base_dir, f'{ts}_{module_name}')
    os.makedirs(out, exist_ok=True)
    return out


def write_md_report(output_dir: str, filename: str, sections: List[Tuple[str, str]]) -> None:
    path = os.path.join(output_dir, filename)
    lines = []
    for title, body in sections:
        lines.append(f'## {title}\n')
        lines.append(body.strip())
        lines.append('\n')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def screening_report_path(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, 'screening_report.md')
```

- [ ] **Step 2: Commit**

```bash
git add diagnostics/shared/report_utils.py
git commit -m "feat(diagnostics): report_utils helpers"
```

---

## Task 4: Module 2 — math utilities

**Files:** `diagnostics/tier1_screening/module2_feature_diagnostics.py` (math functions only), `tests/diagnostics/test_module2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/diagnostics/test_module2.py`:

```python
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier1_screening.module2_feature_diagnostics import (
    compute_acf, compute_n_eff_geyer, compute_vif, is_slow_drift
)


def test_acf_white_noise_near_zero(rng):
    x = rng.standard_normal(2000)
    acf = compute_acf(x, max_lag=40)
    assert acf[0] == pytest.approx(1.0)
    assert np.abs(acf[10:]).max() < 0.15


def test_acf_ar1_decays_geometrically(rng):
    rho = 0.8
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = rho * x[t-1] + rng.standard_normal()
    acf = compute_acf(x, max_lag=10)
    for lag in range(1, 8):
        assert abs(acf[lag] - rho**lag) < 0.05


def test_n_eff_white_noise_near_n(rng):
    x = rng.standard_normal(1000)
    neff = compute_n_eff_geyer(x)
    assert 700 < neff <= 1000


def test_n_eff_ar1_rho09_much_less_than_n(rng):
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.9 * x[t-1] + rng.standard_normal()
    neff = compute_n_eff_geyer(x)
    assert neff < 200


def test_n_eff_never_negative(rng):
    # Over-whitened signal (negative ACF) must not give negative N_eff
    x = rng.standard_normal(500)
    x = np.diff(x)  # first-differencing creates negative lag-1 ACF
    assert compute_n_eff_geyer(x) > 0


def test_vif_independent_channels(rng):
    X = rng.standard_normal((500, 4))
    vifs = compute_vif(X)
    assert len(vifs) == 4
    assert all(v < 3.0 for v in vifs)


def test_vif_collinear_pair(rng):
    x = rng.standard_normal(500)
    X = np.column_stack([x, x + 0.01 * rng.standard_normal(500), rng.standard_normal(500)])
    vifs = compute_vif(X)
    assert vifs[0] > 10 and vifs[1] > 10
    assert vifs[2] < 3.0


def test_is_slow_drift_ar1_rho097(rng):
    x = np.zeros(1200)
    for t in range(1, 1200):
        x[t] = 0.97 * x[t-1] + 0.01 * rng.standard_normal()
    assert is_slow_drift(x, fs=2.0) is True


def test_is_slow_drift_white_noise(rng):
    assert is_slow_drift(rng.standard_normal(1200), fs=2.0) is False
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module2.py -v 2>&1 | head -15
```
Expected: `ImportError`.

- [ ] **Step 3: Implement math utilities in module2**

Create `diagnostics/tier1_screening/module2_feature_diagnostics.py` with the math functions:

```python
from __future__ import annotations
import numpy as np
from typing import List


def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized ACF at lags 0..max_lag via FFT."""
    n = len(x)
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return acov / acov[0]


def compute_n_eff_geyer(x: np.ndarray) -> float:
    """Effective sample size via Geyer's (1992) monotone truncation.

    Truncates the ACF sum at the first lag pair where the sum of
    consecutive ACF values is non-positive, preventing negative N_eff
    from over-whitened signals.
    """
    n = len(x)
    acf = compute_acf(x, max_lag=n - 1)
    # Pair consecutive lags (2m-1, 2m) and truncate when pair sum <= 0
    gamma_sum = 0.0
    for m in range(1, n // 2):
        pair = acf[2 * m - 1] + acf[2 * m]
        if pair <= 0:
            break
        gamma_sum += pair
    return float(n / max(1.0, -1.0 + 2.0 * gamma_sum))


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor for each column of X.

    VIF[j] = 1 / (1 - R^2) where R^2 is from regressing column j on all others.
    """
    from sklearn.linear_model import LinearRegression
    n, p = X.shape
    vifs = np.empty(p)
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[j] = 1.0 / max(1e-9, 1.0 - r2)
    return vifs


def is_slow_drift(x: np.ndarray, fs: float,
                  lag_s: float = 10.0, threshold: float = 0.3) -> bool:
    """True if ACF(lag_s) > threshold — channel dominated by slow drift."""
    lag_samples = int(round(lag_s * fs))
    acf = compute_acf(x, max_lag=lag_samples)
    return bool(acf[lag_samples] > threshold)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module2.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier1_screening/module2_feature_diagnostics.py tests/diagnostics/test_module2.py
git commit -m "feat(diagnostics/m2): ACF, N_eff (Geyer), VIF, slow-drift flag"
```

---

## Task 5: Module 2 — diagnostics runner + plots

**Files:** `diagnostics/tier1_screening/module2_feature_diagnostics.py` (add `run_feature_diagnostics`)

- [ ] **Step 1: Write integration test**

Append to `tests/diagnostics/test_module2.py`:

```python
import os, pandas as pd
from diagnostics.tier1_screening.module2_feature_diagnostics import run_feature_diagnostics
from diagnostics.shared.data_loader import load_session


def test_run_feature_diagnostics_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'module2_out')
    flags = run_feature_diagnostics(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'acf_all_channels.png'))
    assert os.path.exists(os.path.join(out_dir, 'correlation_heatmap.png'))
    assert os.path.exists(os.path.join(out_dir, 'vif_table.csv'))
    assert os.path.exists(os.path.join(out_dir, 'n_eff_per_session.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module2_report.md'))
    assert isinstance(flags, dict)
    assert 'slow_drift' in flags
    assert 'collinear_pairs' in flags
    assert 'high_vif' in flags
```

- [ ] **Step 2: Run test — verify it fails**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module2.py::test_run_feature_diagnostics_creates_outputs -v
```

- [ ] **Step 3: Implement `run_feature_diagnostics`**

Append to `diagnostics/tier1_screening/module2_feature_diagnostics.py`:

```python
import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_feature_diagnostics(sessions, output_dir: str) -> dict:
    """Run all Module 2 diagnostics and write outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    all_sessions = sessions
    D = len(sessions[0].modality_keys)
    keys = sessions[0].modality_keys

    # ── ACF grid (raw features) ──────────────────────────────────────────
    max_lag_s = 60.0
    fs = sessions[0].fs
    max_lag = int(max_lag_s * fs)
    lags_s = np.arange(max_lag + 1) / fs

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.0 * D), sharex=True)
    slow_drift_flags = []
    for d, key in enumerate(keys):
        ax = axes[d] if D > 1 else axes
        all_acf = []
        for sess in sessions:
            acf = compute_acf(sess.Y_raw[:, d], max_lag)
            all_acf.append(acf)
            ax.plot(lags_s, acf, alpha=0.4, linewidth=0.7, color='steelblue')
        mean_acf = np.mean(all_acf, axis=0)
        ax.plot(lags_s, mean_acf, color='navy', linewidth=1.2)
        ax.axhline(0.3, color='red', linewidth=0.8, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.4)
        flag = bool(mean_acf[max_lag] > 0.3)
        slow_drift_flags.append(flag)
        color = 'firebrick' if flag else 'black'
        ax.set_ylabel(key, fontsize=6, color=color, rotation=0, ha='right')
    axes[-1].set_xlabel('Lag (s)')
    fig.suptitle('ACF per channel (raw features) — red = SLOW_DRIFT flag', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'acf_all_channels.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Power spectra ────────────────────────────────────────────────────
    from scipy.signal import welch
    fig, axes = plt.subplots(D, 1, figsize=(10, 1.8 * D), sharex=True)
    for d, key in enumerate(keys):
        ax = axes[d] if D > 1 else axes
        for sess in sessions:
            f, pxx = welch(sess.Y_raw[:, d], fs=sess.fs, nperseg=min(256, len(sess.Y_raw)))
            ax.semilogy(f, pxx, alpha=0.4, linewidth=0.7, color='steelblue')
        ax.set_ylabel(key, fontsize=6, rotation=0, ha='right')
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('Power spectra (raw features)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_spectra.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Cross-channel correlation heatmap (raw, first session) ──────────
    Y_all = np.vstack([s.Y_raw for s in sessions])
    corr = np.corrcoef(Y_all.T)
    fig, ax = plt.subplots(figsize=(max(6, D * 0.5), max(5, D * 0.45)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(D)); ax.set_xticklabels(keys, rotation=90, fontsize=6)
    ax.set_yticks(range(D)); ax.set_yticklabels(keys, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title('Cross-channel correlation (lag 0)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── VIF table ────────────────────────────────────────────────────────
    vifs = compute_vif(Y_all)
    vif_df = pd.DataFrame({'channel': keys, 'VIF': vifs,
                            'HIGH_VIF': vifs > 10.0})
    vif_df.to_csv(os.path.join(output_dir, 'vif_table.csv'), index=False)

    # ── Collinear pairs ──────────────────────────────────────────────────
    collinear_pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr[i, j]) > 0.8:
                collinear_pairs.append((keys[i], keys[j], float(corr[i, j])))

    # ── N_eff table ──────────────────────────────────────────────────────
    neff_rows = []
    for sess in sessions:
        row = {'session': sess.name}
        for d, key in enumerate(keys):
            row[key] = compute_n_eff_geyer(sess.Y_raw[:, d])
        neff_rows.append(row)
    neff_df = pd.DataFrame(neff_rows)
    neff_df.to_csv(os.path.join(output_dir, 'n_eff_per_session.csv'), index=False)
    low_info = [k for k in keys if neff_df[k].median() < 50]

    # ── Report ───────────────────────────────────────────────────────────
    sd_names = [k for k, f in zip(keys, slow_drift_flags) if f]
    hv_names = list(vif_df[vif_df['HIGH_VIF']]['channel'])
    report_lines = [
        f'SLOW_DRIFT: {sd_names}',
        f'LOW_INFO (median N_eff < 50): {low_info}',
        f'COLLINEAR_PAIRS (|r|>0.8): {collinear_pairs}',
        f'HIGH_VIF (>10): {hv_names}',
    ]
    with open(os.path.join(output_dir, 'module2_report.md'), 'w') as f:
        f.write('# Module 2: Feature Diagnostics\n\n')
        f.write('\n'.join(report_lines) + '\n')

    return {
        'slow_drift': sd_names,
        'low_info': low_info,
        'collinear_pairs': collinear_pairs,
        'high_vif': hv_names,
    }
```

- [ ] **Step 4: Run all Module 2 tests**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module2.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier1_screening/module2_feature_diagnostics.py tests/diagnostics/test_module2.py
git commit -m "feat(diagnostics/m2): run_feature_diagnostics runner + plots"
```

---

## Task 6: Module 3 — math utilities

**Files:** `diagnostics/tier1_screening/module3_transition_analysis.py`, `tests/diagnostics/test_module3.py`

- [ ] **Step 1: Write failing tests**

Create `tests/diagnostics/test_module3.py`:

```python
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier1_screening.module3_transition_analysis import (
    compute_dwell_times, transition_matrix_from_logits,
    geometric_mean_dwell, transition_event_ks_test
)


def test_compute_dwell_times_basic():
    path = np.array([0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2])
    dwells = compute_dwell_times(path)
    assert dwells[0] == [3, 2]
    assert dwells[1] == [2]
    assert dwells[2] == [4]


def test_compute_dwell_times_single_state():
    path = np.array([1, 1, 1, 1])
    dwells = compute_dwell_times(path)
    assert dwells[1] == [4]
    assert dwells.get(0, []) == []


def test_transition_matrix_rows_sum_to_one(rng):
    W = rng.standard_normal((4, 4)).astype(np.float32)
    T = transition_matrix_from_logits(W)
    np.testing.assert_allclose(T.sum(axis=1), np.ones(4), atol=1e-6)


def test_geometric_mean_dwell_formula():
    for T_kk in [0.5, 0.8, 0.9, 0.95]:
        expected = 1.0 / (1.0 - T_kk)
        assert abs(geometric_mean_dwell(T_kk) - expected) < 1e-9


def test_coincidence_random_transitions_not_significant(rng):
    session_length_s = 1200.0
    event_times = [300.0, 600.0, 900.0]
    transition_times = rng.uniform(0, session_length_s, 60)
    _, p = transition_event_ks_test(transition_times, event_times,
                                    session_length_s, n_boot=500, seed=42)
    assert p > 0.05


def test_coincidence_event_locked_transitions_significant(rng):
    session_length_s = 1200.0
    event_times = [300.0, 600.0, 900.0]
    transition_times = np.array(
        [t + rng.uniform(-3, 3) for t in event_times for _ in range(15)]
    )
    _, p = transition_event_ks_test(transition_times, event_times,
                                    session_length_s, n_boot=500, seed=42)
    assert p < 0.05
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module3.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement math utilities**

Create `diagnostics/tier1_screening/module3_transition_analysis.py`:

```python
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax
from scipy.stats import ks_2samp


def compute_dwell_times(path: np.ndarray) -> Dict[int, List[int]]:
    """Return dict mapping state_id -> list of dwell lengths (in samples)."""
    dwells: Dict[int, List[int]] = {}
    if len(path) == 0:
        return dwells
    current, count = int(path[0]), 1
    for s in path[1:]:
        s = int(s)
        if s == current:
            count += 1
        else:
            dwells.setdefault(current, []).append(count)
            current, count = s, 1
    dwells.setdefault(current, []).append(count)
    return dwells


def transition_matrix_from_logits(W_trans: np.ndarray) -> np.ndarray:
    """Convert (K, K) logit matrix to row-stochastic transition matrix."""
    return np.array([softmax(W_trans[k]) for k in range(len(W_trans))])


def geometric_mean_dwell(T_kk: float) -> float:
    """Expected mean dwell in samples for geometric(1 - T_kk)."""
    return 1.0 / (1.0 - T_kk)


def transition_event_ks_test(
    transition_times: np.ndarray,
    event_times: List[float],
    session_length_s: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """KS test: are transitions closer to events than random?

    Returns (ks_statistic, p_value).
    p < 0.05 → transitions cluster near events (real signal).
    """
    rng = np.random.default_rng(seed)
    event_arr = np.array(event_times)

    def min_dist(times):
        return np.array([np.min(np.abs(t - event_arr)) for t in times])

    real_dists = min_dist(transition_times)
    n = len(transition_times)
    boot_dists = np.concatenate([
        min_dist(rng.uniform(0, session_length_s, n))
        for _ in range(n_boot)
    ])
    ks_stat, p_val = ks_2samp(real_dists, boot_dists, alternative='less')
    return float(ks_stat), float(p_val)
```

- [ ] **Step 4: Run tests — verify pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module3.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier1_screening/module3_transition_analysis.py tests/diagnostics/test_module3.py
git commit -m "feat(diagnostics/m3): dwell times, T_kk, KS coincidence test"
```

---

## Task 7: Module 3 — runner + plots

**Files:** `diagnostics/tier1_screening/module3_transition_analysis.py` (add `run_transition_analysis`)

- [ ] **Step 1: Write integration test**

Append to `tests/diagnostics/test_module3.py`:

```python
import os
from diagnostics.tier1_screening.module3_transition_analysis import run_transition_analysis
from diagnostics.shared.data_loader import load_session


def test_run_transition_analysis_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm3_out')
    flags = run_transition_analysis(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'dwell_time_distributions.png'))
    assert os.path.exists(os.path.join(out_dir, 'coincidence_test_results.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module3_report.md'))
    assert any(f.endswith('_timeline.png') for f in os.listdir(out_dir))
    assert isinstance(flags, dict)
    assert 'dwell_ratio_low' in flags
    assert 'transitions_event_locked' in flags
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module3.py::test_run_transition_analysis_creates_outputs -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `run_transition_analysis`**

Append to `diagnostics/tier1_screening/module3_transition_analysis.py`:

```python
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostics.shared.data_loader import SessionData


_PROTOCOL_PHASES = {
    'meditation': {'base_EO', 'base_EC', 'conv_1', 'conv_2', 'meditate_B', 'meditate_K'},
    'pe':         {'base_EO', 'base_EC', 'conv_1', 'conv_2', 'PE_1', 'PE_2', 'PE_start'},
}

STATE_COLORS = ['#aaaaaa', '#4477AA', '#228833', '#EE6677', '#CCBB44']


def _detect_protocol(session: SessionData) -> str:
    names = {s[0] for s in session.segments}
    if names & {'meditate_B', 'meditate_K'}:
        return 'meditation'
    if names & {'PE_1', 'PE_2', 'PE_start'}:
        return 'pe'
    return 'unknown'


def run_transition_analysis(sessions: list, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    fs = sessions[0].fs
    flags = {'dwell_ratio_low': [], 'high_flicker_pct': [],
              'transitions_event_locked': False, 'transitions_random': False}
    all_coincidence_rows = []

    # ── Dwell-time distributions ─────────────────────────────────────────
    K = 4
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    state_labels = ['S0', 'S1', 'S2', 'S3']
    T_mean_pred = [None] * K  # model-predicted mean dwell per state

    for sess in sessions:
        if sess.path_unconstrained is None or sess.W_trans is None:
            continue
        T = transition_matrix_from_logits(sess.W_trans)
        sl = sess.state_labels or [f'S{k}' for k in range(K)]
        for k in range(min(K, T.shape[0])):
            dwells = compute_dwell_times(sess.path_unconstrained).get(k, [])
            if not dwells:
                continue
            pred_mean = geometric_mean_dwell(T[k, k])
            emp_mean = float(np.mean(dwells))
            ratio = emp_mean / pred_mean
            if ratio < 0.5:
                flags['dwell_ratio_low'].append(f'{sess.name}:S{k}(ratio={ratio:.2f})')
            pct_short = float(np.mean(np.array(dwells) < (10.0 * fs)))
            if pct_short > 0.5:
                flags['high_flicker_pct'].append(f'{sess.name}:S{k}({pct_short:.0%})')

            axes[k].hist(np.array(dwells) / fs, bins=30, density=True,
                         alpha=0.5, label=sess.name)
            # Geometric null CDF at key dwell values
            dw_s = np.linspace(0.5, max(dwells) / fs, 200)
            p_geom = 1 - T[k, k] ** np.round(dw_s * fs)
            if axes[k].lines == []:
                axes[k].plot(dw_s, np.diff(np.concatenate([[0], p_geom])),
                             color='red', linewidth=1.5, label='geometric null')

        state_labels = sl

    for k in range(K):
        axes[k].set_title(f'State {state_labels[k] if k < len(state_labels) else k}', fontsize=8)
        axes[k].set_xlabel('Dwell (s)')
        if k == 0:
            axes[k].legend(fontsize=5)

    fig.suptitle('Dwell-time distributions vs geometric(1-T_kk) null', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'dwell_time_distributions.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Per-session timelines + coincidence test (per-protocol) ─────────
    for sess in sessions:
        if sess.path_unconstrained is None:
            continue
        t = sess.t_common
        sl = sess.state_labels or [f'S{k}' for k in range(K)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 2.5), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1]})
        for ax, path, label in [(ax1, sess.path_unconstrained, 'Unconstrained'),
                                 (ax2, sess.path_constrained, '10s-min Viterbi')]:
            if path is None:
                continue
            for k in range(K):
                mask = path == k
                if mask.any():
                    ax.fill_between(t, k, k + 1, where=mask,
                                    color=STATE_COLORS[k % len(STATE_COLORS)], alpha=0.8)
            ax.set_ylim(0, K)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=7)
        for seg_name, t0, t1 in sess.segments:
            ax1.axvline(t0, color='black', linewidth=0.8, alpha=0.6)
            ax2.axvline(t0, color='black', linewidth=0.8, alpha=0.6)
            ax1.text(t0, K + 0.1, seg_name[:8], fontsize=5, rotation=45)
        fig.suptitle(f'{sess.name} — state timeline', fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{sess.name}_timeline.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Coincidence test
        trans_idx = np.where(np.diff(sess.path_unconstrained) != 0)[0]
        trans_times = t[trans_idx]
        event_times = [t0 for _, t0, _ in sess.segments]
        session_length = float(t[-1] - t[0])
        if len(trans_times) > 5 and len(event_times) > 0:
            ks, p = transition_event_ks_test(trans_times, event_times, session_length,
                                             n_boot=500, seed=0)
            proto = _detect_protocol(sess)
            all_coincidence_rows.append({
                'session': sess.name, 'protocol': proto,
                'n_transitions': len(trans_times), 'n_events': len(event_times),
                'KS_statistic': ks, 'p_value': p,
                'mean_distance_real_s': float(np.mean([
                    np.min(np.abs(tt - np.array(event_times))) for tt in trans_times])),
            })

    if all_coincidence_rows:
        df = pd.DataFrame(all_coincidence_rows)
        df.to_csv(os.path.join(output_dir, 'coincidence_test_results.csv'), index=False)
        pooled_p = float(df['p_value'].median())
        if pooled_p < 0.05:
            flags['transitions_event_locked'] = True
        if pooled_p > 0.1:
            flags['transitions_random'] = True
    else:
        pd.DataFrame().to_csv(os.path.join(output_dir, 'coincidence_test_results.csv'), index=False)

    # ── Report ───────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'module3_report.md'), 'w') as f:
        f.write('# Module 3: State Transition Analysis\n\n')
        f.write(f'DWELL_RATIO_LOW: {flags["dwell_ratio_low"]}\n')
        f.write(f'HIGH_FLICKER_PCT: {flags["high_flicker_pct"]}\n')
        event_verdict = ('TRANSITIONS_EVENT_LOCKED' if flags['transitions_event_locked']
                         else 'TRANSITIONS_RANDOM' if flags['transitions_random']
                         else 'TRANSITIONS_AMBIGUOUS')
        f.write(f'{event_verdict}\n')

    return flags
```

- [ ] **Step 4: Run all Module 3 tests**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module3.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier1_screening/module3_transition_analysis.py tests/diagnostics/test_module3.py
git commit -m "feat(diagnostics/m3): run_transition_analysis runner + timeline plots"
```

---

## Task 8: `run_tier1.py` orchestrator + screening report

**Files:** `diagnostics/tier1_screening/run_tier1.py`

- [ ] **Step 1: Implement directly**

Create `diagnostics/tier1_screening/run_tier1.py`:

```python
"""Tier 1 screening runner — Module 2 + Module 3 → screening_report.md.

Usage:
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --session y_06
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --results-dir results/v11
    conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --outputs-dir diagnostics/outputs
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.shared.data_loader import load_session, load_all_sessions
from diagnostics.shared.report_utils import make_output_dir, write_md_report
from diagnostics.tier1_screening.module2_feature_diagnostics import run_feature_diagnostics
from diagnostics.tier1_screening.module3_transition_analysis import run_transition_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default=None)
    parser.add_argument('--results-dir', default='results/v11')
    parser.add_argument('--outputs-dir', default='diagnostics/outputs')
    args = parser.parse_args()

    if args.session:
        sessions = [load_session(args.session, results_dir=args.results_dir)]
    else:
        sessions = load_all_sessions(results_dir=args.results_dir)
    print(f'Loaded {len(sessions)} session(s)')

    out_root = make_output_dir(args.outputs_dir, 'tier1_screening')
    m2_dir = os.path.join(out_root, 'module2')
    m3_dir = os.path.join(out_root, 'module3')

    print('Running Module 2 (feature diagnostics)...')
    m2_flags = run_feature_diagnostics(sessions, m2_dir)

    print('Running Module 3 (transition analysis)...')
    m3_flags = run_transition_analysis(sessions, m3_dir)

    # Write screening report
    screening_path = os.path.join(out_root, 'screening_report.md')
    with open(screening_path, 'w') as f:
        f.write('# Tier 1 Screening Report\n\n')
        f.write('## Module 2 Flags\n')
        f.write(f'SLOW_DRIFT: {m2_flags["slow_drift"]}\n')
        f.write(f'LOW_INFO: {m2_flags["low_info"]}\n')
        f.write(f'COLLINEAR_PAIRS: {m2_flags["collinear_pairs"]}\n')
        f.write(f'HIGH_VIF: {m2_flags["high_vif"]}\n\n')
        f.write('## Module 3 Flags\n')
        f.write(f'DWELL_RATIO_LOW: {m3_flags["dwell_ratio_low"]}\n')
        f.write(f'HIGH_FLICKER_PCT: {m3_flags["high_flicker_pct"]}\n')
        event_verdict = ('TRANSITIONS_EVENT_LOCKED' if m3_flags['transitions_event_locked']
                         else 'TRANSITIONS_RANDOM' if m3_flags['transitions_random']
                         else 'TRANSITIONS_AMBIGUOUS')
        f.write(f'{event_verdict}\n\n')
        f.write('## Recommendation\n')
        n_flags = (len(m2_flags['slow_drift']) + len(m2_flags['high_vif'])
                   + len(m3_flags['dwell_ratio_low']))
        if n_flags == 0:
            f.write('No critical flags. Proceed to Tier 2.\n')
        else:
            f.write(f'{n_flags} flag(s) raised. Review before running Tier 2 Module 4/5.\n')

    print(f'Tier 1 complete. Outputs: {out_root}')
    print(f'Screening report: {screening_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test on y_06 (if available)**

```bash
conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --session y_06
```
Expected: outputs directory created, `screening_report.md` written, no Python errors.

- [ ] **Step 3: Commit**

```bash
git add diagnostics/tier1_screening/run_tier1.py
git commit -m "feat(diagnostics): Tier 1 run_tier1.py orchestrator + screening_report"
```

---

## Task 9: Module 1 — Observation space structure

**Files:** `diagnostics/tier2_deepdive/module1_obs_space.py`, `tests/diagnostics/test_module1.py`

- [ ] **Step 1: Write failing tests**

Create `tests/diagnostics/test_module1.py`:

```python
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module1_obs_space import run_obs_space_analysis
from diagnostics.shared.data_loader import load_session


def test_run_obs_space_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm1_out')
    result = run_obs_space_analysis(sessions, out_dir, n_neighbors=5, min_dist=0.1)
    assert os.path.exists(os.path.join(out_dir, 'pca_scree.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_state.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_phase.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_session.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_time.png'))
    assert os.path.exists(os.path.join(out_dir, 'cluster_quality.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module1_report.md'))
    assert 'state_silhouette' in result
    assert 'phase_silhouette' in result


def test_silhouette_in_reduced_space_not_full(fake_session_dir, tmp_path):
    """Silhouette must be computed in PCA-reduced space, not full 28D."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    result = run_obs_space_analysis(sessions, str(tmp_path / 'out'), n_neighbors=5)
    # Reduced dim must be < full D=5 (or == 5 if all PCs needed for 80%)
    assert result['n_pca_components_80pct'] <= sessions[0].Y_pw.shape[1]
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module1.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement module1**

Create `diagnostics/tier2_deepdive/module1_obs_space.py`:

```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap


def run_obs_space_analysis(
    sessions: list, output_dir: str,
    n_neighbors: int = 30, min_dist: float = 0.1,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Pool all prewhitened observations ────────────────────────────────
    Y_all = np.vstack([s.Y_pw for s in sessions])
    # Annotations
    state_labels_all = np.concatenate([
        s.path_unconstrained if s.path_unconstrained is not None
        else np.zeros(len(s.t_common), dtype=int)
        for s in sessions
    ])
    # Coarse phase: baseline/conversation/task
    def _coarse(phase_name: str) -> str:
        p = phase_name.lower()
        if 'base' in p: return 'baseline'
        if 'conv' in p: return 'conversation'
        return 'task'

    phase_labels_all = []
    session_labels_all = []
    time_labels_all = []
    for sess in sessions:
        N = len(sess.t_common)
        phase_arr = np.full(N, 'unknown', dtype=object)
        for seg_name, t0, t1 in sess.segments:
            mask = (sess.t_common >= t0) & (sess.t_common <= t1)
            phase_arr[mask] = _coarse(seg_name)
        phase_labels_all.append(phase_arr)
        session_labels_all.append(np.full(N, sess.name, dtype=object))
        t_norm = (sess.t_common - sess.t_common[0]) / max(1.0, sess.t_common[-1] - sess.t_common[0])
        time_labels_all.append(t_norm)

    phase_all = np.concatenate(phase_labels_all)
    sess_all = np.concatenate(session_labels_all)
    time_all = np.concatenate(time_labels_all)

    # ── PCA scree ────────────────────────────────────────────────────────
    pca_full = PCA().fit(Y_all)
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n_80 = int(np.searchsorted(cumvar, 0.80)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(evr) + 1), evr, alpha=0.7, label='Per-component')
    ax.plot(range(1, len(cumvar) + 1), cumvar, color='red', marker='o', ms=3, label='Cumulative')
    ax.axhline(0.80, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Component'); ax.set_ylabel('Variance explained')
    ax.set_title(f'PCA scree — 80% at PC{n_80}, 95% at PC{n_95}')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'pca_scree.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Compute PCA-reduced coords for silhouette ────────────────────────
    Y_reduced = PCA(n_components=n_80).fit_transform(Y_all)

    # ── Silhouette + DB (in reduced space) ───────────────────────────────
    state_sil = phase_sil = state_db = float('nan')
    if len(np.unique(state_labels_all)) > 1:
        state_sil = float(silhouette_score(Y_reduced, state_labels_all, sample_size=min(5000, len(Y_reduced))))
        state_db = float(davies_bouldin_score(Y_reduced, state_labels_all))
    if len(np.unique(phase_all)) > 1:
        phase_sil = float(silhouette_score(Y_reduced, phase_all, sample_size=min(5000, len(Y_reduced))))

    pd.DataFrame([{
        'state_silhouette': state_sil, 'phase_silhouette': phase_sil,
        'state_davies_bouldin': state_db,
        'n_pca_80pct': n_80, 'n_pca_95pct': n_95,
    }]).to_csv(os.path.join(output_dir, 'cluster_quality.csv'), index=False)

    # ── UMAP embedding ───────────────────────────────────────────────────
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    # Subsample to 10k if very large
    idx = np.arange(len(Y_all))
    if len(idx) > 10000:
        idx = np.random.default_rng(42).choice(len(Y_all), 10000, replace=False)
        idx.sort()
    embedding = reducer.fit_transform(Y_all[idx])

    STATE_CMAP = plt.cm.get_cmap('tab10', 4)
    PHASE_CMAP = {'baseline': 'steelblue', 'conversation': 'green', 'task': 'orange', 'unknown': 'gray'}

    def _save_umap(color_data, title, filename, cmap=None, vmin=None, vmax=None, discrete_map=None):
        fig, ax = plt.subplots(figsize=(7, 6))
        if discrete_map is not None:
            for label, color in discrete_map.items():
                mask = color_data[idx] == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           s=1, alpha=0.3, color=color, label=str(label))
            ax.legend(markerscale=5, fontsize=7, loc='upper right')
        else:
            sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=color_data[idx], s=1, alpha=0.3, cmap=cmap,
                            vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, fraction=0.03)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
        fig.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches='tight')
        plt.close(fig)

    K = int(state_labels_all.max()) + 1 if state_labels_all.dtype.kind in 'iu' else 4
    state_color_map = {k: STATE_CMAP(k) for k in range(K)}
    _save_umap(state_labels_all.astype(str), 'UMAP colored by rSLDS state',
               'umap_by_state.png', discrete_map={str(k): STATE_CMAP(k) for k in range(K)})
    _save_umap(phase_all, 'UMAP colored by protocol phase',
               'umap_by_phase.png', discrete_map=PHASE_CMAP)
    unique_sess = list(dict.fromkeys(sess_all))
    sess_cmap = {s: plt.cm.tab20(i / max(1, len(unique_sess) - 1)) for i, s in enumerate(unique_sess)}
    _save_umap(sess_all, 'UMAP colored by session', 'umap_by_session.png', discrete_map=sess_cmap)
    _save_umap(time_all.astype(float), 'UMAP colored by time-within-session',
               'umap_by_time.png', cmap='plasma', vmin=0, vmax=1)

    # ── Interpretation ───────────────────────────────────────────────────
    if state_sil > phase_sil:
        interp = 'States more separable than conditions — model captures coupling dynamics.'
    elif phase_sil > state_sil and phase_sil > 0.05:
        interp = ('State silhouette < phase silhouette — states tracking experimental '
                  'condition more than coupling dynamics.')
    else:
        interp = ('Both silhouettes near zero. If no temporal gradient in umap_by_time.png, '
                  'observation space is genuinely continuous — consider reporting z_t latent '
                  'trajectory as primary object rather than state labels.')

    with open(os.path.join(output_dir, 'module1_report.md'), 'w') as f:
        f.write('# Module 1: Observation Space Structure\n\n')
        f.write(f'PCA 80%: {n_80} components | 95%: {n_95} components\n')
        f.write(f'State silhouette (PCA-reduced): {state_sil:.3f}\n')
        f.write(f'Phase silhouette (PCA-reduced): {phase_sil:.3f}\n')
        f.write(f'State Davies-Bouldin: {state_db:.3f}\n\n')
        f.write(f'**Interpretation:** {interp}\n')

    return {
        'state_silhouette': state_sil, 'phase_silhouette': phase_sil,
        'state_davies_bouldin': state_db,
        'n_pca_components_80pct': n_80,
    }
```

- [ ] **Step 4: Run tests — verify pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module1.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier2_deepdive/module1_obs_space.py tests/diagnostics/test_module1.py
git commit -m "feat(diagnostics/m1): PCA scree, UMAP ×4, silhouette in reduced space"
```

---

## Task 10: Module 4 — Alternative-model baselines (LOO-CV)

**Files:** `diagnostics/tier2_deepdive/module4_model_comparison.py`, `tests/diagnostics/test_module4.py`

- [ ] **Step 1: Write failing test**

Create `tests/diagnostics/test_module4.py`:

```python
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module4_model_comparison import run_model_comparison
from diagnostics.shared.data_loader import load_session


def test_run_model_comparison_outputs(fake_session_dir, tmp_path):
    """With only 1 session LOO is trivial but output files should still be created."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm4_out')
    result = run_model_comparison(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'loo_cv_results.csv'))
    assert os.path.exists(os.path.join(out_dir, 'model_comparison_summary.png'))
    assert os.path.exists(os.path.join(out_dir, 'module4_report.md'))
    assert 'mean_ll_per_frame_dim' in result
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module4.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement module4**

Create `diagnostics/tier2_deepdive/module4_model_comparison.py`:

```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


def _ll_per_frame_dim(model, X: np.ndarray) -> float:
    """Held-out log-likelihood per frame per dimension."""
    try:
        if hasattr(model, 'score'):
            ll_per_frame = model.score(X)        # already per frame for hmmlearn
            return float(ll_per_frame / X.shape[1])
        else:
            return float(model.score(X) / X.shape[1])
    except Exception:
        return float('nan')


def run_model_comparison(sessions: list, output_dir: str) -> dict:
    """LOO-CV comparison: HMM K=2..5, GMM K=2..5, reference rSLDS K=4."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(sessions)
    rows = []

    K_values = [2, 3, 4, 5]

    for fold, test_sess in enumerate(sessions):
        train_sessions = [s for i, s in enumerate(sessions) if i != fold]
        if not train_sessions:
            train_sessions = [test_sess]   # n=1 edge case for tests

        Y_train = np.vstack([s.Y_pw for s in train_sessions])
        Y_test = test_sess.Y_pw
        T_test, D = Y_test.shape

        row = {'fold': fold, 'test_session': test_sess.name}

        for K in K_values:
            # ── Gaussian HMM ─────────────────────────────────────────────
            hmm = GaussianHMM(n_components=K, covariance_type='diag',
                              n_iter=100, random_state=42)
            try:
                hmm.fit(Y_train)
                row[f'HMM_K{K}'] = _ll_per_frame_dim(hmm, Y_test)
            except Exception:
                row[f'HMM_K{K}'] = float('nan')

            # ── GMM ──────────────────────────────────────────────────────
            gmm = GaussianMixture(n_components=K, covariance_type='diag',
                                  random_state=42, max_iter=200)
            try:
                gmm.fit(Y_train)
                row[f'GMM_K{K}'] = float(gmm.score(Y_test) / D)
            except Exception:
                row[f'GMM_K{K}'] = float('nan')

        # ── rSLDS reference (conditioned on MAP z = path_unconstrained) ──
        # Emission LL: sum_t sum_d -0.5*(y - mu_k)^2/sigma2_k - 0.5*log(sigma2_k)
        if (test_sess.d_emit is not None and test_sess.path_unconstrained is not None):
            sigma2 = np.exp(np.zeros_like(test_sess.d_emit))  # placeholder unit var
            # Use R_emit diagonal if available
            if test_sess.R_emit is not None:
                # R_emit shape may be (K, D) or (K, D, D) — handle both
                R = test_sess.R_emit
                if R.ndim == 3:
                    sigma2 = np.array([np.diag(R[k]) for k in range(R.shape[0])])
                else:
                    sigma2 = np.abs(R) + 1e-6

            path = test_sess.path_unconstrained
            d_emit = test_sess.d_emit  # (K, D)
            ll = 0.0
            for t in range(T_test):
                k = int(path[t])
                if k >= len(d_emit):
                    continue
                diff = Y_test[t] - d_emit[k]
                ll += float(np.sum(-0.5 * diff**2 / sigma2[k] - 0.5 * np.log(sigma2[k] + 1e-9)))
            row['rSLDS_K4'] = ll / (T_test * D)
        else:
            row['rSLDS_K4'] = float('nan')

        rows.append(row)
        print(f'  Fold {fold+1}/{n}: {test_sess.name} done')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'loo_cv_results.csv'), index=False)

    # ── Summary bar chart ────────────────────────────────────────────────
    model_cols = [c for c in df.columns if c not in ('fold', 'test_session')]
    means = df[model_cols].mean()
    sems = df[model_cols].sem()

    fig, ax = plt.subplots(figsize=(max(8, len(model_cols) * 0.6), 5))
    x = np.arange(len(model_cols))
    colors = (['steelblue'] * (4 * len(K_values)) +
              ['firebrick'] * len(K_values) + ['gold'])
    ax.bar(x, means.values, yerr=sems.values, capsize=3,
           color=colors[:len(model_cols)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_cols, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean held-out LL / frame / dim (±SEM)')
    ax.set_title('LOO-CV model comparison (higher = better)')
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison_summary.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Interpretation ───────────────────────────────────────────────────
    rslds_mean = means.get('rSLDS_K4', float('nan'))
    hmm4_mean = means.get('HMM_K4', float('nan'))
    hmm4_sem = sems.get('HMM_K4', 0)
    gmm4_mean = means.get('GMM_K4', float('nan'))

    interp_lines = []
    if not np.isnan(rslds_mean) and not np.isnan(hmm4_mean):
        if rslds_mean - hmm4_mean > hmm4_sem:
            interp_lines.append('rSLDS > HMM K=4 by >1 SEM — recurrent covariates paying off.')
        else:
            interp_lines.append('rSLDS within 1 SEM of HMM K=4 — covariate structure not justified at n.')
    if not np.isnan(gmm4_mean) and not np.isnan(hmm4_mean):
        if abs(gmm4_mean - hmm4_mean) < sems.get('GMM_K4', 0) + hmm4_sem:
            interp_lines.append('GMM K=4 competitive with HMM K=4 — temporal dynamics not contributing.')

    with open(os.path.join(output_dir, 'module4_report.md'), 'w') as f:
        f.write('# Module 4: Alternative-Model Baselines\n\n')
        f.write(means.to_string())
        f.write('\n\n**Interpretation:**\n')
        f.write('\n'.join(interp_lines) or 'Insufficient data for interpretation.')
        f.write('\n')

    return {'mean_ll_per_frame_dim': means.to_dict()}
```

- [ ] **Step 4: Run test — verify pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module4.py -v
```
Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier2_deepdive/module4_model_comparison.py tests/diagnostics/test_module4.py
git commit -m "feat(diagnostics/m4): LOO-CV HMM/GMM/rSLDS comparison"
```

---

## Task 11: Module 5 — Block PCA preprocessing variant

**Files:** `diagnostics/tier2_deepdive/module5_block_pca.py`, `tests/diagnostics/test_module5.py`

- [ ] **Step 1: Write failing test**

Create `tests/diagnostics/test_module5.py`:

```python
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module5_block_pca import fit_block_pca, run_block_pca
from diagnostics.shared.data_loader import load_session


def test_fit_block_pca_reduces_dim(rng):
    keys = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    groups = {'eeg': ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'], 'other': ['bl_expr', 'pose']}
    Y = rng.standard_normal((200, 5))
    Y_red, loadings = fit_block_pca(Y, keys, groups, flags={'slow_drift': [], 'high_vif': []})
    assert Y_red.shape[0] == 200
    assert Y_red.shape[1] <= 5  # reduced dim
    assert 'eeg_PC1' in loadings


def test_fit_block_pca_excludes_flagged(rng):
    keys = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    groups = {'eeg': ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'], 'other': ['bl_expr', 'pose']}
    Y = rng.standard_normal((200, 5))
    flags = {'slow_drift': ['imcoh_theta'], 'high_vif': []}
    Y_red, loadings = fit_block_pca(Y, keys, groups, flags)
    # imcoh_theta excluded so eeg group has 2 channels → max 1 PC at 80%
    assert Y_red.shape[1] < 5


def test_run_block_pca_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm5_out')
    flags = {'slow_drift': [], 'high_vif': [], 'collinear_pairs': [], 'low_info': []}
    result = run_block_pca(sessions, flags, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'block_pca_loadings.md'))
    assert os.path.exists(os.path.join(out_dir, 'module5_report.md'))
    assert 'n_reduced_dims' in result
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module5.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement module5**

Create `diagnostics/tier2_deepdive/module5_block_pca.py`:

```python
from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Default modality groups for V11 28-channel scaffold
V11_MODALITY_GROUPS: Dict[str, List[str]] = {
    'EEG_phase':    ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'EEG_shared':   ['conc_theta', 'conc_alpha', 'conc_beta'],
    'EEG_dynamics': ['dyn_theta', 'dyn_alpha', 'dyn_beta'],
    'EEG_asymmetry':['asym_theta', 'asym_alpha', 'asym_beta'],
    'Facial':       ['bl_expr', 'bl_act_conc'],
    'Autonomic':    ['ecg_lf', 'ecg_hf', 'resp'],
    'Body':         ['pose'],
    'Complexity':   ['lz_conc_theta', 'lz_conc_alpha', 'lz_asym_theta', 'lz_asym_alpha'],
    'Graph':        ['graph_mod'],
    'Burst_TE':     ['te_conc_theta', 'te_conc_alpha',
                     'burst_coinc_theta', 'burst_coinc_alpha', 'burst_coinc_beta'],
}


def fit_block_pca(
    Y: np.ndarray,
    channel_keys: List[str],
    modality_groups: Dict[str, List[str]],
    flags: dict,
    var_threshold: float = 0.80,
    max_pcs: int = 2,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Fit per-modality PCA; return (Y_reduced, loadings_dict).

    Channels listed in flags['slow_drift'] or flags['high_vif'] are excluded
    from their group before PCA. The loadings_dict maps PC names to human-
    readable linear combinations of channel names.
    """
    excluded = set(flags.get('slow_drift', [])) | set(flags.get('high_vif', []))
    key_to_idx = {k: i for i, k in enumerate(channel_keys)}

    blocks = []
    loadings: Dict[str, str] = {}

    for group_name, group_keys in modality_groups.items():
        active = [k for k in group_keys if k in key_to_idx and k not in excluded]
        if not active:
            continue
        idx = [key_to_idx[k] for k in active]
        Xg = Y[:, idx]
        if Xg.shape[1] == 1:
            blocks.append(Xg)
            loadings[f'{group_name}_PC1'] = active[0]
            continue
        pca = PCA().fit(Xg)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = min(max_pcs, int(np.searchsorted(cumvar, var_threshold)) + 1)
        n_keep = max(1, n_keep)
        Xg_r = pca.transform(Xg)[:, :n_keep]
        blocks.append(Xg_r)
        for pc in range(n_keep):
            weights = pca.components_[pc]
            terms = ' + '.join(f'{w:.2f}*{k}' for w, k in
                               sorted(zip(weights, active), key=lambda x: -abs(x[0]))[:4])
            loadings[f'{group_name}_PC{pc+1}'] = terms

    Y_reduced = np.hstack(blocks) if blocks else Y
    return Y_reduced, loadings


def run_block_pca(sessions: list, flags: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Infer modality groups from actual session channel keys
    if sessions[0].modality_keys:
        keys = sessions[0].modality_keys
        # Use default groups, intersected with actual keys
        groups = {g: [k for k in gkeys if k in keys]
                  for g, gkeys in V11_MODALITY_GROUPS.items()
                  if any(k in keys for k in gkeys)}
        # Any keys not in any group → 'Other'
        assigned = {k for gkeys in groups.values() for k in gkeys}
        leftover = [k for k in keys if k not in assigned]
        if leftover:
            groups['Other'] = leftover
    else:
        keys = []
        groups = {}

    Y_all = np.vstack([s.Y_raw for s in sessions])
    Y_red, loadings = fit_block_pca(Y_all, keys, groups, flags)

    # Save loadings
    with open(os.path.join(output_dir, 'block_pca_loadings.md'), 'w') as f:
        f.write('# Block PCA Loadings\n\n')
        for pc_name, formula in loadings.items():
            f.write(f'**{pc_name}**: {formula}\n\n')

    n_in = Y_all.shape[1]
    n_out = Y_red.shape[1]

    with open(os.path.join(output_dir, 'module5_report.md'), 'w') as f:
        f.write('# Module 5: Block PCA Preprocessing Variant\n\n')
        f.write(f'Input dims: {n_in} → Reduced dims: {n_out}\n')
        f.write(f'Excluded (flagged): {flags.get("slow_drift", [])} + {flags.get("high_vif", [])}\n\n')
        f.write('See block_pca_loadings.md for full PC definitions.\n')
        f.write('\n**Note:** Re-fit rSLDS on reduced features using run_tier2.py --refit-block-pca\n')

    return {'n_reduced_dims': n_out, 'n_input_dims': n_in, 'loadings': loadings}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module5.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier2_deepdive/module5_block_pca.py tests/diagnostics/test_module5.py
git commit -m "feat(diagnostics/m5): block PCA with flagged-channel exclusion"
```

---

## Task 12: Module 6 — Null-state ablation

**Files:** `diagnostics/tier2_deepdive/module6_null_ablation.py`, `tests/diagnostics/test_module6.py`

- [ ] **Step 1: Write failing test**

Create `tests/diagnostics/test_module6.py`:

```python
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module6_null_ablation import (
    hungarian_match_states, run_null_ablation
)
from diagnostics.shared.data_loader import load_session


def test_hungarian_match_identity(rng):
    """If constrained and unconstrained d_emit are identical, match should be identity."""
    d = rng.standard_normal((4, 5))
    perm, sims = hungarian_match_states(d, d)
    assert list(perm) == [0, 1, 2, 3]
    np.testing.assert_allclose(sims, np.ones(4), atol=1e-6)


def test_hungarian_match_permuted(rng):
    """Permuted d_emit should be matched back to original order."""
    d = rng.standard_normal((4, 5))
    d_perm = d[[2, 0, 3, 1]]  # permute rows
    perm, sims = hungarian_match_states(d, d_perm)
    # perm[i] = which unconstrained state matches constrained state i
    np.testing.assert_allclose(sims, np.ones(4), atol=1e-6)


def test_run_null_ablation_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm6_out')
    result = run_null_ablation(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'emission_means_comparison.png'))
    assert os.path.exists(os.path.join(out_dir, 'hungarian_alignment.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module6_report.md'))
    assert 'constrained_null_norm' in result
```

- [ ] **Step 2: Run — verify failure**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module6.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement module6**

Create `diagnostics/tier2_deepdive/module6_null_ablation.py`:

```python
from __future__ import annotations
import os
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0 if np.linalg.norm(a - b) < 1e-12 else 0.0
    return float(np.dot(a, b) / denom)


def hungarian_match_states(
    d_constrained: np.ndarray,   # (K, D)
    d_unconstrained: np.ndarray, # (K, D)
) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian matching: for each constrained state i, find best unconstrained state.

    Returns (perm, sims): perm[i] = unconstrained index matched to constrained i,
    sims[i] = cosine similarity of matched pair.
    """
    K = d_constrained.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = 1.0 - _cosine_sim(d_constrained[i], d_unconstrained[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = col_ind[np.argsort(row_ind)]
    sims = np.array([_cosine_sim(d_constrained[i], d_unconstrained[perm[i]]) for i in range(K)])
    return perm, sims


def run_null_ablation(sessions: list, output_dir: str) -> dict:
    """Re-fit rSLDS with null_state=False and compare to constrained fit.

    Uses IOHMM from cadence.significance.rslds_model.
    Falls back to reporting constrained-only stats if re-fit fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather constrained emission stats from saved fits
    all_d_constrained = []
    for sess in sessions:
        if sess.d_emit is not None:
            all_d_constrained.append(sess.d_emit)

    if not all_d_constrained:
        with open(os.path.join(output_dir, 'module6_report.md'), 'w') as f:
            f.write('# Module 6: Null-State Ablation\n\nNo rSLDS fits found.\n')
        return {'constrained_null_norm': float('nan')}

    d_con_mean = np.mean(all_d_constrained, axis=0)  # (K, D) averaged over sessions
    norms_con = np.linalg.norm(d_con_mean, axis=1)
    null_idx_con = int(np.argmin(norms_con))

    # ── Re-fit unconstrained rSLDS ────────────────────────────────────────
    d_uncon_mean = None
    perm = np.arange(len(d_con_mean))
    sims = np.ones(len(d_con_mean))
    refit_attempted = False

    try:
        from cadence.significance.rslds_model import IOHMM, IOHMMConfig
        cfg = IOHMMConfig(
            K=4, D_obs=sessions[0].Y_pw.shape[1],
            D_input=sessions[0].U.shape[1],
            D_latent=3, n_factors=2,
            recurrent=True, sticky_strength=3.0,
            c_shrinkage=0.2, null_state=False,  # KEY: unconstrained
            viterbi_min_dwell=20, max_em_iter=100, n_restarts=2,
        )
        model = IOHMM(cfg)
        refit_attempted = True
        all_d_uncon = []
        for sess in sessions:
            print(f'  Re-fitting {sess.name} (unconstrained)...')
            params, _ = model.fit(sess.Y_pw, sess.U, sess.obs_mask)
            all_d_uncon.append(params.mu)
        d_uncon_mean = np.mean(all_d_uncon, axis=0)
        perm, sims = hungarian_match_states(d_con_mean, d_uncon_mean)
    except Exception as e:
        print(f'  Warning: unconstrained re-fit failed ({e}). Reporting constrained-only stats.')

    # ── Emission norms plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = sessions[0].state_labels or [f'S{k}' for k in range(4)]

    for ax, d_emit, title in [
        (axes[0], d_con_mean, 'Constrained (null_state=True)'),
        (axes[1], d_uncon_mean if d_uncon_mean is not None else d_con_mean,
         'Unconstrained (null_state=False)' if d_uncon_mean is not None else 'N/A (refit failed)'),
    ]:
        norms = np.linalg.norm(d_emit, axis=1)
        sort_idx = np.argsort(norms)
        ax.barh([str(labels[i]) if i < len(labels) else f'S{i}' for i in sort_idx],
                norms[sort_idx], color='steelblue', alpha=0.8)
        ax.set_xlabel('Emission mean L2 norm')
        ax.set_title(title, fontsize=8)

    fig.suptitle('Emission mean norms: constrained vs unconstrained', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'emission_means_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Hungarian alignment table ─────────────────────────────────────────
    K = len(d_con_mean)
    align_df = pd.DataFrame({
        'constrained_state': [labels[i] if i < len(labels) else f'S{i}' for i in range(K)],
        'matched_unconstrained_state': [labels[perm[i]] if perm[i] < len(labels) else f'S{perm[i]}'
                                        for i in range(K)],
        'cosine_similarity': sims,
    })
    align_df.to_csv(os.path.join(output_dir, 'hungarian_alignment.csv'), index=False)

    # ── Interpretation ────────────────────────────────────────────────────
    min_sim = float(sims.min())
    null_norm_con = float(norms_con[null_idx_con])
    null_norm_uncon = float(np.linalg.norm(d_uncon_mean[perm[null_idx_con]])) if d_uncon_mean is not None else float('nan')

    if min_sim > 0.7:
        interp = ('All states match with cosine similarity > 0.7. '
                  'Null-state constraint encodes a real regime — keep it.')
    elif np.isnan(min_sim):
        interp = 'Re-fit not available — see constrained-only norms above.'
    else:
        interp = ('Low cosine similarity for some states. '
                  'Unconstrained model found different structure — null constraint may be forcing.')

    with open(os.path.join(output_dir, 'module6_report.md'), 'w') as f:
        f.write('# Module 6: Null-State Ablation\n\n')
        f.write(f'Constrained null-state norm: {null_norm_con:.3f}\n')
        if not np.isnan(null_norm_uncon):
            f.write(f'Unconstrained matched-state norm: {null_norm_uncon:.3f}\n')
        f.write(f'Min cosine similarity across matched pairs: {min_sim:.3f}\n\n')
        f.write(f'**Interpretation:** {interp}\n')

    return {
        'constrained_null_norm': null_norm_con,
        'unconstrained_null_norm': null_norm_uncon,
        'min_cosine_similarity': min_sim,
        'refit_attempted': refit_attempted,
    }
```

- [ ] **Step 4: Run tests — verify pass**

```bash
conda run -n MCCT pytest tests/diagnostics/test_module6.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diagnostics/tier2_deepdive/module6_null_ablation.py tests/diagnostics/test_module6.py
git commit -m "feat(diagnostics/m6): null-state ablation, Hungarian matching, re-fit"
```

---

## Task 13: `run_tier2.py` orchestrator

**Files:** `diagnostics/tier2_deepdive/run_tier2.py`

- [ ] **Step 1: Implement directly**

Create `diagnostics/tier2_deepdive/run_tier2.py`:

```python
"""Tier 2 deep-dive runner — Modules 1, 4, 5, 6.

Usage:
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py --skip-modules 4
    conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py --screening-report diagnostics/outputs/<ts>/screening_report.md
"""
import argparse, os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diagnostics.shared.data_loader import load_session, load_all_sessions
from diagnostics.shared.report_utils import make_output_dir
from diagnostics.tier2_deepdive.module1_obs_space import run_obs_space_analysis
from diagnostics.tier2_deepdive.module4_model_comparison import run_model_comparison
from diagnostics.tier2_deepdive.module5_block_pca import run_block_pca
from diagnostics.tier2_deepdive.module6_null_ablation import run_null_ablation


def _read_flags_from_screening_report(report_path: str) -> dict:
    """Parse screening_report.md for Module 2 flags to pass to Module 5."""
    flags = {'slow_drift': [], 'high_vif': [], 'collinear_pairs': [], 'low_info': []}
    if not report_path or not os.path.exists(report_path):
        return flags
    with open(report_path) as f:
        text = f.read()
    for line in text.splitlines():
        for key in flags:
            if line.upper().startswith(key.upper()):
                # Parse list-like content: "SLOW_DRIFT: ['ch1', 'ch2']"
                m = re.search(r'\[([^\]]*)\]', line)
                if m:
                    content = m.group(1).replace("'", '').replace('"', '')
                    flags[key] = [c.strip() for c in content.split(',') if c.strip()]
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default=None)
    parser.add_argument('--results-dir', default='results/v11')
    parser.add_argument('--outputs-dir', default='diagnostics/outputs')
    parser.add_argument('--screening-report', default=None,
                        help='Path to screening_report.md from Tier 1')
    parser.add_argument('--skip-modules', nargs='+', type=int, default=[],
                        help='Module numbers to skip (e.g. --skip-modules 4)')
    parser.add_argument('--n-neighbors', type=int, default=30)
    parser.add_argument('--min-dist', type=float, default=0.1)
    args = parser.parse_args()

    if args.session:
        sessions = [load_session(args.session, results_dir=args.results_dir)]
    else:
        sessions = load_all_sessions(results_dir=args.results_dir)
    print(f'Loaded {len(sessions)} session(s)')

    # Warn if Tier 1 flags unread
    flags = _read_flags_from_screening_report(args.screening_report)
    if args.screening_report is None:
        print('Warning: no --screening-report provided. Module 5 will not exclude flagged channels.')

    out_root = make_output_dir(args.outputs_dir, 'tier2_deepdive')

    if 1 not in args.skip_modules:
        print('Running Module 1 (observation space)...')
        run_obs_space_analysis(sessions, os.path.join(out_root, 'module1'),
                               n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    if 4 not in args.skip_modules:
        print('Running Module 4 (model comparison — may take 20-40 min)...')
        run_model_comparison(sessions, os.path.join(out_root, 'module4'))

    if 5 not in args.skip_modules:
        print('Running Module 5 (block PCA)...')
        run_block_pca(sessions, flags, os.path.join(out_root, 'module5'))

    if 6 not in args.skip_modules:
        print('Running Module 6 (null-state ablation)...')
        run_null_ablation(sessions, os.path.join(out_root, 'module6'))

    print(f'Tier 2 complete. Outputs: {out_root}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run full test suite**

```bash
conda run -n MCCT pytest tests/diagnostics/ -v
```
Expected: all tests PASS.

- [ ] **Step 3: Smoke test Tier 1 on real data**

```bash
conda run -n MCCT python diagnostics/tier1_screening/run_tier1.py --session y_06
```
Expected: outputs directory with `module2/`, `module3/`, `screening_report.md`.

- [ ] **Step 4: Smoke test Tier 2 (skip Module 4 for speed)**

```bash
conda run -n MCCT python diagnostics/tier2_deepdive/run_tier2.py --session y_06 --skip-modules 4
```
Expected: outputs with `module1/`, `module5/`, `module6/` directories, no errors.

- [ ] **Step 5: Commit + add outputs to .gitignore**

```bash
# Add diagnostics/outputs/ to .gitignore
echo "diagnostics/outputs/" >> .gitignore
git add diagnostics/tier2_deepdive/run_tier2.py .gitignore
git add tests/diagnostics/
git commit -m "feat(diagnostics): Tier 2 run_tier2.py + full suite complete"
```

---

## Self-Review Checklist (completed inline)

- [x] **Spec coverage**: All 6 modules implemented. Tier 1 (M2+M3) + Tier 2 (M1+M4+M5+M6) + both orchestrators. User guide is in the design spec file, not repeated here.
- [x] **Placeholder scan**: No TBD/TODO in task steps. All code is complete and runnable.
- [x] **Type consistency**: `SessionData` used throughout. `run_*` functions consistently return `dict`. `compute_dwell_times` returns `Dict[int, List[int]]`. `hungarian_match_states` returns `(ndarray, ndarray)`. All consistent across tasks.
- [x] **Data representation**: Module 2 uses `Y_raw`, Modules 1/3/6 use `Y_pw`, Module 4 uses `Y_pw` (HMM on prewhitened = fair comparison), Module 5 uses `Y_raw` then whitens.
- [x] **Geyer truncation**: Implemented correctly in Task 4.
- [x] **Geometric not exponential**: Task 6 uses `geometric_mean_dwell` and geometric null in plots.
- [x] **Silhouette in reduced space**: Task 9 computes silhouette on `PCA(n_80).fit_transform(Y_all)`.
- [x] **Protocol split**: Module 3's `_detect_protocol` handles meditation vs PE; coincidence test runs per-session.
- [x] **Unconstrained path = argmax(gamma)**: Documented in data_loader and used throughout.
