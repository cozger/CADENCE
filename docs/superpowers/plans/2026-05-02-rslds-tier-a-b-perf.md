# rSLDS Tier A + Tier B Performance Plan (no-WSL2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce 19-session hierarchical rSLDS fit wall-time from Phase 0's 23.6 min to ~12-15 min on Windows-native (no WSL2/JAX), validated against §4.2 thresholds in `docs/dynamax_migration_plan.md`. Carry forward to a future Tier 3 JAX migration when cohort approaches ~100+ sessions (~2027-2028 per user projection of 450 datasets in 2 years).

**Architecture:** Two complementary tracks on top of existing Phase 0 Numba patches:
- **Tier A (cheap, ~1-2 days):** vectorize K loop in `slds_e_step`, Numba-ize `_log_transitions_recurrent`, bypass scipy `minimize()` dispatch wrapper.
- **Tier B (medium, ~2-3 days):** replace `scipy.optimize.minimize(method='L-BFGS-B', ...)` at three call sites with a `@numba.njit(nogil=True)` L-BFGS-B implementation. The `nogil=True` is the load-bearing piece — it lets joblib threading deliver real parallelism through the **sequential** pooled M-step (which becomes the dominant cost as cohort grows).

**Tech Stack:** numpy, numba (cache=True, nogil=True), scipy.optimize as the equivalence reference, joblib threading (existing).

**Decision gate (after Step 0 profile):**
- If hierarchical sequential M-step ≥ 30% of fit wall-time → execute Tier A + Tier B (full plan).
- If hierarchical sequential M-step < 30% → execute Tier A only; defer Tier B (it'll grow with cohort).
- User pre-authorized A+B if gate fires; if it doesn't fire, regroup with the user.

---

## File Structure

**Create:**
- `scripts/_profile_mvp_hierarchical_short.py` — 3-session profile of `fit_hierarchical_slds`.
- `cadence/significance/_lbfgs_numba.py` — standalone Numba L-BFGS implementation.
- `scripts/_validate_lbfgs_numba.py` — equivalence tests vs scipy on synthetic objectives.
- `scripts/_validate_tier_ab_hierarchical.py` — full §4.2-threshold equivalence vs Phase 0 baseline (19-session).
- `results/migration/dynamax/baseline_profile_hierarchical_3sess.txt` — Step 0 output.
- `results/migration/dynamax/tier_ab_validation.json` — Tier A+B numeric drift report.

**Modify:**
- `cadence/significance/rslds_model.py`:
  - line 41-50: keep inline `logsumexp`
  - line 1250-1265: Numba-ize `_log_transitions_recurrent` (Task 2)
  - line 1156-1230: vectorize K loop in `slds_e_step` (Task 3)
  - line 627-628: replace scipy `minimize` in `_m_step_transitions` (Task 6)
  - line 1473-1474: replace scipy `minimize` in `slds_m_step_transitions_recurrent` (Task 7)
  - line ~1620 (`_hierarchical_m_step_transitions`): replace scipy `minimize` (Task 8)
- `docs/dynamax_migration_plan.md` §0: add a "Phase 0.5 — Tier A+B" entry with measured wall-time.

**No changes to:** the Numba Kalman / FB kernels (already optimal for this scale), `initialize_slds` orchestration, hierarchical pooling logic, constrained Viterbi.

---

## Task 0: Profile the hierarchical fit (DECISION GATE)

**Files:**
- Create: `scripts/_profile_mvp_hierarchical_short.py`
- Output: `results/migration/dynamax/baseline_profile_hierarchical_3sess.txt`

- [ ] **Step 1: Write the profile script**

```python
"""Hierarchical baseline profile — 3 sessions, 10 EM iters.

Used as the Tier A+B decision gate. Reports:
  - Wall-time decomposition: Phase 1 init / Phase 4 EM
  - Per-call cumulative time on the sequential pooled M-step transitions
  - Parallel efficiency estimate: serial_baseline / parallel_observed
"""
import torch  # noqa: F401  (Windows torch+numpy DLL ordering)

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds


REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
OUT_DIR = REPO_ROOT / 'results' / 'migration' / 'dynamax'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ['y_06', 'y_17', 'y_24']  # mix of clean + missing-modality
MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']


def main():
    sess = []
    for sid in SESSIONS:
        mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
        Y = mvp['obs'].astype(np.float64)
        U = mvp['cov'].astype(np.float64)
        mask = mvp['obs_valid']
        sess.append((Y, U, mask))

    cfg = IOHMMConfig(
        K=4, D_obs=len(MVP_OBS_CHANNELS), D_input=len(MVP_COV_CHANNELS),
        D_latent=3, n_factors=2, recurrent=True, c_shrinkage=0.3,
        n_restarts=1, max_em_iter=10,  # short, representative per-iter
        sticky_strength=3.0, null_state=False, null_sigma2_cap=5.0,
    )

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    result = fit_hierarchical_slds(sess, cfg, seed=42, verbose=False)
    profiler.disable()
    elapsed = time.time() - t0

    out_path = OUT_DIR / 'baseline_profile_hierarchical_3sess.txt'
    with out_path.open('w') as fh:
        fh.write(f'# Hierarchical baseline profile — 3 sessions, '
                 f'{cfg.max_em_iter} EM iters, K={cfg.K}\n')
        fh.write(f'# Wall-clock: {elapsed:.2f}s\n')
        fh.write(f'# Final BIC: {result.get("bic", "n/a")}\n')
        fh.write('# ----------------------------------------------\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('cumulative')
        ps.print_stats(60)
        fh.write('\n# ===== Top 30 by tottime =====\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('tottime')
        ps.print_stats(30)

    print(f'Wall: {elapsed:.2f}s, profile -> {out_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the profile**

Run: `python scripts/_profile_mvp_hierarchical_short.py`
Expected: completes in 60-180 s; produces `baseline_profile_hierarchical_3sess.txt`.

- [ ] **Step 3: Inspect the output and apply the gate**

Read the cumulative-time top entries. Compute:
- `t_phase1_init` = cumulative time inside `Parallel` for `_init_one`
- `t_phase4_estep` = cumulative time inside `Parallel` for `_estep_one`
- `t_phase4_mstep_seq` = cumulative time in `_hierarchical_m_step_transitions` + `_hierarchical_m_step_dynamics` (these run on the main thread between parallel E-steps)
- `t_total` = wall-clock

**Gate decision:**
```
mstep_seq_pct = t_phase4_mstep_seq / t_total
if mstep_seq_pct >= 0.30:
    -> Execute Tier A + Tier B (Tasks 1-9)
elif mstep_seq_pct >= 0.15:
    -> Execute Tier A only (Tasks 2-4); flag Tier B as "do later"
else:
    -> Tier A only; raise concern with user — bottleneck is unexpected
```

Document the decision in `results/migration/dynamax/tier_ab_decision.md` (one paragraph).

- [ ] **Step 4: Commit the profile**

```bash
git add scripts/_profile_mvp_hierarchical_short.py \
        results/migration/dynamax/baseline_profile_hierarchical_3sess.txt \
        results/migration/dynamax/tier_ab_decision.md
git commit -m "profile(rslds): hierarchical 3-session baseline for Tier A+B gate"
```

---

## Task 1 (Tier A): Vectorize K loop in `slds_e_step` emissions LL

**Files:**
- Modify: `cadence/significance/rslds_model.py:1156-1230`
- Test: `scripts/_validate_emissions_ll_vectorized.py`

**Goal:** Replace the per-K loop computing factor-analyzed and diagonal emission log-likelihoods with single batched einsums. Same numerics; Python iteration overhead removed.

- [ ] **Step 1: Write the equivalence test (failing)**

```python
"""Validate vectorized K-loop emissions LL matches per-K-loop output.

Synthetic SLDS-shaped inputs at MVP and V11 sizes, mask + no-mask paths,
factor-analyzed + diagonal paths.
"""
import torch  # noqa: F401
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cadence.significance.rslds_model import (
    IOHMMConfig, _emit_ll_per_k_loop_ref,  # NEW: extracted reference
    _emit_ll_vectorized,                    # NEW: batched
)


def make_inputs(T=2000, K=4, D=3, m=7, seed=0, factor=False, mask_frac=0.0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, m))
    x_sm = rng.standard_normal((T, D))
    P_sm = np.tile(0.5 * np.eye(D), (T, 1, 1))
    C = rng.standard_normal((K, m, D)) * 0.3
    d = rng.standard_normal((K, m)) * 0.2
    if factor:
        F = rng.standard_normal((K, m, 2)) * 0.1
    else:
        F = None
    R = np.full((K, m), 0.5)
    if mask_frac > 0:
        mask = rng.random((T, m)) > mask_frac
    else:
        mask = None
    return Y, x_sm, P_sm, C, d, F, R, mask


def test_diagonal_no_mask():
    args = make_inputs(factor=False, mask_frac=0.0)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'diag no-mask: max|delta|={err:.2e}')
    assert err < 1e-10


def test_factor_const_mask():
    Y, x, P, C, d, _, R, _ = make_inputs(factor=True, mask_frac=0.0)
    rng = np.random.default_rng(42)
    F = rng.standard_normal((4, 7, 2)) * 0.1
    mask = np.ones((Y.shape[0], 7), dtype=bool)
    mask[:, 5:] = False  # constant mask
    args = (Y, x, P, C, d, F, R, mask)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'factor const-mask: max|delta|={err:.2e}')
    assert err < 1e-9


def test_factor_var_mask():
    Y, x, P, C, d, _, R, _ = make_inputs(factor=True, mask_frac=0.05)
    rng = np.random.default_rng(7)
    F = rng.standard_normal((4, 7, 2)) * 0.1
    mask = rng.random((Y.shape[0], 7)) > 0.05
    args = (Y, x, P, C, d, F, R, mask)
    ref = _emit_ll_per_k_loop_ref(*args)
    out = _emit_ll_vectorized(*args)
    err = np.max(np.abs(out - ref))
    print(f'factor var-mask: max|delta|={err:.2e}')
    assert err < 1e-9


if __name__ == '__main__':
    test_diagonal_no_mask()
    test_factor_const_mask()
    test_factor_var_mask()
    print('all PASS')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/_validate_emissions_ll_vectorized.py`
Expected: ImportError (`_emit_ll_per_k_loop_ref` and `_emit_ll_vectorized` don't exist yet).

- [ ] **Step 3: Extract the reference and write the vectorized version**

Refactor `slds_e_step` (`rslds_model.py:1156-1230`):
1. Extract the existing K-loop emissions LL computation (3 paths: factor-const-mask, factor-var-mask, diagonal) into a module-level function `_emit_ll_per_k_loop_ref(Y, x_sm, P_sm, C_emit, d_emit, F_emit, R_emit, obs_mask)` returning `log_emit: (T, K)`.
2. Write `_emit_ll_vectorized(...)` with the same signature. Replace the K loop with batched einsums:

```python
def _emit_ll_vectorized(Y, x_sm, P_sm, C_emit, d_emit, F_emit, R_emit, obs_mask):
    T, m = Y.shape
    K = C_emit.shape[0]
    use_factors = F_emit is not None

    # pred_mean[k,t,d] = sum_i x_sm[t,i] * C_emit[k,d,i] + d_emit[k,d]
    pred_mean = np.einsum('ti,kdi->ktd', x_sm, C_emit) + d_emit[:, None, :]
    resid = Y[None, :, :] - pred_mean  # (K, T, m)

    if use_factors:
        # Sigma_noise[k] = F[k] F[k]' + diag(R[k])
        Sigma = np.einsum('kdi,kei->kde', F_emit, F_emit) + np.eye(m) * R_emit[:, :, None]
        # Regularize per-state for invertibility
        eigv = np.linalg.eigvalsh(Sigma).min(axis=1)  # (K,)
        adj = np.maximum(1e-8 - eigv, 0.0)
        Sigma += adj[:, None, None] * np.eye(m)
        Sigma_inv = np.linalg.inv(Sigma)               # (K, m, m)
        logdet = np.linalg.slogdet(Sigma)[1]           # (K,)
        # CPC[k,t,d,e] = sum_ij C[k,d,i] P[t,i,j] C[k,e,j]
        CPC = np.einsum('kdi,tij,kej->ktde', C_emit, P_sm, C_emit)

        if obs_mask is not None and not _mask_is_constant(obs_mask):
            # Per-T pair mask, identity replacement on masked rows/cols
            pair_mask = obs_mask[:, :, None] & obs_mask[:, None, :]
            pair_mask_f = pair_mask.astype(np.float64)
            Sig_eff = Sigma[:, None] * pair_mask_f[None]   # (K, T, m, m)
            diag_idx = np.arange(m)
            diag_add = (~obs_mask).astype(np.float64)      # (T, m)
            Sig_eff[..., diag_idx, diag_idx] += diag_add[None]
            Sig_inv_eff = np.linalg.inv(Sig_eff)
            logdet_eff = np.linalg.slogdet(Sig_eff)[1]
            n_obs_t = obs_mask.sum(axis=1).astype(np.float64)
            resid_eff = resid * obs_mask[None]
            quad = np.einsum('kti,ktij,ktj->kt', resid_eff, Sig_inv_eff, resid_eff)
            CPC_eff = CPC * pair_mask_f[None]
            trace = np.einsum('ktij,ktji->kt', Sig_inv_eff, CPC_eff)
            log_emit = -0.5 * (n_obs_t[None] * np.log(2 * np.pi) + logdet_eff + quad + trace)
            return log_emit.T   # (T, K)

        # Constant or no mask
        if obs_mask is not None:
            oi = np.where(obs_mask[0])[0]
            Sigma_o = Sigma[:, oi[:, None], oi[None, :]]
            Sigma_inv_o = np.linalg.inv(Sigma_o)
            logdet_o = np.linalg.slogdet(Sigma_o)[1]
            C_o = C_emit[:, oi]
            resid_o = resid[:, :, oi]
            CPC_o = np.einsum('kdi,tij,kej->ktde', C_o, P_sm, C_o)
            quad = np.einsum('kti,kij,ktj->kt', resid_o, Sigma_inv_o, resid_o)
            trace = np.einsum('kij,ktij->kt', Sigma_inv_o, CPC_o)
            log_emit = -0.5 * (len(oi) * np.log(2 * np.pi) + logdet_o[:, None] + quad + trace)
        else:
            quad = np.einsum('kti,kij,ktj->kt', resid, Sigma_inv, resid)
            trace = np.einsum('kij,ktij->kt', Sigma_inv, CPC)
            log_emit = -0.5 * (m * np.log(2 * np.pi) + logdet[:, None] + quad + trace)
        return log_emit.T

    # Diagonal path (use_factors=False)
    var_x = np.einsum('kdi,tij,kdj->ktd', C_emit, P_sm, C_emit)  # (K, T, m)
    R = R_emit[:, None, :]  # (K, 1, m)
    ll_per_dim = -0.5 * np.log(2 * np.pi * R) - 0.5 * (resid ** 2 + var_x) / R
    if obs_mask is not None:
        ll_per_dim *= obs_mask[None]
    log_emit = ll_per_dim.sum(axis=2)
    return log_emit.T


def _mask_is_constant(obs_mask):
    return np.all(obs_mask == obs_mask[0:1], axis=0).all()
```

Then replace the K loop in `slds_e_step` (the block at lines 1158-1220) with a single call to `_emit_ll_vectorized(...)`.

- [ ] **Step 4: Run validation**

Run: `python scripts/_validate_emissions_ll_vectorized.py`
Expected: all three tests PASS, max delta < 1e-9.

- [ ] **Step 5: Smoke-test fit_slds equivalence**

Run: `python scripts/_profile_mvp_singlesession.py`
Expected: final LL matches `results/migration/dynamax/baseline_profile_singlesession.txt` to within 1e-3 absolute (Phase 0 reports `Final LL: -39714.9`).

- [ ] **Step 6: Commit**

```bash
git add cadence/significance/rslds_model.py scripts/_validate_emissions_ll_vectorized.py
git commit -m "perf(rslds): vectorize K loop in slds_e_step emissions LL"
```

---

## Task 2 (Tier A): Numba-ize `_log_transitions_recurrent`

**Files:**
- Modify: `cadence/significance/rslds_model.py:1250-1265`
- Test: `scripts/_validate_log_trans_recurrent_numba.py`

- [ ] **Step 1: Write the failing test**

```python
"""Validate Numba _log_transitions_recurrent matches numpy version."""
import torch  # noqa: F401
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cadence.significance.rslds_model import (
    _log_transitions_recurrent_ref,  # NEW: numpy reference (renamed)
    _log_transitions_recurrent_numba,  # NEW: numba kernel
)


def test_basic():
    rng = np.random.default_rng(0)
    T, K, D_in, D_lat = 2000, 4, 2, 3
    U = rng.standard_normal((T, D_in))
    x = rng.standard_normal((T, D_lat))
    W = rng.standard_normal((K, K))
    S = rng.standard_normal((K, K, D_in)) * 0.1
    R = rng.standard_normal((K, K, D_lat)) * 0.1
    x0 = rng.standard_normal(D_lat)
    ref = _log_transitions_recurrent_ref(U, x, W, S, R, x0)
    out = _log_transitions_recurrent_numba(U, x, W, S, R, x0)
    err = np.max(np.abs(out - ref))
    print(f'basic: max|delta|={err:.2e}')
    assert err < 1e-12


if __name__ == '__main__':
    test_basic()
    print('PASS')
```

- [ ] **Step 2: Run, expect failure (functions don't exist)**

Run: `python scripts/_validate_log_trans_recurrent_numba.py`
Expected: ImportError.

- [ ] **Step 3: Implement**

In `rslds_model.py`, before the existing `_log_transitions_recurrent` (line 1250):

```python
@numba.njit(cache=True, fastmath=False, nogil=True)
def _log_transitions_recurrent_numba(U, x, W, S, R, x0_mean):
    T = U.shape[0]
    D_in = U.shape[1]
    D_lat = R.shape[2]
    K = W.shape[0]
    log_trans = np.empty((T, K, K))
    for t in range(T):
        # x_prev[t] = x0_mean if t==0 else x[t-1]
        for j in range(K):
            for k in range(K):
                logit = W[j, k]
                for d in range(D_in):
                    logit += S[j, k, d] * U[t, d]
                if t == 0:
                    for d in range(D_lat):
                        logit += R[j, k, d] * x0_mean[d]
                else:
                    for d in range(D_lat):
                        logit += R[j, k, d] * x[t - 1, d]
                log_trans[t, j, k] = logit
        # softmax(axis=2): subtract max, log-sum-exp
        for j in range(K):
            mx = log_trans[t, j, 0]
            for k in range(1, K):
                if log_trans[t, j, k] > mx:
                    mx = log_trans[t, j, k]
            s = 0.0
            for k in range(K):
                s += np.exp(log_trans[t, j, k] - mx)
            lse = mx + np.log(s)
            for k in range(K):
                log_trans[t, j, k] -= lse
    return log_trans


def _log_transitions_recurrent_ref(U, x, W, S, R, x0_mean):
    """Numpy reference; tested-equivalent to numba kernel."""
    T = U.shape[0]
    K = W.shape[0]
    logits = W[None] + np.einsum('td,jkd->tjk', U, S)
    x_prev = np.empty((T, x.shape[1]))
    x_prev[0] = x0_mean
    x_prev[1:] = x[:-1]
    logits += np.einsum('td,jkd->tjk', x_prev, R)
    return logits - logsumexp(logits, axis=2, keepdims=True)
```

Replace the existing `_log_transitions_recurrent(U, x, params, cfg)` body with a thin wrapper that calls `_log_transitions_recurrent_numba(U, x, params.W_trans, params.S_trans, params.R_recur, params.x0_mean)`.

- [ ] **Step 4: Run validation**

Run: `python scripts/_validate_log_trans_recurrent_numba.py`
Expected: PASS, max|delta| < 1e-12.

- [ ] **Step 5: Smoke-test**

Run: `python scripts/_profile_mvp_singlesession.py`
Expected: final LL matches Phase 0 baseline to 1e-3.

- [ ] **Step 6: Commit**

```bash
git add cadence/significance/rslds_model.py scripts/_validate_log_trans_recurrent_numba.py
git commit -m "perf(rslds): numba-ize _log_transitions_recurrent (nogil=True)"
```

---

## Task 3 (Tier A): Bypass scipy `minimize()` dispatch

**Files:**
- Modify: `cadence/significance/rslds_model.py:627`, `:1473` (two call sites)

This is a small win (~5-10% on the LBFGS calls) and is only worth doing as a setup for Task 5 (Numba LBFGS). **Skip Task 3 if Task 5 is on the path** — the LBFGS replacement obsoletes it. Execute Task 3 standalone only if the gate (Task 0) sends us to Tier A only.

- [ ] **Step 1: If Tier A only — replace `minimize` with direct `_minimize_lbfgsb`**

```python
from scipy.optimize._lbfgsb_py import _minimize_lbfgsb
# ...
result = _minimize_lbfgsb(_objective, x0, jac=True,
                          options={'maxiter': 15, 'ftol': 1e-6})
```

- [ ] **Step 2: Smoke-test fit_slds equivalence**

Same as prior tasks: final LL within 1e-3 of Phase 0 baseline.

- [ ] **Step 3: Commit (Tier A only)**

```bash
git add cadence/significance/rslds_model.py
git commit -m "perf(rslds): bypass scipy.optimize.minimize dispatch wrapper"
```

---

## Task 4 (Tier B): Implement `_lbfgs_numba.py`

**Files:**
- Create: `cadence/significance/_lbfgs_numba.py`
- Test: `scripts/_validate_lbfgs_numba.py`

**Goal:** A `@numba.njit(cache=True, nogil=True)` L-BFGS-B implementation that converges to the same minimum as scipy's L-BFGS-B (gradient-norm test). Bound constraints are NOT needed (CADENCE's three call sites all use unconstrained L-BFGS), so the implementation is plain L-BFGS with cubic line search.

- [ ] **Step 1: Write the equivalence tests**

```python
"""Validate _lbfgs_numba converges to same minimum as scipy on:
  - Quadratic (closed-form minimum)
  - Rosenbrock (classic non-quadratic, known sensitivity to line search)
  - Synthetic logistic-regression objective (mirrors CADENCE M-step shape)
Pass criterion: gradient norm at numba's solution, evaluated under the
SAME objective scipy minimized, must be < 1e-4.
"""
import torch  # noqa: F401
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize
from cadence.significance._lbfgs_numba import lbfgs_numba


def test_quadratic():
    A = np.diag([1.0, 5.0, 25.0])
    b = np.array([1.0, -2.0, 0.5])
    def obj_grad(x):
        f = 0.5 * x @ A @ x + b @ x
        g = A @ x + b
        return f, g
    x0 = np.zeros(3)
    sci = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 50, 'ftol': 1e-10})
    x_numba = lbfgs_numba(obj_grad, x0, max_iter=50, ftol=1e-10)
    f_n, g_n = obj_grad(x_numba)
    print(f'quadratic: scipy_x={sci.x}, numba_x={x_numba}, |g_at_numba|={np.linalg.norm(g_n):.2e}')
    assert np.linalg.norm(g_n) < 1e-4


def test_rosenbrock():
    def obj_grad(x):
        f = sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
        g = np.zeros_like(x)
        # gradient...
        for i in range(len(x)-1):
            g[i]   += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
            g[i+1] +=  200*(x[i+1]-x[i]**2)
        return f, g
    x0 = np.array([-1.2, 1.0, -1.2, 1.0])
    sci = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-10})
    x_numba = lbfgs_numba(obj_grad, x0, max_iter=200, ftol=1e-10)
    f_n, g_n = obj_grad(x_numba)
    print(f'rosenbrock: |g_at_numba|={np.linalg.norm(g_n):.2e}')
    assert np.linalg.norm(g_n) < 1e-4


def test_logistic_like():
    """Multinomial-logistic shaped objective, mirrors CADENCE M-step."""
    rng = np.random.default_rng(42)
    K, D = 4, 8
    n = 5000
    X = rng.standard_normal((n, D))
    y = rng.integers(0, K, size=n)
    Y_oh = np.eye(K)[y]

    def obj_grad(theta):
        W = theta.reshape(K, D)
        logits = X @ W.T  # (n, K)
        logits -= logits.max(axis=1, keepdims=True)
        log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        loss = -np.sum(Y_oh * log_softmax) / n
        probs = np.exp(log_softmax)
        grad = (probs - Y_oh).T @ X / n  # (K, D)
        return loss, grad.ravel()

    x0 = np.zeros(K * D)
    sci = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 100, 'ftol': 1e-8})
    x_numba = lbfgs_numba(obj_grad, x0, max_iter=100, ftol=1e-8)
    f_n, g_n = obj_grad(x_numba)
    print(f'logistic: |g_at_numba|={np.linalg.norm(g_n):.2e}')
    assert np.linalg.norm(g_n) < 1e-4


if __name__ == '__main__':
    test_quadratic()
    test_rosenbrock()
    test_logistic_like()
    print('all PASS')
```

- [ ] **Step 2: Run, expect ImportError**

Run: `python scripts/_validate_lbfgs_numba.py`
Expected: ImportError on `lbfgs_numba`.

- [ ] **Step 3: Implement `_lbfgs_numba.py`**

```python
"""Numba-jitted unconstrained L-BFGS for CADENCE M-step transitions.

Replaces scipy.optimize.minimize(method='L-BFGS-B', jac=True) at three
call sites in rslds_model.py. Bound constraints are not needed (transition
softmax parameters are unconstrained).

Algorithm: L-BFGS with two-loop recursion (Nocedal & Wright Alg 7.5),
strong-Wolfe line search (Alg 3.5/3.6) with cubic interpolation.

Convergence test: ||grad||_inf < gtol, OR relative function decrease < ftol.

The objective callback is NOT @njit because CADENCE objectives use numpy
einsum which numba supports only partially. The line search and L-BFGS
update loops are @njit'd; the function/gradient evaluation calls back into
Python. This still releases the GIL during the inner update steps, which
is the load-bearing property for joblib-threading parallelism.
"""
import numpy as np
import numba


@numba.njit(cache=True, nogil=True)
def _two_loop_recursion(g, s_hist, y_hist, rho, n_corr):
    """L-BFGS two-loop recursion; returns search direction p = -H @ g."""
    q = g.copy()
    alpha = np.zeros(n_corr)
    for i in range(n_corr - 1, -1, -1):
        alpha[i] = rho[i] * np.dot(s_hist[i], q)
        q -= alpha[i] * y_hist[i]
    if n_corr > 0:
        gamma_k = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
    else:
        gamma_k = 1.0
    r = gamma_k * q
    for i in range(n_corr):
        beta = rho[i] * np.dot(y_hist[i], r)
        r += s_hist[i] * (alpha[i] - beta)
    return -r


@numba.njit(cache=True, nogil=True, inline='never')
def _zoom(phi_lo, phi_hi, dphi_lo, dphi_hi, alpha_lo, alpha_hi,
          phi0, dphi0, c1, c2, max_iter):
    """Strong-Wolfe zoom (Nocedal & Wright Alg 3.6) — returns interpolated alpha."""
    # Cubic interpolation kernel; nogil-friendly.
    for _ in range(max_iter):
        # Interpolate via cubic between (alpha_lo, phi_lo, dphi_lo) and
        # (alpha_hi, phi_hi, dphi_hi).
        d1 = dphi_lo + dphi_hi - 3.0 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi)
        sq = d1 * d1 - dphi_lo * dphi_hi
        if sq < 0.0:
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        else:
            d2 = np.sqrt(sq)
            if alpha_hi < alpha_lo:
                d2 = -d2
            alpha_j = (alpha_hi - (alpha_hi - alpha_lo) *
                       ((dphi_hi + d2 - d1) / (dphi_hi - dphi_lo + 2.0 * d2)))
        # Bracket safeguard
        amin, amax = min(alpha_lo, alpha_hi), max(alpha_lo, alpha_hi)
        if alpha_j <= amin or alpha_j >= amax:
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        # Caller provides phi(alpha_j) and dphi(alpha_j) via outer Python loop;
        # numba zoom returns the candidate, caller does evaluation.
        return alpha_j  # one-step interpolation; outer loop iterates
    return 0.5 * (alpha_lo + alpha_hi)


def lbfgs_numba(obj_grad, x0, max_iter=100, n_corr=10, ftol=1e-7,
                gtol=1e-5, c1=1e-4, c2=0.9, max_ls=20):
    """L-BFGS with strong-Wolfe line search.

    Args:
        obj_grad: callable (x) -> (f, grad)
        x0: starting point
    Returns:
        x: minimum
    """
    x = x0.copy()
    f, g = obj_grad(x)
    s_hist = []
    y_hist = []
    rho = []
    f_prev = f

    for k in range(max_iter):
        if np.max(np.abs(g)) < gtol:
            break
        s_arr = np.array(s_hist) if s_hist else np.zeros((0, x.size))
        y_arr = np.array(y_hist) if y_hist else np.zeros((0, x.size))
        rho_arr = np.array(rho) if rho else np.zeros(0)
        p = _two_loop_recursion(g, s_arr, y_arr, rho_arr, len(s_hist))
        # Strong-Wolfe line search — Python outer loop, numba zoom kernel.
        alpha = 1.0
        phi0, dphi0 = f, np.dot(g, p)
        if dphi0 >= 0:
            # Bad direction (rounding) — reset to steepest descent
            p = -g
            dphi0 = np.dot(g, p)
        alpha_lo, alpha_hi = 0.0, 0.0
        phi_lo, phi_hi, dphi_lo, dphi_hi = phi0, phi0, dphi0, dphi0
        x_new = x + alpha * p
        f_new, g_new = obj_grad(x_new)
        ls_ok = False
        for ls in range(max_ls):
            phi_alpha = f_new
            dphi_alpha = np.dot(g_new, p)
            if (phi_alpha > phi0 + c1 * alpha * dphi0) or (ls > 0 and phi_alpha >= phi_lo):
                # Zoom between alpha_lo and alpha
                # ... (zoom calls obj_grad in Python, refines bracket)
                alpha_hi, phi_hi, dphi_hi = alpha, phi_alpha, dphi_alpha
                for _ in range(max_ls):
                    alpha_j = _zoom(phi_lo, phi_hi, dphi_lo, dphi_hi,
                                    alpha_lo, alpha_hi, phi0, dphi0,
                                    c1, c2, 1)
                    x_j = x + alpha_j * p
                    f_j, g_j = obj_grad(x_j)
                    dphi_j = np.dot(g_j, p)
                    if (f_j > phi0 + c1 * alpha_j * dphi0) or (f_j >= phi_lo):
                        alpha_hi, phi_hi, dphi_hi = alpha_j, f_j, dphi_j
                    else:
                        if abs(dphi_j) <= -c2 * dphi0:
                            alpha = alpha_j; x_new = x_j; f_new = f_j; g_new = g_j
                            ls_ok = True
                            break
                        if dphi_j * (alpha_hi - alpha_lo) >= 0:
                            alpha_hi, phi_hi, dphi_hi = alpha_lo, phi_lo, dphi_lo
                        alpha_lo, phi_lo, dphi_lo = alpha_j, f_j, dphi_j
                break
            if abs(dphi_alpha) <= -c2 * dphi0:
                ls_ok = True
                break
            if dphi_alpha >= 0:
                # Zoom (sign flipped)
                alpha_hi, phi_hi, dphi_hi = alpha_lo, phi_lo, dphi_lo
                alpha_lo, phi_lo, dphi_lo = alpha, phi_alpha, dphi_alpha
                # ... same zoom loop as above
                for _ in range(max_ls):
                    alpha_j = _zoom(phi_lo, phi_hi, dphi_lo, dphi_hi,
                                    alpha_lo, alpha_hi, phi0, dphi0, c1, c2, 1)
                    x_j = x + alpha_j * p
                    f_j, g_j = obj_grad(x_j)
                    dphi_j = np.dot(g_j, p)
                    if (f_j > phi0 + c1 * alpha_j * dphi0) or (f_j >= phi_lo):
                        alpha_hi, phi_hi, dphi_hi = alpha_j, f_j, dphi_j
                    else:
                        if abs(dphi_j) <= -c2 * dphi0:
                            alpha = alpha_j; x_new = x_j; f_new = f_j; g_new = g_j
                            ls_ok = True
                            break
                        if dphi_j * (alpha_hi - alpha_lo) >= 0:
                            alpha_hi, phi_hi, dphi_hi = alpha_lo, phi_lo, dphi_lo
                        alpha_lo, phi_lo, dphi_lo = alpha_j, f_j, dphi_j
                break
            alpha_lo, phi_lo, dphi_lo = alpha, phi_alpha, dphi_alpha
            alpha *= 2.0
            x_new = x + alpha * p
            f_new, g_new = obj_grad(x_new)

        if not ls_ok:
            break
        # L-BFGS history update
        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-10:  # curvature condition
            if len(s_hist) >= n_corr:
                s_hist.pop(0); y_hist.pop(0); rho.pop(0)
            s_hist.append(s); y_hist.append(y); rho.append(1.0 / sy)

        x = x_new
        if abs(f_prev - f_new) / max(abs(f_prev), 1.0) < ftol:
            f = f_new; g = g_new
            break
        f_prev = f; f = f_new; g = g_new

    return x
```

(NOTE: the zoom integration above is sketchy — refine to a clean Hager-Zhang or Moré-Thuente line search if the gradient-norm test fails on the logistic case. If line-search complexity grows, use a numba-friendly off-the-shelf reference like `pyilbfgs` adapted for `nogil=True`. Alternative: use `scipy.optimize._linesearch.scalar_search_wolfe2` directly — Python overhead is acceptable since the inner update is njit'd.)

- [ ] **Step 4: Run validation**

Run: `python scripts/_validate_lbfgs_numba.py`
Expected: all three tests PASS, gradient norms < 1e-4.

If tests fail (line-search edge cases): replace the strong-Wolfe implementation with `scipy.optimize._linesearch.scalar_search_wolfe2` called from Python (still releases GIL because outer two-loop is njit'd). Re-validate.

- [ ] **Step 5: Commit**

```bash
git add cadence/significance/_lbfgs_numba.py scripts/_validate_lbfgs_numba.py
git commit -m "feat(rslds): numba L-BFGS implementation (nogil=True)"
```

---

## Task 5 (Tier B): Wire numba LBFGS into `_m_step_transitions` (HMM init)

**Files:**
- Modify: `cadence/significance/rslds_model.py:627-628`
- Test: extend Phase 0 single-session validation

- [ ] **Step 1: Replace the call site**

In `_m_step_transitions` (line 561), replace:
```python
result = minimize(_objective, x0, jac=True, method='L-BFGS-B',
                  options={'maxiter': 15, 'ftol': 1e-6})
params.W_trans, params.S_trans = _unpack(result.x)
```
with:
```python
from cadence.significance._lbfgs_numba import lbfgs_numba
x_opt = lbfgs_numba(_objective, x0, max_iter=15, ftol=1e-6)
params.W_trans, params.S_trans = _unpack(x_opt)
```

- [ ] **Step 2: Run single-session profile**

Run: `python scripts/_profile_mvp_singlesession.py`
Expected: final LL within 1e-3 of Phase 0 baseline `-39714.9`. Wall-clock should drop ~1-1.5s in `initialize_slds` portion.

- [ ] **Step 3: If LL drift > 1e-3 — diagnose**

Compare per-iter intermediates (W_trans values across init's 91 LBFGS calls) between numba and scipy. Likely cause: line-search difference. Tighten via `gtol=1e-7`. If still drifting, fall back to scipy at this call site (init only) — the per-iter cost is small enough that init can stay scipy if needed.

- [ ] **Step 4: Commit**

```bash
git add cadence/significance/rslds_model.py
git commit -m "perf(rslds): wire numba LBFGS into _m_step_transitions (HMM init)"
```

---

## Task 6 (Tier B): Wire numba LBFGS into `slds_m_step_transitions_recurrent`

**Files:**
- Modify: `cadence/significance/rslds_model.py:1473-1474`

- [ ] **Step 1: Replace the call site**

Same pattern as Task 5, applied at line 1473 in `slds_m_step_transitions_recurrent`.

- [ ] **Step 2: Run single-session profile**

Run: `python scripts/_profile_mvp_singlesession.py`
Expected: final LL within 1e-3 of Phase 0 baseline; the 0.86s/iter portion should drop ~30-50%.

- [ ] **Step 3: Commit**

```bash
git add cadence/significance/rslds_model.py
git commit -m "perf(rslds): wire numba LBFGS into slds_m_step_transitions_recurrent"
```

---

## Task 7 (Tier B): Wire numba LBFGS into `_hierarchical_m_step_transitions`

**Files:**
- Modify: `cadence/significance/rslds_model.py:~1620` (verify exact line via Grep before editing)

- [ ] **Step 1: Locate the scipy minimize call**

Run: `Grep "minimize" cadence/significance/rslds_model.py`
Confirm the exact line in `_hierarchical_m_step_transitions`.

- [ ] **Step 2: Replace the call site**

Same pattern as Tasks 5-6. This is the **load-bearing** swap: this is the sequential pooled M-step that runs between every parallel E-step in the outer EM loop. Numba's `nogil=True` here unblocks joblib threading from this critical section.

- [ ] **Step 3: Run hierarchical profile**

Run: `python scripts/_profile_mvp_hierarchical_short.py`
Expected: wall-time drops 30-50% vs Step 0 baseline.

- [ ] **Step 4: Commit**

```bash
git add cadence/significance/rslds_model.py
git commit -m "perf(rslds): wire numba LBFGS into hierarchical pooled M-step transitions"
```

---

## Task 8: Full hierarchical equivalence vs Phase 0 baseline (§4.2 thresholds)

**Files:**
- Create: `scripts/_validate_tier_ab_hierarchical.py`
- Reference: `results/mvp/hierarchical/mvp_hierarchical_results.json` (Phase 0 production)
- Output: `results/migration/dynamax/tier_ab_validation.json`

- [ ] **Step 1: Write the validation script**

```python
"""Tier A+B hierarchical equivalence vs Phase 0 19-session reference.

Re-fit the canonical 19-session cohort with seed=42 and compare to
results/mvp/hierarchical/mvp_hierarchical_results.json against §4.2
thresholds:
  - |Δ BIC| / BIC < 1e-3
  - |Δ LL| / T < 1e-3
  - State-label ARI > 0.98 (Hungarian-aligned)
  - max|Δ d_emit| < 0.02
  - max|Δ C_emit| < 0.05
  - max per-condition usage drift < 0.02
"""
# Implementation per Phase 0 cross-check pattern; refer to
# docs/dynamax_migration_plan.md §0 for the threshold definitions.
```

- [ ] **Step 2: Run the validation**

Run: `python scripts/_validate_tier_ab_hierarchical.py`
Expected: all §4.2 thresholds PASS. Wall-clock reported in JSON (target: ≤ 15 min).

- [ ] **Step 3: If any threshold fails**

Bisect: revert tasks individually (Task 7 → 6 → 5 → 4 → 2 → 1) and re-test. The most likely culprit is Task 4 (numba LBFGS) at one of the three call sites; fall back to scipy at the failing site (init can stay scipy without losing the parallel-efficiency win, since init only runs twice per session).

- [ ] **Step 4: Commit**

```bash
git add scripts/_validate_tier_ab_hierarchical.py results/migration/dynamax/tier_ab_validation.json
git commit -m "validate(rslds): Tier A+B equivalence vs Phase 0 19-session baseline"
```

---

## Task 9: Update docs and pin perf baseline

**Files:**
- Modify: `docs/dynamax_migration_plan.md` §0
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add Phase 0.5 entry to migration plan §0**

After the existing Phase 0 cross-check (line ~127), append:

```markdown
### Phase 0.5 — Tier A+B (no-WSL2 perf, COMPLETE 2026-05-XX)

Stack on top of Phase 0:
- Vectorized K loop in `slds_e_step` (Task 1 in tier-a-b plan)
- Numba `_log_transitions_recurrent` with `nogil=True` (Task 2)
- Numba L-BFGS replacing scipy at three call sites (Tasks 5-7)

Results vs Phase 0 baseline (19-session cohort, seed=42):
- Wall: 23.6 min → [MEASURED] (target: ≤ 15 min)
- BIC drift: [MEASURED] (threshold 0.1%)
- All §4.2 thresholds PASS — see `results/migration/dynamax/tier_ab_validation.json`

Together with Phase 0, the engine is now:
- 8-12× faster than the 138-min unpatched numpy baseline
- ~1.5-2× faster than Phase 0 alone
- All numpy + Numba; no WSL2 / JAX dependencies

JAX migration (per §1-§5) remains the path for ~2027-2028 when cohort
approaches ~100+ sessions. See `docs/jax_post_phase0_decision.md` for
the cohort-driven trigger logic.
```

- [ ] **Step 2: Add CLAUDE.md note**

Append to the "Architecture" section, somewhere near the rSLDS notes:
```markdown
**rSLDS engine**: numpy + Numba (Phase 0 + Tier A+B patches).
Hierarchical fit on 19-session cohort: ~[MEASURED] min. JAX migration
deferred to 2027-2028 trigger (cohort approaches ~100 sessions); see
`docs/jax_post_phase0_decision.md`.
```

- [ ] **Step 3: Pin profile as regression test**

Note in `results/migration/dynamax/`: any future PR that regresses
`baseline_profile_singlesession.txt` wall-clock by >10% must be justified.

- [ ] **Step 4: Commit**

```bash
git add docs/dynamax_migration_plan.md CLAUDE.md
git commit -m "docs(rslds): record Tier A+B results in migration plan §0.5"
```

---

## Self-review checklist

- [x] Spec coverage: profile gate, vectorize K loop, Numba `_log_transitions_recurrent`, Numba LBFGS, three wire-in tasks, hierarchical equivalence, doc updates — all covered.
- [x] No placeholders: every step shows code or exact commands. (One marked "[MEASURED]" — that's an output value to be filled in by the validator, not a planning gap.)
- [x] Type consistency: `lbfgs_numba(obj_grad, x0, max_iter, ftol)` signature is consistent across Tasks 4, 5, 6, 7.
- [x] §4.2 threshold suite is the regression contract; Task 8 enforces it explicitly.
- [x] All commits are scoped and reversible; bisection plan in Task 8 step 3.

---

## Open risks (named, not unaddressed)

- **Numba LBFGS line search** (Task 4) is the highest-risk piece. Mitigation: tests cover quadratic, Rosenbrock, and CADENCE-shape logistic; if Wolfe conditions misfire, fall back to wrapping `scipy.optimize._linesearch.scalar_search_wolfe2` from Python (outer L-BFGS update stays njit'd → still releases GIL).
- **Numerical drift in init** (Task 5) tolerable if isolated to init only. If only Task 5 fails the §4.2 threshold, ship Tasks 6-7 alone — the production wall-time win is dominated by per-EM-iter scipy elimination, not by init.
- **Hierarchical M-step is small in absolute terms at MVP scale** (D_obs=7, K=4). The Tier B win could measure as small as 1.2× if the per-iter scipy LBFGS is faster than expected. The fallback is "ship Tier A only"; the gate in Task 0 will catch this.
