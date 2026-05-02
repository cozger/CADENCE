"""Phase 0 cross-check — re-run a hierarchical_smoke-equivalent fit with the
patched code and compare to results/mvp/hierarchical_smoke/ baseline.

The baseline was computed with the un-patched (281s/10-iter) numpy code.
With Phase 0 (33x speedup), we expect:
  - Wall-time: 357s -> ~10-20s
  - BIC, final LL: equal within scientifically-equivalent thresholds
    (max|d| BIC < 0.001 relative per plan §4.2)
  - Per-state d_emit: max|d| < 0.05
  - State labels: same set (Hungarian alignment)
"""
# torch must be imported before numpy on Windows
import torch  # noqa: F401

import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import parallel_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds


REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'

MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']


def load_mvp(sid):
    mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
    return (mvp['obs'].astype(np.float64),
            mvp['cov'].astype(np.float64),
            mvp['obs_valid'])


def main():
    # Mirror baseline hierarchical_smoke config (K=4, default null_state=False,
    # default c_shrinkage=0.3, default sticky=3.0, viterbi_min_dwell=20)
    cfg = IOHMMConfig(
        K=4,
        D_obs=len(MVP_OBS_CHANNELS),
        D_input=len(MVP_COV_CHANNELS),
        D_latent=3,
        n_factors=2,
        recurrent=True,
        c_shrinkage=0.3,
        n_restarts=2,
        max_em_iter=80,
        sticky_strength=3.0,
        null_state=False,
        null_sigma2_cap=5.0,
        viterbi_min_dwell=20,
    )

    Y, U, mask = load_mvp('y_06')
    print(f'Loaded y_06: T={Y.shape[0]}, D_obs={Y.shape[1]}')

    print(f'\nRunning fit_hierarchical_slds (N=1, K={cfg.K}, '
          f'max_em_iter={cfg.max_em_iter}, n_restarts={cfg.n_restarts})...')
    t0 = time.time()
    with parallel_backend('threading'):
        result = fit_hierarchical_slds([(Y, U, mask)], cfg, seed=42, verbose=True)
    elapsed = time.time() - t0
    print(f'\nWall-clock: {elapsed:.1f}s')

    new_bic = float(result['bic'])
    new_ll = float(result['final_ll'])
    new_d = np.array(result['shared'].get('d_emit_mean',
                       result['sessions'][0]['d_emit'])) if 'shared' in result \
                       else np.array(result['sessions'][0]['d_emit'])

    # If shared key shape is different, fallback to first session's d_emit
    # Actually the baseline reports mean_d_emit:
    if 'mean_d_emit' in result:
        new_d = np.array(result['mean_d_emit'])
    elif 'shared' in result and 'd_emit_mean' in result['shared']:
        new_d = np.array(result['shared']['d_emit_mean'])
    else:
        new_d = np.array(result['sessions'][0].get('d_emit', []))

    # Load baseline
    base_path = REPO_ROOT / 'results' / 'mvp' / 'hierarchical_smoke' / 'mvp_hierarchical_results.json'
    base = json.loads(base_path.read_text())
    base_bic = base['bic']
    base_ll = base['final_ll']
    base_d = np.array(base['mean_d_emit'])
    base_walls = base['wall_seconds']
    base_labels = base['state_labels']

    print('\n=== Comparison vs results/mvp/hierarchical_smoke/ baseline ===')
    print(f'  Wall-time: baseline {base_walls:.1f}s -> patched {elapsed:.1f}s '
          f'(speedup {base_walls/elapsed:.1f}x)')
    print(f'  BIC:       baseline {base_bic:.4f}  patched {new_bic:.4f}  '
          f'(rel|d| = {abs(new_bic - base_bic)/abs(base_bic):.2e})')
    print(f'  Final LL:  baseline {base_ll:.4f}  patched {new_ll:.4f}  '
          f'(rel|d| = {abs(new_ll - base_ll)/abs(base_ll):.2e})')
    print(f'  baseline state_labels: {base_labels}')

    # The state-label ordering depends on the fit-specific Hungarian alignment;
    # an exact label-by-label match is not expected. We compare the set of
    # labels and their per-state d_emit profiles after Hungarian-aligning.
    print(f'  baseline mean_d_emit shape: {base_d.shape}')
    print(f'  patched  d_emit shape:      {new_d.shape}')

    if new_d.shape == base_d.shape:
        # Try Hungarian alignment
        from scipy.optimize import linear_sum_assignment
        K = new_d.shape[0]
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.sum((base_d[i] - new_d[j]) ** 2)
        row, col = linear_sum_assignment(cost)
        new_d_aligned = new_d[col]
        max_diff = np.abs(base_d - new_d_aligned).max()
        print(f'  After Hungarian alignment (perm={col.tolist()}):')
        print(f'    max|d| per-state d_emit = {max_diff:.4f}')
        for k in range(K):
            d = np.abs(base_d[k] - new_d_aligned[k]).max()
            print(f'    state {k} ({base_labels[k]:6s}): max|d| per-channel = {d:.4f}')

    # Save patched result for posterity
    out = REPO_ROOT / 'results' / 'migration' / 'dynamax' / 'phase0_hierarchical_smoke.json'
    summary = {
        'baseline_path': str(base_path.relative_to(REPO_ROOT)),
        'baseline_bic': base_bic,
        'baseline_ll': base_ll,
        'baseline_wall_seconds': base_walls,
        'patched_bic': new_bic,
        'patched_ll': new_ll,
        'patched_wall_seconds': elapsed,
        'rel_bic_diff': abs(new_bic - base_bic) / abs(base_bic),
        'rel_ll_diff': abs(new_ll - base_ll) / abs(base_ll),
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
