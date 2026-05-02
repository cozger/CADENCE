"""Faster bisection — same patched-vs-baseline comparison but on a 3-session
subset. Gets the answer in ~5 min instead of ~30 min per bisection step.

Procedure:
  1. Load 3 sessions (y_06, Y_10, y_24)
  2. Run fit_hierarchical_slds with current code (patched or partially-reverted)
  3. Save BIC, LL, mean_d_emit, S_trans
  4. Compare to a separately-saved "baseline" 3-session fit
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

SESSIONS_3 = ['y_06', 'Y_10_03182026', 'y_24_022526'.replace('_022526', '022526')]
# Try alternate name forms
ALT_SESSIONS = ['y_06', 'Y_10_03182026', 'y24_022526']


def load(sid):
    mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
    return (mvp['obs'].astype(np.float64),
            mvp['cov'].astype(np.float64),
            mvp['obs_valid'])


def main():
    sids = ALT_SESSIONS
    sessions = []
    for sid in sids:
        if not (MVP_ROOT / sid / 'mvp_scaffold.npz').exists():
            print(f'  Skipping {sid} (no scaffold)')
            continue
        sessions.append(load(sid))
        print(f'  Loaded {sid}: T={sessions[-1][0].shape[0]}')

    cfg = IOHMMConfig(
        K=4, D_obs=7, D_input=2, D_latent=3, n_factors=2,
        recurrent=True, c_shrinkage=0.3,
        n_restarts=2, max_em_iter=80,
        sticky_strength=3.0, null_state=False, null_sigma2_cap=5.0,
    )
    print(f'\nRunning {len(sessions)}-session hierarchical fit...')
    t0 = time.time()
    with parallel_backend('threading'):
        r = fit_hierarchical_slds(sessions, cfg, seed=42, verbose=True)
    elapsed = time.time() - t0
    print(f'\nWall: {elapsed:.1f}s')
    print(f'BIC: {r["bic"]:.1f}')
    print(f'Final LL: {r["final_ll"]:.1f}')


if __name__ == '__main__':
    main()
