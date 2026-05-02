"""Phase 0.1 — baseline cProfile of MVP single-session fit_slds.

Profiles one fit_slds call on y_06 with the production MVP config but with
a reduced EM iter count (10) and single restart, to keep wall-clock under
~60s while still producing per-function cumulative timings that are
representative of the per-iter cost breakdown.

Output: results/migration/dynamax/baseline_profile_singlesession.txt
        (top-50 cumulative-time entries, plus wall-clock summary).
"""
# torch must be imported before numpy on Windows (torch 2.10 + numpy 2.4 DLL bug)
import torch  # noqa: F401

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.significance.rslds_model import IOHMMConfig, fit_slds


REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
OUT_DIR = REPO_ROOT / 'results' / 'migration' / 'dynamax'

MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']


def main():
    sid = 'y_06'
    mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
    Y = mvp['obs'].astype(np.float64)
    U = mvp['cov'].astype(np.float64)
    mask = mvp['obs_valid']

    print(f'Profiling fit_slds on {sid}: T={Y.shape[0]}, D_obs={Y.shape[1]}, '
          f'D_input={U.shape[1]}')
    print(f'mask coverage: {mask.mean():.3f} ({mask.sum()} / {mask.size})')

    # Production MVP config but reduced for profile speed.
    # Use K=4 (current production default per --K=4 in _run_mvp_hierarchical)
    # and recurrent=True (load-bearing, per §0).
    cfg = IOHMMConfig(
        K=4,
        D_obs=len(MVP_OBS_CHANNELS),    # 7
        D_input=len(MVP_COV_CHANNELS),  # 2
        D_latent=3,
        n_factors=2,
        recurrent=True,
        c_shrinkage=0.3,
        n_restarts=1,        # 1 instead of 2 — saves init-phase time
        max_em_iter=10,      # 10 instead of 80 — still ~per-iter representative
        sticky_strength=3.0,
        null_state=False,
        null_sigma2_cap=5.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    params, history = fit_slds(Y, U, mask, cfg, seed=42, verbose=False)
    profiler.disable()
    elapsed = time.time() - t0

    print(f'\nWall-clock: {elapsed:.2f}s for {cfg.max_em_iter} EM iters '
          f'({elapsed/cfg.max_em_iter:.2f}s/iter)')
    print(f'Final LL: {history["ll_trace"][-1]:.1f}')
    print(f'LL trace: {[f"{l:.0f}" for l in history["ll_trace"]]}')

    # Top-50 by cumulative time
    out_path = OUT_DIR / 'baseline_profile_singlesession.txt'
    with out_path.open('w') as fh:
        fh.write(f'# Phase 0.1 baseline profile — fit_slds(y_06, MVP config)\n')
        fh.write(f'# T={Y.shape[0]}, D_obs={Y.shape[1]}, D_input={U.shape[1]}, '
                 f'K={cfg.K}, D_latent={cfg.D_latent}, n_factors={cfg.n_factors}, '
                 f'recurrent={cfg.recurrent}\n')
        fh.write(f'# n_restarts={cfg.n_restarts}, max_em_iter={cfg.max_em_iter}\n')
        fh.write(f'# Wall-clock: {elapsed:.2f}s '
                 f'({elapsed/cfg.max_em_iter:.2f}s per EM iter)\n')
        fh.write(f'# Final LL: {history["ll_trace"][-1]:.1f}\n')
        fh.write('# ----------------------------------------------------\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('cumulative')
        ps.print_stats(50)
        fh.write('\n\n# ===== Top 30 by tottime (excludes called subroutines) =====\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('tottime')
        ps.print_stats(30)

    print(f'\nProfile written to {out_path}')


if __name__ == '__main__':
    main()
