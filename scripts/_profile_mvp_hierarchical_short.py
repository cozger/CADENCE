"""Hierarchical baseline profile — 3 sessions, 10 EM iters.

Used as the Tier A+B decision gate. Reports:
  - Wall-time decomposition: Phase 1 init / Phase 4 EM
  - Per-call cumulative time on the sequential pooled M-step transitions
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

# y_24 scaffold lives at y24_022526 (date-suffixed directory name)
# y_06 and y_17 are the canonical short names
SESSIONS = [
    ('y_06',        'y_06'),
    ('y_17',        'y_17'),
    ('y_24',        'y24_022526'),  # date-suffixed; substituted from spec's y_24
]
MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']


def main():
    sess = []
    for label, sid in SESSIONS:
        path = MVP_ROOT / sid / 'mvp_scaffold.npz'
        if not path.exists():
            print(f'WARNING: {label} ({sid}) scaffold missing at {path}')
            continue
        mvp = np.load(path)
        Y = mvp['obs'].astype(np.float64)
        U = mvp['cov'].astype(np.float64)
        mask = mvp['obs_valid']
        sess.append((Y, U, mask))
        print(f'  Loaded {label} ({sid}): Y={Y.shape}, U={U.shape}')

    if len(sess) < 2:
        print('ERROR: need at least 2 sessions to profile hierarchical fit')
        sys.exit(1)

    # Infer D_obs / D_input from actual scaffold data (avoids hardcode mismatch).
    # Drop sessions whose D_obs or D_input differ from the first session — the
    # hierarchical fit requires all sessions to share the same observation shape.
    d_obs_actual = sess[0][0].shape[1]
    d_input_actual = sess[0][1].shape[1]
    sess = [(Y, U, mask) for Y, U, mask in sess
            if Y.shape[1] == d_obs_actual and U.shape[1] == d_input_actual]
    if len(sess) < 2:
        print(f'ERROR: fewer than 2 sessions share D_obs={d_obs_actual}; '
              'rebuild MVP scaffolds so all sessions match.')
        sys.exit(1)
    print(f'  Using {len(sess)} sessions with D_obs={d_obs_actual}, D_input={d_input_actual}')

    cfg = IOHMMConfig(
        K=4, D_obs=d_obs_actual, D_input=d_input_actual,
        D_latent=3, n_factors=2, recurrent=True, c_shrinkage=0.3,
        n_restarts=1, max_em_iter=10,  # short, representative per-iter
        sticky_strength=3.0, null_state=False, null_sigma2_cap=5.0,
    )
    print(f'\nConfig: K={cfg.K}, D_obs={cfg.D_obs}, D_input={cfg.D_input}, '
          f'D_latent={cfg.D_latent}, n_factors={cfg.n_factors}, '
          f'max_em_iter={cfg.max_em_iter}')
    print(f'Sessions: {len(sess)}\n')

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    result = fit_hierarchical_slds(sess, cfg, seed=42, verbose=False)
    profiler.disable()
    elapsed = time.time() - t0

    out_path = OUT_DIR / 'baseline_profile_hierarchical_3sess.txt'
    with out_path.open('w') as fh:
        fh.write(f'# Hierarchical baseline profile — {len(sess)} sessions, '
                 f'{cfg.max_em_iter} EM iters, K={cfg.K}\n')
        fh.write(f'# Sessions: {[s[0] for s in SESSIONS[:len(sess)]]}\n')
        fh.write(f'# Wall-clock: {elapsed:.2f}s\n')
        fh.write(f'# Final BIC: {result.get("bic", "n/a")}\n')
        fh.write('# ----------------------------------------------\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('cumulative')
        ps.print_stats(60)
        fh.write('\n# ===== Top 30 by tottime =====\n\n')
        ps = pstats.Stats(profiler, stream=fh).sort_stats('tottime')
        ps.print_stats(30)

        # Gate-target functions (load-bearing for Tier A vs A+B decision)
        fh.write('\n# ===== Gate-target sequential M-step functions =====\n\n')
        ps = pstats.Stats(profiler, stream=fh)
        # Filter to specific function names that are the load-bearing measurement
        ps.print_stats('_hierarchical_m_step')
        fh.write('\n# ===== Per-session HMM-init M-step (Phase 1, parallel — for context) =====\n')
        fh.write('# Note: runs in Parallel(prefer="threads"); cProfile only captures main thread.\n')
        fh.write('# Empty section is expected — not a filter miss.\n\n')
        ps.print_stats('rslds_model.py:561')  # exact line filter — avoids _hierarchical_ collision
        fh.write('\n# ===== rSLDS recurrent M-step (called from Phase 1 _init_one) =====\n\n')
        ps.print_stats('slds_m_step_transitions_recurrent')

    print(f'Wall: {elapsed:.2f}s, profile -> {out_path}')
    print(f'Final BIC: {result.get("bic", "n/a")}')


if __name__ == '__main__':
    main()
