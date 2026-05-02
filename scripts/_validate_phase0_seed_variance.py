"""Phase 0 — measure inherent EM-stochasticity by running same patched code
with two different seeds. If seed-to-seed variance in BIC matches the
patched-vs-baseline gap, then the gap is just inherent EM noise from
different local optima, not a bug.

This runs the full 22-session production hierarchical fit.
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

from cadence.ingest.quality import list_canonical_sessions
from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds


REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']


def discover():
    out = []
    for sid in list_canonical_sessions():
        if (MVP_ROOT / sid / 'mvp_scaffold.npz').exists():
            out.append(sid)
    return out


def load(sid):
    mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
    return (mvp['obs'].astype(np.float64),
            mvp['cov'].astype(np.float64),
            mvp['obs_valid'])


def run_one(sessions, seed):
    cfg = IOHMMConfig(
        K=4, D_obs=7, D_input=2, D_latent=3, n_factors=2,
        recurrent=True, c_shrinkage=0.3,
        n_restarts=2, max_em_iter=80,
        sticky_strength=3.0, null_state=False, null_sigma2_cap=5.0,
    )
    t0 = time.time()
    with parallel_backend('threading'):
        r = fit_hierarchical_slds(sessions, cfg, seed=seed, verbose=False)
    # 'usage' is computed post-fit from gamma (the run script does this
    # after the fact via the constrained-Viterbi path, but raw gamma usage
    # is sufficient for our cross-comparison).
    usage = np.array([s['gamma'].mean(axis=0) for s in r['sessions']])
    return {
        'seed': seed,
        'bic': float(r['bic']),
        'final_ll': float(r['final_ll']),
        'wall_seconds': float(time.time() - t0),
        'usage': usage.tolist(),
    }


def main():
    sids = discover()
    print(f'Loading {len(sids)} sessions...')
    sessions = [load(s) for s in sids]
    print('Running seed=42...')
    r42 = run_one(sessions, seed=42)
    print(f'  BIC={r42["bic"]:.1f}  LL={r42["final_ll"]:.1f}  wall={r42["wall_seconds"]:.1f}s')
    print('Running seed=43...')
    r43 = run_one(sessions, seed=43)
    print(f'  BIC={r43["bic"]:.1f}  LL={r43["final_ll"]:.1f}  wall={r43["wall_seconds"]:.1f}s')

    # Load baseline
    base = json.loads((MVP_ROOT / 'hierarchical' / 'mvp_hierarchical_results.json').read_text())
    base_bic = base['bic']
    base_ll = base['final_ll']
    base_usage = np.array([s['usage'] for s in base['sessions']])

    # Load existing seed=42 result for sanity
    p42_existing = json.loads((MVP_ROOT / 'hierarchical_phase0' / 'mvp_hierarchical_results.json').read_text())
    print(f'\nSanity check: existing seed=42 BIC={p42_existing["bic"]:.1f} matches new run={r42["bic"]:.1f}: {abs(p42_existing["bic"] - r42["bic"]) < 0.1}')

    print('\n=== Comparison ===')
    print(f'  baseline (seed=42, un-patched): BIC={base_bic:.1f}  LL={base_ll:.1f}')
    print(f'  patched (seed=42):              BIC={r42["bic"]:.1f}  LL={r42["final_ll"]:.1f}  diff vs baseline: BIC={r42["bic"]-base_bic:+.1f} LL={r42["final_ll"]-base_ll:+.1f}')
    print(f'  patched (seed=43):              BIC={r43["bic"]:.1f}  LL={r43["final_ll"]:.1f}  diff vs baseline: BIC={r43["bic"]-base_bic:+.1f} LL={r43["final_ll"]-base_ll:+.1f}')
    print(f'  seed-to-seed BIC variance:      |seed42 - seed43| = {abs(r42["bic"]-r43["bic"]):.1f}')

    # Per-session usage divergence
    usage_42 = np.array(r42['usage'])
    usage_43 = np.array(r43['usage'])
    print('\n=== Per-session max state-usage divergence ===')
    print(f'  seed42 vs seed43 (same patched code, diff seed): max |d|={np.abs(usage_42-usage_43).max():.3f}')
    print(f'  seed42 patched vs baseline:                       max |d|={np.abs(usage_42-base_usage).max():.3f}')
    print(f'  seed43 patched vs baseline:                       max |d|={np.abs(usage_43-base_usage).max():.3f}')

    out = REPO_ROOT / 'results' / 'migration' / 'dynamax' / 'phase0_seed_variance.json'
    out.write_text(json.dumps({'r42': {k: v for k, v in r42.items() if k != 'usage'},
                                'r43': {k: v for k, v in r43.items() if k != 'usage'},
                                'baseline_bic': base_bic, 'baseline_ll': base_ll}, indent=2))
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
