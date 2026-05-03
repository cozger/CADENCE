"""MVP Hierarchical rSLDS — 7D MVP scaffold + 2D covariates.

Per docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md.

Reads results/mvp/<sid>/mvp_scaffold.npz (from _run_mvp_scaffold.py), fits a
hierarchical rSLDS, writes shared params + per-session results to
results/mvp/hierarchical{suffix}/ and results/mvp/<sid>/mvp_rslds_results{suffix}.npz.

Argparse flags for sensitivity variants (per spec V1-V5):
  --K {3,4}              K=3 or K=4 (default 4)
  --null-state           constrain state 0 to zero emissions (default False per
                         spec — V11 module-6 verdict K=4 no-null wins BIC by 13k)
  --protocol {meditation,pe,all}   filter sessions by protocol from digest JSON
  --share-demit          fit shared d_emit (no per-session deviation); used for
                         §V4 identifiability sensitivity (NOT YET WIRED — flagged)
  --suffix STR           append to output dir + per-session NPZ name; empty
                         string overwrites the production fit
  --viterbi-min-dwell N  Viterbi minimum-dwell samples (default 20 = 10s @ 2 Hz);
                         set to 0 for the unconstrained sensitivity check

Resource discipline: this script does NOT itself parallelize over sessions —
the hierarchical fit's E-step is the parallel level (see rslds_model.py).
Threading-backend per Windows DLL fix (must `import torch` before `import numpy`
in entry script).
"""
# torch must be imported before numpy on Windows (torch 2.10 + numpy 2.4 DLL bug)
import torch  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joblib import parallel_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.ingest.quality import list_canonical_sessions
from cadence.io.resources import log_resources
from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds


# ── Constants ───────────────────────────────────────────────────────

MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_event_coincidence',
                    'pose', 'resp', 'ecg_hf']
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']
FS_OUT = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A']


# ── Constrained Viterbi (carried over from _run_v11_hierarchical.py) ─

def constrained_viterbi(gamma, min_dwell=20):
    """Enforce minimum-dwell on the Viterbi path; iteratively reassign
    short-dwell segments to the higher-margin neighbour."""
    if min_dwell <= 0:
        return np.argmax(gamma, axis=1)
    path = np.argmax(gamma, axis=1).copy()
    for _ in range(10):
        changed = False
        starts = [0]
        for t in range(1, len(path)):
            if path[t] != path[t - 1]:
                starts.append(t)
        starts.append(len(path))
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if e - s >= min_dwell:
                continue
            ls = path[s - 1] if i > 0 else path[s]
            lsc = gamma[s:e, ls].mean() if i > 0 else -np.inf
            rs = path[starts[i + 1]] if i < len(starts) - 2 else path[s]
            rsc = gamma[s:e, rs].mean() if i < len(starts) - 2 else -np.inf
            best = ls if lsc >= rsc else rs
            if best != path[s]:
                path[s:e] = best
                changed = True
        if not changed:
            break
    return path


# ── State labeling (adapted from _run_v11_hierarchical for MVP channels) ─

def label_states(mean_d_emit, K, null_state):
    """Assign NULL / COUP / SHARED / OTHER labels from emission profile.

    MVP heuristic:
      - NULL: lowest-norm state (or state 0 if null_state=True)
      - COUP: highest mean (conc_theta + conc_alpha) loading among remaining
      - SHARED: highest bl_activity_conc loading among remaining
      - OTHER: leftover (autonomic-quiescence at K=4 is here)
    """
    labels = [''] * K
    if null_state:
        labels[0] = 'NULL'
        avail = list(range(1, K))
    else:
        norms = np.linalg.norm(mean_d_emit, axis=1)
        null_k = int(np.argmin(norms))
        labels[null_k] = 'NULL'
        avail = [k for k in range(K) if k != null_k]

    # COUP = highest conc_theta + conc_alpha
    conc_idx = [MVP_OBS_CHANNELS.index('conc_theta'),
                MVP_OBS_CHANNELS.index('conc_alpha')]
    conc_score = {k: mean_d_emit[k, conc_idx].sum() for k in avail}
    if avail:
        coup_k = max(avail, key=lambda k: conc_score[k])
        labels[coup_k] = 'COUP'
        avail = [k for k in avail if k != coup_k]

    # SHARED = highest bl_event_coincidence (was bl_activity_conc; renamed
    # post 2026-05-02 — see cadence/significance/face_event_coincidence.py).
    # If neither is in scaffold, fall back to highest pose loading.
    shared_ch = ('bl_event_coincidence' if 'bl_event_coincidence' in MVP_OBS_CHANNELS
                  else 'bl_activity_conc' if 'bl_activity_conc' in MVP_OBS_CHANNELS
                  else 'pose')
    bla_idx = MVP_OBS_CHANNELS.index(shared_ch)
    if avail:
        shared_k = max(avail, key=lambda k: mean_d_emit[k, bla_idx])
        labels[shared_k] = 'SHARED'
        avail = [k for k in avail if k != shared_k]

    for k in avail:
        labels[k] = 'OTHER'  # autonomic-quiescence at K=4
    return labels


def _parse_bool(s):
    return str(s).lower() in ('true', '1', 'yes', 't', 'y')


# ── Cohort loader ───────────────────────────────────────────────────

def discover_mvp_sessions(protocol_filter: str | None = None) -> list[str]:
    """Return canonical sessions with mvp_scaffold.npz present and matching protocol filter."""
    canonical = list_canonical_sessions()
    out = []
    for sid in canonical:
        if not (MVP_ROOT / sid / 'mvp_scaffold.npz').exists():
            continue
        if protocol_filter and protocol_filter != 'all':
            digest_path = DIGEST_ROOT / f'{sid}.json'
            if not digest_path.exists():
                continue
            digest = json.loads(digest_path.read_text())
            if digest.get('protocol', '').lower() != protocol_filter.lower():
                continue
        out.append(sid)
    return out


def load_mvp_session(sid: str):
    """Load one session's MVP scaffold + segments from V11 sidecar.

    Returns (Y, U, mask, t_common, segments).
    """
    mvp = np.load(MVP_ROOT / sid / 'mvp_scaffold.npz')
    Y = mvp['obs'].astype(np.float64)            # (T, 7)
    U = mvp['cov'].astype(np.float64)            # (T, 2)
    mask = mvp['obs_valid']
    t_common = mvp['t_common']
    # Segments come from V11 sidecar — MVP scaffold doesn't reproduce them
    v11_sidecar = REPO_ROOT / 'results' / 'v11' / sid / 'scaffold_v11_results.json'
    segments = []
    if v11_sidecar.exists():
        info = json.loads(v11_sidecar.read_text())
        segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]
    return Y, U, mask, t_common, segments


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=4,
                    help='Number of discrete states (default 4)')
    ap.add_argument('--null-state', type=_parse_bool, default=False,
                    help='Constrain state 0 to zero emissions (default False per '
                         'spec — V11 module-6 verdict K=4 no-null wins BIC by 13k)')
    ap.add_argument('--protocol', choices=['meditation', 'pe', 'all'],
                    default='all',
                    help='Filter sessions by protocol from digest JSON (default all)')
    ap.add_argument('--share-demit', action='store_true',
                    help='[Spec V4] Fit shared d_emit (no per-session deviation). '
                         'NOT YET WIRED in rslds_model.fit_hierarchical_slds — '
                         'flagged as a TODO.')
    ap.add_argument('--suffix', type=str, default='',
                    help='Output directory suffix (e.g. "_k3", "_med", "_pe", '
                         '"_shared_demit"). Empty string overwrites production fit.')
    ap.add_argument('--viterbi-min-dwell', type=int, default=20,
                    help='Viterbi minimum-dwell samples (default 20 = 10s @ 2 Hz). '
                         '0 disables (unconstrained Viterbi sensitivity check).')
    ap.add_argument('--n-restarts', type=int, default=2)
    ap.add_argument('--max-em-iter', type=int, default=80)
    ap.add_argument('--c-shrinkage', type=float, default=0.3,
                    help='Emission C-loading shrinkage to shared C (default 0.3 — '
                         'V10 default; appropriate for the 7D MVP set).')
    args = ap.parse_args()

    suffix = args.suffix
    print('=' * 80)
    print(f'  MVP Hierarchical rSLDS  (K={args.K}, null_state={args.null_state}, '
          f'protocol={args.protocol}, suffix="{suffix}")')
    print('=' * 80)
    log_resources(prefix='  Resources at start: ')

    if args.share_demit:
        print('\n  WARNING: --share-demit is flagged but not yet wired into '
              'fit_hierarchical_slds. The fit will run with per-session d_emit '
              'as usual; a follow-up patch is needed. Skipping the flag.')

    t_wall = time.time()

    # Discover + load sessions
    sids = discover_mvp_sessions(protocol_filter=args.protocol)
    if not sids:
        print(f'\n  No MVP scaffolds found for protocol={args.protocol}. '
              f'Run scripts/_run_mvp_scaffold.py --all first.')
        return

    print(f'\n  Discovered {len(sids)} MVP-scaffold sessions:')
    sessions = []
    for sid in sids:
        Y, U, mask, t_common, segments = load_mvp_session(sid)
        sessions.append((Y, U, mask, t_common, sid, segments))
        print(f'    {sid:25s}: T={Y.shape[0]:5d} ({Y.shape[0]/FS_OUT:.0f}s), '
              f'D_obs={Y.shape[1]}, D_input={U.shape[1]}, segs={len(segments)}')

    # rSLDS config — per spec §Model specification
    cfg = IOHMMConfig(
        K=args.K,
        D_obs=len(MVP_OBS_CHANNELS),     # 7
        D_input=len(MVP_COV_CHANNELS),   # 2
        D_latent=3,                       # spec: BIC-optimal
        n_factors=2,                      # spec: BIC-optimal
        recurrent=True,
        c_shrinkage=args.c_shrinkage,
        n_restarts=args.n_restarts,
        max_em_iter=args.max_em_iter,
        sticky_strength=3.0,
        null_state=args.null_state,
        null_sigma2_cap=5.0,
        viterbi_min_dwell=args.viterbi_min_dwell,
    )

    session_tuples = [(Y, U, mask) for Y, U, mask, _, _, _ in sessions]
    print(f'\n  Fitting hierarchical rSLDS '
          f'(K={cfg.K}, D_latent={cfg.D_latent}, D_obs={cfg.D_obs}, '
          f'D_input={cfg.D_input}, recurrent={cfg.recurrent}, '
          f'c_shrinkage={cfg.c_shrinkage})...', flush=True)

    # Threading backend mandatory on Windows (torch+numpy DLL ordering bug);
    # rslds_model.py P0.4-patched init now uses limit_blas_threads(1) inside
    # workers, so no oversubscription bloat.
    with parallel_backend('threading'):
        result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)

    print(f'\n  BIC: {result["bic"]:.0f}')
    print(f'  Final LL: {result["final_ll"]:.0f}')

    # State labels from shared d_emit
    all_d = [result['sessions'][i].get('d_emit') for i in range(len(sessions))]
    mean_d = np.mean(all_d, axis=0)
    sl = label_states(mean_d, args.K, args.null_state)
    print(f'\n  State labels: {sl}')
    print(f'\n  Shared d_emit (K x 7):')
    for k in range(args.K):
        loading_str = ' '.join(f'{ch}={mean_d[k, i]:+.2f}'
                                for i, ch in enumerate(MVP_OBS_CHANNELS))
        print(f'    S{k}({sl[k]:>6s}): {loading_str}')

    # Transition covariate effects
    if 'S_trans' in result.get('shared', {}):
        S = result['shared']['S_trans']
        print(f'\n  Transition covariate effects (S_trans):')
        for ci, cov_name in enumerate(MVP_COV_CHANNELS):
            print(f'    {cov_name:25s}: max|S| = {np.abs(S[:, :, ci]).max():.3f}')

    # ── Save ────────────────────────────────────────────────────────
    out_dir = MVP_ROOT / f'hierarchical{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'version': 'mvp_v1',
        'variant': suffix.lstrip('_') or 'production',
        'protocol_filter': args.protocol,
        'n_sessions': len(sessions),
        'sessions_used': [s[4] for s in sessions],
        'config': {
            'K': cfg.K, 'D_obs': cfg.D_obs, 'D_input': cfg.D_input,
            'D_latent': cfg.D_latent, 'n_factors': cfg.n_factors,
            'recurrent': cfg.recurrent, 'c_shrinkage': cfg.c_shrinkage,
            'sticky_strength': cfg.sticky_strength, 'null_state': cfg.null_state,
            'viterbi_min_dwell': cfg.viterbi_min_dwell,
        },
        'bic': float(result['bic']),
        'final_ll': float(result['final_ll']),
        'state_labels': sl,
        'modality_keys': list(MVP_OBS_CHANNELS),
        'covariate_keys': list(MVP_COV_CHANNELS),
        'mean_d_emit': mean_d.tolist(),
        'wall_seconds': float(time.time() - t_wall),
    }
    if 'S_trans' in result.get('shared', {}):
        summary['S_trans'] = result['shared']['S_trans'].tolist()

    np.savez_compressed(
        out_dir / 'mvp_hierarchical_params.npz',
        mean_d_emit=mean_d,
        state_labels=np.array(sl, dtype=object),
        modality_keys=np.array(list(MVP_OBS_CHANNELS), dtype=object),
        covariate_keys=np.array(list(MVP_COV_CHANNELS), dtype=object),
        W_trans=result.get('shared', {}).get('W_trans', np.zeros(1)),
        S_trans=result.get('shared', {}).get('S_trans', np.zeros(1)),
        A_dyn=result.get('shared', {}).get('A_dyn', np.zeros(1)),
        Q_dyn=result.get('shared', {}).get('Q_dyn', np.zeros(1)),
    )

    # Per-session npz with both unconstrained and constrained Viterbi paths
    shared_W = result.get('shared', {}).get('W_trans')
    shared_S = result.get('shared', {}).get('S_trans')
    shared_A = result.get('shared', {}).get('A_dyn')
    shared_Q = result.get('shared', {}).get('Q_dyn')

    all_session_results = []
    for i, (Y, U, mask, t, name, segments) in enumerate(sessions):
        gamma = result['sessions'][i]['gamma']
        path_unconstrained = np.argmax(gamma, axis=1)
        path = constrained_viterbi(gamma, min_dwell=args.viterbi_min_dwell)

        # Per-condition usage (constrained path)
        seg_sorted = sorted(segments, key=lambda x: x[1])
        period_data = []
        for pname, t0, t1 in seg_sorted:
            pmask = (t >= t0) & (t <= t1)
            if pmask.sum() < 3:
                continue
            usage = [float((path[pmask] == k).mean()) for k in range(args.K)]
            period_data.append((pname, float(t1 - t0), usage))

        n_trans = int(np.sum(path[1:] != path[:-1]))
        usage = [float((path == k).mean()) for k in range(args.K)]
        all_session_results.append({
            'session': name, 'state_labels': sl,
            'usage': usage, 'n_transitions': n_trans, 'periods': period_data,
        })

        sess_out_dir = MVP_ROOT / name
        sess_out_dir.mkdir(parents=True, exist_ok=True)
        sess_result = result['sessions'][i]
        save_kw = dict(
            gamma=gamma, path=path, path_unconstrained=path_unconstrained,
            t_common=t, state_labels=np.array(sl, dtype=object),
        )
        if sess_result.get('d_emit') is not None:
            save_kw['d_emit'] = sess_result['d_emit']
        if sess_result.get('C_emit') is not None:
            save_kw['C_emit'] = sess_result['C_emit']
        if sess_result.get('R_emit') is not None:
            save_kw['R_emit'] = sess_result['R_emit']
        if sess_result.get('x_smooth') is not None:
            save_kw['x_smooth'] = sess_result['x_smooth'].astype(np.float32)
        if shared_W is not None: save_kw['W_trans'] = shared_W
        if shared_S is not None: save_kw['S_trans'] = shared_S
        if shared_A is not None: save_kw['A_dyn'] = shared_A
        if shared_Q is not None: save_kw['Q_dyn'] = shared_Q
        np.savez_compressed(sess_out_dir / f'mvp_rslds_results{suffix}.npz',
                            **save_kw)

    summary['sessions'] = all_session_results
    (out_dir / 'mvp_hierarchical_results.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')

    print(f'\n  Wall-clock: {summary["wall_seconds"]:.1f}s '
          f'({summary["wall_seconds"]/60:.1f} min)')
    print(f'  Output: {out_dir}/')
    log_resources(prefix='  Resources at end: ')


if __name__ == '__main__':
    main()
