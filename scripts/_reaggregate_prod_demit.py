"""Re-aggregate the K=4 production fit's per-session emission parameters
using MEDIAN across sessions (robust to unidentified per-session values
in sessions that never visit a state).

Also report the model-predicted mean E[y | state] = median across sessions
of (C[k] · E[x | path==k] + d[k]).

Reads per-session npz files (which carry per-session C_emit, d_emit, etc.)
and the production JSON (for state labels and the canonical exclusion list).

Writes:
  - results/mvp/hierarchical/aggregation_median.txt   — printable table
  - results/mvp/hierarchical/aggregation_median.json  — machine-readable

Usage:
    python scripts/_reaggregate_prod_demit.py
    python scripts/_reaggregate_prod_demit.py --include-excluded   # include y04/y11/y24 too
"""
import torch  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'
_SESS_NPZ = f'mvp_rslds_results{_SUFFIX}.npz'

JSON_PATH = HIER / 'mvp_hierarchical_results.json'


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


def aggregate(only_canonical=True, min_state_usage_for_demit=0.01):
    with open(JSON_PATH) as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    sessions_meta = {s['session']: s for s in d['sessions']}

    canonical = load_canonical_set() if only_canonical else None

    # Discover modality keys from one session
    one_sid = next(iter(sessions_meta))
    with open(REPO / 'results/mvp' / one_sid / 'mvp_scaffold.json') as f:
        mod_keys = json.load(f)['observation_channels']

    rows = []  # per-session per-state summary
    skipped = []
    for sid, sm in sessions_meta.items():
        if canonical is not None and sid not in canonical:
            skipped.append((sid, 'non-canonical'))
            continue
        sd = REPO / 'results/mvp' / sid
        try:
            rs = np.load(sd / _SESS_NPZ, allow_pickle=True)
        except FileNotFoundError:
            skipped.append((sid, 'no rslds npz'))
            continue
        sl = list(rs['state_labels'])
        path = rs['path']
        x = rs['x_smooth']
        T = len(path)
        # Map this session's state ordering to the production label order
        idx_map = [sl.index(lbl) if lbl in sl else None for lbl in state_labels]
        if any(i is None for i in idx_map):
            skipped.append((sid, f'missing states: {[lbl for lbl, i in zip(state_labels, idx_map) if i is None]}'))
            continue
        for k_disp, k_local in enumerate(idx_map):
            in_state = path == k_local
            usage = float(in_state.mean())
            if usage < min_state_usage_for_demit:
                # Per-session d_emit / C·E[x] is unidentified — skip this row
                # (will be excluded from the per-state median)
                continue
            d_e = rs['d_emit'][k_local]            # (n_obs,)
            C   = rs['C_emit'][k_local]            # (n_obs, D_lat)
            x_mean = x[in_state].mean(axis=0)
            pred_y = C @ x_mean + d_e               # (n_obs,) model-predicted mean of y
            rows.append({
                'session': sid, 'state': state_labels[k_disp], 'state_idx': k_disp,
                'usage': usage, 'd_emit': d_e.tolist(),
                'pred_y_mean': pred_y.tolist(),
            })

    return {
        'state_labels': state_labels, 'mod_keys': mod_keys,
        'rows': rows, 'skipped': skipped, 'K': K,
        'only_canonical': only_canonical,
    }


def summarise(agg):
    state_labels = agg['state_labels']
    mod_keys = agg['mod_keys']
    K = agg['K']
    n_obs = len(mod_keys)

    # Stack per-state matrices
    out_d_emit_med = np.full((K, n_obs), np.nan)
    out_d_emit_mean = np.full((K, n_obs), np.nan)
    out_pred_med = np.full((K, n_obs), np.nan)
    out_pred_mean = np.full((K, n_obs), np.nan)
    n_per_state = np.zeros(K, dtype=int)

    for k in range(K):
        rs_k = [r for r in agg['rows'] if r['state_idx'] == k]
        n_per_state[k] = len(rs_k)
        if not rs_k:
            continue
        D = np.array([r['d_emit'] for r in rs_k])
        P = np.array([r['pred_y_mean'] for r in rs_k])
        out_d_emit_med[k]  = np.median(D, axis=0)
        out_d_emit_mean[k] = D.mean(axis=0)
        out_pred_med[k]    = np.median(P, axis=0)
        out_pred_mean[k]   = P.mean(axis=0)

    return {
        'state_labels': state_labels, 'mod_keys': mod_keys,
        'd_emit_median': out_d_emit_med,
        'd_emit_mean':   out_d_emit_mean,
        'pred_y_median': out_pred_med,
        'pred_y_mean':   out_pred_mean,
        'n_per_state':   n_per_state,
    }


def fmt_table(s, kind='d_emit_median'):
    sl = s['state_labels']
    mk = s['mod_keys']
    M = s[kind]
    out = []
    out.append(f'  {"state":12s}  ' + '  '.join(f'{m[:6]:>6s}' for m in mk) + '   ||·||   N')
    for k, lbl in enumerate(sl):
        cols = '  '.join(f'{v:+6.2f}' for v in M[k])
        out.append(f'  {k} ({lbl:6s})  {cols}   '
                   f'{np.linalg.norm(M[k]):>5.3f}  {s["n_per_state"][k]:>2d}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-excluded', action='store_true',
                    help='include y04/y11/y24 etc. (i.e., everything that was in the prod fit)')
    args = ap.parse_args()

    out_dir = HIER
    cohort_suffix = '_all' if args.include_excluded else '_canonical'
    out_txt = out_dir / f'aggregation_median_{_VARIANT}{cohort_suffix}.txt'
    out_json = out_dir / f'aggregation_median_{_VARIANT}{cohort_suffix}.json'

    agg = aggregate(only_canonical=not args.include_excluded,
                     min_state_usage_for_demit=0.01)
    s = summarise(agg)

    lines = []
    lines.append('=' * 80)
    lines.append(f'  K=4 production re-aggregation '
                 f'({"all sessions in fit" if args.include_excluded else "canonical sessions only"})')
    lines.append('=' * 80)
    lines.append(f'  variant     : K=4 production')
    lines.append(f'  N rows      : {len(agg["rows"])} per-(session × state) entries')
    lines.append(f'  per-state N : {s["n_per_state"].tolist()}')
    lines.append(f'  excluded    : ' + (', '.join(f'{sid}({why})' for sid, why in agg['skipped'])
                                          if agg['skipped'] else 'none'))
    lines.append(f'  min state usage cutoff for d_emit aggregation: 1%')
    lines.append('')
    lines.append('[d_emit  — MEDIAN across sessions]   <-- preferred')
    lines.append(fmt_table(s, 'd_emit_median'))
    lines.append('')
    lines.append('[d_emit  — MEAN across sessions  (legacy, included for comparison)]')
    lines.append(fmt_table(s, 'd_emit_mean'))
    lines.append('')
    lines.append('[predicted mean y per state = C[k]·E[x|k] + d[k] — MEDIAN across sessions]')
    lines.append('(this is the right thing to compare to the empirical state-portrait box-medians)')
    lines.append(fmt_table(s, 'pred_y_median'))
    lines.append('')
    lines.append('[predicted mean y per state = C[k]·E[x|k] + d[k] — MEAN across sessions]')
    lines.append(fmt_table(s, 'pred_y_mean'))

    text = '\n'.join(lines)
    print(text)
    out_txt.write_text(text)

    # JSON output
    out_json.write_text(json.dumps({
        'state_labels': s['state_labels'], 'mod_keys': s['mod_keys'],
        'n_per_state': s['n_per_state'].tolist(),
        'd_emit_median': s['d_emit_median'].tolist(),
        'd_emit_mean':   s['d_emit_mean'].tolist(),
        'pred_y_median': s['pred_y_median'].tolist(),
        'pred_y_mean':   s['pred_y_mean'].tolist(),
        'sessions_used': sorted({r['session'] for r in agg['rows']}),
        'sessions_skipped': agg['skipped'],
        'only_canonical': agg['only_canonical'],
    }, indent=2))

    print(f'\nWrote {out_txt} and {out_json}')


if __name__ == '__main__':
    main()
