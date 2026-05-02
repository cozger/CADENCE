"""State portraits: what does each rSLDS coupling state look like?

For each of the K=4 production states, pool all timepoints across all
sessions where Viterbi assigns the state, and visualise:

  - Empirical distribution of all 7 observation channels (boxplot)
    showing what the multimodal coupling profile actually looks like
    while the dyad is in that state
  - Model emission belief (mean_d_emit) as a marker overlay
  - Top conditions during which the state appears
  - Mean dwell duration and state share of cohort time

Reads from the per-session scaffold + rSLDS results (already on disk),
not the hierarchical fit JSON, so we work directly with the data the
model saw.

Usage:
    python scripts/_plot_mvp_state_portraits.py
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',
    'OTHER':  '#9C27B0',
    'SHARED': '#2E7D32',
    'COUP':   '#FF6F00',
}

CHANNEL_DISPLAY = {
    'conc_theta':       'EEG conc θ',
    'conc_alpha':       'EEG conc α',
    'bl_expr':          'BL expression',
    'bl_activity_conc': 'BL activity conc.',
    'pose':             'pose',
    'resp':             'respiration',
    'ecg_hf':           'ECG HF',
}


def parse_periods(markers):
    starts = {}
    out = []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    return sorted(out, key=lambda x: x[1])


def state_runs(path):
    """Return list of (start_idx, end_idx, state) runs."""
    if len(path) == 0:
        return []
    runs = []
    cs = 0
    cur = int(path[0])
    for t in range(1, len(path)):
        if path[t] != cur:
            runs.append((cs, t, cur))
            cs = t
            cur = int(path[t])
    runs.append((cs, len(path), cur))
    return runs


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


def collect_pooled(prod_json, only_canonical=True, min_state_usage=0.01, sess_npz_name='mvp_rslds_results.npz'):
    """Pool obs + path + condition-per-timepoint across canonical sessions.
    Also compute per-session model-predicted E[y|k] = C[k]·E[x|k] + d[k]
    so we can take its median across sessions for the portrait overlay."""
    with open(prod_json) as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    fs = 2.0  # MVP scaffold rate

    canonical = load_canonical_set() if only_canonical else None
    all_sids = [s['session'] for s in d['sessions']]
    sids = [s for s in all_sids if (canonical is None or s in canonical)]
    skipped = [s for s in all_sids if s not in sids]

    with open(REPO / 'results/mvp' / sids[0] / 'mvp_scaffold.json') as f:
        meta0 = json.load(f)
    mod_keys = meta0['observation_channels']

    n_obs_ch = len(mod_keys)

    # Per-session model-predicted mean of y per state, for median overlay.
    # Shape (K, n_obs_ch) at the end after stacking + median.
    pred_per_session = {k: [] for k in range(K)}  # k -> list of (n_obs_ch,) arrays
    for sid in sids:
        sd = REPO / 'results/mvp' / sid
        try:
            rs = np.load(sd / sess_npz_name, allow_pickle=True)
        except FileNotFoundError:
            continue
        sl = list(rs['state_labels'])
        if not all(l in sl for l in state_labels):
            continue
        path = rs['path']
        x = rs['x_smooth']
        for k_disp, lbl in enumerate(state_labels):
            k_local = sl.index(lbl)
            in_state = path == k_local
            if in_state.mean() < min_state_usage:
                continue
            x_mean = x[in_state].mean(axis=0)
            C = rs['C_emit'][k_local]
            d_e = rs['d_emit'][k_local]
            pred_per_session[k_disp].append(C @ x_mean + d_e)
    pred_y_median = np.full((K, n_obs_ch), np.nan)
    pred_y_iqr_lo = np.full((K, n_obs_ch), np.nan)
    pred_y_iqr_hi = np.full((K, n_obs_ch), np.nan)
    n_sess_per_state = np.zeros(K, dtype=int)
    for k in range(K):
        if pred_per_session[k]:
            arr = np.stack(pred_per_session[k])
            pred_y_median[k] = np.median(arr, axis=0)
            pred_y_iqr_lo[k] = np.percentile(arr, 25, axis=0)
            pred_y_iqr_hi[k] = np.percentile(arr, 75, axis=0)
            n_sess_per_state[k] = arr.shape[0]

    # Pooled arrays
    pooled_obs = []          # list of (T, n_obs_ch)
    pooled_valid = []        # list of (T, n_obs_ch)
    pooled_path = []         # list of (T,)
    pooled_cond_idx = []     # list of (T,) — index into cond_names; -1 if none
    pooled_session_idx = []  # list of (T,)

    cond_set = set()
    sess_periods = {}  # sid -> list of (cond_name, t0_lsl, t1_lsl)

    for sid in sids:
        sd = REPO / 'results/mvp' / sid
        scaff = np.load(sd / 'mvp_scaffold.npz')
        rslds = np.load(sd / sess_npz_name, allow_pickle=True)
        with open(REPO / 'data/digest/v1' / f'{sid}.json') as f:
            digest = json.load(f)

        obs = scaff['obs']                # (T, 7)
        valid = scaff['obs_valid']        # (T, 7)
        t_lsl = scaff['t_common']         # (T,) — LSL clock seconds
        path = rslds['path']              # (T,)

        periods = parse_periods(digest['markers'])
        sess_periods[sid] = periods
        cond_set.update(p[0] for p in periods)

        pooled_obs.append(obs)
        pooled_valid.append(valid)
        pooled_path.append(path)
        # Will fill cond_idx after cond_names is defined; for now placeholder
        pooled_cond_idx.append(np.full(len(t_lsl), -1, dtype=int))
        pooled_session_idx.append(np.full(len(t_lsl), len(pooled_path) - 1, dtype=int))

    # Now resolve cond_names ordering
    cond_order = ['base_EO', 'base_EC', 'conv_1',
                  'PE_1', 'PE_2', 'meditate_B', 'meditate_K', 'conv_2']
    cond_names = [c for c in cond_order if c in cond_set]
    cond_names += sorted(cond_set - set(cond_names))

    # Fill cond_idx per session
    for i, sid in enumerate(sids):
        scaff = np.load(REPO / 'results/mvp' / sid / 'mvp_scaffold.npz')
        t_lsl = scaff['t_common']
        ci = np.full(len(t_lsl), -1, dtype=int)
        for name, t0, t1 in sess_periods[sid]:
            if name not in cond_names:
                continue
            j = cond_names.index(name)
            mask = (t_lsl >= t0) & (t_lsl < t1)
            ci[mask] = j
        pooled_cond_idx[i] = ci

    return {
        'state_labels': state_labels,
        'K': K,
        'fs': fs,
        'mod_keys': mod_keys,
        'n_obs_ch': n_obs_ch,
        'cond_names': cond_names,
        'sids': sids,
        'skipped': skipped,
        'obs':       np.concatenate(pooled_obs, axis=0),
        'valid':     np.concatenate(pooled_valid, axis=0),
        'path':      np.concatenate(pooled_path, axis=0),
        'cond_idx':  np.concatenate(pooled_cond_idx, axis=0),
        'sess_idx':  np.concatenate(pooled_session_idx, axis=0),
        'pred_y_median':       pred_y_median,
        'pred_y_iqr_lo':       pred_y_iqr_lo,
        'pred_y_iqr_hi':       pred_y_iqr_hi,
        'n_sess_per_state':    n_sess_per_state,
    }


def state_portraits(prod_json, out_path, only_canonical=True, sess_npz_name='mvp_rslds_results.npz'):
    P = collect_pooled(prod_json, only_canonical=only_canonical, sess_npz_name=sess_npz_name)
    K = P['K']
    fs = P['fs']
    state_labels = P['state_labels']
    mod_keys = P['mod_keys']
    n_ch = P['n_obs_ch']
    state_colors = [STATE_COLOR_BY_LABEL.get(l, '#37474F') for l in state_labels]
    display_keys = [CHANNEL_DISPLAY.get(k, k) for k in mod_keys]

    # ── Per-state aggregations ──
    print('\n[per-state pooled stats]')
    state_stats = []  # list of dicts
    for k in range(K):
        in_state = P['path'] == k
        n_t = int(in_state.sum())
        # Per-session dwell durations
        dwells = []
        # Recompute dwells within each session separately to avoid splicing
        offset = 0
        for i, sid in enumerate(P['sids']):
            sd = REPO / 'results/mvp' / sid
            scaff = np.load(sd / 'mvp_scaffold.npz')
            T = len(scaff['t_common'])
            sess_path = P['path'][offset:offset + T]
            for s, e, st in state_runs(sess_path):
                if st == k:
                    dwells.append((e - s) / fs)
            offset += T
        dwells = np.array(dwells)
        # Conditions while in state
        cond_in = P['cond_idx'][in_state]
        cond_in = cond_in[cond_in >= 0]
        n_per_cond = np.bincount(cond_in, minlength=len(P['cond_names']))
        # Top 3 conditions
        order = np.argsort(-n_per_cond)
        top_conds = [(P['cond_names'][j], int(n_per_cond[j]) / fs)
                     for j in order[:3] if n_per_cond[j] > 0]
        state_stats.append({
            'state': state_labels[k],
            'n_timepoints': n_t,
            'fraction':     n_t / len(P['path']),
            'n_dwells':     len(dwells),
            'mean_dwell_s': float(dwells.mean()) if len(dwells) else 0,
            'med_dwell_s':  float(np.median(dwells)) if len(dwells) else 0,
            'top_conds':    top_conds,
        })
        print(f'  {state_labels[k]:6s}: T={n_t:>6d} ({state_stats[-1]["fraction"]*100:.1f}%)  '
              f'dwells={len(dwells):>4d}  mean_dwell={state_stats[-1]["mean_dwell_s"]:.1f}s  '
              f'med_dwell={state_stats[-1]["med_dwell_s"]:.1f}s  '
              f'top: {", ".join(f"{c}({s:.0f}s)" for c,s in top_conds)}')

    # ── Plot: 2x2 grid of state portraits ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for k in range(K):
        ax = axes[k // 2, k % 2]
        in_state = P['path'] == k
        # Boxplot data: per-channel obs values when in state, masking invalid
        bp_data = []
        for c_idx in range(n_ch):
            mask = in_state & P['valid'][:, c_idx]
            vals = P['obs'][mask, c_idx]
            # finite filter
            vals = vals[np.isfinite(vals)]
            bp_data.append(vals if vals.size else np.array([0.0]))

        positions = np.arange(n_ch)
        bp = ax.boxplot(bp_data, positions=positions, vert=False,
                        widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8, color='#37474F'),
                        capprops=dict(linewidth=0.8, color='#37474F'))
        for patch in bp['boxes']:
            patch.set_facecolor(state_colors[k])
            patch.set_alpha(0.55)
            patch.set_edgecolor('#263238')
            patch.set_linewidth(0.8)

        # Overlay the model-predicted state-mean E[y|k] = C[k]·E[x|k] + d[k]
        # taken as the MEDIAN across sessions (skipping sessions where the
        # state has <1% usage and per-session params are unidentified).
        pred_med = P['pred_y_median'][k]
        pred_lo  = P['pred_y_iqr_lo'][k]
        pred_hi  = P['pred_y_iqr_hi'][k]
        # IQR error bars (horizontal) for the model prediction
        for ch_i in range(n_ch):
            if not np.isfinite(pred_med[ch_i]):
                continue
            ax.plot([pred_lo[ch_i], pred_hi[ch_i]], [ch_i, ch_i],
                    color='#D32F2F', linewidth=1.5, alpha=0.5, zorder=4,
                    solid_capstyle='butt')
        ax.scatter(pred_med, positions, marker='D', s=70, color='#D32F2F',
                   edgecolors='black', linewidths=1.0, zorder=5,
                   label=f'model E[y|k] median (IQR bar; N={P["n_sess_per_state"][k]} sess.)')

        # Reference zero line
        ax.axvline(0, color='gray', linewidth=0.8, alpha=0.5, zorder=1)
        # ±1, ±2 z reference
        for ref in (-2, -1, 1, 2):
            ax.axvline(ref, color='gray', linewidth=0.4, alpha=0.25,
                       linestyle=':', zorder=1)

        ax.set_yticks(positions)
        ax.set_yticklabels(display_keys, fontsize=10)
        ax.invert_yaxis()  # so first channel is on top
        ax.set_xlabel('z-scored observation value (per-session)', fontsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-4, 5)
        ax.grid(axis='x', alpha=0.25, linestyle=':')

        # State title with colored band
        st = state_stats[k]
        title_l1 = f'{state_labels[k]}'
        title_l2 = (f'{st["fraction"]*100:.1f}% of cohort time  ·  '
                    f'{st["n_dwells"]} dwells  ·  '
                    f'mean dwell {st["mean_dwell_s"]:.1f}s '
                    f'(median {st["med_dwell_s"]:.1f}s)')
        title_l3 = ('top conditions: ' +
                    '  '.join(f'{c} ({s:.0f}s)' for c, s in st['top_conds'][:3]))
        ax.set_title(f'{title_l1}\n{title_l2}\n{title_l3}',
                     fontsize=10, color=state_colors[k], fontweight='bold',
                     loc='left', pad=8)
        # Color the spines to match the state
        for spine in ax.spines.values():
            spine.set_edgecolor(state_colors[k])
            spine.set_linewidth(2.5)

        if k == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.85)

    n_sess = len(P['sids'])
    n_skip = len(P['skipped'])
    fig.suptitle(f'MVP rSLDS state portraits — empirical obs distributions per state '
                 f'(N={n_sess} canonical sessions; '
                 f'{n_skip} excluded: {", ".join(P["skipped"]) if P["skipped"] else "none"})\n'
                 f'box: 25–75 %, whiskers: 5–95 %; red ◆ = median model-predicted '
                 f'E[y|state] across sessions, bar = IQR',
                 fontsize=11, fontweight='bold')
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.06,
                        hspace=0.65, wspace=0.30)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'\nSaved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='prod',
                    help='prod | k3 | med | pe | no_dwell | <other suffix>')
    args = ap.parse_args()

    suffix = '' if args.variant == 'prod' else f'_{args.variant}'
    json_path = REPO / f'results/mvp/hierarchical{suffix}/mvp_hierarchical_results.json'
    sess_npz_name = f'mvp_rslds_results{suffix}.npz'
    out = REPO / f'results/mvp/hierarchical{suffix}/state_portraits_{args.variant}.png'
    state_portraits(json_path, out, only_canonical=True, sess_npz_name=sess_npz_name)


if __name__ == '__main__':
    main()
