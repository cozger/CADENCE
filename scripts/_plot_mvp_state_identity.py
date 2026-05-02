"""'State identity cards' — make the case for each rSLDS state via multiple
complementary lenses, not just the instantaneous-observation distribution.

For each of the K=4 production states, render a 4-panel identity card showing:

  (1) Empirical observation distribution (boxplot per channel) — what the
      dyad looks like in this state (the d_emit / E[y|k] view)
  (2) Per-condition residency — when this state appears, by experimental block
  (3) Transition outflow — which states this state tends to go to next
      (probability of leaving for state k', given a leave from this state)
  (4) Dwell-time distribution — how long does this state typically last

The four lenses together characterize the state much more strongly than the
instantaneous-observation view alone, especially for states where the d_emit
offsets are small (NULL, OTHER, SHARED) but the transition / residency /
duration profile is distinctive.

Usage:
    python scripts/_plot_mvp_state_identity.py
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
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
import os
_VARIANT = os.environ.get('MVP_VARIANT', 'prod')
_SUFFIX = '' if _VARIANT == 'prod' else f'_{_VARIANT}'
HIER = REPO / f'results/mvp/hierarchical{_SUFFIX}'
_SESS_NPZ = f'mvp_rslds_results{_SUFFIX}.npz'

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

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1',
                    'PE_1', 'PE_2',
                    'meditate_B', 'meditate_K', 'conv_2']


def load_canonical_set():
    with open(REPO / 'configs/session_quality.yaml') as f:
        yq = yaml.safe_load(f)
    return {sid for sid, e in (yq.get('sessions') or {}).items() if e.get('canonical')}


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


def collect_pooled():
    with open(HIER / 'mvp_hierarchical_results.json') as f:
        d = json.load(f)
    state_labels = d['state_labels']
    K = d['config']['K']
    canonical = load_canonical_set()
    sids = [s['session'] for s in d['sessions'] if s['session'] in canonical]
    fs = 2.0

    with open(REPO / 'results/mvp' / sids[0] / 'mvp_scaffold.json') as f:
        meta0 = json.load(f)
    mod_keys = meta0['observation_channels']
    cov_keys = meta0['covariate_channels']

    obs_list, valid_list, lat_list, path_list, cond_list, cov_list = [], [], [], [], [], []
    dwell_per_state = {k: [] for k in range(K)}
    # Transition outflow: for each (from, to) pair, count how many times we
    # leave 'from' and end up in 'to'.
    trans_counts = np.zeros((K, K), dtype=int)
    pred_per_session = {k: [] for k in range(K)}

    for sid in sids:
        sd = REPO / 'results/mvp' / sid
        scaff = np.load(sd / 'mvp_scaffold.npz')
        rs = np.load(sd / _SESS_NPZ, allow_pickle=True)
        with open(REPO / 'data/digest/v1' / f'{sid}.json') as f:
            digest = json.load(f)

        obs   = scaff['obs']
        valid = scaff['obs_valid']
        t_lsl = scaff['t_common']
        path_local = rs['path']
        x = rs['x_smooth']

        # Map per-session state ordering -> production label ordering
        sl = list(rs['state_labels'])
        idx_map = {sl.index(lbl): k_disp for k_disp, lbl in enumerate(state_labels)}
        path_disp = np.array([idx_map[int(p)] for p in path_local], dtype=np.int8)

        # Dwell durations per state (display-order)
        cur_state = path_disp[0]
        cur_start = 0
        for t in range(1, len(path_disp)):
            if path_disp[t] != cur_state:
                dwell_per_state[int(cur_state)].append((t - cur_start) / fs)
                # Transition: leaving cur_state -> path_disp[t]
                trans_counts[int(cur_state), int(path_disp[t])] += 1
                cur_state = path_disp[t]
                cur_start = t
        dwell_per_state[int(cur_state)].append((len(path_disp) - cur_start) / fs)

        # Per-session model-predicted E[y|k] for diamond overlay
        for k_disp, lbl in enumerate(state_labels):
            k_local = sl.index(lbl)
            in_state = path_local == k_local
            if in_state.mean() < 0.01:
                continue
            x_mean = x[in_state].mean(axis=0)
            C = rs['C_emit'][k_local]
            d_e = rs['d_emit'][k_local]
            pred_per_session[k_disp].append(C @ x_mean + d_e)

        # Condition assignment per timepoint
        periods = parse_periods(digest['markers'])
        ci = np.full(len(t_lsl), '', dtype=object)
        for name, t0, t1 in periods:
            mask = (t_lsl >= t0) & (t_lsl < t1)
            ci[mask] = name

        obs_list.append(obs)
        valid_list.append(valid)
        path_list.append(path_disp)
        cond_list.append(ci)
        cov_list.append(scaff['cov'])

    obs = np.concatenate(obs_list, axis=0)
    valid = np.concatenate(valid_list, axis=0)
    path = np.concatenate(path_list, axis=0)
    cond = np.concatenate(cond_list, axis=0)
    cov = np.concatenate(cov_list, axis=0)

    # Stack per-session predictions
    n_obs = len(mod_keys)
    pred_y_median = np.full((K, n_obs), np.nan)
    pred_y_iqr_lo = np.full((K, n_obs), np.nan)
    pred_y_iqr_hi = np.full((K, n_obs), np.nan)
    n_sess = np.zeros(K, dtype=int)
    for k in range(K):
        if pred_per_session[k]:
            arr = np.stack(pred_per_session[k])
            pred_y_median[k] = np.median(arr, axis=0)
            pred_y_iqr_lo[k] = np.percentile(arr, 25, axis=0)
            pred_y_iqr_hi[k] = np.percentile(arr, 75, axis=0)
            n_sess[k] = arr.shape[0]

    return {
        'state_labels': state_labels, 'mod_keys': mod_keys,
        'cov_keys': cov_keys,
        'obs': obs, 'valid': valid, 'path': path, 'cond': cond, 'cov': cov,
        'sids': sids, 'K': K, 'fs': fs,
        'dwell_per_state': dwell_per_state,
        'trans_counts': trans_counts,
        'pred_y_median': pred_y_median,
        'pred_y_iqr_lo': pred_y_iqr_lo, 'pred_y_iqr_hi': pred_y_iqr_hi,
        'n_sess_per_state': n_sess,
    }


def render_identity_cards(P, out_path, A_dyn=None):
    state_labels = P['state_labels']
    K = P['K']
    n_ch = len(P['mod_keys'])
    state_colors = [STATE_COLOR_BY_LABEL.get(l, '#37474F') for l in state_labels]
    display_keys = [CHANNEL_DISPLAY.get(k, k) for k in P['mod_keys']]
    cov_keys = P['cov_keys']
    n_cov = len(cov_keys)

    # Layout: K rows × 5 cols (5 lenses per state)
    fig = plt.figure(figsize=(24, 4.0 * K))
    gs = fig.add_gridspec(K, 5, width_ratios=[2.6, 1.7, 1.4, 1.4, 1.7],
                          hspace=0.55, wspace=0.32,
                          left=0.05, right=0.98, top=0.95, bottom=0.04)

    for k in range(K):
        col = state_colors[k]

        # ── Lens 1: Empirical obs distribution + model E[y|k] median ──
        ax1 = fig.add_subplot(gs[k, 0])
        in_state = P['path'] == k
        bp_data = []
        for c_idx in range(n_ch):
            mask = in_state & P['valid'][:, c_idx]
            vals = P['obs'][mask, c_idx]
            vals = vals[np.isfinite(vals)]
            bp_data.append(vals if vals.size else np.array([0.0]))
        positions = np.arange(n_ch)
        bp = ax1.boxplot(bp_data, positions=positions, vert=False,
                         widths=0.6, showfliers=False, patch_artist=True,
                         medianprops=dict(color='black', linewidth=1.5),
                         whiskerprops=dict(linewidth=0.8, color='#37474F'),
                         capprops=dict(linewidth=0.8, color='#37474F'))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor('#263238')
            patch.set_linewidth(0.8)
        # Model E[y|k] median diamond + IQR
        for ch_i in range(n_ch):
            if not np.isfinite(P['pred_y_median'][k, ch_i]):
                continue
            ax1.plot([P['pred_y_iqr_lo'][k, ch_i], P['pred_y_iqr_hi'][k, ch_i]],
                     [ch_i, ch_i], color='#D32F2F', linewidth=1.5, alpha=0.5,
                     zorder=4, solid_capstyle='butt')
        ax1.scatter(P['pred_y_median'][k], positions, marker='D', s=70,
                    color='#D32F2F', edgecolors='black', linewidths=1.0, zorder=5)
        ax1.axvline(0, color='gray', linewidth=0.8, alpha=0.5, zorder=1)
        for ref in (-2, -1, 1, 2):
            ax1.axvline(ref, color='gray', linewidth=0.4, alpha=0.25,
                        linestyle=':', zorder=1)
        ax1.set_yticks(positions)
        ax1.set_yticklabels(display_keys, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('z-scored observation', fontsize=8)
        ax1.set_xlim(-4, 5)
        ax1.tick_params(labelsize=8)
        ax1.grid(axis='x', alpha=0.25, linestyle=':')
        if k == 0:
            ax1.set_title('Lens 1: Observation distribution\n'
                          '(box: 25–75 %, whiskers: 5–95 %; ◆ = median model E[y|k])',
                          fontsize=10, pad=10)

        # ── Lens 2: Per-condition residency (% of time IN this state, by condition) ──
        ax2 = fig.add_subplot(gs[k, 1])
        cond_in = P['cond'][in_state]
        # For each condition: P(in state k | condition) = N(state k, cond) / N(cond)
        cond_pcts = []
        cond_labels_used = []
        for cn in CONDITION_ORDER:
            n_cond = (P['cond'] == cn).sum()
            n_state_cond = (in_state & (P['cond'] == cn)).sum()
            if n_cond == 0:
                continue
            cond_pcts.append(100 * n_state_cond / n_cond)
            cond_labels_used.append(cn)
        positions2 = np.arange(len(cond_labels_used))
        bars = ax2.barh(positions2, cond_pcts, color=col, alpha=0.7,
                        edgecolor='#263238', linewidth=0.5)
        # Mark cohort-pooled mean for reference
        cohort_mean = (P['path'] == k).mean() * 100
        ax2.axvline(cohort_mean, color='black', linewidth=1.0,
                    linestyle='--', alpha=0.6, zorder=3)
        ax2.text(cohort_mean, len(cond_labels_used) - 0.5,
                 f' overall {cohort_mean:.0f}%',
                 fontsize=7, color='black', va='center')
        for i, pct in enumerate(cond_pcts):
            ax2.text(pct + 1, i, f'{pct:.0f}%', va='center', fontsize=7)
        ax2.set_yticks(positions2)
        ax2.set_yticklabels(cond_labels_used, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('% of condition time in this state', fontsize=8)
        ax2.set_xlim(0, max(60, max(cond_pcts) * 1.15))
        ax2.tick_params(labelsize=8)
        ax2.grid(axis='x', alpha=0.25, linestyle=':')
        if k == 0:
            ax2.set_title('Lens 2: When this state appears\n'
                          '(% of condition time spent in this state; --- = cohort mean)',
                          fontsize=10, pad=10)

        # ── Lens 3: Transition outflow (where do we go from here?) ──
        ax3 = fig.add_subplot(gs[k, 2])
        out = P['trans_counts'][k].astype(float)
        if out.sum() > 0:
            out_pct = 100 * out / out.sum()
        else:
            out_pct = out
        # Bar per other state (skip self-loop = same state never appears since
        # transitions only count actual state changes)
        bar_colors = [STATE_COLOR_BY_LABEL.get(state_labels[k2], '#37474F')
                      for k2 in range(K)]
        # Mute the self-bar (will be 0 since we count only changes)
        mask_other = np.array([k2 != k for k2 in range(K)])
        positions3 = np.arange(K)
        ax3.bar(positions3, out_pct, color=bar_colors, alpha=0.7,
                edgecolor='#263238', linewidth=0.5)
        for i, pct in enumerate(out_pct):
            if pct > 0.5:
                ax3.text(i, pct + 1, f'{pct:.0f}%', ha='center', fontsize=7)
        ax3.set_xticks(positions3)
        ax3.set_xticklabels(state_labels, fontsize=8, rotation=20)
        ax3.set_ylabel('% of departures', fontsize=8)
        ax3.set_ylim(0, max(50, max(out_pct) * 1.15))
        ax3.tick_params(labelsize=8)
        ax3.grid(axis='y', alpha=0.25, linestyle=':')
        n_trans = int(out.sum())
        if k == 0:
            ax3.set_title('Lens 3: Where it goes next\n'
                          '(% of departures by destination state)',
                          fontsize=10, pad=10)
        ax3.text(0.98, 0.95, f'N={n_trans} departures',
                 transform=ax3.transAxes, ha='right', va='top',
                 fontsize=7, color='#37474F',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # ── Lens 5 (NEW): Per-state covariate distribution + A_dyn signature ──
        ax5 = fig.add_subplot(gs[k, 4])
        cov_data = []
        for j in range(n_cov):
            vals = P['cov'][in_state, j]
            vals = vals[np.isfinite(vals)]
            cov_data.append(vals if vals.size else np.array([0.0]))
        positions5 = np.arange(n_cov)
        bp5 = ax5.boxplot(cov_data, positions=positions5, vert=False,
                          widths=0.6, showfliers=False, patch_artist=True,
                          medianprops=dict(color='black', linewidth=1.5))
        for patch in bp5['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor('#263238')
            patch.set_linewidth(0.8)
        # Mark cohort-pooled mean for each covariate (reference line)
        for j in range(n_cov):
            cohort_med = float(np.nanmedian(P['cov'][:, j]))
            ax5.plot([cohort_med, cohort_med], [j - 0.35, j + 0.35],
                     color='black', linewidth=1.0, linestyle='--', alpha=0.7)
        ax5.set_yticks(positions5)
        ax5.set_yticklabels([k.replace('_', ' ') for k in cov_keys], fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel('z-scored covariate value', fontsize=8)
        ax5.tick_params(labelsize=7)
        ax5.grid(axis='x', alpha=0.25, linestyle=':')

        # Per-state A_dyn eigenvalue signature as text annotation
        if A_dyn is not None:
            eigs = np.linalg.eigvals(A_dyn[k])
            mods = np.abs(eigs)
            angles = np.angle(eigs)  # radians per timestep
            # Decay timescale (s) per mode at fs=2 Hz: tau = -1 / (fs * log|eig|)
            taus = []
            for m in mods:
                if m >= 1.0:
                    taus.append(float('inf'))
                else:
                    taus.append(-1 / (P['fs'] * np.log(m)))
            # Oscillation period for complex eigenvalues
            periods = []
            for ang in angles:
                if abs(ang) > 1e-3:
                    periods.append(2 * np.pi / (abs(ang) * P['fs']))
                else:
                    periods.append(None)
            txt = f'A_dyn[{k}] eigenvalues:\n'
            for i, (m, ang, tau, per) in enumerate(zip(mods, angles, taus, periods)):
                tau_s = '∞' if tau == float('inf') else f'{tau:.0f}s'
                osc = f', osc {per:.0f}s' if per else ''
                txt += f'  λ{i}: |·|={m:.3f}, τ={tau_s}{osc}\n'
            ax5.text(1.02, 0.95, txt.rstrip(), transform=ax5.transAxes,
                     ha='left', va='top', fontsize=7, family='monospace',
                     bbox=dict(facecolor='white', edgecolor=col,
                               boxstyle='round,pad=0.3', linewidth=1.0))

        if k == 0:
            ax5.set_title('Lens 5: Covariate context\n'
                          '(per-state graph-theoretic covariates; --- = cohort median)\n'
                          'inset: A_dyn eigenvalues + decay/oscillation timescale',
                          fontsize=9, pad=10)

        # ── Lens 4: Dwell-time distribution ──
        ax4 = fig.add_subplot(gs[k, 3])
        dwells = np.array(P['dwell_per_state'][k])
        if dwells.size:
            # Log-y or linear histogram with median marker
            ax4.hist(dwells, bins=30, color=col, alpha=0.7, edgecolor='black',
                     linewidth=0.4)
            med_d = float(np.median(dwells))
            mean_d = float(dwells.mean())
            ax4.axvline(med_d, color='black', linewidth=1.4, alpha=0.85,
                        linestyle='-', zorder=3)
            ax4.axvline(mean_d, color='red', linewidth=1.0, alpha=0.7,
                        linestyle='--', zorder=3)
            ax4.text(0.98, 0.95,
                     f'N={len(dwells)} dwells\n'
                     f'median={med_d:.0f}s\n'
                     f'mean={mean_d:.0f}s',
                     transform=ax4.transAxes, ha='right', va='top',
                     fontsize=7,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        ax4.set_xlabel('dwell duration (s)', fontsize=8)
        ax4.set_ylabel('count', fontsize=8)
        ax4.tick_params(labelsize=8)
        ax4.grid(axis='y', alpha=0.25, linestyle=':')
        if k == 0:
            ax4.set_title('Lens 4: How long it lasts\n'
                          '(dwell duration histogram; — = median, --- = mean)',
                          fontsize=10, pad=10)

        # State-name banner on the left
        for ax in (ax1, ax2, ax3, ax4, ax5):
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(2.0)
        # Annotate state name on the leftmost panel
        ax1.text(-0.32, 0.5, state_labels[k], transform=ax1.transAxes,
                 ha='center', va='center', fontsize=18, color=col,
                 fontweight='bold', rotation=90)

    fig.suptitle(f'MVP rSLDS state identity cards — N={len(P["sids"])} canonical sessions, '
                 f'4 complementary lenses per state',
                 fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    print('=== Collecting pooled data ===')
    P = collect_pooled()
    K = P['K']
    print(f'  states: {P["state_labels"]}')
    for k in range(K):
        d = P['dwell_per_state'][k]
        print(f'  {P["state_labels"][k]:6s}: {len(d)} dwells, '
              f'median={np.median(d):.1f}s, mean={np.mean(d):.1f}s')

    # Load A_dyn for eigenvalue annotation
    npz = np.load(HIER / 'mvp_hierarchical_params.npz', allow_pickle=True)
    A_dyn = npz['A_dyn']

    out = HIER / f'state_identity_{_VARIANT}.png'
    render_identity_cards(P, out, A_dyn=A_dyn)


if __name__ == '__main__':
    main()
