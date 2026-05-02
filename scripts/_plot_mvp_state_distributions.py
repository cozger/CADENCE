"""Per-session × per-condition state distribution heatmaps + chi-square test.

Reads the K=4 production hierarchical fit JSON (which already contains
per-session × per-period state usage as [name, dur_s, usage_K] tuples).
Produces:

  1. 4 heatmaps (one per state) — rows = sessions, cols = conditions,
     color = % of that condition's time spent in that state. Lets you
     scan whether each state's cohort-level pattern holds session-by-session.
  2. A pooled stacked-bar showing cohort-mean state usage per condition
     with cross-session error bars.
  3. A chi-square test of state × condition independence on the
     cohort-pooled contingency table.

Usage:
    python scripts/_plot_mvp_state_distributions.py
    python scripts/_plot_mvp_state_distributions.py --variant pe
"""
import torch  # noqa: F401  -- precede numpy on Windows torch+cu128
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

# Display order for conditions (protocol-aware union)
COND_ORDER = ['base_EO', 'base_EC', 'conv_1',
              'PE_1', 'PE_2',
              'meditate_B', 'meditate_K', 'conv_2']

# State label → color (matches timeline plot palette)
STATE_COLOR_BY_LABEL = {
    'NULL':   '#90A4AE',
    'OTHER':  '#9C27B0',
    'SHARED': '#2E7D32',
    'COUP':   '#FF6F00',
}

# Drop noise-condition rows
SKIP_CONDS = {'baseline', 'PE'}  # legacy single-occurrence labels


def fs_from_session_dir(session_id):
    p = REPO / 'results' / 'mvp' / session_id / 'mvp_scaffold.json'
    if not p.exists():
        return 2.0
    with open(p) as f:
        return float(json.load(f).get('fs_out', 2.0))


def collect(json_path):
    """Return (sessions_list_of_dicts, K, state_labels)."""
    with open(json_path) as f:
        d = json.load(f)
    sessions = d['sessions']
    K = d['config']['K']
    state_labels = d['state_labels']
    return sessions, K, state_labels, d


def build_session_cond_usage(sessions, K):
    """Return:
      cond_names_present : sorted list, in COND_ORDER (only those present)
      session_ids        : list of session ids
      usage  : (n_sessions, n_conds, K) — proportion or NaN if missing
      counts : (n_sessions, n_conds, K) — integer timepoint counts (≈ usage * dur * fs)
      durs   : (n_sessions, n_conds) — duration in seconds (0 if missing)
    """
    sids = [s['session'] for s in sessions]
    # Discover which conditions are present anywhere in the cohort
    present = set()
    for s in sessions:
        for p in s.get('periods', []):
            if isinstance(p, list) and len(p) == 3 and p[0] not in SKIP_CONDS:
                present.add(p[0])
    cond_names = [c for c in COND_ORDER if c in present]
    # Add anything not in COND_ORDER at the end (defensive)
    cond_names += sorted(present - set(cond_names))

    n_s = len(sids)
    n_c = len(cond_names)
    usage = np.full((n_s, n_c, K), np.nan, dtype=float)
    durs = np.zeros((n_s, n_c), dtype=float)
    counts = np.zeros((n_s, n_c, K), dtype=int)

    for i, s in enumerate(sessions):
        fs = fs_from_session_dir(s['session'])
        for p in s.get('periods', []):
            if not (isinstance(p, list) and len(p) == 3):
                continue
            name, dur, u = p
            if name not in cond_names:
                continue
            j = cond_names.index(name)
            u = np.array(u)
            usage[i, j] = u
            durs[i, j] = dur
            # Integer counts ~ dur * fs * usage; preserve sum=round(dur*fs)
            T = int(round(dur * fs))
            cnt = np.round(u * T).astype(int)
            # Adjust last-state count so they sum to T exactly
            diff = T - cnt.sum()
            if cnt.size > 0:
                cnt[int(np.argmax(cnt))] += diff
            counts[i, j] = cnt
    return sids, cond_names, usage, counts, durs


def plot_distributions(sessions, K, state_labels, out_path, title_suffix=''):
    sids, cond_names, usage, counts, durs = build_session_cond_usage(sessions, K)
    n_s = len(sids)
    n_c = len(cond_names)

    state_colors = [STATE_COLOR_BY_LABEL.get(lbl, '#37474F') for lbl in state_labels]

    # ── Layout: 2x2 small heatmaps + cohort stacked bar below ──
    fig = plt.figure(figsize=(max(14, 0.7 * n_c + 8), 4 + 0.30 * n_s))
    gs = fig.add_gridspec(3, 4, height_ratios=[5, 5, 4], hspace=0.55, wspace=0.3,
                          left=0.10, right=0.97, top=0.93, bottom=0.07)

    # Per-state heatmaps (2x2). Use viridis for usage; light grey for n/a cells.
    cmap = matplotlib.colormaps.get_cmap('viridis').copy()
    cmap.set_bad('#E0E0E0')  # masked / n/a cells render as light grey
    for k in range(K):
        ax = fig.add_subplot(gs[k // 2, (k % 2) * 2:(k % 2) * 2 + 2])
        # Mask sessions/conditions where dur=0
        data = usage[:, :, k] * 100  # %
        mask = (durs == 0)
        data_masked = np.ma.array(data, mask=mask)
        im = ax.imshow(data_masked, aspect='auto', cmap=cmap, vmin=0, vmax=100,
                       interpolation='nearest')
        # Highlight border in state color
        for spine in ax.spines.values():
            spine.set_edgecolor(state_colors[k])
            spine.set_linewidth(2.5)
        ax.set_title(f'{state_labels[k]} usage  (% of condition time)',
                     fontsize=11, color=state_colors[k], fontweight='bold')
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(cond_names, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(sids, fontsize=6)
        # Annotate cells with %
        for i in range(n_s):
            for j in range(n_c):
                if mask[i, j]:
                    ax.text(j, i, 'n/a', ha='center', va='center', fontsize=6,
                            color='#757575', fontstyle='italic')
                    continue
                v = data[i, j]
                # Viridis: dark at low values (text needs to be white),
                # light/yellow at high values (text needs to be black).
                txt_color = 'white' if v < 50 else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=6, color=txt_color)
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

    # ── Cohort-pooled stacked bar w/ error bars ──
    ax_bar = fig.add_subplot(gs[2, :])
    # Mean per cond across sessions (ignoring nan)
    mean_usage = np.nanmean(usage, axis=0) * 100  # (n_c, K)
    n_per_cond = np.sum(~np.isnan(usage[:, :, 0]), axis=0)         # (n_c,)
    sem_usage = (np.nanstd(usage, axis=0, ddof=1) /
                 np.sqrt(np.maximum(n_per_cond, 1))[:, None]) * 100  # (n_c, K)

    bottom = np.zeros(n_c)
    x = np.arange(n_c)
    for k in range(K):
        ax_bar.bar(x, mean_usage[:, k], bottom=bottom,
                   color=state_colors[k], label=state_labels[k],
                   edgecolor='white', linewidth=0.5)
        # SEM whiskers at top of each segment
        ax_bar.errorbar(x, bottom + mean_usage[:, k], yerr=sem_usage[:, k],
                        fmt='none', ecolor='black', capsize=3, linewidth=0.7, alpha=0.6)
        # Annotate the mean value inside each segment when tall enough
        for xi, m in enumerate(mean_usage[:, k]):
            if m >= 4:
                ax_bar.text(xi, bottom[xi] + m / 2, f'{m:.0f}%',
                            ha='center', va='center', fontsize=7,
                            color='white' if k != 1 else 'black', fontweight='bold')
        bottom += mean_usage[:, k]
    # Annotate total at top of each bar
    for xi, tot in enumerate(bottom):
        ax_bar.text(xi, tot + 1.5, f'Σ={tot:.0f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#263238')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'{c}\n(n={int(nn)})' for c, nn in zip(cond_names, n_per_cond)],
                           fontsize=9)
    ax_bar.set_ylabel('Mean state usage (%)', fontsize=10)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title('Cohort-pooled state distribution per condition '
                     '(mean ± SEM across sessions)', fontsize=11, fontweight='bold')
    ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=K,
                  frameon=False, fontsize=9)
    ax_bar.grid(axis='y', alpha=0.25, linestyle=':')

    fig.suptitle(f'MVP rSLDS state × condition distributions{title_suffix}',
                 fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved {out_path}')

    # ── Chi-square ──
    return run_chi_square(cond_names, state_labels, counts, sids)


def run_chi_square(cond_names, state_labels, counts, sids):
    """Pooled chi-square test of state × condition independence."""
    K = len(state_labels)
    # Pool across sessions: shape (n_conds, K)
    pooled = counts.sum(axis=0)
    n_c = pooled.shape[0]

    print()
    print('=' * 78)
    print('  CHI-SQUARE: state × condition INDEPENDENCE')
    print('=' * 78)

    # Drop conditions with 0 total counts
    keep = pooled.sum(axis=1) > 0
    contingency = pooled[keep]
    kept_conds = [c for c, k in zip(cond_names, keep) if k]

    print(f'\n[pooled contingency table — counts of timepoints]')
    print(f'  {"condition":15s}  ' +
          '  '.join(f'{lbl:>10s}' for lbl in state_labels) +
          f'  {"row_total":>10s}')
    for c, row in zip(kept_conds, contingency):
        print(f'  {c:15s}  ' +
              '  '.join(f'{int(v):>10d}' for v in row) +
              f'  {int(row.sum()):>10d}')
    col_totals = contingency.sum(axis=0)
    print(f'  {"col_total":15s}  ' +
          '  '.join(f'{int(v):>10d}' for v in col_totals) +
          f'  {int(col_totals.sum()):>10d}')

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f'\n[test result]')
    print(f'  chi2          = {chi2:.1f}')
    print(f'  dof           = {dof}')
    print(f'  p-value       = {p:.3e}')
    print(f'  Cramers V     = {np.sqrt(chi2 / (contingency.sum() * (min(contingency.shape) - 1))):.3f}')
    print(f'  N timepoints  = {int(contingency.sum())}')
    print(f'  decision      : {"REJECT independence (p<0.001)" if p < 1e-3 else "fails to reject independence"}')

    # Standardized residuals (which cells drive the effect)
    std_resid = (contingency - expected) / np.sqrt(expected)
    print(f'\n[standardized residuals — |z|>2 are suggestive, |z|>3 are strong]')
    print(f'  Negative = LESS time in state than independence predicts;')
    print(f'  positive = MORE time in state than independence predicts')
    print()
    print(f'  {"condition":15s}  ' +
          '  '.join(f'{lbl:>10s}' for lbl in state_labels))
    for c, row in zip(kept_conds, std_resid):
        cells = []
        for v in row:
            mark = '***' if abs(v) > 4 else ('**' if abs(v) > 3 else (' *' if abs(v) > 2 else '  '))
            cells.append(f'{v:+7.2f}{mark}')
        print(f'  {c:15s}  ' + '  '.join(f'{cell:>10s}' for cell in cells))

    # Per-session chi-square (informative only; usually significant given many timepoints)
    print(f'\n[per-session chi-square (state × condition, this session only)]')
    for i, sid in enumerate(sids):
        ct = counts[i]
        keep_i = ct.sum(axis=1) > 0
        if keep_i.sum() < 2:
            continue
        ct_i = ct[keep_i]
        try:
            chi2_i, p_i, dof_i, _ = stats.chi2_contingency(ct_i)
        except ValueError:
            continue
        cv = np.sqrt(chi2_i / (ct_i.sum() * (min(ct_i.shape) - 1)))
        flag = '  ***' if p_i < 1e-6 else ('  **' if p_i < 1e-3 else '')
        print(f'  {sid:25s}  chi2={chi2_i:>8.1f}  dof={dof_i:>2d}  '
              f'p={p_i:.2e}  V={cv:.2f}{flag}')

    return chi2, p, dof, contingency, kept_conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='prod',
                    help='prod | k3 | med | pe | no_dwell | <other suffix>')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    suffix = '' if args.variant == 'prod' else f'_{args.variant}'
    json_path = REPO / f'results/mvp/hierarchical{suffix}/mvp_hierarchical_results.json'
    out_path = Path(args.out) if args.out else (
        REPO / 'results/mvp' / json_path.parent.name /
        f'state_x_condition_{args.variant}.png'
    )

    sessions, K, state_labels, _ = collect(json_path)
    print(f'Loaded {len(sessions)} sessions, K={K}, state_labels={state_labels}')
    plot_distributions(sessions, K, state_labels, out_path,
                       title_suffix=f'  ·  variant={args.variant}  ·  N={len(sessions)} sessions')


if __name__ == '__main__':
    main()
