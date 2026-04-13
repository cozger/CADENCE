"""CADENCE V11 — Within-session condition statistics.

Loads all V11 scaffold outputs, extracts per-session per-condition means
for 24 coupling metrics, runs Wilcoxon signed-rank tests on 7 contrasts,
applies BH-FDR, and produces JSON + markdown + figures.

Usage:
    python scripts/run_condition_statistics.py
    python scripts/run_condition_statistics.py --session y_06   # single session debug
"""

import os, json, argparse, time
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────

RESULTS_DIR = 'results/v11'
OUT_DIR = 'results/v11/condition_statistics'

# Scaffold channel metrics (mean of raw z-timecourse over condition)
SCAFFOLD_METRICS = [
    ('asym_theta', 'asym_theta'),
    ('asym_alpha', 'asym_alpha'),
    ('asym_beta', 'asym_beta'),
    ('burst_coinc_theta', 'burst_coinc_theta'),
    ('burst_coinc_alpha', 'burst_coinc_alpha'),
    ('burst_coinc_beta', 'burst_coinc_beta'),
]

# Covariate metrics
COVARIATE_METRICS = [
    ('coupling_flexibility', 'coupling_flexibility'),
]

# State usage metrics (fraction of Viterbi path in each state).
# Loaded data-drivenly from the hierarchical fit at runtime — see
# _load_state_labels(). The fit assigns labels by emission structure
# (COUP=argmax imcoh sum, SHARED=argmax conc sum), so the index→label
# permutation is not fixed across refits.
HIERARCHICAL_RESULTS_PATH = 'results/v11/hierarchical/v11_hierarchical_results.json'


def _load_state_labels(path=HIERARCHICAL_RESULTS_PATH):
    """Load index-ordered state labels from the V11 hierarchical fit."""
    with open(path) as f:
        labels = list(json.load(f)['state_labels'])
    assert sorted(labels) == ['COUP', 'NULL', 'OTHER', 'SHARED'], \
        f'Unexpected state labels in {path}: {labels}'
    return labels

# Coupling excess groups (Tier 1): indices into raw scaffold channels
COUPLING_EXCESS_GROUPS = {
    'EEG Phase':  ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'Facial':     ['bl_expr'],
    'LZ Shared':  ['lz_conc_theta', 'lz_conc_alpha'],
    'Respiratory': ['resp'],
    'Postural':   ['pose'],
}

# Burst rate groups: detect bursts from scaffold z > threshold
BURST_RATE_GROUPS = {
    'EEG Phase': ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'EEG Power': ['conc_theta', 'conc_alpha', 'conc_beta'],
    'Face+Body': ['bl_expr', 'bl_activity_conc', 'pose'],
    'LZ':        ['lz_conc_theta', 'lz_conc_alpha'],
}
BURST_Z_THRESH = 2.0
FS_OUT = 2.0

# TE directed episode threshold
TE_EPISODE_Z_THRESH = 2.0

# Contrasts: (name, condition_a, condition_b, session_filter)
CONTRASTS = [
    ('meditate_B vs base_EC', 'meditate_B', 'base_EC', 'meditation'),
    ('meditate_K vs base_EC', 'meditate_K', 'base_EC', 'meditation'),
    ('PE_1 vs base_EO',       'PE_1',       'base_EO', 'pe'),
    ('PE_2 vs base_EO',       'PE_2',       'base_EO', 'pe'),
    ('conv_1 vs conv_2',      'conv_1',     'conv_2',  'all'),
    ('conv_1 vs base_EO',     'conv_1',     'base_EO', 'all'),
    ('conv_2 vs base_EO',     'conv_2',     'base_EO', 'all'),
]


# ── Data loading ─────────────────────────────────────────────────────

def discover_sessions(results_dir):
    """Find all session directories with V11 scaffold + rSLDS outputs."""
    sessions = []
    for d in sorted(os.listdir(results_dir)):
        sess_dir = os.path.join(results_dir, d)
        npz = os.path.join(sess_dir, 'scaffold_v11_ztimecourses.npz')
        js = os.path.join(sess_dir, 'scaffold_v11_results.json')
        rslds = os.path.join(sess_dir, 'v11_rslds_results.npz')
        if os.path.isfile(npz) and os.path.isfile(js) and os.path.isfile(rslds):
            sessions.append({'name': d, 'dir': sess_dir,
                             'npz': npz, 'json': js, 'rslds': rslds})
    return sessions


def load_session(sess):
    """Load scaffold NPZ, results JSON, and rSLDS results for one session."""
    npz = np.load(sess['npz'], allow_pickle=True)
    with open(sess['json']) as f:
        meta = json.load(f)
    rslds = np.load(sess['rslds'], allow_pickle=True)
    return npz, meta, rslds


def get_condition_mask(t_common, segments, cond_name):
    """Return boolean mask for timepoints belonging to a condition."""
    for seg in segments:
        name, t_start, t_end = seg[0], seg[1], seg[2]
        # Handle y01-style 'PE' as PE_1
        if cond_name == 'PE_1' and name == 'PE':
            return (t_common >= t_start) & (t_common <= t_end)
        if name == cond_name:
            return (t_common >= t_start) & (t_common <= t_end)
    return None


def classify_protocol(segments):
    """Classify session as 'meditation' or 'pe' based on segment names."""
    names = [s[0] for s in segments]
    if 'meditate_B' in names or 'meditate_K' in names:
        return 'meditation'
    if 'PE_1' in names or 'PE_2' in names or 'PE' in names:
        return 'pe'
    return 'unknown'


# ── Metric extraction ────────────────────────────────────────────────

def extract_metrics(npz, meta, rslds, state_labels):
    """Extract all metrics for each condition present in this session.

    Args:
        state_labels: index-ordered labels from the hierarchical fit
            (e.g. ['NULL', 'OTHER', 'COUP', 'SHARED']). Index si maps
            to the rSLDS state with that label.

    Returns: dict of {condition_name: {metric_name: float}}
    """
    t_common = npz['t_common']
    segments = meta['segments']
    path = rslds['path']
    obs_mask = npz['obs_mask']  # (T, 28) boolean validity

    # Map modality keys to obs_mask column indices
    modality_keys = list(meta.get('modality_keys', []))

    def _obs_col(key):
        """Find obs_mask column index for a scaffold channel key."""
        for i, mk in enumerate(modality_keys):
            if mk == key:
                return i
        return None

    all_conds = set(s[0] for s in segments)
    result = {}

    for cond_name in all_conds:
        mask = get_condition_mask(t_common, segments, cond_name)
        if mask is None or mask.sum() < 10:
            continue

        metrics = {}
        n_samples = int(mask.sum())

        # 1. Scaffold channel metrics (raw z-timecourse means, obs_mask-aware)
        for display, key in SCAFFOLD_METRICS:
            raw_key = f'z_raw_{key}'
            if raw_key in npz:
                vals = npz[raw_key][mask].copy()
                col = _obs_col(key)
                if col is not None:
                    valid = obs_mask[mask, col]
                    if valid.sum() < 10:
                        continue  # skip metric for this condition
                    vals[~valid] = np.nan
                metrics[display] = float(np.nanmean(vals))

        # 2. Covariate metrics
        for display, key in COVARIATE_METRICS:
            u_key = f'u_{key}'
            if u_key in npz:
                metrics[display] = float(np.nanmean(npz[u_key][mask]))

        # 3. State usage (fraction of Viterbi path).
        # state_labels[si] is the label of the rSLDS state with index si in
        # the hierarchical fit, so path_seg == si correctly counts that state.
        path_seg = path[mask]
        for si, sname in enumerate(state_labels):
            metrics[f'state_{sname}'] = float(np.mean(path_seg == si))

        # 4. Coupling excess (RMS of group channels, obs_mask-aware)
        for group_name, keys in COUPLING_EXCESS_GROUPS.items():
            vals = []
            for k in keys:
                raw_key = f'z_raw_{k}'
                if raw_key in npz:
                    v = npz[raw_key][mask].copy()
                    col = _obs_col(k)
                    if col is not None:
                        valid = obs_mask[mask, col]
                        if valid.sum() < 10:
                            continue
                        v[~valid] = np.nan
                    vals.append(v)
            if vals:
                stacked = np.column_stack(vals)
                rms = np.sqrt(np.nanmean(stacked**2, axis=1))
                metrics[f'excess_{group_name}'] = float(np.nanmean(rms))

        # 5. Burst rates (per minute, obs_mask-aware)
        for group_name, keys in BURST_RATE_GROUPS.items():
            vals = []
            for k in keys:
                raw_key = f'z_raw_{k}'
                if raw_key in npz:
                    v = npz[raw_key][mask].copy()
                    col = _obs_col(k)
                    if col is not None:
                        valid = obs_mask[mask, col]
                        if valid.sum() < 10:
                            continue
                        v[~valid] = np.nan
                    vals.append(v)
            if vals:
                stacked = np.column_stack(vals)
                mean_z = np.nanmean(stacked, axis=1)
                n_bursts = np.nansum(mean_z > BURST_Z_THRESH)
                dur_min = n_samples / FS_OUT / 60.0
                metrics[f'burst_rate_{group_name}'] = float(n_bursts / dur_min) if dur_min > 0 else 0.0

        # 6. TE directed episode fractions (burst-rate gated)
        for band in ['theta', 'alpha']:
            te_key = f'u_te_asym_{band}'
            gate_key = f'burst_gate_{band}'
            if te_key not in npz:
                continue
            te_vals = npz[te_key][mask]
            if gate_key in npz:
                gate = npz[gate_key][mask]
                n_gated = int(gate.sum())
                metrics[f'te_gate_frac_{band}'] = float(gate.mean())
                if n_gated >= 10:
                    te_gated = te_vals[gate]
                    metrics[f'te_T>P_{band}'] = float(np.mean(te_gated > TE_EPISODE_Z_THRESH))
                    metrics[f'te_P>T_{band}'] = float(np.mean(te_gated < -TE_EPISODE_Z_THRESH))
            else:
                # Fallback: ungated (pre-gating scaffold)
                metrics[f'te_T>P_{band}'] = float(np.mean(te_vals > TE_EPISODE_Z_THRESH))
                metrics[f'te_P>T_{band}'] = float(np.mean(te_vals < -TE_EPISODE_Z_THRESH))

        result[cond_name] = metrics

    return result


# ── Statistical testing ──────────────────────────────────────────────

def rank_biserial_r(x, y):
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon)."""
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    r_plus = np.sum(ranks[d > 0])
    r_minus = np.sum(ranks[d < 0])
    denom = r_plus + r_minus
    return float((r_plus - r_minus) / denom) if denom > 0 else 0.0


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    p = np.array(p_values)
    n = len(p)
    sorted_idx = np.argsort(p)
    q = np.empty(n)
    for rank_i, orig_i in enumerate(sorted_idx):
        q[orig_i] = p[orig_i] * n / (rank_i + 1)
    # Enforce monotonicity (descending through sorted order)
    for i in range(n - 2, -1, -1):
        idx = sorted_idx[i]
        idx_next = sorted_idx[i + 1]
        if q[idx] > q[idx_next]:
            q[idx] = q[idx_next]
    return np.minimum(q, 1.0)


def run_all_tests(session_data, sessions):
    """Run all Wilcoxon signed-rank tests across contrasts × metrics."""
    protocol_map = {s['name']: s['protocol'] for s in sessions}

    # Collect all metric names
    all_metrics = set()
    for conds in session_data.values():
        for metrics in conds.values():
            all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    results = []

    for contrast_name, cond_a, cond_b, filt in CONTRASTS:
        for metric in all_metrics:
            pairs_a, pairs_b, pair_sessions = [], [], []

            for sname, conds in session_data.items():
                proto = protocol_map.get(sname, 'unknown')
                if filt == 'meditation' and proto != 'meditation':
                    continue
                if filt == 'pe' and proto != 'pe':
                    continue

                # Resolve condition name (PE → PE_1)
                ca = cond_a
                if ca == 'PE_1' and 'PE' in conds and 'PE_1' not in conds:
                    ca = 'PE'

                if ca not in conds or cond_b not in conds:
                    continue
                if metric not in conds[ca] or metric not in conds[cond_b]:
                    continue

                va = conds[ca][metric]
                vb = conds[cond_b][metric]
                if np.isfinite(va) and np.isfinite(vb):
                    pairs_a.append(va)
                    pairs_b.append(vb)
                    pair_sessions.append(sname)

            n = len(pairs_a)
            a_arr = np.array(pairs_a)
            b_arr = np.array(pairs_b)

            rec = {
                'contrast': contrast_name,
                'metric': metric,
                'n_pairs': n,
                'mean_a': float(a_arr.mean()) if n > 0 else None,
                'mean_b': float(b_arr.mean()) if n > 0 else None,
                'mean_diff': float((a_arr - b_arr).mean()) if n > 0 else None,
                'sem_diff': float((a_arr - b_arr).std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                'median_diff': float(np.median(a_arr - b_arr)) if n > 0 else None,
                'effect_size_r': None,
                'p_uncorrected': None,
                'q_fdr': None,
                'direction': ('A > B' if (a_arr - b_arr).mean() > 0 else 'A < B') if n > 0 else None,
                'sessions': pair_sessions,
                'values_a': pairs_a,
                'values_b': pairs_b,
            }

            if n >= 6:
                try:
                    diffs = a_arr - b_arr
                    # Wilcoxon needs at least one non-zero difference
                    if np.any(diffs != 0):
                        stat, p = wilcoxon(a_arr, b_arr, alternative='two-sided')
                        rec['p_uncorrected'] = float(p)
                        rec['effect_size_r'] = rank_biserial_r(pairs_a, pairs_b)
                except ValueError:
                    pass

            results.append(rec)

    # FDR correction
    p_indices = [(i, r['p_uncorrected']) for i, r in enumerate(results)
                 if r['p_uncorrected'] is not None]
    if p_indices:
        indices, p_vals = zip(*p_indices)
        q_vals = bh_fdr(list(p_vals))
        for idx, q in zip(indices, q_vals):
            results[idx]['q_fdr'] = float(q)

    return results


# ── Output ───────────────────────────────────────────────────────────

def save_json(results, out_path):
    """Save structured results JSON."""
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if k not in ('values_a', 'values_b')}
        clean.append(c)
    with open(out_path, 'w') as f:
        json.dump({'version': 'v11_condition_statistics',
                   'n_tests': len(clean),
                   'fdr_method': 'benjamini-hochberg',
                   'results': clean}, f, indent=2)
    print(f"  Saved {out_path}")


def save_markdown(results, out_path):
    """Save human-readable summary table sorted by p_uncorrected."""
    with_p = [r for r in results if r['p_uncorrected'] is not None]
    with_p.sort(key=lambda r: r['p_uncorrected'])

    n_sig_005 = sum(1 for r in with_p if r['p_uncorrected'] < 0.05)
    n_sig_001 = sum(1 for r in with_p if r['p_uncorrected'] < 0.01)
    n_fdr = sum(1 for r in with_p if r['q_fdr'] is not None and r['q_fdr'] < 0.05)

    lines = ['# V11 Condition Statistics Summary\n']
    lines.append(f'**{len(with_p)} tests with p-values** out of {len(results)} total '
                 f'({len(results) - len(with_p)} had insufficient pairs or zero variance)\n')
    lines.append(f'**Significant uncorrected:** {n_sig_005} at p<0.05, {n_sig_001} at p<0.01')
    lines.append(f'**Significant FDR-corrected:** {n_fdr} at q<0.05\n')

    lines.append('| Contrast | Metric | n | Mean diff | Effect r | p_uncorr | q_FDR | Sig |')
    lines.append('|----------|--------|---|-----------|----------|----------|-------|-----|')

    for r in with_p:
        sig = ''
        p = r['p_uncorrected']
        q = r['q_fdr']
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        if q is not None and q < 0.05:
            sig += '+'

        lines.append(f"| {r['contrast']} | {r['metric']} | {r['n_pairs']} | "
                     f"{r['mean_diff']:+.4f} | {r['effect_size_r']:+.3f} | "
                     f"{p:.4f} | {q:.4f} | {sig} |")

    # Tests without p-values
    no_p = [r for r in results if r['p_uncorrected'] is None and r['n_pairs'] > 0]
    if no_p:
        lines.append(f'\n### Tests with insufficient pairs or zero variance (n < 6 or all diffs = 0)\n')
        lines.append('| Contrast | Metric | n | Mean diff | Direction |')
        lines.append('|----------|--------|---|-----------|-----------|')
        for r in sorted(no_p, key=lambda x: (x['contrast'], x['metric'])):
            md = r['mean_diff']
            if md is not None:
                lines.append(f"| {r['contrast']} | {r['metric']} | {r['n_pairs']} | "
                             f"{md:+.4f} | {r['direction'] or ''} |")

    lines.append('\n---')
    lines.append('Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (uncorrected); '
                 '+ q<0.05 (BH-FDR)')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {out_path}")


def plot_metric(metric_name, results, out_dir):
    """Paired dot plot for one metric across all contrasts with data."""
    metric_results = [r for r in results if r['metric'] == metric_name and r['n_pairs'] > 0]
    if not metric_results:
        return

    n_panels = len(metric_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels + 0.5, 5),
                              squeeze=False, sharey=True)
    axes = axes[0]

    for i, r in enumerate(metric_results):
        ax = axes[i]
        va = np.array(r['values_a'])
        vb = np.array(r['values_b'])
        n = len(va)

        p = r['p_uncorrected']
        if p is not None and p < 0.01:
            color = '#D32F2F'
            alpha = 0.9
        elif p is not None and p < 0.05:
            color = '#1565C0'
            alpha = 0.8
        else:
            color = '#9E9E9E'
            alpha = 0.5

        for j in range(n):
            ax.plot([0, 1], [va[j], vb[j]], color=color, alpha=alpha * 0.6,
                    linewidth=0.8, zorder=1)

        ax.scatter(np.zeros(n), va, color=color, alpha=alpha, s=30, zorder=2, edgecolors='white', linewidths=0.3)
        ax.scatter(np.ones(n), vb, color=color, alpha=alpha, s=30, zorder=2, edgecolors='white', linewidths=0.3)

        ax.plot([0, 1], [np.mean(va), np.mean(vb)], color='black',
                linewidth=2.5, zorder=3, marker='_', markersize=15)

        parts = r['contrast'].split(' vs ')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([parts[0], parts[1]], fontsize=7, rotation=30, ha='right')
        ax.set_xlim(-0.3, 1.3)

        title = f"n={n}"
        if p is not None:
            title += f"\np={p:.3f}"
            er = r['effect_size_r']
            if er is not None:
                title += f", r={er:+.2f}"
        ax.set_title(title, fontsize=7)

        if i == 0:
            ax.set_ylabel(metric_name, fontsize=9)

    fig.suptitle(metric_name, fontsize=11, fontweight='bold')
    fig.tight_layout()

    safe_name = metric_name.replace(' ', '_').replace('+', '_').replace('/', '_')
    safe_name = safe_name.replace('>', '_gt_').replace('<', '_lt_')
    fig_path = os.path.join(out_dir, f'{safe_name}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V11 within-session condition statistics')
    parser.add_argument('--session', type=str, default=None, help='Single session (debug)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # Load state label permutation from the hierarchical fit (data-driven —
    # the index→label mapping changes between refits).
    state_labels = _load_state_labels()
    print(f"Loaded state labels: {state_labels}")

    # Discover sessions
    sessions = discover_sessions(RESULTS_DIR)
    if args.session:
        sessions = [s for s in sessions if args.session in s['name']]
    print(f"Found {len(sessions)} sessions")

    # Load and extract
    session_data = {}
    for sess in sessions:
        print(f"  Loading {sess['name']}...", end='')
        npz, meta, rslds = load_session(sess)
        sess['protocol'] = classify_protocol(meta['segments'])
        session_data[sess['name']] = extract_metrics(npz, meta, rslds, state_labels)
        conds = list(session_data[sess['name']].keys())
        print(f" {sess['protocol']}, {len(conds)} conditions")

    # Count pairs per contrast
    print(f"\nContrast pair counts:")
    for cname, ca, cb, filt in CONTRASTS:
        n = 0
        for sname in session_data:
            proto = next((s['protocol'] for s in sessions if s['name'] == sname), 'unknown')
            if filt == 'meditation' and proto != 'meditation':
                continue
            if filt == 'pe' and proto != 'pe':
                continue
            ca_eff = ca
            if ca == 'PE_1' and 'PE' in session_data[sname] and 'PE_1' not in session_data[sname]:
                ca_eff = 'PE'
            if ca_eff in session_data[sname] and cb in session_data[sname]:
                n += 1
        print(f"  {cname:30s}: n={n}")

    # Run tests
    print(f"\nRunning tests...")
    results = run_all_tests(session_data, sessions)
    n_with_p = sum(1 for r in results if r['p_uncorrected'] is not None)
    print(f"  {len(results)} tests total, {n_with_p} with valid p-values")

    # Summary
    sig_005 = [r for r in results if r['p_uncorrected'] is not None and r['p_uncorrected'] < 0.05]
    sig_001 = [r for r in results if r['p_uncorrected'] is not None and r['p_uncorrected'] < 0.01]
    sig_fdr = [r for r in results if r['q_fdr'] is not None and r['q_fdr'] < 0.05]
    print(f"\n  Significant (uncorrected): {len(sig_005)} at p<0.05, {len(sig_001)} at p<0.01")
    print(f"  Significant (FDR q<0.05):  {len(sig_fdr)}")

    if sig_005:
        print(f"\n  Top results (p<0.05 uncorrected):")
        sig_005.sort(key=lambda r: r['p_uncorrected'])
        for r in sig_005[:20]:
            print(f"    {r['contrast']:30s} | {r['metric']:25s} | "
                  f"n={r['n_pairs']:2d} | diff={r['mean_diff']:+.4f} | "
                  f"r={r['effect_size_r']:+.3f} | p={r['p_uncorrected']:.4f} | "
                  f"q={r['q_fdr']:.4f}")

    # Save outputs
    print(f"\nSaving outputs...")
    save_json(results, os.path.join(OUT_DIR, 'condition_statistics.json'))
    save_markdown(results, os.path.join(OUT_DIR, 'condition_statistics_summary.md'))

    # Figures
    print(f"  Generating figures...")
    all_metrics = sorted(set(r['metric'] for r in results if r['n_pairs'] > 0))
    for m in all_metrics:
        plot_metric(m, results, OUT_DIR)
    print(f"  {len(all_metrics)} figures saved to {OUT_DIR}/")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
