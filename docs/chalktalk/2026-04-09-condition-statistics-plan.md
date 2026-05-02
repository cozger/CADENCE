# Within-Session Condition Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether significant within-session condition differences are emerging across 24 coupling metrics in 15 V11 sessions, with 7 pre-specified contrasts.

**Architecture:** Single runner script loads all V11 scaffold outputs, extracts per-session per-condition means for 24 metrics, runs 168 Wilcoxon signed-rank tests, applies BH-FDR, and produces structured JSON, summary markdown, and paired dot-plot figures.

**Tech Stack:** numpy, scipy.stats (wilcoxon, false_discovery_control), matplotlib, json

---

### Data Layout (from exploration)

**Session directories:** `results/v11/{session_name}/`
- `scaffold_v11_ztimecourses.npz` — 28D z-scored channels + 7D covariates, each as `z_{key}` and `z_raw_{key}`
- `scaffold_v11_results.json` — `segments` list of `[cond_name, t_start_lsl, t_end_lsl]`
- `v11_rslds_results.npz` — `gamma` (T,4), `path` (T,), `t_common` (T,)

**15 sessions discovered.** Session-to-protocol mapping inferred from segment names (presence of `meditate_B`/`meditate_K` vs `PE_1`/`PE_2`).

**Quirks to handle:**
- `y01` has `PE` (unsplit) — treat as PE_1 equivalent for PE_1-vs-base_EO contrast, skip PE_2 contrast
- `y04` has `baseline` (unsplit) — skip (no base_EO/base_EC distinction)
- Some sessions missing conditions (y24 no base_EC, y_17 no base_EO, y_19 no conv_1, y_32 no conv_1)

---

### Task 1: Build the runner script — data loading and metric extraction

**Files:**
- Create: `scripts/run_condition_statistics.py`

- [ ] **Step 1: Write session discovery and loading**

```python
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
from scipy.stats import false_discovery_control
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────

RESULTS_DIR = 'results/v11'
OUT_DIR = 'results/v11/condition_statistics'

# Metrics to extract: (display_name, source, key_or_callable)
# source: 'scaffold' = from NPZ z_raw_{key}, 'covariate' = from NPZ u_{key},
#         'state' = from rSLDS gamma, 'derived' = computed from scaffold channels

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

# State usage metrics (fraction of Viterbi path in each state)
STATE_NAMES = ['NULL', 'COUP', 'SHARED', 'OTHER']

# Coupling excess groups (Tier 1): indices into 28D raw scaffold
# These match the V10 COUPLING_GROUPS from coupling_bursts.py
COUPLING_EXCESS_GROUPS = {
    'EEG Phase':   ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'Facial':      ['bl_expr'],
    'LZ Shared':   ['lz_conc_theta', 'lz_conc_alpha'],
    'Respiratory':  ['resp'],
    'Postural':    ['pose'],
}

# Burst rate groups: detect bursts from scaffold z > threshold
BURST_RATE_GROUPS = {
    'EEG Phase':  ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'EEG Power':  ['conc_theta', 'conc_alpha', 'conc_beta'],
    'Face+Body':  ['bl_expr', 'bl_activity_conc', 'pose'],
    'LZ':         ['lz_conc_theta', 'lz_conc_alpha'],
}
BURST_Z_THRESH = 2.0  # z > 2.0 counts as a coupling burst
FS_OUT = 2.0  # scaffold sample rate

# TE directed episode fractions: from scaffold TE asymmetry covariates
# Therapist→Patient = positive te_asym (by convention), Patient→Therapist = negative
TE_EPISODE_Z_THRESH = 2.0

# Contrasts: (name, condition_a, condition_b, session_filter)
# session_filter: 'all', 'meditation', 'pe'
CONTRASTS = [
    ('meditate_B vs base_EC',  'meditate_B', 'base_EC',  'meditation'),
    ('meditate_K vs base_EC',  'meditate_K', 'base_EC',  'meditation'),
    ('PE_1 vs base_EO',        'PE_1',       'base_EO',  'pe'),
    ('PE_2 vs base_EO',        'PE_2',       'base_EO',  'pe'),
    ('conv_1 vs conv_2',       'conv_1',     'conv_2',   'all'),
    ('conv_1 vs base_EO',      'conv_1',     'base_EO',  'all'),
    ('conv_2 vs base_EO',      'conv_2',     'base_EO',  'all'),
]


def discover_sessions(results_dir):
    """Find all session directories with V11 scaffold + rSLDS outputs."""
    sessions = []
    for d in sorted(os.listdir(results_dir)):
        sess_dir = os.path.join(results_dir, d)
        npz = os.path.join(sess_dir, 'scaffold_v11_ztimecourses.npz')
        js = os.path.join(sess_dir, 'scaffold_v11_results.json')
        rslds = os.path.join(sess_dir, 'v11_rslds_results.npz')
        if os.path.isfile(npz) and os.path.isfile(js) and os.path.isfile(rslds):
            sessions.append({'name': d, 'dir': sess_dir, 'npz': npz, 'json': js, 'rslds': rslds})
    return sessions


def load_session(sess):
    """Load scaffold NPZ, results JSON, and rSLDS results for one session."""
    npz = np.load(sess['npz'], allow_pickle=True)
    with open(sess['json']) as f:
        meta = json.load(f)
    rslds = np.load(sess['rslds'], allow_pickle=True)
    return npz, meta, rslds


def get_condition_mask(t_common, segments, cond_name):
    """Return boolean mask for timepoints belonging to a condition.

    Handles PE (unsplit) as PE_1 equivalent.
    """
    target = cond_name
    for seg in segments:
        name, t_start, t_end = seg[0], seg[1], seg[2]
        # Handle y01-style 'PE' as PE_1
        if cond_name == 'PE_1' and name == 'PE':
            target = 'PE'
        if name == target:
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


def extract_metrics(npz, meta, rslds):
    """Extract all 24 metrics for each condition present in this session.

    Returns: dict of {condition_name: {metric_name: float}}
    """
    t_common = npz['t_common']
    segments = meta['segments']
    path = rslds['path']
    # Align rSLDS t_common to scaffold t_common (should match but be safe)
    rslds_t = rslds['t_common']

    all_conds = set(s[0] for s in segments)
    result = {}

    for cond_name in all_conds:
        mask = get_condition_mask(t_common, segments, cond_name)
        if mask is None or mask.sum() < 10:
            continue

        metrics = {}
        n_samples = int(mask.sum())

        # 1. Scaffold channel metrics (raw z-timecourse means)
        for display, key in SCAFFOLD_METRICS:
            raw_key = f'z_raw_{key}'
            if raw_key in npz:
                metrics[display] = float(np.nanmean(npz[raw_key][mask]))

        # 2. Covariate metrics
        for display, key in COVARIATE_METRICS:
            u_key = f'u_{key}'
            if u_key in npz:
                metrics[display] = float(np.nanmean(npz[u_key][mask]))

        # 3. State usage (fraction of Viterbi path)
        path_seg = path[mask]
        for si, sname in enumerate(STATE_NAMES):
            metrics[f'state_{sname}'] = float(np.mean(path_seg == si))

        # 4. Coupling excess (RMS of group channels)
        for group_name, keys in COUPLING_EXCESS_GROUPS.items():
            vals = []
            for k in keys:
                raw_key = f'z_raw_{k}'
                if raw_key in npz:
                    vals.append(npz[raw_key][mask])
            if vals:
                stacked = np.column_stack(vals)
                rms = np.sqrt(np.nanmean(stacked**2, axis=1))
                metrics[f'excess_{group_name}'] = float(np.nanmean(rms))

        # 5. Burst rates (per minute): fraction of timepoints with z > threshold
        for group_name, keys in BURST_RATE_GROUPS.items():
            vals = []
            for k in keys:
                raw_key = f'z_raw_{k}'
                if raw_key in npz:
                    vals.append(npz[raw_key][mask])
            if vals:
                stacked = np.column_stack(vals)
                # Mean across channels, then threshold
                mean_z = np.nanmean(stacked, axis=1)
                burst_frac = np.mean(mean_z > BURST_Z_THRESH)
                dur_min = n_samples / FS_OUT / 60.0
                metrics[f'burst_rate_{group_name}'] = float(burst_frac * n_samples / dur_min) if dur_min > 0 else 0.0

        # 6. TE directed episode fractions
        te_theta_key = 'u_te_asym_theta'
        te_alpha_key = 'u_te_asym_alpha'
        if te_theta_key in npz:
            te_th = npz[te_theta_key][mask]
            metrics['te_T_to_P_theta'] = float(np.mean(te_th > TE_EPISODE_Z_THRESH))
            metrics['te_P_to_T_theta'] = float(np.mean(te_th < -TE_EPISODE_Z_THRESH))
        if te_alpha_key in npz:
            te_al = npz[te_alpha_key][mask]
            metrics['te_T_to_P_alpha'] = float(np.mean(te_al > TE_EPISODE_Z_THRESH))
            metrics['te_P_to_T_alpha'] = float(np.mean(te_al < -TE_EPISODE_Z_THRESH))

        result[cond_name] = metrics

    return result
```

- [ ] **Step 2: Write the statistical testing and FDR correction**

```python
# ── Statistical testing ──────────────────────────────────────────────

def rank_biserial_r(x, y):
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon)."""
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    r_plus = np.sum(ranks[d > 0])
    r_minus = np.sum(ranks[d < 0])
    return float((r_plus - r_minus) / (r_plus + r_minus)) if (r_plus + r_minus) > 0 else 0.0


def run_all_tests(session_data, sessions):
    """Run all 168 Wilcoxon signed-rank tests.

    Args:
        session_data: dict of {session_name: {condition: {metric: value}}}
        sessions: list of session dicts with protocol info

    Returns:
        list of result dicts, one per test
    """
    # Build protocol map
    protocol_map = {}
    for sess in sessions:
        protocol_map[sess['name']] = sess['protocol']

    # Collect all metric names from first complete session
    all_metrics = set()
    for sname, conds in session_data.items():
        for cond, metrics in conds.items():
            all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    results = []

    for contrast_name, cond_a, cond_b, filt in CONTRASTS:
        for metric in all_metrics:
            pairs_a, pairs_b, pair_sessions = [], [], []

            for sname, conds in session_data.items():
                # Session filter
                proto = protocol_map.get(sname, 'unknown')
                if filt == 'meditation' and proto != 'meditation':
                    continue
                if filt == 'pe' and proto != 'pe':
                    continue

                # Check both conditions present with this metric
                ca = cond_a
                # Handle PE (unsplit) as PE_1
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
            rec = {
                'contrast': contrast_name,
                'metric': metric,
                'n_pairs': n,
                'mean_a': float(np.mean(pairs_a)) if n > 0 else None,
                'mean_b': float(np.mean(pairs_b)) if n > 0 else None,
                'mean_diff': float(np.mean(np.array(pairs_a) - np.array(pairs_b))) if n > 0 else None,
                'sem_diff': float(np.std(np.array(pairs_a) - np.array(pairs_b), ddof=1) / np.sqrt(n)) if n > 1 else None,
                'median_diff': float(np.median(np.array(pairs_a) - np.array(pairs_b))) if n > 0 else None,
                'effect_size_r': None,
                'p_uncorrected': None,
                'q_fdr': None,
                'direction': None,
                'sessions': pair_sessions,
                'values_a': pairs_a,
                'values_b': pairs_b,
            }

            if n >= 6:  # Wilcoxon needs n >= 6 for meaningful p
                try:
                    stat, p = wilcoxon(pairs_a, pairs_b, alternative='two-sided')
                    rec['p_uncorrected'] = float(p)
                    rec['effect_size_r'] = rank_biserial_r(pairs_a, pairs_b)
                    rec['direction'] = 'A > B' if rec['mean_diff'] > 0 else 'A < B'
                except ValueError:
                    pass  # all differences zero
            elif n >= 3:
                # Report descriptives even if too few for Wilcoxon
                rec['direction'] = 'A > B' if rec['mean_diff'] > 0 else 'A < B'

            results.append(rec)

    # FDR correction across all tests with valid p-values
    p_vals = [r['p_uncorrected'] for r in results if r['p_uncorrected'] is not None]
    if p_vals:
        p_arr = np.array(p_vals)
        rejected = false_discovery_control(p_arr, method='bh')
        # Compute q-values manually (adjusted p)
        n_tests = len(p_arr)
        sorted_idx = np.argsort(p_arr)
        q_vals = np.empty(n_tests)
        for rank_i, orig_i in enumerate(sorted_idx):
            q_vals[orig_i] = p_arr[orig_i] * n_tests / (rank_i + 1)
        # Enforce monotonicity (q[i] >= q[i-1] in sorted order)
        for i in range(n_tests - 2, -1, -1):
            idx = sorted_idx[i]
            idx_next = sorted_idx[i + 1]
            if q_vals[idx] > q_vals[idx_next]:
                q_vals[idx] = q_vals[idx_next]
        q_vals = np.minimum(q_vals, 1.0)

        qi = 0
        for r in results:
            if r['p_uncorrected'] is not None:
                r['q_fdr'] = float(q_vals[qi])
                qi += 1

    return results
```

- [ ] **Step 3: Write output generation — JSON, markdown, and figures**

```python
# ── Output ───────────────────────────────────────────────────────────

def save_json(results, out_path):
    """Save structured results JSON (strip numpy arrays for serialization)."""
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if k not in ('values_a', 'values_b', 'sessions')}
        c['sessions'] = r['sessions']
        clean.append(c)
    with open(out_path, 'w') as f:
        json.dump({'version': 'v11_condition_statistics', 'n_tests': len(clean),
                    'fdr_method': 'benjamini-hochberg', 'results': clean}, f, indent=2)
    print(f"  Saved {out_path}")


def save_markdown(results, out_path):
    """Save human-readable summary table sorted by p_uncorrected."""
    # Filter to tests with p-values, sort by p
    with_p = [r for r in results if r['p_uncorrected'] is not None]
    with_p.sort(key=lambda r: r['p_uncorrected'])

    lines = ['# V11 Condition Statistics Summary\n']
    lines.append(f'**{len(with_p)} tests with p-values** out of {len(results)} total '
                 f'({len(results) - len(with_p)} had insufficient pairs)\n')

    n_sig_005 = sum(1 for r in with_p if r['p_uncorrected'] < 0.05)
    n_sig_001 = sum(1 for r in with_p if r['p_uncorrected'] < 0.01)
    n_fdr = sum(1 for r in with_p if r['q_fdr'] is not None and r['q_fdr'] < 0.05)
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

        md = r['mean_diff']
        er = r['effect_size_r']
        lines.append(f"| {r['contrast']} | {r['metric']} | {r['n_pairs']} | "
                     f"{md:+.3f} | {er:+.3f} | {p:.4f} | "
                     f"{q:.4f} | {sig} |")

    # Add tests without p-values
    no_p = [r for r in results if r['p_uncorrected'] is None and r['n_pairs'] > 0]
    if no_p:
        lines.append(f'\n### Tests with insufficient pairs (n < 6)\n')
        lines.append('| Contrast | Metric | n | Mean diff | Direction |')
        lines.append('|----------|--------|---|-----------|-----------|')
        for r in no_p:
            md = r['mean_diff']
            if md is not None:
                lines.append(f"| {r['contrast']} | {r['metric']} | {r['n_pairs']} | "
                             f"{md:+.3f} | {r['direction'] or ''} |")

    lines.append('\n---')
    lines.append('Significance: * p<0.05, ** p<0.01, *** p<0.001 (uncorrected); + q<0.05 (BH-FDR)')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {out_path}")


def plot_metric(metric_name, results, out_dir):
    """Paired dot plot for one metric across all 7 contrasts."""
    metric_results = [r for r in results if r['metric'] == metric_name and r['n_pairs'] > 0]
    if not metric_results:
        return

    fig, axes = plt.subplots(1, len(metric_results), figsize=(3.0 * len(metric_results), 5),
                              squeeze=False, sharey=True)
    axes = axes[0]

    for i, r in enumerate(metric_results):
        ax = axes[i]
        va = np.array(r['values_a'])
        vb = np.array(r['values_b'])
        n = len(va)

        # Color by significance
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

        # Paired lines
        for j in range(n):
            ax.plot([0, 1], [va[j], vb[j]], color=color, alpha=alpha * 0.6,
                    linewidth=0.8, zorder=1)

        # Dots
        ax.scatter(np.zeros(n), va, color=color, alpha=alpha, s=30, zorder=2)
        ax.scatter(np.ones(n), vb, color=color, alpha=alpha, s=30, zorder=2)

        # Means
        ax.plot([0, 1], [np.mean(va), np.mean(vb)], color='black',
                linewidth=2, zorder=3, marker='_', markersize=15)

        # Labels
        parts = r['contrast'].split(' vs ')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([parts[0], parts[1]], fontsize=7, rotation=30, ha='right')
        ax.set_xlim(-0.3, 1.3)

        # Title with stats
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
    fig_path = os.path.join(out_dir, f'{safe_name}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

- [ ] **Step 4: Write the main function**

```python
# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V11 within-session condition statistics')
    parser.add_argument('--session', type=str, default=None, help='Single session (debug)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # Discover sessions
    sessions = discover_sessions(RESULTS_DIR)
    if args.session:
        sessions = [s for s in sessions if args.session in s['name']]
    print(f"Found {len(sessions)} sessions")

    # Load and extract
    session_data = {}
    for sess in sessions:
        print(f"  Loading {sess['name']}...")
        npz, meta, rslds = load_session(sess)
        sess['protocol'] = classify_protocol(meta['segments'])
        session_data[sess['name']] = extract_metrics(npz, meta, rslds)
        conds = list(session_data[sess['name']].keys())
        print(f"    Protocol: {sess['protocol']}, conditions: {conds}")

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
    print(f"\nRunning {len(CONTRASTS)} contrasts × metrics...")
    results = run_all_tests(session_data, sessions)
    n_with_p = sum(1 for r in results if r['p_uncorrected'] is not None)
    print(f"  {len(results)} tests total, {n_with_p} with p-values")

    # Summary
    sig_005 = [r for r in results if r['p_uncorrected'] is not None and r['p_uncorrected'] < 0.05]
    sig_001 = [r for r in results if r['p_uncorrected'] is not None and r['p_uncorrected'] < 0.01]
    sig_fdr = [r for r in results if r['q_fdr'] is not None and r['q_fdr'] < 0.05]
    print(f"\n  Significant (uncorrected): {len(sig_005)} at p<0.05, {len(sig_001)} at p<0.01")
    print(f"  Significant (FDR q<0.05):  {len(sig_fdr)}")

    if sig_005:
        print(f"\n  Top results (p<0.05 uncorrected):")
        sig_005.sort(key=lambda r: r['p_uncorrected'])
        for r in sig_005[:15]:
            print(f"    {r['contrast']:30s} | {r['metric']:25s} | "
                  f"n={r['n_pairs']:2d} | diff={r['mean_diff']:+.3f} | "
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
```

- [ ] **Step 5: Run the script**

Run: `cd C:/Users/optilab/desktop/CADENCE && conda activate MCCT && python scripts/run_condition_statistics.py`

Expected: Loads 15 sessions, runs 168 tests, prints summary of significant results, saves JSON + markdown + figures to `results/v11/condition_statistics/`.

- [ ] **Step 6: Review outputs and fix any issues**

Check:
1. `results/v11/condition_statistics/condition_statistics_summary.md` — verify table renders, p-values look reasonable
2. `results/v11/condition_statistics/condition_statistics.json` — verify structure
3. Spot-check 2-3 figures in `results/v11/condition_statistics/` — verify paired lines, significance coloring

- [ ] **Step 7: Commit**

```bash
git add scripts/run_condition_statistics.py
git commit -m "Add within-session condition statistics (7 contrasts × 24 metrics, Wilcoxon + BH-FDR)"
```
