"""MVP Figures — Figure 1 (dyad variability) + Figure 2 (protocol comparison).

Per docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md
§Figure 1 + §Figure 2.

Reads:
  results/mvp/diagnostics/verification_report.md  — to determine K_winner
  results/mvp/hierarchical[_k3]/mvp_hierarchical_results.json
  results/mvp/<sid>/mvp_rslds_results[_k3].npz
  data/digest/v1/<sid>.json                          — for protocol field

Writes:
  results/mvp/figures/figure1_dyad_variability.png|pdf
  results/mvp/figures/figure2_protocol_comparison.png|pdf

Selection criterion (both figures): mean (COUP+SHARED) gamma during conv_1+conv_2.
Figure 1 selects highest + lowest across the full cohort.
Figure 2 selects highest within each protocol (meditation, PE).

Usage:
    python scripts/_make_mvp_figures.py
    python scripts/_make_mvp_figures.py --K 4              # force K choice
"""
from __future__ import annotations

# Windows torch+numpy DLL ordering: must import torch BEFORE numpy.
import torch as _torch  # noqa: F401

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'
FIG_ROOT = MVP_ROOT / 'figures'

STATE_COLORS = {
    'NULL': '#aaaaaa', 'COUP': '#1565C0', 'SHARED': '#2E7D32',
    'OTHER': '#FF6F00',  # autonomic-quiescence
}
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}

CONV_CONDS = ('conv_1', 'conv_2')
COMMON_CONDS = ('base_EO', 'base_EC', 'conv_1', 'conv_2')
MED_CONDS = ('base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2')
PE_CONDS = ('base_EO', 'base_EC', 'conv_1', 'PE_1', 'PE_2', 'conv_2')


# ── Helpers ─────────────────────────────────────────────────────────

def detect_k_winner() -> int:
    """Read verification_report.md to determine K_winner; default to 4."""
    report = MVP_ROOT / 'diagnostics' / 'verification_report.md'
    if not report.exists():
        return 4
    text = report.read_text(encoding='utf-8')
    m = re.search(r'\*\*Production K = (\d+)\*\*', text)
    if m:
        return int(m.group(1))
    m = re.search(r'K_winner = K=(\d+)', text)
    if m:
        return int(m.group(1))
    return 4


def hier_suffix_for_k(K: int) -> str:
    """Empty suffix = K=4 production; '_k3' = K=3 sensitivity."""
    return '' if K == 4 else f'_k{K}'


def load_per_session_paths(K: int) -> dict[str, dict]:
    """Load per-session gamma + path + segments for the K_winner fit.

    Returns {sid: {'gamma', 'path', 't_common', 'segments', 'state_labels'}}.
    """
    suffix = hier_suffix_for_k(K)
    summary_path = MVP_ROOT / f'hierarchical{suffix}' / 'mvp_hierarchical_results.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'No hierarchical fit at K={K}: {summary_path}')
    summary = json.loads(summary_path.read_text())
    state_labels = summary['state_labels']

    out = {}
    for sess in summary['sessions']:
        sid = sess['session']
        npz_path = MVP_ROOT / sid / f'mvp_rslds_results{suffix}.npz'
        if not npz_path.exists():
            continue
        npz = dict(np.load(npz_path, allow_pickle=True))
        # Segments come from the spec's per-session data — load from V11 sidecar
        v11_sidecar = REPO_ROOT / 'results' / 'v11' / sid / 'scaffold_v11_results.json'
        segments = []
        if v11_sidecar.exists():
            info = json.loads(v11_sidecar.read_text())
            segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]
        out[sid] = {
            'gamma': npz['gamma'],
            'path': npz['path'],
            't_common': npz['t_common'],
            'segments': segments,
            'state_labels': state_labels,
        }
    return out


def load_session_protocols(sids: list[str]) -> dict[str, str]:
    """Read protocol field from each session's digest JSON."""
    out = {}
    for sid in sids:
        path = DIGEST_ROOT / f'{sid}.json'
        if path.exists():
            j = json.loads(path.read_text())
            out[sid] = j.get('protocol', 'unknown')
        else:
            out[sid] = 'unknown'
    return out


def coup_shared_score(sess: dict) -> float:
    """Mean (COUP+SHARED) gamma during conv_1+conv_2 for this session.

    Returns NaN if neither conv condition is present.
    """
    labels = sess['state_labels']
    if 'COUP' not in labels and 'SHARED' not in labels:
        return float('nan')
    coup_idx = labels.index('COUP') if 'COUP' in labels else None
    shared_idx = labels.index('SHARED') if 'SHARED' in labels else None
    gamma = sess['gamma']
    t = sess['t_common']
    masks = []
    for cname, t0, t1 in sess['segments']:
        if cname in CONV_CONDS:
            masks.append((t >= t0) & (t <= t1))
    if not masks:
        return float('nan')
    mask = np.any(masks, axis=0)
    if not mask.any():
        return float('nan')
    score = 0.0
    if coup_idx is not None:
        score += float(gamma[mask, coup_idx].mean())
    if shared_idx is not None:
        score += float(gamma[mask, shared_idx].mean())
    return score


def per_condition_state_usage(sess: dict, conditions: tuple[str, ...]
                                ) -> dict[str, dict[str, float]]:
    """Return {cond: {state_label: usage_frac}} for one session."""
    labels = sess['state_labels']
    K = len(labels)
    path = sess['path']
    t = sess['t_common']
    out = {}
    for cname in conditions:
        cum_mask = np.zeros_like(t, dtype=bool)
        for sname, t0, t1 in sess['segments']:
            if sname == cname:
                cum_mask |= (t >= t0) & (t <= t1)
        if cum_mask.sum() < 3:
            continue
        usage = {labels[k]: float((path[cum_mask] == k).mean()) for k in range(K)}
        out[cname] = usage
    return out


def cohort_mean_per_condition(sessions: dict[str, dict],
                                conditions: tuple[str, ...]
                                ) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {cond: {state_label: (mean_usage, sem_usage)}}."""
    pooled = {c: {} for c in conditions}  # cond -> state -> list of session usages
    for sess in sessions.values():
        psess = per_condition_state_usage(sess, conditions)
        for c, usage in psess.items():
            for state_label, frac in usage.items():
                pooled[c].setdefault(state_label, []).append(frac)
    out = {}
    for c, by_state in pooled.items():
        out[c] = {}
        for state_label, fracs in by_state.items():
            arr = np.array(fracs)
            out[c][state_label] = (float(arr.mean()),
                                   float(arr.std() / np.sqrt(max(arr.size, 1))))
    return out


# ── Plotting ────────────────────────────────────────────────────────

def plot_state_strip(ax, sess: dict, title: str = ''):
    """Headline color strip + dashed phase boundaries."""
    labels = sess['state_labels']
    path = sess['path']
    t = sess['t_common']
    if t.size == 0:
        return
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    # Build per-state color array
    cmap = np.array([STATE_COLORS.get(labels[k], '#000000') for k in range(len(labels))])
    color_strip = np.array([cmap[int(p)] for p in path])
    # Plot as imshow strip
    ax.imshow(color_strip[None, :], aspect='auto',
              extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    # Phase boundaries
    for cname, t0, t1 in sess['segments']:
        x0 = (t0 - t[0]) / max(t[-1] - t[0], 1e-9)
        x1 = (t1 - t[0]) / max(t[-1] - t[0], 1e-9)
        if 0 <= x0 <= 1:
            ax.axvline(x0, color='black', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.text((x0 + x1) / 2, 1.08, cname, ha='center', va='bottom',
                fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_title(title, fontsize=10, fontweight='bold')


def plot_condition_aggregate_bars(ax, agg: dict, conditions: tuple[str, ...],
                                    state_order: list[str], title: str = ''):
    """Stacked bars: x=condition, height=1.0, segments=state usage."""
    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    for state_label in state_order:
        means = np.array([agg.get(c, {}).get(state_label, (0, 0))[0] for c in conditions])
        sems = np.array([agg.get(c, {}).get(state_label, (0, 0))[1] for c in conditions])
        color = STATE_COLORS.get(state_label, '#888888')
        ax.bar(x, means, bottom=bottom, color=color,
               label=state_label, edgecolor='white', linewidth=0.5)
        # Error bars at the segment top
        ax.errorbar(x, bottom + means, yerr=sems, fmt='none',
                    ecolor='black', capsize=2, alpha=0.5)
        bottom += means
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('State usage', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def state_order_from_labels(state_labels: list[str]) -> list[str]:
    """Stack order: NULL bottom, then COUP, SHARED, OTHER (autonomic-quiescence) top."""
    canonical = ['NULL', 'COUP', 'SHARED', 'OTHER']
    return [l for l in canonical if l in state_labels]


# ── Figure builders ────────────────────────────────────────────────

def make_figure1(K: int) -> Path:
    """Figure 1 — dyad variability across cohort."""
    sessions = load_per_session_paths(K)
    if not sessions:
        raise RuntimeError('No sessions available for Figure 1')

    # Score each session
    scores = {sid: coup_shared_score(s) for sid, s in sessions.items()}
    valid = {sid: v for sid, v in scores.items() if not np.isnan(v)}
    if len(valid) < 2:
        raise RuntimeError('Need >= 2 sessions with conv data for Figure 1')

    sorted_sids = sorted(valid, key=lambda s: -valid[s])
    high_sid = sorted_sids[0]
    low_sid = sorted_sids[-1]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 2.5], hspace=0.65)

    ax_a = fig.add_subplot(gs[0])
    plot_state_strip(ax_a, sessions[high_sid],
                     title=f'A. Highest (COUP+SHARED)-during-conv: {high_sid} '
                            f'(score={valid[high_sid]:.2f})')

    ax_b = fig.add_subplot(gs[1])
    plot_state_strip(ax_b, sessions[low_sid],
                     title=f'B. Lowest (COUP+SHARED)-during-conv: {low_sid} '
                            f'(score={valid[low_sid]:.2f})')

    # Cohort aggregate over common conditions
    ax_c = fig.add_subplot(gs[2])
    agg = cohort_mean_per_condition(sessions, COMMON_CONDS)
    state_order = state_order_from_labels(sessions[high_sid]['state_labels'])
    plot_condition_aggregate_bars(
        ax_c, agg, COMMON_CONDS, state_order,
        title=f'C. Cohort condition aggregate (n={len(sessions)}, K={K}, '
              f'common conditions only)')

    fig.suptitle('Figure 1 — Dyad-level variability (cohort)',
                 fontsize=12, fontweight='bold')

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_png = FIG_ROOT / 'figure1_dyad_variability.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_ROOT / 'figure1_dyad_variability.pdf', bbox_inches='tight')
    plt.close(fig)
    return out_png


def make_figure2(K: int) -> Path:
    """Figure 2 — protocol comparison (meditation vs PE)."""
    sessions = load_per_session_paths(K)
    if not sessions:
        raise RuntimeError('No sessions available for Figure 2')

    protos = load_session_protocols(list(sessions.keys()))
    med_sids = [s for s in sessions if protos.get(s, '').lower() == 'meditation']
    pe_sids = [s for s in sessions if protos.get(s, '').lower() == 'pe']

    if not med_sids or not pe_sids:
        raise RuntimeError(f'Need >= 1 session per protocol; '
                           f'have meditation={len(med_sids)}, pe={len(pe_sids)}')

    # Pick highest within each protocol
    med_top = max(med_sids, key=lambda s: coup_shared_score(sessions[s]))
    pe_top = max(pe_sids, key=lambda s: coup_shared_score(sessions[s]))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 2.5], hspace=0.65,
                           width_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, :])
    plot_state_strip(ax_a, sessions[med_top],
                     title=f'A. Top meditation dyad: {med_top} '
                            f'(score={coup_shared_score(sessions[med_top]):.2f})')

    ax_b = fig.add_subplot(gs[1, :])
    plot_state_strip(ax_b, sessions[pe_top],
                     title=f'B. Top PE dyad: {pe_top} '
                            f'(score={coup_shared_score(sessions[pe_top]):.2f})')

    # Per-protocol condition-aggregate bars (meditation left, PE right)
    med_sessions = {s: sessions[s] for s in med_sids}
    pe_sessions = {s: sessions[s] for s in pe_sids}
    med_agg = cohort_mean_per_condition(med_sessions, MED_CONDS)
    pe_agg = cohort_mean_per_condition(pe_sessions, PE_CONDS)
    state_order = state_order_from_labels(sessions[med_top]['state_labels'])

    ax_c1 = fig.add_subplot(gs[2, 0])
    plot_condition_aggregate_bars(
        ax_c1, med_agg, MED_CONDS, state_order,
        title=f'C-Med. Meditation protocol (n={len(med_sids)})')

    ax_c2 = fig.add_subplot(gs[2, 1])
    plot_condition_aggregate_bars(
        ax_c2, pe_agg, PE_CONDS, state_order,
        title=f'C-PE. Psychoeducation protocol (n={len(pe_sids)})')

    fig.suptitle('Figure 2 — Protocol comparison (meditation vs PE)',
                 fontsize=12, fontweight='bold')

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_png = FIG_ROOT / 'figure2_protocol_comparison.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_ROOT / 'figure2_protocol_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    return out_png


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=None,
                    help='Override K_winner detection (default: read verification_report.md)')
    ap.add_argument('--skip-fig1', action='store_true')
    ap.add_argument('--skip-fig2', action='store_true')
    args = ap.parse_args()

    K = args.K if args.K is not None else detect_k_winner()
    print(f'Using K = {K} (from verification_report.md)' if args.K is None
          else f'Using K = {K} (CLI override)')

    if not args.skip_fig1:
        print('\nBuilding Figure 1 (dyad variability)...', flush=True)
        try:
            p = make_figure1(K)
            print(f'  Saved: {p}')
        except Exception as e:
            print(f'  ERROR: Figure 1 failed: {e}')

    if not args.skip_fig2:
        print('\nBuilding Figure 2 (protocol comparison)...', flush=True)
        try:
            p = make_figure2(K)
            print(f'  Saved: {p}')
        except Exception as e:
            print(f'  ERROR: Figure 2 failed: {e}')

    print(f'\nFigures directory: {FIG_ROOT}/')


if __name__ == '__main__':
    main()
