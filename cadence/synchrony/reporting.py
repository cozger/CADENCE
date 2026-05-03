"""Stage 9 — per-dyad reporting + rSLDS cross-tab + REPORT.md."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cadence.synchrony.io import (
    cohort_dir, REPO_ROOT, load_digest, digest_path,
)
from cadence.synchrony.viz import CONDITION_COLORS


def _parse_periods(markers):
    starts, out = {}, []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    return sorted(out, key=lambda x: x[1])


# ── 9a Per-dyad repertoire signature ────────────────────────────────────

def per_dyad_repertoire(save: bool = True) -> pd.DataFrame:
    cdir = cohort_dir()
    coh = dict(np.load(cdir / 'cohort_features.npz', allow_pickle=True))
    cls = dict(np.load(cdir / 'cohort_clusters.npz', allow_pickle=True))

    sids = np.asarray(coh['session_id'])
    labels = np.asarray(cls['labels'])

    cluster_ids = sorted(set(int(c) for c in labels) - {-2, -1})
    unique_sids = sorted(set(sids))

    rows = []
    for s in unique_sids:
        m = sids == s
        n_total = int(m.sum())
        # noise (-1) and excluded (-2) tracked separately
        cluster_counts = {f'c{c}': int(((labels == c) & m).sum())
                            for c in cluster_ids}
        n_noise = int(((labels == -1) & m).sum())
        n_excluded = int(((labels == -2) & m).sum())
        denom = max(n_total - n_excluded, 1)
        cluster_fracs = {f'frac_c{c}': cluster_counts[f'c{c}'] / denom
                          for c in cluster_ids}
        rows.append({
            'session_id':       s,
            'n_episodes_total': n_total,
            'n_excluded_low_quality': n_excluded,
            'n_noise':          n_noise,
            **cluster_counts,
            **cluster_fracs,
        })
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(cdir / 'per_dyad_repertoire.csv', index=False)
        _plot_per_dyad_repertoire(df, cluster_ids,
                                    cdir / 'fig_per_dyad_repertoire.png')
    return df


def _plot_per_dyad_repertoire(df: pd.DataFrame, cluster_ids: list[int],
                                save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    sids = df['session_id'].values
    n_dyads = len(sids)
    bottoms = np.zeros(n_dyads)
    cmap = plt.get_cmap('tab20', max(20, len(cluster_ids)))
    for i, c in enumerate(cluster_ids):
        col = f'frac_c{c}'
        if col not in df.columns:
            continue
        vals = df[col].values
        ax.bar(range(n_dyads), vals, bottom=bottoms, color=cmap(i % 20),
                label=f'c{c}', edgecolor='white', linewidth=0.4)
        bottoms += vals
    # Noise + excluded on top in gray
    noise_frac = df['n_noise'].values / np.maximum(
        df['n_episodes_total'].values - df['n_excluded_low_quality'].values, 1)
    ax.bar(range(n_dyads), noise_frac, bottom=bottoms, color='#bdbdbd',
            label='noise', edgecolor='white', linewidth=0.4)
    ax.set_xticks(range(n_dyads))
    ax.set_xticklabels(sids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('fraction of cluster-eligible episodes', fontsize=10)
    ax.set_title('Per-dyad repertoire signature  ·  cluster-occupancy distribution',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── 9b Per-dyad expressivity profile ────────────────────────────────────

def per_dyad_expressivity(save: bool = True) -> pd.DataFrame:
    """Build the expressivity CSV from per-session episode caches."""
    from cadence.ingest.quality import list_canonical_sessions
    from cadence.synchrony.io import read_stage
    from cadence.constants import AU_REGIONS_7

    rows = []
    for sid in list_canonical_sessions():
        try:
            ev_npz, ev_meta = read_stage(sid, 1)
            ep_npz, ep_meta = read_stage(sid, 2)
        except Exception:
            continue
        n_eps = ep_meta.get('n_episodes', 0)
        digest = load_digest(sid) if digest_path(sid).exists() else {}
        protocol = digest.get('protocol', '')

        per_cond = {}
        if n_eps > 0:
            cond = np.asarray(ep_npz['condition'])
            durs = np.asarray(ep_npz['duration_s'])
            peak_env = np.asarray(ep_npz['peak_env'])
            n_t = np.asarray(ep_npz['n_events_therapist'])
            n_p = np.asarray(ep_npz['n_events_patient'])

            for c in sorted(set(cond) - {''}):
                m = cond == c
                # Use t_start_lsl spans to estimate condition duration
                periods = _parse_periods(digest.get('markers', []))
                cond_dur = next((t1 - t0 for n, t0, t1 in periods if n == c), 0)
                per_cond[c] = (m.sum() / (cond_dur / 60.0)
                                if cond_dur > 0 else 0)
        # Cohort-level metrics
        au_diversity = 0
        if len(np.asarray(ev_npz['au_idx'])) > 0:
            au_diversity = int(len(set(np.asarray(ev_npz['au_idx']).tolist())))

        # Dyadic intensity balance — proxy: ratio of therapist:patient event amps
        if 'amp' in ev_npz and 'role' in ev_npz and len(ev_npz['amp']) > 0:
            roles = np.asarray(ev_npz['role'])
            amps = np.asarray(ev_npz['amp'])
            ta = amps[roles == 'therapist']
            pa = amps[roles == 'patient']
            if len(ta) and len(pa):
                balance = float(ta.mean() / max(pa.mean(), 1e-6))
            else:
                balance = float('nan')
        else:
            balance = float('nan')

        rows.append({
            'session_id':              sid,
            'protocol':                protocol,
            'n_episodes_total':        n_eps,
            'episode_rate_overall':    ep_meta.get('episode_rate_per_min', 0.0),
            'total_active_fraction':   ep_meta.get('fraction_active', 0.0),
            'mean_episode_intensity':  float(np.asarray(ep_npz['peak_env']).mean())
                                          if n_eps > 0 else 0.0,
            'mean_episode_duration':   float(np.asarray(ep_npz['duration_s']).mean())
                                          if n_eps > 0 else 0.0,
            'au_diversity':            au_diversity,
            'dyadic_intensity_balance': balance,
            **{f'episode_rate_{c}': v for c, v in per_cond.items()},
        })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(cohort_dir() / 'per_dyad_expressivity.csv', index=False)
        _plot_per_dyad_expressivity(df, cohort_dir() /
                                          'fig_per_dyad_expressivity.png')
    return df


def _plot_per_dyad_expressivity(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    if df.empty:
        fig.savefig(save_path); plt.close(fig); return

    proto_color = {'meditation': '#7B1FA2', 'PE': '#C62828'}
    for proto in df['protocol'].unique():
        m = df['protocol'] == proto
        sub = df[m]
        ax.scatter(sub['mean_episode_intensity'], sub['episode_rate_overall'],
                    s=120 * sub['total_active_fraction'] + 30,
                    c=proto_color.get(proto, '#9E9E9E'), alpha=0.7,
                    edgecolors='black', linewidths=0.5, label=proto or '?')
        for _, r in sub.iterrows():
            ax.annotate(r['session_id'],
                          (r['mean_episode_intensity'], r['episode_rate_overall']),
                          fontsize=7, ha='center', va='bottom',
                          xytext=(0, 4), textcoords='offset points')
    ax.set_xlabel('mean episode intensity (peak_env)', fontsize=10)
    ax.set_ylabel('episode rate (per minute)', fontsize=10)
    ax.set_title('Per-dyad expressivity profile  ·  '
                  'marker size = total active fraction',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── 9c Repertoire × rSLDS-state cross-tab ───────────────────────────────

def repertoire_x_rslds_state(save: bool = True) -> dict:
    """For each session w/ both MVP rSLDS results AND synchrony repertoire,
    compute P(cluster | rSLDS_state). Aggregate to a single heatmap."""
    cdir = cohort_dir()
    coh = dict(np.load(cdir / 'cohort_features.npz', allow_pickle=True))
    cls = dict(np.load(cdir / 'cohort_clusters.npz', allow_pickle=True))

    sids = np.asarray(coh['session_id'])
    starts = np.asarray(coh['t_start_lsl'])
    ends = np.asarray(coh.get('t_end_lsl', starts + 1))
    labels = np.asarray(cls['labels'])

    cluster_ids = sorted(set(int(c) for c in labels) - {-2, -1})

    # Per-session, load mvp_rslds_results.npz if present
    mvp_root = REPO_ROOT / 'results' / 'mvp'
    state_names = ['NULL', 'OTHER', 'COUP', 'SHARED']  # Per CADENCE convention
    contingency = np.zeros((len(state_names), len(cluster_ids)), dtype=np.int64)
    n_sessions_with_rslds = 0
    n_episodes_assigned = 0
    skipped_reasons = {}

    for s in sorted(set(sids)):
        rslds_path = mvp_root / s / 'mvp_rslds_results.npz'
        if not rslds_path.exists():
            skipped_reasons.setdefault('no mvp_rslds_results.npz', []).append(s)
            continue
        try:
            mvp = np.load(rslds_path, allow_pickle=True)
        except Exception as e:
            skipped_reasons.setdefault(f'load_error: {type(e).__name__}', []).append(s)
            continue
        if 'state' not in mvp.files or 't_common' not in mvp.files:
            skipped_reasons.setdefault('missing state/t_common', []).append(s)
            continue
        state = np.asarray(mvp['state'], dtype=np.int32)
        t_common = np.asarray(mvp['t_common'], dtype=np.float64)
        n_sessions_with_rslds += 1

        # For each episode in this session, get majority rSLDS state during
        # episode → contingency[state, cluster] += 1
        m = sids == s
        for i in np.where(m)[0]:
            cid = int(labels[i])
            if cid < 0:  # skip noise + excluded
                continue
            t0 = float(starts[i]); t1 = float(ends[i])
            mask = (t_common >= t0) & (t_common <= t1)
            if not mask.any():
                continue
            # Majority state
            states_in_ep = state[mask]
            uniq, counts = np.unique(states_in_ep, return_counts=True)
            mode_state = int(uniq[np.argmax(counts)])
            if 0 <= mode_state < len(state_names):
                col = cluster_ids.index(cid)
                contingency[mode_state, col] += 1
                n_episodes_assigned += 1

    out = {
        'state_names':              state_names,
        'cluster_ids':              cluster_ids,
        'contingency':              contingency.tolist(),
        'n_sessions_with_rslds':    n_sessions_with_rslds,
        'n_episodes_assigned':      n_episodes_assigned,
        'skipped_reasons':          {k: v for k, v in skipped_reasons.items()},
    }

    if save and n_episodes_assigned > 0:
        # Normalize per row (P(cluster | state))
        row_sums = contingency.sum(axis=1, keepdims=True)
        prob = np.where(row_sums > 0, contingency / np.maximum(row_sums, 1), 0)

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cluster_ids) + 4), 4))
        im = ax.imshow(prob, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f'c{c}' for c in cluster_ids], rotation=0)
        ax.set_yticks(range(len(state_names)))
        ax.set_yticklabels(state_names)
        ax.set_xlabel('Synchrony cluster', fontsize=10)
        ax.set_ylabel('rSLDS state (MVP)', fontsize=10)
        ax.set_title(f'P(cluster | rSLDS state)  ·  '
                      f'{n_sessions_with_rslds} sessions, '
                      f'{n_episodes_assigned} episodes',
                      fontsize=11, fontweight='bold')
        for i in range(len(state_names)):
            for j in range(len(cluster_ids)):
                ax.text(j, i, f'{prob[i, j]:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if prob[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='probability')
        fig.tight_layout()
        fig.savefig(cdir / 'fig_repertoire_x_rslds_state.png', dpi=120,
                     bbox_inches='tight')
        plt.close(fig)
    elif save:
        # Write a stub showing why no figure was produced
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f'rSLDS cross-tab unavailable\n'
                            f'(n_sessions_with_rslds={n_sessions_with_rslds}, '
                            f'n_episodes_assigned={n_episodes_assigned})',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.axis('off')
        fig.savefig(cdir / 'fig_repertoire_x_rslds_state.png', dpi=120,
                     bbox_inches='tight')
        plt.close(fig)
    return out


# ── REPORT.md ──────────────────────────────────────────────────────────

def write_report_markdown() -> Path:
    cdir = cohort_dir()
    cls_meta = json.loads((cdir / 'cohort_clusters.json').read_text())
    feat_meta = json.loads((cdir / 'cohort_features.json').read_text())
    val_meta = (json.loads((cdir / 'cohort_validation.json').read_text())
                 if (cdir / 'cohort_validation.json').exists() else {})

    cluster_ids = sorted(int(c) for c in cls_meta.get('cluster_sizes', {}))
    sizes = cls_meta.get('cluster_sizes', {})

    md = []
    md.append('# Synchrony Repertoire — Cohort Report\n')
    md.append(f'_generated {time.strftime("%Y-%m-%d %H:%M:%S")}_\n')
    md.append('## Cohort overview\n')
    md.append(f'- Sessions analyzed: **{feat_meta["n_sessions"]}**')
    md.append(f'- Episodes (cluster-eligible): **{cls_meta["n_episodes_clustered"]}** '
                 f'(of {cls_meta["n_episodes_total"]} total; '
                 f'{cls_meta["n_low_quality_excluded"]} low-quality excluded)')
    md.append(f'- Clusters discovered: **{cls_meta["n_clusters"]}** + '
                 f'{cls_meta["n_noise"]} noise-labeled episodes')
    md.append(f'- Mean imputation rate: **{feat_meta["mean_imputation_rate"]:.2%}**\n')

    md.append('## Embedding overview\n')
    md.append('![embedding](fig_cohort_embedding.png)\n')

    md.append('## Per-cluster catalog\n')
    if (cdir / 'cluster_summaries.json').exists():
        cs = json.loads((cdir / 'cluster_summaries.json').read_text())
        for s in cs.get('clusters', []):
            md.append(f'### Cluster {s["cluster_id"]} — {s["tentative_label"]}')
            md.append(f'- n_episodes: **{s["n_episodes"]}** from '
                         f'{s["n_dyads_represented"]} dyads')
            md.append(f'- top conditions: '
                         f'{", ".join(f"{k}={v}" for k, v in list(s["condition_distribution"].items())[:3])}')
            md.append(f'\n![cluster_{s["cluster_id"]}](fig_cluster_{s["cluster_id"]}_profile.png)\n')

    md.append('## Per-dyad repertoire signature\n')
    md.append('![per-dyad repertoire](fig_per_dyad_repertoire.png)\n')

    md.append('## Per-dyad expressivity profile\n')
    md.append('![per-dyad expressivity](fig_per_dyad_expressivity.png)\n')

    md.append('## Repertoire × rSLDS state cross-tab\n')
    md.append('![rSLDS cross-tab](fig_repertoire_x_rslds_state.png)\n')

    md.append('## Validation suite\n')
    if val_meta:
        boot = val_meta.get('bootstrap_stability', {})
        loo  = val_meta.get('leave_one_dyad_out', {})
        thr  = val_meta.get('threshold_sensitivity', {})
        md.append(f'- Bootstrap stability: mean ARI = '
                     f'**{boot.get("mean_ari", float("nan")):.3f}** '
                     f'({boot.get("n_iter", 0)} iter)')
        md.append(f'- Leave-one-dyad-out: mean ARI = '
                     f'**{loo.get("mean_ari", float("nan")):.3f}** '
                     f'({loo.get("n_folds", 0)} folds)')
        md.append(f'- Threshold sensitivity (±20%): count ratio '
                     f'∈ [{thr.get("count_ratio_min", float("nan")):.2f}, '
                     f'{thr.get("count_ratio_max", float("nan")):.2f}]')
    else:
        md.append('_validation suite not yet run_')

    out = cdir / 'REPORT.md'
    out.write_text('\n'.join(md))
    return out
