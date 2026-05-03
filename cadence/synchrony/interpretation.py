"""Stage 7 — per-cluster interpretation.

For each cluster, compute identity / dynamics / synchrony profiles, find
top-3 medoid exemplars, render the per-cluster figure (paired-face
schematic + region radar + medoid exemplars + synchrony profile), and
generate a tentative auto-label.

Outputs:
  cohort/cluster_summaries.json  — per-cluster profile + tentative_label
  cohort/fig_cluster_<k>_profile.png  — one per cluster
  cohort/fig_all_clusters_grid.png    — grid of all-cluster mean faces
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cadence.constants import AU_REGIONS_7
from cadence.synchrony.io import (
    cohort_dir, load_face_npz, load_digest, FACE_PREPROC_ROOT,
)
from cadence.synchrony.events import _resolve_role_to_p
from cadence.significance.face_event_coincidence import (
    _smooth_savgol, _baseline_subtract,
)
from cadence.synchrony.viz import (
    draw_paired_face_schematic, AU_NAMES, CONDITION_COLORS,
)


REGION_NAMES = list(AU_REGIONS_7.keys())


# ── Per-episode AU mean (used for cluster identity profiles) ────────────

from functools import lru_cache


@lru_cache(maxsize=32)
def _load_session_face_data(sid: str):
    """Cached per-session face data: (au_p1, ts_p1, au_p2, ts_p2, role_to_p)."""
    face = load_face_npz(sid)
    digest = load_digest(sid)
    role_to_p = _resolve_role_to_p(face, digest)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    out = {'role_to_p': role_to_p}
    for p_key in ('p1', 'p2'):
        if f'{p_key}_au52' in face:
            au = np.asarray(face[f'{p_key}_au52'], dtype=np.float32)
            ts = np.asarray(face[f'{p_key}_au52_ts'], dtype=np.float64) + lsl_offset
        else:
            au, ts = None, None
        out[f'{p_key}_au'] = au
        out[f'{p_key}_ts'] = ts
    return out


def _episode_au_mean(sid: str, ep_t_start: float, ep_t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (au_mean_therapist, au_mean_patient) — (52,) each — for one episode."""
    cached = _load_session_face_data(sid)
    role_to_p = cached['role_to_p']
    out = {'therapist': None, 'patient': None}
    for role in ('therapist', 'patient'):
        p_key = role_to_p[role]
        au = cached.get(f'{p_key}_au')
        ts = cached.get(f'{p_key}_ts')
        if au is None:
            out[role] = np.zeros(52, dtype=np.float32)
            continue
        i0 = int(np.searchsorted(ts, ep_t_start, side='left'))
        i1 = int(np.searchsorted(ts, ep_t_end, side='right'))
        if i1 - i0 < 1:
            out[role] = np.zeros(52, dtype=np.float32)
        else:
            out[role] = au[i0:i1].mean(axis=0).astype(np.float32)
    return out['therapist'], out['patient']


# ── Cluster summarization ──────────────────────────────────────────────

def _summarize_cluster(cluster_id: int, member_idx: np.ndarray,
                         coh: dict, cls: dict) -> dict:
    """Build the per-cluster profile dict."""
    feature_names_obj = coh.get('feature_names', None)
    if feature_names_obj is not None:
        # NPZ stores object arrays as 0-d; pull element
        try:
            feature_names = list(feature_names_obj.tolist())
        except AttributeError:
            feature_names = list(feature_names_obj)
    else:
        from cadence.synchrony.features.assembly import all_feature_names
        feature_names = all_feature_names()
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    X_raw = np.asarray(coh['features_raw'])
    X_z = np.asarray(coh['features'])

    sids = np.asarray(coh['session_id'])[member_idx]
    eids = np.asarray(coh['episode_id'])[member_idx]
    starts = np.asarray(coh['t_start_lsl'])[member_idx]
    ends_present = 't_end_lsl' in coh
    if ends_present:
        ends = np.asarray(coh['t_end_lsl'])[member_idx]
    else:
        ends = starts + 1.0  # fallback (shouldn't happen)
    conds = np.asarray(coh['condition'])[member_idx]
    durs = np.asarray(coh['duration_s'])[member_idx]
    peak_env = np.asarray(coh['peak_env'])[member_idx]

    # Identity profile — average per-AU activations across member episodes.
    # We pull AU vectors fresh from face npz (rather than re-read region-mean
    # features) so the paired-face schematic uses the underlying 52-AU values.
    au_mean_t = np.zeros((len(member_idx), 52), dtype=np.float32)
    au_mean_p = np.zeros((len(member_idx), 52), dtype=np.float32)
    for i, mi in enumerate(member_idx):
        au_mean_t[i], au_mean_p[i] = _episode_au_mean(
            str(sids[i]), float(starts[i]), float(ends[i]))
    therapist_face = au_mean_t.mean(axis=0)
    patient_face   = au_mean_p.mean(axis=0)

    # Dynamics profile
    def _stat_block(vec):
        v = np.asarray(vec, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {'mean': None, 'std': None, 'p25': None, 'p75': None}
        return {'mean': float(v.mean()), 'std': float(v.std(ddof=0)),
                'p25': float(np.percentile(v, 25)),
                'p75': float(np.percentile(v, 75))}

    def _col(name):
        i = name_to_col.get(name, -1)
        return X_raw[member_idx, i] if i >= 0 else np.full(len(member_idx), np.nan)

    dom_alpha_t = _col('dyn_therapist_dominant_alpha')
    dom_alpha_p = _col('dyn_patient_dominant_alpha')
    skew_t = _col('dyn_therapist_envelope_skew')
    skew_p = _col('dyn_patient_envelope_skew')

    dom_lag = _col('dominant_lag')
    dtw_full = _col('dtw_distance_norm__full52')
    dtw_lag = _col('dtw_mean_lag_s__full52')
    cca_r = _col('cca_peak_r')
    cca_lag = _col('cca_peak_lag_s')
    coh_mean = _col('coh_mean_band')

    # Lead/follow majority
    lead = _col('lead_follow')
    finite_lead = lead[np.isfinite(lead)]
    lead_count = {-1: int((finite_lead == -1).sum()),
                    0: int((finite_lead == 0).sum()),
                   +1: int((finite_lead == +1).sum())}

    # Tentative auto-label
    label = _auto_label(dom_lag, dtw_full, lead_count, peak_env, durs,
                         np.array([len(member_idx)] * 1))

    # Top-3 medoids by distance to centroid in the cluster's UMAP-10D rows
    embed_10d = np.asarray(cls['embedding_10d'])[member_idx]
    centroid_10d = embed_10d.mean(axis=0)
    distances = np.linalg.norm(embed_10d - centroid_10d, axis=1)
    medoid_local_idx = np.argsort(distances)[:3]
    medoids = []
    for li in medoid_local_idx:
        medoids.append({
            'session_id':   str(sids[li]),
            'episode_id':   int(eids[li]),
            't_start_lsl':  float(starts[li]),
            't_end_lsl':    float(ends[li]),
            'condition':    str(conds[li]),
            'duration_s':   float(durs[li]),
            'distance_centroid_10d': float(distances[li]),
        })

    # Per-condition distribution (which conditions contribute episodes here?)
    cond_dist = {}
    for c in conds:
        cond_dist[str(c)] = cond_dist.get(str(c), 0) + 1
    cond_dist = {k: v for k, v in sorted(cond_dist.items(), key=lambda kv: -kv[1])}

    # Per-dyad fraction (what fraction of the cluster is each dyad)
    sid_dist = {}
    for s in sids:
        sid_dist[str(s)] = sid_dist.get(str(s), 0) + 1

    return {
        'cluster_id':            int(cluster_id),
        'n_episodes':            int(len(member_idx)),
        'n_dyads_represented':   len(sid_dist),
        'tentative_label':       label,
        'identity_profile': {
            'therapist_face_mean_per_au': therapist_face.tolist(),
            'patient_face_mean_per_au':   patient_face.tolist(),
        },
        'dynamics_profile': {
            'duration_s':      _stat_block(durs),
            'peak_env':        _stat_block(peak_env),
            'dominant_alpha_therapist': _stat_block(dom_alpha_t),
            'dominant_alpha_patient':   _stat_block(dom_alpha_p),
            'envelope_skew_therapist':  _stat_block(skew_t),
            'envelope_skew_patient':    _stat_block(skew_p),
        },
        'synchrony_profile': {
            'dominant_lag_s':       _stat_block(dom_lag),
            'lead_follow_counts':   lead_count,
            'dtw_distance_norm':    _stat_block(dtw_full),
            'dtw_mean_lag_s':       _stat_block(dtw_lag),
            'cca_peak_r':           _stat_block(cca_r),
            'cca_peak_lag_s':       _stat_block(cca_lag),
            'coh_mean_band':        _stat_block(coh_mean),
        },
        'medoids':           medoids,
        'condition_distribution': cond_dist,
        'session_distribution':   sid_dist,
    }


def _auto_label(dom_lag, dtw_full, lead_count, peak_env, durs, sizes) -> str:
    """Heuristic tentative label — hand-editable in JSON afterwards."""
    finite_lag = dom_lag[np.isfinite(dom_lag)]
    finite_dtw = dtw_full[np.isfinite(dtw_full)]
    if finite_lag.size:
        lag_mean = float(np.mean(finite_lag))
    else:
        lag_mean = 0.0
    dtw_mean = float(np.mean(finite_dtw)) if finite_dtw.size else float('nan')

    # Modal lead/follow direction
    lf_mode = max(lead_count, key=lead_count.get) if lead_count else 0

    if abs(lag_mean) < 0.2 and (np.isnan(dtw_mean) or dtw_mean < 1.5):
        return 'mirrored mutual'
    if lf_mode == +1 and lead_count.get(+1, 0) > 0.5 * sum(lead_count.values()):
        return 'patient-led'
    if lf_mode == -1 and lead_count.get(-1, 0) > 0.5 * sum(lead_count.values()):
        return 'therapist-led'
    if peak_env.size and peak_env.mean() > 3.0:
        return 'high-intensity event'
    if durs.size and durs.mean() > 5.0:
        return 'long episode'
    if durs.size and durs.mean() < 0.7:
        return 'brief micro-event'
    return 'unlabeled'


# ── Per-cluster figure ─────────────────────────────────────────────────

def _draw_one_cluster_figure(summary: dict, save_path: Path):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.6, 1.5, 1.0],
                           width_ratios=[1.4, 1.4, 1.4, 1.4],
                           hspace=0.38, wspace=0.30,
                           left=0.05, right=0.99, top=0.92, bottom=0.06)

    # Header
    fig.suptitle(f'Cluster {summary["cluster_id"]}  ·  '
                  f'{summary["tentative_label"].upper()}  ·  '
                  f'n={summary["n_episodes"]} episodes from '
                  f'{summary["n_dyads_represented"]} dyads',
                  fontsize=13, fontweight='bold')

    # (A,B) Paired-face schematic
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    therapist_face = np.asarray(summary['identity_profile']
                                  ['therapist_face_mean_per_au'])
    patient_face   = np.asarray(summary['identity_profile']
                                  ['patient_face_mean_per_au'])
    vmax = max(0.05, float(max(therapist_face.max(), patient_face.max())))
    draw_paired_face_schematic(therapist_face, patient_face,
                                 ax_t=ax_t, ax_p=ax_p, vmin=0.0, vmax=vmax,
                                 cmap='Reds')

    # (C) Region radar: 7 regions, two roles overlaid
    ax_r = fig.add_subplot(gs[0, 2], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(REGION_NAMES) + 1)[:-1]
    angles_loop = np.concatenate([angles, [angles[0]]])
    therapist_reg = np.array([float(np.nanmean(therapist_face[AU_REGIONS_7[r]]))
                                for r in REGION_NAMES])
    patient_reg = np.array([float(np.nanmean(patient_face[AU_REGIONS_7[r]]))
                              for r in REGION_NAMES])
    therapist_reg_loop = np.concatenate([therapist_reg, [therapist_reg[0]]])
    patient_reg_loop = np.concatenate([patient_reg, [patient_reg[0]]])
    ax_r.plot(angles_loop, therapist_reg_loop, 'o-', linewidth=2,
                color='#C62828', label='therapist')
    ax_r.fill(angles_loop, therapist_reg_loop, alpha=0.15, color='#C62828')
    ax_r.plot(angles_loop, patient_reg_loop, 'o-', linewidth=2,
                color='#1565C0', label='patient')
    ax_r.fill(angles_loop, patient_reg_loop, alpha=0.15, color='#1565C0')
    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(REGION_NAMES, fontsize=7)
    ax_r.set_title('Region-mean activation', fontsize=10, fontweight='bold',
                    pad=14)
    ax_r.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.25, 1.10))

    # (D) Synchrony profile text panel
    ax_s = fig.add_subplot(gs[0, 3])
    sp = summary['synchrony_profile']

    def _fmt(stat):
        if stat is None or stat.get('mean') is None:
            return 'n/a'
        return f"{stat['mean']:+.3f} ± {stat['std']:.3f}"
    text = (
        f"DOMINANT LAG (3a)\n"
        f"  {_fmt(sp['dominant_lag_s'])} s\n"
        f"  lead-follow: T={sp['lead_follow_counts'].get(-1,0)}, "
        f"sync={sp['lead_follow_counts'].get(0,0)}, "
        f"P={sp['lead_follow_counts'].get(+1,0)}\n\n"
        f"DTW (full 52-D)\n"
        f"  distance: {_fmt(sp['dtw_distance_norm'])}\n"
        f"  mean lag: {_fmt(sp['dtw_mean_lag_s'])} s\n\n"
        f"SPARSE CCA\n"
        f"  peak r: {_fmt(sp['cca_peak_r'])}\n"
        f"  peak lag: {_fmt(sp['cca_peak_lag_s'])} s\n\n"
        f"WAVELET COHERENCE 0.1–2 Hz\n"
        f"  mean: {_fmt(sp['coh_mean_band'])}\n"
    )
    ax_s.text(0.0, 1.0, text, transform=ax_s.transAxes, va='top',
               family='monospace', fontsize=9)
    ax_s.set_title('Synchrony profile', fontsize=10, fontweight='bold',
                     loc='left')
    ax_s.axis('off')

    # (E) Per-condition distribution bar
    ax_c = fig.add_subplot(gs[1, 0:2])
    cond_dist = summary['condition_distribution']
    cd_items = list(cond_dist.items())
    if cd_items:
        labels = [c if c else '(no marker)' for c, _ in cd_items]
        counts = [n for _, n in cd_items]
        colors_list = [CONDITION_COLORS.get(c, '#9E9E9E') for c, _ in cd_items]
        ax_c.bar(range(len(labels)), counts, color=colors_list, edgecolor='white')
        ax_c.set_xticks(range(len(labels)))
        ax_c.set_xticklabels(labels, rotation=20, fontsize=8, ha='right')
        ax_c.set_ylabel('# episodes', fontsize=9)
    ax_c.set_title('Cluster\'s episodes by condition', fontsize=10,
                     fontweight='bold', loc='left')
    ax_c.tick_params(labelsize=8)
    ax_c.grid(axis='y', alpha=0.3, linestyle=':')

    # (F) Per-session distribution bar
    ax_d = fig.add_subplot(gs[1, 2:4])
    sid_dist = summary['session_distribution']
    sd_items = sorted(sid_dist.items(), key=lambda kv: -kv[1])[:15]
    if sd_items:
        labels = [s for s, _ in sd_items]
        counts = [n for _, n in sd_items]
        ax_d.barh(range(len(labels)), counts, color='#37474F', edgecolor='white')
        ax_d.set_yticks(range(len(labels)))
        ax_d.set_yticklabels(labels, fontsize=8)
        ax_d.set_xlabel('# episodes', fontsize=9)
        ax_d.invert_yaxis()
    ax_d.set_title('Top dyads contributing to cluster', fontsize=10,
                     fontweight='bold', loc='left')
    ax_d.tick_params(labelsize=8)
    ax_d.grid(axis='x', alpha=0.3, linestyle=':')

    # (G) Top-3 medoid table
    ax_m = fig.add_subplot(gs[2, :])
    medoid_lines = ['Top-3 medoid exemplars:']
    for i, m in enumerate(summary['medoids'], 1):
        medoid_lines.append(
            f"  {i}. {m['session_id']:<22s}  "
            f"ep={m['episode_id']:<4d}  "
            f"t={m['t_start_lsl']:>9.1f}–{m['t_end_lsl']:>9.1f}s  "
            f"({m['condition']:<10s}) "
            f"dur={m['duration_s']:>5.2f}s  "
            f"d={m['distance_centroid_10d']:>5.3f}"
        )
    ax_m.text(0.0, 1.0, '\n'.join(medoid_lines), transform=ax_m.transAxes,
               va='top', family='monospace', fontsize=9)
    ax_m.axis('off')

    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def summarize_clusters(save: bool = True) -> dict:
    """Build per-cluster summaries + figures + JSON."""
    cdir = cohort_dir()
    coh = dict(np.load(cdir / 'cohort_features.npz', allow_pickle=True))
    cls = dict(np.load(cdir / 'cohort_clusters.npz', allow_pickle=True))

    labels = np.asarray(cls['labels'])
    cluster_ids = sorted(set(int(c) for c in labels) - {-2, -1})

    summaries = []
    for cid in cluster_ids:
        member_idx = np.where(labels == cid)[0]
        s = _summarize_cluster(cid, member_idx, coh, cls)
        if save:
            _draw_one_cluster_figure(s, cdir / f'fig_cluster_{cid}_profile.png')
        summaries.append(s)

    out = {
        'n_clusters': len(cluster_ids),
        'cluster_ids': cluster_ids,
        'clusters': summaries,
    }
    if save:
        (cdir / 'cluster_summaries.json').write_text(json.dumps(out, indent=2))
    return out
