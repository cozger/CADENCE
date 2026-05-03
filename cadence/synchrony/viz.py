"""Visualization utilities for the synchrony pipeline.

Sanity-check figures (per-stage) + cohort-level figures (clusters,
repertoire, expressivity, validation, rSLDS cross-tab) + the paired-face
AU heatmap utility used by Stage 7 cluster interpretation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cadence.constants import AU_REGIONS_7
from cadence.synchrony.io import (
    DIGEST_ROOT, FACE_PREPROC_ROOT, session_dir, load_face_npz, load_digest,
)


# ── Condition colors (shared with MVP visualizations for consistency) ──
CONDITION_COLORS = {
    'base_EO':    '#1565C0',
    'base_EC':    '#283593',
    'conv_1':     '#E65100',
    'conv_2':     '#BF360C',
    'meditate_B': '#7B1FA2',
    'meditate_K': '#388E3C',
    'PE_1':       '#C62828',
    'PE_2':       '#B71C1C',
    'PE':         '#C62828',
}


def parse_periods(markers):
    """[(t_lsl, label)] → [(name, t0, t1)] sorted by t0."""
    starts, out = {}, []
    for t, lbl in markers:
        if lbl.endswith('_start'):
            starts[lbl[:-len('_start')]] = t
        elif lbl.endswith('_stop'):
            n = lbl[:-len('_stop')]
            if n in starts:
                out.append((n, starts.pop(n), t))
    return sorted(out, key=lambda x: x[1])


def shade_periods(ax, periods, alpha=0.12):
    for name, t0, t1 in periods:
        c = CONDITION_COLORS.get(name, '#9E9E9E')
        ax.axvspan(t0, t1, color=c, alpha=alpha, linewidth=0)


def label_periods(ax, periods, y_frac=0.92):
    ymin, ymax = ax.get_ylim()
    y_text = ymin + y_frac * (ymax - ymin)
    for name, t0, t1 in periods:
        ax.text((t0 + t1) / 2, y_text, name, ha='center', va='top',
                fontsize=8, color=CONDITION_COLORS.get(name, '#555'),
                fontweight='bold')


# ── Stage 1: events overlay ────────────────────────────────────────────

# AU index → human-readable name (canonical 0-indexed MediaPipe order)
AU_NAMES = [
    "_neutral", "browDownL", "browDownR", "browInnerUp", "browOuterUpL",
    "browOuterUpR", "cheekPuff", "cheekSquintL", "cheekSquintR",
    "eyeBlinkL", "eyeBlinkR", "eyeLookDownL", "eyeLookDownR",
    "eyeLookInL", "eyeLookInR", "eyeLookOutL", "eyeLookOutR",
    "eyeLookUpL", "eyeLookUpR", "eyeSquintL", "eyeSquintR",
    "eyeWideL", "eyeWideR", "jawForward", "jawL", "jawOpen",
    "jawR", "mouthClose", "mouthDimpleL", "mouthDimpleR",
    "mouthFrownL", "mouthFrownR", "mouthFunnel", "mouthL",
    "mouthLowerDownL", "mouthLowerDownR", "mouthPressL",
    "mouthPressR", "mouthPucker", "mouthR", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileL",
    "mouthSmileR", "mouthStretchL", "mouthStretchR",
    "mouthUpperUpL", "mouthUpperUpR", "noseSneerL", "noseSneerR",
]

# Display set for the events-overlay figure: 10 most informative AUs from each
# expression-bearing region (avoids 52-row visual clutter).
DISPLAY_AUS = [
    3, 4, 5,            # brow inner / outer L/R
    9, 19, 21,          # eye blink + squint + wide
    7, 8,               # cheek squint L/R
    44, 45,             # mouth smile L/R
    30, 31,             # mouth frown L/R
    25, 32, 38,         # jaw open + funnel + pucker
    50, 51,             # nose sneer L/R
]


def plot_events_overlay(sid: str, save: bool = True) -> Path:
    """Render 01_events_overlay.png — DISPLAY_AUS rows of trace + event marks."""
    from cadence.synchrony.io import read_stage
    npz, meta = read_stage(sid, 1)
    face = load_face_npz(sid)
    digest = load_digest(sid)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    periods = parse_periods(digest.get('markers', []))
    role_to_p = meta.get('role_resolution', {})

    # Pull events arrays
    au_idx = np.asarray(npz['au_idx'])
    role = np.asarray(npz['role'])
    t_lsl_evt = np.asarray(npz['t_lsl'])
    alpha = np.asarray(npz['alpha'])

    n_au_display = len(DISPLAY_AUS)
    fig, axes = plt.subplots(n_au_display, 1, figsize=(20, 1.3 * n_au_display),
                              sharex=True,
                              gridspec_kw={'hspace': 0.05,
                                           'top': 0.97, 'bottom': 0.04,
                                           'left': 0.06, 'right': 0.99})

    role_color = {'therapist': '#C62828', 'patient': '#1565C0'}

    for row, au_id in enumerate(DISPLAY_AUS):
        ax = axes[row]
        # Plot both participants' trace for this AU
        for r in ('therapist', 'patient'):
            p_key = role_to_p.get(r)
            if not p_key:
                continue
            au_data = np.asarray(face[f'{p_key}_au52'])[:, au_id]
            ts_lsl = np.asarray(face[f'{p_key}_au52_ts']) + lsl_offset
            ax.plot(ts_lsl, au_data, color=role_color[r], linewidth=0.4,
                    alpha=0.55, label=r)
            # Event marks for this AU + role
            mask = (au_idx == au_id) & (role == r)
            if mask.any():
                # interp to get y-value at event times
                t_e = t_lsl_evt[mask]
                y_e = np.interp(t_e, ts_lsl, au_data)
                ax.scatter(t_e, y_e, s=10, c=role_color[r],
                           edgecolors='black', linewidths=0.3, zorder=3,
                           marker='v')
        shade_periods(ax, periods, alpha=0.10)
        ax.set_ylabel(f'{au_id:>2d}\n{AU_NAMES[au_id][:12]}',
                      rotation=0, ha='right', va='center', fontsize=8)
        ax.set_xlim(t_lsl_evt.min() if len(t_lsl_evt) else 0,
                    t_lsl_evt.max() if len(t_lsl_evt) else 1)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        if row == 0:
            ax.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.85)
            label_periods(ax, periods, y_frac=0.92)

    axes[0].set_title(
        f'{sid} — Stage 1 multiscale events  ·  {meta["n_events"]} events  ·  '
        f'GPU={meta["used_gpu"]}  ·  {meta["wall_seconds"]}s',
        fontsize=11, fontweight='bold', loc='left',
    )
    axes[-1].set_xlabel('LSL time (s)', fontsize=10)

    out = session_dir(sid, ensure=True) / '01_events_overlay.png'
    if save:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return out


# ── Stage 2: episodes overlay ──────────────────────────────────────────

def plot_episodes_overlay(sid: str, save: bool = True) -> Path:
    """Render 02_episodes_overlay.png — joint envelope + thresholds +
    episode brackets + per-condition duration histogram."""
    from cadence.synchrony.io import read_stage
    from cadence.synchrony.episodes import (
        _resampled_joint_envelope, compute_cohort_threshold,
    )
    from cadence.synchrony.events import _resolve_role_to_p

    npz, meta = read_stage(sid, 2)
    face = load_face_npz(sid)
    digest = load_digest(sid)
    role_to_p = _resolve_role_to_p(face, digest)
    lsl_offset = float(digest.get('t_start_lsl', 0.0))
    periods = parse_periods(digest.get('markers', []))
    ct = compute_cohort_threshold()
    T_star = ct['T_star']
    T_used = meta['used_threshold']

    joint, ts_rel = _resampled_joint_envelope(face, role_to_p)
    ts_lsl = ts_rel + lsl_offset

    n_eps = meta['n_episodes']

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.0, 1.0, 1.0],
                           width_ratios=[3, 1, 1, 1],
                           hspace=0.35, wspace=0.35,
                           left=0.05, right=0.99, top=0.95, bottom=0.07)

    # Panel A: envelope timecourse + threshold + episode brackets
    axA = fig.add_subplot(gs[0, :])
    axA.plot(ts_lsl, joint, color='#212121', linewidth=0.45, alpha=0.75)
    axA.axhline(T_star, color='#1565C0', linestyle='--', linewidth=1.0,
                 label=f'cohort T*={T_star:.3f}')
    if not np.isnan(T_used) and abs(T_used - T_star) > 1e-6:
        axA.axhline(T_used, color='#E65100', linestyle=':', linewidth=1.0,
                     label=f'fallback T*={T_used:.3f} (no base_EO)')
    # Episode brackets — color by zero-event vs nonzero-event
    n_evt = (np.asarray(npz['n_events_therapist']) +
              np.asarray(npz['n_events_patient']))
    for i in range(n_eps):
        t0_ep = float(npz['t_start_lsl'][i])
        t1_ep = float(npz['t_end_lsl'][i])
        color = '#9E9E9E' if n_evt[i] == 0 else '#388E3C'
        axA.axvspan(t0_ep, t1_ep, color=color, alpha=0.25, linewidth=0)
    shade_periods(axA, periods, alpha=0.10)
    axA.set_xlim(ts_lsl[0], ts_lsl[-1])
    axA.set_ylim(0, np.percentile(joint, 99) * 1.05)
    axA.set_ylabel('joint envelope\n(max of p_T, p_P)', fontsize=10)
    axA.set_title(
        f'{sid} — Stage 2 episodes  ·  n_episodes={n_eps} '
        f'(zero-event: {meta["n_zero_event_episodes"]}, '
        f'cluster-eligible: {n_eps - meta["n_zero_event_episodes"]})  ·  '
        f'episode_rate={meta["episode_rate_per_min"]:.2f}/min  ·  '
        f'fraction_active={meta["fraction_active"]:.1%}',
        fontsize=11, fontweight='bold', loc='left',
    )
    axA.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    axA.tick_params(labelsize=8)
    label_periods(axA, periods, y_frac=0.92)

    # Panel B: per-condition episode COUNT bar chart
    axB = fig.add_subplot(gs[1, 0])
    cond = np.asarray(npz['condition'])
    cond_present = sorted(set(cond) - {''}, key=lambda c: [p[1] for p in periods
                                                            if p[0] == c][0]
                              if any(p[0] == c for p in periods) else 0)
    counts = [int((cond == c).sum()) for c in cond_present]
    nz_counts = [int(((cond == c) & (n_evt > 0)).sum()) for c in cond_present]
    x = np.arange(len(cond_present))
    axB.bar(x, counts, 0.6, color='#9E9E9E', edgecolor='white',
             label='all episodes')
    axB.bar(x, nz_counts, 0.6, color='#388E3C', edgecolor='white',
             label='cluster-eligible')
    axB.set_xticks(x)
    axB.set_xticklabels(cond_present, rotation=20, fontsize=8, ha='right')
    axB.set_ylabel('# episodes', fontsize=9)
    axB.set_title('Per-condition episode counts', fontsize=10, loc='left')
    axB.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.85)
    axB.tick_params(labelsize=8)
    axB.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel C: per-condition mean peak_env
    axC = fig.add_subplot(gs[1, 1])
    means = [float(np.asarray(npz['peak_env'])[cond == c].mean())
              if (cond == c).any() else 0 for c in cond_present]
    axC.bar(x, means, 0.6, color=[CONDITION_COLORS.get(c, '#9E9E9E')
                                       for c in cond_present],
             edgecolor='white')
    axC.set_xticks(x)
    axC.set_xticklabels(cond_present, rotation=20, fontsize=8, ha='right')
    axC.set_ylabel('peak_env', fontsize=9)
    axC.set_title('Per-condition mean peak intensity', fontsize=10, loc='left')
    axC.tick_params(labelsize=8)
    axC.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel D: per-condition mean duration
    axD = fig.add_subplot(gs[1, 2])
    durs = [float(np.asarray(npz['duration_s'])[cond == c].mean())
             if (cond == c).any() else 0 for c in cond_present]
    axD.bar(x, durs, 0.6, color=[CONDITION_COLORS.get(c, '#9E9E9E')
                                      for c in cond_present],
             edgecolor='white')
    axD.set_xticks(x)
    axD.set_xticklabels(cond_present, rotation=20, fontsize=8, ha='right')
    axD.set_ylabel('duration (s)', fontsize=9)
    axD.set_title('Per-condition mean episode duration', fontsize=10, loc='left')
    axD.tick_params(labelsize=8)
    axD.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel E: episode duration histogram (log-y)
    axE = fig.add_subplot(gs[1, 3])
    durations = np.asarray(npz['duration_s'])
    if len(durations):
        axE.hist(durations, bins=30, color='#388E3C', edgecolor='white')
    axE.set_yscale('log')
    axE.set_xlabel('duration (s)', fontsize=9)
    axE.set_ylabel('# episodes', fontsize=9)
    axE.set_title('Duration histogram', fontsize=10, loc='left')
    axE.tick_params(labelsize=8)
    axE.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel F: peak_env histogram
    axF = fig.add_subplot(gs[2, 0])
    axF.hist(np.asarray(npz['peak_env']), bins=40, color='#1565C0',
              edgecolor='white')
    axF.axvline(T_star, color='#1565C0', linestyle='--', linewidth=1.0)
    axF.set_xlabel('peak envelope', fontsize=9)
    axF.set_ylabel('# episodes', fontsize=9)
    axF.set_title('Peak intensity distribution', fontsize=10, loc='left')
    axF.tick_params(labelsize=8)

    # Panel G: events-per-episode histogram
    axG = fig.add_subplot(gs[2, 1])
    axG.hist(n_evt, bins=np.arange(0, max(n_evt.max(), 1) + 2) - 0.5,
              color='#7B1FA2', edgecolor='white')
    axG.set_xlabel('# Stage-1 events in episode', fontsize=9)
    axG.set_ylabel('# episodes', fontsize=9)
    axG.set_title('Member-event count', fontsize=10, loc='left')
    axG.tick_params(labelsize=8)

    # Panel H: per-role events-per-episode scatter
    axH = fig.add_subplot(gs[2, 2:])
    n_th = np.asarray(npz['n_events_therapist'])
    n_pa = np.asarray(npz['n_events_patient'])
    sc = axH.scatter(n_th, n_pa, s=10, alpha=0.5,
                      c=np.asarray(npz['duration_s']), cmap='viridis')
    axH.set_xlabel('# therapist events', fontsize=9)
    axH.set_ylabel('# patient events', fontsize=9)
    axH.set_title('Per-episode: therapist vs patient event counts (color = duration_s)',
                   fontsize=10, loc='left')
    plt.colorbar(sc, ax=axH, label='duration (s)', fraction=0.046, pad=0.02)
    axH.tick_params(labelsize=8)
    axH.plot([0, max(n_th.max(), n_pa.max(), 1)],
              [0, max(n_th.max(), n_pa.max(), 1)],
              ':', color='gray', linewidth=0.5)

    out = session_dir(sid, ensure=True) / '02_episodes_overlay.png'
    if save:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return out


# ── Paired-face AU heatmap ─────────────────────────────────────────────
# A schematic face built from labeled region polygons in normalized
# (x ∈ [-1, 1], y ∈ [-1, 1]) face coordinates. Each region is a
# matplotlib Polygon tinted by the cluster's mean AU activation in that
# region. The whole face is symmetric about x=0.

# Region polygon vertices in normalized face coords (front-facing portrait,
# +x = subject's right side as we see them, +y = up). Vertices are
# anatomically plausible without aiming for photorealism — the goal is
# legibility, not 3D accuracy.

_FACE_OUTLINE = np.array([
    [-0.55,  0.85], [-0.65,  0.55], [-0.70,  0.20], [-0.65, -0.20],
    [-0.50, -0.55], [-0.30, -0.85], [ 0.00, -0.95], [ 0.30, -0.85],
    [ 0.50, -0.55], [ 0.65, -0.20], [ 0.70,  0.20], [ 0.65,  0.55],
    [ 0.55,  0.85], [ 0.30,  0.95], [ 0.00,  1.00], [-0.30,  0.95],
])

_REGION_POLYGONS = {
    # Brow band — a single horizontal strip across the upper face
    'brow': np.array([
        [-0.60,  0.50], [-0.55,  0.65], [-0.20,  0.70], [ 0.20,  0.70],
        [ 0.55,  0.65], [ 0.60,  0.50], [ 0.40,  0.40], [-0.40,  0.40],
    ]),
    # Eye band (both eyes, single polygon for simplicity — color = mean of L+R)
    'eye': np.array([
        [-0.55,  0.30], [-0.50,  0.40], [-0.20,  0.45], [-0.05,  0.40],
        [-0.05,  0.20], [-0.20,  0.15], [-0.50,  0.20],
    ]),
    'eye_R': np.array([  # mirror image, drawn separately for the right eye
        [ 0.55,  0.30], [ 0.50,  0.40], [ 0.20,  0.45], [ 0.05,  0.40],
        [ 0.05,  0.20], [ 0.20,  0.15], [ 0.50,  0.20],
    ]),
    # Nose triangle
    'nose': np.array([
        [-0.10,  0.20], [-0.18, -0.20], [ 0.00, -0.25],
        [ 0.18, -0.20], [ 0.10,  0.20],
    ]),
    # Cheek pads (two)
    'cheek': np.array([
        [-0.55,  0.05], [-0.55, -0.30], [-0.30, -0.40], [-0.20, -0.20],
        [-0.30,  0.00],
    ]),
    'cheek_R': np.array([
        [ 0.55,  0.05], [ 0.55, -0.30], [ 0.30, -0.40], [ 0.20, -0.20],
        [ 0.30,  0.00],
    ]),
    # Mouth: split into smile (corners + upper lip wing), frown (lower
    # lip + chin region), jaw (jaw line + lower face)
    'mouth_smile': np.array([
        [-0.30, -0.50], [-0.10, -0.45], [ 0.10, -0.45], [ 0.30, -0.50],
        [ 0.20, -0.55], [-0.20, -0.55],
    ]),
    'mouth_frown': np.array([
        [-0.20, -0.55], [-0.10, -0.65], [ 0.10, -0.65], [ 0.20, -0.55],
        [ 0.10, -0.50], [-0.10, -0.50],
    ]),
    'mouth_jaw': np.array([
        [-0.40, -0.65], [-0.30, -0.85], [ 0.00, -0.95], [ 0.30, -0.85],
        [ 0.40, -0.65], [ 0.20, -0.65], [-0.20, -0.65],
    ]),
}


def _region_value_to_color(val: float, vmin: float, vmax: float, cmap='Reds'):
    """Map a (clamped) per-region scalar into a matplotlib RGBA via cmap."""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return cm.get_cmap(cmap)(norm(val))


def draw_paired_face_schematic(au_vec_therapist: np.ndarray,
                                 au_vec_patient: np.ndarray,
                                 ax_t=None, ax_p=None,
                                 vmin: float = 0.0, vmax: float = 1.0,
                                 cmap='Reds', annotate: bool = True):
    """Render two schematic faces side-by-side, tinted by per-region mean AU.

    Args:
        au_vec_therapist: (52,) per-AU mean activation in [0, 1] for therapist.
        au_vec_patient:   (52,) for patient.
        ax_t, ax_p: optional matplotlib axes (created if None).
        vmin, vmax: cmap range for tinting.

    Returns:
        (fig, ax_t, ax_p)
    """
    from matplotlib.patches import Polygon as MplPolygon

    if ax_t is None or ax_p is None:
        fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(8, 4.5))
    else:
        fig = ax_t.figure

    for ax, au_vec, role_label in (
        (ax_t, au_vec_therapist, 'therapist'),
        (ax_p, au_vec_patient,   'patient'),
    ):
        # Per-region mean activation
        reg_vals = {}
        for r, idxs in AU_REGIONS_7.items():
            v = float(np.nanmean(au_vec[idxs])) if len(idxs) else 0.0
            reg_vals[r] = max(v, 0.0)  # negative baseline-subtracted means → 0 tint

        ax.set_aspect('equal')
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-1.05, 1.10)
        ax.axis('off')
        # Face outline
        outline = MplPolygon(_FACE_OUTLINE, closed=True, facecolor='#FAFAFA',
                              edgecolor='#212121', linewidth=1.5, zorder=1)
        ax.add_patch(outline)
        # Each region polygon
        for region_key, verts in _REGION_POLYGONS.items():
            base_region = region_key.replace('_R', '')  # 'eye_R' → 'eye'
            v = reg_vals.get(base_region, 0.0)
            color = _region_value_to_color(v, vmin, vmax, cmap)
            poly = MplPolygon(verts, closed=True, facecolor=color,
                                edgecolor='#424242', linewidth=0.6, zorder=2)
            ax.add_patch(poly)
            if annotate and not region_key.endswith('_R'):
                cx, cy = verts.mean(axis=0)
                ax.text(cx, cy, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='#212121', alpha=0.75)
        ax.set_title(role_label, fontsize=10, fontweight='bold')

    return fig, ax_t, ax_p


# ── Cohort embedding overview ──────────────────────────────────────────

def plot_cohort_embedding(save: bool = True) -> Path:
    """Render fig_cohort_embedding.png — 4-panel coloring of the 2D UMAP."""
    from cadence.synchrony.io import cohort_dir
    cdir = cohort_dir()
    coh = dict(np.load(cdir / 'cohort_features.npz', allow_pickle=True))
    cls = dict(np.load(cdir / 'cohort_clusters.npz', allow_pickle=True))

    e2 = np.asarray(cls['embedding_2d'])
    labels = np.asarray(cls['labels'])
    sids = np.asarray(coh['session_id'])
    conds = np.asarray(coh['condition'])
    intensity = np.asarray(coh['peak_env'])
    duration = np.asarray(coh['duration_s'])

    finite = np.isfinite(e2[:, 0])
    e2f = e2[finite]
    labelsf = labels[finite]
    sidsf = sids[finite]
    condsf = conds[finite]
    intensityf = intensity[finite]
    durationf = duration[finite]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # (A) clusters
    cluster_ids = sorted(set(int(c) for c in labelsf))
    cmap = plt.get_cmap('tab20', max(20, len(cluster_ids)))
    for i, c in enumerate(cluster_ids):
        m = labelsf == c
        if c == -1:
            axs[0].scatter(e2f[m, 0], e2f[m, 1], s=4, c='#bdbdbd',
                            label='noise', alpha=0.4)
        else:
            axs[0].scatter(e2f[m, 0], e2f[m, 1], s=8, c=[cmap(i % 20)],
                            label=f'c{c} (n={m.sum()})', alpha=0.75)
    axs[0].set_title('HDBSCAN clusters', fontsize=11, fontweight='bold')
    axs[0].legend(fontsize=6, loc='best', frameon=True, framealpha=0.85,
                   ncol=2)

    # (B) session
    unique_sids = sorted(set(sidsf))
    sid_to_color = {s: plt.get_cmap('tab20', max(20, len(unique_sids)))(i % 20)
                     for i, s in enumerate(unique_sids)}
    for s in unique_sids:
        m = sidsf == s
        axs[1].scatter(e2f[m, 0], e2f[m, 1], s=6, c=[sid_to_color[s]],
                        alpha=0.6, label=s)
    axs[1].set_title('session_id', fontsize=11, fontweight='bold')
    axs[1].legend(fontsize=5, loc='best', frameon=True, ncol=2)

    # (C) condition
    cond_ids = sorted(set(condsf) - {''})
    for c in cond_ids:
        m = condsf == c
        axs[2].scatter(e2f[m, 0], e2f[m, 1], s=6,
                        c=[CONDITION_COLORS.get(c, '#9e9e9e')],
                        alpha=0.6, label=c)
    axs[2].set_title('condition', fontsize=11, fontweight='bold')
    axs[2].legend(fontsize=7, loc='best', frameon=True)

    # (D) intensity
    sc = axs[3].scatter(e2f[:, 0], e2f[:, 1], s=6, c=intensityf, cmap='viridis',
                         alpha=0.7)
    plt.colorbar(sc, ax=axs[3], label='peak_env', fraction=0.046, pad=0.02)
    axs[3].set_title('peak_env (intensity)', fontsize=11, fontweight='bold')

    for ax in axs:
        ax.set_xlabel('UMAP-1', fontsize=9)
        ax.set_ylabel('UMAP-2', fontsize=9)

    fig.suptitle(
        f'Cohort UMAP embedding  ·  '
        f'{len(e2f)} cluster-eligible episodes  ·  '
        f'{len(set(int(c) for c in labelsf if c >= 0))} clusters + '
        f'{int((labelsf == -1).sum())} noise',
        fontsize=12, fontweight='bold')
    fig.tight_layout()

    out = cohort_dir(ensure=True) / 'fig_cohort_embedding.png'
    if save:
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
    return out
