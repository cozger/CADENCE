from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap


def run_obs_space_analysis(
    sessions: list, output_dir: str,
    n_neighbors: int = 30, min_dist: float = 0.1,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Pool all prewhitened observations ────────────────────────────────
    Y_all = np.vstack([s.Y_pw for s in sessions])
    state_labels_all = np.concatenate([
        s.path_unconstrained if s.path_unconstrained is not None
        else np.zeros(len(s.t_common), dtype=int)
        for s in sessions
    ])

    def _coarse(phase_name: str) -> str:
        p = phase_name.lower()
        if 'base' in p: return 'baseline'
        if 'conv' in p: return 'conversation'
        return 'task'

    phase_labels_all = []
    session_labels_all = []
    time_labels_all = []
    for sess in sessions:
        N = len(sess.t_common)
        phase_arr = np.full(N, 'unknown', dtype=object)
        for seg_name, t0, t1 in sess.segments:
            mask = (sess.t_common >= t0) & (sess.t_common <= t1)
            phase_arr[mask] = _coarse(seg_name)
        phase_labels_all.append(phase_arr)
        session_labels_all.append(np.full(N, sess.name, dtype=object))
        t_norm = (sess.t_common - sess.t_common[0]) / max(1.0, sess.t_common[-1] - sess.t_common[0])
        time_labels_all.append(t_norm)

    phase_all = np.concatenate(phase_labels_all)
    sess_all = np.concatenate(session_labels_all)
    time_all = np.concatenate(time_labels_all)

    # ── PCA scree ────────────────────────────────────────────────────────
    pca_full = PCA().fit(Y_all)
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n_80 = int(np.searchsorted(cumvar, 0.80)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(evr) + 1), evr, alpha=0.7, label='Per-component')
    ax.plot(range(1, len(cumvar) + 1), cumvar, color='red', marker='o', ms=3, label='Cumulative')
    ax.axhline(0.80, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Component'); ax.set_ylabel('Variance explained')
    ax.set_title(f'PCA scree — 80% at PC{n_80}, 95% at PC{n_95}')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'pca_scree.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Silhouette in PCA-reduced space ──────────────────────────────────
    Y_reduced = PCA(n_components=n_80).fit_transform(Y_all)

    state_sil = phase_sil = state_db = float('nan')
    if len(np.unique(state_labels_all)) > 1:
        state_sil = float(silhouette_score(Y_reduced, state_labels_all,
                                           sample_size=min(5000, len(Y_reduced))))
        state_db = float(davies_bouldin_score(Y_reduced, state_labels_all))
    if len(np.unique(phase_all)) > 1:
        phase_sil = float(silhouette_score(Y_reduced, phase_all,
                                           sample_size=min(5000, len(Y_reduced))))

    pd.DataFrame([{
        'state_silhouette': state_sil, 'phase_silhouette': phase_sil,
        'state_davies_bouldin': state_db,
        'n_pca_80pct': n_80, 'n_pca_95pct': n_95,
    }]).to_csv(os.path.join(output_dir, 'cluster_quality.csv'), index=False)

    # ── UMAP embedding ───────────────────────────────────────────────────
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    idx = np.arange(len(Y_all))
    if len(idx) > 10000:
        idx = np.random.default_rng(42).choice(len(Y_all), 10000, replace=False)
        idx.sort()
    embedding = reducer.fit_transform(Y_all[idx])

    STATE_CMAP = plt.colormaps['tab10'].resampled(4)
    PHASE_CMAP = {'baseline': 'steelblue', 'conversation': 'green',
                  'task': 'orange', 'unknown': 'gray'}

    def _save_umap(color_data, title, filename, cmap=None, vmin=None, vmax=None, discrete_map=None):
        fig, ax = plt.subplots(figsize=(7, 6))
        if discrete_map is not None:
            for label, color in discrete_map.items():
                mask = color_data[idx] == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           s=1, alpha=0.3, color=color, label=str(label))
            ax.legend(markerscale=5, fontsize=7, loc='upper right')
        else:
            sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=color_data[idx], s=1, alpha=0.3, cmap=cmap,
                            vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, fraction=0.03)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
        fig.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches='tight')
        plt.close(fig)

    K = int(state_labels_all.max()) + 1 if state_labels_all.dtype.kind in 'iu' else 4
    _save_umap(state_labels_all.astype(str), 'UMAP colored by rSLDS state',
               'umap_by_state.png', discrete_map={str(k): STATE_CMAP(k) for k in range(K)})
    _save_umap(phase_all, 'UMAP colored by protocol phase',
               'umap_by_phase.png', discrete_map=PHASE_CMAP)
    unique_sess = list(dict.fromkeys(sess_all))
    sess_cmap = {s: plt.cm.tab20(i / max(1, len(unique_sess) - 1))
                 for i, s in enumerate(unique_sess)}
    _save_umap(sess_all, 'UMAP colored by session', 'umap_by_session.png', discrete_map=sess_cmap)
    _save_umap(time_all.astype(float), 'UMAP colored by time-within-session',
               'umap_by_time.png', cmap='plasma', vmin=0, vmax=1)

    # ── Interpretation ───────────────────────────────────────────────────
    if state_sil > phase_sil:
        interp = 'States more separable than conditions — model captures coupling dynamics.'
    elif phase_sil > state_sil and phase_sil > 0.05:
        interp = ('State silhouette < phase silhouette — states tracking experimental '
                  'condition more than coupling dynamics.')
    else:
        interp = ('Both silhouettes near zero. If no temporal gradient in umap_by_time.png, '
                  'observation space is genuinely continuous — consider reporting z_t latent '
                  'trajectory as primary object rather than state labels.')

    with open(os.path.join(output_dir, 'module1_report.md'), 'w') as f:
        f.write('# Module 1: Observation Space Structure\n\n')
        f.write(f'PCA 80%: {n_80} components | 95%: {n_95} components\n')
        f.write(f'State silhouette (PCA-reduced): {state_sil:.3f}\n')
        f.write(f'Phase silhouette (PCA-reduced): {phase_sil:.3f}\n')
        f.write(f'State Davies-Bouldin: {state_db:.3f}\n\n')
        f.write(f'**Interpretation:** {interp}\n')

    return {
        'state_silhouette': state_sil, 'phase_silhouette': phase_sil,
        'state_davies_bouldin': state_db,
        'n_pca_components_80pct': n_80,
    }
