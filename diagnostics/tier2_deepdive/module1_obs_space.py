from __future__ import annotations
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap


def shuffle_null_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    n_shuffles: int = 200,
    sample_size: int = 5000,
    seed: int = 42,
) -> dict:
    """Compare real silhouette to silhouette under random permutation of labels.

    Answers: "Is the observed (possibly negative) silhouette distinguishable from
    what we'd get assigning random labels with the same marginal distribution?"

    Returns dict with real, shuffle mean/std, z-score, p_right (fraction of
    shuffles that beat real).
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X = X[idx]
        labels = labels[idx]
    if len(np.unique(labels)) < 2:
        return {'real': float('nan'), 'shuffle_mean': float('nan'),
                'shuffle_std': float('nan'), 'z': float('nan'), 'p_right': float('nan')}
    sil_real = float(silhouette_score(X, labels))
    sils = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        sils[i] = float(silhouette_score(X, rng.permutation(labels)))
    z = (sil_real - sils.mean()) / max(sils.std(), 1e-6)
    return {
        'real': sil_real,
        'shuffle_mean': float(sils.mean()),
        'shuffle_std': float(sils.std()),
        'z': float(z),
        'p_right': float((sils >= sil_real).mean()),
        'shuffle_distribution': sils,
    }


def pairwise_d_emit_distances(
    mean_d: np.ndarray,
    labels: Optional[list] = None,
) -> dict:
    """Pairwise L2 distances between K state emission-mean centroids.

    Small min/max ratio → some state pairs are much closer than others (potential
    redundancy). Ratio near 1 → all states roughly equidistant.
    """
    K = mean_d.shape[0]
    dist = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dist[i, j] = np.linalg.norm(mean_d[i] - mean_d[j])
    off_diag = dist[np.triu_indices(K, k=1)]
    return {
        'distances': dist,
        'labels': labels if labels is not None else [f'S{k}' for k in range(K)],
        'norms': np.linalg.norm(mean_d, axis=1),
        'min_off_diag': float(off_diag.min()),
        'mean_off_diag': float(off_diag.mean()),
        'max_off_diag': float(off_diag.max()),
        'min_max_ratio': float(off_diag.min() / max(off_diag.max(), 1e-12)),
    }


def latent_space_silhouette(sessions: list) -> Optional[dict]:
    """Silhouette on x_smooth (latent trajectories) if available across sessions.

    Returns None if x_smooth is missing from any session (older hierarchical runs).
    """
    x_list, path_list = [], []
    for s in sessions:
        if s.x_smooth is None or s.path_unconstrained is None:
            return None
        x_list.append(s.x_smooth)
        path_list.append(s.path_unconstrained)
    X_all = np.vstack(x_list)
    p_all = np.concatenate(path_list)
    if len(np.unique(p_all)) < 2:
        return None
    rng = np.random.default_rng(42)
    sub_n = min(5000, len(X_all))
    sub_idx = rng.choice(len(X_all), sub_n, replace=False)
    X_sub = X_all[sub_idx]
    p_sub = p_all[sub_idx]
    return {
        'silhouette': float(silhouette_score(X_sub, p_sub)),
        'davies_bouldin': float(davies_bouldin_score(X_sub, p_sub)),
        'd_latent': int(X_all.shape[1]),
    }


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

    # ── Shuffle-null silhouette on state labels ───────────────────────────
    shuffle_result = None
    if len(np.unique(state_labels_all)) > 1:
        shuffle_result = shuffle_null_silhouette(
            Y_reduced, state_labels_all, n_shuffles=200, sample_size=5000, seed=42)
        pd.DataFrame([{
            'silhouette_real': shuffle_result['real'],
            'silhouette_shuffle_mean': shuffle_result['shuffle_mean'],
            'silhouette_shuffle_std': shuffle_result['shuffle_std'],
            'z_vs_shuffle': shuffle_result['z'],
            'p_right': shuffle_result['p_right'],
        }]).to_csv(os.path.join(output_dir, 'shuffle_null_silhouette.csv'), index=False)

        # Shuffle distribution histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(shuffle_result['shuffle_distribution'], bins=30, alpha=0.7, color='gray',
                label=f"Shuffle (n=200, mean={shuffle_result['shuffle_mean']:+.4f})")
        ax.axvline(shuffle_result['real'], color='red', linewidth=2,
                   label=f"Real ({shuffle_result['real']:+.4f}, z={shuffle_result['z']:+.2f})")
        ax.set_xlabel('Silhouette score'); ax.set_ylabel('# shuffles')
        ax.set_title('State silhouette vs shuffle-null (observation PCA-reduced)')
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(output_dir, 'silhouette_shuffle_null.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── Pairwise d_emit distances (if hierarchical params available) ──────
    pairwise_result = None
    all_d_emit = [s.d_emit for s in sessions if s.d_emit is not None]
    if all_d_emit:
        mean_d = np.mean(all_d_emit, axis=0)
        state_labels_list = sessions[0].state_labels if sessions[0].state_labels else None
        pairwise_result = pairwise_d_emit_distances(mean_d, state_labels_list)
        dist_df = pd.DataFrame(pairwise_result['distances'],
                               index=pairwise_result['labels'],
                               columns=pairwise_result['labels'])
        dist_df.to_csv(os.path.join(output_dir, 'pairwise_d_emit.csv'))

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pairwise_result['distances'], cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(pairwise_result['labels'])))
        ax.set_xticklabels(pairwise_result['labels'], rotation=45, ha='right')
        ax.set_yticks(range(len(pairwise_result['labels'])))
        ax.set_yticklabels(pairwise_result['labels'])
        for i in range(len(pairwise_result['labels'])):
            for j in range(len(pairwise_result['labels'])):
                ax.text(j, i, f"{pairwise_result['distances'][i, j]:.2f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if pairwise_result['distances'][i, j] <
                        pairwise_result['distances'].max() * 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title('Pairwise |d_emit[i] - d_emit[j]|')
        fig.savefig(os.path.join(output_dir, 'pairwise_d_emit_heatmap.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── Latent-space silhouette (if x_smooth available) ───────────────────
    latent_result = latent_space_silhouette(sessions)
    if latent_result is not None:
        pd.DataFrame([latent_result]).to_csv(
            os.path.join(output_dir, 'latent_silhouette.csv'), index=False)

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
    if shuffle_result is not None and abs(shuffle_result['z']) >= 2:
        if shuffle_result['z'] >= 2:
            interp = (f"States {shuffle_result['z']:.1f}sigma ABOVE shuffle — state labels "
                      f"carry observation-space structure (despite small absolute silhouette).")
        else:
            interp = (f"States {-shuffle_result['z']:.1f}sigma BELOW shuffle — state assignment "
                      f"is MORE scattered than random labels. The model is using temporal/latent "
                      f"structure that actively contradicts observation-space clustering. "
                      f"Validate with latent-space silhouette and Module 4 LOO-CV before "
                      f"concluding states are meaningful.")
    elif state_sil > phase_sil:
        interp = 'States more separable than conditions — model captures coupling dynamics.'
    elif phase_sil > state_sil and phase_sil > 0.05:
        interp = ('State silhouette < phase silhouette — states tracking experimental '
                  'condition more than coupling dynamics.')
    else:
        interp = ('Both silhouettes near zero. If no temporal gradient in umap_by_time.png, '
                  'observation space is genuinely continuous — consider reporting z_t latent '
                  'trajectory as primary object rather than state labels.')

    with open(os.path.join(output_dir, 'module1_report.md'), 'w', encoding='utf-8') as f:
        f.write('# Module 1: Observation Space Structure\n\n')
        f.write(f'PCA 80%: {n_80} components | 95%: {n_95} components\n')
        f.write(f'State silhouette (PCA-reduced): {state_sil:.3f}\n')
        f.write(f'Phase silhouette (PCA-reduced): {phase_sil:.3f}\n')
        f.write(f'State Davies-Bouldin: {state_db:.3f}\n\n')

        if shuffle_result is not None:
            f.write('## Shuffle-null silhouette\n\n')
            f.write(f"Real silhouette: {shuffle_result['real']:+.4f}\n")
            f.write(f"Shuffle mean ± std: {shuffle_result['shuffle_mean']:+.4f} "
                    f"± {shuffle_result['shuffle_std']:.4f}\n")
            f.write(f"z-score vs shuffle: {shuffle_result['z']:+.2f} "
                    f"(p_right={shuffle_result['p_right']:.3f})\n\n")

        if pairwise_result is not None:
            f.write('## Pairwise d_emit distances\n\n')
            f.write(f"State norms: " + ', '.join(
                f"{l}={n:.2f}" for l, n in
                zip(pairwise_result['labels'], pairwise_result['norms'])) + '\n')
            f.write(f"Off-diagonal distances — min: {pairwise_result['min_off_diag']:.3f}, "
                    f"mean: {pairwise_result['mean_off_diag']:.3f}, "
                    f"max: {pairwise_result['max_off_diag']:.3f}\n")
            f.write(f"min/max ratio: {pairwise_result['min_max_ratio']:.3f} "
                    f"(near 1.0 = equidistant; near 0 = some state pairs nearly duplicate)\n\n")

        if latent_result is not None:
            f.write('## Latent-space silhouette (x_smooth)\n\n')
            f.write(f"Latent silhouette (D={latent_result['d_latent']}): "
                    f"{latent_result['silhouette']:+.4f}\n")
            f.write(f"Latent Davies-Bouldin: {latent_result['davies_bouldin']:.3f}\n\n")

        f.write(f'**Interpretation:** {interp}\n')

    out = {
        'state_silhouette': state_sil, 'phase_silhouette': phase_sil,
        'state_davies_bouldin': state_db,
        'n_pca_components_80pct': n_80,
    }
    if shuffle_result is not None:
        out['shuffle_z'] = shuffle_result['z']
        out['shuffle_p_right'] = shuffle_result['p_right']
    if pairwise_result is not None:
        out['d_emit_min_max_ratio'] = pairwise_result['min_max_ratio']
    if latent_result is not None:
        out['latent_silhouette'] = latent_result['silhouette']
    return out
