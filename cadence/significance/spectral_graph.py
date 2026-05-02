"""V9 Spectral Graph Theory — CADENCE

Three applications of spectral graph theory for multimodal coupling analysis:

1. **Inter-brain bipartite Laplacian** — Algebraic connectivity λ₂(t) as a
   principled coupling index. Fiedler vector reveals ROI-pair topology.

2. **AU spectral clustering** — Data-driven AU grouping via normalized
   Laplacian eigenmodes with automatic k selection (eigengap heuristic).

3. **Graph spectral filtering of multimodal features** — Decomposes 18D
   z-timecourses into graph-frequency components. Coupling flexibility
   index from spectral energy ratio.

All graph operations use scipy.linalg on small matrices (8×8 to 52×52).
No GPU needed — eigendecomposition is O(n³) on n < 100.
"""

import numpy as np
from scipy import linalg as la
from scipy.ndimage import gaussian_filter1d
import networkx as nx
import community as community_louvain


# ── MediaPipe Blendshape names (52 channels, standard order) ─────────

MEDIAPIPE_AU_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
    'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
    'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight',
]


# ═════════════════════════════════════════════════════════════════════
#  Shared Utilities
# ═════════════════════════════════════════════════════════════════════

def _normalized_laplacian(W):
    """Normalized graph Laplacian L = I - D^{-1/2} W D^{-1/2}.

    Eigenvalues lie in [0, 2]. Invariant to degree scaling.

    Args:
        W: (N, N) symmetric non-negative weight matrix.

    Returns:
        L: (N, N) normalized Laplacian.
    """
    d = W.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    # L = I - D^{-1/2} W D^{-1/2}
    # Efficiently: L_ij = delta_ij - d_i^{-1/2} W_ij d_j^{-1/2}
    L = -W * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    np.fill_diagonal(L, 1.0 - L.diagonal())  # diagonal = 1 (for connected nodes)
    # Fix isolated nodes: diagonal should be 0 if degree is 0
    isolated = d < 1e-12
    L[isolated, :] = 0.0
    L[:, isolated] = 0.0
    return L


def _eigengap_k(eigvals, k_max=12, k_min=2):
    """Select number of clusters via largest eigengap.

    Skips the trivial gap at λ₁=0 → λ₂.

    Args:
        eigvals: sorted eigenvalues (ascending).
        k_max: maximum k to consider.
        k_min: minimum k.

    Returns:
        k: optimal number of clusters.
        gaps: array of consecutive eigenvalue gaps.
    """
    n = min(len(eigvals), k_max + 1)
    gaps = np.diff(eigvals[:n])
    # Best gap in range [k_min-1, k_max-1] (gap index i → k = i+1)
    search_start = max(k_min - 1, 1)  # skip trivial gap at index 0
    search_end = min(n - 1, k_max)
    if search_start >= search_end:
        return k_min, gaps
    k = int(np.argmax(gaps[search_start:search_end]) + search_start + 1)
    return max(k, k_min), gaps


# ═════════════════════════════════════════════════════════════════════
#  Application 1: Inter-Brain Bipartite Laplacian
# ═════════════════════════════════════════════════════════════════════

def reshape_roi_pairs_to_matrix(coh_flat, n_rois=4):
    """Reshape flattened ROI-pair coherence to (n_rois, n_rois) matrix form.

    eeg_band_coherence returns (n_bands, 16, n_windows) where 16 = 4×4
    ROI pairs in row-major order. This reshapes to (n_bands, 4, 4, n_windows).

    Args:
        coh_flat: (n_bands, n_roi_pairs, n_windows) or (n_roi_pairs, n_windows).
        n_rois: number of ROIs per participant (default 4).

    Returns:
        Reshaped array with ROI pair dimension expanded to (n_rois, n_rois).
    """
    if coh_flat.ndim == 2:
        # (n_roi_pairs, T) → (n_rois, n_rois, T)
        T = coh_flat.shape[1]
        return coh_flat.reshape(n_rois, n_rois, T)
    elif coh_flat.ndim == 3:
        # (n_bands, n_roi_pairs, T) → (n_bands, n_rois, n_rois, T)
        n_bands, _, T = coh_flat.shape
        return coh_flat.reshape(n_bands, n_rois, n_rois, T)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {coh_flat.ndim}D")


def interbrain_laplacian_timeseries(coh_cross, band_names=None):
    """Time-varying bipartite graph spectrum from cross-participant coherence.

    Builds an 8-node bipartite graph (4 P1 ROIs + 4 P2 ROIs) at each
    time point. Edges connect P1↔P2 only (no within-brain edges).

    The bipartite adjacency is:
        W = [0   B ]
            [Bᵀ  0 ]
    where B is the 4×4 cross-participant coherence matrix.

    Args:
        coh_cross: (n_bands, n_rois, n_rois, T) cross-participant coherence,
                   OR (n_rois, n_rois, T) for single band.
        band_names: list of band names (default: band_0, band_1, ...).

    Returns:
        dict keyed by band name, each containing:
            lambda2: (T,) algebraic connectivity timeseries.
            fiedler: (T, 8) Fiedler vector at each time point.
                     First 4 entries = P1 ROIs, last 4 = P2 ROIs.
            spectrum: (T, 8) full eigenvalue spectrum.
            svd_sigma: (T, 4) singular values of cross-block B.
        If single band input, returns the inner dict directly.
    """
    single_band = (coh_cross.ndim == 3)
    if single_band:
        coh_cross = coh_cross[np.newaxis]

    n_bands, n_r, _, T = coh_cross.shape
    n_total = 2 * n_r

    if band_names is None:
        band_names = [f'band_{i}' for i in range(n_bands)]

    results = {}
    for b, bname in enumerate(band_names):
        lam2 = np.empty(T, dtype=np.float64)
        fiedler = np.empty((T, n_total), dtype=np.float64)
        spectrum = np.empty((T, n_total), dtype=np.float64)
        svd_sig = np.empty((T, n_r), dtype=np.float64)

        for t in range(T):
            B = np.abs(coh_cross[b, :, :, t])  # non-negative cross-block

            # Bipartite adjacency
            W = np.zeros((n_total, n_total))
            W[:n_r, n_r:] = B
            W[n_r:, :n_r] = B.T

            L = _normalized_laplacian(W)
            eigvals, eigvecs = la.eigh(L)

            spectrum[t] = eigvals
            lam2[t] = eigvals[1]
            fiedler[t] = eigvecs[:, 1]

            # SVD of cross-block reveals coupling modes
            _, s, _ = la.svd(B, full_matrices=False)
            svd_sig[t] = s

        results[bname] = {
            'lambda2': lam2,
            'fiedler': fiedler,
            'spectrum': spectrum,
            'svd_sigma': svd_sig,
        }

    if single_band:
        return results[band_names[0]]
    return results


def interbrain_fiedler_summary(fiedler, n_rois=4):
    """Summarize Fiedler vector into interpretable ROI-pair loadings.

    The Fiedler vector on the bipartite graph encodes which P1-ROIs
    couple most strongly with which P2-ROIs. This function extracts
    the dominant coupling pattern.

    Args:
        fiedler: (T, 2*n_rois) Fiedler vectors over time.
        n_rois: number of ROIs per participant.

    Returns:
        p1_loadings: (T, n_rois) P1 ROI loadings (mean abs).
        p2_loadings: (T, n_rois) P2 ROI loadings (mean abs).
        coupling_matrix: (n_rois, n_rois) outer product of mean |loadings|.
    """
    p1 = fiedler[:, :n_rois]
    p2 = fiedler[:, n_rois:]
    p1_mean = np.abs(p1).mean(axis=0)
    p2_mean = np.abs(p2).mean(axis=0)
    coupling_matrix = np.outer(p1_mean, p2_mean)
    coupling_matrix /= coupling_matrix.max() + 1e-12
    return np.abs(p1), np.abs(p2), coupling_matrix


# ═════════════════════════════════════════════════════════════════════
#  Application 2: AU Spectral Clustering
# ═════════════════════════════════════════════════════════════════════

def au_similarity_graph(bl_p1, bl_p2, method='correlation', n_au=52):
    """Build AU similarity graph from blendshape timeseries.

    Computes pairwise similarity pooled across both participants.
    The graph captures which AUs co-activate.

    Args:
        bl_p1: (T, n_ch) participant 1 blendshapes.
        bl_p2: (T, n_ch) participant 2 blendshapes.
        method: 'correlation' (absolute Pearson r).
        n_au: number of AU channels to use (default 52, ignores activity ch).

    Returns:
        W: (n_au, n_au) symmetric non-negative similarity matrix.
        active_mask: (n_au,) bool — True for AUs with nonzero variance.
    """
    # Take first n_au channels (skip activity channel if present)
    p1 = bl_p1[:, :n_au].copy()
    p2 = bl_p2[:, :n_au].copy()

    # Pool both participants
    combined = np.vstack([p1, p2])  # (2T, n_au)

    # Z-score each AU (identify zero-variance AUs)
    mu = combined.mean(axis=0)
    std = combined.std(axis=0)
    active_mask = std > 1e-8

    z = np.zeros_like(combined)
    z[:, active_mask] = (combined[:, active_mask] - mu[active_mask]) / std[active_mask]

    if method == 'correlation':
        # Pearson correlation matrix, take absolute value
        W = np.abs(z.T @ z / max(len(z) - 1, 1))
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    np.fill_diagonal(W, 0.0)
    # Zero out rows/cols for inactive AUs
    W[~active_mask, :] = 0.0
    W[:, ~active_mask] = 0.0

    return W, active_mask


def spectral_cluster_aus(W, k_max=12, k_override=None):
    """Spectral clustering of AUs via normalized Laplacian eigenmodes.

    Uses the Ng-Jordan-Weiss algorithm:
    1. Compute normalized Laplacian eigenvectors.
    2. Select k via eigengap heuristic (or override).
    3. Row-normalize the spectral embedding.
    4. K-means on the embedding.

    Args:
        W: (n_au, n_au) non-negative symmetric similarity matrix.
        k_max: maximum clusters to consider.
        k_override: force this many clusters.

    Returns:
        dict with:
            k: selected number of clusters.
            labels: (n_au,) cluster assignments (0-indexed).
            eigvals: (n_au,) Laplacian eigenvalues.
            eigvecs: (n_au, n_au) Laplacian eigenvectors.
            eigengap: consecutive eigenvalue gaps.
            embedding: (n_au, k) row-normalized spectral embedding.
    """
    from sklearn.cluster import KMeans

    n = W.shape[0]
    L = _normalized_laplacian(W)
    eigvals, eigvecs = la.eigh(L)

    if k_override is not None:
        k = k_override
        gaps = np.diff(eigvals[:min(n, k_max + 1)])
    else:
        k, gaps = _eigengap_k(eigvals, k_max=k_max)

    k = min(k, n)

    # Spectral embedding: first k eigenvectors
    embedding = eigvecs[:, :k].copy()

    # Row-normalize (Ng-Jordan-Weiss)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    embedding /= norms

    # K-means
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(embedding)

    return {
        'k': k,
        'labels': labels,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'eigengap': gaps,
        'embedding': embedding,
    }


def format_cluster_report(labels, au_names=None, active_mask=None):
    """Human-readable report of AU cluster assignments.

    Args:
        labels: (n_au,) cluster assignments from spectral_cluster_aus.
        au_names: AU name list (default: MEDIAPIPE_AU_NAMES).
        active_mask: (n_au,) bool — inactive AUs marked separately.

    Returns:
        report: list of (cluster_id, [au_name, ...]) tuples, sorted by size.
    """
    if au_names is None:
        au_names = MEDIAPIPE_AU_NAMES
    n_au = len(labels)

    clusters = {}
    for i in range(n_au):
        if active_mask is not None and not active_mask[i]:
            continue
        cid = int(labels[i])
        if cid not in clusters:
            clusters[cid] = []
        name = au_names[i] if i < len(au_names) else f'AU_{i}'
        clusters[cid].append(name)

    return sorted(clusters.items(), key=lambda x: -len(x[1]))


# ═════════════════════════════════════════════════════════════════════
#  Application 3: Graph Spectral Filtering of Multimodal Features
# ═════════════════════════════════════════════════════════════════════

def build_modality_graph(z_matrix, method='correlation', threshold=0.05):
    """Build a graph over modality channels from temporal relationships.

    Each modality is a node. Edge weight = strength of temporal relationship.
    The graph Laplacian eigenmodes define "modality frequencies":
    low = shared across modalities, high = modality-specific.

    Args:
        z_matrix: (T, D) multimodal feature matrix (D modalities).
        method: 'correlation' (absolute Pearson r).
        threshold: minimum edge weight (sparsifies graph).

    Returns:
        W: (D, D) symmetric non-negative weight matrix.
        L: (D, D) normalized Laplacian.
        eigvals: (D,) graph frequencies (ascending).
        eigvecs: (D, D) graph Fourier basis (columns).
    """
    T, D = z_matrix.shape

    if method == 'correlation':
        mu = z_matrix.mean(axis=0, keepdims=True)
        std = z_matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        z = (z_matrix - mu) / std
        W = np.abs(z.T @ z / max(T - 1, 1))
    else:
        raise ValueError(f"Unknown method: {method}")

    np.fill_diagonal(W, 0.0)
    W[W < threshold] = 0.0  # sparsify

    L = _normalized_laplacian(W)
    eigvals, eigvecs = la.eigh(L)

    return W, L, eigvals, eigvecs


# ── Modality block definitions for edge analysis ─────────────────

MODALITY_BLOCKS = {
    'EEG_ImCoh': [0, 1, 2],
    'EEG_Conc':  [3, 4, 5],
    'EEG_Dyn':   [6, 7, 8],
    'EEG_Asym':  [9, 10, 11],
    'BL':        [12, 13],
    'ECG':       [14, 15],
    'Resp':      [16],
    'Pose':      [17],
}

# Broader grouping for cross-modal edge counting
MODALITY_GROUPS = {
    'EEG':  list(range(12)),
    'BL':   [12, 13],
    'ECG':  [14, 15],
    'Resp': [16],
    'Pose': [17],
}


def _block_of(idx):
    """Return the modality group name for a feature index."""
    for name, indices in MODALITY_GROUPS.items():
        if idx in indices:
            return name
    return 'unknown'


def edge_dynamics(z_matrix, window_s=90, stride_s=15, fs=2.0, threshold=0.05):
    """Track cross-modal edge formation/dissolution over time.

    At each window, builds the modality graph and records:
    - Per-edge weight timeseries for all 153 possible edges
    - Cross-modal edge count (edges between different modality groups)
    - Per-group-pair edge count (e.g., EEG↔BL, EEG↔ECG)

    Args:
        z_matrix: (T, D) multimodal feature matrix.
        window_s, stride_s, fs: windowing parameters.
        threshold: minimum edge weight.

    Returns:
        dict with:
            t_centers: (W,) window center times in seconds.
            edge_weights: (W, D, D) full adjacency matrix per window.
            cross_modal_count: (W,) number of cross-modal edges.
            within_modal_count: (W,) number of within-modal edges.
            group_pair_counts: dict of 'GrpA↔GrpB' → (W,) edge counts.
            group_pair_weights: dict of 'GrpA↔GrpB' → (W,) mean edge weights.
    """
    T, D = z_matrix.shape
    win_samp = int(window_s * fs)
    stride_samp = max(1, int(stride_s * fs))
    starts = np.arange(0, T - win_samp + 1, stride_samp)
    n_windows = len(starts)

    t_centers = (starts + win_samp / 2) / fs
    edge_weights = np.zeros((n_windows, D, D))
    cross_count = np.zeros(n_windows, dtype=int)
    within_count = np.zeros(n_windows, dtype=int)

    # Enumerate group pairs
    group_names = list(MODALITY_GROUPS.keys())
    group_pairs = []
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i:]:
            group_pairs.append(f'{g1}↔{g2}')
    gp_counts = {gp: np.zeros(n_windows, dtype=int) for gp in group_pairs}
    gp_weights = {gp: np.zeros(n_windows) for gp in group_pairs}

    for wi, s in enumerate(starts):
        chunk = z_matrix[s:s + win_samp]
        mu = chunk.mean(axis=0, keepdims=True)
        std = chunk.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        z = (chunk - mu) / std
        W = np.abs(z.T @ z / max(len(z) - 1, 1))
        np.fill_diagonal(W, 0.0)
        W[W < threshold] = 0.0

        edge_weights[wi] = W

        # Count edges by type
        for i in range(D):
            gi = _block_of(i)
            for j in range(i + 1, D):
                if W[i, j] > 0:
                    gj = _block_of(j)
                    if gi == gj:
                        within_count[wi] += 1
                    else:
                        cross_count[wi] += 1
                    # Group pair key (alphabetical order)
                    key = f'{min(gi, gj)}↔{max(gi, gj)}'
                    if key in gp_counts:
                        gp_counts[key][wi] += 1
                        gp_weights[key][wi] += W[i, j]

        # Normalize weights to mean
        for gp in group_pairs:
            if gp_counts[gp][wi] > 0:
                gp_weights[gp][wi] /= gp_counts[gp][wi]

    return {
        't_centers': t_centers,
        'edge_weights': edge_weights,
        'cross_modal_count': cross_count,
        'within_modal_count': within_count,
        'group_pair_counts': gp_counts,
        'group_pair_weights': gp_weights,
        'group_pairs': group_pairs,
    }


def per_condition_coupling_test(z_matrix, t_common, segments,
                                 n_perms=500, threshold=0.05, seed=42):
    """Run coupling-shuffle permutation test within each condition window.

    Tests whether the observed flexibility within each condition depends
    on the temporal alignment of cross-participant features.

    Args:
        z_matrix: (T, D) multimodal feature matrix.
        t_common: (T,) time axis.
        segments: list of (name, t_start, t_end) tuples.
        n_perms: permutations per condition.
        threshold: graph edge threshold.
        seed: random seed.

    Returns:
        dict mapping condition_name → {
            real_flex, null_mean, null_std,
            p_low, p_high, cohens_d
        }
    """
    results = {}
    for seg_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        n_pts = mask.sum()
        if n_pts < 60:  # need at least 30s for stable graph
            continue

        z_cond = z_matrix[mask]
        sr = coupling_shuffle_null(z_cond, n_perms=n_perms,
                                    threshold=threshold, seed=seed)

        # Cohen's d: (real - null_mean) / null_std
        d = (sr['real_flex_mean'] - np.mean(sr['null_flex_mean'])) / max(
            np.std(sr['null_flex_mean']), 1e-8)

        results[seg_name] = {
            'real_flex': sr['real_flex_mean'],
            'null_mean': float(np.mean(sr['null_flex_mean'])),
            'null_std': float(np.std(sr['null_flex_mean'])),
            'p_low': sr['p_value_low'],
            'p_high': sr['p_value_high'],
            'p_two_sided': sr['p_value_two_sided'],
            'cohens_d': float(d),
            'n_timepoints': int(n_pts),
        }

    return results


def graph_fourier_transform(signal, eigvecs):
    """Forward Graph Fourier Transform: project signal onto graph eigenbasis.

    Args:
        signal: (T, D) time × modality signal.
        eigvecs: (D, D) columns = graph Fourier modes.

    Returns:
        coeffs: (T, D) graph spectral coefficients.
    """
    return signal @ eigvecs


def inverse_graph_fourier_transform(coeffs, eigvecs):
    """Inverse Graph Fourier Transform: reconstruct signal from coefficients.

    Args:
        coeffs: (T, D) graph spectral coefficients.
        eigvecs: (D, D) graph Fourier basis.

    Returns:
        signal: (T, D) reconstructed signal.
    """
    return coeffs @ eigvecs.T


def graph_spectral_filter(signal, eigvals, eigvecs, filter_fn):
    """Apply spectral filter on the modality graph.

    Computes:  f_out = Φ · diag(h(λ)) · Φᵀ · f_in

    Args:
        signal: (T, D) multimodal signal.
        eigvals: (D,) graph frequencies.
        eigvecs: (D, D) graph Fourier basis.
        filter_fn: callable, λ → h(λ) filter response.

    Returns:
        filtered: (T, D) filtered signal.
    """
    h = np.array([filter_fn(lam) for lam in eigvals])
    coeffs = signal @ eigvecs
    return (coeffs * h[np.newaxis, :]) @ eigvecs.T


def graph_lowpass(signal, eigvals, eigvecs, cutoff=None):
    """Low-pass graph filter: retains cross-modal (smooth) components.

    All modalities moving together → preserved.
    Modality-specific variation → removed.

    Args:
        signal: (T, D) multimodal signal.
        eigvals, eigvecs: graph spectrum.
        cutoff: eigenvalue threshold (default: median).

    Returns:
        filtered: (T, D) low-pass filtered signal.
    """
    if cutoff is None:
        cutoff = float(np.median(eigvals[eigvals > 1e-8]))  # skip zero eigenvalues
    return graph_spectral_filter(signal, eigvals, eigvecs,
                                 lambda lam: 1.0 if lam <= cutoff else 0.0)


def graph_highpass(signal, eigvals, eigvecs, cutoff=None):
    """High-pass graph filter: retains modality-specific components.

    Args:
        signal: (T, D) multimodal signal.
        eigvals, eigvecs: graph spectrum.
        cutoff: eigenvalue threshold (default: median).

    Returns:
        filtered: (T, D) high-pass filtered signal.
    """
    if cutoff is None:
        cutoff = float(np.median(eigvals[eigvals > 1e-8]))
    return graph_spectral_filter(signal, eigvals, eigvecs,
                                 lambda lam: 0.0 if lam <= cutoff else 1.0)


def graph_heat_kernel(signal, eigvals, eigvecs, tau=1.0):
    """Heat diffusion kernel: h(λ) = exp(-τλ). Smooth low-pass.

    Larger tau → stronger smoothing across modalities.

    Args:
        signal: (T, D) multimodal signal.
        eigvals, eigvecs: graph spectrum.
        tau: diffusion time (larger = more smoothing).

    Returns:
        filtered: (T, D) heat-kernel smoothed signal.
    """
    return graph_spectral_filter(signal, eigvals, eigvecs,
                                 lambda lam: np.exp(-tau * lam))


def coupling_flexibility_index(signal, eigvals, eigvecs, cutoff=None,
                               smooth_s=0, fs=2.0):
    """Coupling flexibility: high-frequency graph energy / total energy.

    Measures how "modality-specific" the signal is at each time point.
    Low flexibility → all modalities in sync (rigid coupling).
    High flexibility → modalities diverge (flexible/independent).

    Excludes the DC component (λ₁=0, constant eigenvector) since it
    captures the global mean, not coupling structure.

    Args:
        signal: (T, D) multimodal features.
        eigvals: (D,) graph frequencies.
        eigvecs: (D, D) graph Fourier basis.
        cutoff: graph frequency cutoff (default: median of nonzero λ).
        smooth_s: optional Gaussian smoothing of output (seconds, 0=none).
        fs: sampling rate (for smoothing).

    Returns:
        flexibility: (T,) in [0, 1], per-timepoint flexibility index.
        energy_low: (T,) energy in low graph frequencies.
        energy_high: (T,) energy in high graph frequencies.
    """
    if cutoff is None:
        nonzero = eigvals[eigvals > 1e-8]
        cutoff = float(np.median(nonzero)) if len(nonzero) > 0 else 0.5

    coeffs = signal @ eigvecs  # (T, D) graph spectral coefficients
    energy = coeffs ** 2

    # Exclude DC component (index 0, λ≈0)
    low_mask = (eigvals > 1e-8) & (eigvals <= cutoff)
    high_mask = eigvals > cutoff

    energy_low = energy[:, low_mask].sum(axis=1)
    energy_high = energy[:, high_mask].sum(axis=1)
    total = energy_low + energy_high + 1e-12

    flexibility = energy_high / total

    if smooth_s > 0 and fs > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma = smooth_s * fs
        flexibility = gaussian_filter1d(flexibility, sigma)
        energy_low = gaussian_filter1d(energy_low, sigma)
        energy_high = gaussian_filter1d(energy_high, sigma)

    return flexibility, energy_low, energy_high


def build_modality_graph_windowed(z_matrix, window_s=90, stride_s=15,
                                  fs=2.0, threshold=0.05):
    """Time-varying modality graph from sliding-window correlations.

    At each window, builds a new graph from local correlations. Tracks
    how the modality network structure evolves over the session.

    Args:
        z_matrix: (T, D) multimodal feature matrix.
        window_s: window length in seconds.
        stride_s: stride between windows in seconds.
        fs: sampling rate.
        threshold: minimum edge weight.

    Returns:
        dict with:
            t_centers: (W,) window center times (in samples / fs).
            lambda2: (W,) algebraic connectivity per window.
            n_components: (W,) connected components per window.
            n_edges: (W,) number of edges above threshold per window.
            flexibility: (W,) coupling flexibility per window.
            mean_weight: (W,) mean edge weight per window.
            spectrum: (W, D) full eigenvalue spectrum per window.
    """
    T, D = z_matrix.shape
    win_samp = int(window_s * fs)
    stride_samp = max(1, int(stride_s * fs))

    starts = np.arange(0, T - win_samp + 1, stride_samp)
    n_windows = len(starts)

    t_centers = (starts + win_samp / 2) / fs
    lam2 = np.empty(n_windows)
    n_comp = np.empty(n_windows, dtype=int)
    n_edges_arr = np.empty(n_windows, dtype=int)
    flex_arr = np.empty(n_windows)
    mean_w = np.empty(n_windows)
    spectra = np.empty((n_windows, D))

    for wi, s in enumerate(starts):
        chunk = z_matrix[s:s + win_samp]

        # Build local graph
        mu = chunk.mean(axis=0, keepdims=True)
        std = chunk.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        z = (chunk - mu) / std
        W = np.abs(z.T @ z / max(len(z) - 1, 1))
        np.fill_diagonal(W, 0.0)
        W[W < threshold] = 0.0

        L = _normalized_laplacian(W)
        eigvals, eigvecs = la.eigh(L)

        spectra[wi] = eigvals
        nc = int((eigvals < 1e-6).sum())
        n_comp[wi] = nc
        lam2[wi] = eigvals[nc] if nc < D else 0.0

        # Edge statistics
        upper = W[np.triu_indices(D, k=1)]
        n_edges_arr[wi] = int((upper > 0).sum())
        mean_w[wi] = float(upper[upper > 0].mean()) if n_edges_arr[wi] > 0 else 0.0

        # Local flexibility (using this window's graph)
        nonzero_eig = eigvals[eigvals > 1e-8]
        cutoff = float(np.median(nonzero_eig)) if len(nonzero_eig) > 0 else 0.5
        coeffs = chunk @ eigvecs
        energy = (coeffs ** 2).mean(axis=0)  # time-averaged per graph freq
        low_mask = (eigvals > 1e-8) & (eigvals <= cutoff)
        high_mask = eigvals > cutoff
        e_low = energy[low_mask].sum()
        e_high = energy[high_mask].sum()
        flex_arr[wi] = e_high / (e_low + e_high + 1e-12)

    return {
        't_centers': t_centers,
        'lambda2': lam2,
        'n_components': n_comp,
        'n_edges': n_edges_arr,
        'flexibility': flex_arr,
        'mean_weight': mean_w,
        'spectrum': spectra,
    }


def graph_spectral_summary(eigvals, eigvecs, modality_names=None):
    """Interpretive summary of the modality graph spectrum.

    Args:
        eigvals: (D,) graph frequencies.
        eigvecs: (D, D) graph Fourier basis.
        modality_names: list of modality labels.

    Returns:
        dict with:
            n_components: number of near-zero eigenvalues (connected components).
            spectral_gap: λ₂ (algebraic connectivity of modality graph).
            effective_dim: number of eigenvalues < 0.5 (low-frequency modes).
            modes: list of dicts describing top eigenmodes.
    """
    n_components = int((eigvals < 1e-6).sum())
    spectral_gap = float(eigvals[n_components]) if n_components < len(eigvals) else 0.0

    effective_dim = int((eigvals < 0.5).sum())

    modes = []
    for i in range(min(5, len(eigvals))):
        v = eigvecs[:, i]
        top_idx = np.argsort(np.abs(v))[::-1][:5]
        mode_info = {
            'index': i,
            'eigenvalue': float(eigvals[i]),
            'top_modalities': [],
        }
        for idx in top_idx:
            name = modality_names[idx] if modality_names and idx < len(modality_names) else f'mod_{idx}'
            mode_info['top_modalities'].append((name, float(v[idx])))
        modes.append(mode_info)

    return {
        'n_components': n_components,
        'spectral_gap': spectral_gap,
        'effective_dim': effective_dim,
        'modes': modes,
    }


# ═════════════════════════════════════════════════════════════════════
#  Permutation Tests for Coupling Flexibility
# ═════════════════════════════════════════════════════════════════════

# Cross-participant feature indices in the 18D MODALITY_KEYS order
# These features are computed from both P1 and P2 together:
#   ImCoh (0-2), BL expression coherence (12), ECG cross-product (14-15),
#   Respiratory phase coherence (16), Pose velocity cross-product (17)
CROSS_PARTICIPANT_IDX = [0, 1, 2, 12, 14, 15, 16, 17]

# Individual-combined features (computed per-participant, then averaged/differenced):
#   Concordance (3-5), Dynamics (6-8), Asymmetry (9-11), BL activity conc (13)
INDIVIDUAL_COMBINED_IDX = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13]


def coupling_shuffle_null(z_matrix, n_perms=1000, threshold=0.05, seed=42):
    """Coupling-shuffle permutation test for flexibility.

    Circular-shifts the cross-participant features independently to destroy
    the temporal alignment between coupling and individual-state features,
    while preserving each feature's autocorrelation and marginal distribution.

    Recomputes the modality graph and flexibility from each permuted matrix.
    Returns the null distribution of mean flexibility.

    Args:
        z_matrix: (T, D) multimodal feature matrix (prewhitened).
        n_perms: number of permutations.
        threshold: graph edge threshold.
        seed: random seed.

    Returns:
        dict with:
            null_flex_mean: (n_perms,) null distribution of session-mean flexibility.
            null_flex_std: (n_perms,) null distribution of flexibility std.
            real_flex_mean: float — observed session-mean flexibility.
            real_flex_std: float — observed flexibility std.
            p_value: float — fraction of null ≤ real (one-sided: real < null).
            p_value_two_sided: float — fraction of null more extreme.
    """
    T, D = z_matrix.shape
    rng = np.random.default_rng(seed)

    # Compute real flexibility
    W, L, eigvals, eigvecs = build_modality_graph(z_matrix, threshold=threshold)
    real_flex, _, _ = coupling_flexibility_index(z_matrix, eigvals, eigvecs)
    real_mean = float(real_flex.mean())
    real_std = float(real_flex.std())

    null_means = np.empty(n_perms)
    null_stds = np.empty(n_perms)

    cross_idx = [i for i in CROSS_PARTICIPANT_IDX if i < D]

    for pi in range(n_perms):
        z_perm = z_matrix.copy()
        # Circular-shift each cross-participant feature independently
        for ci in cross_idx:
            shift = rng.integers(1, T)
            z_perm[:, ci] = np.roll(z_perm[:, ci], shift)

        # Rebuild graph from permuted data
        W_p, _, eigvals_p, eigvecs_p = build_modality_graph(
            z_perm, threshold=threshold)
        flex_p, _, _ = coupling_flexibility_index(z_perm, eigvals_p, eigvecs_p)
        null_means[pi] = flex_p.mean()
        null_stds[pi] = flex_p.std()

    # p-value: how often is null flexibility ≤ real? (tests if real is unusually low)
    p_low = float((null_means <= real_mean).mean())
    p_high = float((null_means >= real_mean).mean())
    p_two = 2 * min(p_low, p_high)

    return {
        'null_flex_mean': null_means,
        'null_flex_std': null_stds,
        'real_flex_mean': real_mean,
        'real_flex_std': real_std,
        'p_value_low': p_low,
        'p_value_high': p_high,
        'p_value_two_sided': min(p_two, 1.0),
    }


def protocol_label_permutation(session_flex, session_protocols,
                                group_a='meditation', group_b='PE',
                                exhaustive=True, n_perms=10000, seed=42):
    """Protocol-label permutation test for between-group flexibility.

    Permutes which sessions get labeled as group_a vs group_b, and
    recomputes the test statistic (difference in means) for each permutation.

    Args:
        session_flex: dict mapping session_name → flexibility value.
        session_protocols: dict mapping session_name → protocol string.
        group_a, group_b: protocol labels to compare.
        exhaustive: if True, enumerate all C(n, n_a) permutations.
        n_perms: number of random permutations (if not exhaustive).
        seed: random seed.

    Returns:
        dict with:
            observed_diff: float — mean(group_a) - mean(group_b).
            null_diffs: array — null distribution of differences.
            p_value: float — two-sided p.
            n_a, n_b: group sizes.
            n_perms_used: int.
    """
    from itertools import combinations

    # Filter to sessions with valid flex and known protocol
    sessions_a = [(s, f) for s, f in session_flex.items()
                  if session_protocols.get(s) == group_a and np.isfinite(f)]
    sessions_b = [(s, f) for s, f in session_flex.items()
                  if session_protocols.get(s) == group_b and np.isfinite(f)]

    n_a, n_b = len(sessions_a), len(sessions_b)
    if n_a < 2 or n_b < 2:
        return {'error': f'Too few sessions: {group_a}={n_a}, {group_b}={n_b}'}

    all_flex = np.array([f for _, f in sessions_a] + [f for _, f in sessions_b])
    n_total = len(all_flex)

    obs_diff = float(all_flex[:n_a].mean() - all_flex[n_a:].mean())

    if exhaustive and n_total <= 20:
        # Enumerate all C(n_total, n_a) permutations
        all_combos = list(combinations(range(n_total), n_a))
        null_diffs = np.empty(len(all_combos))
        for ci, combo in enumerate(all_combos):
            mask_a = np.array(combo)
            mask_b = np.array([i for i in range(n_total) if i not in combo])
            null_diffs[ci] = all_flex[mask_a].mean() - all_flex[mask_b].mean()
        n_used = len(all_combos)
    else:
        rng = np.random.default_rng(seed)
        null_diffs = np.empty(n_perms)
        for pi in range(n_perms):
            perm = rng.permutation(n_total)
            null_diffs[pi] = all_flex[perm[:n_a]].mean() - all_flex[perm[n_a:]].mean()
        n_used = n_perms

    # Two-sided p-value
    p_val = float((np.abs(null_diffs) >= np.abs(obs_diff)).mean())

    return {
        'observed_diff': obs_diff,
        'null_diffs': null_diffs,
        'p_value': p_val,
        'n_a': n_a,
        'n_b': n_b,
        'n_perms_used': n_used,
        'group_a_mean': float(all_flex[:n_a].mean()),
        'group_b_mean': float(all_flex[n_a:].mean()),
    }


# ═════════════════════════════════════════════════════════════════════
#  V10 Extensions: Modularity, Centrality, Change-Points
# ═════════════════════════════════════════════════════════════════════

MODALITY_BLOCKS_V10 = {
    'EEG_ImCoh': [0, 1, 2],
    'EEG_Conc':  [3, 4, 5],
    'EEG_Dyn':   [6, 7, 8],
    'EEG_Asym':  [9, 10, 11],
    'BL':        [12, 13],
    'ECG':       [14, 15],
    'Resp':      [16],
    'Pose':      [17],
    'LZ_Conc':   [18, 19],
    'LZ_Asym':   [20, 21],
    'Graph':     [22],
}

MODALITY_GROUPS_V10 = {
    'EEG':    list(range(12)),
    'BL':     [12, 13],
    'ECG':    [14, 15],
    'Resp':   [16],
    'Pose':   [17],
    'LZ':     [18, 19, 20, 21],
    'Graph':  [22],
}


def _build_windowed_graph(z_matrix, window_s, stride_s, fs, threshold):
    """Shared helper: build correlation-based adjacency per window.

    Returns:
        starts: (W,) start indices.
        t_centers: (W,) center times in seconds.
        adjacencies: list of (D, D) adjacency matrices.
    """
    T, D = z_matrix.shape
    win_samp = int(window_s * fs)
    stride_samp = max(1, int(stride_s * fs))
    starts = np.arange(0, T - win_samp + 1, stride_samp)
    t_centers = (starts + win_samp / 2) / fs
    adjacencies = []

    for s in starts:
        chunk = z_matrix[s:s + win_samp]
        mu = chunk.mean(axis=0, keepdims=True)
        std = chunk.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        z = (chunk - mu) / std
        W = np.abs(z.T @ z / max(len(z) - 1, 1))
        np.fill_diagonal(W, 0.0)
        W[W < threshold] = 0.0
        adjacencies.append(W)

    return starts, t_centers, adjacencies


def graph_modularity_windowed(z_matrix, window_s=90, stride_s=15,
                               fs=2.0, threshold=0.05):
    """Louvain modularity Q on windowed modality graph.

    Uses python-louvain (community.best_partition) on the correlation-based
    adjacency graph per window.

    Args:
        z_matrix: (T, D) multimodal feature matrix (base 18D).
        window_s: window length in seconds.
        stride_s: stride between windows in seconds.
        fs: sampling rate.
        threshold: minimum edge weight.

    Returns:
        dict with:
            t_centers: (W,) window center times.
            modularity_Q: (W,) modularity score per window.
            n_communities: (W,) number of communities per window.
            community_labels: list of (W,) per-window partition dicts.
    """
    starts, t_centers, adjacencies = _build_windowed_graph(
        z_matrix, window_s, stride_s, fs, threshold)
    n_windows = len(starts)

    mod_Q = np.empty(n_windows)
    n_comm = np.empty(n_windows, dtype=int)
    partitions = []

    for wi, W in enumerate(adjacencies):
        G = nx.from_numpy_array(W)
        if G.number_of_edges() == 0:
            mod_Q[wi] = 0.0
            n_comm[wi] = W.shape[0]
            partitions.append({i: i for i in range(W.shape[0])})
            continue

        partition = community_louvain.best_partition(G, random_state=42)
        Q = community_louvain.modularity(partition, G)
        mod_Q[wi] = Q
        n_comm[wi] = len(set(partition.values()))
        partitions.append(partition)

    return {
        't_centers': t_centers,
        'modularity_Q': mod_Q,
        'n_communities': n_comm,
        'community_labels': partitions,
    }


def eigenvector_centrality_windowed(z_matrix, window_s=90, stride_s=15,
                                     fs=2.0, threshold=0.05,
                                     eeg_channels=None):
    """Per-window eigenvector centrality for each modality channel.

    Eigenvector centrality = the leading eigenvector of the adjacency matrix
    (corresponding to largest eigenvalue). Hub modalities have high centrality.

    Args:
        z_matrix: (T, D) multimodal feature matrix (base 18D).
        window_s: window length in seconds.
        stride_s: stride between windows in seconds.
        fs: sampling rate.
        threshold: minimum edge weight.
        eeg_channels: indices of EEG channels for aggregate centrality.
                      Default: 0-11.

    Returns:
        dict with:
            t_centers: (W,) window center times.
            centrality: (W, D) per-modality centrality.
            eeg_centrality: (W,) mean centrality of EEG channels.
    """
    if eeg_channels is None:
        eeg_channels = list(range(12))

    starts, t_centers, adjacencies = _build_windowed_graph(
        z_matrix, window_s, stride_s, fs, threshold)
    n_windows = len(starts)
    D = z_matrix.shape[1]

    centrality = np.zeros((n_windows, D))
    eeg_cent = np.zeros(n_windows)

    for wi, W in enumerate(adjacencies):
        # Leading eigenvector of W (largest eigenvalue)
        eigvals, eigvecs = la.eigh(W)
        # eigh returns ascending order — last eigenvector is the dominant one
        v1 = np.abs(eigvecs[:, -1])
        # Normalize to sum to 1
        v_sum = v1.sum()
        if v_sum > 1e-10:
            v1 /= v_sum

        centrality[wi] = v1
        valid_eeg = [c for c in eeg_channels if c < D]
        if valid_eeg:
            eeg_cent[wi] = v1[valid_eeg].mean()

    return {
        't_centers': t_centers,
        'centrality': centrality,
        'eeg_centrality': eeg_cent,
    }


def graph_changepoint_score(modularity_ts, lambda2_ts, n_edges_ts,
                             sigma_s=5.0, fs_graph=None, stride_s=15.0):
    """Change-point score from graph topology timeseries.

    Combines temporal derivatives of modularity, λ₂, and edge count
    into a single change-point indicator. High scores indicate
    topological transitions (e.g., condition boundaries).

    Args:
        modularity_ts: (W,) modularity Q per window.
        lambda2_ts: (W,) algebraic connectivity per window.
        n_edges_ts: (W,) edge count per window.
        sigma_s: Gaussian smoothing width in seconds before differentiation.
        fs_graph: sampling rate of graph timeseries (1/stride_s if None).
        stride_s: stride between windows in seconds (used if fs_graph is None).

    Returns:
        (W,) change-point score (higher = more topological change).
    """
    if fs_graph is None:
        fs_graph = 1.0 / stride_s

    sigma_samp = max(1, int(sigma_s * fs_graph))

    scores = np.zeros(len(modularity_ts))
    for ts in [modularity_ts, lambda2_ts, n_edges_ts.astype(np.float64)]:
        if len(ts) < 3:
            continue
        smoothed = gaussian_filter1d(ts, sigma=sigma_samp)
        deriv = np.gradient(smoothed)
        std = deriv.std()
        if std > 1e-10:
            deriv /= std
        scores += np.abs(deriv)

    return scores
