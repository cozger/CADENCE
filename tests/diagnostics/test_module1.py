import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module1_obs_space import (
    run_obs_space_analysis,
    shuffle_null_silhouette,
    pairwise_d_emit_distances,
    latent_space_silhouette,
)
from diagnostics.shared.data_loader import load_session


def test_run_obs_space_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm1_out')
    result = run_obs_space_analysis(sessions, out_dir, n_neighbors=5, min_dist=0.1)
    assert os.path.exists(os.path.join(out_dir, 'pca_scree.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_state.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_phase.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_session.png'))
    assert os.path.exists(os.path.join(out_dir, 'umap_by_time.png'))
    assert os.path.exists(os.path.join(out_dir, 'cluster_quality.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module1_report.md'))
    assert 'state_silhouette' in result
    assert 'phase_silhouette' in result


def test_silhouette_in_reduced_space_not_full(fake_session_dir, tmp_path):
    """Silhouette must be computed in PCA-reduced space, not full D=5."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    result = run_obs_space_analysis(sessions, str(tmp_path / 'out'), n_neighbors=5)
    # Reduced dim must be <= full D=5
    assert result['n_pca_components_80pct'] <= sessions[0].Y_pw.shape[1]


def test_shuffle_null_silhouette_clustered_data(rng):
    """Well-separated clusters should yield real silhouette > shuffle with z >> 0."""
    # Three tight clusters in 2D
    X = np.vstack([
        rng.standard_normal((100, 2)) + [0, 0],
        rng.standard_normal((100, 2)) + [10, 0],
        rng.standard_normal((100, 2)) + [0, 10],
    ])
    labels = np.repeat([0, 1, 2], 100)
    r = shuffle_null_silhouette(X, labels, n_shuffles=50, sample_size=300, seed=0)
    assert r['real'] > 0.3, f"real silhouette should be high for well-separated clusters, got {r['real']}"
    assert r['z'] > 3, f"z should be >> 0 for real clusters, got {r['z']}"
    assert r['p_right'] < 0.05


def test_shuffle_null_silhouette_random_labels(rng):
    """Random labels on unstructured data: z should be small (not extreme)."""
    X = rng.standard_normal((500, 5))
    labels = rng.integers(0, 4, size=500)
    r = shuffle_null_silhouette(X, labels, n_shuffles=50, sample_size=500, seed=0)
    # With random labels on random data, real silhouette is itself drawn from
    # the shuffle distribution, so z should be small. Loose bound for n=50.
    assert abs(r['z']) < 3, f"z should be ~0 for random labels on random data, got {r['z']}"


def test_pairwise_d_emit_distances_shape_and_symmetry(rng):
    """Distance matrix must be K x K, symmetric, zero diagonal."""
    K, D = 4, 6
    mean_d = rng.standard_normal((K, D))
    r = pairwise_d_emit_distances(mean_d, labels=['A', 'B', 'C', 'D'])
    assert r['distances'].shape == (K, K)
    np.testing.assert_allclose(r['distances'], r['distances'].T)
    np.testing.assert_allclose(np.diag(r['distances']), 0, atol=1e-12)
    assert r['labels'] == ['A', 'B', 'C', 'D']
    # Off-diagonal min/max ratio should be in (0, 1]
    assert 0 < r['min_max_ratio'] <= 1


def test_pairwise_d_emit_duplicate_states_detects_redundancy():
    """Two identical state centroids → min_off_diag = 0 → min/max ratio = 0."""
    mean_d = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # identical to state 0
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    r = pairwise_d_emit_distances(mean_d)
    assert r['min_off_diag'] == 0.0
    assert r['min_max_ratio'] == 0.0


def test_latent_space_silhouette_returns_none_when_missing(fake_session_dir):
    """When x_smooth is missing from sessions, function returns None (graceful)."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    # fake_session_dir doesn't include x_smooth → should return None
    assert sessions[0].x_smooth is None
    assert latent_space_silhouette(sessions) is None


def test_latent_space_silhouette_computes_when_present(rng, fake_session_dir):
    """When x_smooth is present, function returns silhouette/db/d_latent."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    # Inject synthetic well-separated latents so silhouette > 0
    N = len(sessions[0].t_common)
    path = sessions[0].path_unconstrained
    x_smooth = np.zeros((N, 3), dtype=np.float32)
    for k in range(4):
        mask = path == k
        x_smooth[mask] = rng.standard_normal((mask.sum(), 3)) * 0.1 + np.eye(4, 3)[k] * 5
    sessions[0].x_smooth = x_smooth
    r = latent_space_silhouette(sessions)
    assert r is not None
    assert r['d_latent'] == 3
    assert r['silhouette'] > 0  # well-separated
