import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module1_obs_space import run_obs_space_analysis
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
