import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module5_block_pca import fit_block_pca, run_block_pca
from diagnostics.shared.data_loader import load_session


def test_fit_block_pca_reduces_dim(rng):
    keys = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    groups = {'eeg': ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'], 'other': ['bl_expr', 'pose']}
    Y = rng.standard_normal((200, 5))
    Y_red, loadings = fit_block_pca(Y, keys, groups, flags={'slow_drift': [], 'high_vif': []})
    assert Y_red.shape[0] == 200
    assert Y_red.shape[1] <= 5  # reduced dim
    assert 'eeg_PC1' in loadings


def test_fit_block_pca_excludes_flagged(rng):
    keys = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
    groups = {'eeg': ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'], 'other': ['bl_expr', 'pose']}
    Y = rng.standard_normal((200, 5))
    flags = {'slow_drift': ['imcoh_theta'], 'high_vif': []}
    Y_red, loadings = fit_block_pca(Y, keys, groups, flags)
    # imcoh_theta excluded so eeg group has 2 channels → max 1 PC at 80%
    assert Y_red.shape[1] < 5


def test_run_block_pca_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm5_out')
    flags = {'slow_drift': [], 'high_vif': [], 'collinear_pairs': [], 'low_info': []}
    result = run_block_pca(sessions, flags, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'block_pca_loadings.md'))
    assert os.path.exists(os.path.join(out_dir, 'module5_report.md'))
    assert 'n_reduced_dims' in result
