import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module4_model_comparison import run_model_comparison
from diagnostics.shared.data_loader import load_session


def test_run_model_comparison_outputs(fake_session_dir, tmp_path):
    """With only 1 session LOO is trivial but output files should still be created."""
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm4_out')
    result = run_model_comparison(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'loo_cv_results.csv'))
    assert os.path.exists(os.path.join(out_dir, 'model_comparison_summary.png'))
    assert os.path.exists(os.path.join(out_dir, 'module4_report.md'))
    assert 'mean_ll_per_frame_dim' in result
