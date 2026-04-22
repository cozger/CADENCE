import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from diagnostics.tier2_deepdive.module6_null_ablation import (
    hungarian_match_states, run_null_ablation
)
from diagnostics.shared.data_loader import load_session


def test_hungarian_match_identity(rng):
    """If constrained and unconstrained d_emit are identical, match should be identity."""
    d = rng.standard_normal((4, 5))
    perm, sims = hungarian_match_states(d, d)
    assert list(perm) == [0, 1, 2, 3]
    np.testing.assert_allclose(sims, np.ones(4), atol=1e-6)


def test_hungarian_match_permuted(rng):
    """Permuted d_emit should be matched back to original order."""
    d = rng.standard_normal((4, 5))
    d_perm = d[[2, 0, 3, 1]]  # permute rows
    perm, sims = hungarian_match_states(d, d_perm)
    # All cosine similarities should be 1.0 (same vectors, just reordered)
    np.testing.assert_allclose(sims, np.ones(4), atol=1e-6)


def test_run_null_ablation_creates_outputs(fake_session_dir, tmp_path):
    results_dir, session_name = fake_session_dir
    sessions = [load_session(session_name, results_dir=str(results_dir))]
    out_dir = str(tmp_path / 'm6_out')
    result = run_null_ablation(sessions, out_dir)
    assert os.path.exists(os.path.join(out_dir, 'emission_means_comparison.png'))
    assert os.path.exists(os.path.join(out_dir, 'hungarian_alignment.csv'))
    assert os.path.exists(os.path.join(out_dir, 'module6_report.md'))
    assert 'constrained_null_norm' in result
