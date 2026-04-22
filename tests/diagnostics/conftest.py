import numpy as np
import json
import pytest

N, D, K, D_COV = 120, 5, 4, 3
FS = 2.0
MOD_KEYS = ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta', 'bl_expr', 'pose']
COV_KEYS = ['z_slow_pc1', 'z_slow_pc2', 'flex']
SEGMENTS = [('base_EO', 0.0, 30.0), ('conv_1', 30.0, 90.0), ('meditate_B', 90.0, 60.0)]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fake_session_dir(tmp_path, rng):
    """Write minimal scaffold + rSLDS files for one session."""
    sess_dir = tmp_path / 'fake_sess'
    sess_dir.mkdir()

    Y_raw = rng.standard_normal((N, D)).astype(np.float32)
    Y_pw = rng.standard_normal((N, D)).astype(np.float32)
    obs_mask = np.ones((N, D), dtype=bool)
    U = rng.standard_normal((N, D_COV)).astype(np.float32)
    t_common = np.arange(N, dtype=np.float32) / FS

    save_dict = {'t_common': t_common, 'obs_mask': obs_mask, 'U_covariates': U}
    for i, key in enumerate(MOD_KEYS):
        save_dict[f'z_{key}'] = Y_pw[:, i]
        save_dict[f'z_raw_{key}'] = Y_raw[:, i]
    for i, key in enumerate(COV_KEYS):
        save_dict[f'u_{key}'] = U[:, i]

    np.savez_compressed(str(sess_dir / 'scaffold_v11_ztimecourses.npz'), **save_dict)

    meta = {
        'session': 'fake_sess', 'version': 'v11',
        'n_timepoints': N, 'duration_s': float(N / FS),
        'fs_out': FS,
        'modality_keys': MOD_KEYS,
        'covariate_keys': COV_KEYS,
        'segments': [[s, t0, t1] for s, t0, t1 in SEGMENTS],
    }
    with open(str(sess_dir / 'scaffold_v11_results.json'), 'w') as f:
        json.dump(meta, f)

    # rSLDS results
    gamma = np.abs(rng.standard_normal((N, K)))
    gamma /= gamma.sum(axis=1, keepdims=True)
    path = np.argmax(gamma, axis=1).astype(np.int32)
    d_emit = rng.standard_normal((K, D)).astype(np.float32)
    W_trans = rng.standard_normal((K, K)).astype(np.float32)
    S_trans = rng.standard_normal((K, K, D_COV)).astype(np.float32)
    state_labels = np.array(['NULL', 'COUP', 'SHARED', 'OTHER'], dtype=object)

    np.savez_compressed(str(sess_dir / 'v11_rslds_results.npz'),
                        gamma=gamma, path=path, t_common=t_common,
                        d_emit=d_emit, W_trans=W_trans, S_trans=S_trans,
                        state_labels=state_labels)
    return tmp_path, 'fake_sess'


@pytest.fixture
def fake_scaffold_only_dir(tmp_path, rng):
    """Scaffold only — no rSLDS fit."""
    sess_dir = tmp_path / 'no_rslds'
    sess_dir.mkdir()
    Y_raw = rng.standard_normal((N, D)).astype(np.float32)
    Y_pw = rng.standard_normal((N, D)).astype(np.float32)
    obs_mask = np.ones((N, D), dtype=bool)
    U = rng.standard_normal((N, D_COV)).astype(np.float32)
    t_common = np.arange(N, dtype=np.float32) / FS
    save_dict = {'t_common': t_common, 'obs_mask': obs_mask, 'U_covariates': U}
    for i, key in enumerate(MOD_KEYS):
        save_dict[f'z_{key}'] = Y_pw[:, i]
        save_dict[f'z_raw_{key}'] = Y_raw[:, i]
    np.savez_compressed(str(sess_dir / 'scaffold_v11_ztimecourses.npz'), **save_dict)
    meta = {'session': 'no_rslds', 'version': 'v11',
            'n_timepoints': N, 'duration_s': float(N / FS), 'fs_out': FS,
            'modality_keys': MOD_KEYS, 'covariate_keys': COV_KEYS,
            'segments': [[s, t0, t1] for s, t0, t1 in SEGMENTS]}
    with open(str(sess_dir / 'scaffold_v11_results.json'), 'w') as f:
        json.dump(meta, f)
    return tmp_path, 'no_rslds'
