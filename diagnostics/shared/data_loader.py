from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SessionData:
    name: str
    t_common: np.ndarray
    Y_raw: np.ndarray
    Y_pw: np.ndarray
    U: np.ndarray
    obs_mask: np.ndarray
    modality_keys: List[str]
    covariate_keys: List[str]
    segments: List[Tuple[str, float, float]]
    fs: float = 2.0
    gamma: Optional[np.ndarray] = None
    path_constrained: Optional[np.ndarray] = None
    path_unconstrained: Optional[np.ndarray] = None
    d_emit: Optional[np.ndarray] = None
    C_emit: Optional[np.ndarray] = None
    R_emit: Optional[np.ndarray] = None
    W_trans: Optional[np.ndarray] = None
    S_trans: Optional[np.ndarray] = None
    state_labels: Optional[List[str]] = None


def load_session(session_name: str, results_dir: str = 'results/v11') -> SessionData:
    sess_dir = os.path.join(results_dir, session_name)
    scaffold_npz = os.path.join(sess_dir, 'scaffold_v11_ztimecourses.npz')
    scaffold_json = os.path.join(sess_dir, 'scaffold_v11_results.json')

    npz = np.load(scaffold_npz, allow_pickle=False)
    with open(scaffold_json) as f:
        meta = json.load(f)

    mod_keys = meta['modality_keys']
    cov_keys = meta.get('covariate_keys', [])
    fs = float(meta.get('fs_out', 2.0))

    Y_raw = np.column_stack([npz[f'z_raw_{k}'] for k in mod_keys])
    Y_pw = np.column_stack([npz[f'z_{k}'] for k in mod_keys])
    U = npz['U_covariates'] if 'U_covariates' in npz else np.zeros((len(npz['t_common']), len(cov_keys)))
    obs_mask = npz['obs_mask']
    t_common = npz['t_common']
    segments = [(s[0], float(s[1]), float(s[2])) for s in meta.get('segments', [])]

    rslds_path = os.path.join(sess_dir, 'v11_rslds_results.npz')
    gamma = path_con = path_uncon = d_emit = C_emit = R_emit = W_trans = S_trans = state_labels = None

    if os.path.exists(rslds_path):
        r = np.load(rslds_path, allow_pickle=True)
        gamma = r['gamma']
        path_con = r['path']
        path_uncon = np.argmax(gamma, axis=1).astype(np.int32)
        d_emit = r['d_emit'] if 'd_emit' in r else None
        C_emit = r['C_emit'] if 'C_emit' in r else None
        R_emit = r['R_emit'] if 'R_emit' in r else None
        W_trans = r['W_trans'] if 'W_trans' in r else None
        S_trans = r['S_trans'] if 'S_trans' in r else None
        sl = r['state_labels'] if 'state_labels' in r else None
        state_labels = list(sl) if sl is not None else None

    return SessionData(
        name=session_name, t_common=t_common,
        Y_raw=Y_raw, Y_pw=Y_pw, U=U, obs_mask=obs_mask,
        modality_keys=mod_keys, covariate_keys=cov_keys,
        segments=segments, fs=fs,
        gamma=gamma, path_constrained=path_con, path_unconstrained=path_uncon,
        d_emit=d_emit, C_emit=C_emit, R_emit=R_emit,
        W_trans=W_trans, S_trans=S_trans, state_labels=state_labels,
    )


def discover_session_names(results_dir: str = 'results/v11') -> List[str]:
    names = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if entry.is_dir():
            scaffold = os.path.join(entry.path, 'scaffold_v11_ztimecourses.npz')
            if os.path.exists(scaffold):
                names.append(entry.name)
    return names


def load_all_sessions(results_dir: str = 'results/v11') -> List[SessionData]:
    return [load_session(n, results_dir) for n in discover_session_names(results_dir)]
