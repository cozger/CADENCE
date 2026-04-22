from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Default modality groups for V11 28-channel scaffold
V11_MODALITY_GROUPS: Dict[str, List[str]] = {
    'EEG_phase':    ['imcoh_theta', 'imcoh_alpha', 'imcoh_beta'],
    'EEG_shared':   ['conc_theta', 'conc_alpha', 'conc_beta'],
    'EEG_dynamics': ['dyn_theta', 'dyn_alpha', 'dyn_beta'],
    'EEG_asymmetry':['asym_theta', 'asym_alpha', 'asym_beta'],
    'Facial':       ['bl_expr', 'bl_act_conc'],
    'Autonomic':    ['ecg_lf', 'ecg_hf', 'resp'],
    'Body':         ['pose'],
    'Complexity':   ['lz_conc_theta', 'lz_conc_alpha', 'lz_asym_theta', 'lz_asym_alpha'],
    'Graph':        ['graph_mod'],
    'Burst_TE':     ['te_conc_theta', 'te_conc_alpha',
                     'burst_coinc_theta', 'burst_coinc_alpha', 'burst_coinc_beta'],
}


def fit_block_pca(
    Y: np.ndarray,
    channel_keys: List[str],
    modality_groups: Dict[str, List[str]],
    flags: dict,
    var_threshold: float = 0.80,
    max_pcs: int = 2,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Fit per-modality PCA; return (Y_reduced, loadings_dict).

    Channels in flags['slow_drift'] or flags['high_vif'] are excluded
    from their group before PCA.
    """
    excluded = set(flags.get('slow_drift', [])) | set(flags.get('high_vif', []))
    key_to_idx = {k: i for i, k in enumerate(channel_keys)}

    blocks = []
    loadings: Dict[str, str] = {}

    for group_name, group_keys in modality_groups.items():
        active = [k for k in group_keys if k in key_to_idx and k not in excluded]
        if not active:
            continue
        idx = [key_to_idx[k] for k in active]
        Xg = Y[:, idx]
        if Xg.shape[1] == 1:
            blocks.append(Xg)
            loadings[f'{group_name}_PC1'] = active[0]
            continue
        pca = PCA().fit(Xg)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = min(max_pcs, int(np.searchsorted(cumvar, var_threshold)) + 1)
        n_keep = max(1, n_keep)
        Xg_r = pca.transform(Xg)[:, :n_keep]
        blocks.append(Xg_r)
        for pc in range(n_keep):
            weights = pca.components_[pc]
            terms = ' + '.join(f'{w:.2f}*{k}' for w, k in
                               sorted(zip(weights, active), key=lambda x: -abs(x[0]))[:4])
            loadings[f'{group_name}_PC{pc+1}'] = terms

    Y_reduced = np.hstack(blocks) if blocks else Y
    return Y_reduced, loadings


def run_block_pca(sessions: list, flags: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Infer modality groups from actual session channel keys
    if sessions[0].modality_keys:
        keys = sessions[0].modality_keys
        groups = {g: [k for k in gkeys if k in keys]
                  for g, gkeys in V11_MODALITY_GROUPS.items()
                  if any(k in keys for k in gkeys)}
        assigned = {k for gkeys in groups.values() for k in gkeys}
        leftover = [k for k in keys if k not in assigned]
        if leftover:
            groups['Other'] = leftover
    else:
        keys = []
        groups = {}

    Y_all = np.vstack([s.Y_raw for s in sessions])
    Y_red, loadings = fit_block_pca(Y_all, keys, groups, flags)

    with open(os.path.join(output_dir, 'block_pca_loadings.md'), 'w') as f:
        f.write('# Block PCA Loadings\n\n')
        for pc_name, formula in loadings.items():
            f.write(f'**{pc_name}**: {formula}\n\n')

    n_in = Y_all.shape[1]
    n_out = Y_red.shape[1]

    with open(os.path.join(output_dir, 'module5_report.md'), 'w') as f:
        f.write('# Module 5: Block PCA Preprocessing Variant\n\n')
        f.write(f'Input dims: {n_in} → Reduced dims: {n_out}\n')
        f.write(f'Excluded (flagged): {flags.get("slow_drift", [])} + {flags.get("high_vif", [])}\n\n')
        f.write('See block_pca_loadings.md for full PC definitions.\n')
        f.write('\n**Note:** Re-fit rSLDS on reduced features using run_tier2.py --refit-block-pca\n')

    return {'n_reduced_dims': n_out, 'n_input_dims': n_in, 'loadings': loadings}
