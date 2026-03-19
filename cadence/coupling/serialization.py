"""Save/load full CouplingResult + DiscoveryResult to disk.

Stores all arrays, metadata, and discovery data in a single .npz file
so that visualizations can be regenerated without re-running analysis.
"""

import os
import time
import numpy as np
from dataclasses import fields

import torch


def _key_str(key_tuple):
    """Convert (src, tgt) tuple to serializable string key."""
    return f'{key_tuple[0]}||{key_tuple[1]}'


def _str_key(key_str):
    """Convert serialized string back to (src, tgt) tuple."""
    parts = key_str.split('||')
    return (parts[0], parts[1])


def save_result(result, path, session_name='', runtime_s=0.0,
                peak_gpu_mb=0.0, peak_cpu_mb=0.0):
    """Save a CouplingResult (and its discovery) to a .npz file.

    Args:
        result: CouplingResult dataclass.
        path: Output file path (will add .npz if needed).
        session_name: Session identifier.
        runtime_s: Total analysis runtime in seconds.
        peak_gpu_mb: Peak GPU memory usage in MB.
        peak_cpu_mb: Peak CPU/RSS memory usage in MB.
    """
    data = {}

    # Scalar metadata
    data['_direction'] = np.array([result.direction], dtype=object)
    data['_session_name'] = np.array([session_name], dtype=object)
    data['_n_significant_pathways'] = np.array([result.n_significant_pathways])
    data['_runtime_s'] = np.array([runtime_s])
    data['_peak_gpu_mb'] = np.array([peak_gpu_mb])
    data['_peak_cpu_mb'] = np.array([peak_cpu_mb])
    data['_save_time'] = np.array([time.time()])

    # Times array
    data['times'] = result.times

    # Overall dR2
    if result.overall_dr2 is not None:
        data['overall_dr2'] = result.overall_dr2

    # Dict[Tuple, ndarray] fields
    array_dicts = [
        'pathway_dr2', 'pathway_r2_full', 'pathway_r2_restricted',
        'pathway_kernels', 'pathway_pvalues', 'pathway_f_stat',
        'feature_dr2', 'feature_kernels', 'pathway_times',
    ]
    for field_name in array_dicts:
        d = getattr(result, field_name, {})
        if d:
            keys = []
            for k, v in d.items():
                ks = _key_str(k)
                keys.append(ks)
                if isinstance(v, np.ndarray):
                    data[f'{field_name}/{ks}'] = v
            data[f'{field_name}/_keys'] = np.array(keys, dtype=object)

    # pathway_significant: Dict[Tuple, bool]
    if result.pathway_significant:
        keys = []
        vals = []
        for k, v in result.pathway_significant.items():
            keys.append(_key_str(k))
            vals.append(v)
        data['pathway_significant/_keys'] = np.array(keys, dtype=object)
        data['pathway_significant/_vals'] = np.array(vals, dtype=bool)

    # pathway_feature_dr2: Dict[Tuple, Dict[int, ndarray]]
    if hasattr(result, 'pathway_feature_dr2') and result.pathway_feature_dr2:
        pw_keys = []
        for pw_key, feat_dict in result.pathway_feature_dr2.items():
            pks = _key_str(pw_key)
            pw_keys.append(pks)
            feat_indices = []
            for feat_idx, arr in feat_dict.items():
                feat_indices.append(feat_idx)
                data[f'pathway_feature_dr2/{pks}/f{feat_idx}'] = arr
            data[f'pathway_feature_dr2/{pks}/_feat_indices'] = np.array(
                feat_indices, dtype=int)
        data['pathway_feature_dr2/_pw_keys'] = np.array(pw_keys, dtype=object)

    # pathway_feature_pvalues: Dict[Tuple, Dict[int, ndarray]]
    if hasattr(result, 'pathway_feature_pvalues') and result.pathway_feature_pvalues:
        pw_keys = []
        for pw_key, feat_dict in result.pathway_feature_pvalues.items():
            pks = _key_str(pw_key)
            pw_keys.append(pks)
            feat_indices = []
            for feat_idx, arr in feat_dict.items():
                feat_indices.append(feat_idx)
                data[f'pathway_feature_pvalues/{pks}/f{feat_idx}'] = arr
            data[f'pathway_feature_pvalues/{pks}/_feat_indices'] = np.array(
                feat_indices, dtype=int)
        data['pathway_feature_pvalues/_pw_keys'] = np.array(pw_keys, dtype=object)

    # pathway_src_tgt_dr2: Dict[Tuple, Dict[Tuple[int,int], ndarray]]
    if hasattr(result, 'pathway_src_tgt_dr2') and result.pathway_src_tgt_dr2:
        pw_keys = []
        for pw_key, pair_dict in result.pathway_src_tgt_dr2.items():
            pks = _key_str(pw_key)
            pw_keys.append(pks)
            pair_keys = []
            for (src_idx, tgt_idx), arr in pair_dict.items():
                pair_str = f's{src_idx}_t{tgt_idx}'
                pair_keys.append(pair_str)
                data[f'pathway_src_tgt_dr2/{pks}/{pair_str}'] = arr
            data[f'pathway_src_tgt_dr2/{pks}/_pair_keys'] = np.array(
                pair_keys, dtype=object)
        data['pathway_src_tgt_dr2/_pw_keys'] = np.array(pw_keys, dtype=object)

    # Auxiliary data (e.g., PCA loadings for interpretation)
    aux = getattr(result, 'aux', {})
    if aux:
        for aux_key, aux_arr in aux.items():
            if isinstance(aux_arr, np.ndarray):
                data[f'aux/{aux_key}'] = aux_arr

    # Discovery result (if V2 pipeline)
    discovery = getattr(result, 'discovery', None)
    if discovery is not None:
        _save_discovery(data, discovery)

    if not path.endswith('.npz'):
        path += '.npz'
    np.savez_compressed(path, **data)


def _save_discovery(data, disc):
    """Serialize DiscoveryResult fields into the data dict."""
    data['disc/_session_name'] = np.array([disc.session_name], dtype=object)

    # Dict[Tuple, List[int]]
    for field_name in ['selected_features']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            for k, v in d.items():
                ks = _key_str(k)
                keys.append(ks)
                data[f'disc/{field_name}/{ks}'] = np.array(v, dtype=int)
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)

    # Dict[Tuple, ndarray]
    for field_name in ['coefficients', 'cv_scores', 'stability_scores',
                        'block_hit_counts', 'block_pvalues']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            for k, v in d.items():
                ks = _key_str(k)
                keys.append(ks)
                if isinstance(v, np.ndarray):
                    data[f'disc/{field_name}/{ks}'] = v
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)

    # Dict[Tuple, float]
    for field_name in ['best_lambdas', 'block_pathway_pvalue']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            vals = []
            for k, v in d.items():
                keys.append(_key_str(k))
                vals.append(v)
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)
            data[f'disc/{field_name}/_vals'] = np.array(vals, dtype=float)

    # Dict[Tuple, int]
    for field_name in ['n_selected', 'n_blocks']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            vals = []
            for k, v in d.items():
                keys.append(_key_str(k))
                vals.append(v)
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)
            data[f'disc/{field_name}/_vals'] = np.array(vals, dtype=int)

    # Dict[Tuple, str]
    for field_name in ['selection_method']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            vals = []
            for k, v in d.items():
                keys.append(_key_str(k))
                vals.append(v)
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)
            data[f'disc/{field_name}/_vals'] = np.array(vals, dtype=object)

    # Dict[Tuple, dict] — feature_clusters: serialize as JSON-like
    for field_name in ['feature_clusters']:
        d = getattr(disc, field_name, {})
        if d:
            keys = []
            for k, v in d.items():
                ks = _key_str(k)
                keys.append(ks)
                # Store cluster map: {cluster_idx: [feat_indices]}
                for cidx, feat_list in v.items():
                    data[f'disc/{field_name}/{ks}/c{cidx}'] = np.array(
                        feat_list, dtype=int)
            data[f'disc/{field_name}/_keys'] = np.array(keys, dtype=object)


def load_result(path):
    """Load a CouplingResult from a .npz file.

    Returns:
        result: CouplingResult dataclass.
        meta: dict with runtime_s, peak_gpu_mb, peak_cpu_mb, session_name.
    """
    from cadence.coupling.estimator import CouplingResult

    if not path.endswith('.npz'):
        path += '.npz'
    raw = np.load(path, allow_pickle=True)

    direction = str(raw['_direction'][0])
    times = raw['times']
    result = CouplingResult(direction=direction, times=times)
    result.n_significant_pathways = int(raw['_n_significant_pathways'][0])

    if 'overall_dr2' in raw:
        result.overall_dr2 = raw['overall_dr2']

    # Restore Dict[Tuple, ndarray] fields
    array_dicts = [
        'pathway_dr2', 'pathway_r2_full', 'pathway_r2_restricted',
        'pathway_kernels', 'pathway_pvalues', 'pathway_f_stat',
        'feature_dr2', 'feature_kernels', 'pathway_times',
    ]
    for field_name in array_dicts:
        keys_key = f'{field_name}/_keys'
        if keys_key in raw:
            d = {}
            for ks in raw[keys_key]:
                k = _str_key(str(ks))
                arr_key = f'{field_name}/{ks}'
                if arr_key in raw:
                    d[k] = raw[arr_key]
            setattr(result, field_name, d)

    # pathway_significant
    if 'pathway_significant/_keys' in raw:
        keys = raw['pathway_significant/_keys']
        vals = raw['pathway_significant/_vals']
        result.pathway_significant = {
            _str_key(str(k)): bool(v) for k, v in zip(keys, vals)
        }

    # pathway_feature_dr2
    if 'pathway_feature_dr2/_pw_keys' in raw:
        result.pathway_feature_dr2 = {}
        for pks in raw['pathway_feature_dr2/_pw_keys']:
            pks = str(pks)
            pw_key = _str_key(pks)
            feat_indices = raw[f'pathway_feature_dr2/{pks}/_feat_indices']
            feat_dict = {}
            for fi in feat_indices:
                arr_key = f'pathway_feature_dr2/{pks}/f{fi}'
                if arr_key in raw:
                    feat_dict[int(fi)] = raw[arr_key]
            result.pathway_feature_dr2[pw_key] = feat_dict

    # pathway_feature_pvalues
    if 'pathway_feature_pvalues/_pw_keys' in raw:
        result.pathway_feature_pvalues = {}
        for pks in raw['pathway_feature_pvalues/_pw_keys']:
            pks = str(pks)
            pw_key = _str_key(pks)
            feat_indices = raw[f'pathway_feature_pvalues/{pks}/_feat_indices']
            feat_dict = {}
            for fi in feat_indices:
                arr_key = f'pathway_feature_pvalues/{pks}/f{fi}'
                if arr_key in raw:
                    feat_dict[int(fi)] = raw[arr_key]
            result.pathway_feature_pvalues[pw_key] = feat_dict

    # pathway_src_tgt_dr2
    if 'pathway_src_tgt_dr2/_pw_keys' in raw:
        result.pathway_src_tgt_dr2 = {}
        for pks in raw['pathway_src_tgt_dr2/_pw_keys']:
            pks = str(pks)
            pw_key = _str_key(pks)
            pair_keys = raw[f'pathway_src_tgt_dr2/{pks}/_pair_keys']
            pair_dict = {}
            for pk in pair_keys:
                pk = str(pk)
                arr_key = f'pathway_src_tgt_dr2/{pks}/{pk}'
                if arr_key in raw:
                    # Parse 's3_t5' -> (3, 5)
                    parts = pk.split('_')
                    src_idx = int(parts[0][1:])
                    tgt_idx = int(parts[1][1:])
                    pair_dict[(src_idx, tgt_idx)] = raw[arr_key]
            result.pathway_src_tgt_dr2[pw_key] = pair_dict

    # Auxiliary data
    result.aux = {}
    for key in raw.files:
        if key.startswith('aux/'):
            aux_key = key[4:]  # strip 'aux/' prefix
            result.aux[aux_key] = raw[key]

    # Discovery
    if 'disc/_session_name' in raw:
        result.discovery = _load_discovery(raw)

    meta = {
        'session_name': str(raw['_session_name'][0]),
        'runtime_s': float(raw['_runtime_s'][0]),
        'peak_gpu_mb': float(raw['_peak_gpu_mb'][0]),
        'peak_cpu_mb': float(raw['_peak_cpu_mb'][0]),
        'save_time': float(raw['_save_time'][0]),
    }

    return result, meta


def _load_discovery(raw):
    """Restore DiscoveryResult from npz data."""
    from cadence.coupling.discovery import DiscoveryResult
    disc = DiscoveryResult()
    disc.session_name = str(raw['disc/_session_name'][0])

    # List[int] fields
    for field_name in ['selected_features']:
        keys_key = f'disc/{field_name}/_keys'
        if keys_key in raw:
            d = {}
            for ks in raw[keys_key]:
                ks = str(ks)
                k = _str_key(ks)
                d[k] = raw[f'disc/{field_name}/{ks}'].tolist()
            setattr(disc, field_name, d)

    # ndarray fields
    for field_name in ['coefficients', 'cv_scores', 'stability_scores',
                        'block_hit_counts', 'block_pvalues']:
        keys_key = f'disc/{field_name}/_keys'
        if keys_key in raw:
            d = {}
            for ks in raw[keys_key]:
                ks = str(ks)
                k = _str_key(ks)
                arr_key = f'disc/{field_name}/{ks}'
                if arr_key in raw:
                    d[k] = raw[arr_key]
            setattr(disc, field_name, d)

    # float fields
    for field_name in ['best_lambdas', 'block_pathway_pvalue']:
        keys_key = f'disc/{field_name}/_keys'
        if keys_key in raw:
            keys = raw[keys_key]
            vals = raw[f'disc/{field_name}/_vals']
            setattr(disc, field_name,
                    {_str_key(str(k)): float(v) for k, v in zip(keys, vals)})

    # int fields
    for field_name in ['n_selected', 'n_blocks']:
        keys_key = f'disc/{field_name}/_keys'
        if keys_key in raw:
            keys = raw[keys_key]
            vals = raw[f'disc/{field_name}/_vals']
            setattr(disc, field_name,
                    {_str_key(str(k)): int(v) for k, v in zip(keys, vals)})

    # str fields
    for field_name in ['selection_method']:
        keys_key = f'disc/{field_name}/_keys'
        if keys_key in raw:
            keys = raw[keys_key]
            vals = raw[f'disc/{field_name}/_vals']
            setattr(disc, field_name,
                    {_str_key(str(k)): str(v) for k, v in zip(keys, vals)})

    # feature_clusters
    keys_key = 'disc/feature_clusters/_keys'
    if keys_key in raw:
        disc.feature_clusters = {}
        for ks in raw[keys_key]:
            ks = str(ks)
            k = _str_key(ks)
            clusters = {}
            prefix = f'disc/feature_clusters/{ks}/c'
            for arr_name in raw.files:
                if arr_name.startswith(prefix):
                    cidx = int(arr_name[len(prefix):])
                    clusters[cidx] = raw[arr_name].tolist()
            disc.feature_clusters[k] = clusters

    return disc


def get_peak_gpu_mb():
    """Get peak GPU memory usage in MB (resets counter)."""
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        return peak
    return 0.0


def get_rss_mb():
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
