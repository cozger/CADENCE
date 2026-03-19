"""YAML config loader with defaults for all CADENCE parameters."""

import os
import yaml


DEFAULTS = {
    # Pipeline version
    'pipeline': 'v2',

    # Data pipeline
    'session_cache': 'C:/Users/optilab/desktop/MCCT/session_cache',
    'device': 'cuda',

    # Modality specs (n_features, sample_rate)
    'modalities': {
        'eeg_features':  {'n_features': 8,  'sample_rate': 2.0},
        'ecg_features':  {'n_features': 6,  'sample_rate': 2.0},
        'blendshapes':   {'n_features': 53, 'sample_rate': 30.0},
        'pose_features': {'n_features': 41, 'sample_rate': 12.0},
    },

    # Basis function config
    'basis': {
        'layer1': {
            'n_basis': 8,
            'max_lag_seconds': 5.0,
            'min_lag_seconds': 0.0,
            'log_spacing': True,
        },
        'layer2': {
            'n_basis': 10,
            'max_lag_seconds': 30.0,
            'min_lag_seconds': 0.0,
            'log_spacing': True,
        },
    },

    # EWLS solver config
    'ewls': {
        'tau_seconds': 30.0,
        'lambda_ridge': 1e-3,
        'eval_rate': 2.0,
        'min_effective_n': 20,
    },

    # Autoregressive terms
    'autoregressive': {
        'order': 3,
        'include': True,
    },

    # Pathway discovery
    'discovery': {
        'group_lasso_alpha_range': [1e-4, 1e-1],
        'n_folds': 5,
        'significance_threshold': 0.05,
    },

    # Feature decomposition (Phase 2)
    'decomposition': {
        'enabled': True,
        'blendshape_groups': True,
        'pose_groups': True,
    },

    # Significance testing
    'significance': {
        'f_test_alpha': 0.05,
        'surrogate': {
            'n_surrogates': 200,
            'method': 'circular_shift',
            'min_shift_frac': 0.1,
        },
        'fdr_correction': 'bh',   # Benjamini-Hochberg
        'session_level': {
            'binomial_alpha': 0.01,
            'ttest_alpha': 0.01,
            'min_dr2': 0.001,
        },
        'max_pathway_p': 0.7,
        'timepoint': {
            'enabled': True,
            'n_surrogates': 20,
            'quantile': 0.95,
            'smooth_sec': 30,
            'seed': 42,
            'surrogate_eval_rate': 1.0,
        },
    },

    # Interbrain feature configuration
    'interbrain': {
        'min_freq_hz': 4.0,               # zero channels below this (exclude delta)
        'surrogate_method': 'fourier_phase',  # 'circular_shift' or 'fourier_phase'
    },

    # Layer 3 modulation (deferred)
    'modulation': {
        'enabled': False,
        'moderators': [],
    },

    # Visualization
    'visualization': {
        'figsize': [16, 8],
        'dpi': 150,
    },

    # Sessions to analyze
    'sessions': None,  # None = all discovered sessions
    'excluded_sessions': ['y26_022728'],

    # V2 pipeline settings
    'wavelet': {
        'n_frequencies': 20,
        'freq_range': [2.0, 45.0],
        'n_cycles': [3, 8],
        'output_hz': 10.0,
    },
    'blendshapes_v2': {
        'n_pca_components': 15,
        'derivative_sigma_s': 0.5,
    },
    'v2_discovery': {
        'group_lasso_n_lambdas': 20,
        'n_folds': 5,
        'cv_gap_seconds': 30,
        'consistency_min_sessions': 4,
    },
    'stage2': {
        'moderation': {
            'enabled': True,
            'moderators': ['ecg_hr', 'ecg_rmssd'],
        },
        'nonlinear': {
            'enabled': True,
        },
    },
    'pathway_temporal': {
        'fast': {'max_lag_seconds': 12.0, 'n_basis': 10},
        'medium': {'max_lag_seconds': 18.0, 'n_basis': 10},
        'slow': {'max_lag_seconds': 45.0, 'n_basis': 12},
    },
    'pathway_category': {
        'eeg_wavelet->blendshapes_v2': 'fast',
        'eeg_wavelet->pose_features': 'medium',
        'blendshapes_v2->pose_features': 'medium',
        'default': 'medium',
    },
}


def load_config(path=None):
    """Load CADENCE config from YAML file, merging with defaults.

    Args:
        path: Path to YAML config file. If None, returns defaults.

    Returns:
        dict with all config parameters.
    """
    config = _deep_copy_dict(DEFAULTS)

    if path is not None and os.path.exists(path):
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    return config


def _deep_copy_dict(d):
    """Deep copy a nested dict of primitives and lists."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _deep_merge(base, override):
    """Recursively merge override into base dict."""
    merged = _deep_copy_dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
