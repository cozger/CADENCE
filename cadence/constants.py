"""Shared constants: modality specs, feature names, colors, segment maps."""

# ---------------------------------------------------------------------------
# Modality identifiers
# ---------------------------------------------------------------------------

MODALITY_NAMES = ['EEG', 'ECG', 'Blendshapes', 'Pose']
MODALITY_ORDER = ['eeg_features', 'ecg_features', 'blendshapes', 'pose_features']

MODALITY_COLORS = {
    'eeg_features': '#2196F3',   # blue
    'ecg_features': '#F44336',   # red
    'blendshapes': '#4CAF50',    # green
    'pose_features': '#FF9800',  # orange
}

MODALITY_COLORS_LIST = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

MOD_SHORT = {
    'eeg_features': 'EEG',
    'ecg_features': 'ECG',
    'blendshapes': 'BL',
    'pose_features': 'Pose',
    'overall': 'Overall',
}

# Per-modality native specs: (n_channels_with_activity, sample_rate_hz)
MODALITY_SPECS = {
    'eeg_features':  (8, 2.0),
    'ecg_features':  (6, 2.0),
    'blendshapes':   (53, 30.0),
    'pose_features': (41, 12.0),
}

# Base channels (without activity channel)
MODALITY_BASE_CH = {
    'eeg_features': 7,
    'ecg_features': 6,
    'blendshapes': 52,
    'pose_features': 40,
}

# ---------------------------------------------------------------------------
# ECG feature names (6 HRV channels)
# ---------------------------------------------------------------------------

ECG_FEATURE_NAMES = [
    'ecg_hr', 'ecg_ibi_dev', 'ecg_rmssd',
    'ecg_hr_accel', 'ecg_qrs_amp', 'ecg_hr_trend',
]

# ---------------------------------------------------------------------------
# Pose segment map (8 body segment groups -> channel ranges)
# ---------------------------------------------------------------------------

POSE_SEGMENT_MAP = {
    'pose_head': (0, 8),
    'pose_larm': (8, 13),
    'pose_rarm': (13, 18),
    'pose_torso': (18, 24),
    'pose_lleg': (24, 29),
    'pose_rleg': (29, 34),
    'pose_global': (34, 40),
    'pose_activity': (40, 41),
}

# ---------------------------------------------------------------------------
# Blendshape segment map (11 AU functional groups -> channel lists)
# ---------------------------------------------------------------------------

BLENDSHAPE_SEGMENT_MAP = {
    'bl_brow':         [0, 1, 2, 3, 4],
    'bl_cheek_nose':   [5, 6, 7, 49, 50],
    'bl_eye_blink':    [8, 9],
    'bl_eye_gaze':     [10, 11, 12, 13, 14, 15, 16, 17],
    'bl_eye_lid':      [18, 19, 20, 21],
    'bl_jaw':          [22, 23, 24, 25],
    'bl_mouth_affect': [27, 28, 29, 30, 43, 44],
    'bl_mouth_form':   [26, 31, 37, 39, 40],
    'bl_mouth_move':   [32, 33, 34, 35, 36, 38, 41, 42, 45, 46, 47, 48],
    'bl_neutral':      [51],
    'bl_activity':     [52],
}
BL_SEGMENT_NAMES = list(BLENDSHAPE_SEGMENT_MAP.keys())

# Reverse map: AU channel index -> segment name (for PCA interpretation)
_BL_CH_TO_SEGMENT = {}
for _seg_name, _ch_list in BLENDSHAPE_SEGMENT_MAP.items():
    for _ch in _ch_list:
        _BL_CH_TO_SEGMENT[_ch] = _seg_name.replace('bl_', '')


_BL_SEG_SHORT = {
    'brow': 'brw', 'cheek_nose': 'chk', 'eye_blink': 'blnk',
    'eye_gaze': 'gaze', 'eye_lid': 'lid', 'jaw': 'jaw',
    'mouth_affect': 'maff', 'mouth_form': 'mfrm',
    'mouth_move': 'mmov', 'neutral': 'neut', 'activity': 'act',
}


def bl_pca_label(pc_idx, pca_loadings, top_n=2):
    """Name a blendshape PCA component by its dominant AU group contributions.

    Args:
        pc_idx: PCA component index (0-14)
        pca_loadings: (n_components, 52) PCA loadings matrix (Vt rows)
        top_n: number of top AU groups to include in label

    Returns:
        str like 'PC0(mmov+jaw)' or 'PC3(brw+gaze)'
    """
    if pca_loadings is None or pc_idx >= pca_loadings.shape[0]:
        return f'PC{pc_idx}'

    loadings = pca_loadings[pc_idx]  # (52,)
    abs_loadings = abs(loadings)

    # Accumulate loading energy per AU segment
    segment_energy = {}
    for ch_idx in range(min(52, len(abs_loadings))):
        seg = _BL_CH_TO_SEGMENT.get(ch_idx, 'other')
        segment_energy[seg] = segment_energy.get(seg, 0.0) + abs_loadings[ch_idx] ** 2

    # Sort by energy, take top_n
    ranked = sorted(segment_energy.items(), key=lambda kv: -kv[1])
    top_segs = [_BL_SEG_SHORT.get(name, name) for name, _ in ranked[:top_n]]

    return f'PC{pc_idx}({"+".join(top_segs)})'


# ---------------------------------------------------------------------------
# EEG feature names (8 channels: 7 base + activity)
# ---------------------------------------------------------------------------

EEG_FEATURE_NAMES_V6 = [
    'eeg_engagement_index',
    'eeg_frontal_aperiodic_exponent',
    'eeg_frontal_theta_burst_frac',
    'eeg_phase_frontal_theta_cos', 'eeg_phase_frontal_theta_sin',
    'eeg_phase_frontal_alpha_cos', 'eeg_phase_frontal_alpha_sin',
    'eeg_activity',
]

# ---------------------------------------------------------------------------
# All feature decomposition keys (33 total)
# ---------------------------------------------------------------------------

FEATURE_KEYS = (
    EEG_FEATURE_NAMES_V6
    + ECG_FEATURE_NAMES
    + list(POSE_SEGMENT_MAP.keys())
    + BL_SEGMENT_NAMES
)

# ---------------------------------------------------------------------------
# Synthetic data config
# ---------------------------------------------------------------------------

SYNTH_MODALITY_CONFIG = {
    'eeg_features':  {'n_ch': 8, 'hz': 5.0,  'lag_s': 2.0, 'base_ch': 7},
    'ecg_features':  {'n_ch': 6, 'hz': 2.0,  'lag_s': 2.0, 'base_ch': 6},
    'blendshapes':   {'n_ch': 53, 'hz': 30.0, 'lag_s': 2.0, 'base_ch': 52},
    'pose_features': {'n_ch': 41, 'hz': 12.0, 'lag_s': 2.0, 'base_ch': 40},
}

COUPLING_PROFILES = {
    'eeg_features': {
        'duty_cycle': 0.30,
        'event_range_s': (5, 20),
        'ramp_s': 2.0,
    },
    'ecg_features': {
        'duty_cycle': 0.40,
        'event_range_s': (15, 45),
        'ramp_s': 5.0,
    },
    'blendshapes': {
        'duty_cycle': 0.12,
        'event_range_s': (2, 8),
        'ramp_s': 0.5,
    },
    'pose_features': {
        'duty_cycle': 0.25,
        'event_range_s': (3, 12),
        'ramp_s': 1.0,
    },
}


# ===========================================================================
# V2 Pipeline Constants
# ===========================================================================

import numpy as _np

# ---------------------------------------------------------------------------
# EEG ROIs (14-ch Emotiv EPOC, indices after preprocess_eeg cols 3-16)
# ---------------------------------------------------------------------------

EEG_ROIS = {
    'frontal':     [0, 2, 11, 13],   # AF3, F3, F4, AF4
    'left_temp':   [1, 3, 4],         # F7, FC5, T7
    'right_temp':  [10, 12, 9],       # FC6, F8, T8
    'posterior':   [5, 6, 7, 8],       # P7, O1, O2, P8
}

EEG_ROI_NAMES = list(EEG_ROIS.keys())

# ---------------------------------------------------------------------------
# Wavelet center frequencies (20 log-spaced from 2-45 Hz)
# ---------------------------------------------------------------------------

WAVELET_CENTER_FREQS = _np.logspace(
    _np.log10(2.0), _np.log10(45.0), 20
).astype(_np.float64)

# ---------------------------------------------------------------------------
# Wavelet feature names (2 components x 20 freqs x 4 ROIs = 160)
# ---------------------------------------------------------------------------

# EEG wavelet feature structure (for pre-grouping correlated features)
EEG_WAVELET_N_COMPONENTS = 2   # real, imag
EEG_WAVELET_N_FREQS = 20
EEG_WAVELET_N_ROIS = 4

WAVELET_FEATURE_NAMES = []
for _comp in ['real', 'imag']:
    for _freq in WAVELET_CENTER_FREQS:
        for _roi in EEG_ROI_NAMES:
            WAVELET_FEATURE_NAMES.append(
                f'eeg_w_{_comp}_f{_freq:.1f}_{_roi}')

# ---------------------------------------------------------------------------
# Inter-brain feature names (2 components x 20 freqs x 4 ROIs = 160)
# ---------------------------------------------------------------------------

INTERBRAIN_FEATURE_NAMES = []
for _comp in ['cos', 'sin']:
    for _freq in WAVELET_CENTER_FREQS:
        for _roi in EEG_ROI_NAMES:
            INTERBRAIN_FEATURE_NAMES.append(
                f'eeg_ib_{_comp}_f{_freq:.1f}_{_roi}')

# ---------------------------------------------------------------------------
# V2 modality specs and order
# ---------------------------------------------------------------------------

MODALITY_SPECS_V2 = {
    'eeg_wavelet':     (160, 5.0),
    'eeg_interbrain':  (160, 5.0),
    'ecg_features_v2': (7, 2.0),
    'blendshapes_v2':  (31, 30.0),
    'pose_features':   (41, 12.0),
}

MODALITY_ORDER_V2 = [
    'eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2', 'pose_features',
]

# Inter-brain is source-only (not associated with either participant)
INTERBRAIN_MODALITY = 'eeg_interbrain'

MODALITY_COLORS_V2 = {
    'eeg_wavelet':     '#2196F3',   # blue
    'eeg_interbrain':  '#9C27B0',   # purple
    'ecg_features_v2': '#F44336',   # red
    'blendshapes_v2':  '#4CAF50',   # green
    'pose_features':   '#FF9800',   # orange
}

MOD_SHORT_V2 = {
    'eeg_wavelet':     'EEGw',
    'eeg_interbrain':  'EEGib',
    'ecg_features_v2': 'ECG',
    'blendshapes_v2':  'BL',
    'pose_features':   'Pose',
}

# Blendshape v2 feature names: 15 PCA + 15 derivatives + 1 activity = 31
BL_FEATURE_NAMES_V2 = (
    [f'bl_pca_{i:02d}' for i in range(15)]
    + [f'bl_pca_{i:02d}_dt' for i in range(15)]
    + ['bl_activity']
)

# ECG v2 feature names: 6 original + 1 RMSSD derivative = 7
ECG_FEATURE_NAMES_V2 = [
    'ecg_hr', 'ecg_ibi_dev', 'ecg_rmssd',
    'ecg_hr_accel', 'ecg_qrs_amp', 'ecg_hr_trend',
    'ecg_rmssd_dt',
]

# V2 synthetic config
SYNTH_MODALITY_CONFIG_V2 = {
    'eeg_wavelet':     {'n_ch': 160, 'hz': 5.0,  'lag_s': 2.0, 'base_ch': 160, 'n_coupled': 20},
    'ecg_features_v2': {'n_ch': 7,   'hz': 2.0,  'lag_s': 2.0, 'base_ch': 7,   'n_coupled': 7},
    'blendshapes_v2':  {'n_ch': 31,  'hz': 30.0, 'lag_s': 2.0, 'base_ch': 30,  'n_coupled': 10,
                        'has_derivatives': True, 'n_pca': 15},
    'pose_features':   {'n_ch': 41,  'hz': 12.0, 'lag_s': 2.0, 'base_ch': 40,  'n_coupled': 10},
}

COUPLING_PROFILES_V2 = {
    'eeg_wavelet': {
        'duty_cycle': 0.30,
        'event_range_s': (5, 20),
        'ramp_s': 2.0,
    },
    'ecg_features_v2': {
        'duty_cycle': 0.40,
        'event_range_s': (15, 45),
        'ramp_s': 5.0,
    },
    'blendshapes_v2': {
        'duty_cycle': 0.25,
        'event_range_s': (0.5, 10),
        'ramp_s': 0.3,
    },
    'pose_features': {
        'duty_cycle': 0.25,
        # Bimodal: fast gestures/nods + sustained posture similarity
        'bands': [
            {'event_range_s': (1, 5), 'ramp_s': 0.3, 'weight': 0.5},
            {'event_range_s': (10, 40), 'ramp_s': 2.0, 'weight': 0.5},
        ],
        # Fallback keys for code that reads event_range_s directly
        'event_range_s': (3, 12),
        'ramp_s': 1.0,
    },
}
