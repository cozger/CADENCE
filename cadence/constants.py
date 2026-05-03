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

# Semantic AU groups for coupling analysis (raw 52-AU mode)
# MediaPipe FaceLandmarker blendshape ordering (verified against YQP source)
BLENDSHAPE_COUPLING_GROUPS = {
    'smile':  [44, 45, 7, 8],      # mouthSmileL/R + cheekSquintL/R
    'brow':   [1, 2, 3, 4, 5],     # browDownL/R, browInnerUp, browOuterUpL/R
    'speech': [23, 24, 25, 26,     # jawForward/Left/Open/Right
               27, 32, 33, 38, 39,  # mouthClose/Funnel/Left/Pucker/Right
               34, 35, 36, 37, 40, 41, 42, 43, 46, 47, 48, 49],  # mouth movements
}
# Default coupled AUs for semisynthetic: browInnerUp + cheekSquintL/R + mouthSmileL/R
BLENDSHAPE_MIMICRY_AUS = [3, 7, 8, 44, 45]

# Anatomical AU regions (canonical 0-indexed MediaPipe FaceLandmarker order,
# matching docs/validate_blendshape_isolation.py and bl_wavelet.AFFECT_AUS).
# These are the source of truth for region-based facial coupling analysis.
# Replaces the older AFFECT_AUS functional grouping.
#
# Excluded: index 0 (_neutral), indices 11-18 (eye gaze direction; V12 will
# treat gaze as a separate modality with stereo-calibrated rays).
AU_REGIONS = {
    # Mouth (27 AUs): jaw + all mouth_* AUs
    'mouth': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
              39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    # Eye expression (6 AUs): blink + squint + wide. Gaze (11-18) excluded.
    'eye':   [9, 10, 19, 20, 21, 22],
    # Brow (5 AUs)
    'brow':  [1, 2, 3, 4, 5],
    # Cheek (3 AUs): cheekPuff + cheekSquint L/R
    'cheek': [6, 7, 8],
    # Nose (2 AUs): noseSneer L/R
    'nose':  [50, 51],
}

# AUs intentionally excluded from anatomical-region analysis.
AU_REGIONS_EXCLUDED = {
    '_neutral': [0],
    'gaze':     [11, 12, 13, 14, 15, 16, 17, 18],
}

# Subdivided 7-region map for synchrony-repertoire identity profiles.
# Keeps AU_REGIONS' anatomical groupings but valence-keys the mouth so a
# polite smile, a frown, and a jaw-drop don't all collapse into one
# "mouth lit up" feature. Disjoint with respect to AU_REGIONS — every AU
# appears in exactly one region (gaze + _neutral excluded).
#
# Total: 5 + 6 + 2 + 3 + 4 + 7 + 16 = 43 AUs (52 − 9 excluded).
AU_REGIONS_7 = {
    'brow':        [1, 2, 3, 4, 5],
    'eye':         [9, 10, 19, 20, 21, 22],
    'nose':        [50, 51],
    'cheek':       [6, 7, 8],
    # Smile composite: dimples + zygomaticus (smile L/R). Cheek-squint is
    # part of a Duchenne smile but is kept under 'cheek' so the regions
    # remain disjoint — clusters can still flag "cheek + mouth_smile
    # co-activate" via the per-region presence vector.
    'mouth_smile': [28, 29, 44, 45],
    # Frown / press / pucker / shrug — non-smile non-speech mouth shapes.
    'mouth_frown': [30, 31, 36, 37, 38, 42, 43],
    # Jaw + oromotor (funnel/close/roll/stretch/upperUp/lowerDown) +
    # lateral mouth movements. Speech-related AUs cluster here.
    'mouth_jaw':   [23, 24, 25, 26, 27, 32, 33, 34, 35, 39, 40, 41,
                    46, 47, 48, 49],
}

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

# Channel names in index order (matches preprocess_eeg cols 3-16)
EPOC_CHANNEL_NAMES = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
]

# 2D scalp-projected positions (x=left/right, y=anterior/posterior)
# Derived from standard 10-20 angular coordinates on a unit circle
EPOC_2D_POS = _np.array([
    [-0.31,  0.95],  # AF3
    [-0.81,  0.59],  # F7
    [-0.39,  0.69],  # F3
    [-0.67,  0.35],  # FC5
    [-1.00,  0.00],  # T7
    [-0.81, -0.59],  # P7
    [-0.31, -0.95],  # O1
    [ 0.31, -0.95],  # O2
    [ 0.81, -0.59],  # P8
    [ 1.00,  0.00],  # T8
    [ 0.67,  0.35],  # FC6
    [ 0.39,  0.69],  # F4
    [ 0.81,  0.59],  # F8
    [ 0.31,  0.95],  # AF4
], dtype=_np.float64)

# Pairwise Euclidean distance matrix (14 x 14)
EPOC_DISTANCE = _np.sqrt(
    ((_EPOC_2D := EPOC_2D_POS[:, None] - EPOC_2D_POS[None, :]) ** 2
     ).sum(axis=2))

# Adjacency: True for nearest 10-20 neighbors (distance < 0.65)
EPOC_ADJACENCY = EPOC_DISTANCE < 0.65
_np.fill_diagonal(EPOC_ADJACENCY, False)  # no self-adjacency

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
    'eeg_wavelet':     (160, 10.0),
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
    'eeg_wavelet':     {'n_ch': 160, 'hz': 10.0, 'lag_s': 2.0, 'base_ch': 160, 'n_coupled': 20},
    'ecg_features_v2': {'n_ch': 7,   'hz': 2.0,  'lag_s': 2.0, 'base_ch': 7,   'n_coupled': 7},
    'blendshapes_v2':  {'n_ch': 31,  'hz': 30.0, 'lag_s': 2.0, 'base_ch': 30,  'n_coupled': 4,
                        'has_derivatives': True, 'n_pca': 15},
    'pose_features':   {'n_ch': 41,  'hz': 12.0, 'lag_s': 2.0, 'base_ch': 40,  'n_coupled': 5},
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
        'event_range_s': (3, 15),
        'ramp_s': 1.0,
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

# ===========================================================================
# V8.2 Semi-Synthetic Test Battery Configuration
# ===========================================================================

# Kappa ranges per modality (ecological scale — amplified internally for EEG/ECG)
# EEG raw Kuramoto kappa must be ~5-7x higher than ecological kappa because
# the pipeline averages across 16 ROI pairs and extracts only the imaginary
# part of coherence. The amplification factor is applied in inject_eeg_v82().
V82_KAPPA_RANGES = {
    'eeg':  [0.0, 0.10, 0.20, 0.30, 0.50],  # ecological scale
    'bl':   [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
    'ecg':  [0.0, 0.10, 0.20, 0.30, 0.40, 0.60],
    'resp': [0.0, 0.15, 0.30, 0.50, 0.70],
    'pose': [0.0, 0.10, 0.20, 0.30, 0.40],
}

# Amplification: ecological kappa × amp_factor = raw Kuramoto kappa
# This compensates for pipeline dilution (ROI averaging, ImCoh extraction,
# prewhitening). Calibrated so ecological kappa=0.30 → AUC≈0.70.
V82_EEG_AMP_FACTOR = 3.0  # phase mixing cap=0.95 prevents oversaturation

# EEG injection scenarios
V82_EEG_SCENARIOS = {
    'E1_mutual_gaze': {
        'band': (8.0, 13.0),         # alpha — match pipeline EEG_BANDS exactly
        'channels': [0, 2, 11, 13],  # frontal: AF3, F3, F4, AF4
        'decay_sigma': 0.8,          # broader spread to reach temporal channels
        'lag_ms': 10,                # 10ms = non-zero ImCoh, ecologically valid
        'phase_weight': 1.0,         # Kuramoto phase rotation
        'envelope_weight': 0.3,      # weak concordance accompaniment
        'asymmetry_weight': 0.0,
        'gate': {'duty_cycle': 0.35, 'event_range_s': (8, 30), 'ramp_s': 2.0},
    },
    'E2_cooperative': {
        'band': (4.0, 8.0),          # theta — match pipeline
        'channels': [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],  # frontal + temporal
        'decay_sigma': 1.0,          # broad spatial spread
        'lag_ms': 40,
        'phase_weight': 1.0,
        'envelope_weight': 0.0,      # phase-only
        'asymmetry_weight': 0.0,
        'gate': {'duty_cycle': 0.35, 'event_range_s': (8, 30), 'ramp_s': 2.0},
    },
    'E3_shared_alpha': {
        'band': (8.0, 13.0),         # alpha — match pipeline
        'channels': list(range(14)),  # all channels
        'decay_sigma': 3.0,          # very broad (all channels ~equal)
        'lag_ms': 0,
        'phase_weight': 0.0,
        'envelope_weight': 1.0,      # envelope-only
        'asymmetry_weight': 0.0,
        'gate': {'duty_cycle': 0.35, 'event_range_s': (10, 30), 'ramp_s': 3.0},
    },
    'E4_therapist_leading': {
        'band': (8.0, 13.0),         # alpha — match pipeline
        'channels': [0, 2, 11, 13],  # frontal
        'decay_sigma': 0.8,          # broader spread
        'lag_ms': 75,
        'phase_weight': 0.6,
        'envelope_weight': 0.4,
        'asymmetry_weight': 0.25,    # therapist power boost
        'gate': {'duty_cycle': 0.35, 'event_range_s': (8, 30), 'ramp_s': 2.0},
    },
}

# BL injection scenarios
V82_BL_SCENARIOS = {
    'B1_smile': {
        'au_subset': [44, 45],       # mouthSmileLeft/Right (2/10 AFFECT_AUS)
        'template': 'smile',
        'lag_s': 0.5,                # Short lag: CWT at 1 Hz needs 5s resolution,
        'lag_jitter_s': 0.3,         #   so templates must overlap. 0.5s lag = 5.5s overlap.
        # 40% duty = conversation segments only (~40% of full session)
        'gate': {'duty_cycle': 0.40, 'event_range_s': (15, 40), 'ramp_s': 3.0},
    },
    'B2_duchenne': {
        'au_subset': [44, 45, 7, 8],  # smile + cheekSquint (4/10)
        'template': 'duchenne',
        'lag_s': 2.7,
        'lag_jitter_s': 1.0,
        'gate': {'duty_cycle': 0.35, 'event_range_s': (15, 40), 'ramp_s': 3.0},
    },
    'B3_empathic_frown': {
        'au_subset': [30, 31],        # mouthFrownLeft/Right (2/10)
        'template': 'frown',
        'lag_s': 4.0,
        'lag_jitter_s': 1.5,
        # Frown events are less frequent
        'gate': {'duty_cycle': 0.25, 'event_range_s': (15, 45), 'ramp_s': 3.0},
    },
}

# ECG injection scenarios
V82_ECG_SCENARIOS = {
    'C1_sympathetic_arousal': {
        'band': 'lf',                # 0.04-0.15 Hz
        'lag_s': 4.0,
        'gate': {'duty_cycle': 0.20, 'event_range_s': (30, 90), 'ramp_s': 10.0},
    },
    'C2_vagal_meditation': {
        'band': 'hf',                # 0.15-0.4 Hz
        'lag_s': 0.5,
        'gate': {'duty_cycle': 0.15, 'event_range_s': (15, 45), 'ramp_s': 5.0},
    },
}

# Respiratory injection scenarios
V82_RESP_SCENARIOS = {
    'R1_guided_breathing': {
        'ramp_s': 20.0,              # slow build-up
        # gate = entire meditation segment, no episodic bursts
    },
}

# Pose injection scenarios
V82_POSE_SCENARIOS = {
    'P1_postural_mirror': {
        # Pipeline now uses multi-lag ±5s cross-product bank (max over lags),
        # so nonzero lags ARE detectable. Use realistic 2s lag for postural
        # mirroring (Ramseyer 2011).
        'channels': [0, 1, 6, 7, 8],  # head centroid x/y + torso lean
        'lag_s': 2.0,
        'mode': 'continuous',
        'gate': {
            'duty_cycle': 0.35,
            'bands': [
                {'event_range_s': (15, 45), 'ramp_s': 3.0, 'weight': 0.6},
                {'event_range_s': (3, 8), 'ramp_s': 0.5, 'weight': 0.4},
            ],
            'event_range_s': (10, 30),
            'ramp_s': 2.0,
        },
    },
    'P2_head_nod': {
        'channels': [3],             # head pitch
        'lag_s': 1.0,                # 1s reciprocal nod delay
        'mode': 'event',
        'gate': {'duty_cycle': 0.30, 'event_range_s': (3, 8), 'ramp_s': 0.3},
    },
}

# Multimodal composite patterns
V82_MULTIMODAL_PATTERNS = {
    'M1_conversation': {
        'eeg': ('E1_mutual_gaze', 0.20),
        'bl': ('B1_smile', 0.25),
        'pose': ('P1_postural_mirror', 0.20),
        'ecg': ('C1_sympathetic_arousal', 0.25),
        'resp': None,
    },
    'M2_meditation': {
        'eeg': ('E3_shared_alpha', 0.15),
        'bl': None,
        'pose': None,
        'ecg': ('C2_vagal_meditation', 0.20),
        'resp': ('R1_guided_breathing', 0.50),
    },
    'M3_therapist_leading': {
        'eeg': ('E4_therapist_leading', 0.20),
        'bl': ('B1_smile', 0.20),
        'pose': ('P1_postural_mirror', 0.15),
        'ecg': None,
        'resp': None,
    },
    'M4_null': {
        'eeg': None, 'bl': None, 'pose': None, 'ecg': None, 'resp': None,
    },
}


# ═════════════════════════════════════════════════════════════════════
# V10 Constants — LZ Complexity + Graph Extensions
# ═════════════════════════════════════════════════════════════════════

# Frontal ROI for LZ complexity (AF3, F7, F3, F4, F8, AF4 on Emotiv EPOC)
FRONTAL_ROI = [0, 1, 2, 11, 12, 13]

# V10 21D modality configuration (16 base + 4 LZ + 1 graph; dyn_theta/alpha/beta collapsed to dyn_mean)
# graph_centrality_eeg dropped: near-constant timecourse from 90s windowed
# eigenvector centrality → prewhitening amplifies noise → spurious AUC.
# Centrality info is captured by lambda-2 in transition covariates.
V10_MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_mean',                                  # collapsed from dyn_theta/alpha/beta (r=0.99 collinear)
    'asym_theta', 'asym_alpha', 'asym_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf',
    'resp', 'pose',
    'lz_conc_theta', 'lz_conc_alpha',
    'lz_asym_theta', 'lz_asym_alpha',
    'graph_modularity',
]
V10_MODALITY_NAMES = [
    'ImCoh theta', 'ImCoh alpha', 'ImCoh beta',
    'Conc theta', 'Conc alpha', 'Conc beta',
    'Dyn mean',
    'Asym theta', 'Asym alpha', 'Asym beta',
    'BL expression', 'BL activity conc',
    'ECG LF (SNS)', 'ECG HF (PNS)',
    'Resp phase', 'Pose velocity',
    'LZ conc theta', 'LZ conc alpha',
    'LZ asym theta', 'LZ asym alpha',
    'Graph modularity',
]
V10_MODALITY_COLORS = [
    '#1565C0', '#2196F3', '#64B5F6',
    '#E65100', '#FF9800', '#FFB74D',
    '#B71C1C',                                   # single color for dyn_mean
    '#00695C', '#00897B', '#4DB6AC',
    '#E91E63', '#F48FB1',
    '#FF5722', '#795548',
    '#607D8B', '#4CAF50',
    '#7B1FA2', '#AB47BC',       # LZ concordance (purple)
    '#4A148C', '#CE93D8',       # LZ asymmetry (dark/light purple)
    '#3E2723',                   # graph modularity (brown)
]

# V10 transition covariate keys (5D)
V10_COVARIATE_KEYS = [
    'z_slow_pc1', 'z_slow_pc2',
    'coupling_flexibility', 'lambda2', 'graph_changepoint',
]

# V10 semi-synthetic LZ injection scenarios
V10_LZ_SCENARIOS = {
    'L1_shared_complexity_theta': {
        'band': (4.0, 8.0),
        'channels': FRONTAL_ROI,
        'mechanism': 'amplitude_modulation',
        'target_features': ['lz_conc_theta'],
    },
    'L2_shared_complexity_alpha': {
        'band': (8.0, 13.0),
        'channels': FRONTAL_ROI,
        'mechanism': 'amplitude_modulation',
        'target_features': ['lz_conc_alpha'],
    },
}

V10_KAPPA_RANGES = {
    'lz': [0.0, 0.10, 0.20, 0.30, 0.40],
}


# ═════════════════════════════════════════════════════════════════════
# V11 Constants — Burst Coincidence + Transfer Entropy Extensions
# ═════════════════════════════════════════════════════════════════════
#
# Key design: TE decomposed into concordance (observation) + asymmetry (covariate).
#   te_conc = (z_T→P + z_P→T) / 2  — total bidirectional information flow (state property)
#   te_asym = z_T→P - z_P→T         — who leads (transition modulator)
# Rationale: te_asym conflates "bidirectional coupling" (both high, diff≈0) with
# "no coupling" (both low, diff≈0). te_conc separates these. Asymmetry modulates
# which state transitions are likely, not which state you're in.
# te_asym_beta omitted: collinear with existing asym_beta (r=-0.74).

# V11 26D observation vector (21 V10 + 2 TE concordance + 3 burst coincidence)
V11_MODALITY_KEYS = V10_MODALITY_KEYS + [
    'te_conc_theta', 'te_conc_alpha',            # 23-24: TE concordance (bidirectional flow, surrogate z)
    'burst_coinc_theta', 'burst_coinc_alpha',     # 25-26: burst coincidence z-score
    'burst_coinc_beta',                           # 27: burst coincidence z-score
]
V11_MODALITY_NAMES = V10_MODALITY_NAMES + [
    'TE conc theta', 'TE conc alpha',
    'Burst coinc theta', 'Burst coinc alpha', 'Burst coinc beta',
]
V11_MODALITY_COLORS = V10_MODALITY_COLORS + [
    '#00BFA5', '#00E676',                # TE concordance (teal/green)
    '#FF6D00', '#FFD600', '#FFAB00',     # burst coincidence (orange/yellow/amber)
]

# V11 transition covariates (7D = 5 V10 + 2 TE asymmetry)
V11_COVARIATE_KEYS = V10_COVARIATE_KEYS + [
    'te_asym_theta', 'te_asym_alpha',    # 5-6: TE directionality (transition modulator)
]
