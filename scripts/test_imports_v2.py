"""Deep import test for all v2 modules."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print('=== Deep Import Test ===')

print('Phase A: wavelet + interbrain...')
from cadence.data.wavelet_features import extract_wavelet_features, _morlet_wavelet_bank, _cwt_gpu, _build_roi_signals
from cadence.data.interbrain_features import extract_interbrain_features
print('  OK')

print('Phase B: preprocessors v2...')
from cadence.data.preprocessors import extract_blendshapes_v2, extract_ecg_features_v2
print('  OK')

print('Phase C: group lasso + CV...')
from cadence.regression.group_lasso import GroupLassoSolver
from cadence.regression.time_blocked_cv import create_time_blocks, cross_validate_lambda
print('  OK')

print('Phase D: estimator + discovery + pathways...')
from cadence.coupling.estimator import CouplingEstimator, CouplingResult
from cadence.coupling.discovery import DiscoveryResult, ConsistencyResult, cross_session_consistency, build_stage2_feature_set, discovery_summary
from cadence.coupling.pathways import get_modality_pathways_v2, get_pathway_category, get_feature_groups_v2
print('  OK')

print('Phase E: alignment v2...')
from cadence.data.alignment import _ensure_v2_features
print('  OK')

print('Phase G: visualizations...')
from cadence.visualization.spectral import plot_spectral_coupling_map, plot_coupling_spectrum, extract_spectral_map
from cadence.visualization.discovery import plot_discovery_report, plot_lambda_path, plot_feature_selection_heatmap
from cadence.visualization.heatmaps import plot_coupling_matrix
print('  OK')

print('Constants...')
from cadence.constants import (
    EEG_ROIS, EEG_ROI_NAMES, WAVELET_CENTER_FREQS,
    WAVELET_FEATURE_NAMES, INTERBRAIN_FEATURE_NAMES,
    MODALITY_SPECS_V2, MODALITY_ORDER_V2, MOD_SHORT_V2,
    BL_FEATURE_NAMES_V2, ECG_FEATURE_NAMES_V2,
)
print('  OK')

print('\n=== All imports resolved successfully ===')
