"""EEG preprocessing: bridges MATLAB EEGLAB output (ASR + 1-40 Hz FIR + interp)."""

from cadence.preprocess.eeg.pipeline import (
    EEG_MODALITY_VERSION,
    preprocess_eeg_session,
)
from cadence.preprocess.eeg.matlab_bridge import (
    DEFAULT_MATLAB_DIR,
    MATLAB_PREPROCESS_PARAMS,
    has_fresh_clean_mat,
    load_clean_mat,
    relocate_existing_clean_mats,
)
