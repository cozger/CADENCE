"""EEG feature extraction (8-ch at 2 Hz: engagement, aperiodic, theta burst,
frontal theta/alpha phase, activity).

Re-export shim — the production source still lives at
``cadence.data.eeg_features``. This module is the canonical
post-2026-05-01 import path. ``cadence.preprocess.eeg.pipeline`` already
imports through this surface.
"""

from __future__ import annotations

from cadence.data.eeg_features import (  # noqa: F401
    BANDS,
    EEG_FEATURE_NAMES,
    EEG_FEATURES_SRATE,
    EEG_N_FEATURES,
    FRONTAL,
    WIN_DURATION,
    extract_eeg_features,
)
