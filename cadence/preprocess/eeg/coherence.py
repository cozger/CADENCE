"""EEG coherence features.

Re-export shim — the production source still lives at
``cadence.data.eeg_coherence``. This module is the canonical
post-2026-05-01 import path.
"""

from __future__ import annotations

from cadence.data.eeg_coherence import (  # noqa: F401
    DEFAULT_EEG_BANDS,
    eeg_band_coherence,
    eeg_coherence_features,
    eeg_coherence_surrogates,
)
