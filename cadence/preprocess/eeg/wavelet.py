"""V7 EEG wavelet features (Morlet CWT, ROI-projected, GPU-accelerated).

Re-export shim — the production source still lives at
``cadence.data.wavelet_features``. This module is the canonical
post-2026-05-01 import path.
"""

from __future__ import annotations

from cadence.data.wavelet_features import (  # noqa: F401
    extract_wavelet_features,
    _morlet_wavelet_bank,
    _cwt_gpu,
    _build_roi_signals,
)
