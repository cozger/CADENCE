"""Data pipeline: XDF loading, preprocessing, feature extraction, alignment."""

from cadence.data.alignment import (
    load_session_from_cache,
    discover_cached_sessions,
    load_and_preprocess_cached,
    apply_modality_exclusions,
    EXCLUDED_MODALITIES,
)
