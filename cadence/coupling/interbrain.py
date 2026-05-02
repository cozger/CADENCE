"""Cross-participant (inter-brain) features.

Re-export shim — the production source still lives at
``cadence.data.interbrain_features``. This module is the canonical
post-2026-05-01 import path and exists so consumers can use the new
location without forcing a physical relocation in this migration window.

When the source is later moved here, the shim becomes the implementation
file and the legacy module gets a back-reference.
"""

from __future__ import annotations

from cadence.data.interbrain_features import (  # noqa: F401
    extract_interbrain_features,
)
