"""Stage 0 preprocessing — re-exports from face_event_coincidence.

Single source of truth for the per-AU smoothing + baseline-subtraction
substrate that both the MVP channel and the synchrony pipeline use.
Keeping this as a re-export avoids drift between the two stacks.
"""
from __future__ import annotations

from cadence.significance.face_event_coincidence import (
    _smooth_savgol as smooth_savgol,
    _baseline_subtract as baseline_subtract,
)

__all__ = ['smooth_savgol', 'baseline_subtract']
