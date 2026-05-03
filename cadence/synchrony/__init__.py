"""Synchrony repertoire pipeline (Stages 0-9).

Decomposes each dyadic facial-activity episode into multiple complementary
feature views, then clusters episodes across the cohort to discover the
taxonomy of synchronous events that occur in dyadic interaction.

Two scientific axes per dyad:

- *Repertoire signature* — distribution over the cohort-discovered cluster
  taxonomy ("what kinds of synchronous events").
- *Expressivity profile* — episode rate, intensity, AU diversity, dyadic
  balance ("how much activity").

Two dyads with similar repertoires can have very different expressivity,
and vice versa. Both axes are first-class outputs.

Spec: ``docs/superpowers/specs/2026-05-02-synchrony-repertoire-design.md``.
Builds on ``cadence.significance.face_event_coincidence`` (Stage 0+1).
"""
from __future__ import annotations

__version__ = '0.1.0'
