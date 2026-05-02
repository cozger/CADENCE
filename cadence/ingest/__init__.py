"""CADENCE Layer 1: structural ingestion.

Reads raw XDF files, resolves roles, merges marker streams, detects pose format,
and writes canonical .npz + .json artifacts under data/digest/v1/. No signal
cleaning or value transforms — those live in cadence/preprocess/.

See docs/data_pipeline_v1.md for architecture.
"""

from cadence.ingest.roles import (
    RoleAssignment,
    UnresolvedRolesError,
    THERAPIST_NAME_ALIASES,
    PATIENT_ID_PATTERN,
    resolve_roles,
)
from cadence.ingest.schema import CanonicalSession, SCHEMA_VERSION
