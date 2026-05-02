"""Step 5.2: relocate existing analysis/eeglab_wavelet/cache/<sid>_p{1,2}_clean.mat
into data/matlab/<sid>_p{1,2}_clean.mat with provenance sidecars.

Re-runnable. Idempotent: skips already-up-to-date pairs unless --force.

Usage:
    python scripts/_relocate_clean_mats.py
    python scripts/_relocate_clean_mats.py --force
    python scripts/_relocate_clean_mats.py --canonical-only

The matlab clean.mat data is written under the canonical session_id (the XDF
stem). Legacy cache stems like 'Y_55' get rewritten to 'Y_55_04272026' so the
preproc layer can look up by the same session_id used by the digest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Hoist torch BEFORE numpy on Windows torch 2.10+numpy 2.4 stacks.
import torch  # noqa: F401

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from cadence.preprocess.eeg.matlab_bridge import (
    LEGACY_CACHE_DIR,
    DEFAULT_MATLAB_DIR,
    relocate_existing_clean_mats,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(LEGACY_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_MATLAB_DIR))
    ap.add_argument("--raw-dir", default="raw sessions")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite even if dst is up to date.")
    ap.add_argument("--canonical-only", action="store_true",
                    help="Restrict to sessions flagged canonical: true in YAML.")
    ap.add_argument("--eeglab-version", default="unknown_legacy",
                    help="EEGLAB version string for the sidecar (best-effort retroactive).")
    args = ap.parse_args()

    session_ids: set[str] | None = None
    if args.canonical_only:
        from cadence.ingest.quality import list_canonical_sessions
        session_ids = set(list_canonical_sessions())
        print(f"[relocate] restricting to {len(session_ids)} canonical sessions")

    results = relocate_existing_clean_mats(
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        raw_dir=Path(args.raw_dir),
        session_ids=session_ids,
        force=args.force,
        eeglab_version=args.eeglab_version,
    )

    if not results:
        print("[relocate] no sessions relocated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
