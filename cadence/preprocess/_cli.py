"""Shared CLI helper: build argparse interface for any preprocess_<modality>_session."""

from __future__ import annotations

import argparse
import sys
from typing import Callable


def make_modality_cli(preprocess_fn: Callable, modality_name: str | None = None) -> None:
    """Build and run argparse CLI for a preprocess_<modality>_session function.

    Each cli.py becomes a 3-line wrapper:

        from cadence.preprocess._cli import make_modality_cli
        from cadence.preprocess.eeg.pipeline import preprocess_eeg_session
        if __name__ == "__main__":
            make_modality_cli(preprocess_eeg_session)
    """
    name = modality_name or preprocess_fn.__name__.replace("preprocess_", "").replace("_session", "")
    ap = argparse.ArgumentParser(prog=f"python -m cadence.preprocess.{name}")
    ap.add_argument("session_ids", nargs="*",
                    help="Session IDs to process; mutually exclusive with --all")
    ap.add_argument("--all", action="store_true",
                    help="Process every canonical session per session_quality.yaml")
    ap.add_argument("--digest-dir", default=None,
                    help="Override digest directory (default: data/digest/v1)")
    ap.add_argument("--out-dir", default=None,
                    help=f"Override output directory (default: data/preproc/{name}/v1)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output is up to date")
    args = ap.parse_args()

    if args.all and args.session_ids:
        print("ERROR: --all and explicit session_ids are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    if args.all:
        from cadence.ingest.quality import list_canonical_sessions
        session_ids = list_canonical_sessions()
    elif args.session_ids:
        session_ids = list(args.session_ids)
    else:
        ap.print_help()
        sys.exit(1)

    kwargs = {"force": args.force}
    if args.digest_dir is not None:
        kwargs["digest_dir"] = args.digest_dir
    if args.out_dir is not None:
        kwargs["out_dir"] = args.out_dir

    n_ok = 0
    n_fail = 0
    for sid in session_ids:
        try:
            preprocess_fn(sid, **kwargs)
            n_ok += 1
            print(f"  [OK] {sid}")
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {sid}: {e}", file=sys.stderr)
    print(f"\n{name}: {n_ok}/{len(session_ids)} succeeded ({n_fail} failed)")
    sys.exit(0 if n_fail == 0 else 1)
