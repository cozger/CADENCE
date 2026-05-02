"""CLI entry point: ``python -m cadence.ingest``.

Usage:
    python -m cadence.ingest --session y_06
    python -m cadence.ingest --session Y_55_04272026
    python -m cadence.ingest --all
    python -m cadence.ingest --raw-dir "raw sessions" --out-dir data/digest/v1 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cadence.io.paths import DIGEST_DIR, RAW_SESSIONS_DIR
from cadence.ingest.digest import digest_all, digest_xdf


def _find_xdf(session_id: str, raw_dir: Path) -> Path | None:
    """Locate XDF for a session_id by exact or substring match on basename stem."""
    candidates = sorted(raw_dir.glob("*.xdf"))
    # Exact-stem match first
    for p in candidates:
        if p.stem == session_id:
            return p
    # Substring fallback (case-insensitive)
    sid_lower = session_id.lower()
    matches = [p for p in candidates if sid_lower in p.stem.lower()]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[ingest] WARN: multiple XDFs match {session_id!r}: "
              f"{[m.name for m in matches]}; using {matches[0].name}", file=sys.stderr)
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cadence.ingest")
    ap.add_argument("--session", action="append", default=[],
                    help="Session ID or substring (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="Digest every canonical session per configs/session_quality.yaml")
    ap.add_argument("--every-xdf", action="store_true",
                    help="Digest every XDF in --raw-dir (ignores canonical flag)")
    ap.add_argument("--raw-dir", default=str(RAW_SESSIONS_DIR),
                    help=f"Raw XDF directory (default: {RAW_SESSIONS_DIR})")
    ap.add_argument("--out-dir", default=str(DIGEST_DIR),
                    help=f"Digest output directory (default: {DIGEST_DIR})")
    ap.add_argument("--force", action="store_true",
                    help="Re-digest even if up to date")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers for --all/--every-xdf (threading backend; "
                         "each worker holds one XDF in memory, so 4 ~= 6 GB peak)")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.is_dir():
        print(f"[ingest] ERROR: raw dir not found: {raw_dir}", file=sys.stderr)
        return 1

    if args.all and args.session:
        print("[ingest] ERROR: --all and --session are mutually exclusive", file=sys.stderr)
        return 2
    if args.every_xdf and args.session:
        print("[ingest] ERROR: --every-xdf and --session are mutually exclusive",
              file=sys.stderr)
        return 2

    if args.all or args.every_xdf:
        # pyxdf.load_xdf holds one full XDF in memory; ~1.5 GB/worker realistic
        # for the canonical session sizes (some 4-min recordings are smaller,
        # but multi-stream therapy sessions push that estimate).
        from cadence.io.resources import pick_n_jobs, log_resources
        log_resources(prefix='[ingest --all] ')
        safe_jobs = pick_n_jobs(per_worker_ram_gb=1.5, requested=args.n_jobs)
        if safe_jobs != args.n_jobs:
            print(f"[ingest] resource cap: {args.n_jobs} -> {safe_jobs} workers",
                  flush=True)
        only_canonical = not args.every_xdf
        results = digest_all(raw_dir, out_dir, force=args.force,
                              only_canonical=only_canonical,
                              n_jobs=safe_jobs)
        return 0 if results else 1

    if args.session:
        n_ok = 0
        for sid in args.session:
            xdf = _find_xdf(sid, raw_dir)
            if xdf is None:
                print(f"[ingest] ERROR: no XDF found for {sid!r} in {raw_dir}",
                      file=sys.stderr)
                continue
            result = digest_xdf(xdf, out_dir, force=args.force)
            if result is not None:
                n_ok += 1
        return 0 if n_ok > 0 else 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
