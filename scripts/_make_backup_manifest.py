"""Generate D:\\backup\\2026-05-01\\manifest.md5 — one MD5 per backed-up file.

Plan acceptance criterion: "Verify byte-clean. manifest.md5 verifies byte-clean."

Format: lines of ``<md5_hex>  <relative_path>`` (two spaces, GNU md5sum-compatible).
Re-running the manifest after a crash is safe — files are streamed and read in
binary mode; no temp files are written.
"""
from __future__ import annotations
import hashlib
import sys
import time
from pathlib import Path

BACKUP_ROOT = Path(r"D:\backup\2026-05-01")
SKIP_BASENAMES = {"manifest.md5"}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    if not BACKUP_ROOT.is_dir():
        print(f"backup root missing: {BACKUP_ROOT}", file=sys.stderr)
        return 1

    files = sorted(p for p in BACKUP_ROOT.rglob("*")
                   if p.is_file() and p.name not in SKIP_BASENAMES
                   and not p.name.endswith(".log"))
    print(f"Hashing {len(files)} files in {BACKUP_ROOT}...")

    out_path = BACKUP_ROOT / "manifest.md5"
    t0 = time.perf_counter()
    n_bytes = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, p in enumerate(files, 1):
            digest = md5_of(p)
            rel = p.relative_to(BACKUP_ROOT).as_posix()
            fh.write(f"{digest}  {rel}\n")
            n_bytes += p.stat().st_size
            if i % 250 == 0 or i == len(files):
                elapsed = time.perf_counter() - t0
                rate_mb = n_bytes / 1e6 / max(elapsed, 1e-6)
                print(f"  [{i}/{len(files)}]  {rel}  ({rate_mb:.1f} MB/s)")

    elapsed = time.perf_counter() - t0
    print(f"\nWrote {out_path}")
    print(f"  {len(files)} files  {n_bytes/1e9:.2f} GB  in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
