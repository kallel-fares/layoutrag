"""Fetch CUAD — 510 real commercial contracts with lawyer-annotated clause spans.

CC BY 4.0, cleared for commercial use. The Atticus Project.
https://www.atticusprojectai.org/cuad

Downloads to data/cuad/, which is gitignored. The corpus is fetched, never committed.

    uv run python scripts/fetch_cuad.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
EXPECTED_BYTES = 105_883_672

DATA_DIR = Path("data/cuad")
ARCHIVE = DATA_DIR / "CUAD_v1.zip"


def fetch() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.exists() and ARCHIVE.stat().st_size == EXPECTED_BYTES:
        print(f"Archive already present: {ARCHIVE}")
    else:
        # curl rather than urllib: python.org builds on macOS ship without a CA bundle,
        # so urllib fails TLS verification out of the box. curl uses the system keychain.
        print(f"Downloading CUAD ({EXPECTED_BYTES / 1e6:.0f} MB) from Zenodo")
        subprocess.run(
            ["curl", "-fL", "--progress-bar", "-o", str(ARCHIVE), CUAD_URL],
            check=True,
        )

        size = ARCHIVE.stat().st_size
        if size != EXPECTED_BYTES:
            ARCHIVE.unlink()
            raise RuntimeError(f"expected {EXPECTED_BYTES} bytes, got {size} — download truncated")

    extracted = DATA_DIR / "CUAD_v1"
    if extracted.exists():
        print(f"Already extracted: {extracted}")
    else:
        print("Extracting")
        with zipfile.ZipFile(ARCHIVE) as zf:
            zf.extractall(DATA_DIR)

    return extracted


def summarise(root: Path) -> None:
    pdfs = sorted(root.rglob("*.pdf"))
    total = sum(p.stat().st_size for p in pdfs)
    print(f"\n{len(pdfs)} PDFs, {total / 1e6:.0f} MB total")

    if pdfs:
        sizes = sorted(p.stat().st_size for p in pdfs)
        print(f"  median size {sizes[len(sizes) // 2] / 1e3:.0f} kB")

    for name in ("CUAD_v1.json", "master_clauses.csv"):
        for found in root.rglob(name):
            print(f"  annotations: {found.relative_to(root)} ({found.stat().st_size / 1e6:.1f} MB)")
            break

    print("\nTop-level layout:")
    for child in sorted(root.iterdir())[:12]:
        marker = "/" if child.is_dir() else ""
        print(f"  {child.name}{marker}")


def main() -> int:
    if shutil.disk_usage(".").free < 1_000_000_000:
        print("Less than 1 GB free — aborting rather than filling the disk.", file=sys.stderr)
        return 1

    root = fetch()
    summarise(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
