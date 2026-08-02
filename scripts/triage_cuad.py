"""Triage the CUAD corpus before building anything on it.

Answers two Phase 1 questions:

1. How many contracts are scanned rather than digitally native? OCR is out of scope, so
   those have to be excluded and *counted* — a corpus whose usable size is unknown makes
   every later number unreadable.

2. How fast is pypdfium2 on this machine, per page? Every cost projection depends on a
   measured rate rather than a guess.

    uv run python scripts/triage_cuad.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

PDF_ROOT = Path("data/cuad/CUAD_v1/full_contract_pdf")
OUT = Path("data/cuad/triage.json")

# Below this many extracted characters per page, a PDF has no usable text layer.
# Digitally-native contracts run in the thousands; a scanned page yields near zero.
SCANNED_CHARS_PER_PAGE = 100


def triage_one(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        pdf = pdfium.PdfDocument(path)
        pages = len(pdf)
        chars = 0
        for page in pdf:
            textpage = page.get_textpage()
            chars += len(textpage.get_text_bounded())
            textpage.close()
            page.close()
        pdf.close()
    except Exception as exc:
        return {
            "path": str(path),
            "failed": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "pages": 0,
            "chars": 0,
            "seconds": time.perf_counter() - started,
        }

    elapsed = time.perf_counter() - started
    per_page = chars / pages if pages else 0
    return {
        "path": str(path),
        "failed": False,
        "reason": "",
        "pages": pages,
        "chars": chars,
        "chars_per_page": round(per_page, 1),
        "scanned": per_page < SCANNED_CHARS_PER_PAGE,
        "seconds": elapsed,
    }


def main() -> int:
    pdfs = sorted(p for p in PDF_ROOT.rglob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs under {PDF_ROOT} — run scripts/fetch_cuad.py first")
        return 1

    print(f"Triaging {len(pdfs)} contracts with pypdfium2\n")
    results = []
    for i, path in enumerate(pdfs, 1):
        results.append(triage_one(path))
        if i % 50 == 0 or i == len(pdfs):
            print(f"  {i}/{len(pdfs)}")

    ok = [r for r in results if not r["failed"]]
    failed = [r for r in results if r["failed"]]
    scanned = [r for r in ok if r["scanned"]]
    usable = [r for r in ok if not r["scanned"]]

    total_pages = sum(r["pages"] for r in ok)
    total_seconds = sum(r["seconds"] for r in results)
    usable_pages = sum(r["pages"] for r in usable)
    usable_chars = sum(r["chars"] for r in usable)

    print("\n" + "=" * 60)
    print(f"total contracts     {len(results)}")
    print(f"  parse failures    {len(failed)}")
    print(f"  scanned (no text) {len(scanned)}")
    print(f"  usable            {len(usable)}  ({100 * len(usable) / len(results):.1f}%)")
    print()
    print(f"usable pages        {usable_pages}")
    print(f"  median pages/doc  {sorted(r['pages'] for r in usable)[len(usable) // 2]}")
    print(f"usable characters   {usable_chars / 1e6:.1f} M")
    print(f"  ~tokens (chars/4) {usable_chars / 4 / 1e6:.2f} M")
    print()
    print("pypdfium2 measured on this machine:")
    print(f"  {total_seconds:.1f} s for {total_pages} pages")
    print(f"  {1000 * total_seconds / total_pages:.1f} ms/page")
    print(f"  {total_seconds / len(results):.3f} s/doc")

    if failed:
        print("\nfailures:")
        for r in failed[:10]:
            print(f"  {Path(r['path']).name}: {r['reason']}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nper-document detail written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
