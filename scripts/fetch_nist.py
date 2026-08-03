"""Fetch NIST Special Publication 800-series PDFs — the structure corpus.

Why a second corpus: CUAD's PDFs are all HTML→PDF conversions (Aspose, EVO), so they carry
essentially no heading typography — measured at 0.0-0.5% of text objects set in larger type
than the body. Nothing can recover structure that isn't there, which makes CUAD useless for
comparing parsers. NIST SPs are real publisher PDFs: 9-13 distinct font sizes and up to 19%
of text objects in larger type.

US Government work, public domain. Downloads to data/nist/, which is gitignored.

    uv run python scripts/fetch_nist.py --limit 60

Three quirks of the NIST sites, all handled here:
  - PDF URLs are not derivable from the SP number: they variously live under
    /SpecialPublications/NIST.SP.800-53r5.pdf and /Legacy/SP/nistspecialpublication800-100.pdf,
    with letter suffixes and update markers. So the canonical link is scraped from each
    publication's CSRC detail page rather than guessed.
  - nvlpubs 404s on HEAD but serves 206 on a ranged GET.
  - Concurrent requests trip Cloudflare ("error code: 1015") and silently write a 17-byte
    error body in place of the PDF. Everything here is serial with a delay, and every
    download is verified to start with %PDF.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

LISTING = "https://csrc.nist.gov/publications/sp800"
CSRC = "https://csrc.nist.gov"
DATA_DIR = Path("data/nist")

DETAIL_RE = re.compile(r"/pubs/sp/800/[0-9a-z/-]+/final")
PDF_RE = re.compile(r"https://nvlpubs\.nist\.gov/[^\"'\s]+\.pdf")

DELAY_SECONDS = 0.4


def _get(url: str, timeout: int = 30) -> str:
    proc = subprocess.run(
        ["curl", "-sL", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return proc.stdout


def detail_pages() -> list[str]:
    html = _get(LISTING)
    if not html:
        return []
    return sorted({m.group(0) for m in DETAIL_RE.finditer(html)})


def pdf_url_for(detail_path: str) -> str | None:
    html = _get(CSRC + detail_path)
    match = PDF_RE.search(html)
    return match.group(0) if match else None


def download(url: str) -> Path | None:
    dest = DATA_DIR / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-o", str(dest), url],
        capture_output=True,
    )
    if not dest.exists():
        return None

    # Cloudflare rate-limit bodies are tiny and are not PDFs. Catch them here rather than
    # discovering a corpus of 17-byte "error code: 1015" files at parse time.
    with dest.open("rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        dest.unlink(missing_ok=True)
        return None
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="how many PDFs to download")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading the SP 800 listing from {LISTING}")
    pages = detail_pages()
    if not pages:
        print("Could not read the listing — the page structure may have changed.", file=sys.stderr)
        return 1
    print(f"  {len(pages)} final publications listed\n")

    print(f"Resolving PDF links (serial, {DELAY_SECONDS}s apart to stay under rate limits)")
    downloaded = 0
    skipped = 0
    for i, page in enumerate(pages, 1):
        if downloaded >= args.limit:
            break

        url = pdf_url_for(page)
        time.sleep(DELAY_SECONDS)
        if not url:
            skipped += 1
            continue

        path = download(url)
        time.sleep(DELAY_SECONDS)
        if path:
            downloaded += 1
        else:
            skipped += 1

        if downloaded and downloaded % 10 == 0:
            print(f"  {downloaded} downloaded ({skipped} skipped), {i} pages visited")

    on_disk = sorted(DATA_DIR.glob("*.pdf"))
    size_mb = sum(p.stat().st_size for p in on_disk) / 1e6
    print(f"\n{len(on_disk)} PDFs in {DATA_DIR}, {size_mb:.0f} MB ({skipped} skipped)")
    return 0 if on_disk else 1


if __name__ == "__main__":
    raise SystemExit(main())
