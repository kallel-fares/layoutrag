"""Does an ML layout model find headings that font metrics cannot?

The question the NIST corpus was added for, and the one that decides whether a client needs
an ingestion box with torch on it or one that fits in a couple of hundred megabytes.

pypdfium2-fontsize infers headings by clustering text-object font sizes. It costs 4 ms per
page. docling runs a layout model per page at ~1 s — 258x more — and can in principle see
what typography does not express, such as a section header set bold at body size. Earlier
sampling suggested it finds 2-7x more headings; whether those extra headings improve
retrieval is a different question, and only this comparison answers it.

Only the two heading-sensitive strategies are run. fixed and fixed-no-overlap ignore
structure entirely, so parsing them twice would cost money to produce identical numbers.

    uv run python scripts/parser_comparison.py --parse-only    # free, slow
    uv run python scripts/parser_comparison.py                 # embeds, ~$0.09
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from layoutrag.blocks import BlockType
from layoutrag.budget import BudgetExceeded, SpendGuard
from layoutrag.cache import Cache
from layoutrag.chunkers import ContextualHeadingChunker, FixedChunker
from layoutrag.embedders import OpenAIEmbedder
from layoutrag.parsers import DoclingParser, PdfiumFontSizeParser
from layoutrag.pipeline import build_index, parse_corpus

OUT_DIR = Path("data/indexes")
NIST = Path("data/nist")


def heading_stats(label: str, paths: list[Path], parser: object, cache: Cache) -> None:
    started = time.perf_counter()
    docs, _ = parse_corpus(paths, parser, cache)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started

    pages = sum(d.page_count for d in docs)
    headings = sum(1 for d in docs for b in d.blocks if b.type is BlockType.HEADING)
    structured = sum(1 for d in docs if d.has_structure)
    failed = sum(1 for d in docs if d.parse_failed)

    rate = f"{elapsed / pages:.3f}" if pages else "n/a"
    print(
        f"{label:22} {len(docs):4} docs {pages:6} pages  {headings:6} headings  "
        f"{structured:3}/{len(docs)} with structure  {failed} failed  "
        f"{elapsed:7.1f}s ({rate} s/page)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-only", action="store_true", help="no embedding, no cost")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    paths = sorted(NIST.glob("*.pdf"))
    if args.limit:
        paths = paths[: args.limit]

    cache = Cache()
    print(f"{len(paths)} NIST documents\n")
    heading_stats("pypdfium2-fontsize", paths, PdfiumFontSizeParser(), cache)
    heading_stats("docling", paths, DoclingParser(), cache)

    if args.parse_only:
        print("\nparse-only: nothing embedded, nothing spent")
        return 0

    guard = SpendGuard()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only the heading-sensitive strategies. fixed ignores structure, so re-embedding it
    # from a second parse would pay for identical vectors.
    strategies = {
        "contextual-heading": ContextualHeadingChunker(),
        "fixed": FixedChunker(),
    }

    for name, chunker in strategies.items():
        target = OUT_DIR / f"nistdocling__{name}.pkl"
        if target.exists():
            print(f"\n{name}: already built, skipping")
            continue

        print(f"\n=== docling / {name} ===", flush=True)
        embedder = OpenAIEmbedder(guard=guard)
        try:
            index, report = build_index(paths, DoclingParser(), chunker, embedder, cache=cache)
        except BudgetExceeded as exc:
            print(f"\nSTOPPED: {exc}", file=sys.stderr)
            return 1

        if index is None:
            print("  no chunks")
            continue

        print(f"  {report.describe()}", flush=True)
        with target.open("wb") as fh:
            pickle.dump({"chunks": index.chunks, "vectors": index.vectors}, fh, protocol=5)
        print(f"  wrote {target}", flush=True)

    print(f"\nledger total: ${guard.history().total_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
