"""Embed and index every corpus x strategy combination, then persist the vectors.

Parsing is already cached, so this is network-bound. Each combination is priced before it
runs and the spend guard is consulted per run, so an unexpected token count stops the job
rather than showing up on an invoice.

    uv run python scripts/build_indexes.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from layoutrag.budget import BudgetExceeded, SpendGuard
from layoutrag.cache import Cache
from layoutrag.chunkers import ContextualHeadingChunker, FixedChunker, ParentDocChunker
from layoutrag.embedders import OpenAIEmbedder
from layoutrag.embedders.base import PRICE_PER_MILLION
from layoutrag.parsers import PdfiumFontSizeParser, PdfiumParser
from layoutrag.pipeline import build_index

OUT_DIR = Path("data/indexes")
MODEL = "text-embedding-3-small"

STRATEGIES = {
    "fixed": FixedChunker(),
    "fixed-no-overlap": FixedChunker(overlap_ratio=0.0),
    "contextual-heading": ContextualHeadingChunker(),
    "parent-doc": ParentDocChunker(),
}


def corpora() -> dict[str, tuple[list[Path], object]]:
    cuad = sorted(
        p
        for p in Path("data/cuad/CUAD_v1/full_contract_pdf").rglob("*")
        if p.suffix.lower() == ".pdf"
    )
    nist = sorted(Path("data/nist").glob("*.pdf"))
    # Contracts carry no heading typography, so the structure parser falls back anyway.
    return {"cuad": (cuad, PdfiumParser()), "nist": (nist, PdfiumFontSizeParser())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["cuad", "nist"], help="limit to one corpus")
    ap.add_argument("--max-usd", type=float, default=1.0, help="per-run ceiling")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = Cache()
    guard = SpendGuard(max_usd_per_run=args.max_usd)

    sets = corpora()
    if args.corpus:
        sets = {args.corpus: sets[args.corpus]}

    started = time.perf_counter()
    total_usd = 0.0

    for corpus, (paths, parser) in sets.items():
        for name, chunker in STRATEGIES.items():
            target = OUT_DIR / f"{corpus}__{name}.pkl"
            if target.exists():
                print(f"{corpus}/{name}: already built, skipping")
                continue

            print(f"\n=== {corpus} / {name} ===", flush=True)
            embedder = OpenAIEmbedder(model=MODEL, guard=guard)
            try:
                index, report = build_index(
                    paths,
                    parser,
                    chunker,
                    embedder,
                    cache=cache,  # type: ignore[arg-type]
                )
            except BudgetExceeded as exc:
                print(f"\nSTOPPED: {exc}", file=sys.stderr)
                return 1

            if index is None:
                print("  no chunks produced")
                continue

            spent = report.tokens_embedded / 1e6 * PRICE_PER_MILLION[MODEL]
            total_usd += spent
            print(f"  {report.describe()}", flush=True)

            with target.open("wb") as fh:
                pickle.dump({"chunks": index.chunks, "vectors": index.vectors}, fh, protocol=5)
            print(f"  wrote {target} ({target.stat().st_size / 1e6:.0f} MB)", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\nbuilt in {elapsed / 60:.1f} min, spent ~${total_usd:.4f}")
    print(f"ledger total: ${guard.history().total_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
