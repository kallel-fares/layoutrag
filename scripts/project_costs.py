"""Project what every planned run would cost, without spending anything.

Chunking is free and produces the exact token counts embedding would be billed for, so a
complete cost model can be built with zero API calls. This is what turns "roughly a dollar"
into a table you can decide against.

    uv run python scripts/project_costs.py            # sample, fast
    uv run python scripts/project_costs.py --full     # whole corpora
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from layoutrag.cache import Cache
from layoutrag.chunkers import ContextualHeadingChunker, FixedChunker, ParentDocChunker
from layoutrag.embedders.base import PRICE_PER_MILLION
from layoutrag.parsers import PdfiumFontSizeParser, PdfiumParser
from layoutrag.pipeline import build_index

MODEL = "text-embedding-3-small"


class _NullEmbedder:
    name = MODEL
    model = MODEL
    dimensions = 1536

    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        raise AssertionError("dry run must never embed")


def corpora(sample: int | None) -> dict[str, list[Path]]:
    cuad = sorted(
        p
        for p in Path("data/cuad/CUAD_v1/full_contract_pdf").rglob("*")
        if p.suffix.lower() == ".pdf"
    )
    nist = sorted(Path("data/nist").glob("*.pdf"))
    if sample:
        cuad, nist = cuad[:sample], nist[:sample]
    return {"cuad": cuad, "nist": nist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run over the whole corpora")
    ap.add_argument("--sample", type=int, default=25)
    args = ap.parse_args()

    sample = None if args.full else args.sample
    sets = corpora(sample)
    cache = Cache()

    strategies = [
        ("fixed", FixedChunker()),
        ("fixed-no-overlap", FixedChunker(overlap_ratio=0.0)),
        ("contextual-heading", ContextualHeadingChunker()),
        ("parent-doc", ParentDocChunker()),
    ]

    print(f"model {MODEL} at ${PRICE_PER_MILLION[MODEL]}/M tokens")
    print(f"{'corpus':6} {'strategy':20} {'docs':>5} {'chunks':>8} {'tokens':>12} {'USD':>9}")
    print("-" * 66)

    grand_usd = 0.0
    grand_tokens = 0
    started = time.perf_counter()

    for corpus, paths in sets.items():
        if not paths:
            print(f"{corpus}: no documents found")
            continue
        # Contracts carry no heading typography, so the structure-aware parser would fall
        # back anyway; standards get it.
        parser = PdfiumParser() if corpus == "cuad" else PdfiumFontSizeParser()

        for label, chunker in strategies:
            _, report = build_index(
                paths, parser, chunker, _NullEmbedder(), cache=cache, dry_run=True
            )
            usd = report.tokens_embedded / 1e6 * PRICE_PER_MILLION[MODEL]
            grand_usd += usd
            grand_tokens += report.tokens_embedded
            print(
                f"{corpus:6} {label:20} {report.documents:5} {report.chunks:8,} "
                f"{report.tokens_embedded:12,} ${usd:8.4f}"
            )

    scope = "FULL CORPORA" if args.full else f"SAMPLE of {sample} docs/corpus"
    print("-" * 66)
    print(f"{scope}: {grand_tokens:,} tokens -> ${grand_usd:.4f}")
    if not args.full:
        print("Run with --full for the real projection.")
    print(f"(projected in {time.perf_counter() - started:.1f}s, zero API calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
