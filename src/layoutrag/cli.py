"""Command line entry point.

Two commands, both of which report before they act:

``layoutrag cost``
    price a corpus without touching the API.

``layoutrag query``
    index a folder under several chunking strategies and show, side by side, what each one
    retrieves for a question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layoutrag.cache import Cache
from layoutrag.chunkers import ContextualHeadingChunker, FixedChunker, ParentDocChunker
from layoutrag.chunkers.base import Chunker
from layoutrag.embedders.base import PRICE_PER_MILLION
from layoutrag.parsers import PdfiumFontSizeParser, PdfiumParser
from layoutrag.parsers.base import Parser
from layoutrag.pipeline import build_index

STRATEGIES: dict[str, Chunker] = {
    "fixed": FixedChunker(),
    "fixed-no-overlap": FixedChunker(overlap_ratio=0.0),
    "contextual-heading": ContextualHeadingChunker(),
    "parent-doc": ParentDocChunker(),
}

DEFAULT_STRATEGIES = ["fixed", "contextual-heading", "parent-doc"]


def _pdfs(folder: Path, limit: int | None) -> list[Path]:
    found = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".pdf")
    return found[:limit] if limit else found


def _parser(name: str) -> Parser:
    return PdfiumParser() if name == "pypdfium2" else PdfiumFontSizeParser()


def _embedder(local: bool):  # type: ignore[no-untyped-def]
    if local:
        from layoutrag.embedders import LocalEmbedder

        return LocalEmbedder()
    from layoutrag.embedders import OpenAIEmbedder

    return OpenAIEmbedder()


class _DryEmbedder:
    name = "text-embedding-3-small"
    model = "text-embedding-3-small"
    dimensions = 1536

    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        raise AssertionError("dry run must not embed")


def cmd_cost(args: argparse.Namespace) -> int:
    paths = _pdfs(Path(args.folder), args.limit)
    if not paths:
        print(f"No PDFs under {args.folder}", file=sys.stderr)
        return 1

    cache = Cache()
    total_usd = 0.0
    print(f"{len(paths)} PDFs\n")
    print(f"{'strategy':20} {'chunks':>8} {'tokens':>12} {'USD':>9}")
    print("-" * 52)

    for name in args.strategies:
        _, report = build_index(
            paths,
            _parser(args.parser),
            STRATEGIES[name],
            _DryEmbedder(),
            cache=cache,
            dry_run=True,
        )
        usd = report.tokens_embedded / 1e6 * PRICE_PER_MILLION["text-embedding-3-small"]
        total_usd += usd
        print(f"{name:20} {report.chunks:8,} {report.tokens_embedded:12,} ${usd:8.4f}")

    print("-" * 52)
    print(f"{'TOTAL':20} {'':8} {'':12} ${total_usd:8.4f}")
    print("\nNothing was sent to any API.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    paths = _pdfs(Path(args.folder), args.limit)
    if not paths:
        print(f"No PDFs under {args.folder}", file=sys.stderr)
        return 1

    cache = Cache()
    embedder = _embedder(args.local)
    indexes = {}

    for name in args.strategies:
        print(f"indexing with {name} ...", flush=True)
        index, report = build_index(
            paths, _parser(args.parser), STRATEGIES[name], embedder, cache=cache
        )
        print(f"  {report.describe()}", flush=True)
        if index is not None:
            indexes[name] = index

    query_vector = embedder.embed([args.question])[0]

    for name, index in indexes.items():
        print(f"\n=== {name} ===")
        for hit in index.search(query_vector, k=args.k):
            path = " > ".join(hit.chunk.heading_path)
            head = f"  [{hit.score:.3f}] {path}" if path else f"  [{hit.score:.3f}]"
            print(head)
            print(f"      {hit.chunk.return_text[:200].strip()}...")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="layoutrag")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, handler in (("cost", cmd_cost), ("query", cmd_query)):
        p = sub.add_parser(name)
        p.add_argument("folder")
        p.add_argument("--limit", type=int)
        p.add_argument("--parser", default="pypdfium2-fontsize")
        p.add_argument(
            "--strategies", nargs="+", default=DEFAULT_STRATEGIES, choices=list(STRATEGIES)
        )
        p.set_defaults(func=handler)
        if name == "query":
            p.add_argument("question")
            p.add_argument("-k", type=int, default=3)
            p.add_argument("--local", action="store_true", help="no API, no key")

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
