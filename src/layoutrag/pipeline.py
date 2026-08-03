"""End-to-end pipeline: parse, chunk, embed, index.

Parsing runs in parallel across documents because it is CPU-bound and independent per file.
That matters most for docling at ~1 s/page, where a 60-document corpus goes from an hour to
minutes on a laptop with several cores.

Everything is cached by content hash, so indexing one corpus under four chunking strategies
parses each document once rather than four times. Without that, the side-by-side comparison
the whole tool exists to show would be four times slower than it needs to be.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from layoutrag.blocks import ParsedDoc
from layoutrag.cache import Cache, hash_file, hash_params
from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import Chunker
from layoutrag.embedders.base import Embedder, estimate
from layoutrag.index import VectorIndex
from layoutrag.parsers.base import Parser


def _default_workers() -> int:
    # Leave a core free so the machine stays usable during a long parse.
    return max(1, (os.cpu_count() or 4) - 1)


@dataclass
class IndexReport:
    """What an indexing run actually did. Every number here is measured, not estimated."""

    documents: int = 0
    parse_failures: int = 0
    chunks: int = 0
    degraded_chunks: int = 0
    pages: int = 0
    parse_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    tokens_embedded: int = 0
    usd_spent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.parse_seconds + self.chunk_seconds + self.embed_seconds

    def describe(self) -> str:
        rate = self.pages / self.parse_seconds if self.parse_seconds else 0.0
        return (
            f"{self.documents} docs / {self.pages} pages -> {self.chunks} chunks "
            f"({self.degraded_chunks} degraded, {self.parse_failures} parse failures)\n"
            f"  parse {self.parse_seconds:6.1f}s ({rate:6.1f} pages/s)  "
            f"chunk {self.chunk_seconds:5.1f}s  embed {self.embed_seconds:6.1f}s\n"
            f"  cache {self.cache_hits} hits / {self.cache_misses} misses  "
            f"tokens {self.tokens_embedded:,}  spent ${self.usd_spent:.4f}"
        )


def _parse_one(job: tuple[Parser, Path]) -> ParsedDoc:
    """Module-level so it can be sent to a worker process."""
    parser, path = job
    return parser.parse(path)


def parse_corpus(
    paths: Sequence[Path],
    parser: Parser,
    cache: Cache | None = None,
    workers: int | None = None,
) -> tuple[list[ParsedDoc], float]:
    """Parse documents in parallel, reusing cached parses.

    Processes rather than threads. PDFium is not thread-safe, and calling it concurrently
    from threads aborts the interpreter outright (SIGABRT) rather than raising something
    catchable. Parsing is CPU-bound anyway, so threads would not have helped past the GIL.

    The cache is read and written in the parent, so workers stay stateless and only the
    documents that actually need parsing are dispatched.
    """
    cache = cache if cache is not None else Cache()
    params = hash_params(parser=parser.name)
    started = time.perf_counter()

    hashes = [hash_file(p) for p in paths]
    results: dict[int, ParsedDoc] = {}
    pending: list[tuple[int, Path]] = []

    for i, (path, content) in enumerate(zip(paths, hashes, strict=True)):
        cached = cache.get("parse", content, params)
        if cached is not None:
            results[i] = cached
        else:
            pending.append((i, path))

    if pending:
        count = workers or _default_workers()
        # One document is not worth the cost of spawning a process.
        if count == 1 or len(pending) == 1:
            parsed_docs = [_parse_one((parser, path)) for _, path in pending]
        else:
            with ProcessPoolExecutor(max_workers=count) as pool:
                parsed_docs = list(pool.map(_parse_one, [(parser, p) for _, p in pending]))

        for (i, _), parsed in zip(pending, parsed_docs, strict=True):
            results[i] = parsed
            cache.put("parse", hashes[i], params, parsed)

    return [results[i] for i in range(len(paths))], time.perf_counter() - started


def build_index(
    paths: Sequence[Path],
    parser: Parser,
    chunker: Chunker,
    embedder: Embedder,
    cache: Cache | None = None,
    workers: int | None = None,
    dry_run: bool = False,
) -> tuple[VectorIndex | None, IndexReport]:
    """Parse, chunk, embed, and index a corpus.

    With ``dry_run`` set, everything up to embedding runs and the cost is reported without
    a single API call. That is the cheap way to find out what a full corpus would cost:
    chunking is free, and the token count it produces is the whole basis of the estimate.
    """
    cache = cache if cache is not None else Cache()
    report = IndexReport(documents=len(paths))

    docs, report.parse_seconds = parse_corpus(paths, parser, cache, workers)
    for doc in docs:
        report.pages += doc.page_count
        if doc.parse_failed:
            report.parse_failures += 1
            report.failures.append(f"{doc.doc_id}: {doc.failure_reason}")

    started = time.perf_counter()
    chunks: list[Chunk] = []
    chunk_params = hash_params(chunker=chunker.name, parser=parser.name)
    for doc in docs:
        if doc.parse_failed:
            continue
        key = f"{doc.doc_id}:{len(doc.blocks)}"
        produced = cache.get("chunk", key, chunk_params)
        if produced is None:
            produced = chunker.chunk(doc)
            cache.put("chunk", key, chunk_params, produced)
        chunks.extend(produced)
    report.chunk_seconds = time.perf_counter() - started
    report.chunks = len(chunks)
    report.degraded_chunks = sum(1 for c in chunks if c.degraded)
    report.cache_hits, report.cache_misses = cache.hits, cache.misses

    texts = [c.embed_text for c in chunks]
    model = getattr(embedder, "model", embedder.name)
    report.tokens_embedded = estimate(texts, model).tokens

    if dry_run or not chunks:
        return None, report

    started = time.perf_counter()
    vectors = embedder.embed(texts)
    report.embed_seconds = time.perf_counter() - started
    report.usd_spent = float(getattr(embedder, "usd_spent", 0.0))

    return VectorIndex(chunks, np.asarray(vectors)), report
