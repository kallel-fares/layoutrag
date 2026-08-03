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
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from layoutrag.blocks import ParsedDoc
from layoutrag.cache import Cache, hash_file, hash_params
from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import Chunker
from layoutrag.embedders.base import Embedder, estimate
from layoutrag.index import VectorIndex
from layoutrag.parsers.base import Parser


def _default_workers(parser: Parser | None = None) -> int:
    """Worker count, bounded by how much memory *this parser* needs per process.

    One number cannot serve both parsers, and assuming it could is what made a laptop
    unusable. Each worker is a separate process with its own address space:

    - ``pypdfium2`` holds a document, tens of megabytes, so parallelism is bounded by cores.
    - ``docling`` holds torch and its layout models. Two workers measured at 3306 MB and
      2981 MB resident — 6.3 GB on an 8 GB machine, which drove swap to 20.6 GB of 21.5 GB
      and left the kernel thrashing at 103% CPU.

    So docling runs single-process by default. It is slower in theory and faster in
    practice, because a machine in swap does no useful work.

    ``LAYOUTRAG_WORKERS`` overrides, for a machine with the memory to spare.
    """
    override = os.environ.get("LAYOUTRAG_WORKERS")
    if override and override.isdigit() and int(override) > 0:
        return int(override)

    cores = max(1, (os.cpu_count() or 4) - 1)

    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 1

    per_worker = _memory_per_worker(parser)
    by_memory = int(total_bytes / per_worker) - 1
    return max(1, min(cores, by_memory))


def _memory_per_worker(parser: Parser | None) -> float:
    """Resident memory to budget per worker process, in bytes.

    Model-loading parsers are identified by name rather than by type so a client's own
    heavyweight parser can opt in without importing anything from here.
    """
    name = getattr(parser, "name", "") or ""
    if "docling" in name or "marker" in name or "unstructured" in name:
        return 6.0 * 1024**3  # measured ~3 GB resident, doubled for headroom
    return 1.5 * 1024**3


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


class Progress:
    """Single-line progress with an ETA, for work slow enough that silence looks like a hang.

    docling takes about a minute per document. Without this, a sixty-document parse is an
    hour of no output, which is indistinguishable from a crash — and that ambiguity is
    exactly what makes someone kill a run that was working.

    Writes to stderr and rewrites one line, so piping stdout to a file stays clean. Falls
    back to one line per item when not attached to a terminal, since carriage returns in a
    log file are unreadable.
    """

    def __init__(self, total: int, label: str = "parsing", stream: TextIO | None = None) -> None:
        self.total = total
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.done = 0
        self.started = time.perf_counter()
        self.interactive = hasattr(self.stream, "isatty") and self.stream.isatty()

    def advance(self, name: str = "") -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.started
        rate = elapsed / self.done
        remaining = rate * (self.total - self.done)
        pct = 100.0 * self.done / self.total if self.total else 100.0

        line = (
            f"  {self.label}: {self.done}/{self.total} ({pct:.0f}%) "
            f"· {_duration(elapsed)} elapsed · ~{_duration(remaining)} left"
        )
        if name:
            line += f" · {name[:38]}"

        if self.interactive:
            self.stream.write(f"\r{line:<110}")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.started
        message = f"  {self.label}: {self.done}/{self.total} in {_duration(elapsed)}"
        prefix = "\r" if self.interactive else ""
        self.stream.write(f"{prefix}{message:<110}\n")
        self.stream.flush()


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def parse_corpus(
    paths: Sequence[Path],
    parser: Parser,
    cache: Cache | None = None,
    workers: int | None = None,
    progress: bool = True,
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
        count = workers or _default_workers(parser)
        bar = Progress(len(pending), f"{parser.name}") if progress else None

        # One document is not worth the cost of spawning a process.
        if count == 1 or len(pending) == 1:
            for i, path in pending:
                parsed = _parse_one((parser, path))
                results[i] = parsed
                cache.put("parse", hashes[i], params, parsed)
                if bar:
                    bar.advance(path.stem)
        else:
            try:
                with ProcessPoolExecutor(max_workers=count) as pool:
                    # submit + as_completed rather than map, so progress reports when each
                    # document actually finishes instead of when the whole batch does.
                    futures = {
                        pool.submit(_parse_one, (parser, path)): (i, path) for i, path in pending
                    }
                    for future in as_completed(futures):
                        i, path = futures[future]
                        parsed = future.result()
                        results[i] = parsed
                        cache.put("parse", hashes[i], params, parsed)
                        if bar:
                            bar.advance(path.stem)
            except BrokenProcessPool:
                # macOS spawns rather than forks, so worker processes re-import the calling
                # module. From a REPL, a notebook, or a script piped through stdin there is
                # no module to re-import and the pool dies before doing any work. Parsing
                # serially is slower but always correct, and beats failing on the caller's
                # execution context.
                for i, path in pending:
                    if i in results:
                        continue
                    parsed = _parse_one((parser, path))
                    results[i] = parsed
                    cache.put("parse", hashes[i], params, parsed)
                    if bar:
                        bar.advance(path.stem)

        if bar:
            bar.finish()

    return [results[i] for i in range(len(paths))], time.perf_counter() - started


def build_index(
    paths: Sequence[Path],
    parser: Parser,
    chunker: Chunker,
    embedder: Embedder,
    cache: Cache | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> tuple[VectorIndex | None, IndexReport]:
    """Parse, chunk, embed, and index a corpus.

    With ``dry_run`` set, everything up to embedding runs and the cost is reported without
    a single API call. That is the cheap way to find out what a full corpus would cost:
    chunking is free, and the token count it produces is the whole basis of the estimate.
    """
    cache = cache if cache is not None else Cache()
    report = IndexReport(documents=len(paths))

    docs, report.parse_seconds = parse_corpus(paths, parser, cache, workers, progress)
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
