"""Build a contextual-llm index: enrich each chunk with a locator line, then embed.

Priced before anything is sent, guarded per run, cached per chunk. A rerun costs nothing for
chunks already enriched.

    uv run python scripts/build_enriched_index.py --corpus nist --limit 5   # try it small
    uv run python scripts/build_enriched_index.py --corpus nist
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from layoutrag.budget import BudgetExceeded, SpendGuard
from layoutrag.cache import Cache, hash_params
from layoutrag.chunk_type import Chunk
from layoutrag.chunkers import FixedChunker
from layoutrag.embedders import OpenAIEmbedder
from layoutrag.enrich import DEFAULT_WINDOW_TOKENS, ContextualEnricher, estimate
from layoutrag.index import VectorIndex
from layoutrag.parsers import PdfiumFontSizeParser, PdfiumParser
from layoutrag.pipeline import Progress, parse_corpus

OUT_DIR = Path("data/indexes")

CORPORA = {
    "nist": (Path("data/nist"), "*.pdf", PdfiumFontSizeParser),
    "cuad": (Path("data/cuad/CUAD_v1/full_contract_pdf"), "*", PdfiumParser),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), default="nist")
    ap.add_argument("--limit", type=int, help="first N documents only")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_TOKENS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    root, pattern, parser_cls = CORPORA[args.corpus]
    paths = sorted(p for p in root.rglob(pattern) if p.suffix.lower() == ".pdf")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"No PDFs under {root}", file=sys.stderr)
        return 1

    cache = Cache()
    docs, _ = parse_corpus(paths, parser_cls(), cache)
    documents = {d.doc_id: d.text for d in docs if not d.parse_failed}

    chunker = FixedChunker()
    chunks: list[Chunk] = []
    for doc in docs:
        if not doc.parse_failed:
            chunks.extend(chunker.chunk(doc))

    # Priced from the exact windows that will be sent, before a single request. Chunks
    # already enriched cost nothing on a rerun, so they are excluded from the projection.
    params = hash_params(enricher="contextual-llm", model="gpt-5-nano", window=args.window)
    todo = [c for c in chunks if cache.get("enrich", c.chunk_id, params) is None]
    projected = estimate(todo, documents, args.window)
    print(f"\n{len(paths)} documents, {len(chunks):,} chunks")
    print(f"window {args.window} tokens")
    print(f"projected: {projected.tokens_in:,} in + {projected.tokens_out:,} out")
    print(f"projected cost: ${projected.usd:.4f}")

    guard = SpendGuard()
    print(f"ledger so far: ${guard.history().total_usd:.4f} of ${guard.max_usd_total:.2f}")
    try:
        guard.check(projected.usd, label=f"enriching {len(chunks):,} chunks")
    except BudgetExceeded as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if not args.yes:
        print("\nRe-run with --yes to proceed.")
        return 0

    enricher = ContextualEnricher(window_tokens=args.window, guard=guard)
    bar = Progress(len(chunks), "enriching")

    def one(chunk: Chunk) -> Chunk:
        cached = cache.get("enrich", chunk.chunk_id, params)
        if cached is not None:
            bar.advance()
            return cached  # type: ignore[no-any-return]
        enriched = enricher.enrich(chunk, documents.get(chunk.doc_id, ""))
        cache.put("enrich", chunk.chunk_id, params, enriched)
        bar.advance()
        return enriched

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        enriched = list(pool.map(one, chunks))
    bar.finish()

    done = sum(1 for c in enriched if c.is_enriched)
    minutes = (time.perf_counter() - started) / 60
    print(f"  {done:,} of {len(enriched):,} enriched in {minutes:.1f} min")
    print(f"  actual enrichment cost: ${enricher.usd_spent:.4f}")

    if done:
        print("\n  sample locator lines:")
        for chunk in [c for c in enriched if c.is_enriched][:3]:
            print(f"    {chunk.embed_text.split(chr(10))[0][:110]}")

    embedder = OpenAIEmbedder(guard=guard)
    vectors = embedder.embed([c.embed_text for c in enriched])
    index = VectorIndex(enriched, np.asarray(vectors))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{args.corpus}__contextual-llm.pkl"
    with target.open("wb") as fh:
        pickle.dump({"chunks": index.chunks, "vectors": index.vectors}, fh, protocol=5)

    print(f"\nwrote {target}")
    print(f"embedding cost: ${embedder.usd_spent:.4f}")
    guard.commit()
    print(f"ledger total: ${guard.history().total_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
