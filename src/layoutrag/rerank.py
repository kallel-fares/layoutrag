"""Reranking — the stage between retrieval and generation.

A vector search scores a query against a chunk by comparing two embeddings that were
computed independently. A cross-encoder reads the query and the chunk *together* and scores
the pair directly, which is far more accurate and far too slow to run over a whole corpus.

So the pattern is: retrieve wide and cheap, rerank narrow and expensive. Pull 50 candidates
by vector similarity, then have the cross-encoder reorder them and keep the top 10.

This runs locally. No API, no key, no per-query cost — which matters because it sits in the
request path, unlike everything in the ingest pipeline.

It is also the stage most likely to change the conclusions drawn elsewhere. A cross-encoder
reads text rather than vectors, so it may recover from a poor chunking choice that vector
search alone could not — in which case chunking findings only apply to pipelines without
one.

**Rerank with the user's question, not the retrieval query.** These are usually the same
variable in an implementation and should not be. A retrieval query is often augmented —
with a document title, conversation history, a rewritten form — to help the vector stage
find the right neighbourhood. A cross-encoder scores the pair directly, so that augmentation
becomes boilerplate matching every candidate equally and drowning the actual question.

Measured on the NIST corpus, nDCG@10:

    no reranking                        0.436
    reranking on the retrieval query    0.346    <- worse than not reranking
    reranking on the user's question    0.570

Same model, same candidates, same depth. A 22-point swing decided by which string is passed
in.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from layoutrag.index import Hit

# 22M parameters, ~90 MB. Chosen over bge-reranker-base (278M) because it sits in the query
# path: measured planning estimates put the larger model at ~80 min per eval run against
# ~6 min for this one, for a quality difference that does not justify query latency.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Candidates pulled from vector search before reranking. Deeper finds more that the vector
# stage ranked poorly, and costs linearly more cross-encoder work.
DEFAULT_DEPTH = 50


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Reorder ``hits`` by relevance to ``query`` and return the best ``top_k``."""
        ...


class NoReranker:
    """The control arm. Keeps vector order, so 'no reranking' is a real configuration
    rather than a missing stage."""

    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker:
    """Scores each (query, chunk) pair with a cross-encoder."""

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        self.name = f"cross-encoder:{model.rsplit('/', 1)[-1]}"
        self.model_id = model
        self.batch_size = batch_size
        self._model: Any | None = None

    def _get_model(self) -> Any:
        # Lazy: importing sentence-transformers pulls in torch, and nothing should pay that
        # for merely importing this module.
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []

        model = self._get_model()
        # Scored against the returned text, matching how relevance is judged. Scoring the
        # embedded text would rank contextual-heading on breadcrumbs the reader never sees.
        pairs = [(query, hit.chunk.return_text) for hit in hits]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [
            Hit(chunk=hits[i].chunk, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order[:top_k])
        ]
