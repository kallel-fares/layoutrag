"""Vector index and retrieval.

Exact search, deliberately, rather than an approximate index.

Qdrant and friends trade recall for speed via ANN, and that trade is a confound here: an
arm could lose because the index missed a neighbour rather than because the chunking was
worse, and the two are indistinguishable in the results. At this corpus size the trade buys
nothing anyway — 13k chunks at 1536 dimensions is ~80 MB and a full scan is milliseconds.

Retrieval is scored at two budgets off the same search, because they answer different
questions and disagreeing is the interesting case:

``top-k``
    the conventional k=10, comparable with published numbers.

``token budget``
    fill ~4000 tokens of returned context, whatever k that implies.

Fixed k quietly rewards large chunks: an arm returning 2000-token sections hands the
generator ten times the text of one returning 200-token windows, and scores it as an equal
result. The token budget is what makes "concentrated beats diluted" a number rather than a
caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import count_tokens

DEFAULT_K = 10
DEFAULT_TOKEN_BUDGET = 4000


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    """Exact cosine similarity over normalised vectors."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        self.chunks = chunks
        # Normalising once turns every later cosine similarity into a plain dot product.
        self.vectors = _normalise(np.asarray(vectors, dtype=np.float32))
        self._token_cache: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes)

    def search(self, query_vector: np.ndarray, k: int = DEFAULT_K) -> list[Hit]:
        if not self.chunks:
            return []

        query = _normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ query

        k = min(k, len(self.chunks))
        # argpartition finds the top k without sorting the whole corpus; only those k are
        # then sorted. Matters once this runs for every query in a corpus.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top)
        ]

    def search_token_budget(
        self,
        query_vector: np.ndarray,
        budget: int = DEFAULT_TOKEN_BUDGET,
        max_k: int = 200,
    ) -> list[Hit]:
        """Retrieve until ``budget`` tokens of returned text are filled.

        Counts ``return_text``, since that is what a generator would actually receive.
        Deduplicates by returned text: parent-doc emits many children sharing one parent, and
        counting that parent repeatedly would exhaust the budget on a single section.
        """
        hits = self.search(query_vector, k=min(max_k, len(self.chunks)))

        kept: list[Hit] = []
        seen: set[str] = set()
        used = 0

        for hit in hits:
            text = hit.chunk.return_text
            if text in seen:
                continue
            size = self._tokens_of(hit.chunk)
            if kept and used + size > budget:
                break
            kept.append(hit)
            seen.add(text)
            used += size
            if used >= budget:
                break

        return kept

    def _tokens_of(self, chunk: Chunk) -> int:
        cached = self._token_cache.get(chunk.chunk_id)
        if cached is None:
            cached = count_tokens(chunk.return_text)
            self._token_cache[chunk.chunk_id] = cached
        return cached


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Zero vectors would divide by zero; leaving them at zero makes them score 0 against
    # everything, which is the right behaviour for an empty chunk.
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)
