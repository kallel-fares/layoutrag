"""Reranker contract.

The cross-encoder itself is a downloaded model, so what is tested here is the wiring around
it: ordering, truncation, rank renumbering, and that the no-op arm is a genuine control.
"""

from __future__ import annotations

from layoutrag.chunk_type import Chunk
from layoutrag.index import Hit
from layoutrag.rerank import CrossEncoderReranker, NoReranker


def _hits(n: int) -> list[Hit]:
    return [
        Hit(
            chunk=Chunk(doc_id="d", chunk_id=f"c{i}", raw_text=f"chunk {i}"),
            score=1.0 - i / 10,
            rank=i,
        )
        for i in range(n)
    ]


class _ReverseModel:
    """Stands in for the cross-encoder: scores later candidates higher, so a correct
    implementation visibly reorders rather than passing vector order through."""

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [float(i) for i in range(len(pairs))]


def test_no_reranker_preserves_order_and_truncates() -> None:
    hits = _hits(10)
    kept = NoReranker().rerank("q", hits, top_k=3)
    assert [h.chunk.chunk_id for h in kept] == ["c0", "c1", "c2"]


def test_cross_encoder_reorders_by_its_own_scores() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _ReverseModel()

    kept = reranker.rerank("q", _hits(5), top_k=3)
    # The stand-in scores the last candidate highest, so the vector ordering must be undone.
    assert [h.chunk.chunk_id for h in kept] == ["c4", "c3", "c2"]


def test_ranks_are_renumbered_after_reranking() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _ReverseModel()

    kept = reranker.rerank("q", _hits(5), top_k=3)
    # Metrics key off rank, so stale ranks from the vector stage would corrupt nDCG and MRR.
    assert [h.rank for h in kept] == [0, 1, 2]


def test_scores_come_from_the_reranker_not_the_vector_stage() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _ReverseModel()

    kept = reranker.rerank("q", _hits(5), top_k=2)
    assert kept[0].score == 4.0


def test_empty_input() -> None:
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []
    assert NoReranker().rerank("q", [], top_k=5) == []


def test_top_k_larger_than_candidates() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = _ReverseModel()
    assert len(reranker.rerank("q", _hits(3), top_k=10)) == 3
