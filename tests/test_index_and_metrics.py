"""Index behaviour and metric arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from layoutrag.chunk_type import Chunk
from layoutrag.eval import GoldSpan, score_query, score_run
from layoutrag.eval.metrics import QueryResult
from layoutrag.index import Hit, VectorIndex


def _chunks(n: int, text: str = "chunk") -> list[Chunk]:
    return [Chunk(doc_id="d", chunk_id=f"c{i}", raw_text=f"{text} {i}") for i in range(n)]


def test_search_ranks_by_similarity() -> None:
    chunks = _chunks(3)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    index = VectorIndex(chunks, vectors)

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c2", "c1"]
    assert [h.rank for h in hits] == [0, 1, 2]
    assert hits[0].score > hits[1].score > hits[2].score


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="chunks but"):
        VectorIndex(_chunks(3), np.zeros((2, 4), dtype=np.float32))


def test_k_larger_than_the_corpus_is_clamped() -> None:
    index = VectorIndex(_chunks(2), np.eye(2, dtype=np.float32))
    assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), k=50)) == 2


def test_zero_vectors_do_not_divide_by_zero() -> None:
    index = VectorIndex(_chunks(2), np.zeros((2, 3), dtype=np.float32))
    hits = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert all(h.score == 0.0 for h in hits)


def test_token_budget_stops_filling() -> None:
    long_text = "word " * 400  # ~400 tokens each
    chunks = [Chunk(doc_id="d", chunk_id=f"c{i}", raw_text=f"{long_text}{i}") for i in range(20)]
    vectors = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    index = VectorIndex(chunks, vectors)

    hits = index.search_token_budget(vectors[0], budget=1000)
    assert 0 < len(hits) < 20


def test_token_budget_does_not_pay_twice_for_one_parent() -> None:
    """parent-doc emits many children sharing one parent.

    Counting that parent once per child would exhaust the budget on a single section and
    make the arm look artificially expensive.
    """
    parent = "shared section text " * 100
    chunks = [
        Chunk(doc_id="d", chunk_id=f"c{i}", raw_text=f"child {i}", return_text=parent)
        for i in range(10)
    ]
    vectors = np.random.default_rng(1).normal(size=(10, 8)).astype(np.float32)
    index = VectorIndex(chunks, vectors)

    hits = index.search_token_budget(vectors[0], budget=1000)
    assert len(hits) == 1, "identical returned text should be charged once"


def test_score_query_finds_gold_at_its_rank() -> None:
    gold_text = "the term is five years"
    hits = [
        Hit(Chunk(doc_id="d", chunk_id="a", raw_text="unrelated"), 0.9, 0),
        Hit(Chunk(doc_id="d", chunk_id="b", raw_text=gold_text), 0.8, 1),
    ]
    result = score_query("how long?", hits, [GoldSpan(text=gold_text)])
    assert result.relevant_ranks == (1,)
    assert result.reciprocal_rank == pytest.approx(0.5)
    assert result.recall_at(5) == 1.0
    assert result.recall_at(1) == 0.0


def test_perfect_ranking_scores_one() -> None:
    gold = "the term is five years"
    hits = [Hit(Chunk(doc_id="d", chunk_id="a", raw_text=gold), 0.9, 0)]
    result = score_query("q", hits, [GoldSpan(text=gold)])
    assert result.ndcg_at(10) == pytest.approx(1.0)
    assert result.recall_at(1) == 1.0


def test_missing_gold_scores_zero() -> None:
    hits = [Hit(Chunk(doc_id="d", chunk_id="a", raw_text="nothing relevant"), 0.9, 0)]
    result = score_query("q", hits, [GoldSpan(text="the term is five years")])
    assert result.relevant_ranks == ()
    assert result.ndcg_at(10) == 0.0
    assert result.reciprocal_rank == 0.0


def test_ndcg_rewards_earlier_ranks() -> None:
    early = QueryResult("q", (0,), 10, 500, 1)
    late = QueryResult("q", (9,), 10, 500, 1)
    assert early.ndcg_at(10) > late.ndcg_at(10)


def test_efficiency_metric_penalises_returning_more_text() -> None:
    lean = score_run([QueryResult("q", (0,), 10, 500, 1)])
    bloated = score_run([QueryResult("q", (0,), 10, 4000, 1)])
    assert lean.ndcg_at_10 == pytest.approx(bloated.ndcg_at_10)
    # Same quality, eight times the context. The efficiency metric is what shows it.
    assert lean.ndcg_per_1k_tokens > bloated.ndcg_per_1k_tokens


def test_empty_run_does_not_crash() -> None:
    assert score_run([]).queries == 0
