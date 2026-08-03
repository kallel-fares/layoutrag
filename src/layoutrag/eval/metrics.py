"""Retrieval metrics.

Deliberately the standard set and nothing more: recall@k, nDCG@10, MRR, plus the context
cost of achieving them. No significance testing — that apparatus exists to survive peer
review, and this study exists to pick defensible defaults. Where a difference is inside
noise it is reported as such rather than ranked.

Every result carries ``tokens_returned`` alongside the quality numbers. A strategy that
wins by handing back four times the context has not won, it has spent, and reporting the
two together is what makes that visible instead of hidden.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from layoutrag.chunkers.base import count_tokens
from layoutrag.eval.relevance import DEFAULT_OVERLAP_THRESHOLD, GoldSpan, is_relevant
from layoutrag.index import Hit


@dataclass(frozen=True)
class QueryResult:
    """Per-query scores. Kept so a run can be summarised without re-running retrieval."""

    query: str
    relevant_ranks: tuple[int, ...]
    retrieved: int
    tokens_returned: int
    gold_count: int

    def recall_at(self, k: int) -> float:
        if not self.gold_count:
            return 0.0
        found = sum(1 for r in self.relevant_ranks if r < k)
        return min(1.0, found / self.gold_count)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / (min(self.relevant_ranks) + 1) if self.relevant_ranks else 0.0

    def ndcg_at(self, k: int = 10) -> float:
        """Binary-gain nDCG. Ideal ranking puts every gold span at the top."""
        if not self.gold_count:
            return 0.0
        gain = sum(1.0 / math.log2(r + 2) for r in self.relevant_ranks if r < k)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(self.gold_count, k)))
        return gain / ideal if ideal else 0.0


@dataclass
class Metrics:
    """Aggregated scores for one arm on one corpus."""

    queries: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    median_tokens_returned: float = 0.0
    ndcg_per_1k_tokens: float = 0.0
    per_query: list[QueryResult] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"n={self.queries:4}  R@1={self.recall_at_1:.3f}  R@5={self.recall_at_5:.3f}  "
            f"R@10={self.recall_at_10:.3f}  nDCG@10={self.ndcg_at_10:.3f}  "
            f"MRR={self.mrr:.3f}  tokens={self.median_tokens_returned:.0f}"
        )


def score_query(
    query: str,
    hits: list[Hit],
    gold: list[GoldSpan],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> QueryResult:
    """Score one query's hits against its gold spans.

    A hit counts once per gold span it satisfies, and each gold span is credited to its
    best (earliest) hit, so a chunk containing two gold spans does not inflate recall and a
    gold span spread over two chunks is not counted twice.
    """
    ranks: list[int] = []
    tokens = 0
    seen_text: set[str] = set()

    for hit in hits:
        if hit.chunk.return_text not in seen_text:
            tokens += count_tokens(hit.chunk.return_text)
            seen_text.add(hit.chunk.return_text)

    for span in gold:
        for hit in hits:
            if is_relevant(hit.chunk, span, threshold=threshold):
                ranks.append(hit.rank)
                break

    return QueryResult(
        query=query,
        relevant_ranks=tuple(sorted(ranks)),
        retrieved=len(hits),
        tokens_returned=tokens,
        gold_count=len(gold),
    )


def score_run(results: list[QueryResult]) -> Metrics:
    if not results:
        return Metrics()

    median_tokens = statistics.median(r.tokens_returned for r in results)
    ndcg = statistics.mean(r.ndcg_at(10) for r in results)

    return Metrics(
        queries=len(results),
        recall_at_1=statistics.mean(r.recall_at(1) for r in results),
        recall_at_5=statistics.mean(r.recall_at(5) for r in results),
        recall_at_10=statistics.mean(r.recall_at(10) for r in results),
        ndcg_at_10=ndcg,
        mrr=statistics.mean(r.reciprocal_rank for r in results),
        median_tokens_returned=median_tokens,
        # Quality per unit of context spent. This is the number that separates "retrieved
        # well" from "returned a lot and got lucky".
        ndcg_per_1k_tokens=(ndcg / (median_tokens / 1000)) if median_tokens else 0.0,
        per_query=results,
    )
