"""Evaluation: relevance scoring and retrieval metrics.

Nothing outside this package under ``layoutrag`` imports it. Eval depends on the pipeline;
the pipeline must never depend on eval.
"""

from layoutrag.eval.metrics import Metrics, QueryResult, score_query, score_run
from layoutrag.eval.relevance import GoldSpan, coverage, is_relevant

__all__ = [
    "GoldSpan",
    "Metrics",
    "QueryResult",
    "coverage",
    "is_relevant",
    "score_query",
    "score_run",
]
