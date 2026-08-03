"""Embedder interface and cost estimation.

Nothing here calls an API without the caller having first been able to see what it would
cost. :func:`estimate` is deliberately separate from :meth:`Embedder.embed` so a run can be
priced, shown, and confirmed before a single token is spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# USD per million input tokens. text-embedding-3-small is the default because it is what
# actually ships to production at this price point; ada-002 is deliberately absent, being
# both more expensive ($0.10) and lower quality.
PRICE_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


@dataclass(frozen=True)
class CostEstimate:
    """What a run would cost, before it runs."""

    texts: int
    tokens: int
    model: str
    usd: float

    def describe(self) -> str:
        return f"{self.texts:,} texts, {self.tokens:,} tokens, {self.model} -> ${self.usd:.4f}"


def estimate(texts: list[str], model: str) -> CostEstimate:
    """Price a batch without embedding it."""
    from layoutrag.chunkers.base import get_encoding

    encoding = get_encoding()
    tokens = sum(len(encoding.encode(t)) for t in texts)
    rate = PRICE_PER_MILLION.get(model, 0.0)
    return CostEstimate(texts=len(texts), tokens=tokens, model=model, usd=tokens / 1e6 * rate)


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts, returning one row per text in the order given."""
        ...
