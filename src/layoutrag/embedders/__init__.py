"""Embedders.

Neither backend is imported eagerly: OpenAI and sentence-transformers are both loaded on
first use, so importing this package costs nothing.
"""

from layoutrag.embedders.base import CostEstimate, Embedder, estimate
from layoutrag.embedders.local import LocalEmbedder
from layoutrag.embedders.openai_embedder import MissingAPIKey, OpenAIEmbedder

__all__ = [
    "CostEstimate",
    "Embedder",
    "LocalEmbedder",
    "MissingAPIKey",
    "OpenAIEmbedder",
    "estimate",
]
