"""OpenAI embeddings — the default path.

Hosted because that is what reaches production for a client, which makes the results
decision-relevant rather than academic. The trade is reproducibility: hosted models change
silently behind a stable name, so numbers produced here are labelled accordingly and the
local embedder exists as the reproducible counterpart.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from layoutrag.embedders.base import CostEstimate, estimate

DEFAULT_MODEL = "text-embedding-3-small"
DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

# The API accepts far larger batches, but a smaller one keeps a rate-limit retry cheap and
# makes progress reporting granular enough to be useful on a long ingest.
BATCH_SIZE = 256
MAX_RETRIES = 5


class MissingAPIKey(RuntimeError):
    """Raised with instructions rather than a stack trace."""

    def __init__(self) -> None:
        super().__init__(
            "OPENAI_API_KEY is not set.\n"
            "  Either export it, put it in a .env file at the project root,\n"
            "  or run with --local to use a local model and no API at all."
        )


class OpenAIEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.name = model
        self.model = model
        self.dimensions = DIMENSIONS.get(model, 1536)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any | None = None

        self.tokens_used = 0
        self.usd_spent = 0.0

    def _get_client(self) -> Any:
        if not self._api_key:
            raise MissingAPIKey
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def estimate(self, texts: list[str]) -> CostEstimate:
        return estimate(texts, self.model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        client = self._get_client()
        vectors: list[list[float]] = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self._embed_batch(client, batch)
            # The API does not guarantee ordering, but every item carries its index.
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)

            usage = getattr(response, "usage", None)
            if usage is not None:
                self.tokens_used += usage.total_tokens

        from layoutrag.embedders.base import PRICE_PER_MILLION

        self.usd_spent = self.tokens_used / 1e6 * PRICE_PER_MILLION.get(self.model, 0.0)
        return np.asarray(vectors, dtype=np.float32)

    def _embed_batch(self, client: Any, batch: list[str]) -> Any:
        # Empty strings are rejected by the API, and a chunk can be whitespace-only after
        # cleaning. Substituting a space keeps row alignment with the input list, which
        # callers rely on to match vectors back to chunks.
        cleaned = [text if text.strip() else " " for text in batch]

        for attempt in range(MAX_RETRIES):
            try:
                return client.embeddings.create(model=self.model, input=cleaned)
            except Exception as exc:
                transient = any(
                    marker in str(exc).lower()
                    for marker in ("rate limit", "429", "timeout", "connection", "503")
                )
                if not transient or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("unreachable")
