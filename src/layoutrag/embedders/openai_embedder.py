"""OpenAI embeddings — the default path.

Hosted because that is what reaches production for a client, which makes the results
decision-relevant rather than academic. The trade is reproducibility: hosted models change
silently behind a stable name, so numbers produced here are labelled accordingly and the
local embedder exists as the reproducible counterpart.

Spend is guarded inside :meth:`OpenAIEmbedder.embed` rather than by the caller, so there is
no code path that reaches the API without passing a ceiling check first.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from layoutrag.budget import SpendGuard
from layoutrag.embedders.base import PRICE_PER_MILLION, CostEstimate, estimate

DEFAULT_MODEL = "text-embedding-3-small"
DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

# The API accepts larger batches, but a smaller one keeps a rate-limit retry cheap and makes
# progress reporting granular enough to be useful during a long ingest.
BATCH_SIZE = 256

# Batches are independent, so they run concurrently. Kept modest: the ceiling here is the
# account's tokens-per-minute limit, and overshooting it turns into retries, which cost
# wall-clock rather than saving it.
DEFAULT_CONCURRENCY = 8
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
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        guard: SpendGuard | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.name = model
        self.model = model
        self.dimensions = DIMENSIONS.get(model, 1536)
        self.concurrency = max(1, concurrency)
        # A guard is always present. Passing None gets the default ceilings, not no guard.
        self.guard = guard if guard is not None else SpendGuard()

        if api_key is None:
            from layoutrag.config import load_env

            load_env()
            api_key = os.environ.get("OPENAI_API_KEY")
        self._api_key = api_key
        self._client: Any | None = None
        self._client_lock = threading.Lock()

        self.tokens_used = 0
        self.usd_spent = 0.0

    def _get_client(self) -> Any:
        if not self._api_key:
            raise MissingAPIKey
        with self._client_lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            return self._client

    def estimate(self, texts: list[str]) -> CostEstimate:
        return estimate(texts, self.model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        # Priced and checked before a single request is made. Over the ceiling, this raises
        # and nothing is spent.
        projected = self.estimate(texts)
        self.guard.check(projected.usd, label=f"embedding {len(texts):,} texts")

        client = self._get_client()
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda b: self._embed_batch(client, b), batches))

        vectors: list[list[float]] = []
        for response in results:
            # Ordering is not guaranteed, but every item carries its index.
            vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))

        self.guard.commit()
        return np.asarray(vectors, dtype=np.float32)

    def _embed_batch(self, client: Any, batch: list[str]) -> Any:
        # Empty strings are rejected by the API, and a chunk can be whitespace-only after
        # cleaning. Substituting a space keeps row alignment with the input list, which
        # callers rely on to match vectors back to chunks.
        cleaned = [text if text.strip() else " " for text in batch]

        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(model=self.model, input=cleaned)
            except Exception as exc:
                transient = any(
                    marker in str(exc).lower()
                    for marker in ("rate limit", "429", "timeout", "connection", "503")
                )
                if not transient or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
                continue

            usage = getattr(response, "usage", None)
            tokens = usage.total_tokens if usage is not None else 0
            usd = tokens / 1e6 * PRICE_PER_MILLION.get(self.model, 0.0)

            with self._client_lock:
                self.tokens_used += tokens
                self.usd_spent += usd

            # Re-checked as the job proceeds, because an estimate can be wrong and a long
            # run must not be able to drift past the ceiling after it has started.
            self.guard.record(usd, tokens)
            return response

        raise RuntimeError("unreachable")
