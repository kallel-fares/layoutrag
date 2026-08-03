"""Local embeddings — the offline path.

Exists for two reasons. Commercially, some clients cannot send documents to a third-party
API at all, and being able to run the whole pipeline inside their network is the difference
between a demo they can try and one they can't. Methodologically, it is the reproducible
counterpart to the hosted model: these numbers can be regenerated years from now, which
hosted ones cannot.

Whether the free model is good enough is one of the questions the study answers. If it gets
close, a client's ingestion box drops an ML-sized dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from layoutrag.embedders.base import CostEstimate

# 33M parameters, 384 dimensions, MIT licensed. Small enough that embedding is never the
# bottleneck on a laptop, which matters when four strategies index the same corpus at once.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.name = f"local:{model}"
        self.model_id = model
        self.dimensions = 384
        self._model: Any | None = None

        self.tokens_used = 0
        self.usd_spent = 0.0

    def _get_model(self) -> Any:
        # Imported lazily: sentence-transformers pulls in torch, and nothing should pay
        # that cost merely for importing this module.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
            self.dimensions = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def estimate(self, texts: list[str]) -> CostEstimate:
        return CostEstimate(texts=len(texts), tokens=0, model=self.name, usd=0.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        model = self._get_model()
        cleaned = [text if text.strip() else " " for text in texts]
        vectors = model.encode(
            cleaned,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            # Cosine similarity on normalised vectors is a dot product, which is what the
            # index computes anyway.
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
