"""Fixed-size chunking — the baseline, and the control for the overlap question.

``fixed`` cuts every N tokens with an overlap. ``fixed-no-overlap`` is the same thing with
the overlap set to zero.

Both exist because the baseline would otherwise be confounded. If the baseline has overlap
and no other strategy does, then every comparison against it measures two things at once —
the cut points *and* the overlap — and there is no way to attribute a difference to either.
Measuring overlap once, explicitly, is cheaper than footnoting it in every later result.
"""

from __future__ import annotations

from layoutrag.blocks import ParsedDoc
from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import get_encoding

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_RATIO = 0.125


class FixedChunker:
    """Cut every ``chunk_tokens`` tokens, optionally overlapping."""

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        name: str | None = None,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")

        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = int(chunk_tokens * overlap_ratio)
        self.name = name or ("fixed" if overlap_ratio else "fixed-no-overlap")

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.parse_failed or not doc.blocks:
            return []

        encoding = get_encoding()

        # Chunk over the whole document rather than per block, so a fixed-size cut really
        # is size-driven and does not accidentally inherit the parser's block boundaries —
        # which would make this a structure-aware strategy wearing a fixed-size label.
        text = doc.text
        tokens = encoding.encode(text)

        stride = self.chunk_tokens - self.overlap_tokens
        chunks: list[Chunk] = []

        for index, start in enumerate(range(0, len(tokens), stride)):
            window = tokens[start : start + self.chunk_tokens]
            if not window:
                break
            body = encoding.decode(window).strip()
            if body:
                chunks.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        chunk_id=f"{doc.doc_id}::{self.name}::{index}",
                        raw_text=body,
                        strategy=self.name,
                    )
                )
            if start + self.chunk_tokens >= len(tokens):
                break

        return chunks
