"""Contextual enrichment: give each chunk a line saying where it sits.

A chunk pulled out of a document loses the thing that made it findable. "Either party may
terminate on thirty days notice" appears in hundreds of contracts. Embedded alone, it
matches every one of them equally, and retrieval cannot tell which document the reader
wanted.

So an LLM writes one line per chunk situating it, and that line is prepended to what gets
embedded. The returned text is untouched, so the reader still sees the document, not our
annotation.

**Context comes from a window, not the whole document.** The published form of this
technique passes the entire document alongside every chunk, which on this corpus means
resending a 44,000-token document 80 times. That costs about $1.46 with prompt caching
working and about $10.65 without, and whether it works depends on a cache that expires in
minutes. A window of a few thousand tokens around the chunk carries the same information
that situates it, costs about $0.30, and cannot fail expensively.

Enrichment is cached by content hash like every other stage, so it is paid for once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import get_encoding

DEFAULT_MODEL = "gpt-5-nano"

# USD per million tokens.
PRICE_IN, PRICE_OUT = 0.05, 0.40

# Tokens of surrounding document sent with each chunk. Enough to name the document and the
# section it sits in, short enough that the cost stays linear and small.
DEFAULT_WINDOW_TOKENS = 2000

# The blurb only has to locate the chunk. Anything longer starts competing with the chunk's
# own text for the embedding's attention.
MAX_BLURB_TOKENS = 60

# Output tokens actually billed per chunk, which is far more than the sentence returned.
# Reasoning models bill their internal reasoning as output, and measured over 317 chunks
# that came to ~304 tokens against the ~60 the visible sentence accounts for. Estimating
# from the sentence alone under-priced a corpus run by 65%.
BILLED_OUTPUT_TOKENS = 320

PROMPT = """Here is part of a document, followed by one excerpt from it.

DOCUMENT EXCERPT CONTEXT:
{window}

THE CHUNK:
{chunk}

Write ONE short sentence, at most 25 words, that says what this chunk is and where it sits:
name the document or agreement if you can tell, and the section or topic. This is used to
help a search engine find this chunk later, so be specific about identifiers and names.

Return only the sentence."""


@dataclass
class EnrichmentCost:
    chunks: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def usd(self) -> float:
        return self.tokens_in / 1e6 * PRICE_IN + self.tokens_out / 1e6 * PRICE_OUT


def window_around(document_text: str, chunk_text: str, tokens: int = DEFAULT_WINDOW_TOKENS) -> str:
    """The slice of ``document_text`` surrounding ``chunk_text``.

    Centred on the chunk so context from both sides survives. Falls back to the head of the
    document when the chunk cannot be located, which happens when a chunker rewrote the text
    it was built from.
    """
    encoding = get_encoding()
    doc_tokens = encoding.encode(document_text)
    if len(doc_tokens) <= tokens:
        return document_text

    position = document_text.find(chunk_text[:200])
    if position < 0:
        # The head of a document names it, which is most of what situating needs.
        return encoding.decode(doc_tokens[:tokens]).strip()

    before = len(encoding.encode(document_text[:position]))
    start = max(0, before - tokens // 2)
    return encoding.decode(doc_tokens[start : start + tokens]).strip()


def estimate(chunks: list[Chunk], documents: dict[str, str], window: int) -> EnrichmentCost:
    """Price enrichment without calling anything."""
    encoding = get_encoding()
    cost = EnrichmentCost(chunks=len(chunks))
    overhead = len(encoding.encode(PROMPT))

    for chunk in chunks:
        text = documents.get(chunk.doc_id, "")
        cost.tokens_in += (
            min(window, len(encoding.encode(text)))
            + len(encoding.encode(chunk.raw_text))
            + overhead
        )
        cost.tokens_out += BILLED_OUTPUT_TOKENS
    return cost


class ContextualEnricher:
    """Prepends an LLM-written locator line to each chunk's embedded text."""

    name = "contextual-llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.window_tokens = window_tokens
        self._client: Any | None = None

        if api_key is None:
            from layoutrag.config import load_env

            load_env()
            api_key = os.environ.get("OPENAI_API_KEY")
        self._api_key = api_key

        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def usd_spent(self) -> float:
        return self.tokens_in / 1e6 * PRICE_IN + self.tokens_out / 1e6 * PRICE_OUT

    def _get_client(self) -> Any:
        if not self._api_key:
            from layoutrag.embedders import MissingAPIKey

            raise MissingAPIKey
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def blurb(self, chunk: Chunk, document_text: str) -> str:
        window = window_around(document_text, chunk.raw_text, self.window_tokens)
        prompt = PROMPT.format(window=window, chunk=chunk.raw_text)

        response = self._get_client().responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": "low"},
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tokens_in += getattr(usage, "input_tokens", 0)
            self.tokens_out += getattr(usage, "output_tokens", 0)

        return (response.output_text or "").strip().replace("\n", " ")

    def enrich(self, chunk: Chunk, document_text: str) -> Chunk:
        """Return ``chunk`` with the locator line prepended to its embedded text.

        ``return_text`` is deliberately untouched: the reader gets the document, never our
        annotation. A failed call returns the chunk unchanged rather than aborting a long
        run, and leaves it visibly un-enriched.
        """
        try:
            line = self.blurb(chunk, document_text)
        except Exception:
            return chunk

        if not line:
            return chunk

        return chunk.model_copy(
            update={
                "embed_text": f"{line}\n\n{chunk.raw_text}",
                "strategy": self.name,
            }
        )
