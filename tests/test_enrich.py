"""Contextual enrichment.

The property that matters is that enrichment changes what gets embedded and never what gets
returned. If it leaks into the returned text, the reader is shown an LLM's annotation as
though it were the document, and relevance scoring credits the strategy for text nobody
actually receives.
"""

from __future__ import annotations

from layoutrag.chunk_type import Chunk
from layoutrag.enrich import ContextualEnricher, estimate, window_around

DOC = (
    "COLLABORATION AGREEMENT between Foundation Medicine and Roche. "
    + "Recitals and definitions follow. " * 60
    + "8. TERMINATION. Either party may terminate on thirty days notice. "
    + "Further provisions continue. " * 60
)
CHUNK_TEXT = "8. TERMINATION. Either party may terminate on thirty days notice."


def _chunk(text: str = CHUNK_TEXT) -> Chunk:
    return Chunk(doc_id="d", chunk_id="c", raw_text=text)


class _StubClient:
    """Stands in for the API so the wiring is testable without a key or a charge."""

    def __init__(
        self, text: str = "Termination clause of the Foundation Medicine agreement."
    ) -> None:
        self.text = text
        self.prompts: list[str] = []
        self.responses = self

    def create(self, model: str, input: str, **_: object) -> object:
        self.prompts.append(input)
        return type("R", (), {"output_text": self.text, "usage": None})()


def test_window_is_centred_on_the_chunk() -> None:
    window = window_around(DOC, CHUNK_TEXT, tokens=200)
    assert CHUNK_TEXT[:30] in window
    assert len(window) < len(DOC)


def test_short_documents_are_returned_whole() -> None:
    short = "A brief agreement. It terminates on notice."
    assert window_around(short, "It terminates", tokens=2000) == short


def test_missing_chunk_falls_back_to_the_document_head() -> None:
    # A chunker may rewrite text, so the chunk is not always findable in the source.
    window = window_around(DOC, "text that never appears anywhere", tokens=200)
    assert "COLLABORATION AGREEMENT" in window


def test_enrichment_changes_embedded_text_only() -> None:
    enricher = ContextualEnricher(api_key="test")
    enricher._client = _StubClient()

    before = _chunk()
    after = enricher.enrich(before, DOC)

    assert after.embed_text != before.embed_text
    assert "Foundation Medicine" in after.embed_text
    assert after.is_enriched
    # The reader gets the document, never the annotation.
    assert after.return_text == before.return_text
    assert "Foundation Medicine agreement." not in after.return_text
    assert after.raw_text == before.raw_text


def test_the_window_and_the_chunk_both_reach_the_model() -> None:
    enricher = ContextualEnricher(api_key="test")
    client = _StubClient()
    enricher._client = client

    enricher.enrich(_chunk(), DOC)
    prompt = client.prompts[0]
    assert CHUNK_TEXT in prompt
    assert "COLLABORATION AGREEMENT" in prompt


def test_a_failed_call_leaves_the_chunk_untouched() -> None:
    class _Boom:
        responses = property(lambda self: self)

        def create(self, **_: object) -> object:
            raise RuntimeError("rate limited")

    enricher = ContextualEnricher(api_key="test")
    enricher._client = _Boom()

    before = _chunk()
    after = enricher.enrich(before, DOC)
    # One bad chunk must not end a corpus run, and must stay visibly un-enriched.
    assert after == before
    assert not after.is_enriched


def test_an_empty_response_leaves_the_chunk_untouched() -> None:
    enricher = ContextualEnricher(api_key="test")
    enricher._client = _StubClient(text="   ")
    assert not enricher.enrich(_chunk(), DOC).is_enriched


def test_estimate_scales_with_window_and_costs_nothing_to_compute() -> None:
    chunks = [_chunk() for _ in range(10)]
    docs = {"d": DOC}

    small = estimate(chunks, docs, window=200)
    large = estimate(chunks, docs, window=2000)

    assert small.chunks == large.chunks == 10
    assert large.tokens_in > small.tokens_in
    assert small.usd > 0
