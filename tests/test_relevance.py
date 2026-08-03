"""Relevance scoring.

The size-invariance test is the one that protects the whole study. If relevance depends on
how big a chunk is, every result becomes a measurement of chunk size wearing a chunking
strategy's name, and nothing downstream can detect it.
"""

from __future__ import annotations

import pytest

from layoutrag.chunk_type import Chunk
from layoutrag.eval.relevance import GoldSpan, coverage, is_relevant, normalise

GOLD = "Either party may terminate this agreement on thirty days written notice."


def _chunk(text: str, doc_id: str = "d", **kw: object) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_id="c", raw_text=text, **kw)  # type: ignore[arg-type]


def test_relevance_is_invariant_to_chunk_size() -> None:
    """The test that protects the study.

    The same gold span, embedded in chunks of wildly different sizes, must score the same.
    Defining relevance as 'half of this chunk is gold' would fail this and quietly rig the
    results against parent-doc and hierarchical.
    """
    padding = "This section concerns miscellaneous provisions of the agreement. "

    tiny = _chunk(GOLD)
    medium = _chunk(padding * 5 + GOLD + padding * 5)
    huge = _chunk(padding * 100 + GOLD + padding * 100)

    gold = GoldSpan(text=GOLD)
    assert is_relevant(tiny, gold)
    assert is_relevant(medium, gold)
    assert is_relevant(huge, gold), "large chunks must not be penalised for being large"


def test_a_chunk_without_the_gold_span_is_not_relevant() -> None:
    other = _chunk("Invoices shall be paid within thirty days of receipt." * 10)
    assert not is_relevant(other, GoldSpan(text=GOLD))


def test_partial_coverage_below_threshold_fails() -> None:
    partial = _chunk("Either party may terminate")  # ~4 of 11 words
    assert coverage(partial.return_text, GOLD) < 0.5
    assert not is_relevant(partial, GoldSpan(text=GOLD))


def test_scattered_words_do_not_count_as_a_match() -> None:
    """Gold words must appear together, not sprinkled through a long chunk."""
    scattered = _chunk(
        "Either the invoice is late. " * 20
        + "party hats are provided. " * 20
        + "may we terminate the call. " * 20
    )
    assert coverage(scattered.return_text, GOLD) < 0.5


def test_matching_tolerates_parser_differences() -> None:
    # Different parsers produce different whitespace, quotes, dashes, and line-break
    # hyphenation for the same text. None of those are content differences.
    variants = [
        # Collapsed vs expanded whitespace.
        "Either  party   may terminate this agreement on thirty days written notice.",
        # A word broken across a line by hyphenation.
        "Either party may termina-\nte this agreement on thirty days written notice.",
        # Small-caps read as capitals by some parsers.
        GOLD.upper(),
        # Non-breaking spaces, which survive extraction from several producers. Built
        # here rather than written literally so the source stays plain ASCII.
        GOLD.replace(" ", "\u00a0"),
    ]
    for text in variants:
        assert is_relevant(_chunk(text), GoldSpan(text=GOLD)), text


def test_scoring_uses_returned_text_not_embedded_text() -> None:
    """A generator sees return_text, so relevance is judged on it.

    Judging on embed_text would credit contextual-heading for breadcrumbs that never reach
    the generator, and credit parent-doc for context it embeds but does not return.
    """
    misleading = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text="unrelated filler text about invoicing schedules",
        embed_text=GOLD,  # gold is only in what was embedded
        return_text="unrelated filler text about invoicing schedules",
    )
    assert not is_relevant(misleading, GoldSpan(text=GOLD))

    honest = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text=GOLD,
        embed_text="Agreement > Termination\n\n" + GOLD,
        return_text=GOLD,
    )
    assert is_relevant(honest, GoldSpan(text=GOLD))


def test_gold_from_a_different_document_never_matches() -> None:
    chunk = _chunk(GOLD, doc_id="doc-a")
    assert not is_relevant(chunk, GoldSpan(text=GOLD, doc_id="doc-b"))
    assert is_relevant(chunk, GoldSpan(text=GOLD, doc_id="doc-a"))


def test_empty_inputs_are_not_relevant() -> None:
    assert coverage("", GOLD) == 0.0
    assert coverage(GOLD, "") == 0.0


def test_normalise_folds_only_presentation() -> None:
    assert normalise("Hello,  World!") == normalise("hello world")
    assert normalise("termina-\ntion") == "termination"
    assert normalise("“quoted”") == normalise('"quoted"')


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.8])
def test_threshold_is_configurable_and_recorded(threshold: float) -> None:
    chunk = _chunk(GOLD)
    assert is_relevant(chunk, GoldSpan(text=GOLD), threshold=threshold)
