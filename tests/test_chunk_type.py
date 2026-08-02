"""The three text fields have to stay genuinely distinct.

If these tests pass but the fields were collapsed into one, sentence-window and parent-doc
would silently become fixed-size chunking and the study would compare six names for the
same thing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from layoutrag.chunk_type import Chunk


def test_fields_default_to_raw_text() -> None:
    c = Chunk(doc_id="d", chunk_id="c", raw_text="hello")
    assert c.embed_text == "hello"
    assert c.return_text == "hello"
    assert not c.is_expanded
    assert not c.is_enriched


def test_sentence_window_embeds_less_than_it_returns() -> None:
    c = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text="The term is five years.",
        embed_text="The term is five years.",
        return_text=(
            "This agreement begins on signing. The term is five years. Renewal is automatic."
        ),
        strategy="sentence-window",
    )
    assert c.embed_text != c.return_text
    assert c.is_expanded
    assert not c.is_enriched


def test_parent_doc_returns_the_enclosing_section() -> None:
    c = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text="Either party may terminate on 30 days notice.",
        return_text="8. TERMINATION\n\nEither party may terminate on 30 days notice. "
        "Termination does not affect accrued rights.",
        strategy="parent-doc",
    )
    assert c.embed_text == c.raw_text
    assert c.return_text != c.raw_text
    assert c.is_expanded


def test_contextual_heading_enriches_what_is_embedded_but_not_what_is_returned() -> None:
    raw = "Either party may terminate on 30 days notice."
    c = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text=raw,
        embed_text=f"Master Services Agreement > Termination\n\n{raw}",
        heading_path=("Master Services Agreement", "Termination"),
        strategy="contextual-heading",
    )
    assert c.is_enriched
    # The generator must not be fed synthetic heading text as if it were the contract.
    assert c.return_text == raw


def test_degraded_chunks_must_explain_themselves() -> None:
    with pytest.raises(ValidationError):
        Chunk(doc_id="d", chunk_id="c", raw_text="x", degraded=True)

    c = Chunk(
        doc_id="d",
        chunk_id="c",
        raw_text="x",
        degraded=True,
        degraded_reason="parser produced no headings",
        strategy="contextual-heading",
    )
    assert c.degraded
    # A degraded contextual-heading chunk is just a raw chunk, and must be visible as such.
    assert not c.is_enriched


def test_chunks_are_immutable() -> None:
    c = Chunk(doc_id="d", chunk_id="c", raw_text="x")
    with pytest.raises(ValidationError):
        c.raw_text = "y"
