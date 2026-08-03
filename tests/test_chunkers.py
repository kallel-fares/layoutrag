"""Chunker behaviour.

Two properties matter most here: that the strategies genuinely differ in which text field
they populate, and that a strategy needing structure it wasn't given says so.
"""

from __future__ import annotations

import pytest

from layoutrag.blocks import Block, BlockType, ParsedDoc
from layoutrag.chunkers import ContextualHeadingChunker, FixedChunker, ParentDocChunker


def _structured() -> ParsedDoc:
    return ParsedDoc(
        doc_id="d",
        source_path="/tmp/d.pdf",
        parser="test",
        blocks=(
            Block(type=BlockType.HEADING, text="Master Agreement", level=1, page=1),
            Block(type=BlockType.HEADING, text="Termination", level=2, page=1),
            Block(type=BlockType.PARAGRAPH, text="Either party may terminate." * 8, page=1),
            Block(type=BlockType.HEADING, text="Payment", level=2, page=2),
            Block(type=BlockType.PARAGRAPH, text="Invoices are net thirty." * 8, page=2),
        ),
    )


def _flat() -> ParsedDoc:
    return ParsedDoc(
        doc_id="d",
        source_path="/tmp/d.pdf",
        parser="test",
        blocks=tuple(
            Block(type=BlockType.PARAGRAPH, text=f"Paragraph {i}. " * 20) for i in range(6)
        ),
    )


def test_fixed_produces_uniform_chunks() -> None:
    chunks = FixedChunker(chunk_tokens=64, overlap_ratio=0.0).chunk(_flat())
    assert len(chunks) > 1
    assert all(c.strategy == "fixed-no-overlap" for c in chunks)
    # Nothing interesting is happening to the text fields, and that is the point.
    assert all(c.embed_text == c.raw_text == c.return_text for c in chunks)


def test_overlap_yields_more_chunks_than_no_overlap() -> None:
    doc = _flat()
    plain = FixedChunker(chunk_tokens=64, overlap_ratio=0.0).chunk(doc)
    lapped = FixedChunker(chunk_tokens=64, overlap_ratio=0.5).chunk(doc)
    assert len(lapped) > len(plain)


def test_fixed_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="chunk_tokens"):
        FixedChunker(chunk_tokens=0)
    with pytest.raises(ValueError, match="overlap_ratio"):
        FixedChunker(overlap_ratio=1.0)
    with pytest.raises(ValueError, match="overlap_ratio"):
        FixedChunker(overlap_ratio=-0.1)


def test_contextual_heading_embeds_the_path_but_returns_the_span() -> None:
    chunks = ContextualHeadingChunker(chunk_tokens=64).chunk(_structured())
    enriched = [c for c in chunks if c.heading_path]
    assert enriched

    c = enriched[0]
    assert c.is_enriched
    assert " > ".join(c.heading_path) in c.embed_text
    # The generator sees the document, not our breadcrumbs.
    assert c.return_text == c.raw_text
    assert " > ".join(c.heading_path) not in c.return_text


def test_contextual_heading_marks_itself_degraded_without_structure() -> None:
    chunks = ContextualHeadingChunker(chunk_tokens=64).chunk(_flat())
    assert chunks
    assert all(c.degraded for c in chunks)
    assert all(c.degraded_reason for c in chunks)
    # Degraded means it really did nothing, not that it did something ineffective.
    assert all(not c.is_enriched for c in chunks)


def test_parent_doc_returns_more_than_it_embeds() -> None:
    chunks = ParentDocChunker(child_tokens=16).chunk(_structured())
    assert chunks
    expanded = [c for c in chunks if c.is_expanded]
    assert expanded
    c = expanded[0]
    assert c.embed_text == c.raw_text
    assert len(c.return_text) > len(c.embed_text)


def test_parent_doc_children_of_a_section_share_one_return_text() -> None:
    chunks = ParentDocChunker(child_tokens=16).chunk(_structured())
    by_section: dict[str, set[str]] = {}
    for c in chunks:
        by_section.setdefault(c.return_text, set()).add(c.embed_text)
    multi = [embeds for embeds in by_section.values() if len(embeds) > 1]
    assert multi, "expected at least one section split into several embedded children"


def test_chunkers_return_nothing_for_a_failed_parse() -> None:
    failed = ParsedDoc(
        doc_id="d",
        source_path="/tmp/d.pdf",
        parser="test",
        parse_failed=True,
        failure_reason="scanned",
    )
    assert FixedChunker().chunk(failed) == []
    assert ContextualHeadingChunker().chunk(failed) == []
    assert ParentDocChunker().chunk(failed) == []


def test_chunk_ids_are_unique_within_a_document() -> None:
    for chunker in (FixedChunker(chunk_tokens=64), ContextualHeadingChunker(chunk_tokens=64)):
        chunks = chunker.chunk(_structured())
        assert len({c.chunk_id for c in chunks}) == len(chunks)
