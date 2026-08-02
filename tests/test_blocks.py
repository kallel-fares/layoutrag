"""Heading-path resolution and the structure/no-structure distinction.

``heading_path_at`` is what ``contextual-heading`` is built on, and ``has_structure`` is
what stops a structure-aware strategy from silently reporting a null result when it never
actually ran.
"""

from __future__ import annotations

import pytest

from layoutrag.blocks import Block, BlockType, ParsedDoc


def _doc(*blocks: Block, parser: str = "test") -> ParsedDoc:
    return ParsedDoc(doc_id="d", source_path="/tmp/d.pdf", parser=parser, blocks=blocks)


def test_flat_parse_has_no_structure() -> None:
    doc = _doc(
        Block(type=BlockType.PARAGRAPH, text="one"),
        Block(type=BlockType.PARAGRAPH, text="two"),
    )
    assert not doc.has_structure
    assert doc.heading_path_at(1) == ()
    assert doc.text == "one\n\ntwo"


def test_nested_headings_accumulate_outermost_first() -> None:
    doc = _doc(
        Block(type=BlockType.HEADING, text="Master Services Agreement", level=1),
        Block(type=BlockType.HEADING, text="Termination", level=2),
        Block(type=BlockType.PARAGRAPH, text="Either party may terminate."),
    )
    assert doc.has_structure
    assert doc.heading_path_at(2) == ("Master Services Agreement", "Termination")


def test_sibling_headings_do_not_stack() -> None:
    # A paragraph under "Payment" must not inherit "Termination" just because it came first.
    doc = _doc(
        Block(type=BlockType.HEADING, text="Agreement", level=1),
        Block(type=BlockType.HEADING, text="Termination", level=2),
        Block(type=BlockType.PARAGRAPH, text="notice provisions"),
        Block(type=BlockType.HEADING, text="Payment", level=2),
        Block(type=BlockType.PARAGRAPH, text="net 30"),
    )
    assert doc.heading_path_at(4) == ("Agreement", "Payment")
    assert doc.heading_path_at(2) == ("Agreement", "Termination")


def test_deeper_nesting_skips_intervening_siblings() -> None:
    doc = _doc(
        Block(type=BlockType.HEADING, text="A", level=1),
        Block(type=BlockType.HEADING, text="B", level=2),
        Block(type=BlockType.HEADING, text="C", level=3),
        Block(type=BlockType.HEADING, text="D", level=3),
        Block(type=BlockType.PARAGRAPH, text="body"),
    )
    assert doc.heading_path_at(4) == ("A", "B", "D")


def test_headings_without_levels_still_resolve() -> None:
    # Some parsers report "this is a heading" without a depth. Still usable.
    doc = _doc(
        Block(type=BlockType.HEADING, text="Outer"),
        Block(type=BlockType.HEADING, text="Inner"),
        Block(type=BlockType.PARAGRAPH, text="body"),
    )
    assert doc.heading_path_at(2) == ("Outer", "Inner")


def test_out_of_range_index_is_an_error() -> None:
    doc = _doc(Block(type=BlockType.PARAGRAPH, text="only"))
    with pytest.raises(IndexError):
        doc.heading_path_at(5)


def test_parse_failure_is_recorded_not_raised() -> None:
    doc = ParsedDoc(
        doc_id="d",
        source_path="/tmp/scanned.pdf",
        parser="pypdfium2",
        parse_failed=True,
        failure_reason="no extractable text — likely scanned",
    )
    assert doc.parse_failed
    assert not doc.has_structure
    assert doc.text == ""
