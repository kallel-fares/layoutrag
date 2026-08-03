"""Fragment assembly and heading detection.

Assembly is the part that matters: a PDF text object is not a line, and without merging
them the parser emits ~180 fragments per page instead of ~15 blocks, which would make every
downstream chunking decision meaningless.
"""

from __future__ import annotations

from layoutrag.parsers.pdfium import Fragment, _assemble, _looks_like_heading


def _frag(size: float, left: float, bottom: float, text: str) -> Fragment:
    """A fragment at a given size and position, one unit tall per point of size."""
    return (size, left, bottom, left + 10 * len(text), bottom + size, text)


def test_flush_fragments_merge_without_a_space() -> None:
    # Real case: producers split "NIST Special Publication 800-140F" into three objects.
    # Rendered flush, so no spaces belong around the hyphen.
    frags = [
        _frag(16.0, 0, 700, "NIST Special Publication 800"),
        _frag(16.0, 280, 700, "-"),
        _frag(16.0, 290, 700, "140F"),
    ]
    assembled = _assemble(frags)
    assert len(assembled) == 1
    assert assembled[0][1] == "NIST Special Publication 800-140F"


def test_separated_fragments_on_one_line_get_a_space() -> None:
    # Same line, but a real gap between them — two words, not one.
    frags = [
        _frag(10.0, 0, 700, "Peter Mell"),
        _frag(10.0, 200, 700, "Timothy Grance"),
    ]
    assembled = _assemble(frags)
    assert len(assembled) == 1
    assert assembled[0][1] == "Peter Mell Timothy Grance"


def test_lines_at_different_heights_stay_separate() -> None:
    frags = [
        _frag(10.0, 0, 700, "first line"),
        _frag(10.0, 0, 600, "far below"),
    ]
    assert len(_assemble(frags)) == 2


def test_consecutive_body_lines_join_into_a_paragraph() -> None:
    frags = [
        _frag(10.0, 0, 700, "The quick brown fox"),
        _frag(10.0, 0, 688, "jumps over the lazy dog"),
    ]
    assembled = _assemble(frags)
    assert len(assembled) == 1
    assert assembled[0][1] == "The quick brown fox jumps over the lazy dog"


def test_a_size_change_breaks_the_paragraph() -> None:
    # A heading directly above body text must not be absorbed into it.
    frags = [
        _frag(16.0, 0, 700, "1. INTRODUCTION"),
        _frag(10.0, 0, 685, "This document describes"),
    ]
    assembled = _assemble(frags)
    assert len(assembled) == 2
    assert assembled[0] == (16.0, "1. INTRODUCTION")


def test_reading_order_is_top_down() -> None:
    frags = [
        _frag(10.0, 0, 100, "last"),
        _frag(10.0, 0, 700, "first"),
        _frag(10.0, 0, 400, "middle"),
    ]
    assert [t for _, t in _assemble(frags)] == ["first", "middle", "last"]


def test_empty_input() -> None:
    assert _assemble([]) == []


def test_heading_filter_rejects_rules_and_punctuation() -> None:
    # These appear in large type on NIST cover pages but are not headings.
    assert not _looks_like_heading("_" * 55)
    assert not _looks_like_heading("-")
    assert not _looks_like_heading("...")
    assert not _looks_like_heading("A")


def test_heading_filter_accepts_real_headings() -> None:
    assert _looks_like_heading("1. INTRODUCTION")
    assert _looks_like_heading("The NIST Definition of Cloud Computing")
    assert _looks_like_heading("8. TERMINATION")


def test_heading_filter_rejects_overlong_text() -> None:
    assert not _looks_like_heading("word " * 40)
