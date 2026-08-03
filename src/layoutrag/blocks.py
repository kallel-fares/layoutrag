"""Typed document structure.

A parsed document is a list of typed blocks, not a flat string. That distinction is the
whole reason this project exists: ``contextual-heading`` needs to know which section a
span sits in, and comparing parsers is meaningless if every parser's output is collapsed
to text before anything looks at it.

Parsers that cannot recover structure say so via :attr:`ParsedDoc.has_structure`. Strategies
that need structure then mark their chunks degraded rather than silently emitting something
that looks like a real result.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field

# Synthetic depth for the first heading found that carries no level. Arbitrary but large,
# so any explicitly-levelled heading above it reads as shallower.
_UNLEVELLED_DEPTH = 1_000_000

# Text repeating on at least this share of pages is page furniture, not content.
FURNITURE_PAGE_RATIO = 0.30

# Below this many pages, repetition is not evidence of anything — a three-page contract may
# legitimately say the same short line twice.
FURNITURE_MIN_PAGES = 5


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTER = "footer"
    OTHER = "other"


class Block(BaseModel):
    """One structural element of a document."""

    model_config = {"frozen": True}

    type: BlockType
    text: str
    page: int | None = None

    level: int | None = None
    """Heading depth, 1 = outermost. Only meaningful for :attr:`BlockType.HEADING`."""

    meta: dict[str, str] = Field(default_factory=dict)


class ParsedDoc(BaseModel):
    """The output of a parser: one document, as typed blocks."""

    model_config = {"frozen": True}

    doc_id: str
    source_path: str
    parser: str
    """Name of the parser that produced this, e.g. ``pypdfium2`` or ``docling``."""

    blocks: tuple[Block, ...] = ()
    page_count: int = 0

    parse_failed: bool = False
    """Corrupt, encrypted, and scanned PDFs exist. A parser that fails on part of a corpus
    is disqualified regardless of its quality on the rest, so failures are recorded as data
    rather than raised away."""

    failure_reason: str = ""

    @property
    def has_structure(self) -> bool:
        """True when the parser recovered headings.

        Text-extraction parsers return everything as paragraphs. Strategies that need
        headings must check this and mark their chunks degraded when it is False —
        otherwise a run where the strategy did nothing is indistinguishable from a run
        where it did something that didn't help.
        """
        return any(b.type is BlockType.HEADING for b in self.blocks)

    @property
    def content_blocks(self) -> tuple[Block, ...]:
        """Blocks that carry document content, excluding page furniture.

        Running headers and footers repeat on every page. Left in, they are embedded into
        every chunk of the document — the same string dozens of times — which adds noise to
        each vector and lets a query match furniture instead of content.
        """
        return tuple(b for b in self.blocks if b.type is not BlockType.FOOTER and b.text)

    @property
    def text(self) -> str:
        """Whole document as plain text, for strategies that don't care about structure."""
        return "\n\n".join(b.text for b in self.content_blocks)

    def heading_path_at(self, index: int) -> tuple[str, ...]:
        """Enclosing headings for the block at ``index``, outermost first.

        Walks backwards accumulating headings whose level strictly decreases, so a level-3
        heading under a level-1 heading yields both, while a sibling level-3 seen earlier is
        skipped.

        Some parsers report "this is a heading" without a depth. Those are treated as each
        being one level shallower than the previous one found, so consecutive unlevelled
        headings still nest instead of collapsing to just the nearest.
        """
        if not 0 <= index < len(self.blocks):
            raise IndexError(f"block index {index} out of range for {len(self.blocks)} blocks")

        path: list[str] = []
        depth_seen: int | None = None

        for block in reversed(self.blocks[:index]):
            if block.type is not BlockType.HEADING:
                continue

            if block.level is not None:
                level = block.level
            elif depth_seen is not None:
                level = depth_seen - 1
            else:
                level = _UNLEVELLED_DEPTH

            if depth_seen is None or level < depth_seen:
                path.append(block.text)
                depth_seen = level
                # Only an explicit top-level heading ends the walk; a synthetic depth
                # means we still don't know how deep we really are.
                if block.level is not None and block.level <= 1:
                    break

        return tuple(reversed(path))


def mark_page_furniture(blocks: list[Block], page_count: int) -> list[Block]:
    """Retype running headers and footers as :attr:`BlockType.FOOTER`.

    Detected by repetition rather than position: a running header is text that appears on
    many pages, and that is true regardless of where a given producer places it. Position
    heuristics need per-document margins and get fooled by rotated pages and two-column
    layouts; counting is not fooled by either.

    Only short text is considered. A long paragraph appearing on many pages is a genuine
    repeated clause, which is content, not furniture.
    """
    if page_count < FURNITURE_MIN_PAGES:
        return blocks

    pages_with: dict[str, set[int]] = {}
    for block in blocks:
        if block.page is None or len(block.text) > 120:
            continue
        pages_with.setdefault(_furniture_key(block.text), set()).add(block.page)

    threshold = max(2, int(page_count * FURNITURE_PAGE_RATIO))
    furniture = {key for key, pages in pages_with.items() if len(pages) >= threshold}
    if not furniture:
        return blocks

    return [
        block.model_copy(update={"type": BlockType.FOOTER})
        if block.page is not None
        and len(block.text) <= 120
        and _furniture_key(block.text) in furniture
        else block
        for block in blocks
    ]


def _furniture_key(text: str) -> str:
    """Normalised form for repetition counting.

    Digits are dropped so "Page 4 of 60" and "Page 5 of 60" collapse to one key — otherwise
    every page number is unique and no page numbering is ever detected as furniture.
    """
    return re.sub(r"\d+", "", text).lower().strip()
