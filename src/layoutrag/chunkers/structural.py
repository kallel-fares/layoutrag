"""Structure-aware chunkers.

These are the two strategies neither LangChain nor LlamaIndex ships, and both depend on the
parser having recovered headings. When it hasn't, they mark their output ``degraded`` rather
than quietly behaving like fixed-size chunking — otherwise a results table cannot distinguish
"this made no difference" from "this never ran".

Both also exercise the three text fields for real:

``contextual-heading``
    embeds the heading path prepended to the span, returns the span alone. The generator
    must not be handed synthetic breadcrumb text as though it were document content.

``parent-doc``
    embeds a small span for retrieval precision, returns the whole enclosing section so the
    generator gets the context the span sits in.
"""

from __future__ import annotations

from layoutrag.blocks import Block, BlockType, ParsedDoc
from layoutrag.chunk_type import Chunk
from layoutrag.chunkers.base import get_encoding

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHILD_TOKENS = 128

# Ceiling on what parent-doc hands back for one hit. See _bounded_parent for why an
# unbounded parent silently breaks the comparison rather than merely wasting tokens.
DEFAULT_PARENT_TOKENS = 2048

_NO_STRUCTURE = "parser recovered no headings"


def _split_oversized(index: int, block: Block, limit: int) -> list[tuple[int, Block]]:
    """Break a single block that is longer than ``limit`` into token-sized pieces.

    Without this, a parser that returns one block per page (which flat extraction does)
    produces chunks at whatever size that page happened to be — so a 512-token target
    silently becomes 1000+, and the strategy is no longer being run at the size it claims.
    """
    encoding = get_encoding()
    tokens = encoding.encode(block.text)
    if len(tokens) <= limit:
        return [(index, block)]

    pieces = []
    for start in range(0, len(tokens), limit):
        text = encoding.decode(tokens[start : start + limit]).strip()
        if text:
            pieces.append((index, block.model_copy(update={"text": text})))
    return pieces


def _pack(blocks: list[tuple[int, Block]], limit: int) -> list[list[tuple[int, Block]]]:
    """Group consecutive blocks into runs of at most ``limit`` tokens."""
    encoding = get_encoding()
    runs: list[list[tuple[int, Block]]] = []
    current: list[tuple[int, Block]] = []
    budget = 0

    for raw_index, raw_block in blocks:
        for index, block in _split_oversized(raw_index, raw_block, limit):
            size = len(encoding.encode(block.text))
            if current and budget + size > limit:
                runs.append(current)
                current, budget = [], 0
            current.append((index, block))
            budget += size

    if current:
        runs.append(current)
    return runs


class ContextualHeadingChunker:
    """Fixed-size chunks with the enclosing heading path prepended to the embedded text."""

    name = "contextual-heading"

    def __init__(self, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> None:
        self.chunk_tokens = chunk_tokens

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.parse_failed or not doc.blocks:
            return []

        degraded = not doc.has_structure
        indexed = list(enumerate(doc.blocks))
        chunks: list[Chunk] = []

        for order, run in enumerate(_pack(indexed, self.chunk_tokens)):
            body = "\n\n".join(block.text for _, block in run).strip()
            if not body:
                continue

            first_index = run[0][0]
            path = doc.heading_path_at(first_index)
            embed = f"{' > '.join(path)}\n\n{body}" if path else body
            pages = [block.page for _, block in run if block.page is not None]

            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}::{self.name}::{order}",
                    raw_text=body,
                    embed_text=embed,
                    return_text=body,
                    heading_path=path,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    strategy=self.name,
                    degraded=degraded,
                    degraded_reason=_NO_STRUCTURE if degraded else "",
                )
            )

        return chunks


class ParentDocChunker:
    """Embed a small span, return the section around it, bounded."""

    name = "parent-doc"

    def __init__(
        self,
        child_tokens: int = DEFAULT_CHILD_TOKENS,
        max_parent_tokens: int = DEFAULT_PARENT_TOKENS,
    ) -> None:
        self.child_tokens = child_tokens
        self.max_parent_tokens = max_parent_tokens

    def _bounded_parent(self, section_text: str, child_text: str) -> str:
        """The section, trimmed to a window around the child when it is too long.

        An unbounded parent is not just wasteful, it corrupts the comparison. On documents
        where few headings are detected, a "section" becomes most of the document — measured
        at a 38,000-token median on NIST — so this arm would return a large fraction of the
        corpus for every hit and score near-perfect recall without retrieving anything well.
        """
        encoding = get_encoding()
        tokens = encoding.encode(section_text)
        if len(tokens) <= self.max_parent_tokens:
            return section_text

        position = section_text.find(child_text)
        if position < 0:
            return encoding.decode(tokens[: self.max_parent_tokens]).strip()

        # Centre the window on the child so it keeps context from both sides.
        before = len(encoding.encode(section_text[:position]))
        child_size = len(encoding.encode(child_text))
        margin = max(0, (self.max_parent_tokens - child_size) // 2)
        start = max(0, before - margin)
        return encoding.decode(tokens[start : start + self.max_parent_tokens]).strip()

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.parse_failed or not doc.blocks:
            return []

        degraded = not doc.has_structure
        chunks: list[Chunk] = []
        order = 0

        for section in self._sections(doc):
            section_text = "\n\n".join(block.text for _, block in section).strip()
            if not section_text:
                continue

            first_index = section[0][0]
            path = doc.heading_path_at(first_index)
            pages = [block.page for _, block in section if block.page is not None]

            for child in _pack(section, self.child_tokens):
                body = "\n\n".join(block.text for _, block in child).strip()
                if not body:
                    continue
                chunks.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        chunk_id=f"{doc.doc_id}::{self.name}::{order}",
                        raw_text=body,
                        embed_text=body,
                        # The section around the hit, so a precise match still returns the
                        # context it sits in — bounded, so it cannot return the document.
                        return_text=self._bounded_parent(section_text, body),
                        heading_path=path,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        strategy=self.name,
                        degraded=degraded,
                        degraded_reason=_NO_STRUCTURE if degraded else "",
                    )
                )
                order += 1

        return chunks

    @staticmethod
    def _sections(doc: ParsedDoc) -> list[list[tuple[int, Block]]]:
        """Split the document at headings. Without headings, the document is one section."""
        sections: list[list[tuple[int, Block]]] = []
        current: list[tuple[int, Block]] = []

        for index, block in enumerate(doc.blocks):
            if block.type is BlockType.HEADING and current:
                sections.append(current)
                current = []
            current.append((index, block))

        if current:
            sections.append(current)
        return sections
