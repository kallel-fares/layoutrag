"""pypdfium2 parsers — the fast path.

Two of them, and the difference between them is a finding in its own right:

``pypdfium2``
    Plain text extraction. Everything comes back as paragraphs, no headings. This is what
    most RAG pipelines actually do.

``pypdfium2-fontsize``
    The same extraction, plus heading detection by clustering text-object font sizes.
    Costs almost nothing on top of the flat parse.

Having both makes the expensive question cheap to ask: docling runs an ML layout model at
seconds per page, so it is worth knowing how much of its benefit is recoverable from the
font sizes already sitting in the file. If the answer is "most of it", that matters far
more to a client than a leaderboard position.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from layoutrag.blocks import Block, BlockType, ParsedDoc

# A text object must be this much larger than body text to read as a heading. Chosen from
# the measured corpora: NIST bodies sit at 9-12pt with headings at 11-16pt, so a 15% margin
# separates them without catching bold-but-same-size runs.
HEADING_SIZE_RATIO = 1.15

# Headings are short. This rejects the large-type opening paragraphs and pull quotes that
# would otherwise dominate the heading list in report-style documents.
MAX_HEADING_CHARS = 120

# Walking text objects must recover at least this share of what plain extraction returns,
# or the structured parse is discarded in favour of the flat one. See parse() for why.
MIN_TEXT_RETENTION = 0.90

_BLANK_LINES = re.compile(r"\n\s*\n+")
_WS = re.compile(r"[ \t]+")


def _clean(text: str) -> str:
    # \xa0 and friends survive extraction and would otherwise split words oddly downstream.
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


# Two fragments are on the same line if their vertical centres sit within this fraction of
# the larger font size. Generous enough for subscripts and mixed sizes within a line.
_SAME_LINE_TOLERANCE = 0.5

# Consecutive lines join into one paragraph when the baseline gap is under this multiple of
# the font size. Beyond it, the gap reads as a paragraph break.
_PARAGRAPH_GAP = 1.8

# Fragments on one line separated by more than this multiple of the font size get a space
# between them; closer than that they are one word rendered in pieces.
_SPACE_GAP = 0.25


_WORD = re.compile(r"[A-Za-z0-9]")


def _looks_like_heading(text: str) -> bool:
    """Reject things that are merely set in large type but aren't headings.

    Large type catches horizontal rules drawn as underscores, page furniture, and stray
    punctuation from cover-page typography. A heading is short and mostly alphanumeric.
    """
    if len(text) > MAX_HEADING_CHARS:
        return False
    word_chars = len(_WORD.findall(text))
    if word_chars < 2:
        return False
    return word_chars / len(text) >= 0.5


Fragment = tuple[float, float, float, float, float, str]


def _assemble(fragments: list[Fragment]) -> list[tuple[float, str]]:
    """Merge text-object fragments into lines, then lines into paragraphs."""
    if not fragments:
        return []

    # Reading order: top of page down, then left to right.
    ordered = sorted(fragments, key=lambda f: (-f[4], f[1]))

    lines: list[tuple[float, float, float, str]] = []  # (size, centre_y, right_edge, text)
    for size, left, bottom, right, top, text in ordered:
        centre = (top + bottom) / 2
        if lines:
            prev_size, prev_centre, prev_right, prev_text = lines[-1]
            tolerance = max(size, prev_size) * _SAME_LINE_TOLERANCE
            if abs(centre - prev_centre) <= tolerance:
                # Whether a space belongs between two fragments is a question about the
                # horizontal gap, not about the characters. "800" + "-" + "140F" rendered
                # flush is the title "800-140F"; the same fragments spaced apart are not.
                gap = left - prev_right
                joiner = " " if gap > size * _SPACE_GAP else ""
                lines[-1] = (
                    max(size, prev_size),
                    prev_centre,
                    max(right, prev_right),
                    prev_text + joiner + text,
                )
                continue
        lines.append((size, centre, right, text))

    paragraphs: list[tuple[float, str]] = []
    last_centre: float | None = None
    last_size: float | None = None
    for size, centre, _right, text in lines:
        joinable = (
            last_centre is not None
            and last_size == size
            and abs(last_centre - centre) <= size * _PARAGRAPH_GAP
        )
        if joinable and paragraphs:
            paragraphs[-1] = (size, f"{paragraphs[-1][1]} {text}")
        else:
            paragraphs.append((size, text))
        last_centre, last_size = centre, size

    return [(size, _clean(text)) for size, text in paragraphs if _clean(text)]


class PdfiumParser:
    """Flat text extraction. No structure."""

    name = "pypdfium2"

    def parse(self, path: Path) -> ParsedDoc:
        try:
            pdf = pdfium.PdfDocument(path)
        except Exception as exc:
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                parse_failed=True,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        blocks: list[Block] = []
        try:
            for page_no in range(len(pdf)):
                page = pdf[page_no]
                textpage = page.get_textpage()
                raw = textpage.get_text_bounded()
                textpage.close()
                page.close()

                for para in _BLANK_LINES.split(raw):
                    cleaned = _clean(para)
                    if cleaned:
                        blocks.append(
                            Block(type=BlockType.PARAGRAPH, text=cleaned, page=page_no + 1)
                        )
            page_count = len(pdf)
        finally:
            pdf.close()

        if not blocks:
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                page_count=page_count,
                parse_failed=True,
                failure_reason="no extractable text — likely a scanned document",
            )

        return ParsedDoc(
            doc_id=path.stem,
            source_path=str(path),
            parser=self.name,
            blocks=tuple(blocks),
            page_count=page_count,
        )


class PdfiumFontSizeParser:
    """Text extraction plus heading detection from text-object font sizes."""

    name = "pypdfium2-fontsize"

    def parse(self, path: Path) -> ParsedDoc:
        try:
            pdf = pdfium.PdfDocument(path)
        except Exception as exc:
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                parse_failed=True,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        try:
            spans = self._collect_spans(pdf)
            page_count = len(pdf)
            plain_length = self._plain_text_length(pdf)
        finally:
            pdf.close()

        # Object bounds do not always map onto extractable regions. On some producers'
        # files — CUAD's HTML->PDF conversions especially — walking text objects recovers a
        # fraction of the page, or nothing at all. Retention is checked against plain
        # extraction so that never passes silently: losing 90% of a contract while still
        # emitting a plausible-looking ParsedDoc would corrupt every downstream number
        # invisibly. Falling back costs one cheap extra pass and loses only the structure,
        # which was not recoverable on these files anyway.
        assembled_length = sum(len(text) for _, _, text in spans)
        if assembled_length < plain_length * MIN_TEXT_RETENTION:
            fallback = PdfiumParser().parse(path)
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                blocks=fallback.blocks,
                page_count=page_count,
                parse_failed=fallback.parse_failed,
                failure_reason=fallback.failure_reason,
            )

        body_size = self._body_size(spans)
        levels = self._heading_levels(spans, body_size)

        blocks = []
        for page_no, size, text in spans:
            level = levels.get(size)
            if level is not None and _looks_like_heading(text):
                blocks.append(Block(type=BlockType.HEADING, text=text, page=page_no, level=level))
            else:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text, page=page_no))

        return ParsedDoc(
            doc_id=path.stem,
            source_path=str(path),
            parser=self.name,
            blocks=tuple(blocks),
            page_count=page_count,
        )

    def _collect_spans(self, pdf: pdfium.PdfDocument) -> list[tuple[int, float, str]]:
        """Assembled runs of text: (page number, effective font size, text).

        A PDF text object is not a line. Producers split a single visual line across many
        objects at kerning pairs, style changes, and drop caps — which is why raw objects
        yield fragments like "NIST Special Publication 800" / "-" / "140F" instead of one
        heading. So objects are assembled into lines by vertical position, then lines into
        paragraphs by font size and spacing.

        pdfium also reports nominal font size separately from the transform applied to it,
        and some producers set the nominal size to 1.0 and do all scaling in the matrix.
        Multiplying by the matrix scale recovers the size actually rendered.
        """
        spans: list[tuple[int, float, str]] = []

        for page_no in range(len(pdf)):
            page = pdf[page_no]
            textpage = page.get_textpage()

            fragments = []
            for obj in page.get_objects():
                if obj.type != 1:  # text objects only
                    continue
                try:
                    size = round(obj.get_font_size() * abs(obj.get_matrix().a), 1)
                    left, bottom, right, top = obj.get_bounds()
                    text = _clean(textpage.get_text_bounded(left, bottom, right, top))
                except Exception:
                    continue
                if text:
                    fragments.append((size, left, bottom, right, top, text))

            textpage.close()
            page.close()

            spans.extend((page_no + 1, size, text) for size, text in _assemble(fragments))

        return spans

    @staticmethod
    def _plain_text_length(pdf: pdfium.PdfDocument) -> int:
        """Characters plain extraction finds, as the yardstick for retention."""
        total = 0
        for page_no in range(len(pdf)):
            page = pdf[page_no]
            textpage = page.get_textpage()
            total += len(_clean(textpage.get_text_bounded()))
            textpage.close()
            page.close()
        return total

    @staticmethod
    def _body_size(spans: list[tuple[int, float, str]]) -> float:
        """Most common font size weighted by characters, not by object count.

        Weighting by object count would let a document with many short headings and few
        long paragraphs elect a heading size as the body size.
        """
        weighted: Counter[float] = Counter()
        for _, size, text in spans:
            weighted[size] += len(text)
        return weighted.most_common(1)[0][0]

    @staticmethod
    def _heading_levels(spans: list[tuple[int, float, str]], body_size: float) -> dict[float, int]:
        """Map each above-body font size to a heading level, largest size = level 1."""
        threshold = body_size * HEADING_SIZE_RATIO
        sizes = sorted({size for _, size, _ in spans if size >= threshold}, reverse=True)
        return {size: level for level, size in enumerate(sizes, start=1)}
