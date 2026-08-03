"""docling parser — the ML layout model, and the expensive end of the comparison.

docling runs a layout model over each page rather than reading font metrics, so it can in
principle recover structure that typography alone doesn't express: section headers set in
bold at body size, table boundaries, reading order in multi-column pages.

It costs roughly 1000x more per page than pypdfium2 and drags in torch, so it is gated
behind the ``structure`` extra and imported lazily. The point of having it is to find out
whether the cheap ``pypdfium2-fontsize`` parser gets close enough — because if it does, a
client's ingestion box drops from a few GB with an ML stack to a couple of hundred MB
without one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from layoutrag.blocks import Block, BlockType, ParsedDoc, mark_page_furniture

# docling's own label vocabulary, mapped onto ours. Anything unlisted becomes OTHER rather
# than being silently dropped, so an unexpected label shows up in the block counts instead
# of quietly shrinking the document.
_LABEL_MAP = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "paragraph": BlockType.PARAGRAPH,
    "text": BlockType.PARAGRAPH,
    "table": BlockType.TABLE,
    "list_item": BlockType.LIST_ITEM,
    "caption": BlockType.CAPTION,
    "page_footer": BlockType.FOOTER,
    "page_header": BlockType.FOOTER,
    "footnote": BlockType.FOOTER,
    "formula": BlockType.OTHER,
    "code": BlockType.OTHER,
    "picture": BlockType.OTHER,
}


class DoclingParser:
    """Structure-aware parsing via docling's layout model."""

    name = "docling"

    def __init__(self) -> None:
        self._converter: Any | None = None

    def _get_converter(self) -> Any:
        # Imported and constructed lazily: importing docling pulls in torch, and building
        # the converter downloads layout models on first use. Neither should happen just
        # because something imported this module.
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            # OCR is out of scope, and leaving it on made docling download PaddleOCR
            # weights and run character recognition over born-digital pages that already
            # carry a text layer — most of the measured cost, for nothing. Scanned
            # documents are excluded from these corpora and counted, not recovered.
            options.do_ocr = False
            # Table structure is the part worth paying for: it is the clearest thing a
            # layout model recovers that font metrics cannot.
            options.do_table_structure = True

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, path: Path) -> ParsedDoc:
        try:
            result = self._get_converter().convert(str(path))
            document = result.document
        except Exception as exc:
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                parse_failed=True,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        blocks: list[Block] = []
        for item, level in document.iterate_items():
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue

            label = str(getattr(item, "label", "") or "").lower()
            block_type = _LABEL_MAP.get(label, BlockType.OTHER)

            blocks.append(
                Block(
                    type=block_type,
                    text=text,
                    page=_page_of(item),
                    # docling reports nesting depth directly, which is a better heading
                    # level than anything inferable from font size.
                    level=(level + 1) if block_type is BlockType.HEADING else None,
                )
            )

        page_count = len(getattr(document, "pages", ()) or ())

        if not blocks:
            return ParsedDoc(
                doc_id=path.stem,
                source_path=str(path),
                parser=self.name,
                page_count=page_count,
                parse_failed=True,
                failure_reason="docling produced no text — likely a scanned document",
            )

        return ParsedDoc(
            doc_id=path.stem,
            source_path=str(path),
            parser=self.name,
            blocks=tuple(mark_page_furniture(blocks, page_count)),
            page_count=page_count,
        )


def _page_of(item: Any) -> int | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    page = getattr(provenance[0], "page_no", None)
    return int(page) if page is not None else None
