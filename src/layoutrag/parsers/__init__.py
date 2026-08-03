"""PDF parsers.

``DoclingParser`` is importable from here without pulling in torch — docling itself is
imported only when a converter is first built, so the ``structure`` extra stays genuinely
optional at import time.
"""

from layoutrag.parsers.base import Parser
from layoutrag.parsers.docling_parser import DoclingParser
from layoutrag.parsers.pdfium import PdfiumFontSizeParser, PdfiumParser

__all__ = ["DoclingParser", "Parser", "PdfiumFontSizeParser", "PdfiumParser"]
