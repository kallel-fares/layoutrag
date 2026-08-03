"""The parser interface.

A parser turns a PDF into a :class:`~layoutrag.blocks.ParsedDoc`. Structural typing, so a
client can supply their own parser without importing anything from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from layoutrag.blocks import ParsedDoc


@runtime_checkable
class Parser(Protocol):
    name: str

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one PDF.

        Must not raise on a corrupt, encrypted, or unreadable file — return a
        :class:`ParsedDoc` with ``parse_failed`` set instead. Failure rate is a metric the
        parser comparison depends on, so failures have to survive as data.
        """
        ...
