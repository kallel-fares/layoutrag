"""The chunk type.

The one thing to get right early. A chunk carries three *different* pieces of text,
and conflating them quietly breaks half the strategies:

- ``embed_text``    what gets embedded and searched over
- ``return_text``   what gets handed to the generator when this chunk is retrieved
- ``raw_text``      the verbatim span from the parsed document, with no decoration

For plain fixed-size chunking all three are the same string, which is exactly why it is
tempting to use one field. They diverge as soon as a strategy is doing anything interesting:

- ``sentence-window`` embeds one sentence but returns that sentence plus its neighbours
- ``parent-doc``      embeds a small piece but returns the whole enclosing section
- ``contextual-*``    embeds the raw span with a heading path or LLM-written summary
  prepended, but returns the raw span so the generator isn't fed synthetic text

Scoring uses ``return_text``: that is what a generator would actually see, so it is what
relevance has to be judged against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A retrievable unit of a document."""

    model_config = {"frozen": True}

    doc_id: str
    """Identifies the source document. Stable across parsers."""

    chunk_id: str
    """Unique within a (document, chunking strategy) pair."""

    raw_text: str
    """The verbatim span from the parsed document. Never decorated."""

    embed_text: str = ""
    """What gets embedded. Defaults to ``raw_text``."""

    return_text: str = ""
    """What gets returned on retrieval. Defaults to ``raw_text``."""

    page_start: int | None = None
    page_end: int | None = None

    heading_path: tuple[str, ...] = ()
    """Enclosing headings, outermost first. Empty when the parser found no structure."""

    strategy: str = ""
    """Name of the chunking strategy that produced this chunk."""

    degraded: bool = False
    """True when the strategy could not do its real job.

    A structure-aware chunker fed a flat-text parse still has to emit chunks, but it did
    not do what it claims to do. Recording that is the difference between "this strategy
    had no effect" and "this strategy never ran" — which look identical in a results table
    unless the distinction is carried explicitly.
    """

    degraded_reason: str = ""

    meta: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_text_fields(self) -> Chunk:
        # Both default to raw_text, so simple strategies stay simple, but the fields
        # exist from day one and nothing needs retrofitting when a strategy diverges.
        if not self.embed_text:
            object.__setattr__(self, "embed_text", self.raw_text)
        if not self.return_text:
            object.__setattr__(self, "return_text", self.raw_text)
        if self.degraded and not self.degraded_reason:
            raise ValueError("degraded chunks must say why")
        return self

    @property
    def is_expanded(self) -> bool:
        """True when more text is returned than was embedded (sentence-window, parent-doc)."""
        return len(self.return_text) > len(self.embed_text)

    @property
    def is_enriched(self) -> bool:
        """True when the embedded text carries context the raw span doesn't (contextual-*)."""
        return len(self.embed_text) > len(self.raw_text)
