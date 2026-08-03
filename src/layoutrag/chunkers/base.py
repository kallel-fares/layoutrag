"""The chunker interface and shared tokenisation.

A chunker turns a :class:`~layoutrag.blocks.ParsedDoc` into :class:`~layoutrag.chunk_type.Chunk`
objects. Structural typing, so a client can drop in their own without importing our types.

Chunk sizes are measured in tokens rather than characters throughout, because that is the
unit the embedding model and the context budget are both denominated in. Counting
characters would make "512" mean different amounts of text in different documents.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import tiktoken

from layoutrag.blocks import ParsedDoc
from layoutrag.chunk_type import Chunk

# text-embedding-3-small tokenises with cl100k_base.
ENCODING_NAME = "cl100k_base"


@functools.lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    """The tokeniser, loaded once.

    tiktoken fetches its vocabulary on first use and caches it on disk, so building this
    repeatedly would be slow for no reason.
    """
    import tiktoken

    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        """Split a parsed document into retrievable chunks.

        A chunker that needs structure the parse doesn't have must still return chunks, and
        must mark them ``degraded`` with a reason. Silently returning ordinary chunks would
        make "this strategy had no effect" indistinguishable from "this strategy never ran".
        """
        ...
