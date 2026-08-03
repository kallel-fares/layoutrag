"""Relevance: does a retrieved chunk contain the gold evidence?

The single most important definition in the project, and the easiest place to get a wrong
answer that looks right.

A chunk counts as relevant when **it contains at least 50% of the gold snippet**. That is
recall *of the gold span*, not the fraction of the chunk that is gold.

Defining it the other way — "at least half this chunk is gold text" — makes large chunks
structurally unable to score, no matter how good they are. A 2000-token section containing
the answer verbatim would be judged irrelevant simply for having other text around it. The
study would then be measuring chunk size while reporting it as chunking strategy, and every
number would look plausible.

Gold evidence is stored as a text snippet rather than character offsets because offsets
shift between parsers. Storing offsets would make the parser comparison silently measure
nothing: the same gold span would land at different positions per parser and resolve
against none of them. Matching is fuzzy for the same reason — extraction differs in
whitespace, ligatures, and hyphenation between parsers, so exact substring matching would
report a parser as having destroyed evidence that it merely spelled differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from layoutrag.chunk_type import Chunk

DEFAULT_OVERLAP_THRESHOLD = 0.50

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Fold away the differences between parsers that are not differences in content."""
    # These are the characters being folded, so they must appear literally here.
    text = text.replace("\xa0", " ").replace("’", "'").replace("‘", "'")  # noqa: RUF001
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")  # noqa: RUF001
    # Hyphenation introduced by line wrapping: "termina-\ntion" is one word.
    text = re.sub(r"-\s*\n\s*", "", text)
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True)
class GoldSpan:
    """Evidence that answers a question, as text plus a page hint."""

    text: str
    page: int | None = None
    doc_id: str = ""


def coverage(chunk_text: str, gold: str) -> float:
    """Share of ``gold`` present in ``chunk_text``, from 0.0 to 1.0.

    Word-level rather than character-level, so a near-match that differs only in
    punctuation or spacing is not penalised. Uses a sliding window so the gold words have to
    appear *together*, not scattered across the chunk — otherwise a long chunk would score
    highly just for containing common words somewhere.
    """
    gold_words = normalise(gold).split()
    if not gold_words:
        return 0.0

    chunk_words = normalise(chunk_text).split()
    if not chunk_words:
        return 0.0

    window = len(gold_words)
    if len(chunk_words) <= window:
        return _bag_overlap(chunk_words, gold_words)

    # Slide a gold-sized window and keep the best local match, so overlap is measured
    # against a contiguous region rather than the whole document.
    best = 0.0
    step = max(1, window // 4)
    for start in range(0, len(chunk_words) - window + 1, step):
        best = max(best, _bag_overlap(chunk_words[start : start + window], gold_words))
        if best == 1.0:
            break
    return best


def _bag_overlap(candidate: list[str], gold: list[str]) -> float:
    from collections import Counter

    available = Counter(candidate)
    matched = 0
    for word in gold:
        if available[word] > 0:
            available[word] -= 1
            matched += 1
    return matched / len(gold)


def is_relevant(chunk: Chunk, gold: GoldSpan, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> bool:
    """Whether a chunk carries the gold evidence.

    Judged on ``return_text``, because that is what a generator would receive. Judging on
    ``embed_text`` would credit sentence-window and parent-doc for context they never
    return, and credit contextual-heading for breadcrumbs that are not document content.
    """
    if gold.doc_id and chunk.doc_id != gold.doc_id:
        return False
    return coverage(chunk.return_text, gold.text) >= threshold
