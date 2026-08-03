"""Sample clean NIST passages for hand-written questions.

Questions for this corpus are written by hand rather than generated. An LLM shown a passage
and asked for a question it answers reuses that passage's vocabulary, so retrieval then
succeeds on lexical overlap and every arm scores high for the wrong reason. Writing them by
hand means the wording can be deliberately different from the source — "chip-off" becomes
"physically detaching memory" — which is the inflation removed rather than caveated.

This script only does the selection. Passages have to be clean before they are worth
reading: a first sample returned tables of contents and mangled cryptographic notation,
neither of which can serve as gold evidence.

    uv run python scripts/sample_passages.py --per-doc 2 --out passages.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from layoutrag.blocks import Block, BlockType
from layoutrag.cache import Cache
from layoutrag.parsers import PdfiumFontSizeParser
from layoutrag.pipeline import parse_corpus

MIN_CHARS, MAX_CHARS = 450, 1200

_BOILERPLATE = re.compile(
    r"this publication is available|nist\.sp\.|doi\.org|acknowledgement|"
    r"abstract$|keywords$|table of contents|list of (figures|tables)",
    re.IGNORECASE,
)


def is_clean(text: str) -> bool:
    """Reject passages that are not prose.

    Three failure modes seen in the first sample, all of which would be useless as gold
    evidence and none of which a length filter catches:

    - tables of contents and figure lists, which are mostly dot leaders
    - mangled mathematical notation, where extraction interleaves subscripts into words
    - heavily numeric tables rendered as a paragraph
    """
    if _BOILERPLATE.search(text):
        return False

    # Dot leaders: "Table 4 PIV Authentication Mechanisms......... 13"
    if text.count(".") / len(text) > 0.08:
        return False

    letters = sum(1 for c in text if c.isalpha())
    if letters / len(text) < 0.65:
        return False

    words = text.split()
    if len(words) < 60:
        return False

    # Real prose is mostly ordinary words. Mangled notation produces long runs of glued
    # tokens like "K(ix=2" and "KOLKOUT", which show up as unusually long "words".
    odd = sum(1 for w in words if len(w) > 18 or (any(c.isdigit() for c in w) and len(w) > 8))
    return odd / len(words) <= 0.06


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="passages.json")
    ap.add_argument("--start", type=int, default=0, help="skip this many documents")
    ap.add_argument("--docs", type=int, help="how many documents to draw from")
    args = ap.parse_args()

    paths = sorted(Path("data/nist").glob("*.pdf"))[args.start :]
    if args.docs:
        paths = paths[: args.docs]

    docs, _ = parse_corpus(paths, PdfiumFontSizeParser(), Cache())
    rng = random.Random(args.seed)
    sampled = []

    for doc in docs:
        if doc.parse_failed:
            continue
        headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
        title = next((h for h in headings if len(h) > 20), doc.doc_id)

        candidates: list[Block] = [
            b
            for b in doc.content_blocks
            if b.type is BlockType.PARAGRAPH
            and MIN_CHARS <= len(b.text) <= MAX_CHARS
            and (b.page or 0) > 4
            and is_clean(b.text)
        ]
        rng.shuffle(candidates)
        for block in candidates[: args.per_doc]:
            sampled.append(
                {
                    "doc_id": doc.doc_id,
                    "title": title[:80],
                    "page": block.page,
                    "text": block.text,
                }
            )

    Path(args.out).write_text(json.dumps(sampled, indent=2))
    print(f"{len(sampled)} clean passages from {len({s['doc_id'] for s in sampled})} docs")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
