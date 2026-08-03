"""Assemble the NIST question set from sampled passages and hand-written questions.

No API call. The questions in ``nist_questions`` were written by reading the passages, so
their wording is deliberately different from the source text — see that module for why that
matters more than the volume it costs.

    uv run python scripts/sample_passages.py --per-doc 2 --out data/nist_passages.json
    uv run python scripts/build_questions_nist.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nist_questions import QUESTIONS

PASSAGES = Path("data/nist_passages.json")
OUT = Path("data/questions_nist.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default=str(PASSAGES))
    args = ap.parse_args()

    source = Path(args.passages)
    if not source.exists():
        print(f"{source} missing — run scripts/sample_passages.py first")
        return 1

    passages = json.loads(source.read_text())
    questions = []

    for index, text in sorted(QUESTIONS.items()):
        if index >= len(passages):
            print(f"  passage {index} out of range, skipping")
            continue
        passage = passages[index]
        questions.append(
            {
                # Anchored to the publication, as the CUAD set had to be: without it, one
                # query is searched against 60 documents while the answer sits in exactly
                # one of them, and retrieval measures chance.
                "question": f"{passage['title']}: {text}",
                "base_question": text,
                "doc_id": passage["doc_id"],
                "page": passage["page"],
                "gold": [passage["text"]],
                "source": "passage sampled from the document; question written by hand",
                "curated": True,
            }
        )

    OUT.write_text(json.dumps(questions, indent=2))
    docs = len({q["doc_id"] for q in questions})
    print(f"{len(questions)} hand-written questions over {docs} documents -> {OUT}")
    print(f"distinct queries: {len({q['question'] for q in questions})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
