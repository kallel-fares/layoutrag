"""Turn CUAD's lawyer-annotated clauses into a retrieval question set.

CUAD ships 13,000+ clause spans annotated under the supervision of practising lawyers. That
is better ground truth than anything produced by hand here, and it already exists — so the
work is selecting and phrasing, not judging.

Two filters do the real work:

**Front matter is excluded.** Document name, parties, and the various dates all sit on page
one of every contract. Every chunking strategy finds them, so questions about them measure
nothing and would compress the differences between arms toward zero.

**Rare categories are excluded.** A category present in a handful of contracts cannot carry
enough questions to say anything.

    uv run python scripts/build_questions.py --per-category 12
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd

CSV = Path("data/cuad/CUAD_v1/master_clauses.csv")
OUT = Path("data/questions_cuad.json")

# Categories worth asking about, with the question a person would actually type. Chosen for
# being real clauses buried in the body of an agreement rather than cover-page facts.
QUESTIONS = {
    "Governing Law": "Which state or country's law governs this agreement?",
    "Anti-Assignment": "Can this agreement be assigned to another party?",
    "Cap On Liability": "Is there a cap on liability, and what is it?",
    "Audit Rights": "What audit or inspection rights does each party have?",
    "Termination For Convenience": "Can either party terminate for convenience?",
    "License Grant": "What licence is granted under this agreement?",
    "Post-Termination Services": "What obligations survive after termination?",
    "Exclusivity": "Is there an exclusivity obligation?",
    "Revenue/Profit Sharing": "How is revenue or profit shared?",
    "Insurance": "What insurance is each party required to carry?",
    "Minimum Commitment": "Is there a minimum purchase or volume commitment?",
    "Non-Compete": "Is there a non-compete restriction?",
    "Change Of Control": "What happens on a change of control?",
    "Uncapped Liability": "Which liabilities are uncapped?",
    "Ip Ownership Assignment": "Who owns intellectual property created under this agreement?",
    "Non-Transferable License": "Is the licence transferable?",
    "Renewal Term": "How does renewal work?",
    "Covenant Not To Sue": "Is there a covenant not to sue?",
    "Warranty Duration": "How long does the warranty last?",
    "Liquidated Damages": "Are there liquidated damages?",
}

# A gold span shorter than this is usually a fragment ("Nevada") rather than a clause, and
# short spans match by accident across unrelated contracts.
MIN_SPAN_CHARS = 80


def spans(cell: object) -> list[str]:
    if not isinstance(cell, str) or len(cell) < 4:
        return []
    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and len(s.strip()) >= MIN_SPAN_CHARS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not CSV.exists():
        print(f"{CSV} missing — run scripts/fetch_cuad.py first")
        return 1

    df = pd.read_csv(CSV)
    rng = random.Random(args.seed)
    questions = []

    for category, text in QUESTIONS.items():
        if category not in df.columns:
            print(f"  (no column {category!r}, skipping)")
            continue

        candidates = []
        for _, row in df.iterrows():
            found = spans(row[category])
            if found:
                candidates.append((str(row["Filename"]), found))

        rng.shuffle(candidates)
        for filename, found in candidates[: args.per_category]:
            questions.append(
                {
                    "question": text,
                    "category": category,
                    # Stems must match the PDF filenames the parser derives doc_id from.
                    "doc_id": Path(filename).stem,
                    "gold": found,
                    "source": "CUAD v1 clause annotations (lawyer-supervised)",
                }
            )

    by_category: dict[str, int] = {}
    for q in questions:
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1

    OUT.write_text(json.dumps(questions, indent=2))
    print(f"{len(questions)} questions over {len(by_category)} categories -> {OUT}")
    print(f"{'category':32} {'n':>4}")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {category:30} {count:4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
