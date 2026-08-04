"""Score every built index against the question set.

Each arm is scored twice off one retrieval pass: at k=10, and at a ~4000-token context
budget. Reporting only the first would let an arm win by returning more text; reporting
both makes the trade visible as a number.

    uv run python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from layoutrag.embedders import OpenAIEmbedder
from layoutrag.eval import GoldSpan, score_query, score_run
from layoutrag.index import VectorIndex
from layoutrag.rerank import DEFAULT_DEPTH, CrossEncoderReranker, NoReranker

INDEX_DIR = Path("data/indexes")
QUESTIONS_FOR = {
    "cuad": Path("data/questions_cuad.json"),
    "nist": Path("data/questions_nist.json"),
    "nistdocling": Path("data/questions_nist.json"),
}
RESULTS = Path("results")


def load_index(path: Path) -> VectorIndex:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return VectorIndex(payload["chunks"], payload["vectors"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cuad")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument(
        "--rerank",
        action="store_true",
        help=f"pull {DEFAULT_DEPTH} candidates and reorder with a local cross-encoder",
    )
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_FOR[args.corpus].read_text())
    indexes = sorted(INDEX_DIR.glob(f"{args.corpus}__*.pkl"))
    if not indexes:
        print(f"No indexes for {args.corpus} — run scripts/build_indexes.py first")
        return 1

    # Query embedding is charged once and reused across every arm: the questions do not
    # change between strategies, so re-embedding per arm would multiply cost by the number
    # of arms for no reason.
    embedder = OpenAIEmbedder()
    texts = [q["question"] for q in questions]
    unique = sorted(set(texts))
    print(f"embedding {len(unique)} distinct questions (of {len(texts)})")
    vectors = embedder.embed(unique)
    by_text = dict(zip(unique, vectors, strict=True))
    query_vectors = np.array([by_text[t] for t in texts])

    # Built once and reused across arms: the model load dominates, and reloading it per
    # arm would multiply a one-off cost by the number of strategies.
    reranker = CrossEncoderReranker() if args.rerank else NoReranker()
    if args.rerank:
        print(f"reranking with {reranker.name}, depth {DEFAULT_DEPTH}")

    RESULTS.mkdir(exist_ok=True)
    rows = []

    print(
        f"\n{'strategy':22} {'scoring':14} {'R@1':>6} {'R@5':>6} {'R@10':>6} "
        f"{'nDCG':>6} {'MRR':>6} {'tokens':>8} {'nDCG/1k':>8}"
    )
    print("-" * 96)

    for path in indexes:
        strategy = path.stem.split("__", 1)[1]
        index = load_index(path)

        for label, budget in (("k=10", None), (f"{args.budget}tok", args.budget)):
            started = time.perf_counter()
            per_query = []
            for question, vector in zip(questions, query_vectors, strict=True):
                gold = [GoldSpan(text=g, doc_id=question["doc_id"]) for g in question["gold"]]
                # The reranker gets the user's actual question, not the augmented
                # retrieval query. The document-title prefix helps the vector stage find
                # the right document and actively hurts the cross-encoder, which scores
                # the pair directly: measured at 0.570 nDCG on the question against 0.346
                # on the retrieval query, versus 0.436 for no reranking at all.
                rerank_query = question.get("base_question") or question["question"]
                if budget is None:
                    # Retrieve wide, then let the cross-encoder pick. Without reranking the
                    # depth collapses to k, so the control arm is unaffected.
                    depth = DEFAULT_DEPTH if args.rerank else 10
                    hits = reranker.rerank(rerank_query, index.search(vector, depth), 10)
                else:
                    hits = index.search_token_budget(vector, budget=budget)
                    if args.rerank:
                        hits = reranker.rerank(rerank_query, hits, len(hits))
                per_query.append(score_query(question["question"], hits, gold))

            metrics = score_run(per_query)
            elapsed = time.perf_counter() - started
            print(
                f"{strategy:22} {label:14} {metrics.recall_at_1:6.3f} "
                f"{metrics.recall_at_5:6.3f} {metrics.recall_at_10:6.3f} "
                f"{metrics.ndcg_at_10:6.3f} {metrics.mrr:6.3f} "
                f"{metrics.median_tokens_returned:8.0f} {metrics.ndcg_per_1k_tokens:8.3f}",
                flush=True,
            )
            rows.append(
                {
                    "corpus": args.corpus,
                    "strategy": strategy,
                    "reranker": reranker.name,
                    "scoring": label,
                    "queries": metrics.queries,
                    "recall_at_1": metrics.recall_at_1,
                    "recall_at_5": metrics.recall_at_5,
                    "recall_at_10": metrics.recall_at_10,
                    "ndcg_at_10": metrics.ndcg_at_10,
                    "mrr": metrics.mrr,
                    "median_tokens_returned": metrics.median_tokens_returned,
                    "ndcg_per_1k_tokens": metrics.ndcg_per_1k_tokens,
                    "chunks_indexed": len(index),
                    "seconds": round(elapsed, 1),
                }
            )

    suffix = "-rerank" if args.rerank else ""
    out = RESULTS / f"{args.corpus}{suffix}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")
    print(f"query embedding cost: ${embedder.usd_spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
