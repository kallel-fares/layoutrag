# layoutrag

Compare chunking strategies for RAG over real PDFs — and see what each one actually
retrieves.

Point it at a folder of PDFs. It indexes them under several chunking strategies at once,
then for any question shows you, side by side, the chunks each strategy returned, their
scores, and the document structure the parser found behind them.

```bash
uv run layoutrag cost  ./my-pdfs                          # price it, zero API calls
uv run layoutrag query ./my-pdfs "what is the notice period?"
uv run layoutrag query ./my-pdfs "..." --local            # no API, no key
```

Runs on your machine. Documents are never uploaded anywhere. Embeddings go to the OpenAI
API with your own key by default; `--local` runs fully offline with a local model.

---

## Why this exists

Any RAG build forces a choice of parser, chunker, chunk size, and whether to rerank. Those
get picked either way — the only question is whether the answer to *"why did you chunk it
that way?"* is a measured number or a shrug.

So the defaults here are measured, on two corpora of real PDFs, with the method written
down and the numbers committed.

---

## What the numbers say

### 1. The best chunking strategy depends on the document

The same strategy, opposite results, decided by whether the PDF has real heading typography.

| corpus | `contextual-heading` | `fixed` | verdict |
|---|---|---|---|
| **NIST standards** (headings exist) | **0.436** | 0.353 | **+8.3 nDCG** |
| **CUAD contracts** (no headings) | 0.411 | **0.460** | **−4.9 nDCG** |

On documents with genuine heading structure, prepending the heading path to what gets
embedded is worth **+8.3 nDCG and +12.7 recall@10**, and it holds at an equal token budget
so it is not buying the win with context. On contracts converted from HTML — which carry no
heading typography at all — the same strategy *loses* to plain fixed-size chunking.

The pipeline knew which case applied before any score was computed: all 15,013 CUAD chunks
were flagged `degraded`, because the parser found no headings to use.

**Practical form:** check whether your PDFs have heading typography. If they do,
heading-aware chunking is worth ~8 points. If they don't, it is worse than doing nothing.

### 2. The expensive ML parser lost — and the reason is silent text loss

| parser | strategy | nDCG@10 | R@10 | headings found |
|---|---|---|---|---|
| `pypdfium2-fontsize` | contextual-heading | **0.436** | **0.611** | 478 |
| `pypdfium2-fontsize` | fixed | 0.353 | 0.484 | — |
| `docling` | contextual-heading | 0.305 | 0.432 | 6,579 |
| `docling` | fixed | 0.302 | 0.411 | — |

docling finds **13.8x more headings** and retrieval gets *worse*. The heading advantage
disappears entirely: +8.3 nDCG under font metrics, **+0.3** under docling.

The cause is text loss. **docling recovers 79.2% of the characters plain extraction finds**
— median 86% per document, 17.4% on the worst — while reporting success on all sixty. A
layout model classifies regions; regions it does not classify simply do not appear. No
retrieval strategy can return an answer that is not in the index.

**Practical form:** on born-digital PDFs, the parser that is 258x slower and needs a
torch-sized ingestion box produced worse retrieval than reading font metrics already in the
file. Its case is scanned documents needing OCR — out of scope here, and untested.

### 3. `parent-doc` wins on rank-1, not on recall — and only if you can afford the context

| corpus | scoring | R@1 | R@10 | median tokens |
|---|---|---|---|---|
| CUAD | k=10 | **0.344** | **0.772** | **18,158** |
| CUAD | 4000-token budget | **0.344** | 0.381 | 2,048 |
| CUAD, `fixed` | k=10 | 0.145 | 0.682 | 5,118 |
| CUAD, `fixed` | 4000-token budget | 0.145 | **0.614** | 3,584 |

At k=10 `parent-doc` looks decisive. It is also returning **3.5x the context**. Held to an
equal token budget its recall@10 collapses from 0.772 to 0.381 while `fixed` holds 0.614.

What survives is real: `parent-doc` is **2.4x better at rank 1** and has the best
quality-per-token. So it is the choice when you want the top hit correct and can pay for
context, and `fixed` is the choice when you want breadth per token.

Every arm is scored twice off one retrieval pass for exactly this reason. A conventional
fixed-k table would have published the k=10 column as a chunking result.

### 4. Overlap is worth 2–3 points

`fixed` beats `fixed-no-overlap` by 2.8 nDCG on CUAD and 2.4 on NIST. Small, consistent,
cheap.

---

## Method

**Corpora.** [CUAD](https://www.atticusprojectai.org/cuad) — 510 commercial contracts,
CC BY 4.0, with clause spans annotated under lawyer supervision. NIST Special Publications
— 60 documents, US Government work, public domain. Both fetched by script, never committed.

**Questions.** 300 for CUAD, built from the lawyer-annotated clause spans. 95 for NIST,
written by hand after reading each passage.

Hand-written matters: a model shown a passage and asked for a question reuses that
passage's vocabulary, so retrieval then succeeds on lexical overlap and every strategy
scores well for a reason unrelated to chunking. The NIST questions are deliberately worded
away from their source — "chip-off" asked as "physically detaching memory", "Bluetooth" as
"wireless personal-area devices".

Questions also name the document they are about. Without that, one query is searched against
every document while the answer sits in exactly one — an early version scored recall@10 of
**0.027**, which is what 1-in-437 looks like.

**Relevance.** A chunk counts when it contains **at least 50% of the gold span** — recall of
the gold span, not the fraction of the chunk that is gold. Defined the other way, a
2,000-token section containing the answer verbatim would be judged irrelevant for having
other text around it, and the study would measure chunk size while reporting it as chunking
strategy. `tests/test_relevance.py` asserts size invariance directly.

Scoring uses the **returned** text, since that is what a generator receives — not the
embedded text, which would credit `contextual-heading` for breadcrumbs nobody sees.

**Search is exact**, not approximate. ANN trades recall for speed, and that trade is a
confound: an arm could lose because the index missed a neighbour rather than because its
chunking was worse.

**No significance testing.** That apparatus exists to survive peer review; this exists to
pick defensible defaults. At n=300 the standard error on a paired difference is roughly
±2.6 points, so the 8-point gaps are solid and the 2–3 point ones are suggestive. Where a
difference is inside noise, it is said to be.

---

## Cost

**$0.73 total** — two corpora, 10 indexes, 112,000 chunks, 32.4M tokens embedded. Projected
$0.6471 before any API call was made; actual came in 1.8% over.

Nothing spends without pricing itself first. `layoutrag cost` reports what a corpus would
cost with zero API calls, because chunking is free and yields the exact token counts
embedding is billed for. Ceilings are enforced in code — per-run and cumulative, persisted
across processes — so a loop that re-embeds a corpus stops instead of arriving on an
invoice.

Full measured breakdown in [`costs.md`](costs.md): per-page parse rates, storage, local
embedding throughput, and the 3.6x cost of docling's OCR default.

---

## Reproducing

```bash
uv sync --extra dev --extra run --extra eval
uv run python scripts/fetch_cuad.py
uv run python scripts/fetch_nist.py --limit 60
uv run python scripts/build_questions.py
uv run python scripts/build_indexes.py          # ~$0.65
uv run python scripts/run_eval.py --corpus cuad
uv run python scripts/run_eval.py --corpus nist
```

For the parser comparison, add `--extra structure` and run
`scripts/parser_comparison.py --batch 10` repeatedly — parses are content-hashed, so it
resumes where it left off.

Results land in `results/`.

---

## Limits

One embedder, two corpora, no reranking, no significance testing, and OCR disabled
throughout. The CUAD numbers use a hosted embedding model, which can change behind a stable
name — the `--local` path is the reproducible counterpart. What was deliberately cut, and
why, is in [`not-implemented.md`](not-implemented.md).

## License

Apache-2.0. Question sets CC-BY-4.0.
