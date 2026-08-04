# layoutrag

## What this is

A RAG pipeline for PDF collections. It reads your documents, chunks them, indexes them, and
finds the passages that answer a question.

Every default in it was chosen by measurement, and the measurements are in this repo.

```bash
uv run layoutrag cost  ./my-docs                      # price it, zero API calls
uv run layoutrag query ./my-docs "what is the notice period?"
uv run layoutrag query ./my-docs "..." --local        # fully offline, no key
```

Runs on your machine. Your documents are never uploaded anywhere.

## Who it is for

Anyone building search or Q&A over a pile of PDFs: contracts, standards, reports, manuals,
policies.

Building one forces a dozen choices. Which parser. How to chunk the documents. Whether to
add a reranker. Most of those choices get made by guesswork, and the wrong guess can cost
you 13 points of accuracy or a server four times bigger than you need.

This repo shows what those choices are worth, with numbers, so you can pick on evidence.

## Scope

**In:** parsing PDFs, chunking, embedding, indexing, searching, reranking. Cost
control and caching throughout.

**Not in yet:** writing answers, keyword search, file types other than PDF, a web interface.
See [Status](#status).

**Not planned:** scanned documents and OCR, model training, hosting.

## Data

Two public sets, both downloaded by script and never committed here.

| Set | What | Size | Licence |
|---|---|---|---|
| CUAD | Commercial contracts, clauses marked up by lawyers | 510 documents, 9,348 pages | CC BY 4.0 |
| NIST SP | Technical security standards | 60 documents, 4,060 pages | Public domain |

Questions: 300 built from CUAD's lawyer markup, 95 written by hand for NIST after reading
each passage.

Hand writing the NIST questions matters. A model shown a passage and asked to write a
question reuses that passage's wording, and then search succeeds on matching words alone.
These are worded differently from the source on purpose.

## Pipeline

```
PDF  ->  parse  ->  chunk  ->  embed  ->  index  ->  search  ->  rerank  ->  passages
```

| Stage | What runs | Options |
|---|---|---|
| Parse | pypdfium2 | plain text, font-size heading detection, or docling |
| Chunk | 512 tokens, 12.5% overlap | fixed, no-overlap, heading-aware, parent-document |
| Embed | text-embedding-3-small | or a local model, fully offline |
| Index | exact vector search | |
| Search | top 50 by similarity | |
| Rerank | ms-marco-MiniLM-L6-v2, local | or off |

Parsing and chunking results are cached by content, so changing one stage does not redo the
others.

## Results

Tested on both document sets, 395 questions. Numbers are nDCG@10, a standard retrieval
score from 0 to 1. Full data in [`results/`](results/).

**Effect of each change:**

| Change | Quality | Cost |
|---|---|---|
| Add a reranker, done right | **+13.4** | free |
| Add a reranker, done wrong | −9.0 | free |
| Match chunking to the document type | **+8.3** | free |
| Switch to an AI page parser | **−13.1** | 258x slower, 1 GB more dependencies |

**Reranking, on the standards set:**

| | Quality |
|---|---|
| No reranker | 0.436 |
| Reranker, fed the search query | 0.346 |
| Reranker, fed the user's question | **0.570** |

**Heading-aware chunking, by document type:**

| Documents | Heading-aware | Plain | Result |
|---|---|---|---|
| Standards, real headings | **0.436** | 0.353 | **+8.3** |
| Contracts converted from HTML, no headings | 0.411 | **0.460** | **−4.9** |

**Parsers, on the standards set:**

| Parser | Quality | Headings found | Text recovered | Speed |
|---|---|---|---|---|
| Font-size detection | **0.436** | 478 | 100% | 0.004 s/page |
| docling | 0.305 | 6,579 | 79.2% | 1.03 s/page |

## Observations

**Rerankers pay more than anything else and cost nothing.** A reranker re-reads the top
results with the question in hand and reorders them. It runs locally, so there is no
per-query charge.

**A reranker fed the wrong text does worse than no reranker.** Search queries get padded
with extra context to help find the right document. That padding confuses a reranker, which
compares the question directly against each result. 22 points swing on one variable. This
pipeline keeps the two apart.

**Chunking strategy depends on your documents.** If your PDFs have real heading formatting,
adding the section heading to each chunk is worth 8 points. If they do not, it costs you 5.
The pipeline detects which case applies and tells you before you spend anything.

**The expensive parser lost.** docling found 13.8x more headings and still retrieved worse,
because it only recovers 79.2% of the text a basic reader finds while reporting success on
every document. Worst case was 17.4%. Search cannot find an answer that never made it in.
Its strength is scanned paper, which we did not test.

**Watch how much text a chunking strategy returns.** One looked best until we counted. It was handing back 3.5x more text than the others. Give every strategy the same
amount of room and its score drops from 0.772 to 0.381. Everything here is scored both ways.

## Cost

**$0.73** for the whole study: two document sets, 10 indexes, 112,000 chunks, 32.4M tokens.
Predicted $0.6471 before spending anything, came in 1.8% over.

`layoutrag cost` prices any folder without calling the API. You see the number first.

Spending limits are enforced in code, per run and in total, and persist across runs. A loop
that re-processes the same documents gets stopped before it reaches your invoice.

Full breakdown in [`costs.md`](costs.md).

## Status

| Stage | Working | Next |
|---|---|---|
| Reading files | PDF, 3 parsers, 4 chunking strategies, cloud and local embedding | markdown, txt, docx, 3 more chunking strategies, adding files without a rebuild |
| Searching | vector search, reranking | keyword search blended in, filters |
| Answering | | written answers with sources |
| Using it | command line | web page |
| Running costs | spending limits, caching, resumable processing, progress | |

No answer writing yet. Today it finds and ranks the right passages.

## Running it

```bash
uv sync --extra dev --extra run --extra eval
uv run python scripts/fetch_cuad.py
uv run python scripts/fetch_nist.py --limit 60
uv run python scripts/build_questions.py
uv run python scripts/build_indexes.py                    # ~$0.65
uv run python scripts/run_eval.py --corpus nist --rerank
```

## Limits

One embedding model, two document sets, no statistical significance testing, no scanned
documents, no answer writing yet. Cloud embedding models change quietly, so `--local` is the
version anyone can reproduce years from now.

One open question. On the contracts, reranking with the bare question loses 17 points at
k=10 while gaining 10 when every strategy gets the same amount of room. Those 300 questions
come from 20 templates, so the bare question cannot say which of 510 near-identical
contracts you mean. That fits the reranking finding above, but it is unconfirmed, so it is
not being reported as a result.

Things left out on purpose are in [`not-implemented.md`](not-implemented.md).

## Licence

Apache-2.0. Question sets CC BY 4.0.
