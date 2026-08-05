# layoutrag

**A RAG pipeline for document collections. Advanced techniques and settings, measured on
your own files.**

Every RAG recommendation is published as a universal rule. Add a reranker. Use heading-aware
chunking. Use a layout-aware parser. Measured across two different document sets, each of
those rules reverses.

layoutrag runs the pipeline both ways on your own files and reports which configuration wins
for them.

---

## Overview

```bash
uv run layoutrag cost  ./my-docs                      # price indexing, zero API calls
uv run layoutrag query ./my-docs "what is the notice period?"
uv run layoutrag query ./my-docs "..." --local        # fully offline, no key
```

Runs on your machine. Your documents are never uploaded anywhere.

The pipeline covers parse, chunk, embed, index, search and rerank. Each stage ships several
implementations and a scoring harness, so you can measure a configuration on your corpus
before building on it.

---

## The problem this addresses

Published guidance treats pipeline choices as settled. Measured across two document sets,
the same choice helps one and harms the other.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results-dark.svg">
  <img alt="nDCG@10 by chunking strategy, with and without reranking, on technical standards and commercial contracts" src="docs/results-light.svg" width="100%">
</picture>

| Technique | Technical standards | Commercial contracts |
|---|---|---|
| Cross-encoder reranking, small chunks | +3 to +8 | **−11 to −12** |
| Cross-encoder reranking, best strategy | **−29.8** | **−15.1** |
| Heading-aware chunking | −2.0 | −4.9 |
| Contextual LLM enrichment | −6.1 | not measured |
| Layout-model parsing | −2.7 to +0.5 | not applicable |

Reranking is the clearest case. It gains up to 8 points on standards, loses 11 to 15 on
contracts, and removes 30 points from the best configuration on both.

Scores are nDCG@10, 0 to 1. Full data in [`results/`](results/).

A pipeline built on published defaults would be correct on one of these corpora and wrong on
the other, with no signal that anything was off. Retrieval degrades quietly.

---

## Findings

### Chunking

`parent-doc` embeds a small span for precision and returns the section around it. It won on
both document sets by a wide margin.

| Strategy | Technical standards | Commercial contracts |
|---|---|---|
| **parent-doc** | **0.742** | **0.647** |
| fixed | 0.509 | 0.461 |
| contextual-heading | 0.489 | 0.411 |
| contextual-llm | 0.448 | not measured |
| fixed-no-overlap | 0.447 | 0.433 |

Heading-aware chunking, widely recommended, placed third on standards and third on
contracts. On contracts it loses to plain fixed chunking by 5 points, because those PDFs are
HTML conversions carrying no heading typography: 0.0 to 0.5% of text objects are set above
body size. All 15,013 chunks are flagged as degraded before any score is computed, so the
pipeline reports that case rather than producing a quiet null.

`parent-doc` returns 20,479 median tokens against roughly 5,100 for the others. That cost is
real, and it is examined under Retrieval depth below.

### Reranking

A cross-encoder re-reads the top 50 results with the question in hand and reorders them. It
runs locally at no per-query cost.

| Strategy | Standards | Contracts |
|---|---|---|
| parent-doc | **−29.8** | **−15.1** |
| fixed | +3.0 | −11.9 |
| contextual-heading | +7.8 | −11.8 |
| contextual-llm | +3.6 | not measured |
| fixed-no-overlap | +5.2 | −10.6 |

Reranking helps small-chunk strategies on standards by 3 to 8 points, harms every strategy
on contracts by 11 to 15, and removes 30 points from the best configuration on both sets.

Contracts repeat party names on nearly every page, so a query naming the contract scores
similarly against every chunk in it and the clause-specific part is diluted. `parent-doc`
suffers because its results are already whole sections, leaving a cross-encoder little to
reorder and much to truncate.

**The query passed to the reranker decides its sign.** On standards, reranking the padded
search query scored 0.346 against 0.570 for the user's question. Search queries carry
document identifiers to help the vector stage. A cross-encoder compares the query directly
against each candidate, so that padding matches everything equally. The two queries are kept
separate, and the correct form is selected from the question set at runtime.

### Parsing

docling, an ML layout model, against heading detection from font metrics.

| Parser | contextual-heading | fixed | Headings found | Text recovered | Speed |
|---|---|---|---|---|---|
| Font metrics | 0.489 | **0.509** | 478 | 100% | **0.004 s/page** |
| docling | **0.494** | 0.482 | 6,579 | 79.2% | 1.03 s/page |

docling identifies 13.8x more headings and finishes level, within a point either way.

It also recovers 79.2% of the text plain extraction finds, median 86% per document and 17.4%
at worst, while reporting success on every document. Retrieval cannot return an answer that
was never indexed, so the parser now records a shortfall below 70% as a failure.

For born-digital PDFs the question is what 258x the runtime and 1 GB of additional
dependencies buy. Measured here, a tie. Its applicable case is scanned documents, which this
audit did not cover.

### Retrieval depth

`parent-doc` leads at k=10 while returning 3.5x the context. Holding every strategy to the
same token budget is what separates retrieving well from spending more.

| Standards, 4000-token budget | Score | Tokens returned |
|---|---|---|
| **parent-doc** | **0.600** | 2,048 |
| contextual-heading | 0.483 | 3,784 |
| fixed | 0.477 | 3,584 |

On standards it wins both ways, and leads on quality per token returned. On contracts its
recall falls from 0.772 to 0.381 once the budget is fixed, so the win there is bought with
context. Every configuration is scored both ways for this reason.

---

## Running an audit on your documents

```bash
uv run layoutrag cost ./my-docs
```

Reports the token count and price of indexing your corpus under each configuration. No API
calls are made, because chunking is free and produces the exact token counts embedding is
billed for.

```bash
uv run layoutrag query ./my-docs "a real question from your business"
```

Indexes under several chunking strategies simultaneously and returns what each retrieved,
with scores and the document structure behind them.

Parsing and chunking are cached by content hash, so comparing configurations does not
reprocess your documents.

Producing the scored tables above for your own corpus needs labelled questions, which
currently means a JSON file in the format under `data/` and a run of
`scripts/run_eval.py`. There is no command for it yet.

---

## Pipeline

```
PDF  ->  parse  ->  chunk  ->  embed  ->  index  ->  search  ->  rerank  ->  passages
```

| Stage | Default | Alternatives measured |
|---|---|---|
| Parse | pypdfium2 | plain text, font-size heading detection, docling |
| Chunk | fixed, 512 tokens, 12.5% overlap | no-overlap, heading-aware, parent-document, contextual-llm |
| Embed | text-embedding-3-small | local model, fully offline |
| Index | exact vector search | |
| Search | top 50 by similarity | |
| Rerank | ms-marco-MiniLM-L6-v2, local | disabled |

---

## Evaluation method

**Document sets.** [CUAD](https://www.atticusprojectai.org/cuad): 510 commercial contracts
with clauses annotated under lawyer supervision, CC BY 4.0. NIST Special Publications: 60
technical standards, public domain. Both retrieved by script and not committed here.

**Questions.** 300 derived from CUAD's lawyer annotations. 95 written by hand for NIST after
reading each source passage, deliberately worded away from the source. A model shown a
passage and asked to write a question reuses its vocabulary, after which retrieval succeeds
on word overlap and every configuration scores well for reasons unrelated to the pipeline.

**Relevance.** A result counts when it contains at least half the annotated answer.
Measuring the proportion of the chunk that is relevant would make large chunks structurally
unable to score, and the audit would be measuring chunk size while reporting strategy.
`tests/test_relevance.py` asserts size invariance.

**Search is exact**, so no configuration loses because an approximate index missed a
neighbour.

**No significance testing.** At n=300 the standard error on a paired difference is
approximately ±2.6 points. The 8 to 13 point differences reported here sit well clear of
that. Differences of 2 to 3 points are indicative.

---

## Cost

**$1.88** for everything measured here: two document sets, 11 indexes, 117,000 chunks,
39M tokens embedded, and one LLM enrichment pass over 4,606 chunks. The largest single run
was projected at $1.1073 and came in at $1.0939.

Spending controls are enforced in code:

- every run is priced before it starts, from exact token counts
- per-run and cumulative ceilings, persisted across processes
- a corpus re-embedded in a loop is stopped before it reaches an invoice
- reranking runs locally at no per-query cost

Measured breakdown in [`costs.md`](costs.md).

---

## Coverage

| Stage | Available | Roadmap |
|---|---|---|
| Ingestion | PDF, 3 parsers, 5 chunking strategies, cloud and local embedding | markdown, txt, docx; 3 further chunking strategies; incremental indexing |
| Retrieval | vector search, cross-encoder reranking | BM25 hybrid with RRF, metadata filters |
| Generation | | answer synthesis with citations |
| Interface | command line | web interface |
| Operations | spend ceilings, content-hash caching, resumable processing, progress reporting | |

Answer generation is not yet implemented. The pipeline currently retrieves and ranks
passages.

---

## Limitations

One embedding model, two document sets, no significance testing, no scanned documents, no
generation stage. Cloud embedding models change without notice, so `--local` is the
configuration reproducible over time.

Deliberate exclusions and their reasoning are in
[`not-implemented.md`](not-implemented.md).

---

## Licence

Apache-2.0. Question sets CC BY 4.0.
