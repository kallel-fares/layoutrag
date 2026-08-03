# Measured costs

Every number here was measured on the machine below, not estimated. Where an estimate
appears it is labelled as one.

**Hardware:** Apple M1, 8 GB RAM, macOS. CPU only — no GPU, no cloud, no rented compute.

---

## Corpora

| | CUAD (contracts) | NIST (standards) |
|---|---|---|
| documents | 510 | 60 |
| pages | 9,348 | 4,060 |
| median pages/doc | 11 | 48 |
| characters | 26.3 M | 10.6 M |
| tokens (cl100k) | 6.6 M | 2.66 M |
| parse failures | 0 | 0 |
| scanned / no text layer | 0 | 0 |
| download | 106 MB | 95 MB |

Both are fully usable, which was not a given — the CUAD triage existed to find scanned
documents and found none.

---

## Parsing

Per page, measured over the full corpora.

| parser | s/page | full NIST corpus | notes |
|---|---|---|---|
| `pypdfium2` | 0.0024 | 22 s | text only, no structure |
| `pypdfium2-fontsize` | 0.0040 | 16 s | + heading detection from font metrics |
| `docling` | 1.03 | ~70 min | ML layout model |

`docling` is **258x** slower than reading font metrics.

**The OCR default costs 3.6x.** Out of the box `docling` downloads PaddleOCR weights and
runs character recognition over born-digital pages that already carry a text layer. Turning
it off took the average from 3.70 to 1.03 s/page, and the worst document in the sample from
34.3 to 1.1 s/page. It is one configuration flag.

---

## Embedding

`text-embedding-3-small` at $0.02 per million tokens.

| corpus | strategy | chunks | tokens | USD |
|---|---|---|---|---|
| cuad | fixed | 12,761 | 6,421,390 | $0.1284 |
| cuad | fixed-no-overlap | 11,264 | 5,637,435 | $0.1127 |
| cuad | contextual-heading | 15,013 | 5,631,187 | $0.1126 |
| cuad | parent-doc | 48,279 | 5,630,721 | $0.1126 |
| nist | fixed | 4,841 | 2,464,622 | $0.0493 |
| nist | fixed-no-overlap | 4,245 | 2,158,616 | $0.0432 |
| nist | contextual-heading | 4,784 | 2,260,568 | $0.0452 |
| nist | parent-doc | 21,169 | 2,151,059 | $0.0430 |
| | **total** | **112,356** | **32.4 M** | **$0.6471** |

**Projected $0.6471, actual $0.6589 — 1.8% over**, and the difference is earlier test calls.
The projection was produced with zero API calls: chunking is free and yields the exact token
counts embedding is billed for, so a full cost model is obtainable for nothing.

Wall clock, network-bound with 8 concurrent batches and parsing already cached: **1.7 min**
for NIST, **13.1 min** for CUAD.

`ada-002` was considered and rejected — $0.10/M against $0.02/M, and lower quality. It is a
previous-generation model that costs five times more.

---

## Local embedding, for comparison

`bge-small-en-v1.5`, no API, fully offline.

- one-time model load: 63 s
- throughput: **12 chunks/s** on CPU
- 112,356 chunks would take **~2.6 hours**

Free and reproducible, but slow enough that it is a fallback for clients who cannot send
documents to a third party, not a default.

---

## Storage

| | |
|---|---|
| parse + chunk cache | 832 MB |
| built indexes | 1.5 GB (777 MB is CUAD parent-doc alone) |
| corpora | 360 MB |
| `.venv` | 1.5 GB (torch, pulled in by docling) |
| docling models | 728 MB |

`parent-doc` produces 3.8x the chunks of `fixed` for the same text, which is where its index
size comes from.

---

## Total spend

**$0.66** for the entire study: two corpora, four chunking strategies each, all indexes
built and scored.

The spend guard's ceilings are $1 per run and $10 cumulative, both deliberately low against
a study that costs well under a dollar. A guard that never fires is decoration.

---

## Machine impact

Worth recording, because it was noticed in practice rather than predicted.

Parsing originally used `cpu_count - 1` = 7 worker processes. Each holds a document and its
parser state, and on 8 GB that pages heavily enough to make the whole machine feel slow.
Worker count is now bounded by memory (~1.5 GB headroom per worker) rather than by cores,
which gives 4 on this machine. `LAYOUTRAG_WORKERS` overrides it.

Sizing parallelism to core count is the wrong instinct on a memory-constrained machine.
