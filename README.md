# layoutrag

Compare chunking strategies for RAG over real PDFs — and see what each one actually retrieves.

Point it at a folder of PDFs. It indexes them under several chunking strategies at once,
then for any question shows you, side by side: the chunks each strategy returned, their
scores, and the document structure the parser found behind them.

Not a chatbot. A window into why retrieval works or doesn't.

```bash
layoutrag demo ./my-contracts
```

Runs on your machine. Your documents are never uploaded anywhere. Embeddings go to the
OpenAI API by default (your own key); `--local` runs fully offline with a local model.

## Why

Any RAG build forces a choice of chunker, chunk size, and whether to rerank. Those get
picked either way — the only question is whether the answer to *"why did you chunk it that
way?"* is a measured number or a shrug.

So the defaults here are measured. `results/` carries the numbers, run locally against a
public corpus of real commercial contracts, with the method written down.

## Status

Early. Phase 1 — see [the plan](#) for what's built and what isn't, and
[`not-implemented.md`](not-implemented.md) for what was deliberately cut and why.

## License

Apache-2.0.
