# Not implemented — and why

Being able to say why something was cut is part of the point. Nothing here is an oversight.

## Cut from the study

| Cut | Why |
|---|---|
| **`late` chunking** | Needs token-level embeddings over a long context. The OpenAI embeddings API returns pooled vectors only, so it cannot be implemented against the primary embedder. Running it on a different model would confound the arm, which is worse than omitting it. |
| **Proposition chunking** | Most expensive arm to build and run; the published evidence for it is the weakest of any strategy considered. |
| **Query transforms (HyDE, multi-query)** | Published benchmarks report limited benefit, and HyDE degrades on domains the model hasn't seen — which describes every client corpus. Adds query-time latency for little gain. |
| **Generation and LLM-judge evaluation** | Model-dependent and noisy. Cutting it makes this a clean retrieval study. |
| **Vector store comparison** | They compute the same cosine similarity. |
| **Embedding dimension sweeps** | A knob, not an idea. |
| **Embedding model rankings** | MTEB covers this with far more compute. Stated as methodology, not omission. |
| **k / rerank-depth / alpha sweeps** | Production tuning, not a finding. |
| **Significance testing (bootstrap CIs, Holm-Bonferroni)** | That apparatus exists to survive peer review. This study exists to pick defensible defaults. Where a difference is inside noise, it is reported as such rather than ranked. |
| **Multi-corpus generalisation claims** | One corpus is studied. "It holds across document types" is not claimed, because it is not tested. |

## Cut from the tool

| Cut | Why |
|---|---|
| **OCR / scanned documents** | Out of scope. Scanned PDFs in the corpus are excluded and counted, not silently dropped. |
| **HTML and non-PDF sources** | The whole point is PDF layout and structure. |
| **PyPI packaging** | Nobody needs to `pip install` this. It is cloned and run. |
| **Hosted / uploader demo** | Asking a prospect to upload contracts to a third party at proposal stage is the hardest possible ask. Running locally sidesteps it entirely. |
| **Plugin registry / entry points** | Nobody is registering third-party strategies. |
