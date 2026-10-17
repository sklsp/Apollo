# RAG Retrieval Evaluation

Apollo ships a small, self-contained evaluation harness for its document-RAG
retrieval pipeline. It answers one question: **when the user asks something, do
the chunks that get retrieved actually come from the documents that contain the
answer?** The harness is deterministic, dependency-light (no LLM calls), and fast
enough to run locally on every retrieval-related change.

Everything lives in [`evals/`](../evals/) — data, code, and generated reports:

| Path | What it is |
|------|------------|
| `evals/corpus/*.md` | 12 ground-truth documents (local-AI topics: diffusion, LoRA, ComfyUI, vector search, ...) |
| `evals/questions.json` | 48 questions, each with its ground-truth source docs and key facts |
| `evals/dataset.py` | Loads + validates the corpus and question set (referential integrity) |
| `evals/index.py` | Builds an in-memory index using the **production** chunker and vector store |
| `evals/runner.py` | Runs retrieval over a bare index or through the real `RAGService` |
| `evals/metrics.py` | Document-level recall@k, MRR, hit-rate |
| `evals/report.py` | Deterministic JSON report writer |
| `evals/grid.py` | Chunking-parameter grid sweep (item 27) |
| `evals/cli.py` | Command-line entry point: `python -m evals.cli` |

## Metrics

The unit of scoring is the **document**, not the chunk. A question names one or
more ground-truth documents (`source_docs`); retrieval returns a ranked list of
chunks, and each chunk carries its document id. For a cutoff *k*:

- **recall@k** — fraction of the question's relevant documents present among the
  top-*k* chunks (each document counts once). Averaged over all questions.
- **MRR** — mean reciprocal rank: for each question, `1 / rank` of the first
  chunk from a relevant document; averaged over all questions.
- **hit_rate@k** — fraction of questions with at least one relevant-document
  chunk in the top *k*.

The cutoff applies to the raw ranked chunk list (repeated chunks occupy slots),
then documents are deduplicated for scoring — exactly how the pipeline consumes
results.

## Running it

```bash
# Index mode (default): bare in-memory index, production chunking defaults
python -m evals.cli

# Override chunking (index mode only) and cutoff
python -m evals.cli --chunk-size 600 --overlap 120 --top-k 5

# Service mode: full RAGService end to end (temp persist dir, never DATA_DIR)
python -m evals.cli --service

# Service mode with an explicit retrieval path (see "Hybrid modes" below)
python -m evals.cli --service --hybrid-mode vector

# Chunking grid sweep: one row per cell of the default grid
python -m evals.cli --grid
```

Reports are written to `evals/reports/rag_eval.json` by default; use `--out PATH`
to redirect. Generated reports are **not** committed — they are reproducible from
the corpus, questions, and settings recorded inside each report's `config` block.

### Embedding backend

The CLI uses the same `EmbeddingClient` as production: it probes Ollama for the
configured embedding model (`OLLAMA_EMBEDDING_MODEL`, default `nomic-embed-text`)
and falls back to sentence-transformers (`EMBEDDING_FALLBACK_MODEL`, default
`all-MiniLM-L6-v2`) when Ollama is unreachable. The report's `config` block
records which backend actually produced the vectors (`embedding_backend` and, on
fallback, the real model name), so a report always tells you what it was made with.

### Hybrid modes (service mode only)

`--hybrid-mode {auto,vector,keyword}` selects the retrieval path measured in
service mode; without the flag the service's `RAG_HYBRID` setting applies
(default `auto`). The chosen path is recorded as `retrieval_mode` in the report.

- **vector** — pure FAISS similarity search (`query_vector`).
- **keyword** — BM25 keyword index only (no embeddings at query time).
- **auto** — hybrid: both legs run and are fused with reciprocal-rank fusion
  (`query_hybrid`).

## Report format

Single-run reports (`rag_eval.json`) have `schema_version`, a `config` block, an
aggregate `summary`, and one entry per question (retrieved + relevant doc ids and
per-question metrics). Grid reports carry the same config plus one flat row per
cell. Output is deterministic — no timestamps, floats rounded to four decimals —
so two runs over identical inputs diff cleanly, which is what the CI regression
gate relies on.

## Baselines (2026-10-17)

Measured with the MiniLM fallback backend (Ollama was down at measurement time),
top_k=4, 48 questions / 12 docs.

Chunking grid (index mode):

| chunk/overlap | chunks | recall@4 | MRR    | hit_rate@4 |
|---------------|--------|----------|--------|------------|
| 400 / 75      | 128    | 0.9583   | 0.9028 | 0.9792     |
| 600 / 120     | 68     | 0.9583   | 0.9583 | 0.9792     |
| **800 / 150 (production)** | **46** | **0.9896** | **0.9757** | **1.0000** |
| 1200 / 200    | 27     | 0.9792   | 0.9444 | 1.0000     |

Retrieval modes (service mode, production chunking):

| mode     | recall@4 | MRR    | hit_rate@4 |
|----------|----------|--------|------------|
| vector   | 0.9896   | 0.9757 | 1.0000     |
| auto     | 0.9896   | 0.9514 | 1.0000     |
| keyword  | 0.9896   | 0.9635 | 1.0000     |

Reading: the production chunking (800/150) is the best cell of the grid, and all
three retrieval modes hit every question at k=4 on this corpus — the questions
are close paraphrases of document text, so vectors alone already rank correctly.
Hybrid fusion costs a little MRR here but never loses recall; it exists to help
where embeddings miss (exact identifiers, code-like tokens). Re-measure after any
change to chunking, embedding model, or retrieval logic and compare against these
numbers before merging.

## Extending the dataset

- **Add a document**: drop a `*.md` file into `evals/corpus/`. Its `doc_id` is
  the file stem; its title is the first `# ` heading (falling back to the stem).
- **Add questions**: append objects to `questions.json` with fields `id`,
  `question`, `source_docs` (list of corpus stems), and `key_facts`. Ids must be
  unique, every `source_docs` entry must name an existing document, and both
  lists must be non-empty — `load_dataset()` enforces all of this and fails the
  run loudly otherwise.

Keep questions answerable from the cited documents alone; `key_facts` are for
human review (they are not scored by retrieval metrics).

## Tests

The harness itself is covered by `tests/test_eval_*.py` (dataset validation,
metrics math, index building with a deterministic fake embedder, runner and CLI
wiring including grid and hybrid-mode routing) — all of it runs without Ollama or
a GPU.