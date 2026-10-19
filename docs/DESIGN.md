# Design decisions and trade-offs

Why Apollo is built the way it is, where each decision lives in the code, and what it costs.

## In one minute

Apollo is a local AI workspace: you upload documents and ask questions about them, generate images through
ComfyUI, and build and train LoRA models, all on your own machine with no cloud API. The part I care most about is
retrieval. It is measured with a small evaluation set instead of judged by feel, and changes such as hybrid search
are only switched on when the numbers support them. The rest is plain engineering: a FastAPI service layer, a
persistent FAISS index that survives restarts and corruption, background jobs that survive a restart, and tests
that run without Ollama or ComfyUI.

## How a question becomes an answer

1. The request arrives at `chat()` in `app/api/routes.py`, where FastAPI validates the body as a `ChatRequest` (session id, prompt, optional model and template id, and the use_documents flag).
2. If a template id was sent, `apply_template()` in `app/services/prompt_service.py` expands the stored template with the user's message; otherwise the raw message is used unchanged.
3. `format_history_for_prompt()` in `app/services/memory_service.py` renders the session's persisted messages as USER:/ASSISTANT: lines for prompt injection.
4. When use_documents is on, `query()` in `app/services/rag/service.py` retrieves chunks. In the default auto mode it embeds the question with `embed()` in `app/services/rag/embeddings.py`, searches the FAISS index through `search()` in `app/services/rag/vector_store.py`, scores the same chunks with BM25 via `search()` in `app/services/rag/bm25.py`, and merges both rankings with `reciprocal_rank_fuse()` (k=60) in `app/services/rag/fusion.py`.
5. `format_context()` in `app/services/rag/service.py` renders the winning chunks into a CONTEXT block, and `_build_system_prompt()` in `app/api/routes.py` assembles system instructions (the "default" template from `get_prompt_by_name()`, or a fallback string), context, history, and question.
6. The user's message is stored before the model call: `add_message()` in `app/services/memory_service.py` appends it to the session list and writes it through `save()` in `app/core/persistence.py` to data/sessions.json.
7. `LLMService.chat()` in `app/services/llm_service.py` sends the assembled prompt as a single user message through `chat_completion()` in `app/clients/ollama_client.py` to Ollama's OpenAI-compatible /v1/chat/completions endpoint and extracts choices[0].message.content from the reply.
8. The assistant's reply is stored with a second `add_message()` call (role "assistant"), which persists it the same way; the endpoint then returns a ChatResponse carrying the text and session id.

## Decisions

### 1. A flat FAISS index on disk, not a vector database server

**Decision:** Chunks live in a single `faiss.IndexFlatIP` over L2-normalized float32 vectors, persisted as three files under DATA_DIR/rag: index.faiss, chunks.json (one metadata record per row), and meta.json.
**Why:** A local single-user app does not need a server process to run alongside it, and with normalized vectors the inner product is cosine similarity, so the flat index answers exactly rather than approximately.
**Trade-off:** Search scans every chunk in memory each time, which is fine for thousands of chunks but not millions; there is no concurrent writing, replication, or query API beyond what this app needs.
**Where:** `app/services/rag/vector_store.py` (`VectorStore`, `_rebuild()`); normalization in `app/services/rag/embeddings.py` (`_normalize()`).

### 2. Ollama embeddings with a sentence-transformers fallback

**Decision:** `embed_batch()` tries Ollama's embed endpoint first (default model nomic-embed-text) and, when Ollama is unreachable, falls back to a local sentence-transformers model (default all-MiniLM-L6-v2), recording which backend served the batch.
**Why:** Everything stays on one machine, and retrieval keeps working with an installed local model instead of failing whenever Ollama is down or still starting up.
**Trade-off:** The two backends produce different vector spaces, so an index built under one cannot be searched under the other; the store refuses to mix them by quarantining the old index (decision 4).
**Where:** `app/services/rag/embeddings.py` (`EmbeddingClient.embed_batch()`, `_embed_with_ollama()`, `_embed_with_sentence_transformers()`); defaults in `app/core/config.py`.

### 3. Incremental indexing by content hash and model

**Decision:** Before embedding, `add_document()` compares the SHA-256 of the document text and the embedding model against a per-document version record; when both match it returns without touching the index, and changed documents have their stale chunks dropped first so two versions never coexist.
**Why:** Re-uploading or restarting should not re-embed an unchanged knowledge base, and hashing the text rather than the file time means only real edits trigger work.
**Trade-off:** A document is one unit: any edit re-embeds all of its chunks, and the version records are derived from chunk metadata at startup, so they cannot be inspected independently of the index.
**Where:** `app/services/rag/service.py` (`RAGService.add_document()`, `is_indexed_current()`, `_rebuild_version_map()`), hash in `content_hash()`.

### 4. Index metadata and quarantine instead of deletion

**Decision:** meta.json records the index version, embedding model, dimension, vector count, and update time; loading an index whose model or dimension no longer matches raises `IndexIncompatibleError`, and an unreadable FAISS file is treated as corruption. In both cases the three files are moved into the `quarantined` subdirectory with a timestamp appended to their names rather than deleted.
**Why:** Mixing vectors from different models would silently produce garbage answers, and keeping the old files means a bad upgrade can be inspected or restored instead of being thrown away.
**Trade-off:** After quarantine the index is empty, so retrieval returns nothing until documents are uploaded again; quarantined copies also accumulate in DATA_DIR/rag/quarantined until deleted by hand.
**Where:** `app/services/rag/vector_store.py` (`IndexMeta`, `VectorStore.__init__()`); `app/services/rag/service.py` (`_load_store()`, `_quarantine_files()`).

### 5. Sentence-boundary chunking with character overlap

**Decision:** `chunk_text()` splits text on sentence-ending punctuation, packs sentences into chunks of about 800 characters, and carries the trailing sentences that fit within a 150-character window over into the next chunk; text without sentence boundaries falls back to fixed windows split at spaces.
**Why:** Chunks that end mid-sentence hand the model broken context, and overlap keeps material straddling a boundary available in both neighbouring chunks.
**Trade-off:** The size target is approximate because one long sentence can push a chunk past it; the overlap also duplicates text, so the index holds more characters than the source documents contain.
**Where:** `app/services/rag/chunking.py` (`chunk_text()`, `_overlap_tail()`, `_chunk_by_characters()`); defaults in `app/core/config.py` (RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP).

### 6. Hybrid retrieval behind a flag, judged by the eval

**Decision:** `query()` follows RAG_HYBRID: in the default auto mode it fuses vector similarity with BM25 keyword ranking via reciprocal rank fusion (k=60), while vector and keyword modes use only that single signal; hybrid degrades to plain vector search when no keyword index is available.
**Why:** Embeddings generalize across paraphrase but miss distinctive terms like error codes, and rank fusion combines the two rankings without calibrating scores that live on incompatible scales.
**Trade-off:** Every auto query pays for an embedding call and a BM25 pass over all chunks, and the fused score is a rank statistic rather than a similarity, so it cannot be compared across queries or used as a threshold.
**Where:** `app/services/rag/service.py` (`RAGService.query()`, `query_hybrid()`); `app/services/rag/bm25.py` (`KeywordIndex`); `app/services/rag/fusion.py` (`reciprocal_rank_fuse()`); the eval harness in `evals/cli.py`.

### 7. Background jobs in a thread pool, persisted as JSON

**Decision:** `JobStore` runs work on a small ThreadPoolExecutor (two workers by default) and persists every job to a JSON file; on restart, any job still queued or running is marked failed with "Interrupted by application restart" rather than left spinning.
**Why:** A local single-user app does not need Redis or Celery, and two workers matches reality: ComfyUI and AI-Toolkit both want the whole GPU at once.
**Trade-off:** The file keeps only the most recent 200 jobs, and cancellation is cooperative, so a worker that never checks `is_cancelled()` runs to completion anyway.
**Where:** `app/core/jobs.py` (`JobStore`, `_load()`, `_run()`, `cancel()`).

### 8. ComfyUI workflows as API-format JSON plus a field map

**Decision:** Each workflow is two files on disk: `workflows/<id>.json`, the raw ComfyUI API-format export, and `<id>.map.json`, which names the node id and field each logical input (prompt, seed, steps, cfg, ...) writes to; Python never hardcodes a node id.
**Why:** A workflow can be swapped or extended by editing JSON without touching code, and the mapping keeps the UI controls in sync with whatever nodes the export contains.
**Trade-off:** The map must be written by hand for each new workflow, and only the logical inputs the service knows about are editable through it; the rest of the graph is sent to ComfyUI untouched.
**Where:** `app/services/comfyui_service.py` (workflow loading and input injection); `workflows/sdxl_txt2img.map.json`.

### 9. Argument-list subprocesses and one traversal guard

**Decision:** External processes are launched with fixed argument lists and no shell, and every path built from browser input goes through `safe_join` or `sanitize_filename`, which refuse absolute paths, drive letters, backslashes, and anything that resolves outside the data root.
**Why:** A command line assembled by string concatenation is a shell-injection hole, and one guard at the boundary means no other code in the app may concatenate user strings onto directories.
**Trade-off:** `safe_join` refuses backslash components on every platform, so Windows-style names cannot be used even where they would be harmless; uploads are limited to the extension sets the guards allow (images for generation, .safetensors for LoRA).
**Where:** `app/core/paths.py` (`safe_join`, `sanitize_filename`, `validate_image_upload`); `app/services/lora_training_service.py` (training subprocess launch).

### 10. An access token for the public tunnel

**Decision:** When APOLLO_ACCESS_TOKEN is set, every request needs that token in the X-Apollo-Token header or a cookie, except /live and /health which stay open for probes; opening the link with ?token=... once sets an HttpOnly cookie for 30 days. Unset (a plain local run) leaves Apollo fully open.
**Why:** launcher.py publishes Apollo through a public Cloudflare tunnel whose traffic reaches uvicorn from localhost, so "local" proves nothing and the token is the only gate between the dashboard and the internet.
**Trade-off:** The cookie lives for 30 days on any machine that opens the link, and there are no per-user accounts or revocation: losing the token means regenerating it in .env.
**Where:** `app/main.py` (`require_access_token`).

## Questions people ask

### Why FAISS and not Chroma, pgvector or Qdrant?

I chose a flat FAISS index persisted as three files under DATA_DIR/rag because this is a single-user local app and I do not want to run a database server alongside it. With L2-normalized vectors the inner product equals cosine similarity, so `faiss.IndexFlatIP` in `app/services/rag/vector_store.py` answers exactly rather than approximately. Chroma or Qdrant would add an embedded or remote engine with no benefit at this scale, and pgvector would mean a Postgres process for one user's documents. The cost is that every query scans all chunks, which I accept while the index stays in the thousands of chunks (see "How would this work with a million chunks?").

### How do you know the retrieval is any good?

I score it against a committed set of 48 questions over 12 documents, run by `python -m evals.cli` in `evals/cli.py`, with document-level recall@k, MRR and hit-rate from `evals/metrics.py`. With production chunking (800/150) and the MiniLM fallback backend, vector retrieval hits every question at k=4: hit_rate@4 = 1.0, MRR = 0.9757, recall@4 = 0.9896. `tests/test_eval_regression.py` re-runs that pipeline in CI and fails if a summary metric drops by more than 1e-3 or a question loses its hit against the committed baseline in `evals/baselines/vector_service.json`. So "good" is a number I can diff, not a feeling.

### What happens when the embedding model changes?

meta.json records which model and dimension built the index, so `VectorStore.__init__()` in `app/services/rag/vector_store.py` raises `IndexIncompatibleError` on a mismatch instead of scoring vectors from different spaces against each other. The three index files are then moved into DATA_DIR/rag/quarantined with timestamps rather than deleted (decision 4), and the version map in `app/services/rag/service.py` keys documents by content hash plus model, so nothing is re-embedded until it changes again. In practice: switch models, upload your documents once, done; until then retrieval returns an empty index instead of wrong answers.

### How do you avoid embedding the same document twice?

`add_document()` in `app/services/rag/service.py` hashes the document text with SHA-256 and compares that hash, plus the embedding model name, against a per-document version record before it embeds anything. When both match it returns without touching the index, so re-uploading an unchanged file is free; when they differ, the stale chunks are dropped first so two versions of one document never coexist in the ranking. The records are rebuilt from chunk metadata at startup (`_rebuild_version_map()`), which means the check survives a restart with no extra state to corrupt.

### Why is hybrid search not the default, or why is it?

It is the default here: RAG_HYBRID in `app/core/config.py` defaults to auto, and `query()` in `app/services/rag/service.py` fuses vector and BM25 rankings with reciprocal rank fusion. The eval shows that on this corpus all three retrieval modes hit every question at k=4 (hit_rate@4 = 1.0), so fusion buys nothing here: plain vector scores MRR 0.9757 while auto fusion scores 0.9514, a small cost I accept because BM25 covers exact identifiers and code-like tokens where embeddings miss. The flag exists precisely because the numbers are this close; set RAG_HYBRID to vector or keyword if you want the other ranking back.

### How would this work with a million chunks?

What breaks first is the flat index: `faiss.IndexFlatIP` in `app/services/rag/vector_store.py` is exact but linear, so every query scans all one million vectors and latency grows in proportion to the corpus. I would move to an approximate index such as IVF or HNSW that trades a little recall for much shorter search time, batch the embedding calls harder (they already go through `embed_batch()`), and stop loading chunks.json whole into memory at startup. None of that was built: the indexed corpus here is 46 chunks, and I would not add approximate-search machinery until the numbers say I need it.
