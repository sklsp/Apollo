# Architecture

Apollo is a local FastAPI application for working with documents and images on
your own machine. It has three independent parts: document storage with
retrieval-augmented chat (RAG), image generation through ComfyUI, and LoRA
training through the Ostris AI Toolkit. Each part works without the others;
what they share is configuration, path safety, job tracking, and the API layer
that serves a single-page frontend.

## Module map

| Module | Responsibility |
|--------|----------------|
| `app/main.py` | Application entry point: `create_app()` wires settings, CORS, the three API routers, static frontend serving, and startup checks into one FastAPI instance. |
| `app/api/routes.py` | Core endpoints: health probes (`/live`, `/ready`, `/health`), model listing, chat with session history, document upload/list/delete, RAG status and debug query, dashboard data, and the saved-prompt library. |
| `app/api/comfyui_routes.py` | Endpoints for ComfyUI generation, the workflow library, and background job tracking. |
| `app/api/lora_routes.py` | Endpoints for LoRA projects, datasets, captioning, training runs, and the trained-LoRA library. |
| `app/services/rag/service.py` | Orchestrates RAG: owns the persistent vector index under `DATA_DIR/rag` (a document is re-embedded only when its content hash changes), rebuilds the in-memory BM25 index at startup, and answers queries via vector or hybrid retrieval. |
| `app/services/rag/vector_store.py` | Persistent FAISS store for chunks with atomic temp-file saves and a self-describing `meta.json` that refuses to load an index built with a different embedding model. |
| `app/services/rag/embeddings.py` | Embedding client that prefers Ollama and falls back to local sentence-transformers when it is unreachable. |
| `app/services/rag/chunking.py` | Sentence-aware text chunker (default 800 characters, 150 overlap) used by both the app and the eval harness. |
| `app/services/rag/bm25.py` | In-memory Okapi BM25 keyword index (hand-rolled, no new dependency) that complements vectors with exact-term matching for hybrid retrieval. |
| `app/services/rag/fusion.py` | Reciprocal rank fusion of the vector and keyword rankings into one list; a pure function over opaque chunk ids so it needs no score calibration. |
| `app/core/jobs.py` | Stdlib-only background job store (threading plus uuid) shared by ComfyUI generation and LoRA training, deliberately avoiding Redis or Celery. |
| `app/services/comfyui_service.py` | ComfyUI orchestration: on-disk workflow library with per-workflow input mappings (no hardcoded node ids), input injection, and generation jobs. |
| `app/services/lora_training_service.py` | Treats the Ostris AI Toolkit strictly as an external process: writes a training config (verified against v0.12.26) and runs its `run.py`. |
| `app/services/dataset_validation.py` | Dataset quality control on bytes already on disk (no network): decodability, resolution sanity, exact and near-duplicate detection, and a per-image quality score. |
| `evals/runner.py` | Runs every question in the eval dataset through a built index or a live RAGService with an injected embedder, producing one scored result per question. |
