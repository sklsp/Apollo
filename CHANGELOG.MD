# Changelog

## v1.1.0
- Retrieval evaluation: 48 questions, hit@k and MRR, a hashing baseline and MiniLM (`python -m evals.cli`)
- Optional hybrid retrieval: BM25 plus reciprocal rank fusion behind `RAG_HYBRID`
- CI: ruff, mypy, coverage reporting (TOTAL 81%), retrieval regression gate, Docker build
- Dockerfile and docker-compose.yml (CPU-only image)
- Test suite runs in about 89 seconds instead of 323 (network probes fail fast in tests)
- docs/ARCHITECTURE.md, docs/DESIGN.md and docs/EVAL.md

## v1.0.0
- Initial release of AI Document Agent
- Added RAG pipeline with FAISS
- Added document upload system
- Added prompt library
- Added session memory
- Added Cloudflare tunnel deployment support
