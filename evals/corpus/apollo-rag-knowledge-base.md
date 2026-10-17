# Apollo's Knowledge Base (RAG)

Apollo includes a local knowledge base so the chat assistant can answer questions about your own documents. The feature is built as a retrieval-augmented generation pipeline: uploaded files are chunked, embedded into vectors, and stored in an index that the chat service queries before composing an answer.

## Ingestion

When a document is added, its text is split into overlapping chunks of 800 characters with 150 characters of overlap between neighbors, splitting on sentence boundaries so no sentence is cut in half. Each chunk is embedded by the configured embedding model and appended to the index. Documents are tracked by content hash: re-uploading an unchanged file skips re-embedding entirely, while a changed file has its old chunks removed before the new ones are added, so stale text never lingers in results.

## The index

Vectors live in a FAISS flat inner-product index on disk under the application data directory. Embeddings are L2-normalized before storage, which makes the inner product equivalent to cosine similarity — higher score means more similar. The index file is written atomically (temp file plus rename) so a crash mid-save cannot corrupt the previous good state, and a small metadata file records the embedding model name and vector dimension; loading an index built with a different model is refused rather than silently mixing incompatible vectors.

## Retrieval at query time

When you ask a question in chat, the question is embedded with the same model and the top four most similar chunks are retrieved by default (configurable via `RAG_TOP_K`). Those chunks are formatted into a context block that is injected into the assistant's prompt, so the answer can cite your documents instead of relying on general knowledge. A developer view shows exactly which chunks were retrieved with their scores, which makes it easy to trace a bad answer back to weak retrieval or to a missing document.

## Limits

The knowledge base is local and single-user: there is no multi-tenant isolation because there is only one tenant. Very large corpora are supported but embedding time grows linearly with chunk count, and the flat index searches every vector on each query, which stays fast up to roughly a hundred thousand chunks before approximate methods become worthwhile.
