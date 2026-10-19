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
