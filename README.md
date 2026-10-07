# Apollo

**A unified local AI workspace, documents, image generation, datasets, LoRA training, and ComfyUI. No cloud required.**

Apollo connects the whole local AI workflow in one application: upload PDF, DOCX, or TXT files and ask questions with RAG context backed by a persistent vector index. Generate images through a local ComfyUI instance. Build training datasets, caption them with a local vision model, validate them for duplicates and quality issues, then train LoRA models with the Ostris AI Toolkit, with hardware-aware preflight checks against your actual GPU. Every generated asset keeps its provenance. Built with FastAPI, Ollama, FAISS, ComfyUI, and AI Toolkit.

<p align="center">
  <a href="https://github.com/sklsp/Apollo/actions/workflows/ci.yml"><img src="https://github.com/sklsp/Apollo/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat" alt="Ollama" />
  <img src="https://img.shields.io/badge/FAISS-0466C8?style=flat" alt="FAISS" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/RAG-Enabled-7C3AED?style=flat" alt="RAG" />
  <img src="https://img.shields.io/badge/ComfyUI-Image%20Gen-CC4A31?style=flat" alt="ComfyUI" />
  <img src="https://img.shields.io/badge/LoRA-Training-F5A623?style=flat" alt="LoRA" />
</p>

---

## Live Demo Overview

Apollo is a full AI workspace that runs entirely on your machine:

| Capability | Description |
|------------|-------------|
| **Chat** | Conversational AI powered by local Ollama models |
| **Document RAG** | Upload files, embed chunks, retrieve relevant context at query time, the FAISS index persists across restarts |
| **Prompt library** | Save, edit, and reuse templates with `{input}` variables |
| **Session memory** | Multiple chats with per-session history and settings, persisted across restarts |
| **ComfyUI generation** | Run local ComfyUI workflows from the dashboard, with importable workflow JSON and pre-generation validation |
| **LoRA Studio** | Build datasets, caption them with a local vision model, train LoRAs via Ostris AI Toolkit |
| **Dataset quality control** | One-click validation: exact + near duplicates (multi-signal), broken images, caption coverage, 0-100 quality score |
| **Training presets** | Character / Style / Product / Concept starting points, plus hardware-aware conservative/balanced/quality variants |
| **Hardware preflight** | Detects your GPU/VRAM/RAM/disk and classifies a training config as safe / heavy / risky before you start |
| **One-click LoRA testing** | After training, jump straight into a prefilled ComfyUI generation with the LoRA loaded |
| **Provenance** | Generated images keep workflow/prompt/seed/LoRA sidecars; every training run records its dataset, config and hardware |
| **Unified dashboard** | Home view with document/dataset/job counts, system health, and recent generations at a glance |

Each module is independent: **chat and RAG keep working when ComfyUI, Ollama, or AI Toolkit are unavailable.**

**Quick start:**

```bash
python launcher.py
```

Open **http://localhost:8000**, or use the auto-generated Cloudflare tunnel URL for a public demo.

---

## Feature Demo

> GIFs live in `images/`. 

### Basic Chat

Simple conversation with a local LLM, fast, private, and fully offline-capable.

![Basic chat demo](images/chat.gif)

---

### Session Memory

Create new chats and switch between sessions. Conversation history persists per session during runtime.

![Session memory demo](images/session.gif)

---

### Document Q&A (RAG)

Upload PDF, DOCX, or TXT files and ask questions about their content. Documents are chunked, embedded, and indexed in FAISS.

![Document RAG demo](images/rag.gif)

---

### Toggle Document Usage

Enable or disable document context per session. Compare pure LLM answers vs RAG-enhanced responses with one click.

![Document toggle demo](images/toggle.gif)

---

### Prompt Library

Create reusable prompt templates and apply them from the chat header. Use `{input}` as a placeholder for the user's message.

![Prompt library demo](images/prompts.gif)

---

### Prompt Management

Edit and delete saved prompts from the library panel. Templates sync via the REST API (`GET/POST/PUT/DELETE /prompts`).

![Prompt edit demo](images/prompt-edit.gif)

---

## Why this project

This project demonstrates a full local AI system combining:

- **Retrieval-Augmented Generation (RAG)**, semantic search over uploaded documents
- **Local LLM inference**, no cloud API dependency for chat or embeddings
- **Document understanding pipeline**, PDF, DOCX, and TXT extraction with chunking and indexing
- **Real-time chat UI**, sessions, prompt library, and optional document mode
- **Production-style API architecture**, modular services, dependency injection, OpenAPI docs

Built as a portfolio-ready example of how to ship a private, self-hosted AI knowledge assistant.

---

## Architecture

```
Frontend (Vanilla JS Dashboard, Home · Chat · Docs · ComfyUI · LoRA Studio)
        │
        ▼
FastAPI Backend
        │
        ├── Document / RAG stack ────────────────────┐
        │     /chat · /documents · /prompts · /health │
        │     RAG Service (chunk → embed → retrieve)  │
        │     FAISS vector store (persistent, top-k)  ▼
        │                                          Ollama
        │                                  (chat + embeddings + vision)
        │
        ├── ComfyUIService ──► ComfyUIClient ──► ComfyUI ──► generated images
        │     /comfyui/*                                     + provenance sidecars
        │
        └── LoRA modules
              /loras/*
              LoRAProjectService    ──► dataset on disk (images + .txt captions)
              LoRATrainingService   ──► AIToolkitProcess ──► Ostris AI Toolkit
              RunHistory            ──► per-project run records (runs.json)
                                                                  │
                                                                  ▼
                                                        LoRA .safetensors
                                                                  │
                                                                  ▼
                                                    ComfyUI inference with LoRA
```

The three modules are decoupled, routes call services, services call a client
or process adapter. Nothing in the document pipeline imports ComfyUI code.

**Request flow (chat with documents):**

1. User sends a message from the dashboard
2. FastAPI optionally queries FAISS for top-k relevant chunks
3. Context + history + prompt template are assembled
4. Ollama generates a response
5. Reply is stored in session memory and returned to the UI

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| LLM | Ollama (`llama3.2`, etc.) |
| Embeddings | `nomic-embed-text` + `sentence-transformers` fallback |
| Vector DB | FAISS (persistent on disk, atomic saves, corruption quarantine) |
| Document parsing | pypdf, python-docx |
| Frontend | Vanilla JS (chat UI dashboard) |
| Image generation | ComfyUI (external local instance, HTTP API) |
| LoRA training | Ostris AI Toolkit (external local process) |
| Captioning | Ollama vision model (e.g. `llava`) |
| Job system | stdlib `ThreadPoolExecutor`, JSON-persisted |
| Deployment | Cloudflare Tunnel + LAN support (`0.0.0.0` binding) |

---

## Setup

### Prerequisites

- Python 3.11+ (3.14 supported)
- [Ollama](https://ollama.com/) running locally (default `http://localhost:11434`)
- A chat model, e.g. `llama3.2`
- An embedding model for RAG, e.g. `nomic-embed-text`
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) (optional, for `launcher.py` public tunnel)

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Optional, only needed for the new AI generation features:**

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation
- [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) for LoRA training
- A vision model for AI captioning: `ollama pull llava`

Everything else keeps working if these are absent, the dashboard shows a clear
"unavailable" card instead of failing.

### Install

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. (Optional) Configure environment variables or a `.env` file in the project root.

### Run

**Full stack (recommended):**

```bash
python launcher.py
```

Starts the API, waits for `/health`, launches a Cloudflare quick tunnel, copies the public URL to clipboard, and opens the dashboard.

**Local API only:**

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Interactive CLI (legacy agent):**

```bash
python main.py
```

**API mode via root entrypoint:**

```bash
python main.py --api
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_DEFAULT_MODEL` | `llama3.2` | Default chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `EMBEDDING_FALLBACK_MODEL` | `all-MiniLM-L6-v2` | Fallback if Ollama embeddings fail |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RAG_TOP_K` | `4` | Retrieved chunks per query |
| `OLLAMA_VISION_MODEL` | `llava` | Vision model for AI dataset captions |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | Local ComfyUI instance |
| `COMFYUI_WORKFLOW_DIR` | `./workflows` | Where workflow JSON + mappings are stored |
| `COMFYUI_LORA_DIR` | *(empty)* | ComfyUI's `models/loras`, so manually added LoRAs appear in the library |
| `COMFYUI_GENERATION_TIMEOUT` | `600` | Seconds before a generation job gives up |
| `AI_TOOLKIT_PATH` | *(empty)* | Ostris AI Toolkit checkout, e.g. `C:/ai-toolkit` |
| `AI_TOOLKIT_PYTHON` | *(empty)* | That toolkit's interpreter, e.g. `C:/ai-toolkit/venv/Scripts/python.exe` |
| `LORA_DATA_DIR` | `./data/loras` | LoRA projects, datasets, configs, outputs |
| `GENERATED_DIR` | `./data/generated` | Images pulled back from ComfyUI |
| `DATA_DIR` | `./data` | Where `documents.json` / `sessions.json` persistence lives |
| `MAX_IMAGE_UPLOAD_MB` | `25` | Per-image cap for dataset uploads |
| `PORT` | `8000` | Server port (`launcher.py`) |
| `CLOUDFLARED_PATH` | auto-detect | Path to `cloudflared` executable |

A `.env` file in the project root is loaded automatically (real environment
variables take precedence). Copy `.env.example` to `.env` to start.

**Training is disabled unless both `AI_TOOLKIT_PATH` and `AI_TOOLKIT_PYTHON` are
set and the files actually exist.** The rest of the app is unaffected.

**Remote Ollama (LAN)**, Windows PowerShell:

```powershell
$env:OLLAMA_BASE_URL = "http://192.168.1.50:11434"
python launcher.py
```

Or in `.env`:

```text
OLLAMA_BASE_URL=http://192.168.1.50:11434
```

---

## API Reference

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/health` | Health check + Ollama status |
| `GET` | `/models` | List available Ollama models |
| `POST` | `/chat` | Chat with optional RAG and prompt template |
| `POST` | `/documents/upload` | Upload and index `.txt`, `.pdf`, `.docx` |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `DELETE` | `/documents` | Clear all documents |
| `GET` | `/prompts` | List prompt templates |
| `POST` | `/prompts` | Create a prompt |
| `GET` | `/prompts/{prompt_id}` | Get one prompt |
| `PUT` | `/prompts/{prompt_id}` | Update a prompt |
| `DELETE` | `/prompts/{prompt_id}` | Delete a prompt |
| `GET` | `/sessions/{session_id}/history` | Session chat history |
| `DELETE` | `/sessions/{session_id}/history` | Clear session history |
| `GET` | `/rag/status` | Vector index state: what is indexed, with which model |
| `GET` | `/rag/debug-query?q=...` | Developer view of one retrieval: chunks, scores, context |
| `GET` | `/dashboard` | Unified workspace overview: counts, activity, system health |

### ComfyUI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/comfyui/status` | Connection state, version, VRAM, installed checkpoints/LoRAs |
| `POST` | `/comfyui/test` | Explicit connection test |
| `GET` | `/comfyui/workflows` | List saved workflows and their mapped inputs |
| `POST` | `/comfyui/workflows` | Import an API-format workflow (inputs auto-mapped) |
| `GET` | `/comfyui/workflows/{id}` | Full graph + node list + input mapping |
| `DELETE` | `/comfyui/workflows/{id}` | Delete a workflow |
| `POST` | `/comfyui/generate` | Queue a generation, returns a job immediately |
| `POST` | `/comfyui/validate-generation` | Pre-flight a request without queueing (missing models/LoRAs caught early) |
| `GET` | `/comfyui/generated` | Recent generated images with provenance records |
| `GET` | `/comfyui/jobs/{job_id}` | Generation job status |
| `GET` | `/comfyui/images/{filename}` | Serve a generated image |
| `GET` | `/comfyui/test-lora` | One-click LoRA test setup: picks a workflow, prefills inputs |

### LoRA

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/loras` | LoRA library (project outputs + ComfyUI's folder) |
| `GET` | `/loras/toolkit/status` | Whether AI Toolkit is configured |
| `GET` | `/loras/training/presets` | Training presets with plain-language explanations |
| `GET` | `/loras/training/hardware-presets` | Conservative/balanced/quality variants for the detected GPU |
| `GET` | `/loras/hardware` | Detected hardware: GPU, VRAM, RAM, disk |
| `POST` | `/loras/projects` | Create a LoRA project |
| `GET` | `/loras/projects` | List projects |
| `GET` | `/loras/projects/{id}` | Project detail |
| `PUT` | `/loras/projects/{id}` | Update name / trigger word / base model |
| `DELETE` | `/loras/projects/{id}` | Delete project and its dataset |
| `POST` | `/loras/projects/{id}/images` | Upload training images (PNG/JPG/JPEG) |
| `GET` | `/loras/projects/{id}/images` | List dataset images + captions |
| `GET` | `/loras/projects/{id}/images/{image_id}/file` | Serve a dataset image |
| `DELETE` | `/loras/projects/{id}/images/{image_id}` | Delete an image and its caption |
| `PUT` | `/loras/projects/{id}/captions/{image_id}` | Write a caption (marks it hand-edited) |
| `POST` | `/loras/projects/{id}/generate-captions` | Caption the dataset with a local vision model |
| `POST` | `/loras/projects/{id}/config` | Generate `training.yml` without starting a run |
| `POST` | `/loras/projects/{id}/preflight` | Validate a training config against the detected hardware |
| `POST` | `/loras/projects/{id}/train` | Start training |
| `POST` | `/loras/projects/{id}/stop` | Stop this project's training process |
| `GET` | `/loras/projects/{id}/training` | Training status + progress |
| `GET` | `/loras/projects/{id}/training/log` | Tail of the training log |
| `GET` | `/loras/projects/{id}/runs` | Training run history: dataset, config, hardware, outcome per run |
| `GET` | `/loras/projects/{id}/validate` | Dataset quality report: duplicates, broken images, score |

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/jobs` | List jobs (`?job_type=comfyui_generation`) |
| `GET` | `/jobs/{job_id}` | Job detail |
| `DELETE` | `/jobs/{job_id}` | Request cancellation |

### Chat request example

```json
{
  "prompt": "What is the main topic of my upload?",
  "session_id": "default",
  "model": "llama3.2",
  "use_documents": true,
  "prompt_id": "prompt_3"
}
```

- `use_documents: true`, retrieve relevant chunks from the vector store (default)
- `use_documents: false`, pure LLM chat, no document context
- `prompt_id`, optional; merges a saved template (`{input}` replaced with `prompt`)

---

## How it works

### Document upload flow

1. File is saved temporarily on disk
2. Text is extracted via unified parser (`extract_text`), PDF (pypdf), DOCX (python-docx), TXT
3. Full text is stored in `DocumentService`
4. Text is chunked, embedded, and indexed in FAISS via `RAGService`
5. On chat (when `use_documents` is true), top-k similar chunks are injected into the LLM prompt

### Dashboard

The frontend at `/` supports:

- Multi-session chat
- **Use Documents** toggle (per session)
- Prompt template dropdown
- Prompt library (create, edit, delete)
- Drag-and-drop upload for PDF, DOCX, TXT

The API base URL is resolved automatically from `window.location.origin` (works on localhost, LAN, and Cloudflare tunnel).

### Embedding fallback

RAG prefers Ollama embeddings (`nomic-embed-text`). If Ollama embeddings are unavailable, the system falls back to `sentence-transformers` (`all-MiniLM-L6-v2`):

```bash
ollama pull nomic-embed-text
```

---

## Project structure

```
Apollo/
├── app/                          # FastAPI application (main product)
│   ├── main.py                   # App factory + route registration
│   ├── api/
│   │   ├── routes.py             # Chat / documents / prompts / dashboard endpoints
│   │   ├── comfyui_routes.py     # ComfyUI + job endpoints
│   │   └── lora_routes.py        # LoRA project / dataset / training endpoints
│   ├── models/
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── comfyui_schemas.py    # Generation + workflow + job schemas
│   │   └── lora_schemas.py       # Project / dataset / training schemas
│   ├── core/
│   │   ├── config.py             # Settings (Ollama, RAG, ComfyUI, AI Toolkit)
│   │   ├── exceptions.py         # API error types
│   │   ├── paths.py              # Path traversal guards, upload validation
│   │   ├── persistence.py        # Atomic JSON persistence helper
│   │   └── jobs.py               # Background job store (threadpool + JSON)
│   ├── clients/
│   │   ├── ollama_client.py      # Ollama HTTP client (chat, embed, vision)
│   │   └── comfyui_client.py     # ComfyUI HTTP client
│   ├── services/
│   │   ├── document_service.py   # Document storage (persisted)
│   │   ├── prompt_service.py     # Prompt library CRUD
│   │   ├── memory_service.py     # Session chat history (persisted)
│   │   ├── llm_service.py        # Chat + model listing
│   │   ├── comfyui_service.py    # Workflow library, injection, generation + provenance
│   │   ├── dataset_validation.py # Multi-signal duplicate detection + quality score
│   │   ├── hardware.py           # GPU/VRAM/RAM/disk detection (NVIDIA/AMD/CPU)
│   │   ├── training_preflight.py # Hardware-aware config advisor + failure diagnostics
│   │   ├── run_history.py        # Per-project training run records
│   │   ├── dashboard.py          # Unified workspace overview aggregation
│   │   ├── lora_dataset_service.py   # Projects, images, captions, LoRA library
│   │   ├── lora_training_service.py  # AI Toolkit config gen + subprocess
│   │   └── rag/                  # RAG pipeline
│   │       ├── ingestion.py      # PDF / DOCX / TXT extraction
│   │       ├── chunking.py       # Text chunking
│   │       ├── embeddings.py     # Ollama + fallback embeddings
│   │       ├── vector_store.py   # Persistent FAISS index (+ quarantine recovery)
│   │       └── service.py        # RAGService orchestration (incremental indexing)
│   └── frontend/
│       └── index.html            # Dashboard UI (Home · Chat · Docs · ComfyUI · LoRA)
├── ai_agent/                     # Interactive CLI agent (python main.py)
├── scripts/                      # Dev & maintenance utilities
│   ├── verify_system.py          # Smoke-test services + routes
│   ├── diagnostics_ollama.py     # Ollama connectivity probe
│   ├── smoke_preflight.py        # Live training-preflight demo
│   └── smoke_rag_persistence.py  # Restart-survival verification for the index
├── workflows/                    # ComfyUI workflows + node-input mappings
├── tests/                        # Test suite (250 tests) + legacy debug scripts
├── data/                         # Documents, sessions, jobs, LoRA projects, generated images, rag index
├── images/                       # Feature demo GIFs (README showcase)
├── launcher.py                   # One-command startup (API + tunnel)
├── main.py                       # CLI / API entrypoint
├── .env.example                  # Example environment variables
├── pytest.ini                    # Test configuration
└── requirements.txt
```

---

## AI Generation & LoRA Studio

Two optional modules that turn the document workspace into a local AI creation
platform. Both are independent, if either is missing, chat and RAG are unaffected.

### 1. ComfyUI setup

Install and start [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

```bash
python main.py --listen 127.0.0.1 --port 8188
```

Point the app at it in `.env`:

```text
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_LORA_DIR=C:/path/to/ComfyUI/models/loras
```

Open the dashboard, click **ComfyUI** in the left rail. The header shows the
connection state, version and free VRAM. Checkpoint and LoRA dropdowns are
populated from the running instance, not hardcoded.

### 2. Workflows

Workflows are stored as two files so they can be swapped without touching Python:

| File | Purpose |
|------|---------|
| `workflows/<id>.json` | The graph, in ComfyUI **API format** |
| `workflows/<id>.map.json` | Which node + field each logical input writes to |

The mapping is what keeps node IDs out of the code:

```json
{
  "name": "SDXL Text to Image",
  "arch": "sdxl",
  "inputs": {
    "prompt":    { "node": "6", "field": "text" },
    "seed":      { "node": "3", "field": "seed" },
    "lora_name": { "node": "10", "field": "lora_name" }
  }
}
```

**Importing your own:** in ComfyUI use *Workflow → Export (API)*, not the plain
save, which produces a UI-format file the importer will reject with an explanatory
message. Then **Import workflow JSON** in the dashboard. Inputs are auto-detected:
prompts are found by following the sampler's own `positive`/`negative` links, so
it is a lookup rather than a guess. Edit the `.map.json` afterwards to adjust.

The UI only enables controls a workflow actually maps, pick a workflow without a
LoRA node and the LoRA selector greys out.

### 3. Ostris AI Toolkit setup

Install [AI Toolkit](https://github.com/ostris/ai-toolkit) **separately**, it is
never vendored into this repo:

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then in `.env` (forward slashes are fine on Windows):

```text
AI_TOOLKIT_PATH=C:/ai-toolkit
AI_TOOLKIT_PYTHON=C:/ai-toolkit/venv/Scripts/python.exe
```

Both must exist or training stays disabled with a clear message. Verify with:

```bash
curl http://localhost:8000/loras/toolkit/status
```

Config generation targets **AI Toolkit v0.12.26**. If you upgrade and the schema
moves, `app/services/lora_training_service.py` is the only file to update.

### 4. End-to-end: train a LoRA and use it

1. **LoRA Studio → Projects**, create a project, choose an architecture
   (`sdxl`, `flux`, `qwen_image`, `krea2`) and a trigger word.
2. **Dataset**, drag in PNG/JPG images. Each gets a matching `.txt` caption file,
   which is the pairing AI Toolkit expects:
   ```
   data/loras/<project>/dataset/image001.png
   data/loras/<project>/dataset/image001.txt
   ```
3. **Captions**, write them by hand, or click **Generate captions with AI** to
   run a local Ollama vision model over the dataset. Generated captions are saved
   for review; **captions you edited by hand are never overwritten** unless you
   tick *Overwrite my edits*.
4. **Training**, set steps, learning rate, rank/alpha, resolution. **Preview
   config** writes `config/training.yml` so you can inspect it first. **Start
   training** launches AI Toolkit as a subprocess.
5. **Progress**, step count, loss and the live log are parsed from AI Toolkit's
   output. When step info is not parseable the UI says *"Training in progress"*
   rather than inventing a percentage. **Stop** terminates only that run's process
   tree.
6. **Library**, the finished `.safetensors` is detected automatically. Files are
   validated by reading the safetensors header, so a renamed file is not listed.
7. **Use in ComfyUI**, pick a LoRA in the library and press *Use in ComfyUI*. It
   selects a LoRA-capable workflow, sets the LoRA, and prefills the trigger word.

> ComfyUI can only load LoRAs that live in its own `models/loras` folder. Copy the
> trained file there (or set AI Toolkit's output path to it) and restart ComfyUI.
> The app warns you when the selected LoRA is not yet visible to ComfyUI.

### Jobs

Generation and training are long-running, so neither blocks a request. Both use
one job abstraction, `POST` returns a `job_id` immediately, then poll:

```bash
curl http://localhost:8000/jobs?job_type=comfyui_generation
curl http://localhost:8000/jobs/<job_id>
```

Statuses: `queued · running · completed · failed · cancelled`. Jobs persist to
`data/jobs.json`; anything caught mid-flight by a restart is marked failed with
*"Interrupted by application restart"* rather than left spinning.

### When services are unavailable

| Situation | Behaviour |
|-----------|-----------|
| ComfyUI offline | ComfyUI view shows a disconnected card with the configured URL. Chat/RAG unaffected |
| Ollama offline | Chat and AI captions report it clearly; manual captioning still works |
| AI Toolkit unconfigured | Dataset building works fully; training returns a 503 explaining which variables to set |
| Invalid workflow JSON | Rejected at import with the reason (UI-format exports get a specific hint) |
| Training crashes | Exit code and last log line are surfaced; project marked `failed` |

### Security

Local-first does not mean unguarded:

- Upload allowlist (PNG/JPG/JPEG only) plus a size cap
- Filenames sanitised; directory components stripped
- Every user-supplied path goes through a traversal guard that rejects `..`,
  absolute paths, drive letters and UNC prefixes
- Subprocesses are launched with argument arrays, never `shell=True`, and no
  shell interpolation of user input
- The browser is served images by name from managed folders; raw filesystem paths
  are never exposed
- Uploaded files are never executed

---

## Development notes

- FastAPI binds to `0.0.0.0` for LAN access
- The RAG vector index persists under `data/rag/` and survives restarts; a corrupted index is quarantined, not deleted
- Documents, sessions, jobs, LoRA projects, workflows and generated images all persist on disk
- Upload debug logs appear in the server console: `[UPLOAD] filename`, `extracted chars`, `rag chunks indexed`
- Utility scripts: `python scripts/verify_system.py`, `python scripts/diagnostics_ollama.py`
- Run the tests with `python -m pytest` (250 tests); stress tests: `pytest tests/test_stress.py -m slow`
- Log prefixes: `[COMFYUI]`, `[COMFYUI JOB]`, `[LORA]`, `[LORA TRAINING]`, `[DATASET]`, `[RAG]`, `[RUN HISTORY]`
- `tests/` also contains standalone diagnostic scripts (`test_endpoints.py`,
  `test_ollama_detailed.py`, …) that predate the suite. They print at import time
  and some need a live Ollama, so `tests/conftest.py` excludes them from
  collection. Run them by hand: `python tests/test_endpoints.py`

### Future improvements

- Optional ChromaDB backend as an alternative vector store

---

## License

See repository license file.
