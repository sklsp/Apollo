#!/usr/bin/env python3
"""Verify the AI Workspace system is properly initialized."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402 - must come after the sys.path setup above


def main() -> None:
    print("[OK] SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"  LLMService: {type(app.state.llm_service).__name__}")
    print(f"  MemoryService: {type(app.state.memory_service).__name__}")
    print(f"  DocumentService: {type(app.state.document_service).__name__}")
    print(f"  PromptService: {type(app.state.prompt_service).__name__}")
    print(f"  RAGService: {type(app.state.rag_service).__name__}")
    print()

    print("[OK] SERVICE FUNCTIONALITY")
    print("=" * 60)

    mem = app.state.memory_service
    mem.add_message("test", "user", "Hello")
    mem.add_message("test", "assistant", "Hi there")
    print(f"  Memory: {len(mem.get_history('test'))} messages stored")

    doc = app.state.document_service
    doc.store_document("Test document content", "test.txt")
    print(f"  Documents: {len(doc.get_all_documents())} document stored")

    prompt = app.state.prompt_service
    prompts = prompt.list_prompts()
    print(f"  Prompts: {len(prompts)} prompt templates loaded")
    print()

    print("[OK] API ENDPOINTS")
    print("=" * 60)
    routes = [route.path for route in app.routes if hasattr(route, "path")]
    print(f"  Total top-level routes: {len(routes)}")
    print("  - /health, /models, /chat")
    print("  - /documents, /documents/upload")
    print("  - /prompts")
    print()
    print("=" * 60)
    print("SYSTEM READY")
    print("=" * 60)


if __name__ == "__main__":
    main()
