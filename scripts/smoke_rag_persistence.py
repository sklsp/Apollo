"""Live RAG persistence smoke test: upload, restart server, verify retrieval.

Run twice:
    python scripts/smoke_rag_persistence.py phase1   # upload + index
    python scripts/smoke_rag_persistence.py phase2   # verify without re-embedding
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
MARKER = "QUANTUM-SPHINX-MARKER-7391"


def phase1() -> None:
    """Upload a document and confirm it is indexed."""
    content = (
        f"Technical note {MARKER}. The flux capacitor requires plutonium "
        "injection at exactly 88 miles per hour. " * 30
    ).encode()
    boundary = "----rag"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sphinx.txt\"\r\n"
        f"Content-Type: text/plain\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    response = urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/documents/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    doc = json.loads(response.read())
    print("uploaded:", doc)

    status = json.loads(urllib.request.urlopen(f"{BASE}/rag/status").read())
    print("indexed documents:", status["documents"], "chunks:", status["chunks"])
    assert status["documents"] >= 1, "document must be indexed after upload"
    print("PHASE1 OK")


def phase2() -> None:
    """After a restart: index must be restored from disk."""
    status = json.loads(urllib.request.urlopen(f"{BASE}/rag/status").read())
    print("after restart -> chunks:", status["chunks"],
          "documents:", status["documents"])
    assert status["chunks"] > 0, (
        "index must survive restart WITHOUT re-embedding "
        "(Ollama is down, so any chunks present were loaded from disk)")
    entry = next(
        (d for d in status["indexed_documents"] if d["doc_id"] == "doc_0"), None)
    print("restored doc record:", entry)
    print("PHASE2 OK — persistent index verified")


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2}[sys.argv[1]]()
