#!/usr/bin/env python3
"""One-command launcher for the Apollo stack.

Starts FastAPI (uvicorn) and a Cloudflare quick tunnel, captures the public
URL, copies it to the clipboard, and opens the dashboard in the browser.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

try:
    import pyperclip
except ImportError:  # pragma: no cover - optional until requirements installed
    pyperclip = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")
UVICORN_APP = os.environ.get("UVICORN_APP", "app.main:app")
TUNNEL_TARGET = f"http://localhost:{PORT}"
TUNNEL_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

CLOUDFLARED_CANDIDATES = [
    os.environ.get("CLOUDFLARED_PATH"),
    r"C:\Users\Jayde\Desktop\cloudflared-windows-amd64.exe",
    r"C:\cloudflared\cloudflared-windows-amd64.exe",
    "cloudflared-windows-amd64.exe",
    "cloudflared",
]

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_processes: list[subprocess.Popen[str]] = []
_shutdown = threading.Event()
_public_url: str | None = None
_url_lock = threading.Lock()
_url_announced = threading.Event()


def _resolve_cloudflared() -> str:
    """Find the cloudflared executable."""
    for candidate in CLOUDFLARED_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
        # Allow bare executable names on PATH
        try:
            result = subprocess.run(
                ["where" if os.name == "nt" else "which", candidate],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0]
        except OSError:
            continue
    raise FileNotFoundError(
        "cloudflared not found. Set CLOUDFLARED_PATH or install cloudflared.\n"
        "Expected: C:\\Users\\Jayde\\Desktop\\cloudflared-windows-amd64.exe"
    )


def _stream_output(proc: subprocess.Popen[str], label: str, on_line=None) -> None:
    """Read process stdout line-by-line in a background thread."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if _shutdown.is_set():
            break
        text = line.rstrip()
        if text:
            print(f"[{label}] {text}", flush=True)
            if on_line:
                on_line(text)
    proc.stdout.close()


def _on_tunnel_line(line: str) -> None:
    """Capture the public Cloudflare URL from tunnel output."""
    global _public_url
    match = TUNNEL_URL_PATTERN.search(line)
    if not match:
        return
    with _url_lock:
        if _public_url is not None:
            return
        _public_url = match.group(0)
    _announce_public_url(_public_url)


def _announce_public_url(url: str) -> None:
    """Print, copy, and open the public dashboard URL."""
    print("\n" + "=" * 70)
    print("  APOLLO — LIVE")
    print("=" * 70)
    print(f"\n  Local:   http://localhost:{PORT}")
    print(f"  Public:  {url}")
    print(f"  API:     {url}/chat")
    print()

    if pyperclip is not None:
        try:
            pyperclip.copy(url)
            print("  Clipboard: public URL copied")
        except pyperclip.PyperclipException as exc:
            print(f"  Clipboard: could not copy ({exc})")
    else:
        print("  Clipboard: install pyperclip to enable auto-copy")

    print("\n  Opening dashboard in browser...")
    print("=" * 70 + "\n", flush=True)

    webbrowser.open(url)
    _url_announced.set()


def _wait_for_backend(timeout: float = 45.0) -> bool:
    """Poll /health until the FastAPI server is ready."""
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{PORT}/health"
    print(f"Waiting for FastAPI at {health_url} ...", flush=True)
    while time.monotonic() < deadline and not _shutdown.is_set():
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print("FastAPI is ready.\n", flush=True)
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _start_uvicorn() -> subprocess.Popen[str]:
    """Start the FastAPI backend via uvicorn."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        UVICORN_APP,
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    print(f"Starting FastAPI: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _processes.append(proc)
    threading.Thread(
        target=_stream_output,
        args=(proc, "api"),
        daemon=True,
    ).start()
    return proc


def _start_tunnel(cloudflared: str) -> subprocess.Popen[str]:
    """Start the Cloudflare quick tunnel."""
    cmd = [cloudflared, "tunnel", "--url", TUNNEL_TARGET]
    print(f"Starting tunnel: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _processes.append(proc)
    threading.Thread(
        target=_stream_output,
        args=(proc, "tunnel"),
        kwargs={"on_line": _on_tunnel_line},
        daemon=True,
    ).start()
    return proc


def _shutdown_all(signum: int | None = None, _frame=None) -> None:
    """Terminate all child processes cleanly."""
    if _shutdown.is_set():
        return
    _shutdown.set()
    if signum is not None:
        print("\nShutting down...", flush=True)
    for proc in _processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in _processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("All processes stopped. Goodbye.", flush=True)


def _monitor() -> int:
    """Run until interrupted or a child process exits unexpectedly."""
    while not _shutdown.is_set():
        for proc in _processes:
            code = proc.poll()
            if code is not None:
                label = "api" if proc is _processes[0] else "tunnel"
                print(f"\n[{label}] process exited with code {code}", flush=True)
                _shutdown_all()
                return code if code != 0 else 1
        time.sleep(0.25)
    return 0


def main() -> int:
    """Entry point for the all-in-one dev launcher."""
    print("=" * 70)
    print("  APOLLO — STARTUP")
    print("=" * 70)
    print(f"  Project:  {PROJECT_ROOT}")
    print(f"  Backend:  {UVICORN_APP} on {HOST}:{PORT}")
    print(f"  Ollama:   {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print("=" * 70 + "\n", flush=True)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    signal.signal(signal.SIGINT, _shutdown_all)
    signal.signal(signal.SIGTERM, _shutdown_all)
    atexit.register(_shutdown_all)

    try:
        cloudflared = _resolve_cloudflared()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    uvicorn_proc = _start_uvicorn()
    if not _wait_for_backend():
        print("ERROR: FastAPI did not become ready in time.", file=sys.stderr)
        _shutdown_all()
        return 1

    _start_tunnel(cloudflared)

    print("Waiting for Cloudflare public URL (Ctrl+C to stop)...\n", flush=True)

    # Fallback: open local dashboard if tunnel URL takes too long
    def _local_fallback() -> None:
        if not _url_announced.wait(timeout=60):
            print(
                f"\nTunnel URL not detected yet — opening local dashboard "
                f"http://localhost:{PORT}\n",
                flush=True,
            )
            webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_local_fallback, daemon=True).start()

    try:
        return _monitor()
    except KeyboardInterrupt:
        _shutdown_all()
        return 0
    finally:
        if uvicorn_proc.poll() is None:
            _shutdown_all()


if __name__ == "__main__":
    raise SystemExit(main())
