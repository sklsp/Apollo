#!/usr/bin/env python3
"""One-command launcher for the Apollo stack.

Starts FastAPI (uvicorn) and a Cloudflare quick tunnel, captures the public
URL, copies it to the clipboard, and opens the dashboard in the browser.
"""

from __future__ import annotations

import atexit
import os
import re
import secrets
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
        "cloudflared not found. Set CLOUDFLARED_PATH or install cloudflared."
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


def _with_token(base: str) -> str:
    """The dashboard link that carries the access token (opening it once sets a cookie)."""
    token = os.environ.get("APOLLO_ACCESS_TOKEN")
    return f"{base}/?token={token}" if token else base


def _announce_public_url(url: str) -> None:
    """Print, copy, and open the public dashboard URL."""
    link = _with_token(url)
    print("\n" + "=" * 70)
    print("  APOLLO — LIVE")
    print("=" * 70)
    print(f"\n  Local:   {_with_token(f'http://localhost:{PORT}')}")
    print(f"  Public:  {link}")
    print(f"  API:     {url}/chat  (send the X-Apollo-Token header)")
    print("  Only people with this link can open Apollo; keep it private.")
    print()

    if pyperclip is not None:
        try:
            pyperclip.copy(link)
            print("  Clipboard: public URL copied")
        except pyperclip.PyperclipException as exc:
            print(f"  Clipboard: could not copy ({exc})")
    else:
        print("  Clipboard: install pyperclip to enable auto-copy")

    print("\n  Opening dashboard in browser...")
    print("=" * 70 + "\n", flush=True)

    webbrowser.open(link)
    _url_announced.set()


class StartupResult:
    """Outcome of the backend startup wait, with diagnostics on failure."""

    def __init__(self) -> None:
        self.ok = False
        self.last_url: str | None = None
        self.last_status: int | None = None
        self.last_body: str | None = None
        self.last_error: str | None = None
        self.dependency_warnings: list[str] = []


def _http_probe(url: str, timeout: float = 3.0) -> tuple[int | None, str | None, str | None]:
    """GET a URL. Returns (status, body, error); exactly one of status/error is set."""
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(2048).decode("utf-8", "replace"), None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(2048).decode("utf-8", "replace")
        except OSError:
            body = None
        return exc.code, body, None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, None, str(exc.reason or exc)


def _wait_for_backend(timeout: float = 45.0) -> StartupResult:
    """Wait until the API process is responding, then report dependency state.

    Readiness semantics:

    * **/live** decides startup success — it only means "the HTTP server is
      up". Optional dependencies (Ollama etc.) being down must NOT fail the
      launch, so any HTTP response (including 503 from /health) counts as the
      process being alive.
    * **/health** is then probed once to *report* dependency warnings without
      influencing the outcome.

    Addresses are tried in order: explicit loopback first (Windows can resolve
    ``localhost`` to IPv6, which uvicorn on 0.0.0.0 may not answer), then the
    hostname as a fallback.
    """
    result = StartupResult()
    deadline = time.monotonic() + timeout
    candidate_hosts = ["127.0.0.1", "localhost"]

    print(f"Waiting for the Apollo API on port {PORT} ...", flush=True)

    while time.monotonic() < deadline and not _shutdown.is_set():
        for host in candidate_hosts:
            live_url = f"http://{host}:{PORT}/live"
            result.last_url = live_url
            status, body, error = _http_probe(live_url)
            if status is not None:
                # Any HTTP response proves the server process is up.
                result.ok = True
                result.last_status = status
                result.last_body = body
                result.last_error = None
                print(f"API is responding at http://{host}:{PORT} (HTTP {status}).",
                      flush=True)

                # Dependency report: informational only.
                health_status, health_body, _ = _http_probe(
                    f"http://{host}:{PORT}/health")
                if health_status is not None and health_status != 200:
                    result.dependency_warnings.append(
                        f"/health returned HTTP {health_status}: "
                        f"{(health_body or '')[:200]}"
                    )
                return result
            result.last_error = error

        time.sleep(0.5)

    return result


def _report_startup_failure(result: StartupResult) -> None:
    """Print an actionable diagnosis instead of a bare timeout message."""
    proc_alive = _processes and _processes[0].poll() is None

    print("\n" + "=" * 70, flush=True)
    print("  APOLLO STARTUP FAILED", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  API process: {'RUNNING' if proc_alive else 'EXITED'}", flush=True)
    print(f"  Healthcheck URL: {result.last_url}", flush=True)

    if result.last_status is not None:
        print(f"  Last HTTP status: {result.last_status}", flush=True)
        if result.last_body:
            print(f"  Response: {result.last_body[:300]}", flush=True)

    if result.last_error:
        print(f"\n  Last connection error:\n    {result.last_error}", flush=True)

    if proc_alive and result.last_error:
        print(
            "\n  The API process is running but not accepting connections yet.\n"
            "  Common causes:\n"
            f"    - Another process already holds port {PORT}\n"
            "      (check with: netstat -ano | findstr "
            f"\":{PORT}.*LISTENING\")\n"
            "    - The app failed during import/startup — see [api] output above\n"
            "    - Slow first start (embedding model download)",
            flush=True,
        )
    elif not proc_alive:
        print(
            "\n  The API process exited during startup — see the [api] output\n"
            "  above for the traceback.",
            flush=True,
        )

    if result.dependency_warnings:
        print("\n  Dependency warnings (do NOT block startup):", flush=True)
        for warning in result.dependency_warnings:
            print(f"    - {warning}", flush=True)

    print("=" * 70 + "\n", flush=True)


def _check_port_available() -> bool:
    """Fail fast with a clear message when something else owns our port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            print(
                f"\nERROR: Port {PORT} is already in use.\n"
                f"Another Apollo instance (or another app) is listening on it.\n"
                f"Find it with:   netstat -ano | findstr \":{PORT}.*LISTENING\"\n"
                f"Stop it with:   taskkill /F /PID <pid>\n"
                f"Or use another port:  set PORT=8010 && python launcher.py\n",
                file=sys.stderr,
                flush=True,
            )
            return False
    return True


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

    if not _check_port_available():
        return 1

    try:
        cloudflared = _resolve_cloudflared()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # The quick tunnel makes Apollo reachable from the internet, so the API gets an access token
    # (keep your own by setting APOLLO_ACCESS_TOKEN). uvicorn inherits it from this environment.
    os.environ.setdefault("APOLLO_ACCESS_TOKEN", secrets.token_urlsafe(24))
    uvicorn_proc = _start_uvicorn()
    startup = _wait_for_backend()
    if not startup.ok:
        _report_startup_failure(startup)
        _shutdown_all()
        return 1

    if startup.dependency_warnings:
        print("  Dependency status (the API is up; these are optional):", flush=True)
        for warning in startup.dependency_warnings:
            print(f"    - {warning}", flush=True)
        print(f"  Dashboard: http://localhost:{PORT}  (degraded features are "
              f"marked in the UI)\n", flush=True)

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
            webbrowser.open(_with_token(f"http://localhost:{PORT}"))

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
