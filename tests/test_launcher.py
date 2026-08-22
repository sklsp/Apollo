"""Launcher startup semantics: /live decides readiness, dependencies don't block.

The launcher module is imported (not subprocess-run) so the wait loop can be
exercised against real local HTTP servers started per test.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_launcher():
    """Import launcher.py as a module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "apollo_launcher", PROJECT_ROOT / "launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launcher():
    return _load_launcher()


class _Handler(BaseHTTPRequestHandler):
    """Configurable stand-in for the FastAPI app."""

    server_config = {"live_status": 200, "health_status": 503, "delay": 0.0}

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.server_config["delay"]:
            time.sleep(self.server_config["delay"])
        if self.path == "/live":
            status = self.server_config["live_status"]
        elif self.path == "/health":
            status = self.server_config["health_status"]
        else:
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):  # silence test output
        pass


class _DeadHandler(_Handler):
    def do_GET(self):  # noqa: N802
        raise ConnectionAbortedError


def _serve(handler_class, port=None):
    """Start a test HTTP server on ``port`` (or ephemeral). Returns the server."""
    server = ThreadingHTTPServer(("127.0.0.1", port if port is not None else 0),
                                 handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def free_port():
    """A usable loopback port.

    On Windows, bind -> close -> rebind on the same port frequently fails
    with WinError 10013, so we let the OS hand out an unused port via a
    server-style bind with SO_REUSEADDR (matching ThreadingHTTPServer), then
    release it for the test's server to take.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        yield sock.getsockname()[1]


class TestWaitForBackend:
    def _patch_port(self, launcher, port):
        original = launcher.PORT
        launcher.PORT = port
        return original

    def test_api_up_with_unhealthy_dependencies_still_starts(self, launcher, free_port):
        """/health returning 503 must NOT fail startup — that's the core bug."""
        original = self._patch_port(launcher, free_port)
        server = _serve(_Handler, port=free_port)  # live=200, health=503
        try:
            result = launcher._wait_for_backend(timeout=5)
            assert result.ok is True
            assert result.last_status == 200
            # The dependency problem is surfaced, not hidden.
            assert result.dependency_warnings, "503 from /health should be reported"
            assert "503" in result.dependency_warnings[0]
        finally:
            launcher.PORT = original
            server.shutdown()

    def test_waits_until_api_comes_up(self, launcher, free_port):
        original = self._patch_port(launcher, free_port)

        class LateHandler(_Handler):
            pass

        server_holder = {}

        def start_late():
            time.sleep(1.5)
            server_holder["server"] = _serve(LateHandler, port=free_port)

        thread = threading.Thread(target=start_late, daemon=True)
        thread.start()
        try:
            result = launcher._wait_for_backend(timeout=10)
            assert result.ok is True, "launcher must wait for a slow-but-starting API"
        finally:
            launcher.PORT = original
            thread.join(timeout=3)
            if "server" in server_holder:
                server_holder["server"].shutdown()

    def test_genuinely_dead_api_fails_with_diagnostic(self, launcher, free_port):
        original = self._patch_port(launcher, free_port)
        try:
            result = launcher._wait_for_backend(timeout=3)
            assert result.ok is False
            assert result.last_error, "failure must carry the actual error"
            assert result.last_url and "/live" in result.last_url
        finally:
            launcher.PORT = original

    def test_uses_explicit_loopback_first(self, launcher, free_port):
        """Windows may resolve localhost to ::1; 127.0.0.1 must be tried first."""
        original = self._patch_port(launcher, free_port)
        server = _serve(_Handler, port=free_port)
        tried = []
        real_probe = launcher._http_probe

        def recording_probe(url, timeout=3.0):
            tried.append(url)
            return real_probe(url, timeout)

        launcher._http_probe = recording_probe
        try:
            result = launcher._wait_for_backend(timeout=5)
            assert result.ok is True
            assert tried[0].startswith("http://127.0.0.1:"), (
                f"first probe must be explicit loopback, got {tried[0]}")
        finally:
            launcher._http_probe = real_probe
            launcher.PORT = original
            server.shutdown()


class TestHttpProbe:
    def test_http_error_returns_status_not_exception(self, launcher):
        # Port 1 is reliably closed -> connection refused -> error path.
        status, body, error = launcher._http_probe("http://127.0.0.1:1/live", timeout=2)
        assert status is None
        assert error

    def test_success_returns_status_and_body(self, launcher, free_port):
        server = _serve(_Handler, port=free_port)
        try:
            status, body, error = launcher._http_probe(
                f"http://127.0.0.1:{free_port}/live")
            assert status == 200
            assert error is None
        finally:
            server.shutdown()


class TestPortCheck:
    def test_port_in_use_is_detected_before_starting(self, launcher, free_port):
        server = _serve(_Handler, port=free_port)
        try:
            original = launcher.PORT
            launcher.PORT = free_port
            try:
                assert launcher._check_port_available() is False, (
                    "must refuse to start when the port is already owned")
            finally:
                launcher.PORT = original
        finally:
            server.shutdown()

    def test_free_port_passes(self, launcher, free_port):
        original = launcher.PORT
        launcher.PORT = free_port
        try:
            assert launcher._check_port_available() is True
        finally:
            launcher.PORT = original


class TestStartupResultDiagnostics:
    def test_failure_report_mentions_process_and_url(self, launcher, capsys):
        result = launcher.StartupResult()
        result.ok = False
        result.last_url = "http://127.0.0.1:8000/live"
        result.last_error = "[WinError 10061] target machine actively refused"

        if launcher._processes:
            launcher._processes.clear()

        launcher._report_startup_failure(result)
        output = capsys.readouterr().out
        assert "APOLLO STARTUP FAILED" in output
        assert "http://127.0.0.1:8000/live" in output
        assert "actively refused" in output
