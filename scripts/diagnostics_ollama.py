#!/usr/bin/env python3
"""Probe local Ollama connectivity and available HTTP endpoints."""

import socket
from typing import Any

import requests

hosts = [
    "http://localhost:11434",
    "https://localhost:11434",
]
endpoints = [
    "/",
    "/api/generate",
    "/generate",
    "/api/chat",
    "/chat",
    "/v1/completions",
    "/v1/chat/completions",
    "/completion",
]

payload: dict[str, Any] = {"model": "ggml-small", "prompt": "Hello diagnostics", "stream": False}


def check_socket(hostname: str, port: int = 11434, timeout: float = 3.0):
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            return True, None
    except Exception as e:
        return False, e


def try_requests(url: str, verify: bool):
    try:
        r = requests.post(url, json=payload, timeout=5, verify=verify)
        return (r.status_code, r.headers.get("Content-Type"), r.text[:1000])
    except Exception as e:
        return e


def main() -> None:
    print("Socket checks:")
    ok, err = check_socket("localhost", 11434)
    print(f" - localhost:11434 open? {ok}")
    if err:
        print(f"   socket error: {err}")

    print("\nHTTP probe (no TLS verification):")
    for host in hosts:
        for ep in endpoints:
            url = host.rstrip("/") + ep
            print(f"\nTrying POST {url} (verify=False)")
            res = try_requests(url, verify=False)
            print(" ->", type(res).__name__)
            if isinstance(res, tuple):
                status, ctype, body = res
                print(f"    status={status} content-type={ctype}")
                print(f"    body (truncated): {body[:300]!r}")
            else:
                print(f"    exception: {res}")

    print("\nHTTP probe (with TLS verification where applicable):")
    for host in hosts:
        for ep in endpoints:
            url = host.rstrip("/") + ep
            print(f"\nTrying POST {url} (verify=True)")
            res = try_requests(url, verify=True)
            print(" ->", type(res).__name__)
            if isinstance(res, tuple):
                status, ctype, body = res
                print(f"    status={status} content-type={ctype}")
                print(f"    body (truncated): {body[:300]!r}")
            else:
                print(f"    exception: {res}")

    print("\nDone.")


if __name__ == "__main__":
    main()
