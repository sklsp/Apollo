"""Unit tests for ``app.clients.comfyui_client`` (HTTP layer).

The client is exercised with a fake ``requests.request`` injected via
monkeypatch, so no ComfyUI server or network access is needed. Covers the
request plumbing and every error branch of ``_request``, each public API
method including its failure paths, model discovery helpers, and the
module-level ``_extract_error`` helper.
"""

from __future__ import annotations

import json

import pytest
import requests

from app.clients.comfyui_client import ComfyUIClient, _extract_error
from app.core.exceptions import ComfyUIServiceError


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self.text = text or (content.decode("utf-8", errors="replace") if content else "")
        if content:
            self.content = content
        elif payload is not None:
            self.content = json.dumps(payload).encode("utf-8")
        else:
            self.content = b""
        self._payload = payload

    def json(self):
        if self._payload is None and self.content:
            return json.loads(self.content)
        return self._payload


def make_client(monkeypatch, captured, response=None, base_url="http://localhost:8188", timeout=60.0):
    """Build a ``ComfyUIClient`` whose HTTP calls land in ``captured``."""

    def fake_request(*args, **kwargs):
        captured.append((args, kwargs))
        return response if response is not None else FakeResponse()

    monkeypatch.setattr("app.clients.comfyui_client.requests.request", fake_request)
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.clients.comfyui_client.settings",
        SimpleNamespace(comfyui_base_url=base_url + "/", comfyui_timeout=timeout),
    )
    return ComfyUIClient()


# --------------------------------------------------------------------------- #
# construction and transport
# --------------------------------------------------------------------------- #


def test_init_defaults_strip_slash_and_use_settings(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured)

    assert client.base_url == "http://localhost:8188"
    assert client.timeout == 60.0
    assert len(client.client_id) == 32


def test_init_explicit_base_and_timeout(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.clients.comfyui_client.settings",
        SimpleNamespace(comfyui_base_url="http://other:1/", comfyui_timeout=99.0),
    )

    client = ComfyUIClient(base_url="http://mine:2/", timeout=7)

    assert client.base_url == "http://mine:2"
    assert client.timeout == 7


def test_request_success_returns_json(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(200, payload={"ok": True}))

    result = client._request("GET", "/system_stats")

    assert result == {"ok": True}
    (args, kwargs), = captured
    assert args[0] == "GET"
    assert args[1] == "http://localhost:8188/system_stats"
    assert kwargs["timeout"] == 60.0


def test_request_raw_returns_content_bytes(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(200, content=b"PNGDATA"))

    result = client._request("GET", "/view", raw=True)

    assert result == b"PNGDATA"


def test_request_empty_body_returns_empty_dict(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(204))

    result = client._request("GET", "/queue")

    assert result == {}


def test_request_malformed_json_raises_502(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, content=b"{not json"),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError, match="malformed response"):
        client._request("GET", "/queue")


def test_request_timeout_raises_504(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("app.clients.comfyui_client.requests.request", boom)
    client = ComfyUIClient(base_url="http://localhost:8188", timeout=60.0)

    with pytest.raises(ComfyUIServiceError, match="timed out after 60s") as excinfo:
        client._request("GET", "/system_stats")
    assert excinfo.value.status_code == 504


def test_request_unreachable_raises_503(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr("app.clients.comfyui_client.requests.request", boom)
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError, match="not reachable at http://localhost:8188"):
        client._request("GET", "/system_stats")


def test_request_http_error_raises_502_with_extracted_detail(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(400, payload={"error": {"message": "bad node"}}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError) as excinfo:
        client._request("POST", "/prompt")
    assert excinfo.value.status_code == 502
    assert "rejected the request (400)" in str(excinfo.value)
    assert excinfo.value.detail == "bad node"


# --------------------------------------------------------------------------- #
# api methods
# --------------------------------------------------------------------------- #


def test_ping_returns_stats_and_uses_short_timeout(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"system": {"os": "linux"}})
    )

    stats = client.ping()

    assert stats == {"system": {"os": "linux"}}
    (args, kwargs), = captured
    assert args[1].endswith("/system_stats")
    assert kwargs["timeout"] == 5


def test_ping_non_dict_raises(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=[1, 2]),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError, match="unexpected system stats"):
        client.ping()


def test_object_info_all_nodes_path(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(200, payload={"KSampler": {}}))

    result = client.object_info()

    assert result == {"KSampler": {}}
    (args, kwargs), = captured
    assert args[1].endswith("/object_info")


def test_object_info_single_node_path(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(200, payload={"KSampler": {}}))

    result = client.object_info("KSampler")

    assert result == {"KSampler": {}}
    (args, kwargs), = captured
    assert args[1].endswith("/object_info/KSampler")


def test_object_info_non_dict_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=["nope"]),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.object_info() == {}



def test_queue_prompt_returns_prompt_id(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"prompt_id": "abc-123"})
    )

    prompt_id = client.queue_prompt({"1": {"class": "KSampler"}})

    assert prompt_id == "abc-123"
    (args, kwargs), = captured
    assert args[1].endswith("/prompt")
    assert kwargs["json"]["client_id"] == client.client_id
    assert kwargs["json"]["prompt"] == {"1": {"class": "KSampler"}}


def test_queue_prompt_without_prompt_id_raises_with_node_errors(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"node_errors": {"1": "missing input"}}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError) as excinfo:
        client.queue_prompt({"1": {}})
    assert excinfo.value.status_code == 502
    assert "missing input" in excinfo.value.detail


def test_queue_prompt_without_prompt_id_uses_payload_in_detail(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"status": 400}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    with pytest.raises(ComfyUIServiceError) as excinfo:
        client.queue_prompt({"1": {}})
    assert excinfo.value.detail == "{}"


def test_history_returns_record_for_prompt_id(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"pid-1": {"status": "complete"}}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.history("pid-1") == {"status": "complete"}


def test_history_missing_key_or_null_value_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"other": 1, "pid-1": None}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.history("pid-1") == {}
    assert client.history("unknown") == {}


def test_history_non_dict_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=[1]),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.history("pid-1") == {}


def test_queue_state_returns_dict(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"queue_running": [], "queue_pending": []})
    )

    assert client.queue_state() == {"queue_running": [], "queue_pending": []}


def test_view_image_passes_params_and_returns_bytes(monkeypatch):
    captured = []
    client = make_client(monkeypatch, captured, FakeResponse(200, content=b"IMGBYTES"))

    data = client.view_image("a.png", subfolder="sf", folder_type="temp")

    assert data == b"IMGBYTES"
    (args, kwargs), = captured
    assert args[1].endswith("/view")
    assert kwargs["params"] == {"filename": "a.png", "subfolder": "sf", "type": "temp"}


def test_upload_image_posts_files_and_overwrite_flag(monkeypatch):
    captured = []
    client = make_client(
        monkeypatch, captured, FakeResponse(200, payload={"name": "b.png", "subfolder": ""})
    )

    result = client.upload_image("b.png", b"X", overwrite=False)

    assert result == {"name": "b.png", "subfolder": ""}
    (args, kwargs), = captured
    assert args[1].endswith("/upload/image")
    assert kwargs["files"] == {"image": ("b.png", b"X")}
    assert kwargs["data"] == {"overwrite": "false"}


def test_upload_image_non_dict_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=["nope"]),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.upload_image("b.png", b"X") == {}


# --------------------------------------------------------------------------- #
# model discovery
# --------------------------------------------------------------------------- #


def test_list_loras_extracts_dropdown_options(monkeypatch):
    payload = {
        "LoraLoader": {
            "input": {
                "required": {
                    "lora_name": [["a.safetensors", 2], {"options": ["a.safetensors"]}]
                }
            }
        }
    }
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=payload),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.list_loras() == ["a.safetensors", "2"]


def test_list_checkpoints_missing_spec_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload={"CheckpointLoaderSimple": {}}),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.list_checkpoints() == []


def test_combo_options_non_list_spec_returns_empty(monkeypatch):
    payload = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": "not-a-list"}}}
    }
    monkeypatch.setattr(
        "app.clients.comfyui_client.requests.request",
        lambda *a, **kw: FakeResponse(200, payload=payload),
    )
    client = ComfyUIClient(base_url="http://localhost:8188")

    assert client.list_checkpoints() == []


# --------------------------------------------------------------------------- #
# _extract_error helper
# --------------------------------------------------------------------------- #


def test_extract_error_dict_with_message():
    response = FakeResponse(400, payload={"error": {"message": "boom", "code": 7}})
    assert _extract_error(response) == "boom"


def test_extract_error_dict_without_message_falls_back_to_str():
    response = FakeResponse(400, payload={"error": {"code": 7}})
    result = _extract_error(response)
    assert "code" in result and "7" in result


def test_extract_error_plain_string_error():
    response = FakeResponse(400, payload={"error": "plain problem"})
    assert _extract_error(response) == "plain problem"


def test_extract_error_node_errors_fallback():
    response = FakeResponse(400, payload={"node_errors": {"1": "bad edge"}})
    assert _extract_error(response) == str({"1": "bad edge"})[:500]


def test_extract_error_plain_body_truncated():
    response = FakeResponse(400, payload={"foo": "x" * 600})
    assert len(_extract_error(response)) <= 500


def test_extract_error_bad_json_returns_text():
    response = FakeResponse(400, content=b"oops", text="plain error text")
    assert _extract_error(response) == "plain error text"[:500]


def test_extract_error_non_dict_body():
    response = FakeResponse(400, payload=[1, 2, 3])
    assert _extract_error(response) == "[1, 2, 3]"

