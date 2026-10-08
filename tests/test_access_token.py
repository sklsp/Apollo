"""Optional access token: off by default, required everywhere (but /live and /health) once set."""

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token-123"


class StubLLM:
    def health_check(self):
        return {"status": "ok", "ollama_reachable": True, "model": "llama3.2"}

    def chat(self, prompt, model=None):
        return f"echo:{prompt}"

    def list_models(self):
        return ["llama3.2"]


@pytest.fixture
def make_client(tmp_settings):
    from app.main import create_app

    def _make():
        return TestClient(create_app(service=StubLLM()))

    return _make


def test_without_a_token_configured_apollo_stays_open(make_client, monkeypatch):
    monkeypatch.delenv("APOLLO_ACCESS_TOKEN", raising=False)
    with make_client() as client:
        assert client.get("/").status_code == 200


def test_with_a_token_every_page_and_route_needs_it(make_client, monkeypatch):
    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    with make_client() as client:
        assert client.get("/").status_code == 401
        assert client.get("/prompts").status_code == 401
        assert client.get("/", headers={"X-Apollo-Token": "wrong"}).status_code == 401
        assert client.get("/", headers={"X-Apollo-Token": TOKEN}).status_code == 200


def test_probes_stay_open_for_the_launcher(make_client, monkeypatch):
    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    with make_client() as client:
        assert client.get("/live").status_code != 401
        assert client.get("/health").status_code != 401


def test_the_token_link_sets_a_cookie_and_drops_the_token_from_the_url(make_client, monkeypatch):
    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    with make_client() as client:
        first = client.get(f"/?token={TOKEN}&tab=docs", follow_redirects=False)
        assert first.status_code == 303
        assert first.headers["location"] == "/?tab=docs"
        cookie = first.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie
        assert client.get("/").status_code == 200  # the client kept the cookie


def test_a_wrong_token_link_sets_nothing(make_client, monkeypatch):
    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    with make_client() as client:
        response = client.get("/?token=nope", follow_redirects=False)
        assert response.status_code == 401
        assert "set-cookie" not in response.headers


def test_launcher_links_carry_the_token(monkeypatch):
    import launcher

    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    assert launcher._with_token("https://x.trycloudflare.com") == f"https://x.trycloudflare.com/?token={TOKEN}"
    monkeypatch.delenv("APOLLO_ACCESS_TOKEN")
    assert launcher._with_token("http://localhost:8000") == "http://localhost:8000"


def test_other_sites_cannot_read_responses_with_the_cookie(make_client, monkeypatch):
    """With CORS credentials on, Starlette echoed any Origin that sent a cookie."""
    monkeypatch.setenv("APOLLO_ACCESS_TOKEN", TOKEN)
    with make_client() as client:
        response = client.get("/prompts", headers={"Origin": "https://evil.example", "Cookie": f"apollo_access={TOKEN}"})
        assert response.headers.get("access-control-allow-origin") != "https://evil.example"
        assert response.headers.get("access-control-allow-credentials") is None
