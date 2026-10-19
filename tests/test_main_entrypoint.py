

from typing import Any


def test_main_defaults_to_interactive_agent(monkeypatch) -> None:
    import main as main_module

    called = {}

    def fake_main() -> None:
        called["interactive"] = True

    monkeypatch.setattr(main_module, "run_cli", fake_main)
    monkeypatch.setattr(main_module, "run_server", lambda *args, **kwargs: called.setdefault("server", True))

    main_module.run_entrypoint([])

    assert called["interactive"] is True
    assert "server" not in called


def test_main_can_start_api_server(monkeypatch) -> None:
    import main as main_module

    called: dict[str, Any] = {}

    monkeypatch.setattr(main_module, "run_cli", lambda: called.setdefault("interactive", True))

    def fake_run_server(host: str, port: int) -> None:
        called["server"] = (host, port)

    monkeypatch.setattr(main_module, "run_server", fake_run_server)

    main_module.run_entrypoint(["--api"])

    assert called["server"] == ("0.0.0.0", 8000)
