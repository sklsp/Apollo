"""CLI and API entrypoints for Apollo."""

import argparse

from ai_agent.cli import main as run_cli
from app.main import app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    print(f"Starting FastAPI backend on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_entrypoint(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the AI document agent")
    parser.add_argument(
        "--api",
        action="store_true",
        help="start the FastAPI backend instead of the interactive CLI",
    )
    args = parser.parse_args(argv)

    if args.api:
        run_server("0.0.0.0", 8000)
        return

    print("Starting interactive AI document agent CLI")
    run_cli()


if __name__ == "__main__":
    run_entrypoint()
