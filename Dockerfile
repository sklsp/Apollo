# Apollo - FastAPI backend (portfolio item 38)
#
# Build:   docker build -t apollo .
# Run:     docker run --rm -p 8000:8000 -v apollo-data:/app/data \
#              -e OLLAMA_BASE_URL=http://host.docker.internal:11434 apollo
#
# The app expects Ollama (and optionally ComfyUI) to run OUTSIDE the container;
# from inside, reach them via host.docker.internal (Docker Desktop) or with
# --network=host on Linux.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first, so sentence-transformers does not pull the multi-GB
# CUDA build (same trick as .github/workflows/ci.yml). Runtime deps only -
# ruff/mypy/pytest stay out of the image.
COPY requirements.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Run as an unprivileged user; /app/data is a volume for runtime state
# (LoRA library, generated images, jobs.json - created at startup). The dir
# must exist in the image owned by appuser: Docker initializes an empty
# volume from the image contents on first run, which preserves ownership.
RUN useradd --create-home appuser \
    && mkdir -p data/loras data/generated \
    && chown -R appuser:appuser /app/data

USER appuser

COPY --chown=appuser:appuser . .

VOLUME ["/app/data"]

EXPOSE 8000

# Liveness check: /live is deliberately dependency-free (Ollama/ComfyUI down
# does not make the API "not started"); /health reports dependency state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/live', timeout=4)" || exit 1

CMD ["python", "main.py", "--api"]
