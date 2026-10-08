import hmac
import logging
import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse

from app.api.comfyui_routes import jobs_router
from app.api.comfyui_routes import router as comfyui_router
from app.api.lora_routes import router as lora_router
from app.api.routes import router
from app.core.config import settings
from app.core.jobs import JobStore
from app.services.comfyui_service import ComfyUIService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.lora_dataset_service import LoRAProjectService
from app.services.lora_training_service import LoRATrainingService
from app.services.memory_service import MemoryService
from app.services.prompt_service import PromptService
from app.services.rag.service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _iter_api_routes(routes):
    """Yield API routes, including those nested in included routers (FastAPI 0.138+)."""
    for route in routes:
        if type(route).__name__ == "_IncludedRouter":
            yield from _iter_api_routes(route.original_router.routes)
        elif hasattr(route, "routes"):
            yield from _iter_api_routes(route.routes)
        elif hasattr(route, "path") and hasattr(route, "methods"):
            yield route


ACCESS_COOKIE = "apollo_access"


def create_app(service: object | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Initializes all services (LLM, Memory, Document, Prompt, ComfyUI, LoRA).

    Args:
        service: Optional LLM service override, for tests that want to stub
            Ollama without a live server.
    """
    app = FastAPI(
        title="Apollo API",
        version="2.0.0",
        description=(
            "Unified local AI workspace: chat memory, document RAG, prompt templates, "
            "ComfyUI generation, and Ostris AI Toolkit LoRA training"
        ),
    )

    # ============ CORS MIDDLEWARE ============
    # Allow frontend to call backend from same origin and across network
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (safe for local network)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )

    # ============ ACCESS TOKEN (optional) ============
    # launcher.py publishes Apollo through a public Cloudflare tunnel whose traffic reaches
    # uvicorn from localhost, so "local" proves nothing. With APOLLO_ACCESS_TOKEN set, every
    # request needs that token: the X-Apollo-Token header, or the cookie set by opening
    # /?token=<token> once. Unset (a plain local run) leaves Apollo open, as before.
    @app.middleware("http")
    async def require_access_token(request: Request, call_next):
        token = os.environ.get("APOLLO_ACCESS_TOKEN", "")
        if not token or request.url.path in ("/live", "/health"):
            return await call_next(request)
        offered = request.query_params.get("token")
        if offered is not None and hmac.compare_digest(offered.encode(), token.encode()):
            rest = urlencode([(k, v) for k, v in request.query_params.multi_items() if k != "token"])
            response = RedirectResponse(request.url.path + (f"?{rest}" if rest else ""), status_code=303)
            https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            response.set_cookie(ACCESS_COOKIE, token, max_age=30 * 24 * 3600, httponly=True,
                                samesite="lax", secure=https)
            return response
        presented = request.headers.get("x-apollo-token") or request.cookies.get(ACCESS_COOKIE) or ""
        if hmac.compare_digest(presented.encode(), token.encode()):
            return await call_next(request)
        return PlainTextResponse(
            "Apollo is private. Open the link launcher.py printed (it ends in ?token=...), "
            "or send the X-Apollo-Token header.",
            status_code=401,
        )

    # ============ INITIALIZE SERVICES ============
    app.state.llm_service = service or LLMService()
    app.state.memory_service = MemoryService()
    app.state.document_service = DocumentService()
    app.state.prompt_service = PromptService()
    app.state.rag_service = RAGService()

    # ---- AI generation / LoRA training (independent of the document stack) ----
    # Constructed eagerly but connect lazily: neither ComfyUI nor AI Toolkit
    # needs to be running for the app to boot or for chat/RAG to work.
    app.state.job_store = JobStore(persist_path=settings.jobs_file)
    app.state.comfyui_service = ComfyUIService(jobs=app.state.job_store)
    app.state.lora_project_service = LoRAProjectService()
    app.state.lora_training_service = LoRATrainingService(
        projects=app.state.lora_project_service,
        jobs=app.state.job_store,
    )

    # ============ INCLUDE API ROUTES ============
    app.include_router(router)
    app.include_router(comfyui_router)
    app.include_router(lora_router)
    app.include_router(jobs_router)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        """Stop worker threads so a reload does not leave jobs running."""
        app.state.job_store.shutdown()

    # ============ FRONTEND - SERVE AT ROOT ============
    frontend_path = Path(__file__).parent / "frontend" / "index.html"

    @app.get("/", include_in_schema=False)
    async def root():
        """Serve the frontend HTML."""
        return FileResponse(str(frontend_path))

    # ============ DEBUG: LOG ALL REGISTERED ROUTES ============
    @app.on_event("startup")
    async def log_routes():
        """Log all registered routes on startup for debugging."""
        print("\n" + "=" * 70)
        print("AI WORKSPACE API STARTED")
        print("=" * 70)
        print("\nREGISTERED ENDPOINTS:")
        print("-" * 70)
        
        # Collect all routes (including nested included routers)
        routes = {}
        for route in _iter_api_routes(app.routes):
            if hasattr(route, "path") and hasattr(route, "methods"):
                path = route.path
                methods = ", ".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
                if path not in routes:
                    routes[path] = []
                routes[path].append(methods)
        
        # Print routes organized by path
        for path in sorted(routes.keys()):
            methods_list = routes[path]
            for methods in methods_list:
                print(f"  {methods:10} {path}")
        
        print("-" * 70)
        print("\nCRITICAL ROUTES:")
        print("  POST    /chat              <- Frontend calls this")
        print("  POST    /documents/upload  <- File uploads")
        print("  GET     /prompts           <- Load prompt templates")
        print("  GET     /documents         <- Load documents")
        print("  GET     /health            <- Health check")
        print("-" * 70)
        
        # Verify /chat endpoint exists
        chat_route_found = any(
            r.path == "/chat" and "POST" in r.methods
            for r in _iter_api_routes(app.routes)
        )
        
        if chat_route_found:
            print("\n[OK] /chat endpoint is REGISTERED")
        else:
            print("\n[WARN] /chat endpoint NOT FOUND!")
        
        print("\nFrontend API base:")
        print("   Same origin as this server (localhost, LAN, or Cloudflare tunnel)")
        print("   POST /chat with JSON: {\"prompt\": \"...\", \"session_id\": \"...\", \"model\": \"...\"}")
        print("\n" + "=" * 70 + "\n")

    return app


app = create_app()
