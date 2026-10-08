"""FastAPI application factory."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.deps import Container
from .api.routes import api_router
from .config import Settings, settings as default_settings

log = logging.getLogger("vectordb")


def create_app(settings: Settings = default_settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = Container.build(settings)
        app.state.container = container

        up = await container.ollama.is_available()
        print("=== VectorDB Engine (Python) ===")
        print(f"http://localhost:{settings.port}")
        print(f"{len(container.vector_db)} demo vectors | {settings.demo_dims} dims | HNSW+KD-Tree+BruteForce")
        print(f"Ollama: {'ONLINE' if up else 'OFFLINE (install from ollama.com)'}")
        if up:
            print(f"  embed model: {settings.embed_model}  gen model: {settings.gen_model}")

        yield
        await container.ollama.aclose()

    app = FastAPI(title="VectorDB — HNSW + RAG", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = f"{field}: {first.get('msg', 'invalid request')}" if field else "invalid request"
        return JSONResponse({"error": msg}, status_code=422)

    # On Vercel, routes must live under /api to match vercel.json routing.
    # Locally, routes stay at the root.
    prefix = "/api" if os.getenv("VERCEL") else ""
    app.include_router(api_router, prefix=prefix)

    # Serve the frontend last so API routes take precedence.
    # On Vercel, static files are served by Vercel's CDN, not FastAPI.
    if not os.getenv("VERCEL") and settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    elif not os.getenv("VERCEL"):
        log.warning("Frontend directory not found: %s", settings.frontend_dir)

    return app


app = create_app()
