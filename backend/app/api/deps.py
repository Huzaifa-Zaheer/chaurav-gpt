"""Dependency container + FastAPI dependency getters."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse

from ..config import Settings
from ..db import DocumentDB, VectorDB
from ..services import OllamaClient, RagService, load_demo


@dataclass
class Container:
    settings: Settings
    vector_db: VectorDB
    doc_db: DocumentDB
    ollama: OllamaClient
    rag: RagService

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        vector_db = VectorDB(settings.demo_dims, settings.hnsw_m,
                             settings.hnsw_ef_build, settings.hnsw_ef_search)
        load_demo(vector_db)
        doc_db = DocumentDB(settings.hnsw_m, settings.hnsw_ef_build, settings.hnsw_ef_search)
        ollama = OllamaClient(settings.ollama_url, settings.embed_model, settings.gen_model)
        rag = RagService(doc_db, ollama, settings.chunk_words,
                         settings.chunk_overlap, settings.max_doc_distance)
        return cls(settings, vector_db, doc_db, ollama, rag)


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_vector_db(request: Request) -> VectorDB:
    return get_container(request).vector_db


def get_doc_db(request: Request) -> DocumentDB:
    return get_container(request).doc_db


def get_ollama(request: Request) -> OllamaClient:
    return get_container(request).ollama


def get_rag(request: Request) -> RagService:
    return get_container(request).rag


def error(message: str, status: int = 400) -> JSONResponse:
    """Uniform error payload: {"error": "..."} (what the frontend expects)."""
    return JSONResponse({"error": message}, status_code=status)
