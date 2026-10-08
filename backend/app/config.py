"""Application configuration.

All settings can be overridden with environment variables, e.g.

    set GEN_MODEL=llama3.2:1b
    python run.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # HTTP server
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))

    # Ollama
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    gen_model: str = field(default_factory=lambda: os.getenv("GEN_MODEL", "llama3.2"))

    # Demo vector index
    demo_dims: int = 16

    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_build: int = 200
    hnsw_ef_search: int = 50

    # Document chunking / retrieval
    chunk_words: int = 250
    chunk_overlap: int = 30
    max_doc_distance: float = 0.7

    # Static frontend
    frontend_dir: Path = PROJECT_ROOT / "frontend"


settings = Settings()
