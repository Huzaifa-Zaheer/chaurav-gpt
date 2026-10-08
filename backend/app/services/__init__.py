"""Business-logic services: Ollama client, chunking, RAG, demo seed data."""
from .chunker import chunk_text
from .demo_data import load_demo
from .ollama_client import OllamaClient, OllamaUnavailable
from .rag import RagService

__all__ = ["OllamaClient", "OllamaUnavailable", "RagService", "chunk_text", "load_demo"]
