"""Async client for the local Ollama REST API.

Install:  https://ollama.com
Models:   ollama pull nomic-embed-text
          ollama pull llama3.2
"""
from __future__ import annotations

import httpx


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama cannot be reached or returns an error."""


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, gen_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def embed(self, text: str) -> list[float]:
        try:
            r = await self._client.post(
                "/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=httpx.Timeout(30.0, connect=3.0),
            )
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        if r.status_code != 200:
            raise OllamaUnavailable(f"embed failed ({r.status_code}): {r.text[:200]}")
        emb = r.json().get("embedding") or []
        if not emb:
            raise OllamaUnavailable("Ollama returned an empty embedding")
        return emb

    async def generate(self, prompt: str) -> str:
        try:
            r = await self._client.post(
                "/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=httpx.Timeout(180.0, connect=3.0),  # LLMs can be slow
            )
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        if r.status_code != 200:
            raise OllamaUnavailable(f"generate failed ({r.status_code}): {r.text[:200]}")
        return r.json().get("response", "")
