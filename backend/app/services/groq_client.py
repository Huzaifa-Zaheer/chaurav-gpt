"""Cloud LLM client: Groq (generation) + HuggingFace Inference API (embeddings).

Used when deploying to Vercel / cloud where Ollama can't run locally.
Groq provides fast LLM inference; HuggingFace provides free embeddings.

No extra SDK required — uses httpx (already a project dependency).
"""
from __future__ import annotations

import httpx

from .ollama_client import OllamaUnavailable

_GROQ_URL = "https://api.groq.com/openai/v1"
_HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"


class GroqClient:
    """Drop-in replacement for OllamaClient that calls cloud APIs."""

    def __init__(
        self,
        groq_api_key: str,
        gen_model: str = "llama-3.3-70b-versatile",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.gen_model = gen_model
        self.embed_model = embed_model
        self._groq_key = groq_api_key
        self._client = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        """Check if Groq API is reachable with the provided key."""
        try:
            r = await self._client.get(
                f"{_GROQ_URL}/models",
                headers={"Authorization": f"Bearer {self._groq_key}"},
                timeout=5.0,
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def embed(self, text: str) -> list[float]:
        """Get embeddings via HuggingFace free Inference API (no auth required)."""
        try:
            r = await self._client.post(
                f"{_HF_URL}/{self.embed_model}",
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"HF embedding request failed: {exc}") from exc
        if r.status_code == 503:
            raise OllamaUnavailable("Embedding model is loading, try again in a few seconds")
        if r.status_code != 200:
            raise OllamaUnavailable(f"HF embedding failed ({r.status_code}): {r.text[:200]}")
        data = r.json()
        # HF returns [[float, ...]] for single string input.
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        return data

    async def generate(self, prompt: str) -> str:
        """Generate text via Groq's OpenAI-compatible chat API."""
        try:
            r = await self._client.post(
                f"{_GROQ_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.gen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Groq generation failed: {exc}") from exc
        if r.status_code != 200:
            raise OllamaUnavailable(f"Groq generation failed ({r.status_code}): {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]
