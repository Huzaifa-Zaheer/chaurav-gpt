"""RAG pipeline: embed → retrieve → build prompt → generate."""
from __future__ import annotations

from dataclasses import dataclass

from ..core import DocItem
from ..db import DocumentDB
from .chunker import chunk_text
from .ollama_client import OllamaClient

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the user's question directly. "
    "Use the provided context if it contains relevant information. "
    "If it doesn't, just use your own general knowledge. "
    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
    "'the context doesn't mention'. Just answer the question naturally.\n\n"
    "Context:\n{context}"
    "Question: {question}\n\n"
    "Answer:"
)


@dataclass
class InsertResult:
    ids: list[int]
    chunks: int
    dims: int


@dataclass
class AskResult:
    answer: str
    model: str
    contexts: list[tuple[float, DocItem]]


class RagService:
    def __init__(self, docs: DocumentDB, ollama: OllamaClient,
                 chunk_words: int = 250, chunk_overlap: int = 30, max_dist: float = 0.7) -> None:
        self.docs = docs
        self.ollama = ollama
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.max_dist = max_dist

    async def ingest(self, title: str, text: str) -> InsertResult:
        """Chunk the text, embed every chunk and store it."""
        chunks = chunk_text(text, self.chunk_words, self.chunk_overlap)
        # Embed everything first so a mid-way Ollama failure doesn't leave partial docs.
        embeddings = [await self.ollama.embed(c) for c in chunks]
        ids: list[int] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            ids.append(self.docs.insert(chunk_title, chunk, emb))
        return InsertResult(ids=ids, chunks=len(chunks), dims=self.docs.dims)

    async def retrieve(self, question: str, k: int) -> list[tuple[float, DocItem]]:
        q_emb = await self.ollama.embed(question)
        return self.docs.search(q_emb, k, self.max_dist)

    @staticmethod
    def build_prompt(question: str, hits: list[tuple[float, DocItem]]) -> str:
        context = "".join(f"[{i + 1}] {doc.title}:\n{doc.text}\n\n" for i, (_, doc) in enumerate(hits))
        return PROMPT_TEMPLATE.format(context=context, question=question)

    async def ask(self, question: str, k: int) -> AskResult:
        hits = await self.retrieve(question, k)
        answer = await self.ollama.generate(self.build_prompt(question, hits))
        return AskResult(answer=answer, model=self.ollama.gen_model, contexts=hits)
