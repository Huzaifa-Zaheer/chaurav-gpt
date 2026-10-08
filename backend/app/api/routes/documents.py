"""Document + RAG endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...db import DocumentDB
from ...services import OllamaUnavailable, RagService
from ..deps import error, get_doc_db, get_rag
from ..schemas import DocInsertRequest, QuestionRequest

router = APIRouter(prefix="/doc", tags=["documents"])

_OLLAMA_HELP = (
    "Ollama unavailable. Install from https://ollama.com then run: "
    "ollama pull nomic-embed-text && ollama pull llama3.2"
)


@router.post("/insert")
async def insert_document(body: DocInsertRequest, rag: RagService = Depends(get_rag)):
    """Chunk the text, embed each chunk via Ollama, store in DocumentDB."""
    try:
        res = await rag.ingest(body.title, body.text)
    except OllamaUnavailable:
        return error(_OLLAMA_HELP, 503)
    return {"ids": res.ids, "chunks": res.chunks, "dims": res.dims}


@router.delete("/delete/{doc_id}")
def delete_document(doc_id: int, docs: DocumentDB = Depends(get_doc_db)):
    return {"ok": docs.remove(doc_id)}


@router.get("/list")
def list_documents(docs: DocumentDB = Depends(get_doc_db)):
    out = []
    for d in docs.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": len(d.text.split()),
        })
    return out


@router.post("/search")
async def search_documents(body: QuestionRequest, rag: RagService = Depends(get_rag)):
    """Fast retrieval only (used by the UI visualizer)."""
    try:
        hits = await rag.retrieve(body.question, body.k)
    except OllamaUnavailable:
        return error("Ollama unavailable", 503)
    return {
        "contexts": [
            {"id": doc.id, "title": doc.title, "distance": round(dist, 4)}
            for dist, doc in hits
        ]
    }


@router.post("/ask")
async def ask(body: QuestionRequest, rag: RagService = Depends(get_rag)):
    """Full RAG pipeline: embed → retrieve → generate."""
    try:
        res = await rag.ask(body.question, body.k)
    except OllamaUnavailable:
        return error("Ollama unavailable. Run: ollama serve", 503)
    return {
        "answer": res.answer,
        "model": res.model,
        "contexts": [
            {"id": doc.id, "title": doc.title, "text": doc.text, "distance": round(dist, 4)}
            for dist, doc in res.contexts
        ],
        "docCount": len(rag.docs),
    }
