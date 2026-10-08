"""Status / stats endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.distance import METRIC_NAMES
from ...db import VectorDB
from ..deps import Container, get_container, get_vector_db

router = APIRouter(tags=["system"])


@router.get("/status")
async def status(c: Container = Depends(get_container)):
    return {
        "ollamaAvailable": await c.ollama.is_available(),
        "embedModel": c.ollama.embed_model,
        "genModel": c.ollama.gen_model,
        "docCount": len(c.doc_db),
        "docDims": c.doc_db.dims,
        "demoDims": c.vector_db.dims,
        "demoCount": len(c.vector_db),
    }


@router.get("/stats")
def stats(db: VectorDB = Depends(get_vector_db)):
    return {
        "count": len(db),
        "dims": db.dims,
        "algorithms": list(VectorDB.ALGORITHMS),
        "metrics": list(METRIC_NAMES),
    }
