"""Demo 16-D vector endpoints: search, insert, delete, list, benchmark, hnsw-info."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...core import get_dist_fn
from ...db import VectorDB
from ..deps import error, get_vector_db
from ..schemas import InsertVectorRequest

router = APIRouter(tags=["demo-vectors"])


def _parse_vec(raw: str) -> list[float]:
    out: list[float] = []
    for tok in raw.split(","):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _round_vec(v) -> list[float]:
    return [round(float(x), 4) for x in v]


@router.get("/search")
def search(
    v: str = Query("", description="Comma-separated query vector"),
    k: int = Query(5, ge=1, le=100),
    metric: str = "cosine",
    algo: str = "hnsw",
    db: VectorDB = Depends(get_vector_db),
):
    q = _parse_vec(v)
    if len(q) != db.dims:
        return error(f"need {db.dims}D vector")
    out = db.search(q, k, metric or "cosine", algo or "hnsw")
    return {
        "results": [
            {
                "id": h.id,
                "metadata": h.metadata,
                "category": h.category,
                "distance": round(h.distance, 6),
                "embedding": _round_vec(h.emb),
            }
            for h in out.hits
        ],
        "latencyUs": out.latency_us,
        "algo": out.algo,
        "metric": out.metric,
    }


@router.post("/insert")
def insert(body: InsertVectorRequest, db: VectorDB = Depends(get_vector_db)):
    if len(body.embedding) != db.dims:
        return error("invalid body")
    new_id = db.insert(body.metadata, body.category, body.embedding, get_dist_fn("cosine"))
    return {"id": new_id}


@router.delete("/delete/{item_id}")
def delete(item_id: int, db: VectorDB = Depends(get_vector_db)):
    return {"ok": db.remove(item_id)}


@router.get("/items")
def items(db: VectorDB = Depends(get_vector_db)):
    return [
        {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": _round_vec(v.emb)}
        for v in db.all()
    ]


@router.get("/benchmark")
def benchmark(
    v: str = "",
    k: int = Query(5, ge=1, le=100),
    metric: str = "cosine",
    db: VectorDB = Depends(get_vector_db),
):
    q = _parse_vec(v)
    if len(q) != db.dims:
        return error(f"need {db.dims}D vector")
    b = db.benchmark(q, k, metric or "cosine")
    return {
        "bruteforceUs": b.bruteforce_us,
        "kdtreeUs": b.kdtree_us,
        "hnswUs": b.hnsw_us,
        "itemCount": b.item_count,
    }


@router.get("/hnsw-info")
def hnsw_info(db: VectorDB = Depends(get_vector_db)):
    gi = db.hnsw_info()
    return {
        "topLayer": gi.top_layer,
        "nodeCount": gi.node_count,
        "nodesPerLayer": gi.nodes_per_layer,
        "edgesPerLayer": gi.edges_per_layer,
        "nodes": gi.nodes,
        "edges": gi.edges,
    }
