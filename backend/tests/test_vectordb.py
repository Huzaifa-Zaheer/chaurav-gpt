"""Run with:  python -m pytest -q   (from the backend/ folder)"""
from __future__ import annotations

import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import VectorItem, cosine, euclidean, manhattan
from app.indexes import HNSW, BruteForce, KDTree
from app.main import create_app
from app.services import chunk_text


# ── unit: metrics ──────────────────────────────────────────────
def test_metrics():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert euclidean(a, b) == pytest.approx(2 ** 0.5)
    assert manhattan(a, b) == pytest.approx(2.0)
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, a) == pytest.approx(0.0, abs=1e-6)


# ── unit: indexes agree with brute force ───────────────────────
@pytest.mark.parametrize("dist", [euclidean, cosine])
def test_indexes_match_brute_force(dist):
    rng = random.Random(0)
    dims, n, k = 8, 300, 5
    bf, kd, hn = BruteForce(), KDTree(dims), HNSW(16, 200)
    for i in range(n):
        v = VectorItem(i, f"v{i}", "x", np.array([rng.random() for _ in range(dims)], dtype=np.float32))
        bf.insert(v); kd.insert(v); hn.insert(v, dist)

    recall_hnsw = 0
    for _ in range(20):
        q = np.array([rng.random() for _ in range(dims)], dtype=np.float32)
        truth = {i for _, i in bf.knn(q, k, dist)}
        if dist is euclidean:  # KD-tree pruning is exact for euclidean
            assert {i for _, i in kd.knn(q, k, dist)} == truth
        recall_hnsw += len(truth & {i for _, i in hn.knn(q, k, 50, dist)})
    assert recall_hnsw / (20 * k) >= 0.9


def test_hnsw_remove_entry_point():
    h = HNSW(4, 20)
    for i in range(30):
        h.insert(VectorItem(i, "", "", np.array([float(i), 0.0])), euclidean)
    for i in range(30):
        h.remove(i)
        if i < 29:
            assert h.knn(np.array([29.0, 0.0]), 1, 10, euclidean)[0][1] == 29
    assert h.knn(np.array([0.0, 0.0]), 1, 10, euclidean) == []


def test_chunker():
    words = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_text(words, 250, 30)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w220"
    assert chunk_text("short text") == ["short text"]
    assert chunk_text("   ") == []


# ── integration: HTTP API ──────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


Q = ",".join(["0.9"] * 4 + ["0.1"] * 12)


def test_items_and_stats(client):
    assert len(client.get("/items").json()) == 20
    s = client.get("/stats").json()
    assert s["count"] == 20 and s["dims"] == 16


@pytest.mark.parametrize("algo", ["hnsw", "kdtree", "bruteforce"])
def test_search(client, algo):
    r = client.get(f"/search?v={Q}&k=3&metric=cosine&algo={algo}").json()
    assert len(r["results"]) == 3
    assert all(x["category"] == "cs" for x in r["results"])
    assert r["algo"] == algo and "latencyUs" in r


def test_search_bad_vector(client):
    r = client.get("/search?v=1,2,3")
    assert r.status_code == 400 and "16D" in r.json()["error"]


def test_insert_delete(client):
    emb = [0.1] * 8 + [0.9] * 4 + [0.1] * 4
    new_id = client.post("/insert", json={"metadata": "Pho", "category": "food", "embedding": emb}).json()["id"]
    assert any(i["id"] == new_id for i in client.get("/items").json())
    assert client.delete(f"/delete/{new_id}").json() == {"ok": True}
    assert client.delete(f"/delete/{new_id}").json() == {"ok": False}


def test_validation_error_shape(client):
    r = client.post("/insert", json={"metadata": ""})
    assert r.status_code == 422 and "error" in r.json()


def test_benchmark_and_hnsw_info(client):
    b = client.get(f"/benchmark?v={Q}&k=5").json()
    assert {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"} <= b.keys()
    h = client.get("/hnsw-info").json()
    assert h["nodeCount"] == 20 and h["nodesPerLayer"][0] == 20


def test_status_and_frontend(client):
    s = client.get("/status").json()
    assert "ollamaAvailable" in s and s["demoDims"] == 16
    assert "<canvas id=\"scatter\">" in client.get("/").text
    assert client.get("/js/main.js").status_code == 200
