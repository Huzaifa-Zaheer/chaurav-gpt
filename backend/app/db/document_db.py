"""DocumentDB — HNSW index over real Ollama embeddings (dimension set at runtime)."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from ..core import DocItem, VectorItem, cosine
from ..indexes import HNSW, BruteForce

# Below this many chunks, exact brute force is both faster and more accurate.
_BRUTE_FORCE_THRESHOLD = 10


class DocumentDB:
    def __init__(self, hnsw_m: int = 16, ef_build: int = 200, ef_search: int = 50) -> None:
        self.ef_search = ef_search
        self._store: dict[int, DocItem] = {}
        self._hnsw = HNSW(hnsw_m, ef_build)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self.dims = 0  # determined from the first inserted embedding

    def insert(self, title: str, text: str, emb: Sequence[float]) -> int:
        vec = np.asarray(emb, dtype=np.float32)
        with self._lock:
            if self.dims == 0:
                self.dims = int(vec.shape[0])
            item = DocItem(self._next_id, title, text, vec)
            self._next_id += 1
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", vec)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: Sequence[float], k: int, max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        """Semantic search — top-k chunks whose cosine distance is <= max_dist."""
        qv = np.asarray(q, dtype=np.float32)
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < _BRUTE_FORCE_THRESHOLD:
                raw = self._bf.knn(qv, k, cosine)
            else:
                raw = self._hnsw.knn(qv, k, self.ef_search, cosine)
            return [(d, self._store[i]) for d, i in raw if i in self._store and d <= max_dist]

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bf.remove(item_id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
