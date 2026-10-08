"""VectorDB — unified interface over Brute Force, KD-Tree and HNSW (demo 16D index)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core import DistFn, VectorItem, get_dist_fn
from ..indexes import HNSW, BruteForce, GraphInfo, KDTree


@dataclass
class Hit:
    id: int
    metadata: str
    category: str
    emb: np.ndarray
    distance: float


@dataclass
class SearchOut:
    hits: list[Hit] = field(default_factory=list)
    latency_us: int = 0
    algo: str = "hnsw"
    metric: str = "cosine"


@dataclass
class BenchOut:
    bruteforce_us: int
    kdtree_us: int
    hnsw_us: int
    item_count: int


def _elapsed_us(fn: Callable[[], object]) -> int:
    t0 = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t0) // 1000


class VectorDB:
    ALGORITHMS = ("bruteforce", "kdtree", "hnsw")

    def __init__(self, dims: int, hnsw_m: int = 16, ef_build: int = 200, ef_search: int = 50) -> None:
        self.dims = dims
        self.ef_search = ef_search
        self._store: dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(hnsw_m, ef_build)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: Sequence[float],
               dist: DistFn | None = None) -> int:
        dist = dist or get_dist_fn("cosine")
        with self._lock:
            item = VectorItem(self._next_id, metadata, category, np.asarray(emb, dtype=np.float32))
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            self._kdt.rebuild(self._store.values())
            return True

    def _run(self, algo: str, q: np.ndarray, k: int, dfn: DistFn) -> list[tuple[float, int]]:
        if algo == "bruteforce":
            return self._bf.knn(q, k, dfn)
        if algo == "kdtree":
            return self._kdt.knn(q, k, dfn)
        return self._hnsw.knn(q, k, self.ef_search, dfn)

    def search(self, q: Sequence[float], k: int, metric: str, algo: str) -> SearchOut:
        qv = np.asarray(q, dtype=np.float32)
        dfn = get_dist_fn(metric)
        with self._lock:
            t0 = time.perf_counter_ns()
            raw = self._run(algo, qv, k, dfn)
            us = (time.perf_counter_ns() - t0) // 1000
            out = SearchOut(latency_us=us, algo=algo, metric=metric)
            for d, i in raw:
                item = self._store.get(i)
                if item is not None:
                    out.hits.append(Hit(i, item.metadata, item.category, item.emb, d))
            return out

    def benchmark(self, q: Sequence[float], k: int, metric: str) -> BenchOut:
        qv = np.asarray(q, dtype=np.float32)
        dfn = get_dist_fn(metric)
        with self._lock:
            return BenchOut(
                bruteforce_us=_elapsed_us(lambda: self._bf.knn(qv, k, dfn)),
                kdtree_us=_elapsed_us(lambda: self._kdt.knn(qv, k, dfn)),
                hnsw_us=_elapsed_us(lambda: self._hnsw.knn(qv, k, self.ef_search, dfn)),
                item_count=len(self._store),
            )

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> GraphInfo:
        with self._lock:
            return self._hnsw.get_info()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
