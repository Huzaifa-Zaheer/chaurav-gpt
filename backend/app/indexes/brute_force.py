"""Brute-force exact k-NN — O(N·d) baseline."""
from __future__ import annotations

import numpy as np

from ..core import DistFn, VectorItem

Hits = list[tuple[float, int]]


class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]

    def knn(self, q: np.ndarray, k: int, dist: DistFn) -> Hits:
        scored = sorted((dist(q, v.emb), v.id) for v in self.items)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.items)
