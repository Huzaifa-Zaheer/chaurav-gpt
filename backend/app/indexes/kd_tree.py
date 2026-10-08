"""KD-Tree — exact k-NN via axis-aligned binary space partitioning.

Works well for low dimensions (<= ~20D); degrades toward brute force in high
dimensions because the pruning bound rarely eliminates subtrees.
"""
from __future__ import annotations

import heapq
from typing import Iterable, Optional

import numpy as np

from ..core import DistFn, VectorItem

Hits = list[tuple[float, int]]


class _Node:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[_Node] = None

    # ── insert ────────────────────────────────────────────────────
    def insert(self, item: VectorItem) -> None:
        new = _Node(item)
        if self.root is None:
            self.root = new
            return
        node, depth = self.root, 0
        while True:
            ax = depth % self.dims
            if item.emb[ax] < node.item.emb[ax]:
                if node.left is None:
                    node.left = new
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = new
                    return
                node = node.right
            depth += 1

    def rebuild(self, items: Iterable[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)

    # ── search ────────────────────────────────────────────────────
    def knn(self, q: np.ndarray, k: int, dist: DistFn) -> Hits:
        # Max-heap of (−distance, id) holding the best k so far.
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-d, i) for d, i in heap)

    def _knn(self, node: Optional[_Node], q: np.ndarray, k: int, depth: int,
             dist: DistFn, heap: list[tuple[float, int]]) -> None:
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax = depth % self.dims
        diff = float(q[ax] - node.item.emb[ax])
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)
