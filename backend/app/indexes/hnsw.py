"""HNSW — Hierarchical Navigable Small World graph (approximate k-NN).

Same algorithm family used by Pinecone, Weaviate, Chroma and Milvus.
Nodes live in a multilayer graph; upper layers are exponentially sparser and act
as "highways" so search can zoom in to the right neighbourhood in ~O(log N).
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..core import DistFn, VectorItem

Hits = list[tuple[float, int]]


@dataclass
class _Node:
    item: VectorItem
    max_lyr: int
    nbrs: list[list[int]]


@dataclass
class GraphInfo:
    top_layer: int
    node_count: int
    nodes_per_layer: list[int] = field(default_factory=list)
    edges_per_layer: list[int] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self._rng = random.Random(seed)
        self.G: dict[int, _Node] = {}

    # ── helpers ───────────────────────────────────────────────────
    def _rand_level(self) -> int:
        # 1 - random() is in (0, 1] so log() never sees 0.
        return int(math.floor(-math.log(1.0 - self._rng.random()) * self.mL))

    def _search_layer(self, q: np.ndarray, ep: int, ef: int, lyr: int, dist: DistFn) -> Hits:
        """Beam search on a single layer; returns up to `ef` hits sorted ascending."""
        G = self.G
        visited = {ep}
        d0 = dist(q, G[ep].item.emb)
        cands: list[tuple[float, int]] = [(d0, ep)]    # min-heap
        found: list[tuple[float, int]] = [(-d0, ep)]   # max-heap (negated)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = G[cid]
            if lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in G:
                    continue
                visited.add(nid)
                nd = dist(q, G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, i) for d, i in found)

    @staticmethod
    def _select_nbrs(cands: Hits, max_m: int) -> list[int]:
        return [i for _, i in cands[:max_m]]

    # ── public API ────────────────────────────────────────────────
    def insert(self, item: VectorItem, dist: DistFn) -> None:
        G = self.G
        nid_new = item.id
        lvl = self._rand_level()
        G[nid_new] = _Node(item, lvl, [[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt, self.top_layer = nid_new, lvl
            return

        ep = self.entry_pt
        # Greedy descent through layers above the new node's level.
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        # Connect on every layer from min(top, lvl) down to 0.
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            G[nid_new].nbrs[lc] = sel

            for nid in sel:
                if nid not in G:
                    continue
                nb = G[nid]
                while len(nb.nbrs) <= lc:
                    nb.nbrs.append([])
                conn = nb.nbrs[lc]
                conn.append(nid_new)
                if len(conn) > max_m:
                    # Shrink: keep the max_m closest neighbours.
                    ds = sorted((dist(nb.item.emb, G[c].item.emb), c) for c in conn if c in G)
                    nb.nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer, self.entry_pt = lvl, nid_new

    def knn(self, q: np.ndarray, k: int, ef: int, dist: DistFn) -> Hits:
        if self.entry_pt == -1:
            return []
        G = self.G
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        return self._search_layer(q, ep, max(ef, k), 0, dist)[:k]

    def remove(self, item_id: int) -> None:
        G = self.G
        if item_id not in G:
            return
        for node in G.values():
            for i, layer in enumerate(node.nbrs):
                if item_id in layer:
                    node.nbrs[i] = [n for n in layer if n != item_id]
        del G[item_id]

        if self.entry_pt == item_id:
            if G:
                # Promote the remaining node with the highest layer.
                new_ep = max(G.values(), key=lambda n: n.max_lyr)
                self.entry_pt, self.top_layer = new_ep.item.id, new_ep.max_lyr
            else:
                self.entry_pt, self.top_layer = -1, -1

    def get_info(self) -> GraphInfo:
        max_l = max(self.top_layer + 1, 1)
        gi = GraphInfo(
            top_layer=self.top_layer,
            node_count=len(self.G),
            nodes_per_layer=[0] * max_l,
            edges_per_layer=[0] * max_l,
        )
        for nid, nd in self.G.items():
            gi.nodes.append({
                "id": nid,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                gi.nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for other in nd.nbrs[lc]:
                        if nid < other:
                            gi.edges_per_layer[lc] += 1
                            gi.edges.append({"src": nid, "dst": other, "lyr": lc})
        return gi

    def __len__(self) -> int:
        return len(self.G)
