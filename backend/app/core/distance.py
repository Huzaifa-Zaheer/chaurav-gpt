"""Distance metrics.

All functions take two 1-D numpy float arrays of equal length and return a
Python float where *smaller means more similar*.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DistFn = Callable[[np.ndarray, np.ndarray], float]


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine *distance* (1 - cosine similarity)."""
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / (np.sqrt(na) * np.sqrt(nb))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


_METRICS: dict[str, DistFn] = {
    "euclidean": euclidean,
    "cosine": cosine,
    "manhattan": manhattan,
}

METRIC_NAMES = tuple(_METRICS)


def get_dist_fn(name: str) -> DistFn:
    """Return the metric by name, defaulting to euclidean (same as the C++ version)."""
    return _METRICS.get(name, euclidean)
