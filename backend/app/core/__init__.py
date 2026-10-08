"""Core primitives: data models and distance metrics."""
from .distance import DistFn, cosine, euclidean, get_dist_fn, manhattan
from .models import DocItem, VectorItem

__all__ = [
    "DistFn",
    "DocItem",
    "VectorItem",
    "cosine",
    "euclidean",
    "get_dist_fn",
    "manhattan",
]
