"""Plain data containers shared by indexes and databases."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: np.ndarray


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: np.ndarray
