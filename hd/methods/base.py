from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MetricOutput:
    name: str
    scores: np.ndarray  # shape (n,)
    flags: np.ndarray   # shape (n,) boolean
    extras: Optional[Dict[str, Any]] = None


class BaseMetric:
    name: str

    def compute(self, *, q: np.ndarray, c: np.ndarray, a: np.ndarray, gt: np.ndarray | None, centroid: np.ndarray, **kwargs: Any) -> MetricOutput:
        raise NotImplementedError


def cosine_similarity_rows(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x,y: (n,d)
    x2 = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    y2 = y / (np.linalg.norm(y, axis=1, keepdims=True) + eps)
    return (x2 * y2).sum(axis=1)


def cosine_similarity_vec(x: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (n,d), v: (d,)
    x2 = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    v2 = v / (np.linalg.norm(v) + eps)
    return (x2 * v2[None, :]).sum(axis=1)
