from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def pool_vectors(vectors: np.ndarray, mode: str, weights: np.ndarray | None = None) -> np.ndarray:
    if vectors.size == 0:
        raise ValueError("Cannot pool empty vectors")

    if mode == "mean":
        return vectors.mean(axis=0)
    if mode == "max":
        return vectors.max(axis=0)
    if mode == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires weights")
        w = weights.astype(np.float32)
        w = w / (w.sum() + 1e-12)
        return (vectors * w[:, None]).sum(axis=0)

    raise ValueError(f"Unknown pooling mode: {mode}")
