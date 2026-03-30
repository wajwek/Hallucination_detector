from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from ..base import BaseMetric, MetricOutput, cosine_similarity_vec
from ...registry import register_metric


@register_metric("cluster")
class ClusterMetric(BaseMetric):
    name = "cluster"

    def __init__(
        self,
        threshold: float = 0.5,
        k_clusters: int = 5,
        min_chunks: int = 3,
    ):
        self.threshold = float(threshold)
        self.k_clusters = int(k_clusters)
        self.min_chunks = int(min_chunks)

    def compute(
        self,
        *,
        q: np.ndarray,
        c: np.ndarray,
        a: np.ndarray,
        gt: np.ndarray | None,
        centroid: np.ndarray,
        answer_chunks: Optional[List[np.ndarray]] = None,
        progress_cb: Callable[[float, str], None] | None = None,
        **kwargs,
    ) -> MetricOutput:
        if answer_chunks is None:
            raise ValueError("Cluster metric requires answer_chunks embeddings")

        n = c.shape[0]
        scores = np.zeros((n,), dtype=np.float32)
        flags = np.zeros((n,), dtype=bool)

        step = max(1, n // 100)

        for i in range(n):
            if progress_cb is not None and (i % step == 0 or i == n - 1):
                progress_cb((i + 1) / n, f"Cluster progress: {i + 1}/{n}")

            chunks = answer_chunks[i]
            if chunks is None or chunks.size == 0 or chunks.shape[0] < self.min_chunks:
                scores[i] = 0.0
                flags[i] = False
                continue

            k = min(self.k_clusters, chunks.shape[0])
            if k <= 1:
                scores[i] = 0.0
                flags[i] = False
                continue

            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            km.fit(chunks)
            centroids = km.cluster_centers_.astype(np.float32)

            # Distance of chunk-centroids to context vector (1 - cosine)
            cos = cosine_similarity_vec(centroids, c[i])
            dist = 1.0 - cos

            # Score = mean distance, clipped to [0,1]
            s = float(np.clip(dist.mean(), 0.0, 1.0))
            scores[i] = s
            flags[i] = s > self.threshold

        return MetricOutput(name=self.name, scores=scores, flags=flags)
