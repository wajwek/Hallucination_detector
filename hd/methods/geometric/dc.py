from __future__ import annotations

import numpy as np

from ..base import BaseMetric, MetricOutput, cosine_similarity_rows
from ...registry import register_metric


@register_metric("dc")
class DCMetric(BaseMetric):
    name = "dc"

    def __init__(self, threshold: float = 0.6):
        self.threshold = float(threshold)

    def compute(self, *, q: np.ndarray, c: np.ndarray, a: np.ndarray, gt: np.ndarray | None, centroid: np.ndarray, **kwargs):
        if gt is None:
            raise ValueError("DC requires ground-truth embeddings (gt)")

        dir_gt = gt - q
        dir_gen = a - q

        # normalize direction vectors
        eps = 1e-12
        dir_gt = dir_gt / (np.linalg.norm(dir_gt, axis=1, keepdims=True) + eps)
        dir_gen = dir_gen / (np.linalg.norm(dir_gen, axis=1, keepdims=True) + eps)

        dc = cosine_similarity_rows(dir_gen, dir_gt)
        flags = dc < self.threshold
        return MetricOutput(name=self.name, scores=dc.astype(np.float32), flags=flags.astype(bool))
