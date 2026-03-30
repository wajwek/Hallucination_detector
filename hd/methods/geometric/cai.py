from __future__ import annotations

import numpy as np

from ..base import BaseMetric, MetricOutput, cosine_similarity_rows, cosine_similarity_vec
from ...registry import register_metric


@register_metric("cai")
class CAIMetric(BaseMetric):
    name = "cai"

    def __init__(self, threshold: float = 0.1):
        self.threshold = float(threshold)

    def compute(self, *, q: np.ndarray, c: np.ndarray, a: np.ndarray, gt: np.ndarray | None, centroid: np.ndarray, **kwargs):
        cos_a_c = cosine_similarity_rows(a, c)
        cos_a_centroid = cosine_similarity_vec(a, centroid)

        dist_a_c = 1.0 - cos_a_c
        dist_a_centroid = 1.0 - cos_a_centroid

        cai = dist_a_centroid - dist_a_c
        flags = cai < self.threshold
        return MetricOutput(name=self.name, scores=cai.astype(np.float32), flags=flags.astype(bool))
