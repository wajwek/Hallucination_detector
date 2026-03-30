from __future__ import annotations

import numpy as np

from ..base import BaseMetric, MetricOutput, cosine_similarity_rows
from ...registry import register_metric


@register_metric("sgi")
class SGIMetric(BaseMetric):
    name = "sgi"

    def __init__(
        self,
        threshold: float = 0.3,
        *,
        variant: str = "cosine_diff_norm",
        epsilon: float = 1e-8,
    ):
        self.threshold = float(threshold)
        self.variant = str(variant or "cosine_diff_norm")
        eps = float(epsilon)
        self.epsilon = eps if eps > 0 else 1e-12

    def compute(
        self,
        *,
        q: np.ndarray,
        c: np.ndarray,
        a: np.ndarray,
        gt: np.ndarray | None,
        centroid: np.ndarray,
        **kwargs,
    ) -> MetricOutput:
        if self.variant in {"angular_ratio", "angle_ratio", "theta_ratio"}:
            cos_rq = cosine_similarity_rows(a, q)
            cos_rc = cosine_similarity_rows(a, c)

            theta_rq = np.arccos(np.clip(cos_rq, -1.0, 1.0)).astype(np.float32)
            theta_rc = np.arccos(np.clip(cos_rc, -1.0, 1.0)).astype(np.float32)

            sgi = (theta_rq / (theta_rc + self.epsilon)).astype(np.float32)
        else:
            cos_ac = cosine_similarity_rows(a, c)
            cos_aq = cosine_similarity_rows(a, q)
            denom = cos_ac + cos_aq + self.epsilon
            sgi = ((cos_ac - cos_aq) / denom).astype(np.float32)

        flags = sgi < self.threshold
        return MetricOutput(name=self.name, scores=sgi, flags=flags.astype(bool))
