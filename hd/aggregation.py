from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def hall_component(metric_name: str, scores: np.ndarray) -> np.ndarray:
    # All outputs are mapped to [0,1] where 1 => more likely hallucination.
    if metric_name == "sgi":
        # higher SGI => more grounded => lower hall
        return 1.0 - _sigmoid(5.0 * scores)
    if metric_name == "dc":
        # higher DC => better => lower hall
        s = np.clip(scores, -1.0, 1.0)
        return (1.0 - s) / 2.0
    if metric_name == "cai":
        # higher CAI => better => lower hall
        return 1.0 - _sigmoid(5.0 * scores)
    if metric_name == "cluster":
        # already a distance-like score
        return np.clip(scores, 0.0, 1.0)

    # fallback: treat low as bad
    return 1.0 - _sigmoid(5.0 * scores)


def aggregate_hard(flags_by_metric: Dict[str, np.ndarray]) -> np.ndarray:
    if not flags_by_metric:
        raise ValueError("No flags to aggregate")
    keys = sorted(flags_by_metric)
    out = np.zeros_like(flags_by_metric[keys[0]], dtype=bool)
    for k in keys:
        out = np.logical_or(out, flags_by_metric[k].astype(bool))
    return out


def aggregate_score_based(
    scores_by_metric: Dict[str, np.ndarray],
    weights: Dict[str, float],
    threshold_hall: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not scores_by_metric:
        raise ValueError("No scores to aggregate")

    keys = sorted(scores_by_metric)
    wsum = 0.0
    acc = None
    for k in keys:
        w = float(weights.get(k, 1.0))
        comp = hall_component(k, scores_by_metric[k])
        if acc is None:
            acc = w * comp
        else:
            acc = acc + w * comp
        wsum += w

    assert acc is not None
    hall_score = acc / max(wsum, 1e-12)
    predicted = hall_score > float(threshold_hall)
    return hall_score.astype(np.float32), predicted.astype(bool)
