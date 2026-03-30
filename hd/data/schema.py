from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_COLUMNS = ["sample_id", "question", "context", "answer", "ground_truth", "label"]


@dataclass(frozen=True)
class DatasetStats:
    n_samples: int
    label_present: bool
    label_pos: Optional[int]
    label_neg: Optional[int]


def normalize_context_value(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [s]


def compute_stats(labels: List[Optional[int]]) -> DatasetStats:
    present = any(x is not None for x in labels)
    if not present:
        return DatasetStats(n_samples=len(labels), label_present=False, label_pos=None, label_neg=None)
    pos = sum(1 for x in labels if x == 1)
    neg = sum(1 for x in labels if x == 0)
    return DatasetStats(n_samples=len(labels), label_present=True, label_pos=pos, label_neg=neg)
