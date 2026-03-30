from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_report_basic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def auroc_if_possible(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        y_true = y_true.astype(int)
        if len(set(y_true.tolist())) < 2:
            return None
        return float(roc_auc_score(y_true, y_score.astype(float)))
    except Exception:
        return None
