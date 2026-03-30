from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import DatasetColumns
from .converters import detect_halueval_task
from .schema import compute_stats, normalize_context_value


def _parse_context(v: Any) -> Any:
    # allow JSON list stored as string
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _coerce_label(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        iv = int(v)
        return 1 if iv != 0 else 0
    except Exception:
        s = str(v).strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return 1
        if s in {"false", "no", "n", "0"}:
            return 0
    return None


def _infer_halueval_label_if_possible(
    df: pd.DataFrame, columns: DatasetColumns
) -> tuple[Optional[pd.Series], Optional[str]]:
    """Infer labels for (raw-ish) HaluEval when `label` column is missing.

    If the user maps `answer` to a known HaluEval column name (e.g. `hallucinated_answer`),
    we can infer a constant label for every row. This keeps row count unchanged.

    For a full HaluEval evaluation (pos+neg rows), prefer `hd prepare-halueval` or the GUI
    conversion button.
    """

    answer_col = columns.answer

    pos_cols = {"hallucinated_answer", "hallucinated_response", "hallucinated_summary"}
    neg_cols = {"right_answer", "right_response", "right_summary"}

    if answer_col in pos_cols:
        try:
            task = detect_halueval_task(df) or "unknown"
        except Exception:
            task = "unknown"
        return pd.Series([1] * len(df)), f"halueval.{task}: answer={answer_col} -> label=1"

    if answer_col in neg_cols:
        try:
            task = detect_halueval_task(df) or "unknown"
        except Exception:
            task = "unknown"
        return pd.Series([0] * len(df)), f"halueval.{task}: answer={answer_col} -> label=0"
    return None, None

def _infer_label_from_answer_type_if_possible(df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str]]:
    """Infer labels from an `answer_type` column when present (canonical/HaluEval conversions).

    Expected values include: `hallucinated` -> 1 and `right`/`correct`/`non_hallucinated` -> 0.
    """

    if "answer_type" not in df.columns:
        return None, None

    def _map(v: Any) -> Optional[int]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip().lower()
        s2 = s.replace("-", "_").replace(" ", "_")
        s2 = "_".join([p for p in s2.split("_") if p])

        # handle negatives first (because they contain "halluc" substring)
        if s2 in {"right", "correct", "gold", "non_hallucinated", "nonhallucinated", "not_hallucinated", "non_hallucination", "not_hallucination"}:
            return 0
        if s2 in {"hallucinated", "hallucination", "hallucinated_answer", "hallucinated_response", "hallucinated_summary"}:
            return 1
        if "non" in s2 and "halluc" in s2:
            return 0
        if "halluc" in s2:
            return 1
        return None

    inferred = df["answer_type"].apply(_map)
    if inferred.notna().any():
        return inferred, "answer_type -> label"
    return None, None

def _read_json_flexible(path: Path) -> pd.DataFrame:
    # Accept both JSONL and a JSON array/object
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.read_json(path, lines=False)


def load_dataset(
    path: str | Path,
    columns: DatasetColumns | Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if columns is None:
        columns = DatasetColumns()
    if isinstance(columns, dict):
        columns = DatasetColumns(**columns)

    ext = p.suffix.lower()
    if ext in {".csv"}:
        df = pd.read_csv(p)
    elif ext in {".jsonl", ".jsonlines"}:
        df = pd.read_json(p, lines=True)
    elif ext in {".json"}:
        df = _read_json_flexible(p)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")

    required = {
        "question": columns.question,
        "context": columns.context,
        "answer": columns.answer,
    }
    missing = [k for k, col in required.items() if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {missing}. Available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["sample_id"] = range(len(df))
    out["question"] = df[columns.question].astype(str)

    raw_ctx = df[columns.context].apply(_parse_context)
    out["context"] = raw_ctx.apply(normalize_context_value)

    out["answer"] = df[columns.answer].astype(str)

    if columns.ground_truth and columns.ground_truth in df.columns:
        out["ground_truth"] = df[columns.ground_truth].astype(str)
    else:
        out["ground_truth"] = None

    label_note: Optional[str]
    if columns.label and columns.label in df.columns:
        out["label"] = df[columns.label].apply(_coerce_label)
        label_note = None
    else:
        inferred, label_note = _infer_halueval_label_if_possible(df, columns)
        if inferred is None:
            inferred, label_note = _infer_label_from_answer_type_if_possible(df)
        if inferred is not None:
            out["label"] = inferred.apply(_coerce_label)
        else:
            out["label"] = None

    stats = compute_stats(out["label"].tolist())

    meta: Dict[str, Any] = {
        "path": str(p),
        "extension": ext,
        "columns_mapping": columns.model_dump(),
        "n_samples": stats.n_samples,
        "label_present": stats.label_present,
        "label_pos": stats.label_pos,
        "label_neg": stats.label_neg,
    }

    if label_note:
        meta["label_inferred"] = True
        meta["label_inferred_rule"] = label_note
    else:
        meta["label_inferred"] = False

    return out, meta
