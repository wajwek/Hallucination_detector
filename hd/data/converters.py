from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


HaluEvalTask = Literal["qa", "dialogue", "summarization", "general"]


@dataclass(frozen=True)
class ConvertResult:
    task: HaluEvalTask
    rows: int


def detect_halueval_task(df: pd.DataFrame) -> Optional[HaluEvalTask]:
    cols = set(df.columns)

    if {"knowledge", "question", "right_answer", "hallucinated_answer"}.issubset(cols):
        return "qa"
    if {"knowledge", "dialogue_history", "right_response", "hallucinated_response"}.issubset(cols):
        return "dialogue"
    if {"document", "right_summary", "hallucinated_summary"}.issubset(cols):
        return "summarization"
    if {"user_query", "chatgpt_response", "hallucination_label"}.issubset(cols):
        return "general"

    return None


def _label_yes_no(v) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().lower()
    s2 = s.replace("-", "_").replace(" ", "_")
    s2 = "_".join([p for p in s2.split("_") if p])

    if s2 in {"yes", "y", "true", "1", "hallucinated", "hallucination", "hallucinate"}:
        return 1
    if s2 in {"no", "n", "false", "0", "right", "correct", "non_hallucinated", "nonhallucinated", "not_hallucinated", "non_hallucination", "not_hallucination"}:
        return 0

    # Heuristic fallbacks for common strings
    if "non" in s2 and "halluc" in s2:
        return 0
    if "halluc" in s2:
        return 1

    return None

def _read_json_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.read_json(path, lines=False)


def load_halueval(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".jsonlines"}:
        return pd.read_json(p, lines=True)
    if p.suffix.lower() == ".json":
        return _read_json_flexible(p)
    raise ValueError(f"Unsupported extension for HaluEval: {p.suffix}")


def convert_halueval_to_canonical(
    df: pd.DataFrame,
    *,
    task: HaluEvalTask,
    include_right: bool = True,
    summarization_question: str = "Summarize the following document.",
) -> pd.DataFrame:
    # Output schema: question, context, answer, ground_truth, label (+ extras)

    if task == "qa":
        base = pd.DataFrame(
            {
                "question": df["question"].astype(str),
                "context": df["knowledge"].astype(str),
                "ground_truth": df["right_answer"].astype(str),
            }
        )

        pos = base.copy()
        pos["answer"] = df["hallucinated_answer"].astype(str)
        pos["label"] = 1
        pos["source_task"] = "qa"
        pos["answer_type"] = "hallucinated"

        out = [pos]
        if include_right:
            neg = base.copy()
            neg["answer"] = df["right_answer"].astype(str)
            neg["label"] = 0
            neg["source_task"] = "qa"
            neg["answer_type"] = "right"
            out.append(neg)

        return pd.concat(out, axis=0, ignore_index=True)

    if task == "dialogue":
        base = pd.DataFrame(
            {
                "question": df["dialogue_history"].astype(str),
                "context": df["knowledge"].astype(str),
                "ground_truth": df["right_response"].astype(str),
            }
        )

        pos = base.copy()
        pos["answer"] = df["hallucinated_response"].astype(str)
        pos["label"] = 1
        pos["source_task"] = "dialogue"
        pos["answer_type"] = "hallucinated"

        out = [pos]
        if include_right:
            neg = base.copy()
            neg["answer"] = df["right_response"].astype(str)
            neg["label"] = 0
            neg["source_task"] = "dialogue"
            neg["answer_type"] = "right"
            out.append(neg)

        return pd.concat(out, axis=0, ignore_index=True)

    if task == "summarization":
        base = pd.DataFrame(
            {
                "question": summarization_question,
                "context": df["document"].astype(str),
                "ground_truth": df["right_summary"].astype(str),
            }
        )

        pos = base.copy()
        pos["answer"] = df["hallucinated_summary"].astype(str)
        pos["label"] = 1
        pos["source_task"] = "summarization"
        pos["answer_type"] = "hallucinated"

        out = [pos]
        if include_right:
            neg = base.copy()
            neg["answer"] = df["right_summary"].astype(str)
            neg["label"] = 0
            neg["source_task"] = "summarization"
            neg["answer_type"] = "right"
            out.append(neg)

        return pd.concat(out, axis=0, ignore_index=True)

    if task == "general":
        # Not a true RAG dataset (no external context). We map it so the pipeline can run.
        out = pd.DataFrame(
            {
                "question": df["user_query"].astype(str),
                "context": df["user_query"].astype(str),
                "answer": df["chatgpt_response"].astype(str),
                "ground_truth": "",
                "label": df["hallucination_label"].apply(_label_yes_no),
                "source_task": "general",
                "answer_type": "chatgpt",
            }
        )
        return out

    raise ValueError(f"Unknown task: {task}")


def write_canonical(df: pd.DataFrame, output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".csv":
        df.to_csv(p, index=False)
    elif ext in {".jsonl", ".jsonlines"}:
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    elif ext == ".json":
        # Prefer JSONL for size; keep extension if user asked for .json
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")
