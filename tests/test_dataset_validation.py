from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hd.config import DatasetColumns
from hd.data.loader import load_dataset


def test_loader_requires_question_context_answer(tmp_path: Path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"q": ["a"], "c": ["b"], "a": ["c"]}).to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_dataset(p, DatasetColumns(question="question", context="context", answer="answer"))


def test_loader_with_mapping(tmp_path: Path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"q": ["What?"], "c": ["Ctx"], "a": ["Ans"], "lbl": [1]}).to_csv(p, index=False)

    df, meta = load_dataset(
        p,
        DatasetColumns(question="q", context="c", answer="a", label="lbl", ground_truth=None),
    )

    assert len(df) == 1
    assert df.loc[0, "question"] == "What?"
    assert df.loc[0, "context"] == ["Ctx"]
    assert df.loc[0, "label"] == 1
    assert meta["label_present"] is True


def test_loader_infers_constant_label_for_raw_halueval_when_label_missing(tmp_path: Path):
    p = tmp_path / "halueval_qa.jsonl"
    df0 = pd.DataFrame(
        {
            "knowledge": ["k"],
            "question": ["q"],
            "right_answer": ["ra"],
            "hallucinated_answer": ["ha"],
        }
    )
    df0.to_json(p, orient="records", lines=True)

    df, meta = load_dataset(
        p,
        DatasetColumns(
            question="question",
            context="knowledge",
            answer="hallucinated_answer",
            ground_truth="right_answer",
            label=None,
        ),
    )

    assert len(df) == 1
    assert df.loc[0, "label"] == 1
    assert meta["label_present"] is True
    assert meta["label_inferred"] is True


def test_loader_infers_label_from_answer_type_when_label_missing(tmp_path: Path):
    p = tmp_path / "x.csv"
    pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "context": ["c1", "c2"],
            "answer": ["a1", "a2"],
            "answer_type": ["hallucinated", "right"],
        }
    ).to_csv(p, index=False)

    df, meta = load_dataset(
        p,
        DatasetColumns(
            question="question",
            context="context",
            answer="answer",
            ground_truth=None,
            label=None,
        ),
    )

    assert df["label"].tolist() == [1, 0]
    assert meta["label_present"] is True
    assert meta["label_inferred"] is True