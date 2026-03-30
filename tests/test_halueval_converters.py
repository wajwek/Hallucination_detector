from __future__ import annotations

import pandas as pd

from hd.data.converters import convert_halueval_to_canonical, detect_halueval_task


def test_detect_qa():
    df = pd.DataFrame(
        {
            "knowledge": ["k"],
            "question": ["q"],
            "right_answer": ["ra"],
            "hallucinated_answer": ["ha"],
        }
    )
    assert detect_halueval_task(df) == "qa"


def test_convert_qa_doubles_rows_when_include_right():
    df = pd.DataFrame(
        {
            "knowledge": ["k"],
            "question": ["q"],
            "right_answer": ["ra"],
            "hallucinated_answer": ["ha"],
        }
    )
    out = convert_halueval_to_canonical(df, task="qa", include_right=True)
    assert len(out) == 2
    assert set(out["label"].tolist()) == {0, 1}
    assert (out["ground_truth"] == "ra").all()


def test_convert_dialogue_fields():
    df = pd.DataFrame(
        {
            "knowledge": ["k"],
            "dialogue_history": ["hist"],
            "right_response": ["rr"],
            "hallucinated_response": ["hr"],
        }
    )
    out = convert_halueval_to_canonical(df, task="dialogue", include_right=False)
    assert len(out) == 1
    assert out.loc[0, "question"] == "hist"
    assert out.loc[0, "context"] == "k"
    assert out.loc[0, "ground_truth"] == "rr"
    assert out.loc[0, "label"] == 1
