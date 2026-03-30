from __future__ import annotations

from pathlib import Path

import pandas as pd

from hd.storage import compute_dataset_id, compute_embed_config_id


def test_embed_config_id_stable():
    a = {"embedding_model": "all-MiniLM-L6-v2", "batch_size": 32, "normalize_to_unit_sphere": True, "reduction": {"kind": "none"}}
    b = {"batch_size": 32, "embedding_model": "all-MiniLM-L6-v2", "reduction": {"kind": "none"}, "normalize_to_unit_sphere": True}
    assert compute_embed_config_id(a) == compute_embed_config_id(b)


def test_dataset_id_changes_with_mapping(tmp_path: Path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"q": ["x"], "c": ["y"], "a": ["z"]}).to_csv(p, index=False)

    m1 = {"question": "q", "context": "c", "answer": "a"}
    m2 = {"question": "q", "context": "c", "answer": "a", "label": "lbl"}

    assert compute_dataset_id(p, m1) != compute_dataset_id(p, m2)
