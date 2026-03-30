from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import json_dumps_stable, read_json, sha256_bytes, sha256_file, sha256_text, write_json


def artifacts_root() -> Path:
    env = os.environ.get("HD_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "artifacts"


def compute_dataset_id(dataset_path: str | Path, columns_mapping: Dict[str, Any]) -> str:
    file_hash = sha256_file(dataset_path)
    mapping_hash = sha256_text(json_dumps_stable(columns_mapping))
    return sha256_text(file_hash + ":" + mapping_hash)[:24]


def compute_embed_config_id(embed_config: Dict[str, Any]) -> str:
    return sha256_text(json_dumps_stable(embed_config))[:24]


def compute_run_config_hash(run_config: Dict[str, Any]) -> str:
    return sha256_text(json_dumps_stable(run_config))[:12]


def compute_run_id(prefix: str, run_config: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = compute_run_config_hash(run_config)
    safe_prefix = (prefix or "run").strip().replace(" ", "_")
    return f"{safe_prefix}_{ts}_{h}"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    raw_path: Path
    meta_path: Path


@dataclass(frozen=True)
class EmbeddingPaths:
    embed_dir: Path
    config_path: Path
    q_path: Path
    c_path: Path
    a_path: Path
    gt_path: Path
    centroid_path: Path
    reduction_model_path: Path
    chunk_cache_path: Path
    chunk_meta_path: Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    run_config_path: Path
    metrics_long_path: Path
    predictions_path: Path
    results_summary_path: Path


def dataset_paths(root: Path, dataset_id: str, extension: str) -> DatasetPaths:
    ddir = root / "datasets" / dataset_id
    return DatasetPaths(
        dataset_dir=ddir,
        raw_path=ddir / f"raw{extension}",
        meta_path=ddir / "dataset_meta.json",
    )


def embedding_paths(root: Path, dataset_id: str, embed_config_id: str) -> EmbeddingPaths:
    edir = root / "embeddings" / dataset_id / embed_config_id
    return EmbeddingPaths(
        embed_dir=edir,
        config_path=edir / "embed_config.json",
        q_path=edir / "q.npy",
        c_path=edir / "c.npy",
        a_path=edir / "a.npy",
        gt_path=edir / "gt.npy",
        centroid_path=edir / "centroid.npy",
        reduction_model_path=edir / "reduction_model.joblib",
        chunk_cache_path=edir / "answer_chunks.joblib",
        chunk_meta_path=edir / "answer_chunks_meta.json",
    )


def run_paths(root: Path, run_id: str) -> RunPaths:
    rdir = root / "runs" / run_id
    return RunPaths(
        run_dir=rdir,
        run_config_path=rdir / "run_config.json",
        metrics_long_path=rdir / "metrics_long.csv",
        predictions_path=rdir / "predictions.csv",
        results_summary_path=rdir / "results_summary.json",
    )


def list_runs(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = root or artifacts_root()
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []

    out: List[Dict[str, Any]] = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        summary = d / "results_summary.json"
        cfg = d / "run_config.json"
        rec: Dict[str, Any] = {"run_id": d.name, "path": str(d)}
        if summary.exists():
            try:
                rec.update({"summary": read_json(summary)})
            except Exception:
                rec.update({"summary": None})
        if cfg.exists():
            try:
                rec.update({"config": read_json(cfg)})
            except Exception:
                rec.update({"config": None})
        out.append(rec)
    return out


def write_dataset_bundle(
    root: Path,
    dataset_id: str,
    src_path: Path,
    extension: str,
    meta: Dict[str, Any],
) -> DatasetPaths:
    dp = dataset_paths(root, dataset_id, extension)
    dp.dataset_dir.mkdir(parents=True, exist_ok=True)
    dp.raw_path.write_bytes(src_path.read_bytes())
    write_json(dp.meta_path, meta)
    return dp
