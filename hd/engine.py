from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .aggregation import aggregate_hard, aggregate_score_based
from .config import RunConfig
from .data import load_dataset
from .embeddings import load_or_compute_embeddings
from .eval import auroc_if_possible, classification_report_basic
from .registry import REGISTRY
from .storage import (
    artifacts_root,
    compute_dataset_id,
    compute_embed_config_id,
    compute_run_id,
    run_paths,
    write_dataset_bundle,
)
from .utils import write_json

# Ensure metric classes are registered
from .methods.geometric import cai as _cai  # noqa: F401
from .methods.geometric import cluster as _cluster  # noqa: F401
from .methods.geometric import dc as _dc  # noqa: F401
from .methods.geometric import sgi as _sgi  # noqa: F401


ProgressCallback = Callable[[float, str], None]


def _progress(cb: ProgressCallback | None, frac: float, message: str) -> None:
    if cb is None:
        return
    f = max(0.0, min(1.0, float(frac)))
    cb(f, str(message))


def _extension_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".csv", ".json", ".jsonl", ".jsonlines"}:
        return ext
    raise ValueError(f"Unsupported dataset extension: {ext}")


def run_detection(
    *,
    dataset_path: str | Path,
    run_config: RunConfig,
    artifacts_dir: Optional[str | Path] = None,
    force_recompute_embeddings: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    root = Path(artifacts_dir) if artifacts_dir else artifacts_root()
    root.mkdir(parents=True, exist_ok=True)

    _progress(progress_cb, 0.02, "Loading dataset")
    df, meta = load_dataset(dataset_path, run_config.dataset_columns)
    notes: List[str] = []

    # Validate ground_truth usage early (before expensive embeddings).
    gt_series = df.get("ground_truth")
    if gt_series is not None:
        gt_missing = gt_series.isna() | (
            gt_series.astype(str).str.strip().str.lower().isin({"", "none", "nan", "null"})
        )
        gt_any_present = not bool(gt_missing.all())
    else:
        gt_missing = None
        gt_any_present = False

    if run_config.dc.enabled:
        missing_n = int(gt_missing.sum()) if gt_missing is not None else int(len(df))
        if missing_n > 0:
            mapped = run_config.dataset_columns.ground_truth
            raise ValueError(
                "DC is enabled but ground_truth is missing/empty for "
                f"{missing_n}/{len(df)} rows. Provide a complete ground-truth column and map it via "
                f"dataset_columns.ground_truth (currently: {mapped!r})."
            )
    elif gt_any_present:
        notes.append(
            "ground_truth is present in the dataset, but DC is disabled, so ground truth is not used in this run."
        )

    _progress(progress_cb, 0.06, "Computing IDs")
    dataset_id = compute_dataset_id(dataset_path, meta["columns_mapping"])
    embed_config_id = compute_embed_config_id(run_config.embeddings.model_dump())

    _progress(progress_cb, 0.08, "Saving dataset bundle")
    write_dataset_bundle(
        root,
        dataset_id,
        dataset_path,
        extension=_extension_for(dataset_path),
        meta={**meta, "dataset_id": dataset_id},
    )

    def emb_cb(frac: float, msg: str) -> None:
        # map embeddings progress into [0.10, 0.70]
        _progress(progress_cb, 0.10 + 0.60 * frac, msg)

    _progress(progress_cb, 0.10, "Embeddings")
    emb = load_or_compute_embeddings(
        run_config,
        dataset_id,
        embed_config_id,
        root,
        df,
        force_recompute=bool(force_recompute_embeddings),
        progress_cb=emb_cb,
    )

    enabled = run_config.enabled_methods()

    _progress(progress_cb, 0.70, "Computing metrics")

    scores_by_metric: Dict[str, np.ndarray] = {}
    flags_by_metric: Dict[str, np.ndarray] = {}

    metric_span = 0.18
    per_metric = metric_span / len(enabled) if enabled else metric_span

    for idx, name in enumerate(enabled):
        m_start = 0.70 + idx * per_metric
        m_end = 0.70 + (idx + 1) * per_metric

        _progress(progress_cb, m_start, f"Metric: {name}")

        if name == "sgi":
            metric = REGISTRY.create(
                "sgi",
                threshold=run_config.sgi.threshold,
                variant=getattr(run_config.sgi, "variant", "cosine_diff_norm"),
                epsilon=getattr(run_config.sgi, "epsilon", 1e-8),
            )
            out = metric.compute(q=emb.q, c=emb.c, a=emb.a, gt=emb.gt, centroid=emb.centroid)
        elif name == "dc":
            metric = REGISTRY.create("dc", threshold=run_config.dc.threshold)
            out = metric.compute(q=emb.q, c=emb.c, a=emb.a, gt=emb.gt, centroid=emb.centroid)
        elif name == "cai":
            metric = REGISTRY.create("cai", threshold=run_config.cai.threshold)
            out = metric.compute(q=emb.q, c=emb.c, a=emb.a, gt=emb.gt, centroid=emb.centroid)
        elif name == "cluster":
            metric = REGISTRY.create(
                "cluster",
                threshold=run_config.cluster.threshold,
                k_clusters=run_config.cluster.k_clusters,
                min_chunks=run_config.cluster.min_chunks,
            )

            def cluster_cb(frac: float, msg: str) -> None:
                _progress(progress_cb, m_start + (m_end - m_start) * frac, msg)

            out = metric.compute(
                q=emb.q,
                c=emb.c,
                a=emb.a,
                gt=emb.gt,
                centroid=emb.centroid,
                answer_chunks=emb.answer_chunks,
                progress_cb=cluster_cb,
            )
        else:
            metric = REGISTRY.create(name)
            out = metric.compute(
                q=emb.q,
                c=emb.c,
                a=emb.a,
                gt=emb.gt,
                centroid=emb.centroid,
                answer_chunks=emb.answer_chunks,
            )

        scores_by_metric[out.name] = out.scores
        flags_by_metric[out.name] = out.flags
        _progress(progress_cb, m_end, f"Metric done: {name}")

    _progress(progress_cb, 0.90, "Aggregating")
    hall_score: Optional[np.ndarray] = None
    if run_config.aggregation.kind.value == "hard":
        predicted = aggregate_hard(flags_by_metric)
    else:
        hall_score, predicted = aggregate_score_based(
            scores_by_metric,
            run_config.aggregation.weights,
            run_config.aggregation.threshold_hall,
        )

    _progress(progress_cb, 0.92, "Evaluating")
    y_true_list = df["label"].tolist()
    labeled_mask = np.array([x is not None for x in y_true_list], dtype=bool)

    summary_metrics: Dict[str, Any] = {}
    if labeled_mask.any():
        y_true_int = np.array(
            [int(x) for x in np.array(y_true_list, dtype=object)[labeled_mask]],
            dtype=int,
        )
        if len(set(y_true_int.tolist())) < 2:
            summary_metrics["warning"] = (
                "Only one label class present in this run (all 0 or all 1). "
                "Metrics can look artificially good; for HaluEval use the canonical conversion to include both classes."
            )
        y_pred_int = predicted[labeled_mask].astype(int)
        report = classification_report_basic(y_true_int, y_pred_int)
        summary_metrics.update(report)
        if hall_score is not None:
            auroc = auroc_if_possible(y_true_int, hall_score[labeled_mask])
            if auroc is not None:
                summary_metrics["roc_auc"] = float(auroc)

    _progress(progress_cb, 0.94, "Saving run")
    run_id = compute_run_id(run_config.run_prefix, run_config.model_dump())
    rp = run_paths(root, run_id)
    rp.run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        rp.run_config_path,
        {
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "dataset_id": dataset_id,
            "embed_config_id": embed_config_id,
            "enabled_methods": enabled,
            "force_recompute_embeddings": bool(force_recompute_embeddings),
            "config": run_config.model_dump(),
        },
    )

    _progress(progress_cb, 0.96, "Writing metrics")
    rows: List[Dict[str, Any]] = []
    for metric_name, scores in scores_by_metric.items():
        flags = flags_by_metric[metric_name]
        for i, sample_id in enumerate(df["sample_id"].tolist()):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "method": "geometric",
                    "metric_name": metric_name,
                    "score": float(scores[i]),
                    "flag": bool(flags[i]),
                }
            )

    pd.DataFrame(rows).to_csv(rp.metrics_long_path, index=False)

    _progress(progress_cb, 0.975, "Writing predictions")
    pred_df = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(int),
            "predicted_label": predicted.astype(int),
            "true_label": [int(x) if x is not None else None for x in y_true_list],
        }
    )
    if hall_score is not None:
        pred_df["hall_score"] = hall_score

    for metric_name, scores in scores_by_metric.items():
        pred_df[f"score_{metric_name}"] = scores
        pred_df[f"flag_{metric_name}"] = flags_by_metric[metric_name].astype(int)

    pred_df.to_csv(rp.predictions_path, index=False)

    _progress(progress_cb, 0.99, "Writing summary")
    write_json(
        rp.results_summary_path,
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "embed_config_id": embed_config_id,
            "mode": run_config.aggregation.kind.value,
            "used_methods": enabled,
            "thresholds": {
                "sgi": run_config.sgi.threshold,
                "sgi_variant": getattr(run_config.sgi, "variant", "cosine_diff_norm"),
                "sgi_epsilon": float(getattr(run_config.sgi, "epsilon", 1e-8)),
                "dc": run_config.dc.threshold,
                "cai": run_config.cai.threshold,
                "cluster": run_config.cluster.threshold,
                "threshold_hall": run_config.aggregation.threshold_hall,
            },
            "metrics": summary_metrics,
            "notes": notes,
        },
    )

    _progress(progress_cb, 1.0, "Done")

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "embed_config_id": embed_config_id,
        "paths": {
            "run_dir": str(rp.run_dir),
            "predictions": str(rp.predictions_path),
            "metrics_long": str(rp.metrics_long_path),
            "results_summary": str(rp.results_summary_path),
        },
        "summary": summary_metrics,
        "notes": notes,
    }
