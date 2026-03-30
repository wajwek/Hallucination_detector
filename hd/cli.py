from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import (
    AggregationConfig,
    AggregationKind,
    DatasetColumns,
    EmbeddingConfig,
    PoolingMode,
    ReductionConfig,
    ReductionKind,
    RunConfig,
)
from .data.converters import (
    convert_halueval_to_canonical,
    detect_halueval_task,
    load_halueval,
    write_canonical,
)
from .engine import run_detection
from .storage import list_runs

app = typer.Typer(add_completion=False, help="Hallucination detector (RAG) with geometric methods.")


@app.command("prepare-halueval")
def prepare_halueval(
    input_path: Path = typer.Option(..., exists=True, help="Path to HaluEval JSON/JSONL."),
    output_path: Path = typer.Option(..., help="Output CSV/JSONL path (canonical schema)."),
    task: str = typer.Option("auto", help="qa|dialogue|summarization|general|auto"),
    include_right_answers: bool = typer.Option(
        True, help="If true, add non-hallucinated rows (label=0) where available."
    ),
):
    df = load_halueval(input_path)

    if task == "auto":
        detected = detect_halueval_task(df)
        if detected is None:
            raise typer.BadParameter(f"Cannot auto-detect HaluEval task. Columns: {list(df.columns)}")
        task2 = detected
    else:
        task2 = task

    out_df = convert_halueval_to_canonical(
        df,
        task=task2,  # type: ignore[arg-type]
        include_right=include_right_answers,
    )

    write_canonical(out_df, output_path)
    print({"task": task2, "rows": int(len(out_df)), "output": str(output_path)})


@app.command()
def run(
    dataset: Path = typer.Option(..., exists=True, help="Path to CSV/JSON/JSONL dataset."),
    run_prefix: str = typer.Option("run", help="Prefix for run_id."),
    force_recompute_embeddings: bool = typer.Option(False, help="Recompute embeddings even if cached."),
    question_col: str = typer.Option("question"),
    context_col: str = typer.Option("context"),
    answer_col: str = typer.Option("answer"),
    ground_truth_col: Optional[str] = typer.Option("ground_truth"),
    label_col: Optional[str] = typer.Option("label"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2"),
    batch_size: int = typer.Option(32),
    question_pooling: PoolingMode = typer.Option(PoolingMode.mean),
    context_pooling: PoolingMode = typer.Option(PoolingMode.mean),
    answer_pooling: PoolingMode = typer.Option(PoolingMode.mean),
    normalize: bool = typer.Option(True, help="L2 normalize vectors"),
    reduction: ReductionKind = typer.Option(ReductionKind.none),
    n_components: int = typer.Option(128),
    whitening: bool = typer.Option(False),
    enable_sgi: bool = typer.Option(True),
    sgi_threshold: float = typer.Option(0.3),
    enable_dc: bool = typer.Option(False),
    dc_threshold: float = typer.Option(0.6),
    enable_cai: bool = typer.Option(False),
    cai_threshold: float = typer.Option(0.1),
    enable_cluster: bool = typer.Option(False),
    cluster_threshold: float = typer.Option(0.5),
    k_clusters: int = typer.Option(5),
    agg: AggregationKind = typer.Option(AggregationKind.hard),
    threshold_hall: float = typer.Option(0.5),
):
    columns = DatasetColumns(
        question=question_col,
        context=context_col,
        answer=answer_col,
        ground_truth=ground_truth_col,
        label=label_col,
    )

    emb = EmbeddingConfig(
        embedding_model=embedding_model,
        batch_size=batch_size,
        question_pooling=question_pooling,
        context_pooling=context_pooling,
        answer_pooling=answer_pooling,
        normalize_to_unit_sphere=normalize,
        reduction=ReductionConfig(kind=reduction, n_components=n_components, whitening=whitening),
    )

    cfg = RunConfig(
        dataset_columns=columns,
        embeddings=emb,
        run_prefix=run_prefix,
    )

    cfg.sgi.enabled = bool(enable_sgi)
    cfg.sgi.threshold = float(sgi_threshold)

    cfg.dc.enabled = bool(enable_dc)
    cfg.dc.threshold = float(dc_threshold)

    if cfg.dc.enabled:
        if ground_truth_col is None or str(ground_truth_col).strip() == "":
            raise typer.BadParameter(
                "DC is enabled -> you must provide a non-empty --ground-truth-col (and the dataset must contain that column with non-empty values)."
            )

    cfg.cai.enabled = bool(enable_cai)
    cfg.cai.threshold = float(cai_threshold)

    cfg.cluster.enabled = bool(enable_cluster)
    cfg.cluster.threshold = float(cluster_threshold)
    cfg.cluster.k_clusters = int(k_clusters)

    cfg.aggregation = AggregationConfig(kind=agg, threshold_hall=float(threshold_hall))

    res = run_detection(
        dataset_path=dataset,
        run_config=cfg,
        force_recompute_embeddings=bool(force_recompute_embeddings),
    )
    print(res)

    notes = res.get("notes") or []
    for n in notes:
        print(f"[yellow]{n}[/yellow]")


@app.command()
def history(limit: int = typer.Option(25)):
    def fmt(v) -> str:
        return "n/a" if v is None else str(v)

    runs = list_runs()
    for r in runs[: int(limit)]:
        summary = r.get("summary") or {}
        metrics = (summary.get("metrics") or {}) if isinstance(summary, dict) else {}
        print(
            f"{r['run_id']} | methods={summary.get('used_methods')} | "
            f"acc={fmt(metrics.get('accuracy'))} | f1={fmt(metrics.get('f1'))} | auc={fmt(metrics.get('roc_auc'))}"
        )


if __name__ == "__main__":
    app()

