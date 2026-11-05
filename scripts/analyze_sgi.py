"""Evaluate SGI decision rules and generate the experiment analysis report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASKS = ["qa", "dialogue", "summarization", "all"]


def threshold_metrics(frame, thresholds, prediction_direction="below", tasks=TASKS):
    if prediction_direction not in {"below", "above"}:
        raise ValueError("prediction_direction must be below or above")
    required = {"subset", "sgi", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    scores = pd.to_numeric(frame["sgi"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid = np.isfinite(scores) & labels.isin([0, 1])
    work = frame.loc[valid, ["subset"]].copy()
    work["sgi"] = scores.loc[valid]
    work["label"] = labels.loc[valid].astype(int)
    result = []
    for task in tasks:
        original = frame if task == "all" else frame.loc[frame["subset"] == task]
        group = work if task == "all" else work.loc[work["subset"] == task]
        truth = group["label"].to_numpy() == 1
        values = group["sgi"].to_numpy()
        for threshold in thresholds:
            predicted = values < float(threshold) if prediction_direction == "below" else values > float(threshold)
            tp = int(np.sum(predicted & truth))
            fp = int(np.sum(predicted & ~truth))
            tn = int(np.sum(~predicted & ~truth))
            fn = int(np.sum(~predicted & truth))
            n = len(group)
            result.append(dict(
                subset=task, threshold=float(threshold), rows=len(original),
                evaluated_rows=n, excluded_rows=len(original)-n,
                tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=(tp+tn)/n if n else np.nan,
                precision=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                predicted_hallucination_rate=(tp+fp)/n if n else np.nan,
            ))
    return pd.DataFrame(result)


def accuracy_markdown(table):
    lines = ["| SGI threshold | QA accuracy | Dialogue accuracy | Summarization accuracy | All accuracy |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    pivot = table.pivot(index="threshold", columns="subset", values="accuracy")
    for threshold, row in pivot.sort_index().iterrows():
        cells = [f"{100*row[task]:.2f}%" if pd.notna(row[task]) else "NaN" for task in TASKS]
        lines.append(f"| {threshold:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def estimate_midpoints(frame, thresholds):
    scores = pd.to_numeric(frame["sgi"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    clean = frame.loc[np.isfinite(scores) & labels.isin([0, 1]), ["subset"]].copy()
    clean["sgi"] = scores.loc[clean.index]
    clean["label"] = labels.loc[clean.index].astype(int)
    estimates = []
    for task in TASKS:
        group = clean if task == "all" else clean.loc[clean["subset"] == task]
        grounded = group.loc[group.label == 0, "sgi"]
        hallucinated = group.loc[group.label == 1, "sgi"]
        gmean, hmean = grounded.mean(), hallucinated.mean()
        midpoint = (gmean + hmean) / 2
        # This direction is a separate diagnostic, not a change to the default rule.
        direction = "above" if hmean > gmean else "below"
        estimates.append(dict(
            subset=task, grounded_rows=len(grounded), hallucinated_rows=len(hallucinated),
            mean_sgi_grounded=gmean, mean_sgi_hallucinated=hmean,
            midpoint_threshold=midpoint, class_mean_direction=direction,
            outside_grid=bool(midpoint < thresholds.min() or midpoint > thresholds.max())
                         if np.isfinite(midpoint) else None,
        ))
    return pd.DataFrame(estimates)


def evaluate_midpoints(frame, estimates):
    rows = []
    for estimate in estimates.to_dict("records"):
        task, threshold = estimate["subset"], estimate["midpoint_threshold"]
        for policy, direction in [("fixed_below", "below"),
                                  ("class_mean_direction", estimate["class_mean_direction"])]:
            if np.isfinite(threshold):
                row = threshold_metrics(frame, [threshold], direction, tasks=[task]).iloc[0].to_dict()
            else:
                # Missing classes cannot produce a midpoint or a decision evaluation.
                row = dict(subset=task, threshold=np.nan)
            row.update(policy=policy, prediction_direction=direction, **{
                k: v for k, v in estimate.items() if k != "subset"
            })
            rows.append(row)
    return pd.DataFrame(rows)


def midpoint_markdown(estimates, evaluated):
    lines = ["| Task | Mean SGI: grounded | Mean SGI: hallucinated | Midpoint | Accuracy: SGI < midpoint | Mean-based rule | Accuracy: mean-based rule |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for e in estimates.to_dict("records"):
        task = e["subset"]
        fixed = evaluated.loc[(evaluated.subset == task) & (evaluated.policy == "fixed_below")]
        oriented = evaluated.loc[(evaluated.subset == task) & (evaluated.policy == "class_mean_direction")]
        def percent(frame):
            value = frame.iloc[0].get("accuracy", np.nan)
            return f"{100*value:.2f}%" if pd.notna(value) else "NaN"
        rule = "SGI > midpoint" if e["class_mean_direction"] == "above" else "SGI < midpoint"
        lines.append(f"| {task} | {e['mean_sgi_grounded']:.6f} | {e['mean_sgi_hallucinated']:.6f} | "
                     f"{e['midpoint_threshold']:.6f} | {percent(fixed)} | {rule} | {percent(oriented)} |")
    return "\n".join(lines)

def summarize_scores(frame):
    scores = pd.to_numeric(frame["sgi"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid = np.isfinite(scores) & labels.isin([0, 1])
    clean = frame.loc[valid, ["subset"]].copy()
    clean["sgi"] = scores.loc[valid]
    clean["label"] = labels.loc[valid].astype(int)
    rows = []
    for task in TASKS:
        original = frame if task == "all" else frame.loc[frame.subset == task]
        group = clean if task == "all" else clean.loc[clean.subset == task]
        grounded = group.loc[group.label == 0, "sgi"].to_numpy()
        hallucinated = group.loc[group.label == 1, "sgi"].to_numpy()
        d = np.nan
        if len(grounded) > 1 and len(hallucinated) > 1:
            variance = ((len(grounded)-1)*grounded.var(ddof=1)
                        + (len(hallucinated)-1)*hallucinated.var(ddof=1)) / (len(group)-2)
            if variance > 0:
                d = (grounded.mean()-hallucinated.mean()) / np.sqrt(variance)
        rows.append(dict(subset=task, rows=len(original), excluded_rows=len(original)-len(group),
                         mean_sgi_grounded=grounded.mean() if len(grounded) else np.nan,
                         mean_sgi_hallucinated=hallucinated.mean() if len(hallucinated) else np.nan,
                         cohens_d=d,
                         auroc=roc_auc_score(group.label, -group.sgi) if group.label.nunique() == 2 else np.nan))
    return pd.DataFrame(rows)


def analysis_report(summary, grid, estimates, midpoint_results, start, stop, step):
    baseline_lines = ["| Task | Response rows | Mean SGI: grounded | Mean SGI: hallucinated | Cohen's d | AUROC (-SGI) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        baseline_lines.append(f"| {row.subset} | {row.rows:,} | {row.mean_sgi_grounded:.6f} | "
                              f"{row.mean_sgi_hallucinated:.6f} | {row.cohens_d:.6f} | {row.auroc:.6f} |")
    comparison_lines = ["| Task | Accuracy at 1.00 | Highest accuracy in this grid | Grid threshold | Midpoint | Accuracy at midpoint (below rule) |",
                        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for task in TASKS:
        task_grid = grid.loc[grid.subset == task]
        reference = task_grid.loc[np.isclose(task_grid.threshold, 1.0)].iloc[0]
        available = task_grid.dropna(subset=["accuracy"])
        best = available.loc[available.accuracy.idxmax()] if len(available) else reference
        mid = midpoint_results.loc[(midpoint_results.subset == task)
                                   & (midpoint_results.policy == "fixed_below")].iloc[0]
        comparison_lines.append(f"| {task} | {100*reference.accuracy:.2f}% | {100*best.accuracy:.2f}% | "
                                f"{best.threshold:.2f} | {mid.threshold:.6f} | {100*mid.get('accuracy', np.nan):.2f}% |")
    observations = {}
    for task in TASKS:
        current = grid.loc[grid.subset == task]
        reference = current.loc[np.isclose(current.threshold, 1.0)].iloc[0]
        available = current.dropna(subset=["accuracy"])
        best = available.loc[available.accuracy.idxmax()] if len(available) else reference
        fixed = midpoint_results.loc[(midpoint_results.subset == task)
                                     & (midpoint_results.policy == "fixed_below")].iloc[0]
        oriented = midpoint_results.loc[(midpoint_results.subset == task)
                                        & (midpoint_results.policy == "class_mean_direction")].iloc[0]
        if task == "qa":
            observations[task] = (
                f"In this grid, accuracy reaches {100*best.accuracy:.3f}% at {best.threshold:.2f}, "
                f"compared with {100*reference.accuracy:.3f}% at 1.00. "
                f"The midpoint gives {100*fixed.get('accuracy', np.nan):.2f}%. "
                "This compares a geometric boundary with empirical cutoffs; it does not establish an independently optimal threshold.")
        elif task == "dialogue":
            denominator = fixed.get("fp", 0) + fixed.get("tn", 0)
            false_alarm = fixed.get("fp", 0) / denominator if denominator else np.nan
            observations[task] = (
                f"The highest grid accuracy is {100*best.accuracy:.3f}% at {best.threshold:.2f}. "
                f"The midpoint gives {100*fixed.get('accuracy', np.nan):.2f}%, with recall "
                f"{100*fixed.get('recall', np.nan):.2f}% and false positive rate {100*false_alarm:.2f}%. "
                "A high recall with frequent false alarms does not establish a useful detector. "
                "Inspect dispersion and denominator angles before relying on class means.")
        elif task == "summarization":
            reference_note = (
                "At threshold 1.00, all responses are accepted as grounded. "
                if reference.predicted_hallucination_rate == 0 else "At threshold 1.00, some responses are flagged. ")
            observations[task] = (
                reference_note + f"Reference accuracy is {100*reference.accuracy:.2f}% and hallucination recall "
                f"is {100*reference.recall:.2f}%. The midpoint is {fixed.threshold:.6f}. "
                f"It yields {100*fixed.get('accuracy', np.nan):.2f}% under the below rule and "
                f"{100*oriented.get('accuracy', np.nan):.2f}% under the mean-based diagnostic. "
                "This does not establish a cause for the task difference or transfer to new examples. "
                "The constant instruction is an exploratory adaptation, not a question supplied by HaluEval.")
    baseline_table = "\n".join(baseline_lines)
    comparison_table = "\n".join(comparison_lines)
    grid_table = accuracy_markdown(grid)
    midpoint_table = midpoint_markdown(estimates, midpoint_results)
    artifact_dir = f"results/HaluEval/thresholds_{start:.2f}_{stop:.2f}"
    return f"""# Experiment analysis

## Research question

Can the Semantic Grounding Index distinguish reference and hallucinated responses by their position relative to a question and its supporting context? The experiments test the same angular ratio across HaluEval QA, dialogue and summarization, then examine how decision thresholds affect classification.

The project follows the geometric idea in [Marín's paper](https://arxiv.org/html/2602.13224v1), but does not fully replicate its protocol. QA is the published SGI comparison task. Dialogue uses conversation history as q; summarization uses an instruction added by this project. The separate Directional Grounding Index is not implemented.

## Experiment 1: establish the grounding signal

The recorded embedding run used all-mpnet-base-v2, masked mean pooling, L2 normalization and a 384-token limit. Each task contributes 10,000 source pairs, each containing a reference response and a hallucinated response. Labels are 0 for reference and 1 for hallucinated. General queries are excluded because they lack context.

The table is recomputed from saved per-response scores. AUROC uses -SGI, so the assumed direction is lower SGI for hallucinations.

{baseline_table}

For QA, the paper reports AUROC 0.776 and Cohen's d 0.92 for the same encoder name. The recorded baseline in this repository supports the direction of the effect but does not match those values. Differences in sample selection, long-context handling and denominator stabilization remain candidates to test. No cause has been established, and no confidence interval for the discrepancy is available.

## Experiment 2: compare fixed cutoffs

Predict hallucination when SGI < threshold; equality is grounded. The current grid spans {start:.2f} to {stop:.2f} in steps of {step:.2f}. The earlier 1.00-2.00 sweep is retained in the parent results folder. Accuracy changes use 1.00 as the reference.

{grid_table}

These are descriptive measurements on the available labeled examples. The highest accuracy in the grid is an observation on these data, not a validated threshold. Where several thresholds tie, the comparison below lists the first. Changing the cutoff does not change AUROC for a fixed score direction.

## Experiment 3: test a threshold derived from class means

For each task, midpoint = (mean grounded SGI + mean hallucinated SGI) / 2. It is evaluated at full precision, even outside the fixed grid. The pooled estimate is a separate summary and does not replace task-specific thresholds.

{midpoint_table}

The fixed-below policy preserves the original rule. The mean-based diagnostic uses SGI > midpoint when the hallucinated mean is higher; otherwise it is identical to the below rule. Equality is grounded in both policies. Direction changes are explicit and do not alter the baseline AUROC reported above.

A midpoint is a heuristic, not an optimum. It ignores distribution shape, extreme ratios, class priors and error costs. Both the midpoint and the diagnostic direction are estimated on the same labeled data used for evaluation. Any improvement needs confirmation on separate test pairs.

## Comparison of the tested approaches

{comparison_table}

## Findings from the recorded HaluEval experiment

**QA:** {observations['qa']}

**Dialogue:** {observations['dialogue']}

**Summarization:** {observations['summarization']}

**Pooled scores:** Combining tasks hides their different score distributions. Report task-specific results before drawing conclusions from the pooled value. The individual source pairs are also dependent observations: both responses must remain together in any split or bootstrap.

The tables and decision comparisons above are computed from the supplied scores. The encoder settings and literature comparison refer to the recorded HaluEval experiment; document any change to that provenance when evaluating other inputs.

## Next experiments

- [x] Establish the per-task SGI baseline and compare QA with the paper.
- [x] Evaluate both threshold grids and verify the overlapping results.
- [x] Evaluate class-mean midpoint thresholds with explicit directions.
- [x] Compare accuracy, recall and false alarms for the tested approaches.
- [ ] Inspect SGI quantiles, dispersion and small response-context angles.
- [ ] Test whether context truncation changes the QA result.
- [ ] Compare SGI against response-context similarity on the same pairs.
- [ ] Test alternative summarization instructions using development pairs.
- [ ] Select thresholds and directions on development pairs, then evaluate once on separate test pairs.
- [ ] Add bootstrap confidence intervals grouped by pair_id.

## Evidence and reproducibility

[Baseline exports](results/HaluEval/sgi_metrics.csv), [embedding configuration](results/HaluEval/run_config.json) and [threshold configuration]({artifact_dir}/threshold_sweep_config.json) record the data revision, preprocessing and source-score hash. [Grid metrics]({artifact_dir}/threshold_metrics.csv) and [midpoint metrics]({artifact_dir}/midpoint_metrics.csv) retain full-precision values and confusion counts. Package versions are recorded for reproducibility. The embedding model revision was not pinned in the initial run.

Accuracy values in CSV are fractions; Markdown tables show percentages. Nonfinite scores and nonbinary labels are excluded and counted. Undefined ratios are NaN. No embeddings were recomputed for threshold evaluation. This file is the single narrative report; result directories contain numerical exports and run metadata.
"""


def run_analysis(project_root=None, scores_path=None, threshold_start=0.70,
                 threshold_stop=1.40, threshold_step=0.05, update_report=True):
    """Evaluate saved scores and write CSVs plus the repository analysis report."""
    PROJECT_ROOT = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    RESULTS_DIR = PROJECT_ROOT / "results" / "HaluEval"
    SCORES_PATH = Path(scores_path) if scores_path else RESULTS_DIR / "sgi_scores.parquet"
    THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = threshold_start, threshold_stop, threshold_step
    REFERENCE_THRESHOLD = 1.00
    for value in (THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP):
        if not np.isfinite(value) or not np.isclose(value * 100, round(value * 100), atol=1e-8, rtol=0):
            raise ValueError("Threshold settings must be finite and have at most two decimal places.")
    start_units, stop_units, step_units = [int(round(x * 100)) for x in
                                          (THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP)]
    if step_units <= 0 or stop_units < start_units or (stop_units-start_units) % step_units:
        raise ValueError("Use a positive step that reaches the upper endpoint exactly.")
    THRESHOLDS = np.arange(start_units, stop_units + 1, step_units) / 100
    if not np.any(np.isclose(THRESHOLDS, REFERENCE_THRESHOLD)):
        raise ValueError("The grid must include reference threshold 1.00.")
    OUTPUT_DIR = RESULTS_DIR / f"thresholds_{THRESHOLD_START:.2f}_{THRESHOLD_STOP:.2f}"
    if not SCORES_PATH.exists():
        raise FileNotFoundError("Saved SGI scores are missing. Generate sgi_scores.parquet with the embedding notebook or provide --scores PATH.")
    score_frame = pd.read_parquet(SCORES_PATH).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table = threshold_metrics(score_frame, THRESHOLDS)
    threshold_table.to_csv(OUTPUT_DIR / "threshold_metrics.csv", index=False)
    comparison = threshold_table.pivot(index="threshold", columns="subset", values="accuracy")[TASKS]
    comparison.to_csv(OUTPUT_DIR / "threshold_accuracy.csv", float_format="%.8f")
    accuracy_change = comparison.sub(comparison.loc[REFERENCE_THRESHOLD]).mul(100)
    accuracy_change.to_csv(OUTPUT_DIR / "threshold_accuracy_change_pp.csv", float_format="%.8f")
    midpoint_estimates = estimate_midpoints(score_frame, THRESHOLDS)
    midpoint_estimates.to_csv(OUTPUT_DIR / "midpoint_estimates.csv", index=False)
    midpoint_results = evaluate_midpoints(score_frame, midpoint_estimates)
    midpoint_results.to_csv(OUTPUT_DIR / "midpoint_metrics.csv", index=False)

    sha256 = hashlib.sha256()
    with SCORES_PATH.open("rb") as score_file:
        for block in iter(lambda: score_file.read(1024*1024), b""):
            sha256.update(block)
    sweep_config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scores_file=SCORES_PATH.name, scores_sha256=sha256.hexdigest(),
        threshold_start=THRESHOLD_START, threshold_stop=THRESHOLD_STOP,
        threshold_step=THRESHOLD_STEP, thresholds=THRESHOLDS.tolist(),
        reference_threshold=REFERENCE_THRESHOLD,
        prediction_rule="hallucination if sgi < threshold; equality is grounded",
        midpoint_formula="(mean_sgi_grounded + mean_sgi_hallucinated) / 2",
        midpoint_policies=["fixed_below", "class_mean_direction"],
        midpoint_clipped_to_grid=False,
        positive_label=1, evaluation="descriptive: thresholds estimated and evaluated on the same scores",
        nonfinite_scores="excluded", undefined_ratios="NaN",
    )
    (OUTPUT_DIR / "threshold_sweep_config.json").write_text(json.dumps(sweep_config, indent=2))

    summary = summarize_scores(score_frame)
    summary.to_csv(OUTPUT_DIR / "sgi_summary.csv", index=False)
    report = analysis_report(summary, threshold_table, midpoint_estimates, midpoint_results,
                             THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP)
    environment = {name: importlib.metadata.version(name) for name in
                   ["numpy", "pandas", "pyarrow", "scikit-learn"]}
    (OUTPUT_DIR / "analysis_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if update_report:
        (PROJECT_ROOT / "ANALYSIS.md").write_text(report)
    return dict(accuracy=comparison, accuracy_change_pp=accuracy_change,
                midpoint_estimates=midpoint_estimates, midpoint_metrics=midpoint_results,
                threshold_metrics=threshold_table, summary=summary, report=report, output_dir=OUTPUT_DIR)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="Path to saved sgi_scores.parquet")
    parser.add_argument("--start", type=float, default=0.70)
    parser.add_argument("--stop", type=float, default=1.40)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--no-update-report", action="store_true")
    args = parser.parse_args()
    results = run_analysis(scores_path=args.scores, threshold_start=args.start,
                           threshold_stop=args.stop, threshold_step=args.step,
                           update_report=not args.no_update_report)
    print("Accuracy by threshold (fractions):")
    print(results["accuracy"].to_string())
    print("\nClass-mean midpoint decisions:")
    print(results["midpoint_metrics"][["subset", "policy", "threshold", "accuracy", "recall", "precision"]].to_string(index=False))
    print("\nEvaluation complete. CSV exports are in results/HaluEval; interpretation is in ANALYSIS.md.")

if __name__ == "__main__":
    main()
