"""Run saved-score SGI analysis locally without a GPU or Google Colab."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
        raise FileNotFoundError(f"No saved scores at {SCORES_PATH}. Mount Drive and run SGI scoring first.")
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

    report = f"""# Experiment analysis

## Scope

SGI is evaluated on HaluEval QA, dialogue and summarization. This is not a full replication of the paper. Dialogue uses the conversation history as q; summarization uses a fixed instruction added by this project.

## Approach 1: fixed threshold grid

Predict hallucination when SGI < threshold; equality is grounded. The current grid is {THRESHOLD_START:.2f} to {THRESHOLD_STOP:.2f}, inclusive, in steps of {THRESHOLD_STEP:.2f}. Accuracy changes are measured against threshold {REFERENCE_THRESHOLD:.2f} in percentage points. These results describe the saved labeled scores, not an independent test set.

{accuracy_markdown(threshold_table)}

## Approach 2: midpoint of class means

For each task, threshold = (mean grounded SGI + mean hallucinated SGI) / 2. The pooled row is an additional summary, not a replacement for task-specific thresholds. Midpoints are evaluated at full precision and are not clipped to the grid. Means, thresholds and metrics use only finite SGI values with binary labels.

{midpoint_markdown(midpoint_estimates, midpoint_results)}

The fixed-below column preserves the original SGI rule. The mean-based diagnostic flags scores above the midpoint when the hallucinated mean exceeds the grounded mean. Equality is grounded in both policies. This second policy changes the decision direction, not the score or the baseline AUROC. When the direction is below, the policies are identical.

The midpoint is a heuristic, not an optimal threshold. It ignores dispersion, extreme ratios, class priors and error costs. The means, threshold and diagnostic direction were estimated on the same data used for evaluation, so these are exploratory results. Held-out performance remains unknown. Do not report the highest grid accuracy as validated test accuracy.

## Interpretation and next checks

The original baseline AUROC was 0.7399 for QA, 0.4745 for dialogue and 0.4084 for summarization using -SGI. The new cutoffs do not change that ranking. If the summarization midpoint lies outside the grid, it is still evaluated separately. The test asks whether a different cutoff helps, and whether the reversed mean order warrants a separate direction check. It does not establish why the ranking is reversed.

- [x] Implement and execute the 0.70-1.40 grid on saved scores.
- [x] Compute class means, midpoint thresholds and decision metrics.
- [ ] Review whether the midpoint improves accuracy, recall and false alarm rate against threshold 1.00.
- [ ] Inspect dispersion and extreme ratios, especially for dialogue.
- [ ] Select a threshold and direction on development pairs and evaluate once on separate test pairs.
- [ ] Add bootstrap intervals grouped by pair_id.

## Run evidence

This report was generated locally from saved per-response scores. threshold_sweep_config.json records their SHA-256 and the evaluation settings. threshold_metrics.csv contains the grid metrics; midpoint_estimates.csv and midpoint_metrics.csv contain the heuristic estimates and decisions. Accuracy values in CSV are fractions; this report shows percentages. Zero-denominator ratios are NaN. Invalid rows are excluded and counted. The completed 1.00-2.00 exports remain in the parent results folder.
"""
    report = report.strip() + "\n"
    (OUTPUT_DIR / "ANALYSIS.md").write_text(report)
    environment = {name: importlib.metadata.version(name) for name in
                   ["numpy", "pandas", "pyarrow", "scikit-learn"]}
    (OUTPUT_DIR / "analysis_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if update_report:
        report_path = PROJECT_ROOT / "ANALYSIS.md"
        backup = OUTPUT_DIR / "previous_ANALYSIS.md"
        if report_path.exists() and not backup.exists():
            backup.write_text(report_path.read_text())
        report_path.write_text(report)
    return dict(accuracy=comparison, accuracy_change_pp=accuracy_change,
                midpoint_estimates=midpoint_estimates, midpoint_metrics=midpoint_results,
                threshold_metrics=threshold_table, report=report, output_dir=OUTPUT_DIR)


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
    print("\nSaved results to:", results["output_dir"])


if __name__ == "__main__":
    main()
