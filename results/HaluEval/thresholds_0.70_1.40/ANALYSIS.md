# Experiment analysis

## Scope

SGI is evaluated on HaluEval QA, dialogue and summarization. This is not a full replication of the paper. Dialogue uses the conversation history as q; summarization uses a fixed instruction added by this project.

## Approach 1: fixed threshold grid

Predict hallucination when SGI < threshold; equality is grounded. The current grid is 0.70 to 1.40, inclusive, in steps of 0.05. Accuracy changes are measured against threshold 1.00 in percentage points. These results describe the saved labeled scores, not an independent test set.

| SGI threshold | QA accuracy | Dialogue accuracy | Summarization accuracy | All accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.70 | 61.18% | 50.30% | 50.00% | 53.83% |
| 0.75 | 64.53% | 50.62% | 50.00% | 55.05% |
| 0.80 | 67.83% | 51.39% | 50.00% | 56.41% |
| 0.85 | 70.56% | 52.05% | 50.00% | 57.54% |
| 0.90 | 70.17% | 52.49% | 50.00% | 57.55% |
| 0.95 | 66.73% | 50.75% | 50.00% | 55.83% |
| 1.00 | 63.77% | 47.94% | 50.00% | 53.90% |
| 1.05 | 60.98% | 46.41% | 50.00% | 52.47% |
| 1.10 | 59.16% | 45.70% | 49.99% | 51.62% |
| 1.15 | 57.53% | 45.85% | 49.98% | 51.12% |
| 1.20 | 55.81% | 46.35% | 49.98% | 50.71% |
| 1.25 | 54.20% | 46.87% | 49.96% | 50.34% |
| 1.30 | 53.10% | 47.30% | 49.94% | 50.11% |
| 1.35 | 52.04% | 47.80% | 49.94% | 49.93% |
| 1.40 | 51.30% | 48.21% | 49.90% | 49.80% |

## Approach 2: midpoint of class means

For each task, threshold = (mean grounded SGI + mean hallucinated SGI) / 2. The pooled row is an additional summary, not a replacement for task-specific thresholds. Midpoints are evaluated at full precision and are not clipped to the grid. Means, thresholds and metrics use only finite SGI values with binary labels.

| Task | Mean SGI: grounded | Mean SGI: hallucinated | Midpoint | Accuracy: SGI < midpoint | Mean-based rule | Accuracy: mean-based rule |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| qa | 1.066184 | 0.885615 | 0.975899 | 65.08% | SGI < midpoint | 65.08% |
| dialogue | 2.142929 | 1.057075 | 1.600002 | 49.20% | SGI < midpoint | 49.20% |
| summarization | 2.500901 | 2.683187 | 2.592044 | 43.34% | SGI > midpoint | 56.66% |
| all | 1.903338 | 1.541959 | 1.722648 | 49.43% | SGI < midpoint | 49.43% |

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
