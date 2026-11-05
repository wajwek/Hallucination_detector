# Experiment analysis

## Scope

This project tests SGI on HaluEval. It follows the paper's geometric idea but does not fully replicate its experiment. QA is the comparison task. Dialogue uses conversation history as q; summarization uses our fixed instruction as q. Causes of the task differences remain untested.

## Recorded baseline

The November 4 run used all-mpnet-base-v2, a 384-token limit and manual masked mean pooling with L2 normalization. Its AUROC values were QA 0.73986109, dialogue 0.47446514, summarization 0.40841045 and pooled 0.53972089. These historical values are not recomputed by this threshold sweep.

## Approach 1: threshold sensitivity

Rule: predict hallucination when SGI < threshold. Equality is classified as grounded. The fixed grid is 1.00 through 2.00, inclusive, in steps of 0.05. This keeps the same scoring direction for every task. AUROC is independent of this chosen grid; accuracy, precision, recall, F1 and confusion counts depend on the threshold.

The table below reports descriptive accuracy on the saved scores. It is not an independent test of a selected threshold. Inspecting all rows and reporting the highest value as test performance would overstate the result. The current grid does not cover thresholds below 1.00, where QA may also have useful operating points.

<!-- THRESHOLD_TABLE_START -->
| SGI threshold | QA accuracy | Dialogue accuracy | Summarization accuracy | All accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 63.77% | 47.94% | 50.00% | 53.90% |
| 1.05 | 60.98% | 46.41% | 50.00% | 52.47% |
| 1.10 | 59.16% | 45.70% | 49.99% | 51.62% |
| 1.15 | 57.53% | 45.85% | 49.98% | 51.12% |
| 1.20 | 55.81% | 46.35% | 49.98% | 50.71% |
| 1.25 | 54.20% | 46.87% | 49.96% | 50.34% |
| 1.30 | 53.10% | 47.30% | 49.94% | 50.11% |
| 1.35 | 52.04% | 47.80% | 49.94% | 49.93% |
| 1.40 | 51.30% | 48.21% | 49.90% | 49.80% |
| 1.45 | 50.81% | 48.55% | 49.83% | 49.73% |
| 1.50 | 50.47% | 48.87% | 49.77% | 49.70% |
| 1.55 | 50.29% | 49.10% | 49.57% | 49.65% |
| 1.60 | 50.11% | 49.20% | 49.38% | 49.57% |
| 1.65 | 50.03% | 49.31% | 49.28% | 49.54% |
| 1.70 | 49.95% | 49.51% | 48.97% | 49.48% |
| 1.75 | 49.94% | 49.58% | 48.54% | 49.35% |
| 1.80 | 49.89% | 49.66% | 48.19% | 49.25% |
| 1.85 | 49.90% | 49.77% | 47.88% | 49.18% |
| 1.90 | 49.94% | 49.80% | 47.41% | 49.05% |
| 1.95 | 49.95% | 49.83% | 46.99% | 48.92% |
| 2.00 | 49.97% | 49.84% | 46.47% | 48.76% |
<!-- THRESHOLD_TABLE_END -->

Accuracy is the fraction of labels predicted correctly. CSV values are fractions; this table shows percentages. The sweep exports precision, recall, F1 and TP/FP/TN/FN for every task and threshold. Undefined ratios are NaN. Nonfinite SGI values and invalid labels are excluded and counted.

## Conclusions and next checks

The baseline suggests useful QA ranking, weak dialogue ranking and reversed summarization ranking. It does not establish why. The sweep measures how decisions change as the cutoff increases; it does not change or improve AUROC. Increasing the threshold can only increase or preserve the number of hallucination predictions. Recall cannot decrease, while false alarms cannot decrease; accuracy and precision can move in either direction.

An always-one-class prediction would score 50% on the balanced November 4 task subsets. Compare threshold accuracy with that baseline. The pooled result does not replace task-specific conclusions. No threshold has been selected or validated for deployment.

- [ ] Inspect score distributions and small response-context angles.
- [ ] Compare SGI against response-context similarity.
- [ ] Test alternative summarization inputs using a development protocol.
- [ ] Select thresholds on development pairs and evaluate once on separate test pairs.
- [ ] Add bootstrap intervals grouped by pair_id.

## Run evidence

Source score hash and grid settings are saved in `results/HaluEval/threshold_sweep_config.json`. Detailed counts are in `results/HaluEval/threshold_metrics.csv`; accuracy is in `results/HaluEval/threshold_accuracy.csv` and changes from threshold 1.00 are in `results/HaluEval/threshold_accuracy_change_pp.csv`. Preserve these files with the run before changing settings.
