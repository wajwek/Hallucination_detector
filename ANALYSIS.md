# Experiment analysis

## Scope

This project tests SGI on HaluEval. It follows the paper's geometric idea but does not fully replicate its experiment. QA is the comparison task. Dialogue uses conversation history as q; summarization uses our fixed instruction as q. Causes of the task differences remain untested.

## Recorded baseline

The November 4 run used all-mpnet-base-v2, a 384-token limit and manual masked mean pooling with L2 normalization. Its AUROC values were QA 0.73986109, dialogue 0.47446514, summarization 0.40841045 and pooled 0.53972089. These historical values are not recomputed by this threshold sweep.

## Status

Threshold code is ready. The table is pending a Colab run; accuracy cannot be recovered from the existing class means or AUROC. No threshold results have been measured locally.

## Approach 1: threshold sensitivity

Rule: predict hallucination when SGI < threshold. Equality is classified as grounded. The fixed grid is 1.00 through 2.00, inclusive, in steps of 0.05. This keeps the same scoring direction for every task. AUROC is independent of this chosen grid; accuracy, precision, recall, F1 and confusion counts depend on the threshold.

The table below reports descriptive accuracy on the saved scores. It is not an independent test of a selected threshold. Inspecting all rows and reporting the highest value as test performance would overstate the result. The current grid does not cover thresholds below 1.00, where QA may also have useful operating points.

<!-- THRESHOLD_TABLE_START -->
| SGI threshold | QA accuracy | Dialogue accuracy | Summarization accuracy | All accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | Pending | Pending | Pending | Pending |
| 1.05 | Pending | Pending | Pending | Pending |
| 1.10 | Pending | Pending | Pending | Pending |
| 1.15 | Pending | Pending | Pending | Pending |
| 1.20 | Pending | Pending | Pending | Pending |
| 1.25 | Pending | Pending | Pending | Pending |
| 1.30 | Pending | Pending | Pending | Pending |
| 1.35 | Pending | Pending | Pending | Pending |
| 1.40 | Pending | Pending | Pending | Pending |
| 1.45 | Pending | Pending | Pending | Pending |
| 1.50 | Pending | Pending | Pending | Pending |
| 1.55 | Pending | Pending | Pending | Pending |
| 1.60 | Pending | Pending | Pending | Pending |
| 1.65 | Pending | Pending | Pending | Pending |
| 1.70 | Pending | Pending | Pending | Pending |
| 1.75 | Pending | Pending | Pending | Pending |
| 1.80 | Pending | Pending | Pending | Pending |
| 1.85 | Pending | Pending | Pending | Pending |
| 1.90 | Pending | Pending | Pending | Pending |
| 1.95 | Pending | Pending | Pending | Pending |
| 2.00 | Pending | Pending | Pending | Pending |
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

Source score hash and grid settings are saved in threshold_sweep_config.json. Detailed counts are in threshold_metrics.csv; accuracy is in threshold_accuracy.csv and changes from threshold 1.00 are in threshold_accuracy_change_pp.csv. Preserve these files with the run before changing settings. Replace the repository's ANALYSIS.md with this report after reviewing the Colab output.
