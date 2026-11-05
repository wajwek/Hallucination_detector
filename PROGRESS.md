# Project progress

Last updated: November 5, 2026.

This file tracks established findings and open investigations. Mark an item complete only when supporting evidence is available. Keep hypotheses separate from verified explanations. The README describes the project; this checklist tracks the investigation.

## Established findings

- [x] Limit the project to HaluEval. Other dataset pipelines have been removed.
- [x] Prepare QA, dialogue and summarization as question-context-response triplets. Dialogue uses conversation history as the question; only summarization uses a fixed synthetic instruction.
- [x] Exclude general queries from SGI evaluation because they have no supplied context.
- [x] Complete the first Colab evaluation on 60,000 response rows, representing 30,000 source pairs. Each task has 10,000 reference and 10,000 hallucinated responses.
- [x] Save full-precision metrics, run configuration and dependency versions under `results/HaluEval/`.
- [x] Establish useful QA ranking in the expected direction: AUROC 0.73986109, Cohen's d 0.80575347.
- [x] Record weak dialogue ranking in the assumed direction: AUROC 0.47446514. Its significance relative to chance has not been assessed.
- [x] Record reversed summarization ranking under the same score convention: AUROC 0.40841045. Its cause is not yet established.
- [x] Record pooled AUROC 0.53972089 and zero exclusions under the current denominator tolerance.
- [x] Document the numerical QA comparison with the paper in README: AUROC 0.7399 versus 0.776, Cohen's d 0.8058 versus 0.92. This comparison does not establish matched protocols or a statistically significant difference.

Evidence: [metrics](results/HaluEval/sgi_metrics.csv), [run configuration](results/HaluEval/run_config.json), [environment](results/HaluEval/environment.json). These are exports from the November 4 Colab run. The local notebook has no saved execution outputs; individual score and embedding files remain on Google Drive. The summary metrics have not been independently recomputed here.

## Next investigation: explain the QA difference

- [x] Complete the first paper-to-notebook protocol audit, including the original SGI paper cited by the target article. See the protocol comparison in README.md.
- [ ] Locate a verifiable historical experiment script or obtain missing sample and long-context details. No such script was identified in the checked sources; the exact protocol remains unresolved.
- [ ] Resolve whether the paper's sample count refers to source items or individual responses, and establish how each class was selected.
- [x] Compare the documented model, pooling, normalization, token limits, text handling and denominator rules with the notebook. Record matches, differences and missing details in the audit.
- [ ] Test which confirmed differences change the metrics. No causal explanation has been established.
- [ ] Read the existing `sgi_scores.parquet` and inspect distributions without recomputing embeddings.
- [ ] Compute class-wise medians, standard deviations, quantiles and maxima for SGI and response-context angles.
- [ ] Plot SGI distributions and ROC curves separately for each task.
- [ ] Investigate dialogue's large mean gap alongside its tiny Cohen's d. Check extreme ratios and small denominators above the exclusion cutoff.
- [ ] Inspect representative response pairs and whether truncation removes evidence.
- [ ] Add confidence intervals using bootstrap resampling grouped by `pair_id`.
- [ ] Run controlled comparisons for verified protocol differences, changing one setting at a time and preserving the first run as a baseline.
- [ ] Write a supported explanation of the QA discrepancy, distinguishing measured causes from unresolved hypotheses.

## Extension: dialogue and summarization

- [x] Verify scope in the selected v1: the SGI result table covers QA; reported dialogue results use the separate Directional Grounding Index. Our dialogue and summarization SGI runs extend that reported comparison.
- [x] Inspect and document one actual q, c, r pair per task from the run's dataset revision. See examples in README.md. These illustrative records do not replace distribution-level error analysis.
- [ ] Explain dialogue results using score distributions and inspected examples.
- [ ] Test whether changing the synthetic summarization instruction changes the ranking, with an explicit development/evaluation protocol.
- [ ] Separate effects of the question input, truncation and response characteristics before assigning a cause to the reversed summarization result.
- [ ] Write task-specific conclusions and limitations, avoiding general factuality claims.

## Later evaluation

- [ ] Compare SGI with response-context similarity baselines on the same rows.
- [ ] Select a binary threshold on development data and report precision, recall, F1, accuracy and the confusion matrix on a separate test set, grouping by `pair_id`.
- [ ] Pin model and package revisions for subsequent reported runs.

## Working notes

- SGI measures relative semantic proximity, not factual truth. An SGI above 1 is not a validated truth decision.
- AUROC uses `-SGI` with hallucination as the positive label. AUROC 0.74 is not 74% classification accuracy.
- A fixed synthetic question is currently used only for summarization. Its role in the observed result is a hypothesis to test.
- No confidence intervals or validated binary threshold are available yet.
- Local CPU embedding generation was impractically slow for this workflow, so embeddings are computed on a cloud GPU in Colab. No controlled timing comparison has been recorded.

## Update log

- November 4, 2026: first Colab evaluation completed; summary exports added and README results updated.
- November 5, 2026: created this investigation checklist. Protocol audit, distribution diagnostics and causal explanations remain open.

- November 5, 2026: completed the first protocol audit and documented actual task examples. Confirmed sample-selection, long-context handling and denominator-rule mismatches against the original SGI description. Exact source rows, historical code and causes of the metric gap remain unresolved.

- November 5, 2026: added the descriptive threshold sweep (1.00 to 2.00, step 0.05), moved public examples into README.md and added tracked ANALYSIS.md. Code checks are separate from experiment execution; threshold results remain pending.
