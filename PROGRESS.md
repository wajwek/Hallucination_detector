# Research progress

Updated: November 6, 2026.

This checklist records completed experiments and open questions. A checked item means there is a result or an inspected artifact, not that a hypothesis has been proved. Read [ANALYSIS.md](ANALYSIS.md) for measurements and conclusions.

## Milestone 1: define the experiment

- [x] Limit the current benchmark scope to HaluEval.
- [x] Define QA triplets from question, knowledge and each candidate answer.
- [x] Define dialogue triplets using conversation history and supplied knowledge.
- [x] Define an exploratory summarization variant with a fixed instruction as q.
- [x] Exclude general queries from SGI evaluation because they lack context.
- [x] Preserve reference/hallucinated pairs and document label direction.
- [x] Document actual q, c, r examples in README.md.

## Milestone 2: establish the SGI baseline

- [x] Prepare 60,000 response rows from 30,000 source pairs.
- [x] Generate embeddings with all-mpnet-base-v2, masked mean pooling and L2 normalization.
- [x] Record dataset revision, token limit, pooling, label convention and dependency versions.
- [x] Compute angular distances, SGI, class means, Cohen's d and AUROC.
- [x] Verify the saved per-task AUROC and means against the individual scores.
- [x] Report task-specific results rather than relying on the pooled value.

Observed AUROC using -SGI: QA 0.73986109, dialogue 0.47446514 and summarization 0.40841045. The pooled value is 0.53972089. Zero rows were excluded under the recorded angle tolerance. These are descriptive findings from the recorded experiment.

## Milestone 3: compare with the publication

- [x] Audit the selected paper and its earlier SGI implementation description.
- [x] Compare QA with the reported all-mpnet-base-v2 result: AUROC 0.776 and Cohen's d 0.92.
- [x] Record matching geometry and differences in sampling, context handling and denominator stabilization.
- [x] Verify that the selected paper's dialogue comparison uses the separate directional method.
- [ ] Resolve the exact source-row selection and meaning of the reported sample counts.
- [ ] Locate a verifiable historical script or obtain missing preprocessing details.
- [ ] Test confirmed protocol differences one at a time.
- [ ] Explain the QA gap using measured effects and uncertainty estimates.

The comparison supports the direction of the QA effect. It does not establish an exact replication or identify the cause of the numerical gap.

## Milestone 4: evaluate decision rules

- [x] Complete the historical 1.00-2.00 threshold sweep, step 0.05.
- [x] Complete the 0.70-1.40 sweep, step 0.05, using 1.00 as the accuracy reference.
- [x] Verify confusion counts, derived metrics and overlapping historical results.
- [x] Estimate a midpoint from the two class means for each task.
- [x] Evaluate midpoints at full precision, including values outside the grid.
- [x] Keep the original below-threshold rule separate from the mean-based direction diagnostic.
- [x] Compare accuracy, precision, recall and false alarms for the tested approaches.
- [x] Consolidate numerical interpretation in ANALYSIS.md and retain CSV evidence.

Within the tested grid, QA accuracy reaches 70.565% at 0.85. Its midpoint gives 65.08%. The summarization midpoint gives 43.34% with the original direction and 56.66% with the diagnostic reversed direction. These values are not independent test estimates.

## Next: explain task differences

- [x] Compute class-wise means, medians, sample standard deviations, quantiles, maxima and interquartile ranges from the saved scores.
- [ ] Inspect small response-context angles and dialogue's large score dispersion.
- [x] Plot per-task central histograms and full-range empirical cumulative distributions, with visual clipping counts.
- [ ] Plot ROC curves separately for each task.
- [ ] Inspect representative errors and responses affected by context truncation.
- [ ] Test whether the synthetic summarization instruction changes the score direction.
- [ ] Compare SGI with response-context cosine similarity on the same pairs.
- [ ] Separate input-design effects from length, style and truncation effects before assigning causes.

## Next: validate a detector

- [ ] Define development and test splits grouped by pair_id.
- [ ] Select threshold and direction on development pairs only.
- [ ] Evaluate the selected rule once on separate test pairs.
- [ ] Add paired bootstrap confidence intervals for ranking and decision metrics.
- [ ] Check encoder and token-limit sensitivity without tuning on test data.
- [ ] Pin encoder revisions for subsequent experiments.

## Experiment log

| Date | Work completed | Evidence or consequence |
| --- | --- | --- |
| November 4, 2026 | First HaluEval SGI evaluation | Per-task baseline, configuration and score exports |
| November 5, 2026 | Protocol audit and task examples | Documented replication limits and experimental input choices |
| November 5, 2026 | Fixed threshold sweeps | Measured cutoff sensitivity and verified overlapping results |
| November 5, 2026 | Class-mean midpoint experiment | Compared the original rule with an explicit direction diagnostic |
| November 5, 2026 | Repeatable saved-score evaluation | Recomputed results without another embedding run; consolidated report and research checklist |
| November 6, 2026 | Class-wise distribution analysis | Measured quantiles and dispersion; plotted central and full ranges; found opposite mean and median orders for dialogue |

## Interpretation rules

SGI measures relative semantic proximity, not factual truth. AUROC measures ranking rather than accuracy at a selected cutoff. A midpoint is a heuristic, and a reversed score direction must be declared explicitly. Neither a new cutoff nor the current benchmark results establish a general hallucination detector.
