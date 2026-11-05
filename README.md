# Geometric Hallucination Detection with SGI

Can sentence-embedding geometry distinguish a grounded answer from a plausible hallucination? This project evaluates the **Semantic Grounding Index (SGI)** on **HaluEval** and investigates where its signal holds, weakens or reverses.

The project is based on Javier Marín's [*A Geometric Taxonomy of Hallucinations in LLMs*](https://arxiv.org/html/2602.13224v1). It follows the context-based SGI concept, but **does not fully replicate the paper's experiment**. It also extends the reported QA comparison to dialogue and summarization. The separate context-free Directional Grounding Index is outside the current scope.

## Research workflow

1. Map HaluEval into question-context-response triplets and preserve reference/hallucinated pairs.
2. Encode the three inputs separately and compute SGI from angular distances.
3. Measure per-task ranking and class separation; compare QA with the published reference.
4. Test fixed decision thresholds and thresholds derived from the two class means.
5. Investigate task differences before selecting a detector on development data and testing it on separate examples.

**Current findings:** QA AUROC is 0.740. QA accuracy reaches 70.565% at threshold 0.85 within the tested 0.70-1.40 grid. Dialogue and summarization behave differently, and a single threshold does not work uniformly across tasks. These are descriptive results, not held-out classifier performance.

Read [ANALYSIS.md](ANALYSIS.md) for tables, comparisons and interpretation. [PROGRESS.md](PROGRESS.md) records completed experiments, unresolved questions and the next steps.

## Method

Let q be a question, c its supporting context and r a candidate response. The encoder maps each text to an L2-normalized vector. Angular distance and SGI are:

```text
theta(a, b) = arccos(clip(embedding(a) · embedding(b), -1, 1))
SGI(r; q, c) = theta(r, q) / theta(r, c)
```

SGI above 1 means the response is angularly closer to the context than to the question. The hypothesis predicts higher SGI for grounded responses. This is a measure of relative semantic proximity, not factual truth: a wrong detail can preserve the vocabulary and topic of a correct answer.

The embedding run uses `sentence-transformers/all-mpnet-base-v2`, masked mean pooling, L2 normalization, a 384-token limit and batch size 32. Long inputs are truncated. Scores with response-context angles at most `1e-6` radians are excluded; the recorded run excluded zero rows.

Embedding inference is run on a cloud GPU in Google Colab, with a reusable cache. Saved-score experiments do not repeat embedding inference. No generator is queried, classifier trained or embedding model fine-tuned in the current experiments.

## Dataset and task design

The only benchmark currently evaluated is [HaluEval](https://github.com/RUCAIBox/HaluEval). Each task-specific source item contributes two response rows: reference (`label=0`) and hallucinated (`label=1`).

| Task | q | c | Evaluated response rows |
| --- | --- | --- | ---: |
| QA | Source question | Provided knowledge | 20,000 |
| Dialogue | Entire conversation history | Provided knowledge | 20,000 |
| Summarization | Added instruction: `Summarize the document faithfully.` | Source document | 20,000 |
| General queries | User query | Unavailable | Excluded from SGI |

The 60,000 evaluated rows represent **30,000 source pairs**, not 60,000 independent questions. The preparation stage retained 4,507 general-query rows, but they lack context and are not scored. The fixed summarization instruction is this project's adaptation; it is not a question supplied by HaluEval and does not generate the provided summaries.

### Examples of q, c and r

The examples below come from the first item in each task at the recorded dataset revision. Each reference/hallucinated pair shares q and c. Whitespace and dash typography are normalized for display. The summarization context is an excerpt; inference reads the full document before token truncation.

### QA: qa_0

[Source file](https://github.com/RUCAIBox/HaluEval/blob/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json), first JSONL record.

**q**

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

**c**

```text
Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.
```

**r, reference, label 0**

```text
Arthur's Magazine
```

**r, hallucinated, label 1**

```text
First for Women was started first.
```

The question asks about magazine chronology. The reference names Arthur's Magazine; the negative selects First for Women. The reference is much shorter than the negative in this example, so response form is worth checking across the dataset. This single example does not establish a dataset-wide length effect.

### Dialogue: dialogue_0

[Source file](https://github.com/RUCAIBox/HaluEval/blob/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/dialogue_data.json), first JSONL record.

**q**

```text
[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.
```

**c**

```text
Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal
```

**r, reference, label 0**

```text
I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well.
```

**r, hallucinated, label 1**

```text
I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.
```

The history discusses Iron Man, Robert Downey Jr. and Zodiac. The knowledge also names Jake Gyllenhaal. The reference uses that name; the negative introduces Tom Hanks. The preference for crime movies is conversational content, not the factual error being illustrated.

### Summarization: summarization_0

[Source file](https://github.com/RUCAIBox/HaluEval/blob/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/summarization_data.json), first JSONL record.

**q**

```text
Summarize the document faithfully.
```

**c, opening excerpt only**

```text
Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was [...]
```

**r, reference, label 0**

```text
Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports. Journalists at Bild and Paris Match are "very confident" the video clip is real, an editor says. Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says.
```

**r, hallucinated, label 1**

```text
A video showing the final moments of Germanwings Flight 9525 has been recovered by investigators from the wreckage site. Marseille prosecutor Brice Robin urged anyone who might have more footage to turn it over immediately. Andreas Lubitz, the co-pilot accused of deliberately crashing the plane, had a history of severe depression and suicidal tendencies.
```

The full document reports a dispute over a claimed crash video: investigators had not confirmed it, while publications defended their report. The reference preserves that distinction. The negative presents recovery of a video by investigators as established. Both summaries still discuss the same event, people and topic. The q string was added by our notebook and is absent from the source record.

## Relation to the paper

The paper's SGI table reports HaluEval QA results. Its dialogue comparison uses a separate directional method, so our dialogue SGI result is not a replication of that number. Our summarization input is an additional experimental design choice.

| QA, all-mpnet-base-v2 | Paper | This experiment |
| --- | ---: | ---: |
| Mean SGI, grounded | 1.142 | 1.066184 |
| Mean SGI, hallucinated | 0.921 | 0.885615 |
| Cohen's d | 0.92 | 0.805753 |
| AUROC | 0.776 | 0.739861 |

Source: [paper, Section 6.1](https://arxiv.org/html/2602.13224v1#S6.SS1). The earlier [SGI implementation description](https://arxiv.org/html/2512.13771v1#S4.SS1) describes sampled evaluation, long-context segmentation and additive denominator epsilon. This experiment evaluates both responses for all QA source pairs, truncates input strings and excludes very small denominator angles. Exact historical sample selection and preprocessing are not matched.

Manual masked mean pooling and the 384-token limit follow the [encoder's model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers); neither alone is evidence of an error. The observed performance gap remains unexplained. Controlled comparisons and uncertainty estimates are needed before assigning a cause.

## Experiments

**Baseline:** compute class means, Cohen's d and AUROC separately for each task and for the pooled data.

**Fixed cutoffs:** the first sweep covered 1.00-2.00. The next sweep covers **0.70-1.40 in steps of 0.05**. Predict hallucination when `SGI < threshold`; equality is grounded. Compare accuracy changes against threshold 1.00.

**Class-mean midpoints:** estimate `(mean_sgi_grounded + mean_sgi_hallucinated) / 2` for each task. Evaluate it at full precision, including values outside the grid. Keep the original below-threshold rule as the main comparison. A separate diagnostic uses an above-threshold rule when the hallucinated mean is higher. Both cutoff and diagnostic direction use labels from the same data on which they are evaluated.

A midpoint is a heuristic, not a guaranteed optimum. It ignores dispersion, extreme values, class priors and error costs. Selecting the best observed grid point also does not establish independent test performance.

## Project structure

| File or directory | Purpose |
| --- | --- |
| [Hallucination_detection_SGI.ipynb](Hallucination_detection_SGI.ipynb) | Dataset preparation, embeddings and SGI scoring |
| [SGI_threshold_evaluation.ipynb](SGI_threshold_evaluation.ipynb) | Saved-score evaluation entry point |
| [scripts/analyze_sgi.py](scripts/analyze_sgi.py) | Reusable metrics and report generation |
| [ANALYSIS.md](ANALYSIS.md) | Measured results and interpretation |
| [PROGRESS.md](PROGRESS.md) | Experiment milestones and open research tasks |
| [results/HaluEval/](results/HaluEval/) | Numerical exports, score hashes and run configurations |
| [requirements-analysis.txt](requirements-analysis.txt) | Dependencies for saved-score evaluation |

## Run the experiments

For embedding generation, open `Hallucination_detection_SGI.ipynb` in Google Colab, select a GPU runtime and run the preparation, encoding and scoring sections. The notebook records dataset revision, preprocessing settings and dependency versions.

For saved-score evaluation, install the analysis dependencies in a Python environment:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_sgi.py
```

The input is `results/HaluEval/sgi_scores.parquet`, which is not versioned. Alternatively, supply `--scores PATH`. The analysis script writes CSVs and configuration files under `results/HaluEval/thresholds_0.70_1.40/` and updates the single narrative report, `ANALYSIS.md`. The evaluation notebook runs the same analysis without repeating inference. Use `--no-update-report` to generate numerical exports without updating the report. Grid settings can be changed with `--start`, `--stop` and `--step`; the grid must include reference threshold 1.00.

## Reading the metrics

Hallucination is the positive class. The confusion matrix distinguishes detected hallucinations (TP), grounded responses incorrectly flagged (FP), grounded responses accepted (TN), and missed hallucinations (FN).

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Mean SGI by class | Average score within each label | Describes class location; sensitive to extreme ratios |
| Cohen's d | `(mean_grounded - mean_hallucinated) / pooled_sample_SD` | Difference in units of pooled standard deviation; positive values support the assumed direction |
| AUROC | Area under the ROC curve using `-SGI` | Ranking across all cutoffs: 0.5 is chance-level, 1 is perfect; below 0.5 indicates opposite ranking on these data |
| Accuracy | `(TP + TN) / N` | Fraction of decisions matching the labels at one cutoff |
| Precision | `TP / (TP + FP)` | Fraction of flagged responses that are hallucinated |
| Recall | `TP / (TP + FN)` | Fraction of hallucinations detected; also called true positive rate |
| F1 | `2TP / (2TP + FP + FN)` | Harmonic mean of precision and recall; does not include TN |
| False positive rate | `FP / (FP + TN)` | Fraction of grounded responses incorrectly flagged |
| Predicted hallucination rate | `(TP + FP) / N` | Fraction of responses flagged, whether correctly or not |
| Accuracy change | `100 * (accuracy(t) - accuracy(1.00))` | Difference from the reference cutoff, in percentage points |

A ROC curve plots recall against false positive rate as the cutoff moves through the observed scores. AUROC also equals the probability that a random hallucinated response receives a higher `-SGI` than a random grounded response, with ties counting as half. **AUROC 0.74 does not mean 74% classification accuracy.** Changing the cutoff changes decisions but leaves AUROC unchanged for the same scores and direction.

Cohen's d compares the mean gap with score dispersion. If two class means differ by 0.2 and their pooled standard deviation is 0.1, d is 2; if it is 1.0, d is 0.2. Absolute values of 0.2, 0.5 and 0.8 are conventional small, medium and large references, not performance targets. The sign records the direction. Extreme scores can inflate the standard deviation and make d small despite a large mean gap.

The exports include source rows, evaluated rows, exclusions and confusion counts. Nonfinite scores and nonbinary labels are excluded. Undefined ratios are `NaN`. Accuracy CSVs use fractions; report tables show percentages. Always predicting either class gives 50% accuracy on the balanced recorded task subsets, so accuracy should be read alongside recall and false alarms.

## Research limitations

- The current results are descriptive. Thresholds and directions have not been validated on separate test pairs, and no bootstrap confidence intervals are available.
- Both responses from each `pair_id` must stay together in future splits and resampling.
- Truncation can remove evidence from long documents. Its impact on the metric gap has not been measured.
- Dialogue history is a different anchor from a focused question; a factual error can remain close to the conversation topic.
- The synthetic summarization instruction changes the meaning of the SGI numerator. Its contribution to the reversed ranking remains untested.
- Extreme denominator ratios may distort means and midpoint thresholds, particularly for dialogue.
- HaluEval's generated negatives can contain length or style artifacts. Benchmark separation does not establish general factuality detection.
- The initial run records package versions and dataset revision, but does not pin the encoder revision.

The next experiments and completed milestones are tracked in [PROGRESS.md](PROGRESS.md).

## References

- Javier Marín. [*A Geometric Taxonomy of Hallucinations in LLMs*, v1](https://arxiv.org/html/2602.13224v1).
- Javier Marín. [*Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems*, v1](https://arxiv.org/html/2512.13771v1).
- Junyi Li et al. [*HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*](https://arxiv.org/abs/2305.11747). [Dataset repository](https://github.com/RUCAIBox/HaluEval).
