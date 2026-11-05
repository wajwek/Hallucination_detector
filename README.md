# Geometric Hallucination Detection with SGI

An experimental study of whether the **Semantic Grounding Index (SGI)** can distinguish grounded and hallucinated responses on **HaluEval** using sentence-embedding geometry.

The project is based on Javier Marín's [*A Geometric Taxonomy of Hallucinations in LLMs* (2026)](https://arxiv.org/html/2602.13224v1), particularly its context-based SGI experiments. The goal is to investigate the proposed signal with a transparent notebook pipeline. The first Colab evaluation is complete: SGI shows useful separation on QA, while dialogue and summarization do not support the same scoring direction. It is not a full replication: we evaluate both responses for every QA source pair, use our own long-context and denominator handling, and extend SGI to dialogue and summarization. The synthetic summarization instruction is our addition. These choices may affect the results, but their effects have not yet been isolated.

## Research idea

A response can sound relevant to a question while failing to use the supplied evidence. SGI asks whether its embedding is closer to the context than to the question. It requires three texts: a question **q**, a context **c**, and a response **r**.

For L2-normalized embeddings, angular distance and SGI are:

```text
theta(a, b) = arccos(clip(embedding(a) · embedding(b), -1, 1))
SGI(r; q, c) = theta(r, q) / theta(r, c)
```

An SGI above 1 means the response is angularly closer to the context; below 1 means it is closer to the question. The hypothesis is that grounded responses tend to move toward the context's semantic content, while unfaithful responses remain closer to the question. Whether this separates the labels must be measured empirically.

This is a measure of semantic grounding, not proof of factual correctness. A plausible factual error can occupy almost the same embedding region as a correct answer. The cited paper also develops a context-free Directional Grounding Index; that method is outside this project's scope.

## HaluEval data

The notebook downloads the [official HaluEval repository](https://github.com/RUCAIBox/HaluEval) and converts its task-specific reference/hallucinated response pairs into individual labeled rows. Labels are **0 = reference/grounded** and **1 = hallucinated**.

| Subset | Question input | Context input | Response inputs |
| --- | --- | --- | --- |
| QA | Question | Provided knowledge | Right and hallucinated answers |
| Dialogue | Dialogue history | Provided knowledge | Right and hallucinated responses |
| Summarization | Fixed instruction: `Summarize the document faithfully.` | Source document | Right and hallucinated summaries |
| General queries | User query | Unavailable | Annotated ChatGPT response |

Only rows with nonempty question, context and response and a binary label enter the SGI pipeline. General queries are retained in the normalized export but excluded from SGI because they lack context. The synthetic summarization question makes its geometry different from natural QA; results should be reported separately.

## Example inputs

Each source item supplies two triplets with the same q and c: one reference response and one hallucinated response. The following records come from the first item of each task at the dataset revision used in the run. Whitespace and dash typography are normalized for display. The summarization context is only an excerpt; the notebook reads the complete document before applying its token limit. No individual SGI values are claimed for these examples.

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

## Pipeline

1. Set up Google Colab, mount Google Drive and initialize random seeds.
2. Download HaluEval and record the source revision.
3. Normalize whitespace, expand response pairs and export CSV/Parquet data.
4. Select complete labeled triplets.
5. Encode questions, contexts and responses independently with `sentence-transformers/all-mpnet-base-v2`, attention-mask-aware mean pooling and L2 normalization.
6. Cache embeddings with input/configuration checks.
7. Compute angular distances and SGI, then export per-subset and pooled descriptive metrics.

The encoder uses a **384-token limit** and **batch size 32**. Long texts are truncated, which can remove relevant evidence. No generator is queried, embedding model fine-tuned or classifier trained. The evaluation reports mean SGI by class, Cohen's d and AUROC. A separate threshold sweep computes accuracy, precision, recall, F1 and confusion counts for each task and the pooled data. Since lower SGI is expected for hallucinations, AUROC uses **−SGI**, with label 1 as the positive class. Near-zero response-context angles are excluded from ratio evaluation and counted explicitly.

## Results

### Historical data preparation

The supplied notebook's original saved outputs establish the following preparation progress:

| Stage/subset | Recorded response rows |
| --- | ---: |
| QA | 20,000 |
| Dialogue | 20,000 |
| Summarization | 20,000 |
| General queries | 4,507 |
| Total normalized rows | 64,507 |
| Complete triplets selected for embeddings | 60,000 |

The 60,000 task-specific rows represent 30,000 source items, each expanded into a reference and a hallucinated response. They are not 60,000 independent questions. The general-query count is the retained count printed by the original run, not the advertised size of that subset.

The original embedding cell reported a **cache hit** for all 60,000 complete triplets using `all-mpnet-base-v2`. The cache resides on Google Drive and is not included in this repository; its contents have not been independently verified here.

### First Colab evaluation: November 4, 2026

The revised pipeline was executed on 60,000 complete HaluEval triplets. Each task contains 10,000 grounded and 10,000 hallucinated responses, with **zero rows excluded** by the near-zero denominator check.

| Subset | Rows | Mean SGI: grounded | Mean SGI: hallucinated | Cohen's d | AUROC using -SGI |
| --- | ---: | ---: | ---: | ---: | ---: |
| QA | 20,000 | 1.0662 | 0.8856 | 0.8058 | 0.7399 |
| Dialogue | 20,000 | 2.1429 | 1.0571 | 0.0189 | 0.4745 |
| Summarization | 20,000 | 2.5009 | 2.6832 | -0.3135 | 0.4084 |
| All tasks | 60,000 | 1.9033 | 1.5420 | 0.0109 | 0.5397 |

Full-precision exports are committed as [sgi_metrics.csv](results/HaluEval/sgi_metrics.csv), [run_config.json](results/HaluEval/run_config.json) and [environment.json](results/HaluEval/environment.json). They were read from the Colab output folder on Google Drive. Embedding arrays remain on Drive. The per-response score file is now available locally and ignored by Git. Local checks reproduced the recorded per-task AUROC and class means from those saved scores; embeddings were not recomputed. The local notebook now includes saved outputs from the completed Colab evaluation and threshold sweep.

The recorded run used `all-mpnet-base-v2`, a 384-token limit, batch size 32, manual mean pooling and L2 normalization. The HaluEval source revision is `b7253db3cdaa0ab2c382f92b26b390109174f77e`; the configuration also records the input hash, seed, exclusion tolerance and installed dependency versions. The embedding model revision was not pinned.

**QA supports the expected SGI signal.** Grounded responses have a higher average SGI, and AUROC is approximately 0.740. This is evidence of useful ranking on this subset, not 74% classification accuracy.

**The signal does not generalize uniformly across tasks.** Dialogue ranking is slightly below chance in the assumed direction. Summarization reverses that direction: hallucinated summaries have a higher mean SGI and AUROC is below 0.5. The fixed summarization instruction is one possible contributor to investigate, not an established explanation. The pooled AUROC of 0.540 gives little evidence of useful ranking across the combined tasks.

**Dialogue needs a distribution-level investigation.** Its mean SGI gap is large relative to its tiny Cohen's d, suggesting substantial dispersion or extreme ratios. A near-zero exclusion count of zero does not rule out small denominators above the cutoff. Quantiles, maximum scores and response-context angles should be inspected before interpreting class means. The summary table alone cannot establish the cause.

These are descriptive results from a single run. Confidence intervals, fitted decision thresholds and held-out binary classification metrics are still pending.

For comparison only, the paper's Table 1 reports the following **published HaluEval QA results**, which are not results produced by this project:

| Encoder | Mean SGI: valid | Mean SGI: hallucinated | Cohen's d | AUROC |
| --- | ---: | ---: | ---: | ---: |
| all-mpnet-base-v2 | 1.142 | 0.921 | 0.92 | 0.776 |

Source: [Marín, Section 6.1, Table 1](https://arxiv.org/html/2602.13224v1#S6.SS1). This is a QA reference point; it should not be compared directly with a pooled result across all three tasks.

The project's QA AUROC is **0.0361 lower** than the published 0.776, and Cohen's d is 0.8058 versus 0.92. The direction agrees, but the cause of the gap remains unresolved. The earlier [SGI implementation description](https://arxiv.org/html/2512.13771v1#S4.SS1) describes a sampled evaluation, segmentation of long contexts and an additive denominator epsilon. Our run uses all QA pairs, truncates strings and excludes very small denominator angles. Exact historical row selection and preprocessing are not fully matched.

Manual masked mean pooling and the 384-token limit agree with the [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers). Neither is evidence of an implementation error on its own. The dialogue and summarization runs extend the paper's reported QA SGI comparison. No synthetic instruction is used for dialogue, and the summarization instruction does not generate the dataset's summaries.

[ANALYSIS.md](ANALYSIS.md) records approaches, threshold comparisons and conclusions. [PROGRESS.md](PROGRESS.md) tracks completed work and open questions.

## Run analysis locally in VS Code and WSL

Embedding generation stays in the Colab notebook. Saved-score analysis runs locally on CPU using [SGI_analysis_local.ipynb](SGI_analysis_local.ipynb) or [scripts/analyze_sgi.py](scripts/analyze_sgi.py). The local path does not import PyTorch, Transformers or Google Colab.

Create a Python environment and install the analysis dependencies once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
```

The input is `results/HaluEval/sgi_scores.parquet`. This local data file is ignored by Git. It must be available before running analysis; the aggregate metrics CSV cannot replace individual scores.

Open the local notebook in a VS Code window connected to WSL and select the `.venv/bin/python` interpreter. Run all cells to calculate the threshold grid and midpoint decisions. The notebook retains output tables when saved.

The same analysis can run from the repository terminal:

```bash
.venv/bin/python scripts/analyze_sgi.py
```

The command writes CSVs and the run report under `results/HaluEval/thresholds_0.70_1.40/` and updates the root `ANALYSIS.md` directly. The first previous root report is preserved in that output folder. Use `--no-update-report` to leave the root report unchanged, or `--scores PATH` to read a different saved-score file. Changing the grid is supported with `--start`, `--stop` and `--step`; it must include the reference threshold 1.00.

## Run the notebook

Open [Hallucination_detection_SGI.ipynb](Hallucination_detection_SGI.ipynb) in Google Colab, select a GPU runtime and run the cells in order. Authorize Google Drive mounting when prompted. CPU execution is supported but slower.

The setup cell installs `transformers`, `torch`, `numpy`, `pandas`, `pyarrow`, `scikit-learn` and `tqdm`. Internet access is required for the dataset and model downloads. Dependencies are not pinned; installed package versions are exported with each run.

By default, artifacts are written under:

```text
/content/drive/MyDrive/Hallucination_Project/
├── data/raw/HaluEval/
├── data/processed/by_dataset/HaluEval/
│   ├── native.csv
│   └── native.parquet
├── embeddings/manual_meanpool_l2_v2/
│   └── sentence-transformers_all-mpnet-base-v2/HaluEval/complete_triplets/
│       ├── question.npy
│       ├── context.npy
│       ├── response.npy
│       ├── metadata.parquet
│       └── config.json
└── results/HaluEval/
    ├── environment.json
    ├── run_config.json
    ├── sgi_scores.parquet
    ├── sgi_metrics.csv
    ├── threshold_metrics.csv
    ├── threshold_accuracy.csv
    ├── threshold_accuracy_change_pp.csv
    ├── threshold_sweep_config.json
    └── ANALYSIS.md
```

Edit `PROJECT_ROOT` to change the output location. The revised cache namespace uses content and configuration validation; it intentionally recomputes legacy caches that lack this provenance. Existing dataset checkouts are reused, and their Git revision is recorded.

### Run only the threshold comparison

Once `sgi_scores.parquet` exists on Drive, mount Drive and run section 6, **Compare SGI thresholds**. In a fresh runtime install `numpy`, `pandas` and `pyarrow` first. Set `PROJECT_ROOT` in that cell to your project folder. No embedding model or GPU is needed for this section.

The default grid is **0.70, 0.75, ..., 1.40**, with both endpoints included. The prediction rule is **hallucination if SGI < threshold**; a score equal to the threshold is classified as grounded. The same direction is used for all tasks. The upper endpoint is an exploratory setting, not a theoretical limit. It includes thresholds below 1.00. Accuracy changes still use 1.00 as the reference.

The cell prints a task-by-threshold accuracy table and changes from threshold 1.00 in percentage points. It exports detailed metrics, the source-score hash and a Markdown report.

The completed historical 1.00-2.00 comparison is available in [ANALYSIS.md](ANALYSIS.md). The revised grid and midpoint decision metrics have now been evaluated locally and are recorded in ANALYSIS.md. The sweep compares fixed cutoffs on the available data. It does not select or validate a threshold on a separate test set.

## Understanding the evaluation metrics

The notebook treats **hallucination as the positive class** (`label=1`). The SGI summary measures ranking and class separation. Section 6 computes descriptive decision metrics for the fixed grid and class-mean midpoint thresholds. Historical and revised sweep results are saved in CSV, and the local notebook includes execution outputs.

### Ranking and separation metrics

| Output column | Meaning and interpretation |
| --- | --- |
| `mean_sgi_grounded` | Average SGI for reference responses (`label=0`). The hypothesis predicts a higher value than for hallucinated responses. |
| `mean_sgi_hallucinated` | Average SGI for hallucinated responses (`label=1`). A lower mean supports the expected direction of the signal, but does not establish reliable individual predictions. |
| `cohens_d` | Difference between the grounded and hallucinated mean SGI, divided by their pooled sample standard deviation. Positive values support the hypothesis; negative values indicate the opposite direction. Conventional absolute magnitudes of 0.2, 0.5 and 0.8 are often called small, medium and large, but are only rough guides. This is an effect size, not an accuracy percentage. |
| `auroc` | Area under the receiver operating characteristic curve, using `-SGI` as the hallucination score. It measures how often a randomly selected hallucinated response receives a higher hallucination score than a randomly selected grounded response, with ties counting as half. A value of 0.5 indicates chance-level ranking, 1.0 perfect ranking, and below 0.5 ranking opposite to the assumed direction. An AUROC of 0.8 does **not** mean 80% of binary decisions are correct. |
| `rows` | Number of complete labeled triplets supplied to evaluation for the subset. |
| `excluded_rows` | Rows excluded from SGI metrics because the response-context angle is at most `1e-6` radians, making the ratio unstable. |
| `grounded_rows`, `hallucinated_rows` | Number of usable responses in each class after those exclusions. |

Class means, Cohen's d and AUROC use only rows with a valid SGI. Metrics without enough usable observations or class variation are reported as missing (`NaN`). Compare the QA row with the paper's QA results; the `all` row pools different tasks and can hide differences between them.

### Class-mean midpoint thresholds

Section 6 also estimates a separate threshold for each task as `(mean_sgi_grounded + mean_sgi_hallucinated) / 2`. It evaluates this threshold at full precision, even when it falls outside the fixed grid. The original below-threshold rule remains the main comparison. A second, explicitly labeled diagnostic uses the direction of the class means; reversed means lead to an above-threshold rule. Equality is grounded in both policies.

A midpoint is a heuristic, not a guaranteed optimum. Means can be distorted by extreme scores, and unequal distributions need not be best separated at their midpoint. These thresholds and diagnostic directions are estimated and evaluated on the same data. They do not establish held-out performance or change the baseline AUROC.

New grid and midpoint exports are saved under `results/HaluEval/thresholds_0.70_1.40/`, including `midpoint_estimates.csv`, `midpoint_metrics.csv` and an updated analysis report. Existing parent-folder exports preserve the earlier run.

### How AUROC works

For a chosen cutoff, recall or true positive rate is `TP / (TP + FN)`. The false positive rate is `FP / (FP + TN)`: the fraction of grounded responses incorrectly flagged as hallucinations. A ROC curve plots true positive rate against false positive rate as the cutoff moves across the observed scores. AUROC is the area under this curve.

Our hallucination score is `-SGI`, so a lower SGI means stronger predicted hallucination. AUROC uses the ranking across the full score range, not just the 0.70 to 1.40 grid. An AUROC of 0.74 means a random hallucinated response outranks a random grounded response about 74% of the time, counting ties as half. It does not mean 74% of responses are correctly classified at threshold 1.00.

Changing the decision threshold leaves AUROC unchanged for the same scored rows. It changes accuracy and the confusion matrix. Below-chance AUROC suggests an opposite ranking direction on the evaluated data; it does not establish that reversing the score will generalize to new data.

### Metrics requiring a binary decision threshold

Section 6 now computes the following metrics at each fixed threshold `t`, predicting hallucination when `SGI < t`. These results are descriptive until a threshold is selected on development data and evaluated on a separate test set. Keep both responses from each `pair_id` in the same split.

The confusion matrix counts the four possible outcomes:

| Actual HaluEval label | Predicted hallucination | Predicted grounded |
| --- | --- | --- |
| Hallucinated | True positive (TP): hallucination detected | False negative (FN): hallucination missed |
| Grounded | False positive (FP): false alarm | True negative (TN): grounded response accepted |

These counts define the decision metrics:

| Metric | Formula | What it answers |
| --- | --- | --- |
| Precision | `TP / (TP + FP)` | Of the responses flagged as hallucinations, what fraction actually have the hallucination label? |
| Recall | `TP / (TP + FN)` | Of all labeled hallucinations, what fraction did the detector catch? |
| F1 | `2TP / (2TP + FP + FN)` | What is the harmonic mean of precision and recall? It penalizes an imbalance between them and does not account for true negatives. |
| Accuracy | `(TP + TN) / (TP + FP + FN + TN)` | What fraction of all binary decisions match the labels? It can be misleading when one class dominates. |
| False positive rate | `FP / (FP + TN)` | What fraction of grounded responses are incorrectly flagged? This is the horizontal axis of the ROC curve. |
| Predicted hallucination rate | `(TP + FP) / N` | What fraction of usable responses are flagged, regardless of whether the flag is correct? |
| Accuracy change | `100 * (accuracy(t) - accuracy(1.00))` | How many percentage points accuracy changes from the starting threshold. This is not a relative percentage improvement. |

Undefined ratios are reported as `NaN`. Accuracy exports use fractions from 0 to 1; Markdown tables show percentages. The current task subsets are balanced, so always predicting either class would give 50% accuracy. Each threshold result includes evaluated and excluded row counts; nonfinite scores and invalid labels are excluded. Changing the threshold trades missed hallucinations against false alarms. These metrics measure agreement with HaluEval annotations, not universal factual correctness, and their values can change with the class balance and task.

## Interpretation and limitations

The evaluation is descriptive over the available labeled data. It compares prespecified cutoffs without fitting a threshold or establishing held-out classifier performance. SGI = 1 has a geometric interpretation but is not a validated decision threshold. Reference and hallucinated responses share source items, so any future data split or bootstrap must group them by `pair_id`.

Results depend on the encoder, text length, task and benchmark construction. HaluEval's generated hallucinations may carry stylistic artifacts, and separation on this benchmark would not establish general factuality detection. The first run supports the expected effect on QA but does not establish an exact reproduction or a universal detector. Transfer and confidence intervals have not been evaluated. The next priorities are inspecting score distributions, checking extreme ratios and adding paired bootstrap intervals before selecting and testing a binary decision threshold.

## TODOs

The next steps focus on completing and assessing the HaluEval experiment before expanding its scope.

- [x] **Run the revised notebook in Colab.** Generate or validate the embeddings, execute SGI evaluation and save the actual metrics with the dataset revision and run configuration.
- [x] **Document the first measured results.** Add per-subset means, Cohen's d and AUROC to this README. Compare QA with the paper's QA reference and explain any preprocessing differences rather than assuming an exact replication.
- [ ] **Investigate extreme SGI ratios.** Inspect dialogue quantiles, maximum scores and small response-context angles to explain the large mean gap alongside a tiny effect size. Report diagnostics before considering any change to exclusions.
- [ ] **Inspect the score distributions.** Plot SGI by class, ROC curves and the relationship between question-context angular distance and detection performance. Check whether class separation changes as the question and context become farther apart.
- [ ] **Add uncertainty estimates.** Compute bootstrap confidence intervals for AUROC and effect sizes, resampling source pairs together by `pair_id`.
- [x] **Implement a descriptive threshold sweep.** The historical 1.00-2.00 run is complete. The next grid covers 0.70-1.40 in steps of 0.05.
- [x] **Implement class-mean midpoint evaluation.** Calculate each midpoint and compare the fixed-below rule with a separately labeled mean-based direction diagnostic.
- [x] **Run the revised grid and midpoint evaluation.** Evaluated locally from the saved scores; recorded new decision metrics and verified agreement with overlapping historical thresholds.
- [x] **Run and review the threshold sweep.** Save its tables and replace the pending table in ANALYSIS.md with the Colab report. Check confusion counts, derived metrics and pooled results for consistency.
- [ ] **Evaluate binary decisions on held-out data.** Create development and test splits grouped by `pair_id`, choose a threshold on development data using a stated objective, then report test precision, recall, F1, accuracy and the confusion matrix. Do not choose the threshold using test labels.
- [ ] **Analyze failures.** Inspect false alarms and missed hallucinations, including plausible factual errors, unusually short responses and cases where evidence was truncated.
- [ ] **Compare with simple baselines.** Evaluate response-context cosine similarity and response-context angular distance on the same rows to establish whether the SGI ratio adds useful information.
- [ ] **Check robustness within HaluEval.** Compare encoders and token limits, report QA, dialogue and summarization separately, and keep test data out of model and parameter selection.
- [ ] **Improve reproducibility and runtime.** Pin dependencies and model revisions for reported experiments, measure embedding time and memory use, and consider encoding repeated texts once. Add resumable batches if interrupted runs become a practical problem.

## Encountered problems

### Slow embedding generation on CPU

Local CPU execution has been too slow for practical embedding generation in this project. The selected data contains 60,000 response rows, and the current pipeline encodes a question, context and response for every row. This makes the embedding stage substantially more demanding than calculating SGI from cached vectors.

For this reason, I use **Google Colab to run embedding inference on a cloud GPU**, with Google Drive storing the generated artifacts for reuse. The notebook still supports CPU execution, but the GPU workflow is the practical choice for the full run. No controlled CPU-versus-GPU timing benchmark has been recorded yet.

### Incompatible inputs across HaluEval subsets

SGI needs a question, context and response. General queries have no supplied context and cannot be evaluated with this formula. Summarization has no natural question, so the notebook uses a fixed instruction. These differences prevent treating every HaluEval subset as the same geometric task.

### Truncation and limits of the signal

The 384-token limit can remove evidence from long contexts. Separately, semantically plausible false claims may remain close to the context even when their details are wrong. Both are limitations to investigate through error analysis; their impact has not yet been measured in this project.

### Uneven task performance and unstable summaries

The first run gives a QA AUROC of 0.7399, but only 0.4745 for dialogue and 0.4084 for summarization under the same scoring direction. A single pooled metric obscures these differences. Dialogue also combines a large mean gap with Cohen's d of only 0.0189, which calls for inspection of dispersion, extreme ratios and denominator angles. No rows were excluded by the existing tolerance, and the cause has not yet been established from individual scores.

### From a score to a decision

The evaluation measures ranking, class separation and descriptive decisions at fixed cutoffs. It does not yet establish a validated threshold or held-out precision and recall. The paired responses also require grouped splits to avoid evaluating on another response from the same source item used during development. SGI = 1 should therefore not be presented as an empirically validated cutoff.

## References

- Javier Marín. [*A Geometric Taxonomy of Hallucinations in LLMs*](https://arxiv.org/html/2602.13224v1). arXiv:2602.13224v1, 2026.
- Junyi Li et al. [*HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*](https://arxiv.org/abs/2305.11747), 2023. [Dataset and source code](https://github.com/RUCAIBox/HaluEval).
