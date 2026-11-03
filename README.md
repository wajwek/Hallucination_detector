# Geometric Hallucination Detection with SGI

An independent reproduction attempt exploring whether the **Semantic Grounding Index (SGI)** can distinguish grounded and hallucinated responses on **HaluEval** using sentence-embedding geometry.

The project is inspired by Javier Marín's [*A Geometric Taxonomy of Hallucinations in LLMs* (2026)](https://arxiv.org/html/2602.13224v1), particularly its context-based SGI experiments. The goal is to investigate the proposed signal with a transparent notebook pipeline. This is an exploratory implementation, with reproduction of detection performance still pending.

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

## Pipeline

1. Set up Google Colab, mount Google Drive and initialize random seeds.
2. Download HaluEval and record the source revision.
3. Normalize whitespace, expand response pairs and export CSV/Parquet data.
4. Select complete labeled triplets.
5. Encode questions, contexts and responses independently with `sentence-transformers/all-mpnet-base-v2`, attention-mask-aware mean pooling and L2 normalization.
6. Cache embeddings with input/configuration checks.
7. Compute angular distances and SGI, then export per-subset and pooled descriptive metrics.

The encoder uses a **384-token limit** and **batch size 32**. Long texts are truncated, which can remove relevant evidence. No generator is queried, embedding model fine-tuned or classifier trained. The new evaluation reports mean SGI by class, Cohen's d and AUROC. Since lower SGI is expected for hallucinations, AUROC uses **−SGI**, with label 1 as the positive class. Near-zero response–context angles are excluded from ratio evaluation and counted explicitly.

## Results recorded so far

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

**No SGI scores, AUROC, accuracy or effect sizes were recorded in the supplied notebook.** Consequently, no detection-performance result can currently be claimed for this implementation. The SGI calculation and evaluation cells were added during cleanup and have not yet been run on HaluEval. Historical execution outputs were cleared, and the evidence above was preserved as a written record.

For comparison only, the paper's Table 1 reports the following **published HaluEval QA results**, which are not results produced by this project:

| Encoder | Mean SGI: valid | Mean SGI: hallucinated | Cohen's d | AUROC |
| --- | ---: | ---: | ---: | ---: |
| all-mpnet-base-v2 | 1.142 | 0.921 | 0.92 | 0.776 |

Source: [Marín, Section 6.1, Table 1](https://arxiv.org/html/2602.13224v1#S6.SS1). This is a QA reference point; it should not be compared directly with a pooled result across all three tasks.

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
    └── sgi_metrics.csv
```

Edit `PROJECT_ROOT` to change the output location. The revised cache namespace uses content and configuration validation; it intentionally recomputes legacy caches that lack this provenance. Existing dataset checkouts are reused, and their Git revision is recorded.

## Understanding the evaluation metrics

The notebook treats **hallucination as the positive class** (`label=1`). Its final table describes how SGI separates the HaluEval labels; it does not yet report the percentage of correct binary predictions.

### Metrics currently computed

| Output column | Meaning and interpretation |
| --- | --- |
| `mean_sgi_grounded` | Average SGI for reference responses (`label=0`). The hypothesis predicts a higher value than for hallucinated responses. |
| `mean_sgi_hallucinated` | Average SGI for hallucinated responses (`label=1`). A lower mean supports the expected direction of the signal, but does not establish reliable individual predictions. |
| `cohens_d` | Difference between the grounded and hallucinated mean SGI, divided by their pooled sample standard deviation. Positive values support the hypothesis; negative values indicate the opposite direction. Conventional absolute magnitudes of 0.2, 0.5 and 0.8 are often called small, medium and large, but are only rough guides. This is an effect size, not an accuracy percentage. |
| `auroc` | Area under the receiver operating characteristic curve, using `-SGI` as the hallucination score. It measures how often a randomly selected hallucinated response receives a higher hallucination score than a randomly selected grounded response, with ties counting as half. A value of 0.5 indicates chance-level ranking, 1.0 perfect ranking, and below 0.5 ranking opposite to the assumed direction. An AUROC of 0.8 does **not** mean 80% of binary decisions are correct. |
| `rows` | Number of complete labeled triplets supplied to evaluation for the subset. |
| `excluded_rows` | Rows excluded from SGI metrics because the response–context angle is at most `1e-6` radians, making the ratio unstable. |
| `grounded_rows`, `hallucinated_rows` | Number of usable responses in each class after those exclusions. |

Class means, Cohen's d and AUROC use only rows with a valid SGI. Metrics without enough usable observations or class variation are reported as missing (`NaN`). Compare the QA row with the paper's QA results; the `all` row pools different tasks and can hide differences between them.

### Metrics requiring a binary decision threshold

The following metrics are **not currently computed**. To obtain them, first choose a threshold `t` and predict hallucination when `SGI < t`. Select the threshold on a development set, then evaluate it on a separate test set, keeping both responses from each `pair_id` in the same split.

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

Undefined ratios need an explicit reporting convention, such as `NaN`. Changing the threshold trades missed hallucinations against false alarms. These metrics measure agreement with HaluEval annotations, not universal factual correctness, and their values can change with the class balance and task.

## Interpretation and limitations

The evaluation is descriptive over the available labeled data. It does not fit a threshold or establish held-out classifier performance. SGI = 1 has a geometric interpretation but is not a validated decision threshold. Reference and hallucinated responses share source items, so any future data split or bootstrap must group them by `pair_id`.

Results depend on the encoder, text length, task and benchmark construction. HaluEval's generated hallucinations may carry stylistic artifacts, and separation on this benchmark would not establish general factuality detection. In particular, this project has not reproduced the paper's performance claims, evaluated transfer, or supplied confidence intervals. The most direct next experiment is to execute the QA evaluation and compare its metrics with the published reference under documented preprocessing.

## References

- Javier Marín. [*A Geometric Taxonomy of Hallucinations in LLMs*](https://arxiv.org/html/2602.13224v1). arXiv:2602.13224v1, 2026.
- Junyi Li et al. [*HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*](https://arxiv.org/abs/2305.11747), 2023. [Dataset and source code](https://github.com/RUCAIBox/HaluEval).
