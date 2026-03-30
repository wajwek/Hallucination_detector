# Hallucination_detector
Free-time project focused on building a tool that helps detect LLMs hallucinations using various methods.

## What Is This?
This repo contains a modular hallucination detector for RAG-style datasets (question, context, answer, optional
ground-truth, optional label). v1 focuses on **geometric methods**:
- SGI (Semantic Grounding Index)
- DC (Directional Consistency)
- CAI (Context Alignment Index)
- Cluster-based hallucination score (pragmatic v1)

It supports:
- persistent caching of embeddings/vectors
- saving every run with config + metrics + predictions
- Streamlit GUI for interactive experimentation

Artifacts are written to `./artifacts/` (ignored by git).

## Requirements
- Windows / CPU is fine
- Python 3.11+

Note: on this machine Python was not detected in PATH earlier. Make sure you can run `python --version`.

## Install (recommended: venv)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run CLI
```powershell
python -m hd.cli run --dataset path\to\data.csv --methods sgi dc cai --agg hard
```

## Run GUI (Streamlit)
```powershell
streamlit run hd/gui/app.py
```

## HaluEval
Raw HaluEval files (e.g. QA columns: `knowledge`, `question`, `right_answer`, `hallucinated_answer`) don’t contain a `label`
column, so metrics like accuracy/F1 will show as `None` unless you convert / add labels.

- CLI conversion:
  ```powershell
  hd prepare-halueval --input-path path\\to\\qa.json --output-path artifacts\\halueval_qa_canonical.jsonl --task auto --include-right-answers true
  ```
- GUI conversion: upload the raw file in the **Dataset** tab and click **Convert HaluEval to canonical dataset**.
