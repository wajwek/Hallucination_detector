from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import time
import shutil
from datetime import datetime

from hd.config import (
    AggregationConfig,
    AggregationKind,
    DatasetColumns,
    EmbeddingConfig,
    PoolingMode,
    ReductionConfig,
    ReductionKind,
    RunConfig,
)
from hd.engine import run_detection
from hd.storage import artifacts_root, list_runs
from hd.data.converters import convert_halueval_to_canonical, detect_halueval_task, write_canonical
from hd.utils import read_json


st.set_page_config(page_title="Hallucination Detector", layout="wide")

st.title("Hallucination Detector (RAG) v1")

_RUN_ID_TS_RE = re.compile(r"_(\d{8}_\d{6})_")


def _run_created_at_str(run: Dict[str, Any]) -> str:
    """Best-effort created-at string for History table.

    Primary source: timestamp embedded in run_id: <prefix>_YYYYMMDD_HHMMSS_<hash>
    Fallback: filesystem mtime of artifacts/runs/<run_id>/
    """

    run_id = str(run.get("run_id") or "")
    matches = _RUN_ID_TS_RE.findall(run_id)
    if matches:
        try:
            dt = datetime.strptime(matches[-1], "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    p = run.get("path")
    if p:
        try:
            dt = datetime.fromtimestamp(Path(p).stat().st_mtime)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return ""


def _delete_run_artifacts(run_id: str) -> Tuple[bool, str]:
    """Delete artifacts for a single run under artifacts/runs/<run_id> (safe-guarded)."""

    runs_dir = artifacts_root() / "runs"
    target = runs_dir / str(run_id)

    try:
        runs_dir_res = runs_dir.resolve()
        target_res = target.resolve()
    except Exception as exc:
        return False, f"Failed to resolve run path: {exc}"

    if runs_dir_res == target_res or runs_dir_res not in target_res.parents:
        return False, "Refusing to delete: run path is outside artifacts/runs"

    if not target.exists() or not target.is_dir():
        return False, f"Run not found: {run_id}"

    try:
        shutil.rmtree(target)
    except Exception as exc:
        return False, f"Failed to delete run {run_id}: {exc}"

    return True, f"Deleted run: {run_id}"


def _safe_rmtree_under(base_dir: Path, target: Path) -> Tuple[bool, str]:
    """Recursively delete a directory, but only if it is under base_dir."""

    try:
        base_res = base_dir.resolve()
        target_res = target.resolve()
    except Exception as exc:
        return False, f"Failed to resolve delete path: {exc}"

    if base_res == target_res or base_res not in target_res.parents:
        return False, "Refusing to delete: path is outside the allowed base directory"

    if not target.exists():
        return True, f"Already deleted: {target.name}"
    if not target.is_dir():
        return False, f"Not a directory: {target}"

    try:
        shutil.rmtree(target)
    except Exception as exc:
        return False, f"Failed to delete directory {target.name}: {exc}"

    return True, f"Deleted directory: {target.name}"


def _safe_unlink_under(base_dir: Path, target: Path) -> Tuple[bool, str]:
    """Delete a file, but only if it is under base_dir."""

    try:
        base_res = base_dir.resolve()
        target_res = target.resolve()
    except Exception as exc:
        return False, f"Failed to resolve delete path: {exc}"

    if base_res == target_res or base_res not in target_res.parents:
        return False, "Refusing to delete: path is outside the allowed base directory"

    if not target.exists():
        return True, f"Already deleted: {target.name}"
    if target.is_dir():
        return False, f"Not a file: {target}"

    try:
        target.unlink()
    except Exception as exc:
        return False, f"Failed to delete file {target.name}: {exc}"

    return True, f"Deleted file: {target.name}"


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip())
    cleaned = cleaned.strip("-").lower()
    return cleaned or "run"


def _format_num(value: float) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _slugify(str(value))
    txt = f"{num:.3f}".rstrip("0").rstrip(".")
    if txt == "-0":
        txt = "0"
    return txt.replace(".", "p")


def _short_model_name(name: str) -> str:
    if not name:
        return "model"
    base = str(name).split("/")[-1].split("\\")[-1]
    base = base.replace("sentence-transformers-", "")
    return base or "model"


def _auto_run_prefix(cfg: Dict[str, Any]) -> str:
    tokens = []

    methods = cfg.get("methods", {})
    sgi_cfg = methods.get("sgi", {}) or {}
    if sgi_cfg.get("enabled"):
        tokens.append(f"sgi{_format_num(sgi_cfg.get('threshold'))}")
    if methods.get("dc", {}).get("enabled"):
        tokens.append(f"dc{_format_num(methods['dc'].get('threshold'))}")
    if methods.get("cai", {}).get("enabled"):
        tokens.append(f"cai{_format_num(methods['cai'].get('threshold'))}")
    if methods.get("cluster", {}).get("enabled"):
        tokens.append(f"cluster{_format_num(methods['cluster'].get('threshold'))}")

    emb = cfg.get("emb", {})
    tokens.append(_short_model_name(emb.get("embedding_model", "model")))
    tokens.append(f"q{emb.get('question_pooling', 'mean')}")
    tokens.append(f"c{emb.get('context_pooling', 'mean')}")
    tokens.append(f"a{emb.get('answer_pooling', 'mean')}")
    tokens.append("norm" if emb.get("normalize", True) else "nonorm")
    rk = emb.get("reduction", "none")
    if rk == "none":
        tokens.append("none")
    else:
        tokens.append(f"{rk}{emb.get('n_components', 0)}")

    agg = cfg.get("agg", {})
    tokens.append(f"{agg.get('kind', 'hard')}{_format_num(agg.get('threshold_hall'))}")

    prefix = "-".join(_slugify(t) for t in tokens if t)
    if len(prefix) > 64:
        prefix = prefix[:64].rstrip("-")
    return prefix or "run"


def _save_upload(upload) -> Path:
    root = artifacts_root()
    up_dir = root / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    path = up_dir / upload.name
    path.write_bytes(upload.getvalue())
    return path


def _load_preview(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".jsonl", ".jsonlines"}:
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path, lines=False)
    raise ValueError(f"Unsupported extension: {ext}")


def _reset_column_mapping_widgets() -> None:
    for k in [
        "cfg_question_col",
        "cfg_context_col",
        "cfg_answer_col",
        "cfg_ground_truth_col",
        "cfg_label_col",
    ]:
        st.session_state.pop(k, None)
    st.session_state.pop("cfg", None)
    st.session_state.pop("cfg_valid", None)


def _apply_run_config(run_id: str) -> Tuple[bool, str]:
    cfg_path = artifacts_root() / "runs" / run_id / "run_config.json"
    if not cfg_path.exists():
        return False, f"Run config not found: {run_id}"
    try:
        data = read_json(cfg_path)
    except Exception as exc:
        return False, f"Failed to read run config: {exc}"

    config = data.get("config") or {}
    dataset_path = data.get("dataset_path")
    columns: Optional[list[str]] = None
    if dataset_path and Path(dataset_path).exists():
        st.session_state["dataset_path"] = dataset_path
        try:
            df = _load_preview(Path(dataset_path))
            columns = list(df.columns)
        except Exception:
            columns = None

    def _choose_column(value: Optional[str], allow_none: bool) -> Optional[str]:
        if columns:
            if value is None:
                return "(none)" if allow_none else columns[0]
            if value in columns:
                return value
            return "(none)" if allow_none else columns[0]
        if value is None and allow_none:
            return "(none)"
        return value

    dataset_cols = config.get("dataset_columns") or {}
    if columns:
        st.session_state["cfg_question_col"] = _choose_column(dataset_cols.get("question"), allow_none=False)
        st.session_state["cfg_context_col"] = _choose_column(dataset_cols.get("context"), allow_none=False)
        st.session_state["cfg_answer_col"] = _choose_column(dataset_cols.get("answer"), allow_none=False)
        st.session_state["cfg_ground_truth_col"] = _choose_column(dataset_cols.get("ground_truth"), allow_none=True)
        st.session_state["cfg_label_col"] = _choose_column(dataset_cols.get("label"), allow_none=True)

    embeddings = config.get("embeddings") or {}
    reduction = embeddings.get("reduction") or {}
    sgi = config.get("sgi") or {}
    dc = config.get("dc") or {}
    cai = config.get("cai") or {}
    cluster = config.get("cluster") or {}
    agg = config.get("aggregation") or {}
    weights = agg.get("weights") or {}

    st.session_state["cfg_embedding_model"] = embeddings.get(
        "embedding_model",
        st.session_state.get("cfg_embedding_model", "all-MiniLM-L6-v2"),
    )
    st.session_state["cfg_batch_size"] = int(
        embeddings.get("batch_size", st.session_state.get("cfg_batch_size", 32))
    )
    st.session_state["cfg_q_pool"] = embeddings.get(
        "question_pooling", st.session_state.get("cfg_q_pool", PoolingMode.mean.value)
    )
    st.session_state["cfg_c_pool"] = embeddings.get(
        "context_pooling", st.session_state.get("cfg_c_pool", PoolingMode.mean.value)
    )
    st.session_state["cfg_a_pool"] = embeddings.get(
        "answer_pooling", st.session_state.get("cfg_a_pool", PoolingMode.mean.value)
    )
    st.session_state["cfg_normalize"] = bool(
        embeddings.get("normalize_to_unit_sphere", st.session_state.get("cfg_normalize", True))
    )
    st.session_state["cfg_reduction_kind"] = reduction.get(
        "kind", st.session_state.get("cfg_reduction_kind", ReductionKind.none.value)
    )
    st.session_state["cfg_n_components"] = int(
        reduction.get("n_components", st.session_state.get("cfg_n_components", 128))
    )
    st.session_state["cfg_whitening"] = bool(
        reduction.get("whitening", st.session_state.get("cfg_whitening", False))
    )

    st.session_state["cfg_enable_sgi"] = bool(
        sgi.get("enabled", st.session_state.get("cfg_enable_sgi", True))
    )
    st.session_state["cfg_sgi_threshold"] = float(
        sgi.get("threshold", st.session_state.get("cfg_sgi_threshold", 0.3))
    )
    st.session_state["cfg_enable_dc"] = bool(
        dc.get("enabled", st.session_state.get("cfg_enable_dc", False))
    )
    st.session_state["cfg_dc_threshold"] = float(
        dc.get("threshold", st.session_state.get("cfg_dc_threshold", 0.6))
    )
    st.session_state["cfg_enable_cai"] = bool(
        cai.get("enabled", st.session_state.get("cfg_enable_cai", False))
    )
    st.session_state["cfg_cai_threshold"] = float(
        cai.get("threshold", st.session_state.get("cfg_cai_threshold", 0.1))
    )
    st.session_state["cfg_enable_cluster"] = bool(
        cluster.get("enabled", st.session_state.get("cfg_enable_cluster", False))
    )
    st.session_state["cfg_cluster_threshold"] = float(
        cluster.get("threshold", st.session_state.get("cfg_cluster_threshold", 0.5))
    )
    st.session_state["cfg_k_clusters"] = int(
        cluster.get("k_clusters", st.session_state.get("cfg_k_clusters", 5))
    )

    st.session_state["cfg_min_chunks"] = int(
        cluster.get("min_chunks", st.session_state.get("cfg_min_chunks", 3))
    )
    st.session_state["cfg_sentence_min_chars"] = int(
        cluster.get("sentence_min_chars", st.session_state.get("cfg_sentence_min_chars", 25))
    )

    st.session_state["cfg_agg_kind"] = agg.get(
        "kind", st.session_state.get("cfg_agg_kind", AggregationKind.hard.value)
    )
    st.session_state["cfg_threshold_hall"] = float(
        agg.get("threshold_hall", st.session_state.get("cfg_threshold_hall", 0.5))
    )

    st.session_state["cfg_w_sgi"] = float(weights.get("sgi", st.session_state.get("cfg_w_sgi", 1.0)))
    st.session_state["cfg_w_dc"] = float(weights.get("dc", st.session_state.get("cfg_w_dc", 1.0)))
    st.session_state["cfg_w_cai"] = float(weights.get("cai", st.session_state.get("cfg_w_cai", 1.0)))
    st.session_state["cfg_w_cluster"] = float(weights.get("cluster", st.session_state.get("cfg_w_cluster", 1.0)))

    st.session_state["run_prefix_manual"] = config.get(
        "run_prefix", st.session_state.get("run_prefix_manual", "exp")
    )

    if not columns:
        return True, f"Applied settings from {run_id}. Dataset columns were not updated."
    return True, f"Applied settings from {run_id}."



# Apply copied settings before widgets are instantiated (Streamlit restriction).
if "pending_apply_run_id" in st.session_state:
    run_id_to_apply = st.session_state.pop("pending_apply_run_id")
    _reset_column_mapping_widgets()
    ok, msg = _apply_run_config(run_id_to_apply)
    if ok:
        st.session_state["copy_notice"] = msg
    else:
        st.session_state["copy_error"] = msg

tab_dataset, tab_config, tab_run, tab_history = st.tabs(["Dataset", "Config", "Run", "History"])

with tab_dataset:
    st.subheader("Upload dataset")
    upload = st.file_uploader(
        "Dataset CSV/JSONL",
        type=["csv", "json", "jsonl", "jsonlines"],
        key="dataset_upload",
        help=(
            "Upload a dataset of already generated RAG answers. "
            "The tool expects (at minimum) `question`, `context`, and `answer` columns; "
            "`label` (0/1) is recommended and required to compute accuracy/F1/AUROC; "
            "`ground_truth` is optional and only needed if you enable DC. "
            "If you upload a raw HaluEval file, you can convert it to the canonical schema here."
        ),
    )

    raw_path = None
    if upload is not None:
        upload_size = getattr(upload, "size", None)
        prev_size = st.session_state.get("raw_upload_size")
        same_upload = (
            st.session_state.get("raw_upload_name") == upload.name
            and st.session_state.get("raw_dataset_path")
            and (
                upload_size is None
                or prev_size is None
                or int(prev_size) == int(upload_size)
            )
        )

        if not same_upload:
            raw_path = _save_upload(upload)
            st.session_state["raw_dataset_path"] = str(raw_path)
            st.session_state["raw_upload_name"] = upload.name
            st.session_state["raw_upload_size"] = upload_size
            st.session_state.pop("converted_dataset_path", None)
            st.session_state["dataset_is_converted"] = False
            st.session_state["dataset_path"] = str(raw_path)
            _reset_column_mapping_widgets()
        else:
            raw_path = Path(st.session_state["raw_dataset_path"])

        # Keep the selected dataset_path stable across Streamlit reruns.
        if not st.session_state.get("dataset_path"):
            st.session_state["dataset_path"] = str(raw_path)
            _reset_column_mapping_widgets()

        df_raw = _load_preview(raw_path)
        st.write("Preview (raw upload)")
        st.dataframe(df_raw.head(20))
        st.write({"rows": int(len(df_raw)), "columns": list(df_raw.columns)})

        detected_task = detect_halueval_task(df_raw)
        if detected_task is not None:
            st.info(
                f"Detected raw HaluEval ({detected_task}). "
                "Convert it to the canonical schema to get labels and proper metrics."
            )
            include_right = st.checkbox(
                "Include right answers (label=0) (recommended)",
                value=True,
                key="halueval_include_right",
                help=(
                    "HaluEval raw files often contain only hallucinated examples. "
                    "This option adds the paired right (non-hallucinated) samples as label=0, "
                    "so the dataset contains both classes (0 and 1). "
                    "That makes evaluation metrics like accuracy/F1/AUROC meaningful."
                ),
            )
            overwrite_converted = False
            if st.session_state.get("converted_dataset_path"):
                overwrite_converted = st.checkbox(
                    "Overwrite previous converted dataset",
                    value=False,
                    key="halueval_overwrite_converted",
                    help="Reuses and overwrites the previous converted canonical file instead of creating a new one.",
                )

            if st.button(
                "Convert HaluEval to canonical dataset",
                key="halueval_convert_btn",
                help=(
                    "Creates a new canonical JSONL file with unified column names "
                    "(`question`, `context`, `answer`, `ground_truth`, `label`). "
                    "Use this when your raw dataset has dataset-specific columns and you want reliable metrics. Label is auto-generated for HaluEval."
                ),
            ):
                out_df = convert_halueval_to_canonical(
                    df_raw,
                    task=detected_task,
                    include_right=bool(include_right),
                )
                if overwrite_converted and st.session_state.get("converted_dataset_path"):
                    out_path = Path(st.session_state["converted_dataset_path"])
                else:
                    out_path = artifacts_root() / "uploads" / f"{raw_path.stem}_canonical_{int(time.time())}.jsonl"

                write_canonical(out_df, out_path)
                st.session_state["dataset_path"] = str(out_path)
                st.session_state["converted_dataset_path"] = str(out_path)
                st.session_state["dataset_is_converted"] = True
                _reset_column_mapping_widgets()
                st.success(f"Converted dataset saved: {out_path.name}")

    active_path = st.session_state.get("dataset_path")
    if active_path:
        st.write({"active_dataset_path": str(active_path)})

        if raw_path is None and st.session_state.get("raw_dataset_path"):
            raw_path = Path(st.session_state["raw_dataset_path"])
        converted_path = st.session_state.get("converted_dataset_path")

        if converted_path and not st.session_state.get("dataset_is_converted"):
            if st.button(
                "Switch to converted dataset",
                key="switch_to_converted",
                help="Sets the active dataset to the converted canonical file (recommended for running/evaluation).",
            ):
                st.session_state["dataset_path"] = str(converted_path)
                st.session_state["dataset_is_converted"] = True
                _reset_column_mapping_widgets()

        if raw_path is not None and st.session_state.get("dataset_is_converted"):
            if st.button(
                "Switch to raw dataset",
                key="switch_to_raw",
                help="Sets the active dataset back to the original raw upload (useful for checking original columns).",
            ):
                st.session_state["dataset_path"] = str(raw_path)
                st.session_state["dataset_is_converted"] = False
                _reset_column_mapping_widgets()

        # Preview active dataset (useful after conversion)
        try:
            ap = Path(active_path)
            df_active = _load_preview(ap)
            st.write("Preview (active dataset)")
            st.dataframe(df_active.head(20))
            st.write({"rows": int(len(df_active)), "columns": list(df_active.columns)})
        except Exception:
            pass
with tab_config:
    st.subheader("Configuration")

    dataset_path = st.session_state.get("dataset_path")
    if not dataset_path:
        st.info("Upload a dataset first.")
    else:
        df0 = _load_preview(Path(dataset_path))
        cols = list(df0.columns)
        detected_task = detect_halueval_task(df0)
        if detected_task is not None and "label" not in cols:
            st.warning(
                "Raw HaluEval detected. For proper metrics with both classes (label=0/1), "
                "use the conversion button in the Dataset tab (recommended)."
            )

        st.markdown("### Dataset column mapping")
        default_context = "context"
        default_answer = "answer"
        default_gt = "ground_truth"

        if detected_task == "qa":
            default_context = "knowledge"
            default_answer = "hallucinated_answer"
            default_gt = "right_answer"
        elif detected_task == "dialogue":
            default_context = "knowledge"
            default_answer = "hallucinated_response"
            default_gt = "right_response"
        elif detected_task == "summarization":
            default_context = "document"
            default_answer = "hallucinated_summary"
            default_gt = "right_summary"
        elif detected_task == "general":
            default_context = "user_query"
            default_answer = "chatgpt_response"

        context_default_idx = cols.index(default_context) if default_context in cols else 0
        answer_default_idx = cols.index(default_answer) if default_answer in cols else 0
        gt_default_idx = (cols.index(default_gt) + 1) if default_gt in cols else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            question_col = st.selectbox(
                "question",
                options=cols,
                index=cols.index("question") if "question" in cols else 0,
                key="cfg_question_col",
                help=(
                    "Which column contains the user question/prompt. "
                    "This vector is used as the anchor for several geometric checks (for example DC uses directions from the question)."
                ),
            )
        with c2:
            context_col = st.selectbox(
                "context",
                options=cols,
                index=context_default_idx,
                key="cfg_context_col",
                help=(
                    "Which column contains the retrieved context for RAG. "
                    "It can be a single string or a list of documents; internally it is pooled into one context embedding per sample. "
                    "Most geometric methods compare the answer against this context embedding."
                ),
            )
        with c3:
            answer_col = st.selectbox(
                "answer",
                options=cols,
                index=answer_default_idx,
                key="cfg_answer_col",
                help=(
                    "Which column contains the generated model answer you want to evaluate for hallucinations. "
                    "This is the text that will be embedded and scored by SGI/DC/CAI/cluster."
                ),
            )
        with c4:
            ground_truth_col = st.selectbox(
                "ground_truth (optional)",
                options=["(none)"] + cols,
                index=gt_default_idx,
                key="cfg_ground_truth_col",
                help=(
                    "Optional: a reference/correct answer column. "
                    "If you enable DC (Direction Consistency), this must be provided, because DC compares the direction "
                    "from question->generated answer vs question->ground truth."
                ),
            )
        with c5:
            label_col = st.selectbox(
                "label",
                options=["(none)"] + cols,
                index=(cols.index("label") + 1) if "label" in cols else 0,
                key="cfg_label_col",
                help=(
                    "True hallucination label used for evaluation metrics (recommended). "
                    "Use 1 for hallucinated and 0 for non-hallucinated (or booleans). "
                    "If you leave this empty, the tool can still run detection, but accuracy/precision/recall/F1/AUROC will not be computed."
                ),
            )

        cfg_valid = True
        if question_col == context_col or question_col == answer_col or context_col == answer_col:
            cfg_valid = False
            st.error("Invalid mapping: `question`, `context`, and `answer` must be different columns.")

        if detected_task == "qa":
            if answer_col not in {"hallucinated_answer", "right_answer"}:
                st.warning(
                    "HaluEval QA: `answer` should usually be `hallucinated_answer` (hallucinations) "
                    "or `right_answer` (non-hallucinations)."
                )
            if context_col != "knowledge":
                st.warning("HaluEval QA: `context` should usually be `knowledge`.")

        if label_col == "(none)":
            st.info(
                "No `label` selected -> metrics (accuracy/precision/recall/F1/AUROC) will be empty. Convert HaluEval to canonical or provide a label column."
            )

        st.session_state["cfg_valid"] = bool(cfg_valid)
        st.markdown("### Embeddings")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            embedding_model = st.text_input(
                "embedding_model",
                value="all-MiniLM-L6-v2",
                key="cfg_embedding_model",
                help=(
                    "Sentence-Transformers / Hugging Face model id used to embed text into vectors. "
                    "All geometric methods operate on these embeddings, so changing the model can noticeably change results. "
                    "On the first run the model may be downloaded; later runs reuse cached embeddings."
                ),
            )
        with e2:
            batch_size = st.number_input(
                "batch_size",
                min_value=1,
                max_value=512,
                value=32,
                step=1,
                key="cfg_batch_size",
                help=(
                    "How many texts are embedded per batch. "
                    "Higher values are usually faster but use more RAM/CPU at once. "
                    "If you see memory errors or slowdowns, reduce this (for example 16 or 32)."
                ),
            )
        with e3:
            normalize = st.checkbox(
                "normalize_to_unit_sphere",
                value=True,
                key="cfg_normalize",
                help=(
                    "L2-normalize embeddings (make each vector length = 1). "
                    "This makes cosine similarity behave more consistently across samples and models, "
                    "and is typically a good default for geometry-based scoring."
                ),
            )
        with e4:
            reduction_kind = st.selectbox(
                "reduction",
                options=[k.value for k in ReductionKind],
                index=0,
                key="cfg_reduction_kind",
                help=(
                    "Optional dimensionality reduction applied after embeddings (and before scoring). "
                    "`none` keeps the original embedding size; `pca` and `svd` reduce vectors to `n_components`. "
                    "Reducing dimensions can speed up runs and sometimes improve stability, but it can also remove useful signal. "
                    "The reduction model is fit on the current dataset and cached, so reruns are consistent."
                ),
            )

        r1, r2 = st.columns(2)
        with r1:
            n_components = st.number_input(
                "n_components",
                min_value=2,
                max_value=2048,
                value=128,
                step=1,
                key="cfg_n_components",
                help=(
                    "Target number of dimensions after PCA/SVD. "
                    "Smaller values run faster and use less disk in the embedding cache, but may lose semantic detail. "
                    "Ignored when reduction is `none`."
                ),
            )
        with r2:
            whitening = st.checkbox(
                "whitening (PCA only)",
                value=False,
                key="cfg_whitening",
                help=(
                    "Only used for PCA: enables `PCA(whiten=True)`, which scales principal components to unit variance. "
                    "This can help when some dimensions dominate, but it can also amplify noise. "
                    "Ignored for `svd` and `none`."
                ),
            )

        p1, p2, p3 = st.columns(3)
        with p1:
            q_pool = st.selectbox(
                "question_pooling",
                options=[p.value for p in PoolingMode],
                index=0,
                key="cfg_q_pool",
                help=(
                    "How to pool token-level model outputs into one vector for the question. "
                    "For most Sentence-Transformers models, `mean` is a solid default; `max` can emphasize rare features. "
                    "(Note: context pooling is applied differently when context is a list of documents.)"
                ),
            )
        with p2:
            c_pool = st.selectbox(
                "context_pooling",
                options=[p.value for p in PoolingMode],
                index=0,
                key="cfg_c_pool",
                help=(
                    "How to combine context text into a single context vector per sample. "
                    "If your context is a list of retrieved documents, each is embedded and then pooled/aggregated. "
                    "This choice directly affects how close to context an answer appears under cosine similarity."
                ),
            )
        with p3:
            a_pool = st.selectbox(
                "answer_pooling",
                options=[p.value for p in PoolingMode],
                index=0,
                key="cfg_a_pool",
                help=(
                    "How to pool the answer into a single vector (used by SGI/DC/CAI). "
                    "This does not affect the cluster method, which embeds answer chunks separately."
                ),
            )

        st.markdown("### Methods")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            enable_sgi = st.checkbox(
                "Enable SGI",
                value=True,
                key="cfg_enable_sgi",
                help=(
                    "SGI (Score Geometry Inconsistency): compares how much the answer aligns with the context vs the question. "
                    "It is computed from cosine similarities between (answer, context) and (answer, question). "
                    "Lower SGI suggests the answer is not well supported by the provided context."
                ),
            )
            sgi_threshold = st.number_input(
                "sgi_threshold",
                value=0.3,
                key="cfg_sgi_threshold",
                help=(
                    "SGI decision threshold. A sample is flagged as hallucination when `SGI < threshold`. "
                    "Raising the threshold makes SGI stricter (more samples will be flagged). "
                    "Start around 0.2-0.4 and tune using the History ranking."
                ),
            )
        with m2:
            enable_dc = st.checkbox(
                "Enable DC",
                value=False,
                key="cfg_enable_dc",
                help=(
                    "DC (Direction Consistency): checks whether the direction from the question to the generated answer "
                    "matches the direction from the question to the ground-truth answer in embedding space. "
                    "This method requires a `ground_truth` column."
                ),
            )
            cfg_valid2 = bool(st.session_state.get("cfg_valid", True))
            if enable_dc and ground_truth_col == "(none)":
                cfg_valid2 = False
                st.error("DC is enabled but no `ground_truth` column is selected.")
            elif (not enable_dc) and ground_truth_col != "(none)":
                st.info("`ground_truth` is selected but DC is disabled, so ground truth will not be used.")
            st.session_state["cfg_valid"] = bool(cfg_valid2)

            dc_threshold = st.number_input(
                "dc_threshold",
                value=0.6,
                key="cfg_dc_threshold",
                help=(
                    "DC decision threshold (cosine similarity in [-1, 1]). A sample is flagged as hallucination when `DC < threshold`. "
                    "Higher threshold is stricter (it demands closer alignment to the ground truth)."
                ),
            )
        with m3:
            enable_cai = st.checkbox(
                "Enable CAI",
                value=False,
                key="cfg_enable_cai",
                help=(
                    "CAI (Context Anchoring Index): compares how generic the answer is versus how context-specific it is. "
                    "It uses a dataset-level centroid as a proxy for the general domain and checks whether the answer is closer to its context. "
                    "Higher CAI usually indicates better grounding in the provided context."
                ),
            )
            cai_threshold = st.number_input(
                "cai_threshold",
                value=0.1,
                key="cfg_cai_threshold",
                help=(
                    "CAI decision threshold. A sample is flagged as hallucination when `CAI < threshold`. "
                    "Raising the threshold makes CAI stricter (more samples flagged)."
                ),
            )
        with m4:
            enable_cluster = st.checkbox(
                "Enable cluster",
                value=False,
                key="cfg_enable_cluster",
                help=(
                    "Cluster-based score: splits the answer into chunks (sentences), embeds each chunk, clusters them, "
                    "and measures how far chunk-centroids are from the context embedding. "
                    "This can catch off-topic sub-parts inside a longer answer, but is slower than SGI/DC/CAI."
                ),
            )
            cluster_threshold = st.number_input(
                "cluster_threshold",
                value=0.5,
                key="cfg_cluster_threshold",
                help=(
                    "Cluster decision threshold. The cluster score is a distance-like value in [0, 1] where higher means further from context. "
                    "A sample is flagged as hallucination when `cluster_score > threshold`. "
                    "Raising the threshold makes this method less strict (fewer samples flagged)."
                ),
            )
            k_clusters = st.number_input(
                "k_clusters",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                key="cfg_k_clusters",
                help=(
                    "How many clusters to use inside each answer (KMeans). "
                    "Higher values can separate more topics in an answer, but can be unstable when answers are short. "
                    "The effective k is capped by the number of chunks available for the sample."
                ),
            )
            min_chunks = st.number_input(
                "min_chunks",
                min_value=1,
                max_value=200,
                value=3,
                step=1,
                key="cfg_min_chunks",
                help=(
                    "Minimum number of chunks (sentences) required to compute the cluster score. "
                    "If an answer is too short to produce at least this many chunks, the method returns score=0 and does not flag. "
                    "Lower it for short answers; raise it to avoid noisy scores from very small samples."
                ),
            )
            sentence_min_chars = st.number_input(
                "sentence_min_chars",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
                key="cfg_sentence_min_chars",
                help=(
                    "Chunking parameter for the cluster method. "
                    "Sentences shorter than this are merged/ignored during chunking to avoid tiny, noisy chunks. "
                    "Lower values create more chunks (more sensitive), higher values create fewer chunks (more stable)."
                ),
            )

        st.markdown("### Aggregation")
        a1, a2 = st.columns(2)
        with a1:
            agg_kind = st.selectbox(
                "aggregation_strategy",
                options=[k.value for k in AggregationKind],
                index=0,
                key="cfg_agg_kind",
                help=(
                    "How individual methods are combined into the final hallucination prediction. "
                    "`hard` means: if ANY enabled method flags the sample, the final prediction is hallucination. "
                    "`score_based` means: compute a weighted average hallucination score from enabled methods and compare it to `threshold_hall`."
                ),
            )
        with a2:
            threshold_hall = st.number_input(
                "threshold_hall",
                value=0.5,
                key="cfg_threshold_hall",
                help=(
                    "Only used for `score_based`: final prediction is hallucination when `hall_score > threshold_hall`. "
                    "Higher values make the final decision less strict (fewer positives), lower values make it more strict (more positives)."
                ),
            )

        st.markdown("### Aggregation weights (score_based)")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            w_sgi = st.number_input(
                "weight_sgi",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="cfg_w_sgi",
                help=(
                    "Weight for SGI inside `score_based` aggregation. "
                    "Higher weight makes SGI influence the final `hall_score` more; set to 0 to ignore SGI in the score."
                ),
            )
        with w2:
            w_dc = st.number_input(
                "weight_dc",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="cfg_w_dc",
                help=(
                    "Weight for DC inside `score_based` aggregation. "
                    "Higher weight makes DC influence the final `hall_score` more; set to 0 to ignore DC in the score. "
                    "Remember DC requires `ground_truth`."
                ),
            )
        with w3:
            w_cai = st.number_input(
                "weight_cai",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="cfg_w_cai",
                help=(
                    "Weight for CAI inside `score_based` aggregation. "
                    "Higher weight makes CAI influence the final `hall_score` more; set to 0 to ignore CAI in the score."
                ),
            )
        with w4:
            w_cluster = st.number_input(
                "weight_cluster",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="cfg_w_cluster",
                help=(
                    "Weight for the cluster score inside `score_based` aggregation. "
                    "Higher weight makes the cluster method influence the final `hall_score` more; set to 0 to ignore it in the score."
                ),
            )

        st.session_state["cfg"] = {
            "columns": {
                "question": question_col,
                "context": context_col,
                "answer": answer_col,
                "ground_truth": None if ground_truth_col == "(none)" else ground_truth_col,
                "label": None if label_col == "(none)" else label_col,
            },
            "emb": {
                "embedding_model": embedding_model,
                "batch_size": int(batch_size),
                "question_pooling": q_pool,
                "context_pooling": c_pool,
                "answer_pooling": a_pool,
                "normalize": bool(normalize),
                "reduction": reduction_kind,
                "n_components": int(n_components),
                "whitening": bool(whitening),
            },
            "methods": {
                "sgi": {"enabled": bool(enable_sgi), "threshold": float(sgi_threshold)},
                "dc": {"enabled": bool(enable_dc), "threshold": float(dc_threshold)},
                "cai": {"enabled": bool(enable_cai), "threshold": float(cai_threshold)},
                "cluster": {
                    "enabled": bool(enable_cluster),
                    "threshold": float(cluster_threshold),
                    "k_clusters": int(k_clusters),
                    "min_chunks": int(min_chunks),
                    "sentence_min_chars": int(sentence_min_chars),
                },
            },
            "agg": {
                "kind": agg_kind,
                "threshold_hall": float(threshold_hall),
                "weights": {
                    "sgi": float(w_sgi),
                    "dc": float(w_dc),
                    "cai": float(w_cai),
                    "cluster": float(w_cluster),
                },
            },
        }

with tab_run:
    st.subheader("Run detection")
    dataset_path = st.session_state.get("dataset_path")
    cfg0 = st.session_state.get("cfg")
    cfg_valid = st.session_state.get("cfg_valid", True)

    auto_name = st.checkbox(
        "Auto-generate run_id prefix",
        value=True,
        key="run_prefix_auto",
        help=(
            "When enabled, the tool generates a short, filesystem-safe prefix based on your current configuration. "
            "This helps you recognize runs in the History tab without opening each config."
        ),
    )
    auto_prefix = _auto_run_prefix(cfg0) if cfg0 else "run"
    if auto_name:
        st.text_input(
            "run_id prefix (auto)",
            value=auto_prefix,
            disabled=True,
            help=(
                "Read-only preview of the auto-generated prefix (methods + thresholds + embeddings + aggregation). "
                "The final run_id also includes a timestamp/hash to stay unique."
            ),
        )
        run_prefix = auto_prefix
    else:
        run_prefix = st.text_input(
            "run_id prefix",
            value=st.session_state.get("run_prefix_manual", "exp"),
            key="run_prefix_manual",
            help=(
                "Manual prefix used as the human-readable part of run_id. "
                "Use letters/numbers/dashes; keep it short (for example `baseline`, `tune-sgi`, `pca128`). "
                "The tool will still append a timestamp/hash so runs remain unique."
            ),
        )

    force_recompute_embeddings = st.checkbox(
        "Overwrite cached embeddings (recompute)",
        value=False,
        key="force_recompute_embeddings",
        help=(
            "Deletes cached embeddings for this dataset+embedding config and recomputes them. "
            "Use if you suspect stale/corrupted cache or you want a clean rerun."
        ),
    )

    if st.button(
        "Run Detection",
        disabled=not (dataset_path and cfg0 and cfg_valid),
        help=(
            "Runs the full pipeline: load dataset, compute/reuse cached embeddings, compute selected geometric methods, "
            "aggregate predictions, evaluate metrics (if `label` exists), and save artifacts under `artifacts/runs/<run_id>/`."
        ),
    ):
        c = cfg0
        run_cfg = RunConfig(
            dataset_columns=DatasetColumns(**c["columns"]),
            embeddings=EmbeddingConfig(
                embedding_model=c["emb"]["embedding_model"],
                batch_size=c["emb"]["batch_size"],
                question_pooling=PoolingMode(c["emb"]["question_pooling"]),
                context_pooling=PoolingMode(c["emb"]["context_pooling"]),
                answer_pooling=PoolingMode(c["emb"]["answer_pooling"]),
                normalize_to_unit_sphere=c["emb"]["normalize"],
                reduction=ReductionConfig(
                    kind=ReductionKind(c["emb"]["reduction"]),
                    n_components=c["emb"]["n_components"],
                    whitening=c["emb"]["whitening"],
                ),
            ),
            run_prefix=run_prefix,
        )

        run_cfg.sgi.enabled = c["methods"]["sgi"]["enabled"]
        run_cfg.sgi.threshold = c["methods"]["sgi"]["threshold"]

        run_cfg.dc.enabled = c["methods"]["dc"]["enabled"]
        run_cfg.dc.threshold = c["methods"]["dc"]["threshold"]

        run_cfg.cai.enabled = c["methods"]["cai"]["enabled"]
        run_cfg.cai.threshold = c["methods"]["cai"]["threshold"]

        run_cfg.cluster.enabled = c["methods"]["cluster"]["enabled"]
        run_cfg.cluster.threshold = c["methods"]["cluster"]["threshold"]
        run_cfg.cluster.k_clusters = c["methods"]["cluster"]["k_clusters"]
        run_cfg.cluster.min_chunks = c["methods"]["cluster"].get("min_chunks", 3)
        run_cfg.cluster.sentence_min_chars = c["methods"]["cluster"].get("sentence_min_chars", 25)

        run_cfg.aggregation = AggregationConfig(
            kind=AggregationKind(c["agg"]["kind"]),
            threshold_hall=c["agg"]["threshold_hall"],
            weights=c["agg"].get("weights") or {"sgi": 1.0, "dc": 1.0, "cai": 1.0, "cluster": 1.0},
        )

        start_t = time.time()
        bar = st.progress(0, text="Starting...")

        def cb(frac: float, msg: str) -> None:
            # frac in [0,1]
            elapsed = time.time() - start_t
            eta_txt = ""
            f = float(max(0.0, min(1.0, frac)))
            if f > 0.02:
                eta = elapsed * (1.0 - f) / f
                mins = int(eta // 60)
                secs = int(eta % 60)
                eta_txt = f" | ETA ~{mins:02d}:{secs:02d}"
            pct = f * 100.0
            bar.progress(f, text=f"{msg} ({pct:.1f}%{eta_txt})")

        with st.status("Running...", expanded=True) as status:
            res = run_detection(
                dataset_path=Path(dataset_path),
                run_config=run_cfg,
                force_recompute_embeddings=bool(force_recompute_embeddings),
                progress_cb=cb,
            )
            status.update(label="Done", state="complete")
        bar.progress(1.0, text="Done")

        st.success(f"Run saved: {res['run_id']}")
        summary = res.get("summary", {}) or {}
        if not summary:
            st.warning(
                "No labels found in the dataset; metrics were not computed "
                "(convert HaluEval or provide a label column)."
            )
        st.write(summary)
        st.write(res.get("paths", {}))

        notes = res.get("notes") or []
        for n in notes:
            st.info(n)

with tab_history:
    st.subheader("Previous runs")
    runs = list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        if "copy_notice" in st.session_state:
            st.success(st.session_state.pop("copy_notice"))
        if "copy_error" in st.session_state:
            st.error(st.session_state.pop("copy_error"))
        if "delete_notice" in st.session_state:
            st.success(st.session_state.pop("delete_notice"))
        if "delete_error" in st.session_state:
            st.error(st.session_state.pop("delete_error"))

        rows = []
        for r in runs:
            summary = r.get("summary") or {}
            metrics = (summary.get("metrics") or {}) if isinstance(summary, dict) else {}
            methods = summary.get("used_methods")
            if isinstance(methods, list):
                methods = ", ".join(methods)
            rows.append(
                {
                    "run_id": r.get("run_id"),
                    "created_at": _run_created_at_str(r),
                    "methods": methods,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                }
            )

        hist_df = pd.DataFrame(rows)
        ranked = hist_df.sort_values("accuracy", ascending=False, na_position="last").reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))

        st.markdown("### Ranking")

        ranked_view = ranked.copy()
        ranked_view.insert(0, "delete", False)

        edited = st.data_editor(
            ranked_view[
                [
                    "delete",
                    "rank",
                    "created_at",
                    "run_id",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "methods",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            key="history_ranking_editor",
            column_config={
                "delete": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Mark runs to delete from artifacts/runs/<run_id> (cannot be undone).",
                )
            },
            disabled=[
                "rank",
                "created_at",
                "run_id",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "methods",
            ],
        )

        to_delete = edited.loc[edited["delete"] == True, "run_id"].astype(str).tolist()  # noqa: E712
        if to_delete:
            st.warning(
                "This will permanently delete only the selected run folders under `artifacts/runs/<run_id>/`. "
                "It will NOT delete your dataset uploads or cached embeddings unless you use Permanent delete."
            )

            run_meta: Dict[str, Dict[str, Any]] = {}
            for rr in runs:
                rid = rr.get("run_id")
                cfg = rr.get("config") or {}
                if not rid:
                    continue
                run_meta[str(rid)] = {
                    "dataset_id": cfg.get("dataset_id"),
                    "embed_config_id": cfg.get("embed_config_id"),
                    "dataset_path": cfg.get("dataset_path"),
                }

            remaining_ids = {str(rr.get("run_id")) for rr in runs if str(rr.get("run_id")) not in set(to_delete)}
            used_dataset_ids = {
                str(run_meta[rid]["dataset_id"])
                for rid in remaining_ids
                if rid in run_meta and run_meta[rid].get("dataset_id")
            }
            used_embed_pairs = {
                (str(run_meta[rid]["dataset_id"]), str(run_meta[rid]["embed_config_id"]))
                for rid in remaining_ids
                if rid in run_meta and run_meta[rid].get("dataset_id") and run_meta[rid].get("embed_config_id")
            }
            used_dataset_paths = {
                str(run_meta[rid]["dataset_path"])
                for rid in remaining_ids
                if rid in run_meta and run_meta[rid].get("dataset_path")
            }

            cleanup_dataset_ids = sorted({
                str(run_meta[rid]["dataset_id"])
                for rid in to_delete
                if rid in run_meta and run_meta[rid].get("dataset_id") and str(run_meta[rid]["dataset_id"]) not in used_dataset_ids
            })
            cleanup_embed_pairs = sorted({
                (str(run_meta[rid]["dataset_id"]), str(run_meta[rid]["embed_config_id"]))
                for rid in to_delete
                if rid in run_meta
                and run_meta[rid].get("dataset_id")
                and run_meta[rid].get("embed_config_id")
                and (str(run_meta[rid]["dataset_id"]), str(run_meta[rid]["embed_config_id"])) not in used_embed_pairs
            })

            root = artifacts_root()
            uploads_dir = root / "uploads"
            candidate_uploads = sorted({
                str(run_meta[rid]["dataset_path"])
                for rid in to_delete
                if rid in run_meta and run_meta[rid].get("dataset_path")
            })
            cleanup_upload_paths: list[Path] = []
            for p in candidate_uploads:
                if p in used_dataset_paths:
                    continue
                try:
                    pp = Path(p)
                    if uploads_dir.resolve() in pp.resolve().parents:
                        cleanup_upload_paths.append(pp)
                except Exception:
                    continue

            with st.expander("Permanent delete preview", expanded=False):
                st.write({"runs": to_delete})
                st.write({"embeddings_to_delete": [f"{a}/{b}" for a, b in cleanup_embed_pairs]})
                st.write({"datasets_to_delete": cleanup_dataset_ids})
                st.write({"uploads_to_delete": [p.name for p in cleanup_upload_paths]})

            cdel1, cdel2 = st.columns(2)
            with cdel1:
                confirm_delete = st.checkbox(
                    f"Confirm delete runs ({len(to_delete)} run(s))",
                    key="confirm_delete_runs_only",
                )
                if st.button(
                    "Delete selected runs",
                    type="secondary",
                    disabled=not bool(confirm_delete),
                    help="Deletes only artifacts/runs/<run_id>/ for the selected runs.",
                ):
                    msgs = []
                    any_error = False
                    for rid in to_delete:
                        ok, msg = _delete_run_artifacts(rid)
                        msgs.append(msg)
                        if not ok:
                            any_error = True
                    if any_error:
                        st.session_state["delete_error"] = "\\n".join(msgs)
                    else:
                        st.session_state["delete_notice"] = "\\n".join(msgs)
                    st.session_state.pop("confirm_delete_runs_only", None)
                    st.rerun()

            with cdel2:
                confirm_perm = st.checkbox(
                    "Confirm Permanent delete (runs + unused caches)",
                    key="confirm_delete_runs_perm",
                    help="Also deletes unused embeddings cache and dataset bundles if no remaining runs reference them.",
                )
                cleanup_uploads = st.checkbox(
                    "Also delete unused uploads (artifacts/uploads)",
                    value=False,
                    key="confirm_delete_uploads_perm",
                    help="Only deletes upload files under artifacts/uploads that are not referenced by remaining runs.",
                )
                if st.button(
                    "Permanent delete",
                    type="primary",
                    disabled=not bool(confirm_perm),
                    help="Deletes runs and cleans up caches/uploads that become unused.",
                ):
                    msgs = []
                    any_error = False
                    # First delete runs
                    for rid in to_delete:
                        ok, msg = _delete_run_artifacts(rid)
                        msgs.append(msg)
                        if not ok:
                            any_error = True

                    # Then delete unused embedding caches
                    for dsid, ecid in cleanup_embed_pairs:
                        target = root / "embeddings" / dsid / ecid
                        ok, msg = _safe_rmtree_under(root / "embeddings", target)
                        msgs.append(f"embeddings/{dsid}/{ecid}: {msg}")
                        if not ok:
                            any_error = True

                        # Remove now-empty dataset embeddings dir
                        ds_dir = root / "embeddings" / dsid
                        try:
                            if ds_dir.exists() and ds_dir.is_dir() and not any(ds_dir.iterdir()):
                                ds_dir.rmdir()
                        except Exception:
                            pass

                    # Then delete unused dataset bundles
                    for dsid in cleanup_dataset_ids:
                        target = root / "datasets" / dsid
                        ok, msg = _safe_rmtree_under(root / "datasets", target)
                        msgs.append(f"datasets/{dsid}: {msg}")
                        if not ok:
                            any_error = True

                    # Optionally delete unused uploads
                    if cleanup_uploads:
                        for up in cleanup_upload_paths:
                            ok, msg = _safe_unlink_under(root / "uploads", up)
                            msgs.append(f"uploads/{up.name}: {msg}")
                            if not ok:
                                any_error = True

                    if any_error:
                        st.session_state["delete_error"] = "\\n".join(msgs)
                    else:
                        st.session_state["delete_notice"] = "\\n".join(msgs)
                    st.session_state.pop("confirm_delete_runs_perm", None)
                    st.session_state.pop("confirm_delete_uploads_perm", None)
                    st.rerun()
        st.markdown("### Copy settings")
        choices = hist_df["run_id"].tolist()
        selected_run = st.selectbox(
            "Run to copy",
            choices,
            help=(
                "Select a previous run_id to load its saved run_config.json and apply those settings into the current GUI. "
                "This is useful for iterating: copy a good baseline, tweak a threshold/weight, and rerun."
            ),
        )
        if st.button("Apply settings"):
            st.session_state["pending_apply_run_id"] = selected_run
            st.rerun()










