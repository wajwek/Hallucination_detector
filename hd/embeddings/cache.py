from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from joblib import dump, load

from ..config import EmbeddingConfig, RunConfig
from ..storage import embedding_paths
from ..utils import write_json
from .embedder import SentenceTransformersEmbedder, chunk_sentences
from .pooling import l2_normalize, pool_vectors
from .reduction import (
    ReductionModel,
    apply_reduction,
    fit_reduction,
    load_reduction_model,
    save_reduction_model,
)


ProgressCallback = Callable[[float, str], None]


def _progress(cb: ProgressCallback | None, frac: float, message: str) -> None:
    if cb is None:
        return
    f = max(0.0, min(1.0, float(frac)))
    cb(f, str(message))


def _stage_cb(
    cb: ProgressCallback | None,
    *,
    start: float,
    end: float,
) -> ProgressCallback | None:
    if cb is None:
        return None

    s = float(start)
    e = float(end)

    def inner(frac: float, msg: str) -> None:
        _progress(cb, s + (e - s) * float(frac), msg)

    return inner


@dataclass
class EmbeddingsBundle:
    q: np.ndarray
    c: np.ndarray
    a: np.ndarray
    gt: Optional[np.ndarray]
    centroid: np.ndarray
    answer_chunks: Optional[List[np.ndarray]] = None


def _encode_contexts(
    embedder: SentenceTransformersEmbedder,
    contexts: List[List[str]],
    batch_size: int,
    pooling_mode: str,
    *,
    embed_progress_cb: ProgressCallback | None = None,
    pool_progress_cb: ProgressCallback | None = None,
    progress_every: int = 250,
) -> np.ndarray:
    flat: List[str] = []
    ranges: List[Tuple[int, int]] = []
    for docs in contexts:
        start = len(flat)
        flat.extend([d for d in docs if d and str(d).strip()])
        end = len(flat)
        ranges.append((start, end))

    if not flat:
        raise ValueError("Context is empty for all samples. RAG requires context.")

    embs = embedder.encode(
        flat,
        batch_size=batch_size,
        progress_cb=embed_progress_cb,
        progress_prefix="Embedding contexts",
    )

    out = np.zeros((len(contexts), embs.shape[1]), dtype=np.float32)
    n_ctx = len(contexts)
    step = max(1, int(progress_every))
    for i, (s, e) in enumerate(ranges):
        if s == e:
            out[i] = 0.0
        else:
            block = embs[s:e]
            if pooling_mode == "weighted_mean":
                weights = np.array([max(1, len(t)) for t in flat[s:e]], dtype=np.float32)
                out[i] = pool_vectors(block, "weighted_mean", weights=weights)
            else:
                out[i] = pool_vectors(block, pooling_mode)

        if pool_progress_cb is not None and (i % step == 0 or i + 1 == n_ctx):
            pool_progress_cb((i + 1) / max(1, n_ctx), f"Pooling contexts {i + 1}/{n_ctx}")

    return out


def _maybe_reduction_and_norm(
    cfg: EmbeddingConfig,
    q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    gt: Optional[np.ndarray],
    centroid: np.ndarray,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    kind = cfg.reduction.kind.value if hasattr(cfg.reduction.kind, "value") else str(cfg.reduction.kind)
    whitening = bool(cfg.reduction.whitening)

    if kind == "none":
        rm = ReductionModel(kind="none", model=None)
    else:
        model_path = cache_dir / "reduction_model.joblib"
        if model_path.exists():
            rm = load_reduction_model(str(model_path))
        else:
            pieces = [q, c, a]
            if gt is not None:
                pieces.append(gt)
            x_fit = np.concatenate(pieces, axis=0)
            rm = fit_reduction(
                kind,
                x_fit,
                n_components=int(cfg.reduction.n_components),
                whitening=whitening,
            )
            save_reduction_model(str(model_path), rm)

    q2 = apply_reduction(rm, q)
    c2 = apply_reduction(rm, c)
    a2 = apply_reduction(rm, a)
    gt2 = apply_reduction(rm, gt) if gt is not None else None
    centroid2 = apply_reduction(rm, centroid.reshape(1, -1)).reshape(-1)

    if cfg.normalize_to_unit_sphere:
        q2 = l2_normalize(q2)
        c2 = l2_normalize(c2)
        a2 = l2_normalize(a2)
        if gt2 is not None:
            gt2 = l2_normalize(gt2)
        centroid2 = l2_normalize(centroid2)

    return q2, c2, a2, gt2, centroid2


def _compute_answer_chunks(
    *,
    run_config: RunConfig,
    embedder: SentenceTransformersEmbedder,
    answers: List[str],
    progress_cb: ProgressCallback | None = None,
) -> Optional[List[np.ndarray]]:
    cfg = run_config.embeddings

    _progress(progress_cb, 0.05, "Chunking answers")
    chunks: List[List[str]] = [
        chunk_sentences(t, min_chars=run_config.cluster.sentence_min_chars) for t in answers
    ]
    flat: List[str] = []
    ranges: List[Tuple[int, int]] = []
    for cs in chunks:
        s = len(flat)
        flat.extend(cs)
        e = len(flat)
        ranges.append((s, e))

    if not flat:
        _progress(progress_cb, 1.0, "No chunks")
        return None

    _progress(progress_cb, 0.20, "Embedding answer chunks")
    emb_chunks = embedder.encode(
        flat,
        batch_size=cfg.batch_size,
        progress_cb=_stage_cb(progress_cb, start=0.20, end=0.85),
        progress_prefix="Embedding answer chunks",
    )

    per_sample: List[np.ndarray] = []
    for s, e in ranges:
        per_sample.append(emb_chunks[s:e])

    if cfg.normalize_to_unit_sphere:
        per_sample = [l2_normalize(x) if x.size else x for x in per_sample]

    _progress(progress_cb, 1.0, "Chunks ready")
    return per_sample


def _cache_answer_chunks(out_dir: Path, per_sample: List[np.ndarray], sentence_min_chars: int) -> None:
    chunk_cache_path = out_dir / "answer_chunks.joblib"
    chunk_meta_path = out_dir / "answer_chunks_meta.json"

    dump(per_sample, chunk_cache_path)
    write_json(
        chunk_meta_path,
        {
            "sentence_min_chars": int(sentence_min_chars),
            "n_samples": int(len(per_sample)),
        },
    )


def load_or_compute_embeddings(
    run_config: RunConfig,
    dataset_id: str,
    embed_config_id: str,
    artifacts_root: Path,
    df,
    force_recompute: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> EmbeddingsBundle:
    ep = embedding_paths(artifacts_root, dataset_id, embed_config_id)
    ep.embed_dir.mkdir(parents=True, exist_ok=True)

    cfg = run_config.embeddings
    enabled_methods = set(run_config.enabled_methods())

    if force_recompute:
        _progress(progress_cb, 0.02, "Clearing cached embeddings")
        for fp in [
            ep.q_path,
            ep.c_path,
            ep.a_path,
            ep.gt_path,
            ep.centroid_path,
            ep.config_path,
            ep.reduction_model_path,
            ep.chunk_cache_path,
            ep.chunk_meta_path,
        ]:
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass

    # Cache hit
    if (not force_recompute) and ep.q_path.exists() and ep.c_path.exists() and ep.a_path.exists() and ep.centroid_path.exists():
        _progress(progress_cb, 0.10, "Loading cached embeddings")
        q = np.load(ep.q_path)
        c = np.load(ep.c_path)
        a = np.load(ep.a_path)
        centroid = np.load(ep.centroid_path)
        gt = np.load(ep.gt_path) if ep.gt_path.exists() else None

        answer_chunks = None
        if "cluster" in enabled_methods:
            if ep.chunk_cache_path.exists():
                _progress(progress_cb, 0.50, "Loading cached answer chunks")
                answer_chunks = load(ep.chunk_cache_path)
            else:
                _progress(progress_cb, 0.45, "Computing answer chunks")
                embedder = SentenceTransformersEmbedder(cfg.embedding_model)

                def ch_cb(frac: float, msg: str) -> None:
                    _progress(progress_cb, 0.45 + 0.45 * frac, msg)

                answers = df["answer"].astype(str).tolist()
                per_sample = _compute_answer_chunks(
                    run_config=run_config,
                    embedder=embedder,
                    answers=answers,
                    progress_cb=ch_cb,
                )
                if per_sample is not None:
                    _cache_answer_chunks(ep.embed_dir, per_sample, run_config.cluster.sentence_min_chars)
                answer_chunks = per_sample

        _progress(progress_cb, 1.0, "Embeddings ready (cache)")
        return EmbeddingsBundle(q=q, c=c, a=a, gt=gt, centroid=centroid, answer_chunks=answer_chunks)

    # Compute
    _progress(progress_cb, 0.05, "Loading embedding model")
    embedder = SentenceTransformersEmbedder(cfg.embedding_model)

    q_raw = df["question"].astype(str).tolist()
    a_raw = df["answer"].astype(str).tolist()
    c_raw = df["context"].tolist()

    _progress(progress_cb, 0.10, "Embedding questions")
    q = embedder.encode(
        q_raw,
        batch_size=cfg.batch_size,
        progress_cb=_stage_cb(progress_cb, start=0.10, end=0.30),
        progress_prefix="Embedding questions",
    )

    _progress(progress_cb, 0.30, "Embedding answers")
    a = embedder.encode(
        a_raw,
        batch_size=cfg.batch_size,
        progress_cb=_stage_cb(progress_cb, start=0.30, end=0.50),
        progress_prefix="Embedding answers",
    )

    _progress(progress_cb, 0.50, "Embedding contexts")
    c = _encode_contexts(
        embedder,
        c_raw,
        batch_size=cfg.batch_size,
        pooling_mode=cfg.context_pooling.value,
        embed_progress_cb=_stage_cb(progress_cb, start=0.50, end=0.72),
        pool_progress_cb=_stage_cb(progress_cb, start=0.72, end=0.78),
    )

    gt = None
    if run_config.dc.enabled:
        _progress(progress_cb, 0.78, "Embedding ground truth")
        if df["ground_truth"].isna().all() or (df["ground_truth"].astype(str) == "None").all():
            raise ValueError("DC is enabled but ground_truth is missing in dataset.")
        gt_raw = df["ground_truth"].astype(str).tolist()
        gt = embedder.encode(
            gt_raw,
            batch_size=cfg.batch_size,
            progress_cb=_stage_cb(progress_cb, start=0.78, end=0.84),
            progress_prefix="Embedding ground truth",
        )

    _progress(progress_cb, 0.84, "Computing centroid")
    centroid_src = gt if gt is not None else a
    centroid = centroid_src.mean(axis=0).astype(np.float32)

    _progress(progress_cb, 0.88, "Applying reduction/normalization")
    q, c, a, gt, centroid = _maybe_reduction_and_norm(cfg, q, c, a, gt, centroid, ep.embed_dir)

    _progress(progress_cb, 0.92, "Saving embeddings")
    np.save(ep.q_path, q)
    np.save(ep.c_path, c)
    np.save(ep.a_path, a)
    if gt is not None:
        np.save(ep.gt_path, gt)
    np.save(ep.centroid_path, centroid)

    write_json(
        ep.config_path,
        {
            "dataset_id": dataset_id,
            "embed_config_id": embed_config_id,
            "embedding_config": cfg.model_dump(),
            "enabled_methods": sorted(enabled_methods),
        },
    )

    answer_chunks = None
    if "cluster" in enabled_methods:
        _progress(progress_cb, 0.95, "Preparing answer chunks")

        def ch_cb(frac: float, msg: str) -> None:
            _progress(progress_cb, 0.95 + 0.05 * frac, msg)

        per_sample = _compute_answer_chunks(
            run_config=run_config,
            embedder=embedder,
            answers=a_raw,
            progress_cb=ch_cb,
        )
        if per_sample is not None:
            _cache_answer_chunks(ep.embed_dir, per_sample, run_config.cluster.sentence_min_chars)
        answer_chunks = per_sample

    _progress(progress_cb, 1.0, "Embeddings ready")
    return EmbeddingsBundle(q=q, c=c, a=a, gt=gt, centroid=centroid, answer_chunks=answer_chunks)
