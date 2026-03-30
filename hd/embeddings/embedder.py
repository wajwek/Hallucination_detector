from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np


ProgressCallback = Callable[[float, str], None]


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies: pip install -e ."
            ) from e

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        *,
        progress_cb: Optional[ProgressCallback] = None,
        progress_prefix: str = "Embedding",
    ) -> np.ndarray:
        texts_list = list(texts)
        n = len(texts_list)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Fast path (no UI progress needed)
        if progress_cb is None:
            return self.model.encode(
                texts_list,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)

        # Progress path: embed in batches to report progress to the GUI.
        embs: List[np.ndarray] = []
        done = 0
        bs = max(1, int(batch_size))
        for start in range(0, n, bs):
            end = min(n, start + bs)
            batch = texts_list[start:end]
            batch_emb = self.model.encode(
                batch,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            embs.append(batch_emb)
            done = end
            progress_cb(done / n, f"{progress_prefix} {done}/{n}")

        return np.vstack(embs).astype(np.float32)


def chunk_sentences(text: str, min_chars: int = 25) -> List[str]:
    import re

    # simple rule-based splitter (good enough for v1)
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = [p.strip() for p in parts if len(p.strip()) >= int(min_chars)]
    if not out and text:
        out = [text.strip()]
    return out
