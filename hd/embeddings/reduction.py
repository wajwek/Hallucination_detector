from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA, TruncatedSVD


@dataclass
class ReductionModel:
    kind: Literal["none", "pca", "svd"]
    model: object | None


def fit_reduction(kind: str, x: np.ndarray, n_components: int, whitening: bool) -> ReductionModel:
    if kind == "none":
        return ReductionModel(kind="none", model=None)

    if kind == "pca":
        # PCA(whiten=True) scales principal components to unit variance.
        m = PCA(n_components=n_components, whiten=bool(whitening), random_state=42)
        m.fit(x)
        return ReductionModel(kind="pca", model=m)

    if kind == "svd":
        m = TruncatedSVD(n_components=n_components, random_state=42)
        m.fit(x)
        return ReductionModel(kind="svd", model=m)

    raise ValueError(f"Unknown reduction kind: {kind}")


def apply_reduction(rm: ReductionModel, x: np.ndarray) -> np.ndarray:
    if rm.kind == "none" or rm.model is None:
        return x
    return rm.model.transform(x)


def save_reduction_model(path: str, rm: ReductionModel) -> None:
    dump(rm, path)


def load_reduction_model(path: str) -> ReductionModel:
    return load(path)
