from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PoolingMode(str, Enum):
    mean = "mean"
    max = "max"
    weighted_mean = "weighted_mean"


class ReductionKind(str, Enum):
    none = "none"
    pca = "pca"
    svd = "svd"


class DatasetColumns(BaseModel):
    question: str = "question"
    context: str = "context"
    answer: str = "answer"
    ground_truth: Optional[str] = "ground_truth"
    label: Optional[str] = "label"


class ReductionConfig(BaseModel):
    kind: ReductionKind = ReductionKind.none
    n_components: int = 128
    whitening: bool = False


class EmbeddingConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32

    question_pooling: PoolingMode = PoolingMode.mean
    context_pooling: PoolingMode = PoolingMode.mean
    answer_pooling: PoolingMode = PoolingMode.mean

    normalize_to_unit_sphere: bool = True
    reduction: ReductionConfig = Field(default_factory=ReductionConfig)


class SGIConfig(BaseModel):
    enabled: bool = True
    variant: str = "cosine_diff_norm"
    threshold: float = 0.3
    epsilon: float = 1e-8


class DCConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.6


class CAIConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.1


class ClusterConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.5
    k_clusters: int = 5
    min_chunks: int = 3
    sentence_min_chars: int = 25


class AggregationKind(str, Enum):
    hard = "hard"
    score_based = "score_based"


class AggregationConfig(BaseModel):
    kind: AggregationKind = AggregationKind.hard

    threshold_hall: float = 0.5
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"sgi": 1.0, "dc": 1.0, "cai": 1.0, "cluster": 1.0}
    )


class RunConfig(BaseModel):
    architecture: Literal["RAG"] = "RAG"
    dataset_columns: DatasetColumns = Field(default_factory=DatasetColumns)

    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    sgi: SGIConfig = Field(default_factory=SGIConfig)
    dc: DCConfig = Field(default_factory=DCConfig)
    cai: CAIConfig = Field(default_factory=CAIConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)

    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    run_prefix: str = "run"

    def enabled_methods(self) -> List[str]:
        out: List[str] = []
        if self.sgi.enabled:
            out.append("sgi")
        if self.dc.enabled:
            out.append("dc")
        if self.cai.enabled:
            out.append("cai")
        if self.cluster.enabled:
            out.append("cluster")
        return out


