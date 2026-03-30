from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .methods.base import BaseMetric


class MethodRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Type[BaseMetric]] = {}

    def register(self, name: str, metric_cls: Type[BaseMetric]) -> None:
        if name in self._metrics:
            raise KeyError(f"Metric already registered: {name}")
        self._metrics[name] = metric_cls

    def create(self, name: str, **kwargs: Any) -> BaseMetric:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name}. Available: {sorted(self._metrics)}")
        return self._metrics[name](**kwargs)  # type: ignore[misc]

    def available(self) -> Dict[str, Type[BaseMetric]]:
        return dict(self._metrics)


REGISTRY = MethodRegistry()


def register_metric(name: str) -> Callable[[Type[BaseMetric]], Type[BaseMetric]]:
    def _wrap(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        REGISTRY.register(name, cls)
        return cls

    return _wrap
