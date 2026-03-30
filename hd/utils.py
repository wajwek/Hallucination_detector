from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_json(obj: Any) -> Any:
    # Pydantic enums / stdlib enums
    if isinstance(obj, Enum):
        return obj.value

    # Pydantic models
    if hasattr(obj, "model_dump"):
        # mode="json" ensures enums become their values
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Path):
        return str(obj)

    # numpy scalars (avoid importing numpy hard)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, default=_default_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, default=_default_json, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
