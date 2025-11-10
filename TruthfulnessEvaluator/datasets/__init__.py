"""Dataset loader registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import DatasetLoader, DatasetExample
from .truthfulqa import create_truthfulqa_loader

DATASET_FACTORIES: Dict[str, Callable[[Optional[int]], DatasetLoader]] = {
    "truthfulqa": create_truthfulqa_loader,
}


def create_dataset_loader(name: str, *, limit: Optional[int] = None) -> DatasetLoader:
    """Instantiate a dataset loader using the preset registered unden `name`."""
    try:
        factory = DATASET_FACTORIES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset '{name}'. Registered datasets: {list(DATASET_FACTORIES)}"
        ) from exc
    return factory(limit)


__all__ = [
    "DatasetLoader",
    "DatasetExample",
    "create_dataset_loader"
]
