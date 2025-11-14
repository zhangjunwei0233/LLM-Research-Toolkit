"""Dataset loader registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import DatasetLoader, DatasetExample
from .truthfulqa import create_truthfulqa_loader
from .simpleqa import create_simpleqa_loader
from .haluevalqa import create_haluevalqa_loader
from .hotpotqa import create_hotpotqa_loader

DATASET_FACTORIES: Dict[str, Callable[[Optional[int]], DatasetLoader]] = {
    "truthfulqa": create_truthfulqa_loader,
    "simpleqa": create_simpleqa_loader,
    "haluevalqa": create_haluevalqa_loader,
    "hotpotqa": create_hotpotqa_loader,
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
