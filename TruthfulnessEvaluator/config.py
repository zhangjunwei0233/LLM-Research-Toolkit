"""Dataclasses holding configuration knobs for the truthfulness pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DatasetSelection:
    """Dataset choice plus optional sampling limit."""

    name: str
    limit: Optional[int] = None  # allow quick smoke tests


@dataclass(slots=True)
class ModelSelection:
    """Identifies which preset module and engine to use for a model."""

    name: str
    engine: str = "transformers"


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration that threads through the pipeline."""

    dataset: DatasetSelection
    test_model: ModelSelection
    judge_model: ModelSelection
    output_dir: Path
