"""
TruthfulnessEvaluator package entrypoints.

The package exposes a single high-level helper so other modules can
trigger a full evaluation run without digging into implementation details.
"""

from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .pipeline import TruthfulnessPipeline


def run_truthfulness_evaluation(
    config: PipelineConfig,
    *,
    resume_dir: Optional[Path] = None,
) -> None:
    pipeline = TruthfulnessPipeline(config, resume_dir=resume_dir)
    pipeline.run()


__all__ = [
    "PipelineConfig",
    "run_truthfulness_evaluation",
]
