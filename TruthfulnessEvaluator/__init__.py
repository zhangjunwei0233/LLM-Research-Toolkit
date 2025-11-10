"""
TruthfulnessEvaluator package entrypoints.

The package exposes a single high-level helper so other modules can
trigger a full evaluation run without digging into implementation details.
"""

from .config import PipelineConfig
from .pipeline import TruthfulnessPipeline


def run_truthfulness_evaluation(config: PipelineConfig) -> None:
    pipeline = TruthfulnessPipeline(config)
    pipeline.run()


__all__ = [
    "PipelineConfig",
    "run_truthfulness_evaluation",
]
