"""CLI entrypoint so the package can run with `python -m TruthfulnessEvaluator`."""

from __future__ import annotations

import argparse
from pathlib import Path

from TruthfulnessEvaluator import run_truthfulness_evaluation
from TruthfulnessEvaluator.config import DatasetSelection, ModelSelection, PipelineConfig


def pare_args() -> argparse.Namespace:
    """Collect knobs to run the truthfulness pipeline."""
    parser = argparse.ArgumentParser(
        description="TruthfulnessEvaluator: run local model truthfulness checks."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset preset name backed by modules under TruthfulnessEvaluator/datasets/ ."
    )
    parser.add_argument(
        "--test-model",
        required=True,
        help="Model preset name under TruthfulnessEvaluator/models for the under-test model."
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Model preset name under TruthfulnessEvaluator/models/ for the judge model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("truthfulness_evaluation_results"),
        help="Directory where run artifacts will be stored."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate for a quicker run."
    )
    return parser.parse_args()


def main() -> None:
    """Translate CLI args into dataclass configs and trigger the pipeline."""
    args = pare_args()

    dataset_selection = DatasetSelection(name=args.dataset, limit=args.limit)
    test_model_selection = ModelSelection(name=args.test_model)
    judge_model_selection = ModelSelection(name=args.judge_model)

    pipeline_cfg = PipelineConfig(
        dataset=dataset_selection,
        test_model=test_model_selection,
        judge_model=judge_model_selection,
        output_dir=args.output_dir,
    )

    run_truthfulness_evaluation(pipeline_cfg)


if __name__ == "__main__":
    main()
