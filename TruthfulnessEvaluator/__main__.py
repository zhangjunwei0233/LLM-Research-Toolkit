"""CLI entrypoint so the package can run with `python -m TruthfulnessEvaluator`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from TruthfulnessEvaluator import run_truthfulness_evaluation
from TruthfulnessEvaluator.config import DatasetSelection, ModelSelection, PipelineConfig


def parse_args() -> argparse.Namespace:
    """Collect knobs to run the truthfulness pipeline."""
    parser = argparse.ArgumentParser(
        description="TruthfulnessEvaluator: run local model truthfulness checks."
    )
    parser.add_argument(
        "--dataset",
        help="Dataset preset name backed by modules under TruthfulnessEvaluator/datasets/ .",
    )
    parser.add_argument(
        "--test-model",
        help="Model preset used for the under-test model.",
    )
    parser.add_argument(
        "--judge-model",
        help="Model preset used for the judge model.",
    )
    parser.add_argument(
        "--engine",
        default="transformers",
        choices=("transformers", "vllm"),
        help="Engine to run the under-test model (default: %(default)s).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Override tensor parallel degree for vLLM. Defaults to using all visible GPUs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate for a quicker run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("truthfulness_evaluation_results"),
        help="Directory where run artifacts will be stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of dataset examples to send to the models per generation call.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a previous run from the provided directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Translate CLI args into dataclass configs and trigger the pipeline."""
    args = parse_args()

    resume_dir: Path | None = None
    if args.resume:  # Resume old run
        resume_dir = args.resume
        pipeline_cfg = _load_config_from_run_dir(resume_dir)  # type: ignore
    else:  # Start a fresh run
        # Check missing arguments
        missing = [
            flag
            for flag in ("dataset", "test_model", "judge_model")
            if getattr(args, flag) is None
        ]
        if missing:
            missing_flags = ", ".join(
                f"--{flag.replace('_', '-')}" for flag in missing)
            raise SystemExit(
                f"The following arguments are required when not resuming: {missing_flags}"
            )
        # Select dataset and models
        dataset_selection = DatasetSelection(
            name=args.dataset, limit=args.limit)
        test_model_selection = ModelSelection(
            name=args.test_model,
            engine=args.engine,
            vllm_tensor_parallel_size=args.tensor_parallel_size,
        )
        judge_model_selection = ModelSelection(
            name=args.judge_model,
            engine=args.engine,
            vllm_tensor_parallel_size=args.tensor_parallel_size,
        )

        pipeline_cfg = PipelineConfig(
            dataset=dataset_selection,
            test_model=test_model_selection,
            judge_model=judge_model_selection,
            output_dir=args.output_dir,
            batch_size=max(1, args.batch_size),
        )

    run_truthfulness_evaluation(pipeline_cfg, resume_dir=resume_dir)


def _load_config_from_run_dir(run_dir: Path) -> PipelineConfig:
    config_path = run_dir / "config.json"

    if not config_path.exists():
        raise SystemExit(
            f"Cannot resume from {run_dir}: missing config file at {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    dataset_payload = payload.get("dataset", {})
    test_model_payload = payload.get("test_model", {})
    judge_model_payload = payload.get("judge_model", {})

    dataset_selection = DatasetSelection(
        name=dataset_payload.get("name"),
        limit=dataset_payload.get("limit"),
    )
    test_model_selection = ModelSelection(**test_model_payload)
    judge_model_selection = ModelSelection(**judge_model_payload)

    output_dir = Path(payload["output_dir"])
    batch_size = payload.get("batch_size", 1)

    return PipelineConfig(
        dataset=dataset_selection,
        test_model=test_model_selection,
        judge_model=judge_model_selection,
        output_dir=output_dir,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
