"""High-level orchestration for the truthfulness evaluation pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from TruthfulnessEvaluator.config import PipelineConfig
from TruthfulnessEvaluator.datasets import create_dataset_loader
from TruthfulnessEvaluator.models import create_judge_model_runner, create_test_model_runner
from TruthfulnessEvaluator.inference import InferenceRunner
from TruthfulnessEvaluator.judging import JudgeRunner
from TruthfulnessEvaluator.reporting import ReportBuilder


class TruthfulnessPipeline:
    """Coordinates dataset loading, inference, judging and reporting."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(config.output_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        dataset_loader = create_dataset_loader(
            self.config.dataset.name,
            limit=self.config.dataset.limit,
        )

        test_model = create_test_model_runner(self.config.test_model.name)
        judge_model = create_judge_model_runner(self.config.judge_model.name)
        inference_runner = InferenceRunner(
            model=test_model,
            output_path=self.run_dir / "inference_outputs.jsonl",
        )
        judge_runner = JudgeRunner(
            model=judge_model,
            output_path=self.run_dir / "judgements.jsonl",
        )

        inference_records = inference_runner.run(dataset_loader.load())
        judgements = judge_runner.run(inference_records)

        report = ReportBuilder().build(judgements)
        report_path = self.run_dir / "summary.txt"
        report_path.write_text(report, encoding="utf-8")
        print(report)
