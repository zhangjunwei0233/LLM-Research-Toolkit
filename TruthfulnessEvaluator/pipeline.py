"""High-level orchestration for the truthfulness evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from TruthfulnessEvaluator.config import PipelineConfig
from TruthfulnessEvaluator.datasets import create_dataset_loader
from TruthfulnessEvaluator.models import create_model_runner
from TruthfulnessEvaluator.inference import InferenceRunner, InferenceRecord
from TruthfulnessEvaluator.judging import JudgeRunner, JudgementRecord
from TruthfulnessEvaluator.reporting import ReportBuilder


class TruthfulnessPipeline:
    """Coordinates dataset loading, inference, judging and reporting."""

    def __init__(self, config: PipelineConfig, *, resume_dir: Optional[Path] = None):
        self.config = config
        if resume_dir is not None:
            self.run_dir = Path(resume_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = Path(config.output_dir) / f"run_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_config()

    def run(self) -> None:
        inference_path = self.run_dir / "inference_outputs.json"
        judgements_path = self.run_dir / "judgements.json"
        summary_path = self.run_dir / "summary.txt"

        inference_records: List[InferenceRecord]
        if inference_path.exists():
            # Try to load inference records on resume
            print(f"[resume] Reusing inference outputs from {inference_path}")
            inference_records = InferenceRunner.load_records(inference_path)
        else:
            # Else create a new run
            dataset_loader = create_dataset_loader(
                self.config.dataset.name,
                limit=self.config.dataset.limit,
            )

            test_model = create_model_runner(
                self.config.test_model, role="test")
            if hasattr(test_model, "config"):
                test_model.config.batch_size = self.config.batch_size
            try:
                inference_runner = InferenceRunner(
                    model=test_model,
                    output_path=inference_path,
                    batch_size=self.config.batch_size,
                )
                inference_records = inference_runner.run(dataset_loader.load())
            finally:
                test_model.unload()

        judgements: List[JudgementRecord]
        if judgements_path.exists():
            # Try to load judgement records on resume
            print(f"[resume] Reusing judgements from {judgements_path}")
            judgements = JudgeRunner.load_records(judgements_path)
        else:
            # Else create a new run
            judge_model = create_model_runner(
                self.config.judge_model, role="judge")
            if hasattr(judge_model, "config"):
                judge_model.config.batch_size = self.config.batch_size
            try:
                judge_runner = JudgeRunner(
                    model=judge_model,
                    output_path=judgements_path,
                    batch_size=self.config.batch_size,
                )
                judgements = judge_runner.run(inference_records)
            finally:
                judge_model.unload()

        if summary_path.exists():
            # Try to load an existing report on resume
            print(f"[resume] Existing summary found at {summary_path}.")
            report = summary_path.read_text(encoding="utf-8")
            print(report)
            return

        # Else create a new run
        report = ReportBuilder().build(judgements)
        summary_path.write_text(report, encoding="utf-8")
        print(report)

    def _write_run_config(self) -> None:
        config_payload = {
            "dataset": asdict(self.config.dataset),
            "test_model": asdict(self.config.test_model),
            "judge_model": asdict(self.config.judge_model),
            "output_dir": str(self.config.output_dir),
            "batch_size": self.config.batch_size,
        }
        config_path = self.run_dir / "config.json"
        config_path.write_text(
            json.dumps(config_payload, indent=2),
            encoding="utf-8",
        )
