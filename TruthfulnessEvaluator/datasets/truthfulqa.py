"""Loader for the TruthfulQA dataset shipped via Hugging Face Datasets."""

from __future__ import annotations

from typing import Optional, Iterable, List, Mapping

from datasets import load_dataset

from .base import DatasetExample, DatasetLoader

SUBSET = "generation"
SPLIT = "validation"


class TruthfulQADatasetLoader(DatasetLoader):
    """
    Loads TruthfulQA and normalizes every record into DatasetExample objects.

    The dataset ships with multiple fields; the loader keeps the original
    question and packs the reference answers so downstream components can use
    them for analysis or prompt construction.
    """

    def __init__(
        self,
        split: str = SPLIT,
        subset: str = SUBSET,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.subset = subset
        self.limit = limit

    def load(self) -> Iterable[DatasetExample]:
        # Load the dataset
        dataset = load_dataset("truthful_qa", self.subset, split=self.split)

        # Compute budget: num of examples to use
        total = len(dataset)  # type: ignore
        budget = self.limit or total

        # A generator to generate examples in the desired format
        for idx, row in enumerate(dataset.select(range(min(budget, total)))):  # type: ignore
            references = self._extract_references(row)
            metadata: Mapping[str, object] = {
                "type": row.get("type"),  # type: ignore
                "category": row.get("category"),  # type: ignore
                "source": row.get("source"),  # type: ignore
            }
            example_id = idx

            yield DatasetExample(
                example_id=example_id,
                question=row["question"],  # type: ignore
                references=references,
                metadata=metadata,
            )

    @staticmethod
    def _extract_references(row) -> List[str]:
        references: List[str] = []

        best_answer = row.get("bestanswer")
        if isinstance(best_answer, str) and best_answer.strip():
            references.append(best_answer.strip())

        correct_answers = row.get("correct_answers")
        if isinstance(correct_answers, list):
            references.extend(
                ans.strip()
                for ans in correct_answers
                if isinstance(ans, str) and ans.strip()
            )

        return references


def create_truthfulqa_loader(limit: Optional[int] = None) -> TruthfulQADatasetLoader:
    """Factory used by the registry to keep preset wiring in one place."""
    return TruthfulQADatasetLoader(
        split=SPLIT,
        subset=SUBSET,
        limit=limit,
    )
