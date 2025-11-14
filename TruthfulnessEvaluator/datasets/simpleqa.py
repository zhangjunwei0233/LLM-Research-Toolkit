"""Loader for the SimpleQA dataset shipped via Hugging Face Datasets."""

from __future__ import annotations

from typing import Optional, Iterable, List, Mapping

from datasets import load_dataset

from .base import DatasetExample, DatasetLoader

SPLIT = "eval"


class SimpleQADatasetLoader(DatasetLoader):
    """
    Loads SimpleQA and normalizes every record into DatasetExample objects.

    The dataset ships with multiple fields; the loader keeps the original
    question and packs the reference answers so downstream components can use
    them for analysis or prompt construction.
    """

    def __init__(
        self,
        split: str = SPLIT,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.limit = limit

    def load(self) -> Iterable[DatasetExample]:
        # Load the dataset
        dataset = load_dataset("google/simpleqa-verified", split=self.split)

        # Compute budget: num of examples to use
        total = len(dataset)  # type: ignore
        budget = self.limit or total

        # A generator to generate examples in the desired format
        for idx, row in enumerate(dataset.select(range(min(budget, total)))):  # type: ignore
            references = self._extract_references(row)
            metadata: Mapping[str, object] = {
                "original_index": row.get("original_index"),  # type: ignore
                "topic": row.get("topic"),  # type: ignore
                "answer_type": row.get("answer_type"),  # type: ignore
                "multi_step": row.get("multi_step"),  # type: ignore
                "requires_reasoning": row.get("requires_reasoning"),  # type: ignore
                "urls": row.get("urls"),  # type: ignore
            }
            example_id = idx

            yield DatasetExample(
                example_id=example_id,
                question=row["problem"],  # type: ignore
                references=references,
                metadata=metadata,
            )

    @staticmethod
    def _extract_references(row) -> List[str]:
        references: List[str] = []

        answer = row.get("answer")
        if isinstance(answer, str) and answer.strip():
            references.append(answer.strip())

        return references


def create_simpleqa_loader(limit: Optional[int] = None) -> SimpleQADatasetLoader:
    """Factory used by the registry to keep preset wiring in one place."""
    return SimpleQADatasetLoader(
        split=SPLIT,
        limit=limit,
    )
