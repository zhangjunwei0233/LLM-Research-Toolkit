"""Loader for the HotpotQA dataset shipped via Hugging Face Datasets."""

from __future__ import annotations

from typing import Optional, Iterable, List, Mapping

from datasets import load_dataset

from .base import DatasetExample, DatasetLoader

SUBSET = "distractor"
SPLIT = "validation"


class HotpotQADatasetLoader(DatasetLoader):
    """
    Loads HotpotQA and normalizes every record into DatasetExample objects.

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
        dataset = load_dataset("hotpotqa/hotpot_qa", self.subset, split=self.split)

        # Compute budget: num of examples to use
        total = len(dataset)  # type: ignore
        budget = self.limit or total

        # A generator to generate examples in the desired format
        for idx, row in enumerate(dataset.select(range(min(budget, total)))):  # type: ignore
            references = self._extract_references(row)
            metadata: Mapping[str, object] = {
                "id": row.get("id"),  # type: ignore
                "type": row.get("type"),  # type: ignore
                "level": row.get("level"),  # type: ignore
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

        answer = row.get("answer")
        if isinstance(answer, str) and answer.strip():
            references.append(answer.strip())

        return references


def create_hotpotqa_loader(limit: Optional[int] = None) -> HotpotQADatasetLoader:
    """Factory used by the registry to keep preset wiring in one place."""
    return HotpotQADatasetLoader(
        split=SPLIT,
        subset=SUBSET,
        limit=limit,
    )
