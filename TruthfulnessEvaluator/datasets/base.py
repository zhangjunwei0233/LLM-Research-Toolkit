"""Dataset abstractions used across the evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Protocol, Iterable


@dataclass(slots=True)
class DatasetExample:
    """Normalized representation of a question/answer example."""

    example_id: int
    question: str
    references: List[str]
    metadata: Mapping[str, object]


class DatasetLoader(Protocol):
    """Interface every dataset loader must provide."""

    def load(self) -> Iterable[DatasetExample]:  # type: ignore
        """Return an iterable of dataset examples."""
