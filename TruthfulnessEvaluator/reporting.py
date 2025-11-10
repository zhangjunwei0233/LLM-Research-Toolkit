"""Utilities for summarising and visualizing judge verdicts."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from TruthfulnessEvaluator.judging import JudgementRecord


class ReportBuilder:
    """Aggregates judgements and prints a textual report."""

    def build(self, judgements: Iterable[JudgementRecord]) -> str:
        counts = Counter(j.verdict for j in judgements)
        total = sum(counts.values()) or 1  # avoild zero division

        lines = ["Truthfulness evaluation summary"]
        for label in ("A", "B", "C", "D"):
            fraction = counts.get(label, 0) / total
            bar = self._bar(fraction)
            description = self._label_description(label)
            lines.append(f"{label}: {counts.get(label, 0):3d} "
                         f"({fraction*100:5.1f}%) {bar} {description}")
        return "\n".join(lines)

    @staticmethod
    def _bar(fraction: float, width: int = 30) -> str:
        """Render a quick ASCII progress bar for a given fraction."""
        filled = int(fraction * width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    @staticmethod
    def _label_description(label: str) -> str:
        """Human readable description for verdict labels."""
        return {
            "A": "truthful",
            "B": "false",
            "C": "abstain",
            "D": "judge fail",
        }.get(label, "unknown")
