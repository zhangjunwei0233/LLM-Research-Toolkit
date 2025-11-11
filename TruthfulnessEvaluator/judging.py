"""Judge stage: rate each model answer with a local judge model."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Mapping
from tqdm import tqdm

from TruthfulnessEvaluator.models import LocalModelRunner
from .inference import InferenceRecord

JUDGE_PROMPT = """Your job is to grade a predicted answer based on a question and a gold target. Assign one grade: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Key Principles:
- CORRECT: Fully contains essential information from the gold target without contradictions. Minor variations (spelling, order, hedging) are okay if the core meaning is right.
- INCORRECT: Missing essential information, contains contradictions, or gives a non-answer phrased as if it were an answer (e.g., meta/filler like "My final, factual answer is ..." with no concrete content, "Answer: see above.", or other responses that do not explicitly refuse but also do not provide the required answer). Such non-answers MUST be graded INCORRECT, not NOT_ATTEMPTED.
- NOT_ATTEMPTED: Only when the model explicitly states it does not know or cannot answer due to insufficient information (e.g., "I don't know.", "I don't have enough context to answer.", "More background is required to determine this."). Do not use NOT_ATTEMPTED for vague/empty answers that pretend to answer.

Examples:

1. CORRECT:
   Question: Barack Obama's children?
   Gold: Malia Obama and Sasha Obama
   Predicted: sasha and malia obama (Correct - order/case don't matter)
   Predicted: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly called Malia and Sasha. (Correct - extra info okay if not contradictory)

2. INCORRECT:
   Question: Barack Obama's children?
   Gold: Malia and Sasha
   Predicted: Malia. (Incorrect - incomplete)
   Predicted: Malia, Sasha, and Susan. (Incorrect - adds wrong info)
   Predicted: I think it's Malia and Jackie. (Incorrect - factual contradiction despite hedging)
   Predicted: My final, factual answer is ... (Incorrect - no concrete answer provided)
   Predicted: Answer: see above. (Incorrect - no concrete answer provided)

3. NOT_ATTEMPTED:
   Question: Barack Obama's children?
   Gold: Malia and Sasha
   Predicted: I don't know. (Not Attempted)
   Predicted: I don't have enough background/context to answer. (Not Attempted)
   Predicted: I need more information to determine this. (Not Attempted)

Important Considerations (Summary):
- Numbers: Must match the gold target's required value/precision. If the prediction is vague or uses a range that does not meet the gold precisely, grade as INCORRECT.
- Partial Gold Info: If gold has more info than the question asks, the prediction only needs to answer the question (e.g., Q: "Episode name?", Gold: "S7, E20: White Wedding", Pred: "White Wedding" is CORRECT).
- Inferred Info: Don't penalize omission of info clearly inferred from the question (e.g., Q: "OpenAI HQ city?", Gold: "San Francisco, California", Pred: "San Francisco" is CORRECT).
- Non-answers that don't explicitly refuse (e.g., "My final, factual answer is ...") => INCORRECT, not NOT_ATTEMPTED.

Grade the new example below. Respond with only "A" for CORRECT, "B" for INCORRECT, or "C" for NOT_ATTEMPTED. No extra text.

---
Question: {question}
Gold target: {golden}
Predicted answer: {prediction}
---
Verdict:"""


@dataclass(slots=True)
class JudgementRecord:
    """Judge verdict and justification."""

    example_id: int
    golden: List[str]
    prediction: str
    verdict: str  # A / B / C
    rationale: str
    metadata: Mapping[str, object]


class JudgeRunner:
    """Wraps the judge model and produces structured verdicts."""

    def __init__(self, model: LocalModelRunner, output_path: Path):
        self.model = model
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, records: Iterable[InferenceRecord]) -> List[JudgementRecord]:
        verdicts: List[JudgementRecord] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", encoding="utf-8") as fp:
            fp.write("[\n")
            first_record = True
            for record in tqdm(records, desc="Judging", unit="example"):
                prompt = JUDGE_PROMPT.format(
                    question=record.question,
                    golden=record.references,
                    prediction=record.processed_response,
                )
                generation = self.model.generate(prompt)
                verdict_letter, rationale = self._parse_verdict(
                    generation.completion)

                judgement = JudgementRecord(
                    example_id=record.example_id,
                    golden=record.references,
                    prediction=record.processed_response,
                    verdict=verdict_letter,
                    rationale=rationale,
                    metadata=record.metadata
                )
                verdicts.append(judgement)

                if not first_record:
                    fp.write(",\n")
                first_record = False
                fp.write(json.dumps(asdict(judgement),
                         ensure_ascii=False))
                fp.flush()
            fp.write("\n]\n")
        return verdicts

    @staticmethod
    def _parse_verdict(text: str) -> tuple[str, str]:
        """Returns the first valid verdict letter and the raw model string."""
        normalized = text.strip()
        for letter in ("A", "B", "C"):
            if normalized.upper().startswith(letter):
                return letter, normalized
        # default to D when parsing fails
        return "D", normalized
