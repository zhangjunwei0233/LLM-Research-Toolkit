"""Model utilities and preset runners used by the evaluator."""

from __future__ import annotations

from typing import Callable, Dict

from .base import LocalModelRunner, ModelGeneration
from .gpt2 import create_test_model_runner as gpt2_create_test_model_runner
from .gpt2 import create_judge_model_runner as gpt2_create_judge_model_runner

TEST_MODEL_RUNNER_FACTORIES: Dict[str, Callable[[], LocalModelRunner]] = {
    "gpt2": gpt2_create_test_model_runner,
}

JUDGE_MODEL_RUNNER_FACTORIES: Dict[str, Callable[[], LocalModelRunner]] = {
    "gpt2": gpt2_create_judge_model_runner,
}


def create_test_model_runner(name: str) -> LocalModelRunner:
    """Instantiate a LocalModelRunner for the requested preset name."""
    try:
        factory = TEST_MODEL_RUNNER_FACTORIES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown test model '{name}'. Available: {list(TEST_MODEL_RUNNER_FACTORIES)}"
        ) from exc
    return factory()


def create_judge_model_runner(name: str) -> LocalModelRunner:
    """Instantiate a LocalModelRunner configured for judging."""
    try:
        factory = JUDGE_MODEL_RUNNER_FACTORIES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge model '{name}'. Available: {list(JUDGE_MODEL_RUNNER_FACTORIES)}"
        ) from exc
    return factory()


__all__ = [
    "LocalModelRunner",
    "ModelGeneration",
    "create_test_model_runner",
    "create_judge_model_runner",
]
