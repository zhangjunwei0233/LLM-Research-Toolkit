"""Model utilities and preset runners used by the evaluator."""

from __future__ import annotations

from typing import Callable, Dict, Literal, TYPE_CHECKING

from TruthfulnessEvaluator.config import ModelSelection

from .base import ModelConfig, ModelGeneration
from .transformers_runner import TransformersModelRunner
from .gpt2 import (
    get_judge_model_config as gpt2_get_judge_model_config,
    get_test_model_config as gpt2_get_test_model_config,
)

if TYPE_CHECKING:
    from .vllm_runner import VLLMModelRunner

TEST_MODEL_CONFIG_FACTORIES: Dict[str, Callable[[], ModelConfig]] = {
    "gpt2": gpt2_get_test_model_config,
}

JUDGE_MODEL_CONFIG_FACTORIES: Dict[str, Callable[[], ModelConfig]] = {
    "gpt2": gpt2_get_judge_model_config,
}


def _get_model_config(name: str, role: Literal["test", "judge"]) -> ModelConfig:
    registry = (
        TEST_MODEL_CONFIG_FACTORIES if role == "test" else JUDGE_MODEL_CONFIG_FACTORIES
    )
    try:
        factory = registry[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {role} model '{name}'. Available: {list(registry)}"
        ) from exc
    return factory()


def create_model_runner(selection: ModelSelection, role: Literal["test", "judge"]):
    """Instantiate a model runner using the configured engine."""
    config = _get_model_config(selection.name, role)
    engine = selection.engine.lower()
    if engine == "vllm":
        try:
            from .vllm_runner import VLLMModelRunner
        except ImportError as exc:
            raise RuntimeError(
                "Engine 'vllm' requested but the vllm package is not installed. "
                "Install it via `pip install vllm` or select engine='transformers'."
            ) from exc
        return VLLMModelRunner(config)
    if engine == "transformers":
        return TransformersModelRunner(config)
    raise ValueError(
        f"Unknown engine '{selection.engine}'. Supported engines: transformers, vllm."
    )


__all__ = [
    "ModelGeneration",
    "create_model_runner",
]
