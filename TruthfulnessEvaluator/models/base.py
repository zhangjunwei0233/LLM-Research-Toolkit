"""Shared config/dataclasses for model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class ModelConfig:
    """Common knobs for any locally hosted causal LM."""

    model_name: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    device: Optional[str] = None
    device_map: Optional[str] = None
    tensor_parallel_size: Optional[int] = None  # for vllm
    batch_size: int = 1
    is_reasoning_model: bool = False


@dataclass(slots=True)
class ModelGeneration:
    """Structured output from the model runner."""

    prompt: str
    completion: str
    prompt_tokens_num: int
    completion_tokens_num: int
