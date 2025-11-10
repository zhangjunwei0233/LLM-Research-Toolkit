"""Configuration helpers for GPT-2 based runs."""

from __future__ import annotations

from dataclasses import replace

from .base import ModelConfig, LocalModelRunner

_TEST_MODEL_CONFIG = ModelConfig(
    model_name="openai-community/gpt2",
    max_new_tokens=48,
    temperature=0.0,
    device=None,
    batch_size=1,
    is_reasoning_model=False,
)

_JUDGE_MODEL_CONFIG = ModelConfig(
    model_name="openai-community/gpt2",
    max_new_tokens=8,
    temperature=0.0,
    device=None,
    batch_size=1,
    is_reasoning_model=False,
)


def create_test_model_runner() -> LocalModelRunner:
    """Return default generation settings for GPT-2 as the testes model."""
    return LocalModelRunner(replace(_TEST_MODEL_CONFIG))


def create_judge_model_runner() -> LocalModelRunner:
    """Return judge-specific settings for GPT-2."""
    return LocalModelRunner(replace(_JUDGE_MODEL_CONFIG))
